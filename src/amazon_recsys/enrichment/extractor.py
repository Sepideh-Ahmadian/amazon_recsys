"""Run the extractor LLM over reviews: cache, parse, normalise, write enrichment + aspects."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import ValidationError

from amazon_recsys.config import PipelineConfig, require_resolved
from amazon_recsys.data.schemas import ASPECTS, ENRICHMENT
from amazon_recsys.enrichment.cache import ResponseCache, make_input_hash
from amazon_recsys.enrichment.llm_client import LLMClient, LLMError, estimate_tokens
from amazon_recsys.enrichment.prompts import (
    ExtractionOutput,
    load_aspect_map,
    load_extractor_template,
    render_extractor,
)
from amazon_recsys.utils.io import read_parquet, write_json, write_parquet
from amazon_recsys.utils.paths import Paths

ROLE = "extractor"
NEUTRAL_BAND = 0.1  # |sentiment| <= band -> polarity 0
ENRICHMENT_COLUMNS = list(ENRICHMENT.columns)
ASPECT_COLUMNS = list(ASPECTS.columns)
_SINGULAR_EXCEPTIONS = {"lens", "series", "species", "news", "bias", "canvas", "gas", "plus"}
_log = logging.getLogger(__name__)


_MEDIA_MARKER = re.compile(r"\[\[(?:VIDEOID|ASIN|IMAGEID):[^\]]*\]\]", re.I)


def clean_review_text(text: str | None) -> str:
    """Remove Amazon media markers such as [[VIDEOID:...]] from prompt input (data is untouched)."""
    return re.sub(r"\s+", " ", _MEDIA_MARKER.sub(" ", text or "")).strip()


class ParseError(ValueError):
    """The model output could not be turned into a valid ExtractionOutput."""


# --- parsing ------------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    """Find the first JSON object in `text`, tolerating code fences and surrounding prose."""
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S | re.I)
    if fence:
        body = fence.group(1).strip()
    start = body.find("{")
    if start == -1:
        raise ParseError("no JSON object found in response")
    try:
        obj, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError as err:
        raise ParseError(f"invalid JSON: {err}") from err
    if not isinstance(obj, dict):
        raise ParseError("top-level JSON value is not an object")
    return obj


def parse_response(text: str) -> ExtractionOutput:
    try:
        return ExtractionOutput.model_validate(extract_json_object(text))
    except ValidationError as err:
        raise ParseError(f"schema validation failed: {err.error_count()} error(s)") from err


# --- normalisation ------------------------------------------------------------------------


def singularize(word: str) -> str:
    """Light rule-based singular form (no external lemmatiser)."""
    if len(word) <= 3 or word in _SINGULAR_EXCEPTIONS:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalise_aspect(raw: str, synonyms: dict[str, str] | None = None) -> str:
    """lowercase -> strip punctuation -> singularise words -> synonym map (phrase, then word)."""
    synonyms = synonyms or {}
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", raw.lower())
    words = [singularize(w) for w in cleaned.split()]
    phrase = " ".join(words)
    if phrase in synonyms:
        return synonyms[phrase]
    return " ".join(synonyms.get(w, w) for w in words)


def sentiment_to_polarity(sentiment: float) -> int:
    if sentiment > NEUTRAL_BAND:
        return 1
    if sentiment < -NEUTRAL_BAND:
        return -1
    return 0


def aspect_rows(
    review_id: str, domain: str, output: ExtractionOutput, synonyms: dict[str, str]
) -> list[dict[str, Any]]:
    """Normalised aspect rows; a repeated aspect keeps its strongest mention."""
    best: dict[str, dict[str, Any]] = {}
    for item in output.aspects:
        name = normalise_aspect(item.aspect, synonyms)
        if not name:
            continue
        row = {
            "review_id": review_id,
            "domain": domain,
            "aspect": name,
            "aspect_raw": item.aspect,
            "polarity": sentiment_to_polarity(item.sentiment),
            "intensity": min(1.0, abs(item.sentiment)),
            "evidence": item.evidence or None,
        }
        if name not in best or row["intensity"] > best[name]["intensity"]:
            best[name] = row
    return list(best.values())


# --- running ------------------------------------------------------------------------------


@dataclass
class RunStats:
    total: int = 0
    cached: int = 0
    called: int = 0
    ok: int = 0
    parse_error: int = 0
    api_error: int = 0
    skipped_by_limit: int = 0
    api_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "api_errors"}


class Extractor:
    def __init__(
        self,
        client: LLMClient | None,
        cache: ResponseCache,
        template: str,
        prompt_hash: str,
        deployment: str,
        prompt_version: str,
        domain: str,
        synonyms: dict[str, str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
        price_per_million_input_tokens: float = 0.0,
        concurrency: int = 1,
    ) -> None:
        self.client, self.cache = client, cache
        self.template, self.prompt_hash = template, prompt_hash
        self.deployment, self.prompt_version, self.domain = deployment, prompt_version, domain
        self.synonyms = synonyms or {}
        self.temperature, self.max_tokens = temperature, max_tokens
        self.price = price_per_million_input_tokens
        self.concurrency = max(1, concurrency)

    # -- inputs --
    @staticmethod
    def _variables(row: Any, product_title: str) -> dict[str, str]:
        return {
            "product_title": product_title or "",
            "review_title": clean_review_text(row.title),
            "review_text": clean_review_text(row.text),
        }

    def _input_hash(self, variables: dict[str, str]) -> str:
        return make_input_hash(
            variables["product_title"], variables["review_title"], variables["review_text"]
        )

    def _cached(self, review_id: str, input_hash: str) -> str | None:
        return self.cache.get(ROLE, review_id, self.prompt_hash, self.deployment, input_hash)

    # -- dry run --
    def dry_run(self, reviews: pd.DataFrame, titles: dict[str, str]) -> dict[str, Any]:
        uncached, tokens = 0, 0
        for row in reviews.itertuples():
            variables = self._variables(row, titles.get(row.item_id, ""))
            if self._cached(row.review_id, self._input_hash(variables)) is not None:
                continue
            uncached += 1
            messages = render_extractor(self.template, **variables)
            tokens += sum(estimate_tokens(m["content"]) for m in messages)
        return {
            "rows_total": len(reviews),
            "rows_uncached": uncached,
            "estimated_input_tokens": tokens,
            "price_per_million_input_tokens": self.price,
            "estimated_cost": tokens * self.price / 1_000_000,
        }

    # -- real run --
    def _prefetch(
        self, reviews: pd.DataFrame, titles: dict[str, str], limit: int | None, stats: RunStats
    ) -> tuple[set[str], set[str]]:
        """Call the API concurrently for uncached reviews (at most `limit`).

        Returns (fetched_ids, failed_ids). Responses land in the cache, so the sequential loop
        in `run` then builds identical rows to a purely sequential run.
        """
        if self.concurrency <= 1 or self.client is None:
            return set(), set()
        todo: list[tuple[str, dict[str, str], str]] = []
        for row in reviews.itertuples():
            variables = self._variables(row, titles.get(row.item_id, ""))
            input_hash = self._input_hash(variables)
            if self._cached(row.review_id, input_hash) is not None:
                continue
            if limit is not None and len(todo) >= limit:
                break
            todo.append((row.review_id, variables, input_hash))
        if not todo:
            return set(), set()

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(lambda t: (t[0], *self._fetch(*t)), todo))
        stats.called += len(todo)
        fetched, failed = set(), set()
        for review_id, text, error in results:
            if text is None:
                failed.add(review_id)
                stats.api_errors.append(f"{review_id}: {error}")
            else:
                fetched.add(review_id)
        return fetched, failed

    def run(
        self, reviews: pd.DataFrame, titles: dict[str, str], limit: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, RunStats]:
        stats = RunStats(total=len(reviews))
        enrich_rows: list[dict[str, Any]] = []
        asp_rows: list[dict[str, Any]] = []
        fetched, failed = self._prefetch(reviews, titles, limit, stats)

        for row in reviews.itertuples():
            variables = self._variables(row, titles.get(row.item_id, ""))
            input_hash = self._input_hash(variables)
            if row.review_id in failed:  # already attempted (and failed) in the prefetch
                raw = None
            else:
                raw = self._cached(row.review_id, input_hash)
            if raw is not None:
                if row.review_id not in fetched:  # prefetched ones were counted as `called`
                    stats.cached += 1
            elif row.review_id not in failed:
                if limit is not None and stats.called >= limit:
                    stats.skipped_by_limit += 1
                    continue
                raw = self._call(row.review_id, variables, input_hash, stats)

            record: dict[str, Any] = {
                "review_id": row.review_id,
                "domain": self.domain,
                "overall_sentiment": float("nan"),
                "extractor_model": self.deployment,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
            }
            if raw is None:
                record["status"] = "api_error"
                stats.api_error += 1
            else:
                try:
                    output = parse_response(raw)
                except ParseError as err:
                    record["status"] = "parse_error"
                    stats.parse_error += 1
                    _log.warning("parse_error for %s: %s", row.review_id, err)
                else:
                    record["status"] = "ok"
                    record["overall_sentiment"] = output.overall_sentiment
                    asp_rows += aspect_rows(row.review_id, self.domain, output, self.synonyms)
                    stats.ok += 1
            enrich_rows.append(record)

        enrichment = pd.DataFrame(enrich_rows, columns=ENRICHMENT_COLUMNS)
        enrichment = enrichment.astype({"overall_sentiment": "float64"})
        aspects = pd.DataFrame(asp_rows, columns=ASPECT_COLUMNS)
        aspects = aspects.astype({"polarity": "int64", "intensity": "float64"})
        return enrichment, aspects, stats

    def _fetch(
        self, review_id: str, variables: dict[str, str], input_hash: str
    ) -> tuple[str | None, str | None]:
        """One API call; caches the response. Returns (text, None) or (None, error message)."""
        if self.client is None:
            raise RuntimeError("Extractor has no client; use dry_run() instead")
        messages = render_extractor(self.template, **variables)
        try:
            response = self.client.complete(
                messages,
                review_id=review_id,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
        except LLMError as err:
            _log.error("api_error for %s: %s", review_id, err)
            return None, str(err)  # not cached: retried on the next run
        self.cache.put(
            ROLE, review_id, self.prompt_hash, self.deployment, input_hash, response.text
        )
        return response.text, None

    def _call(
        self, review_id: str, variables: dict[str, str], input_hash: str, stats: RunStats
    ) -> str | None:
        stats.called += 1
        text, error = self._fetch(review_id, variables, input_hash)
        if text is None:
            stats.api_errors.append(f"{review_id}: {error}")
        return text


def build_extractor(
    cfg: PipelineConfig, paths: Paths, client: LLMClient | None, cache: ResponseCache
) -> Extractor:
    ext = cfg.enrichment.extractor
    prompt_dir = paths.resolve(cfg.domain.prompt_dir)
    template, prompt_hash = load_extractor_template(prompt_dir, ext.prompt_version)
    return Extractor(
        client=client,
        cache=cache,
        template=template,
        prompt_hash=prompt_hash,
        deployment=ext.deployment,
        prompt_version=ext.prompt_version,
        domain=cfg.domain.name,
        synonyms=load_aspect_map(prompt_dir),
        temperature=ext.temperature,
        max_tokens=ext.max_tokens,
        price_per_million_input_tokens=ext.price_per_million_input_tokens,
        concurrency=ext.concurrency,
    )


def enrich_domain(
    cfg: PipelineConfig,
    paths: Paths,
    client: LLMClient | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Enrich `reviews_filtered.parquet`; with `dry_run` only estimate rows, tokens and cost."""
    from amazon_recsys.data.schemas import ITEMS, REVIEWS

    reviews = read_parquet(paths.reviews_filtered, REVIEWS)
    items = read_parquet(paths.items, ITEMS)
    titles = dict(zip(items["item_id"], items["title"].fillna(""), strict=True))

    with ResponseCache(paths.llm_cache) as cache:
        extractor = build_extractor(cfg, paths, client, cache)
        if dry_run:
            return extractor.dry_run(reviews, titles)
        require_resolved(cfg.enrichment.extractor)
        enrichment, aspects, stats = extractor.run(reviews, titles, limit)

    write_parquet(enrichment, paths.processed("enrichment"), ENRICHMENT)
    write_parquet(aspects, paths.processed("aspects"), ASPECTS)
    summary = {"domain": cfg.domain.name, **stats.as_dict(), "api_errors": stats.api_errors[:20]}
    write_json(summary, paths.processed_dir / "enrichment_summary.json")
    return summary
