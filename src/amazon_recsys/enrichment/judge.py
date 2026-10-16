"""Judge each extraction with Jev: build state/questions, combine answers, write tables."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import pandas as pd

from amazon_recsys.config import JudgeCfg, PipelineConfig
from amazon_recsys.data.schemas import (
    ASPECT_JUDGEMENTS,
    ASPECTS,
    ENRICHMENT,
    JUDGEMENTS,
    REVIEWS,
)
from amazon_recsys.enrichment.cache import ResponseCache, make_input_hash
from amazon_recsys.enrichment.jev_client import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    JevError,
    JevResponse,
    JevTooLargeError,
    JudgeClient,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    parse_response,
)
from amazon_recsys.enrichment.llm_client import estimate_tokens
from amazon_recsys.enrichment.prompts import QuestionsSpec, load_judge_questions
from amazon_recsys.utils.hashing import hash_obj, hash_text
from amazon_recsys.utils.io import read_parquet, write_json, write_parquet
from amazon_recsys.utils.paths import Paths

ROLE = "judge"
POLARITY_LABEL = {1: "positive", -1: "negative", 0: "neutral"}
COMPONENTS = (
    "faithfulness",
    "polarity_accuracy",
    "completeness",
    "format_compliance",
    "sentiment_agreement",
)
JUDGEMENT_COLUMNS = list(JUDGEMENTS.columns)
ASPECT_JUDGEMENT_COLUMNS = list(ASPECT_JUDGEMENTS.columns)


@dataclass(frozen=True)
class AspectIn:
    aspect: str  # normalised key used in the tables
    aspect_raw: str  # what the extractor wrote; shown to the judge
    polarity: int
    intensity: float
    evidence: str | None


# --- state and questions ------------------------------------------------------------------


def build_state(
    title: str | None,
    text: str | None,
    rating: float,
    overall_sentiment: float,
    aspects: Sequence[AspectIn],
    max_chars: int,
) -> tuple[dict[str, Any], bool]:
    """The content Jev judges. Review text is cut at `max_chars`; returns (state, truncated)."""
    body = text or ""
    truncated = len(body) > max_chars
    state = {
        # plain Python scalars: numpy types are not JSON-serialisable by the SDK
        "review": {"title": title or "", "text": body[:max_chars], "rating": float(rating)},
        "extraction": {
            "overall_sentiment": float(overall_sentiment),
            "aspects": [
                {
                    "aspect": a.aspect_raw,
                    "polarity": POLARITY_LABEL[a.polarity],
                    "intensity": float(a.intensity),
                    "evidence": a.evidence or "",
                }
                for a in aspects
            ],
        },
    }
    return state, truncated


def extraction_hash(state: Mapping[str, Any]) -> str:
    return hash_obj(state["extraction"])


def aspect_question_names(index: int) -> tuple[str, str]:
    return f"aspect_{index}_supported", f"aspect_{index}_polarity"


def build_requests(
    spec: QuestionsSpec, aspects: Sequence[AspectIn], max_per_request: int
) -> list[dict[str, Question]]:
    """Questions for one review, split into several requests above `max_per_request` aspects.

    Review-level questions (`completeness`, `overall_sentiment`) go in the first request.
    """
    review_level: dict[str, Question] = {}
    review_level["completeness"] = ScoreQuestion(
        tuple(spec.completeness.levels), spec.completeness.instructions
    )
    review_level["overall_sentiment"] = ScoreQuestion(
        tuple(spec.overall_sentiment.levels), spec.overall_sentiment.instructions
    )

    requests: list[dict[str, Question]] = [dict(review_level)]
    for index, aspect in enumerate(aspects):
        if index > 0 and index % max_per_request == 0:
            requests.append({})
        supported, polarity = aspect_question_names(index)
        requests[-1][supported] = NoulQuestion(
            {
                "extracted_aspect": aspect.aspect_raw,
                "evidence": aspect.evidence or "",
                "question": spec.aspect_supported.instructions,
            }
        )
        requests[-1][polarity] = ChoiceQuestion(
            tuple(spec.aspect_polarity.options),
            {
                "extracted_aspect": aspect.aspect_raw,
                "question": spec.aspect_polarity.instructions,
            },
        )
    return requests


# --- pure scoring -------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def normalise_text(text: str | None) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


MIN_LABEL_WORDS, MAX_LABEL_WORDS = 2, 4


def aspect_is_well_formed(aspect: AspectIn, review_text: str | None) -> bool:
    """Deterministic format check: evidence appears verbatim in the review and the label is a
    short noun phrase (2-4 words, i.e. roughly one adjective + one noun)."""
    evidence = normalise_text(aspect.evidence)
    verbatim = bool(evidence) and evidence in normalise_text(review_text)
    n_words = len(normalise_text(aspect.aspect_raw).split())
    return verbatim and MIN_LABEL_WORDS <= n_words <= MAX_LABEL_WORDS


def format_compliance(aspects: Sequence[AspectIn], review_text: str | None) -> float | None:
    """Fraction of aspects that are well formed; None when there are no aspects."""
    if not aspects:
        return None
    return sum(aspect_is_well_formed(a, review_text) for a in aspects) / len(aspects)


def faithfulness(p_supported: Sequence[float]) -> float | None:
    return sum(p_supported) / len(p_supported) if p_supported else None


def polarity_accuracy(
    choices: Sequence[ChoiceAnswer], extracted_polarities: Sequence[int]
) -> float | None:
    """Mean probability Jev assigns to the *extracted* polarity."""
    if not choices:
        return None
    hits = [
        c.probabilities.get(POLARITY_LABEL[p], 0.0)
        for c, p in zip(choices, extracted_polarities, strict=True)
    ]
    return sum(hits) / len(hits)


def normalise_score(answer: ScoreAnswer) -> float:
    """score / (n_levels - 1), clamped to [0, 1]."""
    return _clamp01(answer.score / max(answer.n_levels - 1, 1))


def sentiment_agreement(jev_norm: float, extractor_sentiment: float) -> float:
    """1 - |jev_norm - (extractor_sentiment + 1) / 2|."""
    return _clamp01(1.0 - abs(jev_norm - (extractor_sentiment + 1.0) / 2.0))


def weighted_overall(
    components: Mapping[str, float | None], weights: Mapping[str, float]
) -> float | None:
    """Weighted mean over components that are present; weights renormalised over those."""
    pairs = [
        (components[k], w) for k, w in weights.items() if components.get(k) is not None and w > 0
    ]
    total = sum(w for _, w in pairs)
    if total == 0:
        return None
    return _clamp01(sum(v * w for v, w in pairs) / total)


def answer_confidence(answers: Mapping[str, Answer]) -> float | None:
    """Minimum over Choice/Score confidences and 2|p - 0.5| for every Noul."""
    values: list[float] = []
    for answer in answers.values():
        if isinstance(answer, NoulAnswer):
            values.append(2.0 * abs(answer.noul - 0.5))
        else:
            values.append(answer.confidence)
    return _clamp01(min(values)) if values else None


def decide_verdict(
    overall: float | None, confidence: float | None, pass_threshold: float, min_confidence: float
) -> str | None:
    if overall is None or confidence is None:
        return None
    if confidence < min_confidence:
        return "uncertain"
    return "pass" if overall >= pass_threshold else "fail"


@dataclass(frozen=True)
class ReviewScores:
    faithfulness: float | None
    polarity_accuracy: float | None
    completeness: float | None
    format_compliance: float | None
    sentiment_agreement: float | None
    overall: float | None
    confidence: float | None
    verdict: str | None


def score_review(
    answers: Mapping[str, Answer],
    aspects: Sequence[AspectIn],
    extractor_sentiment: float,
    cfg: JudgeCfg,
    review_text: str | None = None,
) -> ReviewScores:
    supported, choices = [], []
    for i in range(len(aspects)):
        s_name, p_name = aspect_question_names(i)
        supported.append(answers[s_name].noul)
        choices.append(answers[p_name])

    completeness = normalise_score(answers["completeness"])
    components: dict[str, float | None] = {
        "faithfulness": faithfulness(supported),
        "polarity_accuracy": polarity_accuracy(choices, [a.polarity for a in aspects]),
        "completeness": completeness,
        "format_compliance": format_compliance(aspects, review_text),
        "sentiment_agreement": sentiment_agreement(
            normalise_score(answers["overall_sentiment"]), extractor_sentiment
        ),
    }
    overall = weighted_overall(components, cfg.weights)
    confidence = answer_confidence(answers)
    return ReviewScores(
        **components,
        overall=overall,
        confidence=confidence,
        verdict=decide_verdict(overall, confidence, cfg.pass_threshold, cfg.min_confidence),
    )


def aspect_judgement_rows(
    review_id: str, domain: str, aspects: Sequence[AspectIn], answers: Mapping[str, Answer]
) -> list[dict[str, Any]]:
    rows = []
    for i, aspect in enumerate(aspects):
        s_name, p_name = aspect_question_names(i)
        choice: ChoiceAnswer = answers[p_name]
        probs = choice.probabilities
        rows.append(
            {
                "review_id": review_id,
                "domain": domain,
                "aspect": aspect.aspect,
                "p_supported": answers[s_name].noul,
                "p_polarity_match": probs.get(POLARITY_LABEL[aspect.polarity], 0.0),
                "p_positive": probs.get("positive", 0.0),
                "p_negative": probs.get("negative", 0.0),
                "p_neutral": probs.get("neutral", 0.0),
                "polarity_confidence": choice.confidence,
            }
        )
    return rows


# --- serialisation for the cache ------------------------------------------------------------


def response_to_dict(response: JevResponse) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for name, a in response.answers.items():
        if isinstance(a, NoulAnswer):
            answers[name] = {"type": "noul", "noul": a.noul}
        elif isinstance(a, ChoiceAnswer):
            answers[name] = {
                "type": "choice",
                "choice": a.choice,
                "confidence": a.confidence,
                "probabilities": a.probabilities,
            }
        else:
            answers[name] = {
                "type": "score",
                "score": a.score,
                "confidence": a.confidence,
                "probabilities": {str(k): v for k, v in a.probabilities.items()},
            }
    return {
        "model": response.model,
        "answers": answers,
        "usage": {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
    }


def merge_responses(responses: Sequence[JevResponse]) -> JevResponse:
    answers: dict[str, Answer] = {}
    tokens = 0
    for r in responses:
        answers.update(r.answers)
        tokens += r.input_tokens or 0
    return JevResponse(responses[0].model, answers, tokens)


# --- sampling --------------------------------------------------------------------------------


def in_sample(review_id: str, rate: float, seed: int) -> bool:
    """Seeded, order-independent membership test; rate 1.0 keeps everything."""
    if rate >= 1.0:
        return True
    fraction = int(hash_text(f"{seed}|{review_id}")[:8], 16) / 0xFFFFFFFF
    return fraction < rate


# --- runner ----------------------------------------------------------------------------------


@dataclass
class JudgeStats:
    total: int = 0
    sampled_out: int = 0
    cached: int = 0
    called: int = 0
    skipped_by_limit: int = 0
    ok: int = 0
    no_aspects: int = 0
    api_error: int = 0
    too_large: int = 0


class Judge:
    def __init__(
        self,
        client: JudgeClient | None,
        cache: ResponseCache,
        spec: QuestionsSpec,
        questions_hash: str,
        cfg: JudgeCfg,
        domain: str,
        seed: int = 0,
        concurrency: int = 1,
    ) -> None:
        if set(spec.aspect_polarity.options) != set(POLARITY_LABEL.values()):
            raise ValueError("aspect_polarity options must be positive, negative and neutral")
        self.client, self.cache, self.spec = client, cache, spec
        self.questions_hash, self.cfg, self.domain, self.seed = questions_hash, cfg, domain, seed
        self.concurrency = max(1, concurrency)

    def _prepare(self, review: Any, sentiment: float, aspects: list[AspectIn]):
        state, truncated = build_state(
            review.title, review.text, review.rating, sentiment, aspects, self.cfg.max_state_chars
        )
        requests = build_requests(self.spec, aspects, self.cfg.max_aspects_per_request)
        ext_hash = extraction_hash(state)
        input_hash = make_input_hash(state["review"], ext_hash)
        return state, truncated, requests, ext_hash, input_hash

    def _cache_get(self, review_id: str, input_hash: str) -> JevResponse | None:
        raw = self.cache.get(ROLE, review_id, self.questions_hash, self.cfg.model, input_hash)
        if raw is None:
            return None
        try:
            return parse_response(json.loads(raw))
        except (JevError, ValueError):  # unreadable cache entry: treat as a miss, never crash
            return None

    def dry_run(
        self, reviews: pd.DataFrame, sentiments: Mapping[str, float], by_review: Mapping[str, list]
    ) -> dict[str, Any]:
        uncached = tokens = requests_total = 0
        for review in reviews.itertuples():
            if review.review_id not in sentiments or not in_sample(
                review.review_id, self.cfg.sample_rate, self.seed
            ):
                continue
            aspects = by_review.get(review.review_id, [])
            state, _, requests, _, input_hash = self._prepare(
                review, sentiments[review.review_id], aspects
            )
            if self._cache_get(review.review_id, input_hash) is not None:
                continue
            uncached += 1
            for questions in requests:
                requests_total += 1
                payload = json.dumps(
                    {"state": state, "questions": {k: q.to_payload() for k, q in questions.items()}}
                )
                tokens += estimate_tokens(payload)
        price = self.cfg.price_per_million_input_tokens
        return {
            "rows_total": len(reviews),
            "rows_uncached": uncached,
            "requests": requests_total,
            "estimated_input_tokens": tokens,
            "price_per_million_input_tokens": price,
            "estimated_cost": tokens * price / 1_000_000,
        }

    def _fetch(
        self,
        review_id: str,
        state: dict[str, Any],
        requests: list[dict[str, Question]],
        input_hash: str,
    ) -> tuple[JevResponse | None, str | None]:
        """Ask Jev for one review and cache it. Returns (response, None) or (None, status)."""
        try:
            merged = self._ask(state, requests)
        except JevTooLargeError:
            return None, "too_large"
        except JevError:
            return None, "api_error"
        self.cache.put(
            ROLE, review_id, self.questions_hash, self.cfg.model, input_hash,
            json.dumps(response_to_dict(merged)),
        )  # fmt: skip
        return merged, None

    def _prefetch(
        self,
        reviews: pd.DataFrame,
        sentiments: Mapping[str, float],
        by_review: Mapping[str, list[AspectIn]],
        limit: int | None,
        stats: JudgeStats,
    ) -> dict[str, tuple[JevResponse | None, str | None]]:
        """Judge uncached reviews concurrently (at most `limit`); results are cached too."""
        if self.concurrency <= 1 or self.client is None:
            return {}
        todo = []
        for review in reviews.itertuples():
            rid = review.review_id
            if rid not in sentiments or not in_sample(rid, self.cfg.sample_rate, self.seed):
                continue
            state, _, requests, _, input_hash = self._prepare(
                review, sentiments[rid], by_review.get(rid, [])
            )
            if self._cache_get(rid, input_hash) is not None:
                continue
            if limit is not None and len(todo) >= limit:
                break
            todo.append((rid, state, requests, input_hash))
        if not todo:
            return {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(lambda t: (t[0], self._fetch(*t)), todo))
        stats.called += len(todo)
        return dict(results)

    def run(
        self,
        reviews: pd.DataFrame,
        sentiments: Mapping[str, float],
        by_review: Mapping[str, list[AspectIn]],
        limit: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, JudgeStats]:
        stats = JudgeStats()
        j_rows: list[dict[str, Any]] = []
        a_rows: list[dict[str, Any]] = []
        outcomes = self._prefetch(reviews, sentiments, by_review, limit, stats)

        for review in reviews.itertuples():
            if review.review_id not in sentiments:  # extraction failed: nothing to judge
                continue
            stats.total += 1
            if not in_sample(review.review_id, self.cfg.sample_rate, self.seed):
                stats.sampled_out += 1
                continue
            aspects = by_review.get(review.review_id, [])
            sentiment = sentiments[review.review_id]
            state, truncated, requests, ext_hash, input_hash = self._prepare(
                review, sentiment, aspects
            )
            row: dict[str, Any] = {
                "review_id": review.review_id,
                "domain": self.domain,
                "judge_status": "ok",
                "truncated": truncated,
                "jev_model": None,
                "questions_hash": self.questions_hash,
                "extraction_hash": ext_hash,
                "input_tokens": None,
            }

            failure: str | None = None
            if review.review_id in outcomes:  # judged concurrently; already counted as `called`
                merged, failure = outcomes[review.review_id]
            else:
                merged = self._cache_get(review.review_id, input_hash)
                if merged is not None:
                    stats.cached += 1
                else:
                    if limit is not None and stats.called >= limit:
                        stats.skipped_by_limit += 1
                        continue
                    stats.called += 1
                    merged, failure = self._fetch(review.review_id, state, requests, input_hash)
            if failure is not None:
                row["judge_status"] = failure
                if failure == "too_large":
                    stats.too_large += 1
                else:
                    stats.api_error += 1

            if merged is not None:
                scores = score_review(merged.answers, aspects, sentiment, self.cfg, review.text)
                row.update(
                    {
                        k: getattr(scores, k)
                        for k in (*COMPONENTS, "overall", "confidence", "verdict")
                    }
                )
                row["jev_model"] = merged.model
                row["input_tokens"] = merged.input_tokens
                if not aspects:
                    row["judge_status"] = "no_aspects"
                    stats.no_aspects += 1
                else:
                    stats.ok += 1
                a_rows += aspect_judgement_rows(
                    review.review_id, self.domain, aspects, merged.answers
                )
            j_rows.append(row)

        judgements = pd.DataFrame(j_rows, columns=JUDGEMENT_COLUMNS)
        for col in (*COMPONENTS, "overall", "confidence"):
            judgements[col] = judgements[col].astype("float64")
        judgements["truncated"] = judgements["truncated"].astype(bool)
        judgements["input_tokens"] = judgements["input_tokens"].astype("Int64")
        aspect_df = pd.DataFrame(a_rows, columns=ASPECT_JUDGEMENT_COLUMNS)
        return judgements, aspect_df, stats

    def _ask(self, state: dict[str, Any], requests: list[dict[str, Question]]) -> JevResponse:
        if self.client is None:
            raise RuntimeError("Judge has no client; use dry_run() instead")
        return merge_responses([self.client.ask(state, q) for q in requests])


# --- domain-level entry point ----------------------------------------------------------------


def _aspects_by_review(aspects: pd.DataFrame) -> dict[str, list[AspectIn]]:
    grouped: dict[str, list[AspectIn]] = {}
    for r in aspects.itertuples():
        grouped.setdefault(r.review_id, []).append(
            AspectIn(r.aspect, r.aspect_raw, int(r.polarity), float(r.intensity), r.evidence)
        )
    return grouped


def judge_domain(
    cfg: PipelineConfig,
    paths: Paths,
    client: JudgeClient | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    reviews = read_parquet(paths.reviews_filtered, REVIEWS)
    if not paths.processed("enrichment").exists():
        raise FileNotFoundError("No enrichment found; run `recsys enrich` first")
    enrichment = read_parquet(paths.processed("enrichment"), ENRICHMENT)
    aspects = read_parquet(paths.processed("aspects"), ASPECTS)

    ok = enrichment[enrichment["status"] == "ok"]
    sentiments = dict(zip(ok["review_id"], ok["overall_sentiment"], strict=True))
    by_review = _aspects_by_review(aspects)

    spec, questions_hash = load_judge_questions(
        paths.resolve(cfg.domain.prompt_dir), cfg.enrichment.judge.questions_version
    )
    with ResponseCache(paths.llm_cache) as cache:
        judge = Judge(
            client, cache, spec, questions_hash, cfg.enrichment.judge, cfg.domain.name, cfg.seed,
            concurrency=cfg.enrichment.judge.concurrency,
        )  # fmt: skip
        if dry_run:
            return judge.dry_run(reviews, sentiments, by_review)
        judgements, aspect_df, stats = judge.run(reviews, sentiments, by_review, limit)

    write_parquet(judgements, paths.processed("judgements"), JUDGEMENTS)
    write_parquet(aspect_df, paths.processed("aspect_judgements"), ASPECT_JUDGEMENTS)
    summary = {"domain": cfg.domain.name, "jev_model": cfg.enrichment.judge.model, **stats.__dict__}
    write_json(summary, paths.processed_dir / "judge_summary.json")
    return summary
