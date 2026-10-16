"""Trace ONE review through the whole enrichment path, stage by stage. Read-only: it uses the
cache and the stored tables, never the network, so it is free to run as often as you like.

    python scripts/trace_review.py                       # first review
    python scripts/trace_review.py --index 42
    python scripts/trace_review.py --review-id <id>
    python scripts/trace_review.py --pick uncertain      # first review with that verdict
    python scripts/trace_review.py --pick lowest_faithfulness | lowest_format | lowest_confidence
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd

from amazon_recsys.config import load_pipeline_config, resolve_domain
from amazon_recsys.data.schemas import REVIEWS
from amazon_recsys.enrichment.cache import ResponseCache, make_input_hash
from amazon_recsys.enrichment.extractor import (
    aspect_rows,
    build_extractor,
    extract_json_object,
    parse_response,
)
from amazon_recsys.enrichment.jev_client import ChoiceAnswer, NoulAnswer, ScoreAnswer, parse_response as parse_jev
from amazon_recsys.enrichment.judge import (
    AspectIn,
    aspect_is_well_formed,
    aspect_question_names,
    build_requests,
    build_state,
    extraction_hash,
    normalise_score,
    score_review,
)
from amazon_recsys.enrichment.prompts import load_judge_questions, render_extractor
from amazon_recsys.utils.io import read_parquet
from amazon_recsys.utils.paths import Paths


def heading(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\nSTAGE {n}: {title}\n{'=' * 78}")


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def pick_review(args, judgements: pd.DataFrame, reviews: pd.DataFrame) -> str:
    if args.review_id:
        return args.review_id
    if args.pick == "first":
        return reviews.review_id.iloc[args.index]
    j = judgements.set_index("review_id")
    if args.pick == "uncertain":
        return j.index[j.verdict == "uncertain"][args.index]
    column = {
        "lowest_faithfulness": "faithfulness",
        "lowest_format": "format_compliance",
        "lowest_confidence": "confidence",
    }[args.pick]
    return j[column].sort_values().index[args.index]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--review-id")
    ap.add_argument("--index", type=int, default=0, help="n-th review (or n-th of --pick)")
    ap.add_argument(
        "--pick",
        default="first",
        choices=["first", "uncertain", "lowest_faithfulness", "lowest_format", "lowest_confidence"],
    )
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    cfg = load_pipeline_config(resolve_domain(None, args.configs_dir), args.configs_dir)
    paths = Paths.from_config(cfg, root=Path(args.root))
    reviews = read_parquet(paths.reviews_filtered, REVIEWS)
    judgements = read_parquet(paths.processed("judgements"))
    review_id = pick_review(args, judgements, reviews)
    row = next(reviews[reviews.review_id == review_id].itertuples())
    items = read_parquet(paths.items)
    title = items.set_index("item_id")["title"].fillna("").get(row.item_id, "")

    # ---- stage 1: input ---------------------------------------------------------------------
    heading(1, "INPUT  (reviews_filtered.parquet + items.parquet)")
    print(f"  review_id : {review_id}\n  user/item : {row.user_id} / {row.item_id}   rating: {row.rating}")
    print(f"  product   : {title}\n  title     : {row.title}\n  text      :")
    print(wrap(row.text or ""))

    with ResponseCache(paths.llm_cache) as cache:
        extractor = build_extractor(cfg, paths, None, cache)
        variables = extractor._variables(row, title)

        # ---- stage 2: prompt ----------------------------------------------------------------
        heading(2, "PROMPT  (prompts/<domain>/extractor_<version>.txt rendered with this review)")
        messages = render_extractor(extractor.template, **variables)
        input_hash = extractor._input_hash(variables)
        print(f"  template hash : {extractor.prompt_hash[:16]}...   input hash : {input_hash[:16]}...")
        print(f"  model         : {extractor.deployment}")
        print("  cleaned review text sent to the model:")
        print(wrap(variables["review_text"]))
        print(f"  (system {len(messages[0]['content'])} chars, user {len(messages[1]['content'])} chars)")

        # ---- stage 3: cache -----------------------------------------------------------------
        heading(3, "CACHE  (llm_cache.sqlite: role, review_id, prompt hash, model, input hash)")
        raw = extractor._cached(review_id, input_hash)
        if raw is None:
            print("  MISS: no cached extractor response for this exact input.")
            print("  -> the prompt, model or review text changed since the last run, or it was never run.")
            return
        print("  HIT. Raw model response:")
        print(wrap(raw))

        # ---- stage 4: parse and normalise ---------------------------------------------------
        heading(4, "PARSE + NORMALISE  (extractor.parse_response, aspect_rows)")
        print("  JSON object found in the response:")
        print(wrap(json.dumps(extract_json_object(raw))[:600]))
        output = parse_response(raw)
        print(f"\n  overall_sentiment = {output.overall_sentiment:+.2f}")
        print(f"  {'raw label':28s} {'normalised':26s} {'sent':>6s} {'pol':>4s} {'int':>5s}  evidence")
        rows = aspect_rows(review_id, cfg.domain.name, output, extractor.synonyms)
        for a, r in zip(output.aspects, rows, strict=False):
            print(f"  {a.aspect:28s} {r['aspect']:26s} {a.sentiment:+6.2f} {r['polarity']:+4d} "
                  f"{r['intensity']:5.2f}  {a.evidence[:40]!r}")
        if len(rows) != len(output.aspects):
            print(f"  note: {len(output.aspects)} extracted -> {len(rows)} after merging duplicates")
        aspects = [
            AspectIn(r["aspect"], r["aspect_raw"], r["polarity"], r["intensity"], r["evidence"])
            for r in rows
        ]
        enrichment = read_parquet(paths.processed("enrichment")).set_index("review_id").loc[review_id]
        print(f"\n  stored in enrichment.parquet: status={enrichment.status} "
              f"overall_sentiment={enrichment.overall_sentiment:+.2f}")

        # ---- stage 5: judge input -----------------------------------------------------------
        heading(5, "JUDGE INPUT  (judge.build_state, build_requests)")
        state, truncated = build_state(
            row.title, row.text, row.rating, output.overall_sentiment, aspects,
            cfg.enrichment.judge.max_state_chars,
        )  # fmt: skip
        spec, qhash = load_judge_questions(
            paths.resolve(cfg.domain.prompt_dir), cfg.enrichment.judge.questions_version
        )
        requests = build_requests(spec, aspects, cfg.enrichment.judge.max_aspects_per_request)
        ext_hash = extraction_hash(state)
        j_input_hash = make_input_hash(state["review"], ext_hash)
        print(f"  truncated: {truncated}   extraction hash: {ext_hash[:16]}...   questions hash: {qhash[:16]}...")
        print(f"  requests: {len(requests)}  (max {cfg.enrichment.judge.max_aspects_per_request} aspects each)")
        for n, req in enumerate(requests):
            kinds = {}
            for q in req.values():
                kinds[type(q).__name__] = kinds.get(type(q).__name__, 0) + 1
            print(f"    request {n}: {len(req)} questions  {kinds}")

        # ---- stage 6: judge answers ---------------------------------------------------------
        heading(6, "JUDGE ANSWERS  (Jev probabilities, exactly as returned and cached)")
        raw_j = cache.get("judge", review_id, qhash, cfg.enrichment.judge.model, j_input_hash)
        if raw_j is None:
            print("  MISS: no cached judge response (not judged yet, or the extraction/questions changed).")
            return
        answers = parse_jev(json.loads(raw_j)).answers
        print(f"  {'aspect':26s} {'P(supported)':>12s} {'P(+)':>6s} {'P(-)':>6s} {'P(0)':>6s} {'conf':>6s}  extracted -> Jev")
        names = {1: "positive", -1: "negative", 0: "neutral"}
        for i, a in enumerate(aspects):
            s_name, p_name = aspect_question_names(i)
            c: ChoiceAnswer = answers[p_name]
            p = c.probabilities
            flag = "" if c.choice == names[a.polarity] else "   <-- DISAGREES"
            print(f"  {a.aspect:26s} {answers[s_name].noul:12.2f} {p.get('positive', 0):6.2f} "
                  f"{p.get('negative', 0):6.2f} {p.get('neutral', 0):6.2f} {c.confidence:6.2f}  "
                  f"{names[a.polarity]} -> {c.choice}{flag}")
        for name in ("completeness", "overall_sentiment"):
            s: ScoreAnswer = answers[name]
            dist = " ".join(f"{k}:{v:.2f}" for k, v in sorted(s.probabilities.items()))
            print(f"  {name:26s} score {s.score:.2f} of 0..{s.n_levels - 1}  conf {s.confidence:.2f}   [{dist}]")

        # ---- stage 7: scores ----------------------------------------------------------------
        heading(7, "SCORES  (judge.score_review: how the answers become one verdict)")
        scores = score_review(answers, aspects, output.overall_sentiment, cfg.enrichment.judge, row.text)
        w = cfg.enrichment.judge.weights
        for name in w:
            v = getattr(scores, name)
            print(f"  {name:22s} {v if v is None else round(v, 3)!s:>8}   weight {w[name]:.2f}")
        print(f"  {'overall':22s} {scores.overall:8.3f}   (weighted mean of the above)")
        well = [aspect_is_well_formed(a, row.text) for a in aspects]
        print(f"  format check per aspect: {['ok' if x else 'FAIL' for x in well]}")
        conf = {}
        for name, a in answers.items():
            conf[name] = 2 * abs(a.noul - 0.5) if isinstance(a, NoulAnswer) else a.confidence
        weakest = min(conf, key=conf.get)
        print(f"  {'confidence':22s} {scores.confidence:8.3f}   = the minimum over all answers; weakest: "
              f"{weakest} ({conf[weakest]:.2f})")
        j = cfg.enrichment.judge
        print(f"  verdict: {scores.verdict}   (confidence >= {j.min_confidence}? "
              f"{scores.confidence >= j.min_confidence};  overall >= {j.pass_threshold}? "
              f"{scores.overall >= j.pass_threshold})")
        print(f"  completeness normalised = {normalise_score(answers['completeness']):.3f}")

        # ---- stage 8: stored ----------------------------------------------------------------
        heading(8, "STORED  (judgements.parquet)  vs the recomputation above")
        stored = judgements.set_index("review_id").loc[review_id]
        for name in (*w, "overall", "confidence", "verdict"):
            fresh = getattr(scores, name)
            ok = (stored[name] == fresh) or (
                isinstance(fresh, float) and abs(float(stored[name]) - fresh) < 1e-9
            )
            print(f"  {name:22s} stored {stored[name]!s:>10}  recomputed {fresh!s:>22}  {'OK' if ok else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
