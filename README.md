# amazon_recsys

Does LLM-based review enrichment, checked by a Jev judge, improve graph-based recommendation?

One Amazon review domain (All_Beauty) is enriched with an extractor LLM (aspects, sentiment,
evidence), each extraction is judged by TypeSafe's Jev, and the results feed a weighted
user-item graph and a knowledge graph. GAT and KG-GAT are compared against LightGCN and BPR-MF
under one evaluation protocol. See `CLAUDE.md` for the full specification.

## Setup

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env        # then fill in the keys (never commit .env)
```

| Variable | Used for |
|---|---|
| `OPENROUTER_API_KEY` | extractor (`extractor.provider: openrouter`) |
| `AZURE_OPENAI_*`, `EXTRACTOR_DEPLOYMENT` | extractor if `provider: azure` |
| `TYPESAFE_API_KEY` | Jev judge |

Place the raw files in `data/raw/` (never modified):
`All_Beauty.jsonl.gz` and `meta_All_Beauty.jsonl.gz`.

## Reproduce

```bash
make data                     # ingest -> filter (k-core + user cap from the domain config)
recsys enrich --dry-run       # rows, tokens and cost estimate; no API call
make enrich                   # extractor LLM (cached, resumable); LIMIT=N to cap
make judge                    # Jev judge (cached, resumable); LIMIT=N to cap
make calibrate                # export a blind labelling sheet; label it, then:
recsys calibrate score labels.csv
recsys calibrate thresholds labels.csv    # writes reports/judge_calibration.md
make split                    # interactions -> leave-one-out splits
recsys kg-embed --exp kg_gat  # knowledge graph + cached PyKEEN embeddings
for exp in mf lightgcn gat_plain gat_rating gat_enriched gat_enriched_judged kg_gat kg_gat_unjudged; do
  for seed in 0 1 2; do make train EXP=$exp SEED=$seed; done
done
make report                   # reports/results.md, results.csv, ndcg_at_10.png
```

Every command that calls a model supports `--dry-run` and `--limit N`. Re-running is free:
responses are cached in `data/<domain>/interim/llm_cache.sqlite`, keyed by prompt/question hash,
model and input.

## Experiments

| Experiment | Graph | Weighting | Judge filter | Model |
|---|---|---|---|---|
| `mf` | none | none | none | BPR-MF |
| `lightgcn` | bipartite | none | none | LightGCN |
| `gat_plain` | bipartite | none | none | GAT |
| `gat_rating` | bipartite | rating | none | GAT |
| `gat_enriched` | bipartite | rating_sentiment | off | GAT |
| `gat_enriched_judged` | bipartite | rating_sentiment | 0.7 | GAT |
| `kg_gat` | knowledge | none | judged-pass aspects only | KG-GAT |
| `kg_gat_unjudged` | knowledge | none | every extracted aspect | KG-GAT |

`gat_enriched` vs `gat_enriched_judged` (and `kg_gat` vs `kg_gat_unjudged`) isolate the value
of the judge.

## Development

```bash
make test        # fast suite (not slow, not integration)
make test-all    # includes the end-to-end smoke test
RUN_INTEGRATION=1 make test-integration   # real API calls; costs money
make lint
```

## Things worth knowing

- The development subset is small by design (`max_users` in `configs/domains/all_beauty.yaml`),
  so metrics are noisy and many held-out items are cold. Treat early numbers as a pipeline check.
- On the 60-user subset every model is at chance level (a random scorer gets ndcg@10 0.015,
  hit@10 0.031; the best model 0.019 / 0.039; train popularity is below random). Always read
  results against a random scorer, and do not read differences between experiments as findings
  until the data is large enough to separate them from noise.
- Long runs should be started under `caffeinate -i`: an idle laptop can throttle a background
  process for many minutes (this stretched a ~25 min enrichment to ~105 min of wall time).
- Jev is not perfectly deterministic between calls; the cache is what makes a run reproducible.
- `format_compliance` is a deterministic code check, not a Jev question (see `CLAUDE.md`).
- The aspect knowledge graph carries no category relation: All_Beauty has two distinct categories.
