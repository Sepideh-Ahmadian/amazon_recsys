# CLAUDE.md — amazon_recsys

Instructions for Claude Code working in this repository. Read this file fully before writing code. When this file and the existing code disagree, this file wins.

---

## 1. Purpose

Answer one research question:

> **Does LLM-based review enrichment, checked by a Jev judge, improve graph-based recommendation?**

The pipeline:

1. Load one Amazon review domain (currently **All_Beauty**).
2. **Enrich** each review with an extractor LLM (aspects, sentiment, evidence).
3. **Judge** each extraction with **Jev** (TypeSafe's decision model), and keep calibrated quality probabilities per review and per aspect.
4. Build two graphs from the training split: a **weighted user–item graph** and a **knowledge graph**.
5. Train graph models (GAT on weighted graph, GAT initialised from KG embeddings) and compare them against **LightGCN** and **BPR-MF** under one evaluation protocol.

---

## 2. Scope

### In scope now
- One domain, selected by config (`all_beauty`).
- Extractor LLM (Azure OpenAI) + Jev judge, both cached and resumable.
- Graph builders: `bipartite` (with a weighting option) and `knowledge`.
- Models: `mf` (BPR-MF), `lightgcn`, `gat`, `kg_gat`.
- One trainer, one evaluator, one report script.

### Out of scope — do NOT implement
- Cross-domain logic (no source/target domains, no domain masks, no user overlap across domains).
- Hypergraphs / HGNN.
- Hydra, W&B/MLflow, hyperparameter sweeps, significance tests.
- Popularity-weighted or hard negative sampling (uniform only).

### Must stay extensible to
- **More domains**: adding a domain = one YAML file + one prompt file. No code changes.
- **Hypergraphs**: adding one = one new builder + one new model, registered by name. No changes to trainer, evaluator or existing builders.
- **Cross-domain later**: every table already carries a `domain` column, and all paths are namespaced by domain.

If a task seems to need an out-of-scope feature, stop and ask instead of building it.

---

## 3. Directory layout

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml            # package, pinned deps, ruff + pytest config
├── Makefile                  # make data | enrich | judge | train EXP=... | report | test | lint
├── .env.example              # AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, EXTRACTOR_DEPLOYMENT, TYPESAFE_API_KEY
├── .gitignore                # data/, runs/, reports/, .env, *.pt, *.pkl, __pycache__/
├── configs/
│   ├── base.yaml             # seed, device, paths
│   ├── domains/all_beauty.yaml
│   ├── enrichment.yaml       # extractor + judge settings
│   ├── split.yaml
│   ├── graphs/{bipartite,knowledge}.yaml
│   ├── models/{mf,lightgcn,gat,kg_gat}.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── experiments/*.yaml    # one file per row of the results table
├── prompts/
│   └── all_beauty/
│       ├── extractor_v1.txt
│       ├── judge_questions_v1.yaml   # Jev question set (instructions + criteria/levels)
│       └── aspect_map.yaml           # optional synonym map
├── src/amazon_recsys/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── schemas.py
│   │   ├── ingest.py
│   │   ├── filter.py
│   │   ├── interactions.py
│   │   └── splits.py
│   ├── enrichment/
│   │   ├── llm_client.py     # extractor client (Azure OpenAI + Fake)
│   │   ├── jev_client.py     # judge client (TypeSafe Jev + Fake)
│   │   ├── prompts.py
│   │   ├── extractor.py
│   │   ├── judge.py
│   │   ├── cache.py
│   │   └── calibration.py
│   ├── graphs/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── bipartite.py
│   │   ├── knowledge.py
│   │   └── kg_embeddings.py
│   ├── models/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── mf.py
│   │   ├── lightgcn.py
│   │   ├── gat.py
│   │   └── kg_gat.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── sampling.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       ├── io.py
│       ├── paths.py
│       └── hashing.py
├── scripts/
│   └── make_report.py
├── tests/                    # mirrors src/; see section 9
├── legacy/                   # old scripts, read-only reference
├── data/                     # untracked
│   ├── raw/                  # original downloads, never modified
│   └── {domain}/
│       ├── interim/          # reviews.parquet, items.parquet, llm_cache.sqlite
│       ├── processed/        # enrichment, judgements, interactions, splits/
│       └── embeddings/       # KG embeddings, named by config hash
├── runs/<run_id>/            # untracked: config.yaml, metrics.json, checkpoint.pt, git_hash.txt, log
└── reports/                  # untracked, regenerable
```

Dependency direction is one-way:

```
config ─▶ cli ─▶ data ─▶ enrichment ─▶ graphs ─▶ models ─▶ training ─▶ evaluation
                  └────────────── utils (used by all, depends on nothing) ──────────┘
```

`data` never imports `enrichment`; `graphs` never imports `models`; `models` never imports `training`; `evaluation` depends only on the `Recommender` interface and split files.

---

## 4. Configuration

- Plain YAML loaded into **pydantic** models in `config.py`. No Hydra.
- An experiment file lists which component files to use and any overrides:

```yaml
# configs/experiments/gat_weighted_judged.yaml
domain: all_beauty
graph: bipartite
model: gat
overrides:
  graph.weighting: rating_sentiment
  graph.min_judge_score: 0.7
  train.epochs: 100
```

- CLI overrides: `recsys train --exp gat_weighted_judged --set train.lr=0.0005 --seed 1`.
- Every run writes its fully resolved config to `runs/<run_id>/config.yaml`.
- Invalid keys must raise at load time (`extra="forbid"`).

### Domain config

```yaml
# configs/domains/all_beauty.yaml
name: all_beauty
reviews_file: data/raw/All_Beauty.jsonl.gz
meta_file: data/raw/meta_All_Beauty.jsonl.gz
columns: {user: user_id, item: parent_asin, rating: rating, text: text, title: title, time: timestamp}
prompt_dir: prompts/all_beauty
filter:
  min_user_interactions: 5
  min_item_interactions: 5
  k_core: true
  max_users: 60             # sized to a 30-minute enrichment budget; null = every user that survives k-core
  sample_seed: 42
  sample_strategy: most_active          # random | most_active
  min_item_interactions_after_sample: 2 # item minimum for the k-core after sampling (null = min_item_interactions)
```

**Development subset.** The raw All_Beauty file has 631,986 users but only 253 survive an iterative 5-core (356 items, 2,535 interactions). The first round used `max_users: 20` (498 reviews). The current subset is `max_users: 60` (1,171 reviews), sized to a 30-minute enrichment budget from the measured rates (extraction ~0.8 s and judge ~1.5 s per review at 8 parallel calls); the plan is to expand it further (up to `null` for the full core: 253 users, 2,544 reviews). The cap is only a config value: setting `max_users: null` (or a larger number) expands to the full core with no code change, and a different domain brings its own `max_users`.

**Filter order is fixed:** k-core → sample of `max_users` users → k-core again (sampling can push items below the minimum). Never sample before the first k-core: almost every raw user has a single review, so sampling first would leave an almost empty dataset.

**Why `most_active` and a relaxed item minimum.** Measured on All_Beauty: a strict 5-core on a *random* sample of 20, 50, 100 or 150 core users is empty, because an item needs 5 distinct users and a small sample rarely overlaps that much (200 random users leave 97 users / 132 items). So the development subset takes the `max_users` users with the most distinct items (ties broken by user id, deterministic) and re-runs the k-core with `min_item_interactions_after_sample: 2` (users still need 5 items). Result for 20 users: 20 users, 186 items, 498 reviews. `sample_strategy: random` and a null `min_item_interactions_after_sample` restore the strict behaviour when the cap is large enough.

**Raw data location:** `data/raw/` at the project root (not inside `src/`). Files are never modified.

---

## 5. Data contracts

All stage outputs are parquet, validated with **pandera** schemas in `data/schemas.py` on write and on read. Every table has a `domain` column.

| Table | Key columns |
|---|---|
| `reviews` | review_id, domain, user_id, item_id, rating, title, text, timestamp |
| `items` | item_id, domain, title, category, store, description |
| `enrichment` | review_id, domain, status (`ok`/`parse_error`/`api_error`), overall_sentiment ∈ [-1, 1], extractor_model, prompt_version, prompt_hash |
| `aspects` (long) | review_id, domain, aspect, polarity ∈ {-1, 0, 1}, intensity ∈ [0, 1], evidence |
| `judgements` | review_id, domain, judge_status (`ok`/`api_error`/`too_large`/`no_aspects`), faithfulness, polarity_accuracy, completeness, format_compliance, sentiment_agreement (each ∈ [0, 1]), overall ∈ [0, 1], confidence ∈ [0, 1], verdict (`pass`/`fail`/`uncertain`), truncated (bool), jev_model (version returned by the API), questions_hash, extraction_hash, input_tokens |
| `aspect_judgements` (long) | review_id, domain, aspect, p_supported, p_polarity_match, p_positive, p_negative, p_neutral, polarity_confidence |
| `interactions` | domain, user_id, item_id, rating, timestamp, review_id, overall_sentiment, judge_overall, judge_verdict (`pass`/`fail`/`uncertain`, nullable; lets the graph builder apply the uncertain rule) |
| `splits/{train,val,test}` | same columns as interactions |
| `splits/manifest.json` | counts, seed, strategy, input hashes, leakage check result |

`review_id` is a stable hash of `(domain, user_id, item_id, timestamp)`.

Aspect vocabulary: the extractor returns free-text aspects; `extractor.py` normalises them (lowercase, lemmatise, map through an optional per-domain synonym file `prompts/{domain}/aspect_map.yaml`). Keep the raw string in a `aspect_raw` column.

---

## 6. Module specifications

### data/
- **ingest.py** — stream `.jsonl.gz` → `reviews.parquet`, `items.parquet`. Skip and count malformed lines. Normalise timestamps to UTC ms. Record row counts and file checksum in a sidecar JSON.
- **filter.py** — apply the domain `filter` config in this order: iterative k-core, then optional `max_users` sample (`null` = no cap; strategy `random` is seeded, `most_active` takes the users with the most distinct items), then k-core again (item minimum `min_item_interactions_after_sample`, defaulting to `min_item_interactions`) so the final frame meets the minimums. Output the filtered reviews and a summary of what was dropped and why at each stage (counts before/after k-core, after sampling, after the second k-core). The final user count may be below `max_users`.
- **interactions.py** — join filtered reviews with enrichment and judgements (left join; missing enrichment is allowed and yields nulls). Produce `interactions.parquet`.
- **splits.py** — leave-one-out by time: per user, last interaction → test, second-to-last → val, rest → train. Seeded tie-breaking. Write manifest. Assert no (user, item) pair from val/test appears in train.
- Dataset statistics live as functions here (`interactions.dataset_stats`) or in a notebook; no separate stats module.
- Duplicate reviews (same `review_id`) are dropped, keeping the first, and counted in the ingest manifest; there is no option for another policy.

### enrichment/  (the core of this project)
- **llm_client.py** — `LLMClient` protocol with `complete(messages, *, review_id=None, **params) -> LLMResponse` (text, model, tokens in/out, latency). Two provider implementations selected by `extractor.provider`: `AzureOpenAIClient` and `OpenRouterClient` (OpenAI-compatible, `OPENROUTER_API_KEY`, model slug in `extractor.deployment`), both created lazily (never at import) and built through `make_llm_client`, which wraps them with retries (transient errors only) and a rate limiter. `FakeLLMClient` for tests returns canned responses keyed by review_id.
- **prompts.py** — loads `prompts/{domain}/extractor_{version}.txt` and `prompts/{domain}/judge_questions_{version}.yaml`, renders variables, and returns `(content, hash)`. Defines the pydantic schema for extractor JSON output. The extractor file holds the system prompt, a `---USER---` line, then the user prompt, with `${review_title}`, `${review_text}`, `${product_title}` variables; a missing variable raises. The v1 prompt is adapted from the legacy `LLM_enrichment.py` prompt (strict JSON, per-review `overall_sentiment`, evidence phrase per aspect) and, like the question wording, is a placeholder the owner will refine.
- **cache.py** — SQLite cache keyed by `(role, review_id, prompt_or_questions_hash, model, input_hash)`. Stores the raw response JSON. The judge's `input_hash` includes the extraction hash, so changing the extractor invalidates judgements automatically.
- **extractor.py** — one LLM call per review (the legacy script batched many reviews per call; per-review calls keep parse errors, retries and cache entries isolated; `batch_size` is only the number of reviews processed between progress/commit points), skip cached rows, call the client with retries (exponential backoff, configurable max), parse with the pydantic schema, normalise aspects, write `enrichment.parquet` + `aspects.parquet`. Parse failures get `status=parse_error` and keep the raw text in the cache; never drop them silently.
- **jev_client.py**, **judge.py**, **calibration.py** — see "Judge: Jev" below.

#### Judge: Jev

Jev is TypeSafe AI's "System One" model. It does **not** generate text: it takes a `state` (the content to judge) plus a set of typed `questions`, and returns probabilities. That suits a judge well: no JSON parsing failures, calibrated probabilities instead of a made-up 1–5 score, and very low cost per call. Its known weakness is multi-step reasoning, which this task doesn't need — the evidence (the review text) is always in the state.

Three question types are used:

| Type | Returns | Used for |
|---|---|---|
| **Noul** | `noul` = P(yes) | "Is this aspect actually mentioned in the review?" |
| **Choice** | `choice`, `probabilities` per option, `confidence` | "What is the reviewer's sentiment toward this aspect?" (positive / negative / neutral) |
| **Score** | `score` (expected level, 0…n−1), `probabilities` per level, `confidence` | "How completely does the extraction cover the aspects in the review?"; overall review sentiment on an ordered scale |

API facts (verify against https://docs.typesafe.ai before coding; install the TypeSafe agent skill if available):
- Python SDK: `pip install typesafe-sdk`; `from typesafe_sdk import TypeSafeClient, Noul, NoulCriteria, Score, Choice`; `client.system_one(model=..., state=..., questions={...})`; key from `TYPESAFE_API_KEY`.
- HTTP fallback: `POST https://api.typesafe.ai/v1/systemone` with `{"model", "state", "questions"}`.
- `state` may be a string or a JSON object. `instructions` may be a string or an object (use an object to pass the extracted aspect alongside the question).
- Response: `{"model": "jev-x.y.z", "answers": {question_id: {...}}, "usage": {"input_tokens", "output_tokens"}}`.
- All questions in one request are answered in parallel; send one request per review.
- Score `criteria` is an ordered list of ≤ 10 level descriptions, worst first. Each level is judged on its own, so describe situations, not degrees ("mentions every product attribute the reviewer comments on", not "very complete").
- **Pin a concrete model version** in config (not `jev-latest`) so results are reproducible; store the returned `model` string in every row.

**jev_client.py**
- `JudgeClient` protocol: `ask(state: dict | str, questions: dict[str, Question]) -> JevResponse`.
- `JevResponse` / answer dataclasses: `NoulAnswer(noul)`, `ChoiceAnswer(choice, probabilities, confidence)`, `ScoreAnswer(score, probabilities, confidence, n_levels)`, plus `model`, `input_tokens`.
- `TypeSafeJevClient` wraps the SDK (lazy creation, retries with backoff on rate-limit/5xx, timeout, rate limiter). Maps an oversized-input error to `JevTooLargeError`.
- `FakeJevClient` returns responses from recorded JSON files in `tests/fixtures/jev/`, or from a callable for property-style tests. No network.

**judge.py** — per review with `status=ok`:
1. **Build the state**: `{"review": {"title", "text", "rating"}, "extraction": {"overall_sentiment", "aspects": [...]}}`. Truncate review text to `max_state_chars` and set `truncated=True` if cut.
2. **Build questions in code** from `judge_questions_v1.yaml`:
   - per aspect *i*: `aspect_{i}_supported` (Noul, instructions object `{extracted_aspect, question}`), `aspect_{i}_polarity` (Choice over `positive|negative|neutral`)
   - per review: `completeness` (Score), `overall_sentiment` (Score, 5 levels from very negative to very positive)
   - cap at `max_aspects_per_request`; extra aspects go into a second request
3. **Combine answers into scores** (pure functions, all in [0, 1]):
   - `faithfulness` = mean over aspects of `p_supported`
   - `polarity_accuracy` = mean over aspects of P(Jev choice == extracted polarity)
   - `completeness` = `score / (n_levels − 1)`
   - `format_compliance` = fraction of aspects that are well formed, a **deterministic code check, not a Jev question**: the evidence phrase appears verbatim in the review (ignoring case and punctuation) and the label has 2–4 words. (A Jev Noul for this disagreed with the code check on real data: 24 of 26 evidence phrases were verbatim, yet Jev scored 0.11–0.41.) `None` when a review has no aspects.
   - `sentiment_agreement` = `1 − |jev_sentiment_norm − (extractor_sentiment + 1) / 2|`
   - `overall` = weighted mean with weights from config (weights renormalised over available components)
   - `confidence` = min of the Choice/Score confidences and of `2·|p − 0.5|` for every Noul
4. **Verdict**: `uncertain` if `confidence < min_confidence`; otherwise `pass` if `overall ≥ pass_threshold`, else `fail`. Downstream treats `uncertain` as `fail` unless `graph.accept_uncertain: true`.
5. Reviews with no extracted aspects get `judge_status=no_aspects` and only the review-level questions.
6. Write `judgements.parquet` and `aspect_judgements.parquet`. The KG uses `aspect_judgements.p_supported ≥ kg.min_aspect_support` to decide which aspect triples to keep.

Supports `sample_rate` (judge a seeded subset during development) and resumes from cache.

**calibration.py** — Jev's probabilities are only useful once checked against your own labels.
- `export`: stratified sample (by verdict, rating and confidence bucket; default 150 reviews) to CSV with one row per aspect, columns for human labels (`supported`, `polarity`) and per-review `completeness` (0…n−1) and `format_ok` (yes/no).
- `score`: read the labelled CSV back and report accuracy, Cohen's κ, **Brier score** and **expected calibration error** per question type, and a reliability table by probability bucket.
- `thresholds`: sweep `pass_threshold`, `min_confidence` and `min_aspect_support` on the labelled set and report precision/recall for each, so thresholds come from data, not defaults. Write the result to `reports/judge_calibration.md`.

Required CLI safeguards for any command that calls an external model:
- `--dry-run` prints row count, estimated input tokens and estimated cost (per-token prices in config), then exits.
- `--limit N` processes at most N uncached rows.
- Real API calls require the `.env` keys; tests must never make them.

Config sketch:

```yaml
# configs/enrichment.yaml
extractor:
  deployment: ${EXTRACTOR_DEPLOYMENT}
  prompt_version: v1
  temperature: 0.0
  max_tokens: 600
  batch_size: 20
judge:
  provider: typesafe
  model: jev-1.13.0            # pin a version; never jev-latest for reported results
  questions_version: v1
  max_state_chars: 6000
  max_aspects_per_request: 15
  weights: {faithfulness: 0.35, polarity_accuracy: 0.25, completeness: 0.15, sentiment_agreement: 0.15, format_compliance: 0.10}
  pass_threshold: 0.7          # replace with calibrated value
  min_confidence: 0.5          # replace with calibrated value
  sample_rate: 1.0
  price_per_million_input_tokens: 0.042
retry: {max_attempts: 5, backoff_seconds: 2}
rate_limit: {requests_per_minute: 300}
```

### graphs/
- **base.py** — the graph contract:

```python
class GraphData(Protocol):
    kind: str                       # "bipartite", "knowledge", later "hypergraph"
    num_users: int
    num_items: int
    user_index: dict[str, int]
    item_index: dict[str, int]

@dataclass
class BipartiteGraph:               # kind = "bipartite"
    edge_index: Tensor              # [2, E], users 0..U-1, items U..U+I-1, both directions
    edge_weight: Tensor | None

@dataclass
class KnowledgeGraph:               # kind = "knowledge"
    triples: pd.DataFrame           # head, relation, tail (string ids)
    entity_index: dict[str, int]
    relation_index: dict[str, int]
    interaction_graph: BipartiteGraph   # the train user–item graph the GNN runs on

class GraphBuilder(ABC):
    kind: str
    def build(self, train: pd.DataFrame, cfg) -> GraphData: ...
```

- **registry.py** — `@register_builder("bipartite")`, `build_graph(name, train, cfg)`. A future `hyper.py` registers `"hypergraph"` and returns a `HyperGraph` with an incidence matrix; nothing else changes.
- **bipartite.py** — one builder with `weighting ∈ {none, rating, rating_sentiment}`:
  - `none` → all weights 1 (used by LightGCN, MF needs no graph).
  - `rating` → `rating / 5`.
  - `rating_sentiment` → `α · rating/5 + (1−α) · (sentiment+1)/2`, with `α` in config.
  - `min_judge_score`: when set, sentiment from extractions below the threshold (or failed/missing) is replaced by the rating-only weight. The edge is kept — the judge filters the *enrichment*, not the interaction.
- **knowledge.py** — builds triples **from the train split only**:
  - `user —likes→ item` (rating ≥ 4), `user —dislikes→ item` (rating ≤ 2)
  - `item —has_positive_aspect→ aspect`, `item —has_negative_aspect→ aspect`. A mention in a **train** review counts only when (a) Jev's `p_supported ≥ graph.min_aspect_support` and (b) the review's verdict is `pass` (or `uncertain` with `graph.accept_uncertain`). Its polarity is the extractor's, unless Jev's polarity choice differs with confidence ≥ `graph.min_polarity_confidence`. The two relations are emitted independently once the number of distinct reviews reaches `graph.min_aspect_count` (so a mixed aspect gets both). `graph.judged_only: false` keeps every extracted aspect (the unjudged ablation).
  - `user —cares_about→ aspect` (aspects the user mentions in train reviews, same filters, at least `graph.min_user_aspect_count` reviews)
  - No category relation for now (All_Beauty has only two distinct categories, so it carries no information); adding one later is a new relation in this builder.
  Relation set is configurable; namespace entity ids (`u:`, `i:`, `a:`). Entity ids are ordered users, then items (matching `user_index` / `item_index`), then aspects, so KG embeddings align with the GNN's node order.
- **kg_embeddings.py** — train PyKEEN embeddings (model, dim, epochs from config; default TransE or RotatE). Save to `data/{domain}/embeddings/{config_hash}.pt`; reuse if present.

### models/
- **base.py** — `Recommender(nn.Module, ABC)`:
  - `requires_graph: str | None` (`None`, `"bipartite"`, `"knowledge"`)
  - `forward(graph) -> (user_emb, item_emb)`
  - `score(user_idx, item_idx) -> Tensor` and `score_all(user_idx) -> Tensor[n_users, n_items]`
  - `save(path)`, `load(path)`
- **registry.py** — `@register_model("gat")`, `build_model(cfg, graph)`. Raises a clear error if `graph.kind` doesn't match `requires_graph`.
- **mf.py** — BPR matrix factorisation. Baseline, no graph.
- **lightgcn.py** — standard LightGCN on the unweighted bipartite graph. Baseline.
- **gat.py** — GAT on the bipartite graph; learned ID embeddings; layers, heads, dropout from config; edge weights optionally passed as `edge_attr`. Run with `weighting=none` vs `rating_sentiment` to isolate the effect of enrichment.
- **kg_gat.py** — same GAT architecture, but user/item embeddings initialised from KG embeddings (`freeze: true|false`). Should reuse `gat.py` layers rather than duplicate them.

### training/
- **trainer.py** — one loop for every model: epochs, mini-batches of (user, pos, neg), validation every N epochs with the evaluator, early stopping on a configured metric (default `ndcg@10` on val), gradient clipping, checkpoint best model, resume from checkpoint. Logs per-epoch loss and val metrics.
- **losses.py** — `bpr` (default) and `bce`, same signature.
- **sampling.py** — uniform negative sampling, seeded, excluding the user's train positives.

### evaluation/
- **metrics.py** — pure functions over ranked lists: Recall@K, NDCG@K, HitRate@K, Precision@K, MRR. Unit-tested on hand-computed cases.
- **evaluator.py** — full ranking over all items in the domain, excluding the user's train (and for test, val) positives. Uses `model.score_all`. Returns a dict of metrics for each K in config. Every model is evaluated by this code path only.

### utils/
- **seed.py** — seed random, numpy, torch, CUDA, PyG; set deterministic flags.
- **logging.py** — standard `logging` setup; per-run log file. No `print` debugging.
- **io.py** — parquet/JSON/checkpoint helpers; create run directory with resolved config and git hash.
- **paths.py** — all paths resolved here from `configs/base.yaml` + domain name. No hardcoded paths elsewhere.
- **hashing.py** — stable hashes for configs, files and prompts.

### cli.py
Typer app. Each command loads config and calls the library; no logic of its own.

```
recsys ingest       --domain all_beauty
recsys filter       --domain all_beauty
recsys enrich       --domain all_beauty [--dry-run] [--limit N]
recsys judge        --domain all_beauty [--dry-run] [--limit N]
recsys calibrate    --domain all_beauty export|score <csv>
recsys interactions --domain all_beauty
recsys split        --domain all_beauty
recsys kg-embed     --exp kg_gat
recsys train        --exp <name> [--seed N] [--set key=value ...]
recsys evaluate     --run runs/<run_id>
```

`scripts/make_report.py` reads `runs/*/metrics.json`, groups by experiment, reports mean ± std over seeds, and writes Markdown/CSV tables and plots to `reports/`.

---

## 7. Planned experiments

One YAML per row in `configs/experiments/`. Run each with seeds 0, 1, 2.

| Experiment | Graph | Weighting | Judge filter | Model |
|---|---|---|---|---|
| `mf` | — | — | — | mf |
| `lightgcn` | bipartite | none | — | lightgcn |
| `gat_plain` | bipartite | none | — | gat |
| `gat_rating` | bipartite | rating | — | gat |
| `gat_enriched` | bipartite | rating_sentiment | off | gat |
| `gat_enriched_judged` | bipartite | rating_sentiment | 0.7 | gat |
| `kg_gat` | knowledge | — | aspects from judged-pass only | kg_gat |
| `kg_gat_unjudged` | knowledge | — | off (every extracted aspect) | kg_gat |

The comparison `gat_enriched` vs `gat_enriched_judged` is the direct measure of the judge's value for edge weights, and `kg_gat` vs `kg_gat_unjudged` is the same measure for the knowledge graph.

---

## 8. Build order

Work in phases. A phase is done only when **every test listed for its modules in section 9 exists and passes**, coverage targets are met, and `make lint` is clean. Write the tests together with the module, not afterwards. After each phase, summarise what was built and stop for review.

**Phase 0 — Skeleton.** `pyproject.toml`, Makefile, `.gitignore`, `.env.example`, `config.py`, `utils/*`, empty registries, `tests/conftest.py` with all shared fixtures from section 9. ✅ `pip install -e .[dev]` works; `test_config.py` and `tests/utils/*` pass.

**Phase 1 — Data + evaluation + MF.** `schemas`, `ingest`, `filter`, `splits`, `metrics`, `evaluator`, `losses`, `sampling`, `trainer`, `models/base`, `models/registry`, `mf`. ✅ Tests pass; `recsys train --exp mf` produces `runs/<id>/metrics.json` on All_Beauty. This gives a working end-to-end baseline before any LLM cost.

**Phase 2 — Extraction.** `llm_client`, `prompts`, `cache`, `extractor`. ✅ Tests pass; re-running `enrich` on real data makes zero API calls when everything is cached; `--dry-run` reports cost.

**Phase 3 — Jev judge.** `jev_client`, `judge`, `calibration`. Record 5–10 real Jev responses (one per question type plus an error case) into `tests/fixtures/jev/` with an `integration` test, then build the rest against `FakeJevClient`. ✅ Tests pass; `judge --limit 200` runs on real data; calibration report generated once labels exist.

**Phase 4 — Weighted graph + GNNs.** `interactions`, `graphs/base`, `graphs/registry`, `graphs/bipartite`, `lightgcn`, `gat`. ✅ Tests pass, including the end-to-end smoke test.

**Phase 5 — Knowledge graph.** `graphs/knowledge`, `kg_embeddings`, `kg_gat`. ✅ Tests pass.

**Phase 6 — Report + CLI.** `make_report.py`, `cli.py` tests, all experiment YAMLs, README with reproduction steps.

---

## 9. Testing

### Layout and tooling
- `tests/` mirrors `src/amazon_recsys/`: `tests/data/test_ingest.py`, `tests/enrichment/test_judge.py`, etc. One test file per module.
- `pytest`, `pytest-cov`, `hypothesis` (property tests for metrics and score combination), `pytest-mock`. Pin them in the `dev` extra.
- Markers (registered in `pyproject.toml`):
  - `slow` — anything over ~5 s (e.g. multi-epoch training). Excluded from `make test`, included in `make test-all`.
  - `integration` — calls a real API (Azure or Jev). Skipped unless `RUN_INTEGRATION=1` **and** the keys are set. Never run in CI.
- Default `make test` = `pytest -m "not slow and not integration" --cov=amazon_recsys`, must finish in under 60 s on CPU.
- Coverage targets: ≥ 90 % for `evaluation/`, `data/splits.py`, `enrichment/judge.py`, `enrichment/cache.py`; ≥ 80 % for the package overall.
- Tests are deterministic: seed via the `seed` fixture, no wall-clock or network dependence, all I/O under `tmp_path`.

### Shared fixtures (`tests/conftest.py`)
| Fixture | Contents |
|---|---|
| `seed` | Calls `utils.seed.set_seed(0)` (autouse) |
| `tmp_paths` | Points `utils.paths` at `tmp_path` so no test touches `data/` or `runs/` |
| `tiny_cfg` | A fully resolved config for the `all_beauty` domain with tiny sizes (dim 8, 2 epochs, batch 16) |
| `raw_jsonl_gz` | Writes small reviews + meta `.jsonl.gz` files (including 2 malformed lines, 1 duplicate, 1 missing field) |
| `tiny_reviews` | ~60 users × ~40 items review frame with timestamps, ratings 1–5 and short texts; some users with < 5 interactions so filtering has work to do |
| `tiny_interactions` / `tiny_splits` | Output of the real pipeline on `tiny_reviews` |
| `hand_split` | A 3-user, 6-item split written by hand, used where exact answers matter |
| `fake_llm` | `FakeLLMClient` with canned extractor responses: valid, valid-with-extra-text, malformed JSON, empty aspects, API error |
| `fake_jev` | `FakeJevClient` loading `tests/fixtures/jev/*.json` (Noul, Choice, Score answers, too-large error, rate-limit error) |
| `tiny_enrichment` | Enrichment + aspects + judgements + aspect_judgements tables for `tiny_reviews`, built with the fakes |
| `tiny_bipartite`, `tiny_kg` | Graphs built from `tiny_splits.train` |
| `tiny_models` | Parametrised over every registered model name |

`tests/fixtures/` holds: `extractor/*.json` (saved real and hand-made extractor outputs, including samples migrated from the legacy `Output/` folder), `jev/*.json` (recorded Jev responses), `calibration_labels.csv` (a small hand-labelled file).

### Required tests per module

**`test_config.py`**
- Loads every file in `configs/experiments/` without error.
- Unknown key → validation error; wrong type → validation error.
- `--set a.b=c` overrides are applied with type coercion; unknown override path → error.
- Env var interpolation (`${TYPESAFE_API_KEY}`) resolves, and a missing var only errors when that section is used.
- Resolved config round-trips through YAML unchanged.

**`utils/`**
- `test_seed.py`: two runs with the same seed produce identical numpy/torch/random draws; different seeds differ.
- `test_hashing.py`: hash is stable across dict key order and runs; changes when any value changes; file hash matches a known value.
- `test_paths.py`: all paths are namespaced by domain; no path resolves outside the configured root.
- `test_io.py`: parquet/JSON round-trip preserves dtypes; run directory contains resolved config and git hash (git mocked when unavailable).

**`data/test_schemas.py`**
- Each schema accepts a valid frame and rejects: missing column, wrong dtype, out-of-range value (rating 6, sentiment 1.5, probability −0.1), null in a non-nullable column.

**`data/test_ingest.py`**
- Row counts match input minus malformed lines; malformed count recorded in the sidecar.
- Timestamps normalised to UTC ms; `review_id` deterministic and unique; duplicates dropped (first kept) and counted.
- Column mapping comes from the domain config (rename a column in the config fixture and ingestion still works).

**`data/test_filter.py`**
- After k-core, every user and item meets the minimum (property test with `hypothesis` on random small frames).
- k-core is iterated to a fixed point (case where removing items drops a user below the threshold).
- `max_users` sampling is seeded and reproducible; drop summary counts add up.
- Sampling happens after the first k-core and is followed by a second k-core (a case where sampling leaves an item below the minimum); `max_users: null` keeps every user that survives the first k-core; a cap larger than the core size is a no-op.

**`data/test_interactions.py`**
- Left join keeps reviews with no enrichment (nulls, not dropped).
- Row count equals filtered review count; schema validates.

**`data/test_splits.py`**
- Leave-one-out by time: each user's test item is their latest, val the second latest (checked on `hand_split`).
- No (user, item) pair appears in more than one split; every val/test user appears in train.
- Same seed → identical split; ties in timestamp broken deterministically.
- Users with too few interactions are handled per config; manifest counts match the files.

**`enrichment/test_llm_client.py`**
- Client is not created at import time (import the module with no env vars set).
- Retries on transient errors with backoff (time mocked), gives up after `max_attempts`, and surfaces the final error.
- `FakeLLMClient` returns the canned response for a given review_id.

**`enrichment/test_prompts.py`**
- Template renders with all variables; missing variable → error.
- Hash changes when template text changes and is stable otherwise.
- Judge question YAML loads into valid Noul/Choice/Score objects; Score levels ≤ 10 and ≥ 2; Choice options unique.

**`enrichment/test_cache.py`**
- Put/get round-trip; miss on any key component change (model, prompt hash, input hash).
- Judge entry invalidated when the extraction hash changes.
- Survives reopening the SQLite file; concurrent writes from two threads do not corrupt it.

**`enrichment/test_extractor.py`**
- Parses every file in `tests/fixtures/extractor/`; malformed output → `status=parse_error`, raw text kept in the cache.
- Aspect normalisation (case, lemma, synonym map); `aspect_raw` preserved.
- Second run makes zero client calls (call count on `fake_llm`); `--limit` respected.
- Dry-run cost estimate equals tokens × configured price.

**`enrichment/test_jev_client.py`**
- Request builder produces the documented payload shape for each question type (snapshot test against `tests/fixtures/jev/request_*.json`).
- Response parser maps recorded JSON to `NoulAnswer` / `ChoiceAnswer` / `ScoreAnswer`; Score probability keys parsed as ints; probabilities sum to 1 (± 1e-6).
- Oversized-input error → `JevTooLargeError`; rate-limit → retried; model version captured from the response.
- `integration`: one real call per question type succeeds and matches the recorded schema.

**`enrichment/test_judge.py`**
- State builder: truncation at `max_state_chars` sets `truncated=True`; aspect objects embedded correctly.
- Question builder: one Noul + one Choice per aspect, two review-level Scores; splits into multiple requests above `max_aspects_per_request`.
- Score combination (pure functions), with hand-computed expected values:
  - `faithfulness` = mean of `p_supported`; `polarity_accuracy` uses the probability of the *extracted* polarity
  - `completeness` normalisation for 3-, 4- and 5-level scales
  - `sentiment_agreement` at the extremes (perfect agreement → 1, opposite → 0)
  - weights renormalised when a component is missing (no-aspects review)
  - `confidence` is the minimum across answers; Noul confidence `2·|p − 0.5|`
- Property tests (`hypothesis`): every component and `overall` stay in [0, 1] for any valid answer set.
- Verdict boundaries: exactly at `pass_threshold` → pass; just below → fail; low confidence → uncertain regardless of score.
- Error paths: API error → `judge_status=api_error`, row kept; too large → `too_large`; no aspects → `no_aspects`.
- `sample_rate` is seeded and reproducible; cached reviews make no calls.
- Output tables validate against their schemas.

**`enrichment/test_calibration.py`**
- Export is stratified (every verdict/confidence bucket represented) and reproducible.
- `score` on `calibration_labels.csv` returns the hand-computed accuracy, κ, Brier and ECE.
- Threshold sweep returns monotone recall as the threshold falls; output file written.

**`graphs/test_registry.py`**
- Registered names resolve; unknown name → clear error listing available builders; duplicate registration → error.

**`graphs/test_bipartite.py`**
- Node count = users + items; every train interaction appears once in each direction; no val/test edges.
- `weighting=none` → all ones; `rating` → rating/5; `rating_sentiment` → hand-computed values for given α.
- All weights in [0, 1]; index maps are bijective.
- Judge filter: review below `min_judge_score`, failed, uncertain or missing → rating-only weight; edge still present.

**`graphs/test_knowledge.py`**
- Relation counts on `hand_split` match hand-computed triples.
- No val/test (user, item) pair appears as `likes`/`dislikes`.
- Aspect triples only for `p_supported ≥ min_aspect_support`; polarity override rule applied only above `min_polarity_confidence`.
- Entity ids namespaced; relation set restricted by config.

**`graphs/test_kg_embeddings.py`**
- Tiny PyKEEN run (1 epoch, dim 8) produces one embedding per entity with the configured dimension.
- Second call with the same config loads from cache (training function not called); config change → new file.

**`models/test_registry.py`**
- Every model name in `configs/models/` is registered; building a model with the wrong graph kind raises a clear error.

**`models/test_models.py`** (parametrised over `tiny_models`)
- `forward` returns user/item embeddings of the right shapes; `score` and `score_all` agree on the same pairs.
- Gradients flow to all trainable parameters after one backward pass.
- `save` → `load` reproduces identical scores.
- Model-specific: LightGCN output equals the mean of layer embeddings (check on a 2-node graph); GAT with `weighting=none` vs weights gives different outputs; `kg_gat` initial embeddings equal the KG embeddings, and stay fixed when `freeze=true`.

**`training/test_losses.py`**
- BPR and BCE match hand-computed values; BPR decreases when positive scores rise.

**`training/test_sampling.py`**
- Negatives never include the user's train positives (property test); seeded reproducibility; correct shape for `n_neg`.

**`training/test_trainer.py`**
- Loss decreases over a few epochs on `tiny_splits` for MF.
- Early stopping stops after `patience` epochs without improvement (mocked evaluator returning a fixed sequence) and restores the best checkpoint.
- Resume from checkpoint continues at the right epoch with identical optimizer state.
- Run directory contains config, metrics and checkpoint.

**`evaluation/test_metrics.py`**
- Recall@K, NDCG@K, HitRate@K, Precision@K, MRR against hand-computed rankings.
- Edge cases: no relevant items, K larger than the list, ties, all-relevant list.
- Property tests: all metrics in [0, 1]; NDCG = 1 for a perfect ranking.

**`evaluation/test_evaluator.py`**
- Train positives (and val positives when scoring test) are excluded from candidates.
- An oracle model that scores the test item highest gets HitRate@1 = 1; a reversed oracle gets 0.
- Results identical for the same model and seed; metrics returned for every configured K.

**`test_cli.py`** (Typer `CliRunner`)
- Every command's `--help` works.
- `enrich --dry-run` and `judge --dry-run` print a cost estimate and make no client calls.
- `train --exp mf --set train.epochs=1` on fixtures creates a run directory.

**`scripts/test_make_report.py`**
- Given fake `runs/*/metrics.json` for two experiments × three seeds, produces the correct mean ± std table.

**`test_smoke_e2e.py`** (`slow`)
- Full pipeline on `raw_jsonl_gz` with fakes: ingest → filter → enrich → judge → interactions → split → each graph → each model for 1 epoch → evaluate. Asserts every stage output validates and every model produces metrics.

---

## 10. Coding rules

- Python 3.11, type hints everywhere, `ruff` + `ruff format`, line length 100.
- Small pure functions; side effects (I/O, API calls) at the edges.
- Validate every parquet against its schema on write and read.
- Seed everything through `utils/seed.py`; no unseeded randomness.
- No hardcoded paths, domain names or column names outside `configs/` and `paths.py`.
- No LLM or Jev calls, network access or GPU requirement in tests, except tests marked `integration`. Use `FakeLLMClient`, `FakeJevClient` and CPU fixtures.
- Every new function with logic gets a test in the mirrored test file in the same change. Bug fixes start with a failing test.
- Jev's `probabilities` and `confidence` are stored as returned; never round them or overwrite them with thresholded booleans. Thresholds are applied downstream, from config.
- Never modify `data/raw/`. Never commit data, checkpoints, embeddings or `.env`.
- Don't add dependencies without listing them in `pyproject.toml` with a pinned version.
- Graphs and KG triples are built from the **train split only**. Enrichment may be computed for all reviews, but val/test reviews must never feed graph structure, edge weights or KG triples.
- When adding a new component (domain, graph, model), register it; do not add `if name == ...` branches in the trainer or evaluator.

## 11. Legacy code

Old scripts live in `legacy/`. Read them to recover logic (prompt formats, parsing, GAT details, PyKEEN setup), but never import from them. Useful references:

| New module | Legacy source |
|---|---|
| `data/ingest.py` | `SQLiteManager.jsonl_gz_to_sqlite`, `import_datasets_into_database` |
| `enrichment/extractor.py`, `prompts.py`, `llm_client.py` | `LLM_enrichment.py`, `parse_llm_response*` |
| `tests/fixtures/extractor/` | saved LLM responses in `Output/` |
| `graphs/bipartite.py` | `DatabaseToGraphConverter` in `WG_emb_model_*`, `simple_graph.py` |
| `graphs/knowledge.py`, `kg_embeddings.py` | `knowledge_graph.py`, `KnowledgeGraphGenerator` |
| `models/gat.py` | `SimpleAdvancedGAT`, `GATModel*` |
| `training/trainer.py` | `GATTrainer` |
| `evaluation/*` | `RecommendationEvaluator` classes |

Ignore all `*_CD`, `HGNN_*` and `Hyper_graph_*` files for now.

## 12. Commands

```bash
pip install -e ".[dev]"
make data                       # ingest → filter
make enrich LIMIT=200           # extractor (use --dry-run first)
make judge  LIMIT=200           # Jev judge (use --dry-run first)
make calibrate                  # export sample / score labels / sweep thresholds
make split                      # interactions → split
make train EXP=lightgcn SEED=0
make report
make test                       # fast suite: not slow, not integration
make test-all                   # includes slow (smoke e2e)
RUN_INTEGRATION=1 make test-integration   # real Azure + Jev calls; costs money
make lint
```
