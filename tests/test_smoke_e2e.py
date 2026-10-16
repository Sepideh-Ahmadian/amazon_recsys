"""Full pipeline through the CLI on a synthetic raw dataset, external services faked."""

import gzip
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from amazon_recsys import cli
from amazon_recsys.data.schemas import (
    ASPECT_JUDGEMENTS,
    ASPECTS,
    ENRICHMENT,
    INTERACTIONS,
    JUDGEMENTS,
)
from amazon_recsys.enrichment.jev_client import (
    ChoiceAnswer,
    ChoiceQuestion,
    FakeJevClient,
    JevResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
)
from amazon_recsys.enrichment.llm_client import LLMResponse
from amazon_recsys.utils.io import read_json, read_parquet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = str(ROOT / "configs")
N_USERS, N_ITEMS = 12, 6  # a complete block: every user has 6 items, every item 12 users
EXPERIMENTS = [
    "mf",
    "lightgcn",
    "gat_plain",
    "gat_rating",
    "gat_enriched",
    "gat_enriched_judged",
    "kg_gat",
    "kg_gat_unjudged",
]

pytestmark = pytest.mark.slow


class EchoLLM:
    """Valid extractor output built from the review text (evidence is a verbatim prefix)."""

    def complete(self, messages, *, review_id=None, **params):
        review = messages[-1]["content"].rsplit("Review: ", 1)[-1]
        evidence = " ".join(review.split()[:3])
        body = {
            "overall_sentiment": 0.4,
            "aspects": [
                {"aspect": "Pleasant Scent", "sentiment": 0.8, "evidence": evidence},
                {"aspect": "Sticky Feel", "sentiment": -0.3, "evidence": evidence},
            ],
        }
        return LLMResponse(json.dumps(body), "echo-llm", 10, 10, 0.0)


def jev_responder(state, questions):
    answers = {}
    for name, q in questions.items():
        if isinstance(q, NoulQuestion):
            answers[name] = NoulAnswer(0.95)
        elif isinstance(q, ChoiceQuestion):
            probs = {o: 0.05 for o in q.options}
            probs["positive"] = 0.9
            answers[name] = ChoiceAnswer("positive", probs, 0.9)
        else:
            n = len(q.levels)
            answers[name] = ScoreAnswer(0.8 * (n - 1), {i: 1 / n for i in range(n)}, 0.8)
    return JevResponse("jev-1.13.0", answers, 100)


def _write_gz(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.fixture
def project(tmp_path):
    reviews = [
        {
            "user_id": f"u{u}",
            "parent_asin": f"i{i}",
            "rating": float(1 + (u + i) % 5),
            "title": f"title {u} {i}",
            "text": f"Smells great and lasts long, set {u} item {i}.",
            "timestamp": 1_600_000_000_000 + 1000 * (u * 10 + i),
        }
        for u in range(N_USERS)
        for i in range(N_ITEMS)
    ]
    meta = [{"parent_asin": f"i{i}", "title": f"Lotion {i}", "main_category": "All Beauty"}
            for i in range(N_ITEMS)]  # fmt: skip
    _write_gz(tmp_path / "data/raw/All_Beauty.jsonl.gz", reviews)
    _write_gz(tmp_path / "data/raw/meta_All_Beauty.jsonl.gz", meta)
    shutil.copytree(ROOT / "prompts" / "all_beauty", tmp_path / "prompts" / "all_beauty")
    return tmp_path


def test_full_pipeline_through_the_cli(project, monkeypatch):
    from amazon_recsys.enrichment import jev_client, llm_client

    monkeypatch.setattr(llm_client, "make_llm_client", lambda *a, **k: EchoLLM())
    monkeypatch.setattr(
        jev_client, "TypeSafeJevClient", lambda *a, **k: FakeJevClient(jev_responder)
    )
    runner = CliRunner(mix_stderr=False)
    base = ["--configs-dir", CONFIGS, "--root", str(project)]

    def run(*args):
        result = runner.invoke(cli.app, [str(a) for a in args])
        assert result.exit_code == 0, f"{args}\n{result.stdout}\n{result.stderr}"
        return json.loads(result.stdout)

    # --- data
    ingested = run("ingest", *base)
    assert ingested["reviews"]["rows"] == N_USERS * N_ITEMS
    filtered = run("filter", *base)
    assert filtered["final"] == {"rows": N_USERS * N_ITEMS, "users": N_USERS, "items": N_ITEMS}

    # --- enrichment
    enriched = run("enrich", *base)
    assert enriched["ok"] == N_USERS * N_ITEMS and enriched["parse_error"] == 0
    again = run("enrich", *base)
    assert again["called"] == 0 and again["cached"] == N_USERS * N_ITEMS  # fully resumable

    judged = run("judge", *base)
    assert judged["ok"] == N_USERS * N_ITEMS and judged["api_error"] == 0
    assert run("judge", *base)["called"] == 0

    paths = {
        t: project / "data/all_beauty/processed" / f"{t}.parquet"
        for t in ("enrichment", "aspects", "judgements", "aspect_judgements")
    }
    ENRICHMENT.validate(read_parquet(paths["enrichment"]))
    ASPECTS.validate(read_parquet(paths["aspects"]))
    judgements = JUDGEMENTS.validate(read_parquet(paths["judgements"]))
    ASPECT_JUDGEMENTS.validate(read_parquet(paths["aspect_judgements"]))
    assert (judgements.format_compliance == 1.0).all()  # evidence is verbatim by construction
    assert set(judgements.verdict) == {"pass"}

    # --- interactions and splits carry enrichment through
    assert run("interactions", *base)["rows"] == N_USERS * N_ITEMS
    interactions = INTERACTIONS.validate(
        read_parquet(project / "data/all_beauty/processed/interactions.parquet")
    )
    assert interactions.overall_sentiment.notna().all()
    assert (interactions.judge_verdict == "pass").all()
    manifest = run("split", *base)
    assert manifest["counts"] == {"train": 48, "val": 12, "test": 12}
    assert manifest["leakage_check"]["passed"]

    # --- the knowledge graph is built from enrichment and its embeddings are cached
    fast_kg = ["--set", "graph.embedding.epochs=2"]
    kg = run("kg-embed", "--exp", "kg_gat", *fast_kg, *base)
    assert kg["embedding_shape"] == [kg["entities"], 64] and kg["triples"] > 0
    embeddings_dir = project / "data/all_beauty/embeddings"
    cached = sorted(embeddings_dir.glob("*.pt"))
    assert len(cached) == 1
    run("kg-embed", "--exp", "kg_gat", *fast_kg, *base)
    assert sorted(embeddings_dir.glob("*.pt")) == cached  # reused, not retrained
    unjudged = run("kg-embed", "--exp", "kg_gat_unjudged", *fast_kg, *base)
    assert unjudged["triples"] >= kg["triples"]  # the ablation keeps at least as many aspects

    # --- every experiment trains and evaluates
    run_dirs = {}
    for exp in EXPERIMENTS:
        extra = fast_kg if exp.startswith("kg_gat") else []
        result = run(
            "train", "--exp", exp, "--set", "train.epochs=1", "--set", "train.batch_size=16",
            *extra, *base,
        )  # fmt: skip
        assert result["epochs_run"] == 1
        assert 0.0 <= result["test"]["ndcg@10"] <= 1.0
        run_dirs[exp] = Path(result["run_dir"])
        assert (run_dirs[exp] / "metrics.json").exists()

    evaluated = run("evaluate", "--run", run_dirs["gat_enriched_judged"], "--root", str(project))
    stored = read_json(run_dirs["gat_enriched_judged"] / "metrics.json")
    assert evaluated["test"]["ndcg@10"] == pytest.approx(stored["test"]["ndcg@10"])

    # --- calibration sheet can be exported from the judged data
    sheet = run("calibrate", "export", "--n", "5", *base)
    assert sheet["reviews"] == 5 and Path(sheet["path"]).exists()
    exported = pd.read_csv(sheet["path"])
    assert not {"p_supported", "verdict"} & set(exported.columns)  # blind labelling


def test_weighted_experiments_see_different_graphs(project, monkeypatch):
    """The enrichment actually reaches the graph: judged vs unjudged weights differ."""
    import amazon_recsys.graphs  # noqa: F401
    from amazon_recsys.config import load_config
    from amazon_recsys.data.interactions import build_index
    from amazon_recsys.graphs.registry import build_graph

    train = pd.DataFrame(
        {
            "domain": "d", "user_id": ["u1", "u1", "u2"], "item_id": ["i1", "i2", "i1"],
            "rating": [4.0, 4.0, 4.0], "timestamp": [1, 2, 3], "review_id": ["a", "b", "c"],
            "overall_sentiment": [1.0, 1.0, 1.0],
            "judge_overall": [0.9, 0.2, 0.9], "judge_verdict": ["pass", "fail", "pass"],
        }
    )  # fmt: skip
    users, items = build_index(train)
    plain = load_config("gat_enriched", CONFIGS)
    judged = load_config("gat_enriched_judged", CONFIGS)
    w_plain = build_graph("bipartite", train, plain, user_index=users, item_index=items).edge_weight
    w_judged = build_graph(
        "bipartite", train, judged, user_index=users, item_index=items
    ).edge_weight
    assert w_plain[:3].tolist() == pytest.approx([0.9, 0.9, 0.9])  # sentiment everywhere
    assert w_judged[:3].tolist() == pytest.approx([0.9, 0.8, 0.9])  # low-judge edge falls back
