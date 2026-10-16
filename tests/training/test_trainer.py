import pytest
import torch

import amazon_recsys.models  # noqa: F401  (registers models)
from amazon_recsys.config import SplitCfg
from amazon_recsys.data.interactions import build_index
from amazon_recsys.data.schemas import INTERACTIONS
from amazon_recsys.data.splits import make_splits
from amazon_recsys.evaluation.evaluator import Evaluator
from amazon_recsys.models.registry import build_model
from amazon_recsys.training.trainer import GraphDims, Trainer, run_experiment
from amazon_recsys.utils.io import read_json, write_parquet
from amazon_recsys.utils.seed import set_seed

N_USERS, N_ITEMS = 8, 10


def _splits(interactions_factory):
    """Each user interacts with the 5 items sharing their parity, in increasing time."""
    rows = []
    for u in range(N_USERS):
        items = [i for i in range(N_ITEMS) if i % 2 == u % 2]
        rows += [(f"u{u}", f"i{i}", 100 + n) for n, i in enumerate(items)]
    return make_splits(interactions_factory(rows), SplitCfg(min_user_interactions=3))[0]


def _cfg(tiny_cfg, **train):
    model = tiny_cfg.model.model_copy(update={"name": "mf", "dim": 8})
    return tiny_cfg.model_copy(
        update={"model": model, "train": tiny_cfg.train.model_copy(update=train)}
    )


def _build(cfg, splits):
    user_index, item_index = build_index(*splits.values())
    set_seed(cfg.seed)
    model = build_model(cfg, GraphDims(len(user_index), len(item_index)))
    evaluator = Evaluator(splits, user_index, item_index, cfg.eval.ks)
    return model, evaluator, user_index, item_index


def _trainer(cfg, splits, run_dir=None, evaluator=None):
    model, ev, ui, ii = _build(cfg, splits)
    return Trainer(model, cfg, splits["train"], ui, ii, evaluator or ev, run_dir)


def test_loss_decreases_for_mf(tiny_cfg, interactions_factory):
    cfg = _cfg(tiny_cfg, epochs=15, lr=0.05, patience=100, batch_size=8)
    result = _trainer(cfg, _splits(interactions_factory)).fit()
    losses = [h["loss"] for h in result["history"]]
    assert len(losses) == 15
    assert losses[-1] < losses[0]


class ScriptedEvaluator:
    """Returns a fixed validation sequence and records the model weights at every call."""

    def __init__(self, values, metric="ndcg@10"):
        self.values, self.metric, self.snapshots, self.calls = values, metric, [], 0

    def evaluate(self, model, split):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        self.snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
        return {self.metric: value, "n_users": 1.0}


def test_early_stopping_and_best_checkpoint_restored(tiny_cfg, interactions_factory):
    cfg = _cfg(tiny_cfg, epochs=10, patience=2, lr=0.05, eval_every=1)
    scripted = ScriptedEvaluator([0.1, 0.3, 0.2, 0.2, 0.2, 0.2])
    trainer = _trainer(cfg, _splits(interactions_factory), evaluator=scripted)
    result = trainer.fit()

    assert result["best_epoch"] == 2
    assert result["stopped_early"] is True
    assert result["epochs_run"] == 4  # 2 epochs after the best without improvement
    best = scripted.snapshots[1]  # model weights when epoch 2 was validated
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(tensor, best[name])


def test_resume_matches_uninterrupted_run(tiny_cfg, interactions_factory, tmp_path):
    splits = _splits(interactions_factory)
    full_cfg = _cfg(tiny_cfg, epochs=5, lr=0.05, patience=100, batch_size=8)
    full = _trainer(full_cfg, splits, run_dir=tmp_path / "full")
    full_result = full.fit()

    part_cfg = _cfg(tiny_cfg, epochs=3, lr=0.05, patience=100, batch_size=8)
    _trainer(part_cfg, splits, run_dir=tmp_path / "part").fit()

    resumed = _trainer(full_cfg, splits, run_dir=tmp_path / "part")
    resumed_result = resumed.fit(resume=True)

    assert [h["epoch"] for h in resumed_result["history"]] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [h["loss"] for h in resumed_result["history"]] == [
        h["loss"] for h in full_result["history"]
    ]
    for (_, a), (_, b) in zip(
        full.model.state_dict().items(), resumed.model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b)
    opt_a, opt_b = full.optimizer.state_dict(), resumed.optimizer.state_dict()
    for key, state in opt_a["state"].items():
        for name, value in state.items():
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(opt_b["state"][key][name]))


def test_run_experiment_creates_run_directory(tiny_cfg, tmp_paths, interactions_factory):
    splits = _splits(interactions_factory)
    for name, frame in splits.items():
        write_parquet(frame, tmp_paths.split(name), INTERACTIONS)
    cfg = _cfg(tiny_cfg, epochs=2, batch_size=8)

    result = run_experiment(cfg, tmp_paths)

    run_dir = tmp_paths.runs_root / result["run_dir"].split("/")[-1]
    for name in ("config.yaml", "git_hash.txt", "metrics.json", "checkpoint.pt", "log.txt"):
        assert (run_dir / name).exists(), name
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["experiment"] == cfg.name and metrics["seed"] == 0
    assert "ndcg@10" in metrics["test"] and "ndcg@10" in metrics["val"]
    assert len(metrics["history"]) == 2


@pytest.mark.parametrize("model_name", ["lightgcn", "gat"])
def test_graph_models_train_end_to_end_through_run_experiment(
    tiny_cfg, tmp_paths, interactions_factory, model_name
):
    splits = _splits(interactions_factory)
    for name, frame in splits.items():
        write_parquet(frame, tmp_paths.split(name), INTERACTIONS)
    model_cfg = tiny_cfg.model.model_copy(update={"name": model_name, "dim": 8, "heads": 2})
    cfg = tiny_cfg.model_copy(
        update={
            "model": model_cfg,
            "train": tiny_cfg.train.model_copy(update={"epochs": 2, "batch_size": 8}),
        }
    )
    result = run_experiment(cfg, tmp_paths)
    assert result["epochs_run"] == 2
    assert 0.0 <= result["test"]["ndcg@10"] <= 1.0
    assert result["history"][0]["loss"] > 0


def test_kg_gat_trains_end_to_end_and_caches_kg_embeddings(
    tiny_cfg, tmp_paths, interactions_factory
):
    splits = _splits(interactions_factory)
    for name, frame in splits.items():
        write_parquet(frame, tmp_paths.split(name), INTERACTIONS)
    graph_cfg = tiny_cfg.graph.model_copy(update={"name": "knowledge"})
    model_cfg = tiny_cfg.model.model_copy(update={"name": "kg_gat", "dim": 8, "heads": 2})
    cfg = tiny_cfg.model_copy(
        update={
            "graph": graph_cfg,
            "model": model_cfg,
            "train": tiny_cfg.train.model_copy(update={"epochs": 2, "batch_size": 8}),
        }
    )
    result = run_experiment(cfg, tmp_paths)
    assert result["epochs_run"] == 2 and 0.0 <= result["test"]["ndcg@10"] <= 1.0
    assert len(list(tmp_paths.embeddings_dir.glob("*.pt"))) == 1
    run_experiment(cfg, tmp_paths)  # second run reuses the cached embeddings
    assert len(list(tmp_paths.embeddings_dir.glob("*.pt"))) == 1


def test_graph_model_without_a_graph_in_the_experiment_is_a_clear_error(
    tiny_cfg, tmp_paths, interactions_factory
):
    splits = _splits(interactions_factory)
    for name, frame in splits.items():
        write_parquet(frame, tmp_paths.split(name), INTERACTIONS)
    model_cfg = tiny_cfg.model.model_copy(update={"name": "lightgcn"})
    cfg = tiny_cfg.model_copy(update={"model": model_cfg, "graph": None})
    with pytest.raises(ValueError, match="no `graph`"):
        run_experiment(cfg, tmp_paths)
