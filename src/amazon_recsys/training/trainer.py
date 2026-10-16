"""One training loop for every model, plus `run_experiment` to wire a whole run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import amazon_recsys.graphs  # noqa: F401  (registers graph builders)
import amazon_recsys.models  # noqa: F401  (registers models)
from amazon_recsys.config import Config
from amazon_recsys.data.interactions import build_index
from amazon_recsys.data.schemas import INTERACTIONS
from amazon_recsys.evaluation.evaluator import Evaluator
from amazon_recsys.graphs.registry import build_graph
from amazon_recsys.models.registry import build_model, get_model_class
from amazon_recsys.training.losses import get_loss
from amazon_recsys.training.sampling import UniformNegativeSampler, epoch_batches
from amazon_recsys.utils.io import (
    load_checkpoint,
    make_run_dir,
    make_run_id,
    read_parquet,
    save_checkpoint,
    write_json,
)
from amazon_recsys.utils.logging import setup_logging
from amazon_recsys.utils.paths import Paths
from amazon_recsys.utils.seed import set_seed

CHECKPOINT = "checkpoint.pt"
BEST_MODEL = "best_model.pt"
METRICS = "metrics.json"


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Config,
        train: pd.DataFrame,
        user_index: dict[str, int],
        item_index: dict[str, int],
        evaluator: Any,
        run_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.evaluator = evaluator
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.log = logger or logging.getLogger("amazon_recsys.trainer")

        self.users = train["user_id"].map(user_index).to_numpy(dtype=np.int64)
        self.items = train["item_id"].map(item_index).to_numpy(dtype=np.int64)
        self.sampler = UniformNegativeSampler(
            self.users, self.items, len(item_index), seed=cfg.seed
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.loss_fn = get_loss(cfg.train.loss)

        self.epoch = 0
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.history: list[dict[str, float]] = []
        self.stopped_early = False

    # --- one epoch ----------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        cfg = self.cfg.train
        self.model.train()
        total, count = 0.0, 0
        for u, pos, neg in epoch_batches(
            self.users,
            self.items,
            self.sampler,
            cfg.batch_size,
            cfg.n_neg,
            seed=self.cfg.seed + epoch,
        ):
            user_emb, item_emb = self.model()
            ue = user_emb[u]
            pos_scores = (ue * item_emb[pos]).sum(-1)
            neg_scores = (ue.unsqueeze(1) * item_emb[neg]).sum(-1)
            loss = self.loss_fn(pos_scores, neg_scores)
            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            total += loss.item() * len(u)
            count += len(u)
        return total / max(count, 1)

    # --- checkpointing ------------------------------------------------------------------
    def _save_checkpoint(self) -> None:
        if self.run_dir is None:
            return
        save_checkpoint(
            {
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "best_state": self.best_state,
                "history": self.history,
                "stopped_early": self.stopped_early,
                "sampler_rng": self.sampler._rng.bit_generator.state,
            },
            self.run_dir / CHECKPOINT,
        )

    def _resume(self) -> bool:
        if self.run_dir is None or not (self.run_dir / CHECKPOINT).exists():
            return False
        ckpt = load_checkpoint(self.run_dir / CHECKPOINT)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"]
        self.best_metric = ckpt["best_metric"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.history = ckpt["history"]
        self.stopped_early = ckpt["stopped_early"]
        self.sampler._rng.bit_generator.state = ckpt["sampler_rng"]
        self.log.info("Resumed from %s at epoch %d", self.run_dir / CHECKPOINT, self.epoch)
        return True

    # --- main loop ----------------------------------------------------------------------
    def fit(self, resume: bool = False) -> dict[str, Any]:
        cfg = self.cfg.train
        if resume:
            self._resume()

        while self.epoch < cfg.epochs and not self.stopped_early:
            epoch = self.epoch + 1
            record: dict[str, float] = {"epoch": float(epoch), "loss": self._train_epoch(epoch)}
            self.epoch = epoch

            if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                val = self.evaluator.evaluate(self.model, "val")
                if cfg.metric not in val:
                    raise KeyError(f"Metric {cfg.metric!r} not produced; got {sorted(val)}")
                record.update({f"val_{k}": v for k, v in val.items()})
                if val[cfg.metric] > self.best_metric:
                    self.best_metric = val[cfg.metric]
                    self.best_epoch = epoch
                    self.best_state = _snapshot(self.model)
                elif epoch - self.best_epoch >= cfg.patience:
                    self.stopped_early = True
            self.history.append(record)
            self.log.info(
                "epoch %d loss %.4f %s=%s",
                epoch,
                record["loss"],
                cfg.metric,
                record.get(f"val_{cfg.metric}"),
            )
            self._save_checkpoint()

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.run_dir is not None:
            self.model.save(self.run_dir / BEST_MODEL)
        return self._result()

    def _result(self) -> dict[str, Any]:
        val = self.evaluator.evaluate(self.model, "val")
        test = self.evaluator.evaluate(self.model, "test")
        result = {
            "experiment": self.cfg.name,
            "seed": self.cfg.seed,
            "epochs_run": self.epoch,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
            "metric": self.cfg.train.metric,
            "val": val,
            "test": test,
            "history": self.history,
        }
        if self.run_dir is not None:
            write_json(result, self.run_dir / METRICS)
        return result


@dataclass(frozen=True)
class GraphDims:
    """Size provider for graph-free models (MF)."""

    num_users: int
    num_items: int
    kind: str = "none"


def _enrichment_tables(paths: Paths) -> dict[str, pd.DataFrame | None]:
    """Aspect tables for builders that use them; None when enrichment has not been run."""
    from amazon_recsys.data.schemas import ASPECT_JUDGEMENTS, ASPECTS

    def load(table: str, schema: Any) -> pd.DataFrame | None:
        path = paths.processed(table)
        return read_parquet(path, schema) if path.exists() else None

    return {
        "aspects": load("aspects", ASPECTS),
        "aspect_judgements": load("aspect_judgements", ASPECT_JUDGEMENTS),
    }


def _prepare(cfg: Config, paths: Paths) -> tuple[Any, ...]:
    """Load splits and build index, model and evaluator for `cfg`."""
    splits = {n: read_parquet(paths.split(n), INTERACTIONS) for n in ("train", "val", "test")}
    user_index, item_index = build_index(*splits.values())

    model_cls = get_model_class(cfg.model.name)
    if getattr(model_cls, "requires_graph", None) is None:
        graph: Any = GraphDims(len(user_index), len(item_index))
    else:
        if cfg.graph is None:
            raise ValueError(
                f"Model {cfg.model.name!r} needs a {model_cls.requires_graph!r} graph; "
                "the experiment has no `graph`"
            )
        graph = build_graph(
            cfg.graph.name,
            splits["train"],
            cfg,
            user_index=user_index,
            item_index=item_index,
            **_enrichment_tables(paths),
        )
        prepare = getattr(graph, "prepare", None)  # e.g. KnowledgeGraph trains KG embeddings
        if prepare is not None:
            prepare(cfg, paths)
    model = build_model(cfg, graph)
    evaluator = Evaluator(splits, user_index, item_index, cfg.eval.ks)
    return splits, user_index, item_index, model, evaluator


def run_experiment(cfg: Config, paths: Paths, resume_dir: Path | None = None) -> dict[str, Any]:
    """Load splits, build index/model/evaluator, train, evaluate, and write the run directory."""
    set_seed(cfg.seed)
    splits, user_index, item_index, model, evaluator = _prepare(cfg, paths)

    if resume_dir is not None:
        run_dir = Path(resume_dir)
    else:
        run_dir = make_run_dir(paths.runs_root, make_run_id(cfg.name, cfg.seed), cfg.to_yaml())
    logger = setup_logging("amazon_recsys.trainer", run_dir / "log.txt")

    trainer = Trainer(
        model, cfg, splits["train"], user_index, item_index, evaluator, run_dir, logger
    )
    result = trainer.fit(resume=resume_dir is not None)
    result["run_dir"] = str(run_dir)
    return result


def evaluate_run(run_dir: Path, paths: Paths) -> dict[str, Any]:
    """Re-evaluate a finished run's best model on val and test; writes evaluation.json."""
    run_dir = Path(run_dir)
    cfg = Config.from_yaml((run_dir / "config.yaml").read_text())
    set_seed(cfg.seed)
    _, _, _, model, evaluator = _prepare(cfg, paths)
    model.load(run_dir / BEST_MODEL)
    result = {
        "experiment": cfg.name,
        "seed": cfg.seed,
        "val": evaluator.evaluate(model, "val"),
        "test": evaluator.evaluate(model, "test"),
    }
    write_json(result, run_dir / "evaluation.json")
    return result


def embed_experiment(cfg: Config, paths: Paths) -> dict[str, Any]:
    """Build the experiment's knowledge graph and train (or load cached) KG embeddings."""
    _, _, _, model, _ = _prepare(cfg, paths)
    graph = model.graph
    return {
        "experiment": cfg.name,
        "entities": len(graph.entity_index),
        "relations": len(graph.relation_index),
        "triples": len(graph.triples),
        "embedding_shape": list(graph.entity_embeddings.shape),
    }
