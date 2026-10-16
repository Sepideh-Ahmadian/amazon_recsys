"""Model registry: `@register_model("gat")`, `build_model(cfg, graph)`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_MODELS: dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    """Class decorator registering a Recommender subclass under `name`."""

    def decorator(cls: type) -> type:
        if name in _MODELS:
            raise ValueError(f"Model {name!r} is already registered")
        _MODELS[name] = cls
        return cls

    return decorator


def available_models() -> list[str]:
    return sorted(_MODELS)


def get_model_class(name: str) -> type:
    try:
        return _MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(available_models()) or 'none'}"
        ) from None


def build_model(cfg: Any, graph: Any) -> Any:
    """Build `cfg.model.name`, first checking `graph.kind` against `requires_graph`."""
    cls = get_model_class(cfg.model.name)
    required = getattr(cls, "requires_graph", None)
    if required is not None:
        kind = getattr(graph, "kind", None)
        if kind != required:
            raise ValueError(
                f"Model {cfg.model.name!r} requires a {required!r} graph, got {kind!r}"
            )
    return cls.from_config(cfg, graph)
