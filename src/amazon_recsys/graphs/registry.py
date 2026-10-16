"""Graph builder registry: `@register_builder("bipartite")`, `build_graph(name, train, cfg)`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_BUILDERS: dict[str, type] = {}


def register_builder(name: str) -> Callable[[type], type]:
    """Class decorator registering a GraphBuilder subclass under `name`."""

    def decorator(cls: type) -> type:
        if name in _BUILDERS:
            raise ValueError(f"Graph builder {name!r} is already registered")
        _BUILDERS[name] = cls
        return cls

    return decorator


def available_builders() -> list[str]:
    return sorted(_BUILDERS)


def get_builder(name: str) -> type:
    try:
        return _BUILDERS[name]
    except KeyError:
        names = ", ".join(available_builders()) or "none"
        raise KeyError(f"Unknown graph builder {name!r}. Available: {names}") from None


def build_graph(name: str, train: Any, cfg: Any, **kwargs: Any) -> Any:
    """Instantiate the registered builder and build the graph from the train split.

    Extra keyword arguments (e.g. `user_index`, `item_index`) are passed to the builder.
    """
    return get_builder(name)().build(train, cfg, **kwargs)
