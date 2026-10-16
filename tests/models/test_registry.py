from types import SimpleNamespace

import pytest

from amazon_recsys.models import registry


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "_MODELS", {})


def _cfg(name):
    return SimpleNamespace(model=SimpleNamespace(name=name))


def test_build_model_passes_cfg_and_graph():
    @registry.register_model("toy")
    class Toy:
        requires_graph = None

        @classmethod
        def from_config(cls, cfg, graph):
            return ("toy", cfg.model.name, graph)

    assert registry.build_model(_cfg("toy"), None) == ("toy", "toy", None)


def test_wrong_graph_kind_raises_clear_error():
    @registry.register_model("needs_kg")
    class NeedsKG:
        requires_graph = "knowledge"

        @classmethod
        def from_config(cls, cfg, graph):
            return cls()

    bipartite = SimpleNamespace(kind="bipartite")
    with pytest.raises(ValueError, match="requires a 'knowledge' graph, got 'bipartite'"):
        registry.build_model(_cfg("needs_kg"), bipartite)
    with pytest.raises(ValueError, match="got None"):
        registry.build_model(_cfg("needs_kg"), None)
    assert registry.build_model(_cfg("needs_kg"), SimpleNamespace(kind="knowledge"))


def test_unknown_and_duplicate():
    @registry.register_model("a")
    class A: ...

    with pytest.raises(KeyError, match="Available: a"):
        registry.get_model_class("zzz")
    with pytest.raises(ValueError, match="already registered"):

        @registry.register_model("a")
        class B: ...
