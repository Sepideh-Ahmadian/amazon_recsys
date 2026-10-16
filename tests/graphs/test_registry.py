import pytest

from amazon_recsys.graphs import registry


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "_BUILDERS", {})


def test_registered_name_resolves_and_builds():
    @registry.register_builder("toy")
    class Toy:
        kind = "toy"

        def build(self, train, cfg):
            return ("built", train, cfg)

    assert registry.get_builder("toy") is Toy
    assert registry.build_graph("toy", "train", "cfg") == ("built", "train", "cfg")
    assert registry.available_builders() == ["toy"]


def test_unknown_name_lists_available():
    @registry.register_builder("a")
    class A: ...

    with pytest.raises(KeyError, match="Available: a"):
        registry.get_builder("nope")


def test_duplicate_registration_errors():
    @registry.register_builder("dup")
    class First: ...

    with pytest.raises(ValueError, match="already registered"):

        @registry.register_builder("dup")
        class Second: ...
