import pytest

from amazon_recsys.utils.paths import PathError, Paths


@pytest.fixture
def paths(tmp_path):
    return Paths(root=tmp_path, domain="all_beauty")


def test_domain_paths_are_namespaced(paths):
    for p in (
        paths.interim_dir,
        paths.processed_dir,
        paths.embeddings_dir,
        paths.reviews,
        paths.llm_cache,
        paths.processed("judgements"),
        paths.split("train"),
        paths.split_manifest,
        paths.embedding("abc"),
        paths.prompt_dir,
    ):
        assert "all_beauty" in p.parts


def test_all_paths_inside_root(paths, tmp_path):
    for p in (paths.raw_dir, paths.run_dir("r1"), paths.reports, paths.split("val")):
        assert p.is_relative_to(tmp_path.resolve())


def test_layout_matches_spec(paths, tmp_path):
    root = tmp_path.resolve()
    assert paths.reviews == root / "data/all_beauty/interim/reviews.parquet"
    assert (
        paths.processed("interactions") == root / "data/all_beauty/processed/interactions.parquet"
    )
    assert paths.split("test") == root / "data/all_beauty/processed/splits/test.parquet"
    assert paths.run_dir("x") == root / "runs/x"


def test_escaping_root_is_rejected(paths):
    with pytest.raises(PathError):
        paths.resolve("../outside.txt")
    with pytest.raises(PathError):
        paths.run_dir("../../etc")


@pytest.mark.parametrize("bad", ["", "..", "a/b", "../x"])
def test_invalid_domain_rejected(tmp_path, bad):
    with pytest.raises(PathError):
        Paths(root=tmp_path, domain=bad)


def test_unknown_split_rejected(paths):
    with pytest.raises(PathError):
        paths.split("holdout")
