from pathlib import Path


def test_tiny_cfg_has_tiny_sizes(tiny_cfg):
    assert tiny_cfg.domain.name == "all_beauty"
    assert tiny_cfg.model.dim == 8
    assert tiny_cfg.train.epochs == 2
    assert tiny_cfg.train.batch_size == 16
    assert tiny_cfg.eval.ks == [5, 10]


def test_tmp_paths_stay_inside_tmp(tmp_paths, tmp_path):
    root = tmp_path.resolve()
    assert tmp_paths.root == root
    assert tmp_paths.interim_dir.is_relative_to(root)
    assert not Path("data").resolve().is_relative_to(root)
