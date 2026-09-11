import random
from pathlib import Path

import pytest

from zerong.data.splits import (
    Sample,
    SplitConfig,
    load_manifest,
    make_splits,
    save_manifest,
    scan_mvtec_category,
    subsample_train,
)


def _dataset(n_train=100, n_good_test=20, defects=(("crack", 10), ("scratch", 7), ("hole", 1))):
    train = [Sample(f"train/good/{i:03d}.png", 0) for i in range(n_train)]
    test = [Sample(f"test/good/{i:03d}.png", 0) for i in range(n_good_test)]
    for name, n in defects:
        test += [Sample(f"test/{name}/{i:03d}.png", 1, name) for i in range(n)]
    return train, test


def test_splits_are_a_partition():
    train, test = _dataset()
    splits = make_splits(train, test, SplitConfig(seed=0))
    assert sorted(splits["train"] + splits["val_good"], key=lambda s: s.path) == train
    assert sorted(splits["val"] + splits["test"], key=lambda s: s.path) == sorted(
        test, key=lambda s: s.path
    )
    assert not {s.path for s in splits["val"]} & {s.path for s in splits["test"]}


def test_splits_deterministic_and_order_independent():
    train, test = _dataset()
    a = make_splits(train, test, SplitConfig(seed=3))
    shuffled_train, shuffled_test = train[:], test[:]
    random.Random(1).shuffle(shuffled_train)
    random.Random(2).shuffle(shuffled_test)
    b = make_splits(shuffled_train, shuffled_test, SplitConfig(seed=3))
    assert a == b


def test_different_seeds_differ():
    train, test = _dataset()
    a = make_splits(train, test, SplitConfig(seed=0))
    b = make_splits(train, test, SplitConfig(seed=1))
    assert a["val"] != b["val"]


def test_stratified_every_defect_type_in_both_val_and_test():
    train, test = _dataset()
    splits = make_splits(train, test, SplitConfig(seed=0, val_fraction=0.3))
    for name in ("good", "crack", "scratch"):
        assert any(s.defect_type == name for s in splits["val"])
        assert any(s.defect_type == name for s in splits["test"])
    # A defect type with a single image cannot be in both; it must not be dropped.
    assert sum(s.defect_type == "hole" for s in splits["val"] + splits["test"]) == 1


def test_val_good_size():
    train, test = _dataset(n_train=100)
    splits = make_splits(train, test, SplitConfig(seed=0, val_good_fraction=0.1))
    assert len(splits["val_good"]) == 10
    assert len(splits["train"]) == 90
    no_holdout = make_splits(train, test, SplitConfig(seed=0, val_good_fraction=0.0))
    assert no_holdout["val_good"] == []


def test_rejects_defects_in_train():
    train, test = _dataset()
    with pytest.raises(ValueError):
        make_splits(train + [Sample("train/good/bad.png", 1, "crack")], test, SplitConfig())


def test_subsample_nested_and_deterministic():
    train, _ = _dataset()
    subsets = [subsample_train(train, k, seed=0) for k in (1, 5, 10, 50)]
    for small, large in zip(subsets, subsets[1:], strict=False):
        assert large[: len(small)] == small
    assert subsample_train(train, 10, seed=0) == subsets[2]
    assert subsample_train(train, 10, seed=1) != subsets[2]
    with pytest.raises(ValueError):
        subsample_train(train, 101, seed=0)


def test_manifest_roundtrip(tmp_path: Path):
    train, test = _dataset()
    config = SplitConfig(seed=7)
    splits = make_splits(train, test, config)
    save_manifest(splits, config, tmp_path / "split.json")
    loaded, loaded_config = load_manifest(tmp_path / "split.json")
    assert loaded == splits
    assert loaded_config == config


def test_scan_mvtec_category(tmp_path: Path):
    for rel in (
        "train/good/000.png",
        "train/good/001.png",
        "test/good/000.png",
        "test/crack/000.png",
        "ground_truth/crack/000_mask.png",
        "test/crack/readme.txt",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    train, test = scan_mvtec_category(tmp_path)
    assert [s.path for s in train] == ["train/good/000.png", "train/good/001.png"]
    assert Sample("test/good/000.png", 0) in test
    assert Sample("test/crack/000.png", 1, "crack", "ground_truth/crack/000_mask.png") in test
    assert len(test) == 2


def test_scan_accepts_visa_mask_names(tmp_path: Path):
    for rel in ("train/good/a.JPG", "test/bad/b.JPG", "ground_truth/bad/b.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    _, test = scan_mvtec_category(tmp_path)
    assert test == [Sample("test/bad/b.JPG", 1, "bad", "ground_truth/bad/b.png")]
