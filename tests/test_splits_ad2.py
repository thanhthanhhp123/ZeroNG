from pathlib import Path

import pytest

from zerong.data.splits import Sample, SplitConfig, make_splits, scan_mvtec_ad2_category


def _touch(root: Path, *rels: str) -> None:
    for rel in rels:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")


def test_scan_mvtec_ad2_layout(tmp_path: Path):
    _touch(
        tmp_path,
        "train/good/000_regular.png",
        "validation/good/100_regular.png",
        "test_public/good/200_regular.png",
        "test_public/bad/300_regular.png",
        "test_public/ground_truth/bad/300_regular_mask.png",
        "test_private/400_regular.png",
        "test_private_mixed/500_overexposed.png",
    )
    train, val_good, test = scan_mvtec_ad2_category(tmp_path)
    assert train == [Sample("train/good/000_regular.png", 0)]
    assert val_good == [Sample("validation/good/100_regular.png", 0)]
    assert Sample("test_public/good/200_regular.png", 0) in test
    assert (
        Sample(
            "test_public/bad/300_regular.png",
            1,
            "bad",
            "test_public/ground_truth/bad/300_regular_mask.png",
        )
        in test
    )
    # Private test sets have no public ground truth: never part of the protocol splits.
    assert all("private" not in s.path for s in train + val_good + test)


def test_provided_val_good_is_used_as_is():
    train = [Sample(f"train/good/{i:03d}.png", 0) for i in range(20)]
    val_good = [Sample(f"validation/good/{i:03d}.png", 0) for i in range(5)]
    test = [Sample(f"test_public/good/{i}.png", 0) for i in range(4)]
    test += [Sample(f"test_public/bad/{i}.png", 1, "bad") for i in range(6)]
    splits = make_splits(train, test, SplitConfig(seed=0, val_good_fraction=0.5), val_good)
    assert splits["val_good"] == val_good
    assert splits["train"] == train  # nothing held out from training
    with pytest.raises(ValueError):
        make_splits(train, test, SplitConfig(), [Sample("validation/bad.png", 1, "bad")])
