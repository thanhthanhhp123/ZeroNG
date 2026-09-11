"""Seeded split protocol for MVTec-style datasets.

MVTec-style datasets ship only good training images and a mixed test set. There is no defect
validation split, so anything tuned on the official test set leaks. This module defines the
protocol used for every number in ZeroNG:

- ``train``: good training images used to fit the model (optionally subsampled to k images).
- ``val_good``: good images never used for fitting. Used for threshold calibration in the true
  zero-NG setting (no real defects available). Held out from the training images, unless the
  dataset ships its own good-only validation set (MVTec AD 2), which is then used as-is.
- ``val``: part of the official test set (good + defects), stratified by defect type. Used for
  oracle threshold calibration and model selection.
- ``test``: the rest of the official test set. Used **only** for final numbers.

All splits are deterministic given a seed and independent of input order: samples are sorted by
path before shuffling. Paths are stored as POSIX strings relative to the dataset root so a split
manifest produced on Windows is valid on Linux.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GOOD = "good"
SPLIT_NAMES = ("train", "val_good", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG"}


@dataclass(frozen=True)
class Sample:
    """One image. ``label`` is 0 for good, 1 for anomalous."""

    path: str
    label: int
    defect_type: str = GOOD
    mask_path: str | None = None


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    val_fraction: float = 0.3
    val_good_fraction: float = 0.1
    min_val_good: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if not 0.0 <= self.val_good_fraction < 1.0:
            raise ValueError(f"val_good_fraction must be in [0, 1), got {self.val_good_fraction}")


def _rng(seed: int, *salt: str) -> np.random.Generator:
    """Independent RNG stream per (seed, purpose) so changing one split never shifts another."""
    salt_ints = [int.from_bytes(s.encode(), "little") % (2**32) for s in salt]
    return np.random.default_rng([seed, *salt_ints])


def _shuffled(samples: list[Sample], rng: np.random.Generator) -> list[Sample]:
    ordered = sorted(samples, key=lambda s: s.path)
    return [ordered[i] for i in rng.permutation(len(ordered))]


def make_splits(
    train_good: list[Sample],
    test: list[Sample],
    config: SplitConfig,
    val_good: list[Sample] | None = None,
) -> dict[str, list[Sample]]:
    """Build the four protocol splits.

    The official test set is split per defect type (``good`` included) so every defect type
    appears in both ``val`` and ``test`` whenever it has at least two images.

    If ``val_good`` is given (a dataset-provided good-only validation set), it is used as-is and
    no training image is held out; ``config.val_good_fraction`` is then ignored.
    """
    if any(s.label != 0 for s in train_good):
        raise ValueError("train_good must contain only good (label 0) samples")

    if val_good is not None:
        if any(s.label != 0 for s in val_good):
            raise ValueError("val_good must contain only good (label 0) samples")
        train = list(train_good)
    else:
        pool = _shuffled(train_good, _rng(config.seed, "val_good"))
        n_val_good = 0
        if config.val_good_fraction > 0:
            n_val_good = max(config.min_val_good, round(len(pool) * config.val_good_fraction))
            n_val_good = min(n_val_good, len(pool) - 1)
        val_good, train = pool[:n_val_good], pool[n_val_good:]

    by_type: dict[str, list[Sample]] = defaultdict(list)
    for s in test:
        by_type[s.defect_type].append(s)

    val: list[Sample] = []
    final_test: list[Sample] = []
    for defect_type in sorted(by_type):
        group = _shuffled(by_type[defect_type], _rng(config.seed, "val", defect_type))
        n_val = round(len(group) * config.val_fraction)
        if len(group) >= 2:
            n_val = min(max(n_val, 1), len(group) - 1)
        val.extend(group[:n_val])
        final_test.extend(group[n_val:])

    splits = {"train": train, "val_good": val_good, "val": val, "test": final_test}
    return {name: sorted(items, key=lambda s: s.path) for name, items in splits.items()}


def subsample_train(train: list[Sample], k: int, seed: int) -> list[Sample]:
    """Pick k training images for the data-efficiency curve.

    Subsets are nested for a fixed seed: the k=5 subset contains the k=1 subset, and so on. This
    makes the curve measure the effect of adding images rather than of drawing a new sample.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(train):
        raise ValueError(f"k={k} exceeds the {len(train)} available training images")
    return _shuffled(train, _rng(seed, "subsample"))[:k]


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix in IMAGE_EXTENSIONS)


def scan_mvtec_category(root: Path) -> tuple[list[Sample], list[Sample]]:
    """List the train-good and test samples of one category in the MVTec AD folder layout.

    Layout: ``train/good/*.png``, ``test/<defect_type>/*.png``,
    ``ground_truth/<defect_type>/<stem>_mask.png``. VisA is converted to this layout by
    anomalib's datamodule during preparation, except that its masks are named ``<stem>.png``;
    both names are accepted.
    """
    root = Path(root)

    def rel(p: Path) -> str:
        return p.relative_to(root).as_posix()

    train_good = [Sample(rel(p), 0) for p in _images(root / "train" / GOOD)]
    test: list[Sample] = []
    for defect_dir in sorted(d for d in (root / "test").iterdir() if d.is_dir()):
        defect_type = defect_dir.name
        for p in _images(defect_dir):
            if defect_type == GOOD:
                test.append(Sample(rel(p), 0))
                continue
            mask_dir = root / "ground_truth" / defect_type
            candidates = (mask_dir / f"{p.stem}_mask.png", mask_dir / f"{p.stem}.png")
            mask = next((m for m in candidates if m.exists()), None)
            test.append(Sample(rel(p), 1, defect_type, rel(mask) if mask else None))
    return train_good, test


def scan_mvtec_ad2_category(root: Path) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """List train-good, validation-good and public-test samples of one MVTec AD 2 category.

    Layout: ``train/good``, ``validation/good`` (official good-only validation set),
    ``test_public/{good,bad}``, ``test_public/ground_truth/bad/<stem>_mask.png``.
    ``test_private`` and ``test_private_mixed`` have no public ground truth (they are scored by
    the MVTec evaluation server) and are not part of the protocol splits.
    """
    root = Path(root)

    def rel(p: Path) -> str:
        return p.relative_to(root).as_posix()

    train_good = [Sample(rel(p), 0) for p in _images(root / "train" / GOOD)]
    val_good = [Sample(rel(p), 0) for p in _images(root / "validation" / GOOD)]
    test = [Sample(rel(p), 0) for p in _images(root / "test_public" / GOOD)]
    mask_dir = root / "test_public" / "ground_truth" / "bad"
    for p in _images(root / "test_public" / "bad"):
        mask = mask_dir / f"{p.stem}_mask.png"
        test.append(Sample(rel(p), 1, "bad", rel(mask) if mask.exists() else None))
    return train_good, val_good, test


def save_manifest(splits: dict[str, list[Sample]], config: SplitConfig, path: Path) -> None:
    payload = {
        "config": asdict(config),
        "splits": {name: [asdict(s) for s in splits[name]] for name in SPLIT_NAMES},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_manifest(path: Path) -> tuple[dict[str, list[Sample]], SplitConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = {name: [Sample(**s) for s in items] for name, items in payload["splits"].items()}
    return splits, SplitConfig(**payload["config"])
