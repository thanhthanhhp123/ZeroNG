from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zerong.data.anomalib_adapter import ManifestDataModule
from zerong.data.splits import SplitConfig, make_splits, scan_mvtec_category


@pytest.fixture
def category(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)

    def save(rel: str, mode: str = "RGB") -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (40, 40, 3) if mode == "RGB" else (40, 40)
        array = rng.integers(0, 255, shape, dtype=np.uint8)
        if mode == "L":
            array = (array > 128).astype(np.uint8) * 255
        Image.fromarray(array, mode).save(path)

    for i in range(10):
        save(f"train/good/{i:03d}.png")
    for i in range(4):
        save(f"test/good/{i:03d}.png")
    for i in range(3):
        save(f"test/crack/{i:03d}.png")
        save(f"ground_truth/crack/{i:03d}_mask.png", "L")
    return tmp_path


def _datamodule(root: Path) -> tuple[ManifestDataModule, dict]:
    splits = make_splits(*scan_mvtec_category(root), SplitConfig(seed=0, val_good_fraction=0.2))
    dm = ManifestDataModule(
        root=root,
        splits=splits,
        dataset_name="toy",
        category="widget",
        image_size=32,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )
    dm.setup()
    return dm, splits


def test_subsets_match_manifest(category: Path):
    dm, splits = _datamodule(category)
    assert len(dm.train_data) == len(splits["train"]) == 8
    # Validation during fit uses held-out good images only: no real defect is seen.
    assert len(dm.val_data) == len(splits["val_good"]) == 2
    assert not dm.val_data.has_anomalous
    # anomalib's own splitting is disabled: test keeps both good and NG images.
    assert len(dm.test_data) == len(splits["test"])
    assert dm.test_data.has_normal and dm.test_data.has_anomalous


def test_fit_loaders_are_built_once(category: Path):
    # EfficientAD requests the val loader every epoch; rebuilding it would respawn workers.
    dm, _ = _datamodule(category)
    assert dm.val_dataloader() is dm.val_dataloader()
    assert dm.train_dataloader() is dm.train_dataloader()


def test_eval_dataloader_resizes_images_and_masks(category: Path):
    dm, splits = _datamodule(category)
    batches = list(dm.eval_dataloader("val"))
    assert sum(b.image.shape[0] for b in batches) == len(splits["val"])
    batch = batches[0]
    assert batch.image.shape[-2:] == (32, 32)
    assert batch.gt_mask.shape[-2:] == (32, 32)
    assert set(batch.gt_label.tolist()) <= {0, 1}
