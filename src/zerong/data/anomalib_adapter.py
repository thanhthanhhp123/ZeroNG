"""Feed ZeroNG split manifests to anomalib.

anomalib's datamodules derive their own validation split (by default a copy of the test set, which
leaks test data into thresholding). ``ManifestDataModule`` replaces that logic: every subset comes
from a ZeroNG split manifest (see ``zerong.data.splits``), and anomalib's automatic splitting is
disabled.

During ``Engine.fit`` the validation subset is ``val_good`` (held-out good images only), so
model-internal statistics such as EfficientAD's map-normalisation quantiles never see a real defect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize

from zerong.data.splits import Sample


class ManifestDataset(AnomalibDataset):
    """An anomalib dataset built from a list of ``Sample``s relative to ``root``."""

    def __init__(
        self,
        root: Path,
        samples: list[Sample],
        split: str,
        dataset_name: str,
        category: str,
        augmentations=None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        self._name = dataset_name
        self.category = category
        root = Path(root)
        frame = pd.DataFrame(
            {
                "image_path": [str(root / s.path) for s in samples],
                "label": ["abnormal" if s.label else "normal" for s in samples],
                "label_index": pd.array([s.label for s in samples], dtype="Int64"),
                "mask_path": [str(root / s.mask_path) if s.mask_path else "" for s in samples],
                "defect_type": [s.defect_type for s in samples],
                "split": split,
            }
        )
        anomalous = [s for s in samples if s.label == 1]
        has_all_masks = bool(anomalous) and all(s.mask_path for s in anomalous)
        frame.attrs["task"] = "segmentation" if has_all_masks else "classification"
        self.samples = frame

    @property
    def name(self) -> str:
        return self._name


class ManifestDataModule(AnomalibDataModule):
    """Datamodule whose train/val/test subsets come verbatim from a split manifest."""

    def __init__(
        self,
        root: Path,
        splits: dict[str, list[Sample]],
        dataset_name: str,
        category: str,
        image_size: int,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        fit_val_split: str = "val_good",
    ) -> None:
        # Resizing inside the dataset keeps images and masks at model resolution and makes
        # batches collatable; anomalib swaps in the model's own Resize when a trainer is attached.
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            augmentations=Resize((image_size, image_size), antialias=True),
        )
        self.root = Path(root)
        self.splits = splits
        self.dataset_name = dataset_name
        self.category = category
        self.fit_val_split = fit_val_split
        self._loaders: dict[str, DataLoader] = {}

    @property
    def name(self) -> str:
        return self.dataset_name

    def make_dataset(self, split: str) -> ManifestDataset:
        return ManifestDataset(
            self.root,
            self.splits[split],
            split,
            self.dataset_name,
            self.category,
            augmentations=self.test_augmentations,
        )

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = self.make_dataset("train")
        self.val_data = self.make_dataset(self.fit_val_split)
        self.test_data = self.make_dataset("test")

    # Subsets come from the manifest: disable anomalib's own splitting.
    def _create_test_split(self) -> None:
        pass

    def _create_val_split(self) -> None:
        pass

    def _cached_loader(
        self, key: str, dataset, shuffle: bool, batch_size: int, num_workers: int
    ) -> DataLoader:
        # Windows spawns DataLoader workers and each one re-imports torch + anomalib (~30 s,
        # ~1 GB RAM). Build each fit-time loader once, with persistent workers: EfficientAD
        # requests the val loader again at every epoch.
        if key not in self._loaders:
            self._loaders[key] = DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                collate_fn=dataset.collate_fn,
            )
        return self._loaders[key]

    def train_dataloader(self) -> DataLoader:
        return self._cached_loader(
            "train", self.train_data, True, self.train_batch_size, self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        # val_good is a few dozen images: load in the main process rather than keeping a second
        # set of worker processes alive next to the training workers.
        return self._cached_loader("val", self.val_data, False, self.eval_batch_size, 0)

    def eval_dataloader(self, split: str, num_workers: int = 0) -> DataLoader:
        """Deterministic dataloader over any manifest split, for inference.

        Defaults to loading in the main process: evaluation sets are a few hundred images, and
        spawning workers costs more than it saves.
        """
        dataset = self.make_dataset(split)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )
