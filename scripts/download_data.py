"""Download public datasets into data/.

Idempotent: a dataset whose extracted folders already exist is skipped. Every archive is verified
against its SHA-256 before extraction (anomalib only checks freshly downloaded files, so a partial
archive left by an interrupted run would otherwise be extracted).

Check data/LICENSES.md before adding a dataset here.

Example:
    uv run python scripts/download_data.py --dataset mvtec_ad visa imagenette
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anomalib.data.datamodules.image.mvtecad import DOWNLOAD_INFO as MVTEC_AD_INFO
from anomalib.data.datamodules.image.mvtecad2 import DOWNLOAD_INFO as MVTEC_AD2_INFO
from anomalib.data.datamodules.image.visa import DOWNLOAD_INFO as VISA_INFO
from anomalib.data.datamodules.image.visa import Visa
from anomalib.data.datasets.image.mvtecad2 import CATEGORIES as MVTEC_AD2_CATEGORIES
from anomalib.data.utils.download import DownloadInfo, check_hash, download_and_extract
from anomalib.models.image.efficient_ad.lightning_model import IMAGENETTE_DOWNLOAD_INFO

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

MVTEC_AD_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill",
    "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)  # fmt: skip
VISA_CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1",
    "pcb2", "pcb3", "pcb4", "pipe_fryum",
)  # fmt: skip

log = logging.getLogger("download_data")


def verified_download(root: Path, info: DownloadInfo) -> None:
    archive = root / (info.filename or info.url.split("/")[-1])
    if archive.exists():
        log.info("Verifying existing archive %s", archive)
        check_hash(archive, info.hashsum)  # raises on a partial or corrupted archive
    download_and_extract(root, info)


def download_mvtec_ad(root: Path) -> None:
    if all((root / c / "test").is_dir() for c in MVTEC_AD_CATEGORIES):
        log.info("MVTec AD already present in %s", root)
        return
    verified_download(root, MVTEC_AD_INFO)


def download_mvtec_ad2(root: Path) -> None:
    """MVTec AD 2 via the archive URL and checksum pinned in anomalib (see data/LICENSES.md)."""
    if all((root / c / "test_public").is_dir() for c in MVTEC_AD2_CATEGORIES):
        log.info("MVTec AD 2 already present in %s", root)
        return
    verified_download(root, MVTEC_AD2_INFO)


def download_visa(root: Path) -> None:
    split_root = root / "visa_pytorch"
    if all((split_root / c / "test").is_dir() for c in VISA_CATEGORIES):
        log.info("VisA already present in %s", split_root)
        return
    if not all((root / c).is_dir() for c in VISA_CATEGORIES):
        verified_download(root, VISA_INFO)
    log.info("Converting VisA to the MVTec layout (official 1cls split)")
    Visa(root=root).apply_cls1_split()


def download_imagenette(root: Path) -> None:
    """Auxiliary images for EfficientAD's ImageNet penalty term. Not an evaluation dataset."""
    if root.is_dir() and any(root.iterdir()):
        log.info("Imagenette already present in %s", root)
        return
    verified_download(root, IMAGENETTE_DOWNLOAD_INFO)


DOWNLOADERS = {
    "mvtec_ad": (download_mvtec_ad, DATA / "mvtec_ad"),
    "mvtec_ad2": (download_mvtec_ad2, DATA / "mvtec_ad2"),
    "visa": (download_visa, DATA / "visa"),
    "imagenette": (download_imagenette, DATA / "imagenette"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="+", choices=sorted(DOWNLOADERS), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for name in args.dataset:
        fn, root = DOWNLOADERS[name]
        fn(root)
        log.info("%s ready", name)


if __name__ == "__main__":
    main()
