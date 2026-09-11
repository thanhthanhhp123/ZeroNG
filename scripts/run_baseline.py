"""Train and evaluate an anomalib baseline under the ZeroNG split protocol.

For each (category, seed): build the seeded split, fit the model, predict raw scores on val_good,
val and test, compute metrics, and log config + seed + git commit + split manifest + per-image
scores to MLflow. The split seed is fixed by the config; the training seed varies, so the spread
across seeds measures model variance only.

Example:
    uv run python scripts/run_baseline.py --config configs/patchcore_mvtec.yaml \
        --categories bottle --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import anomalib
import mlflow
import numpy as np
import torch
import yaml
from anomalib.engine import Engine
from lightning.pytorch import seed_everything

from zerong.data.anomalib_adapter import ManifestDataModule
from zerong.data.splits import (
    SplitConfig,
    make_splits,
    save_manifest,
    scan_mvtec_ad2_category,
    scan_mvtec_category,
)
from zerong.evaluation import SplitPredictions, evaluate, predict
from zerong.models.registry import build_model

REPO = Path(__file__).resolve().parents[1]
EVAL_SPLITS = ("val_good", "val", "test")
# Local MLflow store. The plain-folder backend ("./mlruns") is in maintenance mode in MLflow 3.x
# and refuses to start, so a local SQLite file is used instead (still no server).
TRACKING_URI = f"sqlite:///{(REPO / 'mlflow.db').as_posix()}"
ARTIFACTS = REPO / "mlartifacts"
log = logging.getLogger("run_baseline")


def git_state() -> dict[str, str]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "git_commit": git("rev-parse", "HEAD") or "uncommitted",
        "git_dirty": str(bool(git("status", "--porcelain"))),
    }


def hardware() -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    return f"{platform.processor() or platform.machine()} | {gpu}"


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = str(v)
    return out


def write_scores(path: Path, preds: dict[str, SplitPredictions], root: Path, types: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "path", "label", "defect_type", "score"])
        for split, p in preds.items():
            for img, label, score in zip(p.paths, p.labels, p.scores, strict=True):
                rel = Path(img).relative_to(root).as_posix()
                writer.writerow([split, rel, int(label), types[img], f"{score:.6g}"])


def run_one(
    cfg: dict, config_path: Path, category: str, seed: int, out_root: Path, artifact_root: Path
) -> dict:
    data_cfg, model_cfg, train_cfg = cfg["dataset"], cfg["model"], cfg["train"]
    root = REPO / data_cfg["root"] / category
    split_cfg = SplitConfig(
        seed=cfg["split"].get("seed", 0),
        val_fraction=cfg["split"]["val_fraction"],
        val_good_fraction=cfg["split"]["val_good_fraction"],
    )
    layout = data_cfg.get("layout", "mvtec")
    if layout == "mvtec_ad2":
        train_good, val_good, test = scan_mvtec_ad2_category(root)
        splits = make_splits(train_good, test, split_cfg, val_good=val_good)
    elif layout == "mvtec":
        splits = make_splits(*scan_mvtec_category(root), split_cfg)
    else:
        raise ValueError(f"unknown dataset layout {layout!r}")
    run_dir = out_root / model_cfg["name"] / data_cfg["name"] / category / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(splits, split_cfg, run_dir / "split.json")

    seed_everything(seed, workers=True)
    datamodule = ManifestDataModule(
        root=root,
        splits=splits,
        dataset_name=data_cfg["name"],
        category=category,
        image_size=data_cfg["image_size"],
        train_batch_size=train_cfg["batch_size"],
        eval_batch_size=train_cfg.get("eval_batch_size", 8),
        num_workers=train_cfg["num_workers"],
    )
    model = build_model(
        model_cfg["name"], model_cfg.get("params", {}), data_cfg["image_size"], REPO / "data"
    )
    engine = Engine(
        default_root_dir=run_dir / "anomalib",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        # Lightning stops at whichever limit comes first; models may override (PatchCore: 1 epoch).
        max_epochs=train_cfg.get("max_epochs"),
        max_steps=train_cfg.get("max_steps", -1),
    )
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - t0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = {
        s: predict(model, datamodule.eval_dataloader(s), device, keep_maps=(s == "test"))
        for s in EVAL_SPLITS
    }
    n_images = sum(p.scores.size for p in preds.values())
    forward_ms = 1000 * sum(p.forward_seconds for p in preds.values()) / n_images

    metrics = evaluate(preds, cfg["calibration"]["zero_ng_max_frr"])
    metrics |= {"fit_seconds": fit_seconds, "forward_ms_per_image": forward_ms}

    types = {str(root / s.path): s.defect_type for split in EVAL_SPLITS for s in splits[split]}
    write_scores(run_dir / "scores.csv", preds, root, types)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(cfg["experiment"])
    run_name = f"{model_cfg['name']}-{data_cfg['name']}-{category}-s{seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten(cfg) | {"category": category, "seed": seed})
        mlflow.set_tags(
            git_state()
            | {
                "config": (
                    config_path.relative_to(REPO).as_posix()
                    if config_path.is_relative_to(REPO)
                    else config_path.as_posix()
                ),
                "hardware": hardware(),
                "anomalib": anomalib.__version__,
                "torch": torch.__version__,
            }
        )
        mlflow.log_metrics(metrics)
        for name in ("split.json", "scores.csv", "metrics.json"):
            mlflow.log_artifact(str(run_dir / name))
        mlflow.log_artifact(str(config_path))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", help="default: all categories in the config")
    parser.add_argument("--seeds", nargs="+", type=int, help="default: train.seeds in the config")
    parser.add_argument("--out", type=Path, default=REPO / "results")
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACTS)
    parser.add_argument(
        "--num-workers",
        type=int,
        help="override train.num_workers (e.g. higher on Linux, where workers are forked)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers  # logged to MLflow with the rest of cfg
    mlflow.set_tracking_uri(args.tracking_uri)
    categories = args.categories or cfg["dataset"]["categories"]
    seeds = args.seeds or cfg["train"]["seeds"]

    headline = ("test_image_auroc", "test_aupro", "test_frr_at_zero_escape", "zerong_test_frr")
    for category in categories:
        results = [
            run_one(cfg, config_path, category, seed, args.out, args.artifact_root.resolve())
            for seed in seeds
        ]
        summary = []
        for key in headline:
            values = np.array([r[key] for r in results if key in r])
            if values.size:
                summary.append(f"{key}={values.mean():.4f}+/-{values.std():.4f}")
        log.info("%s (%d seeds): %s", category, len(seeds), "  ".join(summary))


if __name__ == "__main__":
    main()
