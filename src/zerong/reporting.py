"""Aggregate logged runs into the result tables. README numbers are generated, never typed."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Reported metrics, in table order: key -> column title.
METRICS: dict[str, str] = {
    "test_image_auroc": "Image AUROC",
    "test_pixel_auroc": "Pixel AUROC",
    "test_aupro": "AUPRO",
    "test_frr_at_zero_escape": "FRR @ 0 escape",
    "zerong_test_frr": "Zero-NG FRR",
    "zerong_test_escape_rate": "Zero-NG escape",
    "valcal_test_frr": "Val-cal FRR",
    "valcal_test_escape_rate": "Val-cal escape",
}
RATE_METRICS = {k for k in METRICS if "frr" in k or "escape" in k}


@dataclass(frozen=True)
class RunRecord:
    model: str
    dataset: str
    category: str
    seed: int
    git_commit: str
    start_time: int
    metrics: dict[str, float] = field(default_factory=dict)
    hardware: str = ""


def latest_per_seed(runs: list[RunRecord]) -> list[RunRecord]:
    """If a (model, dataset, category, seed) was run more than once, keep the most recent run."""
    latest: dict[tuple, RunRecord] = {}
    for r in runs:
        key = (r.model, r.dataset, r.category, r.seed)
        if key not in latest or r.start_time > latest[key].start_time:
            latest[key] = r
    return list(latest.values())


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    # Sample standard deviation over seeds (ddof=1); 0 for a single seed.
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def aggregate(runs: list[RunRecord]) -> list[dict]:
    """One row per (model, dataset, category): mean and std over seeds of every metric."""
    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in latest_per_seed(runs):
        groups[(r.model, r.dataset, r.category)].append(r)
    rows = []
    for (model, dataset, category), group in sorted(groups.items()):
        row: dict = {
            "model": model,
            "dataset": dataset,
            "category": category,
            "n_seeds": len(group),
            "seeds": sorted(r.seed for r in group),
            "git_commits": sorted({r.git_commit for r in group}),
        }
        for key in METRICS:
            values = [r.metrics[key] for r in group if key in r.metrics]
            row[f"{key}_mean"], row[f"{key}_std"] = _mean_std(values)
        rows.append(row)
    return rows


def dataset_mean(rows: list[dict]) -> dict:
    """Macro average over categories of the per-category means (std across categories)."""
    out: dict = {"category": "**mean**", "n_seeds": min((r["n_seeds"] for r in rows), default=0)}
    for key in METRICS:
        out[f"{key}_mean"], out[f"{key}_std"] = _mean_std([r[f"{key}_mean"] for r in rows])
    return out


HEADLINE_METRICS = (
    "test_image_auroc",
    "test_aupro",
    "test_frr_at_zero_escape",
    "zerong_test_frr",
    "zerong_test_escape_rate",
)


def headline_table(groups: dict[tuple, list[dict]]) -> str:
    """Compact table of the macro averages, one row per (model, dataset)."""
    header = ["Model", "Dataset", *(METRICS[k] for k in HEADLINE_METRICS)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (model, dataset), rows in sorted(groups.items()):
        mean = dataset_mean(rows)
        cells = [model, dataset]
        cells += [format_value(k, mean[f"{k}_mean"], mean[f"{k}_std"]) for k in HEADLINE_METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_value(key: str, mean: float, std: float) -> str:
    if math.isnan(mean):
        return "n/a"
    if key in RATE_METRICS:
        return f"{100 * mean:.1f}% ± {100 * std:.1f}"
    return f"{mean:.3f} ± {std:.3f}"


def markdown_table(rows: list[dict]) -> str:
    """Markdown table for the rows of one (model, dataset), with a macro-average last row."""
    header = ["Category", *METRICS.values(), "Seeds"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in [*rows, dataset_mean(rows)]:
        cells = [row["category"]]
        cells += [format_value(k, row[f"{k}_mean"], row[f"{k}_std"]) for k in METRICS]
        cells.append(str(row["n_seeds"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
