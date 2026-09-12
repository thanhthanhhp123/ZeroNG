"""Generate result tables from logged MLflow runs (README numbers are never typed by hand).

Reads one or more MLflow stores (the local one and/or stores fetched from remote jobs), keeps
FINISHED runs of an experiment logged from a clean git tree, aggregates mean +/- std over seeds
per (model, dataset, category), and writes:

- experiments/<experiment>/<model>_<dataset>.csv   per-category numbers
- experiments/<experiment>/summary.md              markdown tables + provenance
- README.md, between the results markers           the same tables

Example:
    uv run python scripts/make_tables.py \
        --store remote_runs/patchcore-baseline-v4/mlflow.db --experiment baseline
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

import mlflow

from zerong.reporting import METRICS, RunRecord, aggregate, headline_table, markdown_table

# Chart tokens: validated categorical slots 1-2 on the light chart surface (all-pairs CVD
# separation 24.7, normal-vision 33.6), with surface/grid/ink tokens from the same palette.
SURFACE, GRID, AXIS_LINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, SECONDARY_INK, MUTED_INK = "#0b0b0b", "#52514e", "#898781"
SERIES_COLORS = ("#2a78d6", "#eb6834")

REPO = Path(__file__).resolve().parents[1]
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"
log = logging.getLogger("make_tables")


def load_runs(store: Path, experiment: str, allow_dirty: bool) -> list[RunRecord]:
    mlflow.set_tracking_uri(f"sqlite:///{store.resolve().as_posix()}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        log.warning("%s: no experiment named %r", store, experiment)
        return []
    runs, skipped = [], 0
    for run in mlflow.search_runs(
        [exp.experiment_id], filter_string="attributes.status = 'FINISHED'", output_format="list"
    ):
        tags, params = run.data.tags, run.data.params
        if tags.get("git_dirty") != "False" and not allow_dirty:
            skipped += 1
            continue
        runs.append(
            RunRecord(
                model=params["model.name"],
                dataset=params["dataset.name"],
                category=params["category"],
                seed=int(params["seed"]),
                git_commit=tags.get("git_commit", "unknown"),
                start_time=run.info.start_time,
                metrics=dict(run.data.metrics),
                hardware=tags.get("hardware", ""),
            )
        )
    if skipped:
        log.warning("%s: skipped %d runs from a dirty or unknown git tree", store, skipped)
    log.info("%s: %d runs", store, len(runs))
    return runs


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = ["category", "n_seeds", "seeds", "git_commits"]
    columns += [f"{k}_{s}" for k in METRICS for s in ("mean", "std")]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["seeds"] = " ".join(map(str, row["seeds"]))
            out["git_commits"] = " ".join(c[:8] for c in row["git_commits"])
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in out.items()}
            )


def plot_auroc_vs_frr(groups: dict[tuple, list[dict]], path: Path) -> None:
    """One point per category: image AUROC against the false rejects needed for zero escape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, ((_model, dataset), rows) in enumerate(sorted(groups.items())):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        xs = [r["test_image_auroc_mean"] for r in rows]
        ys = [100 * r["test_frr_at_zero_escape_mean"] for r in rows]
        # >=8px markers with a 2px surface ring, so overlapping points stay legible.
        ax.scatter(xs, ys, s=80, color=color, edgecolors=SURFACE, linewidths=2, zorder=3,
                   label=dataset)  # fmt: skip
        # Label selectively: the worst false-reject rate and the weakest AUROC of each dataset.
        worst = max(rows, key=lambda r: r["test_frr_at_zero_escape_mean"])
        weakest = min(rows, key=lambda r: r["test_image_auroc_mean"])
        for row in {id(worst): worst, id(weakest): weakest}.values():
            ax.annotate(
                row["category"],
                (row["test_image_auroc_mean"], 100 * row["test_frr_at_zero_escape_mean"]),
                textcoords="offset points",
                xytext=(9, 6),  # above right: at the top-left cluster, dots sit side by side
                fontsize=9,
                color=SECONDARY_INK,
            )

    ax.set_title(
        "A high AUROC does not mean the line can use it",
        loc="left", fontsize=13, color=INK, pad=18, fontweight="semibold",
    )  # fmt: skip
    ax.text(
        0, 1.03, "One point per category, mean over seeds; PatchCore at 256 px",
        transform=ax.transAxes, fontsize=10, color=SECONDARY_INK,
    )  # fmt: skip
    ax.set_xlabel("Image AUROC", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Good parts rejected to let no defect through", fontsize=10, color=SECONDARY_INK)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylim(-4, 104)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_LINE)
    ax.tick_params(colors=MUTED_INK, labelsize=9, length=0)
    # Lower left: the only quadrant the data leaves empty (a weak model is never cheap).
    ax.margins(x=0.06)
    legend = ax.legend(frameon=False, loc="lower left", fontsize=10)
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)  # identity comes from the dot, not from coloured text
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def render_summary(
    groups: dict[tuple, list[dict]], runs: list[RunRecord], command: str, figure: str | None = None
) -> str:
    commits = sorted({r.git_commit[:8] for r in runs})
    hardware = sorted({r.hardware for r in runs if r.hardware})
    parts = ["Macro averages over categories:", "", headline_table(groups), ""]
    if figure:
        parts += [
            f"![Image AUROC against the false rejects needed for zero escape, "
            f"one point per category]({figure})",
            "",
        ]
    for (model, dataset), rows in sorted(groups.items()):
        parts += [f"### {model} on {dataset}", "", markdown_table(rows), ""]
    parts += [
        "Test split only (the protocol's held-out part of the official test set). Mean ± sample "
        "std over training seeds; the last row is the macro average over categories (± std across "
        "categories).",
        "",
        "- **FRR @ 0 escape**: good parts rejected when the threshold catches every defect on "
        "test (threshold chosen on test: an achievable best case, like AUROC, not deployable).",
        "- **Zero-NG**: threshold from held-out good images only (≤ 1% false rejects on "
        "`val_good`), applied to test. The setting of a factory with no defect images.",
        "- **Val-cal**: threshold = lowest defect score on `val` (uses real defects), applied "
        "to test.",
        "",
        f"Provenance: {len(runs)} runs; git commit(s) {', '.join(commits)}; "
        f"hardware: {'; '.join(hardware) or 'unknown'}.  ",
        f"Generated by `{command}`.",
    ]
    return "\n".join(parts) + "\n"


def update_readme(readme: Path, body: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        return False
    readme.write_text(
        pattern.sub(lambda _: f"{README_START}\n{body}{README_END}", text), encoding="utf-8"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store", type=Path, action="append", help="MLflow SQLite file (repeatable)"
    )
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--out", type=Path, help="default: experiments/<experiment>")
    parser.add_argument("--allow-dirty", action="store_true", help="include uncommitted runs")
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    stores = args.store or [REPO / "mlflow.db"]
    runs = [r for store in stores for r in load_runs(store, args.experiment, args.allow_dirty)]
    if not runs:
        raise SystemExit("no runs found")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in aggregate(runs):
        groups[(row["model"], row["dataset"])].append(row)

    out = args.out or REPO / "experiments" / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    for (model, dataset), rows in groups.items():
        write_csv(out / f"{model}_{dataset}.csv", rows)

    def rel(p: Path) -> str:
        p = p.resolve()
        return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p.name

    command = "uv run python scripts/make_tables.py " + " ".join(
        [*(f"--store {rel(s)}" for s in stores), f"--experiment {args.experiment}"]
    )
    figure_path = out / "auroc_vs_frr.png"
    plot_auroc_vs_frr(groups, figure_path)
    summary = render_summary(groups, runs, command, figure=rel(figure_path))
    (out / "summary.md").write_text(summary, encoding="utf-8")
    log.info("wrote %s", rel(out))
    if not args.no_readme and update_readme(REPO / "README.md", summary):
        log.info("updated README.md results section")


if __name__ == "__main__":
    main()
