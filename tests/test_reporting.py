import math

import pytest

from zerong.reporting import (
    METRICS,
    RunRecord,
    aggregate,
    dataset_mean,
    format_value,
    latest_per_seed,
    markdown_table,
)


def _run(category, seed, auroc, start=0, frr=0.1, commit="abc"):
    return RunRecord(
        model="patchcore",
        dataset="mvtec_ad",
        category=category,
        seed=seed,
        git_commit=commit,
        start_time=start,
        metrics={"test_image_auroc": auroc, "test_frr_at_zero_escape": frr},
    )


def test_latest_per_seed_keeps_most_recent_rerun():
    runs = [
        _run("bottle", 0, 0.5, start=1),
        _run("bottle", 0, 0.9, start=2),
        _run("bottle", 1, 0.7),
    ]
    kept = latest_per_seed(runs)
    assert len(kept) == 2
    assert {r.metrics["test_image_auroc"] for r in kept} == {0.9, 0.7}


def test_aggregate_mean_and_sample_std():
    runs = [_run("bottle", s, v) for s, v in enumerate([0.90, 0.92, 0.94])]
    runs.append(_run("cable", 0, 0.80))
    rows = aggregate(runs)
    assert [r["category"] for r in rows] == ["bottle", "cable"]
    bottle = rows[0]
    assert bottle["n_seeds"] == 3 and bottle["seeds"] == [0, 1, 2]
    assert bottle["test_image_auroc_mean"] == pytest.approx(0.92)
    assert bottle["test_image_auroc_std"] == pytest.approx(0.02)  # ddof=1
    assert rows[1]["test_image_auroc_std"] == 0.0  # single seed
    assert math.isnan(bottle["test_aupro_mean"])  # metric never logged


def test_dataset_mean_is_macro_over_categories():
    rows = aggregate([_run("a", 0, 0.9), _run("a", 1, 0.9), _run("b", 0, 0.5)])
    mean = dataset_mean(rows)
    assert mean["test_image_auroc_mean"] == pytest.approx(0.7)  # not weighted by seed count
    assert mean["n_seeds"] == 1


def test_format_value():
    assert format_value("test_image_auroc", 0.97634, 0.0021) == "0.976 ± 0.002"
    assert format_value("test_frr_at_zero_escape", 0.3071, 0.052) == "30.7% ± 5.2"
    assert format_value("test_aupro", float("nan"), float("nan")) == "n/a"


def test_markdown_table_shape():
    table = markdown_table(aggregate([_run("bottle", 0, 1.0), _run("cable", 0, 0.9)]))
    lines = table.splitlines()
    assert len(lines) == 2 + 2 + 1  # header, separator, 2 categories, mean row
    assert all(line.count("|") == len(METRICS) + 3 for line in lines)
    assert lines[-1].startswith("| **mean** |")
