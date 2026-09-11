"""Factory-facing metrics: what a production line actually pays for.

Convention used throughout: higher anomaly score = more likely NG, and an image is flagged NG
when ``score >= threshold``.

- *Escape*: an NG part classified OK (shipped to the customer). The expensive error.
- *False reject*: a good part classified NG (scrapped or sent to rework). The cheap error.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike


class Verdict(IntEnum):
    OK = 0
    NOT_CLEAR = 1
    NG = 2


def _split_scores(scores: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (good) or 1 (NG)")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    return scores[labels == 0], scores[labels == 1]


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    frr: float
    recall: float


def frr_at_recall(scores: ArrayLike, labels: ArrayLike, recall: float = 1.0) -> OperatingPoint:
    """False reject rate at the highest threshold that still catches ``recall`` of NG parts.

    With ``recall=1.0`` this is FRR at zero escape: the threshold is the lowest NG score, and
    every good part scoring at or above it is falsely rejected.
    """
    if not 0.0 < recall <= 1.0:
        raise ValueError(f"recall must be in (0, 1], got {recall}")
    good, ng = _split_scores(scores, labels)
    if ng.size == 0 or good.size == 0:
        raise ValueError("need at least one good and one NG sample")
    # Smallest number of NG we may miss while keeping recall >= target.
    n_catch = int(np.ceil(recall * ng.size - 1e-9))
    threshold = float(np.sort(ng)[::-1][n_catch - 1])
    frr = float(np.mean(good >= threshold))
    achieved = float(np.mean(ng >= threshold))
    return OperatingPoint(threshold=threshold, frr=frr, recall=achieved)


def frr_at_zero_escape(scores: ArrayLike, labels: ArrayLike) -> OperatingPoint:
    """Primary ZeroNG metric: FRR at 100% NG recall."""
    return frr_at_recall(scores, labels, recall=1.0)


def rates_at_threshold(scores: ArrayLike, labels: ArrayLike, threshold: float) -> dict[str, float]:
    """Apply a fixed threshold (e.g. calibrated on val) to held-out data."""
    good, ng = _split_scores(scores, labels)
    return {
        "frr": float(np.mean(good >= threshold)) if good.size else float("nan"),
        "escape_rate": float(np.mean(ng < threshold)) if ng.size else float("nan"),
        "n_good": int(good.size),
        "n_ng": int(ng.size),
    }


def threshold_from_good_quantile(good_scores: ArrayLike, max_frr: float) -> float:
    """Zero-NG calibration: the threshold that rejects at most ``max_frr`` of good parts.

    Uses only good images, so it is available before any defect has been seen.
    """
    good = np.asarray(good_scores, dtype=np.float64)
    if good.size == 0:
        raise ValueError("need at least one good score")
    if not 0.0 <= max_frr < 1.0:
        raise ValueError(f"max_frr must be in [0, 1), got {max_frr}")
    # Smallest threshold t such that mean(good >= t) <= max_frr: place t just above the
    # (n_reject + 1)-th highest good score.
    n_reject = int(np.floor(max_frr * good.size + 1e-9))
    kth_highest = float(np.sort(good)[::-1][n_reject])
    return float(np.nextafter(kth_highest, np.inf))


def three_way_decision(scores: ArrayLike, t_low: float, t_high: float) -> np.ndarray:
    """OK below ``t_low``, NG at or above ``t_high``, NOT CLEAR (manual review) in between."""
    if t_low > t_high:
        raise ValueError(f"t_low ({t_low}) must be <= t_high ({t_high})")
    scores = np.asarray(scores, dtype=np.float64)
    verdict = np.full(scores.shape, Verdict.NOT_CLEAR, dtype=np.int8)
    verdict[scores < t_low] = Verdict.OK
    verdict[scores >= t_high] = Verdict.NG
    return verdict


def three_way_report(
    scores: ArrayLike, labels: ArrayLike, t_low: float, t_high: float
) -> dict[str, float]:
    """Rates of a three-way decision. NOT CLEAR parts go to manual review and are not errors."""
    _split_scores(scores, labels)
    labels = np.asarray(labels)
    verdict = three_way_decision(scores, t_low, t_high)
    good, ng = labels == 0, labels == 1
    return {
        "escape_rate": float(np.mean(verdict[ng] == Verdict.OK)) if ng.any() else float("nan"),
        "frr": float(np.mean(verdict[good] == Verdict.NG)) if good.any() else float("nan"),
        "not_clear_rate": float(np.mean(verdict == Verdict.NOT_CLEAR)),
        "false_rejects": int(np.sum(verdict[good] == Verdict.NG)),
        "escapes": int(np.sum(verdict[ng] == Verdict.OK)),
        "not_clear": int(np.sum(verdict == Verdict.NOT_CLEAR)),
    }


@dataclass(frozen=True)
class CostModel:
    """Line cost of inspection errors. Units are the client's currency per part."""

    unit_cost: float
    escape_cost: float
    review_cost: float = 0.0

    def __call__(self, false_rejects: int, escapes: int, not_clear: int = 0) -> float:
        return (
            false_rejects * self.unit_cost
            + escapes * self.escape_cost
            + not_clear * self.review_cost
        )
