import numpy as np
import pytest

from zerong.metrics.factory import (
    CostModel,
    Verdict,
    frr_at_recall,
    frr_at_zero_escape,
    rates_at_threshold,
    three_way_decision,
    three_way_report,
    threshold_from_good_quantile,
)


def test_frr_at_zero_escape_perfect_separation():
    op = frr_at_zero_escape([0.1, 0.2, 0.3, 0.8, 0.9], [0, 0, 0, 1, 1])
    assert op.threshold == 0.8
    assert op.frr == 0.0
    assert op.recall == 1.0


def test_frr_at_zero_escape_overlap():
    # Lowest NG score is 0.25: good parts at 0.3 and 0.5 are rejected -> FRR 2/4.
    scores = [0.1, 0.2, 0.3, 0.5, 0.25, 0.9]
    labels = [0, 0, 0, 0, 1, 1]
    op = frr_at_zero_escape(scores, labels)
    assert op.threshold == 0.25
    assert op.frr == 0.5


def test_ties_at_threshold_count_as_rejected():
    op = frr_at_zero_escape([0.5, 0.4, 0.5], [0, 0, 1])
    assert op.frr == 0.5


def test_frr_at_lower_recall():
    # 4 NG, recall 0.75 -> may miss the single lowest NG (0.2); threshold becomes 0.6.
    scores = [0.1, 0.3, 0.5, 0.7, 0.2, 0.6, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    op = frr_at_recall(scores, labels, recall=0.75)
    assert op.threshold == 0.6
    assert op.frr == 0.25
    assert op.recall == 0.75


def test_frr_input_validation():
    with pytest.raises(ValueError):
        frr_at_zero_escape([0.1, 0.2], [0, 0])
    with pytest.raises(ValueError):
        frr_at_zero_escape([0.1, 0.2], [0, 2])
    with pytest.raises(ValueError):
        frr_at_zero_escape([0.1, float("nan")], [0, 1])
    with pytest.raises(ValueError):
        frr_at_recall([0.1, 0.2], [0, 1], recall=0.0)


def test_rates_at_threshold():
    r = rates_at_threshold([0.1, 0.6, 0.4, 0.9], [0, 0, 1, 1], threshold=0.5)
    assert r["frr"] == 0.5
    assert r["escape_rate"] == 0.5


@pytest.mark.parametrize("max_frr", [0.0, 0.05, 0.1, 0.25])
def test_threshold_from_good_quantile_respects_budget(max_frr):
    good = np.random.default_rng(0).normal(size=200)
    t = threshold_from_good_quantile(good, max_frr)
    assert np.mean(good >= t) <= max_frr
    # And it is the tightest such threshold: one more rejection would exceed the budget.
    rejected_if_lower = np.sum(good >= np.nextafter(t, -np.inf))
    assert rejected_if_lower / good.size > max_frr or t <= good.min()


def test_threshold_from_good_quantile_zero_frr_is_above_max():
    t = threshold_from_good_quantile([0.1, 0.2, 0.3], 0.0)
    assert t > 0.3


def test_three_way_decision():
    v = three_way_decision([0.1, 0.4, 0.5, 0.9], t_low=0.3, t_high=0.5)
    assert v.tolist() == [Verdict.OK, Verdict.NOT_CLEAR, Verdict.NG, Verdict.NG]
    with pytest.raises(ValueError):
        three_way_decision([0.1], t_low=0.6, t_high=0.5)


def test_three_way_collapses_to_binary_when_thresholds_equal():
    scores = [0.1, 0.4, 0.5, 0.9]
    assert Verdict.NOT_CLEAR not in three_way_decision(scores, 0.5, 0.5).tolist()


def test_three_way_report():
    scores = [0.1, 0.35, 0.6, 0.2, 0.4, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    r = three_way_report(scores, labels, t_low=0.3, t_high=0.5)
    assert r["escapes"] == 1  # NG at 0.2
    assert r["false_rejects"] == 1  # good at 0.6
    assert r["not_clear"] == 2  # 0.35 and 0.4
    assert r["escape_rate"] == pytest.approx(1 / 3)
    assert r["not_clear_rate"] == pytest.approx(2 / 6)


def test_cost_model():
    cost = CostModel(unit_cost=2.0, escape_cost=500.0, review_cost=0.5)
    assert cost(false_rejects=10, escapes=1, not_clear=4) == 20 + 500 + 2
    assert CostModel(1.0, 100.0)(3, 0) == 3.0
