"""Tests for the metrics and the time split.

The metric implementations are checked against values computed by hand, because
the whole argument for hand-rolling them is that they can be verified by hand.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from sentinel.config import TrainingSettings
from sentinel.evaluation import evaluate, pr_auc, roc_auc, time_split
from sentinel.features import FEATURE_NAMES
from sentinel.history_mining import LabeledCommit
from sentinel.model import ModelError

START = datetime(2020, 1, 1)


# --- ROC-AUC --------------------------------------------------------------


def test_perfect_ranking_scores_one() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_reversed_ranking_scores_zero() -> None:
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_a_constant_predictor_scores_one_half() -> None:
    """Every score tied means no ranking information at all."""
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_roc_auc_of_a_known_case() -> None:
    """One positive ranked above one of two negatives gives 0.5."""
    assert roc_auc([0, 1, 0], [0.9, 0.5, 0.1]) == pytest.approx(0.5)


def test_roc_auc_is_undefined_with_one_class() -> None:
    assert math.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert math.isnan(roc_auc([1, 1], [0.1, 0.2]))


# --- PR-AUC ---------------------------------------------------------------


def test_perfect_ranking_has_average_precision_one() -> None:
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_average_precision_of_a_hand_computed_case() -> None:
    """Ranked p, n, p: precision 1.0 at recall 0.5, then 2/3 at recall 1.0.

    AP = (0.5 - 0.0) * 1.0 + (1.0 - 0.5) * (2/3) = 0.5 + 0.3333 = 0.8333
    """
    assert pr_auc([1, 0, 1], [0.9, 0.5, 0.1]) == pytest.approx(0.8333333, abs=1e-6)


def test_worst_ranking_average_precision() -> None:
    """Ranked n, n, p: the only positive is last, precision 1/3 at recall 1."""
    assert pr_auc([0, 0, 1], [0.9, 0.5, 0.1]) == pytest.approx(1 / 3)


def test_average_precision_is_undefined_without_positives() -> None:
    assert math.isnan(pr_auc([0, 0], [0.1, 0.2]))


def test_tied_scores_are_treated_as_one_threshold() -> None:
    """All tied means one step: precision equals the base rate."""
    assert pr_auc([1, 0, 0, 1], [0.5] * 4) == pytest.approx(0.5)


# --- the time split -------------------------------------------------------


def row(day: int, label: int = 0, lines: float = 1.0) -> LabeledCommit:
    features = [0.0] * len(FEATURE_NAMES)
    features[FEATURE_NAMES.index("lines_changed")] = lines
    return LabeledCommit(
        sha=f"sha{day:03d}",
        when=START + timedelta(days=day),
        features=features,
        label=label,
    )


def test_split_puts_older_commits_in_train_and_newer_in_test() -> None:
    rows = [row(day) for day in range(10)]
    train, test = time_split(rows, 0.75)

    assert len(train) == 7
    assert len(test) == 3
    assert max(r.when for r in train) < min(r.when for r in test)


def test_split_ignores_input_order() -> None:
    """Rows arriving newest-first must not end up in the wrong halves."""
    rows = list(reversed([row(day) for day in range(10)]))
    train, test = time_split(rows, 0.75)
    assert max(r.when for r in train) < min(r.when for r in test)


def test_split_never_returns_an_empty_side() -> None:
    train, test = time_split([row(0), row(1)], 0.99)
    assert train and test


# --- end to end -----------------------------------------------------------


def test_evaluation_rejects_a_dataset_that_is_too_small() -> None:
    with pytest.raises(ModelError, match="at least 4"):
        evaluate([row(0), row(1)], TrainingSettings())


def test_evaluation_rejects_a_training_half_with_no_positives() -> None:
    """Nothing can be learned from a half that contains no bugs."""
    rows = [row(day, label=0) for day in range(8)] + [row(9, label=1)]
    with pytest.raises(ModelError, match="no bug-inducing commits"):
        evaluate(rows, TrainingSettings())


def test_evaluation_reports_both_metrics_and_a_baseline() -> None:
    # Larger diffs are genuinely riskier here, so both model and baseline
    # should find signal.
    rows = []
    for day in range(40):
        risky = day % 4 == 0
        rows.append(row(day, label=1 if risky else 0, lines=500.0 if risky else 5.0))

    result = evaluate(rows, TrainingSettings(min_data_in_leaf=2, num_rounds=20))

    assert result.train_rows + result.test_rows == 40
    assert result.split_at is not None
    assert 0.0 <= result.baseline.roc_auc <= 1.0
    assert 0.0 <= result.model.roc_auc <= 1.0
    assert 0.0 < result.base_rate < 1.0
