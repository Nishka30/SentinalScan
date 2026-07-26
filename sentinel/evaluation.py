"""Honest, time-based evaluation.

Two rules, both non-negotiable:

* **Split by time, never at random.** Train on older commits, test on newer.
  A random split lets the model learn from commits that came *after* the ones
  it is tested on, and reports an accuracy the tool will never reproduce in
  practice. It is the single easiest way to fake a good result.
* **Report metrics that suit rare events, against a baseline.** Bug-inducing
  commits are a few percent of history, so plain accuracy is ~95% for a model
  that predicts "clean" every time. ROC-AUC and PR-AUC say something; and both
  are reported next to a lines-changed-only baseline, because a model that
  cannot beat "big diffs are risky" has not earned its complexity.

The metrics are implemented here rather than imported from scikit-learn, which
is not in this project's dependency list. They are a dozen lines each, they are
unit-tested against hand-computed values, and being able to explain exactly how
your headline number was produced is worth more than the import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from sentinel.config import TrainingSettings
from sentinel.features import FEATURE_NAMES
from sentinel.history_mining import LabeledCommit
from sentinel.model import ModelError, predict, train_with_early_stopping

#: The baseline predicts risk from the size of the diff and nothing else.
BASELINE_FEATURE = "lines_changed"


@dataclass(frozen=True)
class Metrics:
    """Ranking quality for a rare positive class."""

    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class EvaluationResult:
    """Everything needed to judge whether the model is worth using."""

    train_rows: int
    test_rows: int
    train_positives: int
    test_positives: int
    split_at: datetime | None
    model: Metrics
    baseline: Metrics

    @property
    def base_rate(self) -> float:
        """Positive rate in the test set — what a coin flip would score on PR-AUC."""
        return self.test_positives / self.test_rows if self.test_rows else 0.0

    @property
    def beats_baseline(self) -> bool:
        return self.model.pr_auc > self.baseline.pr_auc


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Area under the ROC curve, via the rank (Mann-Whitney U) identity.

    Equivalent to the probability that a randomly chosen positive is ranked
    above a randomly chosen negative. Ties get averaged ranks, which is what
    makes a constant predictor score exactly 0.5 instead of 1.0.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        stop = index
        while stop < len(order) and scores[order[stop]] == scores[order[index]]:
            stop += 1
        # Ranks are 1-based; every tied entry takes the group's average rank.
        average = (index + 1 + stop) / 2.0
        for position in range(index, stop):
            ranks[order[position]] = average
        index = stop

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels: list[int], scores: list[float]) -> float:
    """Average precision: the step-wise area under the precision-recall curve.

    Preferred over ROC-AUC when positives are rare, because it ignores the vast
    pool of true negatives that flatters ROC.
    """
    total_positives = sum(labels)
    if total_positives == 0:
        return math.nan

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0

    index = 0
    while index < len(order):
        stop = index
        # All rows sharing a score are one threshold, so they move together.
        while stop < len(order) and scores[order[stop]] == scores[order[index]]:
            if labels[order[stop]] == 1:
                true_positives += 1
            else:
                false_positives += 1
            stop += 1

        recall = true_positives / total_positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = stop

    return area


def time_split(
    rows: list[LabeledCommit], fraction: float
) -> tuple[list[LabeledCommit], list[LabeledCommit]]:
    """Oldest `fraction` for training, newest remainder for testing."""
    ordered = sorted(rows, key=lambda r: r.when)
    cut = int(len(ordered) * fraction)
    # Never hand back an empty side when there is data to split.
    cut = max(1, min(cut, len(ordered) - 1)) if len(ordered) > 1 else len(ordered)
    return ordered[:cut], ordered[cut:]


def evaluate(rows: list[LabeledCommit], settings: TrainingSettings) -> EvaluationResult:
    """Train on the older commits, score the newer ones, compare to the baseline."""
    if len(rows) < 4:
        raise ModelError(f"need at least 4 labeled commits to evaluate, got {len(rows)}")

    train_rows, test_rows = time_split(rows, settings.train_fraction)

    train_labels = [r.label for r in train_rows]
    test_labels = [r.label for r in test_rows]

    if sum(train_labels) == 0:
        raise ModelError(
            "the training half contains no bug-inducing commits — "
            "widen --max-commits so the older period includes some"
        )

    # Rounds are chosen from a holdout carved out of the training half, never
    # from the test half — tuning against the test set is the other easy way to
    # report a number the tool will not reproduce.
    booster, _rounds = train_with_early_stopping(
        [r.features for r in train_rows], train_labels, settings
    )
    model_scores = predict(booster, [r.features for r in test_rows])

    baseline_index = FEATURE_NAMES.index(BASELINE_FEATURE)
    baseline_scores = [r.features[baseline_index] for r in test_rows]

    return EvaluationResult(
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        train_positives=sum(train_labels),
        test_positives=sum(test_labels),
        split_at=test_rows[0].when if test_rows else None,
        model=Metrics(
            roc_auc=roc_auc(test_labels, model_scores),
            pr_auc=pr_auc(test_labels, model_scores),
        ),
        baseline=Metrics(
            roc_auc=roc_auc(test_labels, baseline_scores),
            pr_auc=pr_auc(test_labels, baseline_scores),
        ),
    )
