"""Tests for training, persistence, and turning a prediction into reasons."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.config import RiskWeights, TrainingSettings
from sentinel.features import FEATURE_NAMES
from sentinel.model import (
    Contribution,
    FileProbability,
    ModelError,
    as_change_risk,
    as_reason,
    class_weight,
    explain,
    load,
    model_paths,
    predict,
    save,
    train,
    train_with_early_stopping,
)

SETTINGS = TrainingSettings(min_data_in_leaf=2, num_rounds=25)
LINES = FEATURE_NAMES.index("lines_changed")
BUGFIXES = FEATURE_NAMES.index("max_file_bugfixes")


def dataset() -> tuple[list[list[float]], list[int]]:
    """A separable toy problem: big diffs to often-fixed files are risky."""
    vectors: list[list[float]] = []
    labels: list[int] = []
    for index in range(60):
        risky = index % 3 == 0
        row = [0.0] * len(FEATURE_NAMES)
        row[LINES] = 400.0 if risky else 8.0
        row[BUGFIXES] = 12.0 if risky else 0.0
        vectors.append(row)
        labels.append(1 if risky else 0)
    return vectors, labels


# --- class imbalance ------------------------------------------------------


def test_class_weight_is_the_negative_to_positive_ratio() -> None:
    assert class_weight([0] * 90 + [1] * 10) == pytest.approx(9.0)


def test_class_weight_never_drops_below_one() -> None:
    assert class_weight([1] * 9 + [0]) == 1.0


def test_no_positives_is_a_clear_error_not_a_useless_model() -> None:
    with pytest.raises(ModelError, match="no bug-inducing commits"):
        class_weight([0, 0, 0])


# --- training -------------------------------------------------------------


def test_training_learns_the_pattern() -> None:
    vectors, labels = dataset()
    booster = train(vectors, labels, SETTINGS)

    risky = [0.0] * len(FEATURE_NAMES)
    risky[LINES], risky[BUGFIXES] = 400.0, 12.0
    safe = [0.0] * len(FEATURE_NAMES)
    safe[LINES], safe[BUGFIXES] = 8.0, 0.0

    risky_score, safe_score = predict(booster, [risky, safe])
    assert risky_score > safe_score


def test_training_is_deterministic() -> None:
    vectors, labels = dataset()
    first = predict(train(vectors, labels, SETTINGS), vectors)
    second = predict(train(vectors, labels, SETTINGS), vectors)
    assert first == second


def test_training_rejects_mismatched_input() -> None:
    with pytest.raises(ModelError, match="mismatch"):
        train([[0.0] * len(FEATURE_NAMES)], [0, 1], SETTINGS)

    with pytest.raises(ModelError, match="no training rows"):
        train([], [], SETTINGS)


def test_early_stopping_picks_a_round_count_and_still_predicts() -> None:
    vectors, labels = dataset()
    booster, rounds = train_with_early_stopping(vectors, labels, SETTINGS)

    assert rounds >= 1
    risky = [0.0] * len(FEATURE_NAMES)
    risky[LINES], risky[BUGFIXES] = 400.0, 12.0
    safe = [0.0] * len(FEATURE_NAMES)
    safe[LINES], safe[BUGFIXES] = 8.0, 0.0
    risky_score, safe_score = predict(booster, [risky, safe])
    assert risky_score > safe_score


def test_early_stopping_falls_back_when_there_is_too_little_data() -> None:
    """Four rows cannot support a holdout; fitting them directly is correct."""
    vectors = [[0.0] * len(FEATURE_NAMES) for _ in range(4)]
    for index, row in enumerate(vectors):
        row[LINES] = float(index * 100)
    labels = [0, 0, 1, 1]

    booster, rounds = train_with_early_stopping(vectors, labels, SETTINGS)
    assert rounds == SETTINGS.num_rounds
    assert len(predict(booster, vectors)) == 4


def test_early_stopping_falls_back_when_a_side_has_one_class() -> None:
    """A holdout of all-clean commits cannot tell you when to stop."""
    vectors = []
    labels = []
    for index in range(40):
        row = [0.0] * len(FEATURE_NAMES)
        row[LINES] = float(index)
        vectors.append(row)
        # Every positive sits in the oldest part, so the newest slice is
        # single-class and unusable as a stopping criterion.
        labels.append(1 if index < 10 else 0)

    booster, rounds = train_with_early_stopping(vectors, labels, SETTINGS)
    assert rounds == SETTINGS.num_rounds
    assert len(predict(booster, vectors)) == 40


def test_predicting_nothing_returns_nothing() -> None:
    vectors, labels = dataset()
    assert predict(train(vectors, labels, SETTINGS), []) == []


# --- persistence ----------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    vectors, labels = dataset()
    booster = train(vectors, labels, SETTINGS)

    save(booster, tmp_path, SETTINGS, rows=60, positives=20, commits_considered=100)
    loaded = load(tmp_path, SETTINGS)

    assert loaded is not None
    assert loaded.rows == 60
    assert loaded.positives == 20
    assert loaded.positive_rate == pytest.approx(1 / 3)
    assert loaded.is_compatible
    assert predict(loaded.booster, vectors) == predict(booster, vectors)


def test_loading_an_untrained_repo_returns_none(tmp_path: Path) -> None:
    """Not trained yet is normal operation, not an error."""
    assert load(tmp_path, SETTINGS) is None


def test_a_model_from_a_different_feature_set_is_flagged(tmp_path: Path) -> None:
    """The quiet failure mode: columns line up numerically but mean nothing."""
    vectors, labels = dataset()
    save(
        train(vectors, labels, SETTINGS),
        tmp_path,
        SETTINGS,
        rows=60,
        positives=20,
        commits_considered=100,
    )

    _model_file, metadata_file = model_paths(tmp_path, SETTINGS)
    metadata_file.write_text('{"feature_names": ["only_one_feature"]}', encoding="utf-8")

    loaded = load(tmp_path, SETTINGS)
    assert loaded is not None
    assert not loaded.is_compatible


def test_a_corrupt_model_file_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    model_file, _ = model_paths(tmp_path, SETTINGS)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("this is not a lightgbm model", encoding="utf-8")

    assert load(tmp_path, SETTINGS) is None


# --- explanations ---------------------------------------------------------


def test_shap_names_the_feature_that_drove_the_prediction() -> None:
    vectors, labels = dataset()
    booster = train(vectors, labels, SETTINGS)

    risky = [0.0] * len(FEATURE_NAMES)
    risky[LINES], risky[BUGFIXES] = 400.0, 12.0

    contributions = explain(booster, risky, top=3)

    assert contributions
    assert {c.feature for c in contributions} & {"lines_changed", "max_file_bugfixes"}
    # Sorted by magnitude of effect.
    magnitudes = [abs(c.impact) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_a_contribution_renders_as_a_reason() -> None:
    reason = as_reason(Contribution("lines_changed", "Size of the change", 400.0, 0.42))

    assert reason.rule == "model:lines_changed"
    assert reason.points == 42
    assert "400" in reason.detail
    assert "+0.420" in reason.detail


def test_model_risk_has_the_same_shape_as_a_rule_result() -> None:
    weights = RiskWeights()
    contributions = [Contribution("max_file_bugfixes", "Bug-fix history", 12.0, 0.5)]
    files = [
        FileProbability("app/calm.py", 0.10, 2, 0),
        FileProbability("app/hot.py", 0.90, 200, 30, top=contributions[0]),
    ]

    risk = as_change_risk(0.72, contributions, files, weights)

    assert risk.scoring_method == "model"
    assert risk.score == 72
    assert risk.band == "high"
    assert risk.recommendation  # a hint was found for the model feature
    # Files ranked riskiest first, each with its own band.
    assert [f.path for f in risk.files] == ["app/hot.py", "app/calm.py"]
    assert risk.files[0].score == 90
    assert risk.files[0].band == "high"
    assert risk.files[1].band == "low"


def test_a_negative_contribution_is_kept_as_risk_reducing() -> None:
    risk = as_change_risk(
        0.20,
        [Contribution("test_files_changed", "Test files touched", 3.0, -0.30)],
        [],
        RiskWeights(),
    )
    assert risk.reasons[0].points == -30
