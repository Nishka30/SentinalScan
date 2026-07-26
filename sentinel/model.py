"""Train and predict with LightGBM, and explain the prediction with SHAP.

Persists to a plain text model file plus a small JSON sidecar. No database and
no server — a trained Sentinel is a file you can delete and rebuild.

The sidecar exists to make the model self-describing. Loading a model trained
against a different feature list is the quiet failure mode of any ML feature:
the columns still line up numerically and the predictions are nonsense. Storing
the feature names lets that be detected instead of guessed at.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from sentinel import __version__
from sentinel.config import RiskWeights, TrainingSettings
from sentinel.features import FEATURE_LABELS, FEATURE_NAMES
from sentinel.risk_rules import ChangeRisk, FileRisk, Reason, band_for, recommend

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """The model cannot be trained, loaded, or trusted."""


@dataclass(frozen=True)
class TrainedModel:
    """A booster plus the metadata needed to use it safely."""

    booster: lgb.Booster
    feature_names: tuple[str, ...]
    rows: int
    positives: int
    trained_at: str
    sentinel_version: str
    commits_considered: int = 0

    @property
    def positive_rate(self) -> float:
        return self.positives / self.rows if self.rows else 0.0

    @property
    def is_compatible(self) -> bool:
        """False when the saved feature list no longer matches the code."""
        return self.feature_names == FEATURE_NAMES


@dataclass(frozen=True)
class Contribution:
    """One feature's SHAP contribution to a single prediction."""

    feature: str
    label: str
    value: float
    #: Signed log-odds contribution. Positive pushes towards "risky".
    impact: float


def model_paths(repo_root: Path, settings: TrainingSettings) -> tuple[Path, Path]:
    """Where the model and its sidecar live for a given repository."""
    directory = repo_root / settings.model_dir
    return directory / settings.model_file, directory / settings.metadata_file


def class_weight(labels: list[int]) -> float:
    """Weight for the positive class: how many negatives per positive.

    Bug-inducing commits are a small minority. Left unweighted, LightGBM
    maximises accuracy by predicting "clean" for everything, which scores well
    and is useless.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        raise ModelError(
            "no bug-inducing commits were found, so there is nothing to learn. "
            "Try a larger --max-commits, or check that commit messages mention fixes."
        )
    return max(1.0, negatives / positives)


def train(
    vectors: list[list[float]],
    labels: list[int],
    settings: TrainingSettings,
    *,
    rounds: int | None = None,
    validation: tuple[list[list[float]], list[int]] | None = None,
) -> lgb.Booster:
    """Fit a gradient-boosted classifier. Deterministic for a given seed.

    Pass `validation` to stop boosting once that slice stops improving, or
    `rounds` to fit a fixed number of iterations.
    """
    if not vectors:
        raise ModelError("no training rows")
    if len(vectors) != len(labels):
        raise ModelError("feature/label length mismatch")

    dataset = lgb.Dataset(
        np.asarray(vectors, dtype=np.float64),
        label=np.asarray(labels, dtype=np.int32),
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": settings.early_stopping_metric,
        "verbosity": -1,
        "seed": settings.seed,
        "num_leaves": settings.num_leaves,
        "learning_rate": settings.learning_rate,
        "min_data_in_leaf": settings.min_data_in_leaf,
        "scale_pos_weight": class_weight(labels),
        # Determinism: without these, thread scheduling changes the tree splits
        # and two runs on the same data disagree.
        "deterministic": True,
        "num_threads": 1,
        "force_row_wise": True,
    }

    valid_sets: list[lgb.Dataset] = []
    callbacks: list = []
    if validation is not None:
        valid_vectors, valid_labels = validation
        valid_sets = [
            lgb.Dataset(
                np.asarray(valid_vectors, dtype=np.float64),
                label=np.asarray(valid_labels, dtype=np.int32),
                feature_name=list(FEATURE_NAMES),
                reference=dataset,
                free_raw_data=False,
            )
        ]
        callbacks = [lgb.early_stopping(settings.early_stopping_rounds, verbose=False)]

    return lgb.train(
        params,
        dataset,
        num_boost_round=settings.num_rounds if rounds is None else rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def train_with_early_stopping(
    vectors: list[list[float]],
    labels: list[int],
    settings: TrainingSettings,
) -> tuple[lgb.Booster, int]:
    """Choose the number of boosting rounds from held-out data, then use it all.

    Two passes on purpose. The first fits the older part and watches the newest
    part to find where improvement stops; the second refits everything for that
    many rounds, so no data is wasted and the shipped model is regularised the
    same way the evaluated one was.

    The holdout is the *newest* slice, never a random sample — the same reason
    `evaluation` splits by time.
    """
    if len(vectors) < 10:
        # Too small to hold anything back; fit it and move on.
        return train(vectors, labels, settings), settings.num_rounds

    cut = max(1, int(len(vectors) * settings.holdout_fraction))
    fit_vectors, holdout_vectors = vectors[:cut], vectors[cut:]
    fit_labels, holdout_labels = labels[:cut], labels[cut:]

    # Early stopping needs both classes present on each side to mean anything.
    if not holdout_vectors or sum(fit_labels) == 0 or sum(holdout_labels) == 0:
        return train(vectors, labels, settings), settings.num_rounds

    probe = train(
        fit_vectors,
        fit_labels,
        settings,
        validation=(holdout_vectors, holdout_labels),
    )
    rounds = probe.best_iteration or settings.num_rounds
    return train(vectors, labels, settings, rounds=rounds), rounds


def save(
    booster: lgb.Booster,
    repo_root: Path,
    settings: TrainingSettings,
    *,
    rows: int,
    positives: int,
    commits_considered: int,
) -> Path:
    """Write the model and its sidecar, creating the directory if needed."""
    model_file, metadata_file = model_paths(repo_root, settings)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    # Make the directory invisible to git. Without this, the next `sentinel
    # diff` in the analyzed repository sees a 1,500-line untracked model file,
    # treats it as part of the change, and lets it dominate the score.
    gitignore = model_file.parent / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Sentinel's own artifacts\n*\n", encoding="utf-8")

    booster.save_model(str(model_file))
    metadata_file.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "rows": rows,
                "positives": positives,
                "commits_considered": commits_considered,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "sentinel_version": __version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_file


def load(repo_root: Path, settings: TrainingSettings) -> TrainedModel | None:
    """Load a trained model, or None when there is not a usable one.

    Returns None rather than raising for the ordinary "not trained yet" case,
    because falling back to the rules is normal operation, not an error.
    """
    model_file, metadata_file = model_paths(repo_root, settings)
    if not model_file.is_file():
        return None

    try:
        booster = lgb.Booster(model_file=str(model_file))
    except Exception as exc:
        logger.warning("ignoring unreadable model at %s: %s", model_file, exc)
        return None

    metadata: dict = {}
    if metadata_file.is_file():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ignoring unreadable model metadata: %s", exc)

    return TrainedModel(
        booster=booster,
        feature_names=tuple(metadata.get("feature_names", FEATURE_NAMES)),
        rows=int(metadata.get("rows", 0)),
        positives=int(metadata.get("positives", 0)),
        trained_at=str(metadata.get("trained_at", "unknown")),
        sentinel_version=str(metadata.get("sentinel_version", "unknown")),
        commits_considered=int(metadata.get("commits_considered", 0)),
    )


def predict(booster: lgb.Booster, vectors: list[list[float]]) -> list[float]:
    """Probability of being bug-inducing, one per row."""
    if not vectors:
        return []
    raw = booster.predict(np.asarray(vectors, dtype=np.float64))
    return [float(p) for p in np.asarray(raw).ravel()]


def explain(
    booster: lgb.Booster, vector: list[float], *, top: int = 6
) -> list[Contribution]:
    """The features that moved this prediction most, largest effect first.

    Uses SHAP's TreeExplainer. Contributions are in log-odds, which is why they
    are reported as signed impact rather than as points out of 100 — rescaling
    them to look like the rule engine's points would be inventing precision.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - shap is a declared dependency
        logger.warning("shap unavailable, no model reasons: %s", exc)
        return []

    try:
        with warnings.catch_warnings():
            # shap warns that its binary-classifier output shape changed between
            # releases. We normalise every shape it can return just below, so
            # the warning is noise on the user's terminal for something already
            # handled. Filtered by message, not blanket-silenced.
            warnings.filterwarnings(
                "ignore",
                message=".*output has changed to a list of ndarray.*",
                category=UserWarning,
            )
            explainer = shap.TreeExplainer(booster)
            raw = explainer.shap_values(np.asarray([vector], dtype=np.float64))
    except Exception as exc:
        logger.warning("could not compute SHAP values: %s", exc)
        return []

    # shap's output shape for binary classifiers has changed between releases:
    # older versions return one array per class, newer ones a single array, and
    # some return a trailing class axis. Normalise to one row of per-feature
    # values for the positive class rather than trusting any one of them.
    if isinstance(raw, list):
        raw = raw[-1]
    row = np.asarray(raw)[0]
    while row.ndim > 1:
        row = row[..., -1]

    contributions = [
        Contribution(
            feature=name,
            label=FEATURE_LABELS.get(name, name),
            value=vector[index],
            impact=float(row[index]),
        )
        for index, name in enumerate(FEATURE_NAMES)
        if index < len(row)
    ]
    contributions.sort(key=lambda c: abs(c.impact), reverse=True)
    return [c for c in contributions[:top] if c.impact != 0.0]


# --------------------------------------------------------------------------
# Turning a prediction into the same result shape the rules produce
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileProbability:
    """One file's own model score, for ranking within a change."""

    path: str
    probability: float
    lines_added: int
    lines_deleted: int
    top: Contribution | None = None


def as_reason(contribution: Contribution, *, path: str | None = None) -> Reason:
    """Render a SHAP contribution as a `Reason` the report can already print.

    `points` holds the log-odds impact scaled by 100 and rounded. That keeps one
    integer column in the report for both engines while staying honest: the sign
    and the ordering are meaningful, and the raw value is in the evidence text.
    """
    return Reason(
        rule=f"model:{contribution.feature}",
        label=contribution.label,
        points=int(round(contribution.impact * 100)),
        detail=f"value {contribution.value:g}; log-odds {contribution.impact:+.3f}",
        path=path,
    )


def as_change_risk(
    probability: float,
    contributions: list[Contribution],
    files: list[FileProbability],
    weights: RiskWeights,
) -> ChangeRisk:
    """Assemble a model-scored `ChangeRisk`, identical in shape to a rule-scored one."""
    score = int(round(probability * weights.max_score))
    band = band_for(score, weights)
    reasons = tuple(as_reason(c) for c in contributions)

    ranked = sorted(files, key=lambda f: (-f.probability, f.path))
    file_risks = tuple(
        FileRisk(
            path=f.path,
            score=int(round(f.probability * weights.max_score)),
            band=band_for(int(round(f.probability * weights.max_score)), weights),
            reasons=() if f.top is None else (as_reason(f.top, path=f.path),),
            lines_added=f.lines_added,
            lines_deleted=f.lines_deleted,
        )
        for f in ranked
    )

    return ChangeRisk(
        score=score,
        band=band,
        reasons=reasons,
        files=file_risks,
        recommendation=recommend(band, reasons),
        scoring_method="model",
    )
