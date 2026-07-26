"""Transparent, points-based risk scoring.

Every rule is a small pure function that either stays silent or returns a
`Reason` carrying its own explanation and its own contribution to the score.
That shape is the whole point: the report is not a separate rendering of the
score, it *is* the list of rules that fired.

Phase 2 replaces the number with a learned probability. These rules stay, as
the fallback for repositories with too little history to train on — and as the
thing a reviewer can argue with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sentinel.config import DistributionSettings, RiskWeights
from sentinel.features import Change, FileChange, RepoDistribution

#: Which band a score falls into. Ordered least to most alarming.
BANDS = ("low", "medium", "high")

#: Concrete advice per rule, so the recommendation says what to *do* rather
#: than restating the score.
RULE_HINTS: dict[str, str] = {
    "missing_tests": "add a test for the changed file before shipping",
    "stale_tests": "the tests for this file were not touched — check they still cover it",
    "hot_file": "this file has been fixed repeatedly; deploy where you can roll back fast",
    "high_complexity": "the most complex function here deserves a second reviewer",
    "author_new_to_file": "ask someone who has worked in this file to review",
    "low_ownership": "loop in the file's main author",
    "large_edit": "consider splitting this into smaller, separately deployable changes",
    "broad_change": "this touches a lot of the tree — stage the rollout",
    "weekend_deploy": "avoid shipping when the people who can fix it are away",
    "late_night_deploy": "hold until working hours if you can",
    "churn": "this file changes constantly; expect it to need another change soon",
    # Model-scored reasons are keyed by feature, so the same advice table serves
    # both engines and a recommendation never comes back empty.
    "model:max_file_bugfixes": "these files have been fixed repeatedly; keep a rollback ready",
    "model:mean_file_bugfixes": "the files touched have a poor track record; deploy carefully",
    "model:lines_changed": "consider splitting this into smaller, separately deployable changes",
    "model:lines_added": "consider splitting this into smaller, separately deployable changes",
    "model:files_changed": "this touches a lot of the tree — stage the rollout",
    "model:folders_touched": "the change spans several areas; check each one has an owner",
    "model:test_files_changed": "add or update tests alongside this change",
    "model:code_files_changed": "review each source file touched, not just the diff summary",
    "model:files_author_never_touched": "ask someone who has worked in these files to review",
    "model:min_author_ownership": "loop in the main author of the least familiar file",
    "model:mean_author_ownership": "get a review from someone who knows this area",
    "model:max_complexity": "the most complex function here deserves a second reviewer",
    "model:max_file_commits": "these files change constantly; expect another change soon",
    "model:is_weekend": "avoid shipping when the people who can fix it are away",
    "model:hour_of_day": "hold until working hours if you can",
    "model:new_files": "new files have no track record — watch them after release",
}


@dataclass(frozen=True)
class Reason:
    """One rule that fired, with the evidence and the points it contributed."""

    rule: str
    label: str
    points: int
    detail: str = ""
    #: Set when the reason is about a specific file rather than the change.
    path: str | None = None


@dataclass(frozen=True)
class FileRisk:
    """Score for a single file and the reasons behind it."""

    path: str
    score: int
    band: str
    reasons: tuple[Reason, ...]
    lines_added: int = 0
    lines_deleted: int = 0


@dataclass(frozen=True)
class ChangeRisk:
    """The answer: how risky is this change, why, and what to do about it."""

    score: int
    band: str
    reasons: tuple[Reason, ...]
    files: tuple[FileRisk, ...] = field(default_factory=tuple)
    recommendation: str = ""
    #: "rules" or "model" — which engine produced `score`.
    scoring_method: str = "rules"
    #: True when the thresholds came from the repository's own percentiles.
    relative_thresholds: bool = False


def calibrate(
    weights: RiskWeights,
    distribution: RepoDistribution,
    settings: DistributionSettings,
) -> tuple[RiskWeights, bool]:
    """Raise the bug-fix and churn thresholds to this repository's percentiles.

    Returns the weights to score with, and whether calibration actually applied.

    The configured absolutes act as floors via `max`, for two reasons: a young
    repository whose 90th percentile is one bug fix must not hand out the top
    tier for a single fix, and a threshold should never get *looser* than the
    policy written in config.
    """
    if not settings.enabled or not distribution.meaningful:
        return weights, False

    return (
        weights.model_copy(
            update={
                "hot_bugfixes": max(weights.hot_bugfixes, distribution.bugfix_hot),
                "very_hot_bugfixes": max(weights.very_hot_bugfixes, distribution.bugfix_very_hot),
                "churn_commits": max(weights.churn_commits, distribution.churn_high),
            }
        ),
        True,
    )


# --------------------------------------------------------------------------
# File-level rules
# --------------------------------------------------------------------------


def _rule_large_edit(f: FileChange, w: RiskWeights) -> list[Reason]:
    """Big edits are harder to review, and unreviewed lines are where bugs live."""
    lines = f.lines_changed
    if lines >= w.file_large_lines:
        return [
            Reason(
                "large_edit",
                "Large edit to a single file",
                w.file_large_points,
                f"{lines} lines changed (threshold {w.file_large_lines})",
                f.path,
            )
        ]
    if lines >= w.file_moderate_lines:
        return [
            Reason(
                "large_edit",
                "Sizeable edit to a single file",
                w.file_moderate_points,
                f"{lines} lines changed (threshold {w.file_moderate_lines})",
                f.path,
            )
        ]
    return []


def _rule_hot_file(f: FileChange, w: RiskWeights) -> list[Reason]:
    """Files that needed many fixes tend to need another one.

    This is the strongest signal available without a model, which is why it
    carries the most points.
    """
    if f.history is None:
        return []
    fixes = f.history.bugfix_commits
    if fixes >= w.very_hot_bugfixes:
        return [
            Reason(
                "hot_file",
                "File has a long history of bug fixes",
                w.very_hot_points,
                f"{fixes} past bug-fix commits (threshold {w.very_hot_bugfixes})",
                f.path,
            )
        ]
    if fixes >= w.hot_bugfixes:
        return [
            Reason(
                "hot_file",
                "File has needed bug fixes before",
                w.hot_points,
                f"{fixes} past bug-fix commits (threshold {w.hot_bugfixes})",
                f.path,
            )
        ]
    return []


def _rule_churn(f: FileChange, w: RiskWeights) -> list[Reason]:
    """High churn without bug fixes is milder, but still instability."""
    if f.history is None or f.history.total_commits < w.churn_commits:
        return []
    return [
        Reason(
            "churn",
            "File changes very frequently",
            w.churn_points,
            f"{f.history.total_commits} commits touched it (threshold {w.churn_commits})",
            f.path,
        )
    ]


def _rule_tests(f: FileChange, w: RiskWeights) -> list[Reason]:
    """No tests is worse than untouched tests, so the two carry different points."""
    if f.tests.is_test_file or not f.tests.is_code:
        return []
    if not f.tests.has_tests:
        return [
            Reason(
                "missing_tests",
                "No test file found for this file",
                w.no_test_file_points,
                "searched the usual naming conventions and test directories",
                f.path,
            )
        ]
    if not f.tests.tests_changed:
        return [
            Reason(
                "stale_tests",
                "Tests exist but were not changed",
                w.stale_test_points,
                f"unchanged: {', '.join(f.tests.test_paths[:3])}",
                f.path,
            )
        ]
    return []


def _rule_complexity(f: FileChange, w: RiskWeights) -> list[Reason]:
    """Branch-heavy code has more paths than any reviewer holds in their head."""
    if f.complexity is None:
        return []
    ccn = f.complexity.max_ccn
    if ccn >= w.very_high_ccn:
        return [
            Reason(
                "high_complexity",
                "Contains a very complex function",
                w.very_high_ccn_points,
                f"peak cyclomatic complexity {ccn} (threshold {w.very_high_ccn})",
                f.path,
            )
        ]
    if ccn >= w.high_ccn:
        return [
            Reason(
                "high_complexity",
                "Contains a complex function",
                w.high_ccn_points,
                f"peak cyclomatic complexity {ccn} (threshold {w.high_ccn})",
                f.path,
            )
        ]
    return []


def _rule_familiarity(f: FileChange, w: RiskWeights) -> list[Reason]:
    """Editing code you have never touched is riskier than editing your own.

    Brand-new files are exempt: nobody is familiar with a file that does not
    exist yet, and penalising that would just tax new code.
    """
    if f.history is None:
        return []
    if f.author_commits == 0:
        return [
            Reason(
                "author_new_to_file",
                "Author has never changed this file before",
                w.new_to_file_points,
                f"{f.author} has 0 of its {f.history.total_commits} past commits",
                f.path,
            )
        ]
    if f.ownership < w.low_ownership:
        return [
            Reason(
                "low_ownership",
                "Author knows this file only slightly",
                w.low_ownership_points,
                f"{f.author} made {f.author_commits} of {f.history.total_commits} "
                f"past commits ({f.ownership:.0%})",
                f.path,
            )
        ]
    return []


FILE_RULES: tuple[Callable[[FileChange, RiskWeights], list[Reason]], ...] = (
    _rule_hot_file,
    _rule_tests,
    _rule_complexity,
    _rule_familiarity,
    _rule_large_edit,
    _rule_churn,
)


# --------------------------------------------------------------------------
# Change-level rules
# --------------------------------------------------------------------------


def _rule_broad_change(c: Change, w: RiskWeights) -> list[Reason]:
    """Breadth is its own risk: more surfaces to break, more places to look."""
    reasons: list[Reason] = []
    if len(c.files) >= w.broad_files:
        reasons.append(
            Reason(
                "broad_change",
                "Change touches many files",
                w.broad_files_points,
                f"{len(c.files)} files (threshold {w.broad_files})",
            )
        )
    if c.total_lines_changed >= w.broad_lines:
        reasons.append(
            Reason(
                "broad_change",
                "Change is large overall",
                w.broad_lines_points,
                f"{c.total_lines_changed} lines across {len(c.files)} files",
            )
        )
    return reasons


def _rule_timing(c: Change, w: RiskWeights) -> list[Reason]:
    """Not superstition: it is about who is around to notice and roll back."""
    reasons: list[Reason] = []
    if c.when.weekday() >= 5:
        reasons.append(
            Reason(
                "weekend_deploy",
                "Shipping at the weekend",
                w.weekend_points,
                c.when.strftime("%A %H:%M"),
            )
        )
    hour = c.when.hour
    if hour >= w.late_night_start_hour or hour < w.late_night_end_hour:
        reasons.append(
            Reason(
                "late_night_deploy",
                "Shipping outside working hours",
                w.late_night_points,
                c.when.strftime("%A %H:%M"),
            )
        )
    return reasons


CHANGE_RULES: tuple[Callable[[Change, RiskWeights], list[Reason]], ...] = (
    _rule_broad_change,
    _rule_timing,
)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_file(f: FileChange, weights: RiskWeights) -> FileRisk:
    """Score one file: the sum of its fired rules, capped."""
    reasons = [r for rule in FILE_RULES for r in rule(f, weights)]
    reasons.sort(key=lambda r: r.points, reverse=True)
    score = min(weights.max_score, sum(r.points for r in reasons))
    return FileRisk(
        path=f.path,
        score=score,
        band=band_for(score, weights),
        reasons=tuple(reasons),
        lines_added=f.lines_added,
        lines_deleted=f.lines_deleted,
    )


def score_change(
    change: Change,
    weights: RiskWeights,
    *,
    include_change_rules: bool = True,
    relative_thresholds: bool = False,
) -> ChangeRisk:
    """Score a whole change.

    The headline number is *the riskiest file, plus context penalties*, not a sum
    across files. Summing would make any large refactor score 100 and say
    nothing; a change is dangerous mainly because of the worst thing in it.

    `include_change_rules` is off for `scan --all`, where "10 files changed" and
    "it is Saturday" describe the repository and the clock rather than a change.
    """
    if not change.files:
        # Context rules would otherwise score an empty change on the clock
        # alone — "it is Saturday" is not a risk when nothing is shipping.
        return ChangeRisk(
            score=0,
            band=band_for(0, weights),
            reasons=(),
            files=(),
            recommendation="Nothing to score — no files in scope.",
            relative_thresholds=relative_thresholds,
        )

    files = sorted(
        (score_file(f, weights) for f in change.files),
        key=lambda r: (-r.score, r.path),
    )

    context_reasons: list[Reason] = []
    if include_change_rules:
        context_reasons = [r for rule in CHANGE_RULES for r in rule(change, weights)]

    riskiest = files[0] if files else None
    riskiest_score = riskiest.score if riskiest is not None else 0
    riskiest_reasons = list(riskiest.reasons) if riskiest is not None else []

    score = min(
        weights.max_score,
        sum(r.points for r in context_reasons) + riskiest_score,
    )

    reasons = context_reasons + riskiest_reasons
    reasons.sort(key=lambda r: r.points, reverse=True)

    band = band_for(score, weights)
    return ChangeRisk(
        score=score,
        band=band,
        reasons=tuple(reasons),
        files=tuple(files),
        recommendation=recommend(band, tuple(reasons)),
        relative_thresholds=relative_thresholds,
    )


def band_for(score: int, weights: RiskWeights) -> str:
    """Map a score onto low / medium / high."""
    if score >= weights.high_band_score:
        return "high"
    if score >= weights.medium_band_score:
        return "medium"
    return "low"


def recommend(band: str, reasons: tuple[Reason, ...], *, max_hints: int = 2) -> str:
    """Turn the band and the top rules into one actionable sentence."""
    headline = {
        "low": "Safe to deploy with a normal review.",
        "medium": "Deployable, but review it properly and watch it after release.",
        "high": "Hold this, or ship it behind a flag with a rollback ready.",
    }[band]

    hints: list[str] = []
    for reason in reasons:
        hint = RULE_HINTS.get(reason.rule)
        if hint and hint not in hints:
            hints.append(hint)
        if len(hints) == max_hints:
            break

    if not hints:
        return headline
    return f"{headline} Suggested: {'; '.join(hints)}."
