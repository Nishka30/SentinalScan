"""Tests for the rule engine.

Each test drives one rule to the point where it must fire (or must not), so a
change to a weight in config.py breaks the test that documents that weight.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from sentinel.config import DistributionSettings, RiskWeights
from sentinel.features import Change, FileChange, FileHistory, RepoDistribution
from sentinel.risk_rules import (
    band_for,
    calibrate,
    score_change,
    score_file,
)
from sentinel.static_analysis import ComplexityInfo, TestSignal

W = RiskWeights()

QUIET_HOUR = datetime(2026, 7, 22, 14, 0)  # a Wednesday afternoon
WEEKEND_NIGHT = datetime(2026, 7, 25, 23, 30)  # a Saturday, late


def history(
    *, total: int = 5, bugfixes: int = 0, author: str = "Alice", author_commits: int | None = None
) -> FileHistory:
    """A FileHistory with the numbers a test cares about and sane filler."""
    mine = total if author_commits is None else author_commits
    others = max(0, total - mine)
    counts = {author: mine}
    if others:
        counts["Someone Else"] = others
    return FileHistory(
        path="app/core.py",
        total_commits=total,
        bugfix_commits=bugfixes,
        commits_per_author=counts,
        last_changed=QUIET_HOUR,
    )


def covered() -> TestSignal:
    return TestSignal(
        is_test_file=False,
        test_paths=("tests/test_core.py",),
        changed_test_paths=("tests/test_core.py",),
    )


def file_change(**overrides) -> FileChange:
    """A deliberately boring change: every rule silent unless overridden."""
    defaults = dict(
        path="app/core.py",
        lines_added=5,
        lines_deleted=1,
        tests=covered(),
        author="Alice",
        history=history(),
        complexity=ComplexityInfo(average_ccn=2.0, max_ccn=3, nloc=40, function_count=5),
    )
    defaults.update(overrides)
    return FileChange(**defaults)  # type: ignore[arg-type]


def rules_fired(risk) -> set[str]:
    return {r.rule for r in risk.reasons}


def points_for(risk, rule: str) -> int:
    return sum(r.points for r in risk.reasons if r.rule == rule)


# --- the baseline ---------------------------------------------------------


def test_a_boring_change_fires_nothing() -> None:
    risk = score_file(file_change(), W)
    assert risk.score == 0
    assert risk.reasons == ()
    assert risk.band == "low"


# --- individual rules -----------------------------------------------------


def test_large_edit_has_two_tiers() -> None:
    moderate = score_file(file_change(lines_added=W.file_moderate_lines, lines_deleted=0), W)
    assert points_for(moderate, "large_edit") == W.file_moderate_points

    large = score_file(file_change(lines_added=W.file_large_lines, lines_deleted=0), W)
    assert points_for(large, "large_edit") == W.file_large_points

    below = score_file(file_change(lines_added=W.file_moderate_lines - 1, lines_deleted=0), W)
    assert "large_edit" not in rules_fired(below)


def test_hot_file_has_two_tiers_and_cites_the_count() -> None:
    hot = score_file(file_change(history=history(total=10, bugfixes=W.hot_bugfixes)), W)
    assert points_for(hot, "hot_file") == W.hot_points
    assert str(W.hot_bugfixes) in next(r.detail for r in hot.reasons if r.rule == "hot_file")

    very_hot = score_file(
        file_change(history=history(total=20, bugfixes=W.very_hot_bugfixes)), W
    )
    assert points_for(very_hot, "hot_file") == W.very_hot_points

    cold = score_file(file_change(history=history(total=10, bugfixes=W.hot_bugfixes - 1)), W)
    assert "hot_file" not in rules_fired(cold)


def test_churn_fires_on_commit_count_alone() -> None:
    risk = score_file(file_change(history=history(total=W.churn_commits)), W)
    assert points_for(risk, "churn") == W.churn_points


def test_missing_tests_outweighs_stale_tests() -> None:
    no_tests = TestSignal(is_test_file=False, test_paths=(), changed_test_paths=())
    missing = score_file(file_change(tests=no_tests), W)
    assert points_for(missing, "missing_tests") == W.no_test_file_points

    stale = score_file(
        file_change(
            tests=TestSignal(
                is_test_file=False,
                test_paths=("tests/test_core.py",),
                changed_test_paths=(),
            )
        ),
        W,
    )
    assert points_for(stale, "stale_tests") == W.stale_test_points
    assert W.stale_test_points < W.no_test_file_points


def test_non_code_files_are_not_penalised_for_having_no_tests() -> None:
    """A TOML file with no unit test is not a deployment risk."""
    risk = score_file(
        file_change(
            path="pyproject.toml",
            tests=TestSignal(
                is_test_file=False, test_paths=(), changed_test_paths=(), is_code=False
            ),
        ),
        W,
    )
    assert "missing_tests" not in rules_fired(risk)


def test_a_test_file_is_not_penalised_for_having_no_tests() -> None:
    risk = score_file(
        file_change(
            path="tests/test_core.py",
            tests=TestSignal(is_test_file=True, test_paths=(), changed_test_paths=()),
        ),
        W,
    )
    assert "missing_tests" not in rules_fired(risk)
    assert "stale_tests" not in rules_fired(risk)


def test_complexity_has_two_tiers_and_stays_silent_without_data() -> None:
    high = score_file(
        file_change(complexity=ComplexityInfo(4.0, W.high_ccn, 100, 10)), W
    )
    assert points_for(high, "high_complexity") == W.high_ccn_points

    very_high = score_file(
        file_change(complexity=ComplexityInfo(9.0, W.very_high_ccn, 200, 10)), W
    )
    assert points_for(very_high, "high_complexity") == W.very_high_ccn_points

    unknown = score_file(file_change(complexity=None), W)
    assert "high_complexity" not in rules_fired(unknown)


def test_author_new_to_the_file_scores_more_than_a_light_contributor() -> None:
    stranger = score_file(file_change(history=history(total=10, author_commits=0)), W)
    assert points_for(stranger, "author_new_to_file") == W.new_to_file_points

    occasional = score_file(file_change(history=history(total=100, author_commits=1)), W)
    assert points_for(occasional, "low_ownership") == W.low_ownership_points

    owner = score_file(file_change(history=history(total=10, author_commits=9)), W)
    assert "author_new_to_file" not in rules_fired(owner)
    assert "low_ownership" not in rules_fired(owner)


def test_a_brand_new_file_is_not_blamed_for_being_unfamiliar() -> None:
    """Nobody is familiar with a file that did not exist until now."""
    risk = score_file(file_change(history=None), W)
    assert "author_new_to_file" not in rules_fired(risk)
    assert "hot_file" not in rules_fired(risk)
    assert "churn" not in rules_fired(risk)


# --- change-level rules ---------------------------------------------------


def test_timing_rules_fire_at_the_weekend_and_late_at_night() -> None:
    change = Change(files=(file_change(),), author="Alice", when=WEEKEND_NIGHT)
    risk = score_change(change, W)

    assert points_for(risk, "weekend_deploy") == W.weekend_points
    assert points_for(risk, "late_night_deploy") == W.late_night_points

    calm = score_change(Change(files=(file_change(),), author="Alice", when=QUIET_HOUR), W)
    assert "weekend_deploy" not in rules_fired(calm)
    assert "late_night_deploy" not in rules_fired(calm)


def test_broad_change_fires_on_file_count_and_on_total_lines() -> None:
    many = tuple(file_change(path=f"app/mod_{i}.py") for i in range(W.broad_files))
    risk = score_change(Change(files=many, author="Alice", when=QUIET_HOUR), W)
    assert "broad_change" in rules_fired(risk)

    detail = " ".join(r.detail for r in risk.reasons if r.rule == "broad_change")
    assert str(W.broad_files) in detail


def test_change_rules_can_be_switched_off_for_scan_all() -> None:
    """`--all` scores files, not a change, so "it is Saturday" is noise."""
    change = Change(files=(file_change(),), author="Alice", when=WEEKEND_NIGHT)
    risk = score_change(change, W, include_change_rules=False)
    assert "weekend_deploy" not in rules_fired(risk)


# --- aggregation ----------------------------------------------------------


def test_change_score_is_the_riskiest_file_plus_context() -> None:
    """Not a sum across files: one bad file plus context, so breadth cannot swamp it."""
    risky = file_change(
        path="app/hot.py",
        history=history(total=40, bugfixes=W.hot_bugfixes, author_commits=40),
        tests=TestSignal(is_test_file=False, test_paths=(), changed_test_paths=()),
    )
    calm = file_change(path="app/calm.py")

    change = Change(files=(calm, risky), author="Alice", when=WEEKEND_NIGHT)
    risk = score_change(change, W)

    riskiest = risk.files[0]
    assert riskiest.path == "app/hot.py"
    assert risk.score == riskiest.score + W.weekend_points + W.late_night_points
    # The calm file contributes nothing, and is ranked last.
    assert risk.files[-1].path == "app/calm.py"
    assert risk.files[-1].score == 0


def test_score_is_capped_at_one_hundred() -> None:
    worst = file_change(
        lines_added=5_000,
        history=history(total=500, bugfixes=99, author_commits=0),
        tests=TestSignal(is_test_file=False, test_paths=(), changed_test_paths=()),
        complexity=ComplexityInfo(30.0, 90, 2_000, 50),
    )
    file_risk = score_file(worst, W)
    assert sum(r.points for r in file_risk.reasons) > W.max_score
    assert file_risk.score == W.max_score

    change_risk = score_change(Change(files=(worst,), author="Alice", when=WEEKEND_NIGHT), W)
    assert change_risk.score == W.max_score
    assert change_risk.band == "high"


def test_reasons_are_labelled_and_ordered_by_contribution() -> None:
    worst = file_change(
        lines_added=1_000,
        history=history(total=100, bugfixes=20, author_commits=0),
        tests=TestSignal(is_test_file=False, test_paths=(), changed_test_paths=()),
        complexity=ComplexityInfo(20.0, 40, 900, 30),
    )
    risk = score_file(worst, W)

    points = [r.points for r in risk.reasons]
    assert points == sorted(points, reverse=True)
    for reason in risk.reasons:
        assert reason.rule and reason.label and reason.detail
        assert reason.points > 0
        assert reason.path == "app/core.py"


def test_an_empty_change_scores_zero() -> None:
    risk = score_change(Change(files=(), author="Alice", when=QUIET_HOUR), W)
    assert risk.score == 0
    assert risk.band == "low"
    assert risk.files == ()


# --- bands and recommendations -------------------------------------------


# --- relative calibration -------------------------------------------------


def test_calibration_raises_thresholds_to_the_repos_own_percentiles() -> None:
    """In a repo where 40 past fixes is normal, 8 must not be "very hot"."""
    distribution = RepoDistribution(
        files=500, bugfix_hot=12, bugfix_very_hot=40, churn_high=120, meaningful=True
    )
    calibrated, applied = calibrate(W, distribution, DistributionSettings())

    assert applied
    assert calibrated.very_hot_bugfixes == 40
    assert calibrated.hot_bugfixes == 12
    assert calibrated.churn_commits == 120

    # A file that would have maxed out the absolute thresholds now says nothing.
    typical = file_change(history=history(total=30, bugfixes=9))
    assert "hot_file" not in rules_fired(score_file(typical, calibrated))
    assert "hot_file" in rules_fired(score_file(typical, W))


def test_calibration_never_loosens_below_the_configured_floor() -> None:
    """A young repo whose 90th percentile is one fix must not reward one fix."""
    distribution = RepoDistribution(
        files=100, bugfix_hot=1, bugfix_very_hot=1, churn_high=2, meaningful=True
    )
    calibrated, applied = calibrate(W, distribution, DistributionSettings())

    assert applied
    assert calibrated.very_hot_bugfixes == W.very_hot_bugfixes
    assert calibrated.hot_bugfixes == W.hot_bugfixes
    assert calibrated.churn_commits == W.churn_commits


def test_a_repo_with_too_little_history_keeps_the_absolute_thresholds() -> None:
    distribution = RepoDistribution(
        files=3, bugfix_hot=1, bugfix_very_hot=2, churn_high=3, meaningful=False
    )
    calibrated, applied = calibrate(W, distribution, DistributionSettings())

    assert not applied
    assert calibrated == W


def test_relative_scoring_can_be_switched_off() -> None:
    distribution = RepoDistribution(
        files=500, bugfix_hot=12, bugfix_very_hot=40, churn_high=120, meaningful=True
    )
    calibrated, applied = calibrate(W, distribution, DistributionSettings(enabled=False))

    assert not applied
    assert calibrated == W


def test_the_effective_threshold_appears_in_the_evidence() -> None:
    """A reader must be able to see which threshold was actually applied."""
    distribution = RepoDistribution(
        files=500, bugfix_hot=12, bugfix_very_hot=40, churn_high=120, meaningful=True
    )
    calibrated, _ = calibrate(W, distribution, DistributionSettings())

    risk = score_file(file_change(history=history(total=200, bugfixes=50)), calibrated)
    detail = next(r.detail for r in risk.reasons if r.rule == "hot_file")
    assert "50 past bug-fix commits" in detail
    assert "threshold 40" in detail


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "low"),
        (W.medium_band_score - 1, "low"),
        (W.medium_band_score, "medium"),
        (W.high_band_score - 1, "medium"),
        (W.high_band_score, "high"),
        (100, "high"),
    ],
)
def test_band_boundaries(score: int, expected: str) -> None:
    assert band_for(score, W) == expected


def test_recommendation_names_something_to_do() -> None:
    untested = file_change(
        tests=TestSignal(is_test_file=False, test_paths=(), changed_test_paths=())
    )
    risk = score_change(Change(files=(untested,), author="Alice", when=QUIET_HOUR), W)
    assert "add a test" in risk.recommendation


def test_recommendation_exists_even_when_no_rule_fires() -> None:
    risk = score_change(Change(files=(file_change(),), author="Alice", when=QUIET_HOUR), W)
    assert risk.recommendation
    assert "Safe to deploy" in risk.recommendation
