"""Tests for the feature vector — the contract between training and prediction."""

from __future__ import annotations

from datetime import datetime

from sentinel.features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    Change,
    FileChange,
    FileHistory,
    percentile,
    vector,
)
from sentinel.static_analysis import ComplexityInfo, TestSignal

SATURDAY_NIGHT = datetime(2026, 7, 25, 23, 0)
WEDNESDAY = datetime(2026, 7, 22, 10, 0)


def history(total: int, bugfixes: int, author_commits: int) -> FileHistory:
    counts = {"Alice": author_commits}
    if total - author_commits > 0:
        counts["Bob"] = total - author_commits
    return FileHistory(
        path="app/core.py",
        total_commits=total,
        bugfix_commits=bugfixes,
        commits_per_author=counts,
        last_changed=None,
    )


def file_change(path: str = "app/core.py", **overrides) -> FileChange:
    defaults = dict(
        path=path,
        lines_added=10,
        lines_deleted=2,
        tests=TestSignal(is_test_file=False, test_paths=(), changed_test_paths=()),
        author="Alice",
        history=history(10, 2, 5),
        complexity=ComplexityInfo(average_ccn=3.0, max_ccn=7, nloc=80, function_count=6),
    )
    defaults.update(overrides)
    return FileChange(**defaults)  # type: ignore[arg-type]


def features_of(change: Change) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, vector(change)))


def test_vector_length_matches_the_declared_names() -> None:
    change = Change(files=(file_change(),), author="Alice", when=WEDNESDAY)
    assert len(vector(change)) == len(FEATURE_NAMES)


def test_every_feature_has_a_human_label() -> None:
    """SHAP output is only useful if each feature can be named in the report."""
    assert set(FEATURE_NAMES) == set(FEATURE_LABELS)


def test_sizes_and_counts_are_summed_across_files() -> None:
    change = Change(
        files=(
            file_change("app/core.py", lines_added=10, lines_deleted=2),
            file_change("web/ui.py", lines_added=5, lines_deleted=3),
        ),
        author="Alice",
        when=WEDNESDAY,
    )
    f = features_of(change)

    assert f["lines_added"] == 15
    assert f["lines_deleted"] == 5
    assert f["lines_changed"] == 20
    assert f["files_changed"] == 2
    assert f["folders_touched"] == 2


def test_files_in_the_same_folder_count_once() -> None:
    change = Change(
        files=(file_change("app/a.py"), file_change("app/b.py")),
        author="Alice",
        when=WEDNESDAY,
    )
    assert features_of(change)["folders_touched"] == 1


def test_root_level_files_have_a_folder() -> None:
    change = Change(files=(file_change("setup.py"),), author="Alice", when=WEDNESDAY)
    assert features_of(change)["folders_touched"] == 1


def test_history_features_take_the_worst_and_the_average() -> None:
    change = Change(
        files=(
            file_change("a.py", history=history(10, 1, 5)),
            file_change("b.py", history=history(30, 9, 0)),
        ),
        author="Alice",
        when=WEDNESDAY,
    )
    f = features_of(change)

    assert f["max_file_commits"] == 30
    assert f["mean_file_commits"] == 20
    assert f["max_file_bugfixes"] == 9
    assert f["mean_file_bugfixes"] == 5
    assert f["files_author_never_touched"] == 1
    assert f["min_author_ownership"] == 0.0


def test_new_files_are_counted_and_do_not_break_the_averages() -> None:
    change = Change(
        files=(file_change("new.py", history=None), file_change("old.py")),
        author="Alice",
        when=WEDNESDAY,
    )
    f = features_of(change)

    assert f["new_files"] == 1
    # The new file contributes no history, so averages come from the other file.
    assert f["mean_file_commits"] == 10


def test_a_change_of_only_new_files_has_zero_history_features() -> None:
    change = Change(files=(file_change(history=None),), author="Alice", when=WEDNESDAY)
    f = features_of(change)

    assert f["max_file_commits"] == 0
    assert f["mean_file_bugfixes"] == 0
    assert f["mean_author_ownership"] == 0


def test_missing_complexity_does_not_poison_the_average() -> None:
    change = Change(
        files=(
            file_change("a.py", complexity=ComplexityInfo(2.0, 4, 10, 2)),
            file_change("b.toml", complexity=None),
        ),
        author="Alice",
        when=WEDNESDAY,
    )
    f = features_of(change)

    assert f["max_complexity"] == 4
    assert f["mean_complexity"] == 4  # averaged over files that had a reading


def test_test_and_code_files_are_counted_separately() -> None:
    change = Change(
        files=(
            file_change("app/core.py"),
            file_change(
                "tests/test_core.py",
                tests=TestSignal(is_test_file=True, test_paths=(), changed_test_paths=()),
            ),
            file_change(
                "pyproject.toml",
                tests=TestSignal(
                    is_test_file=False, test_paths=(), changed_test_paths=(), is_code=False
                ),
            ),
        ),
        author="Alice",
        when=WEDNESDAY,
    )
    f = features_of(change)

    assert f["test_files_changed"] == 1
    assert f["code_files_changed"] == 2  # core.py and test_core.py; not the TOML


def test_timing_features() -> None:
    weekend = features_of(Change(files=(file_change(),), author="Alice", when=SATURDAY_NIGHT))
    assert weekend["hour_of_day"] == 23
    assert weekend["day_of_week"] == 5
    assert weekend["is_weekend"] == 1

    weekday = features_of(Change(files=(file_change(),), author="Alice", when=WEDNESDAY))
    assert weekday["is_weekend"] == 0


# --- percentiles ----------------------------------------------------------


def test_percentile_of_a_simple_range() -> None:
    values = list(range(1, 11))  # 1..10
    assert percentile(values, 0.0) == 1
    assert percentile(values, 1.0) == 10
    assert percentile(values, 0.9) == 9
    # Nearest rank on a 0-based index: round(0.5 * 9) is 4, so the 5th value.
    assert percentile(values, 0.5) == 5


def test_percentile_of_an_empty_or_single_list() -> None:
    assert percentile([], 0.9) == 0
    assert percentile([7], 0.9) == 7
