"""Tests for reading history out of a real (tiny) git repository."""

from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo

from sentinel.config import BugfixDetection
from sentinel.commit_log import (
    looks_like_bugfix,
    parse_commit_date,
    parse_numstat,
    read_commit_log,
    resolve_rename,
    split_rename,
)
from sentinel.git_reader import (
    RepositoryError,
    collect_file_histories,
    configured_author,
    files_changed_since,
    files_in_working_tree,
    head_author,
    open_repo,
    resolve_base,
    tracked_paths,
)
from tests.conftest import BOB, commit_file

BUGFIX = BugfixDetection()


# --- bug-fix detection ----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "fix: off-by-one in run()",
        "Fixed the broken parser",
        "hotfix for prod",
        "bugfix: null deref",
        "resolve crash on empty input",
        "closes #123",
        "resolved PROJ-42",
        "patch the retry loop",
    ],
)
def test_bugfix_messages_are_detected(message: str) -> None:
    assert looks_like_bugfix(message, BUGFIX)


@pytest.mark.parametrize(
    "message",
    [
        "Add connection pooling (#1234)",
        "Bump urllib3 to 2.1 (#5678)",
        "PROJ-42 add pagination support",
    ],
)
def test_a_bare_issue_or_pr_number_is_not_a_bug_fix(message: str) -> None:
    """Squash-merge repos put a PR number on every subject line.

    Counting those as bug-fix evidence labelled 51% of `requests` as bug fixes.
    """
    assert not looks_like_bugfix(message, BUGFIX)


@pytest.mark.parametrize(
    "message",
    [
        "add util helper",
        "refactor run() for clarity",
        "bump dependency versions",
        "document the config module",
        "",
    ],
)
def test_ordinary_messages_are_not_bugfixes(message: str) -> None:
    assert not looks_like_bugfix(message, BUGFIX)


def test_keywords_match_on_word_boundaries_only() -> None:
    """`prefix` and `suffix` contain "fix" but are not bug fixes."""
    assert not looks_like_bugfix("rename prefix and suffix handling", BUGFIX)


def test_only_the_subject_line_counts() -> None:
    """Bodies routinely mention bugs a change does not fix."""
    assert not looks_like_bugfix("add caching\n\nThis does not fix the timeout bug.", BUGFIX)


# --- history aggregation --------------------------------------------------


def test_history_counts_commits_bugfixes_and_authors(tiny_repo: Repo) -> None:
    scan = collect_file_histories(tiny_repo, BUGFIX)

    core = scan.histories["app/core.py"]
    assert core.total_commits == 3
    assert core.bugfix_commits == 1
    assert core.commits_by("Alice") == 2
    assert core.commits_by("Bob") == 1
    assert core.ownership("Alice") == pytest.approx(2 / 3)
    assert core.last_changed is not None

    util = scan.histories["app/util.py"]
    assert util.total_commits == 1
    assert util.bugfix_commits == 0

    # An issue key *with a fixing verb* is treated as a bug-fix signal.
    assert scan.histories["tests/test_core.py"].bugfix_commits == 1

    assert scan.commits_walked == 5
    assert scan.commits_skipped == 0


def test_history_walks_once_and_reports_unknown_authors_as_zero(tiny_repo: Repo) -> None:
    scan = collect_file_histories(tiny_repo, BUGFIX)
    core = scan.histories["app/core.py"]
    assert core.commits_by("Nobody") == 0
    assert core.ownership("Nobody") == 0.0


def test_every_file_is_aggregated_not_just_the_changed_ones(tiny_repo: Repo) -> None:
    """The percentile thresholds need the whole repository, not a subset."""
    scan = collect_file_histories(tiny_repo, BUGFIX)
    assert set(scan.histories) == {"app/core.py", "app/util.py", "tests/test_core.py"}
    assert scan.distribution.files == 3


def test_a_tiny_repo_reports_its_distribution_as_meaningless(tiny_repo: Repo) -> None:
    """Three files are not a distribution; the absolute thresholds must be used."""
    scan = collect_file_histories(tiny_repo, BUGFIX)
    assert not scan.distribution.meaningful


def test_until_measures_history_before_the_change(tiny_repo: Repo) -> None:
    """A change must not be able to inflate its own file's track record."""
    second_commit = list(tiny_repo.iter_commits())[-2].hexsha  # after the initial commit

    scan = collect_file_histories(
        tiny_repo, BUGFIX, until=second_commit
    )

    core = scan.histories["app/core.py"]
    assert core.total_commits == 2  # the third commit is excluded
    assert "app/util.py" not in scan.histories


# --- diffs ----------------------------------------------------------------


def test_files_changed_since_lists_the_range(tiny_repo: Repo) -> None:
    base = list(tiny_repo.iter_commits())[-1].hexsha  # the initial commit
    changed = {c.path: c for c in files_changed_since(tiny_repo, base)}

    assert set(changed) == {"app/core.py", "app/util.py", "tests/test_core.py"}
    assert changed["app/util.py"].lines_added == 2
    assert changed["app/core.py"].lines_changed > 0


def test_working_tree_sees_edits_and_untracked_files(tiny_repo: Repo) -> None:
    root = Path(tiny_repo.working_tree_dir)
    (root / "app/core.py").write_text("def run():\n    return 4\n\n# edited\n", encoding="utf-8")
    (root / "app/brand_new.py").write_text("x = 1\ny = 2\n", encoding="utf-8")

    changed = {c.path: c for c in files_in_working_tree(tiny_repo)}

    assert changed["app/core.py"].lines_added > 0
    assert changed["app/brand_new.py"].lines_added == 2
    assert changed["app/brand_new.py"].lines_deleted == 0


def test_tracked_paths_are_posix(tiny_repo: Repo) -> None:
    paths = tracked_paths(tiny_repo)
    assert "app/core.py" in paths
    assert not any("\\" in p for p in paths)


def test_authors_come_from_head_and_config(tiny_repo: Repo) -> None:
    assert head_author(tiny_repo) == BOB.name  # Bob made the last commit
    assert configured_author(tiny_repo) == "Alice"


# --- failure modes --------------------------------------------------------


def test_opening_a_non_repository_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(RepositoryError, match="not a git repository"):
        open_repo(tmp_path / "nope")


def test_unknown_ref_fails_clearly(tiny_repo: Repo) -> None:
    with pytest.raises(RepositoryError, match="unknown ref"):
        resolve_base(tiny_repo, "no-such-branch")


# --- diff parsing details -------------------------------------------------


def test_numstat_treats_binary_files_as_zero_lines() -> None:
    changed = parse_numstat("-\t-\tlogo.png\n3\t1\tapp/core.py\n")
    assert changed[0].lines_changed == 0
    assert changed[1].lines_changed == 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("app/core.py", "app/core.py"),
        ("old.py => new.py", "new.py"),
        ("app/{old => new}/core.py", "app/new/core.py"),
        ("{a => b}/core.py", "b/core.py"),
        ("{ => src}/core.py", "src/core.py"),
    ],
)
def test_renames_resolve_to_the_current_path(raw: str, expected: str) -> None:
    assert resolve_rename(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("app/core.py", (None, "app/core.py")),
        ("old.py => new.py", ("old.py", "new.py")),
        ("app/{old => new}/core.py", ("app/old/core.py", "app/new/core.py")),
        ("{ => src}/core.py", ("core.py", "src/core.py")),
    ],
)
def test_rename_notation_splits_into_old_and_new(raw: str, expected: tuple) -> None:
    assert split_rename(raw) == expected


def test_history_follows_a_file_across_a_rename(tiny_repo: Repo) -> None:
    """A repo that moved `app/` to `src/app/` must not look brand new."""
    tiny_repo.git.mv("app/core.py", "app/renamed.py")
    tiny_repo.index.commit("move core to renamed", author=BOB, committer=BOB)

    scan = collect_file_histories(tiny_repo, BUGFIX)

    renamed = scan.histories["app/renamed.py"]
    assert renamed.total_commits == 4  # 3 before the move, plus the move itself
    assert renamed.bugfix_commits == 1  # the pre-rename "fix:" commit still counts
    assert renamed.commits_by("Alice") == 2
    assert "app/core.py" not in scan.histories


def test_rename_history_survives_a_directory_move(tiny_repo: Repo) -> None:
    root = Path(tiny_repo.working_tree_dir)
    (root / "src").mkdir()
    tiny_repo.git.mv("app", "src/app")
    tiny_repo.index.commit("move app under src", author=BOB, committer=BOB)

    scan = collect_file_histories(tiny_repo, BUGFIX)

    assert scan.histories["src/app/core.py"].total_commits == 4
    assert "app/core.py" not in scan.histories


def test_commit_dates_survive_a_corrupt_timezone_offset() -> None:
    """A real `requests` commit is recorded at +518:00; it must not be dropped."""
    parsed = parse_commit_date("2011-09-08T02:38:50+518:00")

    assert parsed is not None
    assert parsed.year == 2011
    assert parsed.hour == 2
    assert parsed.tzinfo is not None  # aware, so it can be compared to the rest


def test_normal_commit_dates_keep_their_offset() -> None:
    parsed = parse_commit_date("2011-09-08T02:38:50+02:00")
    assert parsed is not None
    assert parsed.utcoffset().total_seconds() == 7200


def test_genuinely_unreadable_dates_are_reported_as_missing() -> None:
    assert parse_commit_date("not a date at all") is None


def test_untracked_binary_file_does_not_crash_the_line_count(tiny_repo: Repo) -> None:
    root = Path(tiny_repo.working_tree_dir)
    (root / "blob.bin").write_bytes(b"\x00\x01\x02\xff")

    changed = {c.path: c for c in files_in_working_tree(tiny_repo)}
    assert changed["blob.bin"].lines_added == 0


def test_commit_file_helper_keeps_history_deterministic(tiny_repo: Repo) -> None:
    """Guards the fixture itself: extra commits must show up in the aggregate."""
    commit_file(tiny_repo, "app/util.py", "def helper():\n    return 1\n", "fix: helper", BOB)
    scan = collect_file_histories(tiny_repo, BUGFIX)
    assert scan.histories["app/util.py"].bugfix_commits == 1
