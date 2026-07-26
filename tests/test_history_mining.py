"""Tests for SZZ labeling — the part that makes the labels real rather than made up."""

from __future__ import annotations

from pathlib import Path

from git import Repo

from sentinel.config import BugfixDetection, TrainingSettings
from sentinel.features import FEATURE_NAMES
from sentinel.commit_log import read_commit_log
from sentinel.history_mining import (
    _RunningHistory,
    build_rows,
    find_bug_inducing,
    mine,
)
from tests.conftest import ALICE, BOB, commit_file

BUGFIX = BugfixDetection()
TRAINING = TrainingSettings(max_commits=100)


def test_blame_finds_the_commit_that_introduced_the_bug(szz_repo: Repo) -> None:
    log = read_commit_log(szz_repo, BUGFIX)
    inducing, blames = find_bug_inducing(szz_repo.working_tree_dir, log.commits)

    assert blames == 1  # only the "fix:" commit gets blamed
    assert szz_repo.bug_inducing_sha in inducing


def test_unrelated_commits_are_not_blamed(szz_repo: Repo) -> None:
    log = read_commit_log(szz_repo, BUGFIX)
    inducing, _ = find_bug_inducing(szz_repo.working_tree_dir, log.commits)

    notes_commit = next(c for c in log.commits if c.subject == "add notes")
    fix_commit = next(c for c in log.commits if c.subject.startswith("fix:"))

    assert notes_commit.sha not in inducing
    assert fix_commit.sha not in inducing


def test_mining_labels_exactly_the_inducing_commit(szz_repo: Repo) -> None:
    result = mine(szz_repo, BUGFIX, TRAINING)

    assert result.commits_considered == 4
    assert result.bugfix_commits == 1
    assert result.positives == 1

    labelled = {r.sha: r.label for r in result.rows}
    assert labelled[szz_repo.bug_inducing_sha] == 1
    assert sum(labelled.values()) == 1


def test_every_row_has_a_full_feature_vector(szz_repo: Repo) -> None:
    result = mine(szz_repo, BUGFIX, TRAINING)

    assert result.rows
    for row in result.rows:
        assert len(row.features) == len(FEATURE_NAMES)
        assert all(isinstance(value, float) for value in row.features)


def test_rows_come_out_oldest_first(szz_repo: Repo) -> None:
    """Time order is what makes the running-history features leak-free."""
    result = mine(szz_repo, BUGFIX, TRAINING)
    dates = [r.when for r in result.rows]
    assert dates == sorted(dates)


def test_max_commits_is_reported_as_truncated(tiny_repo: Repo) -> None:
    """A capped run must never look like a complete one."""
    result = mine(tiny_repo, BUGFIX, TrainingSettings(max_commits=2))
    assert result.commits_considered == 2
    assert result.truncated

    full = mine(tiny_repo, BUGFIX, TrainingSettings(max_commits=100))
    assert not full.truncated


# --- leak-proofing --------------------------------------------------------


def test_features_never_include_the_commit_being_described(tiny_repo: Repo) -> None:
    """The first commit to a file must see zero prior history, not one."""
    log = read_commit_log(tiny_repo, BUGFIX)
    rows = list(build_rows(tiny_repo, log.commits, set(), TRAINING))

    max_commits_index = FEATURE_NAMES.index("max_file_commits")
    new_files_index = FEATURE_NAMES.index("new_files")

    first = rows[0]
    assert first.features[max_commits_index] == 0.0
    assert first.features[new_files_index] == 1.0


def test_running_history_counts_only_what_came_before(tiny_repo: Repo) -> None:
    log = read_commit_log(tiny_repo, BUGFIX)
    history = _RunningHistory()

    seen: list[int] = []
    for commit in log.commits:
        snapshot = history.snapshot("app/core.py")
        seen.append(0 if snapshot is None else snapshot.total_commits)
        history.apply(commit)

    # core.py is touched by commits 1, 2 and 3, so the tally observed before
    # each of those commits is 0, then 1, then 2 — never 3.
    assert seen[:3] == [0, 1, 2]


def test_bugfix_history_accumulates_from_earlier_fixes(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path)
    with repo.config_writer() as writer:
        writer.set_value("user", "name", "Alice")
        writer.set_value("user", "email", "alice@example.com")

    commit_file(repo, "a.py", "x = 1\n", "add a", ALICE)
    commit_file(repo, "a.py", "x = 2\n", "fix: wrong value", BOB)
    commit_file(repo, "a.py", "x = 3\n", "fix: still wrong", BOB)
    commit_file(repo, "a.py", "x = 4\n", "tidy up", ALICE)

    log = read_commit_log(repo, BUGFIX)
    rows = list(build_rows(repo, log.commits, set(), TRAINING))

    index = FEATURE_NAMES.index("max_file_bugfixes")
    # By the final commit, two earlier fixes have landed on a.py.
    assert rows[-1].features[index] == 2.0
    assert rows[0].features[index] == 0.0
