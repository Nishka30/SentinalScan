"""SZZ labeling — where the training labels come from.

The whole project rests on this file. A risk model needs examples of risky
changes, and the only honest source is the repository's own record of what it
later had to fix:

1. Find commits whose subject reads like a bug fix.
2. For each, blame the lines that fix *deleted* against the parent commit. The
   commits that last touched those lines are the ones that introduced the bug.
3. Those commits are labelled bug-inducing (1); everything else is clean (0).

This is where PyDriller earns its place in the stack. The `git log --numstat`
walk in `git_reader` is far faster for aggregate counts, but it cannot do
line-level blame, and `get_commits_last_modified_lines` is exactly that
algorithm already written and tested.

Two honest caveats, both of which the time-based evaluation is designed around:

* Keyword-matched bug fixes are a proxy. A fix whose subject says "tidy up"
  is missed, and a feature commit mentioning "fix" is a false positive.
* Recent commits look cleaner than they are, simply because nobody has found
  their bugs yet. Never evaluate on a random split — the newest commits belong
  in the test set, which is what `evaluation.py` enforces.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator

import lizard
from git import Repo
from pydriller import Git as PyDrillerGit

from sentinel.config import BugfixDetection, TrainingSettings
from sentinel.features import Change, CommitRecord, FileChange, FileHistory, vector
from sentinel.commit_log import read_commit_log
from sentinel.static_analysis import ComplexityInfo, TestSignal, is_code_path, is_test_path

logger = logging.getLogger(__name__)

ProgressHook = Callable[[int, int, str], None]


@dataclass(frozen=True)
class LabeledCommit:
    """A commit turned into a training row."""

    sha: str
    when: datetime
    features: list[float]
    label: int


@dataclass(frozen=True)
class MiningResult:
    """Everything `train` and `evaluate` need, plus how it was obtained."""

    rows: tuple[LabeledCommit, ...]
    commits_considered: int
    bugfix_commits: int
    bug_inducing_commits: int
    blames_run: int
    truncated: bool

    @property
    def positives(self) -> int:
        return sum(r.label for r in self.rows)

    @property
    def positive_rate(self) -> float:
        return self.positives / len(self.rows) if self.rows else 0.0


# --------------------------------------------------------------------------
# SZZ
# --------------------------------------------------------------------------


def find_bug_inducing(
    repo_path: str,
    commits: tuple[CommitRecord, ...],
    *,
    on_progress: ProgressHook | None = None,
) -> tuple[set[str], int]:
    """Blame every bug-fixing commit to find the commits that introduced the bug.

    Returns the set of bug-inducing commit hashes and how many blames ran.

    Only bug-fixing commits are blamed, which is what makes this affordable:
    they are a minority of history, and the other commits need no blame at all.
    """
    git = PyDrillerGit(repo_path)
    fixes = [c for c in commits if c.is_bugfix]
    inducing: set[str] = set()

    for index, fix in enumerate(fixes, start=1):
        if on_progress is not None:
            on_progress(index, len(fixes), fix.sha[:8])
        try:
            commit = git.get_commit(fix.sha)
            blamed = git.get_commits_last_modified_lines(commit)
        except Exception as exc:
            # A single unblameable commit (binary file, missing parent, shallow
            # clone boundary) must not abandon the whole mining run.
            logger.debug("blame failed for %s: %s", fix.sha[:8], exc)
            continue
        for shas in blamed.values():
            inducing.update(shas)

    return inducing, len(fixes)


# --------------------------------------------------------------------------
# Features for historical commits
# --------------------------------------------------------------------------


class _RunningHistory:
    """Per-file tallies as of "just before" the commit being featurised.

    This is the leak-proofing. Walking forwards and reading the tallies *before*
    applying each commit guarantees no feature can contain information from the
    commit it describes, or from any commit after it.
    """

    def __init__(self) -> None:
        self.commits: Counter[str] = Counter()
        self.fixes: Counter[str] = Counter()
        self.authors: dict[str, Counter[str]] = defaultdict(Counter)

    def snapshot(self, path: str) -> FileHistory | None:
        """History for `path` so far, or None if git has not seen it before."""
        if self.commits[path] == 0:
            return None
        return FileHistory(
            path=path,
            total_commits=self.commits[path],
            bugfix_commits=self.fixes[path],
            commits_per_author=dict(self.authors[path]),
            last_changed=None,
        )

    def apply(self, commit: CommitRecord) -> None:
        for path, _added, _deleted in commit.files:
            self.commits[path] += 1
            self.authors[path][commit.author] += 1
            if commit.is_bugfix:
                self.fixes[path] += 1


def _complexity_at(repo: Repo, sha: str, path: str, cache: dict[str, ComplexityInfo | None]):
    """Complexity of one file as it stood at one commit.

    Read from the object database rather than the working tree, because the
    working tree is the future as far as a historical commit is concerned.
    Cached on blob hash so unchanged content is only ever parsed once.
    """
    if not is_code_path(path):
        return None
    try:
        blob = repo.commit(sha).tree / path
    except Exception:
        return None  # deleted in this commit, or not a blob

    if blob.hexsha in cache:
        return cache[blob.hexsha]

    info: ComplexityInfo | None = None
    try:
        content = blob.data_stream.read().decode("utf-8", errors="replace")
        parsed = lizard.analyze_file.analyze_source_code(path, content)
        functions = list(parsed.function_list)
        if functions:
            values = [int(f.cyclomatic_complexity) for f in functions]
            info = ComplexityInfo(
                average_ccn=round(sum(values) / len(values), 1),
                max_ccn=max(values),
                nloc=int(parsed.nloc),
                function_count=len(functions),
            )
    except Exception as exc:
        logger.debug("complexity failed for %s@%s: %s", path, sha[:8], exc)

    cache[blob.hexsha] = info
    return info


def as_change(
    commit: CommitRecord,
    history: _RunningHistory,
    repo: Repo | None = None,
    complexity_cache: dict[str, ComplexityInfo | None] | None = None,
) -> Change:
    """Rebuild a historical commit as the same `Change` object `scan` scores.

    Using one type for both is the point: the feature vector is computed by the
    same function in training and in production, so the two cannot disagree.
    """
    files = tuple(
        FileChange(
            path=path,
            lines_added=added,
            lines_deleted=deleted,
            tests=TestSignal(
                is_test_file=is_test_path(path),
                test_paths=(),
                changed_test_paths=(),
                is_code=is_code_path(path),
            ),
            author=commit.author,
            history=history.snapshot(path),
            complexity=(
                _complexity_at(repo, commit.sha, path, complexity_cache)
                if repo is not None and complexity_cache is not None
                else None
            ),
        )
        for path, added, deleted in commit.files
    )
    return Change(files=files, author=commit.author, when=commit.when)


def build_rows(
    repo: Repo,
    commits: tuple[CommitRecord, ...],
    inducing: set[str],
    settings: TrainingSettings,
    *,
    on_progress: ProgressHook | None = None,
) -> Iterator[LabeledCommit]:
    """Featurise and label every commit, oldest first."""
    history = _RunningHistory()
    cache: dict[str, ComplexityInfo | None] = {}
    use_complexity = settings.include_complexity

    for index, commit in enumerate(commits, start=1):
        if on_progress is not None:
            on_progress(index, len(commits), commit.sha[:8])
        if commit.files:
            change = as_change(
                commit,
                history,
                repo if use_complexity else None,
                cache if use_complexity else None,
            )
            yield LabeledCommit(
                sha=commit.sha,
                when=commit.when,
                features=vector(change),
                label=1 if commit.sha in inducing else 0,
            )
        # Applied *after* featurising, never before.
        history.apply(commit)


def mine(
    repo: Repo,
    bugfix: BugfixDetection,
    settings: TrainingSettings,
    *,
    max_commits: int | None = None,
    on_blame: ProgressHook | None = None,
    on_features: ProgressHook | None = None,
) -> MiningResult:
    """Run the whole pipeline: read history, blame the fixes, label, featurise."""
    limit = settings.max_commits if max_commits is None else max_commits
    log = read_commit_log(repo, bugfix, max_commits=limit)
    commits = log.commits
    if not commits:
        return MiningResult((), 0, 0, 0, 0, False)

    repo_path = repo.working_tree_dir or "."
    inducing, blames = find_bug_inducing(repo_path, commits, on_progress=on_blame)

    rows = tuple(build_rows(repo, commits, inducing, settings, on_progress=on_features))
    known = {c.sha for c in commits}

    return MiningResult(
        rows=rows,
        commits_considered=len(commits),
        bugfix_commits=sum(1 for c in commits if c.is_bugfix),
        # Blame can name commits older than the window; only those inside it can
        # become training rows, so that is the number worth reporting.
        bug_inducing_commits=len(inducing & known),
        blames_run=blames,
        truncated=log.truncated,
    )
