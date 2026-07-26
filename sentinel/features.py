"""The shared data model, and how a change becomes numbers.

This module is the bottom of the dependency graph on purpose. `git_reader`
produces these types, `risk_rules` consumes them, and `history_mining` builds
them for historical commits — so they cannot live in any of those three.

`vector()` is the single definition of what the model sees. Training and
prediction both call it, which is what stops the two from drifting apart: a
feature added for prediction but absent at training time is the classic way to
ship a model that quietly scores nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sentinel.static_analysis import ComplexityInfo, TestSignal


@dataclass(frozen=True)
class FileHistory:
    """What the commit graph says about one file, up to some point in time."""

    path: str
    total_commits: int
    bugfix_commits: int
    commits_per_author: dict[str, int]
    last_changed: datetime | None

    def commits_by(self, author: str) -> int:
        """How many past commits this author made to this file."""
        return self.commits_per_author.get(author, 0)

    def ownership(self, author: str) -> float:
        """This author's share of the file's past commits, 0.0 to 1.0."""
        if self.total_commits == 0:
            return 0.0
        return self.commits_by(author) / self.total_commits


@dataclass(frozen=True)
class ChangedFile:
    """One file in the scope being scored, with the size of its edit."""

    path: str
    lines_added: int
    lines_deleted: int

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_deleted


@dataclass(frozen=True)
class RepoDistribution:
    """Percentiles of the repository's own history.

    Absolute thresholds cannot work across repositories: eight past bug fixes
    makes a file exceptional in a young service and unremarkable in a fifteen-
    year-old library. These percentiles let the rules ask "is this file unusual
    *for this repository*" instead.
    """

    files: int
    bugfix_hot: int
    bugfix_very_hot: int
    churn_high: int
    #: False when there is too little history for percentiles to mean anything,
    #: in which case the configured absolute thresholds are used instead.
    meaningful: bool


@dataclass(frozen=True)
class CommitRecord:
    """One commit as the log walk sees it: who, when, and what it touched.

    Paths are already resolved through any later renames, so a commit that
    touched `foo/a.py` before a move reports the path that file has today.
    """

    sha: str
    author: str
    when: datetime
    subject: str
    is_bugfix: bool
    #: (path, lines added, lines deleted)
    files: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class CommitLog:
    """A window of history, oldest commit first."""

    commits: tuple[CommitRecord, ...]
    #: True when `max_commits` cut the window short, so a partial run is never
    #: mistaken for a complete one.
    truncated: bool
    skipped: int


@dataclass(frozen=True)
class HistoryScan:
    """Result of the single history walk, including what it could not read."""

    histories: dict[str, FileHistory]
    distribution: RepoDistribution
    commits_walked: int
    commits_skipped: int


@dataclass(frozen=True)
class FileChange:
    """Everything known about one changed file, ready to be scored."""

    path: str
    lines_added: int
    lines_deleted: int
    tests: TestSignal
    author: str
    history: FileHistory | None = None
    complexity: ComplexityInfo | None = None

    @property
    def lines_changed(self) -> int:
        return self.lines_added + self.lines_deleted

    @property
    def is_new_file(self) -> bool:
        """No history means git has never seen this path before."""
        return self.history is None

    @property
    def author_commits(self) -> int:
        return 0 if self.history is None else self.history.commits_by(self.author)

    @property
    def ownership(self) -> float:
        return 0.0 if self.history is None else self.history.ownership(self.author)


@dataclass(frozen=True)
class Change:
    """The whole set of files being scored, plus who and when."""

    files: tuple[FileChange, ...]
    author: str
    when: datetime

    @property
    def total_lines_changed(self) -> int:
        return sum(f.lines_changed for f in self.files)


# --------------------------------------------------------------------------
# The feature vector
# --------------------------------------------------------------------------

#: Feature order is part of the saved model's contract. Append only; never
#: reorder or remove, or an old model file will silently read the wrong columns.
FEATURE_NAMES: tuple[str, ...] = (
    "lines_added",
    "lines_deleted",
    "lines_changed",
    "files_changed",
    "folders_touched",
    "max_file_commits",
    "mean_file_commits",
    "max_file_bugfixes",
    "mean_file_bugfixes",
    "new_files",
    "min_author_ownership",
    "mean_author_ownership",
    "files_author_never_touched",
    "max_complexity",
    "mean_complexity",
    "code_files_changed",
    "test_files_changed",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
)

#: Human phrasing for SHAP output, so a model reason reads like a rule reason.
FEATURE_LABELS: dict[str, str] = {
    "lines_added": "Lines added",
    "lines_deleted": "Lines deleted",
    "lines_changed": "Size of the change",
    "files_changed": "Number of files touched",
    "folders_touched": "Spread across folders",
    "max_file_commits": "Change history of the busiest file",
    "mean_file_commits": "Average change history of the files touched",
    "max_file_bugfixes": "Bug-fix history of the worst file",
    "mean_file_bugfixes": "Average bug-fix history of the files touched",
    "new_files": "Brand-new files",
    "min_author_ownership": "Author's familiarity with the file they know least",
    "mean_author_ownership": "Author's familiarity with these files",
    "files_author_never_touched": "Files the author has never changed",
    "max_complexity": "Complexity of the most complex file",
    "mean_complexity": "Average complexity of the files touched",
    "code_files_changed": "Source files touched",
    "test_files_changed": "Test files touched",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max(values: list[float]) -> float:
    return max(values) if values else 0.0


def vector(change: Change) -> list[float]:
    """Turn a change into the model's feature vector, in `FEATURE_NAMES` order.

    Deliberately excluded:

    * whether a matching test file *exists* — it cannot be reconstructed for a
      historical commit without indexing the whole tree at that commit, so
      including it would mean training on a feature prediction has and training
      does not.
    * **blast radius** (how many files depend on the changed one). It is
      computed for the report, but deliberately not fed to the model: doing so
      honestly would need the import graph *as it stood at each historical
      commit*, meaning every source file re-parsed at every commit — far more
      expensive than the SZZ pass that is already the slow half. Using today's
      graph for a 2015 commit would leak present-day structure into a past
      feature, which is exactly the kind of shortcut the time-based split
      exists to catch.
    * anything derived from the future (later commits, current file contents).
      Every input here is knowable at the moment the change was made.
    """
    files = change.files
    commits = [float(f.history.total_commits) for f in files if f.history]
    bugfixes = [float(f.history.bugfix_commits) for f in files if f.history]
    ownerships = [f.ownership for f in files if f.history]
    complexities = [float(f.complexity.max_ccn) for f in files if f.complexity]
    folders = {f.path.rsplit("/", 1)[0] if "/" in f.path else "" for f in files}

    return [
        float(sum(f.lines_added for f in files)),
        float(sum(f.lines_deleted for f in files)),
        float(change.total_lines_changed),
        float(len(files)),
        float(len(folders)),
        _max(commits),
        _mean(commits),
        _max(bugfixes),
        _mean(bugfixes),
        float(sum(1 for f in files if f.is_new_file)),
        min(ownerships) if ownerships else 0.0,
        _mean(ownerships),
        float(sum(1 for f in files if f.history and f.author_commits == 0)),
        _max(complexities),
        _mean(complexities),
        float(sum(1 for f in files if f.tests.is_code)),
        float(sum(1 for f in files if f.tests.is_test_file)),
        float(change.when.hour),
        float(change.when.weekday()),
        1.0 if change.when.weekday() >= 5 else 0.0,
    ]


def percentile(sorted_values: list[int], quantile: float) -> int:
    """Nearest-rank percentile of an already-sorted list.

    Hand-rolled rather than pulled from numpy: it is four lines, it is exact for
    the integer counts we feed it, and it keeps the dependency list honest.
    """
    if not sorted_values:
        return 0
    index = int(round(quantile * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]
