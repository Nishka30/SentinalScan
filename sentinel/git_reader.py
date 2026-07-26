"""Ask a git repository questions.

Opening repositories, resolving refs, finding what changed, and aggregating
per-file history. The parsing of git's own output formats lives next door in
`commit_log`; this module is about which questions to ask and what the answers
mean together.

Two deliberate choices:

* History is aggregated in **one** pass over the commit graph, and that pass is
  a single `git log --numstat`. PyDriller was the first implementation and was
  abandoned here on measurement, not taste: its `modified_files` reconstructs
  the full textual diff of every commit, which took over ten minutes on a
  6,500-commit repository. We only need paths and line counts, which `git log`
  hands over in one subprocess. PyDriller remains the right tool for the SZZ
  pass in `history_mining`, where line-level blame is exactly what is wanted.
* Working-tree and range diffs also go through GitPython, because `sentinel
  diff` has to score changes that were never committed at all.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo

from sentinel.commit_log import normalize, parse_numstat, read_commit_log
from sentinel.config import BugfixDetection, DistributionSettings
from sentinel.features import (
    ChangedFile,
    FileHistory,
    HistoryScan,
    RepoDistribution,
    percentile,
)

logger = logging.getLogger(__name__)

#: Branch names tried, in order, when the remote does not tell us its default.
DEFAULT_BRANCH_CANDIDATES = ("main", "master", "develop", "trunk")


class RepositoryError(RuntimeError):
    """The repository cannot answer a question Sentinel needs answered."""


# --------------------------------------------------------------------------
# Opening the repository and resolving what "changed" means
# --------------------------------------------------------------------------


def open_repo(repo_path: Path) -> Repo:
    """Open a git repository, or fail with a message a human can act on."""
    try:
        return Repo(repo_path, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise RepositoryError(f"{repo_path} is not a git repository") from exc


def repo_root(repo: Repo) -> Path:
    """Absolute path of the working tree, for resolving files on disk."""
    if repo.working_tree_dir is None:
        raise RepositoryError("bare repositories have no working tree to analyze")
    return Path(repo.working_tree_dir)


def resolve_default_branch(repo: Repo) -> str:
    """Best guess at the branch changes are merged into.

    Prefers what the remote actually says over a hardcoded name, because
    `main` vs `master` vs `develop` is not something we should assume.
    """
    try:
        ref = repo.git.symbolic_ref("refs/remotes/origin/HEAD").strip()
        if ref:
            return ref.removeprefix("refs/remotes/")
    except GitCommandError:
        logger.debug("origin/HEAD is not set; falling back to common branch names")

    for name in DEFAULT_BRANCH_CANDIDATES:
        for candidate in (name, f"origin/{name}"):
            if _ref_exists(repo, candidate):
                return candidate

    raise RepositoryError(
        "could not work out the default branch — pass --since <ref> explicitly"
    )


def _ref_exists(repo: Repo, ref: str) -> bool:
    try:
        repo.rev_parse(ref)
    except Exception:  # GitPython raises several unrelated types here
        return False
    return True


def resolve_base(repo: Repo, ref: str) -> str:
    """Resolve the commit the change should be measured against.

    Uses the merge base, not the tip: comparing against the tip of `main` would
    attribute other people's merged work to this change.
    """
    if not _ref_exists(repo, ref):
        raise RepositoryError(f"unknown ref: {ref}")
    try:
        return repo.git.merge_base(ref, "HEAD").strip()
    except GitCommandError as exc:
        raise RepositoryError(f"{ref} and HEAD have no common ancestor") from exc


def require_commits(repo: Repo) -> None:
    if not repo.head.is_valid():
        raise RepositoryError("repository has no commits yet — nothing to compare against")


# --------------------------------------------------------------------------
# What changed
# --------------------------------------------------------------------------


def files_changed_since(repo: Repo, base: str) -> list[ChangedFile]:
    """Files that differ between `base` and the current HEAD."""
    require_commits(repo)
    return parse_numstat(repo.git.diff("--numstat", base, "HEAD"))


def files_in_working_tree(repo: Repo) -> list[ChangedFile]:
    """Uncommitted work: staged, unstaged, and untracked files."""
    require_commits(repo)
    files = parse_numstat(repo.git.diff("--numstat", "HEAD"))
    seen = {f.path for f in files}
    root = repo_root(repo)

    for raw in repo.untracked_files:
        path = normalize(raw)
        if path in seen:
            continue
        files.append(ChangedFile(path=path, lines_added=_count_lines(root / path), lines_deleted=0))
    return files


def _count_lines(path: Path) -> int:
    """Line count of a new file, treating unreadable/binary content as 0."""
    try:
        return len(path.read_text(encoding="utf-8", errors="strict").splitlines())
    except (OSError, UnicodeDecodeError):
        logger.debug("could not count lines in %s (binary or unreadable)", path)
        return 0


def all_tracked_files(repo: Repo) -> list[ChangedFile]:
    """Every tracked file, with a zero-line edit.

    Used by `scan --all`, where there is no change to measure — only the
    inherent riskiness of each file. Zero lines means the change-size rules
    correctly stay silent.
    """
    return [ChangedFile(path=p, lines_added=0, lines_deleted=0) for p in tracked_paths(repo)]


def tracked_paths(repo: Repo, ref: str | None = None) -> list[str]:
    """All file paths tracked at `ref` (default: the working tree)."""
    if ref is None:
        output = repo.git.ls_files()
    else:
        output = repo.git.ls_tree("-r", "--name-only", ref)
    return [normalize(line) for line in output.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Authors
# --------------------------------------------------------------------------


def head_author(repo: Repo) -> str:
    """Author of HEAD — the person whose committed work `scan` is scoring."""
    require_commits(repo)
    return repo.head.commit.author.name or repo.head.commit.author.email or "unknown"


def configured_author(repo: Repo) -> str:
    """Local `user.name` — the person whose uncommitted work `diff` is scoring."""
    try:
        return str(repo.config_reader().get_value("user", "name"))
    except Exception:
        logger.warning("git user.name is not set; author familiarity will be unavailable")
        return "unknown"


# --------------------------------------------------------------------------
# Aggregating history
# --------------------------------------------------------------------------


def collect_file_histories(
    repo: Repo,
    config: BugfixDetection,
    *,
    until: str | None = None,
    distribution: DistributionSettings | None = None,
) -> HistoryScan:
    """Aggregate per-file history in one pass over the commit graph.

    Every file is aggregated, not just the ones being scored, because the
    percentile thresholds in the returned `RepoDistribution` are only meaningful
    against the whole repository.

    `until` matters for correctness, not speed: when scoring commits that are
    already on the branch, history must be measured *before* them, or the change
    being scored inflates its own file's track record and hides the risk.

    Merge commits are skipped so a merge does not double-count every commit it
    brings in.

    History follows renames. Git log walks newest first, so a rename is always
    seen *before* the older commits that used the old name — recording the alias
    on the way past is enough to stitch the two halves together. Without this, a
    repository that moved `foo/` to `src/foo/` looks like it has no history at
    all, and the strongest signal Sentinel has silently reads zero.
    """
    totals: Counter[str] = Counter()
    fixes: Counter[str] = Counter()
    authors: dict[str, Counter[str]] = defaultdict(Counter)
    last_changed: dict[str, datetime] = {}

    log = read_commit_log(repo, config, until=until)

    for commit in log.commits:
        for path, _added, _deleted in commit.files:
            totals[path] += 1
            authors[path][commit.author] += 1
            if commit.is_bugfix:
                fixes[path] += 1
            if path not in last_changed or commit.when > last_changed[path]:
                last_changed[path] = commit.when

    histories = {
        path: FileHistory(
            path=path,
            total_commits=count,
            bugfix_commits=fixes[path],
            commits_per_author=dict(authors[path]),
            last_changed=last_changed.get(path),
        )
        for path, count in totals.items()
    }
    return HistoryScan(
        histories=histories,
        distribution=_summarize_distribution(histories, distribution),
        commits_walked=len(log.commits),
        commits_skipped=log.skipped,
    )


def _summarize_distribution(
    histories: dict[str, FileHistory], settings: DistributionSettings | None
) -> RepoDistribution:
    """Percentiles of this repository's bug-fix and churn counts.

    Only files that have actually needed a fix count towards the bug-fix
    percentiles. Including the long tail of never-fixed files would drag the
    90th percentile down to zero or one in most repositories, at which point
    "unusually often fixed" would mean "fixed once".
    """
    settings = settings or DistributionSettings()

    fix_counts = sorted(h.bugfix_commits for h in histories.values() if h.bugfix_commits > 0)
    churn_counts = sorted(h.total_commits for h in histories.values())

    meaningful = len(churn_counts) >= settings.min_files and len(fix_counts) >= settings.min_files

    return RepoDistribution(
        files=len(churn_counts),
        bugfix_hot=percentile(fix_counts, settings.hot_quantile),
        bugfix_very_hot=percentile(fix_counts, settings.very_hot_quantile),
        churn_high=percentile(churn_counts, settings.churn_quantile),
        meaningful=meaningful,
    )
