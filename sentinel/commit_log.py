"""Parse the text git gives us: numstat diffs, rename notation, log records.

Split out of `git_reader` so that module can stay about *asking* a repository
questions while this one is about understanding its answers. Everything here is
either a pure function over a string or a single `git log` call, which makes the
fiddly parts — rename notation, corrupt dates, binary markers — testable without
a repository at all.

`git_reader` imports this; this must never import `git_reader`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterator

from git import Repo

from sentinel.config import BugfixDetection
from sentinel.features import ChangedFile, CommitLog, CommitRecord

logger = logging.getLogger(__name__)

#: Record and field separators for the history walk: ASCII RS and US. Both are
#: control characters that do not occur in commit subjects or file paths. NUL
#: would be the conventional choice, but Windows `CreateProcess` rejects a NUL
#: inside an argument, so it cannot travel through `--format=`.
RECORD_SEP = "\x1e"
FIELD_SEP = "\x1f"
LOG_FORMAT = RECORD_SEP + FIELD_SEP.join(("%H", "%an", "%aI", "%s"))


def normalize(path: str) -> str:
    """Compare repo-relative paths as POSIX everywhere.

    Git speaks POSIX paths, Windows does not; mixing them silently breaks every
    dict lookup that joins history to changed files.
    """
    return path.replace("\\", "/")


# --------------------------------------------------------------------------
# Bug-fix detection (the rules, the SZZ pass and the model all share this)
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _compile_bugfix(
    keywords: tuple[str, ...], issue_patterns: tuple[str, ...]
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile once per keyword set — this runs against every commit message."""
    keyword_re = re.compile(
        r"\b(?:" + "|".join(re.escape(word) for word in keywords) + r")\b",
        re.IGNORECASE,
    )
    # Not case-insensitive: Jira keys are uppercase by convention, and lowering
    # the pattern would match ordinary hyphenated words followed by digits.
    issue_re = re.compile("|".join(issue_patterns))
    return keyword_re, issue_re


def looks_like_bugfix(message: str, config: BugfixDetection) -> bool:
    """True when a commit message reads like a bug fix.

    Only the subject line is considered: bodies routinely mention the bugs a
    change does *not* fix, and "fixes #12" in a footer is already an issue ref
    on the subject line in practice.
    """
    if not message:
        return False
    subject = message.strip().splitlines()[0]
    keyword_re, issue_re = _compile_bugfix(config.keywords, config.issue_patterns)
    return bool(keyword_re.search(subject) or issue_re.search(subject))


# --------------------------------------------------------------------------
# Rename notation
# --------------------------------------------------------------------------


def split_rename(raw: str) -> tuple[str | None, str]:
    """Split git's rename notation into (old path, new path).

    Git writes `old => new` or `dir/{a => b}/file`. The old path is not
    cosmetic: it is how history is followed across a move.
    """
    if "=>" not in raw:
        return None, normalize(raw)

    if "{" in raw and "}" in raw:
        prefix, rest = raw.split("{", 1)
        inner, suffix = rest.split("}", 1)
        old_part, _, new_part = inner.partition("=>")
        return (
            _splice(prefix, old_part.strip(), suffix),
            _splice(prefix, new_part.strip(), suffix),
        )

    old_part, _, new_part = raw.partition("=>")
    return normalize(old_part.strip()), normalize(new_part.strip())


def _splice(prefix: str, middle: str, suffix: str) -> str:
    """Rebuild a path from git's brace notation.

    One side of `{ => src}` is empty — a move to or from the repo root — which
    leaves a doubled or leading separator to clean up.
    """
    return normalize(f"{prefix}{middle}{suffix}".replace("//", "/").lstrip("/"))


def resolve_rename(raw: str) -> str:
    """The path a file has *now*, which is what exists on disk to analyze."""
    return split_rename(raw)[1]


# --------------------------------------------------------------------------
# numstat
# --------------------------------------------------------------------------


def iter_numstat(output: str) -> Iterator[tuple[str | None, str, int, int]]:
    """Yield (old path, new path, added, deleted) for each numstat line.

    Binary files report `-` for both counts and are reported as 0 lines.
    """
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, raw_path = parts
        old, new = split_rename(raw_path)
        yield (
            old,
            new,
            0 if added.strip() == "-" else int(added),
            0 if deleted.strip() == "-" else int(deleted),
        )


def parse_numstat(output: str) -> list[ChangedFile]:
    """Parse `git diff --numstat` into changed files at their current paths."""
    return [
        ChangedFile(path=new, lines_added=added, lines_deleted=deleted)
        for _old, new, added, deleted in iter_numstat(output)
    ]


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------


def parse_commit_date(raw: str) -> datetime | None:
    """Parse a commit's ISO-8601 author date, tolerating corrupt offsets.

    Real repositories contain commits with impossible timezone offsets — the
    `requests` history has one recorded at `+518:00` — because git faithfully
    reproduces whatever the original client wrote. The wall-clock half is still
    perfectly good, and dropping the commit would quietly lose its contribution
    to every file it touched, so fall back to the timestamp without the offset.

    Every value returned is timezone-aware, because these dates get compared to
    each other and Python refuses to compare naive against aware.
    """
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass

    try:
        naive = datetime.fromisoformat(raw[:19])  # YYYY-MM-DDTHH:MM:SS
    except ValueError:
        logger.debug("unparseable commit date: %r", raw)
        return None

    logger.debug("commit date %r has a corrupt offset; assuming UTC", raw)
    return naive.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Renames across history
# --------------------------------------------------------------------------


def canonical_path(path: str, aliases: dict[str, str]) -> str:
    """Follow a chain of renames to the path a file has today.

    The `seen` guard is not paranoia: a file renamed A->B->A across history
    would otherwise loop forever.
    """
    seen: set[str] = set()
    while path in aliases and path not in seen:
        seen.add(path)
        path = aliases[path]
    return path


# --------------------------------------------------------------------------
# The log walk
# --------------------------------------------------------------------------


def read_commit_log(
    repo: Repo,
    config: BugfixDetection,
    *,
    until: str | None = None,
    max_commits: int | None = None,
) -> CommitLog:
    """Read a window of history with one `git log --numstat`.

    The single place in Sentinel that parses git history. Both the aggregate
    scan and the training pipeline consume the result, so there is one
    definition of "what a commit touched" rather than two that can disagree.

    Returned oldest-first, because building features from a running tally of
    prior history only works walking forwards. Renames are resolved on the way
    past while iterating newest-first, which is when the aliases are known.
    """
    if not repo.head.is_valid():
        return CommitLog(commits=(), truncated=False, skipped=0)

    args = ["--no-merges", "--numstat", f"--format={LOG_FORMAT}"]
    if max_commits is not None:
        args.append(f"--max-count={max_commits}")
    output = repo.git.log(*args, until if until is not None else "HEAD")

    records = [record for record in output.split(RECORD_SEP) if record.strip()]
    truncated = max_commits is not None and len(records) >= max_commits

    aliases: dict[str, str] = {}
    commits: list[CommitRecord] = []
    skipped = 0

    for record in records:  # newest first
        header, _, body = record.partition("\n")
        fields = header.split(FIELD_SEP)
        if len(fields) != 4:
            skipped += 1
            logger.debug("unparseable log record: %r", header[:80])
            continue

        sha, author, iso_date, subject = fields
        when = parse_commit_date(iso_date)
        if when is None:
            skipped += 1
            continue

        files: list[tuple[str, int, int]] = []
        for old_path, new_path, added, deleted in iter_numstat(body):
            path = canonical_path(new_path, aliases)
            if old_path is not None:
                aliases[old_path] = path
            files.append((path, added, deleted))

        commits.append(
            CommitRecord(
                sha=sha,
                author=author or "unknown",
                when=when,
                subject=subject,
                is_bugfix=looks_like_bugfix(subject, config),
                files=tuple(files),
            )
        )

    if skipped:
        logger.warning("skipped %d commit record(s) that could not be parsed", skipped)

    commits.reverse()
    return CommitLog(commits=tuple(commits), truncated=truncated, skipped=skipped)
