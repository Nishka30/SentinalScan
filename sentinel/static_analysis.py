"""Measure the code itself, not its history.

Split from `git_reader` because these signals need no repository at all — only
the files on disk — which keeps both halves independently testable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lizard

logger = logging.getLogger(__name__)

#: Directory names that mark everything beneath them as test code.
TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__", "spec", "specs", "testing"})

#: Extensions we expect to have unit tests. Config files, docs, lockfiles and
#: templates are excluded: "no test file found for pyproject.toml" is noise, and
#: noise is what stops people trusting the score.
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".java",
        ".kt",
        ".scala",
        ".groovy",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".go",
        ".rb",
        ".rs",
        ".cs",
        ".php",
        ".swift",
        ".m",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
    }
)


@dataclass(frozen=True)
class ComplexityInfo:
    """Complexity of one file. `max_ccn` is what the risk rules care about.

    Averages hide the problem: a 40-branch function in a file of trivial getters
    averages out to "fine", but it is the function that will break.
    """

    average_ccn: float
    max_ccn: int
    nloc: int
    function_count: int


@dataclass(frozen=True)
class TestSignal:
    """Whether a changed file has tests, and whether they were touched too."""

    is_test_file: bool
    test_paths: tuple[str, ...]
    changed_test_paths: tuple[str, ...]
    #: False for files no reasonable person would unit-test (config, docs, data).
    #: The test rules stay silent for those rather than inventing a problem.
    is_code: bool = True

    @property
    def has_tests(self) -> bool:
        return bool(self.test_paths)

    @property
    def tests_changed(self) -> bool:
        return bool(self.changed_test_paths)


def analyze_complexity(path: Path) -> ComplexityInfo | None:
    """Complexity of one file, or None when there is no signal to be had.

    None covers every "we cannot say" case — a language lizard has no parser
    for, a deleted file, a file with no functions in it. The caller treats them
    identically: the complexity rule stays silent rather than guessing.
    """
    if not path.is_file():
        return None
    try:
        result = lizard.analyze_file(str(path))
    except Exception as exc:  # lizard raises assorted parse/decode errors
        logger.debug("no complexity for %s: %s", path, exc)
        return None

    functions = list(result.function_list)
    if not functions:
        return None

    complexities = [int(f.cyclomatic_complexity) for f in functions]
    return ComplexityInfo(
        average_ccn=round(sum(complexities) / len(complexities), 1),
        max_ccn=max(complexities),
        nloc=int(result.nloc),
        function_count=len(functions),
    )


def is_test_path(path: str) -> bool:
    """True when a path is itself test code, by the usual naming conventions."""
    parts = path.split("/")
    if any(part.lower() in TEST_DIR_NAMES for part in parts[:-1]):
        return True

    name = parts[-1]
    stem = name.split(".")[0]
    lowered = name.lower()

    if ".test." in lowered or ".spec." in lowered:
        return True
    if stem.startswith("test_") or stem.endswith("_test"):
        return True

    # `CoreTest.java` and `TestCore.java` yes; a file named exactly `Test.java`
    # is a class called Test, which is production code often enough to matter.
    for affix in ("Test", "Tests"):
        if stem.endswith(affix) and len(stem) > len(affix):
            return True
    return stem.startswith("Test") and len(stem) > len("Test")


def is_code_path(path: str) -> bool:
    """True when a file is the kind of thing that ought to have unit tests."""
    name = path.split("/")[-1]
    if "." not in name:
        return False
    return f".{name.rpartition('.')[2].lower()}" in CODE_EXTENSIONS


def candidate_test_names(path: str) -> tuple[str, ...]:
    """Filenames that would be the tests for `path`, across common conventions.

    Names only, not paths: test files sit anywhere from `tests/` at the repo
    root to right beside the source, and searching by name finds both.
    """
    name = path.split("/")[-1]
    if "." not in name:
        return ()
    stem, _, ext = name.rpartition(".")

    return (
        # Python
        f"test_{stem}.{ext}",
        f"{stem}_test.{ext}",
        # JS / TS and friends
        f"{stem}.test.{ext}",
        f"{stem}.spec.{ext}",
        # Java / C#
        f"{stem}Test.{ext}",
        f"{stem}Tests.{ext}",
        f"Test{stem}.{ext}",
        # Same name, parked in a test directory — matched via the index
        name,
    )


class TestIndex:
    """Filename lookup over every path in the repository.

    Built once per run because the alternative — globbing the tree per changed
    file — is the same work repeated N times.
    """

    def __init__(self, repo_paths: Iterable[str]) -> None:
        self._by_name: dict[str, list[str]] = defaultdict(list)
        for path in repo_paths:
            self._by_name[path.split("/")[-1]].append(path)

    def signal_for(self, path: str, changed_paths: set[str]) -> TestSignal:
        """Find the tests for `path` and note which of them changed too."""
        if is_test_path(path):
            # A test file needs no tests of its own; saying otherwise would
            # penalise the very thing we are asking people to write.
            return TestSignal(is_test_file=True, test_paths=(), changed_test_paths=())

        if not is_code_path(path):
            return TestSignal(
                is_test_file=False, test_paths=(), changed_test_paths=(), is_code=False
            )

        name = path.split("/")[-1]
        matches: set[str] = set()
        for candidate in candidate_test_names(path):
            for found in self._by_name.get(candidate, ()):
                if found == path or not is_test_path(found):
                    continue
                # `name` itself only counts when it lives in a test directory,
                # otherwise every same-named file across the tree would match.
                if candidate == name and not _in_test_dir(found):
                    continue
                matches.add(found)

        tests = tuple(sorted(matches))
        return TestSignal(
            is_test_file=False,
            test_paths=tests,
            changed_test_paths=tuple(t for t in tests if t in changed_paths),
        )


def _in_test_dir(path: str) -> bool:
    return any(part.lower() in TEST_DIR_NAMES for part in path.split("/")[:-1])
