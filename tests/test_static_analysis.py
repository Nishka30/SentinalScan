"""Tests for complexity measurement and test-file detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.static_analysis import (
    TestIndex,
    analyze_complexity,
    candidate_test_names,
    is_test_path,
)

BRANCHY = """
def classify(n):
    if n < 0:
        return "neg"
    elif n == 0:
        return "zero"
    elif n < 10:
        return "small"
    elif n < 100:
        return "medium"
    else:
        return "large"
"""


def test_complexity_reports_the_worst_function(tmp_path: Path) -> None:
    source = tmp_path / "classify.py"
    source.write_text(BRANCHY + "\ndef trivial():\n    return 1\n", encoding="utf-8")

    info = analyze_complexity(source)

    assert info is not None
    assert info.function_count == 2
    assert info.max_ccn >= 5  # the branchy one, not the average
    assert info.max_ccn > info.average_ccn
    assert info.nloc > 0


def test_unsupported_and_missing_files_yield_no_signal(tmp_path: Path) -> None:
    """`None` means "we cannot say", and the complexity rule then stays silent."""
    prose = tmp_path / "README.md"
    prose.write_text("# Just prose\n\nNo functions here.\n", encoding="utf-8")
    assert analyze_complexity(prose) is None

    empty = tmp_path / "constants.py"
    empty.write_text("TIMEOUT = 30\n", encoding="utf-8")
    assert analyze_complexity(empty) is None

    assert analyze_complexity(tmp_path / "deleted.py") is None


def test_binary_content_does_not_crash(tmp_path: Path) -> None:
    blob = tmp_path / "weird.py"
    blob.write_bytes(b"\x00\xff\xfe garbage \x00")
    analyze_complexity(blob)  # must not raise


# --- test-file detection --------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_core.py",
        "test_core.py",
        "core_test.py",
        "app/__tests__/core.py",
        "src/core.test.ts",
        "src/core.spec.js",
        "src/main/java/CoreTest.java",
        "src/main/java/TestCore.java",
        "spec/core_spec.rb",
    ],
)
def test_test_paths_are_recognised(path: str) -> None:
    assert is_test_path(path)


@pytest.mark.parametrize(
    "path",
    ["app/core.py", "src/latest.py", "app/contest.py", "src/main/java/Test.java"],
)
def test_source_paths_are_not_mistaken_for_tests(path: str) -> None:
    assert not is_test_path(path)


def test_candidate_names_cover_the_common_conventions() -> None:
    names = candidate_test_names("app/core.py")
    assert "test_core.py" in names
    assert "core_test.py" in names
    assert "core.test.py" in names

    java = candidate_test_names("src/Core.java")
    assert "CoreTest.java" in java
    assert "TestCore.java" in java


def test_index_finds_the_test_file_and_notices_it_changed() -> None:
    index = TestIndex(["app/core.py", "tests/test_core.py", "app/util.py"])

    signal = index.signal_for("app/core.py", changed_paths={"app/core.py", "tests/test_core.py"})
    assert signal.has_tests
    assert signal.tests_changed
    assert signal.test_paths == ("tests/test_core.py",)


def test_index_notices_when_tests_exist_but_were_not_touched() -> None:
    index = TestIndex(["app/core.py", "tests/test_core.py"])

    signal = index.signal_for("app/core.py", changed_paths={"app/core.py"})
    assert signal.has_tests
    assert not signal.tests_changed


def test_index_reports_no_tests_when_there_are_none() -> None:
    index = TestIndex(["app/core.py", "app/util.py"])

    signal = index.signal_for("app/util.py", changed_paths={"app/util.py"})
    assert not signal.has_tests
    assert not signal.is_test_file


@pytest.mark.parametrize(
    "path",
    ["pyproject.toml", ".claude/settings.json", "README.md", "Dockerfile", "data/fixture.csv"],
)
def test_non_code_files_are_exempt_from_the_test_signal(path: str) -> None:
    """"No test file found for pyproject.toml" is noise, and noise costs trust."""
    index = TestIndex([path, "app/core.py"])

    signal = index.signal_for(path, changed_paths={path})
    assert not signal.is_code
    assert not signal.has_tests


@pytest.mark.parametrize("path", ["app/core.py", "src/Core.java", "web/app.tsx"])
def test_code_files_are_subject_to_the_test_signal(path: str) -> None:
    index = TestIndex([path])
    assert index.signal_for(path, changed_paths=set()).is_code


def test_a_test_file_reports_itself_as_such() -> None:
    index = TestIndex(["app/core.py", "tests/test_core.py"])

    signal = index.signal_for("tests/test_core.py", changed_paths=set())
    assert signal.is_test_file
    assert signal.test_paths == ()


def test_same_named_source_files_elsewhere_are_not_treated_as_tests() -> None:
    """`lib/core.py` is not a test for `app/core.py` just by sharing a name."""
    index = TestIndex(["app/core.py", "lib/core.py"])

    signal = index.signal_for("app/core.py", changed_paths=set())
    assert not signal.has_tests


def test_same_named_file_in_a_test_directory_does_count() -> None:
    index = TestIndex(["app/core.py", "app/__tests__/core.py"])

    signal = index.signal_for("app/core.py", changed_paths=set())
    assert signal.test_paths == ("app/__tests__/core.py",)
