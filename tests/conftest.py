"""Shared fixtures.

The git tests run against a real repository built in a temp directory rather
than a mock: the whole point of `git_reader` is that it agrees with git, and a
mocked git proves nothing. No network is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from git import Actor, Repo

from sentinel.config import get_settings

ALICE = Actor("Alice", "alice@example.com")
BOB = Actor("Bob", "bob@example.com")


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch: pytest.MonkeyPatch):
    """No test may reach the network, whatever is in the developer's `.env`.

    `Settings` reads `.env` from the working directory, so a real key sitting in
    the project root would otherwise make the AI tests call NVIDIA for real. An
    empty environment variable outranks the dotenv file, which turns the LLM off
    at the source. Tests that want a key set it themselves.

    The cache is cleared either side because settings are memoised per process.
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def commit_file(repo: Repo, rel_path: str, content: str, message: str, author: Actor) -> str:
    """Write a file and commit it, returning the commit hash."""
    path = Path(repo.working_tree_dir) / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    repo.index.add([rel_path])
    return repo.index.commit(message, author=author, committer=author).hexsha


@pytest.fixture
def graph_repo(tmp_path: Path) -> Path:
    """A repository with a hand-built import graph, including the awkward cases.

    Edges (importer -> imported)::

        api.py      -> service.py -> repo.py -> settings.py
        worker.py   -> service.py
        report.py   -> repo.py
        cycle_a.py <-> cycle_b.py          (deliberate circular import)
        everything  -> settings.py         (the hub)

    So `repo.py` has 1 direct dependent (service) and 3 transitive
    (api, worker, report), and `settings.py` is imported by nearly everything.
    """
    files = {
        "app/settings.py": "TIMEOUT = 30\n",
        "app/repo.py": "from app import settings\n\n\ndef fetch():\n    return settings.TIMEOUT\n",
        "app/service.py": "from app import repo, settings\n\n\ndef run():\n    return repo.fetch()\n",
        "app/api.py": "from app import service\nfrom app import settings\n\n\ndef handle():\n    return service.run()\n",
        "app/worker.py": "from app.service import run\nimport app.settings\n",
        "app/report.py": "from app.repo import fetch\n",
        "app/cycle_a.py": "from app import cycle_b\nfrom app import settings\n",
        "app/cycle_b.py": "from app import cycle_a\n",
        "app/notes.md": "not code\n",
    }
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    repo = Repo.init(tmp_path)
    with repo.config_writer() as writer:
        writer.set_value("user", "name", "Alice")
        writer.set_value("user", "email", "alice@example.com")
    repo.index.add(list(files))
    repo.index.commit("initial graph", author=ALICE, committer=ALICE)
    return tmp_path


@pytest.fixture
def java_repo(tmp_path: Path) -> Path:
    """A small Java tree, to prove the parser registry is not Python-only."""
    files = {
        "src/main/java/com/shop/PaymentService.java": (
            "package com.shop;\n\n"
            "import com.shop.data.PaymentRepo;\n\n"
            "public class PaymentService {}\n"
        ),
        "src/main/java/com/shop/CheckoutController.java": (
            "package com.shop;\n\n"
            "import com.shop.PaymentService;\n\n"
            "public class CheckoutController {}\n"
        ),
        "src/main/java/com/shop/data/PaymentRepo.java": (
            "package com.shop.data;\n\npublic class PaymentRepo {}\n"
        ),
    }
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def szz_repo(tmp_path: Path) -> Repo:
    """A repository with a bug deliberately introduced and later fixed.

    The shape is what SZZ has to detect:

    * commit 1 writes `calc.py` with a correct line          (clean)
    * commit 2 rewrites that line into a bug                 (bug-inducing)
    * commit 3 is unrelated                                  (clean)
    * commit 4 says "fix:" and repairs the line from commit 2 (the fix)

    Blaming commit 4's deleted line must land on commit 2, and only commit 2.
    """
    repo = Repo.init(tmp_path)
    with repo.config_writer() as writer:
        writer.set_value("user", "name", "Alice")
        writer.set_value("user", "email", "alice@example.com")

    commit_file(repo, "calc.py", "def total(items):\n    return sum(items)\n", "add calc", ALICE)
    bug = commit_file(
        repo,
        "calc.py",
        "def total(items):\n    return sum(items) + 1\n",
        "speed up total()",
        BOB,
    )
    commit_file(repo, "notes.md", "# notes\n", "add notes", ALICE)
    commit_file(
        repo,
        "calc.py",
        "def total(items):\n    return sum(items)\n",
        "fix: total() was off by one",
        ALICE,
    )

    repo.bug_inducing_sha = bug  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Repo:
    """A five-commit repository with two authors and two bug-fix commits.

    Layout and history are fixed so the assertions can be exact:

    * ``app/core.py``      — 3 commits, 1 of them a bug fix, Alice owns 2/3
    * ``app/util.py``      — 1 commit by Alice, no tests anywhere
    * ``tests/test_core.py`` — 1 commit by Bob, closing an issue key
    """
    repo = Repo.init(tmp_path)
    with repo.config_writer() as writer:
        writer.set_value("user", "name", "Alice")
        writer.set_value("user", "email", "alice@example.com")

    commit_file(repo, "app/core.py", "def run():\n    return 1\n", "initial commit", ALICE)
    commit_file(
        repo,
        "app/core.py",
        "def run():\n    return 2\n",
        "fix: off-by-one in run()",
        BOB,
    )
    commit_file(
        repo,
        "app/core.py",
        "def run():\n    return 3\n",
        "refactor run() for clarity",
        ALICE,
    )
    commit_file(repo, "app/util.py", "def helper():\n    pass\n", "add util helper", ALICE)
    commit_file(
        repo,
        "tests/test_core.py",
        "def test_run():\n    assert True\n",
        "resolved PROJ-42 with a regression test",
        BOB,
    )
    return repo
