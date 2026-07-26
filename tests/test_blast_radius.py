"""Tests for the dependency graph and the impact walk.

Asserted against a fixture whose graph is written out by hand in `conftest.py`,
including a deliberate circular import and a hub module, because those are the
two shapes that make a naive implementation either hang or say nothing useful.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.blast_radius import (
    JavaParser,
    PythonParser,
    build_graph,
    compute,
    dependents_by_depth,
    impact_for,
    parser_for,
)
from sentinel.config import BlastRadiusSettings

SETTINGS = BlastRadiusSettings()
PY_FILES = [
    "app/settings.py",
    "app/repo.py",
    "app/service.py",
    "app/api.py",
    "app/worker.py",
    "app/report.py",
    "app/cycle_a.py",
    "app/cycle_b.py",
    "app/notes.md",
]


def graph_of(root: Path, settings: BlastRadiusSettings = SETTINGS):
    return build_graph(root, PY_FILES, settings)


# --- the graph ------------------------------------------------------------


def test_graph_indexes_only_parseable_files(graph_repo: Path) -> None:
    dependency_graph = graph_of(graph_repo)

    assert dependency_graph.files_indexed == 8  # the markdown file is skipped
    assert "app/notes.md" not in dependency_graph.graph
    assert dependency_graph.parsers == ("python",)


def test_edges_point_from_importer_to_imported(graph_repo: Path) -> None:
    graph = graph_of(graph_repo).graph

    assert graph.has_edge("app/service.py", "app/repo.py")
    assert graph.has_edge("app/api.py", "app/service.py")
    assert not graph.has_edge("app/repo.py", "app/service.py")


def test_both_import_styles_are_understood(graph_repo: Path) -> None:
    """`from app.service import run` and `import app.settings` both count."""
    graph = graph_of(graph_repo).graph

    assert graph.has_edge("app/worker.py", "app/service.py")
    assert graph.has_edge("app/worker.py", "app/settings.py")


def test_third_party_imports_are_counted_as_unresolved_not_invented(
    graph_repo: Path,
) -> None:
    (graph_repo / "app/extra.py").write_text(
        "import os\nimport requests\nfrom app import repo\n", encoding="utf-8"
    )
    dependency_graph = build_graph(graph_repo, PY_FILES + ["app/extra.py"], SETTINGS)

    assert dependency_graph.unresolved >= 2  # os, requests
    assert dependency_graph.graph.has_edge("app/extra.py", "app/repo.py")


def test_a_file_never_depends_on_itself(graph_repo: Path) -> None:
    graph = graph_of(graph_repo).graph
    assert not any(a == b for a, b in graph.edges)


# --- dependents -----------------------------------------------------------


def test_direct_and_transitive_dependents_are_distinguished(graph_repo: Path) -> None:
    dependency_graph = graph_of(graph_repo)
    impact = impact_for(dependency_graph, "app/repo.py", SETTINGS)

    assert set(impact.direct) == {"app/report.py", "app/service.py"}
    # api and worker reach repo.py only through service.py.
    assert set(impact.transitive) == {"app/api.py", "app/worker.py"}
    assert impact.direct_count == 2
    assert impact.transitive_count == 2


def test_a_leaf_file_affects_nothing(graph_repo: Path) -> None:
    dependency_graph = graph_of(graph_repo)
    impact = impact_for(dependency_graph, "app/api.py", SETTINGS)

    assert impact.direct == ()
    assert impact.transitive == ()
    assert impact.total == 0


def test_depth_limits_how_far_the_walk_goes(graph_repo: Path) -> None:
    dependency_graph = graph_of(graph_repo)

    shallow = impact_for(dependency_graph, "app/settings.py", BlastRadiusSettings(max_depth=1))
    deep = impact_for(dependency_graph, "app/settings.py", BlastRadiusSettings(max_depth=5))

    assert shallow.transitive_count == 0  # one hop only
    assert deep.transitive_count > shallow.transitive_count
    assert shallow.depth_capped  # there was more to find


def test_depth_cap_is_not_reported_when_the_walk_finished(graph_repo: Path) -> None:
    dependency_graph = graph_of(graph_repo)
    impact = impact_for(dependency_graph, "app/repo.py", BlastRadiusSettings(max_depth=9))
    assert not impact.depth_capped


# --- the awkward cases ----------------------------------------------------


def test_a_circular_import_terminates_and_is_flagged(graph_repo: Path) -> None:
    """Without a visited set this walk would never return."""
    dependency_graph = graph_of(graph_repo)

    assert "app/cycle_a.py" in dependency_graph.cyclic
    assert "app/cycle_b.py" in dependency_graph.cyclic

    impact = impact_for(dependency_graph, "app/cycle_a.py", SETTINGS)
    assert impact.in_cycle
    assert set(impact.direct) == {"app/cycle_b.py"}
    # cycle_a is its own transitive dependent via cycle_b, and must not be listed
    # as impacted by itself.
    assert "app/cycle_a.py" not in impact.transitive


def test_a_file_outside_a_cycle_is_not_flagged(graph_repo: Path) -> None:
    dependency_graph = graph_of(graph_repo)
    assert not impact_for(dependency_graph, "app/repo.py", SETTINGS).in_cycle


def test_a_hub_is_named_as_a_hub_rather_than_listed_exhaustively(
    graph_repo: Path,
) -> None:
    dependency_graph = graph_of(graph_repo)
    # settings.py is imported directly by four files in the fixture.
    impact = impact_for(dependency_graph, "app/settings.py", BlastRadiusSettings(hub_dependents=3))

    assert impact.is_hub
    assert impact.direct_count >= 3


def test_listings_are_truncated_but_counts_stay_complete(graph_repo: Path) -> None:
    dependency_graph = graph_of(graph_repo)
    impact = impact_for(
        dependency_graph, "app/settings.py", BlastRadiusSettings(max_listed=2)
    )

    assert len(impact.direct) == 2
    assert impact.direct_count > 2  # the full number survives truncation


# --- whole-change rollup --------------------------------------------------


def test_compute_unions_impact_across_the_change(graph_repo: Path) -> None:
    blast = compute(graph_repo, PY_FILES, ["app/repo.py", "app/service.py"], SETTINGS)

    assert blast.analyzed
    # service.py is part of the change, so it is not "impacted by" it.
    assert "app/service.py" not in blast.direct
    assert set(blast.direct) == {"app/api.py", "app/report.py", "app/worker.py"}


def test_a_file_is_never_both_direct_and_transitive(graph_repo: Path) -> None:
    blast = compute(graph_repo, PY_FILES, ["app/settings.py"], SETTINGS)
    assert not set(blast.direct) & set(blast.transitive)


def test_compute_reports_hubs_and_cycles_for_the_change(graph_repo: Path) -> None:
    blast = compute(
        graph_repo,
        PY_FILES,
        ["app/settings.py", "app/cycle_a.py"],
        BlastRadiusSettings(hub_dependents=3),
    )

    assert "app/settings.py" in blast.hubs
    assert "app/cycle_a.py" in blast.cycle_files


def test_per_file_impact_is_ordered_by_reach(graph_repo: Path) -> None:
    blast = compute(graph_repo, PY_FILES, ["app/api.py", "app/repo.py"], SETTINGS)
    assert blast.files[0].path == "app/repo.py"  # api.py affects nothing


def test_too_many_changed_files_are_capped_and_reported(graph_repo: Path) -> None:
    blast = compute(
        graph_repo, PY_FILES, PY_FILES, BlastRadiusSettings(max_analyzed_files=2)
    )
    assert len(blast.files) == 2
    assert blast.files_omitted == len(PY_FILES) - 2


def test_disabling_blast_radius_reports_not_analyzed(graph_repo: Path) -> None:
    blast = compute(graph_repo, PY_FILES, ["app/repo.py"], BlastRadiusSettings(enabled=False))
    assert not blast.analyzed
    assert blast.total == 0


def test_a_repo_with_no_parseable_files_is_not_analyzed(tmp_path: Path) -> None:
    """"Could not look" must be distinguishable from "nothing depends on it"."""
    (tmp_path / "README.md").write_text("# docs\n", encoding="utf-8")
    blast = compute(tmp_path, ["README.md"], ["README.md"], SETTINGS)
    assert not blast.analyzed


# --- parser registry ------------------------------------------------------


def test_parsers_are_chosen_by_extension() -> None:
    assert isinstance(parser_for("app/core.py"), PythonParser)
    assert isinstance(parser_for("src/Main.java"), JavaParser)
    assert parser_for("README.md") is None
    assert parser_for("Makefile") is None


def test_java_imports_build_a_graph(java_repo: Path) -> None:
    paths = [
        "src/main/java/com/shop/PaymentService.java",
        "src/main/java/com/shop/CheckoutController.java",
        "src/main/java/com/shop/data/PaymentRepo.java",
    ]
    dependency_graph = build_graph(java_repo, paths, SETTINGS)

    assert dependency_graph.parsers == ("java",)
    assert dependency_graph.graph.has_edge(
        "src/main/java/com/shop/PaymentService.java",
        "src/main/java/com/shop/data/PaymentRepo.java",
    )

    impact = impact_for(dependency_graph, "src/main/java/com/shop/data/PaymentRepo.java", SETTINGS)
    assert impact.direct == ("src/main/java/com/shop/PaymentService.java",)
    assert impact.transitive == ("src/main/java/com/shop/CheckoutController.java",)


def test_java_wildcard_imports_resolve_to_the_package(tmp_path: Path) -> None:
    (tmp_path / "com/x").mkdir(parents=True)
    (tmp_path / "com/x/Foo.java").write_text("package com.x;\nclass Foo {}\n", encoding="utf-8")
    (tmp_path / "com/x/Bar.java").write_text("package com.x;\nclass Bar {}\n", encoding="utf-8")
    (tmp_path / "com/Main.java").write_text(
        "package com;\nimport com.x.*;\nclass Main {}\n", encoding="utf-8"
    )

    paths = ["com/x/Foo.java", "com/x/Bar.java", "com/Main.java"]
    graph = build_graph(tmp_path, paths, SETTINGS).graph

    assert graph.has_edge("com/Main.java", "com/x/Foo.java")
    assert graph.has_edge("com/Main.java", "com/x/Bar.java")


def test_static_java_imports_resolve_to_the_class(tmp_path: Path) -> None:
    (tmp_path / "com").mkdir()
    (tmp_path / "com/Util.java").write_text(
        "package com;\nclass Util { static int f() { return 1; } }\n", encoding="utf-8"
    )
    (tmp_path / "com/Caller.java").write_text(
        "package com;\nimport static com.Util.f;\nclass Caller {}\n", encoding="utf-8"
    )

    graph = build_graph(tmp_path, ["com/Util.java", "com/Caller.java"], SETTINGS).graph
    assert graph.has_edge("com/Caller.java", "com/Util.java")


# --- Python parser details ------------------------------------------------


def test_relative_imports_resolve_upwards(tmp_path: Path) -> None:
    (tmp_path / "pkg/sub").mkdir(parents=True)
    for rel, content in {
        "pkg/__init__.py": "",
        "pkg/shared.py": "X = 1\n",
        "pkg/sub/__init__.py": "",
        "pkg/sub/leaf.py": "from .. import shared\nfrom . import sibling\n",
        "pkg/sub/sibling.py": "Y = 2\n",
    }.items():
        (tmp_path / rel).write_text(content, encoding="utf-8")

    paths = [
        "pkg/__init__.py",
        "pkg/shared.py",
        "pkg/sub/__init__.py",
        "pkg/sub/leaf.py",
        "pkg/sub/sibling.py",
    ]
    graph = build_graph(tmp_path, paths, SETTINGS).graph

    assert graph.has_edge("pkg/sub/leaf.py", "pkg/shared.py")
    assert graph.has_edge("pkg/sub/leaf.py", "pkg/sub/sibling.py")


def test_a_package_is_named_by_its_directory() -> None:
    parser = PythonParser()
    assert "pkg" in parser.module_names("pkg/__init__.py", "")
    assert "pkg.mod" in parser.module_names("pkg/mod.py", "")


def test_an_unparseable_python_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def (((:\n", encoding="utf-8")
    (tmp_path / "fine.py").write_text("import broken\n", encoding="utf-8")

    dependency_graph = build_graph(tmp_path, ["broken.py", "fine.py"], SETTINGS)

    assert dependency_graph.files_indexed == 2
    # The broken file yields no references, but is still importable by others.
    assert dependency_graph.graph.has_edge("fine.py", "broken.py")


def test_an_ambiguous_module_name_is_dropped_rather_than_guessed(tmp_path: Path) -> None:
    """Two files named `utils.py` must not invent an edge to a coin flip."""
    for rel in ("a/utils.py", "b/utils.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("import utils\n", encoding="utf-8")

    dependency_graph = build_graph(
        tmp_path, ["a/utils.py", "b/utils.py", "main.py"], SETTINGS
    )

    assert dependency_graph.ambiguous >= 1
    assert dependency_graph.graph.number_of_edges() == 0


# --- traversal primitive --------------------------------------------------


def test_walking_from_an_unknown_path_is_empty(graph_repo: Path) -> None:
    graph = graph_of(graph_repo).graph
    layers, capped = dependents_by_depth(graph, "does/not/exist.py", 3)
    assert layers == []
    assert not capped


@pytest.mark.parametrize("depth", [1, 2, 3, 10])
def test_the_walk_always_terminates_on_a_cyclic_graph(graph_repo: Path, depth: int) -> None:
    graph = graph_of(graph_repo).graph
    layers, _ = dependents_by_depth(graph, "app/cycle_b.py", depth)
    seen = [node for layer in layers for node in layer]
    assert len(seen) == len(set(seen))  # no node visited twice
