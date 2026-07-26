"""What else this change can break.

Builds an in-memory import graph with networkx and walks it *backwards* from the
changed files: an edge points from importer to imported, so the files that
depend on a change are its ancestors in that graph.

In memory on purpose — a graph database would be permanent infrastructure for a
graph that is rebuilt from scratch in under a second.

Three real-world messes are handled deliberately rather than incidentally:

* **Cycles.** `a` imports `b` imports `a` happens in real code. The traversal
  carries a visited set, so a cycle terminates instead of looping, and files
  that sit in one are flagged — their "dependents" include things they also
  depend on, which is worth knowing.
* **Hubs.** A settings or types module is imported by everything. Reporting
  "this affects all 400 files" is technically true and completely useless, so
  depth is capped, listings are truncated with the full count kept, and hubs are
  named as hubs.
* **Direct vs transitive.** Something that imports the changed file directly is
  a different proposition from something five hops away. They are never merged.

Language support is pluggable: a parser declares its extensions, how to name the
modules a file provides, and what a file references. The graph logic below never
needs to know which language it is looking at.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import networkx as nx

from sentinel.config import BlastRadiusSettings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# The parser contract
# --------------------------------------------------------------------------


@runtime_checkable
class LanguageParser(Protocol):
    """How to read one language's dependencies.

    Adding a language means adding one of these to `PARSERS`. Nothing in the
    graph or traversal code changes.
    """

    name: str
    extensions: frozenset[str]

    def module_names(self, path: str, source: str) -> set[str]:
        """Names by which other files may refer to this file."""

    def references(self, path: str, source: str) -> set[str]:
        """Module names this file depends on."""


def _dotted_suffixes(parts: list[str]) -> set[str]:
    """Every trailing dotted name for a path, longest first.

    `src/requests/utils.py` can legitimately be imported as `requests.utils`
    from inside the package or `src.requests.utils` from outside it, and we do
    not know the source root. Indexing every suffix covers both; ambiguity is
    resolved later by requiring a unique match, so a name shared by two files
    is simply dropped rather than guessed at.
    """
    return {".".join(parts[index:]) for index in range(len(parts))}


class PythonParser:
    """Python imports, via `ast` rather than regex.

    `ast` is used because it gets relative imports, parenthesised multi-line
    imports and `as` aliases right for free, and because a file that does not
    parse is better skipped than half-understood.
    """

    name = "python"
    extensions = frozenset({".py", ".pyi"})

    def module_names(self, path: str, source: str) -> set[str]:
        parts = path.removesuffix(".pyi").removesuffix(".py").split("/")
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]  # a package is named by its directory
        return _dotted_suffixes(parts) if parts else set()

    def references(self, path: str, source: str) -> set[str]:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            logger.debug("could not parse %s: %s", path, exc)
            return set()

        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.update(self._from_import(path, node))
        return found

    def _from_import(self, path: str, node: ast.ImportFrom) -> set[str]:
        base = node.module or ""
        if node.level:
            base = self._relative_base(path, node.level, node.module)
            if base is None:
                return set()

        if not base:
            return set()
        # `from pkg import thing` — `thing` may be a submodule or just a name,
        # so offer both and let resolution decide which exists.
        return {base} | {f"{base}.{alias.name}" for alias in node.names}

    def _relative_base(self, path: str, level: int, module: str | None) -> str | None:
        """Turn `from ..pkg import x` into an absolute dotted name."""
        parts = path.removesuffix(".pyi").removesuffix(".py").split("/")
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        package = parts[:-1] if parts and parts[-1] != "" else parts

        climb = level - 1
        if climb:
            if climb > len(package):
                return None
            package = package[: len(package) - climb]

        if module:
            package = package + module.split(".")
        return ".".join(package) if package else None


class JavaParser:
    """Java imports and package declarations.

    Same-package references need no import statement and are therefore invisible
    here; that is a known gap, not an oversight — catching them needs real type
    resolution rather than a line-oriented read.
    """

    name = "java"
    extensions = frozenset({".java"})

    _PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
    _IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+(?:\.\*)?)\s*;", re.MULTILINE)

    def module_names(self, path: str, source: str) -> set[str]:
        class_name = path.split("/")[-1].removesuffix(".java")
        names = _dotted_suffixes(path.removesuffix(".java").split("/"))

        declared = self._PACKAGE.search(source)
        if declared:
            names.add(f"{declared.group(1)}.{class_name}")
        return names

    def references(self, path: str, source: str) -> set[str]:
        found = set()
        for match in self._IMPORT.finditer(source):
            target = match.group(1)
            # `import static com.x.Foo.bar;` refers to the class, not the method.
            if "static" in match.group(0) and not target.endswith(".*"):
                target = target.rpartition(".")[0]
            if target:
                found.add(target)
        return found


#: Registry. Append a parser here to support another language.
PARSERS: tuple[LanguageParser, ...] = (PythonParser(), JavaParser())


def parser_for(path: str) -> LanguageParser | None:
    suffix = f".{path.rpartition('.')[2].lower()}" if "." in path else ""
    for parser in PARSERS:
        if suffix in parser.extensions:
            return parser
    return None


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------


@dataclass
class DependencyGraph:
    """An import graph: an edge from importer to imported."""

    graph: nx.DiGraph
    files_indexed: int
    unresolved: int
    ambiguous: int
    parsers: tuple[str, ...]
    #: Files that sit in a strongly connected component larger than one node.
    cyclic: frozenset[str]

    @property
    def edges(self) -> int:
        return self.graph.number_of_edges()


def build_graph(
    root: Path, paths: Iterable[str], settings: BlastRadiusSettings
) -> DependencyGraph:
    """Read every parseable file once and wire up the graph."""
    sources: dict[str, str] = {}
    index: dict[str, set[str]] = {}
    used_parsers: set[str] = set()

    for path in paths:
        if len(sources) >= settings.max_files:
            logger.warning("dependency graph capped at %d files", settings.max_files)
            break
        parser = parser_for(path)
        if parser is None:
            continue
        source = _read(root / path, settings.max_bytes_per_file)
        if source is None:
            continue
        sources[path] = source
        used_parsers.add(parser.name)
        for name in parser.module_names(path, source):
            index.setdefault(name, set()).add(path)

    graph = nx.DiGraph()
    graph.add_nodes_from(sources)
    unresolved = 0
    ambiguous = 0

    for path, source in sources.items():
        parser = parser_for(path)
        assert parser is not None  # only parseable files reached `sources`
        for target in parser.references(path, source):
            matches = _resolve(target, index)
            if not matches:
                unresolved += 1  # third-party or stdlib: not our problem
                continue
            # A wildcard (`import com.x.*`) is *supposed* to hit many files;
            # only a specific reference matching several is genuinely ambiguous.
            if len(matches) > 1 and not target.endswith(".*"):
                ambiguous += 1
                continue
            for imported in matches:
                if imported != path:
                    graph.add_edge(path, imported)

    cyclic = frozenset(
        node
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1
        for node in component
    )

    return DependencyGraph(
        graph=graph,
        files_indexed=len(sources),
        unresolved=unresolved,
        ambiguous=ambiguous,
        parsers=tuple(sorted(used_parsers)),
        cyclic=cyclic,
    )


def _read(path: Path, max_bytes: int) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("could not read %s: %s", path, exc)
        return None


def _resolve(target: str, index: dict[str, set[str]]) -> set[str]:
    """Map a dotted reference onto repository files.

    Longest prefix wins, because `from a.b import c` should resolve to `a.b.c`
    when that module exists and to `a.b` when `c` is just a name inside it. A
    prefix matching more than one file is returned as-is so the caller can
    discard it: guessing between two same-named modules invents dependencies.
    """
    if target.endswith(".*"):
        prefix = target[:-2]
        depth = len(prefix.split(".")) + 1
        return {
            path
            for name, paths in index.items()
            if name.startswith(f"{prefix}.") and len(name.split(".")) == depth
            for path in paths
        }

    parts = target.split(".")
    for stop in range(len(parts), 0, -1):
        candidate = ".".join(parts[:stop])
        if candidate in index:
            return set(index[candidate])
    return set()


# --------------------------------------------------------------------------
# Impact
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileImpact:
    """Who depends on one changed file."""

    path: str
    direct: tuple[str, ...]
    transitive: tuple[str, ...]
    direct_count: int
    transitive_count: int
    in_cycle: bool
    depth_capped: bool
    is_hub: bool

    @property
    def total(self) -> int:
        return self.direct_count + self.transitive_count


@dataclass(frozen=True)
class BlastRadius:
    """The impact of a whole change."""

    files: tuple[FileImpact, ...]
    direct: tuple[str, ...]
    transitive: tuple[str, ...]
    direct_count: int
    transitive_count: int
    hubs: tuple[str, ...]
    cycle_files: tuple[str, ...]
    depth: int
    depth_capped: bool
    listed_limit: int
    graph_files: int
    graph_edges: int
    parsers: tuple[str, ...]
    analyzed: bool
    files_omitted: int = 0

    @property
    def total(self) -> int:
        return self.direct_count + self.transitive_count


def empty(reason_analyzed: bool = False) -> BlastRadius:
    """A blast radius for "we did not or could not look"."""
    return BlastRadius(
        files=(),
        direct=(),
        transitive=(),
        direct_count=0,
        transitive_count=0,
        hubs=(),
        cycle_files=(),
        depth=0,
        depth_capped=False,
        listed_limit=0,
        graph_files=0,
        graph_edges=0,
        parsers=(),
        analyzed=reason_analyzed,
    )


def dependents_by_depth(
    graph: nx.DiGraph, path: str, max_depth: int
) -> tuple[list[set[str]], bool]:
    """Breadth-first walk *up* the graph, one layer per hop.

    Returns the layers and whether the walk stopped at the depth cap with more
    still to find. The visited set is what makes cycles terminate.
    """
    if path not in graph:
        return [], False

    seen = {path}
    frontier = {path}
    layers: list[set[str]] = []

    for _ in range(max_depth):
        nxt: set[str] = set()
        for node in frontier:
            for predecessor in graph.predecessors(node):
                if predecessor not in seen:
                    seen.add(predecessor)
                    nxt.add(predecessor)
        if not nxt:
            return layers, False
        layers.append(nxt)
        frontier = nxt

    # Anything left unvisited beyond the last layer means the cap bit.
    capped = any(
        predecessor not in seen
        for node in frontier
        for predecessor in graph.predecessors(node)
    )
    return layers, capped


def impact_for(
    dependency_graph: DependencyGraph, path: str, settings: BlastRadiusSettings
) -> FileImpact:
    """Direct and transitive dependents of one file, truncated for readability."""
    layers, capped = dependents_by_depth(
        dependency_graph.graph, path, settings.max_depth
    )
    direct = layers[0] if layers else set()
    transitive: set[str] = set()
    for layer in layers[1:]:
        transitive |= layer

    return FileImpact(
        path=path,
        direct=_listed(direct, settings.max_listed),
        transitive=_listed(transitive, settings.max_listed),
        direct_count=len(direct),
        transitive_count=len(transitive),
        in_cycle=path in dependency_graph.cyclic,
        depth_capped=capped,
        is_hub=len(direct) >= settings.hub_dependents,
    )


def _listed(paths: set[str], limit: int) -> tuple[str, ...]:
    """Names to actually show. Counts are reported separately and in full."""
    return tuple(sorted(paths)[:limit])


def compute(
    root: Path,
    all_paths: Iterable[str],
    changed_paths: Iterable[str],
    settings: BlastRadiusSettings,
    *,
    graph: DependencyGraph | None = None,
) -> BlastRadius:
    """Blast radius for a change: per file, and unioned across the change."""
    if not settings.enabled:
        return empty()

    changed = list(changed_paths)
    dependency_graph = graph or build_graph(root, all_paths, settings)
    if dependency_graph.files_indexed == 0:
        return empty(reason_analyzed=False)

    considered = changed[: settings.max_analyzed_files]
    omitted = len(changed) - len(considered)

    impacts = [impact_for(dependency_graph, path, settings) for path in considered]

    changed_set = set(changed)
    direct: set[str] = set()
    transitive: set[str] = set()
    for path in considered:
        layers, _ = dependents_by_depth(
            dependency_graph.graph, path, settings.max_depth
        )
        if layers:
            direct |= layers[0]
        for layer in layers[1:]:
            transitive |= layer

    # A file inside the change is not "impacted by" the change; and a direct
    # dependent should not be double-counted as a transitive one.
    direct -= changed_set
    transitive -= changed_set | direct

    return BlastRadius(
        files=tuple(sorted(impacts, key=lambda i: (-i.total, i.path))),
        direct=_listed(direct, settings.max_listed),
        transitive=_listed(transitive, settings.max_listed),
        direct_count=len(direct),
        transitive_count=len(transitive),
        hubs=tuple(sorted(i.path for i in impacts if i.is_hub)),
        cycle_files=tuple(sorted(i.path for i in impacts if i.in_cycle)),
        depth=settings.max_depth,
        depth_capped=any(i.depth_capped for i in impacts),
        listed_limit=settings.max_listed,
        graph_files=dependency_graph.files_indexed,
        graph_edges=dependency_graph.edges,
        parsers=dependency_graph.parsers,
        analyzed=True,
        files_omitted=omitted,
    )
