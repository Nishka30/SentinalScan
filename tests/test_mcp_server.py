"""Tests for the FastMCP server.

Strategy
--------
* **Protocol-level tests** (tool listing, schema shape, server name) use the
  FastMCP in-process async client so the full MCP handshake is exercised
  without spawning a subprocess.

* **Tool-logic tests** (return values, error handling) call ``get_deployment_risk``
  directly.  The function is still a plain Python callable; testing it directly
  is faster and gives clearer failure messages than going through the async
  transport for every assertion.

The ``asyncio_mode = "auto"`` setting in pyproject.toml means async test
functions are collected and run automatically — no ``@pytest.mark.asyncio``
needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client
from git import Repo

from mcp_server.server import get_deployment_risk, mcp
from sentinel.git_reader import RepositoryError
from sentinel.serialization import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Shared async fixture — one in-process client for the whole session
# ---------------------------------------------------------------------------


@pytest.fixture
async def mcp_client():
    """Yield a connected FastMCP in-process client."""
    async with Client(mcp) as client:
        yield client


# ---------------------------------------------------------------------------
# Protocol-level tests
# ---------------------------------------------------------------------------


async def test_exactly_one_tool_is_exposed(mcp_client) -> None:
    tools = await mcp_client.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "get_deployment_risk"


async def test_tool_schema_has_required_repo_path(mcp_client) -> None:
    tools = await mcp_client.list_tools()
    schema = tools[0].inputSchema

    assert "repo_path" in schema.get("required", [])
    assert set(schema["properties"]) >= {"repo_path", "scope", "since", "explain"}


async def test_tool_description_says_not_llm(mcp_client) -> None:
    """A calling model must not think it may adjust the score."""
    tools = await mcp_client.list_tools()

    assert "never produced by a language model" in tools[0].description


async def test_tool_call_returns_content(mcp_client, graph_repo: Path) -> None:
    """End-to-end: call via the MCP client and get parseable JSON back."""
    result = await mcp_client.call_tool(
        "get_deployment_risk",
        {"repo_path": str(graph_repo), "scope": "all"},
    )

    # FastMCP 3.x returns a CallToolResult; content is in .content list.
    assert result.content, "expected at least one content item"
    payload = json.loads(result.content[0].text)
    assert 0 <= payload["score"] <= 100
    assert payload["band"] in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Tool-logic tests (direct function calls — faster, clearer failures)
# ---------------------------------------------------------------------------


def test_the_tool_returns_the_json_contract(graph_repo: Path) -> None:
    payload = get_deployment_risk(repo_path=str(graph_repo), scope="all")

    assert payload["schema_version"] == SCHEMA_VERSION
    assert 0 <= payload["score"] <= 100
    assert payload["band"] in {"low", "medium", "high"}
    assert payload["recommendation"]
    assert payload["scoring_method"] in {"rules", "model"}
    assert payload["scope"]["mode"] == "all"
    assert isinstance(payload["files"], list)
    assert payload["blast_radius"]["analyzed"] is True
    assert payload["ai_explanation"] is None  # not requested
    assert payload["warnings"] == []


def test_the_tool_reports_blast_radius_for_a_real_graph(graph_repo: Path) -> None:
    """The same dependency data the CLI shows, in structured form."""
    payload = get_deployment_risk(repo_path=str(graph_repo), scope="all")
    blast = payload["blast_radius"]

    assert blast["graph"]["parsers"] == ["python"]
    assert blast["graph"]["edges"] > 0
    per_file = {entry["path"]: entry for entry in blast["per_file"]}
    assert per_file["app/repo.py"]["direct_count"] == 2
    assert per_file["app/cycle_a.py"]["in_cycle"] is True


def test_scope_diff_sees_uncommitted_work(graph_repo: Path) -> None:
    (graph_repo / "app/service.py").write_text(
        "from app import repo\n\n\ndef run():\n    return 1\n", encoding="utf-8"
    )
    payload = get_deployment_risk(repo_path=str(graph_repo), scope="diff")

    assert payload["scope"]["mode"] == "diff"
    assert any(f["path"] == "app/service.py" for f in payload["files"])


def test_since_is_passed_through(tiny_repo: Repo) -> None:
    payload = get_deployment_risk(
        repo_path=tiny_repo.working_tree_dir, scope="scan", since="HEAD~2"
    )
    assert payload["scope"]["base_ref"] == "HEAD~2"
    assert payload["scope"]["files_analyzed"] > 0


def test_the_narrative_is_skipped_without_a_key(graph_repo: Path) -> None:
    """`explain` is opt-in; the analysis is unaffected when no key is set."""
    payload = get_deployment_risk(
        repo_path=str(graph_repo), scope="all", explain=True
    )

    assert payload["ai_explanation"]["available"] is False
    assert "NVIDIA_API_KEY" in payload["ai_explanation"]["skipped_reason"]
    assert payload["score"] >= 0


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


def test_a_missing_repo_path_raises_value_error() -> None:
    """FastMCP converts the ValueError into a tool error on the wire."""
    with pytest.raises(ValueError, match="repo_path is required"):
        get_deployment_risk(repo_path="")


def test_a_path_that_is_not_a_repository_raises_repository_error(
    tmp_path: Path,
) -> None:
    """RepositoryError surfaces so the client can show the model what went wrong."""
    with pytest.raises(RepositoryError, match="not a git repository"):
        get_deployment_risk(repo_path=str(tmp_path / "nowhere"))


def test_an_unknown_scope_raises_value_error(graph_repo: Path) -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        get_deployment_risk(repo_path=str(graph_repo), scope="sideways")
