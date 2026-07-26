"""Tests for the MCP server: the protocol handshake and the one tool."""

from __future__ import annotations

import io
import json
from pathlib import Path

from git import Repo

from mcp_server.server import (
    PROTOCOL_VERSION,
    TOOL_NAME,
    TOOL_SCHEMA,
    get_deployment_risk,
    handle_request,
    serve,
)
from sentinel.serialization import SCHEMA_VERSION


def request(method: str, params: dict | None = None, request_id: int | str = 1) -> dict:
    message: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def call_tool(arguments: dict, request_id: int = 7) -> dict:
    return handle_request(
        request("tools/call", {"name": TOOL_NAME, "arguments": arguments}, request_id)
    )


# --- the handshake --------------------------------------------------------


def test_initialize_advertises_tools() -> None:
    response = handle_request(request("initialize", {"protocolVersion": PROTOCOL_VERSION}))

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    result = response["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "sentinel"


def test_notifications_are_never_answered() -> None:
    """Replying to a notification is a protocol violation."""
    assert handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_exactly_one_tool_is_exposed() -> None:
    tools = handle_request(request("tools/list"))["result"]["tools"]

    assert len(tools) == 1
    assert tools[0]["name"] == TOOL_NAME
    assert tools[0]["inputSchema"]["required"] == ["repo_path"]
    assert set(tools[0]["inputSchema"]["properties"]) == {
        "repo_path",
        "scope",
        "since",
        "explain",
    }


def test_the_description_says_the_score_is_not_written_by_an_llm() -> None:
    """A calling model must not think it may adjust the number."""
    assert "never produced by a language model" in TOOL_SCHEMA["description"]


def test_an_unknown_method_is_a_jsonrpc_error() -> None:
    response = handle_request(request("resources/list"))
    assert response["error"]["code"] == -32601


def test_an_unknown_tool_is_rejected() -> None:
    response = handle_request(request("tools/call", {"name": "rm_rf", "arguments": {}}))
    assert response["error"]["code"] == -32602


def test_ping_is_answered() -> None:
    assert handle_request(request("ping"))["result"] == {}


# --- the tool ------------------------------------------------------------


def test_the_tool_returns_the_json_contract(graph_repo: Path) -> None:
    payload = get_deployment_risk({"repo_path": str(graph_repo), "scope": "all"})

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
    payload = get_deployment_risk({"repo_path": str(graph_repo), "scope": "all"})
    blast = payload["blast_radius"]

    assert blast["graph"]["parsers"] == ["python"]
    assert blast["graph"]["edges"] > 0
    per_file = {entry["path"]: entry for entry in blast["per_file"]}
    assert per_file["app/repo.py"]["direct_count"] == 2
    assert per_file["app/cycle_a.py"]["in_cycle"] is True


def test_a_tool_call_wraps_the_payload_both_ways(graph_repo: Path) -> None:
    result = call_tool({"repo_path": str(graph_repo), "scope": "all"})["result"]

    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    # Text and structured forms must agree.
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_scope_diff_sees_uncommitted_work(graph_repo: Path) -> None:
    (graph_repo / "app/service.py").write_text(
        "from app import repo\n\n\ndef run():\n    return 1\n", encoding="utf-8"
    )
    payload = get_deployment_risk({"repo_path": str(graph_repo), "scope": "diff"})

    assert payload["scope"]["mode"] == "diff"
    assert any(f["path"] == "app/service.py" for f in payload["files"])


def test_since_is_passed_through(tiny_repo: Repo) -> None:
    payload = get_deployment_risk(
        {"repo_path": tiny_repo.working_tree_dir, "scope": "scan", "since": "HEAD~2"}
    )
    assert payload["scope"]["base_ref"] == "HEAD~2"
    assert payload["scope"]["files_analyzed"] > 0


def test_the_narrative_is_skipped_without_a_key(graph_repo: Path) -> None:
    """`explain` is opt-in and optional; the analysis is unaffected."""
    payload = get_deployment_risk(
        {"repo_path": str(graph_repo), "scope": "all", "explain": True}
    )

    assert payload["ai_explanation"]["available"] is False
    assert "NVIDIA_API_KEY" in payload["ai_explanation"]["skipped_reason"]
    assert payload["score"] >= 0  # still a complete analysis


# --- failure handling ----------------------------------------------------


def test_a_missing_repo_path_is_a_tool_error_not_a_crash() -> None:
    result = call_tool({})["result"]
    assert result["isError"] is True
    assert "repo_path is required" in result["content"][0]["text"]


def test_a_path_that_is_not_a_repository_is_a_tool_error(tmp_path: Path) -> None:
    """The client should show the model what went wrong, not see a dead server."""
    result = call_tool({"repo_path": str(tmp_path / "nowhere")})["result"]

    assert result["isError"] is True
    assert "not a git repository" in result["content"][0]["text"]


def test_an_unknown_scope_is_rejected(graph_repo: Path) -> None:
    result = call_tool({"repo_path": str(graph_repo), "scope": "sideways"})["result"]
    assert result["isError"] is True
    assert "unknown scope" in result["content"][0]["text"]


# --- the stdio loop ------------------------------------------------------


def test_the_loop_answers_requests_and_skips_notifications(graph_repo: Path) -> None:
    lines = [
        json.dumps(request("initialize", {}, 1)),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps(request("tools/list", None, 2)),
        "",  # blank lines are tolerated
    ]
    stdout = io.StringIO()
    serve(io.StringIO("\n".join(lines) + "\n"), stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert [r["id"] for r in responses] == [1, 2]  # the notification got no reply


def test_unparseable_input_does_not_kill_the_server() -> None:
    stdout = io.StringIO()
    serve(io.StringIO("this is not json\n" + json.dumps(request("ping", None, 5)) + "\n"), stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert responses[0]["error"]["code"] == -32700
    assert responses[1]["id"] == 5  # and it carried on
