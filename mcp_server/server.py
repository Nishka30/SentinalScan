"""One MCP server, one tool: `get_deployment_risk`.

Speaks the Model Context Protocol over stdio — newline-delimited JSON-RPC 2.0 on
stdin and stdout — so Cursor, Claude Desktop and Claude Code can register it.

The protocol subset is written out by hand rather than pulled from the `mcp`
SDK, because that SDK is not in this project's locked dependency list. It is a
small surface: `initialize`, `tools/list`, `tools/call`, and ignoring
notifications. If you would rather depend on the official SDK, this module is
the only thing that changes.

Two rules it follows:

* **It calls the library, never the CLI.** `analysis.run_analysis` is imported
  directly, so there is no subprocess, no argument quoting, and no parsing of
  output that was formatted for a human.
* **Nothing is written to stdout except protocol messages.** On a stdio
  transport, one stray `print` corrupts the stream. Logging goes to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

from sentinel import __version__
from sentinel.analysis import add_explanation, run_analysis
from sentinel.config import get_settings
from sentinel.git_reader import RepositoryError
from sentinel.serialization import to_dict

logger = logging.getLogger(__name__)

#: The version of the MCP spec this server implements.
PROTOCOL_VERSION = "2024-11-05"

SERVER_INFO = {"name": "sentinel", "version": __version__}

TOOL_NAME = "get_deployment_risk"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Assess whether a change in a git repository is safe to deploy. Returns a "
        "risk score out of 100, a band (low/medium/high), the specific reasons "
        "behind the score with their evidence, and the blast radius (which other "
        "files depend on the changed ones). The score is computed from the "
        "repository's own bug history — either by a transparent rule engine or, "
        "if `sentinel train` has been run there, by a model trained on that "
        "history. It is never produced by a language model."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string",
                "description": "Absolute path to the git repository to analyze.",
            },
            "scope": {
                "type": "string",
                "enum": ["scan", "diff", "all"],
                "default": "scan",
                "description": (
                    "'scan' scores committed changes against the base ref, "
                    "'diff' scores uncommitted work in the working tree, "
                    "'all' ranks every tracked file by inherent risk."
                ),
            },
            "since": {
                "type": "string",
                "description": (
                    "For scope 'scan': the ref to compare against, e.g. 'main' or "
                    "'HEAD~20'. Defaults to the repository's default branch."
                ),
            },
            "explain": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Also request a plain-English narrative from the configured "
                    "LLM. Requires NVIDIA_API_KEY; the score is identical either "
                    "way."
                ),
            },
        },
        "required": ["repo_path"],
    },
}


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------


def get_deployment_risk(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the analysis and return the same payload the CLI's `--json` emits."""
    raw_path = arguments.get("repo_path")
    if not raw_path:
        raise ValueError("repo_path is required")

    scope = arguments.get("scope") or "scan"
    if scope not in ("scan", "diff", "all"):
        raise ValueError(f"unknown scope {scope!r}; expected scan, diff or all")

    result = run_analysis(
        Path(str(raw_path)).expanduser(),
        mode=scope,
        since=arguments.get("since"),
    )

    # The narrative is opt-in and optional: with no key it is skipped and the
    # deterministic analysis is returned unchanged.
    if arguments.get("explain"):
        result = add_explanation(result, get_settings())

    return to_dict(result)


# --------------------------------------------------------------------------
# JSON-RPC
# --------------------------------------------------------------------------


def _result(request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message. Returns None for notifications.

    A notification has no `id` and must never be answered — replying to one is a
    protocol violation that some clients treat as fatal.
    """
    method = request.get("method")
    request_id = request.get("id")

    if request_id is None:
        logger.debug("ignoring notification %s", method)
        return None

    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )

    if method == "tools/list":
        return _result(request_id, {"tools": [TOOL_SCHEMA]})

    if method == "tools/call":
        return _handle_tool_call(request_id, request.get("params") or {})

    if method == "ping":
        return _result(request_id, {})

    return _error(request_id, -32601, f"method not found: {method}")


def _handle_tool_call(request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call, reporting failures as tool errors.

    A repository that cannot be analyzed is a *tool* error (`isError: true`), not
    a transport error: the client should show the model what went wrong so it can
    try a different path, rather than treating the server as broken.
    """
    name = params.get("name")
    if name != TOOL_NAME:
        return _error(request_id, -32602, f"unknown tool: {name}")

    try:
        payload = get_deployment_risk(params.get("arguments") or {})
    except (RepositoryError, ValueError) as exc:
        return _result(
            request_id,
            {
                "content": [{"type": "text", "text": f"Sentinel could not analyze that: {exc}"}],
                "isError": True,
            },
        )
    except Exception as exc:  # a crash must not take the server down
        logger.exception("unexpected failure in %s", TOOL_NAME)
        return _result(
            request_id,
            {
                "content": [
                    {"type": "text", "text": f"Sentinel failed: {type(exc).__name__}: {exc}"}
                ],
                "isError": True,
            },
        )

    # Both forms on purpose: `structuredContent` for clients that read typed
    # results, and the JSON as text for those that only read content blocks.
    return _result(
        request_id,
        {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
            "structuredContent": payload,
            "isError": False,
        },
    )


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Read newline-delimited JSON-RPC from stdin until the stream closes."""
    source = stdin or sys.stdin
    sink = stdout or sys.stdout

    for line in source:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("discarding unparseable message: %s", exc)
            _write(sink, _error(None, -32700, "parse error"))
            continue

        response = handle_request(request)
        if response is not None:
            _write(sink, response)


def _write(sink: TextIO, message: dict[str, Any]) -> None:
    sink.write(json.dumps(message) + "\n")
    sink.flush()


def main() -> int:
    # stderr, never stdout: stdout is the protocol channel.
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    serve()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
