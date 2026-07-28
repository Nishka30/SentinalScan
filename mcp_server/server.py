"""MCP server — powered by FastMCP.

FastMCP handles the JSON-RPC 2.0 protocol, tool schema generation from type
annotations, and the stdio/HTTP transport.  This file is a thin wrapper:
one registered tool that delegates all real work to the sentinel library.

Nothing in sentinel/ was changed.  The analysis logic lives in
sentinel/analysis.py; this file only wires it to the MCP protocol.

Install and start:
    pip install sentinel-risk
    sentinel-mcp

For remote HTTP instead of stdio, replace ``mcp.run()`` in main() with:
    mcp.run(transport="http", host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastmcp import FastMCP

from sentinel import __version__
from sentinel.analysis import add_explanation, run_analysis
from sentinel.config import get_settings
from sentinel.git_reader import RepositoryError  # noqa: F401 — re-exported for tests
from sentinel.serialization import to_dict

logger = logging.getLogger(__name__)

# FastMCP uses the name for serverInfo.name in the initialize handshake.
mcp = FastMCP("sentinel", version=__version__)


@mcp.tool
def get_deployment_risk(
    repo_path: str,
    scope: str = "scan",
    since: str | None = None,
    explain: bool = False,
) -> dict:
    """Assess whether a change in a git repository is safe to deploy.

    Returns a risk score out of 100, a band (low / medium / high), the
    specific reasons behind the score with their evidence, and the blast
    radius (which other files depend on the changed ones).  The score is
    computed from the repository's own bug history — either by a transparent
    rule engine or, if ``sentinel train`` has been run there, by a model
    trained on that history.  It is never produced by a language model.

    Args:
        repo_path: Absolute path to the git repository to analyze.
        scope: 'scan' scores committed changes against the base ref,
               'diff' scores uncommitted work in the working tree,
               'all' ranks every tracked file by inherent risk.
        since: For scope 'scan': the ref to compare against, e.g. 'main' or
               'HEAD~20'.  Defaults to the repository's default branch.
        explain: Also request a plain-English narrative from the configured
                 LLM.  Requires NVIDIA_API_KEY; the score is identical either
                 way.
    """
    if not repo_path:
        raise ValueError("repo_path is required")
    if scope not in ("scan", "diff", "all"):
        raise ValueError(f"unknown scope {scope!r}; expected scan, diff or all")

    result = run_analysis(
        Path(repo_path).expanduser(),
        mode=scope,
        since=since,
    )

    if explain:
        result = add_explanation(result, get_settings())

    return to_dict(result)


def main() -> None:
    """Start the MCP server on stdio — the transport Claude Desktop / Cursor use."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
