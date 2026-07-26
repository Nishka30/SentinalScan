"""An MCP server exposing Sentinel's analysis as a single tool."""

from mcp_server.server import TOOL_NAME, handle_request, serve

__all__ = ["TOOL_NAME", "handle_request", "serve"]
