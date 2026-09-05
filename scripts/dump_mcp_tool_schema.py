"""Dump the MCP tool list (what the LLM sees) to a markdown report.

This is a one-off diagnostic: it imports the FastMCP server, introspects its
registered tools via mcp._tool_manager._tools (the standard internal API), and
prints the JSON schema for each parameter so we can verify the per-parameter
descriptions are visible to the LLM.

Usage:
    python scripts/dump_mcp_tool_schema.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_voice_studio.server import mcp  # noqa: E402


def main() -> None:
    tools = mcp._tool_manager._tools
    out_lines: list[str] = ["# MCP Tool Schema (what the LLM sees)\n"]

    for name in sorted(tools):
        tool = tools[name]
        out_lines.append(f"## `{name}`\n")
        desc = tool.description or "(no description)"
        out_lines.append(f"**Description:** {desc}\n")
        out_lines.append("\n**Parameters:**\n")
        params = tool.parameters or {}
        properties = params.get("properties", {})
        required = set(params.get("required", []))
        if not properties:
            out_lines.append("  (none)\n")
        else:
            for pname, spec in properties.items():
                ptype = spec.get("type", "any")
                pdesc = spec.get("description", "(no description)")
                is_req = "✅ required" if pname in required else "optional"
                default = spec.get("default", "")
                if default:
                    out_lines.append(f"- `{pname}` ({ptype}, {is_req}, default=`{default}`): {pdesc}\n")
                else:
                    out_lines.append(f"- `{pname}` ({ptype}, {is_req}): {pdesc}\n")
        out_lines.append("\n---\n")

    print("".join(out_lines))
    print(f"\n[OK] {len(tools)} tools dumped.")


if __name__ == "__main__":
    main()
