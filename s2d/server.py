"""MCP server: the diagram tools, callable by any MCP client (an agent, an IDE, a chat application).

Run:  python -m s2d.server            (stdio transport)
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from mcp.server.mcpserver import MCPServer

from .graph import Graph, render_svg, validate
from .sketch import recognise
from .text import parse

mcp = MCPServer("sketch2diagram", instructions="Turn a process description or a photo of a hand-drawn flowchart "
                                               "into a Mermaid diagram, then validate and render it.")


def _result(g: Graph) -> dict:
    return dict(mermaid=g.to_mermaid(), graph=g.to_dict(), issues=validate(g))


@mcp.tool()
def text_to_diagram(description: str) -> dict:
    """Turn a process description (sentences with first / then / if ... otherwise ...) into a flowchart."""
    return _result(parse(description))


@mcp.tool()
def sketch_to_diagram(png_base64: str) -> dict:
    """Read a photo or scan of a hand-drawn flowchart (PNG or JPEG, base64) and return the flowchart."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(png_base64), np.uint8), cv2.IMREAD_GRAYSCALE)
    g, _ = recognise(img)
    return _result(g)


@mcp.tool()
def validate_mermaid(mermaid: str) -> dict:
    """Check that a Mermaid diagram renders; returns the error message of the Mermaid CLI if it does not."""
    ok, out = render_svg(mermaid)
    return dict(valid=ok, error=None if ok else out)


@mcp.tool()
def render_diagram(mermaid: str) -> dict:
    """Render a Mermaid diagram to SVG."""
    ok, out = render_svg(mermaid)
    return dict(svg=out if ok else None, error=None if ok else out)


if __name__ == "__main__":
    mcp.run()
