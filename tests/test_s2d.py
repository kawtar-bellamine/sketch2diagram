import asyncio
import random

from s2d.gen import describe, random_graph
from s2d.graph import Edge, Graph, Node, compare, render_svg, validate
from s2d.sketch import recognise, render
from s2d.text import action_label, parse


def test_labels_whatever_the_voice():
    assert action_label("the parcel is packed") == "Pack the parcel"
    assert action_label("the technician welds the frame") == "Weld the frame"
    assert action_label("the parts are picked") == "Pick the parts"


def test_decision_with_loop_back():
    g = parse("The process starts when a part arrives. First, the part is inspected. If the part is within "
              "tolerance, the process continues; otherwise, the part is reworked and the part is inspected again. "
              "The process then ends.")
    lab = {n.id: n.label for n in g.nodes}
    edges = {(lab[e.src], lab[e.dst], e.label) for e in g.edges}
    assert ("Rework the part", "Inspect the part", "") in edges
    assert ("the part is within tolerance?", "End", "yes") in edges
    assert not validate(g)


def test_validate_finds_problems():
    g = Graph([Node("S", "s", "start"), Node("D", "d?", "decision"), Node("X", "x")], [Edge("S", "D")])
    issues = " ".join(validate(g))
    assert "1 outgoing" in issues or "0 outgoing" in issues
    assert "cannot be reached" in issues


def test_generated_text_round_trip():
    rng = random.Random(0)
    g, plan = random_graph(rng)
    assert compare(parse(describe(plan, g.nodes[0].label, rng, "B")), g)["node_f1"] == 1.0


def test_sketch_round_trip_structure():
    rng = random.Random(1)
    g, _ = random_graph(rng)
    img, _ = render(g, seed=1)
    p, _ = recognise(img)
    c = compare(p, g)
    assert c["node_f1"] > 0.9 and c["edge_f1"] > 0.7


def test_mermaid_renders():
    ok, svg = render_svg(Graph([Node("S", "a", "start"), Node("E", "End", "end")], [Edge("S", "E")]).to_mermaid())
    assert ok and "<svg" in svg


def test_mcp_tools():
    from s2d.server import mcp
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert {"text_to_diagram", "sketch_to_diagram", "validate_mermaid", "render_diagram"} <= set(names)
