"""Random process graphs and their descriptions, used as ground truth for the evaluation.

Two phrasing sets: set A was used while writing the text parser; set B (different connectors, sentence orders and
voices) was written afterwards and never used to tune it, so it measures how the parser copes with new phrasing.
"""
from __future__ import annotations

import random

from .graph import Edge, Graph, Node

STEPS = [("check", "checked", "checks", "the order"), ("reserve", "reserved", "reserves", "the stock"),
         ("pick", "picked", "picks", "the parts"), ("pack", "packed", "packs", "the parcel"),
         ("ship", "shipped", "ships", "the parcel"), ("send", "sent", "sends", "the invoice"),
         ("contact", "contacted", "contacts", "the customer"), ("inspect", "inspected", "inspects", "the part"),
         ("measure", "measured", "measures", "the dimensions"), ("rework", "reworked", "reworks", "the part"),
         ("record", "recorded", "records", "the result"), ("calibrate", "calibrated", "calibrates", "the machine"),
         ("load", "loaded", "loads", "the program"), ("clean", "cleaned", "cleans", "the tool"),
         ("replace", "replaced", "replaces", "the filter"), ("test", "tested", "tests", "the pump"),
         ("approve", "approved", "approves", "the request"), ("update", "updated", "updates", "the stock level"),
         ("print", "printed", "prints", "the label"), ("archive", "archived", "archives", "the report"),
         ("assemble", "assembled", "assembles", "the gearbox"), ("weld", "welded", "welds", "the frame"),
         ("paint", "painted", "paints", "the housing"), ("order", "ordered", "orders", "the spare part")]
CONDS = ["the order is complete", "the part is within tolerance", "the test passes", "the stock is available",
         "the pressure is correct", "the request is urgent", "the weld is accepted"]
STARTS = ["a customer order is received", "a part arrives at the inspection station",
          "a maintenance request is created", "a new batch is released", "the machine is switched on"]
ACTORS = ["the operator", "the technician", "the planner", "the team"]


def label(step):
    return f"{step[0].capitalize()} {step[3]}"


def random_graph(rng: random.Random) -> tuple[Graph, list]:
    """Linear process with 0 to 2 decisions; the 'no' branch either joins the flow or loops back."""
    n = rng.randint(3, 6)
    steps = rng.sample(STEPS, n + 2)
    g = Graph([Node("S", rng.choice(STARTS), "start")])
    plan = []                                   # recipe for the text generator
    prev = "S"
    k = 0
    n_dec = rng.choice([0, 1, 1, 2])
    dec_pos = sorted(rng.sample(range(1, n), min(n_dec, n - 1)))
    alt_pool = steps[n:]
    for i in range(n):
        nid = f"N{i}"
        g.nodes.append(Node(nid, label(steps[i]), "process"))
        g.edges.append(Edge(prev, nid, "yes" if prev.startswith("D") else ""))
        plan.append(("step", steps[i], nid))
        prev = nid
        if i in dec_pos and alt_pool:
            k += 1
            cond = rng.choice([c for c in CONDS if c not in [p[1] for p in plan if p[0] == "dec"]])
            did, aid = f"D{k}", f"A{k}"
            alt = alt_pool.pop()
            loop = rng.random() < 0.5
            g.nodes += [Node(did, cond + "?", "decision"), Node(aid, label(alt), "process")]
            g.edges += [Edge(prev, did), Edge(did, aid, "no")]
            back = prev if loop else None
            if loop:
                g.edges.append(Edge(aid, prev))
            plan.append(("dec", cond, did, alt, aid, back, steps[i]))
            prev = did
    g.nodes.append(Node("E", "End", "end"))
    g.edges.append(Edge(prev, "E", "yes" if prev.startswith("D") else ""))
    # 'no' branches that do not loop join the next node
    for p in plan:
        if p[0] == "dec" and p[5] is None:
            nxt = next((e.dst for e in g.edges if e.src == p[2] and e.label == "yes"), "E")
            g.edges.append(Edge(p[4], nxt))
    return g, plan


PLURAL = {"the parts", "the dimensions"}


def clause(step, rng, voice=None):
    v = voice or rng.choice(["passive", "active"])
    be = "are" if step[3] in PLURAL else "is"
    return f"{step[3]} {be} {step[1]}" if v == "passive" else f"{rng.choice(ACTORS)} {step[2]} {step[3]}"


def describe(plan, start_label, rng: random.Random, style: str = "A") -> str:
    conn_a = ["First", "Then", "Next", "After that"]
    conn_b = ["To begin", "Subsequently", "Once this is done", "Following this", "Afterwards"]
    out = [f"The process starts when {start_label}." if style == "A" else f"It all begins when {start_label}."]
    first = True
    for p in plan:
        if p[0] == "step":
            c = (conn_a if style == "A" else conn_b)
            w = c[0] if first else rng.choice(c[1:])
            out.append(f"{w}, {clause(p[1], rng)}.")
            first = False
        else:
            _, cond, did, alt, aid, back, prev_step = p
            again = f" and {clause(prev_step, rng, 'passive')} again" if back else ""
            if style == "A":
                out.append(f"If {cond}, the process continues; otherwise, {clause(alt, rng)}{again}.")
            else:
                out.append(f"In case {cond}, the flow goes on; if not, {clause(alt, rng)}{again}.")
    out.append("The process then ends." if style == "A" else "That completes the process.")
    return " ".join(out).replace(" the the", " the")
