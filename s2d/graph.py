"""Diagram model, Mermaid export, structural validation and comparison with a reference."""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz


@dataclass
class Node:
    id: str
    label: str
    kind: str = "process"          # start | end | process | decision


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def node(self, i):
        return next(n for n in self.nodes if n.id == i)

    def to_mermaid(self) -> str:
        shape = {"start": ("([", "])"), "end": ("([", "])"), "process": ("[", "]"), "decision": ("{", "}")}
        lines = ["flowchart TD"]
        for n in self.nodes:
            a, b = shape[n.kind]
            lines.append(f'    {n.id}{a}"{_esc(n.label)}"{b}')
        for e in self.edges:
            lines.append(f"    {e.src} -->|{_esc(e.label)}| {e.dst}" if e.label else f"    {e.src} --> {e.dst}")
        return "\n".join(lines)

    def to_dict(self):
        return dict(nodes=[n.__dict__ for n in self.nodes], edges=[e.__dict__ for e in self.edges])


def _esc(s: str) -> str:
    return s.replace('"', "'").replace("|", "/")


def validate(g: Graph) -> list[str]:
    """Structural checks a reviewer would make before accepting a flowchart."""
    issues = []
    ids = {n.id for n in g.nodes}
    for e in g.edges:
        if e.src not in ids or e.dst not in ids:
            issues.append(f"edge {e.src}->{e.dst} points to a missing node")
    starts = [n for n in g.nodes if n.kind == "start"]
    if len(starts) != 1:
        issues.append(f"{len(starts)} start nodes (expected 1)")
    out = {n.id: [e for e in g.edges if e.src == n.id] for n in g.nodes}
    for n in g.nodes:
        if n.kind == "decision" and len(out[n.id]) < 2:
            issues.append(f"decision '{n.label}' has {len(out[n.id])} outgoing branch(es)")
        if n.kind != "end" and not out[n.id]:
            issues.append(f"'{n.label}' is a dead end")
    if starts:
        seen, stack = set(), [starts[0].id]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k); stack += [e.dst for e in out.get(k, [])]
        for n in g.nodes:
            if n.id not in seen:
                issues.append(f"'{n.label}' cannot be reached from the start")
    return issues


def render_svg(mermaid: str) -> tuple[bool, str]:
    """Render with the Mermaid CLI; a syntax error is reported instead of an image."""
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "g.mmd", Path(d) / "g.svg"
        src.write_text(mermaid)
        r = subprocess.run(["mmdc", "-i", str(src), "-o", str(out), "-q", "-p", str(Path(__file__).with_name("puppeteer.json"))], capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not out.exists():
            return False, (r.stderr or r.stdout)[-500:]
        return True, out.read_text()


def compare(pred: Graph, ref: Graph, threshold: int = 75) -> dict:
    """Match nodes by label similarity (one-to-one, best pairs first), then edges through the node matching."""
    pairs = sorted(((fuzz.token_set_ratio(_norm(a.label), _norm(b.label)), a.id, b.id)
                    for a in pred.nodes for b in ref.nodes), reverse=True)
    m, used = {}, set()
    for s, a, b in pairs:
        if s >= threshold and a not in m and b not in used:
            m[a] = b; used.add(b)
    tp_n = len(m)
    pe = {(m.get(e.src), m.get(e.dst)) for e in pred.edges}
    re_ = {(e.src, e.dst) for e in ref.edges}
    tp_e = len(pe & re_)
    kinds_ok = sum(pred.node(a).kind == ref.node(b).kind for a, b in m.items())
    f = lambda tp, p, r: 2 * tp / max(p + r, 1)  # noqa: E731
    return dict(node_f1=f(tp_n, len(pred.nodes), len(ref.nodes)), edge_f1=f(tp_e, len(pred.edges), len(ref.edges)),
                kind_acc=kinds_ok / max(tp_n, 1),
                exact=tp_n == len(ref.nodes) == len(pred.nodes) and tp_e == len(re_) == len(pe))


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())
