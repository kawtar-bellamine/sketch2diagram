"""Text to diagram: a process description becomes a flowchart.

Sentences are split, connectors removed, and each action clause is turned into a short label "Verb object" from its
dependency parse (spaCy), whatever the voice: "the order is checked" and "the planner checks the order" both give
"Check the order". Conditions ("if ..., otherwise ...") become decision nodes with yes / no branches; "... again"
loops back to the step with the same label.
"""
from __future__ import annotations

import re

import spacy

from .graph import Edge, Graph, Node

_nlp = None
CONNECTORS = r"^(first|then|next|after that|afterwards|to begin|subsequently|once this is done|following this|" \
             r"finally|after this|later)\s*,?\s*"
START = re.compile(r"(?:process starts|process begins|all begins|it begins|it starts|starts) when (.+)", re.I)
END = re.compile(r"(process (then )?ends|completes the process|end of the process|process is complete)", re.I)
DEC = re.compile(r"^(?:if|in case|when|should it be that|provided that)\s+(.+?),\s*(.+?)[;,]\s*"
                 r"(?:otherwise|if not|else|or else)\s*,?\s*(.+)$", re.I)
CONTINUE = re.compile(r"(process continues|flow goes on|carry on|nothing (else )?is done|continue)", re.I)


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md")
    return _nlp


def action_label(clause: str) -> str | None:
    doc = nlp()(clause.strip().rstrip("."))
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None
    verb = root if root.pos_ in ("VERB", "AUX") else next((t for t in doc if t.pos_ == "VERB"), root)
    obj = next((t for t in verb.children if t.dep_ in ("dobj", "nsubjpass", "obj")), None)
    if obj is None:
        obj = next((t for t in doc if t.dep_ in ("dobj", "nsubjpass")), None)
    lemma = verb.lemma_.lower()
    if lemma == "be":
        # "the parcel is packed" is sometimes parsed as be + adjective: take the participle as the verb
        part = next((t for t in root.children if t.dep_ in ("acomp", "attr", "advmod", "oprd")), None)
        if part is not None:
            lemma = _verb_lemma(part.text)
            obj = obj or next((t for t in root.children if t.dep_ == "nsubj"), None)
    if obj is None:
        return lemma.capitalize()
    span = doc[obj.left_edge.i: obj.right_edge.i + 1].text
    if not span.lower().startswith(("the ", "a ", "an ")):
        span = "the " + span
    return f"{lemma.capitalize()} {span.lower()}"


def _verb_lemma(word: str) -> str:
    """Lemma of a past participle, read in a sentence where it can only be a verb."""
    doc = nlp()(f"The item was {word} by the operator.")
    return doc[3].lemma_.lower()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def parse(text: str) -> Graph:
    g = Graph()
    prev, pending_yes, k, d = None, None, 0, 0
    open_no = []                                  # 'no' branch nodes waiting to join the next node

    def add(label, kind="process"):
        nonlocal k
        if kind == "start":
            nid = "S"
        elif kind == "end":
            nid = "E"
        else:
            nid = f"N{k}"; k += 1
        g.nodes.append(Node(nid, label, kind))
        return nid

    def link(nid):
        nonlocal prev, pending_yes
        if prev is not None:
            g.edges.append(Edge(prev, nid, "yes" if pending_yes == prev else ""))
        for o in open_no:
            g.edges.append(Edge(o, nid))
        open_no.clear()
        pending_yes = None
        prev = nid

    for s in sentences(text):
        m = START.search(s)
        if m and not g.nodes:
            prev = add(m.group(1).rstrip(". "), "start")
            continue
        if END.search(s):
            link(add("End", "end"))
            continue
        body = re.sub(CONNECTORS, "", s.rstrip("."), flags=re.I)
        m = DEC.match(body)
        if m:
            cond, yes, no = m.groups()
            d += 1
            did = f"D{d}"
            g.nodes.append(Node(did, cond.strip() + "?", "decision"))
            link(did)
            pending_yes = did
            if not CONTINUE.search(yes):
                lab = action_label(yes)
                if lab:
                    link(add(lab)); g.edges[-1].label = "yes"
            back = re.search(r"\s+and\s+(.+?)\s+again$", no)
            no_clause = no[:back.start()] if back else no
            lab = action_label(no_clause)
            nid = add(lab)
            g.edges.append(Edge(did, nid, "no"))
            if back:
                target_lab = action_label(back.group(1))
                target = next((n.id for n in g.nodes if n.label == target_lab and n.id != nid), None)
                if target:
                    g.edges.append(Edge(nid, target))
                else:
                    open_no.append(nid)
            else:
                open_no.append(nid)
            continue
        for c in re.split(r",?\s+(?:and then|then)\s+", body):
            lab = action_label(c)
            if lab:
                link(add(lab))
    return g
