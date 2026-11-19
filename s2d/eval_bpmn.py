"""Evaluate the sketch recogniser on real hand-drawn BPMN (hdBPMN) and the text parser on the real exercise texts,
against the human models."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
from rapidfuzz import fuzz

from .bpmn_sketch import DEFAULT, load, recognise
from .hdbpmn import ROOT, diagrams, exercise_texts


def match(gt: dict, pred: dict):
    """Greedy one-to-one matching: a predicted shape matches a ground-truth node if its centre falls inside the
    ground-truth box (enlarged by 15%); closest centres first."""
    g = list(gt["nodes"].items())
    pairs = []
    for pi, pn in enumerate(pred["nodes"]):
        cx, cy = (pn["box"][0] + pn["box"][2]) / 2, (pn["box"][1] + pn["box"][3]) / 2
        for gi, (gid, gn) in enumerate(g):
            x0, y0, x1, y1 = gn["box"]; mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
            if x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my:
                d = np.hypot(cx - (x0 + x1) / 2, cy - (y0 + y1) / 2)
                pairs.append((d, pi, gid))
    used_p, used_g, m = set(), set(), {}
    for d, pi, gid in sorted(pairs):
        if pi not in used_p and gid not in used_g:
            used_p.add(pi); used_g.add(gid); m[pi] = gid
    return m


def score(gt: dict, pred: dict) -> dict:
    m = match(gt, pred)
    gn = gt["nodes"]
    r = dict(gt_nodes=len(gn), pred_nodes=len(pred["nodes"]), matched=len(m),
             typed=sum(pred["nodes"][pi]["kind"] == gn[gid]["kind"] for pi, gid in m.items()))
    for k in ("task", "event", "gateway"):
        r[f"gt_{k}"] = sum(n["kind"] == k for n in gn.values())
        r[f"found_{k}"] = sum(gn[gid]["kind"] == k and pred["nodes"][pi]["kind"] == k for pi, gid in m.items())
        r[f"pred_{k}"] = sum(n["kind"] == k for n in pred["nodes"])
    ge = set(gt["edges"])
    pe = {(m[a], m[b]) for a, b in pred["edges"] if a in m and b in m}
    r.update(gt_edges=len(ge), pred_edges=len(pred["edges"]), correct_edges=len(ge & pe),
             reversed_edges=len({(b, a) for a, b in pe} & ge - pe))
    sims = [fuzz.token_set_ratio(pred["nodes"][pi].get("label", "").lower(), gn[gid]["label"].lower())
            for pi, gid in m.items() if gn[gid]["kind"] == "task" and pred["nodes"][pi]["kind"] == "task"]
    r["label_sims"] = sims
    return r


def agg(rows):
    s = lambda k: sum(r[k] for r in rows)  # noqa: E731
    P = s("matched") / max(s("pred_nodes"), 1); R = s("matched") / max(s("gt_nodes"), 1)
    eP = s("correct_edges") / max(s("pred_edges"), 1); eR = s("correct_edges") / max(s("gt_edges"), 1)
    sims = [x for r in rows for x in r["label_sims"]]
    out = dict(diagrams=len(rows), node_precision=round(P, 3), node_recall=round(R, 3),
               node_f1=round(2 * P * R / max(P + R, 1e-9), 3), type_accuracy=round(s("typed") / max(s("matched"), 1), 3),
               edge_precision=round(eP, 3), edge_recall=round(eR, 3), edge_f1=round(2 * eP * eR / max(eP + eR, 1e-9), 3),
               reversed_edges=s("reversed_edges"),
               diagrams_all_nodes_and_edges=sum(r["matched"] == r["gt_nodes"] == r["pred_nodes"] and
                                                r["correct_edges"] == r["gt_edges"] == r["pred_edges"] for r in rows))
    for k in ("task", "event", "gateway"):
        out[f"recall_{k}"] = round(s(f"found_{k}") / max(s(f"gt_{k}"), 1), 3)
    if sims:
        out["task_label_similarity_median"] = float(np.median(sims))
        out["task_labels_readable_share"] = round(float(np.mean(np.array(sims) >= 70)), 3)
    return out


def run(split, params=None, ocr=False):
    rows = []
    for d in diagrams(split):
        pred = recognise(load(d["image"]), params, ocr=ocr)
        rows.append(dict(name=d["name"], **score(d, pred)))
    return rows


# ---------- text to diagram against the human models ----------
def task_f1(a: list[str], b: list[str], thr: int = 70) -> float:
    """Tasks matched one to one by label similarity (token set ratio >= thr), greedy on the best pairs."""
    pairs = sorted(((fuzz.token_set_ratio(x.lower(), y.lower()), i, j) for i, x in enumerate(a)
                    for j, y in enumerate(b)), reverse=True)
    ui, uj, m = set(), set(), 0
    for s, i, j in pairs:
        if s >= thr and i not in ui and j not in uj:
            ui.add(i); uj.add(j); m += 1
    p = m / max(len(a), 1); r = m / max(len(b), 1)
    return 2 * p * r / max(p + r, 1e-9)


def text_eval():
    from .text import parse
    texts = exercise_texts()
    out = {}
    for ex in ("ex00", "ex01"):
        humans = [d for d in diagrams() if d["exercise"] == ex]
        H = [[n["label"] for n in d["nodes"].values() if n["kind"] == "task"] for d in humans]
        hg = [sum(n["kind"] == "gateway" for n in d["nodes"].values()) for d in humans]
        g = parse(texts[ex])
        ours = [n.label for n in g.nodes if n.kind == "process"]
        f = [task_f1(ours, h) for h in H]
        hh = [task_f1(H[i], H[j]) for i, j in itertools.combinations(range(len(H)), 2)]
        out[ex] = dict(text=texts[ex], human_models=len(H), our_tasks=ours,
                       our_decisions=sum(n.kind == "decision" for n in g.nodes),
                       human_tasks_median=float(np.median([len(h) for h in H])),
                       human_gateways_median=float(np.median(hg)),
                       task_f1_vs_humans_mean=round(float(np.mean(f)), 3),
                       task_f1_human_vs_human_mean=round(float(np.mean(hh)), 3),
                       mermaid=g.to_mermaid())
    return out


def main():
    t0 = time.time()
    res = dict(params=DEFAULT)
    try:
        res["validation_grid"] = json.loads((ROOT / "results/val_grid.json").read_text())
    except FileNotFoundError:
        pass
    res["validation"] = agg(run("val"))
    test = run("test", ocr=True); train = run("train", ocr=True)
    res["test"] = agg(test); res["train_writers_unseen"] = agg(train); res["all_unseen"] = agg(test + train)
    res["text_to_diagram"] = text_eval()
    res["seconds"] = round(time.time() - t0)
    (ROOT / "results/metrics_hdbpmn.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (ROOT / "results/per_diagram_hdbpmn.json").write_text(json.dumps(test + train, indent=0, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "validation_grid"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
