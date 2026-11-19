"""Evaluation of the two paths on graphs never seen while writing the code."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .gen import describe, random_graph
from .graph import compare, render_svg, validate
from .handwritten import CASES
from .sketch import degrade, recognise, render
from .text import parse

ROOT = Path(__file__).resolve().parents[1]


def mean(rows):
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}


def geometry(pred_boxes, true_boxes, pred, ref):
    """Structure only (no OCR): nodes matched by box overlap, then edges through that matching."""
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u else 0
    m = {}
    for pid, pb in pred_boxes.items():
        best = max(true_boxes, key=lambda t: iou(pb, true_boxes[t]))
        if iou(pb, true_boxes[best]) > 0.5:
            m[pid] = best
    pe = {(m.get(e.src), m.get(e.dst)) for e in pred.edges}
    te = {(e.src, e.dst) for e in ref.edges}
    kinds = sum(pred.node(p).kind == ref.node(t).kind for p, t in m.items())
    return dict(node_f1=2 * len(m) / (len(pred.nodes) + len(ref.nodes)),
                edge_f1=2 * len(pe & te) / max(len(pe) + len(te), 1), kind_acc=kinds / max(len(m), 1),
                direction_ok=len(pe & te) / max(len(pe & te) + len({(b, a) for a, b in pe} & te), 1))


def main(n=120):
    res = {}
    for style in ("A", "B"):
        rng = random.Random(100 + ord(style)); rows = []
        for _ in range(n):
            g, plan = random_graph(rng)
            rows.append(compare(parse(describe(plan, g.nodes[0].label, rng, style)), g))
        res[f"text_set_{style}"] = mean(rows)
    rows, fails = [], []
    for text, ref in CASES:
        p = parse(text); c = compare(p, ref); rows.append(c)
        if not c["exact"]:
            fails.append(dict(text=text, got=p.to_mermaid()))
    res["text_set_C_handwritten"] = dict(mean(rows), n=len(CASES))
    res["text_set_C_failures"] = fails
    for variant in ("clean", "photo-like"):
        rng = random.Random(2024); rows, geo, times = [], [], []   # not the seeds used while tuning
        for k in range(n):
            g, _ = random_graph(rng)
            img, boxes = render(g, seed=1000 + k)
            if variant != "clean":
                img = degrade(img, seed=1000 + k)
            t = time.perf_counter(); p, pb = recognise(img); times.append(time.perf_counter() - t)
            rows.append(compare(p, g)); geo.append(geometry(pb, boxes, p, g))
            if k < 3 and variant == "clean":
                import cv2
                cv2.imwrite(str(ROOT / f"results/figures/sketch_{k}.png"), img)
                (ROOT / f"results/sketch_{k}.mmd").write_text(p.to_mermaid())
        res[f"sketch_{variant}"] = dict(with_ocr=mean(rows), structure_only=mean(geo),
                                       seconds_per_sketch=round(float(np.mean(times)), 2))
    # every output must render with the Mermaid CLI and pass the structural checks
    rng = random.Random(5); ok = valid = 0
    for k in range(20):
        g, plan = random_graph(rng)
        p = parse(describe(plan, g.nodes[0].label, rng, "B"))
        ok += render_svg(p.to_mermaid())[0]; valid += not validate(p)
    res["mermaid_render_ok"] = f"{ok}/20"; res["structurally_valid"] = f"{valid}/20"
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "text_set_C_failures"}, indent=1))
    for f in fails:
        print(f["text"][:150]); print(f["got"]); print()


if __name__ == "__main__":
    main()
