"""Sketch to diagram.

render(): draws a graph as a hand-drawn sketch (matplotlib xkcd style: wobbly lines, uneven shapes), with the true
pixel box of every node, to build an evaluation set.
recognise(): reads a sketch with classical computer vision, no training data needed:
  1. nodes are the closed regions of the drawing (holes of the ink contours); their fill ratio gives the shape:
     rectangle = process, ellipse = start / end, diamond = decision;
  2. the text inside each node is read by OCR (Tesseract);
  3. nodes are erased, what remains are arrows: each connected stroke that touches two nodes is an edge, oriented
     towards the end that carries more ink (the arrowhead);
  4. the words left outside nodes ("yes", "no") are attached to the nearest arrow.
"""
from __future__ import annotations

import io
import random
import subprocess
import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np

from .graph import Edge, Graph, Node

# ---------- rendering ----------


def layout(g: Graph):
    """Main path top-down in column 0; 'no' branches in column 1, next to their decision."""
    main, side = [], []
    for n in g.nodes:
        (side if n.id.startswith("A") else main).append(n.id)
    pos = {nid: (0, i) for i, nid in enumerate(main)}
    for n in side:
        d = next(e.src for e in g.edges if e.dst == n and e.label == "no")
        pos[n] = (1, pos[d][1] + 0.5)
    return pos


def render(g: Graph, seed: int = 0, dpi: int = 100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
    rng = random.Random(seed)
    pos = layout(g)
    rows = max(y for _, y in pos.values()) + 1
    W, Hh = 7.0, 1.15 * rows + 0.6
    with plt.xkcd(scale=0.8, length=120, randomness=2):
        fig = plt.figure(figsize=(W, Hh), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Hh, 0); ax.axis("off")
        centre, size = {}, {}
        for n in g.nodes:
            cx = 2.2 + pos[n.id][0] * 3.0 + rng.uniform(-0.15, 0.15)
            cy = 0.7 + pos[n.id][1] * 1.15 + rng.uniform(-0.05, 0.05)
            w, h = rng.uniform(1.9, 2.3), rng.uniform(0.62, 0.72)
            if n.kind == "decision":
                w, h = w * 1.15, h * 1.35
            centre[n.id], size[n.id] = (cx, cy), (w, h)
            kw = dict(fill=False, lw=rng.uniform(1.6, 2.2), ec="k")
            if n.kind == "process":
                ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0", **kw))
            elif n.kind == "decision":
                ax.add_patch(Polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], **kw))
            else:
                ax.add_patch(Ellipse((cx, cy), w, h, **kw))
            lab = n.label if n.kind != "decision" else n.label.rstrip("?") + "?"
            wrap = 16 if n.kind != "decision" else 14
            ax.text(cx, cy, "\n".join(textwrap.wrap(lab, wrap)), ha="center", va="center",
                    fontsize=rng.uniform(8.0, 9.0) if n.kind != "decision" else 7.4, family="DejaVu Sans")
        for e in g.edges:
            (x1, y1), (x2, y2) = centre[e.src], centre[e.dst]
            (w1, h1), (w2, h2) = size[e.src], size[e.dst]
            if abs(x1 - x2) < 0.5:            # same column
                if y2 > y1:
                    a, b = (x1, y1 + h1 / 2), (x2, y2 - h2 / 2)
                else:                         # loop back on the main column: go round on the left
                    a, b = (x1 - w1 / 2, y1), (x2 - w2 / 2, y2)
            else:
                a = (x1 + w1 / 2, y1) if x2 > x1 else (x1 - w1 / 2, y1)
                b = (x2 - w2 / 2, y2) if x2 > x1 else (x2 + w2 / 2, y2)
            style = "arc3,rad=-0.5" if (abs(x1 - x2) < 0.5 and y2 < y1) else "arc3,rad=0"
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.3",
                                                             lw=1.6, color="k", connectionstyle=style,
                                                             shrinkA=2, shrinkB=2))
            if e.label:
                ax.text(a[0] + 0.12 + (0.1 if x2 > x1 else 0), a[1] + (0.22 if abs(x1 - x2) < 0.5 else -0.12),
                        e.label, fontsize=8, family="DejaVu Sans")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi); plt.close(fig)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
    boxes = {}
    for nid, (cx, cy) in centre.items():
        w, h = size[nid]
        boxes[nid] = (int((cx - w / 2) * dpi), int((cy - h / 2) * dpi), int((cx + w / 2) * dpi), int((cy + h / 2) * dpi))
    return img, boxes


def degrade(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """Photo-like version: small rotation, uneven lighting, blur and sensor noise."""
    rng = np.random.default_rng(seed)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderValue=255).astype(float)
    yy, xx = np.mgrid[0:h, 0:w]
    out = out * (0.8 + 0.2 * (xx / w)) - 10 * (yy / h)
    out = cv2.GaussianBlur(out, (3, 3), 0.8) + rng.normal(0, 8, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- recognition ----------
def _ocr(img: np.ndarray, psm: int = 6) -> str:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.png"; cv2.imwrite(str(p), img)
        r = subprocess.run(["tesseract", str(p), "-", "--psm", str(psm)], capture_output=True, text=True)
    return " ".join(r.stdout.split())


def _ocr_words(img: np.ndarray):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.png"; cv2.imwrite(str(p), img)
        r = subprocess.run(["tesseract", str(p), "-", "--psm", "11", "tsv"], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines()[1:]:
        f = line.split("\t")
        if len(f) == 12 and f[11].strip() and float(f[10]) > 30:
            x, y, w, h = map(int, f[6:10])
            out.append((f[11].strip().lower(), x + w / 2, y + h / 2))
    return out


def recognise(img: np.ndarray) -> tuple[Graph, dict]:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    nodes = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] == -1:              # outer contour: not a hole
            continue
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area < 3000 or w < 60 or h < 25 or w > 0.9 * W:
            continue
        fill = area / (w * h)
        kind = "process" if fill > 0.86 else ("start" if fill > 0.66 else "decision")
        if kind == "decision" and not (0.4 < fill < 0.62 and 1.4 < w / h < 4 and area > 6000 and _is_diamond(c)):
            continue                     # closed gaps between strokes and arrows, not a drawn shape
        mask = np.zeros_like(ink); cv2.drawContours(mask, [c], -1, 255, -1)
        nodes.append(dict(box=(x, y, x + w, y + h), kind=kind, mask=mask, fill=fill))
    nodes = [n for n in nodes if not any(m is not n and _inside(n["box"], m["box"]) for m in nodes)]
    nodes.sort(key=lambda n: (n["box"][1], n["box"][0]))
    # erase nodes (with their border) and read their text
    rest = ink.copy()
    near = []
    for n in nodes:
        x0, y0, x1, y1 = n["box"]
        inner = cv2.erode(n["mask"], np.ones((9, 9), np.uint8))          # inside the border only
        clean = np.where(inner > 0, gray, 255).astype(np.uint8)
        pad = 2
        crop = clean[max(0, y0 + pad):y1 - pad, max(0, x0 + pad):x1 - pad]
        if n["kind"] == "decision":
            crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        n["label"] = _ocr(cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        # the interior of a diamond stops short of its sharp tips: reach further around decisions
        e_k, n_k = (17, 41) if n["kind"] == "decision" else (11, 25)
        big = cv2.dilate(n["mask"], np.ones((e_k, e_k), np.uint8))
        rest[big > 0] = 0
        near.append(cv2.dilate(n["mask"], np.ones((n_k, n_k), np.uint8)) > 0)
    # arrows: connected strokes touching two nodes, oriented towards the heavier end (arrowhead)
    dist = [cv2.distanceTransform(255 - n["mask"], cv2.DIST_L2, 3) for n in nodes]
    n_lab, comp, stats, _ = cv2.connectedComponentsWithStats(rest, 8)
    edges = []
    for k in range(1, n_lab):
        if stats[k, cv2.CC_STAT_AREA] < 25:
            continue
        pix = comp == k
        touch = [i for i, z in enumerate(near) if (pix & z).sum() >= 12]
        if len(touch) < 2:
            continue
        ys, xs = np.nonzero(pix)
        mass = []
        for i in touch:                            # ink within 14 px of the stroke end nearest to node i
            d = dist[i][ys, xs]
            j = int(np.argmin(d))
            mass.append(int((np.hypot(xs - xs[j], ys - ys[j]) < 14).sum()))
        dst = touch[int(np.argmax(mass))]          # the arrowhead puts the most ink next to the target
        for src in touch:                          # strokes that merge before a common target: one edge each
            if src != dst:
                edges.append(dict(src=src, dst=dst, label="", pts=np.column_stack([xs, ys])))
    # edge labels: words outside the nodes
    outside = gray.copy()
    for z in near:
        outside[z] = 255
    for word, cx, cy in _ocr_words(outside):
        w = word.strip(".,:;").replace("0", "o")
        if w not in ("yes", "no", "oui", "non"):
            continue
        best = min(edges, key=lambda e: np.min(np.hypot(e["pts"][:, 0] - cx, e["pts"][:, 1] - cy)), default=None)
        if best is not None and nodes[best["src"]]["kind"] == "decision":
            best["label"] = {"oui": "yes", "non": "no"}.get(w, w)
    # start and end: the ellipse without incoming edge is the start
    g = Graph()
    for i, n in enumerate(nodes):
        kind = n["kind"]
        if kind == "start":
            kind = "start" if not any(e["dst"] == i for e in edges) else "end"
        nid = {"start": "S", "end": "E"}.get(kind, f"N{i}")
        if any(m.id == nid for m in g.nodes):
            nid = f"N{i}"
        label = n["label"] if kind != "end" else (n["label"] or "End")
        if kind == "decision" and not label.endswith("?"):
            label += "?"
        g.nodes.append(Node(nid, label, kind))
    ids = [n.id for n in g.nodes]
    g.edges = [Edge(ids[e["src"]], ids[e["dst"]], e["label"]) for e in edges]
    return g, {ids[i]: n["box"] for i, n in enumerate(nodes)}


def _is_diamond(c) -> bool:
    """Four corners, each close to the middle of a side of the bounding box."""
    ap = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(ap) != 4 or not cv2.isContourConvex(ap.reshape(-1, 1, 2)):
        return False
    x, y, w, h = cv2.boundingRect(c)
    mids = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]
    return all(min(np.hypot(px - mx, py - my) for px, py in ap) < 0.2 * max(w, h) for mx, my in mids)


def _inside(a, b):
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]
