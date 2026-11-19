"""Recognise a photographed hand-drawn BPMN diagram (tasks, events, gateways, sequence flows) with classical computer
vision, no training data. Built for hdBPMN photos: squared or lined paper, blue or black ink, gateways with an X or
a + inside, labels of events written outside the circle.

1. ink: each pixel against the local paper brightness, on the image resized to a fixed width (grid lines of squared
   paper are only slightly darker than the paper, ink much darker);
2. shapes: the ink plus the regions it encloses gives solid blobs; a morphological opening removes the thin strokes
   (arrows, handwriting) and keeps the solid shapes; each blob is classified by how much of its bounding box it fills
   (rectangle about 0.9, circle 0.79, diamond 0.5) and by its size;
3. arrows: the shapes are erased, each remaining stroke touching two shapes is a flow, oriented towards the end with
   the most ink (the arrowhead);
4. task labels: OCR (Tesseract) inside the rectangle.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

WIDTH = 1600
DEFAULT = dict(t=0.75, open_k=15, min_side=22, near=45, gap=3, rect_fill=0.80)   # chosen on the validation writers


def _ocr(img: np.ndarray) -> str:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.png"; cv2.imwrite(str(p), img)
        r = subprocess.run(["tesseract", str(p), "-", "--psm", "6"], capture_output=True, text=True)
    return " ".join("".join(ch for ch in r.stdout if ch.isalnum() or ch.isspace()).split())


def load(path) -> np.ndarray:
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.array(im)


def recognise(gray: np.ndarray, p: dict | None = None, ocr: bool = True) -> dict:
    p = {**DEFAULT, **(p or {})}
    s = WIDTH / gray.shape[1]
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))   # paper brightness, ink removed
    norm = g.astype(np.float32) / np.maximum(bg, 1)                          # grid lines are only slightly darker
    ink = ((norm < p["t"]) * 255).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # solid shapes: ink + enclosed regions, then remove thin strokes; a closing first bridges small gaps in outlines
    H, W = ink.shape
    flood = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["gap"], p["gap"])))
    flood = cv2.copyMakeBorder(flood, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)   # the fill must start on paper
    mask = np.zeros((H + 6, W + 6), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 128)
    solid = np.where(flood[2:-2, 2:-2] == 128, 0, 255).astype(np.uint8)
    k = p["open_k"]
    blobs = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
    nodes = []
    for i in range(1, n_lab):
        x, y, w, h, area = stats[i]
        if min(w, h) < p["min_side"] or w > 0.6 * W or h > 0.6 * H or x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        fill = area / (w * h)
        ar = w / h
        if fill > p["rect_fill"] and w >= 1.15 * h and w > 70:
            kind = "task"
        elif fill > 0.68 and 0.6 < ar < 1.6 and (fill < p["rect_fill"] or max(w, h) < 70):
            kind = "event"
        elif 0.38 < fill <= 0.68 and 0.6 < ar < 1.7:
            kind = "gateway"
        elif fill > p["rect_fill"] and w > 70:
            kind = "task"
        else:
            continue
        nodes.append(dict(kind=kind, box=(x, y, x + w, y + h), lab=i, fill=round(float(fill), 2)))
    # arrows: erase the shapes, keep the strokes that touch two shapes
    node_map = np.zeros(lab.shape, np.int32)
    for j, n in enumerate(nodes):
        node_map[lab == n["lab"]] = j + 1
    any_node = (node_map > 0).astype(np.uint8)
    rest = ink.copy()
    rest[cv2.dilate(any_node, np.ones((9, 9), np.uint8)) > 0] = 0
    near_map = cv2.dilate(node_map.astype(np.float32), np.ones((p["near"], p["near"]), np.uint8)).astype(np.int32)
    n_c, comp, cst, _ = cv2.connectedComponentsWithStats(rest, 8)
    edges = []
    for c in range(1, n_c):
        if cst[c, cv2.CC_STAT_AREA] < 30:
            continue
        x, y, w, h = cst[c, :4]
        sub = comp[y:y + h, x:x + w] == c
        nm = near_map[y:y + h, x:x + w][sub]
        ids, cnt = np.unique(nm[nm > 0], return_counts=True)
        touch = [int(i) - 1 for i, k in zip(ids, cnt) if k >= 8]
        if len(touch) < 2:
            continue
        ys, xs = np.nonzero(sub); xs = xs + x; ys = ys + y
        mass = []
        for i in touch:
            bx0, by0, bx1, by1 = nodes[i]["box"]
            d = np.hypot(np.clip(xs, bx0, bx1) - xs, np.clip(ys, by0, by1) - ys)   # distance to the node's box
            j = int(np.argmin(d))
            mass.append(int((np.hypot(xs - xs[j], ys - ys[j]) < 18).sum()))
        dst = touch[int(np.argmax(mass))]
        for src in touch:
            if src != dst:
                edges.append((src, dst))
    # labels of tasks
    if ocr:
        for n in nodes:
            if n["kind"] != "task":
                continue
            x0, y0, x1, y1 = n["box"]
            crop = g[y0 + 6:y1 - 6, x0 + 6:x1 - 6]
            n["label"] = _ocr(cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)) if crop.size else ""
    out_nodes = [dict(kind=n["kind"], label=n.get("label", ""), box=tuple(float(v) / s for v in n["box"]))
                 for n in nodes]
    return dict(nodes=out_nodes, edges=sorted(set(edges)))
