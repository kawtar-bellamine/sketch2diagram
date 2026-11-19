"""Figures for the README: a real test photo with the recognised shapes, and synthetic against real results."""
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .bpmn_sketch import WIDTH, load, recognise  # noqa: E402
from .hdbpmn import ROOT, diagrams  # noqa: E402

COL = {"task": (31, 95, 168), "event": (230, 145, 56), "gateway": (46, 139, 87)}


def overlay(d):
    g = load(d["image"]); s = WIDTH / g.shape[1]
    im = cv2.cvtColor(cv2.resize(g, None, fx=s, fy=s), cv2.COLOR_GRAY2RGB)
    pr = recognise(g, ocr=False)
    for n in d["nodes"].values():
        a, b, c, e = (int(v * s) for v in n["box"])
        cv2.rectangle(im, (a, b), (c, e), (200, 200, 200), 2)
    for n in pr["nodes"]:
        a, b, c, e = (int(v * s) for v in n["box"])
        cv2.rectangle(im, (a, b), (c, e), COL[n["kind"]], 4)
    cents = [((n["box"][0] + n["box"][2]) / 2 * s, (n["box"][1] + n["box"][3]) / 2 * s) for n in pr["nodes"]]
    for i, j in pr["edges"]:
        cv2.arrowedLine(im, tuple(int(v) for v in cents[i]), tuple(int(v) for v in cents[j]), (192, 57, 43), 3,
                        tipLength=0.08)
    return im


def main():
    r = json.loads((ROOT / "results/metrics_hdbpmn.json").read_text())
    rows = json.loads((ROOT / "results/per_diagram_hdbpmn.json").read_text())
    test = {x["name"]: x for x in rows}
    ds = [d for d in diagrams("test")]
    good = max(ds, key=lambda d: test[d["name"]]["matched"] / test[d["name"]]["gt_nodes"])
    mid = sorted(ds, key=lambda d: test[d["name"]]["matched"] / test[d["name"]]["gt_nodes"])[len(ds) // 2]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=120)
    for ax, d, t in zip(axes, (good, mid), ("best test photo", "median test photo")):
        ax.imshow(overlay(d)); ax.axis("off")
        x = test[d["name"]]
        ax.set_title(f"{t} ({d['name']}): {x['matched']}/{x['gt_nodes']} shapes, {x['correct_edges']}/{x['gt_edges']} "
                     "flows", fontsize=10)
    fig.suptitle("Real hand-drawn BPMN (hdBPMN test writers). Grey: ground truth; blue task, orange event, green "
                 "gateway: recognised; red arrows: recognised flows", fontsize=10.5)
    fig.tight_layout(); fig.savefig(ROOT / "results/figures/fig_hdbpmn_examples.png"); plt.close(fig)

    syn = json.loads((ROOT / "results/metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=130)
    labels = ["Shapes F1", "Flows F1", "Task labels read"]
    sk = syn.get("sketch_clean") or {}
    real = [r["test"]["node_f1"], r["test"]["edge_f1"], r["test"]["task_labels_readable_share"]]
    x = np.arange(3); w = 0.38
    synv = [sk["structure_only"]["node_f1"], sk["structure_only"]["edge_f1"], sk["with_ocr"]["node_f1"]]
    b1 = ax.bar(x - w / 2, synv, w, color="#9aa7b5", label="synthetic sketches (generated, clean scan)")
    b2 = ax.bar(x + w / 2, real, w, color="#1f5fa8", label="real hand-drawn photos (hdBPMN test, 22)")
    ax.bar_label(b1, labels=[f"{v:.2f}" for v in synv], fontsize=8.5)
    ax.bar_label(b2, labels=[f"{v:.2f}" for v in real], fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.12); ax.legend(fontsize=8.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Same recogniser, synthetic against real drawings", fontsize=10.5)
    fig.tight_layout(); fig.savefig(ROOT / "results/figures/fig_synthetic_vs_real.png"); plt.close(fig)


if __name__ == "__main__":
    main()
