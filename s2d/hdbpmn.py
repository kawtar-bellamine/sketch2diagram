"""hdBPMN: hand-drawn BPMN diagrams, photographed or scanned, with their ground-truth BPMN models (Schäfer et al.,
CAiSE 2021, github.com/dwslab/hdBPMN, CC BY 4.0). Fetched at a pinned commit, not redistributed.

Only the diagrams drawn for exercises 0 and 1 use nothing but tasks, events, gateways and sequence flows, the
vocabulary of a flowchart: 137 of them (writer split: 85 train, 30 validation, 22 test). This module reads the
images, the ground-truth graphs with their boxes in image pixels, and the two exercise texts.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/hdBPMN"
REPO = "https://github.com/dwslab/hdBPMN.git"
COMMIT = "1d7e5fcd17deeb6894dd0552699c88e21bcab9cb"
NS = {"b": "http://www.omg.org/spec/BPMN/20100524/MODEL", "di": "http://www.omg.org/spec/BPMN/20100524/DI",
      "dc": "http://www.omg.org/spec/DD/20100524/DC"}
SIMPLE = {"process", "task", "startEvent", "endEvent", "exclusiveGateway", "parallelGateway", "sequenceFlow",
          "intermediateCatchEvent", "intermediateThrowEvent", "eventBasedGateway"}
KIND = {"task": "task", "startEvent": "event", "endEvent": "event", "intermediateCatchEvent": "event",
        "intermediateThrowEvent": "event", "exclusiveGateway": "gateway", "parallelGateway": "gateway",
        "eventBasedGateway": "gateway"}


def _git(*a):
    subprocess.run(["git", "-C", str(DEST), *a], check=True)


def fetch(exercises=("ex00", "ex01")) -> Path:
    """Annotations of the chosen exercises, their images, the writer split and the exercise texts."""
    if not (DEST / "data/writer_split.csv").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO, str(DEST)], check=True)
        _git("checkout", "-q", COMMIT, "--", "data/writer_split.csv", "data/exercises.pdf",
             *[f"data/annotations/{e}" for e in exercises])
    names = {p.stem for e in exercises for p in (DEST / "data/annotations" / e).glob("*.bpmn")}
    have = {p.stem for p in (DEST / "data/images").glob("*/*")} if (DEST / "data/images").exists() else set()
    if names - have:
        files = subprocess.run(["git", "-C", str(DEST), "ls-tree", "-r", "--name-only", COMMIT, "data/images"],
                               capture_output=True, text=True, check=True).stdout.split()
        need = [f for f in files if Path(f).stem in names - have]
        for i in range(0, len(need), 40):
            _git("checkout", "-q", COMMIT, "--", *need[i:i + 40])
    return DEST


def parse(path: Path, img_w: int) -> dict:
    """Ground truth: nodes (kind, label, box in image pixels) and directed edges."""
    txt = path.read_text()
    bg = json.loads(re.search(r"<!-- (\{.*?\}) -->", txt).group(1))["backgroundSize"]
    s = img_w / bg
    root = ET.fromstring(txt)
    proc = root.find("b:process", NS)
    tags = {el.tag.split("}")[1] for el in proc}
    nodes, edges = {}, []
    for el in proc:
        t = el.tag.split("}")[1]
        if t in KIND:
            nodes[el.get("id")] = dict(kind=KIND[t], type=t, label=" ".join((el.get("name") or "").split()))
        elif t == "sequenceFlow":
            edges.append((el.get("sourceRef"), el.get("targetRef")))
    for sh in root.iter(f"{{{NS['di']}}}BPMNShape"):
        b = sh.find("dc:Bounds", NS)
        if sh.get("bpmnElement") in nodes:
            x, y, w, h = (float(b.get(k)) * s for k in ("x", "y", "width", "height"))
            nodes[sh.get("bpmnElement")]["box"] = (x, y, x + w, y + h)
    return dict(nodes=nodes, edges=[e for e in edges if e[0] in nodes and e[1] in nodes], simple=tags <= SIMPLE)


def diagrams(split: str | None = None):
    from PIL import Image, ImageOps
    root = fetch()
    sp = {r["writer"]: r["split"] for r in csv.DictReader(open(root / "data/writer_split.csv"))}
    out = []
    for ann in sorted((root / "data/annotations").glob("ex0[01]/*.bpmn")):
        w = re.search(r"writer\d+", ann.stem).group(0)
        if split and sp[w] != split:
            continue
        img = next((root / "data/images" / ann.parent.name).glob(ann.stem + ".*"))
        im = ImageOps.exif_transpose(Image.open(img))
        g = parse(ann, im.size[0])
        if g["simple"]:
            out.append(dict(name=ann.stem, exercise=ann.parent.name, split=sp[w], image=img, size=im.size, **g))
    return out


def exercise_texts() -> dict:
    import pymupdf
    t = "".join(p.get_text() for p in pymupdf.open(fetch() / "data/exercises.pdf"))
    out = {}
    for m in re.finditer(r"Exercise (\d+)\n(.*?)Instructions:", t, re.S):
        body = m.group(2).split("\n", 1)[1]                    # drop "Model the following business process ..."
        out[f"ex{int(m.group(1)):02d}"] = " ".join(body.replace("-\n", "").split())
    return out
