from s2d.bpmn_sketch import load, recognise
from s2d.eval_bpmn import match, task_f1
from s2d.hdbpmn import diagrams, exercise_texts


def test_real_dataset_subset():
    d = diagrams()
    assert len(d) == 137 and {x["split"] for x in d} == {"train", "val", "test"}
    x = d[0]
    assert all("box" in n for n in x["nodes"].values()) and x["edges"]
    assert {n["kind"] for n in x["nodes"].values()} <= {"task", "event", "gateway"}


def test_ground_truth_boxes_lie_inside_the_photo():
    for x in diagrams("test"):
        w, h = x["size"]
        assert all(n["box"][2] <= w * 1.02 and n["box"][3] <= h * 1.02 for n in x["nodes"].values()), x["name"]


def test_exercise_texts():
    t = exercise_texts()
    assert t["ex00"].startswith("When the bank receives") and "disbursement" in t["ex01"]


def test_matching_and_task_f1():
    gt = dict(nodes={"a": dict(kind="task", label="x", box=(0, 0, 100, 50))}, edges=[])
    pred = dict(nodes=[dict(kind="task", box=(10, 5, 90, 45)), dict(kind="event", box=(300, 300, 320, 320))])
    assert match(gt, pred) == {0: "a"}
    assert task_f1(["Evaluate the application"], ["evaluate application", "notify customer"]) == 2 / 3


def test_recogniser_finds_shapes_on_a_real_photo():
    x = next(d for d in diagrams("test") if d["name"] == "ex01_writer0088")
    r = recognise(load(x["image"]), ocr=False)
    assert len(match(x, r)) >= 8 and r["edges"]
