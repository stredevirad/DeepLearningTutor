"""Chart helpers: Altair specs carry the right data and labels; Grad-CAM overlays are valid images."""

import json

import numpy as np
import pytest

from components import visualizations as V
from config.translations import UI

LABEL_KEYS = ["train_loss", "val_loss", "train_acc", "val_acc", "epoch_axis", "loss_axis", "acc_axis", "plot_loss",
              "plot_acc", "best_epoch_line", "confusion_title", "actual", "predicted", "confidence_pct", "class_probs",
              "cm_count", "cm_share", "params_summary", "pc_class", "pc_precision", "pc_recall", "pc_f1"]


def labels(lang):
    return {k: UI[lang][k] for k in LABEL_KEYS}


def spec(chart):
    return json.dumps(chart.to_dict(), ensure_ascii=False)


HISTORY = {k: list(np.linspace(1, 0.2, 8)) for k in ("loss", "val_loss", "accuracy", "val_accuracy")}


@pytest.mark.parametrize("lang", ["en", "ml"])
def test_history_charts_have_both_series_in_the_ui_language(lang):
    lab = labels(lang)
    loss, acc = V.history_charts(HISTORY, lab)
    for chart, a, b, title in ((loss, "train_loss", "val_loss", "plot_loss"), (acc, "train_acc", "val_acc", "plot_acc")):
        text = spec(chart)
        assert lab[a] in text and lab[b] in text and lab[title] in text
    assert '".0%"' in spec(acc)                                   # accuracy axis shown as a percentage


def test_best_epoch_marker_only_when_weights_were_restored():
    lab = labels("en")
    loss, _ = V.history_charts(HISTORY, lab, best_epoch=5)
    assert "layer" in loss.to_dict() and lab["best_epoch_line"] in spec(loss)
    loss, _ = V.history_charts(HISTORY, lab, best_epoch=8)          # best == last epoch: nothing to mark
    assert "layer" not in loss.to_dict()


def test_confusion_chart_cells_and_row_shares():
    lab = labels("ml")
    chart = V.confusion_chart([[3, 1], [0, 4]], ["a", "b"], lab)
    data = next(iter(chart.to_dict()["datasets"].values()))
    cells = {(d["actual"], d["predicted"]): (d["count"], round(d["share"], 2)) for d in data}
    assert cells == {("a", "a"): (3, 0.75), ("a", "b"): (1, 0.25), ("b", "a"): (0, 0.0), ("b", "b"): (4, 1.0)}
    assert lab["actual"] in spec(chart) and lab["predicted"] in spec(chart)
    assert "title" not in chart.to_dict()          # a long (Malayalam) title would squeeze the cells


def test_confusion_chart_handles_a_class_with_no_samples():
    chart = V.confusion_chart(np.zeros((3, 3), dtype=int), ["x", "y", "z"], labels("en"))
    data = next(iter(chart.to_dict()["datasets"].values()))
    assert len(data) == 9 and all(d["share"] == 0.0 for d in data)


@pytest.mark.parametrize("lang", ["en", "ml"])
def test_per_class_table_has_one_row_per_class_in_the_ui_language(lang):
    from core.training import per_class_metrics
    lab = labels(lang)
    rows = per_class_metrics([[3, 1], [0, 4]], ["a", "b"])          # a: 3 of 4 found; b: 4 of 5 predictions right
    table = V.per_class_table(rows, lab)
    assert list(table.columns) == [lab["pc_class"], lab["pc_precision"], lab["pc_recall"], lab["pc_f1"], lab["cm_count"]]
    assert table.values.tolist() == [["a", "100.0%", "75.0%", "85.7%", 4], ["b", "80.0%", "100.0%", "88.9%", 4]]


def test_probabilities_chart_marks_the_predicted_class():
    chart = V.probabilities_chart(["cat", "dog", "owl"], [0.1, 0.7, 0.2], 1, labels("en"))
    data = next(iter(chart.to_dict()["datasets"].values()))
    assert [d["class"] for d in data if d["predicted"]] == ["dog"]
    assert sum(d["probability"] for d in data) == pytest.approx(1.0)


def test_gradcam_overlay_is_a_crisp_rgb_image():
    image = np.zeros((28, 28), np.float32)
    heat = np.zeros((28, 28), np.float32)
    heat[:14] = 1.0                                                # top half: strongest evidence
    out = V.gradcam_overlay(image, heat, scale=8)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    top, bottom = out[10, 10], out[200, 200]
    assert top[0] > top[2] and bottom[2] > bottom[0]               # red where heat is high, blue where it is low


def test_enlarge_keeps_pixels_sharp():
    small = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
    big = V.enlarge(small, scale=4)
    assert big.shape == (8, 8) and set(np.unique(big)) == {0, 255}


def test_code_download_only_for_python_blocks(monkeypatch):
    buttons = []
    monkeypatch.setattr(V.st, "download_button", lambda label, data, **kw: buttons.append(data))
    V.render_code_download("x\n```python\nprint(1)\n```\n```bash\nls\n```\n```py\nx = 2\n```", "Download")
    assert buttons == ["print(1)", "x = 2"]
