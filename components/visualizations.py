"""
Visualization Helpers
=====================
Charts for training curves, confusion matrices, class probabilities, agent
experiments and Grad-CAM overlays.

The charts are Altair (Vega-Lite) specifications that the browser draws. Building
one takes a few milliseconds, whereas a matplotlib figure had to be rendered to a
200-dpi PNG on the server (0.3-0.9 s each, on every rerun). They are interactive
(tooltips), follow the light/dark theme and show Malayalam labels with the
browser's own fonts. The Grad-CAM overlay is composed as an image with NumPy.
"""

import hashlib

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from PIL import Image

TRAIN_COLOR = "#4F46E5"      # indigo
VAL_COLOR = "#F97316"        # orange
PREDICTED_COLOR = "#16A34A"  # green
OTHER_COLOR = "#94A3B8"      # slate


def _curve_chart(history, keys, names, title, y_title, epoch_title, best_epoch=None, best_label="", percent=False):
    n = len(history.get(keys[0], []))
    rows = [{"epoch": i + 1, "series": name, "value": float(v)}
            for key, name in zip(keys, names) for i, v in enumerate(history.get(key, [])[:n])]
    df = pd.DataFrame(rows, columns=["epoch", "series", "value"])
    y_axis = alt.Axis(format=".0%") if percent else alt.Axis()
    lines = alt.Chart(df).mark_line(point=n <= 40, strokeWidth=2).encode(
        x=alt.X("epoch:Q", title=epoch_title, axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y("value:Q", title=y_title, axis=y_axis, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, sort=list(names),
                        scale=alt.Scale(domain=list(names), range=[TRAIN_COLOR, VAL_COLOR]),
                        legend=alt.Legend(orient="bottom", labelLimit=320)),
        tooltip=[alt.Tooltip("epoch:Q", title=epoch_title), alt.Tooltip("series:N", title=" "),
                 alt.Tooltip("value:Q", title=y_title, format=".1%" if percent else ".4f")],
    )
    chart = lines
    if best_epoch and best_epoch != n:
        rule = alt.Chart(pd.DataFrame({"epoch": [best_epoch], "label": [best_label]})).mark_rule(
            strokeDash=[5, 4], color="#64748B").encode(x="epoch:Q", tooltip=[alt.Tooltip("label:N", title=" ")])
        chart = lines + rule
    return chart.properties(title=title, height=280)


def history_charts(history, labels, best_epoch=None):
    """(loss chart, accuracy chart) for a Keras history dict; marks the epoch whose weights were kept."""
    loss = _curve_chart(history, ("loss", "val_loss"), (labels["train_loss"], labels["val_loss"]),
                        labels["plot_loss"], labels["loss_axis"], labels["epoch_axis"], best_epoch,
                        labels["best_epoch_line"])
    acc = _curve_chart(history, ("accuracy", "val_accuracy"), (labels["train_acc"], labels["val_acc"]),
                       labels["plot_acc"], labels["acc_axis"], labels["epoch_axis"], best_epoch,
                       labels["best_epoch_line"], percent=True)
    return loss, acc


def confusion_chart(cm, class_names, labels):
    """Confusion-matrix heatmap. Rows = actual class, columns = predicted class.
    (No chart title: a long title - e.g. in Malayalam - would squeeze the cells; the
    caller shows the title as text.)"""
    cm = np.asarray(cm)
    names = [str(c) for c in class_names]
    totals = cm.sum(axis=1)
    rows = [{"actual": names[i], "predicted": names[j], "count": int(cm[i, j]),
             "share": float(cm[i, j] / totals[i]) if totals[i] else 0.0}
            for i in range(cm.shape[0]) for j in range(cm.shape[1])]
    df = pd.DataFrame(rows)
    size = int(min(480, max(300, 44 * len(names))))
    many = max(len(n) for n in names) * len(names) > 45       # rotate the x labels only if they would collide
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=names, title=labels["predicted"],
                axis=alt.Axis(labelAngle=-40 if many else 0, labelLimit=110, labelOverlap=False)),
        y=alt.Y("actual:N", sort=names, title=labels["actual"], axis=alt.Axis(labelLimit=120, labelOverlap=False)),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
        tooltip=[alt.Tooltip("actual:N", title=labels["actual"]),
                 alt.Tooltip("predicted:N", title=labels["predicted"]),
                 alt.Tooltip("count:Q", title=labels["cm_count"]),
                 alt.Tooltip("share:Q", title=labels["cm_share"], format=".0%")],
    )
    text = base.mark_text(fontSize=11 if many else 13).encode(
        text="count:Q",
        color=alt.condition(alt.datum.share > 0.5, alt.value("white"), alt.value("#1E293B")),
    )
    return (heat + text).properties(width=size, height=size)


def probabilities_chart(class_names, probs, pred_idx, labels):
    """Horizontal bars of the model's softmax output for every class (predicted class highlighted)."""
    names = [str(c) for c in class_names]
    df = pd.DataFrame({"class": names, "probability": [float(p) for p in np.asarray(probs)],
                       "predicted": [i == int(pred_idx) for i in range(len(names))]})
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("probability:Q", title=labels["confidence_pct"], scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1])),
        y=alt.Y("class:N", sort=names, title=None, axis=alt.Axis(labelLimit=140, labelOverlap=False)),
        color=alt.condition(alt.datum.predicted, alt.value(PREDICTED_COLOR), alt.value(OTHER_COLOR)),
        tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".2%")],
    ).properties(title=labels["class_probs"], height=alt.Step(24))


def gradcam_overlay(image, heatmap, alpha=0.45, scale=8):
    """RGB uint8 image: the 28x28 model input (grayscale) blended with the heatmap
    (jet colour map: blue = little influence, red = strong), enlarged with crisp pixels."""
    img = np.clip(np.asarray(image, dtype=np.float32).squeeze(), 0.0, 1.0)
    heat = np.clip(np.asarray(heatmap, dtype=np.float32).squeeze(), 0.0, 1.0)
    rgb = np.repeat(img[..., None], 3, axis=-1)
    colored = colormaps["jet"](heat)[..., :3]
    out = (1.0 - alpha) * rgb + alpha * colored
    out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
    return enlarge(out, scale)


def enlarge(array, scale=8):
    """Upscale a small image with nearest-neighbour pixels (a 28x28 input stays readable)."""
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(arr.squeeze())
    return np.asarray(img.resize((img.width * scale, img.height * scale), Image.NEAREST))


def show_chart(chart):
    st.altair_chart(chart, width="stretch")


def render_experiment_chart(result, labels):
    """Inline loss/accuracy charts for an agent experiment result."""
    if result.get("status") != "success" or "history" not in result:
        return
    loss, acc = history_charts(result["history"], labels)
    c1, c2 = st.columns(2)
    # An explicit element height: inside the fixed-height chat box the chart would otherwise shrink.
    with c1:
        st.altair_chart(loss.properties(height=180), width="stretch", height=300)
    with c2:
        st.altair_chart(acc.properties(height=180), width="stretch", height=300)


def render_experiment_confusion_matrix(result, labels):
    """Confusion matrix heatmap from a run_experiment result."""
    if result.get("status") != "success":
        return
    cm_data = result.get("confusion_matrix")
    cn = result.get("class_names", [])
    if not cm_data or not cn:
        return
    st.markdown(f"**{labels['confusion_title']}**")
    st.altair_chart(confusion_chart(cm_data, cn, labels), width="content")


def render_parameter_summary(result, labels):
    """Show a compact parameter summary for the experiment."""
    if result.get("status") != "success":
        return
    params = result.get("parameters", {})
    if params:
        st.markdown(labels["params_summary"].format(
            total=f"{params.get('total', 0):,}", trainable=f"{params.get('trainable', 0):,}",
            non_trainable=f"{params.get('non_trainable', 0):,}"))


def render_code_download(text, label, experiment_idx=0):
    """Extract Python code blocks from text and render download buttons."""
    from components.chat_helpers import split_markdown_and_code
    blocks = [c for kind, c, lang in split_markdown_and_code(text) if kind == "code" and lang == "python"]
    for i, code in enumerate(blocks):
        code = code.strip()
        if code:
            digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]
            st.download_button(
                label,
                data=code,
                file_name="experiment_model.py",
                mime="text/x-python",
                key=f"dl_code_{experiment_idx}_{i}_{digest}",
                width="stretch",
            )
