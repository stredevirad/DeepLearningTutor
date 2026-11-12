"""
Deep Learning Tutor - performance benchmarks
============================================
Measures start-up, Streamlit rerun cost and the individual building blocks
(data loading, model building, inference, image preprocessing, Grad-CAM,
charts, RAG, prompt building, Ollama status checks, agent tools, MCP).

    python benchmarks/run_benchmarks.py --out results.json            # everything
    python benchmarks/run_benchmarks.py --out results.json --only app  # one group

Groups: startup, app, components, mcp.

Hermetic by design: `.env` is not loaded and no Gemini key is visible, so no
request is ever sent to Gemini. LLM calls inside the app scenarios are replaced
by instant fakes (their network latency is not what is being measured).
Ollama "server down" is simulated with a closed local port.

Every number is wall-clock time on the machine that runs the script; compare
runs from the same machine only. Results are written as JSON.
"""

import argparse
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERMETIC_ENV = {
    "DLT_SKIP_DOTENV": "1", "GEMINI_API_KEYS": "", "GEMINI_API_KEY": "",
    "TF_CPP_MIN_LOG_LEVEL": "2", "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8",
}
os.environ.update(HERMETIC_ENV)
os.environ.pop("MCP_AUTH_TOKEN", None)
sys.path.insert(0, str(ROOT))

RESULTS = {}


# ─────────────────────────── helpers ───────────────────────────

def record(group, name, value, unit="ms", note=None):
    entry = {"value": value, "unit": unit}
    if note:
        entry["note"] = note
    RESULTS.setdefault(group, {})[name] = entry
    shown = f"{value:.3g} {unit}" if isinstance(value, (int, float)) else value
    print(f"  [{group}] {name}: {shown}" + (f"  ({note})" if note else ""), flush=True)


def measure(fn, n=5, warmup=1):
    """Median / min / max wall time in ms of fn() over n runs (after warmup runs)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {"median": statistics.median(times), "min": min(times), "max": max(times), "n": n}


def record_stats(group, name, stats, note=None):
    note = f"median of {stats['n']}, min {stats['min']:.3g}, max {stats['max']:.3g}" + (f"; {note}" if note else "")
    record(group, name, round(stats["median"], 3), note=note)


def once_ms(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000, out


def run_python(code, timeout=600):
    """Run `code` in a fresh interpreter (hermetic env) and return its last stdout line."""
    env = {**os.environ, **HERMETIC_ENV}
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return proc.stdout.strip().splitlines()[-1]


def safe(group, name, fn):
    try:
        fn()
    except Exception as e:  # keep going: one failing measurement must not hide the others
        record(group, name, f"ERROR {type(e).__name__}: {e}"[:300], unit="")


# ─────────────────────────── startup ───────────────────────────

def bench_startup(repeats=3):
    g = "startup"
    tf_times = [float(run_python(
        "import time; t=time.perf_counter(); import tensorflow; print((time.perf_counter()-t)*1000)"))
        for _ in range(repeats)]
    record(g, "import_tensorflow", round(statistics.median(tf_times), 1), note=f"fresh process, median of {repeats}")

    st_times = [float(run_python(
        "import time; t=time.perf_counter(); import streamlit; print((time.perf_counter()-t)*1000)"))
        for _ in range(repeats)]
    record(g, "import_streamlit", round(statistics.median(st_times), 1), note=f"fresh process, median of {repeats}")

    code = (
        "import time, streamlit as st\n"
        "from streamlit.testing.v1 import AppTest\n"
        "t=time.perf_counter()\n"
        f"at=AppTest.from_file({str(ROOT / 'app.py')!r}, default_timeout=300); at.run()\n"
        "assert not at.exception, [e.value for e in at.exception]\n"
        "print((time.perf_counter()-t)*1000)\n"
    )
    cold = [float(run_python(code)) for _ in range(repeats)]
    record(g, "first_script_run_cold", round(statistics.median(cold), 1),
           note=f"fresh process: all app imports + first page render (streamlit already imported), median of {repeats}")

    # Real server: time until /_stcore/health answers (no browser session yet).
    import httpx
    port = 8597
    env = {**os.environ, **HERMETIC_ENV}
    boot = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                                 "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                                cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while time.perf_counter() - t0 < 120:
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).text == "ok":
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            boot.append((time.perf_counter() - t0) * 1000)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    record(g, "server_boot_to_health_ok", round(statistics.median(boot), 1),
           note=f"`streamlit run --server.headless`, median of {repeats}")


# ─────────────────────────── app (AppTest) scenarios ───────────────────────────

def _widget(at, kind, *keys):
    for key in keys:
        try:
            return getattr(at, kind)(key=key)
        except (KeyError, IndexError, ValueError):
            continue
    raise KeyError(f"no {kind} with keys {keys}")


def _select_tab(at, index):
    """Open a main tab. Returns True if the app has server-side (lazy) tabs; with the
    old always-render-every-tab layout tab switching is purely client-side."""
    if "active_tab" not in at.session_state:
        return False
    at.session_state["active_tab"] = index
    return True


def _timed_run(at, timeout=600):
    t0 = time.perf_counter()
    at.run(timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    if at.exception:
        raise RuntimeError([e.value for e in at.exception])
    return dt


def _alternate(at, action_a, action_b, n):
    times = []
    for i in range(n):
        (action_a if i % 2 == 0 else action_b)()
        times.append(_timed_run(at))
    return {"median": statistics.median(times), "min": min(times), "max": max(times), "n": n}


def _fake_llm():
    """Instant, offline stand-ins for Gemini so scenarios never touch the network.
    Returns a function that restores the real implementations."""
    from google.genai import types
    from backend import llm_api

    def generate(contents, model_id=None, config=None, **kw):
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text="Dropout randomly disables neurons.")]))])

    saved = {name: getattr(llm_api, name) for name in ("generate", "generate_stream", "generate_text", "embed_texts")}
    llm_api.API_KEYS[:] = ["bench-key"]
    llm_api.generate = generate
    llm_api.generate_stream = lambda contents, **kw: iter(["Dropout randomly ", "disables neurons."])
    llm_api.generate_text = lambda prompt, **kw: "**Solid result.** Try 50 epochs."
    llm_api.embed_texts = lambda texts, task_type=None, **kw: [[1.0, float(len(t) % 7)] for t in texts]

    def restore():
        for name, fn in saved.items():
            setattr(llm_api, name, fn)
        llm_api.API_KEYS[:] = []
    return restore


def bench_app(n=5):
    import streamlit as st
    import tensorflow as tf
    from streamlit.testing.v1 import AppTest

    g = "app"
    app_path = str(ROOT / "app.py")
    import tts_engine
    tts_engine.text_to_audio = lambda text, lang="en": None   # gTTS needs the internet: never call it here

    def new_app():
        at = AppTest.from_file(app_path, default_timeout=600)
        return at

    # --- New session on an already-warm server process (modules imported, caches warm) ---
    warm = new_app()
    warm.run()
    t_warm = []
    for _ in range(3):
        at = new_app()
        t_warm.append(_timed_run(at))
    record(g, "new_session_first_run_warm", round(statistics.median(t_warm), 1),
           note="new browser session, server process already warm; median of 3")

    # --- Untrained app ---
    at = new_app()
    at.run()
    record_stats(g, "rerun_noop_untrained", measure(lambda: _timed_run(at), n=n))
    epochs = _widget(at, "slider", "epochs_nn")
    record_stats(g, "slider_change_untrained", _alternate(
        at, lambda: epochs.set_value(20), lambda: epochs.set_value(40), n))
    lang = _widget(at, "radio", "lang_choice")
    record_stats(g, "language_switch_untrained", _alternate(
        at, lambda: lang.set_value("Malayalam"), lambda: lang.set_value("English"), 4))

    # --- Iris trained (2x16, lr 0.01, batch 16, 50 epochs) ---
    at = new_app()
    at.run()
    _widget(at, "slider", "epochs_nn").set_value(50)
    _widget(at, "select_slider", "learning_rate_nn", "learning_rate").set_value(0.01)
    _widget(at, "selectbox", "batch_size_nn", "batch_size").set_value(16)
    at.button(key="train_button").click()
    train_ms = _timed_run(at)
    record(g, "train_click_iris_50_epochs", round(train_ms, 1),
           note="whole rerun: data prep + 50 epochs + evaluation + result rendering (LLM review not configured)")
    acc = at.session_state["training_outcome"]["final"]["val_accuracy"]
    record(g, "iris_val_accuracy", round(acc * 100, 2), unit="%")

    record_stats(g, "rerun_noop_trained_iris", measure(lambda: _timed_run(at), n=n),
                 note="default tab open")
    epochs = _widget(at, "slider", "epochs_nn")
    record_stats(g, "slider_change_trained_iris", _alternate(
        at, lambda: epochs.set_value(60), lambda: epochs.set_value(50), n))
    lang = _widget(at, "radio", "lang_choice")
    record_stats(g, "language_switch_trained_iris", _alternate(
        at, lambda: lang.set_value("Malayalam"), lambda: lang.set_value("English"), 4))
    ds = _widget(at, "selectbox", "dataset_nn")
    record_stats(g, "dataset_change_trained_iris", _alternate(
        at, lambda: ds.set_value("Wine"), lambda: ds.set_value("Iris"), 4))

    if _select_tab(at, 1):
        record(g, "tab_switch_to_visualizations_iris", round(_timed_run(at), 1),
               note="first visit renders the charts")
        record_stats(g, "rerun_noop_visualizations_tab_iris", measure(lambda: _timed_run(at), n=n))
        _select_tab(at, 3)
        record(g, "tab_switch_to_test_iris", round(_timed_run(at), 1))
        _select_tab(at, 0)
        _timed_run(at)
    else:
        record(g, "tab_switch_to_visualizations_iris", 0.0,
               note="client-side only: every tab is already rendered on every rerun")

    # --- Chat message (fake instant LLM, so this is the app's own overhead) ---
    restore_llm = _fake_llm()
    at = new_app()
    at.run()
    if _select_tab(at, 2):
        _timed_run(at)
    times = []
    for i in range(3):
        at.chat_input[0].set_value(f"What does dropout do? ({i})")
        times.append(_timed_run(at))
    record(g, "chat_message_roundtrip_fake_llm", round(statistics.median(times), 1),
           note="median of 3; LLM replaced by an instant fake")
    restore_llm()                     # back to "no key" so the CNN scenario matches the Iris one

    # --- CNN on a real MNIST subset + uploaded handwritten digit ---
    (X_tr, y_tr), (X_va, y_va) = tf.keras.datasets.mnist.load_data()
    subset = ((X_tr[:6000], y_tr[:6000]), (X_va[:1000], y_va[:1000]))
    real_load = tf.keras.datasets.mnist.load_data
    tf.keras.datasets.mnist.load_data = lambda: subset
    st.cache_resource.clear()
    try:
        at = new_app()
        at.run()
        _widget(at, "selectbox", "network_type_en", "network_type").set_value("CNN")
        _timed_run(at)
        _widget(at, "slider", "epochs_cnn").set_value(1)
        _widget(at, "number_input", "conv_layers").set_value(2)
        _widget(at, "selectbox", "base_filters").set_value(16)
        at.button(key="train_button").click()
        record(g, "train_click_cnn_mnist_subset_1_epoch", round(_timed_run(at), 1),
               note="6,000-image MNIST subset, 2 conv layers, 16 filters")
        if _select_tab(at, 3):
            _timed_run(at)
        version = at.session_state["model_version"]
        png = (ROOT / "handwritten_seven.png").read_bytes()
        up = _widget(at, "file_uploader", f"test_img_{version}_0", f"test_img_{version}")
        up.upload("handwritten_seven.png", png, "image/png")
        record(g, "image_upload_run", round(_timed_run(at), 1),
               note="upload of handwritten_seven.png (690 KB)")
        try:
            btn = at.button(key="gradcam_button")
        except (KeyError, IndexError, ValueError):
            btn = None
        if btn is not None:
            btn.click()
            record(g, "gradcam_click_run", round(_timed_run(at), 1))
        record_stats(g, "rerun_noop_with_uploaded_image", measure(lambda: _timed_run(at), n=n),
                     note="Test tab state with an uploaded image")
        epochs = _widget(at, "slider", "epochs_cnn")
        record_stats(g, "slider_change_with_uploaded_image", _alternate(
            at, lambda: epochs.set_value(2), lambda: epochs.set_value(1), n))
    finally:
        tf.keras.datasets.mnist.load_data = real_load


# ─────────────────────────── components ───────────────────────────

def bench_components(n=5):
    import numpy as np
    import tensorflow as tf
    g = "components"

    from core import data as cdata
    from core.models import build_model

    for name in cdata.TABULAR_DATASETS:
        record_stats(g, f"prepare_{name.replace(' ', '_').lower()}",
                     measure(lambda name=name: cdata.prepare_builtin_tabular(name), n=n))

    t, raw = once_ms(tf.keras.datasets.mnist.load_data)
    record(g, "mnist_load_data_from_disk", round(t, 1), note="~/.keras/datasets/mnist.npz")
    record_stats(g, "prepare_mnist_normalise", measure(
        lambda: cdata.prepare_builtin_images("MNIST", loader=lambda: raw), n=3))

    nn_cfg = {"network_type": "Simple NN", "hidden_layers": 2, "nodes_per_layer": 16, "dropout_rate": 0.0,
              "optimizer": "Adam", "learning_rate": 0.01}
    cnn_cfg = {"network_type": "CNN", "conv_layers": 2, "base_filters": 16, "dropout_rate": 0.0,
               "optimizer": "Adam", "learning_rate": 0.001}
    record_stats(g, "build_simple_nn", measure(lambda: build_model(nn_cfg, (4,), 3), n=n))
    record_stats(g, "build_cnn", measure(lambda: build_model(cnn_cfg, (28, 28, 1), 10), n=n))

    # Training: Iris 2x16, lr 0.01, batch 16, 100 epochs (the recommended screenshot setting)
    from core.training import set_global_seed, train_model, validate_config
    cfg = validate_config({"network_type": "Simple NN", "dataset": "Iris", "epochs": 100, "learning_rate": 0.01,
                           "batch_size": 16, "hidden_layers": 2, "nodes_per_layer": 16, "seed": 42})
    set_global_seed(42)
    iris = cdata.prepare_builtin_tabular("Iris", seed=42)
    model = build_model(cfg, iris.input_shape, iris.num_classes)
    stamps = []

    class Stamp(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            stamps.append(time.perf_counter())

        def on_epoch_end(self, epoch, logs=None):
            stamps.append(time.perf_counter())
    t0 = time.perf_counter()
    outcome = train_model(model, iris, cfg, callbacks=[Stamp()])
    total = (time.perf_counter() - t0) * 1000
    record(g, "train_iris_100_epochs_total", round(total, 1),
           note=f"val acc {outcome['final']['val_accuracy']:.4f}")
    record(g, "train_iris_first_epoch", round((stamps[1] - stamps[0]) * 1000, 1), note="includes graph tracing")
    later = [(b - a) * 1000 for a, b in zip(stamps[2:-1], stamps[3:])]
    record(g, "train_iris_later_epoch_median", round(statistics.median(later), 2))

    # Inference: one sample
    x1 = iris.X_val[:1]
    record_stats(g, "predict_simple_nn_model.predict", measure(lambda: model.predict(x1, verbose=0), n=20))
    record_stats(g, "predict_simple_nn_direct_call", measure(lambda: model(x1, training=False).numpy(), n=20))

    set_global_seed(42)
    cnn = build_model(cnn_cfg, (28, 28, 1), 10)
    mn = cdata.prepare_builtin_images("MNIST", loader=lambda: raw)
    t, _ = once_ms(lambda: cnn.fit(mn.X_train[:6000], mn.y_train[:6000], epochs=1, batch_size=32, verbose=0))
    record(g, "train_cnn_6000_images_1_epoch", round(t, 1), note="plain model.fit on NumPy arrays")
    try:
        from core.training import batch_pipeline
        set_global_seed(42)
        cnn2 = build_model(cnn_cfg, (28, 28, 1), 10)
        ds, steps = batch_pipeline(mn.X_train[:6000], mn.y_train[:6000], 32, shuffle=True)
        t, _ = once_ms(lambda: cnn2.fit(ds, steps_per_epoch=steps, epochs=1, verbose=0, shuffle=False))
        record(g, "train_cnn_6000_images_1_epoch_pipeline", round(t, 1), note="the app's repeating tf.data pipeline")
    except ImportError:
        pass
    xi = mn.X_val[:1]
    record_stats(g, "predict_cnn_model.predict", measure(lambda: cnn.predict(xi, verbose=0), n=20))
    record_stats(g, "predict_cnn_direct_call", measure(lambda: cnn(xi, training=False).numpy(), n=20))
    record_stats(g, "predict_cnn_predict_on_batch", measure(lambda: cnn.predict_on_batch(xi), n=20),
                 note="what the app uses (core.training.predict_proba)")

    # Image preprocessing
    from core.image_preprocessing import open_image, preprocess_image
    for fname, style in (("handwritten_seven.png", "digit"), ("fashion_sneaker.png", "object"),
                         ("fashion_ankle_boot.png", "object")):
        raw_bytes = (ROOT / fname).read_bytes()
        record_stats(g, f"preprocess_{fname.split('.')[0]}", measure(
            lambda b=raw_bytes, s=style: preprocess_image(open_image(b), style=s), n=10))

    # Grad-CAM
    from core.gradcam import compute_gradcam
    arr, _ = preprocess_image(open_image((ROOT / "handwritten_seven.png").read_bytes()), style="digit")
    t, _ = once_ms(lambda: compute_gradcam(cnn, arr))
    record(g, "gradcam_first_call", round(t, 1))
    record_stats(g, "gradcam_warm", measure(lambda: compute_gradcam(cnn, arr), n=10))

    # Charts: old = matplotlib figure encoded like st.pyplot (200 dpi PNG);
    # new = Altair spec serialised to JSON (what st.altair_chart sends to the browser).
    def charts():
        from components import visualizations as viz
        from config.translations import UI
        labels = {k: UI["en"][k] for k in UI["en"]}
        h = outcome["history"]
        cm = outcome["confusion_matrix"]
        probs = cnn(xi, training=False).numpy()[0]
        heat, _ = compute_gradcam(cnn, arr)
        cm10 = np.eye(10, dtype=int) * 90
        names10 = [str(i) for i in range(10)]
        if hasattr(viz, "history_figure"):
            import matplotlib.pyplot as plt

            def render(make):
                fig = make()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                plt.close(fig)
            items = {
                "chart_history": lambda: render(lambda: viz.history_figure(h, labels, best_epoch=40)),
                "chart_confusion_3_classes": lambda: render(lambda: viz.confusion_figure(cm, ["a", "b", "c"], labels)),
                "chart_confusion_10_classes": lambda: render(lambda: viz.confusion_figure(cm10, names10, labels)),
                "chart_probabilities_10": lambda: render(lambda: viz.probabilities_figure(
                    names10, probs, int(probs.argmax()), labels)),
                "chart_gradcam_overlay": lambda: render(lambda: viz.gradcam_figure(arr, heat)),
            }
        else:
            def spec(chart):
                return json.dumps(chart.to_dict())
            items = {
                "chart_history": lambda: [spec(c) for c in viz.history_charts(h, labels, best_epoch=40)],
                "chart_confusion_3_classes": lambda: spec(viz.confusion_chart(cm, ["a", "b", "c"], labels)),
                "chart_confusion_10_classes": lambda: spec(viz.confusion_chart(cm10, names10, labels)),
                "chart_probabilities_10": lambda: spec(viz.probabilities_chart(names10, probs, int(probs.argmax()), labels)),
                "chart_gradcam_overlay": lambda: viz.gradcam_overlay(arr, heat),
            }
        for name, fn in items.items():
            record_stats(g, name, measure(fn, n=n))
    safe(g, "charts", charts)

    def later_sections():
        # RAG
        from core import rag
        guide = (ROOT / "deep_learning_study_guide.txt").read_bytes()
        record_stats(g, "rag_ingest_study_guide_txt", measure(lambda: rag.build_chunks([("guide.txt", guide)]), n=n))
        big = ("Backpropagation computes gradients layer by layer. " * 40 + "\n\n") * 120   # ~250 KB
        t, (chunks, _) = once_ms(lambda: rag.build_chunks([("big.txt", big.encode())]))
        record(g, "rag_ingest_250kb_txt", round(t, 1), note=f"{len(chunks)} chunks")
        report = next(iter(sorted(ROOT.glob("*.pdf"))), None)   # optional: any PDF you drop next to app.py
        if report is not None:
            pdf = report.read_bytes()
            t, (pchunks, _) = once_ms(lambda: rag.build_chunks([("report.pdf", pdf)]))
            record(g, "rag_ingest_local_pdf", round(t, 1),
                   note=f"{len(pdf) / 1e6:.1f} MB local PDF -> {len(pchunks)} chunks (content not recorded)")

        rng = np.random.default_rng(0)
        words = ["gradient", "dropout", "epoch", "neuron", "layer", "loss", "optimizer", "batch", "kernel", "pooling"]
        kb = [{"id": i, "source": "s.txt", "embeddings": {},
               "text": " ".join(rng.choice(words, 150))} for i in range(500)]
        tag = "gemini:gemini-embedding-001"
        for c in kb:
            c["embeddings"][tag] = rng.normal(size=3072).tolist()
        q = rng.normal(size=3072).tolist()
        record_stats(g, "rag_retrieve_bm25_500_chunks", measure(
            lambda: rag.retrieve("how does dropout reduce overfitting", kb, None, None, k=3), n=n))
        record_stats(g, "rag_retrieve_cosine_500x3072", measure(
            lambda: rag.retrieve("dropout", kb, tag, lambda _q: q, k=3), n=n))
        if hasattr(rag, "get_index"):          # what the app does: reuse the per-knowledge-base index
            cache = {}
            record_stats(g, "rag_retrieve_bm25_500_chunks_cached_index", measure(
                lambda: rag.retrieve("how does dropout reduce overfitting", kb, None, None, k=3,
                                     index=rag.get_index(kb, cache)), n=n))
            record_stats(g, "rag_retrieve_cosine_500x3072_cached_index", measure(
                lambda: rag.retrieve("dropout", kb, tag, lambda _q: q, k=3, index=rag.get_index(kb, cache)), n=n))

        # Prompt preparation + cached Gemini call (fake client)
        from backend import llm_api
        from config.prompts import build_agent_context, build_base_context, get_agent_level_rules, get_level_instructions
        record_stats(g, "prompt_build_chat_context", measure(
            lambda: build_base_context(get_level_instructions()["Intermediate"], "ml")
            + rag.format_context([(1.0, c) for c in kb[:3]]), n=20))
        record_stats(g, "prompt_build_agent_context", measure(
            lambda: build_agent_context(get_agent_level_rules()["Advanced"], "en"), n=20))
        calls = []

        class FakeModels:
            def generate_content(self, model, contents, config=None):
                calls.append(model)
                from types import SimpleNamespace
                return SimpleNamespace(text="cached answer")

        from types import SimpleNamespace
        llm_api.API_KEYS[:] = ["bench-key"]
        real_get_client = llm_api.get_client
        llm_api.get_client = lambda: SimpleNamespace(models=FakeModels())
        try:
            llm_api.clear_caches()
            t, _ = once_ms(lambda: llm_api.generate_text("same prompt", cache=True))
            record(g, "gemini_generate_text_miss_fake_client", round(t, 3), note="fake client: SDK call is instant")
            record_stats(g, "gemini_generate_text_cache_hit", measure(
                lambda: llm_api.generate_text("same prompt", cache=True), n=50))
            record(g, "gemini_cache_calls_made", len(calls), unit="calls", note="1 miss + 51 identical requests")
        finally:
            llm_api.get_client = real_get_client
            llm_api.API_KEYS[:] = []

        # Ollama status checks
        from backend import ollama_api
        closed = ollama_api.make_client("http://127.0.0.1:11999")
        def down():
            try:
                ollama_api.list_models(closed)
            except ollama_api.OllamaError:
                pass
        record_stats(g, "ollama_list_models_server_down", measure(down, n=3, warmup=0),
                     note="closed localhost port; Windows retries refused loopback connections")
        record_stats(g, "ollama_find_executable", measure(ollama_api.find_executable, n=20))
        up_client = ollama_api.make_client()
        try:
            ollama_api.list_models(up_client)
            record_stats(g, "ollama_list_models_server_up", measure(lambda: ollama_api.list_models(up_client), n=10))
        except ollama_api.OllamaError:
            record(g, "ollama_list_models_server_up", "n/a (no local Ollama server)", unit="")
        if hasattr(ollama_api, "server_status"):
            record_stats(g, "ollama_server_status_probe_down", measure(
                lambda: ollama_api.server_status("http://127.0.0.1:11999", refresh=True), n=3, warmup=0),
                note="first check against a stopped server (short TCP probe)")
            record_stats(g, "ollama_server_status_cached", measure(
                lambda: ollama_api.server_status("http://127.0.0.1:11999"), n=20), note="later reruns")

        from components.chat_helpers import call_ollama

        class FakeOllama:
            def chat(self, **kw):
                return {"message": {"content": "ok"}}
        from PIL import Image
        img = Image.new("RGB", (512, 512), "white")
        record_stats(g, "ollama_request_prep_with_image_fake_client", measure(
            lambda: call_ollama(FakeOllama(), "llama3.2:1b", "prompt " * 200, [img]), n=10))

        # TTS: cache hit only (a real gTTS call needs the internet)
        import tts_engine
        tts_engine._audio_cache[("en", "cached sentence")] = b"mp3"
        record_stats(g, "tts_cache_hit", measure(lambda: tts_engine.text_to_audio("cached sentence", "en"), n=50))

        # Agent tools
        import json as _json
        from agent_tools import execute_tool
        hist = _json.dumps([1.0 - i / 40 for i in range(30)])
        record_stats(g, "agent_tool_diagnose_training", measure(lambda: execute_tool(
            "diagnose_training", {"train_loss_json": hist, "val_loss_json": hist,
                                  "train_acc_json": hist, "val_acc_json": hist}), n=20))
        record_stats(g, "agent_tool_explain_concept", measure(lambda: execute_tool("explain_concept", {"topic": "dropout"}), n=20))
        t, res = once_ms(lambda: execute_tool("run_experiment", {"network_type": "Simple NN", "dataset": "Iris",
                                                                 "epochs": 30, "learning_rate": 0.01}))
        record(g, "agent_tool_run_experiment_iris_30_epochs", round(t, 1),
               note=f"val acc {res.get('results', {}).get('final_val_accuracy')}")
    safe(g, "later_sections", later_sections)


# ─────────────────────────── MCP ───────────────────────────

def bench_mcp(n=10):
    import httpx
    from mcp_client import MCPClient
    g = "mcp"
    port = 8798
    env = {**os.environ, **HERMETIC_ENV}
    proc = subprocess.Popen([sys.executable, "mcp_server.py", "--port", str(port)], cwd=str(ROOT), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 120:
            try:
                httpx.get(f"{url}/health", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.05)
        record(g, "server_boot_to_health", round((time.perf_counter() - t0) * 1000, 1))
        client = MCPClient(url)
        record_stats(g, "health_request", measure(client.check_health, n=n))
        polls = []
        t0 = time.perf_counter()
        res = client.train_remote({"network_type": "Simple NN", "dataset": "Iris", "epochs": 30,
                                   "learning_rate": 0.01}, poll_interval=0.25, on_progress=polls.append)
        total = (time.perf_counter() - t0) * 1000
        record(g, "train_iris_30_epochs_roundtrip", round(total, 1),
               note=f"status={res.get('status')}, {len(polls)} progress polls, model bytes "
                    f"{len(res.get('model_b64') or '') * 3 // 4}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


# ─────────────────────────── main ───────────────────────────

GROUPS = {"startup": bench_startup, "app": bench_app, "components": bench_components, "mcp": bench_mcp}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="where to write the JSON results")
    parser.add_argument("--only", choices=sorted(GROUPS), action="append", help="run only these groups")
    args = parser.parse_args(argv)

    import numpy
    import streamlit
    import tensorflow as tf
    meta = {
        "python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor(),
        "tensorflow": tf.__version__, "streamlit": streamlit.__version__, "numpy": numpy.__version__,
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True,
                              text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True,
                                         text=True).stdout.strip()),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(meta, indent=1))
    for name in (args.only or list(GROUPS)):
        print(f"== {name} ==", flush=True)
        safe(name, "_group", GROUPS[name])
    Path(args.out).write_text(json.dumps({"meta": meta, "results": RESULTS}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
