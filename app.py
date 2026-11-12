"""
Deep Learning Tutor - Streamlit UI
==================================
The sidebar holds the configuration; four tabs follow the workflow
Configure & Train -> Visualizations -> AI Tutor -> Test & Export.
All machine-learning logic lives in core/, LLM access in backend/; this file
only wires the user interface and session state together.

Responsiveness (details in docs/PERFORMANCE.md):
  * tabs are lazy: a rerun only executes the tab that is open;
  * slow or paid work (AI review, speech, Grad-CAM) runs only when the user asks,
    and every result is kept in session state for the model it belongs to, so
    later reruns just redraw it;
  * uploads are identified by a content hash, so an unchanged file is never parsed,
    decoded or embedded twice.
"""

import base64
import hashlib
import logging
import os
import sys
import time

import streamlit as st

from config.session_defaults import init_session_state, reset_trained_model
from config.translations import UI, t

st.set_page_config(page_title="Deep Learning Tutor", layout="wide", page_icon="\U0001f9e0")
init_session_state()

LANG_OPTIONS = ("English", "Malayalam")
l = "ml" if st.session_state.get("lang_choice") == "Malayalam" else "en"


def T(key, **kwargs):
    return t(l, key, **kwargs)


st.markdown("""
<style>
.block-container {padding-top: 2.4rem; padding-bottom: 3rem;}
h1 {font-size: 2.1rem !important; margin-bottom: 0 !important;}
.dlt-section {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
              opacity: 0.6; margin: 1.2rem 0 0.25rem 0;}
[data-testid="stMetric"] {border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 0.6rem; padding: 0.6rem 0.9rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem;}
[data-testid="stMetricLabel"] p {white-space: normal !important;}
</style>
""", unsafe_allow_html=True)
st.title(T("title"))
st.caption(T("subtitle"))

# Heavy imports come after the header so that the very first page load after a
# server start shows the page at once instead of a blank screen while TensorFlow
# loads. On later reruns these imports are instant (modules are cached).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # keep TensorFlow's C++ info messages out of the console
with st.spinner(T("starting_up")):
    import numpy as np
    import tensorflow as tf

    from backend import llm_api, ollama_api
    from backend.agent import build_user_parts, run_agent
    from components.chat_helpers import (
        call_ollama, parse_bilingual, render_response, run_hallucination_check,
    )
    from components.training_ui import LiveTrainingCallback
    from components.visualizations import (
        confusion_chart, enlarge, gradcam_overlay, history_charts, probabilities_chart,
        render_code_download, render_experiment_chart, render_experiment_confusion_matrix,
        render_parameter_summary, show_chart,
    )
    from config.prompts import (
        BENCHMARKS, build_agent_context, build_base_context, get_agent_level_rules,
        get_level_instructions, get_ollama_level_hints,
    )
    from core import rag
    from core.codegen import generate_training_code
    from core.data import (
        IMAGE_DATASETS, TABULAR_DATASETS, DataError, PreparedData, describe_csv, load_image_zip,
        prepare_builtin_images, prepare_builtin_tabular, prepare_custom_csv, prepare_image_arrays,
        read_csv_upload, transform_rows, zip_summary,
    )
    from core.gradcam import GradCamError, compute_gradcam
    from core.image_preprocessing import ImageError, open_image, preprocess_image, style_for_dataset
    from core.models import build_model, count_parameters, model_from_bytes, model_to_bytes, summary_text
    from core.training import (
        DEFAULTS as TRAINING_DEFAULTS, ConfigError, predict_proba, set_global_seed, train_model,
        validate_config,
    )
    from mcp_client import MCPClient
    from tts_engine import generate_training_narrative, performance_verdict, text_to_audio

logger = logging.getLogger("deep_learning_tutor")

# Per-file upload limits (MB). The global ceiling is in .streamlit/config.toml.
MAX_CSV_MB, MAX_ZIP_MB, MAX_DOC_MB, MAX_IMAGE_MB, MAX_ATTACHMENT_MB = 25, 100, 20, 10, 10
MAX_ATTACHMENT_TEXT = 8000
MAX_KEPT_TEST_IMAGES = 8          # predictions / heatmaps kept per model for quick re-display
BACKEND_GEMINI, BACKEND_OLLAMA = "Gemini (Online)", "Ollama (Offline)"
LEVELS = ["Beginner", "Intermediate", "Advanced"]
TAB_IDS = ("train", "viz", "tutor", "test")
ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGES = {   # sample files shipped with the repo, offered in the Test tab
    "MNIST": [("handwritten_seven.png", "sample_seven")],
    "Fashion-MNIST": [("fashion_sneaker.png", "sample_sneaker"), ("fashion_ankle_boot.png", "sample_boot")],
}
# Sensible starting points per network type (a tuned Iris model reaches ~97 % with these).
UI_DEFAULTS = {"Simple NN": {"epochs": 50, "learning_rate": 0.01, "batch_size": 16},
               "CNN": {"epochs": 5, "learning_rate": 0.001, "batch_size": 32}}


def _skip_streamlit_keras_reset():
    """Streamlit calls keras.backend.clear_session() after EVERY script run
    (streamlit/runtime/scriptrunner/script_runner.py: _clean_problem_modules). With
    TensorFlow loaded that costs ~0.3 s per click and throws away TensorFlow's kernel
    cache, so the next prediction is slower too. This app already clears the Keras
    session itself right before it builds a new model, so only Streamlit's other
    clean-up step (closing matplotlib figures) is kept."""
    from streamlit.runtime.scriptrunner import script_runner
    if getattr(script_runner, "_dlt_keeps_keras_state", False) or not hasattr(script_runner, "_clean_problem_modules"):
        return

    def _clean_problem_modules():
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is not None:
            try:
                plt.close("all")
            except Exception:  # never let clean-up break a rerun
                logger.debug("matplotlib clean-up failed", exc_info=True)

    script_runner._clean_problem_modules = _clean_problem_modules
    script_runner._dlt_keeps_keras_state = True


_skip_streamlit_keras_reset()


# ═══════════════════════════════════════════════
# Caches (content-addressed, so they can never mix up two users' files)
# ═══════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_mnist():
    return tf.keras.datasets.mnist.load_data()


@st.cache_resource(show_spinner=False)
def _load_fashion_mnist():
    return tf.keras.datasets.fashion_mnist.load_data()


IMAGE_LOADERS = {"MNIST": _load_mnist, "Fashion-MNIST": _load_fashion_mnist}


@st.cache_data(max_entries=4, show_spinner=False)
def _parse_csv(sha, _raw_bytes):
    return read_csv_upload(_raw_bytes)


@st.cache_data(max_entries=8, show_spinner=False)
def _describe_csv(sha, target, _df):
    return describe_csv(_df, target)


@st.cache_data(max_entries=4, show_spinner=False)
def _zip_summary(sha, _raw_bytes):
    return zip_summary(_raw_bytes)


@st.cache_data(max_entries=2, show_spinner=False)
def _decode_zip(sha, _raw_bytes):
    """Decoding every image is the slow part of a custom ZIP; retraining with other
    hyperparameters reuses the result (st.cache_data hands out copies)."""
    return load_image_zip(_raw_bytes, style="object")


@st.cache_resource(show_spinner=False)
def _ollama_client(host):
    return ollama_api.make_client(host)


def file_sha(uploaded):
    """sha256 of an uploaded file, computed once per upload (keyed by Streamlit's file_id)."""
    hashes = st.session_state.file_hashes
    if uploaded.file_id not in hashes:
        if len(hashes) > 32:
            hashes.clear()
        hashes[uploaded.file_id] = hashlib.sha256(uploaded.getvalue()).hexdigest()
    return hashes[uploaded.file_id]


# ═══════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════

def pair():
    """Two side-by-side slots for small controls. Malayalam labels are much wider, so
    in Malayalam the controls are stacked instead of squeezed into half the sidebar."""
    return st.columns(2) if l == "en" else (st.container(), st.container())


def section(title):
    st.markdown(f"<div class='dlt-section'>{title}</div>", unsafe_allow_html=True)


def translated_choice(widget, label, options, state_key, label_keys, **kwargs):
    """A radio / selectbox whose option labels are translated.

    Streamlit remembers the *displayed text* of the chosen option. That text changes
    when the language changes, and the widget would then return the old label instead
    of the option. So the widget key includes the language, and the chosen option
    itself is kept in st.session_state[state_key]."""
    current = st.session_state.get(state_key)
    if current not in options:
        current = options[0]
    value = widget(label, options, index=options.index(current), key=f"{state_key}_{l}",
                   format_func=lambda o: T(label_keys[o]), **kwargs)
    st.session_state[state_key] = value
    return value


def go_to_tab(name):
    """Button callback: open another main tab on the next run."""
    st.session_state["_goto_tab"] = TAB_IDS.index(name)


def ask_tutor(prompt):
    """Button callback: send a prepared question to the AI Tutor tab."""
    st.session_state["_prefilled_prompt"] = prompt
    go_to_tab("tutor")


def describe_config(cfg):
    if not cfg:
        return ""
    if cfg.get("network_type") == "Simple NN":
        arch = f"{cfg.get('hidden_layers')}×{cfg.get('nodes_per_layer')} dense"
    else:
        arch = f"{cfg.get('conv_layers')} conv, {cfg.get('base_filters')} filters"
    return (f"{cfg.get('network_type')} · {cfg.get('dataset')} · {arch} · "
            f"{cfg.get('optimizer')} lr={cfg.get('learning_rate')} · batch {cfg.get('batch_size')} · "
            f"{cfg.get('epochs')} epochs · dropout {cfg.get('dropout_rate')} · seed {cfg.get('seed')}")


def config_signature(cfg, custom_id=None):
    keys = ["network_type", "dataset", "epochs", "learning_rate", "batch_size", "optimizer",
            "dropout_rate", "early_stopping", "seed"]
    keys += ["hidden_layers", "nodes_per_layer"] if cfg.get("network_type") == "Simple NN" else ["conv_layers", "base_filters"]
    return repr(tuple((k, cfg.get(k)) for k in keys) + (("custom", custom_id),))


CHART_KEYS = ["train_loss", "val_loss", "train_acc", "val_acc", "epoch_axis", "loss_axis", "acc_axis",
              "plot_loss", "plot_acc", "best_epoch_line", "confusion_title", "actual", "predicted",
              "confidence_pct", "class_probs", "epoch_running", "epoch_done", "cm_count", "cm_share",
              "params_summary"]


def chart_labels():
    return {k: UI[l][k] for k in CHART_KEYS}


def fmt_pct(x):
    return f"{x * 100:.1f}%"


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
ss = st.session_state
ollama_client = None
ollama_model_name = None
ollama_embed_ready = False
custom_csv = custom_zip = target_column = None
csv_df = None

with st.sidebar:
    st.radio("Language / ഭാഷ", LANG_OPTIONS, key="lang_choice", horizontal=True, label_visibility="collapsed",
             format_func=lambda o: "English" if o == "English" else "മലയാളം")

    # --- Model & data ---
    section(T("sec_model"))
    network_type = translated_choice(st.selectbox, T("net_type"), ["Simple NN", "CNN"], "network_type",
                                     {"Simple NN": "net_simple", "CNN": "net_cnn"}, help=T("net_type_help"))
    is_nn = network_type == "Simple NN"
    datasets = list(TABULAR_DATASETS) if is_nn else list(IMAGE_DATASETS)
    dataset_name = st.selectbox(T("dataset"), datasets, key="dataset_nn" if is_nn else "dataset_cnn",
                                help=T("dataset_help_nn" if is_nn else "dataset_help_cnn"))

    has_custom = bool(ss.get("custom_csv_upload" if is_nn else "custom_zip_upload"))
    with st.expander(T("custom_data_header"), expanded=has_custom):
        if is_nn:
            custom_csv = st.file_uploader(T("upload_csv"), type=["csv"], key="custom_csv_upload",
                                          max_upload_size=MAX_CSV_MB)
            if custom_csv is not None:
                csv_sha = file_sha(custom_csv)
                try:
                    csv_df = _parse_csv(csv_sha, custom_csv.getvalue())
                    cols = list(csv_df.columns)
                    target_column = st.selectbox(T("target_column"), cols, index=len(cols) - 1,
                                                 help=T("target_column_help"), key=f"target_col_{csv_sha[:12]}")
                    info = _describe_csv(csv_sha, target_column, csv_df)
                    st.caption(T("csv_summary", rows=info["rows"], cols=info["columns"]) + " · "
                               + T("csv_classes", n=info["num_classes"]))
                    lines = []
                    if info["numeric"]:
                        lines.append(T("csv_numeric", cols=", ".join(info["numeric"])))
                    if info["categorical"]:
                        lines.append(T("csv_categorical", cols=", ".join(info["categorical"])))
                    if info["dropped"]:
                        lines.append(T("csv_dropped", cols=", ".join(f"{c} ({why})" for c, why in info["dropped"].items())))
                    removed = info["rows_without_target"] + info["rows_with_bad_numbers"]
                    if removed:
                        lines.append(T("csv_rows_removed", n=removed, usable=info["usable_rows"]))
                    st.markdown("\n".join(f"- {line}" for line in lines))
                    if info["regression_like"]:
                        st.error(T("csv_regression"))
                    elif info["num_classes"] < 2:
                        st.error(T("csv_one_class"))
                    elif info["imbalance_warning"]:
                        st.warning(info["imbalance_warning"])
                    st.success(T("custom_overrides", dataset=dataset_name))
                except DataError as e:
                    st.error(str(e))
                    custom_csv = None
        else:
            custom_zip = st.file_uploader(T("upload_zip"), type=["zip"], key="custom_zip_upload",
                                          max_upload_size=MAX_ZIP_MB)
            if custom_zip is not None:
                try:
                    zinfo = _zip_summary(file_sha(custom_zip), custom_zip.getvalue())
                    st.caption(T("zip_summary", images=zinfo["images"], classes=len(zinfo["class_counts"])))
                    st.markdown(", ".join(f"**{name}** {n}" for name, n in zinfo["class_counts"].items()))
                    for w in zinfo["warnings"]:
                        st.warning(w)
                    st.success(T("custom_overrides", dataset=dataset_name))
                except DataError as e:
                    st.error(str(e))
        st.caption(T("data_help"))

    # --- Architecture ---
    section(T("sec_architecture"))
    hidden_layers = nodes_per_layer = conv_layers = filters = None
    c1, c2 = pair()
    if is_nn:
        hidden_layers = c1.number_input(T("hidden_layers"), 1, 5, 2, help=T("hidden_layers_help"), key="hidden_layers")
        nodes_per_layer = c2.number_input(T("nodes_layer"), 2, 128, 16, step=2, help=T("nodes_layer_help"),
                                          key="nodes_per_layer")
    else:
        conv_layers = c1.number_input(T("conv_layers"), 1, 3, 2, help=T("conv_layers_help"), key="conv_layers")
        filters = c2.selectbox(T("filters"), [8, 16, 32], index=1, help=T("filters_help"), key="base_filters")

    # --- Training ---
    section(T("sec_training"))
    d = UI_DEFAULTS[network_type]
    suffix = "nn" if is_nn else "cnn"
    epochs = st.slider(T("epochs"), *((10, 200) if is_nn else (1, 50)), d["epochs"], key=f"epochs_{suffix}",
                       help=T("epochs_help" if is_nn else "epochs_help_cnn"))
    learning_rate = st.select_slider(T("learning_rate"), options=[0.0001, 0.001, 0.01, 0.1], value=d["learning_rate"],
                                     key=f"learning_rate_{suffix}", help=T("learning_rate_help"))
    c1, c2 = pair()
    batch_size = c1.selectbox(T("batch_size"), [16, 32, 64, 128], index=[16, 32, 64, 128].index(d["batch_size"]),
                              key=f"batch_size_{suffix}", help=T("batch_size_help"))
    optimizer_name = c2.selectbox(T("optimizer"), ["Adam", "SGD", "RMSprop"], key="optimizer", help=T("optimizer_help"))
    dropout_rate = st.slider(T("dropout_rate"), 0.0, 0.5, 0.0, 0.05, key="dropout_rate", help=T("dropout_rate_help"))
    use_early_stopping = st.toggle(T("early_stopping"), value=False, key="early_stopping", help=T("early_stopping_help"))
    seed = st.number_input(T("seed"), 0, 2**31 - 1, 42, step=1, key="seed", help=T("seed_help"))

    # --- AI tutor ---
    section(T("sec_ai"))
    backend_options = [BACKEND_GEMINI] + ([BACKEND_OLLAMA] if ollama_api.OLLAMA_PACKAGE_AVAILABLE else [])
    translated_choice(st.radio, T("backend_select"), backend_options, "ai_backend",
                      {BACKEND_GEMINI: "backend_gemini", BACKEND_OLLAMA: "backend_ollama"}, horizontal=True)
    translated_choice(st.radio, T("learning_level"), LEVELS, "learning_level",
                      {v: f"level_{v.lower()}" for v in LEVELS}, horizontal=True, help=T("learning_level_help"))

    with st.expander(T("backend_settings"), expanded=False):
        if ss.ai_backend == BACKEND_GEMINI:
            if not llm_api.is_configured():
                st.warning(T("no_gemini_key"))
            labels_ = list(llm_api.GEMINI_MODELS)
            chosen = ss.get("gemini_model_select", ss.get("gemini_model_label", llm_api.DEFAULT_MODEL_LABEL))
            st.selectbox(T("gemini_model"), labels_, index=labels_.index(chosen) if chosen in labels_ else 1,
                         key="gemini_model_select")
            st.selectbox(T("thinking_level"), list(llm_api.THINKING_LEVELS), key="thinking_level_select")
        else:
            requested_model = st.text_input(T("ollama_model"), value="llama3.2:1b", help=T("ollama_model_help"),
                                            key="ollama_model_input").strip()
            host = ollama_api.get_host()
            server_up, installed_models = ollama_api.server_status(host)
            if server_up:
                ollama_client = _ollama_client(host)
                if requested_model and ollama_api.has_model(installed_models, requested_model):
                    st.success(T("ollama_connected", model=requested_model))
                    ollama_model_name = requested_model
                else:
                    st.warning(T("ollama_model_missing", model=requested_model))
                    if installed_models:
                        st.caption(T("ollama_installed_models", models=", ".join(installed_models)))
                    if requested_model and st.button(T("ollama_download", model=requested_model), key="ollama_pull",
                                                     width="stretch"):
                        with st.spinner(T("ollama_downloading", model=requested_model)):
                            try:
                                ollama_client.pull(requested_model)
                                ollama_api.invalidate_status(host)
                                st.rerun()
                            except Exception as e:
                                st.error(T("ollama_download_failed", reason=ollama_api.friendly_error(e, requested_model)))
                ollama_embed_ready = ollama_api.has_model(installed_models, ollama_api.DEFAULT_EMBED_MODEL)
                if ollama_embed_ready:
                    st.caption(T("ollama_embed_ready", model=ollama_api.DEFAULT_EMBED_MODEL))
                else:
                    st.caption(T("ollama_embed_missing", model=ollama_api.DEFAULT_EMBED_MODEL))
                    if st.button(T("ollama_download", model=ollama_api.DEFAULT_EMBED_MODEL), key="ollama_pull_embed",
                                 width="stretch"):
                        with st.spinner(T("ollama_downloading", model=ollama_api.DEFAULT_EMBED_MODEL)):
                            try:
                                ollama_client.pull(ollama_api.DEFAULT_EMBED_MODEL)
                                ollama_api.invalidate_status(host)
                                st.rerun()
                            except Exception as e:
                                st.error(T("ollama_download_failed",
                                           reason=ollama_api.friendly_error(e, ollama_api.DEFAULT_EMBED_MODEL)))
                if l == "ml":
                    st.caption(T("ollama_english_only"))
            else:
                st.error(T("ollama_not_running"))
                exe = ollama_api.find_executable()
                if exe:
                    if st.button(T("ollama_start"), key="ollama_start", width="stretch"):
                        with st.spinner(T("ollama_starting")):
                            try:
                                if ollama_api.start_server(exe, ollama_api.make_client(host)):
                                    st.rerun()
                                else:
                                    st.error(T("ollama_start_failed"))
                            except ollama_api.OllamaError as e:
                                st.error(str(e))
                else:
                    st.info(T("ollama_not_installed"))
                st.markdown(T("ollama_setup_steps"))
            if st.button(T("ollama_recheck"), key="ollama_recheck", width="stretch"):
                ollama_api.invalidate_status(host)
                st.rerun()

    # Gemini model for this session (the selectbox value is already in session_state).
    chosen_label = ss.get("gemini_model_select", ss.get("gemini_model_label", llm_api.DEFAULT_MODEL_LABEL))
    if chosen_label not in llm_api.GEMINI_MODELS:
        chosen_label = llm_api.DEFAULT_MODEL_LABEL
    llm = llm_api.configure_gemini(chosen_label)
    using_ollama = bool(ss.ai_backend == BACKEND_OLLAMA and ollama_client and ollama_model_name)

    # --- Knowledge base ---
    def active_embedder():
        """(backend_tag, embed_documents_fn, embed_query_fn) for the active backend, or Nones."""
        if ss.ai_backend == BACKEND_OLLAMA:
            if ollama_client is not None and ollama_embed_ready:
                return (rag.ollama_tag(ollama_api.DEFAULT_EMBED_MODEL),
                        lambda texts: ollama_api.embed(ollama_client, texts),
                        lambda q: ollama_api.embed(ollama_client, [q])[0])
            return None, None, None
        if llm_api.is_configured():
            return (rag.GEMINI_EMBED_TAG,
                    lambda texts: llm_api.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"),
                    lambda q: llm_api.embed_texts([q], task_type="RETRIEVAL_QUERY")[0])
        return None, None, None

    def learn_documents(files):
        kb = ss.knowledge_base
        next_id = max((c["id"] for c in kb), default=-1) + 1
        new_chunks, messages = rag.build_chunks([(f.name, f.getvalue()) for f in files], start_id=next_id,
                                                known_hashes=rag.document_hashes(kb),
                                                capacity=rag.MAX_CHUNKS - len(kb))
        for level, msg in messages:
            {"error": st.error, "warning": st.warning}.get(level, st.caption)(msg)
        kb.extend(new_chunks)
        tag, embed_docs, _ = active_embedder()
        if embed_docs and kb:
            try:   # embeds new chunks AND older ones that lack this backend's vectors
                rag.add_embeddings(kb, tag, embed_docs)
            except (llm_api.LLMError, ollama_api.OllamaError) as e:
                st.warning(T("kb_embed_failed", reason=str(e)))
        if new_chunks:
            st.success(T("kb_learned", n=len(new_chunks), files=len({c['source'] for c in new_chunks})))

    with st.expander(T("kb_header"), expanded=False):
        uploaded_files = st.file_uploader(T("kb_upload"), type=["pdf", "txt"], accept_multiple_files=True,
                                          help=T("kb_upload_help"), key="kb_files", max_upload_size=MAX_DOC_MB)
        c1, c2 = st.columns(2)
        if c1.button(T("kb_learn"), key="kb_learn", width="stretch"):
            if not uploaded_files:
                st.warning(T("kb_no_files"))
            else:
                with st.spinner(T("kb_learning")):
                    learn_documents(uploaded_files)
        if c2.button(T("kb_clear"), key="kb_clear", width="stretch"):
            ss.knowledge_base = []
            ss.kb_index_cache = {}
            st.rerun()
        if ss.knowledge_base:
            tag, _, _ = active_embedder()
            semantic = bool(tag) and rag.embedding_coverage(ss.knowledge_base, tag) == 1.0
            docs = len(rag.document_hashes(ss.knowledge_base))
            st.info(T("kb_active", n=len(ss.knowledge_base), docs=docs,
                      method=T("kb_method_semantic") if semantic else T("kb_method_keyword")))
            if not semantic:
                st.caption(T("kb_keyword_hint_ollama" if ss.ai_backend == BACKEND_OLLAMA else "kb_keyword_hint"))
        elif uploaded_files:
            st.caption(T("kb_click_learn"))

    # --- Advanced: remote training over MCP ---
    with st.expander(T("mcp_header"), expanded=bool(ss.use_mcp)):
        ss.use_mcp = st.toggle(T("mcp_use"), value=ss.use_mcp, key="use_mcp_toggle", help=T("mcp_use_help"))
        if ss.use_mcp:
            ss.mcp_server_url = st.text_input(T("mcp_url_label"), value=ss.mcp_server_url, help=T("mcp_url_help"),
                                              key="mcp_url_input")
            if st.button(T("mcp_connect"), key="mcp_connect", width="stretch"):
                mcp = MCPClient(ss.mcp_server_url)
                health = mcp.check_health()
                ss.mcp_connected = bool(health["connected"])
                ss.mcp_client = mcp if ss.mcp_connected else None
                if ss.mcp_connected and health.get("server_info"):
                    info = health["server_info"]
                    st.caption(T("mcp_server_info", server=info.get("server", "?"), jobs=info.get("active_jobs", 0)))
                elif not ss.mcp_connected:
                    st.error(f"{T('mcp_disconnected')} — {health.get('error', '')}")
            if ss.mcp_connected:
                st.success(T("mcp_connected"))
            else:
                st.caption(T("mcp_disconnected"))
        st.caption(T("mcp_guide"))


# --- The configuration a click on Train would use ---
ui_config = {
    "network_type": network_type, "dataset": dataset_name, "epochs": int(epochs),
    "learning_rate": float(learning_rate), "batch_size": int(batch_size), "optimizer": optimizer_name,
    "dropout_rate": float(dropout_rate), "early_stopping": bool(use_early_stopping), "seed": int(seed),
}
if is_nn:
    ui_config.update(hidden_layers=int(hidden_layers), nodes_per_layer=int(nodes_per_layer))
else:
    ui_config.update(conv_layers=int(conv_layers), base_filters=int(filters))
custom_file = custom_csv if is_nn else custom_zip
custom_id = (file_sha(custom_file), target_column) if custom_file is not None else None
if custom_file is not None:
    ui_config.update(dataset=custom_file.name, using_custom=True)
    if target_column:
        ui_config["target_column"] = target_column
current_signature = config_signature(ui_config, custom_id)
trained_cfg = ss.model_config or {}
model_is_stale = bool(ss.trained_model is not None and trained_cfg.get("signature") != current_signature)


# ═══════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════

def prepare_training_data(cfg):
    """Load + preprocess the dataset for `cfg`. Returns (PreparedData, meta for code export)."""
    if cfg["network_type"] == "Simple NN":
        if custom_csv is not None:
            data = prepare_custom_csv(csv_df, target_column, seed=cfg["seed"])
            numeric = [c for c in data.feature_columns if c not in data.categorical_levels]
            meta = {"target_column": target_column, "numeric_columns": numeric,
                    "categorical_columns": list(data.categorical_levels), "stratified": data.stratified,
                    "file_name": custom_csv.name}
            return data, meta
        return prepare_builtin_tabular(cfg["dataset"], seed=cfg["seed"]), {}
    if custom_zip is not None:
        X, y, class_names, report = _decode_zip(file_sha(custom_zip), custom_zip.getvalue())
        data = prepare_image_arrays(X, y, class_names, seed=cfg["seed"], image_style="object")
        data.warnings = report["warnings"] + data.warnings
        meta = {"class_names": class_names, "stratified": data.stratified,
                "folder": os.path.splitext(custom_zip.name)[0]}
        return data, meta
    return prepare_builtin_images(cfg["dataset"], loader=IMAGE_LOADERS[cfg["dataset"]], seed=cfg["seed"]), {}


def store_training_results(cfg, data, meta, model, outcome):
    """Replace the current model and everything derived from it (no stale scaler/class names)."""
    reset_trained_model()
    n = len(data.y_train) + len(data.y_val)
    ss.model_config = {**cfg, "custom_meta": meta, "signature": current_signature,
                       "trained_at": time.strftime("%H:%M:%S"),
                       "data_notes": list(data.info) + list(data.warnings),
                       "data_facts": {"classes": data.num_classes, "samples": n,
                                      "train": len(data.y_train), "val": len(data.y_val),
                                      "class_counts": dict(data.class_counts) if cfg.get("using_custom") else {}}}
    ss.history = outcome["history"]
    ss.training_outcome = {k: outcome[k] for k in ("epochs_run", "best_epoch", "stopped_early", "weights_restored", "final")}
    ss.confusion_data = {"matrix": np.asarray(outcome["confusion_matrix"]), "class_names": list(data.class_names)}
    ss.model_summary_text = summary_text(model)
    ss.model_params = count_parameters(model)
    ss.trained_model = model
    ss.class_names = list(data.class_names)
    ss.data_prep = {
        "kind": data.kind,
        "preprocessor": data.preprocessor,          # fitted on the training split only
        "feature_columns": list(data.feature_columns),
        "categorical_levels": dict(data.categorical_levels),
        "feature_defaults": dict(data.feature_defaults),
        "image_style": data.image_style,
    }
    if data.kind == "tabular":
        ss.val_raw = data.X_val_raw
        ss.val_labels = np.asarray(data.y_val)


def train_remotely(cfg, live, progress):
    """Train on the MCP server. Returns True on success, False to fall back to local training."""
    client = ss.mcp_client or MCPClient(ss.mcp_server_url)

    def on_progress(status):
        hist = status.get("history") or {}
        done = len(hist.get("loss", []))
        if done:
            live.load_history(hist)
        progress.progress(min(done / cfg["epochs"], 1.0),
                          text=T("mcp_epoch", epoch=done, total=cfg["epochs"]) if done else T("mcp_training_remote"))

    result = client.train_remote({k: cfg[k] for k in TRAINING_DEFAULTS}, on_progress=on_progress)
    if result.get("status") != "completed" or not result.get("model_b64"):
        st.warning(T("mcp_remote_failed", reason=result.get("error", "unknown error")))
        return False
    res = result["result"]
    model = model_from_bytes(base64.b64decode(result["model_b64"]))
    if cfg["network_type"] == "Simple NN":
        # Same dataset + seed -> identical split and train-fitted scaler for the Test tab.
        data = prepare_builtin_tabular(cfg["dataset"], seed=cfg["seed"])
    else:
        data = PreparedData(X_train=np.empty((0, 28, 28, 1)), X_val=np.empty((0, 28, 28, 1)),
                            y_train=np.zeros(res["train_samples"], np.int64), y_val=np.zeros(res["val_samples"], np.int64),
                            class_names=res["class_names"], input_shape=(28, 28, 1), kind="image",
                            image_style=style_for_dataset(cfg["dataset"]))
    r = res["results"]
    outcome = {
        "history": res["history"], "epochs_run": res["epochs_trained"], "best_epoch": res["best_epoch"],
        "stopped_early": res["stopped_early"], "weights_restored": res["best_epoch"] != res["epochs_trained"],
        "final": {"train_loss": r["final_train_loss"], "train_accuracy": r["final_train_accuracy"],
                  "val_loss": r["final_val_loss"], "val_accuracy": r["final_val_accuracy"]},
        "confusion_matrix": res["confusion_matrix"],
    }
    store_training_results(cfg, data, {}, model, outcome)
    progress.progress(1.0, text=T("mcp_done"))
    return True


def handle_train_click():
    try:
        cfg = validate_config(ui_config, allow_custom_dataset=True)
    except ConfigError as e:
        st.error(str(e))
        return
    remote = ss.use_mcp and ss.mcp_connected and not cfg.get("using_custom")
    if ss.use_mcp and cfg.get("using_custom"):
        st.info(T("mcp_custom_local"))
    elif ss.use_mcp and not ss.mcp_connected:
        st.info(T("mcp_not_connected_local"))

    status = st.container(border=True)
    with status:
        progress = st.progress(0.0, text=T("loading_data"))
        c1, c2 = st.columns(2)
        with c1:
            loss_chart = st.empty()
        with c2:
            acc_chart = st.empty()
    live = LiveTrainingCallback(cfg["epochs"], progress, loss_chart, acc_chart, chart_labels())

    try:
        if remote:
            if train_remotely(cfg, live, progress):
                st.toast(T("success_remote"))
                return
            st.warning(T("mcp_fallback"))
        data, meta = prepare_training_data(cfg)
        progress.progress(0.0, text=T("training"))
        set_global_seed(cfg["seed"])
        tf.keras.backend.clear_session()
        model = build_model(cfg, data.input_shape, data.num_classes)
        outcome = train_model(model, data, cfg, callbacks=[live])
    except DataError as e:
        progress.empty()
        st.error(f"{T('data_error')} {e}")
        return
    except tf.errors.ResourceExhaustedError:
        progress.empty()
        st.error(T("oom"))
        return
    except Exception as e:  # show a friendly message; details go to the server log
        logger.exception("Training failed")
        progress.empty()
        st.error(T("train_failed", reason=type(e).__name__))
        return

    if outcome["stopped_early"] or outcome["weights_restored"]:
        progress.progress(1.0, text=T("stopped_early", epoch=outcome["epochs_run"], total=cfg["epochs"],
                                      best=outcome["best_epoch"]))
    else:
        progress.progress(1.0, text=T("training_done"))
    store_training_results(cfg, data, meta, model, outcome)


def render_result_card():
    """Headline numbers of the trained model, shown right after training."""
    cfg = ss.model_config
    outcome = ss.training_outcome or {}
    final = outcome.get("final") or {}
    params = ss.model_params or {}
    st.markdown(f"#### {T('result_header')}")
    st.caption(T("current_model", desc=describe_config(cfg), time=cfg.get("trained_at", "")))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(T("metrics_val_acc"), fmt_pct(final.get("val_accuracy", 0.0)), help=T("metrics_val_acc_help"))
    m2.metric(T("metrics_val_loss"), f"{final.get('val_loss', 0.0):.3f}", help=T("metrics_val_loss_help"))
    m3.metric(T("metrics_epochs"), f"{outcome.get('epochs_run', 0)} / {cfg.get('epochs')}")
    m4.metric(T("metrics_params"), f"{params.get('total', 0):,}")
    notes = []
    if outcome.get("weights_restored"):
        notes.append(T("weights_restored_note", best=outcome["best_epoch"], run=outcome["epochs_run"]))
    if cfg.get("dataset") in BENCHMARKS:
        lo, hi, _ = BENCHMARKS[cfg["dataset"]]
        tier, _ = performance_verdict(cfg["dataset"], final.get("val_accuracy", 0.0))
        notes.append(T(f"verdict_{tier}", dataset=cfg["dataset"], lo=f"{lo:.0%}", hi=f"{hi:.1%}"))
    gap = final.get("train_accuracy", 0.0) - final.get("val_accuracy", 0.0)
    if gap > 0.1:
        notes.append(T("gap_note", gap=f"{gap * 100:.0f}"))
    for note in notes:
        st.caption(note)
    if cfg.get("data_notes"):
        with st.expander(T("data_notes", n=len(cfg["data_notes"]))):
            for note in cfg["data_notes"]:
                st.markdown(f"- {note}")
    c1, c2, c3 = st.columns(3)
    c1.button(T("next_viz"), key="next_viz", on_click=go_to_tab, args=("viz",), width="stretch")
    c2.button(T("next_tutor"), key="next_tutor", on_click=go_to_tab, args=("tutor",), width="stretch")
    c3.button(T("next_test"), key="next_test", on_click=go_to_tab, args=("test",), width="stretch")


def render_train_tab():
    facts = T(f"ds_info_{dataset_name.replace(' ', '_').replace('-', '_').lower()}") \
        if not custom_file and dataset_name in TABULAR_DATASETS + IMAGE_DATASETS else ""
    if is_nn:
        arch = T("arch_nn", layers=hidden_layers, nodes=nodes_per_layer)
    else:
        arch = T("arch_cnn", layers=conv_layers, filters=filters)
    st.markdown(T("config_summary", dataset=ui_config["dataset"], arch=arch, opt=optimizer_name, lr=learning_rate,
                  epochs=epochs, batch=batch_size))
    if facts:
        st.caption(facts)
    clicked = st.button(T("btn_train"), type="primary", key="train_button")
    if clicked:
        handle_train_click()
    if ss.trained_model is not None:
        if (ss.model_config or {}).get("signature") != current_signature:     # checked after a possible retrain
            st.info(T("config_changed_info"))
        render_result_card()
    elif not clicked:
        st.caption(T("getting_started"))
    with st.expander(T("tips_header")):
        st.markdown(T("hyper_tips"))


# ═══════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════

def most_confused(cm, names):
    cm = np.asarray(cm)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.size == 0 or off.max() == 0:
        return None
    i, j = np.unravel_index(int(np.argmax(off)), off.shape)
    return names[i], names[j], int(off[i, j])


def review_language():
    """Small local models write English only, so an Ollama review (and its audio) is English."""
    return "en" if using_ollama else l


def generate_review():
    """AI Professor's review of the current model (on request, kept per language and level)."""
    lang = review_language()
    key = (lang, ss.learning_level)
    if using_ollama:
        def gen(prompt):
            resp = ollama_client.chat(model=ollama_model_name, messages=[{"role": "user", "content": prompt}])
            msg = getattr(resp, "message", None) or resp["message"]
            return getattr(msg, "content", None) or msg["content"]
    else:
        gen = None                    # default: cached Gemini call
    with st.spinner(T("narrative_generating")):
        ss.narratives[key] = generate_training_narrative(
            ss.model_config, ss.history, ss.confusion_data, ss.model_summary_text,
            final_metrics=ss.training_outcome["final"], lang=lang, learning_level=ss.learning_level,
            training_info=ss.training_outcome, generate_fn=gen)


def render_viz_tab():
    if ss.history is None:
        st.info(T("empty_viz"))
        return
    cfg = ss.model_config
    outcome = ss.training_outcome or {}
    final = outcome.get("final") or {}
    if model_is_stale:
        st.warning(T("config_changed_warning"))
    st.caption(T("viz_summary", ds=cfg.get("dataset"), net=cfg.get("network_type"),
                 acc=fmt_pct(final.get("val_accuracy", 0.0)), loss=f"{final.get('val_loss', 0.0):.3f}",
                 run=outcome.get("epochs_run"), epochs=cfg.get("epochs")))
    labels = chart_labels()
    loss, acc = history_charts(ss.history, labels, best_epoch=outcome.get("best_epoch"))
    c1, c2 = st.columns(2)
    with c1:
        show_chart(loss)
    with c2:
        show_chart(acc)
    st.caption(T("curves_help"))

    if ss.confusion_data:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**{T('confusion_title')}**")
            st.altair_chart(confusion_chart(ss.confusion_data["matrix"], ss.confusion_data["class_names"], labels),
                            width="content")
        with c2:
            st.markdown(f"##### {T('cm_read_header')}")
            st.markdown(T("cm_read"))
            worst = most_confused(ss.confusion_data["matrix"], [str(c) for c in ss.confusion_data["class_names"]])
            if worst:
                st.markdown(T("cm_worst", actual=worst[0], predicted=worst[1], n=worst[2]))
            else:
                st.markdown(T("cm_perfect"))

    with st.expander(T("model_arch")):
        params = ss.model_params or {}
        st.markdown(labels["params_summary"].format(total=f"{params.get('total', 0):,}",
                                                    trainable=f"{params.get('trainable', 0):,}",
                                                    non_trainable=f"{params.get('non_trainable', 0):,}"))
        st.code(ss.model_summary_text or "", language="text")

    # --- AI Professor's review (on request) ---
    st.markdown(f"#### {T('narrative_header')}")
    key = (review_language(), ss.learning_level)
    review = ss.narratives.get(key)
    if not (llm or using_ollama):
        st.caption(T("narrative_needs_ai"))
    elif review is None:
        st.caption(T("narrative_intro"))
        if st.button(T("narrative_button"), key="review_button"):
            generate_review()
            review = ss.narratives.get(key)
    if review:
        st.markdown(review)
        if not review.startswith("⚠️"):
            audio = ss.narrative_audio.get(key)
            if audio:
                st.audio(audio, format="audio/mp3")
            elif st.button(T("listen_button"), key="listen_button"):
                with st.spinner(T("audio_generating")):
                    audio = text_to_audio(review, lang=key[0])
                if audio:
                    ss.narrative_audio[key] = audio
                    st.audio(audio, format="audio/mp3")
                else:
                    st.caption(T("narrative_audio_error"))

    # --- Ask the tutor about these results ---
    st.markdown(f"#### {T('ask_tutor_header')}")
    net, ds = cfg.get("network_type", "model"), cfg.get("dataset", "the dataset")
    h = ss.history
    prompts = {
        "explain_cm": T("explain_cm_prompt", net=net, ds=ds),
        "why_overfit": T("why_overfit_prompt", net=net, ds=ds,
                         tr_acc=fmt_pct(final.get("train_accuracy", h["accuracy"][-1])),
                         val_acc=fmt_pct(final.get("val_accuracy", 0.0)),
                         tr_loss=f"{final.get('train_loss', h['loss'][-1]):.4f}",
                         val_loss=f"{final.get('val_loss', 0.0):.4f}"),
        "suggest_improve": T("suggest_improve_prompt", net=net, ds=ds,
                             layers=cfg.get("hidden_layers", cfg.get("conv_layers", "?")),
                             lr=cfg.get("learning_rate", 0.001), dropout=cfg.get("dropout_rate", 0),
                             opt=cfg.get("optimizer", "Adam"), epochs=cfg.get("epochs"),
                             val_acc=fmt_pct(final.get("val_accuracy", 0.0))),
    }
    cols = st.columns(3)
    for col, (key_, prompt) in zip(cols, prompts.items()):
        col.button(T(key_), key=f"explain_{key_}", on_click=ask_tutor, args=(prompt,), width="stretch")


# ═══════════════════════════════════════════════
# AI TUTOR
# ═══════════════════════════════════════════════

def read_attachments(files):
    """(PIL images, image bytes for display, extracted text, file names) from chat attachments."""
    images, image_bytes, text, names = [], [], "", []
    for f in files or []:
        names.append(f.name)
        data = f.getvalue()
        try:
            if f.type and f.type.startswith("image/"):
                images.append(open_image(data))
                image_bytes.append(data)
            else:
                pages = 50 if (f.type == "application/pdf" or f.name.lower().endswith(".pdf")) else None
                content, _ = rag.extract_text(f.name, data, **({"max_pages": pages} if pages else {}))
                text += f"\n--- Content of {f.name} ---\n{content[:MAX_ATTACHMENT_TEXT]}\n"
        except (ImageError, rag.DocumentError) as e:
            st.warning(T("attachment_unreadable", name=f.name, reason=str(e)))
    return images, image_bytes, text, names


def history_block(messages, limit, max_chars=300):
    lines = []
    for msg in messages[-limit:]:
        text = msg["content"].get("en", "") if isinstance(msg["content"], dict) else str(msg["content"])
        lines.append(f"{msg['role'].upper()}: {text[:max_chars] + '...' if len(text) > max_chars else text}")
    return "\n".join(lines)


def project_status():
    cfg = ss.model_config or {}
    if not cfg:
        return "No model has been trained yet."
    out = f"Trained model: {describe_config(cfg)}\n"
    outcome = ss.training_outcome or {}
    final = outcome.get("final")
    if final:
        out += (f"Final Training Loss: {final['train_loss']:.4f}, Final Validation Loss: {final['val_loss']:.4f}.\n"
                f"Final Training Accuracy: {final['train_accuracy']:.4f}, "
                f"Final Validation Accuracy: {final['val_accuracy']:.4f}.\n"
                f"Epochs run: {outcome.get('epochs_run')} (best epoch {outcome.get('best_epoch')}).\n")
    if ss.model_params:
        out += f"Parameters: {ss.model_params['total']:,} ({ss.model_params['trainable']:,} trainable).\n"
    return out


def answer(prompt, attachments, chat_container):
    """Build the context, call the active backend and render + store the reply."""
    images, image_bytes, attached_text, names = read_attachments(attachments)
    previous = list(ss.chat_messages)           # history BEFORE this question (it is sent separately below)
    entry = {"role": "user", "content": {"en": prompt, "ml": prompt}}
    if image_bytes:
        entry["images"] = image_bytes
    if names:
        entry["file_names"] = names
    ss.chat_messages.append(entry)
    ss.chat_upload_nonce += 1                   # attachments belong to this message only

    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
            for ib in image_bytes:
                st.image(ib, width=150)
            for fn in names:
                if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    st.caption(f"📎 {fn}")

    agent_active = ss.agent_mode and not using_ollama
    level_block = "" if agent_active else get_level_instructions().get(ss.learning_level, "")
    context = build_base_context(level_block, "en" if using_ollama else l)

    retrieved, method = [], None
    if ss.knowledge_base:
        with st.spinner(T("searching_kb")):
            tag, _, query_fn = active_embedder()
            retrieved, method = rag.retrieve(prompt, ss.knowledge_base, tag, query_fn, k=3,
                                             index=rag.get_index(ss.knowledge_base, ss.kb_index_cache))
        if retrieved:
            context += ("\n--- KNOWLEDGE BASE CONTENT (extracted from user's uploaded documents) ---\n"
                        f"{rag.format_context(retrieved)}\n")
    retrieved_texts = [c["text"] for _, c in retrieved]
    if previous:
        context += f"\n--- CONVERSATION HISTORY ---\n{history_block(previous, 6)}\n"
    context += f"\n--- CURRENT PROJECT STATUS ---\n{project_status()}"

    if agent_active:
        rules = get_agent_level_rules().get(ss.learning_level, get_agent_level_rules()["Intermediate"])
        full_prompt = f"{context}\n{build_agent_context(rules, l)}\nUser Question: {prompt}"
    else:
        full_prompt = f"{context}\n\nUser Question: {prompt}"
    if attached_text:
        full_prompt += f"\n\n--- ATTACHED FILE CONTENT ---\n{attached_text}"

    def check(answer_text):
        run_hallucination_check(answer_text, l, UI, using_ollama, sources=retrieved_texts,
                                ollama_client=ollama_client, ollama_model_name=ollama_model_name)

    def save_and_render(json_data):
        ss.chat_messages.append({"role": "assistant", "content": json_data})
        render_response(json_data.get(l) or json_data.get("en", ""))

    with chat_container:
        with st.chat_message("assistant"):
            if retrieved:
                st.caption(T("kb_used", n=len(retrieved),
                             method=T("kb_method_semantic") if method == "cosine" else T("kb_method_keyword")))
            if agent_active:
                run_agent_turn(full_prompt, images, save_and_render, check)
            elif using_ollama:
                with st.spinner(T("thinking")):
                    hint = get_ollama_level_hints().get(ss.learning_level, "")
                    ollama_prompt = (f"You are a helpful AI assistant inside a Deep Learning Tutor app. {hint} "
                                     "Answer concisely in English.\n")
                    if retrieved:
                        ollama_prompt += "\n--- REFERENCE DOCS ---\n" + rag.format_context(retrieved, max_chars=3000) + "\n"
                    if previous:
                        ollama_prompt += f"\n--- RECENT CHAT ---\n{history_block(previous, 4)}\n"
                    if attached_text:
                        ollama_prompt += f"\n--- ATTACHED FILES ---\n{attached_text[:3000]}\n"
                    ollama_prompt += f"\nUser: {prompt}"
                    json_data = parse_bilingual(call_ollama(ollama_client, ollama_model_name, ollama_prompt,
                                                            images or None))
                    save_and_render(json_data)
                    if not json_data["en"].startswith("⚠️"):
                        check(json_data.get("en", ""))
            else:
                with st.spinner(T("thinking")):
                    try:
                        model_id = llm_api.current_model_id()
                        budget = llm_api.thinking_budget(ss.get("thinking_level_select"), model_id)
                        config = llm_api.build_config(json_mode=(l == "ml"), thinking_budget=budget)
                        contents = [llm_api.types.Content(role="user", parts=build_user_parts(full_prompt, images))]
                        raw = (llm_api.generate(contents, config=config).text or "").strip()
                        if not raw:
                            st.warning(T("empty_reply"))
                        else:
                            json_data = parse_bilingual(raw)
                            save_and_render(json_data)
                            check(json_data.get("en", ""))
                    except llm_api.LLMError as e:
                        st.error(str(e))
                    except Exception:
                        logger.exception("Gemini chat failed")
                        st.error(T("gemini_failed"))


def run_agent_turn(full_prompt, images, save_and_render, check):
    """Gemini function-calling loop with live tool-call status boxes."""
    statuses = {}

    def on_iteration(i):
        if i:
            st.caption(f"🔄 {T('agent_iteration')} {i + 1}…")

    def on_tool_start(name, args):
        display = name.replace("_", " ").title()
        box = st.status(f"🔧 {T('agent_tool_call')}: **{display}**", expanded=True)
        box.write(f"{T('agent_args')}: `{args}`")
        statuses["box"], statuses["name"] = box, display

    def on_tool_end(name, args, result):
        box = statuses["box"]
        if name == "run_experiment" and result.get("status") == "success":
            r = result["results"]
            box.write(T("agent_exp_result", loss=r["final_val_loss"], acc=r["final_val_accuracy"]))
        elif result.get("status") == "error":
            box.write(T("agent_error", error=result.get("error")))
        else:
            box.write(f"✅ {T('agent_done')}")
        failed = result.get("status") == "error"
        box.update(label=f"{'⚠️' if failed else '✅'} {statuses['name']}", state="error" if failed else "complete")

    try:
        with st.spinner(T("agent_thinking")):
            out = run_agent(build_user_parts(full_prompt, images), model_id=llm_api.current_model_id(),
                            on_iteration=on_iteration, on_tool_start=on_tool_start, on_tool_end=on_tool_end)
        level = ss.learning_level
        labels = chart_labels()
        for exp in out["experiments"]:
            ecfg = exp.get("config", {})
            arch_keys = (("hidden_layers", "nodes_per_layer") if exp.get("network_type") == "Simple NN"
                         else ("conv_layers", "base_filters"))
            parts = [f"{k.replace('_', ' ').title()}: {ecfg[k]}"
                     for k in arch_keys + ("epochs", "learning_rate", "batch_size", "optimizer", "dropout_rate")
                     if ecfg.get(k) not in (None, 0, 0.0)]
            st.caption(f"📊 {exp['network_type']} · {exp.get('dataset', '?')} — {' | '.join(parts)}")
            if level in ("Intermediate", "Advanced"):
                render_experiment_chart(exp, labels=labels)
            if level == "Advanced":
                render_experiment_confusion_matrix(exp, labels=labels)
                render_parameter_summary(exp, labels=labels)
        text = out["text"]
        if text.strip():
            json_data = parse_bilingual(text)
            save_and_render(json_data)
            if level in ("Intermediate", "Advanced"):
                render_code_download(json_data.get("en", ""), T("download_script"), experiment_idx=len(out["experiments"]))
            check(json_data.get("en", ""))
        else:
            st.warning(T("agent_no_text"))
    except llm_api.LLMError as e:
        st.error(str(e))
    except Exception:
        logger.exception("Agent mode failed")
        st.error(T("agent_failed"))


def render_tutor_tab():
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="center")
    ss.agent_mode = c1.toggle(T("agent_desc"), value=ss.agent_mode, key="agent_toggle", help=T("agent_help"))
    ss.hallucination_detector = c2.toggle(T("hallucination_desc"), value=ss.hallucination_detector,
                                          key="hallucination_toggle", help=T("hallucination_help"))
    if c3.button(T("chat_clear"), key="chat_clear", width="stretch"):
        ss.chat_messages = []
        st.rerun()

    if llm or using_ollama:
        backend = (T("status_ollama", model=ollama_model_name) if using_ollama
                   else T("status_gemini", model=llm_api.current_model_id()))
        st.caption(T("tutor_status", backend=backend, level=T(f"level_{ss.learning_level.lower()}"),
                     kb=T("status_kb", n=len(ss.knowledge_base)) if ss.knowledge_base else T("status_no_kb")))
    if ss.agent_mode:
        st.caption(T("agent_tools_available") if not using_ollama else T("agent_ollama_note"))
    if ss.hallucination_detector and not ss.knowledge_base:
        st.caption(T("hallucination_no_kb"))

    if not (llm or using_ollama):
        st.warning(T("tutor_warning"))
        return
    if ss.ai_backend == BACKEND_OLLAMA and not using_ollama:
        st.caption(T("ollama_not_ready_using_gemini"))

    quick_prompts = st.empty()
    if not ss.chat_messages:
        with quick_prompts.container():
            st.markdown(f"##### {T('try_prompts')}")
            for col, i in zip(st.columns(3), (1, 2, 3)):
                col.button(T(f"qp{i}"), key=f"qp{i}", on_click=ask_tutor, args=(T(f"qp{i}_prompt"),), width="stretch")

    chat_container = st.container(height=520)
    with chat_container:
        if not ss.chat_messages:
            st.info(T("tutor_welcome"))
        for message in ss.chat_messages:
            with st.chat_message(message["role"]):
                render_response(message["content"].get(l, message["content"].get("en", "")))
                for img_data in message.get("images", []):
                    st.image(img_data, width=150)
                for fn in message.get("file_names", []):
                    if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                        st.caption(f"📎 {fn}")

    attachments = st.file_uploader(
        T("attach_files"), type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "csv", "py", "json"],
        accept_multiple_files=True, key=f"chat_file_upload_{ss.chat_upload_nonce}",
        label_visibility="collapsed", max_upload_size=MAX_ATTACHMENT_MB)
    typed = st.chat_input(T("chat_placeholder"))
    prompt = ss.pop("_prefilled_prompt", None) or typed
    if prompt:
        quick_prompts.empty()                    # the conversation has started
        answer(prompt, attachments, chat_container)


# ═══════════════════════════════════════════════
# TEST & EXPORT
# ═══════════════════════════════════════════════

def set_test_image(name, data):
    sha = hashlib.sha256(data).hexdigest()
    if not ss.test_image or ss.test_image["sha"] != sha:
        ss.test_image = {"sha": sha, "name": name, "bytes": data}


def clear_test_image():
    ss.test_image = None
    ss.test_upload_nonce += 1


def image_result(model, style):
    """Preprocess + predict the current test image once; later reruns reuse the result."""
    img = ss.test_image
    if img["sha"] not in ss.test_results:
        pil = open_image(img["bytes"])
        array, info = preprocess_image(pil, style=style)
        probs = predict_proba(model, array.reshape(1, 28, 28, 1))[0]
        while len(ss.test_results) >= MAX_KEPT_TEST_IMAGES:            # keep memory bounded
            ss.gradcam_results.pop(next(iter(ss.test_results)), None)
            ss.test_results.pop(next(iter(ss.test_results)))
        ss.test_results[img["sha"]] = {"image": pil, "array": array, "info": info, "probs": probs}
    return ss.test_results[img["sha"]]


def explain_with_gradcam(model, sha, array, class_index):
    """Button callback: runs before the next rerun, so that run already shows the heatmap."""
    try:
        heatmap, _ = compute_gradcam(model, array, class_index=class_index)
        ss.gradcam_results[sha] = heatmap
    except GradCamError as e:
        ss["_gradcam_error"] = str(e)
    except Exception:
        logger.exception("Grad-CAM failed")
        ss["_gradcam_error"] = T("unexpected_error")


def render_image_test(model, class_names, prep, cfg):
    dataset = cfg.get("dataset", "")
    style = prep.get("image_style") or style_for_dataset(dataset)
    samples = [] if cfg.get("using_custom") else SAMPLE_IMAGES.get(dataset, [])
    if samples:
        cols = st.columns([2, 3, 3, 4], vertical_alignment="center")
        cols[0].caption(T("try_sample"))
        for col, (fname, label_key) in zip(cols[1:], samples):
            if col.button(T(label_key), key=f"sample_{fname}", width="stretch"):
                with open(os.path.join(ROOT, fname), "rb") as f:
                    set_test_image(fname, f.read())
    uploaded = st.file_uploader(T("upload_image"), type=["png", "jpg", "jpeg", "bmp", "gif", "webp"],
                                key=f"test_img_{ss.model_version}_{ss.test_upload_nonce}", help=T("upload_image_help"),
                                max_upload_size=MAX_IMAGE_MB)
    if uploaded is not None:
        set_test_image(uploaded.name, uploaded.getvalue())
    if not ss.test_image:
        return

    try:
        res = image_result(model, style)
    except ImageError as e:
        st.error(str(e))
        clear_test_image()
        return
    probs = res["probs"]
    pred = int(np.argmax(probs))
    if res["info"]["blank"]:
        st.warning(T("image_blank_warning"))
    st.success(T("prediction", cls=class_names[pred], conf=f"{probs[pred] * 100:.1f}"))
    st.caption(T("probability_note"))

    sha = ss.test_image["sha"]
    heatmap = ss.gradcam_results.get(sha)
    gradcam_error = ss.pop("_gradcam_error", None)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(T("your_upload"))
        st.image(res["image"], width="stretch")
    with c2:
        st.caption(T("ai_input"))
        st.image(enlarge(res["array"]), width="stretch")
    with c3:
        st.caption(T("gradcam_title"))
        if heatmap is None:
            st.markdown(T("gradcam_intro"))
            st.button(T("gradcam_button"), key="gradcam_button", type="primary",
                      on_click=explain_with_gradcam, args=(model, sha, res["array"], pred))
            if gradcam_error:
                st.warning(T("gradcam_failed", reason=gradcam_error))
        else:
            st.image(gradcam_overlay(res["array"], heatmap), width="stretch")
    if heatmap is not None:
        st.caption(T("gradcam_empty") if heatmap.max() <= 0 else T("gradcam_caption"))
    show_chart(probabilities_chart(class_names, probs, pred, chart_labels()))
    st.button(T("clear_image"), key="clear_image", on_click=clear_test_image)


def render_tabular_test(model, class_names, prep):
    version = ss.model_version
    feature_columns = prep.get("feature_columns", [])
    categorical = prep.get("categorical_levels", {})
    defaults = prep.get("feature_defaults", {})
    values = ss.feature_values

    def key_for(i):
        return f"feat_{version}_{i}"

    if st.button(T("fill_random"), key=f"fill_random_{version}"):
        val_raw = ss.get("val_raw")
        if val_raw is not None and len(val_raw):
            idx = int(np.random.default_rng().integers(0, len(val_raw)))
            row = val_raw.iloc[idx]
            for i, col in enumerate(feature_columns):
                values[col] = str(row[col]) if col in categorical else float(row[col])
                ss[key_for(i)] = values[col]
            ss["_test_sample_label"] = int(ss.val_labels[idx])
    sample_label = ss.get("_test_sample_label")
    if sample_label is not None and 0 <= sample_label < len(class_names):
        st.info(T("sample_actual", cls=class_names[sample_label]))

    inputs = {}
    with st.expander(T("enter_features"), expanded=True):
        cols = st.columns(2)
        for i, col in enumerate(feature_columns):
            key = key_for(i)
            with cols[i % 2]:
                if col in categorical:
                    options = categorical[col] or [""]
                    if key not in ss or ss[key] not in options:
                        start = values.get(col, defaults.get(col))
                        ss[key] = start if start in options else options[0]
                    inputs[col] = st.selectbox(col, options, key=key)
                else:
                    if key not in ss:
                        ss[key] = float(values.get(col, defaults.get(col, 0.0)))
                    inputs[col] = st.number_input(col, format="%.4f", key=key)
                values[col] = inputs[col]

    if st.button(T("predict"), type="primary", key=f"predict_{version}"):
        try:
            probs = predict_proba(model, transform_rows(prep.get("preprocessor"), inputs))[0]   # TRAIN-fitted scaler
            ss.nn_prediction = {"inputs": dict(inputs), "probs": probs}
        except Exception:
            logger.exception("Prediction failed")
            st.error(T("prediction_failed"))
    result = ss.nn_prediction
    if result:
        if result["inputs"] != inputs:
            st.caption(T("inputs_changed"))
        else:
            probs = result["probs"]
            pred = int(np.argmax(probs))
            st.success(T("prediction", cls=class_names[pred], conf=f"{probs[pred] * 100:.1f}"))
            st.caption(T("probability_note"))
            show_chart(probabilities_chart(class_names, probs, pred, chart_labels()))


def render_test_tab():
    if ss.trained_model is None:
        st.info(T("empty_test"))
        return
    cfg = ss.model_config
    st.caption(T("current_model", desc=describe_config(cfg), time=cfg.get("trained_at", "")))
    if model_is_stale:
        st.warning(T("config_changed_warning"))
    st.markdown(f"#### {T('test_header')}")
    model, class_names, prep = ss.trained_model, ss.class_names, ss.data_prep or {}
    if cfg.get("network_type") == "CNN":
        render_image_test(model, class_names, prep, cfg)
    else:
        render_tabular_test(model, class_names, prep)

    st.markdown(f"#### {T('export_header')}")
    c1, c2 = st.columns(2)
    c1.download_button(T("download_model"), data=lambda m=model: model_to_bytes(m), file_name="trained_model.keras",
                       mime="application/octet-stream", key="download_model", width="stretch")
    code = generate_training_code(cfg, cfg.get("custom_meta") or {})
    c2.download_button(T("download_script"), data=code, file_name="train_model.py", mime="text/x-python",
                       key="download_script", width="stretch")
    with st.expander(T("show_code")):
        st.caption(T("code_caption"))
        st.code(code, language="python")


# ═══════════════════════════════════════════════
# MAIN AREA: four lazy tabs (only the open one runs)
# ═══════════════════════════════════════════════
tab_labels = [T(f"tab_{name}") for name in TAB_IDS]
pending_tab = ss.pop("_goto_tab", None)
if pending_tab is not None:
    ss.active_tab = pending_tab
if ss.get("main_tabs") != tab_labels[ss.active_tab]:      # after a language switch or a "go to" button
    ss["main_tabs"] = tab_labels[ss.active_tab]


def _remember_tab(labels):
    selected = st.session_state.get("main_tabs")
    if selected in labels:
        st.session_state.active_tab = labels.index(selected)


tabs = st.tabs(tab_labels, key="main_tabs", on_change=_remember_tab, args=(tab_labels,))
renderers = (render_train_tab, render_viz_tab, render_tutor_tab, render_test_tab)
for tab, render in zip(tabs, renderers):
    if tab.open:
        with tab:
            render()
