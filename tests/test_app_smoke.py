"""End-to-end smoke tests of the Streamlit app with AppTest (all LLM calls mocked, no network).

The main tabs are lazy (only the open tab runs), so tests open a tab by setting
`active_tab` in session state, exactly what clicking the tab does in the browser.
"""

import io
import json
import os

import numpy as np
import pandas as pd
import pytest
from google.genai import types
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from backend import llm_api
from tests.helpers import class_zip, fake_image_loader

pytestmark = pytest.mark.slow
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
TRAIN, VIZ, TUTOR, TEST = range(4)


def new_app():
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def texts(elements):
    return " ".join(str(e.value) for e in elements)


def open_tab(at, index):
    at.session_state["active_tab"] = index
    at.run(timeout=TIMEOUT)
    no_errors(at)
    return at


def train(at):
    at.button(key="train_button").click().run(timeout=TIMEOUT)
    no_errors(at)
    return at


def charts(at):
    return at.get("vega_lite_chart")


# ─────────────────────────── fakes ───────────────────────────

class FakeGemini:
    """Stands in for llm_api.generate / generate_text / embed_texts."""

    def __init__(self, reply="Here is code:\n```python\n# 1. Load the dataset\nprint('hi')\n```\nDone.",
                 agent_calls=None):
        self.reply = reply
        self.agent_calls = list(agent_calls or [])
        self.prompts = []
        self.text_prompts = []

    def generate(self, contents, model_id=None, config=None, **kw):
        self.prompts.append((contents, config))
        if config is not None and config.tools and self.agent_calls:
            name, args = self.agent_calls.pop(0)
            parts = [types.Part(function_call=types.FunctionCall(name=name, args=args))]
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text=self.reply)]))])

    def generate_text(self, prompt, model_id=None, json_mode=False, **kw):
        self.text_prompts.append(prompt)
        if json_mode and "Fact-Checker" in prompt:
            return json.dumps({"grounding_score": 90, "hallucinated_claims": [], "verdict": "Supported by the notes."})
        return "**Below the typical range.** Train for more epochs."

    @staticmethod
    def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", **kw):
        return [[1.0 if "dropout" in t.lower() else 0.1, 0.5] for t in texts]

    def last_prompt_text(self):
        contents, _ = self.prompts[-1]
        return contents[0].parts[0].text if isinstance(contents, list) else str(contents)


@pytest.fixture
def gemini(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(llm_api, "API_KEYS", ["test-key"])
    monkeypatch.setattr(llm_api, "generate", fake.generate)
    monkeypatch.setattr(llm_api, "generate_text", fake.generate_text)
    monkeypatch.setattr(llm_api, "embed_texts", fake.embed_texts)
    return fake


def seven_png():
    img = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(img).line((30, 20, 90, 20, 50, 100), fill=(0, 0, 0, 255), width=10)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def train_small_cnn(monkeypatch):
    import tensorflow as tf
    monkeypatch.setattr(tf.keras.datasets.mnist, "load_data", fake_image_loader(160, 40))
    at = new_app()
    at.selectbox(key="network_type_en").set_value("CNN").run(timeout=TIMEOUT)
    at.number_input(key="conv_layers").set_value(1)
    at.selectbox(key="base_filters").set_value(8)
    at.slider(key="epochs_cnn").set_value(5)
    return train(at)


# ─────────────────────────── layout & training ───────────────────────────

def test_app_loads_without_keys():
    at = new_app()
    assert "Deep Learning Tutor" in texts(at.title)
    assert len(at.tabs) == 4
    open_tab(at, TUTOR)
    assert "GEMINI_API_KEYS" in texts(at.warning)           # tutor explains how to enable itself


def test_only_the_open_tab_runs():
    at = new_app()
    assert at.button(key="train_button")                     # Train tab is open by default
    assert not [b for b in at.button if b.key == "chat_clear"]
    open_tab(at, TUTOR)
    assert [b for b in at.button if b.key == "chat_clear"]
    assert not [b for b in at.button if b.key == "train_button"]


def test_sensible_defaults_per_network_type():
    at = new_app()
    assert at.slider(key="epochs_nn").value == 50
    assert at.select_slider(key="learning_rate_nn").value == 0.01
    assert at.selectbox(key="batch_size_nn").value == 16
    at.selectbox(key="network_type_en").set_value("CNN").run(timeout=TIMEOUT)
    assert at.select_slider(key="learning_rate_cnn").value == 0.001
    assert at.slider(key="epochs_cnn").value == 5


def test_train_iris_and_every_tab_renders():
    at = new_app()
    at.slider(key="epochs_nn").set_value(10)
    at.select_slider(key="learning_rate_nn").set_value(0.01)
    train(at)

    metrics = {m.label: m.value for m in at.metric}
    # The headline is the accuracy on the 22 held-out test rows; its size is shown beside it.
    assert set(metrics) == {"Test accuracy (22 examples)", "Validation accuracy", "Epochs trained", "Parameters"}
    assert metrics["Epochs trained"] == "10 / 10" and metrics["Parameters"] == "403"
    final = at.session_state["training_outcome"]["final"]
    assert metrics["Test accuracy (22 examples)"] == f"{final['test_accuracy'] * 100:.1f}%"
    assert metrics["Validation accuracy"] == f"{final['val_accuracy'] * 100:.1f}%"
    assert "typical range for Iris" in texts(at.caption)
    assert "106 training · 22 validation" in texts(at.caption) and "22 test" in texts(at.caption)
    assert "too small for a separate test set" not in texts(at.warning)   # that note is for tiny datasets only

    open_tab(at, VIZ)
    assert len(charts(at)) == 3                              # loss, accuracy, confusion matrix
    assert "Most frequent mistake" in texts(at.markdown) or "No mistakes" in texts(at.markdown)
    assert "(test set, 22 examples)" in texts(at.markdown)
    assert int(at.session_state["confusion_data"]["matrix"].sum()) == 22
    # Per-class precision / recall / F1 beside the confusion matrix, on the same 22 test rows.
    assert "Precision, recall and F1 per class" in texts(at.markdown)
    table = at.dataframe[0].value
    assert list(table.columns) == ["Class", "Precision", "Recall", "F1", "Examples"]
    assert table["Class"].tolist() == ["setosa", "versicolor", "virginica"] and int(table["Examples"].sum()) == 22
    assert "Add a Gemini key or start Ollama" in texts(at.caption)   # no AI configured: no review button

    # Test & Export: model download reloads, code is leak-free and matches the config.
    open_tab(at, TEST)
    from core.models import model_from_bytes, model_to_bytes
    assert model_from_bytes(model_to_bytes(at.session_state["trained_model"])).output_shape == (None, 3)
    code = texts(at.code)
    assert "data = load_iris()" in code and "stratify=y" in code and "learning_rate=0.01" in code

    # Simple NN form uses the dataset's real feature names and the train-fitted scaler.
    ss = at.session_state
    feats = sorted((n for n in at.number_input if n.key and n.key.startswith("feat_")),
                   key=lambda n: int(n.key.rsplit("_", 1)[1]))           # 2-column layout: order by index
    assert [n.label for n in feats] == ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    at.button(key=f"fill_random_{ss['model_version']}").click().run(timeout=TIMEOUT)
    at.button(key=f"predict_{ss['model_version']}").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert "Prediction:" in texts(at.success)
    assert "true class" in texts(at.info)
    assert len(charts(at)) == 1                              # probability bars

    # The prediction and the form values survive a visit to another tab.
    open_tab(at, TRAIN)
    open_tab(at, TEST)
    assert "Prediction:" in texts(at.success)


def test_changing_config_after_training_is_flagged():
    at = new_app()
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    at.selectbox(key="dataset_nn").set_value("Wine").run(timeout=TIMEOUT)
    no_errors(at)
    assert "sidebar settings differ" in texts(at.info)
    open_tab(at, TEST)
    assert "changed after this model was trained" in texts(at.warning)
    assert "load_iris" in texts(at.code)                      # export still describes the trained model


def test_language_switch_keeps_the_model_and_the_open_tab():
    at = new_app()
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    version = at.session_state["model_version"]
    open_tab(at, VIZ)
    at.radio(key="lang_choice").set_value("Malayalam").run(timeout=TIMEOUT)
    no_errors(at)
    assert "ഡീപ് ലേണിംഗ് ട്യൂട്ടർ" in texts(at.title)
    assert at.session_state["active_tab"] == VIZ and at.session_state["model_version"] == version
    specs = json.dumps([json.loads(c.proto.spec) for c in charts(at)], ensure_ascii=False)
    assert "വാലിഡേഷൻ കൃത്യത" in specs and "യഥാർത്ഥ ക്ലാസ്" in specs          # chart labels are translated
    assert "കൺഫ്യൂഷൻ മാട്രിക്സ്" in texts(at.markdown)


def test_malayalam_ui():
    at = new_app()
    at.radio(key="lang_choice").set_value("Malayalam").run(timeout=TIMEOUT)
    no_errors(at)
    assert "ഡീപ് ലേണിംഗ് ട്യൂട്ടർ" in texts(at.title)
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    labels = [m.label for m in at.metric]
    assert "വാലിഡേഷൻ കൃത്യത" in labels and "ടെസ്റ്റ് കൃത്യത (22 ഉദാഹരണങ്ങൾ)" in labels


def test_next_step_buttons_switch_tabs():
    at = new_app()
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    at.button(key="next_test").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert at.session_state["active_tab"] == TEST and at.button(key=f"predict_{at.session_state['model_version']}")


# ─────────────────────────── CNN: prediction & Grad-CAM ───────────────────────────

def test_cnn_training_upload_and_gradcam(monkeypatch):
    at = train_small_cnn(monkeypatch)
    assert at.session_state["model_config"]["network_type"] == "CNN"
    open_tab(at, TEST)
    at.file_uploader(key=f"test_img_{at.session_state['model_version']}_0").upload(
        "seven.png", seven_png(), "image/png").run(timeout=TIMEOUT)
    no_errors(at)
    assert "Prediction:" in texts(at.success)
    assert at.session_state["gradcam_results"] == {}          # Grad-CAM only runs when asked for

    at.button(key="gradcam_button").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert len(at.session_state["gradcam_results"]) == 1
    assert "Could not generate Grad-CAM" not in texts(at.warning)
    assert "red = strong influence" in texts(at.caption) or "heatmap is empty" in texts(at.caption)
    assert "tensorflow.keras.datasets.mnist" not in texts(at.code) and "mnist.load_data()" in texts(at.code)

    # Rerunning (e.g. moving a sidebar slider) does not recompute anything.
    results_before = dict(at.session_state["test_results"])
    at.slider(key="epochs_cnn").set_value(6).run(timeout=TIMEOUT)
    assert at.session_state["test_results"].keys() == results_before.keys()

    # The tested image and its heatmap survive a tab switch; "Clear image" removes them.
    open_tab(at, VIZ)
    open_tab(at, TEST)
    assert "Prediction:" in texts(at.success) and at.session_state["test_image"]["name"] == "seven.png"
    at.button(key="clear_image").click().run(timeout=TIMEOUT)
    assert at.session_state["test_image"] is None and "Prediction:" not in texts(at.success)


def test_sample_image_button(monkeypatch):
    at = train_small_cnn(monkeypatch)
    open_tab(at, TEST)
    at.button(key="sample_handwritten_seven.png").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert "Prediction:" in texts(at.success)
    assert at.session_state["test_image"]["name"] == "handwritten_seven.png"


# ─────────────────────────── custom data ───────────────────────────

def test_custom_csv_with_categorical_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"size": rng.normal(size=60), "colour": ["red", "green", "blue"] * 20,
                       "id": [f"r{i}" for i in range(60)], "label": ["yes", "no"] * 30})
    at = new_app()
    at.file_uploader(key="custom_csv_upload").upload("pets.csv", df.to_csv(index=False).encode(), "text/csv").run(timeout=TIMEOUT)
    no_errors(at)
    target = [s for s in at.selectbox if s.key and s.key.startswith("target_col_")][0]
    assert target.value == "label"                            # defaults to the last column
    summary = texts(at.markdown)
    assert "one-hot encoded): colour" in summary and "Ignored: id" in summary
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    notes = " ".join(at.session_state["model_config"]["data_notes"])
    assert "One-hot" in notes and "id" in notes
    open_tab(at, TEST)
    assert "TARGET = 'label'" in texts(at.code)
    cat_inputs = [s for s in at.selectbox if s.key and s.key.startswith("feat_")]
    assert [s.label for s in cat_inputs] == ["colour"]
    at.button(key=f"predict_{at.session_state['model_version']}").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert "Prediction:" in texts(at.success)


def test_same_csv_uploaded_again_is_not_stale_but_changed_content_is():
    df = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) % 7, "t": ["x", "y"] * 20})
    raw = df.to_csv(index=False).encode()
    at = new_app()
    at.file_uploader(key="custom_csv_upload").upload("d.csv", raw, "text/csv").run(timeout=TIMEOUT)
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    at.file_uploader(key="custom_csv_upload").upload("d.csv", raw, "text/csv").run(timeout=TIMEOUT)   # same bytes
    assert "sidebar settings differ" not in texts(at.info)
    changed = raw.replace(b"\n3.0,3.0,y", b"\n4.0,3.0,y")                                         # same name + size
    assert len(changed) == len(raw) and changed != raw
    at.file_uploader(key="custom_csv_upload").upload("d.csv", changed, "text/csv").run(timeout=TIMEOUT)
    assert "sidebar settings differ" in texts(at.info)


def test_custom_zip_training():
    at = new_app()
    at.selectbox(key="network_type_en").set_value("CNN").run(timeout=TIMEOUT)
    at.file_uploader(key="custom_zip_upload").upload("shapes.zip", class_zip(per_class=8), "application/zip").run(timeout=TIMEOUT)
    no_errors(at)
    assert "16 images in 2 class folders" in texts(at.caption) and "**circle** 8" in texts(at.markdown)
    at.number_input(key="conv_layers").set_value(1)
    at.slider(key="epochs_cnn").set_value(5)
    train(at)
    assert at.session_state["class_names"] == ["circle", "square"]
    # 16 images -> 12 train / 2 validation / 2 test: a real but tiny test set, and the card says so.
    assert "Test accuracy (2 examples)" in [m.label for m in at.metric]
    assert "Only 2 test examples" in texts(at.caption)
    open_tab(at, TEST)
    assert "CLASS_NAMES = ['circle', 'square']" in texts(at.code)


def test_tiny_dataset_without_test_split_is_labelled_validation():
    """9 rows in 3 classes cannot fill three splits: the app falls back to train/validation
    and says clearly that the number is a validation accuracy."""
    df = pd.DataFrame({"a": np.arange(9.0), "b": np.arange(9.0) % 4, "t": ["x", "y", "z"] * 3})
    at = new_app()
    at.file_uploader(key="custom_csv_upload").upload("tiny.csv", df.to_csv(index=False).encode(), "text/csv").run(timeout=TIMEOUT)
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    labels = [m.label for m in at.metric]
    assert "Validation accuracy" in labels and not any(label.startswith("Test accuracy") for label in labels)
    assert "too small for a separate test set" in texts(at.warning)
    assert at.session_state["training_outcome"]["final"]["test_accuracy"] is None
    open_tab(at, VIZ)
    assert "(validation set, " in texts(at.markdown)
    open_tab(at, TEST)
    assert at.button(key=f"fill_random_{at.session_state['model_version']}").label == "Fill in a random validation example"


def test_bad_zip_shows_friendly_error():
    at = new_app()
    at.selectbox(key="network_type_en").set_value("CNN").run(timeout=TIMEOUT)
    at.file_uploader(key="custom_zip_upload").upload("bad.zip", b"not a zip", "application/zip").run(timeout=TIMEOUT)
    assert not at.exception and "not a valid ZIP" in texts(at.error)
    at.button(key="train_button").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert "not a valid ZIP" in texts(at.error)


def test_mcp_enabled_without_server_trains_locally():
    at = new_app()
    at.toggle(key="use_mcp_toggle").set_value(True).run(timeout=TIMEOUT)
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    assert "training runs on this computer" in texts(at.info)
    assert at.session_state["trained_model"] is not None


def test_ollama_backend_status_is_cached(monkeypatch):
    from backend import ollama_api
    ollama_api.invalidate_status()
    probes = []
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: probes.append(host) or False)
    at = new_app()
    at.radio(key="ai_backend_en").set_value("Ollama (Offline)").run(timeout=TIMEOUT)
    at.slider(key="epochs_nn").set_value(20).run(timeout=TIMEOUT)
    at.slider(key="epochs_nn").set_value(30).run(timeout=TIMEOUT)
    assert "Ollama server is not running" in texts(at.error)
    assert len(probes) == 1                                   # three reruns, one probe
    ollama_api.invalidate_status()


# ─────────────────────────── AI tutor (mocked Gemini) ───────────────────────────

def test_review_is_on_demand_and_chat_renders_code(gemini):
    at = new_app()
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    assert gemini.prompts == []                               # training never calls the LLM by itself
    open_tab(at, VIZ)
    at.button(key="review_button").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert "Below the typical range" in texts(at.markdown)    # AI Professor's review (mocked)
    test_acc = at.session_state["training_outcome"]["final"]["test_accuracy"]
    assert f"Test Acc: {test_acc:.4f} (held-out test set" in gemini.text_prompts[-1]
    at.button(key="listen_button").click().run(timeout=TIMEOUT)
    assert "gTTS needs an internet connection" in texts(at.caption)   # network is blocked in tests

    open_tab(at, TUTOR)
    at.chat_input[0].set_value("Show me code").run(timeout=TIMEOUT)
    no_errors(at)
    assert any("# 1. Load the dataset" in c.value for c in at.code)   # rendered as code, not an H1
    assert not any("# 1. Load the dataset" in m.value for m in at.markdown)


def test_quick_action_opens_the_tutor_and_asks(gemini):
    at = new_app()
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    open_tab(at, VIZ)
    at.button(key="explain_explain_cm").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert at.session_state["active_tab"] == TUTOR
    assert "confusion matrix" in gemini.last_prompt_text()
    test_acc = at.session_state["training_outcome"]["final"]["test_accuracy"]
    assert f"Test Accuracy (held-out test set, evaluated once after training): {test_acc:.4f}" in gemini.last_prompt_text()


def test_chat_history_is_not_duplicated(gemini):
    at = new_app()
    open_tab(at, TUTOR)
    at.chat_input[0].set_value("first question about dropout").run(timeout=TIMEOUT)
    at.chat_input[0].set_value("second question about pooling").run(timeout=TIMEOUT)
    no_errors(at)
    prompt = gemini.last_prompt_text()
    history = prompt.split("--- CONVERSATION HISTORY ---")[1].split("---")[0]
    assert "first question about dropout" in history and "second question" not in history
    assert prompt.count("second question about pooling") == 1
    assert at.session_state["chat_upload_nonce"] == 2          # attachments are cleared after each message


def test_tutor_malayalam_bilingual_reply(gemini):
    gemini.reply = json.dumps({"en": "Dropout helps.", "ml": "ഡ്രോപ്പ്ഔട്ട് സഹായിക്കുന്നു."})
    at = new_app()
    at.radio(key="lang_choice").set_value("Malayalam").run(timeout=TIMEOUT)
    open_tab(at, TUTOR)
    at.chat_input[0].set_value("dropout?").run(timeout=TIMEOUT)
    no_errors(at)
    assert "ഡ്രോപ്പ്ഔട്ട് സഹായിക്കുന്നു." in texts(at.markdown)
    _, config = gemini.prompts[-1]
    assert config.response_mime_type == "application/json"      # bilingual JSON requested in ml mode


def test_agent_mode_runs_tools(gemini):
    gemini.agent_calls = [("explain_concept", {"topic": "dropout"})]
    gemini.reply = "Agent answer with `explain_concept` results."
    at = new_app()
    open_tab(at, TUTOR)
    at.toggle(key="agent_toggle").set_value(True).run(timeout=TIMEOUT)
    at.chat_input[0].set_value("Explain dropout").run(timeout=TIMEOUT)
    no_errors(at)
    assert "Agent answer" in texts(at.markdown)
    assert any("Explain Concept" in s.label for s in at.status)


def test_knowledge_base_and_hallucination_check(gemini):
    notes = b"Dropout randomly disables neurons to reduce overfitting. " * 30
    at = new_app()
    at.file_uploader(key="kb_files").upload("notes.txt", notes, "text/plain").run(timeout=TIMEOUT)
    at.button(key="kb_learn").click().run(timeout=TIMEOUT)
    no_errors(at)
    assert "semantic search" in texts(at.info)
    n_chunks = len(at.session_state["knowledge_base"])
    at.button(key="kb_learn").click().run(timeout=TIMEOUT)     # learning the same file again adds nothing
    assert len(at.session_state["knowledge_base"]) == n_chunks
    assert "already in the knowledge base" in texts(at.caption)

    open_tab(at, TUTOR)
    at.toggle(key="hallucination_toggle").set_value(True).run(timeout=TIMEOUT)
    at.chat_input[0].set_value("What does dropout do?").run(timeout=TIMEOUT)
    no_errors(at)
    assert "passage(s) from your documents" in texts(at.caption)
    assert any("Document support" in e.label and "90%" in e.label for e in at.expander)
    prompt_text = gemini.last_prompt_text()
    assert "KNOWLEDGE BASE CONTENT" in prompt_text and "Dropout randomly" in prompt_text


def test_translated_choices_survive_a_language_switch():
    """Streamlit stores the displayed label of a choice; switching language must not turn
    the chosen option into a stale label (this crashed the real browser once)."""
    at = new_app()
    at.selectbox(key="network_type_en").set_value("CNN").run(timeout=TIMEOUT)
    at.radio(key="learning_level_en").set_value("Advanced").run(timeout=TIMEOUT)
    at.radio(key="lang_choice").set_value("Malayalam").run(timeout=TIMEOUT)
    no_errors(at)
    assert at.selectbox(key="network_type_ml").value == "CNN"
    assert at.radio(key="learning_level_ml").value == "Advanced"
    assert at.radio(key="ai_backend_ml").value == "Gemini (Online)"
    at.selectbox(key="network_type_ml").set_value("Simple NN").run(timeout=TIMEOUT)
    at.radio(key="lang_choice").set_value("English").run(timeout=TIMEOUT)
    no_errors(at)
    assert at.selectbox(key="network_type_en").value == "Simple NN"
    assert at.session_state["learning_level"] == "Advanced"


def test_retraining_clears_the_stale_note_in_the_same_run():
    at = new_app()
    at.slider(key="epochs_nn").set_value(10)
    train(at)
    at.slider(key="epochs_nn").set_value(12).run(timeout=TIMEOUT)
    assert "sidebar settings differ" in texts(at.info)
    train(at)
    assert "sidebar settings differ" not in texts(at.info)


def test_corrupt_test_image_gives_one_friendly_error(monkeypatch):
    at = train_small_cnn(monkeypatch)
    open_tab(at, TEST)
    at.file_uploader(key=f"test_img_{at.session_state['model_version']}_0").upload(
        "broken.png", b"this is not an image", "image/png").run(timeout=TIMEOUT)
    assert not at.exception
    assert "could not be read as an image" in texts(at.error)
    assert at.session_state["test_image"] is None and at.session_state["test_upload_nonce"] == 1


# ─────────────────────────── API usage meter ───────────────────────────

def test_api_usage_expander_shows_this_sessions_numbers():
    from backend import usage
    at = new_app()
    labels = [e.label for e in at.sidebar.expander]
    assert "API usage (this session)" in labels
    assert "No Gemini requests yet in this session." in texts(at.sidebar.caption)

    totals = usage.new_totals()
    totals.update(calls=3, failed=1, retries=1, cache_hits=2, cache_misses=1, prompt_tokens=1234, output_tokens=56,
                  seconds=4.5, calls_by_kind={"generate": 2, "embed": 1})
    at.session_state["api_usage"] = totals
    at.run(timeout=TIMEOUT)
    no_errors(at)
    shown = texts(at.sidebar.markdown)
    assert "**Requests sent:** 3 (2 generate, 1 embed) · failed: 1 · retries: 1" in shown
    assert "prompt 1,234 · output 56" in shown and "2 answered from memory" in shown and "4.5 s in total" in shown

    at.radio(key="lang_choice").set_value("Malayalam").run(timeout=TIMEOUT)
    no_errors(at)
    assert "API ഉപയോഗം (ഈ സെഷൻ)" in [e.label for e in at.sidebar.expander]
