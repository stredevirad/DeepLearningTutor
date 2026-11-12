"""
Session State Defaults
======================
Initializes all st.session_state variables with their default values.
Called at the top of every script run (only missing keys are filled in).

Dependency rules:
  * training a model replaces everything in TRAINED_MODEL_KEYS at once, so a
    model can never be paired with an older scaler, class list, prediction,
    Grad-CAM heatmap or AI review;
  * changing the language, the AI backend or the knowledge base never touches
    the trained model.
"""

import streamlit as st

# Everything that describes or was derived from the currently trained model.
TRAINED_MODEL_KEYS = {
    "history": None,
    "training_outcome": None,
    "model_config": {},
    "trained_model": None,
    "confusion_data": None,
    "model_summary_text": None,
    "model_params": None,
    "data_prep": None,
    "class_names": None,
    "val_raw": None,
    "val_labels": None,
    "narratives": {},          # (language, level) -> AI Professor review text
    "narrative_audio": {},     # (language, level) -> MP3 bytes (made only when "Listen" is clicked)
    "test_image": None,        # the image being tested: {"sha", "name", "bytes"}
    "test_results": {},        # image sha -> {"image", "array", "info", "probs"}
    "gradcam_results": {},     # image sha -> heatmap (computed only when requested)
    "feature_values": {},      # Simple NN test form values (survive tab switches)
    "nn_prediction": None,     # last Simple NN prediction: {"inputs", "probs", "actual"}
}


def init_session_state():
    """Initialize all session state variables with defaults if not already set."""
    defaults = {
        **TRAINED_MODEL_KEYS,
        "model_version": 0,
        "active_tab": 0,
        "chat_messages": [],
        "chat_upload_nonce": 0,
        "test_upload_nonce": 0,
        "knowledge_base": [],
        "kb_index_cache": {},
        "file_hashes": {},
        "agent_mode": False,
        "learning_level": "Intermediate",
        "ai_backend": "Gemini (Online)",
        "hallucination_detector": False,
        "use_mcp": False,
        "mcp_server_url": "http://localhost:8765",
        "mcp_connected": False,
        "mcp_client": None,
        "current_key_index": 0,
        "key_cooldowns": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


def reset_trained_model():
    """Forget the current model and everything derived from it (incl. Test-tab inputs)."""
    for key, value in TRAINED_MODEL_KEYS.items():
        st.session_state[key] = value.copy() if isinstance(value, dict) else value
    for key in [k for k in st.session_state.keys() if str(k).startswith(("_test_sample", "feat_"))]:
        del st.session_state[key]
    st.session_state.model_version = st.session_state.get("model_version", 0) + 1
