"""
TTS Engine — Vivid Training Narratives + Audio Narration
=========================================================
Generates unique, context-aware critiques of model training runs
using the Gemini LLM, then converts them to playable MP3 audio
using Google Text-to-Speech (gTTS).

The critique is calibrated: the verdict (below / within / above the typical
accuracy for the dataset) is computed here in Python and handed to the LLM,
so an 83% Iris model is never praised as "exceptional".
"""

import io
import re
import threading
from collections import OrderedDict

from config.prompts import BENCHMARKS


def _build_confusion_report(confusion_data):
    """Extract human-readable misclassification patterns from a confusion matrix."""
    if not confusion_data:
        return "No confusion matrix available."
    cm = confusion_data["matrix"]
    classes = confusion_data["class_names"]
    lines = []
    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes):
            if i != j and cm[i, j] > 0:
                lines.append((cm[i, j], f"- Misclassified '{class_i}' as '{class_j}' {cm[i, j]} times"))
    # Keep only top 5 worst misclassifications to save tokens
    lines.sort(key=lambda x: x[0], reverse=True)
    top_lines = [l[1] for l in lines[:5]]
    return "\n".join(top_lines) if top_lines else "- Perfect classification — no misclassifications."


def _detect_training_patterns(history):
    """Analyze epoch-by-epoch metrics to identify patterns like overfitting, underfitting, etc."""
    if not history:
        return "No training history available."

    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_accuracy", [])

    patterns = []

    if len(train_loss) >= 5 and len(val_loss) >= 5:
        # Overfitting: train loss keeps dropping but val loss rises
        last_5_train = train_loss[-5:]
        last_5_val = val_loss[-5:]
        if last_5_train[-1] < last_5_train[0] and last_5_val[-1] > last_5_val[0]:
            patterns.append("OVERFITTING detected: training loss decreased but validation loss increased in the last 5 epochs.")

        # Underfitting: both losses remain high
        if train_loss[-1] > 0.5 and val_loss[-1] > 0.5:
            patterns.append("UNDERFITTING likely: both training and validation loss remain high (>0.5).")

        # Good convergence
        if abs(train_loss[-1] - val_loss[-1]) < 0.05 and val_loss[-1] < 0.3:
            patterns.append("GOOD CONVERGENCE: train and validation losses are close and low.")

        # Still improving at the end -> more epochs (or a higher LR) would help
        if val_loss[-1] < min(val_loss[:-1]) and train_loss[-1] < train_loss[-3]:
            patterns.append("STILL IMPROVING: validation loss was still falling at the last epoch - training more epochs or a higher learning rate would likely help.")

    if len(val_acc) >= 2:
        # Plateau detection
        last_half = val_acc[len(val_acc)//2:]
        if max(last_half) - min(last_half) < 0.01:
            patterns.append("PLATEAU detected: validation accuracy barely changed in the second half of training.")

    return "\n".join(patterns) if patterns else "No strong patterns detected — the model trained normally."


def majority_baseline(confusion_data):
    """Accuracy of always predicting the most common validation class."""
    if not confusion_data:
        return None
    cm = confusion_data["matrix"]
    total = float(cm.sum())
    return float(cm.sum(axis=1).max()) / total if total else None


def performance_verdict(dataset, val_accuracy, confusion_data=None):
    """Return (tier, sentence) comparing the result with the typical range for this dataset.
    tier is one of: "below", "within", "above", "baseline", "unknown"."""
    if dataset in BENCHMARKS:
        lo, hi, _ = BENCHMARKS[dataset]
        if val_accuracy < lo:
            return "below", (f"BELOW the typical range for {dataset} ({lo:.0%}-{hi:.1%}) by "
                             f"{(lo - val_accuracy) * 100:.1f} percentage points. The model is under-performing; "
                             "the critique must say so plainly and focus on how to fix it.")
        if val_accuracy > hi:
            return "above", f"ABOVE the typical range for {dataset} ({lo:.0%}-{hi:.1%}). Excellent result."
        return "within", f"WITHIN the typical range for {dataset} ({lo:.0%}-{hi:.1%}). A solid, expected result."
    base = majority_baseline(confusion_data)
    if base is not None:
        if val_accuracy <= base + 0.02:
            return "baseline", (f"Custom dataset: validation accuracy is no better than always guessing the most "
                                f"common class ({base:.0%}). The model has not learned a useful pattern yet.")
        return "unknown", (f"Custom dataset: no standard benchmark. Always guessing the most common class would "
                           f"score {base:.0%}, so the model beats that baseline by {(val_accuracy - base) * 100:.1f} points.")
    return "unknown", "No benchmark available for this dataset."


def build_narrative_prompt(config, history, confusion_data, model_summary, final_metrics=None,
                           lang="en", learning_level="Intermediate", training_info=None):
    """Build the professor-critique prompt (pure function, easy to test)."""
    confusion_report = _build_confusion_report(confusion_data)
    detected_patterns = _detect_training_patterns(history)

    # Compress the history to save tokens (sample every N epochs)
    def _compress_list(lst, max_points=10):
        if len(lst) <= max_points:
            return [round(float(x), 3) for x in lst]
        step = max(1, len(lst) // max_points)
        return [round(float(lst[i]), 3) for i in range(0, len(lst), step)]

    # Extract just the parameter count line from the model summary (saves ~400 tokens)
    _summary_compact = "Not available"
    if model_summary:
        summary_lines = model_summary.strip().split("\n")
        param_lines = [l for l in summary_lines if "param" in l.lower() or "total" in l.lower()]
        _summary_compact = "\n".join(param_lines[-3:]) if param_lines else summary_lines[-1]

    fm = final_metrics or {
        "train_accuracy": float(history.get("accuracy", [0])[-1]),
        "val_accuracy": float(history.get("val_accuracy", [0])[-1]),
        "val_loss": float(history.get("val_loss", [0])[-1]),
    }
    dataset = config.get("dataset", "?")
    tier, verdict_sentence = performance_verdict(dataset, fm["val_accuracy"], confusion_data)
    gap = fm["train_accuracy"] - fm["val_accuracy"]
    # The held-out test split (None when the data was too small for one): evaluated once, after training.
    if fm.get("test_accuracy") is not None:
        test_line = (f"Test Acc: {fm['test_accuracy']:.4f} (held-out test set, never used for training or early "
                     "stopping; this is the number to quote for how good the model is)")
    else:
        test_line = "Test Acc: not available (the dataset was too small for a separate test set)"
    info = training_info or {}
    epochs_line = f"Epochs run: {len(history.get('loss', []))} of {config.get('epochs', '?')}"
    if info.get("stopped_early"):
        epochs_line += " (early stopping ended training)"
    if info.get("weights_restored"):
        epochs_line += f"; weights restored from the best epoch ({info.get('best_epoch')})"

    level_style = {
        "Beginner": "The student is a beginner: use plain language and one everyday analogy, no formulas.",
        "Intermediate": "The student is intermediate: be technical but concise.",
        "Advanced": "The student is advanced: be precise and technical; mention optimisation dynamics.",
    }.get(learning_level, "")
    language_rule = ("Write the critique in Malayalam (keep technical terms such as epoch, learning rate, "
                     "validation accuracy in English)." if lang == "ml" else "Write in English.")

    return f"""You are a warm but honest Deep Learning Professor reviewing a student's training run.
Be specific and educational. Keep it under 200 words. Use markdown. {level_style} {language_rule}

═══ SESSION DATA ═══
Network: {config.get('network_type', '?')} | Dataset: {dataset}
Optimizer: {config.get('optimizer', 'Adam')} (LR={config.get('learning_rate', 0.001)}) | Dropout: {config.get('dropout_rate', 0.0)} | Batch: {config.get('batch_size', 32)}
Params: {_summary_compact}
{epochs_line}

Metrics (sampled): Loss {_compress_list(history.get('loss', []))} | Val Loss {_compress_list(history.get('val_loss', []))}
Acc {_compress_list(history.get('accuracy', []))} | Val Acc {_compress_list(history.get('val_accuracy', []))}

Patterns: {detected_patterns}
Confusion: {confusion_report}

Final (evaluated on the weights the model ended with) — Train Acc: {fm['train_accuracy']:.4f}, Val Acc: {fm['val_accuracy']:.4f}, Val Loss: {fm['val_loss']:.4f}, Train-Val gap: {gap * 100:.1f} points
{test_line}

═══ PERFORMANCE VERDICT (computed, you must follow it) ═══
{verdict_sentence}

Rules:
- Your opening bold verdict MUST match the performance verdict above. Only use words like "excellent" or "exceptional" if the verdict is ABOVE the typical range; never use "brilliant".
- Analyze the SPECIFIC numbers and trends. If overfitting, explain WHY for this architecture. If classes are confused, hypothesize WHY.
- Be encouraging: mention one thing that went well, then give ONE concrete, actionable recommendation (e.g. a specific learning rate or epoch count).
- When a test accuracy is given, state it as the model's result and say it is a test accuracy; use the validation numbers to discuss training behaviour (overfitting, early stopping). Never call a validation accuracy a test accuracy.
- No code blocks.

Write your critique now. Start with a bold verdict."""


def generate_training_narrative(config, history, confusion_data, model_summary, final_metrics=None,
                                lang="en", learning_level="Intermediate", training_info=None, generate_fn=None):
    """
    Send the full training context to Gemini and get back a unique, calibrated,
    professor-style critique of the training session.

    `generate_fn(prompt) -> str` defaults to a cached, low-retry Gemini call.

    Returns:
        str: Markdown-formatted narrative text, or a message starting with "⚠️".
    """
    from backend import llm_api

    prompt = build_narrative_prompt(config, history, confusion_data, model_summary, final_metrics,
                                    lang, learning_level, training_info)
    if generate_fn is None:
        def generate_fn(p):
            # Few retries: the narrative is a nice-to-have and must not burn through API keys.
            return llm_api.generate_text(p, cache=True, max_attempts=2)
    try:
        text = generate_fn(prompt)
    except Exception as e:  # never break training because of the narrative
        return f"⚠️ Could not generate the training review: {llm_api.friendly_error(e)}"
    text = (text or "").strip()
    return text or "⚠️ Could not generate a training review — the AI returned an empty response."


_audio_cache = OrderedDict()
_audio_lock = threading.Lock()


def clean_for_speech(text):
    clean = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)             # links -> link text
    clean = re.sub(r'[*#_`>|~]', '', clean)                        # markdown characters
    clean = re.sub(r'[\U0001F000-\U0001FAFF☀-➿️]', '', clean)  # emoji
    clean = re.sub(r'\n{2,}', '. ', clean)                         # paragraph breaks -> pauses
    clean = re.sub(r'\n', ' ', clean)                              # single newlines -> spaces
    return re.sub(r'\s{2,}', ' ', clean).strip()


def text_to_audio(text, lang="en"):
    """
    Convert text to MP3 audio bytes using Google Text-to-Speech.

    Args:
        text: The narrative text to convert.
        lang: "en" or "ml" (gTTS supports Malayalam).

    Returns:
        bytes: MP3 audio data, or None if TTS fails.
    """
    try:
        from gtts import gTTS
    except ImportError:
        return None

    clean = clean_for_speech(text or "")
    if not clean:
        return None
    lang = lang if lang in ("en", "ml") else "en"

    key = (lang, clean)
    with _audio_lock:
        if key in _audio_cache:
            return _audio_cache[key]
    try:
        tts = gTTS(text=clean, lang=lang, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        data = audio_fp.read()
    except Exception:  # gTTS needs internet; a missing voice is not an error for the user
        return None
    with _audio_lock:
        _audio_cache[key] = data
        while len(_audio_cache) > 32:
            _audio_cache.popitem(last=False)
    return data
