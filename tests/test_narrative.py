"""AI Professor's review: calibrated verdicts, language, TTS (LLM and gTTS mocked)."""

import numpy as np
import pytest

import tts_engine as T
from config import prompts as P


def iris_history(final_val=0.83, n=30):
    return {
        "loss": list(np.linspace(1.1, 0.5, n)), "val_loss": list(np.linspace(1.1, 0.52, n)),
        "accuracy": list(np.linspace(0.4, 0.85, n)), "val_accuracy": list(np.linspace(0.4, final_val, n)),
    }


CFG = {"network_type": "Simple NN", "dataset": "Iris", "optimizer": "Adam", "learning_rate": 0.001,
       "dropout_rate": 0.0, "batch_size": 32, "epochs": 30}
CM = {"matrix": np.array([[10, 0, 0], [0, 7, 3], [0, 2, 8]]), "class_names": ["setosa", "versicolor", "virginica"]}


@pytest.mark.parametrize("dataset,acc,tier", [
    ("Iris", 0.83, "below"), ("Iris", 0.967, "within"), ("MNIST", 0.975, "below"),
    ("MNIST", 0.989, "within"), ("Fashion-MNIST", 0.95, "above"),
])
def test_performance_verdict(dataset, acc, tier):
    assert T.performance_verdict(dataset, acc)[0] == tier


def test_custom_dataset_uses_majority_baseline():
    cm = {"matrix": np.array([[45, 5], [5, 5]]), "class_names": ["a", "b"]}
    assert T.performance_verdict("my.csv", 0.76, cm)[0] == "baseline"      # majority class = 83%
    assert T.performance_verdict("my.csv", 0.95, cm)[0] == "unknown"


def test_prompt_for_83_percent_iris_is_honest():
    prompt = T.build_narrative_prompt(CFG, iris_history(), CM, "Total params: 403", {
        "train_accuracy": 0.85, "val_accuracy": 0.83, "val_loss": 0.52})
    assert "BELOW the typical range for Iris" in prompt
    assert "never use \"brilliant\"" in prompt
    assert "enthusiastic" not in prompt.lower()
    assert "STILL IMPROVING" in prompt                                    # tells the LLM why it under-performs


def test_prompt_uses_evaluated_metrics_and_early_stopping_info():
    prompt = T.build_narrative_prompt(CFG, iris_history(0.9), CM, None,
                                      {"train_accuracy": 0.99, "val_accuracy": 0.97, "val_loss": 0.1},
                                      training_info={"stopped_early": True, "weights_restored": True, "best_epoch": 12})
    assert "Val Acc: 0.9700" in prompt and "best epoch (12)" in prompt and "WITHIN" in prompt
    assert "Test Acc: not available" in prompt                # no test split in these metrics


def test_prompt_quotes_the_test_accuracy_when_there_is_one():
    prompt = T.build_narrative_prompt(CFG, iris_history(0.9), CM, None, {
        "train_accuracy": 0.99, "val_accuracy": 0.97, "val_loss": 0.1, "test_loss": 0.2, "test_accuracy": 0.9545})
    assert "Test Acc: 0.9545 (held-out test set" in prompt and "Val Acc: 0.9700" in prompt
    assert "Never call a validation accuracy a test accuracy" in prompt
    none = T.build_narrative_prompt(CFG, iris_history(0.9), CM, None, {
        "train_accuracy": 0.99, "val_accuracy": 0.97, "val_loss": 0.1, "test_loss": None, "test_accuracy": None})
    assert "Test Acc: not available" in none and "0.9545" not in none


@pytest.mark.parametrize("level", ["Beginner", "Intermediate", "Advanced"])
def test_levels_change_prompt(level):
    p = T.build_narrative_prompt(CFG, iris_history(), CM, None, learning_level=level)
    assert level.lower() in p.lower()


def test_malayalam_prompt():
    assert "Malayalam" in T.build_narrative_prompt(CFG, iris_history(), CM, None, lang="ml")


def test_generate_narrative_with_fake_llm():
    seen = []
    out = T.generate_training_narrative(CFG, iris_history(), CM, None, generate_fn=lambda p: seen.append(p) or "**Needs work.**")
    assert out == "**Needs work.**" and "Iris" in seen[0]


def test_generate_narrative_failure_is_friendly():
    from backend.llm_api import LLMUnavailableError

    def boom(p):
        raise LLMUnavailableError("All Gemini API keys have hit their rate limit.")
    out = T.generate_training_narrative(CFG, iris_history(), CM, None, generate_fn=boom)
    assert out.startswith("⚠️") and "rate limit" in out and "Traceback" not in out


def test_text_to_audio_uses_language(monkeypatch):
    calls = []

    class FakeTTS:
        def __init__(self, text, lang, slow):
            calls.append((text, lang))

        def write_to_fp(self, fp):
            fp.write(b"ID3fake")

    import gtts
    monkeypatch.setattr(gtts, "gTTS", FakeTTS)
    T._audio_cache.clear()
    assert T.text_to_audio("**Good** job 🎉", lang="ml") == b"ID3fake"
    assert calls == [("Good job", "ml")]
    T.text_to_audio("**Good** job 🎉", lang="ml")
    assert len(calls) == 1                                                 # cached


def test_text_to_audio_offline_returns_none(monkeypatch):
    import gtts

    class Broken:
        def __init__(self, *a, **k):
            raise ConnectionError("offline")
    monkeypatch.setattr(gtts, "gTTS", Broken)
    T._audio_cache.clear()
    assert T.text_to_audio("hello") is None


def test_benchmarks_cover_all_builtin_datasets():
    from core.data import IMAGE_DATASETS, TABULAR_DATASETS
    assert set(P.BENCHMARKS) == set(TABULAR_DATASETS) | set(IMAGE_DATASETS)


@pytest.mark.parametrize("lang", ["en", "ml"])
def test_chat_contexts_follow_language_and_level(lang):
    for level, block in P.get_level_instructions().items():
        ctx = P.build_base_context(block, lang)
        assert level.upper() in ctx
        assert ('"ml"' in ctx) == (lang == "ml")
        assert "typical range" in ctx
    agent = P.build_agent_context(P.get_agent_level_rules()["Advanced"], lang)
    assert "ADVANCED" in agent and ('"ml"' in agent) == (lang == "ml")


def test_level_texts_differ():
    for table in (P.get_level_instructions(), P.get_agent_level_rules(), P.get_ollama_level_hints()):
        assert len(set(table.values())) == 3
