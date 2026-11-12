"""Agent tools called directly with valid and invalid arguments (no LLM involved)."""

import ast
import inspect
import json

import numpy as np
import pytest

import agent_tools as T
from tests.helpers import fake_image_loader


# ─────────────────────────── run_experiment ───────────────────────────

@pytest.mark.slow
def test_run_experiment_simple_nn():
    r = T.run_experiment(network_type="Simple NN", dataset="Iris", epochs=2, hidden_layers=1, nodes_per_layer=8)
    assert r["status"] == "success", r
    json.dumps(r)
    assert set(r["results"]) == {"final_train_loss", "final_val_loss", "final_train_accuracy", "final_val_accuracy"}
    assert len(r["history"]["loss"]) == r["epochs_trained"] == 2
    assert np.array(r["confusion_matrix"]).shape == (3, 3)


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["Wine", "Breast Cancer", "Digits"])
def test_run_experiment_uses_the_requested_dataset(dataset):
    """Previously anything but Iris/Wine silently trained on make_moons."""
    r = T.run_experiment(dataset=dataset, epochs=1)
    assert r["status"] == "success" and r["dataset"] == dataset
    expected = {"Wine": 3, "Breast Cancer": 2, "Digits": 10}[dataset]
    assert r["num_classes"] == expected and np.array(r["confusion_matrix"]).shape == (expected, expected)


@pytest.mark.slow
def test_run_experiment_cnn_uses_subset(monkeypatch):
    import tensorflow as tf
    monkeypatch.setattr(tf.keras.datasets.mnist, "load_data", fake_image_loader(n_train=300, n_val=60))
    monkeypatch.setattr(T, "AGENT_CNN_MAX_TRAIN", 120)
    r = T.run_experiment(network_type="CNN", dataset="MNIST", epochs=1, conv_layers=1, base_filters=4)
    assert r["status"] == "success" and r["train_samples"] == 120


@pytest.mark.parametrize("kwargs,fragment", [
    ({"network_type": "RNN"}, "network_type"),
    ({"dataset": "Moons"}, "dataset"),
    ({"network_type": "CNN", "dataset": "Iris"}, "dataset"),
    ({"hidden_layers": 1000}, "hidden_layers"),
    ({"epochs": 5000}, "epochs"),
    ({"epochs": "many"}, "epochs"),
    ({"optimizer": "Nadam"}, "optimizer"),
    ({"learning_rate": -1}, "learning_rate"),
    ({"batch_size": 0}, "batch_size"),
    ({"conv_layers": 9, "network_type": "CNN", "dataset": "MNIST"}, "conv_layers"),
])
def test_run_experiment_invalid_args_return_errors(kwargs, fragment):
    r = T.run_experiment(**kwargs)
    assert r["status"] == "error" and fragment in r["error"]


# ─────────────────────────── other tools ───────────────────────────

def test_generate_code_valid_and_teaches_no_leakage():
    r = T.generate_code("Add dropout to a CNN")
    assert r["status"] == "ready" and "BEFORE any scaling" in r["instruction"] and "stratify" in r["instruction"]
    assert T.generate_code("x", framework="jax")["framework"] == "tensorflow"


def test_generate_code_empty_task():
    assert T.generate_code("")["status"] == "error"


def test_explain_concept():
    assert T.explain_concept("backpropagation", "advanced")["depth"] == "advanced"
    assert T.explain_concept("dropout", "expert")["depth"] == "beginner"      # unknown depth falls back
    assert T.explain_concept("   ")["status"] == "error"


def _exp(acc, lr):
    return {"config": {"learning_rate": lr, "optimizer": "Adam"}, "results": {"final_val_accuracy": acc, "final_val_loss": 1 - acc}}


def test_compare_experiments():
    r = T.compare_experiments(json.dumps([_exp(0.80, 0.001), _exp(0.93, 0.01), _exp(0.70, 0.1)]))
    assert r["status"] == "success" and r["best_experiment"]["index"] == 1 and r["worst_experiment"]["index"] == 2
    assert r["accuracy_spread"] == pytest.approx(0.23)
    assert [x["rank"] for x in r["ranking"]] == [1, 2, 3]


@pytest.mark.parametrize("bad", ["not json", "[]", "{}", json.dumps([1, 2]), None])
def test_compare_experiments_invalid(bad):
    assert T.compare_experiments(bad)["status"] == "error"


def test_diagnose_training_overfitting():
    n = 10
    r = T.diagnose_training(json.dumps(list(np.linspace(1, 0.05, n))), json.dumps(list(np.linspace(0.5, 0.9, n))),
                            json.dumps([0.99] * n), json.dumps([0.70] * n))
    assert r["status"] == "success"
    assert any(d["issue"] == "Overfitting" for d in r["diagnoses"]) and r["overall_health"] == "needs_attention"


def test_diagnose_training_healthy():
    down = json.dumps([0.9, 0.5, 0.3, 0.2, 0.15, 0.1])
    r = T.diagnose_training(down, down, json.dumps([0.6, 0.8, 0.9, 0.95, 0.97, 0.98]), json.dumps([0.6, 0.79, 0.88, 0.93, 0.96, 0.97]))
    assert any(d["issue"] == "Good Convergence" for d in r["diagnoses"])


@pytest.mark.parametrize("args", [
    ("oops", "[1]", "[1]", "[1]"),
    ("[1, 2]", "[1, 2]", "[1, 2]", "[1, 2]"),                 # too short
    ("[1, 2, 3]", "[1, 2]", "[1, 2, 3]", "[1, 2, 3]"),        # mismatched lengths
    ('["a", "b", "c"]', "[1, 2, 3]", "[1, 2, 3]", "[1, 2, 3]"),
])
def test_diagnose_training_invalid(args):
    assert T.diagnose_training(*args)["status"] == "error"


# ─────────────────────────── execute_tool ───────────────────────────

def test_execute_tool_unknown_name():
    r = T.execute_tool("delete_everything", {})
    assert r["status"] == "error" and "Unknown tool" in r["error"]


def test_execute_tool_unknown_argument():
    r = T.execute_tool("explain_concept", {"topic": "CNN", "rm_rf": True})
    assert r["status"] == "error" and "rm_rf" in r["error"]


def test_execute_tool_bad_arg_container():
    assert T.execute_tool("explain_concept", ["CNN"])["status"] == "error"
    assert T.execute_tool("explain_concept", None)["status"] == "error"          # missing required arg


def test_execute_tool_catches_tool_crash(monkeypatch):
    def boom(topic: str):
        raise RuntimeError("kaboom")
    monkeypatch.setitem(T.TOOL_FUNCTIONS, "explain_concept", boom)
    r = T.execute_tool("explain_concept", {"topic": "x"})
    assert r["status"] == "error" and "kaboom" in r["error"]


def test_execute_tool_dispatches():
    assert T.execute_tool("explain_concept", {"topic": "pooling"})["status"] == "ready"


# ─────────────────────────── declarations ───────────────────────────

def test_declarations_match_functions():
    assert [d["name"] for d in T.TOOL_DECLARATIONS] == [f.__name__ for f in T.AGENT_TOOLS]
    for decl in T.TOOL_DECLARATIONS:
        params = inspect.signature(T.TOOL_FUNCTIONS[decl["name"]]).parameters
        assert set(decl["parameters"]["properties"]) <= set(params)
        assert set(decl["parameters"]["required"]) <= set(decl["parameters"]["properties"])


def test_generate_code_instruction_is_plain_text():
    # The instruction is fed to the LLM; make sure it is a normal string literal-safe text.
    ast.parse(repr(T.generate_code("CNN")["instruction"]))


def test_identical_experiment_reuses_the_earlier_result(monkeypatch):
    calls = []
    real = T._run_experiment_core

    def counting(config, **kw):
        calls.append(config)
        return real(config, **kw)
    monkeypatch.setattr(T, "_run_experiment_core", counting)
    monkeypatch.setattr(T, "_EXPERIMENT_CACHE", T.OrderedDict())
    first = T.run_experiment(network_type="Simple NN", dataset="Wine", epochs=2)
    again = T.run_experiment(network_type="Simple NN", dataset="Wine", epochs=2.0)   # same after validation
    other = T.run_experiment(network_type="Simple NN", dataset="Wine", epochs=3)
    assert len(calls) == 2
    assert again["results"] == first["results"] and "reused" in again["notes"][-1]
    assert other["epochs_trained"] == 3
    again["results"]["final_val_accuracy"] = -1                                          # callers get copies
    assert T.run_experiment(network_type="Simple NN", dataset="Wine", epochs=2)["results"] == first["results"]


def test_comparison_is_worded_as_these_runs_only():
    r = T.compare_experiments(json.dumps([_exp(0.80, 0.001), _exp(0.93, 0.01)]))
    assert "best validation result in these runs" in r["note"] and "noise" in r["note"]
    assert "Best validation result in these runs" in r["recommendation"]
    close = T.compare_experiments(json.dumps([_exp(0.93, 0.001), _exp(0.94, 0.01)]))
    assert "does not separate them" in close["recommendation"]
    assert all("val_loss" in entry and "parameters" in entry for entry in r["ranking"])


def test_experiment_config_only_lists_settings_of_that_network():
    r = T.run_experiment(network_type="Simple NN", dataset="Iris", epochs=2)
    assert "conv_layers" not in r["config"] and "hidden_layers" in r["config"]
