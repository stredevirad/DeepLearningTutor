"""Training: config validation, seeding, leakage-free pipeline, early-stopping bookkeeping."""

import json

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from core import data as D
from core.models import build_model
from core.training import (ConfigError, compact_history, run_experiment, set_global_seed,
                           train_model, validate_config)
from tests.helpers import fake_image_loader

pytestmark = pytest.mark.slow


# ─────────────────────────── validate_config ───────────────────────────

def test_defaults_and_coercion():
    cfg = validate_config({"hidden_layers": "3", "epochs": 2.0, "learning_rate": "0.01", "early_stopping": "true"})
    assert cfg["hidden_layers"] == 3 and cfg["epochs"] == 2 and cfg["learning_rate"] == 0.01
    assert cfg["early_stopping"] is True and cfg["dataset"] == "Iris" and cfg["seed"] == 42


@pytest.mark.parametrize("bad", [
    {"network_type": "RNN"},
    {"network_type": "CNN", "dataset": "Iris"},
    {"dataset": "Moons"},
    {"optimizer": "Nadam"},
    {"epochs": 10 ** 9},
    {"hidden_layers": 0},
    {"nodes_per_layer": 2.5},
    {"batch_size": "lots"},
    {"learning_rate": 5},
    {"dropout_rate": 0.99},
    {"evil": "__import__('os')"},
])
def test_invalid_configs_rejected(bad):
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_non_dict_config():
    with pytest.raises(ConfigError):
        validate_config(["Iris"])


# ─────────────────────────── leakage & training ───────────────────────────

def test_simple_nn_scaler_fitted_on_train_only(monkeypatch):
    fitted_rows = []
    real_fit = StandardScaler.fit

    def spy_fit(self, X, y=None, **kw):
        fitted_rows.append(len(X))
        return real_fit(self, X, y, **kw)

    monkeypatch.setattr(StandardScaler, "fit", spy_fit)
    cfg = validate_config({"dataset": "Iris", "epochs": 2, "seed": 1})
    set_global_seed(cfg["seed"])
    data = D.prepare_builtin_tabular("Iris", seed=1)
    outcome = train_model(build_model(cfg, data.input_shape, data.num_classes), data, cfg)
    assert fitted_rows == [120]                 # 80% of 150 rows, never the full 150
    assert outcome["epochs_run"] == 2
    assert outcome["confusion_matrix"].shape == (3, 3)


def test_cnn_short_training(synthetic_images):
    X, y, _ = synthetic_images
    data = D.prepare_image_arrays(X, y, ["top-left", "bottom-right"])
    cfg = validate_config({"network_type": "CNN", "dataset": "MNIST", "conv_layers": 2, "base_filters": 8,
                           "epochs": 2, "batch_size": 16, "learning_rate": 0.01})
    set_global_seed(0)
    outcome = train_model(build_model(cfg, data.input_shape, 2), data, cfg)
    assert outcome["epochs_run"] == 2 and set(outcome["history"]) >= {"loss", "val_loss", "accuracy", "val_accuracy"}
    assert 0.0 <= outcome["final"]["val_accuracy"] <= 1.0


def test_final_metrics_describe_the_returned_weights():
    """With early stopping, Keras restores the best weights at the end of training.
    The reported final metrics must match model.evaluate on the final model."""
    cfg = validate_config({"dataset": "Iris", "epochs": 25, "learning_rate": 0.1, "early_stopping": True, "seed": 3})
    set_global_seed(cfg["seed"])
    data = D.prepare_builtin_tabular("Iris", seed=3)
    model = build_model(cfg, data.input_shape, data.num_classes)
    outcome = train_model(model, data, cfg)
    val_loss, val_acc = model.evaluate(data.X_val, data.y_val, verbose=0)
    assert outcome["final"]["val_loss"] == pytest.approx(val_loss, rel=1e-5)
    assert outcome["final"]["val_accuracy"] == pytest.approx(val_acc)
    assert 1 <= outcome["best_epoch"] <= outcome["epochs_run"] <= 25
    # Restored weights are the best-val-loss epoch's weights.
    assert min(outcome["history"]["val_loss"]) == pytest.approx(outcome["history"]["val_loss"][outcome["best_epoch"] - 1])
    assert outcome["weights_restored"] == (outcome["best_epoch"] != outcome["epochs_run"])


def test_confusion_matrix_has_every_class_even_if_absent_from_val():
    X = np.random.default_rng(0).normal(size=(30, 3)).astype(np.float32)
    y = np.array([0] * 15 + [1] * 15)
    data = D.PreparedData(X_train=X[:20], X_val=X[20:], y_train=y[:20], y_val=np.ones(10, np.int64),
                          class_names=["a", "b", "c"], input_shape=(3,), kind="tabular")
    cfg = validate_config({"epochs": 1})
    outcome = train_model(build_model(cfg, (3,), 3), data, cfg)
    assert outcome["confusion_matrix"].shape == (3, 3)


def test_same_seed_same_result():
    runs = [run_experiment({"dataset": "Wine", "epochs": 3, "seed": 11}) for _ in range(2)]
    assert runs[0]["history"] == runs[1]["history"]
    assert runs[0]["results"] == runs[1]["results"]


def test_run_experiment_result_is_json_ready():
    result = run_experiment({"dataset": "Breast Cancer", "epochs": 2})
    json.dumps(result)
    assert result["status"] == "success" and result["num_classes"] == 2
    assert result["parameters"]["total"] == result["parameters"]["trainable"]
    assert result["history_summary"]["sampled_epochs"] == [1, 2]


def test_run_experiment_cnn_with_fake_loader():
    result = run_experiment({"network_type": "CNN", "dataset": "Fashion-MNIST", "epochs": 1, "conv_layers": 1,
                             "base_filters": 4}, image_loader=fake_image_loader(), max_train_samples=100)
    assert result["train_samples"] == 100 and result["class_names"][7] == "Sneaker"


def test_compact_history():
    h = {k: list(np.linspace(1, 0, 20)) for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    c = compact_history(h)
    assert c["sampled_epochs"] == [1, 6, 11, 16, 20] and len(c["loss"]) == 5
    assert compact_history({})["sampled_epochs"] == []
