"""
Training
========
Config validation, seeding, training and evaluation shared by the app, the
agent tools and the MCP server.

Important detail: when early stopping is on, Keras restores the best epoch's
weights at the end of training (Keras 3 does this even if training was never
stopped early). The "final" metrics reported here are therefore computed on
the weights the model actually ends up with, so the numbers always describe
the model the user tests and downloads.

Speed: with NumPy inputs Keras builds a fresh tf.data iterator for every epoch
and for every predict/evaluate call. On a CPU that fixed cost (~100-200 ms)
dwarfs the maths for small datasets such as Iris. Training therefore feeds one
repeating tf.data pipeline (its iterator is created once per fit), and the
final evaluation runs the model's forward pass directly.
"""

import logging
import math

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from core.data import (DEFAULT_SEED, IMAGE_DATASETS, TABULAR_DATASETS, DataError,
                       prepare_builtin_images, prepare_builtin_tabular)
from core.models import OPTIMIZERS, build_model, count_parameters, summary_text

NETWORK_TYPES = ("Simple NN", "CNN")
EARLY_STOPPING_PATIENCE = 5

# (min, max) for numeric settings. The UI uses a narrower range; these are the
# hard limits enforced for every caller (UI, AI agent, remote MCP clients).
LIMITS = {
    "hidden_layers": (1, 10),
    "nodes_per_layer": (2, 256),
    "conv_layers": (1, 4),            # 28 -> 14 -> 7 -> 3 -> 1: a 5th pooling layer is impossible
    "base_filters": (4, 64),
    "epochs": (1, 200),
    "batch_size": (4, 512),
    "learning_rate": (1e-5, 1.0),
    "dropout_rate": (0.0, 0.9),
    "seed": (0, 2**31 - 1),
}

DEFAULTS = {
    "network_type": "Simple NN",
    "dataset": "Iris",
    "hidden_layers": 2,
    "nodes_per_layer": 16,
    "conv_layers": 2,
    "base_filters": 16,
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "dropout_rate": 0.0,
    "early_stopping": False,
    "seed": DEFAULT_SEED,
}

INT_KEYS = ("hidden_layers", "nodes_per_layer", "conv_layers", "base_filters", "epochs", "batch_size", "seed")
FLOAT_KEYS = ("learning_rate", "dropout_rate")


class ConfigError(ValueError):
    """Invalid training configuration (message is safe to show to users / LLMs)."""


def datasets_for(network_type):
    return TABULAR_DATASETS if network_type == "Simple NN" else IMAGE_DATASETS


def validate_config(raw, allow_custom_dataset=False):
    """Return a clean config dict (defaults filled in, types coerced) or raise ConfigError.

    Unknown keys are rejected so remote callers cannot smuggle options in.
    """
    if not isinstance(raw, dict):
        raise ConfigError("The training configuration must be a JSON object.")
    allowed = set(DEFAULTS) | {"using_custom", "target_column"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown setting(s): {', '.join(unknown)}.")

    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in raw.items() if v is not None})
    errors = []

    if cfg["network_type"] not in NETWORK_TYPES:
        errors.append(f"network_type must be one of {list(NETWORK_TYPES)}")
    elif not (allow_custom_dataset and cfg.get("using_custom")) and cfg["dataset"] not in datasets_for(cfg["network_type"]):
        errors.append(f"dataset for {cfg['network_type']} must be one of {list(datasets_for(cfg['network_type']))}")
    if cfg["optimizer"] not in OPTIMIZERS:
        errors.append(f"optimizer must be one of {list(OPTIMIZERS)}")

    for key in INT_KEYS:
        try:
            value = float(cfg[key])
            if not value.is_integer():
                raise ValueError
            cfg[key] = int(value)
        except (TypeError, ValueError):
            errors.append(f"{key} must be a whole number")
            continue
        lo, hi = LIMITS[key]
        if not lo <= cfg[key] <= hi:
            errors.append(f"{key} must be between {lo} and {hi} (got {cfg[key]})")
    for key in FLOAT_KEYS:
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError):
            errors.append(f"{key} must be a number")
            continue
        lo, hi = LIMITS[key]
        if not lo <= cfg[key] <= hi:
            errors.append(f"{key} must be between {lo} and {hi} (got {cfg[key]})")
    if isinstance(cfg["early_stopping"], str):
        cfg["early_stopping"] = cfg["early_stopping"].strip().lower() in ("true", "1", "yes")
    cfg["early_stopping"] = bool(cfg["early_stopping"])

    if errors:
        raise ConfigError("Invalid training settings: " + "; ".join(errors) + ".")
    return cfg


def set_global_seed(seed):
    """Seed Python `random`, NumPy and TensorFlow so runs are reproducible on CPU."""
    tf.keras.utils.set_random_seed(int(seed))


def early_stopping_callback():
    return tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                            restore_best_weights=True)


class _SkipPerModelTracingNotice(logging.Filter):
    """Every trained model gets its own compiled forward pass (_logits_function), traced
    exactly once. TensorFlow counts traces per source line across models and, after a
    few training runs, logs a misleading "excessive retracing" warning for it."""

    def filter(self, record):
        message = record.getMessage()
        return not ("_logits_function" in message and "retracing" in message)


tf.get_logger().addFilter(_SkipPerModelTracingNotice())

VAL_BATCH = 512      # validation / evaluation batch size (inference only, so larger is fine)
EVAL_CHUNK = 1024


def batch_pipeline(X, y, batch_size, shuffle):
    """Endless tf.data stream of (X[idx], y[idx]) batches, one pass per epoch.

    Batching happens before `repeat`, so every epoch sees each sample exactly once
    (same as fit(..., batch_size=...)) and is reshuffled each epoch. The random
    order comes from TensorFlow's global seed (see set_global_seed), so runs are
    reproducible. Only indices are shuffled; the arrays are not copied per epoch.
    """
    X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
    y = tf.convert_to_tensor(np.asarray(y, dtype=np.int64))
    n = int(X.shape[0])
    ds = tf.data.Dataset.range(n)
    if shuffle:
        ds = ds.shuffle(n, reshuffle_each_iteration=True)
    gather = tf.autograph.experimental.do_not_convert(lambda idx: (tf.gather(X, idx), tf.gather(y, idx)))
    ds = ds.batch(int(batch_size)).map(gather)
    return ds.repeat().prefetch(tf.data.AUTOTUNE), math.ceil(n / int(batch_size))


def _logits_function(model):
    """A compiled forward pass returning the pre-softmax scores, or None when the
    model does not end in a softmax Dense layer (all models built by this app do)."""
    layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    last = layers[-1] if layers else None
    if not (isinstance(last, tf.keras.layers.Dense) and getattr(last.activation, "__name__", "") == "softmax"):
        return None

    spec = tf.TensorSpec(shape=(None,) + tuple(model.input_shape[1:]), dtype=tf.float32)

    @tf.function(input_signature=[spec], autograph=False)   # traced once; the layer loop unrolls at trace time
    def logits(x):
        h = x
        for layer in layers[:-1]:
            h = layer(h, training=False)
        out = tf.matmul(h, last.kernel)
        return out + last.bias if last.use_bias else out
    return logits


def evaluate_split(model, X, y, logits_fn=None):
    """(loss, accuracy, probabilities) of the model's current weights on X, y.

    The loss is sparse categorical cross-entropy computed from the logits, which is
    exactly what model.evaluate reports for a softmax output layer, but without
    Keras' per-call data-pipeline setup.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if logits_fn is None:
        loss, acc = model.evaluate(X, y, batch_size=VAL_BATCH, verbose=0)
        return float(loss), float(acc), predict_proba(model, X)
    losses, probs = [], []
    for start in range(0, len(X), EVAL_CHUNK):
        z = logits_fn(tf.convert_to_tensor(X[start:start + EVAL_CHUNK]))
        losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[start:start + EVAL_CHUNK],
                                                                    logits=z).numpy())
        probs.append(tf.nn.softmax(z).numpy())
    if not probs:
        return float("nan"), float("nan"), np.zeros((0, model.output_shape[-1]), np.float32)
    probs = np.concatenate(probs)
    loss = float(np.mean(np.concatenate(losses)))
    acc = float(np.mean(np.argmax(probs, axis=1) == y))
    return loss, acc, probs


def predict_proba(model, X):
    """Class probabilities for a small batch (e.g. one uploaded image). Uses
    predict_on_batch: about 4 ms instead of ~140 ms for model.predict on one sample."""
    X = np.asarray(X, dtype=np.float32)
    out = [np.asarray(model.predict_on_batch(X[i:i + EVAL_CHUNK])) for i in range(0, len(X), EVAL_CHUNK)]
    return np.concatenate(out) if out else np.zeros((0, model.output_shape[-1]), np.float32)


def train_model(model, data, cfg, callbacks=None):
    """Fit `model` on prepared data, then evaluate the weights it ends up with.

    Returns a dict with the Keras history, epoch bookkeeping, final metrics
    (evaluated on the final weights, not history[-1]), predictions and the
    confusion matrix.
    """
    cbs = list(callbacks or [])
    es = None
    if cfg.get("early_stopping"):
        es = early_stopping_callback()
        cbs.insert(0, es)

    train_ds, steps = batch_pipeline(data.X_train, data.y_train, cfg["batch_size"], shuffle=True)
    val_ds, val_steps = batch_pipeline(data.X_val, data.y_val, VAL_BATCH, shuffle=False)
    history = model.fit(train_ds, steps_per_epoch=steps, epochs=int(cfg["epochs"]),
                        validation_data=val_ds, validation_steps=val_steps, shuffle=False,  # pipeline shuffles
                        verbose=0, callbacks=cbs)
    h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    epochs_run = len(h.get("loss", []))

    best_epoch = epochs_run
    stopped_early = False
    if es is not None:
        stopped_early = es.stopped_epoch > 0
        if es.best_weights is not None:
            best_epoch = int(es.best_epoch) + 1

    logits_fn = _logits_function(model)
    train_loss, train_acc, _ = evaluate_split(model, data.X_train, data.y_train, logits_fn)
    val_loss, val_acc, probs = evaluate_split(model, data.X_val, data.y_val, logits_fn)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(data.y_val, y_pred, labels=list(range(data.num_classes)))

    return {
        "history": h,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "weights_restored": es is not None and best_epoch != epochs_run,
        "final": {
            "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss, "val_accuracy": val_acc,
        },
        "y_pred": y_pred,
        "confusion_matrix": cm,
    }


def compact_history(h, max_points=5):
    """Downsample an epoch history to a few representative points (saves LLM tokens)."""
    n = len(h.get("loss", []))
    if n == 0:
        return {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "sampled_epochs": []}
    idx = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    idx = [i for i in idx if 0 <= i < n][:max_points]
    out = {k: [round(float(h[k][i]), 4) for i in idx] for k in ("loss", "val_loss", "accuracy", "val_accuracy") if k in h}
    out["sampled_epochs"] = [i + 1 for i in idx]
    return out


def _unused_keys(cfg):
    """Architecture settings that do not apply to this network type."""
    return ("conv_layers", "base_filters") if cfg["network_type"] == "Simple NN" else ("hidden_layers", "nodes_per_layer")


def prepare_builtin_data(cfg, image_loader=None, max_train_samples=None, max_val_samples=None):
    """PreparedData for a built-in dataset described by a validated config."""
    if cfg["network_type"] == "Simple NN":
        return prepare_builtin_tabular(cfg["dataset"], seed=cfg["seed"])
    return prepare_builtin_images(cfg["dataset"], loader=image_loader, max_train_samples=max_train_samples,
                                  max_val_samples=max_val_samples, seed=cfg["seed"])


def run_experiment(raw_config, image_loader=None, max_train_samples=None, max_val_samples=None,
                   callbacks=None, return_model=False):
    """Validate -> seed -> load data -> build -> train -> evaluate.

    Returns the JSON-friendly result dict used by the AI agent and the MCP server
    (and optionally the trained Keras model as a second value).
    """
    cfg = validate_config(raw_config)
    set_global_seed(cfg["seed"])
    tf.keras.backend.clear_session()
    data = prepare_builtin_data(cfg, image_loader, max_train_samples, max_val_samples)
    model = build_model(cfg, data.input_shape, data.num_classes)
    outcome = train_model(model, data, cfg, callbacks=callbacks)

    h = outcome["history"]
    final = outcome["final"]
    result = {
        "status": "success",
        "network_type": cfg["network_type"],
        "dataset": cfg["dataset"],
        "class_names": data.class_names,
        "num_classes": data.num_classes,
        "config": {k: cfg[k] for k in DEFAULTS if k not in ("network_type", "dataset") + _unused_keys(cfg)},
        "epochs_trained": outcome["epochs_run"],
        "best_epoch": outcome["best_epoch"],
        "stopped_early": outcome["stopped_early"],
        "results": {
            "final_train_loss": round(final["train_loss"], 4),
            "final_val_loss": round(final["val_loss"], 4),
            "final_train_accuracy": round(final["train_accuracy"], 4),
            "final_val_accuracy": round(final["val_accuracy"], 4),
        },
        "history": {k: [round(float(x), 4) for x in h.get(k, [])] for k in ("loss", "val_loss", "accuracy", "val_accuracy")},
        "history_summary": compact_history(h),
        "confusion_matrix": outcome["confusion_matrix"].tolist(),
        "parameters": count_parameters(model),
        "train_samples": int(len(data.y_train)),
        "val_samples": int(len(data.y_val)),
        "notes": data.info + data.warnings,
    }
    if return_model:
        return result, model, data
    return result


__all__ = [
    "ConfigError", "DataError", "DEFAULTS", "LIMITS", "NETWORK_TYPES",
    "validate_config", "set_global_seed", "train_model", "evaluate_split", "predict_proba", "compact_history",
    "prepare_builtin_data", "run_experiment", "summary_text", "datasets_for",
]
