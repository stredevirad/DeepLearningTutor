"""
Agent Tools for the Deep Learning Tutor
========================================
Contains tool functions that the Gemini AI agent can autonomously call
during conversations. Each function has detailed docstrings and type hints,
and TOOL_DECLARATIONS describes the same tools to Gemini as JSON schemas.

All arguments come from an LLM, so every tool validates its inputs and
returns {"status": "error", "error": "..."} instead of raising.
"""

import copy
import json
import threading
from collections import OrderedDict

import tensorflow as tf

from core.data import DataError, IMAGE_DATASETS, TABULAR_DATASETS
from core.training import LIMITS, ConfigError, run_experiment as _run_experiment_core, validate_config

# CNN experiments requested by the agent train on a random subset so a single
# chat turn stays interactive on a CPU (the Train tab always uses the full data).
AGENT_CNN_MAX_TRAIN = 12_000
AGENT_CNN_MAX_VAL = 5_000
AGENT_MAX_EPOCHS = 100

# Training is deterministic for a given configuration and seed, so an identical
# experiment requested again (agents like to re-run the baseline) reuses the
# earlier result. Only built-in datasets are ever trained here: nothing private.
_EXPERIMENT_CACHE = OrderedDict()
_EXPERIMENT_CACHE_SIZE = 16
_cache_lock = threading.Lock()

# --- TOOL REGISTRY ---
TOOL_FUNCTIONS = {}


def _register(func):
    """Decorator to register a function as an available agent tool."""
    TOOL_FUNCTIONS[func.__name__] = func
    return func


@_register
def run_experiment(network_type: str = "Simple NN", dataset: str = "Iris",
                   hidden_layers: int = 2, nodes_per_layer: int = 16,
                   epochs: int = 30, conv_layers: int = 2,
                   base_filters: int = 16, early_stopping: bool = False,
                   learning_rate: float = 0.001, batch_size: int = 32,
                   optimizer: str = "Adam", dropout_rate: float = 0.0) -> dict:
    """Build and train a TensorFlow/Keras neural network on a REAL dataset, returning metrics.

    Call this tool when the user wants to:
    - See what happens with different hyperparameters
    - Compare model architectures
    - Understand the effect of changing epochs, nodes, or filters
    - Run a quick experiment to test a hypothesis

    Args:
        network_type: "Simple NN" or "CNN".
        dataset: For Simple NN: "Iris", "Wine", "Breast Cancer" or "Digits". For CNN: "MNIST" or "Fashion-MNIST".
        hidden_layers: Number of hidden Dense layers (Simple NN only). Range: 1-10.
        nodes_per_layer: Neurons per hidden layer (Simple NN only). Range: 2-256.
        epochs: Training epochs. Range: 1-100.
        conv_layers: Number of Conv2D layers (CNN only). Range: 1-4.
        base_filters: Starting filter count (CNN only), doubled in each conv block. Range: 4-64.
        early_stopping: Whether to stop training when validation loss stops improving.
        learning_rate: Learning rate for the optimizer. Typical: 0.0001, 0.001, 0.01, 0.1.
        batch_size: Batch size for training. Range: 4-512.
        optimizer: Optimizer to use. Options: "Adam", "SGD", "RMSprop".
        dropout_rate: Dropout rate after dense layers. Range: 0.0-0.9.

    Returns:
        Dictionary with training history, final metrics, and confusion matrix data.
    """
    config = {
        "network_type": network_type, "dataset": dataset, "hidden_layers": hidden_layers,
        "nodes_per_layer": nodes_per_layer, "epochs": epochs, "conv_layers": conv_layers,
        "base_filters": base_filters, "early_stopping": early_stopping,
        "learning_rate": learning_rate, "batch_size": batch_size,
        "optimizer": optimizer, "dropout_rate": dropout_rate,
    }
    try:
        if float(epochs) > AGENT_MAX_EPOCHS:
            return {"status": "error", "error": f"epochs must be at most {AGENT_MAX_EPOCHS} for agent experiments."}
    except (TypeError, ValueError):
        return {"status": "error", "error": "epochs must be a whole number."}

    try:
        is_cnn = network_type == "CNN"
        key = json.dumps(validate_config(config), sort_keys=True)
        with _cache_lock:
            cached = _EXPERIMENT_CACHE.get(key)
        if cached is not None:
            result = copy.deepcopy(cached)
            result["notes"] = result.get("notes", []) + [
                "Identical to an earlier experiment: the earlier result was reused "
                "(training is deterministic for the same settings and seed)."]
            return result
        result = _run_experiment_core(
            config,
            max_train_samples=AGENT_CNN_MAX_TRAIN if is_cnn else None,
            max_val_samples=AGENT_CNN_MAX_VAL if is_cnn else None,
        )
        with _cache_lock:
            _EXPERIMENT_CACHE[key] = copy.deepcopy(result)
            while len(_EXPERIMENT_CACHE) > _EXPERIMENT_CACHE_SIZE:
                _EXPERIMENT_CACHE.popitem(last=False)
        return result
    except (ConfigError, DataError) as e:
        return {"status": "error", "error": str(e)}
    except tf.errors.ResourceExhaustedError:
        return {"status": "error", "error": "Ran out of memory. Try a smaller batch size or fewer filters/nodes."}


@_register
def generate_code(task: str, framework: str = "tensorflow") -> dict:
    """Signal that the user needs a working code snippet for a deep learning task.

    Call this tool when the user asks to:
    - Write or show code for a specific architecture or technique
    - Implement a feature like dropout, batch normalization, or data augmentation
    - Create a training loop, data pipeline, or custom layer

    Args:
        task: What the code should accomplish (e.g. "Add dropout regularization to a CNN").
        framework: The ML framework to use. Default "tensorflow". Options: "tensorflow", "pytorch".

    Returns:
        Structured context for generating the code response.
    """
    task = str(task or "").strip()[:500]
    if not task:
        return {"status": "error", "error": "Describe what the code should do in 'task'."}
    framework = framework if framework in ("tensorflow", "pytorch") else "tensorflow"
    library = "TensorFlow/Keras" if framework == "tensorflow" else "PyTorch"
    return {
        "status": "ready",
        "task": task,
        "framework": framework,
        "instruction": (
            f"Now generate a COMPLETE, STANDALONE Python script for: '{task}' using {library}. "
            "The script MUST be fully runnable outside this workbench (e.g. in Google Colab or a local terminal).\n\n"
            "FORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n"
            "1. Put ALL the Python code in ONE SINGLE ```python ``` fenced code block. "
            "Do NOT split the code across multiple blocks. Do NOT use markdown headers (# ##) outside the code block to label code sections. "
            "Use Python comments (#) inside the code for section labels.\n"
            "2. After the code block, add a section titled **Explanation:** that walks through the code step by step.\n"
            "3. After the explanation, add a section titled **Model Architecture:** describing the layers, their output shapes, and total parameter count.\n\n"
            "THE CODE MUST INCLUDE:\n"
            "- All necessary imports (tensorflow, numpy, pandas, sklearn, etc.)\n"
            "- A fixed random seed (e.g. tf.keras.utils.set_random_seed(42)) for reproducibility\n"
            "- Data loading — if using a custom CSV, use `pd.read_csv('your_dataset.csv')` as a placeholder, "
            "name the target column explicitly, convert numeric columns with pd.to_numeric(errors='coerce') "
            "and one-hot encode low-cardinality text columns\n"
            "- A train/validation split with train_test_split(..., stratify=y) done BEFORE any scaling; "
            "fit the scaler/encoder on the TRAINING split only and use it to transform the validation split "
            "(never fit preprocessing on the full dataset - that is data leakage)\n"
            "- Full model architecture with detailed inline comments\n"
            "- Model compilation and training\n"
            "- Evaluation with printed metrics\n"
            "- Model saving with `model.save('my_model.keras')` at the end\n"
            "Make the code copy-paste ready with no placeholders except the CSV filename."
        )
    }


@_register
def explain_concept(topic: str, depth: str = "beginner") -> dict:
    """Create a structured educational explanation of a deep learning concept.

    Call this tool when the user asks:
    - "What is X?" or "How does X work?" where X is a DL concept
    - For an explanation of theory, math, or architecture details
    - To understand why something behaves a certain way in neural networks

    Args:
        topic: The concept to explain (e.g. "backpropagation", "attention mechanism").
        depth: Detail level. "beginner" for analogies, "intermediate" for technical, "advanced" for full math.

    Returns:
        Structured context for generating the explanation.
    """
    depth_guide = {
        "beginner": "Use a simple real-world analogy. Avoid heavy math. Focus on intuition.",
        "intermediate": "Include technical details and light math. Explain the 'why'.",
        "advanced": "Full mathematical formulation with derivations. Include implementation notes."
    }
    topic = str(topic or "").strip()[:300]
    if not topic:
        return {"status": "error", "error": "Provide the concept to explain in 'topic'."}
    depth = str(depth or "beginner").lower()
    if depth not in depth_guide:
        depth = "beginner"
    return {
        "status": "ready",
        "topic": topic,
        "depth": depth,
        "instruction": (
            f"Now explain '{topic}' at the {depth} level. "
            f"{depth_guide[depth]} "
            "Structure: 1) One-line summary, 2) Core explanation, "
            "3) Practical example or code snippet, 4) Common pitfalls."
        )
    }


def _test_accuracy_of(exp):
    """The run's test accuracy, or None (no test split, or a result from before the test set existed)."""
    try:
        value = exp.get("results", {}).get("final_test_accuracy")
        return None if value is None else float(value)
    except (TypeError, ValueError, AttributeError):
        return None


def _accuracy_of(exp):
    try:
        return float(exp.get("results", {}).get("final_val_accuracy", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


@_register
def compare_experiments(experiment_summaries: str) -> dict:
    """Compare multiple experiment results and identify the best configuration.

    Call this tool when:
    - You have run multiple experiments and want to pick the best one
    - You want to understand which hyperparameter changes had the most impact
    - You need to summarize a series of training runs for the user

    Args:
        experiment_summaries: A JSON string containing a list of experiment result
            dictionaries (each from a previous run_experiment call). Each entry should
            have at minimum: config, results (with final_val_accuracy, final_val_loss;
            final_test_accuracy is reported too when present).

    Returns:
        Dictionary with comparison analysis: best model, ranking, and recommendations.
    """
    try:
        experiments = json.loads(experiment_summaries)
    except (TypeError, json.JSONDecodeError):
        return {"status": "error", "error": "Invalid JSON. Pass a JSON string of experiment result dicts."}

    if not isinstance(experiments, list) or len(experiments) == 0:
        return {"status": "error", "error": "Provide a non-empty list of experiment results."}
    if not all(isinstance(e, dict) for e in experiments):
        return {"status": "error", "error": "Each experiment must be a JSON object with 'config' and 'results'."}

    # Rank by validation accuracy (descending)
    ranked = sorted(enumerate(experiments), key=lambda x: _accuracy_of(x[1]), reverse=True)

    best_idx, best = ranked[0]
    worst_idx, worst = ranked[-1]

    comparison = {
        "status": "success",
        "total_experiments": len(experiments),
        "best_experiment": {
            "index": best_idx,
            "val_accuracy": best.get("results", {}).get("final_val_accuracy"),
            "val_loss": best.get("results", {}).get("final_val_loss"),
            "test_accuracy": _test_accuracy_of(best),
            "config": best.get("config", {}),
        },
        "worst_experiment": {
            "index": worst_idx,
            "val_accuracy": worst.get("results", {}).get("final_val_accuracy"),
            "val_loss": worst.get("results", {}).get("final_val_loss"),
            "test_accuracy": _test_accuracy_of(worst),
            "config": worst.get("config", {}),
        },
        "ranking": [
            {
                "rank": rank + 1,
                "index": idx,
                "val_accuracy": exp.get("results", {}).get("final_val_accuracy"),
                "val_loss": exp.get("results", {}).get("final_val_loss"),
                "test_accuracy": _test_accuracy_of(exp),
                "parameters": (exp.get("parameters") or {}).get("total"),
                "key_config": {
                    k: exp.get("config", {}).get(k)
                    for k in ["learning_rate", "dropout_rate", "optimizer", "hidden_layers", "nodes_per_layer",
                              "conv_layers", "base_filters", "epochs", "batch_size"]
                    if exp.get("config", {}).get(k) is not None
                },
            }
            for rank, (idx, exp) in enumerate(ranked)
        ],
        "accuracy_spread": round(_accuracy_of(best) - _accuracy_of(worst), 4),
    }

    # Recommendation, worded for what one validation split can and cannot show.
    best_cfg = best.get("config", {})
    val_samples = best.get("val_samples")
    comparison["note"] = (
        "Ranked by validation accuracy on one validation split"
        + (f" of {val_samples} examples" if val_samples else "")
        + ". Call the top run the best validation result in these runs, not a universally better model; "
          "on small datasets a difference of one or two examples is within noise. Also weigh validation loss "
          "and model size (parameters).")
    if _test_accuracy_of(best) is not None:
        comparison["note"] += (
            f" The runs are ranked by validation accuracy on purpose: choosing by test accuracy would turn the test "
            f"set into a second validation set. Report the chosen run's test accuracy ({_test_accuracy_of(best):.4f}) "
            "as its final result, and say that it is a test accuracy.")
    else:
        comparison["note"] += " No test accuracy is available for the top run, so quote its validation accuracy as such."
    if comparison["accuracy_spread"] < 0.02:
        comparison["recommendation"] = (
            "The runs performed about the same (validation accuracy within 2 points), so this comparison does not "
            "separate them. Try larger changes (architecture, much higher/lower learning rate) or compare validation loss.")
    else:
        comparison["recommendation"] = (
            f"Best validation result in these runs: LR={best_cfg.get('learning_rate')}, "
            f"dropout={best_cfg.get('dropout_rate')}, optimizer={best_cfg.get('optimizer')}. "
            "Try small changes around these values to check that the difference is stable.")

    return comparison


def _parse_metric_list(raw, name):
    values = json.loads(raw)
    if not isinstance(values, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise ValueError(f"{name} must be a JSON list of numbers")
    return [float(v) for v in values]


@_register
def diagnose_training(train_loss_json: str, val_loss_json: str,
                      train_acc_json: str, val_acc_json: str) -> dict:
    """Analyze a single training run's history to diagnose issues like overfitting,
    underfitting, exploding gradients, or convergence problems.

    Call this tool when:
    - You want to understand WHY a model performed a certain way
    - You need to decide what hyperparameter to change next
    - The user asks about training behavior or learning curve patterns

    Args:
        train_loss_json: JSON string of the training loss list (epoch by epoch).
        val_loss_json: JSON string of the validation loss list (epoch by epoch).
        train_acc_json: JSON string of the training accuracy list (epoch by epoch).
        val_acc_json: JSON string of the validation accuracy list (epoch by epoch).

    Returns:
        Dictionary with diagnoses, severity levels, and actionable recommendations.
    """
    try:
        train_loss = _parse_metric_list(train_loss_json, "train_loss_json")
        val_loss = _parse_metric_list(val_loss_json, "val_loss_json")
        train_acc = _parse_metric_list(train_acc_json, "train_acc_json")
        val_acc = _parse_metric_list(val_acc_json, "val_acc_json")
    except (TypeError, json.JSONDecodeError):
        return {"status": "error", "error": "Invalid JSON. Pass JSON strings of metric lists."}
    except ValueError as e:
        return {"status": "error", "error": str(e)}

    n = len(train_loss)
    if len({n, len(val_loss), len(train_acc), len(val_acc)}) != 1:
        return {"status": "error", "error": "All four metric lists must have the same number of epochs."}
    if n < 3:
        return {"status": "error", "error": "Need at least 3 epochs of data to diagnose."}

    diagnoses = []
    recommendations = []

    # ── Overfitting Detection ──
    if n >= 5:
        late_train_loss = train_loss[-5:]
        late_val_loss = val_loss[-5:]
        train_dropping = late_train_loss[-1] < late_train_loss[0]
        val_rising = late_val_loss[-1] > late_val_loss[0]
        gap = abs(train_acc[-1] - val_acc[-1])

        if train_dropping and val_rising:
            diagnoses.append({
                "issue": "Overfitting",
                "severity": "high" if gap > 0.15 else "moderate",
                "detail": f"Train loss decreased but val loss increased in last 5 epochs. Accuracy gap: {gap:.2%}",
            })
            recommendations.append("Increase dropout rate, add data augmentation, or reduce model capacity.")

    # ── Underfitting Detection ──
    if train_acc[-1] < 0.6 and val_acc[-1] < 0.6:
        diagnoses.append({
            "issue": "Underfitting",
            "severity": "high",
            "detail": f"Both train ({train_acc[-1]:.2%}) and val ({val_acc[-1]:.2%}) accuracy are low.",
        })
        recommendations.append("Increase model capacity (more layers/nodes), train longer, or increase learning rate.")

    # ── Plateau Detection ──
    if n >= 6:
        second_half_acc = val_acc[n//2:]
        if max(second_half_acc) - min(second_half_acc) < 0.005:
            diagnoses.append({
                "issue": "Accuracy Plateau",
                "severity": "moderate",
                "detail": "Validation accuracy barely changed in the second half of training.",
            })
            recommendations.append("Try a learning rate schedule (reduce on plateau) or change optimizer.")

    # ── Exploding Loss Detection ──
    if any(v > 100 for v in train_loss) or any(v > 100 for v in val_loss):
        diagnoses.append({
            "issue": "Exploding Loss",
            "severity": "critical",
            "detail": "Loss exceeded 100 during training — likely exploding gradients.",
        })
        recommendations.append("Reduce learning rate drastically (try 0.0001) and add gradient clipping.")

    # ── Convergence Check ──
    if n >= 5 and abs(train_loss[-1] - val_loss[-1]) < 0.05 and val_loss[-1] < 0.3:
        diagnoses.append({
            "issue": "Good Convergence",
            "severity": "none",
            "detail": f"Train/val losses are close ({train_loss[-1]:.4f} vs {val_loss[-1]:.4f}) and low.",
        })

    # ── Summary ──
    overall = "healthy"
    if any(d["severity"] == "critical" for d in diagnoses):
        overall = "critical"
    elif any(d["severity"] == "high" for d in diagnoses):
        overall = "needs_attention"
    elif any(d["severity"] == "moderate" for d in diagnoses):
        overall = "minor_issues"

    return {
        "status": "success",
        "overall_health": overall,
        "epochs_analyzed": n,
        "final_metrics": {
            "train_accuracy": round(float(train_acc[-1]), 4),
            "val_accuracy": round(float(val_acc[-1]), 4),
            "train_loss": round(float(train_loss[-1]), 4),
            "val_loss": round(float(val_loss[-1]), 4),
        },
        "diagnoses": diagnoses,
        "recommendations": recommendations,
    }


def execute_tool(name: str, args: dict) -> dict:
    """Execute a registered tool by name with the given arguments."""
    if name not in TOOL_FUNCTIONS:
        return {"status": "error", "error": f"Unknown tool: {name}. Available tools: {', '.join(sorted(TOOL_FUNCTIONS))}."}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {"status": "error", "error": "Tool arguments must be an object of named parameters."}
    func = TOOL_FUNCTIONS[name]
    valid = set(func.__code__.co_varnames[:func.__code__.co_argcount])
    unknown = sorted(set(args) - valid)
    if unknown:
        return {"status": "error",
                "error": f"Unknown argument(s) for {name}: {', '.join(unknown)}. Valid: {', '.join(sorted(valid))}."}
    try:
        return func(**args)
    except TypeError as e:
        return {"status": "error", "error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # a tool must never crash the chat
        return {"status": "error", "error": f"{name} failed: {type(e).__name__}: {e}"}


# List of tool functions for Gemini's function calling API
AGENT_TOOLS = [run_experiment, generate_code, explain_concept, compare_experiments, diagnose_training]


# --- JSON-schema declarations sent to Gemini (explicit, so no SDK introspection is needed) ---
_json_list_arg = {"type": "string", "description": "JSON string of a list of numbers, one per epoch."}

TOOL_DECLARATIONS = [
    {
        "name": "run_experiment",
        "description": ("Build and train a real TensorFlow/Keras model on a real dataset and return metrics, "
                        "history, confusion matrix and parameter counts. Use for any 'what if', comparison "
                        "or optimisation question. CNN experiments train on a 12,000-image subset for speed."),
        "parameters": {
            "type": "object",
            "properties": {
                "network_type": {"type": "string", "enum": ["Simple NN", "CNN"]},
                "dataset": {"type": "string", "enum": list(TABULAR_DATASETS) + list(IMAGE_DATASETS),
                            "description": "Simple NN: Iris, Wine, Breast Cancer, Digits. CNN: MNIST, Fashion-MNIST."},
                "hidden_layers": {"type": "integer", "minimum": LIMITS["hidden_layers"][0], "maximum": LIMITS["hidden_layers"][1]},
                "nodes_per_layer": {"type": "integer", "minimum": LIMITS["nodes_per_layer"][0], "maximum": LIMITS["nodes_per_layer"][1]},
                "epochs": {"type": "integer", "minimum": 1, "maximum": AGENT_MAX_EPOCHS},
                "conv_layers": {"type": "integer", "minimum": LIMITS["conv_layers"][0], "maximum": LIMITS["conv_layers"][1]},
                "base_filters": {"type": "integer", "minimum": LIMITS["base_filters"][0], "maximum": LIMITS["base_filters"][1]},
                "early_stopping": {"type": "boolean"},
                "learning_rate": {"type": "number", "minimum": LIMITS["learning_rate"][0], "maximum": LIMITS["learning_rate"][1]},
                "batch_size": {"type": "integer", "minimum": LIMITS["batch_size"][0], "maximum": LIMITS["batch_size"][1]},
                "optimizer": {"type": "string", "enum": ["Adam", "SGD", "RMSprop"]},
                "dropout_rate": {"type": "number", "minimum": 0.0, "maximum": LIMITS["dropout_rate"][1]},
            },
            "required": ["network_type", "dataset"],
        },
    },
    {
        "name": "generate_code",
        "description": "Signal that the user needs a complete, runnable code example for a deep learning task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What the code should accomplish."},
                "framework": {"type": "string", "enum": ["tensorflow", "pytorch"]},
            },
            "required": ["task"],
        },
    },
    {
        "name": "explain_concept",
        "description": "Produce a structured explanation of a deep learning concept.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "depth": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "compare_experiments",
        "description": "Rank several run_experiment results and recommend the best configuration.",
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_summaries": {"type": "string",
                                         "description": "JSON string: list of run_experiment result objects."},
            },
            "required": ["experiment_summaries"],
        },
    },
    {
        "name": "diagnose_training",
        "description": "Diagnose overfitting, underfitting, plateaus or exploding loss from one run's history.",
        "parameters": {
            "type": "object",
            "properties": {
                "train_loss_json": _json_list_arg,
                "val_loss_json": _json_list_arg,
                "train_acc_json": _json_list_arg,
                "val_acc_json": _json_list_arg,
            },
            "required": ["train_loss_json", "val_loss_json", "train_acc_json", "val_acc_json"],
        },
    },
]
