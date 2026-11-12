"""
Accuracy and Grad-CAM verification with the app's own training code
===================================================================
Trains the recommended configurations on the full built-in datasets (same code
path as the Train button: core.data -> core.models -> core.training) and checks:

  * Iris  (Simple NN 2x16, Adam, lr 0.01, batch 16, 50 and 100 epochs)  >= 93 %
  * MNIST (CNN 2 conv, 16 filters, Adam, lr 0.001, batch 32, 3 epochs)  >= 97 %
  * Fashion-MNIST (same CNN, 3 epochs) and the repo's sample images
  * Grad-CAM on correctly classified MNIST validation digits: how much of the
    heatmap lies on the digit's pixels compared with chance, and how many maps are empty.

    python benchmarks/verify_models.py --out results.json

MNIST / Fashion-MNIST are downloaded by Keras on first use (~30 MB each).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

from core.data import prepare_builtin_images, prepare_builtin_tabular  # noqa: E402
from core.gradcam import compute_gradcam  # noqa: E402
from core.image_preprocessing import open_image, preprocess_image  # noqa: E402
from core.models import build_model, count_parameters  # noqa: E402
from core.training import predict_proba, set_global_seed, train_model, validate_config  # noqa: E402


def train(cfg, data):
    import tensorflow as tf
    cfg = validate_config(cfg)
    set_global_seed(cfg["seed"])
    tf.keras.backend.clear_session()
    model = build_model(cfg, data.input_shape, data.num_classes)
    t0 = time.perf_counter()
    outcome = train_model(model, data, cfg)
    return model, outcome, time.perf_counter() - t0


def gradcam_quality(model, data, n=300, seed=0):
    """Heat on the digit vs chance for correctly classified validation images."""
    probs = predict_proba(model, data.X_val)
    correct = np.flatnonzero(np.argmax(probs, axis=1) == data.y_val)
    rng = np.random.default_rng(seed)
    idx = rng.choice(correct, size=min(n, len(correct)), replace=False)
    ratios, empty = [], 0
    t0 = time.perf_counter()
    for i in idx:
        image = data.X_val[i, ..., 0]
        heat, _ = compute_gradcam(model, image)
        if heat.max() <= 0:
            empty += 1
            continue
        on_digit = image > 0.1
        share_heat = float(heat[on_digit].sum() / heat.sum())
        share_pixels = float(on_digit.mean())
        ratios.append(share_heat / share_pixels)
    ms = (time.perf_counter() - t0) * 1000 / len(idx)
    return {"images": int(len(idx)), "empty_heatmaps": int(empty),
            "median_heat_on_digit_vs_chance": round(float(np.median(ratios)), 2),
            "p10_heat_on_digit_vs_chance": round(float(np.percentile(ratios, 10)), 2),
            "ms_per_heatmap": round(ms, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-fashion", action="store_true")
    args = ap.parse_args()
    results = {}

    iris = prepare_builtin_tabular("Iris", seed=42)
    for epochs in (50, 100):
        _, outcome, secs = train({"network_type": "Simple NN", "dataset": "Iris", "hidden_layers": 2,
                                  "nodes_per_layer": 16, "optimizer": "Adam", "learning_rate": 0.01,
                                  "batch_size": 16, "epochs": epochs, "seed": 42}, iris)
        results[f"iris_{epochs}_epochs"] = {"val_accuracy": round(outcome["final"]["val_accuracy"], 4),
                                            "val_loss": round(outcome["final"]["val_loss"], 4),
                                            "train_seconds": round(secs, 1)}
        print("Iris", epochs, results[f"iris_{epochs}_epochs"], flush=True)

    cnn = {"network_type": "CNN", "conv_layers": 2, "base_filters": 16, "optimizer": "Adam",
           "learning_rate": 0.001, "batch_size": 32, "epochs": 3, "seed": 42}
    mnist = prepare_builtin_images("MNIST", seed=42)
    model, outcome, secs = train({**cnn, "dataset": "MNIST"}, mnist)
    h = outcome["history"]["val_accuracy"]
    results["mnist_3_epochs"] = {"val_accuracy": round(outcome["final"]["val_accuracy"], 4),
                                 "val_accuracy_per_epoch": [round(v, 4) for v in h],
                                 "val_loss": round(outcome["final"]["val_loss"], 4),
                                 "parameters": count_parameters(model)["total"], "train_seconds": round(secs, 1)}
    print("MNIST", results["mnist_3_epochs"], flush=True)
    seven, _ = preprocess_image(open_image((ROOT / "handwritten_seven.png").read_bytes()), style="digit")
    p = predict_proba(model, seven.reshape(1, 28, 28, 1))[0]
    heat, _ = compute_gradcam(model, seven)
    stroke = seven > 0.1
    results["handwritten_seven"] = {"predicted": int(np.argmax(p)), "probability": round(float(p.max()), 4),
                                    "gradcam_heat_on_stroke_vs_chance":
                                        round(float(heat[stroke].sum() / heat.sum() / stroke.mean()), 2)}
    print("seven", results["handwritten_seven"], flush=True)
    results["gradcam_mnist"] = gradcam_quality(model, mnist)
    print("gradcam", results["gradcam_mnist"], flush=True)

    if not args.skip_fashion:
        fashion = prepare_builtin_images("Fashion-MNIST", seed=42)
        model, outcome, secs = train({**cnn, "dataset": "Fashion-MNIST"}, fashion)
        results["fashion_3_epochs"] = {"val_accuracy": round(outcome["final"]["val_accuracy"], 4),
                                       "train_seconds": round(secs, 1)}
        for fname in ("fashion_sneaker.png", "fashion_ankle_boot.png"):
            arr, _ = preprocess_image(open_image((ROOT / fname).read_bytes()), style="object")
            p = predict_proba(model, arr.reshape(1, 28, 28, 1))[0]
            results[fname] = {"predicted": fashion.class_names[int(np.argmax(p))], "probability": round(float(p.max()), 4)}
        print("fashion", results["fashion_3_epochs"], results["fashion_sneaker.png"], results["fashion_ankle_boot.png"])

    checks = {
        "iris_50_epochs >= 0.93": results["iris_50_epochs"]["val_accuracy"] >= 0.93,
        "iris_100_epochs >= 0.93": results["iris_100_epochs"]["val_accuracy"] >= 0.93,
        "mnist_3_epochs >= 0.97": results["mnist_3_epochs"]["val_accuracy"] >= 0.97,
        "handwritten seven predicted as 7": results["handwritten_seven"]["predicted"] == 7,
        "gradcam heat concentrates on the digit (> 1.5x chance)":
            results["gradcam_mnist"]["median_heat_on_digit_vs_chance"] > 1.5,
    }
    results["checks"] = checks
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
