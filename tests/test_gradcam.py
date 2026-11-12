"""Grad-CAM on tiny CNNs (1-3 conv layers), fresh and reloaded from .keras."""

import os

import numpy as np
import pytest
import tensorflow as tf

from core import data as D
from core.gradcam import GradCamError, compute_gradcam
from core.models import build_model, build_simple_nn
from core.training import set_global_seed, train_model, validate_config

pytestmark = pytest.mark.slow


def _train(synthetic_images, conv_layers):
    X, y, _ = synthetic_images
    data = D.prepare_image_arrays(X, y, ["top-left", "bottom-right"], seed=0)
    cfg = validate_config({"network_type": "CNN", "dataset": "MNIST", "conv_layers": conv_layers,
                           "base_filters": 8, "epochs": 2, "batch_size": 16, "learning_rate": 0.01})
    set_global_seed(0)
    model = build_model(cfg, data.input_shape, 2)
    train_model(model, data, cfg)
    return model


@pytest.mark.parametrize("conv_layers", [1, 2, 3])
def test_gradcam_shape_range_and_focus(synthetic_images, conv_layers):
    X, _, masks = synthetic_images
    model = _train(synthetic_images, conv_layers)
    ratios = []
    for i in range(24):
        heatmap, cls = compute_gradcam(model, X[i])
        assert heatmap.shape == (28, 28) and heatmap.dtype == np.float32
        assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
        assert cls in (0, 1)
        if heatmap.max() > 0:
            ratios.append(heatmap[masks[i]].mean() / (heatmap[~masks[i]].mean() + 1e-6))
    # The heatmap should mostly light up the object, not the background.
    assert ratios and np.median(ratios) > 1.5


def test_gradcam_accepts_all_input_shapes(synthetic_images):
    X, _, _ = synthetic_images
    model = _train(synthetic_images, 2)
    for img in (X[0, ..., 0], X[0], X[:1]):
        heatmap, _ = compute_gradcam(model, img, class_index=1)
        assert heatmap.shape == (28, 28)


def test_gradcam_after_save_and_reload(synthetic_images, tmp_path):
    X, _, _ = synthetic_images
    model = _train(synthetic_images, 2)
    path = os.path.join(tmp_path, "m.keras")
    model.save(path)
    reloaded = tf.keras.models.load_model(path)
    a, _ = compute_gradcam(model, X[3])
    b, _ = compute_gradcam(reloaded, X[3])
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_gradcam_rejects_dense_only_model():
    with pytest.raises(GradCamError, match="Conv2D"):
        compute_gradcam(build_simple_nn(784, 10), np.zeros((28, 28), np.float32))


def test_gradcam_rejects_batches():
    model = build_model(validate_config({"network_type": "CNN", "dataset": "MNIST", "conv_layers": 1}),
                        (28, 28, 1), 10)
    with pytest.raises(GradCamError):
        compute_gradcam(model, np.zeros((2, 28, 28, 1), np.float32))
