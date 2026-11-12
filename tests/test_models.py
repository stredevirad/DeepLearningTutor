"""Model building: output shapes and parameter counts across a range of settings."""

import pytest
import tensorflow as tf

from core.models import build_cnn, build_model, build_optimizer, build_simple_nn
from core.training import ConfigError, validate_config


def expected_dense_params(d, hidden, k, c):
    return (d * k + k) + (hidden - 1) * (k * k + k) + (k * c + c)


@pytest.mark.parametrize("hidden", [1, 2, 3, 5])
@pytest.mark.parametrize("nodes", [2, 16, 128])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_simple_nn_shapes_and_params(hidden, nodes, dropout):
    model = build_simple_nn(4, 3, hidden, nodes, dropout)
    assert model.output_shape == (None, 3)
    assert model.count_params() == expected_dense_params(4, hidden, nodes, 3)
    n_dropout = sum(isinstance(l, tf.keras.layers.Dropout) for l in model.layers)
    assert n_dropout == (hidden if dropout > 0 else 0)


def expected_cnn_params(conv_layers, f, classes=10, dense=64):
    total, in_ch, side = 0, 1, 28
    for _ in range(conv_layers):
        total += 3 * 3 * in_ch * f + f
        in_ch, f, side = f, f * 2, side // 2
    flat = side * side * in_ch
    return total + (flat * dense + dense) + (dense * classes + classes)


@pytest.mark.parametrize("conv_layers", [1, 2, 3, 4])
@pytest.mark.parametrize("filters", [8, 16, 32])
@pytest.mark.parametrize("dropout", [0.0, 0.25])
def test_cnn_shapes_and_params(conv_layers, filters, dropout):
    model = build_cnn((28, 28, 1), 10, conv_layers, filters, dropout)
    assert model.output_shape == (None, 10)
    assert model.count_params() == expected_cnn_params(conv_layers, filters)
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    assert [c.filters for c in convs] == [filters * 2 ** i for i in range(conv_layers)]


def test_models_are_built_without_being_called():
    """The Input layer makes shapes known immediately (no 'never been called' surprises)."""
    model = build_cnn((28, 28, 1), 10, 2, 8, 0.0)
    assert model.built and model.input_shape == (None, 28, 28, 1)


@pytest.mark.parametrize("name,cls", [("Adam", tf.keras.optimizers.Adam), ("SGD", tf.keras.optimizers.SGD),
                                      ("RMSprop", tf.keras.optimizers.RMSprop)])
def test_optimizers(name, cls):
    opt = build_optimizer(name, 0.01)
    assert isinstance(opt, cls)
    assert float(opt.learning_rate) == pytest.approx(0.01)


def test_unknown_optimizer():
    with pytest.raises(ValueError):
        build_optimizer("Nadam", 0.01)


def test_build_model_compiles_from_config():
    cfg = validate_config({"network_type": "CNN", "dataset": "MNIST", "conv_layers": 3, "base_filters": 8,
                           "optimizer": "RMSprop", "learning_rate": 0.01, "dropout_rate": 0.2})
    model = build_model(cfg, (28, 28, 1), 10)
    assert model.optimizer is not None and model.loss == "sparse_categorical_crossentropy"


def test_too_many_conv_layers_rejected():
    with pytest.raises(ConfigError, match="conv_layers"):
        validate_config({"network_type": "CNN", "dataset": "MNIST", "conv_layers": 5})
