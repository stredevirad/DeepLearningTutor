"""
Model Construction
==================
Builds the Simple NN (dense) and CNN architectures used everywhere in the app.
Models are Keras Sequential models that start with an explicit `Input`, so they
are fully built (and have known shapes) as soon as they are created.
"""

import os
import shutil
import tempfile

import tensorflow as tf

OPTIMIZERS = ("Adam", "SGD", "RMSprop")
CNN_DENSE_UNITS = 64


def build_optimizer(name, learning_rate):
    lr = float(learning_rate)
    if name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Unknown optimizer '{name}'. Choose one of: {', '.join(OPTIMIZERS)}.")


def build_simple_nn(input_dim, num_classes, hidden_layers=2, nodes_per_layer=16, dropout_rate=0.0):
    """Fully-connected network for tabular data: [Dense(relu) (+Dropout)] x N -> Dense(softmax)."""
    layers = [tf.keras.Input(shape=(int(input_dim),))]
    for _ in range(int(hidden_layers)):
        layers.append(tf.keras.layers.Dense(int(nodes_per_layer), activation="relu"))
        if dropout_rate > 0:
            layers.append(tf.keras.layers.Dropout(float(dropout_rate)))
    layers.append(tf.keras.layers.Dense(int(num_classes), activation="softmax"))
    return tf.keras.Sequential(layers, name="simple_nn")


def build_cnn(input_shape, num_classes, conv_layers=2, base_filters=16, dropout_rate=0.0):
    """Convolutional network for 28x28 images:
    [Conv2D(3x3, relu, same) -> MaxPool(2x2)] x N (filters double each block)
    -> Flatten -> Dense(64, relu) (+Dropout) -> Dense(softmax)."""
    layers = [tf.keras.Input(shape=tuple(input_shape))]
    filters = int(base_filters)
    for _ in range(int(conv_layers)):
        layers.append(tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
        filters *= 2
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(CNN_DENSE_UNITS, activation="relu"))
    if dropout_rate > 0:
        layers.append(tf.keras.layers.Dropout(float(dropout_rate)))
    layers.append(tf.keras.layers.Dense(int(num_classes), activation="softmax"))
    return tf.keras.Sequential(layers, name="cnn")


def build_model(cfg, input_shape, num_classes):
    """Build and compile the model described by a validated config dict."""
    if cfg["network_type"] == "Simple NN":
        model = build_simple_nn(input_shape[0], num_classes, cfg["hidden_layers"],
                                cfg["nodes_per_layer"], cfg["dropout_rate"])
    else:
        model = build_cnn(input_shape, num_classes, cfg["conv_layers"],
                          cfg["base_filters"], cfg["dropout_rate"])
    model.compile(optimizer=build_optimizer(cfg["optimizer"], cfg["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def count_parameters(model):
    trainable = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def summary_text(model):
    lines = []
    model.summary(print_fn=lambda x, **_: lines.append(x))
    return "\n".join(lines)


def model_to_bytes(model):
    """Serialise a Keras model to .keras bytes (private temp dir, always cleaned up)."""
    tmp_dir = tempfile.mkdtemp(prefix="dlt_model_")
    try:
        path = os.path.join(tmp_dir, "trained_model.keras")
        model.save(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def model_from_bytes(data):
    """Load a model saved by model_to_bytes (or any .keras file's bytes)."""
    tmp_dir = tempfile.mkdtemp(prefix="dlt_model_")
    try:
        path = os.path.join(tmp_dir, "model.keras")
        with open(path, "wb") as f:
            f.write(data)
        return tf.keras.models.load_model(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
