"""
Grad-CAM (Selvaraju et al., 2017)
=================================
Shows which parts of an image the CNN relied on for a prediction.

Why not `tf.keras.Model(inputs=model.inputs, outputs=[last_conv.output, model.output])`?
Under Keras 3 that fails for Sequential models ("The layer sequential has never
been called and thus has no defined output"), especially after the model has
been stored in `st.session_state`. Instead the forward pass is replayed layer by
layer inside a GradientTape, capturing the last Conv2D activation on the way.
This works for any linear stack of layers (all models built by this app) with
1-4 conv layers, whether freshly trained or reloaded from a .keras file.

The class score is taken BEFORE the final softmax (the logit). Gradients of a
saturated softmax (e.g. 100% confidence) vanish and give an empty heatmap.
"""

import numpy as np
import tensorflow as tf


class GradCamError(ValueError):
    """Grad-CAM cannot be computed for this model."""


def _as_batch(image):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:          # (H, W)
        x = x[np.newaxis, ..., np.newaxis]
    elif x.ndim == 3:        # (H, W, C)
        x = x[np.newaxis, ...]
    if x.ndim != 4 or x.shape[0] != 1:
        raise GradCamError(f"Expected a single image, got an array of shape {np.shape(image)}.")
    return tf.convert_to_tensor(x)


def _layers_in_order(model):
    layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    if not layers:
        raise GradCamError("The model has no layers.")
    return layers


def compute_gradcam(model, image, class_index=None):
    """Return (heatmap, class_index) where heatmap is a float32 array with the
    image's height x width, values in [0, 1].

    `image` is one preprocessed image: (28, 28), (28, 28, 1) or (1, 28, 28, 1).
    `class_index` defaults to the model's predicted class.
    """
    x = _as_batch(image)
    layers = _layers_in_order(model)
    conv_positions = [i for i, layer in enumerate(layers) if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_positions:
        raise GradCamError("Grad-CAM needs a convolutional (Conv2D) layer; this model has none.")
    last_conv = conv_positions[-1]

    last = layers[-1]
    split_softmax = (isinstance(last, tf.keras.layers.Dense)
                     and getattr(last.activation, "__name__", "") == "softmax")

    with tf.GradientTape() as tape:
        h = x
        conv_out = None
        for i, layer in enumerate(layers):
            if i == len(layers) - 1 and split_softmax:
                logits = tf.matmul(h, last.kernel)
                if last.use_bias:
                    logits = logits + last.bias
                h = logits
            else:
                h = layer(h, training=False)
            if i == last_conv:
                conv_out = h
                tape.watch(conv_out)
        scores = h                                  # logits (or model output if not softmax-Dense)
        if class_index is None:
            class_index = int(tf.argmax(scores[0]))
        target = scores[:, int(class_index)]

    grads = tape.gradient(target, conv_out)
    if grads is None:
        raise GradCamError("Could not compute gradients for this model.")

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))                   # one weight per feature map
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))    # (h, w)
    cam = cam.numpy().astype(np.float32)

    peak = float(cam.max())
    if not np.isfinite(peak) or peak <= 1e-12:
        # Nothing positively supports this class at the last conv layer.
        cam = np.zeros_like(cam)
    else:
        cam = cam / peak

    height, width = int(x.shape[1]), int(x.shape[2])
    resized = tf.image.resize(cam[..., np.newaxis], (height, width), method="bilinear").numpy()[..., 0]
    return np.clip(resized, 0.0, 1.0).astype(np.float32), int(class_index)
