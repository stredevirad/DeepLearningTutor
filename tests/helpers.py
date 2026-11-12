"""Small builders shared by several test modules (no network, no real datasets)."""

import io
import zipfile

import numpy as np
from PIL import Image, ImageDraw


def png_bytes(size=(32, 32), color=255, mode="L", draw=None):
    img = Image.new(mode, size, color=color)
    if draw:
        draw(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def shape_png(kind, background=255, foreground=0, size=40):
    """A light-background image with a dark circle or square (like a photo of a drawing)."""
    def _draw(d):
        if kind == "circle":
            d.ellipse((8, 8, size - 8, size - 8), fill=foreground)
        else:
            d.rectangle((10, 10, size - 10, size - 10), fill=foreground)
    return png_bytes((size, size), color=background, draw=_draw)


def make_zip(entries):
    """entries: dict of archive name -> bytes (or str)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def class_zip(classes=("circle", "square"), per_class=6, root=None):
    entries = {}
    for cls in classes:
        for i in range(per_class):
            prefix = f"{root}/" if root else ""
            entries[f"{prefix}{cls}/{cls}_{i}.png"] = shape_png(cls)
    return make_zip(entries)


def fake_image_loader(n_train=200, n_val=50, num_classes=10, seed=0):
    """Stand-in for tf.keras.datasets.*.load_data returning uint8 arrays."""
    rng = np.random.default_rng(seed)

    def load():
        X_tr = rng.integers(0, 255, size=(n_train, 28, 28), dtype=np.uint8)
        X_va = rng.integers(0, 255, size=(n_val, 28, 28), dtype=np.uint8)
        y_tr = np.arange(n_train) % num_classes
        y_va = np.arange(n_val) % num_classes
        return (X_tr, y_tr.astype(np.uint8)), (X_va, y_va.astype(np.uint8))
    return load
