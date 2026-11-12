"""Uploaded-image preprocessing (corner-based inversion, crop, MNIST-style centring)."""

import io
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.image_preprocessing import (ImageError, background_brightness, open_image, preprocess_image,
                                      style_for_dataset)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = {
    "handwritten_seven.png": ("digit", "7", "MNIST"),
    "fashion_sneaker.png": ("object", "Sneaker", "Fashion-MNIST"),
    "fashion_ankle_boot.png": ("object", "Ankle boot", "Fashion-MNIST"),
}


def _bytes(img, fmt="PNG", **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _check(arr):
    assert arr.shape == (28, 28) and arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


@pytest.mark.parametrize("name", list(SAMPLES))
def test_repo_sample_images(name):
    style = SAMPLES[name][0]
    arr, info = preprocess_image(open_image(os.path.join(ROOT, name)), style=style)
    _check(arr)
    assert not info["blank"]
    # Background ends up dark and the object bright, like MNIST / Fashion-MNIST.
    assert arr[:3, :3].mean() < 0.1 and arr.max() > 0.8


def test_digit_is_centred_in_20px_box():
    img = Image.new("L", (300, 200), 250)
    ImageDraw.Draw(img).rectangle((20, 30, 60, 170), fill=10)   # dark stroke far from centre
    arr, info = preprocess_image(img, "digit")
    assert info["inverted"]
    ys, xs = np.nonzero(arr > 0.2)
    assert ys.min() >= 2 and ys.max() <= 25                      # fits inside the frame
    cy, cx = np.average(np.indices(arr.shape)[0], weights=arr), np.average(np.indices(arr.shape)[1], weights=arr)
    assert abs(cy - 13.5) < 1.5 and abs(cx - 13.5) < 1.5         # centred by mass


def test_dark_background_is_not_inverted():
    img = Image.new("L", (64, 64), 5)
    ImageDraw.Draw(img).ellipse((16, 16, 48, 48), fill=240)
    arr, info = preprocess_image(img, "object")
    assert not info["inverted"] and arr.max() > 0.8


def test_thin_border_frame_is_ignored_when_cropping():
    img = Image.new("L", (200, 200), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 199, 199), outline=120)                  # 1-px grey frame
    d.rectangle((80, 90, 120, 110), fill=0)                      # small object
    arr, _ = preprocess_image(img, "object")
    cols = np.where((arr > 0.3).any(axis=0))[0]
    assert cols.min() <= 1 and cols.max() >= 26                  # object was scaled to fill the frame


@pytest.mark.parametrize("fg", [0, 255])
def test_rgba_transparent_background(fg):
    img = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((20, 20, 60, 60), fill=(fg, fg, fg, 255))
    arr, info = preprocess_image(open_image(_bytes(img)), "object")
    _check(arr)
    assert not info["blank"]
    assert arr[:2, :2].mean() < 0.1                              # background dark, object visible


def test_palette_with_transparency_and_animated_gif():
    frames = [Image.new("P", (40, 40), i) for i in (0, 1)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], transparency=0)
    arr, _ = preprocess_image(open_image(buf.getvalue()), "digit")
    _check(arr)


@pytest.mark.parametrize("size", [(1, 1), (2, 3), (5, 5), (28, 28)])
def test_tiny_images(size):
    arr, info = preprocess_image(Image.new("L", size, 200), "digit")
    _check(arr)
    arr2, _ = preprocess_image(Image.new("RGB", size, (10, 200, 30)), "object")
    _check(arr2)


def test_huge_image_is_downscaled():
    img = Image.new("L", (5000, 3000), 255)
    ImageDraw.Draw(img).ellipse((1000, 500, 4000, 2500), fill=0)
    opened = open_image(_bytes(img))
    assert max(opened.size) <= 1024
    arr, info = preprocess_image(opened, "object")
    _check(arr)
    assert not info["blank"]


def test_decompression_bomb_guard():
    with pytest.raises(ImageError, match="too large"):
        open_image(_bytes(Image.new("L", (50, 50))), max_pixels=100)


def test_sixteen_bit_image():
    arr16 = (np.linspace(0, 65535, 32 * 32).reshape(32, 32)).astype(np.uint16)
    img = Image.fromarray(arr16)
    arr, _ = preprocess_image(open_image(_bytes(img)), "object")
    _check(arr)


@pytest.mark.parametrize("raw", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_corrupt_files(raw):
    with pytest.raises(ImageError):
        open_image(raw)


def test_blank_image_flagged():
    _, info = preprocess_image(Image.new("L", (50, 50), 255), "digit")
    assert info["blank"]


def test_exif_rotation_applied():
    img = Image.new("RGB", (60, 30), "white")
    exif = img.getexif()
    exif[0x0112] = 6                                             # "rotate 90° CW" orientation tag
    opened = open_image(_bytes(img, "JPEG", exif=exif.tobytes()))
    assert opened.size == (30, 60)


def test_background_brightness_uses_corners():
    g = np.full((80, 80), 250.0)
    g[20:60, 20:60] = 0                                          # big dark centre must not fool it
    assert background_brightness(g) > 200


def test_style_for_dataset():
    assert style_for_dataset("MNIST") == "digit"
    assert style_for_dataset("Fashion-MNIST") == "object"
    assert style_for_dataset("my_custom.zip") == "object"


# ─────────── Sensible predictions on the repo samples (uses locally cached datasets) ───────────

def _cached(name):
    base = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
    return os.path.exists(os.path.join(base, name))


@pytest.mark.slow
@pytest.mark.parametrize("dataset,cache_name", [("MNIST", "mnist.npz"), ("Fashion-MNIST", "fashion-mnist")])
def test_sample_predictions_are_sensible(dataset, cache_name):
    if not _cached(cache_name):
        pytest.skip(f"{dataset} is not cached locally (tests never download data)")
    from core.training import run_experiment
    _, model, data = run_experiment({"network_type": "CNN", "dataset": dataset, "epochs": 2, "conv_layers": 2,
                                     "base_filters": 16, "batch_size": 64},
                                    max_train_samples=20_000, max_val_samples=2_000, return_model=True)
    for name, (style, expected, ds) in SAMPLES.items():
        if ds != dataset:
            continue
        arr, _ = preprocess_image(open_image(os.path.join(ROOT, name)), style=style_for_dataset(dataset))
        probs = model.predict(arr[None, ..., None], verbose=0)[0]
        ranked = [data.class_names[i] for i in np.argsort(probs)[::-1]]
        if name == "fashion_sneaker.png":
            # Outline (line-art) drawing: out of distribution for Fashion-MNIST photos,
            # so only require the right class to be among the top guesses.
            assert expected in ranked[:3], ranked[:3]
        else:
            assert ranked[0] == expected, ranked[:3]
