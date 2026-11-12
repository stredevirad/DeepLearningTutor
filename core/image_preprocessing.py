"""
Image Preprocessing
===================
Turns an arbitrary user image (photo, scan, screenshot, transparent PNG) into
the 28x28 grayscale format the CNNs are trained on: bright object on a dark
background, values in [0, 1].

The SAME function is used for custom-ZIP training images and for images
uploaded in the Test tab, so training and prediction see identically
prepared inputs.

Styles:
  * "digit"  - MNIST-like: threshold, crop to the stroke, fit into a 20x20 box
               and centre it by centre of mass inside 28x28 (how MNIST was made).
  * "object" - Fashion-MNIST / custom datasets: keep texture, crop to the
               object and scale it to fill the 28x28 frame.
"""

import io

import numpy as np
from PIL import Image, ImageOps

MAX_PIXELS = 40_000_000      # refuse decompression bombs (~40 megapixels)
WORKING_MAX_SIDE = 1024      # huge photos are downscaled before any maths
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


class ImageError(ValueError):
    """The file is not a usable image (corrupt, unsupported, or too large)."""


def open_image(source, max_pixels=MAX_PIXELS):
    """Open an image from a path, bytes or file-like object, safely.

    Checks the pixel count from the header before decoding, applies the EXIF
    orientation, takes the first frame of animated images, flattens any
    transparency, and downscales very large images.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        img = Image.open(source)
    except Exception as e:  # PIL raises many different types for bad files
        raise ImageError("This file could not be read as an image.") from e

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ImageError("The image has no pixels.")
    if w * h > max_pixels:
        raise ImageError(f"The image is too large ({w}x{h} pixels). Please upload a smaller image.")

    try:
        img.seek(0)            # first frame of GIF/WebP animations
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ImageError("The image file appears to be damaged.") from e

    img = _flatten_transparency(img)
    if max(img.size) > WORKING_MAX_SIDE:
        img = img.copy()
        img.thumbnail((WORKING_MAX_SIDE, WORKING_MAX_SIDE), Image.LANCZOS)
    return img


def _flatten_transparency(img):
    """Composite transparent images onto a background that contrasts with the
    visible content (black behind light objects, white behind dark ones)."""
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if not has_alpha:
        return img
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32)
    alpha = arr[..., 3] / 255.0
    luminance = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    opaque = alpha > 0.5
    content_brightness = float(luminance[opaque].mean()) if opaque.any() else 0.0
    bg_value = 0 if content_brightness > 127 else 255
    background = Image.new("RGBA", rgba.size, (bg_value, bg_value, bg_value, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def to_grayscale_array(img):
    """Grayscale float32 array in 0..255 for any PIL mode (incl. 16-bit and float)."""
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(img).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        return np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo) * 255.0
    return np.asarray(img.convert("L")).astype(np.float32)


def background_brightness(gray):
    """Mean brightness of the four corner patches (corners are almost always background)."""
    h, w = gray.shape
    k = max(1, min(h, w) // 8)
    corners = np.concatenate([
        gray[:k, :k].ravel(), gray[:k, -k:].ravel(),
        gray[-k:, :k].ravel(), gray[-k:, -k:].ravel(),
    ])
    return float(corners.mean())


def _resize(gray, size):
    return np.asarray(
        Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    ).astype(np.float32)


def _content_bbox(gray, rel_threshold):
    """Bounding box (r0, r1, c0, c1) of pixels brighter than rel_threshold * max,
    ignoring a thin outer border (screenshots and exports often have a 1-px frame)."""
    peak = gray.max()
    if peak <= 0:
        return None
    mask = gray > peak * rel_threshold
    b = max(1, int(round(0.01 * min(gray.shape)))) if min(gray.shape) > 8 else 0
    if b:
        mask[:b, :] = mask[-b:, :] = False
        mask[:, :b] = mask[:, -b:] = False
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not (rows.any() and cols.any()):
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1), int(c0), int(c1)


def _digit_layout(gray):
    """MNIST-style: threshold, crop to the digit, fit into 20x20, centre by mass in 28x28."""
    thresholded = gray.copy()
    if thresholded.max() > 0:
        thresholded[thresholded < thresholded.max() * 0.3] = 0

    bbox = _content_bbox(thresholded, 0.0)
    if bbox:
        r0, r1, c0, c1 = bbox
        margin = max(2, int(0.05 * max(r1 - r0, c1 - c0)))
        r0, r1 = max(0, r0 - margin), min(thresholded.shape[0] - 1, r1 + margin)
        c0, c1 = max(0, c0 - margin), min(thresholded.shape[1] - 1, c1 + margin)
        thresholded = thresholded[r0:r1 + 1, c0:c1 + 1]

    h, w = thresholded.shape
    if h > w:
        pad = (h - w) // 2
        thresholded = np.pad(thresholded, ((0, 0), (pad, h - w - pad)))
    elif w > h:
        pad = (w - h) // 2
        thresholded = np.pad(thresholded, ((pad, w - h - pad), (0, 0)))

    digit = _resize(thresholded, (20, 20))
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = digit

    # Shift so the centre of mass sits at the image centre, as in MNIST.
    total = canvas.sum()
    if total > 0:
        ys, xs = np.indices(canvas.shape)
        cy, cx = (ys * canvas).sum() / total, (xs * canvas).sum() / total
        dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
        dy, dx = int(np.clip(dy, -4, 4)), int(np.clip(dx, -4, 4))
        canvas = np.roll(np.roll(canvas, dy, axis=0), dx, axis=1)
    return canvas


def _object_layout(gray):
    """Fashion-MNIST-style: crop to the object, pad to a square and scale it to
    fill the 28x28 frame (Fashion-MNIST items touch the frame on their long side).
    Texture is kept - no thresholding of the pixels themselves."""
    bbox = _content_bbox(gray, 0.15)
    if bbox:
        r0, r1, c0, c1 = bbox
        gray = gray[r0:r1 + 1, c0:c1 + 1]
    h, w = gray.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = gray
    return _resize(square, (28, 28))


def preprocess_image(img, style="object"):
    """Convert a PIL image into a (28, 28) float32 array in [0, 1].

    Returns (array, info) where info has:
      inverted  - True if a light background was inverted to dark
      blank     - True if the result has (almost) no content
    """
    if style not in ("digit", "object"):
        raise ValueError(f"Unknown preprocessing style: {style!r}")
    gray = to_grayscale_array(img)
    # No contrast in the original image (e.g. an empty white page) -> nothing to classify.
    low_contrast = float(gray.max() - gray.min()) < 12.0

    bg = background_brightness(gray)
    inverted = bg > 127
    if inverted:
        gray = 255.0 - gray

    if style == "digit":
        out = _digit_layout(gray)
    else:
        out = _object_layout(gray)

    out = np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)
    blank = bool(low_contrast or out.max() < 0.1)
    return out, {"inverted": bool(inverted), "blank": blank, "background_brightness": round(bg, 1)}


def style_for_dataset(dataset_name):
    """Which preprocessing style a trained model expects."""
    return "digit" if dataset_name == "MNIST" else "object"
