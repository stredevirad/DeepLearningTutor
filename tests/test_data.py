"""Data loading & preprocessing: built-in datasets, custom CSV, custom ZIP."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from core import data as D
from core.data import DataError
from tests.helpers import class_zip, fake_image_loader, make_zip, png_bytes, shape_png


# ─────────────────────────── built-in tabular ───────────────────────────

@pytest.mark.parametrize("name,n_features,n_classes", [
    ("Iris", 4, 3), ("Wine", 13, 3), ("Breast Cancer", 30, 2), ("Digits", 64, 10),
])
def test_builtin_tabular_split_then_scale(name, n_features, n_classes):
    data = D.prepare_builtin_tabular(name, seed=42)
    assert data.kind == "tabular"
    assert data.X_train.shape[1] == n_features == data.input_shape[0]
    assert data.num_classes == n_classes
    assert len(data.X_val) == int(np.ceil((len(data.X_train) + len(data.X_val)) * 0.2))
    assert data.stratified
    # The scaler was fitted on the training rows only: train columns are exactly
    # standardised, validation columns are not (they were only transformed).
    varying = data.X_train.std(axis=0) > 1e-6
    np.testing.assert_allclose(data.X_train[:, varying].mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(data.X_train[:, varying].std(axis=0), 1, atol=1e-3)
    assert not np.allclose(data.X_val[:, varying].mean(axis=0), 0, atol=1e-5)
    # Raw (unscaled) validation rows are kept for the Test tab and match the scaled ones.
    np.testing.assert_allclose(D.transform_rows(data, data.X_val_raw), data.X_val, rtol=1e-5, atol=1e-5)


def test_stratified_split_keeps_class_proportions():
    data = D.prepare_builtin_tabular("Iris", seed=0)
    assert np.bincount(data.y_val).tolist() == [10, 10, 10]


def test_split_is_reproducible_with_seed():
    a = D.prepare_builtin_tabular("Wine", seed=7)
    b = D.prepare_builtin_tabular("Wine", seed=7)
    c = D.prepare_builtin_tabular("Wine", seed=8)
    np.testing.assert_array_equal(a.y_val, b.y_val)
    assert not np.array_equal(a.y_val, c.y_val)


def test_stratify_falls_back_when_a_class_has_one_sample():
    X = pd.DataFrame({"f": np.arange(20, dtype=float)})
    y = np.array([0] * 10 + [1] * 9 + [2])
    data = D.prepare_tabular(X, y, ["a", "b", "c"])
    assert not data.stratified
    assert any("stratified" in w for w in data.warnings)


def test_unknown_builtin_dataset():
    with pytest.raises(DataError):
        D.load_builtin_tabular("Moons")


# ─────────────────────────── built-in images ───────────────────────────

@pytest.mark.parametrize("name", ["MNIST", "Fashion-MNIST"])
def test_builtin_images_normalised_with_channel(name):
    data = D.prepare_builtin_images(name, loader=fake_image_loader())
    assert data.X_train.shape == (200, 28, 28, 1) and data.X_val.shape == (50, 28, 28, 1)
    assert data.X_train.dtype == np.float32
    assert 0.0 <= data.X_train.min() and data.X_train.max() <= 1.0
    assert data.num_classes == 10
    assert data.image_style == ("digit" if name == "MNIST" else "object")
    if name == "Fashion-MNIST":
        assert data.class_names[9] == "Ankle boot"


def test_builtin_images_subset():
    data = D.prepare_builtin_images("MNIST", loader=fake_image_loader(), max_train_samples=64, max_val_samples=16)
    assert len(data.X_train) == 64 and len(data.X_val) == 16
    assert any("subset" in m for m in data.info)


def test_unknown_image_dataset():
    with pytest.raises(DataError):
        D.load_builtin_images("CIFAR", loader=fake_image_loader())


# ─────────────────────────── custom CSV ───────────────────────────

def _csv(df):
    return df.to_csv(index=False).encode()


def test_csv_numeric_target_last():
    df = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) % 7, "label": ["x", "y"] * 20})
    data = D.prepare_custom_csv(D.read_csv_upload(_csv(df)), "label")
    assert data.class_names == ["x", "y"]
    assert data.feature_columns == ["a", "b"]
    assert data.X_train.shape[1] == 2


def test_csv_target_not_last_column():
    df = pd.DataFrame({"species": ["cat", "dog"] * 15, "w": np.linspace(1, 9, 30), "h": np.linspace(3, 5, 30)})
    data = D.prepare_custom_csv(df, "species")
    assert data.class_names == ["cat", "dog"]
    assert data.feature_columns == ["w", "h"]


def test_csv_non_numeric_columns_one_hot_or_dropped():
    n = 60
    df = pd.DataFrame({
        "id": [f"row-{i}" for i in range(n)],                 # high-cardinality text -> dropped
        "colour": ["red", "green", "blue"] * 20,               # low-cardinality -> one-hot
        "size": np.linspace(0, 1, n),
        "target": [0, 1] * 30,
    })
    data = D.prepare_custom_csv(df, "target")
    assert "id" not in data.feature_columns
    assert "colour" in data.feature_columns and data.categorical_levels["colour"] == ["blue", "green", "red"]
    assert data.X_train.shape[1] == 1 + 3                      # size + 3 one-hot columns
    assert any("id" in w for w in data.warnings)
    assert any("One-hot" in m for m in data.info)
    # A single raw row (as typed in the Test tab) goes through the same fitted transform,
    # and an unseen category is handled (all-zero one-hot) instead of crashing.
    row = D.transform_rows(data, {"colour": "purple", "size": 0.5})
    assert row.shape == (1, 4)


def test_csv_nans_are_counted_not_silent():
    df = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0).astype(object), "t": ["p", "q"] * 15})
    df.loc[[1, 2, 3], "a"] = np.nan
    df.loc[4, "b"] = "oops"                                  # non-numeric value in numeric column
    df.loc[5, "t"] = np.nan                                  # missing target
    data = D.prepare_custom_csv(D.read_csv_upload(_csv(df)), "t")
    assert len(data.y_train) + len(data.y_val) == 30 - 5
    assert any("4 row(s)" in w for w in data.warnings)       # 3 NaN + 1 non-numeric
    assert any("target" in w for w in data.warnings)


def test_csv_wrong_target_column():
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    with pytest.raises(DataError, match="not in the CSV"):
        D.prepare_custom_csv(df, "missing")


def test_csv_regression_like_target_rejected():
    df = pd.DataFrame({"a": np.arange(200.0), "price": np.random.default_rng(0).normal(size=200)})
    with pytest.raises(DataError, match="regression"):
        D.prepare_custom_csv(df, "price")


def test_csv_single_class_rejected():
    df = pd.DataFrame({"a": np.arange(20.0), "t": ["only"] * 20})
    with pytest.raises(DataError, match="one class"):
        D.prepare_custom_csv(df, "t")


def test_csv_no_usable_features():
    df = pd.DataFrame({"name": [f"n{i}" for i in range(30)], "t": ["a", "b"] * 15})
    with pytest.raises(DataError, match="No usable feature"):
        D.prepare_custom_csv(df, "t")


@pytest.mark.parametrize("raw,match", [
    (b"", "empty"),
    (b"\x00\x01\x02 not,a\ncsv", None),
    (b"only_one_column\n1\n2\n3\n4\n5\n6\n", "two columns"),
])
def test_csv_malformed_files(raw, match):
    with pytest.raises(DataError, match=match):
        D.read_csv_upload(raw)


def test_csv_semicolon_separated():
    raw = b"a;b;t\n" + b"".join(f"{i};{i * 2};{'x' if i % 2 else 'y'}\n".encode() for i in range(20))
    assert list(D.read_csv_upload(raw).columns) == ["a", "b", "t"]


def test_csv_too_large():
    with pytest.raises(DataError, match="too large"):
        D.read_csv_upload(b"a,b\n" + b"1,2\n" * 100, max_bytes=50)


# ─────────────────────────── custom ZIP ───────────────────────────

def test_zip_valid_class_folders():
    X, y, class_names, report = D.load_image_zip(class_zip())
    assert class_names == ["circle", "square"]
    assert X.shape == (12, 28, 28, 1) and X.dtype == np.float32
    assert report["class_counts"] == {"circle": 6, "square": 6}
    assert 0.0 <= X.min() and X.max() <= 1.0
    # Light-background images are inverted (object bright, background dark) - the
    # same convention the Test tab uses, so training and prediction match.
    circles = X[y == class_names.index("circle")]
    assert circles[:, :2, :2].mean() < 0.1 and circles.max() > 0.8


def test_zip_single_root_folder_is_stripped():
    _, _, class_names, _ = D.load_image_zip(class_zip(root="my_dataset"))
    assert class_names == ["circle", "square"]


def test_zip_empty_class_is_left_out_with_warning():
    entries = {f"a/{i}.png": shape_png("circle") for i in range(5)}
    entries.update({f"b/{i}.png": shape_png("square") for i in range(5)})
    entries["empty_class/readme.txt"] = "nothing here"
    entries["broken/bad.png"] = b"not really a png"
    X, y, class_names, report = D.load_image_zip(make_zip(entries))
    assert class_names == ["a", "b"]
    assert report["unreadable"] == 1 and report["skipped_non_image"] == 1
    assert any("broken" in w for w in report["warnings"])


def test_zip_needs_two_classes():
    entries = {f"only/{i}.png": shape_png("circle") for i in range(8)}
    with pytest.raises(DataError, match="At least 2 class"):
        D.load_image_zip(make_zip(entries))


def test_zip_bad_files_only():
    entries = {"a/x.png": b"garbage", "b/y.jpg": b"garbage"}
    with pytest.raises(DataError):
        D.load_image_zip(make_zip(entries))


@pytest.mark.parametrize("evil", ["../evil.png", "a/../../evil.png", "/abs/evil.png", "C:/evil.png", "..\\evil.png"])
def test_zip_path_traversal_rejected(evil):
    entries = {f"a/{i}.png": shape_png("circle") for i in range(4)}
    entries.update({f"b/{i}.png": shape_png("square") for i in range(4)})
    entries[evil] = shape_png("circle")
    with pytest.raises(DataError, match="(unsafe|absolute) path"):
        D.load_image_zip(make_zip(entries))


def test_zip_limits_enforced():
    good = class_zip(per_class=6)
    with pytest.raises(DataError, match="limit"):
        D.load_image_zip(good, limits={"max_images": 5})
    with pytest.raises(DataError, match="too large"):
        D.load_image_zip(good, limits={"max_bytes": 100})
    big = make_zip({"a/big.png": png_bytes((600, 600)), "b/1.png": png_bytes()})
    with pytest.raises(DataError):
        D.load_image_zip(big, limits={"max_uncompressed": 10_000})


def test_zip_not_a_zip():
    with pytest.raises(DataError, match="not a valid ZIP"):
        D.load_image_zip(b"PK-but-not-really")


def test_zip_temp_dir_is_cleaned_up(monkeypatch):
    created = []
    real = tempfile.mkdtemp

    def recording_mkdtemp(*a, **k):
        path = real(*a, **k)
        created.append(path)
        return path

    monkeypatch.setattr(D.tempfile, "mkdtemp", recording_mkdtemp)
    D.load_image_zip(class_zip())
    assert created and not os.path.exists(created[0])
    # ...also when loading fails part-way.
    with pytest.raises(DataError):
        D.load_image_zip(make_zip({"only/1.png": shape_png("circle"), "only/2.png": shape_png("circle")}))
    assert not any(os.path.exists(p) for p in created)


def test_prepare_image_arrays_stratified():
    X, y, class_names, _ = D.load_image_zip(class_zip(per_class=10))
    data = D.prepare_image_arrays(X, y, class_names)
    assert data.input_shape == (28, 28, 1)
    assert sorted(np.bincount(data.y_val).tolist()) == [2, 2]


# ─────────────────────────── pre-training summaries ───────────────────────────

def test_describe_csv_reports_roles_and_row_losses():
    df = pd.DataFrame({
        "size": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
        "colour": ["red", "blue", "red", "blue", "red", "blue", "red", "blue"],
        "id": [f"row{i}" for i in range(8)],
        "label": ["a", "b", "a", None, "a", "b", "a", "b"],
    })
    info = D.describe_csv(df, "label")
    assert info["rows"] == 8 and info["columns"] == 4
    assert info["numeric"] == ["size"] and info["categorical"] == ["colour"]
    assert "ID" in info["dropped"]["id"]              # all-unique text is an identifier, even in a tiny file
    assert info["rows_without_target"] == 1 and info["rows_with_bad_numbers"] == 1
    assert info["usable_rows"] == 6 and info["num_classes"] == 2 and not info["regression_like"]


def test_describe_csv_flags_regression_like_target():
    df = pd.DataFrame({"x": np.arange(200.0), "price": np.linspace(0, 1, 200)})
    assert D.describe_csv(df, "price")["regression_like"] is True


def test_zip_summary_counts_without_decoding(monkeypatch):
    import core.data as data_mod
    monkeypatch.setattr(data_mod, "preprocess_image", lambda *a, **k: pytest.fail("must not decode images"))
    entries = {f"root/cats/{i}.png": b"not really a png" for i in range(9)}
    entries.update({f"root/dogs/{i}.jpg": b"x" for i in range(2)})
    entries.update({"root/notes.txt": b"hi", "root/empty/": b"", "root/__MACOSX/cats/._0.png": b""})
    info = D.zip_summary(make_zip(entries))
    assert info["class_counts"] == {"cats": 9, "dogs": 2} and info["images"] == 11
    assert info["empty_classes"] == ["empty"] and info["skipped_non_image"] == 1
    text = " ".join(info["warnings"])
    assert "empty" in text and "non-image" in text and "imbalanced" in text


def test_zip_summary_rejects_unsafe_and_bad_archives():
    with pytest.raises(D.DataError):
        D.zip_summary(make_zip({"../evil/1.png": b"x", "ok/2.png": b"y"}))
    with pytest.raises(D.DataError):
        D.zip_summary(b"definitely not a zip")
    with pytest.raises(D.DataError):
        D.zip_summary(make_zip({"only_text/readme.txt": b"hi"}))
