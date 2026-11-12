"""
Dataset Loading & Preprocessing
===============================
One leakage-free pipeline shared by the app, the agent tools and the MCP server:

    1. split the data into train / validation / test (stratified by class),
    2. fit the preprocessing (StandardScaler, one-hot encoder) on the TRAIN split only,
    3. apply that fitted preprocessing to the other splits and to live predictions.

What each split is for:
    train        the model learns from these rows;
    validation   drives the live curves and early stopping (so it influences the model);
    test         evaluated once, after training, for the reported accuracy. Nothing is
                 ever chosen by looking at it.

Also contains the custom CSV analysis and the safe custom-ZIP image loader.
"""

import io
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.image_preprocessing import IMAGE_EXTENSIONS, ImageError, open_image, preprocess_image

TABULAR_DATASETS = ("Iris", "Wine", "Breast Cancer", "Digits")
IMAGE_DATASETS = ("MNIST", "Fashion-MNIST")
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Tables and custom uploads: a stratified 70 / 15 / 15 split.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# Too little data for three parts: the older two-way split (80 % train / 20 % validation).
FALLBACK_VAL_FRACTION = 0.2
# MNIST / Fashion-MNIST: one tenth of the official 60,000 training images (6,000) becomes
# the validation set, 54,000 remain for training; the official 10,000 test images are the test set.
IMAGE_VAL_DIVISOR = 10
DEFAULT_SEED = 42
NO_TEST_SET_NOTE = ("The dataset is too small for a separate test set (every class needs examples in the "
                    "training, validation and test parts), so it was split into training and validation only "
                    "(80/20). The reported accuracy is a validation accuracy.")

# --- Custom CSV limits ---
MAX_CSV_BYTES = 25 * 1024 * 1024
MAX_CATEGORY_LEVELS = 15          # text columns with <= this many values are one-hot encoded
NUMERIC_PARSE_RATIO = 0.95        # a text column is numeric if >= 95% of values parse as numbers
MAX_CLASSES_WARN = 20
MAX_CLASSES = 50

# --- Custom ZIP limits ---
MAX_ZIP_BYTES = 100 * 1024 * 1024
MAX_ZIP_MEMBERS = 20_000
MAX_ZIP_IMAGES = 10_000
MAX_ZIP_UNCOMPRESSED = 300 * 1024 * 1024
MAX_ZIP_MEMBER_BYTES = 25 * 1024 * 1024
MIN_IMAGES_TOTAL = 6


class DataError(ValueError):
    """A problem with the user's data that should be shown as a friendly message."""


@dataclass
class PreparedData:
    """Everything training, evaluation and the Test tab need about a dataset."""
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    class_names: list
    input_shape: tuple
    kind: str                                   # "tabular" or "image"
    preprocessor: object = None                 # fitted on X_train only (tabular data)
    feature_columns: list = field(default_factory=list)      # raw input columns, in order
    categorical_levels: dict = field(default_factory=dict)   # column -> known categories
    X_val_raw: object = None                    # un-scaled validation rows (DataFrame) for the Test tab
    feature_defaults: dict = field(default_factory=dict)     # typical value per raw column
    info: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)
    stratified: bool = True
    image_style: str = None                     # "digit" / "object" for image models
    X_test: np.ndarray = None                   # held-out test split; None when the data is too small for one
    y_test: np.ndarray = None
    X_test_raw: object = None                   # un-scaled test rows (DataFrame) for the Test tab
    split_indices: dict = field(default_factory=dict)        # "train" / "val" / "test" -> row numbers in the source data

    @property
    def num_classes(self):
        return len(self.class_names)

    @property
    def has_test(self):
        return self.y_test is not None and len(self.y_test) > 0


# ═══════════════════════════════════════════════
# Splitting
# ═══════════════════════════════════════════════

def three_way_split(y, seed=DEFAULT_SEED):
    """Row numbers for a stratified train / validation / test split (70 / 15 / 15).

    Returns (train, val, test) index arrays, or None when the data is too small for
    every class to appear in all three parts. The two train_test_split calls are the
    same two calls the exported script makes (core/codegen.py), so that script
    reproduces exactly these rows.
    """
    y = np.asarray(y)
    rows = np.arange(len(y))
    # Whole numbers of rows (15 % each), so no floating-point rounding decides a split size.
    n_test, n_val = round(TEST_FRACTION * len(y)), round(VAL_FRACTION * len(y))
    try:
        rest, test = train_test_split(rows, test_size=n_test, random_state=seed, stratify=y)
        train, val = train_test_split(rest, test_size=n_val, random_state=seed, stratify=y[rest])
    except ValueError:          # scikit-learn: a class has too few members, or a part would be empty
        return None
    n_classes = len(np.unique(y))
    if any(len(np.unique(y[part])) < n_classes for part in (train, val, test)):
        return None
    return train, val, test


def split_rows(y, seed=DEFAULT_SEED):
    """Decide which rows go into which split.

    Returns (parts, stratified, warnings). `parts` maps "train", "val" and - unless the
    data is too small - "test" to arrays of row numbers that never overlap.
    """
    three = three_way_split(y, seed)
    if three is not None:
        return {"train": three[0], "val": three[1], "test": three[2]}, True, []
    rows = np.arange(len(y))
    train, val, _, _, stratified, warning = stratified_split(rows, y, seed=seed)
    return {"train": train, "val": val}, stratified, [NO_TEST_SET_NOTE] + ([warning] if warning else [])


def stratified_split(X, y, seed=DEFAULT_SEED, test_size=FALLBACK_VAL_FRACTION):
    """Stratified train/validation split, falling back to a plain random split
    when stratification is impossible (a class with < 2 samples, or fewer
    validation slots than classes). Returns (X_tr, X_val, y_tr, y_val, stratified, warning)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_val = int(np.ceil(len(y) * test_size))
    warning = None
    if counts.min() < 2:
        warning = ("Some classes have only 1 example, so the validation split could not be "
                   "stratified (class proportions may differ between train and validation).")
    elif n_val < len(classes) or len(y) - n_val < len(classes):
        warning = ("The dataset is too small to give every class a place in both splits, "
                   "so a plain random split was used.")
    if warning is None:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        return X_tr, X_val, y_tr, y_val, True, None
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_tr, X_val, y_tr, y_val, False, warning


# ═══════════════════════════════════════════════
# Tabular data
# ═══════════════════════════════════════════════

def load_builtin_tabular(name):
    """Return (features DataFrame, integer labels, class names) for a built-in sklearn dataset."""
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
    loaders = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer, "Digits": load_digits}
    if name not in loaders:
        raise DataError(f"Unknown dataset '{name}'. Choose one of: {', '.join(TABULAR_DATASETS)}.")
    data = loaders[name]()
    X = pd.DataFrame(data.data, columns=[str(c) for c in data.feature_names])
    class_names = [str(c) for c in data.target_names]
    return X, np.asarray(data.target, dtype=np.int64), class_names


def make_tabular_preprocessor(numeric_columns, categorical_columns):
    """StandardScaler for numeric columns + one-hot encoding for categorical columns.
    Must only ever be fitted on training rows."""
    transformers = []
    if numeric_columns:
        transformers.append(("num", StandardScaler(), list(numeric_columns)))
    if categorical_columns:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                             list(categorical_columns)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def prepare_tabular(X_df, y, class_names, seed=DEFAULT_SEED, categorical_columns=()):
    """Split FIRST, then fit the preprocessing on the training rows only."""
    categorical_columns = [c for c in categorical_columns if c in X_df.columns]
    numeric_columns = [c for c in X_df.columns if c not in categorical_columns]
    y = np.asarray(y, dtype=np.int64)

    parts, stratified, split_warnings = split_rows(y, seed)
    X_tr_df, X_val_df = X_df.iloc[parts["train"]], X_df.iloc[parts["val"]]
    X_test_df = X_df.iloc[parts["test"]] if "test" in parts else None
    preprocessor = make_tabular_preprocessor(numeric_columns, categorical_columns)
    X_train = preprocessor.fit_transform(X_tr_df).astype(np.float32)   # fit on train only
    X_val = preprocessor.transform(X_val_df).astype(np.float32)        # transform validation
    X_test = preprocessor.transform(X_test_df).astype(np.float32) if X_test_df is not None else None

    levels = {c: sorted(str(v) for v in X_tr_df[c].dropna().unique()) for c in categorical_columns}
    defaults = {}
    for c in X_df.columns:
        if c in categorical_columns:
            defaults[c] = str(X_tr_df[c].mode().iloc[0]) if len(X_tr_df[c].mode()) else ""
        else:
            defaults[c] = float(X_tr_df[c].median())

    data = PreparedData(
        X_train=X_train, X_val=X_val, y_train=y[parts["train"]], y_val=y[parts["val"]],
        class_names=list(class_names), input_shape=(X_train.shape[1],), kind="tabular",
        preprocessor=preprocessor, feature_columns=list(X_df.columns), categorical_levels=levels,
        X_val_raw=X_val_df.reset_index(drop=True), feature_defaults=defaults,
        class_counts=_class_counts(y, class_names), stratified=stratified,
        X_test=X_test, y_test=y[parts["test"]] if "test" in parts else None,
        X_test_raw=X_test_df.reset_index(drop=True) if X_test_df is not None else None,
        split_indices=parts,
    )
    data.warnings.extend(split_warnings)
    return data


def prepare_builtin_tabular(name, seed=DEFAULT_SEED):
    X, y, class_names = load_builtin_tabular(name)
    return prepare_tabular(X, y, class_names, seed=seed)


def transform_rows(data_or_preprocessor, rows):
    """Apply the TRAIN-fitted preprocessing to raw rows (dict, list of dicts or DataFrame)."""
    pre = getattr(data_or_preprocessor, "preprocessor", data_or_preprocessor)
    if isinstance(rows, dict):
        rows = pd.DataFrame([rows])
    elif isinstance(rows, list):
        rows = pd.DataFrame(rows)
    return pre.transform(rows).astype(np.float32)


def _class_counts(y, class_names):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=len(class_names))
    return {str(class_names[i]): int(counts[i]) for i in range(len(class_names))}


# --- Custom CSV ---

def read_csv_upload(raw_bytes, max_bytes=MAX_CSV_BYTES):
    """Parse uploaded CSV bytes into a DataFrame with friendly errors."""
    if raw_bytes is None or len(raw_bytes) == 0:
        raise DataError("The CSV file is empty.")
    if len(raw_bytes) > max_bytes:
        raise DataError(f"The CSV file is too large ({len(raw_bytes) / 1e6:.1f} MB). "
                        f"The limit is {max_bytes / 1e6:.0f} MB.")
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            for sep in (";", "\t", "|"):   # European / tab / pipe separated files
                if df.shape[1] >= 2:
                    break
                alt = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=sep)
                if alt.shape[1] >= 2:
                    df = alt
            break
        except UnicodeDecodeError as e:
            last_error = e
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, TypeError) as e:
            raise DataError("The file could not be read as a CSV table. "
                            "Check that it is a comma-separated file with a header row.") from e
    else:
        raise DataError("The CSV file uses an unsupported text encoding.") from last_error

    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise DataError("The CSV needs at least two columns: one or more features and a target column.")
    if df.columns.duplicated().any():
        raise DataError("The CSV has duplicate column names. Please give every column a unique name.")
    if len(df) < MIN_IMAGES_TOTAL:
        raise DataError(f"The CSV has only {len(df)} rows; at least {MIN_IMAGES_TOTAL} are needed.")
    return df


def analyse_csv(df, target_column):
    """Decide how every column will be used. Returns a plan dict (no fitting happens here)."""
    if target_column not in df.columns:
        raise DataError(f"Target column '{target_column}' is not in the CSV.")

    plan = {"target": target_column, "numeric": [], "categorical": [], "dropped": {}}
    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        if series.isna().all():
            plan["dropped"][col] = "empty column"
            continue
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            plan["numeric"].append(col)
            continue
        non_null = series.dropna()
        parsed = pd.to_numeric(non_null, errors="coerce")
        distinct = non_null.astype(str).nunique()
        if parsed.notna().mean() >= NUMERIC_PARSE_RATIO:
            plan["numeric"].append(col)
        elif distinct == len(non_null) and len(non_null) >= MIN_IMAGES_TOTAL:
            # Every row has its own value: an identifier or name, not a category to learn from.
            plan["dropped"][col] = "every row has a different value (looks like an ID or a name)"
        elif distinct <= MAX_CATEGORY_LEVELS:
            plan["categorical"].append(col)
        else:
            plan["dropped"][col] = f"text with {distinct} different values (IDs, names, dates, free text)"
    return plan


def describe_csv(df, target_column):
    """What training would do with this CSV, computed without fitting anything:
    column roles, rows that will be removed and the classes found in the target."""
    plan = analyse_csv(df, target_column)
    target = df[target_column]
    no_target = int(target.isna().sum())
    work = df[target.notna()]
    bad_numeric = 0
    if plan["numeric"]:
        numeric = work[plan["numeric"]].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        bad_numeric = int(numeric.isna().any(axis=1).sum())
    y = work[target_column]
    regression_like = bool(pd.api.types.is_numeric_dtype(y) and y.nunique() > MAX_CLASSES)
    counts = y.astype(str).value_counts()
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "numeric": plan["numeric"], "categorical": plan["categorical"], "dropped": plan["dropped"],
        "rows_without_target": no_target, "rows_with_bad_numbers": bad_numeric,
        "usable_rows": int(len(work) - bad_numeric),
        "num_classes": int(len(counts)), "regression_like": regression_like,
        "class_counts": {str(k): int(v) for k, v in counts.head(MAX_CLASSES_WARN).items()},
        "imbalance_warning": None if regression_like else _imbalance_warning(dict(counts.items())),
    }


def prepare_custom_csv(df, target_column, seed=DEFAULT_SEED):
    """Clean a user CSV and return PreparedData (split first, preprocessing fitted on train)."""
    plan = analyse_csv(df, target_column)
    info, warnings = [], []

    work = df.copy()
    n_start = len(work)
    work = work[work[target_column].notna()]
    n_no_target = n_start - len(work)
    if n_no_target:
        warnings.append(f"Removed {n_no_target} row(s) with no value in the target column '{target_column}'.")

    for col in plan["numeric"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype(float)
    for col in plan["categorical"]:
        work[col] = work[col].astype(str).where(work[col].notna(), "(missing)")

    if plan["dropped"]:
        details = "; ".join(f"**{c}** ({why})" for c, why in plan["dropped"].items())
        warnings.append(f"Ignored {len(plan['dropped'])} column(s) that cannot be used as numeric features: {details}.")
    if plan["categorical"]:
        info.append("One-hot encoded categorical column(s): "
                    + ", ".join(f"**{c}**" for c in plan["categorical"]) + ".")

    feature_cols = plan["numeric"] + plan["categorical"]
    if not feature_cols:
        raise DataError("No usable feature columns remain after removing non-numeric columns.")

    n_before = len(work)
    work = work.dropna(subset=plan["numeric"]) if plan["numeric"] else work
    n_nan = n_before - len(work)
    if n_nan:
        warnings.append(f"Removed {n_nan} row(s) with missing or non-numeric values in numeric feature columns "
                        f"({n_nan / max(n_before, 1):.0%} of the data).")
    if len(work) < MIN_IMAGES_TOTAL:
        raise DataError(f"Only {len(work)} usable rows remain after cleaning; at least {MIN_IMAGES_TOTAL} are needed.")

    y_raw = work[target_column]
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > MAX_CLASSES:
        raise DataError(
            f"The target column '{target_column}' has {y_raw.nunique()} different numeric values. "
            "It looks like a regression target (a continuous number), but this app trains "
            "classification models. Pick a column with a small number of categories.")
    labels = y_raw.astype(str) if not pd.api.types.is_numeric_dtype(y_raw) else y_raw
    class_values = sorted(labels.unique(), key=lambda v: (isinstance(v, str), v))
    class_names = [_format_class_name(v) for v in class_values]
    if len(class_names) < 2:
        raise DataError(f"The target column '{target_column}' has only one class; at least 2 are needed.")
    if len(class_names) > MAX_CLASSES:
        raise DataError(f"The target column has {len(class_names)} classes; the limit is {MAX_CLASSES}.")
    if len(class_names) > MAX_CLASSES_WARN:
        warnings.append(f"The target has {len(class_names)} classes. Check that this really is a category column.")
    index = {v: i for i, v in enumerate(class_values)}
    y = labels.map(index).to_numpy(dtype=np.int64)

    X_df = work[feature_cols].reset_index(drop=True)
    data = prepare_tabular(X_df, y, class_names, seed=seed, categorical_columns=plan["categorical"])
    data.info = info + data.info
    data.warnings = warnings + data.warnings
    imbalance = _imbalance_warning(data.class_counts)
    if imbalance:
        data.warnings.append(imbalance)
    return data


def _format_class_name(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _imbalance_warning(class_counts):
    counts = [c for c in class_counts.values() if c > 0]
    if len(counts) >= 2 and max(counts) >= 3 * min(counts):
        return (f"Classes are imbalanced (largest class has {max(counts)} examples, smallest has {min(counts)}). "
                "Accuracy can look good even if small classes are mostly wrong, so check the confusion matrix.")
    return None


# ═══════════════════════════════════════════════
# Image data
# ═══════════════════════════════════════════════

def load_builtin_images(name, loader=None):
    """Return (X_train, y_train, X_test, y_test, class_names) for MNIST / Fashion-MNIST:
    the official training and test sets, normalised to [0, 1] with a channel axis.
    `loader` lets the app pass a cached loader."""
    import tensorflow as tf
    if name == "MNIST":
        load = loader or tf.keras.datasets.mnist.load_data
        class_names = [str(i) for i in range(10)]
    elif name == "Fashion-MNIST":
        load = loader or tf.keras.datasets.fashion_mnist.load_data
        class_names = list(FASHION_CLASSES)
    else:
        raise DataError(f"Unknown image dataset '{name}'. Choose one of: {', '.join(IMAGE_DATASETS)}.")
    (X_train, y_train), (X_test, y_test) = load()
    X_train = (X_train.astype(np.float32) / 255.0)[..., np.newaxis]
    X_test = (X_test.astype(np.float32) / 255.0)[..., np.newaxis]
    return X_train, np.asarray(y_train, np.int64), X_test, np.asarray(y_test, np.int64), class_names


def prepare_builtin_images(name, loader=None, max_train_samples=None, max_val_samples=None, seed=DEFAULT_SEED):
    """MNIST / Fashion-MNIST as train / validation / test.

    The validation images are taken from the official TRAINING set (6,000 of the 60,000,
    stratified, chosen by the seed), which leaves 54,000 for training. The official test
    set (10,000 images) stays untouched until the final evaluation.
    """
    X_full, y_full, X_test, y_test, class_names = load_builtin_images(name, loader)
    rows = np.arange(len(y_full))
    n_val = len(rows) // IMAGE_VAL_DIVISOR
    train, val = train_test_split(rows, test_size=n_val, random_state=seed,
                                  stratify=y_full if n_val >= len(class_names) else None)
    rng = np.random.default_rng(seed)
    note = None
    if max_train_samples and len(train) > max_train_samples:
        train = rng.choice(train, max_train_samples, replace=False)
        note = f"Trained on a random subset of {max_train_samples:,} training images to keep the experiment quick."
    if max_val_samples and len(val) > max_val_samples:
        val = rng.choice(val, max_val_samples, replace=False)
    data = PreparedData(
        X_train=X_full[train], X_val=X_full[val], y_train=y_full[train], y_val=y_full[val],
        class_names=class_names, input_shape=(28, 28, 1), kind="image",
        class_counts=_class_counts(np.concatenate([y_full, y_test]), class_names),
        image_style="digit" if name == "MNIST" else "object",
        X_test=X_test, y_test=y_test, split_indices={"train": train, "val": val},
    )
    if note:
        data.info.append(note)
    return data


def prepare_image_arrays(X, y, class_names, seed=DEFAULT_SEED, image_style="object"):
    """Stratified train / validation / test split of an in-memory image dataset (e.g. from a custom ZIP)."""
    y = np.asarray(y, dtype=np.int64)
    parts, stratified, split_warnings = split_rows(y, seed)
    data = PreparedData(
        X_train=X[parts["train"]], X_val=X[parts["val"]], y_train=y[parts["train"]], y_val=y[parts["val"]],
        class_names=list(class_names), input_shape=tuple(X.shape[1:]), kind="image",
        class_counts=_class_counts(y, class_names), stratified=stratified, image_style=image_style,
        X_test=X[parts["test"]] if "test" in parts else None,
        y_test=y[parts["test"]] if "test" in parts else None, split_indices=parts,
    )
    data.warnings.extend(split_warnings)
    return data


# --- Custom ZIP ---

def _member_parts(name):
    """Validate a ZIP member name and return its path components.
    Rejects absolute paths, drive letters and '..' components (zip-slip)."""
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise DataError(f"The ZIP contains an absolute path ('{name}'), which is not allowed.")
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise DataError(f"The ZIP contains an unsafe path ('{name}'), which is not allowed.")
    return parts


def _is_ignored(parts):
    return any(p.startswith(".") or p == "__MACOSX" for p in parts)


def _zip_limits(limits):
    lim = {"max_bytes": MAX_ZIP_BYTES, "max_members": MAX_ZIP_MEMBERS, "max_images": MAX_ZIP_IMAGES,
           "max_uncompressed": MAX_ZIP_UNCOMPRESSED, "max_member_bytes": MAX_ZIP_MEMBER_BYTES}
    lim.update(limits or {})
    return lim


def _open_zip(zip_bytes, lim):
    if not zip_bytes:
        raise DataError("The ZIP file is empty.")
    if len(zip_bytes) > lim["max_bytes"]:
        raise DataError(f"The ZIP file is too large ({len(zip_bytes) / 1e6:.0f} MB). "
                        f"The limit is {lim['max_bytes'] / 1e6:.0f} MB.")
    try:
        return zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise DataError("This file is not a valid ZIP archive.") from e


def _list_zip_images(zf, lim, report):
    """Validated (info, path parts) for every image in the archive, read from the ZIP's
    directory only. Enforces the member / image / size limits and strips a single shared
    root folder. Also returns every folder name that could be a class."""
    infos = zf.infolist()
    if len(infos) > lim["max_members"]:
        raise DataError(f"The ZIP contains {len(infos):,} entries; the limit is {lim['max_members']:,}.")

    candidates, folders = [], []
    for info in infos:
        parts = _member_parts(info.filename)          # raises on traversal / absolute paths
        if not parts or _is_ignored(parts):
            continue
        folders.append(parts if info.is_dir() else parts[:-1])
        if info.is_dir():
            continue
        if not parts[-1].lower().endswith(IMAGE_EXTENSIONS):
            report["skipped_non_image"] += 1
            continue
        candidates.append((info, parts))

    if not candidates:
        raise DataError("No images (.png, .jpg, .jpeg, .bmp, .gif, .webp) were found in the ZIP.")
    if len(candidates) > lim["max_images"]:
        raise DataError(f"The ZIP contains {len(candidates):,} images; the limit is {lim['max_images']:,}.")
    declared = sum(info.file_size for info, _ in candidates)
    if declared > lim["max_uncompressed"]:
        raise DataError(f"The ZIP would expand to {declared / 1e6:.0f} MB; "
                        f"the limit is {lim['max_uncompressed'] / 1e6:.0f} MB.")

    # Strip a single shared root folder ("dataset/cats/1.png" -> "cats/1.png").
    firsts = {parts[0] for _, parts in candidates}
    if len(firsts) == 1 and all(len(parts) >= 3 for _, parts in candidates):
        root = next(iter(firsts))
        candidates = [(info, parts[1:]) for info, parts in candidates]
        folders = [f[1:] for f in folders if f and f[0] == root]
    class_folders = sorted({f[0] for f in folders if f})
    return candidates, class_folders


def zip_summary(zip_bytes, limits=None):
    """Classes and image counts of a class-folder ZIP, read from its directory listing
    only (nothing is extracted or decoded), so it is cheap to show before training."""
    lim = _zip_limits(limits)
    report = {"skipped_non_image": 0}
    with _open_zip(zip_bytes, lim) as zf:
        candidates, class_folders = _list_zip_images(zf, lim, report)
    counts = {name: 0 for name in class_folders}
    loose = 0
    for _, parts in candidates:
        if len(parts) < 2:
            loose += 1
        else:
            counts[parts[0]] = counts.get(parts[0], 0) + 1
    empty = sorted(name for name, n in counts.items() if n == 0)
    counts = {name: n for name, n in sorted(counts.items()) if n > 0}
    warnings = []
    if report["skipped_non_image"]:
        warnings.append(f"{report['skipped_non_image']} non-image file(s) will be ignored.")
    if loose:
        warnings.append(f"{loose} image(s) are not inside a class folder and will be ignored.")
    if empty:
        warnings.append("Class folder(s) without images will be left out: " + ", ".join(empty) + ".")
    if len(counts) < 2:
        warnings.append("At least 2 class folders with images are needed (e.g. cats/ and dogs/).")
    imbalance = _imbalance_warning(counts)
    if imbalance:
        warnings.append(imbalance)
    return {"class_counts": counts, "images": sum(counts.values()), "skipped_non_image": report["skipped_non_image"],
            "loose_files": loose, "empty_classes": empty, "warnings": warnings}


def load_image_zip(zip_bytes, style="object", limits=None):
    """Safely read a ZIP of class folders into a 28x28 image dataset.

    Expected layout:  class_a/img1.png, class_b/img2.jpg  (optionally inside one root folder).
    Returns (X, y, class_names, report) where report has counts, skipped files and warnings.
    Files are extracted into a private temporary directory that is always removed afterwards.
    """
    lim = _zip_limits(limits)
    report = {"skipped_non_image": 0, "unreadable": 0, "loose_files": 0, "warnings": [], "class_counts": {}}
    with _open_zip(zip_bytes, lim) as zf:
        candidates, _ = _list_zip_images(zf, lim, report)

        tmp_dir = tempfile.mkdtemp(prefix="dlt_zip_")
        try:
            extracted = _extract_members(zf, candidates, tmp_dir, lim, report)
            X, y, class_names = _load_extracted_images(extracted, style, report)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if report["skipped_non_image"]:
        report["warnings"].append(f"Skipped {report['skipped_non_image']} non-image file(s).")
    if report["loose_files"]:
        report["warnings"].append(f"Skipped {report['loose_files']} image(s) that were not inside a class folder.")
    if report["unreadable"]:
        report["warnings"].append(f"Skipped {report['unreadable']} image(s) that could not be read.")
    imbalance = _imbalance_warning(report["class_counts"])
    if imbalance:
        report["warnings"].append(imbalance)
    return X, y, class_names, report


def _extract_members(zf, candidates, tmp_dir, lim, report):
    """Stream each accepted member to disk, counting the real bytes written
    (headers can lie about sizes in zip bombs)."""
    root = os.path.realpath(tmp_dir)
    total = 0
    extracted = []  # (class_name, path)
    for info, parts in candidates:
        if len(parts) < 2:
            report["loose_files"] += 1
            continue
        dest = os.path.realpath(os.path.join(root, *parts))
        if os.path.commonpath([root, dest]) != root:
            raise DataError(f"The ZIP contains an unsafe path ('{info.filename}'), which is not allowed.")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        written = 0
        with zf.open(info) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(1 << 16)
                if not chunk:
                    break
                written += len(chunk)
                total += len(chunk)
                if written > lim["max_member_bytes"]:
                    raise DataError(f"'{info.filename}' is larger than "
                                    f"{lim['max_member_bytes'] / 1e6:.0f} MB, which is not allowed.")
                if total > lim["max_uncompressed"]:
                    raise DataError("The ZIP expands to more data than allowed "
                                    f"({lim['max_uncompressed'] / 1e6:.0f} MB).")
                out.write(chunk)
        extracted.append((parts[0], dest))
    return extracted


def _load_extracted_images(extracted, style, report):
    by_class = {}
    for class_name, path in extracted:
        by_class.setdefault(class_name, []).append(path)

    images, labels, class_names = [], [], []
    for class_name in sorted(by_class):
        loaded = []
        for path in sorted(by_class[class_name]):
            try:
                arr, _ = preprocess_image(open_image(path), style=style)
                loaded.append(arr)
            except (ImageError, OSError, ValueError):
                report["unreadable"] += 1
        report["class_counts"][class_name] = len(loaded)
        if not loaded:
            report["warnings"].append(f"Class folder '{class_name}' has no readable images and was left out.")
            continue
        label = len(class_names)
        class_names.append(class_name)
        images.extend(loaded)
        labels.extend([label] * len(loaded))

    report["class_counts"] = {k: v for k, v in report["class_counts"].items() if v > 0}
    if len(class_names) < 2:
        found = ", ".join(sorted(by_class)) or "none"
        raise DataError("At least 2 class folders with images are needed "
                        f"(found usable classes: {', '.join(class_names) or 'none'}; folders: {found}). "
                        "Put each class's images in its own folder, e.g. cats/ and dogs/.")
    if len(images) < MIN_IMAGES_TOTAL:
        raise DataError(f"Only {len(images)} usable images were found; at least {MIN_IMAGES_TOTAL} are needed.")
    X = np.stack(images)[..., np.newaxis].astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, class_names
