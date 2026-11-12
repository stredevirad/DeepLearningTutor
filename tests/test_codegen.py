"""'Show Equivalent Python Code': valid Python, no leakage, faithful to the trained config, and runnable."""

import ast
import os

import numpy as np
import pandas as pd
import pytest

from core import data as D
from core.codegen import generate_training_code
from core.training import validate_config
from tests.helpers import fake_image_loader, shape_png

CSV_META = {"target_column": "species", "numeric_columns": ["w", "h"], "categorical_columns": ["colour"],
            "stratified": True}
ZIP_META = {"class_names": ["circle", "square"], "stratified": True}


def _cfg(**kw):
    custom = kw.pop("using_custom", False)
    cfg = validate_config(kw, allow_custom_dataset=True) if not custom else validate_config(
        dict(kw, using_custom=True), allow_custom_dataset=True)
    return cfg


def _calls(tree, name):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if fname == name:
                out.append(node)
    return out


def _const(node):
    return ast.literal_eval(node)


COMBOS = (
    [dict(network_type="Simple NN", dataset=d, hidden_layers=h, nodes_per_layer=n, dropout_rate=dr,
          optimizer=o, learning_rate=lr, batch_size=b, epochs=e, early_stopping=es, seed=s)
     for d in ("Iris", "Wine", "Breast Cancer", "Digits")
     for (h, n, dr, o, lr, b, e, es, s) in [(1, 8, 0.0, "Adam", 0.01, 16, 100, False, 42),
                                           (3, 32, 0.25, "SGD", 0.1, 64, 12, True, 7)]]
    + [dict(network_type="CNN", dataset=d, conv_layers=c, base_filters=f, dropout_rate=dr, optimizer=o,
            learning_rate=lr, batch_size=b, epochs=e, early_stopping=es)
       for d in ("MNIST", "Fashion-MNIST")
       for (c, f, dr, o, lr, b, e, es) in [(1, 8, 0.0, "RMSprop", 0.001, 128, 3, False),
                                           (3, 32, 0.5, "Adam", 0.0001, 32, 50, True)]]
    + [dict(network_type="Simple NN", dataset="data.csv", using_custom=True, hidden_layers=2, nodes_per_layer=16,
            early_stopping=True),
       dict(network_type="CNN", dataset="shapes.zip", using_custom=True, conv_layers=2, base_filters=16)]
)


@pytest.mark.parametrize("raw", COMBOS, ids=lambda c: f"{c['network_type']}-{c['dataset']}-{c.get('epochs', '')}")
def test_code_is_valid_faithful_and_leak_free(raw):
    cfg = _cfg(**raw)
    meta = CSV_META if cfg.get("using_custom") and cfg["network_type"] == "Simple NN" else ZIP_META
    code = generate_training_code(cfg, meta)
    tree = ast.parse(code)                                         # valid Python

    # --- no leakage: nothing is fitted before the split ---
    split_at = code.index("train_test_split(") if "train_test_split(" in code else None
    preprocessing_fits = [i for i in range(len(code)) if code.startswith(("fit_transform(", ".fit("), i)
                          and not code[max(0, i - 5):i].endswith("model")]
    for i in preprocessing_fits:
        assert split_at is not None and i > split_at, code[i - 20:i + 20]
    assert "fit_transform(X)" not in code

    # --- faithful to the configuration ---
    dense_units = [_const(c.args[0]) for c in _calls(tree, "Dense") if isinstance(c.args[0], ast.Constant)]
    dropouts = [_const(c.args[0]) for c in _calls(tree, "Dropout")]
    if cfg["network_type"] == "Simple NN":
        assert dense_units == [cfg["nodes_per_layer"]] * cfg["hidden_layers"]
        assert dropouts == ([cfg["dropout_rate"]] * cfg["hidden_layers"] if cfg["dropout_rate"] > 0 else [])
    else:
        filters = [_const(c.args[0]) for c in _calls(tree, "Conv2D")]
        assert filters == [cfg["base_filters"] * 2 ** i for i in range(cfg["conv_layers"])]
        assert dense_units == [64]
        assert dropouts == ([cfg["dropout_rate"]] if cfg["dropout_rate"] > 0 else [])

    opt = [c for c in _calls(tree, cfg["optimizer"])]
    assert len(opt) == 1 and _const(opt[0].keywords[0].value) == pytest.approx(cfg["learning_rate"])
    fit = _calls(tree, "fit")[-1]
    kw = {k.arg: k.value for k in fit.keywords}
    assert _const(kw["epochs"]) == cfg["epochs"] and _const(kw["batch_size"]) == cfg["batch_size"]
    assert bool(_calls(tree, "EarlyStopping")) == cfg["early_stopping"]
    assert f"SEED = {cfg['seed']}" in code and "stratify=y" in code

    # --- three parts, and both numbers are printed ---
    assert "X_test" in code and "Validation accuracy" in code and "Test accuracy" in code
    assert "validation only" not in code


def test_right_dataset_loader_is_used():
    for ds, loader in [("Iris", "load_iris"), ("Wine", "load_wine"),
                       ("Breast Cancer", "load_breast_cancer"), ("Digits", "load_digits")]:
        code = generate_training_code(_cfg(dataset=ds))
        assert f"data = {loader}()" in code and ("load_wine" not in code or ds == "Wine")
    assert "fashion_mnist.load_data()" in generate_training_code(_cfg(network_type="CNN", dataset="Fashion-MNIST"))


def test_custom_csv_code_mentions_target_and_columns():
    code = generate_training_code(_cfg(network_type="Simple NN", dataset="x.csv", using_custom=True), CSV_META)
    assert "TARGET = 'species'" in code and "CATEGORICAL = ['colour']" in code and "OneHotEncoder" in code
    no_cat = generate_training_code(_cfg(network_type="Simple NN", dataset="x.csv", using_custom=True),
                                    dict(CSV_META, categorical_columns=[]))
    assert "OneHotEncoder(handle_unknown" not in no_cat


# ─────────────────────────── the scripts actually run ───────────────────────────

def _run(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = {"__name__": "__generated__"}
    exec(compile(code, "generated.py", "exec"), ns)
    assert os.path.exists(tmp_path / "my_model.keras")
    return ns


def _same_rows(ns, data):
    """The script put exactly the app's rows into each part."""
    for part in data.split_indices:
        assert np.array_equal(ns[part], data.split_indices[part]), part
    assert np.allclose(ns["X_train"], data.X_train, atol=1e-5) and np.array_equal(ns["y_train"], data.y_train)
    assert np.allclose(ns["X_val"], data.X_val, atol=1e-5) and np.array_equal(ns["y_val"], data.y_val)
    if data.has_test:
        assert np.allclose(ns["X_test"], data.X_test, atol=1e-5) and np.array_equal(ns["y_test"], data.y_test)
        assert 0.0 <= ns["test_acc"] <= 1.0
    else:
        assert "X_test" not in ns and "test_acc" not in ns


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["Iris", "Wine", "Breast Cancer", "Digits"])
def test_simple_nn_scripts_run(dataset, tmp_path, monkeypatch):
    code = generate_training_code(_cfg(dataset=dataset, epochs=1, early_stopping=True, dropout_rate=0.1, seed=7))
    ns = _run(code, tmp_path, monkeypatch)
    data = D.prepare_builtin_tabular(dataset, seed=7)
    assert data.has_test
    _same_rows(ns, data)


@pytest.mark.slow
@pytest.mark.parametrize("dataset,module", [("MNIST", "mnist"), ("Fashion-MNIST", "fashion_mnist")])
def test_cnn_scripts_run(dataset, module, tmp_path, monkeypatch):
    import tensorflow as tf
    images = fake_image_loader(200, 16)()        # loaded once, so the script and the app see the same images

    def loader():
        return images
    monkeypatch.setattr(getattr(tf.keras.datasets, module), "load_data", loader)
    code = generate_training_code(_cfg(network_type="CNN", dataset=dataset, epochs=1, conv_layers=2, base_filters=4))
    ns = _run(code, tmp_path, monkeypatch)
    data = D.prepare_builtin_images(dataset, loader=loader, seed=42)
    assert len(data.y_val) == 20 and len(data.y_test) == 16      # a tenth of the training set; the official test set
    _same_rows(ns, data)


@pytest.mark.slow
def test_custom_csv_script_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"w": rng.normal(size=40), "h": rng.normal(size=40),
                       "colour": ["red", "blue"] * 20, "species": ["a", "b", "c", "d"] * 10})
    df.loc[3, "w"] = None
    df.to_csv(tmp_path / "your_dataset.csv", index=False)
    data = D.prepare_custom_csv(df, "species", seed=42)
    assert data.has_test
    code = generate_training_code(_cfg(network_type="Simple NN", dataset="x.csv", using_custom=True, epochs=1), CSV_META)
    _same_rows(_run(code, tmp_path, monkeypatch), data)


@pytest.mark.slow
def test_tiny_csv_script_makes_the_fallback_split(tmp_path, monkeypatch):
    """Too few rows for three parts: app and script both fall back to 80 / 20 and report validation only."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"w": rng.normal(size=8), "h": rng.normal(size=8),
                       "colour": ["red", "blue"] * 4, "species": ["a", "b"] * 4})
    df.to_csv(tmp_path / "your_dataset.csv", index=False)
    data = D.prepare_custom_csv(df, "species", seed=42)
    assert not data.has_test
    meta = dict(CSV_META, stratified=data.stratified, has_test=False)
    code = generate_training_code(_cfg(network_type="Simple NN", dataset="x.csv", using_custom=True, epochs=1), meta)
    assert "Test accuracy" not in code and "X_test" not in code
    _same_rows(_run(code, tmp_path, monkeypatch), data)


@pytest.mark.slow
def test_custom_zip_script_runs(tmp_path, monkeypatch):
    for cls in ("circle", "square"):
        folder = tmp_path / "your_dataset" / cls
        folder.mkdir(parents=True)
        for i in range(10):
            (folder / f"{i}.png").write_bytes(shape_png(cls))
    code = generate_training_code(_cfg(network_type="CNN", dataset="s.zip", using_custom=True, epochs=1,
                                       conv_layers=1, base_filters=4), ZIP_META)
    ns = _run(code, tmp_path, monkeypatch)
    data = D.prepare_image_arrays(ns["X"], ns["y"], ["circle", "square"], seed=42)
    assert data.has_test
    _same_rows(ns, data)
