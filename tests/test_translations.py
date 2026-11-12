"""Every UI string exists in English AND Malayalam, and every key the code uses is defined."""

import glob
import os
import re
import string

from config.translations import UI, t

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DYNAMIC_KEYS = (       # keys the code builds at run time (f-strings, ternaries, lookup tables)
    [f"level_{v}" for v in ("beginner", "intermediate", "advanced")]
    + [f"qp{i}" for i in (1, 2, 3)] + [f"qp{i}_prompt" for i in (1, 2, 3)]
    + [f"tab_{name}" for name in ("train", "viz", "tutor", "test")]
    + [f"verdict_{tier}" for tier in ("below", "within", "above")]
    + [f"ds_info_{name}" for name in ("iris", "wine", "breast_cancer", "digits", "mnist", "fashion_mnist")]
    + ["sample_seven", "sample_sneaker", "sample_boot", "explain_cm", "why_overfit", "suggest_improve",
       "net_simple", "net_cnn"]
    + ["backend_gemini", "backend_ollama", "kb_keyword_hint", "kb_keyword_hint_ollama",
       "dataset_help_cnn", "epochs_help_cnn", "cm_on_val", "fill_random_val"]
)


def used_keys():
    keys = set(DYNAMIC_KEYS)
    for path in [os.path.join(ROOT, "app.py")] + glob.glob(os.path.join(ROOT, "components", "*.py")):
        src = open(path, encoding="utf-8").read()
        keys |= set(re.findall(r"""\bT\(\s*["']([a-z0-9_]+)["']""", src))
        keys |= set(re.findall(r"""UI\[\w+\]\[\s*["']([a-z0-9_]+)["']\s*\]""", src))
        keys |= set(re.findall(r"""UI\["en"\]\[\s*["']([a-z0-9_]+)["']\s*\]""", src))
        keys |= set(re.findall(r"""labels\[\s*["']([a-z0-9_]+)["']\s*\]""", src))
    return keys


def placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_both_languages_have_the_same_keys():
    assert set(UI["en"]) == set(UI["ml"]), set(UI["en"]) ^ set(UI["ml"])


def test_every_key_used_in_code_is_defined():
    missing = sorted(k for k in used_keys() if k not in UI["en"] or k not in UI["ml"])
    assert not missing, missing


def test_placeholders_match_between_languages():
    for key in UI["en"]:
        assert placeholders(UI["en"][key]) == placeholders(UI["ml"][key]), key


def test_no_literal_backslash_n_and_no_empty_strings():
    for lang in ("en", "ml"):
        for key, value in UI[lang].items():
            assert value.strip(), (lang, key)
            assert "\\n" not in value, (lang, key)      # would render as a literal "\n"


def test_malayalam_is_actually_translated():
    same = [k for k in UI["en"] if UI["en"][k] == UI["ml"][k]]
    assert len(same) / len(UI["en"]) < 0.1, same        # only tiny technical labels may match


def test_t_helper_formats_and_falls_back():
    assert t("en", "csv_summary", rows=3, cols=2) == "3 rows × 2 columns"
    assert "3" in t("ml", "csv_summary", rows=3, cols=2)
    assert t("xx", "title") == UI["en"]["title"]
    assert t("en", "no_such_key") == "no_such_key"


def test_no_inaccurate_legacy_strings():
    text = " ".join(UI["en"].values()).lower()
    assert "colab" not in text and "dummy image" not in text and "generate data" not in text


def test_no_unused_keys():
    """Every string in the dictionary is shown somewhere (no dead translations)."""
    unused = sorted(set(UI["en"]) - used_keys())
    assert not unused, unused
