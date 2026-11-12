"""parse_bilingual, render_response, call_ollama and the hallucination checker (all LLM calls mocked)."""

import json
from types import SimpleNamespace

import pytest

from components import chat_helpers as C

PY_ANSWER = """Here is a script:

```python
# 1. Load the dataset
import numpy as np
print("a\\nb")
```

That's it."""


# ─────────────────────────── parse_bilingual ───────────────────────────

def test_pure_json():
    out = C.parse_bilingual('{"en": "Hello", "ml": "നമസ്കാരം"}')
    assert out == {"en": "Hello", "ml": "നമസ്കാരം"}


def test_json_wrapped_in_json_fence():
    out = C.parse_bilingual('```json\n{"en": "Hi", "ml": "ഹായ്"}\n```')
    assert out == {"en": "Hi", "ml": "ഹായ്"}


def test_json_with_fenced_code_and_escaped_newlines():
    payload = {"en": "Try this:\n```python\n# 1. Load\nx = 1\nprint(\"a\\nb\")\n```\nDone.", "ml": "ഇത്:\n```python\nx = 1\n```"}
    raw = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    out = C.parse_bilingual(raw)
    assert out["en"] == payload["en"]                      # fences inside the answer survive
    assert '"a\\nb"' in out["en"]                           # string-literal escape in code is untouched
    assert out["ml"].count("```") == 2


def test_json_with_literal_newlines_inside_strings():
    raw = '{"en": "line one\nline two\n```python\nprint(1)\n```", "ml": "ഒന്ന്"}'
    out = C.parse_bilingual(raw)
    assert out["en"].startswith("line one\nline two") and "```python" in out["en"]


def test_plain_text_with_python_fences_keeps_them():
    out = C.parse_bilingual(PY_ANSWER)
    assert out["en"] == out["ml"] == PY_ANSWER
    assert out["en"].count("```") == 2


def test_whole_reply_is_a_python_block_is_not_stripped():
    raw = "```python\nprint('hi')\n```"
    assert C.parse_bilingual(raw)["en"] == raw


def test_malformed_json_falls_back_to_text():
    raw = '{"en": "unterminated'
    out = C.parse_bilingual(raw)
    assert out["en"] == "unterminated" and out["ml"] == "unterminated"


def test_malformed_json_that_is_not_bilingual():
    raw = "{this is not json at all"
    assert C.parse_bilingual(raw) == {"en": raw, "ml": raw}


def test_json_embedded_in_prose():
    raw = 'Sure! Here you go:\n{"en": "Answer", "ml": "ഉത്തരം"}\nHope that helps.'
    assert C.parse_bilingual(raw) == {"en": "Answer", "ml": "ഉത്തരം"}


def test_json_list_and_missing_ml():
    assert C.parse_bilingual('[{"en": "A", "ml": "B"}]') == {"en": "A", "ml": "B"}
    assert C.parse_bilingual('{"en": "only english"}') == {"en": "only english", "ml": "only english"}


def test_double_escaped_single_line_reply_is_unescaped():
    out = C.parse_bilingual("Line 1\\nLine 2\\n```python\\nx = 1\\n```")
    assert out["en"] == "Line 1\nLine 2\n```python\nx = 1\n```"


def test_none_and_empty():
    assert C.parse_bilingual(None) == {"en": "", "ml": ""}
    assert C.parse_bilingual("   ") == {"en": "", "ml": ""}


# ─────────────────────────── render_response ───────────────────────────

class Recorder:
    def __init__(self):
        self.calls = []

    def markdown(self, text, **kw):
        self.calls.append(("markdown", text))

    def code(self, text, language=None, **kw):
        self.calls.append(("code", text, language))


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()
    monkeypatch.setattr(C, "st", r)
    return r


def test_render_python_block(rec):
    C.render_response(PY_ANSWER)
    kinds = [c[0] for c in rec.calls]
    assert kinds == ["markdown", "code", "markdown"]
    code = rec.calls[1]
    assert code[2] == "python" and code[1].startswith("# 1. Load the dataset")
    assert 'print("a\\nb")' in code[1]
    assert all("# 1. Load" not in c[1] for c in rec.calls if c[0] == "markdown")


def test_render_pure_json_parsed_then_rendered(rec):
    data = C.parse_bilingual('```json\n{"en": "Intro\\n```py\\nx = 2\\n```", "ml": "x"}\n```')
    C.render_response(data["en"])
    assert rec.calls == [("markdown", "Intro"), ("code", "x = 2", "python")]


def test_render_unterminated_fence_is_code(rec):
    C.render_response("Start\n```python\n# heading-looking comment\nx = 1")
    assert rec.calls[-1] == ("code", "# heading-looking comment\nx = 1", "python")


def test_render_tilde_fence_and_trailing_spaces(rec):
    C.render_response("~~~bash   \necho hi\n~~~\n\n```   \nplain\n```")
    assert rec.calls == [("code", "echo hi", "bash"), ("code", "plain", "python")]


def test_render_malformed_json_text(rec):
    C.render_response(C.parse_bilingual("{not json")["en"])
    assert rec.calls == [("markdown", "{not json")]


def test_split_markdown_and_code_multiple_blocks():
    segs = C.split_markdown_and_code("a\n```js\nx\n```\nb\n````\ny\n```\nstill code\n````\nc")
    assert [s[0] for s in segs] == ["markdown", "code", "markdown", "code", "markdown"]
    assert segs[3][1] == "y\n```\nstill code"                  # inner ``` inside a ```` fence


# ─────────────────────────── call_ollama ───────────────────────────

def test_call_ollama_success_attribute_style():
    client = SimpleNamespace(chat=lambda **kw: SimpleNamespace(message=SimpleNamespace(content="hi")))
    assert C.call_ollama(client, "llama3.2:1b", "hello") == "hi"


def test_call_ollama_json_mode_sets_format():
    seen = {}

    def chat(**kw):
        seen.update(kw)
        return {"message": {"content": "{}"}}
    C.call_ollama(SimpleNamespace(chat=chat), "m", "p", json_mode=True)
    assert seen["format"] == "json"


def test_call_ollama_error_with_quotes_is_valid_json():
    def chat(**kw):
        raise RuntimeError('model "llama3" said: "boom" \\ and {braces}')
    raw = C.call_ollama(SimpleNamespace(chat=chat), "llama3", "hi")
    data = json.loads(raw)                                  # would fail with the old f-string
    assert set(data) == {"en", "ml"} and data["en"].startswith("⚠️")
    assert C.parse_bilingual(raw)["en"] == data["en"]


def test_call_ollama_model_not_found_message():
    class NotFound(Exception):
        status_code = 404
    def chat(**kw):
        raise NotFound("model 'x' not found")
    data = json.loads(C.call_ollama(SimpleNamespace(chat=chat), "x", "hi"))
    assert "ollama pull x" in data["en"]


# ─────────────────────────── fact checker ───────────────────────────

@pytest.mark.parametrize("raw,score", [
    ('{"grounding_score": 85, "hallucinated_claims": [], "verdict": "ok"}', 85),
    ('```json\n{"grounding_score": "70%", "hallucinated_claims": ["x"], "verdict": "meh"}\n```', 70),
    ('Result: {"grounding_score": 0.9, "hallucinated_claims": "one claim", "verdict": "v"} done', 90),
    ('{"grounding_score": 250, "verdict": "too high"}', 100),
    ('[{"grounding_score": -5}]', 0),
])
def test_parse_fact_check_variants(raw, score):
    assert C.parse_fact_check(raw)["score"] == score


@pytest.mark.parametrize("raw", ["", "no json here", '{"verdict": "missing score"}', '{"grounding_score": "high"}',
                                 '{"grounding_score": 80,'])
def test_parse_fact_check_malformed(raw):
    with pytest.raises(C.FactCheckError):
        C.parse_fact_check(raw)


def test_fact_check_prompt_uses_given_sources():
    p = C.build_fact_check_prompt("answer", ["chunk about dropout", "chunk about relu"])
    assert "[1] chunk about dropout" in p and "[2] chunk about relu" in p


def test_fact_check_gemini_path(monkeypatch):
    calls = []

    def fake_generate_text(prompt, model_id=None, json_mode=False, cache=False, **kw):
        calls.append((model_id, json_mode))
        return '{"grounding_score": 60, "hallucinated_claims": ["c"], "verdict": "partly"}'
    monkeypatch.setattr(C.llm_api, "generate_text", fake_generate_text)
    out = C.fact_check("ans", ["src"])
    assert out == {"score": 60, "claims": ["c"], "verdict": "partly"}
    assert calls == [(C.llm_api.FAST_MODEL_ID, True)]


def test_fact_check_gemini_falls_back_to_selected_model(monkeypatch):
    calls = []

    def fake_generate_text(prompt, model_id=None, json_mode=False, cache=False, **kw):
        calls.append(model_id)
        if model_id == C.llm_api.FAST_MODEL_ID:
            raise C.llm_api.LLMError("model not available")
        return '{"grounding_score": 99, "verdict": "fine"}'
    monkeypatch.setattr(C.llm_api, "generate_text", fake_generate_text)
    assert C.fact_check("ans", ["src"])["score"] == 99
    assert calls == [C.llm_api.FAST_MODEL_ID, None]


def test_fact_check_does_not_start_a_second_round_when_no_key_is_usable(monkeypatch):
    """ "All keys are rate-limited" (or no key at all) is not a problem with the fast model:
    asking again with the selected model would only repeat the same retries and waits."""
    calls = []

    def fake_generate_text(prompt, model_id=None, json_mode=False, cache=False, **kw):
        calls.append(model_id)
        raise C.llm_api.LLMUnavailableError("All Gemini API keys have hit their rate limit.")
    monkeypatch.setattr(C.llm_api, "generate_text", fake_generate_text)
    with pytest.raises(C.llm_api.LLMUnavailableError):
        C.fact_check("ans", ["src"])
    assert calls == [C.llm_api.FAST_MODEL_ID]


def test_fact_check_ollama_path_and_malformed_reply():
    good = SimpleNamespace(chat=lambda **kw: {"message": {"content": '{"grounding_score": 40, "verdict": "low"}'}})
    assert C.fact_check("a", ["s"], using_ollama=True, ollama_client=good, ollama_model_name="m")["score"] == 40
    bad = SimpleNamespace(chat=lambda **kw: {"message": {"content": "I think it's fine!"}})
    with pytest.raises(C.FactCheckError):
        C.fact_check("a", ["s"], using_ollama=True, ollama_client=bad, ollama_model_name="m")
    with pytest.raises(C.FactCheckError):
        C.fact_check("a", ["s"], using_ollama=True, ollama_client=None, ollama_model_name=None)
