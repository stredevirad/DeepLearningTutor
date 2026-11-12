"""LLM retry / key rotation / caching with mocked exceptions (no network)."""

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from backend import llm_api as L


def quota_error(msg="Resource exhausted."):
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": msg, "status": "RESOURCE_EXHAUSTED"}})


def server_error():
    return genai_errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})


def client_error(code, msg):
    return genai_errors.ClientError(code, {"error": {"code": code, "message": msg, "status": "X"}})


@pytest.fixture
def keys(monkeypatch):
    """Three fake keys and a fresh per-session state; sleeps are recorded, not slept."""
    monkeypatch.setattr(L, "API_KEYS", ["k1", "k2", "k3"])
    monkeypatch.setitem(L._fallback_state, "current_key_index", 0)
    monkeypatch.setitem(L._fallback_state, "key_cooldowns", {})
    L.clear_caches()
    return L


class Flaky:
    """Callable that raises the given exceptions in order, then returns 'ok'."""
    def __init__(self, *errors):
        self.errors = list(errors)
        self.keys_seen = []

    def __call__(self):
        self.keys_seen.append(L._get("current_key_index"))
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def test_labels_match_model_ids():
    for label, model_id in L.GEMINI_MODELS.items():
        version = model_id.split("-")[1]                   # e.g. "2.5"
        assert version in label, (label, model_id)
        assert ("Flash-Lite" in label) == ("flash-lite" in model_id)
        assert ("Pro" in label) == model_id.endswith("pro")


def test_success_first_try(keys):
    sleeps = []
    assert L.call_with_retry(Flaky(), sleep=sleeps.append) == "ok" and sleeps == []


def test_quota_error_rotates_to_next_key(keys):
    f, sleeps = Flaky(quota_error()), []
    assert L.call_with_retry(f, sleep=sleeps.append) == "ok"
    assert f.keys_seen == [0, 1]                            # moved to key 2 after the 429
    assert 0 in L._get("key_cooldowns")                     # key 1 is cooling down
    assert len(sleeps) == 1 and sleeps[0] <= 1.0            # short pause when another key is free


def test_all_keys_exhausted_gives_clear_message(keys):
    f = Flaky(*[quota_error()] * 10)
    with pytest.raises(L.LLMUnavailableError, match="rate limit"):
        L.call_with_retry(f, sleep=lambda s: None)
    assert len(f.keys_seen) == 5                            # attempts = keys + 2, then give up


def test_single_key_backs_off_exponentially(monkeypatch):
    monkeypatch.setattr(L, "API_KEYS", ["only"])
    monkeypatch.setitem(L._fallback_state, "current_key_index", 0)
    monkeypatch.setitem(L._fallback_state, "key_cooldowns", {})
    monkeypatch.setattr(L.random, "random", lambda: 1.0)     # remove jitter for exact delays
    sleeps = []
    assert L.call_with_retry(Flaky(server_error(), server_error()), sleep=sleeps.append) == "ok"
    assert sleeps == [1.0, 2.0]                             # 1s, 2s, ... (exponential)


def test_retry_after_hint_is_respected(monkeypatch):
    monkeypatch.setattr(L, "API_KEYS", ["only"])
    monkeypatch.setitem(L._fallback_state, "current_key_index", 0)
    monkeypatch.setitem(L._fallback_state, "key_cooldowns", {})
    sleeps = []
    L.call_with_retry(Flaky(quota_error("Please retry in 7.5s.")), sleep=sleeps.append)
    assert sleeps == [8.0]


def test_long_retry_after_gives_up_quickly(monkeypatch):
    monkeypatch.setattr(L, "API_KEYS", ["only"])
    monkeypatch.setitem(L._fallback_state, "current_key_index", 0)
    monkeypatch.setitem(L._fallback_state, "key_cooldowns", {})
    sleeps = []
    with pytest.raises(L.LLMUnavailableError):
        L.call_with_retry(Flaky(*[quota_error("Please retry in 50s.")] * 5), sleep=sleeps.append)
    assert sleeps == []                                     # no pointless 50 s block of the UI


@pytest.mark.parametrize("err,fragment", [
    (client_error(400, "API key not valid. Please pass a valid API key."), "API key was rejected"),
    (client_error(403, "Permission denied"), "API key was rejected"),
    (client_error(404, "models/gemini-9 is not found"), "not available"),
    (client_error(400, "Invalid argument: request too large"), "rejected the request"),
    (ValueError("something unexpected"), "failed"),
])
def test_non_retryable_errors_fail_fast(keys, err, fragment):
    f = Flaky(err)
    with pytest.raises(L.LLMError, match=fragment):
        L.call_with_retry(f, sleep=lambda s: None)
    assert len(f.keys_seen) == 1                            # not retried pointlessly


def test_friendly_errors_never_include_key(keys):
    msg = L.friendly_error(client_error(400, "API key not valid: k1"))
    assert "k1" not in msg


def test_transient_timeouts_are_retried(keys):
    assert L.call_with_retry(Flaky(TimeoutError("read timed out")), sleep=lambda s: None) == "ok"


def test_no_keys_configured(monkeypatch):
    monkeypatch.setattr(L, "API_KEYS", [])
    with pytest.raises(L.LLMUnavailableError, match="No Gemini API key"):
        L.call_with_retry(lambda: "never")
    assert L.configure_gemini() is None


def test_configure_gemini_handles(keys):
    handle = L.configure_gemini(model_label=list(L.GEMINI_MODELS)[2])
    assert handle.model_name == "gemini-2.5-pro" and L.current_model_id() == "gemini-2.5-pro"


# ─────────────────────────── generation & caching ───────────────────────────

class FakeModels:
    def __init__(self):
        self.generate_calls = []
        self.embed_calls = []

    def generate_content(self, model, contents, config=None):
        self.generate_calls.append((model, contents, config))
        return SimpleNamespace(text=f"reply to {contents}")

    def embed_content(self, model, contents, config=None):
        self.embed_calls.append((model, list(contents), config.task_type))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(t)), 1.0]) for t in contents])


@pytest.fixture
def fake_client(keys, monkeypatch):
    models = FakeModels()
    monkeypatch.setattr(L, "get_client", lambda: SimpleNamespace(models=models))
    return models


def test_generate_text_cache(fake_client):
    a = L.generate_text("narrate this history", cache=True)
    b = L.generate_text("narrate this history", cache=True)
    assert a == b and len(fake_client.generate_calls) == 1          # second call served from cache
    L.generate_text("narrate this history", cache=False)
    assert len(fake_client.generate_calls) == 2


def test_generate_text_config(fake_client):
    L.generate_text("x", json_mode=True, thinking_budget=8192, model_id="gemini-2.5-flash")
    model, _, config = fake_client.generate_calls[0]
    assert model == "gemini-2.5-flash"
    assert config.response_mime_type == "application/json" and config.thinking_config.thinking_budget == 8192


def test_empty_reply_is_an_error(keys, monkeypatch):
    models = SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text=""))
    monkeypatch.setattr(L, "get_client", lambda: SimpleNamespace(models=models))
    with pytest.raises(L.LLMError, match="empty"):
        L.generate_text("x")


def test_embed_texts_batches_and_caches(fake_client, monkeypatch):
    monkeypatch.setattr(L, "EMBED_BATCH_SIZE", 2)
    vecs = L.embed_texts(["a", "bb", "ccc"], task_type="RETRIEVAL_DOCUMENT")
    assert vecs == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert [len(c[1]) for c in fake_client.embed_calls] == [2, 1]
    assert {c[2] for c in fake_client.embed_calls} == {"RETRIEVAL_DOCUMENT"}
    L.embed_texts(["a", "bb"], task_type="RETRIEVAL_DOCUMENT")
    assert len(fake_client.embed_calls) == 2                        # cached, no new call
    L.embed_texts(["a"], task_type="RETRIEVAL_QUERY")
    assert fake_client.embed_calls[-1][2] == "RETRIEVAL_QUERY"      # different task type -> new call


def test_thinking_budget_only_for_models_that_support_it():
    extended = [label for label, budget in L.THINKING_LEVELS.items() if budget][0]
    standard = [label for label, budget in L.THINKING_LEVELS.items() if not budget][0]
    assert L.thinking_budget(extended, "gemini-2.5-flash") == 8192
    assert L.thinking_budget(extended, "gemini-2.5-pro") == 8192
    assert L.thinking_budget(extended, "gemini-2.0-flash-lite") is None
    assert L.thinking_budget(standard, "gemini-2.5-flash") is None
