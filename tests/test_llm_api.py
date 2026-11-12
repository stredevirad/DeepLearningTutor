"""LLM retry / key rotation / caching with mocked exceptions (no network)."""

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors as genai_errors

from backend import llm_api as L
from backend import usage as U


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


# ─────────────────────────── request time limit ───────────────────────────

def test_client_is_built_with_a_time_limit_and_no_sdk_retries(keys, monkeypatch):
    made = []
    monkeypatch.setattr(L, "_clients", {})
    monkeypatch.setattr(L.genai, "Client", lambda **kw: made.append(kw) or SimpleNamespace(models=None))
    L.get_client()
    L.get_client()                                              # cached per key: built once
    assert len(made) == 1 and made[0]["api_key"] == "k1"
    options = made[0]["http_options"]
    assert options.timeout == L.GENERATE_TIMEOUT_SECONDS * 1000        # the SDK takes milliseconds
    assert options.retry_options.attempts == 1                         # the SDK itself never retries
    assert L.EMBED_TIMEOUT_SECONDS < L.GENERATE_TIMEOUT_SECONDS


def test_embedding_requests_carry_the_shorter_time_limit(keys, monkeypatch):
    seen = []

    def embed_content(model, contents, config=None):
        seen.append(config)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0]) for _ in contents])
    monkeypatch.setattr(L, "get_client", lambda: SimpleNamespace(models=SimpleNamespace(embed_content=embed_content)))
    L.embed_texts(["a"])
    assert seen[0].http_options.timeout == L.EMBED_TIMEOUT_SECONDS * 1000
    assert seen[0].http_options.retry_options.attempts == 1 and seen[0].task_type == "RETRIEVAL_DOCUMENT"
    assert seen[0].output_dimensionality == L.EMBEDDING_DIMENSIONS == 768      # a quarter of the full 3,072


def test_embedding_cache_is_separate_per_dimension(fake_client, monkeypatch):
    L.embed_texts(["a"])
    monkeypatch.setattr(L, "EMBEDDING_DIMENSIONS", 3072)
    L.embed_texts(["a"])
    assert len(fake_client.embed_calls) == 2        # a vector of another size is never served from the cache


def test_a_timeout_is_transient_and_retried_once_only(keys):
    import httpx
    assert L.classify_error(httpx.ReadTimeout("The read operation timed out")) == "transient"
    assert L.is_timeout(httpx.ConnectTimeout("x")) and L.is_timeout(TimeoutError()) and not L.is_timeout(ValueError())

    once = Flaky(httpx.ReadTimeout("The read operation timed out"))
    assert L.call_with_retry(once, sleep=lambda s: None) == "ok"       # one timeout: tried again

    hung = Flaky(*[httpx.ReadTimeout("The read operation timed out") for _ in range(6)])
    with pytest.raises(L.LLMError, match="took too long"):
        L.call_with_retry(hung, sleep=lambda s: None)
    assert len(hung.keys_seen) == 1 + L.MAX_TIMEOUT_RETRIES             # 2 attempts, not the 5 a 3-key setup allows
    u = U.snapshot()
    assert u["failed"] == 3 and u["retries"] == 2                       # 1 + 2 failed attempts in this test


def test_other_transient_errors_still_get_every_attempt(keys):
    busy = Flaky(*[server_error() for _ in range(9)])
    with pytest.raises(L.LLMError):
        L.call_with_retry(busy, sleep=lambda s: None)
    assert len(busy.keys_seen) == 5                                     # max(3, min(3 keys + 2, 6))


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
    assert [v.tolist() for v in vecs] == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert all(v.dtype == np.float32 and not v.flags.writeable for v in vecs)      # compact, shared, read-only
    assert [len(c[1]) for c in fake_client.embed_calls] == [2, 1]
    assert {c[2] for c in fake_client.embed_calls} == {"RETRIEVAL_DOCUMENT"}
    again = L.embed_texts(["a", "bb"], task_type="RETRIEVAL_DOCUMENT")
    assert len(fake_client.embed_calls) == 2                        # cached, no new call
    assert again[0] is vecs[0] and again[1] is vecs[1]              # the cache hands out the same array
    L.embed_texts(["a"], task_type="RETRIEVAL_QUERY")
    assert fake_client.embed_calls[-1][2] == "RETRIEVAL_QUERY"      # different task type -> new call


# ─────────────────────────── streaming ───────────────────────────

class StreamingModels:
    """generate_content_stream fake: each call takes the next script from `scripts`.
    A script is a list of chunks; an exception in it is raised when that position is reached."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.opened = 0

    def generate_content_stream(self, model, contents, config=None):
        self.opened += 1
        script = self.scripts.pop(0)

        def chunks():
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item
        return chunks()


def _chunk(text, **usage_fields):
    meta = SimpleNamespace(prompt_token_count=None, candidates_token_count=None, thoughts_token_count=None,
                           cached_content_token_count=None)
    for name, value in usage_fields.items():
        setattr(meta, name, value)
    return SimpleNamespace(text=text, usage_metadata=meta if usage_fields else None)


def _streaming(monkeypatch, *scripts):
    models = StreamingModels(*scripts)
    monkeypatch.setattr(L, "get_client", lambda: SimpleNamespace(models=models))
    return models


def test_stream_yields_the_pieces_and_counts_tokens_from_the_last_chunk(keys, monkeypatch):
    models = _streaming(monkeypatch, [_chunk("Drop"), _chunk(None), _chunk("out "), _chunk(
        "works.", prompt_token_count=500, candidates_token_count=40, thoughts_token_count=90)])
    assert list(L.generate_stream("what is dropout?")) == ["Drop", "out ", "works."]
    u = U.snapshot()
    assert models.opened == 1 and (u["calls"], u["failed"], u["retries"]) == (1, 0, 0)
    assert (u["prompt_tokens"], u["output_tokens"], u["thinking_tokens"]) == (500, 40, 90)


def test_stream_retries_only_before_the_first_piece(keys, monkeypatch):
    models = _streaming(monkeypatch, [server_error()], [_chunk("Hello "), _chunk("there")])
    stream = L.generate_stream("hi", sleep=lambda s: None)        # the failed opening is retried in here
    assert models.opened == 2
    assert "".join(stream) == "Hello there"
    u = U.snapshot()
    assert (u["calls"], u["failed"], u["retries"]) == (2, 1, 1)


def test_stream_error_in_the_middle_is_not_retried(keys, monkeypatch):
    models = _streaming(monkeypatch, [_chunk("The answer is "), server_error()], [_chunk("never asked for")])
    stream = L.generate_stream("hi", sleep=lambda s: None)
    assert next(stream) == "The answer is "
    with pytest.raises(L.LLMError, match="temporarily unavailable"):
        next(stream)
    assert models.opened == 1                                      # no second request after text was shown


def test_stream_that_cannot_be_opened_raises_like_generate(keys, monkeypatch):
    _streaming(monkeypatch, [client_error(400, "bad request")])
    with pytest.raises(L.LLMError, match="rejected the request"):
        L.generate_stream("hi", sleep=lambda s: None)
    empty = _streaming(monkeypatch, [])
    assert list(L.generate_stream("hi")) == [] and empty.opened == 1   # an empty stream is an empty answer


# ─────────────────────────── usage meter ───────────────────────────

def _with_usage(text="ok", prompt=120, output=30, thoughts=None, cached=None):
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(
        prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thoughts,
        cached_content_token_count=cached, total_token_count=prompt + output))


def test_usage_counts_calls_tokens_and_cache(keys, monkeypatch):
    replies = [_with_usage(prompt=120, output=30, thoughts=50), _with_usage(prompt=200, output=10, cached=64)]
    models = SimpleNamespace(generate_content=lambda **kw: replies.pop(0))
    monkeypatch.setattr(L, "get_client", lambda: SimpleNamespace(models=models))
    assert U.snapshot()["calls"] == 0 and U.snapshot()["cache_hit_rate"] is None
    L.generate_text("review this run", cache=True)             # miss -> 1 call
    L.generate_text("review this run", cache=True)             # hit  -> no call
    L.generate_text("a chat question")                         # not cacheable -> 1 call, no cache lookup
    u = U.snapshot()
    assert (u["calls"], u["failed"], u["retries"]) == (2, 0, 0) and u["calls_by_kind"] == {"generate": 2, "embed": 0}
    assert (u["cache_hits"], u["cache_misses"]) == (1, 1) and u["cache_hit_rate"] == 0.5
    assert (u["prompt_tokens"], u["output_tokens"], u["thinking_tokens"], u["cached_tokens"]) == (320, 40, 50, 64)
    assert u["seconds"] >= 0 and u["seconds_per_call"] is not None


def test_usage_counts_failed_attempts_and_retries(keys):
    flaky = Flaky(client_error(429, "quota"), TimeoutError("timed out"))
    assert L.call_with_retry(flaky, sleep=lambda s: None) == "ok"
    u = U.snapshot()
    assert (u["calls"], u["failed"], u["retries"]) == (3, 2, 2)
    with pytest.raises(L.LLMError):
        L.call_with_retry(Flaky(client_error(400, "bad request")), sleep=lambda s: None)
    u = U.snapshot()
    assert (u["calls"], u["failed"], u["retries"]) == (4, 3, 2)   # a fatal error is not retried


def test_usage_counts_embedding_batches_and_per_text_cache(fake_client, monkeypatch):
    monkeypatch.setattr(L, "EMBED_BATCH_SIZE", 2)
    L.embed_texts(["a", "bb", "ccc"])                           # 3 misses, 2 batches
    L.embed_texts(["a", "dddd"])                                # 1 hit, 1 miss, 1 batch
    u = U.snapshot()
    assert u["calls_by_kind"] == {"generate": 0, "embed": 3} and u["calls"] == 3
    assert (u["cache_hits"], u["cache_misses"]) == (1, 4)
    assert u["prompt_tokens"] == 0                              # embedding responses carry no usage_metadata here
    U.reset()
    assert U.snapshot()["calls"] == 0 and U.snapshot()["calls_by_kind"] == {"generate": 0, "embed": 0}


def test_usage_ignores_missing_or_odd_token_fields():
    assert U.read_tokens(SimpleNamespace(text="x")) == {
        "prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cached_tokens": 0}
    odd = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count="12", candidates_token_count=-5,
                                                         thoughts_token_count=True, cached_content_token_count=7))
    assert U.read_tokens(odd) == {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cached_tokens": 7}


def test_thinking_budget_only_for_models_that_support_it():
    extended = [label for label, budget in L.THINKING_LEVELS.items() if budget][0]
    standard = [label for label, budget in L.THINKING_LEVELS.items() if not budget][0]
    assert L.thinking_budget(extended, "gemini-2.5-flash") == 8192
    assert L.thinking_budget(extended, "gemini-2.5-pro") == 8192
    assert L.thinking_budget(extended, "gemini-2.0-flash-lite") is None
    assert L.thinking_budget(standard, "gemini-2.5-flash") is None
