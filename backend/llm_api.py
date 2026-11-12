"""
LLM API Management
==================
Handles Gemini configuration (google-genai SDK), API key rotation with
cooldowns, retry with exponential backoff, response caching and embeddings.

Design notes:
  * Each Streamlit session keeps its own key index/cooldowns in st.session_state;
    Gemini clients are cached per key, so sessions never change each other's key.
  * Errors are classified: quota (429) -> rotate key / back off, transient
    (5xx, timeouts) -> back off and retry, everything else (bad key, bad
    request, unknown model) -> fail fast with a friendly LLMError.
  * Identical cacheable requests (training narrative, fact-checks, embeddings)
    are served from an in-memory cache instead of calling the API again.
"""

import hashlib
import os
import random
import re
import threading
import time
from collections import OrderedDict

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# Real environment variables win over .env (override=False), and tests can
# skip .env entirely so a developer's real key is never used by the test suite.
if os.getenv("DLT_SKIP_DOTENV") != "1":
    load_dotenv(override=False)

# Accept multiple keys separated by commas, fallback to the old singular key variable if needed
API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()]

# --- Model Chooser Options (labels match the real model IDs) ---
# Checked against models.list() on 2026-09-30: gemini-2.0-flash-lite is retired;
# the stable 2.5 family below and gemini-embedding-001 are served.
GEMINI_MODELS = {
    "Gemini 2.5 Flash-Lite — Fastest answers": "gemini-2.5-flash-lite",
    "Gemini 2.5 Flash — All-around help": "gemini-2.5-flash",
    "Gemini 2.5 Pro — Advanced math and code": "gemini-2.5-pro",
}
DEFAULT_MODEL_LABEL = list(GEMINI_MODELS.keys())[1]
FAST_MODEL_ID = "gemini-2.5-flash-lite"      # cheap model for fact-checking
EMBEDDING_MODEL = "gemini-embedding-001"

# Thinking budget (tokens the model may spend reasoning before answering).
# None = the model's default behaviour.
THINKING_LEVELS = {
    "Standard — Best for most questions": None,
    "Extended — Complex problem solving (up to 8,192 thinking tokens)": 8192,
}


def supports_thinking(model_id):
    """Only the Gemini 2.5 (and newer) models accept a thinking budget."""
    return bool(re.match(r"gemini-(2\.5|[3-9])", str(model_id or "")))


def thinking_budget(label, model_id):
    """The thinking budget for the selected level, or None when the model cannot use one."""
    budget = THINKING_LEVELS.get(label)
    return budget if budget and supports_thinking(model_id) else None


KEY_COOLDOWN_SECONDS = 60
MAX_BACKOFF_SECONDS = 16
MAX_WAIT_FOR_RETRY_AFTER = 20
EMBED_BATCH_SIZE = 100


class LLMError(RuntimeError):
    """An LLM call failed; the message is safe and friendly to show to users."""


class LLMUnavailableError(LLMError):
    """No usable key right now (none configured, or all rate-limited)."""


# Global LLM handle: truthy when Gemini is configured (kept for backwards compatibility).
llm = None


class GeminiModel:
    """Lightweight handle for the selected Gemini model (no network, no key stored)."""

    def __init__(self, model_id):
        self.model_name = model_id

    def generate_content(self, contents, config=None):
        return generate(contents, model_id=self.model_name, config=config)

    def __repr__(self):
        return f"GeminiModel({self.model_name!r})"


# ═══════════════════════════════════════════════
# Session state helpers (safe outside Streamlit, e.g. in tests)
# ═══════════════════════════════════════════════

_fallback_state = {"current_key_index": 0, "key_cooldowns": {}, "gemini_model_label": DEFAULT_MODEL_LABEL}


def _state():
    """st.session_state inside a Streamlit run, a module-level dict otherwise (tests, scripts)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(suppress_warning=True) is None:
            return _fallback_state
        ss = st.session_state
    except Exception:
        return _fallback_state
    for k, v in _fallback_state.items():
        if k not in ss:
            ss[k] = {} if isinstance(v, dict) else v
    return ss


def _get(key):
    return _state()[key]


def _set(key, value):
    _state()[key] = value


def _notify(message):
    try:
        st.toast(message)
    except Exception:
        pass


# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

_clients = {}
_clients_lock = threading.Lock()


def get_client():
    """Gemini client for this session's current key (cached per key)."""
    if not API_KEYS:
        raise LLMUnavailableError("No Gemini API key is configured. Add GEMINI_API_KEYS to your .env file.")
    idx = _get("current_key_index")
    if idx >= len(API_KEYS):
        idx = 0
        _set("current_key_index", 0)
    key = API_KEYS[idx]
    with _clients_lock:
        if key not in _clients:
            _clients[key] = genai.Client(api_key=key)
        return _clients[key]


def current_model_id(label=None):
    label = label or _get("gemini_model_label")
    return GEMINI_MODELS.get(label, GEMINI_MODELS[DEFAULT_MODEL_LABEL])


def configure_gemini(model_label=None):
    """Configure the Gemini model for this session. Returns the model handle (or None)."""
    global llm
    if model_label:
        _set("gemini_model_label", model_label)
    if not API_KEYS:
        llm = None
        return None
    llm = GeminiModel(current_model_id())
    return llm


def is_configured():
    return bool(API_KEYS)


# ═══════════════════════════════════════════════
# Retry / rotation
# ═══════════════════════════════════════════════

_QUOTA_WORDS = ("429", "quota", "resource_exhausted", "resource exhausted", "rate limit")
_TRANSIENT_WORDS = ("503", "500", "502", "504", "unavailable", "overloaded", "deadline", "timed out", "timeout",
                    "internal error", "connection reset")


def classify_error(exc):
    """Return 'quota', 'transient' or 'fatal' for an exception from an LLM call."""
    if isinstance(exc, LLMError):
        return "fatal"
    code = getattr(exc, "code", None)
    if isinstance(exc, genai_errors.APIError) and isinstance(code, int):
        if code == 429:
            return "quota"
        if code >= 500 or code == 408:
            return "transient"
        return "fatal"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    name = type(exc).__name__.lower()
    if "timeout" in name or "connecterror" in name or "connectionerror" in name:
        return "transient"
    msg = str(exc).lower()
    if any(w in msg for w in _QUOTA_WORDS):
        return "quota"
    if any(w in msg for w in _TRANSIENT_WORDS):
        return "transient"
    return "fatal"


def friendly_error(exc):
    """Turn an API exception into a message that is safe to show (never includes keys)."""
    code = getattr(exc, "code", None)
    msg = str(exc).lower()
    if isinstance(exc, LLMError):
        return str(exc)
    if code in (401, 403) or "api key not valid" in msg or "permission" in msg or "api_key_invalid" in msg:
        return "The Gemini API key was rejected. Check GEMINI_API_KEYS in your .env file."
    if code == 404 or "not found" in msg:
        return ("The selected Gemini model is not available for this API key. "
                "Choose another model in the AI Backend settings.")
    if code == 400 or "invalid argument" in msg:
        return "Gemini rejected the request (it may be too long or contain an unsupported file)."
    kind = classify_error(exc)
    if kind == "quota":
        return "Gemini's rate limit was reached. Please wait about a minute and try again."
    if kind == "transient":
        return "Gemini is temporarily unavailable. Please try again in a moment."
    return f"The AI request failed ({type(exc).__name__})."


def _retry_after_seconds(exc):
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.I) or re.search(r"retryDelay['\"]?:\s*['\"]?([\d.]+)s", str(exc))
    return float(m.group(1)) if m else None


def _find_available_key():
    """Switch to the next key that isn't on cooldown. Returns True if one was found."""
    now = time.time()
    cooldowns = {k: v for k, v in dict(_get("key_cooldowns")).items() if v > now}
    _set("key_cooldowns", cooldowns)
    current = _get("current_key_index")
    for i in range(1, len(API_KEYS) + 1):
        candidate = (current + i) % len(API_KEYS)
        if candidate not in cooldowns:
            if candidate != current:
                _set("current_key_index", candidate)
                configure_gemini()
            return True
    return False


def call_with_retry(func, *args, max_attempts=None, base_delay=1.0, sleep=time.sleep, **kwargs):
    """Call `func` (which must fetch the client via get_client() on every call),
    rotating keys on quota errors and backing off exponentially (with jitter) on
    quota/transient errors. Raises LLMError with a friendly message on failure."""
    if not API_KEYS:
        raise LLMUnavailableError("No Gemini API key is configured. Add GEMINI_API_KEYS to your .env file.")
    attempts = max_attempts or max(3, min(len(API_KEYS) + 2, 6))
    last_exc = None
    for attempt in range(attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # classified below; unknown errors fail fast
            last_exc = e
            kind = classify_error(e)
            if kind == "fatal":
                raise LLMError(friendly_error(e)) from e
            if attempt == attempts - 1:
                break

            delay = min(MAX_BACKOFF_SECONDS, base_delay * (2 ** attempt)) * (0.5 + random.random() / 2)
            if kind == "quota":
                idx = _get("current_key_index")
                cooldowns = dict(_get("key_cooldowns"))
                cooldowns[idx] = time.time() + KEY_COOLDOWN_SECONDS
                _set("key_cooldowns", cooldowns)
                if len(API_KEYS) > 1 and _find_available_key():
                    _notify(f"🔑 Key {idx + 1}/{len(API_KEYS)} hit its rate limit — switched to key "
                            f"{_get('current_key_index') + 1}.")
                    delay = min(delay, 1.0)
                else:
                    retry_after = _retry_after_seconds(e)
                    if retry_after is not None:
                        if retry_after > MAX_WAIT_FOR_RETRY_AFTER:
                            break
                        delay = retry_after + 0.5
                    _notify(f"⏳ Rate limited — retrying in {delay:.0f}s…")
            sleep(delay)

    if last_exc is not None and classify_error(last_exc) == "quota":
        raise LLMUnavailableError(
            "All Gemini API keys have hit their rate limit. Please wait about a minute and try again "
            "(or add more keys to GEMINI_API_KEYS).") from last_exc
    raise LLMError(friendly_error(last_exc)) from last_exc


# ═══════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════

_response_cache = OrderedDict()
_embedding_cache = OrderedDict()
_cache_lock = threading.Lock()
RESPONSE_CACHE_SIZE = 256
EMBEDDING_CACHE_SIZE = 5000


def _cache_key(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(cache, key):
    with _cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    return None


def _cache_put(cache, key, value, limit):
    with _cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)


def clear_caches():
    with _cache_lock:
        _response_cache.clear()
        _embedding_cache.clear()


# ═══════════════════════════════════════════════
# Generation & embeddings
# ═══════════════════════════════════════════════

def build_config(json_mode=False, thinking_budget=None, system=None, max_output_tokens=None, **extra):
    kwargs = dict(extra)
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if thinking_budget:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(thinking_budget))
    if system:
        kwargs["system_instruction"] = system
    if max_output_tokens:
        kwargs["max_output_tokens"] = int(max_output_tokens)
    return types.GenerateContentConfig(**kwargs) if kwargs else None


def generate(contents, model_id=None, config=None, **retry_kwargs):
    """generate_content with key rotation + retries. Returns the SDK response."""
    model_id = model_id or current_model_id()

    def _call():
        return get_client().models.generate_content(model=model_id, contents=contents, config=config)
    return call_with_retry(_call, **retry_kwargs)


def generate_text(prompt, model_id=None, json_mode=False, thinking_budget=None, system=None,
                  cache=False, **retry_kwargs):
    """Text-in/text-out helper. With cache=True identical requests are answered from memory."""
    model_id = model_id or current_model_id()
    key = _cache_key("text", model_id, json_mode, thinking_budget, system, prompt) if cache else None
    if key:
        hit = _cache_get(_response_cache, key)
        if hit is not None:
            return hit
    config = build_config(json_mode=json_mode, thinking_budget=thinking_budget, system=system)
    response = generate(prompt, model_id=model_id, config=config, **retry_kwargs)
    text = (response.text or "").strip() if response is not None else ""
    if not text:
        raise LLMError("The AI returned an empty response. Please try again.")
    if key:
        _cache_put(_response_cache, key, text, RESPONSE_CACHE_SIZE)
    return text


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", **retry_kwargs):
    """Embed a list of texts with gemini-embedding-001 (batched, cached).
    Use task_type RETRIEVAL_DOCUMENT for knowledge-base chunks and RETRIEVAL_QUERY for questions."""
    texts = list(texts)
    out = [None] * len(texts)
    missing = []
    for i, t in enumerate(texts):
        hit = _cache_get(_embedding_cache, _cache_key("emb", EMBEDDING_MODEL, task_type, t))
        if hit is not None:
            out[i] = hit
        else:
            missing.append(i)

    for start in range(0, len(missing), EMBED_BATCH_SIZE):
        batch_idx = missing[start:start + EMBED_BATCH_SIZE]
        batch = [texts[i] for i in batch_idx]

        def _call():
            return get_client().models.embed_content(
                model=EMBEDDING_MODEL, contents=batch, config=types.EmbedContentConfig(task_type=task_type))
        result = call_with_retry(_call, **retry_kwargs)
        vectors = [list(e.values) for e in (result.embeddings or [])]
        if len(vectors) != len(batch):
            raise LLMError("The embedding service returned an unexpected number of vectors.")
        for i, vec in zip(batch_idx, vectors):
            out[i] = vec
            _cache_put(_embedding_cache, _cache_key("emb", EMBEDDING_MODEL, task_type, texts[i]), vec,
                       EMBEDDING_CACHE_SIZE)
    return out
