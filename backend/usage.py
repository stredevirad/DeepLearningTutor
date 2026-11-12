"""
API usage meter
===============
Counts what the Gemini layer costs, so that efficiency can be shown with numbers:

  * calls        - requests actually sent to Gemini (every attempt counts, failed ones too)
  * failed       - attempts that raised an error
  * retries      - attempts made after a first attempt failed
  * cache hits   - cacheable requests answered from memory (no call made)
  * cache misses - cacheable requests that had to be sent
  * tokens       - prompt / output / thinking / cached, read from response.usage_metadata
  * seconds      - time spent waiting for Gemini (the calls themselves, not the back-off sleeps)

backend/llm_api.py calls record_call / record_retry / record_cache; nothing else
writes here. The totals live in st.session_state (one meter per browser session);
outside a Streamlit run (tests, scripts) a module-level dict is used instead, the
same arrangement as llm_api._state.
"""

SESSION_KEY = "api_usage"
KINDS = ("generate", "embed")
COUNTERS = ("calls", "failed", "retries", "cache_hits", "cache_misses",
            "prompt_tokens", "output_tokens", "thinking_tokens", "cached_tokens")


def new_totals():
    totals = {name: 0 for name in COUNTERS}
    totals["seconds"] = 0.0
    totals["calls_by_kind"] = {kind: 0 for kind in KINDS}
    return totals


_fallback_totals = new_totals()


def _totals():
    """This session's totals inside a Streamlit run, a module-level dict otherwise."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(suppress_warning=True) is None:
            return _fallback_totals
        if SESSION_KEY not in st.session_state:
            st.session_state[SESSION_KEY] = new_totals()
        return st.session_state[SESSION_KEY]
    except Exception:
        return _fallback_totals


def _count(value):
    """A token count as a non-negative int (the SDK leaves unused fields as None)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def read_tokens(response):
    """Token counts from a generate_content response's usage_metadata (zeros when there is none,
    e.g. for embedding responses or a fake client in tests)."""
    meta = getattr(response, "usage_metadata", None)
    return {
        "prompt_tokens": _count(getattr(meta, "prompt_token_count", None)),
        "output_tokens": _count(getattr(meta, "candidates_token_count", None)),
        "thinking_tokens": _count(getattr(meta, "thoughts_token_count", None)),
        "cached_tokens": _count(getattr(meta, "cached_content_token_count", None)),
    }


def record_call(kind, seconds, response=None, ok=True):
    """One request was sent to Gemini and took `seconds`. `response` is None when it failed."""
    totals = _totals()
    totals["calls"] += 1
    totals["calls_by_kind"][kind] = totals["calls_by_kind"].get(kind, 0) + 1
    totals["seconds"] += max(0.0, float(seconds))
    if not ok:
        totals["failed"] += 1
        return
    for name, value in read_tokens(response).items():
        totals[name] += value


def record_stream_end(seconds, last_chunk=None):
    """A streamed answer has finished: add the time spent reading it and the token counts,
    which Gemini sends with the last chunk. The call itself was counted when the stream opened."""
    totals = _totals()
    totals["seconds"] += max(0.0, float(seconds))
    for name, value in read_tokens(last_chunk).items():
        totals[name] += value


def record_retry():
    """A failed attempt is about to be tried again."""
    _totals()["retries"] += 1


def record_cache(hit, n=1):
    """`n` cacheable requests were answered from memory (hit) or had to be sent (miss)."""
    _totals()["cache_hits" if hit else "cache_misses"] += int(n)


def snapshot():
    """A copy of the totals plus two derived numbers, for display and for measurements."""
    totals = _totals()
    out = {name: totals[name] for name in COUNTERS}
    out["seconds"] = round(totals["seconds"], 3)
    out["calls_by_kind"] = dict(totals["calls_by_kind"])
    lookups = out["cache_hits"] + out["cache_misses"]
    out["cache_hit_rate"] = out["cache_hits"] / lookups if lookups else None
    out["seconds_per_call"] = round(totals["seconds"] / out["calls"], 3) if out["calls"] else None
    return out


def reset():
    """Start counting from zero (this session only)."""
    totals = _totals()
    totals.clear()
    totals.update(new_totals())
