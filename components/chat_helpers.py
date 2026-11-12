"""
Chat Helper Functions
=====================
Response rendering, bilingual JSON parsing, Ollama calls, and hallucination detection.
"""

import io
import json
import re

import streamlit as st

from backend import llm_api
from backend.ollama_api import friendly_error as _ollama_friendly_error

# ═══════════════════════════════════════════════
# Markdown / code rendering
# ═══════════════════════════════════════════════

_FENCE_OPEN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*([\w+#.-]*)[^`\n]*$")
_CODE_LANG_ALIASES = {"py": "python", "python3": "python", "sh": "bash", "shell": "bash", "": "python"}


def split_markdown_and_code(text):
    """Split text into [("markdown", str, None) | ("code", str, language)] segments.

    Handles ``` and ~~~ fences, language tags, trailing spaces after the fence,
    and an unterminated final fence (e.g. a truncated reply) - the rest of the
    text is then treated as code instead of becoming giant markdown headings.
    """
    segments = []
    buf = []
    fence = None
    lang = None
    for line in text.split("\n"):
        if fence is None:
            m = _FENCE_OPEN.match(line)
            if m:
                if buf and "".join(buf).strip():
                    segments.append(("markdown", "\n".join(buf).strip("\n"), None))
                buf = []
                fence, lang = m.group(1), m.group(2).lower()
                continue
            buf.append(line)
        else:
            stripped = line.strip()
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= len(fence):
                segments.append(("code", "\n".join(buf), _CODE_LANG_ALIASES.get(lang, lang)))
                buf, fence, lang = [], None, None
                continue
            buf.append(line)
    if fence is not None:
        segments.append(("code", "\n".join(buf), _CODE_LANG_ALIASES.get(lang, lang)))
    elif buf and "\n".join(buf).strip():
        segments.append(("markdown", "\n".join(buf).strip("\n"), None))
    return segments


def render_response(text):
    """Split response into markdown and code blocks, rendering code with st.code()."""
    if not isinstance(text, str):
        text = str(text)
    text = _undo_double_escaping(text)
    for kind, content, lang in split_markdown_and_code(text):
        if kind == "code":
            st.code(content.strip("\n"), language=lang or "python")
        elif content.strip():
            st.markdown(content)


def _undo_double_escaping(text):
    """Some replies arrive with every newline written as a literal backslash-n
    (double-escaped JSON). Only then convert them - a normal multi-line reply
    keeps its "\\n" sequences, which are meaningful inside code (print("a\\nb"))."""
    if "\n" not in text and "\\n" in text:
        return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return text


# ═══════════════════════════════════════════════
# Bilingual JSON parsing
# ═══════════════════════════════════════════════

_OUTER_FENCE = re.compile(r"^\s*(`{3,}|~{3,})[ \t]*(json|javascript|js)?[ \t]*\r?\n(.*)\r?\n[ \t]*\1[ \t]*\s*$",
                          re.S | re.I)


def _strip_outer_json_fence(text):
    """Remove a fence that wraps the WHOLE reply, e.g. ```json { ... } ```.
    Fences inside the reply (code examples) are never touched."""
    m = _OUTER_FENCE.match(text)
    if m and (m.group(2) or m.group(3).lstrip().startswith(("{", "["))):
        return m.group(3).strip()
    return text


def _coerce_bilingual(data):
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None
    en, ml = data.get("en"), data.get("ml")
    if en is None and ml is None:
        return None
    en = en if isinstance(en, str) else json.dumps(en, ensure_ascii=False) if en is not None else None
    ml = ml if isinstance(ml, str) else json.dumps(ml, ensure_ascii=False) if ml is not None else None
    en = en if en is not None else ml
    ml = ml if ml is not None else en
    return {"en": _undo_double_escaping(en), "ml": _undo_double_escaping(ml)}


def _find_embedded_json(text):
    """Find a {...} object with "en"/"ml" keys inside surrounding prose."""
    decoder = json.JSONDecoder(strict=False)
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        result = _coerce_bilingual(obj)
        if result:
            return result
    return None


def _salvage_truncated_json(text):
    """Recover the "en" (and "ml") strings from JSON that was cut off mid-way."""
    out = {}
    for key in ("en", "ml"):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)' % key, text, re.S)
        if m:
            raw = m.group(1)
            try:
                out[key] = json.loads(f'"{raw}"', strict=False)
            except json.JSONDecodeError:
                out[key] = raw.replace('\\"', '"').replace("\\n", "\n")
    if not out:
        return None
    out.setdefault("en", out.get("ml"))
    out.setdefault("ml", out.get("en"))
    return out


def parse_bilingual(raw_text):
    """Parse bilingual JSON response from LLM, with fallback to plain text.

    Always returns {"en": str, "ml": str}. Code fences inside the answer are kept.
    """
    if raw_text is None:
        return {"en": "", "ml": ""}
    text = str(raw_text).strip()
    candidate = _strip_outer_json_fence(text)

    if candidate.startswith(("{", "[")):
        try:
            result = _coerce_bilingual(json.loads(candidate, strict=False))
            if result:
                return result
        except json.JSONDecodeError:
            salvaged = _salvage_truncated_json(candidate)
            if salvaged:
                return salvaged

    embedded = _find_embedded_json(candidate) if '"en"' in candidate else None
    if embedded:
        return embedded

    # Fallback: a normal markdown answer (possibly with ```python blocks) - keep it intact.
    plain = _undo_double_escaping(text)
    return {"en": plain, "ml": plain}


# ═══════════════════════════════════════════════
# Ollama
# ═══════════════════════════════════════════════

def call_ollama(ollama_client, ollama_model_name, prompt_text, images_pil=None, json_mode=False):
    """Call Ollama with the given prompt and optional images.

    On failure returns a bilingual JSON string (built with json.dumps, so quotes
    in error messages cannot break it) that parse_bilingual() turns into a message.
    """
    try:
        msg = {"role": "user", "content": prompt_text}
        if images_pil:
            encoded_images = []
            for img in images_pil:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                encoded_images.append(buf.getvalue())
            msg["images"] = encoded_images

        kwargs = {"model": ollama_model_name, "messages": [msg], "options": {"temperature": 0.7}}
        if json_mode:
            kwargs["format"] = "json"
            kwargs["options"]["temperature"] = 0.0
        response = ollama_client.chat(**kwargs)
        # Handle both dict-style and attribute-style response objects
        if hasattr(response, 'message'):
            msg_resp = response.message
            return msg_resp.content if hasattr(msg_resp, 'content') else msg_resp.get('content', str(msg_resp))
        elif isinstance(response, dict):
            return response['message']['content']
        else:
            return str(response)
    except Exception as e:
        message = f"⚠️ {_ollama_friendly_error(e, ollama_model_name)}"
        return json.dumps({"en": message, "ml": message}, ensure_ascii=False)


# ═══════════════════════════════════════════════
# Hallucination detector (LLM-as-judge)
# ═══════════════════════════════════════════════

class FactCheckError(ValueError):
    """The fact-checker's reply could not be understood."""


def build_fact_check_prompt(ai_answer_text, source_texts):
    sources = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(source_texts))
    return f"""You are a strict Fact-Checker. Given the SOURCE TEXT and the AI ANSWER below, determine:
1) A Grounding Score from 0-100 indicating what percentage of the AI answer is directly supported by the Source Text.
2) Any claims in the answer that appear to be hallucinated (not found in the Source Text).
3) A one-line verdict.

Output ONLY a JSON object:
{{"grounding_score": <number>, "hallucinated_claims": [<string>, ...], "verdict": "<string>"}}

--- SOURCE TEXT ---
{sources}

--- AI ANSWER ---
{ai_answer_text}
"""


def parse_fact_check(raw):
    """Parse the judge's reply into {"score": 0-100, "claims": [...], "verdict": str}.
    Tolerates code fences and prose around the JSON; raises FactCheckError otherwise."""
    if not raw or not str(raw).strip():
        raise FactCheckError("empty reply")
    text = _strip_outer_json_fence(str(raw).strip())
    data = None
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder(strict=False)
        for m in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text, m.start())
                if isinstance(data, dict) and "grounding_score" in data:
                    break
            except json.JSONDecodeError:
                continue
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict) or "grounding_score" not in data:
        raise FactCheckError("no grounding_score in reply")
    try:
        score = float(str(data["grounding_score"]).strip().rstrip("%"))
    except (TypeError, ValueError) as e:
        raise FactCheckError("grounding_score is not a number") from e
    if 0 < score <= 1 and "." in str(data["grounding_score"]):
        score *= 100                      # some models answer 0.85 instead of 85
    claims = data.get("hallucinated_claims") or []
    if isinstance(claims, str):
        claims = [claims]
    return {
        "score": int(round(min(100.0, max(0.0, score)))),
        "claims": [str(c) for c in claims if str(c).strip()][:10],
        "verdict": str(data.get("verdict", "")).strip(),
    }


def fact_check(ai_answer_text, source_texts, using_ollama=False, ollama_client=None, ollama_model_name=None):
    """Run the LLM judge with whichever backend is active. Returns the parsed dict.
    Raises FactCheckError / llm_api.LLMError on failure."""
    prompt = build_fact_check_prompt(ai_answer_text, source_texts)
    if using_ollama:
        if not (ollama_client and ollama_model_name):
            raise FactCheckError("Ollama is not ready")
        raw = call_ollama(ollama_client, ollama_model_name, prompt, json_mode=True)
    else:
        try:
            raw = llm_api.generate_text(prompt, model_id=llm_api.FAST_MODEL_ID, json_mode=True, cache=True)
        except llm_api.LLMError:
            # The fast model may be unavailable for this key - retry once with the selected model.
            raw = llm_api.generate_text(prompt, json_mode=True, cache=True)
    return parse_fact_check(raw)


def run_hallucination_check(ai_answer_text, lang_key, UI, using_ollama, sources=None,
                            ollama_client=None, ollama_model_name=None):
    """Run a second fact-checking call if the hallucination detector is enabled and KB is loaded.

    `sources` are the knowledge-base chunk texts that were retrieved for THIS question.
    """
    if not st.session_state.get("hallucination_detector"):
        return
    if not st.session_state.get("knowledge_base"):
        return
    if not ai_answer_text or not ai_answer_text.strip():
        return
    if not sources:
        st.caption(UI[lang_key]["hallucination_no_sources"])
        return
    if not using_ollama and not llm_api.is_configured():
        return

    with st.spinner(UI[lang_key]["hallucination_checking"]):
        try:
            result = fact_check(ai_answer_text, sources, using_ollama, ollama_client, ollama_model_name)
        except (FactCheckError, llm_api.LLMError):
            st.caption(UI[lang_key]["hallucination_failed"])
            return

    score = result["score"]
    if score >= 80:
        badge_color, badge_text = "🟢", UI[lang_key]["grounded_high"]
    elif score >= 50:
        badge_color, badge_text = "🟡", UI[lang_key]["grounded_mid"]
    else:
        badge_color, badge_text = "🔴", UI[lang_key]["grounded_low"]

    with st.expander(f"{badge_color} {UI[lang_key]['grounding_score']}: **{score}%** — {badge_text}", expanded=False):
        st.write(f"**{UI[lang_key]['verdict']}:** {result['verdict']}")
        if result["claims"]:
            st.write(f"**{UI[lang_key]['hallucinated_claims']}:**")
            for c in result["claims"]:
                st.write(f"- ⚠️ {c}")
        else:
            st.write(UI[lang_key]["no_hallucinated_claims"])
        st.caption(UI[lang_key]["grounding_note"])
