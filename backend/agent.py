"""
Agent Loop (Gemini function calling)
====================================
Runs the tutor's agentic loop: send the conversation to Gemini with the tool
declarations, execute any function calls it requests (run_experiment,
diagnose_training, ...), feed the results back, and repeat until the model
answers in text. Automatic function calling is disabled so the UI can show
every tool call as it happens.

UI-independent: the caller passes callbacks for progress display, and
`generate_fn` can be replaced by a fake in tests.
"""

import json

from google.genai import types

from agent_tools import TOOL_DECLARATIONS, execute_tool
from backend import llm_api

MAX_ITERATIONS = 5  # tool rounds per question (keeps token use bounded)


def tool_config():
    """Tools + disabled automatic calling (we execute tools ourselves)."""
    declarations = [types.FunctionDeclaration(name=d["name"], description=d["description"],
                                              parameters_json_schema=d["parameters"])
                    for d in TOOL_DECLARATIONS]
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def _final_answer_config():
    """Same tools, but the model is not allowed to call them (forces a text answer)."""
    cfg = tool_config()
    cfg.tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))
    return cfg


def slim_for_llm(result):
    """Strip heavy keys from tool results before sending them back to the LLM.
    Keeps history_summary (5 points) instead of the full epoch-by-epoch history."""
    if not isinstance(result, dict):
        return result
    slim = dict(result)
    slim.pop("history", None)
    slim.pop("model_summary", None)
    return slim


def build_user_parts(prompt_text, images=None):
    """Text + optional PIL images as google-genai Parts."""
    import io
    parts = [types.Part(text=prompt_text)]
    for img in images or []:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    return parts


def run_agent(user_parts, model_id=None, generate_fn=None, execute_fn=execute_tool,
              max_iterations=MAX_ITERATIONS, on_iteration=None, on_tool_start=None, on_tool_end=None):
    """Run the agent loop.

    Returns {"text": final_text, "experiments": [successful run_experiment results],
             "tool_calls": [(name, args, result), ...]}.
    Raises llm_api.LLMError (friendly message) if Gemini cannot be reached.
    """
    generate_fn = generate_fn or (lambda contents, config: llm_api.generate(contents, model_id=model_id, config=config))
    contents = [types.Content(role="user", parts=list(user_parts))]
    config = tool_config()
    experiments, tool_calls = [], []
    seen_calls = {}
    response = None

    for iteration in range(max_iterations):
        if on_iteration:
            on_iteration(iteration)
        response = generate_fn(contents, config)
        calls = response.function_calls or []
        if not calls:
            return {"text": (response.text or "").strip(), "experiments": experiments, "tool_calls": tool_calls}

        contents.append(response.candidates[0].content)
        response_parts = []
        for fc in calls:
            name = fc.name or ""
            args = dict(fc.args or {})
            if on_tool_start:
                on_tool_start(name, args)
            key = (name, json.dumps(args, sort_keys=True, default=str))
            if key in seen_calls:            # a loop: same tool, same arguments, same (deterministic) answer
                result = dict(seen_calls[key])
                result["note"] = ("You already called this tool with exactly these arguments; this is the same "
                                  "result. Do not call it again - use it to answer.")
            else:
                result = execute_fn(name, args)
                seen_calls[key] = result
            if on_tool_end:
                on_tool_end(name, args, result)
            tool_calls.append((name, args, result))
            if (name == "run_experiment" and isinstance(result, dict) and result.get("status") == "success"
                    and "note" not in result):
                experiments.append(result)
            response_parts.append(types.Part.from_function_response(
                name=name, response={"result": json.dumps(slim_for_llm(result), default=str)}))
        contents.append(types.Content(role="user", parts=response_parts))

    # Out of tool rounds: ask for the final answer without allowing more tool calls.
    contents.append(types.Content(role="user", parts=[types.Part(
        text="You have used all available tool calls. Now give your final answer based on the results above.")]))
    response = generate_fn(contents, _final_answer_config())
    return {"text": (response.text or "").strip(), "experiments": experiments, "tool_calls": tool_calls}
