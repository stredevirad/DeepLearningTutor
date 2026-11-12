"""Agent mode function-calling loop with mocked Gemini responses (real google-genai types)."""

import json

from google.genai import types

from backend import agent as A


def fc_response(*calls):
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


def text_response(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]))])


class ScriptedGemini:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, contents, config):
        self.calls.append((list(contents), config))
        return self.responses.pop(0)


def fake_execute(name, args):
    if name == "run_experiment":
        return {"status": "success", "network_type": "Simple NN", "dataset": args.get("dataset"),
                "results": {"final_val_accuracy": 0.97}, "history": {"loss": [1] * 100}, "history_summary": {}}
    return {"status": "ready", "name": name}


def test_tool_call_then_final_answer():
    gemini = ScriptedGemini(fc_response(("run_experiment", {"dataset": "Iris", "epochs": 5}),
                                        ("explain_concept", {"topic": "dropout"})),
                            text_response('{"en": "Done", "ml": "കഴിഞ്ഞു"}'))
    started, ended = [], []
    out = A.run_agent(A.build_user_parts("optimise iris"), generate_fn=gemini, execute_fn=fake_execute,
                      on_tool_start=lambda n, a: started.append(n), on_tool_end=lambda n, a, r: ended.append(r["status"]))
    assert out["text"] == '{"en": "Done", "ml": "കഴിഞ്ഞു"}'
    assert started == ["run_experiment", "explain_concept"] and ended == ["success", "ready"]
    assert len(out["experiments"]) == 1 and out["experiments"][0]["dataset"] == "Iris"

    # Second request contains: user turn, model's function-call turn, our function responses.
    second_contents = gemini.calls[1][0]
    assert [c.role for c in second_contents] == ["user", "model", "user"]
    responses = [p.function_response for p in second_contents[2].parts]
    assert [r.name for r in responses] == ["run_experiment", "explain_concept"]
    sent = json.loads(responses[0].response["result"])
    assert "history" not in sent                                    # heavy history is slimmed away


def test_config_declares_all_tools_and_disables_auto_calling():
    cfg = A.tool_config()
    names = [d.name for d in cfg.tools[0].function_declarations]
    assert names == ["run_experiment", "generate_code", "explain_concept", "compare_experiments", "diagnose_training"]
    assert cfg.automatic_function_calling.disable is True


def test_iteration_limit_forces_text_answer():
    looping = [fc_response(("explain_concept", {"topic": "x"})) for _ in range(3)]
    gemini = ScriptedGemini(*looping, text_response("Final summary"))
    out = A.run_agent(A.build_user_parts("q"), generate_fn=gemini, execute_fn=fake_execute, max_iterations=3)
    assert out["text"] == "Final summary"
    last_config = gemini.calls[-1][1]
    assert last_config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.NONE


def test_unknown_tool_from_model_is_reported_not_raised():
    from agent_tools import execute_tool
    gemini = ScriptedGemini(fc_response(("format_disk", {})), text_response("ok"))
    out = A.run_agent(A.build_user_parts("q"), generate_fn=gemini, execute_fn=execute_tool)
    name, _, result = out["tool_calls"][0]
    assert name == "format_disk" and result["status"] == "error"


def test_images_become_inline_parts():
    from PIL import Image
    parts = A.build_user_parts("what is this", [Image.new("RGB", (4, 4))])
    assert parts[1].inline_data.mime_type == "image/png"


def test_identical_repeated_tool_call_is_not_executed_again():
    executed = []

    def execute(name, args):
        executed.append(name)
        return fake_execute(name, args)
    gemini = ScriptedGemini(fc_response(("compare_experiments", {"experiment_summaries": "[1]"})),
                            fc_response(("compare_experiments", {"experiment_summaries": "[1]"})),
                            fc_response(("compare_experiments", {"experiment_summaries": "[1]"})),
                            text_response("final"))
    out = A.run_agent(A.build_user_parts("compare"), generate_fn=gemini, execute_fn=execute)
    assert executed == ["compare_experiments"]                 # executed once, then answered from memory
    assert out["text"] == "final" and len(out["tool_calls"]) == 3
    assert "Do not call it again" in out["tool_calls"][1][2]["note"]


def test_repeated_experiment_is_not_counted_twice():
    gemini = ScriptedGemini(fc_response(("run_experiment", {"dataset": "Iris"})),
                            fc_response(("run_experiment", {"dataset": "Iris"})),
                            text_response("done"))
    out = A.run_agent(A.build_user_parts("run"), generate_fn=gemini, execute_fn=fake_execute)
    assert len(out["experiments"]) == 1
