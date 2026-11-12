"""Unit tests for backend/ollama_api.py (fake clients, mocked subprocess - no real Ollama)."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from backend import ollama_api
from backend.ollama_api import OllamaError


# ─────────────────────────── fakes ───────────────────────────

class FakeClient:
    """Stands in for ollama.Client. `list_results` is consumed one per list() call;
    an Exception instance is raised instead of returned."""

    def __init__(self, list_results=None, embed_fn=None):
        self.list_results = list(list_results or [])
        self.list_calls = 0
        self.embed_fn = embed_fn
        self.embed_calls = []

    def list(self):
        self.list_calls += 1
        result = self.list_results.pop(0) if len(self.list_results) > 1 else self.list_results[0]
        if isinstance(result, Exception):
            raise result
        return result

    def embed(self, model, input):
        self.embed_calls.append((model, list(input)))
        return self.embed_fn(model, input)


class FakeClock:
    """Deterministic time.monotonic / time.sleep pair."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def attr_list_response(*names):
    return SimpleNamespace(models=[SimpleNamespace(model=n) for n in names])


# ─────────────────────────── host / client ───────────────────────────

@pytest.mark.parametrize("env, expected", [
    (None, "http://127.0.0.1:11434"),
    ("", "http://127.0.0.1:11434"),
    ("   ", "http://127.0.0.1:11434"),
    ("localhost:9999", "http://localhost:9999"),
    ("http://10.0.0.5:11434", "http://10.0.0.5:11434"),
    ("https://ollama.example.test", "https://ollama.example.test"),
    ("  127.0.0.1:11434  ", "http://127.0.0.1:11434"),
])
def test_get_host_normalises(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", env)
    assert ollama_api.get_host() == expected


def test_make_client_uses_host(monkeypatch):
    created = []
    monkeypatch.setattr(ollama_api, "OLLAMA_PACKAGE_AVAILABLE", True)
    monkeypatch.setattr(ollama_api, "ollama", SimpleNamespace(Client=lambda host: created.append(host) or "client"))
    monkeypatch.setenv("OLLAMA_HOST", "localhost:1234")
    assert ollama_api.make_client() == "client"
    assert ollama_api.make_client("http://other:1") == "client"
    assert created == ["http://localhost:1234", "http://other:1"]


def test_make_client_without_package(monkeypatch):
    monkeypatch.setattr(ollama_api, "OLLAMA_PACKAGE_AVAILABLE", False)
    assert ollama_api.make_client() is None


# ─────────────────────────── executable discovery ───────────────────────────

def test_find_executable_prefers_install_locations(monkeypatch, tmp_path):
    exe = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(ollama_api.shutil, "which", lambda name: "/should/not/be/used")
    assert ollama_api.find_executable() == str(exe)


def test_find_executable_falls_back_to_path_or_none(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "nothing"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "nothing"))
    monkeypatch.setattr(ollama_api.os.path, "isfile", lambda p: False)
    monkeypatch.setattr(ollama_api.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None)
    assert ollama_api.find_executable() == "/usr/bin/ollama"
    monkeypatch.setattr(ollama_api.shutil, "which", lambda name: None)
    assert ollama_api.find_executable() is None


# ─────────────────────────── model listing ───────────────────────────

def test_list_models_attribute_style():
    client = FakeClient([attr_list_response("llama3.2:1b", "nomic-embed-text:latest")])
    assert ollama_api.list_models(client) == ["llama3.2:1b", "nomic-embed-text:latest"]


def test_list_models_dict_style_model_or_name_keys():
    client = FakeClient([{"models": [{"model": "mistral:7b"}, {"name": "gemma2:2b"}, {"size": 1}]}])
    assert ollama_api.list_models(client) == ["mistral:7b", "gemma2:2b"]


def test_list_models_empty_and_none():
    assert ollama_api.list_models(FakeClient([{"models": []}])) == []
    assert ollama_api.list_models(FakeClient([SimpleNamespace(models=None)])) == []


def test_list_models_unreachable_server_raises_friendly_error():
    client = FakeClient([ConnectionError("[WinError 10061] connection refused")])
    with pytest.raises(OllamaError, match="not running"):
        ollama_api.list_models(client)


@pytest.mark.parametrize("installed, name, expected", [
    (["llama3.2:1b"], "llama3.2:1b", True),
    (["llama3.2:3b"], "llama3.2:1b", False),
    (["nomic-embed-text:latest"], "nomic-embed-text", True),
    (["llama3:latest"], "llama3", True),
    (["llama3.2:1b"], "llama3", False),        # a different model family must not match
    (["mistral:7b"], "mistral:7b-instruct", False),
    ([], "llama3.2:1b", False),
])
def test_has_model(installed, name, expected):
    assert ollama_api.has_model(installed, name) is expected


# ─────────────────────────── starting the server ───────────────────────────

def test_start_server_without_ollama_installed(monkeypatch):
    monkeypatch.setattr(ollama_api, "find_executable", lambda: None)
    popen_calls = []
    monkeypatch.setattr(ollama_api.subprocess, "Popen", lambda *a, **k: popen_calls.append(a))
    with pytest.raises(OllamaError, match="not installed"):
        ollama_api.start_server()
    assert popen_calls == []


def test_start_server_runs_fixed_command_and_polls_until_ready(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(ollama_api.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(ollama_api.time, "sleep", clock.sleep)
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(pid=1)
    monkeypatch.setattr(ollama_api.subprocess, "Popen", fake_popen)

    refused = ConnectionError("refused")
    client = FakeClient([refused, refused, attr_list_response("llama3.2:1b")])
    assert ollama_api.start_server("C:/Ollama/ollama.exe", client, wait_seconds=10, poll=0.5) is True

    args, kwargs = calls[0]
    assert args == ["C:/Ollama/ollama.exe", "serve"]           # fixed argv, no shell, no user text
    assert "shell" not in kwargs
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["stdout"] is subprocess.DEVNULL
    if os.name == "nt":
        assert "creationflags" in kwargs
    else:
        assert kwargs["start_new_session"] is True
    assert client.list_calls == 3 and clock.sleeps == [0.5, 0.5]


def test_start_server_times_out(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(ollama_api.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(ollama_api.time, "sleep", clock.sleep)
    monkeypatch.setattr(ollama_api.subprocess, "Popen", lambda *a, **k: None)
    client = FakeClient([ConnectionError("refused")])
    assert ollama_api.start_server("ollama", client, wait_seconds=2.0, poll=0.5) is False
    assert clock.now >= 2.0


def test_start_server_popen_failure_is_friendly(monkeypatch):
    def broken_popen(*a, **k):
        raise PermissionError("access denied")
    monkeypatch.setattr(ollama_api.subprocess, "Popen", broken_popen)
    with pytest.raises(OllamaError, match="Could not start Ollama"):
        ollama_api.start_server("ollama", FakeClient([ConnectionError()]))


# ─────────────────────────── error messages ───────────────────────────

def test_friendly_error_model_missing():
    exc = type("ResponseError", (Exception,), {})("model 'llama3' not found")
    exc.status_code = 404
    msg = ollama_api.friendly_error(exc, "llama3")
    assert "`llama3` is not downloaded" in msg and "ollama pull llama3" in msg


def test_friendly_error_connection_and_generic():
    assert "Cannot reach the Ollama server" in ollama_api.friendly_error(ConnectionRefusedError("refused"))
    assert "Cannot reach the Ollama server" in ollama_api.friendly_error(RuntimeError("failed to connect"))
    generic = ollama_api.friendly_error(ValueError("weird secret-looking detail"))
    assert generic == "Ollama request failed (ValueError)."     # raw exception text is not shown


# ─────────────────────────── embeddings ───────────────────────────

def test_embed_batches_and_preserves_order():
    def fn(model, texts):
        return SimpleNamespace(embeddings=[[float(len(t)), 1.0] for t in texts])
    client = FakeClient(embed_fn=fn)
    texts = [f"text {'x' * i}" for i in range(5)]
    vectors = ollama_api.embed(client, texts, batch_size=2)
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]
    assert [len(batch) for _, batch in client.embed_calls] == [2, 2, 1]
    assert all(model == ollama_api.DEFAULT_EMBED_MODEL for model, _ in client.embed_calls)


def test_embed_dict_style_response():
    client = FakeClient(embed_fn=lambda model, texts: {"embeddings": [[0.5, 0.25] for _ in texts]})
    assert ollama_api.embed(client, ["a", "b"]) == [[0.5, 0.25], [0.5, 0.25]]


def test_embed_wrong_count_is_an_error():
    client = FakeClient(embed_fn=lambda model, texts: {"embeddings": [[1.0]]})
    with pytest.raises(OllamaError, match="no embeddings"):
        ollama_api.embed(client, ["a", "b"])


def test_embed_missing_model_is_explained():
    def fn(model, texts):
        exc = RuntimeError(f"model '{model}' not found, try pulling it first")
        exc.status_code = 404
        raise exc
    with pytest.raises(OllamaError, match="ollama pull nomic-embed-text"):
        ollama_api.embed(FakeClient(embed_fn=fn), ["a"])


def test_embed_empty_input_makes_no_calls():
    client = FakeClient(embed_fn=lambda model, texts: pytest.fail("should not be called"))
    assert ollama_api.embed(client, []) == []


# ─────────────────────────── cached server status ───────────────────────────

@pytest.fixture
def fresh_status(monkeypatch):
    ollama_api.invalidate_status()
    clock = FakeClock()
    monkeypatch.setattr(ollama_api.time, "monotonic", clock.monotonic)
    yield clock
    ollama_api.invalidate_status()


def test_server_status_down_is_cached(monkeypatch, fresh_status):
    probes = []
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: probes.append(timeout) or False)
    assert ollama_api.server_status("http://127.0.0.1:11434") == (False, [])
    assert ollama_api.server_status("http://127.0.0.1:11434") == (False, [])
    assert probes == [0.3]                         # short loopback probe, asked only once


def test_server_status_up_lists_models_and_expires(monkeypatch, fresh_status):
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: True)
    client = FakeClient([attr_list_response("llama3.2:1b")])
    host = "http://127.0.0.1:11434"
    assert ollama_api.server_status(host, client=client) == (True, ["llama3.2:1b"])
    assert ollama_api.server_status(host, client=client) == (True, ["llama3.2:1b"])
    assert client.list_calls == 1
    fresh_status.now += ollama_api.STATUS_TTL_SECONDS + 1          # cache entry expires
    ollama_api.server_status(host, client=client)
    ollama_api.server_status(host, client=client, refresh=True)    # explicit re-check
    assert client.list_calls == 3


def test_server_status_port_open_but_api_fails(monkeypatch, fresh_status):
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: True)
    client = FakeClient([ConnectionError("reset")])
    assert ollama_api.server_status("http://10.0.0.9:11434", client=client) == (False, [])


def test_server_status_remote_host_uses_longer_probe(monkeypatch, fresh_status):
    probes = []
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: probes.append(timeout) or False)
    ollama_api.server_status("http://10.0.0.9:11434")
    assert probes == [1.5]


def test_invalidate_status_forces_a_new_probe(monkeypatch, fresh_status):
    probes = []
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: probes.append(host) or False)
    ollama_api.server_status("http://127.0.0.1:11434")
    ollama_api.invalidate_status("http://127.0.0.1:11434")
    ollama_api.server_status("http://127.0.0.1:11434")
    assert len(probes) == 2


def test_real_probe_of_blocked_port_is_fast_and_down():
    """The test harness refuses the Ollama port; the probe must report 'down' without hanging."""
    ollama_api.invalidate_status()
    import time as _time
    t0 = _time.perf_counter()
    assert ollama_api.server_status("http://127.0.0.1:11434", refresh=True) == (False, [])
    assert _time.perf_counter() - t0 < 2.0
    ollama_api.invalidate_status()


def test_start_server_success_clears_cached_status(monkeypatch, fresh_status):
    monkeypatch.setattr(ollama_api.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: False)
    ollama_api.server_status("http://127.0.0.1:11434")               # cached as "down"
    assert ollama_api.start_server("ollama", FakeClient([attr_list_response()])) is True
    monkeypatch.setattr(ollama_api, "_port_open", lambda host, timeout: True)
    up = ollama_api.server_status("http://127.0.0.1:11434", client=FakeClient([attr_list_response("m:1")]))
    assert up == (True, ["m:1"])
