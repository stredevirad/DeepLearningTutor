"""
Ollama (local, offline LLM) helpers
===================================
Finding / starting the Ollama server (Windows, macOS, Linux), listing models
and computing embeddings for the knowledge base.

No user-controlled text is ever passed to a subprocess: the executable path
comes from fixed install locations or PATH, and the only argument is "serve".
"""

import os
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

try:
    import ollama
    OLLAMA_PACKAGE_AVAILABLE = True
except ImportError:  # the app still works with Gemini only
    ollama = None
    OLLAMA_PACKAGE_AVAILABLE = False

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaError(RuntimeError):
    """Friendly Ollama failure message."""


def get_host():
    host = os.getenv("OLLAMA_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def make_client(host=None):
    if not OLLAMA_PACKAGE_AVAILABLE:
        return None
    return ollama.Client(host=host or get_host())


def find_executable():
    """Path to the Ollama executable, or None if Ollama is not installed."""
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Ollama", "ollama.exe"),
        "/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/homebrew/bin/ollama",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return shutil.which("ollama")


def list_models(client):
    """Names of locally installed models. Raises OllamaError if the server is unreachable."""
    try:
        response = client.list()
    except Exception as e:
        raise OllamaError("The Ollama server is not running.") from e
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    names = []
    for m in models or []:
        name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
        if name:
            names.append(name)
    return names


STATUS_TTL_SECONDS = 20.0
_status_cache = {}               # host -> (monotonic time, (running, models))
_status_lock = threading.Lock()


def _port_open(host_url, timeout):
    parsed = urlparse(host_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def server_status(host=None, ttl=STATUS_TTL_SECONDS, refresh=False, client=None):
    """(running, installed_model_names) for the Ollama server, remembered for `ttl` seconds.

    Streamlit reruns the whole script on every click, so the sidebar must not ask
    the server each time. A stopped server is detected with a short TCP probe:
    on Windows a refused loopback connection otherwise takes about 2 seconds,
    because the connection attempt is retried.
    """
    host = host or get_host()
    with _status_lock:
        hit = _status_cache.get(host)
        if hit and not refresh and time.monotonic() - hit[0] < ttl:
            return hit[1]
    loopback = urlparse(host).hostname in ("127.0.0.1", "localhost", "::1")
    status = (False, [])
    if _port_open(host, 0.3 if loopback else 1.5):
        try:
            status = (True, list_models(client or make_client(host)))
        except OllamaError:
            pass
    with _status_lock:
        _status_cache[host] = (time.monotonic(), status)
    return status


def invalidate_status(host=None):
    """Forget the cached status (after starting the server or pulling a model)."""
    with _status_lock:
        if host is None:
            _status_cache.clear()
        else:
            _status_cache.pop(host, None)


def has_model(installed, name):
    """True if `name` (e.g. 'llama3.2:1b' or 'nomic-embed-text') is installed."""
    return any(m == name or m.startswith(f"{name}:") or (":" not in name and m.split(":")[0] == name)
               for m in installed)


def start_server(executable=None, client=None, wait_seconds=15.0, poll=0.5):
    """Start `ollama serve` in the background and wait until it answers.
    Returns True if the server is reachable afterwards."""
    exe = executable or find_executable()
    if not exe:
        raise OllamaError("Ollama is not installed. Download it from https://ollama.com and run the installer.")
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([exe, "serve"], **kwargs)
    except OSError as e:
        raise OllamaError("Could not start Ollama. Try starting it manually with `ollama serve`.") from e

    client = client or make_client()
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            list_models(client)
            invalidate_status()
            return True
        except OllamaError:
            time.sleep(poll)
    return False


def friendly_error(exc, model_name=None):
    """Explain an Ollama exception in plain language."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status == 404 or "not found" in text:
        name = model_name or "the model"
        return f"Ollama model `{name}` is not downloaded. Run `ollama pull {name}` (or use the download button)."
    if isinstance(exc, (ConnectionError, OSError)) or "connect" in text or "refused" in text:
        return "Cannot reach the Ollama server. Start it with `ollama serve` (or the Start button in the sidebar)."
    return f"Ollama request failed ({type(exc).__name__})."


def embed(client, texts, model=DEFAULT_EMBED_MODEL, batch_size=64):
    """Embed texts with a local Ollama embedding model. Raises OllamaError on failure."""
    vectors = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            response = client.embed(model=model, input=batch)
            got = getattr(response, "embeddings", None)
            if got is None and isinstance(response, dict):
                got = response.get("embeddings")
            if not got or len(got) != len(batch):
                raise OllamaError("Ollama returned no embeddings.")
            vectors.extend([list(map(float, v)) for v in got])
    except OllamaError:
        raise
    except Exception as e:
        raise OllamaError(friendly_error(e, model)) from e
    return vectors
