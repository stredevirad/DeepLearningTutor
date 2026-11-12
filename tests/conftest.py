"""
Shared pytest configuration.

Keeps the suite hermetic:
  * `.env` is never loaded and no Gemini key is visible to the code under test.
  * Every outbound network connection is refused (only loopback is allowed,
    and even loopback is refused on the Ollama port so a locally running
    Ollama server can never be reached by accident).
  * TensorFlow runs quietly and deterministically.
"""

import os
import socket
import sys
from pathlib import Path

# --- Environment must be set BEFORE any project module is imported ---
os.environ["DLT_SKIP_DOTENV"] = "1"
os.environ["GEMINI_API_KEYS"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ.pop("MCP_AUTH_TOKEN", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_BLOCKED_LOCAL_PORTS = {11434, 8765}  # Ollama, MCP server


def _check_address(address):
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        if host not in _LOOPBACK:
            raise ConnectionRefusedError(f"Network access is blocked in tests (tried {host}:{port})")
        if port in _BLOCKED_LOCAL_PORTS:
            raise ConnectionRefusedError(f"Local service port {port} is blocked in tests")


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        _check_address(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check_address(address)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in _LOOPBACK and host is not None:
            raise socket.gaierror(f"DNS lookups are blocked in tests (tried {host})")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield


@pytest.fixture
def synthetic_images():
    """Tiny, download-free image dataset: a bright 8x8 square whose position
    (top-left vs bottom-right) defines the class. Returns (X, y, masks)."""
    rng = np.random.default_rng(0)
    n = 160
    X = rng.uniform(0.0, 0.08, size=(n, 28, 28, 1)).astype(np.float32)
    y = np.arange(n) % 2
    masks = np.zeros((n, 28, 28), dtype=bool)
    for i in range(n):
        if y[i] == 0:
            r, c = rng.integers(2, 7), rng.integers(2, 7)
        else:
            r, c = rng.integers(14, 19), rng.integers(14, 19)
        X[i, r:r + 8, c:c + 8, 0] = rng.uniform(0.8, 1.0)
        masks[i, r:r + 8, c:c + 8] = True
    return X, y, masks


@pytest.fixture(autouse=True)
def _fresh_usage_meter():
    """The API usage meter falls back to one module-level dict outside Streamlit; start each test at zero."""
    module = sys.modules.get("backend.usage")
    if module is not None:
        module.reset()
    yield


@pytest.fixture(autouse=True)
def _fresh_experiment_cache():
    """agent_tools reuses identical experiments within a process; tests must not share them."""
    module = sys.modules.get("agent_tools")
    if module is not None:
        module._EXPERIMENT_CACHE.clear()
    yield
