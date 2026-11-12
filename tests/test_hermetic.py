"""Guards that the test environment itself is hermetic (rule: no real LLM calls)."""

import socket

import pytest


def test_external_network_is_blocked():
    with pytest.raises((ConnectionRefusedError, socket.gaierror, OSError)):
        socket.create_connection(("example.com", 443), timeout=1)


def test_ollama_port_is_blocked():
    s = socket.socket()
    try:
        with pytest.raises(ConnectionRefusedError):
            s.connect(("127.0.0.1", 11434))
    finally:
        s.close()


def test_no_gemini_keys_visible():
    import backend.llm_api as llm_api
    assert llm_api.API_KEYS == []
