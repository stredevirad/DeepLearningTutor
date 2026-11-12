"""MCP server REST routes and the MCPClient (in-process, no real sockets)."""

import base64
import os
import tempfile
import time

import httpx
import pytest
from starlette.testclient import TestClient

import mcp_server as S
from mcp_client import MCPClient


@pytest.fixture
def client():
    with S._jobs_lock:
        S._training_jobs.clear()
    return TestClient(S.create_standalone_http_server())


def _wait_for(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.post("/call", json={"name": "get_training_status", "arguments": {"job_id": job_id}})
        assert r.status_code == 200
        if r.json()["status"] in ("completed", "failed"):
            return r.json()
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_default_bind_is_localhost():
    assert S.build_arg_parser().parse_args([]).host == "127.0.0.1"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["active_jobs"] == 0


def test_tools(client):
    names = [t["name"] for t in client.get("/tools").json()["tools"]]
    assert names == ["train_model_remote", "get_training_status"]


@pytest.mark.parametrize("body,status", [
    ({"name": "rm_rf", "arguments": {}}, 400),
    ({"name": "train_model_remote", "arguments": {"epochs": 10 ** 9}}, 400),
    ({"name": "train_model_remote", "arguments": {"network_type": "CNN", "dataset": "Iris"}}, 400),
    ({"name": "train_model_remote", "arguments": {"code": "__import__('os').system('x')"}}, 400),
    ({"name": "train_model_remote", "arguments": "not a dict"}, 400),
    ({"name": "get_training_status", "arguments": {"job_id": "../../etc"}}, 400),
    ({"name": "get_training_status", "arguments": {"job_id": "deadbeef"}}, 404),
])
def test_call_rejects_bad_requests(client, body, status):
    r = client.post("/call", json=body)
    assert r.status_code == status and "error" in r.json()


def test_call_rejects_non_json(client):
    r = client.post("/call", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_call_rejects_huge_body(client):
    r = client.post("/call", content=b"x" * (S.MAX_BODY_BYTES + 1))
    assert r.status_code == 413


@pytest.mark.slow
def test_train_job_runs_in_background_and_reports(client):
    r = client.post("/call", json={"name": "train_model_remote",
                                   "arguments": {"network_type": "Simple NN", "dataset": "Iris", "epochs": 2}})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] in ("queued", "running", "completed") and len(job["job_id"]) == 8

    done = _wait_for(client, job["job_id"])
    assert done["status"] == "completed", done
    assert done["result"]["results"]["final_val_accuracy"] >= 0
    assert len(done["history"]["loss"]) == 2
    assert "model_b64" not in done

    logs = client.get(f"/logs/{job['job_id']}").json()["logs"]
    assert any("Training completed" in line for line in logs)
    assert any("Epoch 2/2" in line for line in logs)

    with_model = client.post("/call", json={"name": "get_training_status",
                                            "arguments": {"job_id": job["job_id"], "include_model": True}}).json()
    import tensorflow as tf
    path = os.path.join(tempfile.mkdtemp(), "m.keras")
    with open(path, "wb") as f:
        f.write(base64.b64decode(with_model["model_b64"]))
    assert tf.keras.models.load_model(path).output_shape == (None, 3)


def test_logs_route_validation(client):
    assert client.get("/logs/..%2F..%2Fsecret").status_code in (400, 404)
    assert client.get("/logs/deadbeef").status_code == 404


def test_queue_limit(client, monkeypatch):
    monkeypatch.setattr(S, "MAX_PENDING_JOBS", 0)
    r = client.post("/call", json={"name": "train_model_remote", "arguments": {"epochs": 1}})
    assert r.status_code == 429


def test_auth_token(client, monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "s3cret")
    assert client.get("/health").status_code == 200              # health stays open
    assert client.get("/tools").status_code == 401
    assert client.post("/call", json={"name": "get_training_status", "arguments": {}}).status_code == 401
    assert client.get("/tools", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/tools", headers={"Authorization": "Bearer wrong"}).status_code == 401


# ─────────────────────────── MCPClient ───────────────────────────

def _mcp_client(tc, **kw):
    """MCPClient whose HTTP requests are forwarded to the in-process Starlette app."""
    def forward(request):
        r = tc.request(request.method, str(request.url), content=request.content,
                       headers={k: v for k, v in request.headers.items() if k.lower() != "host"})
        return httpx.Response(r.status_code, content=r.content, headers={"content-type": r.headers.get("content-type", "")})
    return MCPClient("http://testserver", transport=httpx.MockTransport(forward), **kw)


def test_client_health_and_tools(client):
    c = _mcp_client(client)
    h = c.check_health()
    assert h["connected"] and h["server_info"]["server"] == "deep-learning-tutor-gpu"
    assert len(c.list_tools()) == 2


def test_client_rejects_bad_url():
    assert not MCPClient("ftp://example").check_health()["connected"]
    assert MCPClient("localhost:8765").submit_training({})["status"] == "error"


def test_client_connection_error_is_friendly():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)
    c = MCPClient("http://localhost:1", transport=httpx.MockTransport(refuse))
    h = c.check_health()
    assert not h["connected"] and "running" in h["error"]


def test_client_reports_validation_errors(client):
    r = _mcp_client(client).submit_training({"epochs": -5})
    assert r["status"] == "error" and "epochs" in r["error"]


@pytest.mark.slow
def test_client_train_remote_end_to_end(client):
    progress = []
    result = _mcp_client(client).train_remote({"dataset": "Wine", "epochs": 2}, poll_interval=0.1,
                                              on_progress=progress.append)
    assert result["status"] == "completed", result
    assert result["model_b64"] and result["result"]["dataset"] == "Wine"
    assert progress and progress[-1]["status"] == "completed"


def test_client_sends_token(client, monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "abc")
    assert _mcp_client(client, token="abc").list_tools()
    assert _mcp_client(client, token="nope").list_tools() == []


def test_failed_job_does_not_leak_exception_details(client, monkeypatch):
    import core.training

    def explode(*a, **k):
        raise RuntimeError(r"C:\secret\server\path.py exploded")
    monkeypatch.setattr(core.training, "run_experiment", explode)
    job = client.post("/call", json={"name": "train_model_remote",
                                     "arguments": {"network_type": "Simple NN", "dataset": "Iris", "epochs": 1}}).json()
    done = _wait_for(client, job["job_id"])
    assert done["status"] == "failed"
    error = done["result"]["error"]
    assert "RuntimeError" in error and "secret" not in error and "server log" in error
    assert not any("secret" in line for line in client.get(f"/logs/{job['job_id']}").json()["logs"])


def test_startup_messages_are_plain_ascii():
    """A redirected stdout on Windows uses a legacy code page; emoji in the banner crashed the server."""
    import mcp_server
    source = open(mcp_server.__file__, encoding="utf-8").read()
    for line in source.splitlines():
        if "print(" in line:
            assert line.isascii(), line
