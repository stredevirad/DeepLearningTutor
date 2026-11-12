"""
MCP Server — Model Context Protocol Server for Remote GPU Training
===================================================================
Exposes deep learning training tools and resources over the Model Context
Protocol (JSON-RPC / SSE) so that local Streamlit clients or any MCP-compatible
host can trigger training on a remote GPU machine. A plain REST API
(/health, /tools, /call, /logs/{job_id}) is always served as well; the app's
MCPClient uses it.

Training jobs run in a background worker thread: `train_model_remote` returns a
job_id immediately and `get_training_status` reports progress (per-epoch
history), the final result and, on request, the trained model (.keras, base64).

Security:
  * Binds to 127.0.0.1 (this machine only) by default.
  * Only the two known tools are accepted, and every training argument is
    validated against fixed ranges (no arbitrary code, no unbounded jobs).
  * If MCP_AUTH_TOKEN is set, every request except /health must send
    "Authorization: Bearer <token>". Set it before using --host 0.0.0.0.

Usage:
    python mcp_server.py                       # 127.0.0.1:8765
    python mcp_server.py --port 9000           # custom port
    python mcp_server.py --host 0.0.0.0        # expose on the network (set MCP_AUTH_TOKEN first!)
"""

import argparse
import base64
import hmac
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ── Graceful import handling ──
try:
    from mcp.server import Server
    from mcp.server.sse import SseServerTransport
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
    )
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False

try:
    from starlette.applications import Starlette
    from starlette.concurrency import run_in_threadpool
    from starlette.routing import Route, Mount
    from starlette.responses import JSONResponse, Response
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

logger = logging.getLogger("mcp_server")

TOOL_NAMES = ("train_model_remote", "get_training_status")
JOB_ID_RE = re.compile(r"^[0-9a-f]{8}$")
MAX_JOBS_KEPT = 50          # finished jobs beyond this are forgotten (oldest first)
MAX_PENDING_JOBS = 5        # queued + running jobs accepted at once
MAX_BODY_BYTES = 64 * 1024
LOG_LINES_KEPT = 500

TOOL_SCHEMAS = {
    "train_model_remote": {
        "description": ("Start training a neural network on this server. Returns a job_id immediately; "
                        "poll get_training_status for progress and results. Supports Simple NN "
                        "(Iris/Wine/Breast Cancer/Digits) and CNN (MNIST/Fashion-MNIST)."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "network_type": {"type": "string", "enum": ["Simple NN", "CNN"], "default": "Simple NN"},
                "dataset": {"type": "string", "default": "Iris"},
                "epochs": {"type": "integer", "default": 30, "minimum": 1, "maximum": 200},
                "learning_rate": {"type": "number", "default": 0.001},
                "batch_size": {"type": "integer", "default": 32},
                "optimizer": {"type": "string", "enum": ["Adam", "SGD", "RMSprop"], "default": "Adam"},
                "dropout_rate": {"type": "number", "default": 0.0},
                "hidden_layers": {"type": "integer", "default": 2},
                "nodes_per_layer": {"type": "integer", "default": 16},
                "conv_layers": {"type": "integer", "default": 2},
                "base_filters": {"type": "integer", "default": 16},
                "early_stopping": {"type": "boolean", "default": False},
                "seed": {"type": "integer", "default": 42},
            },
            "required": ["network_type", "dataset"],
        },
    },
    "get_training_status": {
        "description": "Check the status, progress, logs and result of a training job by its job_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The job ID returned by train_model_remote"},
                "include_model": {"type": "boolean", "default": False,
                                  "description": "Also return the trained .keras model (base64)."},
            },
            "required": ["job_id"],
        },
    },
}

# ── Training job storage ──
_training_jobs = OrderedDict()   # job_id -> {status, config, result, logs, history, created_at, model_b64}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-train")


class ToolCallError(ValueError):
    """A rejected tool call (unknown tool, bad arguments). Message is safe to return."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _log(job_id, message):
    with _jobs_lock:
        job = _training_jobs.get(job_id)
        if job is not None:
            job["logs"].append(f"[{_now()}] {message}")
            del job["logs"][:-LOG_LINES_KEPT]


def _forget_old_jobs():
    """Keep at most MAX_JOBS_KEPT jobs; never drop queued or running ones."""
    finished = [jid for jid, j in _training_jobs.items() if j["status"] in ("completed", "failed")]
    while len(_training_jobs) > MAX_JOBS_KEPT and finished:
        _training_jobs.pop(finished.pop(0), None)


def _run_training_job(job_id: str, config: dict):
    """Execute a TensorFlow training run (in the worker thread) and store results."""
    import tensorflow as tf
    from core.training import run_experiment

    with _jobs_lock:
        _training_jobs[job_id]["status"] = "running"
    _log(job_id, "Training started")
    total_epochs = int(config.get("epochs", 30))

    class JobProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            with _jobs_lock:
                hist = _training_jobs[job_id]["history"]
                for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
                    if key in logs:
                        hist.setdefault(key, []).append(round(float(logs[key]), 4))
            _log(job_id, f"Epoch {epoch + 1}/{total_epochs} — loss: {logs.get('loss', float('nan')):.4f}, "
                         f"val_acc: {logs.get('val_accuracy', float('nan')):.4f}")

    tmp_dir = None
    try:
        result, model, _ = run_experiment(config, callbacks=[JobProgress()], return_model=True)
        tmp_dir = tempfile.mkdtemp(prefix="mcp_model_")
        path = os.path.join(tmp_dir, "model.keras")
        model.save(path)
        with open(path, "rb") as f:
            model_b64 = base64.b64encode(f.read()).decode("ascii")
        with _jobs_lock:
            job = _training_jobs[job_id]
            job["status"] = "completed"
            job["result"] = result
            job["model_b64"] = model_b64
        _log(job_id, f"Training completed — Val Acc: {result['results']['final_val_accuracy']}")
    except Exception as e:  # report any failure to the client instead of crashing the worker
        from core.training import ConfigError, DataError
        if isinstance(e, (ConfigError, DataError)):
            message = str(e)                         # written for users already
        elif isinstance(e, tf.errors.ResourceExhaustedError):
            message = "The server ran out of memory. Try a smaller batch size or fewer filters/nodes."
        else:                                        # details stay in the server log, not in the reply
            logger.exception("Training job %s failed", job_id)
            message = f"Training failed on the server ({type(e).__name__}). See the server log for details."
        with _jobs_lock:
            job = _training_jobs[job_id]
            job["status"] = "failed"
            job["result"] = {"status": "error", "error": message}
        _log(job_id, f"ERROR: {message}")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def submit_training_job(arguments: dict) -> str:
    """Validate a training request and queue it. Returns the new job_id."""
    from core.training import ConfigError, validate_config
    try:
        config = validate_config(arguments)
    except ConfigError as e:
        raise ToolCallError(str(e)) from e
    with _jobs_lock:
        pending = sum(1 for j in _training_jobs.values() if j["status"] in ("queued", "running"))
        if pending >= MAX_PENDING_JOBS:
            raise ToolCallError("Too many training jobs are already queued. Try again later.", status_code=429)
        job_id = uuid.uuid4().hex[:8]
        _training_jobs[job_id] = {
            "status": "queued",
            "config": config,
            "result": None,
            "model_b64": None,
            "history": {},
            "logs": [f"[{_now()}] Job {job_id} created"],
            "created_at": _now(),
        }
        _forget_old_jobs()
    _executor.submit(_run_training_job, job_id, config)
    return job_id


def job_status(job_id, include_model=False, log_lines=10):
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise ToolCallError("job_id must be the 8-character id returned by train_model_remote.")
    with _jobs_lock:
        job = _training_jobs.get(job_id)
        if job is None:
            raise ToolCallError(f"Unknown job: {job_id}", status_code=404)
        payload = {
            "job_id": job_id,
            "status": job["status"],
            "config": job["config"],
            "history": {k: list(v) for k, v in job["history"].items()},
            "result": job.get("result"),
            "logs": job["logs"][-log_lines:],
        }
        if include_model and job.get("model_b64"):
            payload["model_b64"] = job["model_b64"]
    return payload


def handle_tool_call(name, arguments):
    """Dispatch a validated tool call. Shared by the REST API and the MCP SDK server."""
    if name not in TOOL_NAMES:
        raise ToolCallError(f"Unknown tool: {name!r}. Available tools: {', '.join(TOOL_NAMES)}.")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolCallError("'arguments' must be a JSON object.")

    if name == "train_model_remote":
        job_id = submit_training_job(arguments)
        return job_status(job_id, log_lines=5)

    unknown = set(arguments) - {"job_id", "include_model"}
    if unknown:
        raise ToolCallError(f"Unknown argument(s) for get_training_status: {', '.join(sorted(unknown))}.")
    return job_status(arguments.get("job_id"), include_model=bool(arguments.get("include_model", False)))


def create_mcp_server():
    """Create and configure the MCP server with tools and resources."""
    if not MCP_SDK_AVAILABLE:
        raise ImportError("mcp SDK is not installed. Run: pip install mcp")

    server = Server("deep-learning-tutor-gpu")

    # ── LIST TOOLS ──
    @server.list_tools()
    async def handle_list_tools():
        return [Tool(name=name, description=spec["description"], inputSchema=spec["inputSchema"])
                for name, spec in TOOL_SCHEMAS.items()]

    # ── CALL TOOL ──
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict):
        try:
            payload = handle_tool_call(name, arguments)
        except ToolCallError as e:
            payload = {"error": str(e)}
        return [TextContent(type="text", text=json.dumps(payload, indent=2))]

    # ── LIST RESOURCES ──
    @server.list_resources()
    async def handle_list_resources():
        with _jobs_lock:
            jobs = list(_training_jobs.items())
        return [Resource(
            uri=f"training://{job_id}/logs",
            name=f"Training Logs — Job {job_id}",
            description=f"Training logs for job {job_id} ({job['status']})",
            mimeType="text/plain",
        ) for job_id, job in jobs]

    # ── READ RESOURCE ──
    @server.read_resource()
    async def handle_read_resource(uri: str):
        # Parse URI: training://{job_id}/logs
        job_id = str(uri).replace("training://", "").split("/")[0]
        with _jobs_lock:
            if JOB_ID_RE.match(job_id) and job_id in _training_jobs:
                return "\n".join(_training_jobs[job_id]["logs"])
        return f"Resource not found: {uri}"

    return server


# ── REST API (always served; used by mcp_client.MCPClient) ──

def _auth_error(request):
    """Return a 401 response if MCP_AUTH_TOKEN is set and the request lacks it."""
    token = os.getenv("MCP_AUTH_TOKEN", "")
    if not token:
        return None
    sent = request.headers.get("authorization", "")
    if hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
        return None
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


def _rest_routes():
    async def health(request):
        with _jobs_lock:
            active = sum(1 for j in _training_jobs.values() if j["status"] in ("queued", "running"))
            total = len(_training_jobs)
        return JSONResponse({
            "status": "ok",
            "server": "deep-learning-tutor-gpu",
            "protocol": "mcp+http" if MCP_SDK_AVAILABLE else "http-fallback",
            "active_jobs": active,
            "total_jobs": total,
            "auth_required": bool(os.getenv("MCP_AUTH_TOKEN")),
            "timestamp": _now(),
        })

    async def list_tools(request):
        denied = _auth_error(request)
        if denied:
            return denied
        return JSONResponse({"tools": [{"name": n, **spec} for n, spec in TOOL_SCHEMAS.items()]})

    async def call_tool(request):
        denied = _auth_error(request)
        if denied:
            return denied
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse({"error": "Request body too large"}, status_code=413)
        try:
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "Request body must be JSON"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)
        try:
            # Validation and job submission are quick; training itself runs in the worker thread.
            result = await run_in_threadpool(handle_tool_call, payload.get("name"), payload.get("arguments", {}))
        except ToolCallError as e:
            return JSONResponse({"error": str(e)}, status_code=e.status_code)
        return JSONResponse(result)

    async def read_logs(request):
        denied = _auth_error(request)
        if denied:
            return denied
        job_id = request.path_params["job_id"]
        if not JOB_ID_RE.match(job_id):
            return JSONResponse({"error": "Invalid job id"}, status_code=400)
        with _jobs_lock:
            if job_id not in _training_jobs:
                return JSONResponse({"error": "Job not found"}, status_code=404)
            logs = list(_training_jobs[job_id]["logs"])
        return JSONResponse({"logs": logs})

    return [
        Route("/health", health),
        Route("/tools", list_tools),
        Route("/call", call_tool, methods=["POST"]),
        Route("/logs/{job_id}", read_logs),
    ]


def create_http_app(server):
    """Wrap the MCP server in a Starlette ASGI app with an SSE endpoint plus the REST API."""
    if not WEB_AVAILABLE:
        raise ImportError("starlette/uvicorn not installed. Run: pip install starlette uvicorn")

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        denied = _auth_error(request)
        if denied:
            return denied
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )
        return Response()

    return Starlette(routes=_rest_routes() + [
        Route("/sse", handle_sse),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ])


# ── Standalone HTTP fallback (no MCP SDK required) ──
def create_standalone_http_server():
    """
    A lightweight HTTP server that mimics MCP tool semantics using plain REST.
    This allows the Streamlit client to connect even without the full MCP SDK.
    """
    if not WEB_AVAILABLE:
        raise ImportError("starlette/uvicorn not installed. Run: pip install starlette uvicorn")
    return Starlette(routes=_rest_routes())


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Deep Learning Tutor — MCP GPU Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1 = this machine only)")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.getenv("MCP_AUTH_TOKEN"):
        print("WARNING: binding to a network interface without MCP_AUTH_TOKEN set.")
        print("   Anyone who can reach this port can start training jobs on this machine.")

    if MCP_SDK_AVAILABLE and WEB_AVAILABLE:
        print(f"Starting MCP Server (SSE + REST) on {args.host}:{args.port}")
        app = create_http_app(create_mcp_server())
    elif WEB_AVAILABLE:
        print(f"Starting HTTP fallback server on {args.host}:{args.port}")
        print("   (MCP SDK not found - using REST endpoints instead)")
        app = create_standalone_http_server()
    else:
        print("Cannot start server: starlette and uvicorn are required.")
        print("   Run: pip install starlette uvicorn")
        sys.exit(1)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
