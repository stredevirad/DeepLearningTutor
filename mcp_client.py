"""
MCP Client — Connects the Streamlit app to a remote MCP/GPU server
===================================================================
Provides a simple synchronous interface for the Streamlit app to:
  1. Check server health
  2. Trigger remote training jobs (which run in the background on the server)
  3. Poll job status/progress and retrieve results and the trained model

Uses the REST API that mcp_server.py always serves (with or without the MCP SDK).
If MCP_AUTH_TOKEN is set, it is sent as a Bearer token.
"""

import os
import time
from urllib.parse import urlparse

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class MCPClient:
    """Client for connecting to the Deep Learning Tutor MCP Server."""

    def __init__(self, server_url: str, timeout: float = 1800.0, token: str = None, transport=None):
        """
        Args:
            server_url: Base URL of the MCP server (e.g. http://localhost:8765)
            timeout: Maximum seconds to wait for a remote training job to finish
            token: Optional auth token (defaults to the MCP_AUTH_TOKEN env var)
            transport: Optional httpx transport (used by tests)
        """
        self.server_url = (server_url or "").strip().rstrip("/")
        self.timeout = timeout
        self.token = token if token is not None else os.getenv("MCP_AUTH_TOKEN", "")
        self._transport = transport
        self._connected = False
        self._server_info = None

    def is_available(self) -> bool:
        """Check if httpx is installed."""
        return HTTPX_AVAILABLE

    def _url_error(self):
        parsed = urlparse(self.server_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return "Server URL must start with http:// or https:// (e.g. http://localhost:8765)"
        return None

    def _client(self, timeout):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        kwargs = {"timeout": timeout, "headers": headers}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    @staticmethod
    def _error_from_response(resp):
        try:
            return resp.json().get("error") or f"HTTP {resp.status_code}"
        except ValueError:
            return f"HTTP {resp.status_code}"

    def check_health(self) -> dict:
        """
        Ping the remote server's /health endpoint.

        Returns:
            dict with keys: connected (bool), server_info (dict or None), error (str or None)
        """
        if not HTTPX_AVAILABLE:
            return {"connected": False, "server_info": None, "error": "httpx not installed. Run: pip install httpx"}
        url_error = self._url_error()
        if url_error:
            return {"connected": False, "server_info": None, "error": url_error}

        try:
            with self._client(5.0) as client:
                resp = client.get(f"{self.server_url}/health")
                resp.raise_for_status()
                info = resp.json()
                self._connected = True
                self._server_info = info
                return {"connected": True, "server_info": info, "error": None}
        except httpx.ConnectError:
            return {"connected": False, "server_info": None, "error": "Cannot connect — is the MCP server running?"}
        except httpx.TimeoutException:
            return {"connected": False, "server_info": None, "error": "Connection timed out"}
        except (httpx.HTTPError, ValueError) as e:
            return {"connected": False, "server_info": None, "error": f"Unexpected response from server ({e})"}

    def list_tools(self) -> list:
        """List available tools on the remote server."""
        if not HTTPX_AVAILABLE or self._url_error():
            return []
        try:
            with self._client(10.0) as client:
                resp = client.get(f"{self.server_url}/tools")
                resp.raise_for_status()
                return resp.json().get("tools", [])
        except (httpx.HTTPError, ValueError):
            return []

    def _call(self, name, arguments, timeout=30.0):
        with self._client(timeout) as client:
            resp = client.post(f"{self.server_url}/call", json={"name": name, "arguments": arguments})
        if resp.status_code >= 400:
            return {"status": "error", "error": self._error_from_response(resp)}
        return resp.json()

    def submit_training(self, config: dict) -> dict:
        """Queue a training job. Returns the job status dict (with job_id) or an error dict."""
        if not HTTPX_AVAILABLE:
            return {"status": "error", "error": "httpx not installed"}
        url_error = self._url_error()
        if url_error:
            return {"status": "error", "error": url_error}
        try:
            return self._call("train_model_remote", config)
        except httpx.TimeoutException:
            return {"status": "error", "error": "The MCP server did not respond in time"}
        except httpx.ConnectError:
            return {"status": "error", "error": "Lost connection to MCP server"}
        except (httpx.HTTPError, ValueError) as e:
            return {"status": "error", "error": f"Unexpected response from MCP server ({e})"}

    def get_job_status(self, job_id: str, include_model: bool = False) -> dict:
        """Check the status of a training job."""
        if not HTTPX_AVAILABLE:
            return {"status": "error", "error": "httpx not installed"}
        try:
            return self._call("get_training_status", {"job_id": job_id, "include_model": include_model},
                              timeout=120.0 if include_model else 10.0)
        except (httpx.HTTPError, ValueError) as e:
            return {"status": "error", "error": f"Could not read job status ({e})"}

    def train_remote(self, config: dict, poll_interval: float = 1.0, on_progress=None,
                     include_model: bool = True, sleep=time.sleep) -> dict:
        """
        Start a training job on the remote server and wait for it to finish.

        Args:
            config: Dictionary with training configuration (network_type, dataset, epochs, etc.)
            poll_interval: Seconds between status polls
            on_progress: Optional callback(status_dict) called after every poll (for live charts)
            include_model: Download the trained model (base64 .keras) when the job completes

        Returns:
            Final job status dict ("completed" with result/model_b64) or {"status": "error", ...}
        """
        submitted = self.submit_training(config)
        if submitted.get("status") == "error" or "job_id" not in submitted:
            return submitted if submitted.get("status") == "error" else {"status": "error", "error": "Server did not return a job id"}
        job_id = submitted["job_id"]

        deadline = time.monotonic() + self.timeout
        status = submitted
        while time.monotonic() < deadline:
            if on_progress:
                on_progress(status)
            if status.get("status") in ("completed", "failed"):
                break
            sleep(poll_interval)
            status = self.get_job_status(job_id)
            if status.get("status") == "error":
                return status
        else:
            return {"status": "error", "error": f"Training timed out after {self.timeout:.0f}s (job {job_id})"}

        if status.get("status") == "failed":
            err = (status.get("result") or {}).get("error", "unknown error")
            return {"status": "error", "error": f"Remote training failed: {err}", "job_id": job_id}
        if include_model:
            status = self.get_job_status(job_id, include_model=True)
        return status

    def get_logs(self, job_id: str) -> list:
        """Retrieve training logs for a job."""
        if not HTTPX_AVAILABLE or self._url_error():
            return []
        try:
            with self._client(10.0) as client:
                resp = client.get(f"{self.server_url}/logs/{job_id}")
                resp.raise_for_status()
                return resp.json().get("logs", [])
        except (httpx.HTTPError, ValueError):
            return []
