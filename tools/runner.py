"""Minimal client for the hackathon's remote PTX runner (see its docs/API.md).

Responses are cached on disk keyed by the exact request, so re-running the
test suite only sends kernels whose PTX (or launch) changed. "busy" answers
are retried with a growing back-off.

Sandboxes that only allow HTTPS CONNECT tunnels can set ``RUNNER_PROXY``
(``host:port``); requests are then tunnelled through that proxy.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_URL = "http://runner.riftstack.ai:48701"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "runner"


class RunnerError(Exception):
    pass


class Runner:
    def __init__(self, url: str = DEFAULT_URL, use_cache: bool = True):
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.use_cache = use_cache
        self.proxy = os.environ.get("RUNNER_PROXY")

    def _connection(self) -> http.client.HTTPConnection:
        if self.proxy:
            host, port = self.proxy.rsplit(":", 1)
            conn = http.client.HTTPConnection(host, int(port), timeout=300)
            conn.set_tunnel(self.host, self.port)
            return conn
        return http.client.HTTPConnection(self.host, self.port, timeout=300)

    def post(self, path: str, body: dict) -> dict:
        key = hashlib.sha256((path + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        cache = CACHE_DIR / f"{key}.json"
        if self.use_cache and cache.exists():
            return json.loads(cache.read_text())
        delay = 2.0
        for _ in range(30):
            conn = self._connection()
            try:
                conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = resp.read()
            except OSError as exc:
                raise RunnerError(f"cannot reach the runner: {exc}") from exc
            finally:
                conn.close()
            if resp.status != 200:
                raise RunnerError(f"HTTP {resp.status}: {data[:500]!r}")
            out = json.loads(data)
            if out.get("stage") != "busy":
                break
            time.sleep(delay)
            delay = min(delay * 1.5, 20.0)
        else:
            raise RunnerError("runner stayed busy")
        if self.use_cache and out.get("stage") != "busy":
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out))
        return out

    def compile(self, ptx: str) -> dict:
        return self.post("/compile", {"ptx": ptx})

    def run(self, request: dict) -> dict:
        return self.post("/run", request)
