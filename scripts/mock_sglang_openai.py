#!/usr/bin/env python3
"""Mock the organizers' SGLang serving contract, backed by ANY OpenAI-compatible
upstream (e.g. DashScope compatible-mode) instead of local Ollama.

Presents the exact surface the Phase-2 harness config expects --
    http://127.0.0.1:30000/v1  (OpenAI-compatible)
    served model name "Qwen3.6-27B", api_key "EMPTY", thinking disabled --
and fulfils completions with a remote OpenAI-compatible endpoint. The
submission package runs with its shipped config.toml untouched, zero env
overrides -- byte-for-byte the organizer invocation -- while translation runs
at API speed instead of local-GGUF speed (which cannot fit the 170 s
per-query cap on a laptop).

NO SECRETS IN THIS FILE. Upstream configured via env:
  UPSTREAM_BASE_URL  e.g. https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  UPSTREAM_MODEL     e.g. qwen3.6-27b
  UPSTREAM_API_KEY   bearer token (never logged)

Usage:
  UPSTREAM_BASE_URL=... UPSTREAM_MODEL=... UPSTREAM_API_KEY=... \
  python scripts/mock_sglang_openai.py [--port 30000]
Stdlib-only. Logs one line per request (no payloads, no keys).
"""
import argparse
import json
import os
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
UP_BASE = os.environ.get("UPSTREAM_BASE_URL", "").rstrip("/")
UP_MODEL = os.environ.get("UPSTREAM_MODEL", "")
UP_KEY = os.environ.get("UPSTREAM_API_KEY", "")

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def upstream_chat(payload):
    body = {
        "model": UP_MODEL,
        "messages": payload.get("messages", []),
        "stream": False,
        # DashScope compatible-mode: disable reasoning like the organizers'
        # SGLang serving does at the server level.
        "enable_thinking": False,
    }
    for k in ("temperature", "top_p", "seed", "max_tokens"):
        if payload.get(k) is not None:
            body[k] = payload[k]
    req = urllib.request.Request(
        f"{UP_BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {UP_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


class H(BaseHTTPRequestHandler):
    server_version = "SGLang-mock/0.5.10"

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "Qwen3.6-27B", "object": "model", "created": 0,
                 "owned_by": "sglang-mock"}]})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, {"error": {"message": "bad json"}}); return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}}); return
        model = payload.get("model", "")
        if model != "Qwen3.6-27B":
            # organizers serve exactly this name; be strict like a real server
            self._send(404, {"error": {"message": f"model '{model}' not found"}}); return
        t0 = time.time()
        last_exc = None
        for attempt in range(3):
            try:
                out = upstream_chat(payload)
                break
            except Exception as exc:
                last_exc = exc
                print(f"[mock] upstream error (try {attempt+1}/3): {type(exc).__name__}", flush=True)
                time.sleep(2 * (attempt + 1))
        else:
            self._send(502, {"error": {"message": str(last_exc)}}); return
        try:
            content = out["choices"][0]["message"].get("content") or ""
        except Exception:
            content = ""
        content = _THINK_RE.sub("", content)
        usage = out.get("usage", {}) or {}
        resp = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Qwen3.6-27B",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                      "completion_tokens": usage.get("completion_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0)},
        }
        print(f"[mock] chat ok {time.time()-t0:.1f}s ({len(content)} chars)", flush=True)
        self._send(200, resp)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ARGS = ap.parse_args()
    if not (UP_BASE and UP_MODEL and UP_KEY):
        raise SystemExit("set UPSTREAM_BASE_URL / UPSTREAM_MODEL / UPSTREAM_API_KEY")
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    print(f"[mock] SGLang mock on http://127.0.0.1:{ARGS.port}/v1 -> {UP_BASE} ({UP_MODEL})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
