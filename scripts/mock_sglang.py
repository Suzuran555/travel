#!/usr/bin/env python3
"""Mock the organizers' SGLang serving contract on a Mac.

Exposes the exact surface the Phase-2 harness config expects --
    http://127.0.0.1:30000/v1  (OpenAI-compatible)
    model name "Qwen3.6-27B", api_key "EMPTY", thinking disabled --
and fulfils completions with the LOCAL Ollama qwen3.6 model, using Ollama's
NATIVE /api/chat with think=false (the OpenAI-compat shim of Ollama cannot
disable thinking; native can). This lets the submission package run with its
shipped config.toml untouched, zero env overrides -- byte-for-byte the same
invocation the organizers use.

Endpoints:
  GET  /v1/models             -> lists Qwen3.6-27B
  POST /v1/chat/completions   -> forwarded to Ollama native /api/chat

Usage:
  python scripts/mock_sglang.py [--port 30000] [--ollama http://localhost:11434]
                                [--tag qwen3.6:27b]
Stdlib-only. Logs one line per request.
"""
import argparse
import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None


def ollama_chat(payload):
    body = {
        "model": ARGS.tag,
        "messages": payload.get("messages", []),
        "stream": False,
        "think": False,
        "options": {},
    }
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed")):
        if src in payload and payload[src] is not None:
            body["options"][dst] = payload[src]
    if payload.get("max_tokens"):
        body["options"]["num_predict"] = payload["max_tokens"]
    req = urllib.request.Request(
        f"{ARGS.ollama}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
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

    def log_message(self, fmt, *a):  # quieter default log
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models") or self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "Qwen3.6-27B", "object": "model", "created": 0, "owned_by": "sglang-mock"}]})
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
        try:
            out = ollama_chat(payload)
        except Exception as exc:
            print(f"[mock] upstream error: {exc}", flush=True)
            self._send(502, {"error": {"message": str(exc)}}); return
        content = (out.get("message") or {}).get("content", "")
        resp = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Qwen3.6-27B",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": out.get("prompt_eval_count", 0),
                      "completion_tokens": out.get("eval_count", 0),
                      "total_tokens": out.get("prompt_eval_count", 0) + out.get("eval_count", 0)},
        }
        print(f"[mock] chat ok {time.time()-t0:.1f}s ({len(content)} chars)", flush=True)
        self._send(200, resp)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--tag", default="qwen3.6:27b")
    ARGS = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    print(f"[mock] SGLang mock on http://127.0.0.1:{ARGS.port}/v1 -> {ARGS.ollama} ({ARGS.tag})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
