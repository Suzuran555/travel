"""Backbone LLM client for the Phase-2 harness (self-contained).

Stock main's llms.py has no OpenAI-compatible generic client, so the package
vendors a minimal one implementing the repo's llm(messages, one_line=False,
json_mode=False) protocol with the usual token counters.

Backends (chosen by environment, checked in this order):

  1. CHINATRAVEL_OPENAI_BASE_URL   -> any OpenAI-compatible /chat/completions
       server (SGLang, vLLM, llama-server...). Model name from
       CHINATRAVEL_OPENAI_MODEL; API key from CHINATRAVEL_OPENAI_API_KEY
       (defaults to "EMPTY", which SGLang accepts).
  2. OLLAMA_TAG / OLLAMA_HOST_URL  -> local Ollama /api/chat (our local
       stand-in for the organizers' SGLang serving).

Sampling is pinned for determinism: temperature 0.0 and a fixed seed by
default (override via PENGUINS_TEMPERATURE / PENGUINS_SEED). top_p 0.95 /
top_k 20 are kept for the non-greedy case; thinking blocks are stripped. The display name keys the
translation cache dir (cache/translation_{name}_reflect), so keep it stable
per model: CHINATRAVEL_LLM_NAME overrides, default "Qwen3.6-27B".
"""
import os
import re
import time

import requests

from . import env_fix

# earliest package code executed by run_tpc (init_llm imports this module):
# patch the EN POI-name drift before anything else builds POI tables.
env_fix.apply_class_patches()

try:
    from json_repair import repair_json
except ModuleNotFoundError:  # json_repair is in stock requirements.txt
    def repair_json(s, ensure_ascii=False):
        return s

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _merge_repeated_role(messages):
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))
    return merged


class TPCLLM:
    class ModeError(Exception):
        pass

    def __init__(self):
        self.name = os.environ.get("CHINATRAVEL_LLM_NAME", "Qwen3.6-27B")
        self.input_token_count = 0
        self.output_token_count = 0
        self.input_token_maxx = 0

        self.openai_base = (os.environ.get("CHINATRAVEL_OPENAI_BASE_URL") or "").rstrip("/")
        self.openai_model = os.environ.get("CHINATRAVEL_OPENAI_MODEL", "")
        self.openai_key = os.environ.get("CHINATRAVEL_OPENAI_API_KEY") or "EMPTY"
        self.ollama_host = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
        self.ollama_tag = os.environ.get("OLLAMA_TAG", "")
        self.timeout = float(os.environ.get("CHINATRAVEL_LLM_TIMEOUT", "3600"))
        # Optional per-query wall-clock deadline (epoch seconds).  The agent
        # sets this at the start of every run() so that a single blocking
        # request can never outlive the query's emission deadline (the
        # harness watchdog cannot interrupt a blocking socket read).
        self.request_deadline = None
        # 8192: strictly-constrained json_object servers (SGLang) format the
        # constraint list verbosely; 3072 truncated it mid-array and silently
        # dropped constraints after json repair
        self.max_tokens = int(os.environ.get("CHINATRAVEL_LLM_MAX_TOKENS", "8192"))
        # determinism: greedy decoding + fixed seed unless explicitly overridden
        self.temperature = float(os.environ.get("PENGUINS_TEMPERATURE", "0.0"))
        self.seed = int(os.environ.get("PENGUINS_SEED", "20260322"))

    # ---- repo llm protocol --------------------------------------------------
    def __call__(self, messages, one_line=True, json_mode=False):
        if one_line and json_mode:
            raise self.ModeError("one_line and json_mode cannot both be True")
        return self._get_response(messages, one_line, json_mode)

    def _get_response(self, messages, one_line, json_mode):
        messages = _merge_repeated_role(list(messages))
        if self.openai_base:
            content, n_in, n_out = self._openai_chat(messages, json_mode)
        elif self.ollama_tag:
            content, n_in, n_out = self._ollama_chat(messages, json_mode)
        else:
            raise RuntimeError(
                "TPCLLM: no backend configured. Set CHINATRAVEL_OPENAI_BASE_URL "
                "(+ CHINATRAVEL_OPENAI_MODEL) for an OpenAI-compatible server, "
                "or OLLAMA_TAG (+ OLLAMA_HOST_URL) for local Ollama."
            )
        self.input_token_count += n_in
        self.output_token_count += n_out
        self.input_token_maxx = max(self.input_token_maxx, n_in)

        content = _THINK_RE.sub("", content or "").strip()
        if json_mode:
            content = repair_json(content, ensure_ascii=False)
        if one_line:
            for line in content.splitlines():
                if line.strip():
                    return line.strip()
        return content

    def _effective_timeout(self):
        """Per-request timeout, capped by the agent-set per-query deadline."""
        timeout = self.timeout
        deadline = getattr(self, "request_deadline", None)
        if deadline:
            timeout = min(timeout, max(2.0, deadline - time.time()))
        return timeout

    # ---- backends ------------------------------------------------------------
    def _openai_chat(self, messages, json_mode):
        payload = {
            "model": self.openai_model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking":
                                     os.environ.get("CHINATRAVEL_LLM_THINK", "0") == "1"},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        url = f"{self.openai_base}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=self._effective_timeout())
        if resp.status_code == 400:
            # server may reject non-standard extras (top_k / template kwargs /
            # response_format) -- retry with the bare standard payload
            for k in ("top_k", "chat_template_kwargs", "response_format", "seed"):
                payload.pop(k, None)
            resp = requests.post(url, json=payload, headers=headers, timeout=self._effective_timeout())
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"].get("content", "")
        usage = body.get("usage") or {}
        return content, usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0

    def _ollama_chat(self, messages, json_mode):
        payload = {
            "model": self.ollama_tag,
            "messages": messages,
            "stream": False,
            "think": os.environ.get("OLLAMA_THINK", "0") == "1",
            "options": {
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 20,
                "seed": self.seed,
                "num_predict": self.max_tokens,
                "num_ctx": 32768,
            },
        }
        if json_mode:
            payload["format"] = "json"
        resp = requests.post(f"{self.ollama_host}/api/chat", json=payload, timeout=self._effective_timeout())
        resp.raise_for_status()
        body = resp.json()
        content = body.get("message", {}).get("content", "")
        return (content,
                body.get("prompt_eval_count", 0) or 0,
                body.get("eval_count", 0) or 0)
