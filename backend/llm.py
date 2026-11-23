"""The single LLM interface used by the backend, with three backends.

Selected by the LLM_BACKEND env var (see backend/config.py):
  - "hf":     local HF model. Loads EITHER the merged fine-tuned directory
              (finetune/outputs/merged) OR the base model + LoRA adapter
              (finetune/outputs/adapter) once at startup, then streams with
              transformers.TextIteratorStreamer driven by a background
              thread that runs model.generate.
  - "ollama": an HTTP streaming call to a local Ollama server. Zero Python
              GPU memory in our process, trivially swappable model.
  - "fake":   deterministic word-by-word echo generator, for tests and for
              frontend development without any model installed.

All backends expose the same generator:
    stream(messages: list[dict], **gen_kwargs) -> Iterator[str]
yielding text chunks as they are produced.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

from backend.config import get_settings, resolve_path

_settings = get_settings()


# ---------------------------------------------------------------------------
# Fake backend (tests + frontend dev)
# ---------------------------------------------------------------------------
def _stream_fake(messages: list[dict], **gen_kwargs: Any) -> Iterator[str]:
    """Deterministic 'model': echoes the last user message word by word.

    Latency is intentionally non-zero so SSE and the frontend's streaming
    state can be observed realistically in tests.
    """
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    words = (last_user or "empty").split()
    reply = "This is the FAKE backend echoing: " + " ".join(words)
    for w in reply.split(" "):
        time.sleep(0.01)
        yield w + " "


# ---------------------------------------------------------------------------
# HuggingFace backend (merged dir OR base+adapter)
# ---------------------------------------------------------------------------
class HFBackend:
    """Loads the model once and streams generations token by token.

    The heavy import + load happen on first use (not at module import), so
    `from backend.llm import stream` stays cheap for tests and /health.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_from: Optional[str] = None

    def _load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = get_settings()
        merged_dir = resolve_path(s.hf_model_dir)
        adapter_dir = resolve_path(s.hf_adapter_dir)
        if merged_dir.exists():
            # Preferred: standalone merged fine-tuned model.
            self._model = AutoModelForCausalLM.from_pretrained(
                merged_dir, torch_dtype=torch.float16, device_map="auto"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(merged_dir)
            self._loaded_from = f"merged:{merged_dir}"
        elif adapter_dir.exists():
            # Fallback: base model + LoRA adapter (no merge step needed).
            base = AutoModelForCausalLM.from_pretrained(
                s.hf_base_model, torch_dtype=torch.float16, device_map="auto"
            )
            self._model = PeftModel.from_pretrained(base, adapter_dir)
            self._tokenizer = AutoTokenizer.from_pretrained(s.hf_base_model)
            self._loaded_from = f"adapter:{adapter_dir}"
        else:
            raise FileNotFoundError(
                f"no model found at {merged_dir} or {adapter_dir} - run "
                "finetune/train_qlora.py (and optionally merge_and_test.py), "
                "or set LLM_BACKEND=ollama|fake"
            )

    @property
    def loaded_from(self) -> str:
        if self._model is None:
            self._load()
        return self._loaded_from or "unknown"

    def stream(self, messages: list[dict], **gen_kwargs: Any) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        if self._model is None:
            self._load()
        s = get_settings()
        text = self._tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self._tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(self._model.device)

        # TextIteratorStreamer is a queue that yields generated text chunks;
        # model.generate() blocks, so it runs in a background thread and the
        # generator below yields from the queue in real time.
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs_full = {
            "max_new_tokens": gen_kwargs.get("max_new_tokens", s.max_new_tokens),
            "temperature": gen_kwargs.get("temperature", s.temperature),
            "top_p": gen_kwargs.get("top_p", s.top_p),
            "do_sample": gen_kwargs.get("temperature", s.temperature) > 0,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, **gen_kwargs_full},
            daemon=True,
        )
        thread.start()
        try:
            for chunk in streamer:
                if chunk:  # skip empty strings the streamer can emit
                    yield chunk
        finally:
            thread.join(timeout=5.0)


_HF = HFBackend()


# ---------------------------------------------------------------------------
# Ollama backend (HTTP streaming, JSON lines)
# ---------------------------------------------------------------------------
def _stream_ollama(messages: list[dict], **gen_kwargs: Any) -> Iterator[str]:
    """Stream from a local Ollama server's /api/chat endpoint.

    Ollama answers with newline-delimited JSON objects; each has a "message"
    fragment and a "done" flag, so we parse line by line and yield the text.
    """
    s = get_settings()
    payload = {
        "model": gen_kwargs.get("model", s.ollama_model),
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": gen_kwargs.get("max_new_tokens", s.max_new_tokens),
            "temperature": gen_kwargs.get("temperature", s.temperature),
            "top_p": gen_kwargs.get("top_p", s.top_p),
        },
    }
    req = urllib.request.Request(
        f"{s.ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                break


# ---------------------------------------------------------------------------
# OpenAI-compatible API backend (OpenAI, Groq, OpenRouter, DeepSeek, etc.)
# ---------------------------------------------------------------------------
def _stream_openai(messages: list[dict], **gen_kwargs: Any) -> Iterator[str]:
    """Stream from an OpenAI-compatible /chat/completions endpoint."""
    s = get_settings()
    api_key = s.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for the 'openai' backend. "
            "Set OPENAI_API_KEY in your .env or environment variables."
        )

    model = gen_kwargs.get("model", s.openai_model)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": gen_kwargs.get("max_new_tokens", s.max_new_tokens),
        "temperature": gen_kwargs.get("temperature", s.temperature),
    }

    url = f"{s.openai_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mini-chatgpt/1.0",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("error", {}).get("message", err_body)
        except Exception:
            msg = err_body
        raise RuntimeError(f"API error ({exc.code}): {msg}") from exc


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def stream(messages: list[dict], **gen_kwargs: Any) -> Iterator[str]:
    """Dispatch to the configured backend. Raises ValueError for unknown."""
    backend = _settings.llm_backend.lower()
    if backend == "hf":
        return _HF.stream(messages, **gen_kwargs)
    if backend == "ollama":
        return _stream_ollama(messages, **gen_kwargs)
    if backend in {"openai", "api"}:
        return _stream_openai(messages, **gen_kwargs)
    if backend == "fake":
        return _stream_fake(messages, **gen_kwargs)
    raise ValueError(f"unknown LLM_BACKEND {backend!r} (hf|ollama|openai|fake)")


def describe_backend() -> str:
    """Human-readable backend description for /health and logs."""
    backend = _settings.llm_backend.lower()
    if backend == "hf":
        return f"hf ({_HF.loaded_from})"
    if backend == "ollama":
        return f"ollama ({_settings.ollama_model} @ {_settings.ollama_base_url})"
    if backend in {"openai", "api"}:
        return f"openai ({_settings.openai_model} @ {_settings.openai_base_url})"
    return "fake (echo generator)"
