"""Backend configuration via environment variables (.env supported).

All settings have defaults that work for local development; see .env.example
for the full list and what each does.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated settings loaded from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM ------------------------------------------------------------------
    # "hf": load a local HF model (merged dir, or base + LoRA adapter)
    # "ollama": stream from a local Ollama server (zero VRAM setup)
    # "fake": deterministic echo generator for tests / frontend dev
    # "openai": any OpenAI-compatible API (OpenAI, Groq, OpenRouter, DeepSeek, etc.)
    llm_backend: str = "fake"
    hf_model_dir: str = "finetune/outputs/merged"
    hf_adapter_dir: str = "finetune/outputs/adapter"
    hf_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # --- generation -------------------------------------------------------------
    max_new_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.9

    # --- RAG ---------------------------------------------------------------------
    rag_index_dir: str = "rag/index"
    rag_top_k: int = 4
    rag_score_threshold: float = 0.35
    rag_rerank: bool = False

    # --- API behaviour ------------------------------------------------------------
    # History is trimmed to this many tokens before the prompt is built.
    history_token_budget: int = 1500
    # Hard cap on a single user message (characters; ~1 token per 4 chars).
    max_input_chars: int = 4000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    rate_limit_requests: int = 20
    rate_limit_window_s: int = 60

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so every module sees the same settings object."""
    return Settings()


# Resolve repo-relative paths against the repo root (backend/..), NOT the
# current working directory, so `uvicorn backend.main:app` works from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)
