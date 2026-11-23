"""Pydantic request/response models for the chat API.

Two shapes matter:
  - the JSON the frontend POSTs to /chat;
  - the SSE event payloads the server streams back
    ({"event": "token"|"sources"|"done"|"error", "data": ...}).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single turn of conversation history (frontend -> backend)."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    """POST /chat body.

    `use_rag` toggles retrieval-augmented answering; with False the question
    goes straight to the model (used by the eval harness to compare systems).
    """

    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    use_rag: bool = True
    max_new_tokens: Optional[int] = Field(None, ge=16, le=1024)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class TokenEvent(BaseModel):
    event: Literal["token"] = "token"
    data: str


class SourceInfo(BaseModel):
    source: str
    chunk_id: str
    score: float
    snippet: str = Field(..., description="first ~200 chars of the chunk")
    page: Optional[int] = None


class SourcesEvent(BaseModel):
    event: Literal["sources"] = "sources"
    data: list[SourceInfo]


class DoneEvent(BaseModel):
    event: Literal["done"] = "done"
    data: dict = Field(default_factory=dict)  # e.g. {"request_id": ...}


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    data: dict  # {"detail": "...", "request_id": ...}


class HealthResponse(BaseModel):
    """GET /health payload - one glance at the whole system state."""

    status: str
    backend: str
    model: str
    index_size: int
    rag_enabled: bool
