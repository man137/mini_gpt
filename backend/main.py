"""FastAPI chat backend with Server-Sent Events streaming.

Endpoints:
  POST /chat    - SSE stream: events "sources" (before tokens), then
                  "token" per chunk, then "done" (or "error").
  POST /ingest  - upload documents (.txt/.md/.pdf), rebuild the FAISS index.
  GET  /health  - one-glance status: backend, model, index size.

Why SSE (Server-Sent Events) over WebSockets: chat generation is strictly
server->client streaming of one response per request. SSE is plain HTTP,
auto-reconnects, works with a single GET/POST round trip, needs no extra
protocol upgrade, and FastAPI supports it with a simple async generator -
WebSockets would only pay off for bidirectional real-time features (typing
indicators, collaborative rooms).

Run:  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import llm
from backend.config import get_settings, resolve_path
from backend.schemas import ChatRequest, HealthResponse

logger = logging.getLogger("mini-chatgpt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="mini-chatgpt", version="1.0.0")
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Simple per-IP sliding-window rate limiter (in-memory; fine for a demo).
# A production deployment would use Redis so the state survives restarts
# and is shared across workers.
# ---------------------------------------------------------------------------
_rate_events: dict[str, deque] = defaultdict(deque)


def _rate_limited(client_ip: str) -> bool:
    now = time.time()
    window = settings.rate_limit_window_s
    events = _rate_events[client_ip]
    while events and now - events[0] > window:
        events.popleft()
    if len(events) >= settings.rate_limit_requests:
        return True
    events.append(now)
    return False


# ---------------------------------------------------------------------------
# RAG state (index is optional - the API still works without one)
# ---------------------------------------------------------------------------
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import Retriever

        _retriever = Retriever(
            index_dir=resolve_path(settings.rag_index_dir),
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
            rerank=settings.rag_rerank,
        )
    return _retriever


def index_size() -> int:
    try:
        return get_retriever().index.ntotal
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sse(event: str, data) -> str:
    """Format one SSE frame: `event: <name>\ndata: <json>\n\n`."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _trim_history(messages: list[dict], token_budget: int) -> list[dict]:
    """Keep the system prompt + as many recent turns as fit the budget.

    Approximation: 1 token ~= 4 characters. Precision is not critical here -
    the point is to bound prompt growth in long conversations.
    """
    if not messages:
        return []
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    budget = token_budget * 4
    kept: list[dict] = []
    for m in reversed(rest):  # newest first
        if budget - len(m["content"]) < 0:
            break
        kept.insert(0, m)
        budget -= len(m["content"])
    return system + kept


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + system snapshot for the frontend and monitoring."""
    if settings.llm_backend == "hf":
        active_model = settings.hf_model_dir
    elif settings.llm_backend in {"openai", "api"}:
        active_model = settings.openai_model
    else:
        active_model = settings.ollama_model

    return HealthResponse(
        status="ok",
        backend=llm.describe_backend(),
        model=active_model,
        index_size=index_size(),
        rag_enabled=index_size() > 0,
    )


@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream a chat answer as SSE. Requires `StreamingResponse` (not
    JSONResponse) because the body is an unbounded event stream."""
    request_id = uuid.uuid4().hex[:12]
    client_ip = request.client.host if request.client else "unknown"
    logger.info("chat request_id=%s ip=%s use_rag=%s msgs=%d",
                request_id, client_ip, req.use_rag, len(req.messages))

    if _rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded, retry in a minute")

    # Input length guard: reject oversized user messages early (413).
    for m in req.messages:
        if len(m.content) > settings.max_input_chars:
            raise HTTPException(
                status_code=413,
                detail=f"message too long ({len(m.content)} chars; "
                       f"limit {settings.max_input_chars})",
            )

    messages = [m.model_dump() for m in req.messages]
    history = _trim_history(messages, settings.history_token_budget)
    question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")

    sources_payload: list[dict] = []
    if req.use_rag and question:
        try:
            hits = get_retriever().retrieve(question)
        except FileNotFoundError:
            hits = []  # no index built yet: answer without RAG
        sources_payload = [
            {
                "source": h["source"],
                "chunk_id": h["chunk_id"],
                "score": round(h["score"], 3),
                "snippet": h["text"][:200],
                "page": h.get("page"),
            }
            for h in hits
        ]
        if hits:
            from rag.prompt import build_messages as build_rag_messages

            # Replace history with a grounded prompt: system rules + context +
            # question. Older turns are dropped when RAG supplies the context.
            final_messages = build_rag_messages(question, hits)
        else:
            from rag.prompt import SYSTEM_PROMPT

            final_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *[m for m in history if m["role"] != "system"],
            ]
    else:
        final_messages = history

    gen_kwargs = {}
    if req.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature

    async def event_stream():
        """The SSE body: sources -> tokens -> done, with error capture."""
        try:
            if sources_payload:
                yield _sse("sources", sources_payload)
            for chunk in llm.stream(final_messages, **gen_kwargs):
                yield _sse("token", {"text": chunk})
            yield _sse("done", {"request_id": request_id})
        except Exception as exc:  # noqa: BLE001 - surface ANY failure to the client
            logger.exception("chat failed request_id=%s", request_id)
            yield _sse("error", {"detail": str(exc), "request_id": request_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # never cache a stream
            "X-Accel-Buffering": "no",    # disable proxy buffering (nginx)
        },
    )


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Upload .txt/.md/.pdf documents and rebuild the RAG index from
    rag/docs + the uploaded files (uploads are saved into rag/docs)."""
    saved: list[str] = []
    docs_dir = resolve_path("rag/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            raise HTTPException(400, f"unsupported file type {suffix!r}")
        dest = docs_dir / Path(f.filename).name
        # Untrusted upload: keep the original bytes on disk as-is; parsing
        # happens in rag/ingest with the same code path as CLI ingestion.
        dest.write_bytes(await f.read())
        saved.append(dest.name)
    logger.info("ingest uploaded=%s", saved)

    # Rebuild index by calling the ingest pipeline programmatically.
    from rag.ingest import main as ingest_main
    import sys as _sys

    _sys.argv = ["ingest", "--docs", str(docs_dir),
                 "--index", str(resolve_path(settings.rag_index_dir))]
    ingest_main()

    # Drop the cached retriever so the next request sees the new index.
    global _retriever
    _retriever = None
    return {"saved": saved, "index_size": index_size()}


# ---------------------------------------------------------------------------
# Global exception handler: uniform JSON errors instead of HTML tracebacks
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return StreamingResponse(
        iter([json.dumps({"detail": exc.detail})]),
        status_code=exc.status_code,
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Serve frontend static assets in production if built
# ---------------------------------------------------------------------------
frontend_dist = resolve_path("frontend/dist")
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    from starlette.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

