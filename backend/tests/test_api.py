"""Tests for the FastAPI backend (fake LLM backend, no GPU needed).

Run:  pytest backend/tests/test_api.py -v
Covers: health, SSE event ordering, oversized-input rejection, rate limiting,
history trimming and the /ingest rebuild path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app, _trim_history  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        ev = None
        data = None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                ev = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if ev is not None:
            events.append((ev, data))
    return events


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "backend" in body and "index_size" in body


class TestChat:
    def test_chat_streams_events_in_order(self, client: TestClient) -> None:
        """Events must arrive as (optional sources) -> tokens -> done."""
        r = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "How long does the flu vaccine take to work?"}
                ],
                "use_rag": False,  # fake backend; skip retrieval for determinism
            },
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(r.text)
        names = [e for e, _ in events]
        assert names[0] in {"sources", "token"}
        assert "token" in names
        assert names[-1] == "done"  # done is always last
        # no error event anywhere in a successful stream
        assert "error" not in names
        tokens = [d["text"] for e, d in events if e == "token"]
        assert any("vaccine" in t for t in tokens)  # fake backend echoes the question

    def test_chat_with_rag_emits_sources_first(self, client: TestClient) -> None:
        """With use_rag=True and an index present, sources come BEFORE any
        token so the UI can render the panel while text streams."""
        r = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "How long is flu contagious?"}],
                  "use_rag": True},
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        names = [e for e, _ in events]
        if "sources" in names:  # index exists -> sources precede tokens
            assert names.index("sources") < names.index("token")
            sources = next(d for e, d in events if e == "sources")
            assert isinstance(sources, list) and sources
            s = sources[0]
            assert {"source", "chunk_id", "score", "snippet"} <= set(s)

    def test_oversized_input_rejected(self, client: TestClient) -> None:
        long_msg = "x" * 5000  # over the default 4000-char cap
        r = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": long_msg}], "use_rag": False},
        )
        assert r.status_code == 413
        assert "too long" in r.json()["detail"]

    def test_invalid_role_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/chat",
            json={"messages": [{"role": "wizard", "content": "hi"}]},
        )
        assert r.status_code == 422  # pydantic validation error

    def test_empty_messages_rejected(self, client: TestClient) -> None:
        r = client.post("/chat", json={"messages": []})
        assert r.status_code == 422


class TestRateLimit:
    def test_rate_limit_trips(self, client: TestClient) -> None:
        """The default is 20 req/min; fire enough requests to trip 429."""
        codes = []
        for _ in range(25):
            r = client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "hi"}], "use_rag": False},
            )
            codes.append(r.status_code)
            if r.status_code == 429:
                break
        assert 429 in codes

    def test_rate_limit_isolated_per_ip(self, client: TestClient) -> None:
        """A different client IP must NOT inherit another IP's counter."""
        # exercise one IP heavily first
        for _ in range(25):
            client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "hi"}], "use_rag": False},
            )
        # then a fresh IP still gets through
        client.headers.update({"x-forwarded-for": "10.9.8.7"})
        # NOTE: TestClient's client.host is always testclient, so this checks
        # the limiter data structure rather than a true e2e path; the real
        # per-IP behaviour is covered by the keying on request.client.host.
        from backend.main import _rate_limited

        assert _rate_limited("1.2.3.4") is False  # fresh IP not limited
        assert _rate_limited("1.2.3.4") is False
        assert _rate_limited("1.2.3.4") is False


class TestHistoryTrimming:
    def test_trims_to_budget(self) -> None:
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 100}
            for i in range(40)
        ]
        trimmed = _trim_history(msgs, token_budget=500)  # 500*4 = 2000 chars
        assert trimmed[0]["role"] == "system"  # system always survives
        total_chars = sum(len(m["content"]) for m in trimmed)
        assert total_chars <= 500 * 4 + 100  # within budget (+1 message slack)

    def test_keeps_recent_over_old(self) -> None:
        msgs = [
            {"role": "user", "content": "OLD-" + "a" * 500},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "NEW-question"},
        ]
        trimmed = _trim_history(msgs, token_budget=50)
        contents = " ".join(m["content"] for m in trimmed)
        assert "NEW-question" in contents
        assert "OLD-" not in contents  # oldest dropped first


class TestIngest:
    def test_ingest_rejects_bad_type(self, client: TestClient) -> None:
        r = client.post(
            "/ingest",
            files={"files": ("notes.exe", b"binary", "application/octet-stream")},
        )
        assert r.status_code == 400

    def test_ingest_accepts_md(self, client: TestClient) -> None:
        r = client.post(
            "/ingest",
            files={"files": ("test_upload.md", b"# Test doc\n\nSome test content here.", "text/markdown")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "test_upload.md" in body["saved"]
        assert body["index_size"] >= 1
        # cleanup so other tests see the original docs
        p = Path(__file__).resolve().parents[2] / "rag" / "docs" / "test_upload.md"
        p.unlink(missing_ok=True)
