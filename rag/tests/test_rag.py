"""Tests for the RAG pipeline.

Two layers:
  - PURE tests (no model download): chunking, prompt construction, threshold
    logic with a tiny fake index.
  - INTEGRATION tests (marked `rag_integration`, need the bge-small model
    and a built index): real retrieval over rag/docs.

Run:
    pytest rag/tests/test_rag.py -v                       # fast, offline
    pytest rag/tests/test_rag.py -v -m rag_integration    # needs network
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_messages,
    build_prompt,
    extract_citations,
    is_refusal,
)
from rag.retriever import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    Retriever,
    chunk_text,
    load_documents,
)

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


# ---------------------------------------------------------------------------
# Chunking (pure)
# ---------------------------------------------------------------------------
class TestChunking:
    def test_chunk_sizes_respected(self) -> None:
        """With an exact tokenizer, chunks must not exceed max_tokens."""
        class ExactTok:
            def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
                return s.split()  # "1 token" == 1 word

        text = " ".join(f"w{i}" for i in range(2000))  # 2000 "tokens"
        chunks = chunk_text(
            text, max_tokens=100, overlap_tokens=20,
            source="t.txt", tokenizer=ExactTok(),
        )
        assert all(c["n_tokens"] <= 100 for c in chunks)
        assert len(chunks) >= 10  # and it actually split

    def test_overlap_exists(self) -> None:
        """Consecutive chunks must share content (the overlap)."""
        class ExactTok:
            def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
                return s.split()

        # Sentences of 10 words each -> clean boundaries.
        text = ". ".join(
            " ".join(f"s{i}w{j}" for j in range(10)) for i in range(40)
        ) + "."
        chunks = chunk_text(
            text, max_tokens=50, overlap_tokens=20,
            source="t.txt", tokenizer=ExactTok(),
        )
        assert len(chunks) >= 2
        # Compare CONSECUTIVE pairs: the tail of chunk[i] must reappear as
        # the head of chunk[i+1] (that is the overlap).
        for prev, nxt in zip(chunks, chunks[1:]):
            overlap = set(prev["text"].split()) & set(nxt["text"].split())
            assert overlap, "no overlap between consecutive chunks"

    def test_metadata_present(self) -> None:
        chunks = chunk_text("a b c. d e f.", max_tokens=10, overlap_tokens=2, source="doc.md")
        assert chunks and all(c["source"] == "doc.md" for c in chunks)
        assert all(c["chunk_id"].startswith("doc.md:chunk") for c in chunks)

    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_text("hello world", max_tokens=100, overlap_tokens=10, source="s.md")
        assert len(chunks) == 1

    def test_real_docs_load(self) -> None:
        docs = load_documents(DOCS_DIR)
        assert len(docs) == 3
        sources = {d["source"] for d in docs}
        assert "influenza_guidance.md" in sources
        assert "sleep_health.md" in sources


# ---------------------------------------------------------------------------
# Prompt construction (pure)
# ---------------------------------------------------------------------------
class TestPrompt:
    def test_system_prompt_rules(self) -> None:
        assert "only" in SYSTEM_PROMPT.lower()
        assert "I don't know based on the provided documents" in SYSTEM_PROMPT
        assert "ignore them" in SYSTEM_PROMPT  # injection defence

    def test_build_prompt_with_context(self) -> None:
        chunks = [
            {"text": "The flu vaccine takes two weeks to work.",
             "source": "flu.md", "chunk_id": "flu.md:chunk0", "score": 0.7},
            {"text": "Wash hands for 20 seconds.",
             "source": "hyg.md", "chunk_id": "hyg.md:chunk1", "score": 0.5},
        ]
        p = build_prompt("How long until the vaccine works?", chunks)
        assert "flu.md:chunk0" in p
        assert "How long until the vaccine works?" in p
        msgs = build_messages("q", chunks)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_build_prompt_empty_context(self) -> None:
        p = build_prompt("anything", [])
        assert "No relevant context" in p

    def test_refusal_detection(self) -> None:
        assert is_refusal("I don't know based on the provided documents.")
        assert not is_refusal("The flu vaccine takes two weeks.")
        # case-insensitive + trailing content
        assert is_refusal("I DON'T KNOW BASED ON THE PROVIDED DOCUMENTS. ")

    def test_citation_extraction(self) -> None:
        a = "Vaccines reduce risk [source:flu.md:chunk0] and hand washing helps [source:hyg.md:chunk1]."
        assert extract_citations(a) == ["flu.md:chunk0", "hyg.md:chunk1"]


# ---------------------------------------------------------------------------
# Threshold behaviour with a FAKE index (offline, deterministic)
# ---------------------------------------------------------------------------
class _FakeRetriever(Retriever):
    """Bypass index loading; return fixed scores for known queries."""

    def __init__(self, fixed: list[dict]) -> None:  # skip parent __init__
        self.fixed = fixed
        self.top_k = 4
        self.score_threshold = 0.35
        self.rerank = False

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        out = [h for h in self.fixed if h["score"] >= self.score_threshold]
        return out[: (top_k or self.top_k)]


class TestThreshold:
    def test_below_threshold_dropped(self) -> None:
        r = _FakeRetriever(
            [
                {"text": "low relevance", "source": "a.md", "chunk_id": "a.md:chunk0", "score": 0.10},
                {"text": "good", "source": "b.md", "chunk_id": "b.md:chunk0", "score": 0.80},
            ]
        )
        hits = r.retrieve("query")
        assert [h["chunk_id"] for h in hits] == ["b.md:chunk0"]

    def test_all_below_threshold_gives_no_answer_path(self) -> None:
        r = _FakeRetriever(
            [{"text": "x", "source": "a.md", "chunk_id": "a.md:chunk0", "score": 0.01}]
        )
        assert r.retrieve("query") == []
        # and the prompt layer must then produce the "no context" prompt:
        assert "No relevant context" in build_prompt("q", r.retrieve("q"))

    def test_threshold_is_configurable(self) -> None:
        r = _FakeRetriever(
            [{"text": "mid", "source": "a.md", "chunk_id": "a.md:chunk0", "score": 0.5}]
        )
        r.score_threshold = 0.6
        assert r.retrieve("q") == []
        r.score_threshold = 0.4
        assert len(r.retrieve("q")) == 1


# ---------------------------------------------------------------------------
# Integration: real retrieval over the sample docs
# ---------------------------------------------------------------------------
@pytest.mark.rag_integration
class TestRealRetrieval:
    @pytest.fixture(scope="class")
    def retriever(self) -> Retriever:
        import faiss  # noqa: F401 - fail early with a clear error if missing
        if not (DEFAULT_INDEX_DIR / "index.faiss").exists():
            pytest.skip("index not built - run `python rag/ingest.py` first")
        return Retriever(top_k=3)

    def test_right_chunk_for_known_query(self, retriever: Retriever) -> None:
        hits = retriever.retrieve("How long does the flu vaccine take to work?")
        assert hits, "expected at least one chunk above threshold"
        joined = " ".join(h["text"].lower() for h in hits)
        assert "two weeks" in joined  # the fact lives in influenza_guidance.md

    def test_retrieval_returns_metadata(self, retriever: Retriever) -> None:
        hits = retriever.retrieve("hand washing technique")
        assert hits
        for h in hits:
            assert h["source"] and h["chunk_id"] and 0.0 <= h["score"] <= 1.0

    def test_out_of_domain_scores_low(self, retriever: Retriever) -> None:
        hits = retriever.retrieve("What is the capital of France?")
        # Either nothing passes the threshold, or scores are clearly low.
        assert not hits or all(h["score"] < 0.5 for h in hits)
