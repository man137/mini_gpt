"""RAG building blocks: document loading, token-aware chunking, embedding,
FAISS retrieval with score threshold, optional cross-encoder reranking.

Kept dependency-light on purpose: this module is imported by ingest.py, the
backend and the tests, so the embedding model loads lazily (only when
embedding/retrieving actually happens).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # 384-dim, strong for its size
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"   # cross-encoder, optional
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "index"
DEFAULT_TOP_K = 4
DEFAULT_SCORE_THRESHOLD = 0.35  # cosine sim for bge-small; tune on your data

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> tuple[str, list[int]]:
    """Extract text page-by-page; returns (text, page_number_per_char).

    Keeping a char->page map lets chunks record the page they came from.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    pages: list[int] = []
    for page_no, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        parts.append(t)
        pages.extend([page_no] * len(t))
    return "".join(parts), pages


def load_documents(docs_dir: Path) -> list[dict]:
    """Load every .txt/.md/.pdf in `docs_dir` -> [{source, text, pages?}]."""
    docs: list[dict] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            text, page_map = _read_pdf(path)
            docs.append({"source": path.name, "text": text, "page_map": page_map})
        elif path.suffix.lower() in {".txt", ".md"}:
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


# ---------------------------------------------------------------------------
# Token-aware chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 80,
    source: str = "unknown",
    tokenizer=None,
) -> list[dict]:
    """Split `text` into chunks of ~max_tokens tokens with overlap.

    Token-aware beats character-aware because "500 tokens" is what the
    embedding model actually sees. We default to a lazy heuristic split by
    sentence boundaries first, then pack sentences up to the token budget.
    The OVERLAP (80 tokens) ensures a fact sitting on a boundary appears
    fully in at least one chunk.

    Returns [{"text", "n_tokens", "source", "chunk_id", "page"}].
    """
    if tokenizer is None:
        # Lightweight default: approximate 1 token ~ 4 characters for
        # English text (a common rule of thumb). Tests pass an exact
        # tokenizer when they need precise control.
        def n_tokens(s: str) -> int:
            return max(1, len(s) // 4)
    else:
        def n_tokens(s: str) -> int:
            return len(tokenizer.encode(s, add_special_tokens=False))

    # Sentence-ish split; keeps headings attached to their section text.
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]

    # A single sentence longer than max_tokens (e.g. minified text with no
    # punctuation) must be force-split, otherwise the chunk cap is violated.
    split_sentences: list[str] = []
    for sent in sentences:
        if n_tokens(sent) <= max_tokens:
            split_sentences.append(sent)
            continue
        words = sent.split(" ")
        part: list[str] = []
        part_t = 0
        for w in words:
            t = n_tokens(w)
            if part and part_t + t > max_tokens:
                split_sentences.append(" ".join(part))
                part, part_t = [], 0
            part.append(w)
            part_t += t
        if part:
            split_sentences.append(" ".join(part))
    sentences = split_sentences

    chunks: list[dict] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        t = n_tokens(sent)
        if current_tokens + t > max_tokens and current:
            chunk_txt = " ".join(current)
            chunks.append(
                {
                    "text": chunk_txt,
                    "n_tokens": current_tokens,
                    "source": source,
                    "chunk_id": f"{source}:chunk{len(chunks)}",
                }
            )
            # Keep the last ~overlap_tokens worth of sentences as the start
            # of the next chunk (that is the overlap).
            kept: list[str] = []
            kept_tokens = 0
            for s in reversed(current):
                if kept_tokens + n_tokens(s) > overlap_tokens:
                    break
                kept.insert(0, s)
                kept_tokens += n_tokens(s)
            current, current_tokens = kept, kept_tokens
        current.append(sent)
        current_tokens += t
    if current:
        chunks.append(
            {
                "text": " ".join(current),
                "n_tokens": current_tokens,
                "source": source,
                "chunk_id": f"{source}:chunk{len(chunks)}",
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_embedder = None  # module-level cache so the model loads once per process


def get_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Load (once) and cache the sentence-transformers model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_texts(texts: list[str], model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    """Embed and L2-normalise texts.

    Normalisation is the trick that makes FAISS inner-product (IndexFlatIP)
    return cosine-similarity rankings: for unit vectors, dot == cosine.
    """
    model = get_embedder(model_name)
    emb = model.encode(
        texts,
        normalize_embeddings=True,  # <- normalise here, not after
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(emb, dtype="float32")


# ---------------------------------------------------------------------------
# Index building / persistence
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray):
    """Build a FAISS inner-product index over (normalised) embeddings."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact search, no approximation
    index.add(np.ascontiguousarray(embeddings))
    return index


def load_index(index_dir: Path = DEFAULT_INDEX_DIR):
    """Load the persisted FAISS index + metadata list."""
    import faiss

    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"RAG index not found in {index_dir} - run `python rag/ingest.py` first"
        )
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if index.ntotal != len(metadata):
        raise ValueError(
            f"index/metadata mismatch: {index.ntotal} vectors but {len(metadata)} "
            f"metadata rows - rebuild with rag/ingest.py"
        )
    return index, metadata


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class Retriever:
    """Top-k retrieval with a score threshold and optional cross-encoder rerank."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embed_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        rerank: bool = False,
        rerank_model: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.index, self.metadata = load_index(index_dir)
        self.embed_model = embed_model
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rerank = rerank
        self.rerank_model = rerank_model
        self._reranker = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Return [{text, source, chunk_id, score, page}] best-first.

        Chunks below `score_threshold` are dropped, which is what lets the
        prompt layer say "no relevant context found" for out-of-domain
        questions instead of forcing irrelevant chunks into the prompt.
        """
        k = top_k or self.top_k
        q_emb = embed_texts([query], self.embed_model)  # (1, dim), normalised
        # FAISS: search returns (scores, ids) arrays of shape (1, k)
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(q_emb, k)
        hits = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id == -1:  # FAISS pads with -1 when k > ntotal
                continue
            if float(score) < self.score_threshold:
                continue
            meta = self.metadata[int(row_id)]
            hits.append(
                {
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"],
                    "page": meta.get("page"),
                    "score": float(score),
                }
            )
        if self.rerank and hits:
            hits = self._rerank_hits(query, hits)
        return hits

    def _rerank_hits(self, query: str, hits: list[dict]) -> list[dict]:
        """Cross-encoder reranking: the model scores (query, chunk) pairs
        jointly, which is more accurate than bi-encoder cosine but ~50x
        slower per pair - hence only worth it on the shortlist."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.rerank_model)
        pairs = [(query, h["text"]) for h in hits]
        ce_scores = self._reranker.predict(pairs)
        order = np.argsort(-np.asarray(ce_scores))
        return [hits[i] for i in order]
