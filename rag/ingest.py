"""Ingest documents into the RAG store: load, chunk, embed, index, persist.

Pipeline:
  1. Load .txt / .md / .pdf files from a directory (or list).
  2. Split into ~500-token chunks with 80-token overlap (token-aware, using
     the embedding model's own tokenizer so "500 tokens" means model tokens).
  3. Embed chunks with BAAI/bge-small-en-v1.5 via sentence-transformers,
     L2-NORMALISED (so inner product == cosine similarity; FAISS IndexFlatIP
     then returns cosine-ranked results).
  4. Persist a FAISS IndexFlatIP + a metadata JSON side-by-side.

Usage:
    python rag/ingest.py                     # ingest rag/docs
    python rag/ingest.py --docs my_docs/ --index rag/index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_DIR,
    build_index,
    chunk_text,
    embed_texts,
    load_documents,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=Path, default=Path(__file__).resolve().parent / "docs")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--chunk_tokens", type=int, default=500)
    p.add_argument("--overlap_tokens", type=int, default=80)
    args = p.parse_args()

    # ---- 1. load ------------------------------------------------------------
    docs = load_documents(args.docs)
    if not docs:
        raise SystemExit(f"no .txt/.md/.pdf documents found in {args.docs}")
    print(f"loaded {len(docs)} documents: {[d['source'] for d in docs]}")

    # ---- 2. chunk -------------------------------------------------------------
    all_chunks: list[dict] = []
    for doc in docs:
        chunks = chunk_text(
            doc["text"],
            max_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            source=doc["source"],
        )
        all_chunks.extend(chunks)
    sizes = [c["n_tokens"] for c in all_chunks]
    print(
        f"chunked into {len(all_chunks)} chunks "
        f"(tokens: min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)})"
    )

    # ---- 3. embed ------------------------------------------------------------
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, model_name=args.embed_model)
    print(f"embedded {embeddings.shape[0]} chunks -> dim {embeddings.shape[1]}")

    # ---- 4. index + persist ---------------------------------------------------
    args.index.mkdir(parents=True, exist_ok=True)
    index = build_index(embeddings)
    import faiss

    faiss.write_index(index, str(args.index / "index.faiss"))
    # Metadata is stored NEXT TO the index with the same row order; the i-th
    # FAISS row corresponds to the i-th metadata entry.
    (args.index / "metadata.json").write_text(
        json.dumps(
            [
                {
                    "source": c["source"],
                    "chunk_id": c["chunk_id"],
                    "page": c.get("page"),
                    "text": c["text"],
                    "n_tokens": c["n_tokens"],
                }
                for c in all_chunks
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"index ({index.ntotal} vectors) + metadata written to {args.index}")


if __name__ == "__main__":
    main()
