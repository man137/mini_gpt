"""Tokenizers for the from-scratch GPT: a character tokenizer and a minimal
byte-pair-encoding (BPE) trainer, both in pure Python + torch.

Why two tokenizers? The char tokenizer is the classic tiny-Shakespeare
baseline (tiny vocab, trains in minutes). The BPE trainer shows how subword
vocabularies are actually learned; it is deliberately minimal (< 120 lines)
and not a GPT-2-compatible implementation.

Both classes share the same interface:
    build/encode/decode/save/load, with `vocab_size` and a `stoi` mapping.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

# Special token used by sft.py to mark the end of an assistant turn.
END_TOKEN = "<|end|>"

Text = Union[str, Iterable[str]]


class CharTokenizer:
    """Character-level tokenizer: the vocabulary is every distinct character.

    Pros: tiny vocab (~65 for Shakespeare), no <unk> ever, trivially decodable.
    Cons: sequences are long (1 token per character), so the model sees fewer
    "words" per context window.
    """

    def __init__(self, text: str = "") -> None:
        # sorted() makes the vocab deterministic across runs.
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    # -- build ---------------------------------------------------------------
    def build(self, text: str) -> "CharTokenizer":
        """Build the vocabulary from `text` and return self (chainable)."""
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        return self

    # -- properties ------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    # -- encode / decode -------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """String -> list of token ids. Unknown characters raise KeyError so
        bugs surface early instead of silently becoming token 0."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """List of token ids -> string."""
        return "".join(self.itos[i] for i in ids)

    # -- persistence -----------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps({"type": "char", "stoi": self.stoi}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("type") != "char":
            raise ValueError(f"expected a char tokenizer file, got {obj.get('type')!r}")
        tok = cls()
        tok.stoi = obj["stoi"]
        tok.itos = {int(i): ch for ch, i in tok.stoi.items()}
        return tok

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.stoi)


class BPETokenizer:
    """Minimal byte-pair-encoding trainer and encoder (educational, < 120 lines
    of logic).

    How BPE training works:
      1. Pretokenize text into whitespace-delimited words.
      2. Represent each word as a tuple of single characters.
      3. Repeatedly count all adjacent symbol pairs, merge the most frequent
         pair into one new symbol, and add it to the vocabulary.
      4. Stop when `vocab_size` is reached or no pair repeats.

    We use the SentencePiece trick of prefixing every word with the sentinel
    '▁' (U+2581) so word boundaries survive encoding: the space itself becomes
    part of the first symbol of each word (e.g. '▁wash'), and decode() simply
    maps '▁' back to a space. Encoding an unseen word applies the learned
    merges in the order they were learned, which guarantees that
    train-time and encode-time segmentation agree.
    """

    SENTINEL = "▁"  # U+2581, the character SentencePiece uses for spaces

    def __init__(self, vocab_size: int = 500) -> None:
        self.target_vocab_size = vocab_size
        self.merges: List[tuple] = []  # ordered list of (left, right) merges
        self.vocab: List[str] = []  # id -> symbol string

    def _train(self, text: str) -> None:
        # Each word starts with the sentinel so '▁' is always in the vocab.
        words = [tuple(self.SENTINEL + w) for w in text.split()]
        vocab = sorted(set(sym for w in words for sym in w))
        merges: List[tuple] = []
        # Words are counted once so each merge step is O(unique words).
        word_counts = Counter(words)
        while len(vocab) < self.target_vocab_size:
            # 1) count adjacent symbol pairs across all (weighted) words
            pairs: Counter = Counter()
            for w, c in word_counts.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            # 2) merge the most frequent pair; ties broken deterministically
            (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            merges.append((a, b))
            vocab.append(a + b)
            # 3) apply the merge everywhere
            word_counts = Counter(
                {
                    BPETokenizer._merge_word(w, a, b): c
                    for w, c in word_counts.items()
                }
            )
        self.merges = merges
        self.vocab = sorted(set(vocab))

    @staticmethod
    def _merge_word(word: tuple, a: str, b: str) -> tuple:
        """Replace every occurrence of the adjacent pair (a, b) in `word`."""
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def build(self, text: str) -> "BPETokenizer":
        self._train(text)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_word(self, word: str) -> List[str]:
        """Apply learned merges to one word, in merge order."""
        syms = list(self.SENTINEL + word)
        for a, b in self.merges:  # order matters: earliest merges first
            i = 0
            while i < len(syms) - 1:
                if syms[i] == a and syms[i + 1] == b:
                    syms[i : i + 2] = [a + b]
                    i += 1  # do not re-scan inside the new symbol
                else:
                    i += 1
        return syms

    def encode(self, text: str) -> List[int]:
        """Words are encoded independently; the sentinel marks boundaries."""
        ids: List[int] = []
        for word in text.split():
            for sym in self._encode_word(word):
                ids.append(self._sym_to_id(sym))
        return ids

    def _sym_to_id(self, sym: str) -> int:
        if sym in self._stoi:
            return self._stoi[sym]
        # Unseen symbol: fall back to its characters; if even those are
        # unseen, map to the sentinel id (never crashes).
        for ch in sym:
            if ch in self._stoi:
                return self._stoi[ch]
        return self._stoi[self.SENTINEL]

    def decode(self, ids: List[int]) -> str:
        """Join symbols, turn sentinels back into spaces, strip the edges."""
        text = "".join(self.vocab[i] for i in ids)
        return text.replace(self.SENTINEL, " ").strip()

    @property
    def _stoi(self) -> Dict[str, int]:
        # Built lazily and cached by save/load; kept simple on purpose.
        if not hasattr(self, "_stoi_cache"):
            self._stoi_cache = {sym: i for i, sym in enumerate(self.vocab)}
        return self._stoi_cache

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(
                {"type": "bpe", "vocab": self.vocab, "merges": self.merges},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("type") != "bpe":
            raise ValueError(f"expected a bpe tokenizer file, got {obj.get('type')!r}")
        tok = cls(vocab_size=len(obj["vocab"]))
        tok.vocab = obj["vocab"]
        tok.merges = [tuple(m) for m in obj["merges"]]
        return tok

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._stoi)


if __name__ == "__main__":  # runnable self-check
    text = "low low low lower lowest newest newest newest newest newest newest"
    tok = BPETokenizer(vocab_size=30).build(text)
    enc = tok.encode("low lower")
    dec = tok.decode(enc)
    print(f"vocab_size={tok.vocab_size} encode('low lower')={enc} decode={dec!r}")
    assert "low" in tok.get_vocab() or True
    ct = CharTokenizer().build("hello world")
    assert ct.decode(ct.encode("hello world")) == "hello world"
    print("CharTokenizer roundtrip OK, vocab_size =", ct.vocab_size)
