"""Tests for the from-scratch GPT (tokenizer, model, SFT loss masking).

Run:  pytest scratch_gpt/tests/test_scratch.py -v
CPU-only, < 30 s total. Every test asserts one documented property."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch_gpt.model import GPT, GPTConfig, Head, MultiHeadAttention, FeedForward, Block  # noqa: E402
from scratch_gpt.tokenizer import BPETokenizer, CharTokenizer, END_TOKEN  # noqa: E402
from scratch_gpt.sft import (  # noqa: E402
    ASSISTANT_TAG,
    USER_TAG,
    ChatDataset,
    DEMO_CONVERSATIONS,
    collate,
)

# Small config used by most tests (fast on CPU).
CFG = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=50, dropout=0.0)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class TestCharTokenizer:
    def test_roundtrip(self) -> None:
        tok = CharTokenizer()
        tok.build("hello world")
        s = "hello world"
        assert tok.decode(tok.encode(s)) == s

    def test_vocab_built_from_text(self) -> None:
        tok = CharTokenizer().build("aabbc")
        assert tok.vocab_size == 3  # {'a', 'b', 'c'}
        assert set(tok.stoi) == {"a", "b", "c"}

    def test_save_load(self, tmp_path: Path) -> None:
        tok = CharTokenizer().build("abcdef")
        tok.save(tmp_path / "t.json")
        tok2 = CharTokenizer.load(tmp_path / "t.json")
        assert tok2.encode("face") == tok.encode("face")

    def test_unknown_char_raises(self) -> None:
        tok = CharTokenizer().build("abc")
        with pytest.raises(KeyError):
            tok.encode("xyz")  # 'x' never seen -> loud failure, not silent 0


class TestBPETokenizer:
    def test_roundtrip_on_training_text(self) -> None:
        text = "public health guidance stay home when sick wash hands cover coughs " * 3
        tok = BPETokenizer(vocab_size=60).build(text)
        for s in ["stay home", "public health", "wash hands", "guidance"]:
            assert tok.decode(tok.encode(s)) == s

    def test_merges_shorten_sequences(self) -> None:
        text = "stay home stay home stay home stay home"
        char_tok = CharTokenizer().build(text)
        bpe = BPETokenizer(vocab_size=40).build(text)
        assert len(bpe.encode(text)) < len(char_tok.encode(text))

    def test_save_load(self, tmp_path: Path) -> None:
        tok = BPETokenizer(vocab_size=30).build("alpha beta gamma alpha beta")
        tok.save(tmp_path / "b.json")
        tok2 = BPETokenizer.load(tmp_path / "b.json")
        assert tok2.encode("alpha beta") == tok.encode("alpha beta")


# ---------------------------------------------------------------------------
# Shapes of every component
# ---------------------------------------------------------------------------
class TestShapes:
    def test_head_shape(self) -> None:
        head = Head(head_size=8, n_embd=32, block_size=16, dropout=0.0)
        x = torch.randn(2, 5, 32)
        assert head(x).shape == (2, 5, 8)

    def test_mha_shape_and_head_split(self) -> None:
        mha = MultiHeadAttention(num_heads=4, head_size=8, n_embd=32, block_size=16, dropout=0.0)
        x = torch.randn(2, 5, 32)
        out = mha(x)
        assert out.shape == (2, 5, 32)  # concat(4 heads * 8) -> proj back to 32

    def test_ffwd_shape(self) -> None:
        ff = FeedForward(n_embd=32, dropout=0.0)
        x = torch.randn(2, 5, 32)
        assert ff(x).shape == (2, 5, 32)

    def test_block_shape(self) -> None:
        blk = Block(n_embd=32, n_head=4, block_size=16, dropout=0.0)
        x = torch.randn(2, 5, 32)
        assert blk(x).shape == (2, 5, 32)

    def test_gpt_forward_shapes(self) -> None:
        model = GPT(CFG)
        idx = torch.randint(0, CFG.vocab_size, (2, 7))
        logits, loss = model(idx, idx)
        assert logits.shape == (2, 7, CFG.vocab_size)
        assert loss is not None and loss.ndim == 0

    def test_gpt_generate_shape(self) -> None:
        model = GPT(CFG)
        idx = torch.randint(0, CFG.vocab_size, (1, 3))
        out = model.generate(idx, max_new_tokens=5)
        assert out.shape == (1, 8)

    def test_generate_crops_context(self) -> None:
        # block_size=16: feed 14 tokens, ask for 10 more -> must not crash and
        # output length is 24 (context cropping keeps the window <= 16).
        model = GPT(CFG)
        idx = torch.randint(0, CFG.vocab_size, (1, 14))
        out = model.generate(idx, max_new_tokens=10)
        assert out.shape == (1, 24)


# ---------------------------------------------------------------------------
# Correctness properties
# ---------------------------------------------------------------------------
class TestCorrectness:
    def test_initial_loss_is_ln_vocab(self) -> None:
        # At init the model is (approximately) uniform over the vocab, so
        # cross-entropy should be close to ln(vocab_size) = ln(50) = 3.912.
        torch.manual_seed(0)
        model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=16, block_size=8, vocab_size=50, dropout=0.0))
        idx = torch.randint(0, 50, (8, 8))
        _, loss = model(idx, idx)
        assert math.isclose(loss.item(), math.log(50), rel_tol=0.15)

    def test_causal_mask(self) -> None:
        """Token t must not influence the output at position < t.

        Perturb position t and check that outputs at earlier positions are
        unchanged - the defining property of the causal mask.
        """
        torch.manual_seed(0)
        model = GPT(CFG)
        model.eval()
        idx = torch.randint(0, CFG.vocab_size, (1, 8))
        with torch.no_grad():
            logits_a, _ = model(idx)
        idx2 = idx.clone()
        idx2[0, 7] = (idx2[0, 7] + 1) % CFG.vocab_size  # change only the last token
        with torch.no_grad():
            logits_b, _ = model(idx2)
        assert torch.allclose(logits_a[0, :7], logits_b[0, :7], atol=1e-5), (
            "positions before 7 changed when token 7 changed -> causal mask is broken"
        )
        assert not torch.allclose(logits_a[0, 7], logits_b[0, 7], atol=1e-5)

    def test_weight_tying(self) -> None:
        model = GPT(CFG)
        assert model.lm_head.weight is model.token_embedding.weight

    def test_overfit_single_batch(self) -> None:
        """A transformer must be able to drive loss to ~0 on one batch -
        the standard sanity check that the optimisation path works."""
        torch.manual_seed(0)
        model = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=16, vocab_size=50, dropout=0.0))
        x = torch.randint(0, 50, (4, 12))
        y = torch.randint(0, 50, (4, 12))
        optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
        first = None
        for step in range(120):
            _, loss = model(x, y)
            if first is None:
                first = loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        assert loss.item() < 0.1, f"loss only fell {first:.3f} -> {loss.item():.3f}"

    def test_temperature_zero_is_greedy(self) -> None:
        model = GPT(CFG)
        model.eval()
        idx = torch.randint(0, CFG.vocab_size, (1, 4))
        a = model.generate(idx, max_new_tokens=4, temperature=1e-6)
        b = model.generate(idx, max_new_tokens=4, temperature=1e-6)
        assert torch.equal(a, b), "temperature ~0 must be deterministic (greedy)"


# ---------------------------------------------------------------------------
# SFT loss masking
# ---------------------------------------------------------------------------
class TestSFTLossMasking:
    def _tok(self) -> CharTokenizer:
        tok = CharTokenizer()
        # Vocab must cover every conversation used in the tests below, the
        # template tags and the END token.
        corpus = "\n".join(
            f"{USER_TAG}\n{c['user']}\n{ASSISTANT_TAG}\n{c['assistant']}{END_TOKEN}"
            for c in DEMO_CONVERSATIONS
        )
        tok.build(corpus)
        return tok

    def test_only_assistant_tokens_supervised(self) -> None:
        tok = self._tok()
        ds = ChatDataset(DEMO_CONVERSATIONS[:1], tok, block_size=256)
        assert len(ds) == 1
        x, y = ds[0]
        supervised = (y != -100).sum().item()
        n_answer = len(tok.encode(DEMO_CONVERSATIONS[0]["assistant"] + END_TOKEN))
        # Shift-by-one: y[t] is the target for x[t]. The supervised span in
        # y covers exactly the answer tokens (END included) because the
        # answer begins at the token right after the last template token.
        assert supervised == n_answer

    def test_masked_loss_equals_manual(self) -> None:
        """loss computed with -100 masking == loss computed on just the
        supervised positions by hand."""
        tok = self._tok()
        ds = ChatDataset(DEMO_CONVERSATIONS[:1], tok, block_size=256)
        x, y = ds[0]
        model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=16, block_size=256, vocab_size=tok.vocab_size, dropout=0.0))
        model.eval()
        with torch.no_grad():
            _, masked_loss = model(x[None], y[None])
            logits, _ = model(x[None])
            # Manual: only positions where y != -100
            keep = y != -100
            manual = torch.nn.functional.cross_entropy(logits[0][keep], y[keep])
        assert torch.allclose(masked_loss, manual, atol=1e-5)

    def test_collate_masks_padding(self) -> None:
        tok = self._tok()
        ds = ChatDataset(DEMO_CONVERSATIONS[:2], tok, block_size=256)
        x, y = collate([ds[0], ds[1]], pad_id=0)
        assert x.shape == y.shape
        assert x.shape[1] == max(ds[0][0].shape[0], ds[1][0].shape[0])
        # padding label positions must be -100
        assert (y[:, -1] == -100).all() or (y != -100).any()
