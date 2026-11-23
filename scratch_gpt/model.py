"""A small GPT implemented from scratch in pure PyTorch.

Built strictly bottom-up so every interview question has a one-file answer:
  Head -> MultiHeadAttention -> FeedForward -> Block -> GPT

Design notes (the "why" for each non-obvious line is commented inline):
  - pre-LayerNorm blocks with residual connections (gradient highway)
  - causal mask registered as a buffer (moves with .to(device), not saved in
    the state_dict because persistent=False)
  - attention scores scaled by 1/sqrt(head_size) (variance stabilisation)
  - weight tying between the token embedding and lm_head (parameter sharing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    vocab_size: int = 65  # set by the caller after the tokenizer is built
    dropout: float = 0.2


class Head(nn.Module):
    """One self-attention head: y = softmax(QK^T / sqrt(d)) V."""

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask: position i may attend to positions <= i.
        # persistent=False -> not saved in checkpoints (it is a constant) but
        # still moved by .to(device) because it is a registered buffer.
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)), persistent=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Raw dot products have variance ~ C; without the 1/sqrt(d) scaling the
        # softmax saturates and gradients vanish. (Scaled dot-product
        # attention, Vaswani et al. 2017.)
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))  # (B, T, T)
        # Causal mask: future positions get -inf, so after softmax they have
        # exactly zero attention weight -> token t cannot see token t+1.
        wei = wei.masked_fill(self.tril[: T, : T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # rows sum to 1 over allowed positions
        wei = self.dropout(wei)
        return wei @ self.value(x)  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel, concatenated, then projected."""

    def __init__(
        self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        # nn.ModuleList (not a plain list) so parameters are registered.
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        # Output projection: mixes information across heads after concat.
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, non-linearity, project back."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        # The 4x expansion follows the original Transformer paper; it gives
        # the MLP enough capacity to do per-token "computation" between the
        # attention layers that do per-position "communication".
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # smooth ReLU variant, used by GPT-2 and later
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-LayerNorm + residual around each sub-layer.

    Pre-LN (x = x + Sublayer(LN(x))) is used by GPT-2 and virtually all
    modern LLMs because it keeps the residual stream clean: gradients can
    flow through the identity path untouched, which makes deep stacks stable
    without a warmup-sensitive post-LN normalisation.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))   # residual: communication
        x = x + self.ffwd(self.ln2(x))  # residual: computation
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Learned (not sinusoidal) positional embeddings, as in GPT-2.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(config.n_embd, config.n_head, config.block_size, config.dropout)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)  # final norm before the head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # WEIGHT TYING: the output projection reuses the input embedding
        # matrix. This shares parameters, drastically reduces params for small
        # vocabs, and is known to improve perplexity (Press & Wolf 2017).
        self.lm_head.weight = self.token_embedding.weight
        # Scaled init (GPT-2 paper): residual paths add variance, so shrink
        # the initialised weights of the projections by 1/sqrt(2*n_layer).
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.weight is self.lm_head.weight:
            torch.nn.init.normal_(
                self.token_embedding.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) int64 token ids. targets: (B, T) int64 or None.

        Returns (logits, loss). Loss is None when targets is None. With
        targets given, the shift-by-one happens here: logits at position t
        predict token t+1, which is exactly next-token prediction.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(T, device=idx.device)  # (T,)
        tok_emb = self.token_embedding(idx)        # (B, T, C)
        pos_emb = self.position_embedding(pos)[None, :, :]  # (1, T, C) broadcast
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten (B, T) -> (B*T) because F.cross_entropy wants
            # (N, C) predictions and (N,) targets.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                ignore_index=-100,  # SFT loss masking uses -100 for "don't train"
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling loop.

        temperature: divides logits (<1 sharpens, >1 flattens).
        top_k: keep only the k most likely tokens.
        top_p: nucleus sampling - keep the smallest set of tokens whose
               cumulative probability >= p.
        Context cropping: the model can only condition on `block_size`
        tokens, so we keep the last block_size tokens when the sequence
        grows beyond that.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)            # (B, T, vocab)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # (B, vocab)
            if top_k is not None:
                # Keep only top-k logits; everything else becomes -inf.
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                # Mask tokens once cumulative probability exceeds top_p,
                # but always keep the first token (it may already exceed p).
                mask = cum - probs > top_p
                sorted_logits[mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def resize_vocab(self, new_vocab_size: int) -> None:
        """Grow (or shrink) the vocabulary after pretraining.

        Used by sft.py: the chat template introduces characters that the
        pretraining corpus did not contain, so new rows are appended to the
        (tied) embedding matrix. New rows use the same init distribution.
        """
        # Compare against the ACTUAL matrix rows, not self.config.vocab_size:
        # callers may have already synced config.vocab_size before calling us,
        # in which case the two are equal even though the matrix is smaller.
        old = self.token_embedding.weight.data
        if new_vocab_size == old.shape[0]:
            self.config.vocab_size = new_vocab_size
            return
        new = torch.empty(
            new_vocab_size, self.config.n_embd, dtype=old.dtype, device=old.device
        )
        torch.nn.init.normal_(new, mean=0.0, std=0.02)
        keep = min(old.shape[0], new_vocab_size)
        new[:keep] = old[:keep]
        # Because of weight tying, assigning to token_embedding.weight also
        # updates lm_head.weight (same tensor object).
        self.token_embedding.weight = torch.nn.Parameter(new)
        self.lm_head.weight = self.token_embedding.weight
        self.config.vocab_size = new_vocab_size

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Weight tying means lm_head shares the token embedding matrix,
            # so subtracting the tied matrix once still counts it once.
            n -= self.token_embedding.weight.numel()
        return n
