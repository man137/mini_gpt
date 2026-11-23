"""Train the from-scratch GPT on Tiny Shakespeare (or any plain-text file).

Implements: 90/10 train/val split, AdamW, gradient clipping, linear warmup +
cosine decay, periodic estimated val loss, bf16/fp16 autocast (with
GradScaler for fp16), best-val checkpointing and CSV logging.

Usage:
    python scratch_gpt/train.py                    # defaults (5k iters)
    python scratch_gpt/train.py --max_iters 200    # quick smoke run
    python scratch_gpt/train.py --cpu              # force CPU (small config)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import (  # noqa: E402
    CHECKPOINTS_DIR,
    DATA_DIR,
    DEFAULT_CONFIG,
    LOGS_DIR,
    Config,
    get_device,
    seed_everything,
)
from scratch_gpt.model import GPT, GPTConfig  # noqa: E402
from scratch_gpt.tokenizer import CharTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download_dataset(cfg: Config) -> Path:
    """Download Tiny Shakespeare (1.1 MB) if not already present."""
    cfg.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.dataset_path.exists():
        print(f"downloading {cfg.dataset_url} -> {cfg.dataset_path}")
        urllib.request.urlretrieve(cfg.dataset_url, cfg.dataset_path)
    return cfg.dataset_path


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) pairs.

    For each sample we pick a random offset `i`; x = data[i:i+T] and
    y = data[i+1:i+1+T] is the same window shifted by one token, which is
    exactly next-token prediction.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    """Average loss over `eval_iters` random batches (train and val splits)."""
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data[split], cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup then cosine decay to min_lr
# ---------------------------------------------------------------------------
def get_lr(it: int, cfg: Config) -> float:
    if it < cfg.warmup_iters:  # linear ramp from ~0 to learning_rate
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:  # past the schedule: stay at min_lr
        return cfg.min_lr
    # cosine decay between learning_rate and min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * coeff


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_iters", type=int, default=None, help="override cfg.max_iters")
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="force CPU with a tiny config")
    p.add_argument("--smoke", action="store_true", help="tiny config for a quick run")
    args = p.parse_args()

    cfg = Config()
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.n_layer is not None:
        cfg.n_layer = args.n_layer
    if args.n_head is not None:
        cfg.n_head = args.n_head
    if args.n_embd is not None:
        cfg.n_embd = args.n_embd
    if args.cpu or args.smoke:
        # Tiny config so the whole loop runs in <1 min on a laptop CPU.
        cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size = 2, 2, 64, 64
        cfg.batch_size = 8
    if args.smoke:
        cfg.max_iters, cfg.eval_interval, cfg.warmup_iters = 40, 20, 5

    seed_everything(cfg.seed)
    device = get_device()
    print(f"device={device}  config={cfg}")

    # --- data -------------------------------------------------------------
    download_dataset(cfg)
    text = cfg.dataset_path.read_text(encoding="utf-8")
    tok = CharTokenizer()
    tok.build(text)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(DATA_DIR / "char_tokenizer.json")
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))  # 90/10 split (first 90% train, last 10% val)
    data = {"train": ids[:n], "val": ids[n:]}
    print(
        f"dataset: {len(ids)} chars, vocab={tok.vocab_size}, "
        f"train={len(data['train'])}, val={len(data['val'])}"
    )

    # --- model --------------------------------------------------------------
    model = GPT(
        GPTConfig(
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            block_size=cfg.block_size,
            vocab_size=tok.vocab_size,
            dropout=cfg.dropout,
        )
    ).to(device)
    print(f"model params: {model.num_params():,} (non-embedding: {model.num_params(non_embedding=True):,})")

    # --- optimiser ------------------------------------------------------------
    # AdamW with weight decay 0.1 on 2D params (matrices), none on biases /
    # norms (standard GPT-2 recipe: decay only weights that interact linearly).
    decay, no_decay = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )

    # --- mixed precision ---------------------------------------------------
    from common.config import autocast_dtype

    amp_dtype = autocast_dtype(device)
    use_scaler = amp_dtype == torch.float16  # fp16 needs loss scaling
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"autocast dtype: {amp_dtype}")

    # --- logging / checkpointing ---------------------------------------------
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOGS_DIR / "train_log.csv"
    log_rows: list[dict] = []
    best_val = float("inf")

    model.train()
    for it in range(cfg.max_iters):
        lr = get_lr(it, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, data, cfg, device)
            print(
                f"step {it:5d} | lr {lr:.2e} | train {losses['train']:.4f} "
                f"| val {losses['val']:.4f}"
            )
            log_rows.append(
                {"step": it, "lr": lr, "train_loss": losses["train"], "val_loss": losses["val"]}
            )
            if cfg.checkpoint_best and losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": vars(model.config)},
                    CHECKPOINTS_DIR / "best.pt",
                )

        x, y = get_batch(data["train"], cfg.block_size, cfg.batch_size, device)
        # torch.autocast is a context manager that runs the forward in a
        # lower-precision dtype (matmuls in fp16/bf16, accumulations fp32).
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Unscale before clipping so the clip threshold applies to real
        # gradient magnitudes, not scaled ones.
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optim)
        scaler.update()

    # final save + CSV flush
    torch.save(
        {"model": model.state_dict(), "config": vars(model.config)},
        CHECKPOINTS_DIR / "last.pt",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lr", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(log_rows)
    print(f"done. best val loss {best_val:.4f}; checkpoint: {CHECKPOINTS_DIR / 'best.pt'}")
    print(f"log: {csv_path} ({len(log_rows)} eval rows)")
    print("next: python scratch_gpt/plot_loss.py  &&  python scratch_gpt/generate.py --prompt 'ROMEO:'")


if __name__ == "__main__":
    main()
