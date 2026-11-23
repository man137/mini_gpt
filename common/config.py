"""Shared configuration, seeding and device helpers.

Every runnable script in this repo imports from here, so there is exactly one
place where defaults live. Values can be overridden with environment
variables (see :class:`Config` field defaults) or by passing explicit
arguments to the scripts.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths (repo layout is fixed; ROOT is the parent of the `common` package)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "scratch_gpt" / "data"
CHECKPOINTS_DIR = REPO_ROOT / "scratch_gpt" / "checkpoints"
LOGS_DIR = REPO_ROOT / "scratch_gpt" / "logs"


def seed_everything(seed: int = 1337) -> None:
    """Seed python, numpy and torch so runs are reproducible.

    Note: full determinism also needs `torch.use_deterministic_algorithms(True)`
    and a fixed `CUBLAS_WORKSPACE_CONFIG` on CUDA, which we do not force because
    it slows training and some CUDA kernels have no deterministic variant.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS exists on Apple Silicon; guard with `is_available` for portability.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Mixed-precision dtype for `torch.autocast`.

    bfloat16 has the same exponent range as fp32 so it never overflows and
    needs no loss scaling; on very old GPUs (compute capability < 8.0, e.g.
    T4/Colab free tier) bf16 is not supported so we fall back to fp16 and
    must use a GradScaler to avoid under/overflow in gradients.
    """
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:  # Ampere and newer (A100, RTX 30xx/40xx, L4, ...)
            return torch.bfloat16
        return torch.float16
    # torch.autocast on CPU supports bfloat16 (used for the from-scratch model).
    return torch.bfloat16


@dataclass
class Config:
    """Single source of truth for the from-scratch GPT hyper-parameters.

    Environment variables (all optional): GPT_N_LAYER, GPT_N_HEAD, GPT_N_EMBD,
    GPT_BLOCK_SIZE, GPT_DROPOUT, SEED.
    """

    # --- model -------------------------------------------------------------
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    # Maximum sequence length the model was designed for; longer prompts are
    # cropped to the last `block_size` tokens in `GPT.generate`.
    block_size: int = 256
    dropout: float = 0.2

    # --- data / tokenizer ----------------------------------------------------
    dataset_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dataset_path: Path = field(default_factory=lambda: DATA_DIR / "input.txt")

    # --- optimisation --------------------------------------------------------
    batch_size: int = 64
    # ~10.6M non-embedding params with the default config (see README).
    learning_rate: float = 1e-3
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_iters: int = 5000
    warmup_iters: int = 200
    lr_decay_iters: int = 5000
    min_lr: float = 1e-4
    eval_interval: int = 250
    eval_iters: int = 20
    # Only checkpoint when validation loss improves (early-stopping style).
    checkpoint_best: bool = True

    # --- reproducibility -----------------------------------------------------
    seed: int = 1337

    def __post_init__(self) -> None:
        # Allow env-var overrides without editing code (handy on Colab).
        env_map = {
            "GPT_N_LAYER": ("n_layer", int),
            "GPT_N_HEAD": ("n_head", int),
            "GPT_N_EMBD": ("n_embd", int),
            "GPT_BLOCK_SIZE": ("block_size", int),
            "GPT_DROPOUT": ("dropout", float),
            "SEED": ("seed", int),
        }
        for env, (attr, cast) in env_map.items():
            if os.environ.get(env):
                object.__setattr__(self, attr, cast(os.environ[env]))
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}) "
                "so each head gets an equal slice of the embedding."
            )


DEFAULT_CONFIG = Config()
