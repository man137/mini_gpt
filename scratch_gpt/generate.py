"""Sample text from a trained from-scratch GPT checkpoint.

Usage:
    python scratch_gpt/generate.py --prompt "ROMEO:" --max_new_tokens 300
    python scratch_gpt/generate.py --temperature 0.8 --top_k 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import CHECKPOINTS_DIR, DATA_DIR, seed_everything  # noqa: E402
from scratch_gpt.model import GPT, GPTConfig  # noqa: E402
from scratch_gpt.tokenizer import CharTokenizer  # noqa: E402


def load_model(checkpoint: Path, device: torch.device) -> tuple[GPT, CharTokenizer]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # sft.py extends the vocab with the chat-template characters and saves a
    # matching tokenizer; use it when sampling from an sft checkpoint.
    if "sft" in checkpoint.stem.lower() and (DATA_DIR / "char_tokenizer_sft.json").exists():
        tok = CharTokenizer.load(DATA_DIR / "char_tokenizer_sft.json")
    else:
        tok = CharTokenizer.load(DATA_DIR / "char_tokenizer.json")
    assert tok.vocab_size == cfg.vocab_size, (
        f"tokenizer vocab {tok.vocab_size} != model vocab {cfg.vocab_size} - "
        "retrain or regenerate the tokenizer"
    )
    return model, tok


def main() -> None:
    p = argparse.ArgumentParser(description="sample from the from-scratch GPT")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_samples", type=int, default=1)
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"checkpoint {args.checkpoint} not found - run `python scratch_gpt/train.py` first"
        )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} ({model.num_params():,} params) on {device}\n")

    # Unknown characters in the prompt would crash encode(); be forgiving.
    prompt = "".join(ch for ch in args.prompt if ch in tok.stoi)
    if not prompt:
        prompt = "\n"
    ctx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        out = model.generate(
            ctx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        text = tok.decode(out[0].tolist())
        print(f"===== sample {i + 1} =====")
        print(text)
        print()


if __name__ == "__main__":
    main()
