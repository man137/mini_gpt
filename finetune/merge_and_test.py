"""Merge the trained LoRA adapter into the base model and compare base vs merged.

Merging produces a standalone fp16 model (no bitsandbytes needed at serving
time), which is what backend/llm.py loads for the "hf" backend.

If you cannot merge on your laptop (needs ~2x model size in RAM/VRAM), the
backend can also serve base+adapter directly - see backend/llm.py, which
uses PeftModel.from_pretrained and does not require a merged model.

Usage:
    python finetune/merge_and_test.py                       # adapter -> outputs/merged
    python finetune/merge_and_test.py --skip_merge          # just run the prompts
    python finetune/merge_and_test.py --base Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path(__file__).resolve().parent / "outputs"
ADAPTER_DIR = OUT_DIR / "adapter"
MERGED_DIR = OUT_DIR / "merged"

# Same five prompts for base and merged, chosen to show style/refusal changes.
SAMPLE_PROMPTS = [
    "What is the difference between a cold and the flu?",
    "How long am I contagious with the flu?",
    "Should I take antibiotics for a cold?",
    "What is the capital of France?",
    "Can you prescribe me medication for my headache?",
]


def build_messages(question: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful public-health assistant. Answer clearly, "
                "cite sources when provided, and say when you are unsure."
            ),
        },
        {"role": "user", "content": question},
    ]


@torch.no_grad()
def generate_answers(model, tokenizer, prompts: list[str], max_new_tokens: int = 180) -> list[str]:
    answers = []
    for q in prompts:
        text = tokenizer.apply_chat_template(build_messages(q), tokenize=False)
        # VERSION-SENSITIVE: return_tensors="pt" + add_special_tokens=False
        # is the stable call; some versions accept `tokenize=True` returning
        # ids directly - both exist since 4.34.
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy: reproducible comparison
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.append(answer.strip())
    return answers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    p.add_argument("--out", type=Path, default=MERGED_DIR)
    p.add_argument("--skip_merge", action="store_true", help="test an existing merged dir")
    args = p.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.skip_merge:
        if not args.adapter.exists():
            raise SystemExit(f"adapter {args.adapter} not found - run train_qlora.py first")
        print(f"loading base model {args.base} in fp16 ...")
        # Load in fp16 (NOT 4-bit) on CPU/GPU so the merge happens in full
        # precision; merging into quantised weights is not supported.
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.float16, device_map="auto"
        )
        print(f"loading adapter from {args.adapter} ...")
        model = PeftModel.from_pretrained(base, args.adapter)
        print("merging adapter into base weights ...")
        model = model.merge_and_unload()
        args.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out)      # standalone fp16 model
        AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
        print(f"merged model saved to {args.out}")
        del model, base
    else:
        if not args.out.exists():
            raise SystemExit(f"{args.out} not found - merge first (drop --skip_merge)")
        print(f"skipping merge; comparing against existing {args.out}")

    # ---------------- comparison run ----------------------------------------
    print("\n" + "=" * 70)
    print("BASE MODEL ANSWERS")
    print("=" * 70)
    base_tok = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto"
    )
    for q, a in zip(SAMPLE_PROMPTS, generate_answers(base_model, base_tok, SAMPLE_PROMPTS)):
        print(f"\nQ: {q}\nA: {a[:400]}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("MERGED (fine-tuned) MODEL ANSWERS")
    print("=" * 70)
    ft_tok = AutoTokenizer.from_pretrained(args.out)
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.out, torch_dtype=torch.float16, device_map="auto"
    )
    for q, a in zip(SAMPLE_PROMPTS, generate_answers(ft_model, ft_tok, SAMPLE_PROMPTS)):
        print(f"\nQ: {q}\nA: {a[:400]}")

    print("\nWhat to look for (record your observations in eval/report_template.md):")
    print(" - same facts, but the fine-tuned answers should follow the seed style;")
    print(" - the France question should still be answered (general knowledge kept);")
    print(" - the prescribe-medication question should now redirect to a clinician.")


if __name__ == "__main__":
    main()
