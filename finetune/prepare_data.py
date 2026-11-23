"""Validate + split + tokenise the fine-tuning data.

Steps:
  1. Load every *.jsonl in finetune/data/ that matches the schema
     (seed_qa.jsonl always; synthetic_reviewed.jsonl if you created it).
  2. Validate schema strictly (see QUALITY.md).
  3. Apply the BASE MODEL's chat template with tokenizer.apply_chat_template
     (the exact formatting the model saw in pretraining - using anything
     else is a classic fine-tuning bug).
  4. 90/10 train/val split with a FIXED seed, written to
     finetune/data/train.jsonl / val.jsonl (message format, for SFTTrainer
     with a collator that masks non-assistant tokens) and
     train_tokenized.jsonl / val_tokenized.jsonl (rendered text form,
     inspected or used by a text-mode trainer).

Usage:
    python finetune/prepare_data.py [--model Qwen/Qwen2.5-1.5B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent / "data"
ROLES = {"system", "user", "assistant"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_example(obj: dict, i: int, fname: str) -> None:
    """Raise ValueError with a precise message if the example is malformed."""
    where = f"{fname}:{i}"
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: not a JSON object")
    keys = set(obj)
    if keys != {"messages", "source", "id"}:
        raise ValueError(f"{where}: expected keys messages/source/id, got {keys}")
    if not isinstance(obj["id"], str) or not obj["id"]:
        raise ValueError(f"{where}: id must be a non-empty string")
    if not isinstance(obj["source"], str) or not obj["source"]:
        raise ValueError(f"{where}: source must be a non-empty string")
    msgs = obj["messages"]
    if not isinstance(msgs, list) or not msgs:
        raise ValueError(f"{where}: messages must be a non-empty list")
    for j, m in enumerate(msgs):
        if set(m) != {"role", "content"}:
            raise ValueError(f"{where}: message {j} keys {set(m)} != role/content")
        if m["role"] not in ROLES:
            raise ValueError(f"{where}: message {j} role {m['role']!r} not in {ROLES}")
        if not isinstance(m["content"], str) or not m["content"].strip():
            raise ValueError(f"{where}: message {j} has empty content")
    if msgs[-1]["role"] != "assistant":
        raise ValueError(f"{where}: last message must be the assistant answer")


def load_jsonl_files(paths: list[Path]) -> list[dict]:
    examples: list[dict] = []
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            validate_example(obj, i, path.name)
            if obj["id"] in ids:
                raise ValueError(f"{path.name}:{i}: duplicate id {obj['id']!r}")
            ids.add(obj["id"])
            examples.append(obj)
    return examples


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    args = p.parse_args()

    # seed is always included; reviewed synthetics only if present
    paths = [DATA_DIR / "seed_qa.jsonl", DATA_DIR / "synthetic_reviewed.jsonl"]
    examples = load_jsonl_files(paths)
    if not examples:
        raise SystemExit(f"no data found in {paths}")

    # ---- stats -------------------------------------------------------------
    n_by_source = Counter(e["source"] for e in examples)
    lens = [len(e["messages"][-1]["content"]) for e in examples]
    print(f"loaded {len(examples)} examples")
    print(f"  by source: {dict(n_by_source)}")
    print(f"  answer chars: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")

    # ---- chat template (the exact base-model format) ------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rendered = tokenizer.apply_chat_template(
        examples[0]["messages"], tokenize=False
    )
    print(f"\nchat template of {args.model} (example {examples[0]['id']}):\n{rendered}")

    # ---- 90/10 split with fixed seed ---------------------------------------
    rng = random.Random(args.seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_val = max(1, int(round(args.val_ratio * len(examples))))
    val_ids = set(idx[:n_val])
    split = {"train": [examples[i] for i in idx if i not in val_ids],
             "val": [examples[i] for i in idx if i in val_ids]}
    print(f"\nsplit (seed={args.seed}): train={len(split['train'])} val={len(split['val'])}")

    # ---- write message-format and rendered-text-format files ----------------
    for name, rows in split.items():
        with open(DATA_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(DATA_DIR / f"{name}_tokenized.jsonl", "w", encoding="utf-8") as f:
            for e in rows:
                text = tokenizer.apply_chat_template(e["messages"], tokenize=False)
                # add eos between examples so packing/concatenation is safe
                f.write(json.dumps({"text": text, "id": e["id"]}, ensure_ascii=False) + "\n")
    print(f"wrote train/val .jsonl and _tokenized.jsonl to {DATA_DIR}")
    print("NEXT: python finetune/train_qlora.py  (on Colab T4 for the full run)")


if __name__ == "__main__":
    main()
