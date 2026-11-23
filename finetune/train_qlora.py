"""QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on the public-health Q&A data.

Runs on a free Colab T4 (16 GB VRAM). Read the VRAM table below before
changing batch sizes.

VRAM estimate (Qwen2.5-1.5B-Instruct, NF4, LoRA r=16, seq 1024, approximate):
  | component                          | ~VRAM   |
  |------------------------------------|---------|
  | base weights in 4-bit NF4          | ~1.2 GB |
  | LoRA adapters (fp16, trainable)    | ~0.05 GB|
  | activations w/ grad checkpointing  | ~2-4 GB |
  | optimizer state (adamw, 2 moments) | ~0.1 GB |
  | CUDA context + fragmentation       | ~1-2 GB |
  | TOTAL (bs=1, ga=8)                 | ~5-7 GB |
  So a T4 has comfortable headroom; the numbers above are approximate and
  the student should record the real `torch.cuda.max_memory_allocated()`.

IF OOM, change these IN ORDER (each is safer than the next):
  1. per_device_train_batch_size  1 -> already 1? go to 2
  2. gradient_accumulation_steps  8 -> 16 (keeps effective batch the same)
  3. max_seq_length               1024 -> 512
  4. MODEL_NAME -> "Qwen/Qwen2.5-0.5B-Instruct" (config switch below)
  5. lora r                       16 -> 8, lora_alpha 32 -> 16

Usage (on Colab or any 16 GB GPU):
    python finetune/train_qlora.py
    python finetune/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Config switch: the default fits a 16 GB T4 with room to spare.
#  - 0.5B : CPU-poor / repeated OOM situations, or quick smoke tests
#  - 1.5B : default - best quality/VRAM trade-off on free Colab
#  - 3B   : needs a 24 GB GPU (T4 will OOM with long sequences)
MODEL_CHOICES = {
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
    "base": "Qwen/Qwen2.5-1.5B-Instruct",
    "large": "Qwen/Qwen2.5-3B-Instruct",
}


class AssistantOnlyCollator:
    """Tokenize a messages-list example and MASK everything except the final
    assistant answer with -100 (same loss-masking idea as scratch_gpt/sft.py).

    Why mask? We only want gradients from tokens the assistant should
    produce; training on user text teaches the model to parrot questions.
    """

    def __init__(self, tokenizer, max_seq_length: int = 1024) -> None:
        self.tok = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[dict]) -> dict:
        input_ids_list, labels_list = [], []
        for ex in examples:
            msgs = ex["messages"]
            # Render the full conversation (system + user + assistant) once,
            # and the prompt-only part (everything before the last assistant
            # message) once; the token difference is the assistant span.
            full = self.tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False
            )
            prompt_only = self.tok.apply_chat_template(
                msgs[:-1], tokenize=True, add_generation_prompt=True
            )
            # VERSION-SENSITIVE: apply_chat_template returns a list of ids
            # when tokenize=True. If your transformers version returns a
            # BatchEncoding dict instead, use `.["input_ids"]`.
            if hasattr(full, "input_ids"):
                full = full["input_ids"]
            if hasattr(prompt_only, "input_ids"):
                prompt_only = prompt_only["input_ids"]
            input_ids = torch.tensor(full[: self.max_seq_length], dtype=torch.long)
            labels = input_ids.clone()
            n_prompt = min(len(prompt_only), len(input_ids))
            labels[:n_prompt] = -100  # mask system + user + role scaffolding
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        maxlen = max(t.shape[0] for t in input_ids_list)
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        input_ids = torch.full((len(input_ids_list), maxlen), pad_id, dtype=torch.long)
        labels = torch.full((len(labels_list), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(input_ids_list), maxlen), dtype=torch.long)
        for i, (ii, ll) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[i, : ii.shape[0]] = ii
            labels[i, : ll.shape[0]] = ll
            attn[i, : ii.shape[0]] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=MODEL_CHOICES["base"],
                   help=f"model id or preset name: {list(MODEL_CHOICES.values()) + list(MODEL_CHOICES.keys())}")
    p.add_argument("--train_file", type=Path, default=DATA_DIR / "train.jsonl")
    p.add_argument("--val_file", type=Path, default=DATA_DIR / "val.jsonl")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR / "adapter")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA LRs are ~10-100x pretraining LRs")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to verify the pipeline wiring (CPU/GPU, ~minutes)")
    args = p.parse_args()

    model_name = MODEL_CHOICES.get(args.model, args.model)
    if args.smoke:
        model_name = MODEL_CHOICES["small"]
        args.epochs, args.eval_steps, args.max_seq_length = 1.0, 5, 512

    if not args.train_file.exists():
        raise SystemExit(f"{args.train_file} missing - run `python finetune/prepare_data.py` first")

    # Heavy imports live here so --help stays fast.
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    # ---- tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4-bit quantised base -----------------------------------------------
    # bnb_4bit_quant_type="nf4": NormalFloat4, a 4-bit dtype tuned for the
    #   normal distribution of LLM weights - better accuracy than plain int4.
    # bnb_4bit_use_double_quant=True: quantise the quantisation constants too,
    #   saving ~0.4 GB per model at negligible quality cost.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    # Required for QLoRA: fixes LayerNorms to fp32, enables input gradients
    # for the frozen quantised weights, disables its own dropout etc.
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA ---------------------------------------------------------------
    # r=16, alpha=32 -> effective scale alpha/r = 2 (a common sweet spot).
    # Target only the attention projections; adding MLP layers (gate/up/down)
    # helps quality but costs more VRAM and time.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # typically ~0.5-2% of the base model

    # ---- data -----------------------------------------------------------------
    def load_jsonl(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    train_ds = Dataset.from_list(load_jsonl(args.train_file))
    val_ds = Dataset.from_list(load_jsonl(args.val_file)) if args.val_file.exists() else None

    # ---- training arguments ----------------------------------------------------
    # VERSION-SENSITIVE: `gradient_checkpointing_kwargs={"use_reentrant": False}`
    # is needed with recent torch+peft (the reentrant implementation breaks
    # with LoRA). On very old transformers versions, remove the kwarg and set
    # gradient_checkpointing=True alone.
    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",   # paged optimizer: spills to CPU RAM on spikes
        report_to=[],
        seed=42,
    )

    collator = AssistantOnlyCollator(tokenizer, args.max_seq_length)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    # ---- save the adapter only (a few MB) ------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)      # adapter weights
    tokenizer.save_pretrained(args.out_dir)  # tokenizer next to it
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM: {peak:.2f} GB")
    print(f"adapter saved to {args.out_dir}")
    print("NEXT: python finetune/merge_and_test.py  (needs ~4-6 GB GPU or a CPU with patience)")


if __name__ == "__main__":
    main()
