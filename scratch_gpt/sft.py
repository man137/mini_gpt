"""Supervised fine-tuning (SFT) of the from-scratch GPT on tiny chat data.

WHAT THIS IS: a miniature version of what the QLoRA pipeline does to Qwen -
teaching a pretrained LM a *format* (chat turns) instead of general knowledge.

HONEST EXPECTATION SETTING: this model has ~10M non-embedding parameters and
was pretrained on 1 MB of Shakespeare. Fine-tuning it on a handful of public-
health chats will teach it the *shape* of a conversation (user line, assistant
line, <|end|>) but it will NOT learn reliable facts. That is exactly the
point: it demonstrates loss masking, chat formatting and a low SFT learning
rate - concepts that transfer 1:1 to the real pipeline.

Chat template (kept as plain text so it is explicit):
    <|user|> question <|end|>
    <|assistant|> answer <|end|>

LOSS MASKING: labels are -100 on every token that is not part of an assistant
answer (system/user text, role tags, delimiters). F.cross_entropy ignores
-100 via ignore_index, so gradients flow only from the tokens the assistant
should have produced.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import CHECKPOINTS_DIR, DATA_DIR, seed_everything  # noqa: E402
from scratch_gpt.model import GPT, GPTConfig  # noqa: E402
from scratch_gpt.tokenizer import END_TOKEN, CharTokenizer  # noqa: E402

# 10 hand-written demo examples (public-health FAQ style, deliberately short
# because the char-level model has a 256-token context window).
DEMO_CONVERSATIONS = [
    {"user": "What is the flu?", "assistant": "The flu is a contagious respiratory illness caused by influenza viruses."},
    {"user": "How can I prevent the flu?", "assistant": "Wash hands often, cover coughs, and get the yearly flu vaccine."},
    {"user": "Should I stay home when sick?", "assistant": "Yes. Stay home to rest and to avoid spreading illness to others."},
    {"user": "Is water important for health?", "assistant": "Yes. Drinking water daily supports digestion, circulation and temperature control."},
    {"user": "What is exercise good for?", "assistant": "Regular exercise strengthens the heart, muscles and mood."},
    {"user": "Why sleep matters?", "assistant": "Sleep restores the body and supports memory, mood and immune function."},
    {"user": "What are vitamins?", "assistant": "Vitamins are nutrients the body needs in small amounts to work properly."},
    {"user": "How much water a day?", "assistant": "About eight glasses a day is common advice, needs vary by person."},
    {"user": "Is smoking bad?", "assistant": "Yes. Smoking harms the lungs and heart and raises cancer risk."},
    {"user": "What is a fever?", "assistant": "A fever is a rise in body temperature, often a sign of infection."},
]

# Special role markers are added to the vocab as ordinary characters of the
# CharTokenizer are single chars; instead we keep them as multi-char strings
# that encode() will handle character by character (they are still part of
# the text, and loss masking is done at the character level).
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"


class ChatDataset(Dataset):
    """Turns (user, assistant) pairs into (x, y) training examples.

    Each item is a full rendered conversation; positions belonging to the
    assistant answer keep their token id in `y`, everything else is set to
    -100 (the cross_entropy ignore_index) so it contributes NO loss.
    """

    def __init__(self, conversations: list[dict], tokenizer: CharTokenizer, block_size: int) -> None:
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for conv in conversations:
            rendered = (
                f"{USER_TAG}\n{conv['user']}\n{ASSISTANT_TAG}\n{conv['assistant']}{END_TOKEN}"
            )
            ids = torch.tensor(tokenizer.encode(rendered), dtype=torch.long)
            if len(ids) > block_size:
                continue  # skip too-long conversations (char tokenizer!)

            # Build the mask: find the span of the assistant answer in the
            # rendered string, then map it through the tokenizer.
            prefix = f"{USER_TAG}\n{conv['user']}\n{ASSISTANT_TAG}\n"
            answer = conv["assistant"] + END_TOKEN
            n_prefix = len(tokenizer.encode(prefix))
            n_answer = len(tokenizer.encode(answer))
            labels = ids.clone()
            labels[:] = -100                        # default: ignore everything
            labels[n_prefix : n_prefix + n_answer] = ids[n_prefix : n_prefix + n_answer]

            # x predicts y with a one-token shift: x = ids[:-1], y = labels[1:]
            self.items.append((ids[:-1], labels[1:]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[i]


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length examples so a batch stacks into one tensor.

    Padding positions are also masked with -100 in the labels, so padding
    never contributes loss either.
    """
    max_len = max(x.shape[0] for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - x.shape[0]
        xs.append(torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)]))
        ys.append(torch.cat([y, torch.full((pad,), -100, dtype=torch.long)]))
    return torch.stack(xs), torch.stack(ys)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best.pt")
    p.add_argument("--data", type=Path, default=None, help="optional jsonl of {user, assistant} pairs")
    p.add_argument("--epochs", type=int, default=200,
                   help="200 epochs memorises the 10-example demo set (loss < 0.1); "
                        "the point is the FORMAT, not knowledge")
    p.add_argument("--lr", type=float, default=3e-5, help="SFT uses a much smaller LR than pretraining")
    p.add_argument("--demo", action="store_true", help="also print a demo generation at the end")
    p.add_argument("--out", type=Path, default=CHECKPOINTS_DIR / "sft.pt")
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint {args.checkpoint} not found - run train.py first")

    seed_everything(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    cfg.dropout = 0.0  # no dropout during small SFT: we *want* to memorise format
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    # Rebuild the char tokenizer from the pretraining artefacts and EXTEND it
    # with the chat-template characters ('<' '|' '>' '/') that plain
    # Shakespeare does not contain. Extension is safe: new ids are appended,
    # so all pretrained ids stay identical.
    tok = CharTokenizer.load(DATA_DIR / "char_tokenizer.json")
    extra = sorted(set(USER_TAG + ASSISTANT_TAG + END_TOKEN))
    next_id = tok.vocab_size
    for ch in extra:
        if ch not in tok.stoi:
            tok.stoi[ch] = next_id
            tok.itos[next_id] = ch
            next_id += 1
    # The model's vocab must grow to match (new rows are randomly initialised).
    cfg.vocab_size = tok.vocab_size
    model.config = cfg
    model.resize_vocab(tok.vocab_size)
    # Persist the EXTENDED tokenizer so generate.py can decode sft samples.
    tok.save(DATA_DIR / "char_tokenizer_sft.json")

    if args.data and args.data.exists():
        convs = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        convs = DEMO_CONVERSATIONS

    ds = ChatDataset(convs, tok, cfg.block_size)
    if not ds:
        raise SystemExit(
            "no conversations fit inside block_size - run pretraining with "
            "--block_size 256 (default) or shorten the SFT data"
        )
    print(f"SFT dataset: {len(ds)} examples from {len(convs)} conversations")
    x0, y0 = ds[0]
    supervised = (y0 != -100).sum().item()
    print(f"example 0: {x0.shape[0]} tokens, {supervised} supervised (loss-contributing) tokens")

    pad_id = 0
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for epoch in range(args.epochs):
        total, count = 0.0, 0
        for i in range(0, len(ds), 8):
            batch = [ds[j] for j in range(i, min(i + 8, len(ds)))]
            x, y = collate(batch, pad_id)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)  # -100 positions contribute nothing
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item()
            count += 1
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"epoch {epoch + 1:3d} | loss {total / max(1, count):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, args.out)
    print(f"saved {args.out}")

    if args.demo:
        model.eval()
        prompt = f"{USER_TAG}\nWhat is the flu?\n{ASSISTANT_TAG}\n"
        ctx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(ctx, max_new_tokens=80, temperature=0.7, top_k=20)
        print("\n--- demo generation (memorised-ish, this is a 10M model) ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
