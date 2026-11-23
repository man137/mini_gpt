"""Optional: expand the 40 seed examples into a larger synthetic dataset.

This is deliberately OPTIONAL - the pipeline trains fine on the 40 seeds
(2 epochs over 40 examples already moves style and refusal behaviour).
Synthetic data multiplies what you have; it does not fix bad seeds.

Design (interview talking points):
  - variants are generated from seed + retrieved document context, so answers
    stay grounded in real text instead of free-running model imagination;
  - every generated example is DEDUPLICATED by a normalised-text hash;
  - obvious refusals and length outliers are FILTERED;
  - nothing enters training directly: a REVIEW markdown file is written and a
    human must move approved examples to `synthetic_reviewed.jsonl`.

Backends (choose with SYNTH_BACKEND):
  - "local"  : a local HF model (default Qwen2.5-1.5B-Instruct) - offline,
               free, lower quality. Works on CPU slowly, fast on GPU.
  - "openai" : an OpenAI-compatible HTTP API (set SYNTH_BASE_URL,
               SYNTH_MODEL, OPENAI_API_KEY) - better quality, costs money.

Usage:
    python finetune/generate_synthetic.py --n_variants 3
    # then EDIT finetune/data/REVIEW_synthetic.md, and:
    #   move the examples you keep into finetune/data/synthetic_reviewed.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED_PATH = DATA_DIR / "seed_qa.jsonl"
REVIEW_PATH = DATA_DIR / "REVIEW_synthetic.md"
OUT_PATH = DATA_DIR / "synthetic_reviewed.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful public-health assistant. Answer clearly, cite sources "
    "when provided, and say when you are unsure."
)

# Variation instructions applied to each seed example.
VARIATION_PROMPTS = [
    "Rewrite the question and answer in simpler words for a general audience. Keep the same facts.",
    "Rewrite the question more formally and make the answer slightly more detailed. Keep the same facts.",
    "Rewrite the question as asked by a worried parent, and adjust the tone of the answer to be calm and reassuring. Keep the same facts.",
    "Rewrite the question as a short informal text message, and answer in two or three clear sentences. Keep the same facts.",
]

# Patterns that indicate the generator refused instead of answering.
REFUSAL_PATTERNS = [
    r"\bas an ai\b",
    r"\bi cannot\b",
    r"\bi can't (answer|help|provide)",
    r"\bi'm sorry, but i\b",
    r"\bi am not able to\b",
]

REFUSAL_RE = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace - makes the dedup hash robust
    to formatting differences."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:16]


def load_seeds() -> list[dict]:
    seeds = []
    for line in SEED_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seeds.append(json.loads(line))
    return seeds


# ---------------------------------------------------------------------------
# Backends: each returns the assistant message content for a variation request
# ---------------------------------------------------------------------------
def _extract_assistant(raw: str) -> str:
    """Pull the answer out of a raw completion (handles both plain text and
    JSON responses defensively)."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "content" in obj:
            return str(obj["content"]).strip()
    except json.JSONDecodeError:
        pass
    # strip a leading "assistant:" if the model added one
    return re.sub(r"^assistant:\s*", "", raw, flags=re.IGNORECASE).strip()


def generate_local(instruction: str, question: str, answer: str, model_id: str) -> str:
    """Generate one variant with a local HF instruct model."""
    from transformers import pipeline  # imported lazily: heavy import

    # VERSION-SENSITIVE: transformers.pipeline("text-generation") with
    # return_full_text=False returns only the continuation. If your version
    # returns the full text instead, _extract_assistant still strips a
    # leading repeat of the prompt.
    pipe = generate_local._pipe
    if pipe is None:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        generate_local._pipe = pipe
    prompt = (
        f"{instruction}\n\n"
        f"Original question: {question}\n"
        f"Original answer: {answer}\n\n"
        "Return JSON: {\"question\": \"...\", \"answer\": \"...\"}"
    )
    messages = [
        {"role": "system", "content": "You rewrite public-health Q&A pairs. Always return valid JSON."},
        {"role": "user", "content": prompt},
    ]
    out = pipe(messages, max_new_tokens=400, do_sample=True, temperature=0.8)[0]
    return _extract_assistant(out["generated_text"][-1]["content"])


generate_local._pipe = None


def generate_api(instruction: str, question: str, answer: str, base_url: str, model: str, api_key: str) -> str:
    """Generate one variant through an OpenAI-compatible /chat/completions API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You rewrite public-health Q&A pairs. Always return valid JSON."},
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    f"Original question: {question}\n"
                    f"Original answer: {answer}\n\n"
                    'Return JSON: {"question": "...", "answer": "..."}'
                ),
            },
        ],
        "temperature": 0.8,
        "max_tokens": 500,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return _extract_assistant(data["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
def main() -> None:
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--n_variants", type=int, default=2,
                   help="variations to attempt per seed example")
    p.add_argument("--min_chars", type=int, default=80, help="answer length filter")
    p.add_argument("--max_chars", type=int, default=1500)
    p.add_argument("--model", type=str, default=None, help="override local model id")
    p.add_argument("--limit", type=int, default=None, help="only use the first N seeds (debug)")
    args = p.parse_args()

    backend = os.environ.get("SYNTH_BACKEND", "local").lower()
    seeds = load_seeds()
    if args.limit:
        seeds = seeds[: args.limit]
    print(f"backend={backend} seeds={len(seeds)} variants/seed={args.n_variants}")

    # dedup against seeds AND previously reviewed synthetics
    seen_hashes = {text_hash(s["messages"][1]["content"]) for s in seeds}
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen_hashes.add(text_hash(json.loads(line)["messages"][1]["content"]))

    rng = random.Random(0)
    accepted, rejected = [], []
    for seed in seeds:
        question = seed["messages"][1]["content"]
        answer = seed["messages"][2]["content"]
        for instruction in rng.sample(VARIATION_PROMPTS, k=min(args.n_variants, len(VARIATION_PROMPTS))):
            try:
                if backend == "openai":
                    raw = generate_api(
                        instruction, question, answer,
                        base_url=os.environ.get("SYNTH_BASE_URL", "https://api.openai.com/v1"),
                        model=os.environ.get("SYNTH_MODEL", "gpt-4o-mini"),
                        api_key=os.environ["OPENAI_API_KEY"],
                    )
                else:
                    model_id = args.model or os.environ.get(
                        "SYNTH_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
                    )
                    raw = generate_local(instruction, question, answer, model_id)
            except Exception as exc:  # noqa: BLE001 - log and continue
                rejected.append({"seed": seed["id"], "reason": f"generation failed: {exc}"})
                continue

            # The model was asked for JSON {"question", "answer"}; accept both
            # JSON and plain text gracefully.
            try:
                obj = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
                new_q, new_a = str(obj["question"]).strip(), str(obj["answer"]).strip()
            except (ValueError, KeyError, json.JSONDecodeError):
                new_q, new_a = "", raw

            # ---- filters -------------------------------------------------
            reason = None
            if not new_q or not new_a:
                reason = "unparseable output"
            elif any(rx.search(new_a) for rx in REFUSAL_RE):
                reason = "refusal pattern in answer"
            elif not (args.min_chars <= len(new_a) <= args.max_chars):
                reason = f"answer length {len(new_a)} outside [{args.min_chars},{args.max_chars}]"
            elif text_hash(new_q) in seen_hashes:
                reason = "duplicate (normalised hash)"
            if reason:
                rejected.append({"seed": seed["id"], "question": new_q[:80], "reason": reason})
                continue

            seen_hashes.add(text_hash(new_q))
            accepted.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": new_q},
                    {"role": "assistant", "content": new_a},
                ],
                "source": "synthetic_v1_unreviewed",
                "id": f"syn_{seed['id']}_{text_hash(new_q)[:6]}",
            })

    # ---- REVIEW file (human gate) ----------------------------------------
    lines = [
        "# Synthetic examples - MANUAL REVIEW REQUIRED",
        "",
        f"Generated: {len(accepted)} accepted, {len(rejected)} filtered out.",
        "For each example below write KEEP / EDIT (then edit it) / DROP.",
        "Move the KEEP/EDIT examples into `synthetic_reviewed.jsonl`",
        "(as single-line JSON objects) and only then include them in training.",
        "",
        "## Accepted (review these)",
        "",
    ]
    for ex in accepted:
        q = ex["messages"][1]["content"]
        a = ex["messages"][2]["content"]
        lines += [f"### {ex['id']}", f"- **Q:** {q}", f"- **A:** {a}", "- **Verdict:** ", ""]
    lines += ["## Rejected (already filtered, listed for transparency)", ""]
    for r in rejected:
        lines.append(f"- {r['seed']}: {r['reason']} ({r.get('question', '')})")
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REVIEW_PATH} ({len(accepted)} to review, {len(rejected)} filtered)")
    print("NEXT: open the review file, keep the good ones into synthetic_reviewed.jsonl,")
    print("      then re-run prepare_data.py")


if __name__ == "__main__":
    main()
