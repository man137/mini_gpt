# Dataset Quality Checklist

Apply this to `seed_qa.jsonl` AND to any synthetic expansion before training.
A dataset you cannot defend line-by-line will not survive an interview.

## Schema (machine-checked by `prepare_data.py`)

- [ ] Every line is valid JSON with exactly `messages`, `source`, `id`.
- [ ] `messages` is a non-empty list of `{role, content}` objects.
- [ ] Roles are only `system` | `user` | `assistant`.
- [ ] The final message is the `assistant` answer (what the model learns to produce).
- [ ] `id` is unique across the whole file.
- [ ] `source` says where the example came from (`seed_handwritten`,
      `synthetic_v1_reviewed`, `docs_public_health` ...).

## Content (human-checked - read every example)

- [ ] **Factually correct.** Every claim should be checkable against a
      trusted source (WHO / CDC / NHS pages). No invented numbers.
- [ ] **Consistent voice.** Same persona, same system prompt, similar length
      and register. Wildly varying styles teach the model a noisy target.
- [ ] **Answers refuse when they should.** Examples where the correct answer
      is "I don't know / see a clinician" are as valuable as factual ones.
- [ ] **No contradictions.** Two examples giving different cutoffs for the
      same question is the fastest way to confuse a small model.
- [ ] **Not medical advice for individuals.** Public-health information, not
      diagnosis. Dangerous-if-wrong topics (dosing, emergencies) always
      redirect to professional care.
- [ ] **Length sanity.** Roughly 40-120 words per answer. Very short answers
      teach terseness; very long ones overflow context and slow training.
- [ ] **Deduplicated.** Near-duplicate questions (same question rephrased)
      inflate confidence without adding information. `generate_synthetic.py`
      hashes normalised text to catch exact and near-exact duplicates.

## After synthetic expansion (generate_synthetic.py)

- [ ] The `REVIEW_synthetic.md` file was opened and every example marked
      KEEP / EDIT / DROP by a human before it enters training data.
- [ ] Refusals filtered out (patterns like "I cannot", "as an AI").
- [ ] Length outliers filtered (see `--min_chars` / `--max_chars`).
- [ ] Spot-check 10 random examples against the source documents.

## Just before training

- [ ] `python finetune/prepare_data.py` runs with zero schema errors.
- [ ] Printed split stats look sane (90/10, no empty files).
- [ ] A rendered example from `train_tokenized.jsonl` was decoded by hand and
      looks like a correct chat conversation.
