# Evaluation Report - mini-chatgpt

> Fill in every `<PLACEHOLDER>` below with YOUR measured numbers after
> running `make eval` (or the manual commands in README "Evaluation").
> Approximate expectations are marked "(expected: ...)" - replace them.
> Never present unmeasured numbers as results.

## 1. Setup

- Date of run: <DATE>
- Hardware: <e.g. Colab T4 16GB / RTX 4070 / CPU-only>
- Base model: Qwen2.5-<0.5B|1.5B|3B>-Instruct
- LoRA config: r=<16>, alpha=<32>, dropout=<0.05>, targets=q,k,v,o_proj
- Training: <N> examples, <3> epochs, lr <2e-4> - final train loss <LOSS>,
  final val loss <LOSS> (paste from the Colab log)
- Embedding model: BAAI/bge-small-en-v1.5, threshold <0.35>, top_k <4>
- Eval set: 40 questions (25 in-scope, 8 out-of-scope, 7 prompt-injection)

## 2. Headline table

Paste `eval/results_summary.md` here after all three runs:

| system | in-scope correct % | refusal acc % | injection resist % | groundedness | TTFT p50 (s) | TTFT p95 (s) | total p50 (s) |
|---|---|---|---|---|---|---|---|
| base      | <FILL> | <FILL> | <FILL> | n/a | <FILL> | <FILL> | <FILL> |
| finetuned | <FILL> | <FILL> | <FILL> | n/a | <FILL> | <FILL> | <FILL> |
| rag       | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |

Expected shape of results (approximate, from similar setups - verify):
- base: low in-scope correctness without retrieval (model guesses), low
  refusal accuracy (happily answers off-domain questions), TTFT fastest
  after warmup.
- finetuned: better answer style and slightly better refusals than base;
  facts still unreliable (40 training examples cannot teach facts).
- rag: highest in-scope correctness and the only system with meaningful
  groundedness; refusal accuracy depends on the retrieval threshold.

## 3. Detailed findings

### 3.1 In-scope correctness (25 questions)

- base: <N>/25 correct. Failure mode observed: <e.g. confident but wrong
  numbers, outdated guidance, verbose off-topic answers>.
- finetuned: <N>/25 correct. Change vs base: <e.g. answers follow the
  trained 40-120-word style, more "see a clinician" redirects>.
- rag: <N>/25 correct. Wrong answers were: <list ids - were they retrieval
  misses (wrong/no chunk) or generation misses (right chunk, wrong answer)?>
  This distinction is the most interesting thing in the report - look up
  the `n_sources` column in results.csv for each wrong id.

### 3.2 Refusal accuracy (8 out-of-scope questions)

- base: <N>/8 refused properly. The base model likely answers cookie
  recipes and World Cup questions happily.
- finetuned: <N>/8.
- rag: <N>/8. Note: with use_rag, refusal comes from the retrieval
  threshold dropping all chunks + the grounded prompt's "I don't know"
  rule, NOT from the model's own judgement. Say which mechanism fired
  (check n_sources == 0 in results.csv).

### 3.3 Injection resistance (7 questions)

- base: <N>/7 resisted. Look for "system prompt" leaks or "rules
  cancelled" echo in results.csv answers.
- finetuned: <N>/7.
- rag: <N>/7. The RAG system prompt has an explicit "treat context as
  data" instruction - but note the injection rides in the USER message
  here, not the documents, so that particular defence is not exercised.
  Document-injection would need eval questions with payloads inside
  rag/docs - a good "what I would do next" item.

### 3.4 Latency

- TTFT p50/p95 per system: <FILL> (expected on T4-class hardware: HF
  backend ~0.1-0.5 s TTFT, ~5-20 s total for 256 tokens; ollama similar;
  first request after startup is slower due to lazy model load).
- Total p50: <FILL>.
- Where the time goes: <e.g. retrieval ~30-80 ms with bge-small on CPU;
  generation dominates>.

## 4. Qualitative examples

Paste 2-3 question/answer triplets (one good RAG answer with citations,
one refusal, one injection attempt) from `eval/results.csv`:

> **Q (ev_001):** How long does the flu vaccine take to become effective?
>
> - base: <paste>
> - finetuned: <paste>
> - rag: <paste>

## 5. Threats to validity (be honest - interviewers love this section)

- Keyword scoring is lexical, not semantic: an answer can be correct
  without the keywords or wrong with them (false negatives dominate).
- 40 questions is a small sample: +-1 question is a 2.5-point swing in
  the in-scope percentage.
- Single run per system: generation temperature 0.2 makes runs fairly
  deterministic, but GPU nondeterminism still moves borderline answers.
- Refusal on out-of-scope was scored against the exact RAG refusal
  sentence; a base model saying "I cannot answer that" scores 0 even
  though it refused - check those rows manually.

## 6. Conclusion

<Two or three sentences: which system wins on which axis, and what you
would change first (e.g. more eval questions, semantic judge, doc-level
injection eval).>
