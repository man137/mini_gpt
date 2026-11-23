# Interview Notes - mini-chatgpt

How to use this file: rehearse the walkthrough until you can draw the
architecture from memory, then the 10 design decisions, then skim the 20
questions the day before.

## 1. Architecture walkthrough (90 seconds, memorise the order)

1. **From-scratch GPT** (`scratch_gpt/`): char-level tokenizer -> token +
   positional embeddings -> 6 pre-LN transformer blocks (6 heads, 384 dims)
   -> final LayerNorm -> tied lm_head. Trained with AdamW, warmup+cosine LR,
   bf16/fp16 autocast, gradient clipping on Tiny Shakespeare. SFT demo adds
   a chat template with loss masking (-100 on non-assistant tokens).
2. **Fine-tune** (`finetune/`): 40 hand-written public-health Q&A pairs ->
   rendered with the base model's chat template -> QLoRA (4-bit NF4 base +
   LoRA r=16 on q/k/v/o_proj) on Colab T4 -> optional merge.
3. **RAG** (`rag/`): documents chunked ~500 tokens with 80-token overlap ->
   embedded with bge-small-en-v1.5 (L2-normalised) -> FAISS IndexFlatIP ->
   top-4 above a 0.35 cosine threshold -> grounded system prompt with
   citation format and injection rules.
4. **Backend** (`backend/`): FastAPI, one LLM interface with hf/ollama/fake
   backends, POST /chat streams SSE events (sources -> token -> done/error),
   history trimming, rate limiting, /ingest rebuild.
5. **Frontend** (`frontend/`): React + Vite, streams the POST body with
   fetch + ReadableStream, parses SSE manually, AbortController stop,
   collapsible sources panel.
6. **Eval** (`eval/`): 40 questions (25 in-scope / 8 out-of-scope / 7
   injection), three systems compared on keyword correctness, refusal
   accuracy, injection resistance, groundedness, TTFT p50/p95.

## 2. Ten design decisions and their trade-offs

1. **Why QLoRA instead of full fine-tuning?** A 1.5B model in fp16 needs
   ~3GB for weights + ~12GB optimizer state - full fine-tuning does not fit
   a 16GB T4. QLoRA freezes a 4-bit base and trains ~0.6% of parameters, so
   training fits in ~6-8GB with comparable style-transfer quality at this
   scale. Trade-off: no knowledge updates, and merging is needed for
   standalone serving.
2. **Why fine-tune AND RAG?** They solve different problems. Fine-tune
   moves FORMAT and behaviour (answer length, refusal style, citation
   habit); RAG supplies FACTS. Doing only one leaves the other broken:
   RAG-only has a generic voice and shaky citation discipline; FT-only
   hallucinates facts. The eval ablation (base / FT / FT+RAG) exists
   precisely to show this.
3. **Why bge-small-en-v1.5?** 384-dim, ~130MB, CPU-fast (matters because
   retrieval runs on the request path), strong MTEB scores for its size.
   Trade-off: English-only; a multilingual assistant would need
   bge-m3 or multilingual-e5.
4. **Why SSE instead of WebSockets?** Chat streaming is one-directional
   per request. SSE is plain HTTP, auto-reconnects, no protocol upgrade,
   trivially proxied. We need POST bodies (conversation history), so we
   stream `fetch` + ReadableStream instead of EventSource (GET-only).
   WebSockets pay off only for bidirectional features.
5. **Why loss masking in SFT?** Gradients should come only from tokens the
   assistant should emit. Training on user text teaches question-parroting
   and wastes capacity. Implemented with labels=-100 (cross_entropy's
   ignore_index) in both scratch_gpt/sft.py and the QLoRA collator.
6. **Why pre-LN (not post-LN)?** Pre-LN keeps the residual stream clean:
   x + Sublayer(LN(x)) means gradients always have an identity path, so
   deep stacks train stably without warmup-sensitivity. Post-LN (original
   paper) needs careful warmup and is why early transformers were brittle.
7. **Why weight tying?** lm_head shares the token-embedding matrix:
   fewer parameters (critical at small scale), and input/output spaces are
   the same vocabulary anyway. Cite Press & Wolf 2017 - it also improves
   perplexity.
8. **Why FAISS IndexFlatIP (exact) over HNSW/IVF?** At demo scale (tens to
   a few thousand chunks) exact inner-product search is sub-millisecond and
   has zero recall loss or tuning. Approximate indexes pay off past ~100k
   vectors. Normalised embeddings make IP == cosine.
9. **Why a score threshold (0.35) in retrieval?** It powers the refusal
   path: when nothing clears the threshold, the prompt says "no relevant
   context", and the grounded prompt forces "I don't know based on the
   provided documents." Without it, the model must answer from some chunk
   regardless of relevance - the classic forced-hallucination bug.
10. **Why evaluate three systems on the same set?** The interesting claim
    is attribution: which improvement came from fine-tuning and which from
    retrieval? Single-system evals can't answer that; the ablation can.
    Trade-off: keyword scoring is crude - a semantic judge is the upgrade.

## 3. What I would improve next (pick 2-3 in the interview)

- LLM-as-judge semantic scoring alongside keyword scoring.
- Injection eval through the DOCUMENTS (rag/docs payloads), not just the
  user message - the current system prompt defence is untested there.
- Redis-backed rate limiting + auth (JWT) for multi-user deployment.
- Streaming citations: attach chunk ids per sentence instead of one
  sources panel per answer.
- A reranker ablation (bge-reranker-base flag already exists in retriever).
- KV-cache reuse across turns for the HF backend.

## 4. Twenty likely interview questions

**Attention maths**
1. *Walk me through scaled dot-product attention.* Query/key/value
   projections; scores = QK^T / sqrt(d_k) - the scaling keeps the softmax
   out of saturation because dot products of d_k-dim random vectors have
   variance d_k; softmax turns scores into weights; output = weighted sum
   of values. In code: `scratch_gpt/model.py` Head.forward, 6 lines.
2. *Why divide by sqrt(head_size) and not the embedding dim?* Each head
   works in its own head_size-dimensional space; that is the dot product's
   variance. Multiplying by 1/sqrt(head_size) normalises variance to 1.
3. *Multi-head: why concat + project?* Heads specialise (positional,
   syntactic, coreference patterns); concat would change dimension, so a
   linear projection W_O mixes them back to the model dim and lets heads
   communicate.
4. *What is the KV cache and why does generation need it?* At each new
   token, keys/values of past tokens are unchanged; caching them makes each
   step O(T) instead of O(T^2). Our scratch model recomputes everything -
   correct but slow; saying this unprompted is a strong signal.

**Causal masking**
5. *What does the tril mask do?* Sets attention to -inf for future
   positions, so after softmax future tokens have exactly zero influence -
   required for autoregressive next-token prediction.
6. *How would you prove the mask works?* Perturb token t; outputs at
   positions < t must be bit-identical. That is literally our test
   `test_causal_mask` in scratch_gpt/tests/test_scratch.py.

**Training**
7. *Why does initial loss equal ln(vocab_size)?* At init the model is near-
   uniform over the vocab, so cross-entropy = -ln(1/V). For 65 chars that's
   ~4.18 - and our test asserts it.
8. *Why warmup + cosine decay?* Warmup avoids Adam's noisy second-moment
   estimates early on (large LR + bad variance estimate = divergence);
   cosine decay anneals to a small LR for stable convergence. GPT-2/3
   recipe.
9. *Why gradient clipping at 1.0?* Caps the global gradient norm so a
   single bad batch cannot blow up the weights; cheap insurance.
10. *bf16 vs fp16?* bf16 has fp32's exponent range (no overflow, no
    GradScaler needed) at fp16's memory cost, but needs Ampere+. On T4
    (compute 7.5) we use fp16 + GradScaler - see common/config.py.

**LoRA / fine-tuning**
11. *How does LoRA work mathematically?* W' = W + BA where B is d x r,
    A is r x k, r << min(d,k). Only A, B train (~0.6% of params here).
    alpha/r scales the update; init: A random, B zero so training starts
    as the identity.
12. *Why target q/k/v/o_proj?* Attention projections capture the
    style/behaviour changes SFT needs. Adding MLP layers (gate/up/down)
    improves quality for larger deltas at more VRAM cost.
13. *What is NF4 / double quantization?* NF4 is a 4-bit dtype matched to
    the normal distribution of LLM weights; double-quantization also
    quantizes the per-block scaling constants, saving ~0.4GB.

**RAG & hallucination**
14. *How do you prevent hallucinations?* Layered: (a) grounded prompt -
    answer only from context; (b) retrieval threshold so weak matches never
    reach the prompt; (c) trained refusal sentence; (d) citations the user
    can verify; (e) eval with out-of-scope refusals and groundedness.
    No single layer is sufficient.
15. *Why chunk with overlap?* A fact split at a chunk boundary appears
    fully in at least one chunk; without overlap retrieval can return two
    halves of the fact, neither sufficient.
16. *Bi-encoder vs cross-encoder?* Bi-encoder embeds query and document
    independently (fast, precomputable, weaker). Cross-encoder scores the
    PAIR jointly (accurate, slow) - used as an optional reranker on the
    top-k shortlist only.

**Streaming & deployment**
17. *Walk me through one request.* Browser POSTs history -> backend trims
    to token budget -> retrieves top-4 chunks -> builds grounded prompt ->
    streams tokens from TextIteratorStreamer (background thread) or Ollama
    HTTP -> SSE frames out; sources event precedes tokens so the UI can
    render them while text arrives.
18. *How would you scale this to 1000 users?* Move the model behind a
    batching server (vLLM/TGI) or keep Ollama + replicas; Redis rate
    limiting; auth; horizontal scaling of FastAPI (stateless already);
    vector DB service instead of local FAISS files.
19. *Why is TTFT the metric that matters?* Users perceive streamed answers
    as fast once the first token lands; total time is dominated by length.
    Our harness measures both (p50/p95).
20. *What breaks first in production?* Honestly, in this design: the
    in-memory rate limiter (per-process), the lazy first-request model
    load, and keyword-based eval drifting from real quality. Each has a
    named fix above.

## 5. Numbers to remember (approximate - verify with your runs)

- From-scratch GPT: ~10.6M non-embedding params; loss ln(65) ~ 4.18 at
  init; ~1.3-1.6 after 5k steps (config-dependent).
- QLoRA: ~9M trainable of ~1.5B (~0.6%); peak VRAM ~6-8GB on T4; adapter
  ~20-40MB.
- bge-small: 384-dim embeddings; retrieval ~tens of ms on CPU.
- Eval set: 40 questions = 25 in-scope + 8 out-of-scope + 7 injection.
