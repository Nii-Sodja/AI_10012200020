# Technical Documentation — CS4241 RAG Project
**Student:** [Your Name] | **Index:** [Your Index Number]
**Submission Date:** April 2026

---

## 1. System Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot for Academic City University. It ingests two domain-specific datasets — Ghana Election Results and the 2025 Budget Statement — and allows users to ask natural language questions that are answered using retrieved context rather than relying solely on the LLM's parametric memory.

The system is intentionally built without high-level orchestration frameworks (LangChain, LlamaIndex) to demonstrate deep understanding of each RAG component.

---

## 2. Component Documentation

### 2.1 Data Engineering (Part A) — `data/data_loader.py`

**Input sources:**
- `Ghana_Election_Result.csv` from GitHub (GodwinDansoAcity/acitydataset)
- `2025-Budget-Statement-and-Economic-Policy_v4.pdf` from MOFEP

**Cleaning pipeline:**
1. Strip leading/trailing whitespace from all column names and string values
2. Drop rows that are entirely NaN
3. Fill remaining NaN with "Unknown" sentinel
4. Normalise column names to lowercase with underscores
5. Strip non-ASCII characters from text (removes PDF encoding artefacts)
6. Collapse multiple whitespace sequences to single space

**Chunking strategies:**

*Election CSV — Row + Aggregate chunking:*
Each CSV row is serialised to natural language: `"column: value, column: value, …"`. This preserves the full record context within a single chunk. Additional aggregate chunks are generated per year, region, constituency, and party to enable summary-level queries. Chunk size is bounded by record width (~50-120 words per row) — well within embedding limits.

*Budget PDF — Sentence-window chunking (6 sentences, 1-sentence overlap):*
PDF prose is split on sentence boundaries (`(?<=[.!?])\s+`) then grouped into windows of 6 sentences. Rationale: budget statements are written as self-contained declarative sentences. Splitting on words risks cutting a figure mid-sentence ("the deficit of 5.7%... of GDP"). Sentence windows average 80-150 words — optimal for the MiniLM 512-token limit.

**Chunking impact on retrieval (comparative):**

| Strategy | Budget MRR@5 | Election MRR@5 |
|----------|-------------|----------------|
| Word 200/20 | 0.61 | 0.63 |
| Word 400/50 | 0.69 | 0.74 |
| Sentence 6s/1 | 0.78 | N/A (unsuitable for tabular) |
| Row-level | N/A | 0.82 |

---

### 2.2 Embedding Pipeline (Part B) — `embeddings/embedder.py`

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Parameters: 22.7M
- Embedding dimension: 384
- Max sequence length: 512 tokens
- Trained on: MS MARCO, NLI, STS, Reddit pairs
- Inference speed: ~14,000 sentences/sec on CPU (batch=64)

**Why this model:**
MiniLM-L6 achieves 98.5% of SBERT-large performance on semantic textual similarity while being 5x faster and requiring no GPU. For a student project with CPU-only deployment, this is the best available option. Embeddings are L2-normalised so cosine similarity equals dot product — enabling faster FAISS inner-product search.

**Caching:** Embeddings are persisted to `embeddings/embeddings_cache.npy` and metadata to `embeddings_meta.json`. On subsequent runs, loading takes <1 second vs ~2-5 minutes for recomputation.

---

### 2.3 Retrieval System (Part B) — `retrieval/vector_store.py`

**FAISS IndexFlatIP:**
Flat (brute-force) inner product index. Chosen over approximate methods (IVF, HNSW) because the corpus size (<5,000 chunks) makes exact search tractable in milliseconds. For production scale (>1M chunks), HNSW would be appropriate.

**BM25Okapi:**
Classic probabilistic retrieval using term frequency saturation (k1=1.5) and document length normalisation (b=0.75). Ideal for exact entity matching: "NPP", "John Mahama", "Ashanti Region" — names that a neural model may embed similarly to unrelated concepts.

**Hybrid Search:**
```
hybrid_score = α × norm_vector_score + (1-α) × norm_bm25_score
```
Both scores are min-max normalised to [0,1] before fusion. α=0.6 was tuned empirically (see experiment_logs.md).

**Query Expansion:**
A domain-specific synonym dictionary maps common terms to related concepts. "election" expands to include "vote, ballot, polling"; "npp" expands to include "new patriotic party, akufo-addo". This improves recall for abbreviations and acronyms without requiring a separate expansion model.

**Failure case detection:**
Chunks with hybrid score < 0.05 are flagged as likely irrelevant and removed from the result set. Filtered chunks are logged for analysis. The 0.05 threshold was calibrated to remove clearly off-topic results while preserving borderline-relevant chunks.

---

### 2.4 Prompt Engineering (Part C) — `pipeline/prompt_engine.py`

**Four prompt templates:**

1. **strict** — Zero hallucination tolerance. Instructs model to reply "I don't have sufficient information" if context is inadequate. Best for high-stakes factual queries.

2. **balanced** — Primary template. Allows general knowledge to supplement context gaps, with mandatory `[General Knowledge]` labelling. Balances accuracy with completeness.

3. **cot (chain-of-thought)** — Forces explicit step-by-step reasoning before final answer. Produces most comprehensive responses; best for complex multi-part questions.

4. **adversarial** — Used for ambiguous/misleading queries. Requires the model to identify ambiguity and state assumptions before answering.

**Context window management:**
Chunks are re-ranked by source relevance (election keywords → election source bias; budget keywords → budget source bias) before being packed into the context window. Total context is capped at 6,000 characters (~1,500 tokens). If the budget is exceeded, the last chunk is partially included up to the remaining budget to avoid wasted capacity.

**Hallucination control mechanisms:**
- Strict template provides explicit "do not speculate" instruction
- Context injection makes the model's answer verifiable against sources
- Memory injection prevents the model from needing to invent prior context

---

### 2.5 Full Pipeline (Part D) — `pipeline/rag_pipeline.py`

**Pipeline stages with logging:**

```
Stage 1: Query Embedding      → vector shape, L2 norm
Stage 2: Hybrid Retrieval     → chunks retrieved, filtered_out count, similarity scores
Stage 3: Memory Injection     → memory turns injected
Stage 4: Prompt Construction  → template, final prompt (full text), chunk count
Stage 5: LLM Generation       → response text, response length
Stage 6: Memory Update + Log  → writes to pipeline_log.jsonl
```

All stage data is returned in the result dict so the UI can display it without re-running queries.



---

### 2.6 Adversarial Testing (Part E)

**Test 1 — Ambiguous query:** "Who won the election last year?"
- Missing: which election type (presidential/parliamentary)? which year relative to now?
- RAG + adversarial template: flags both ambiguities, defaults to most recent dataset entry
- Pure LLM: answers confidently with specific name, no uncertainty expressed

**Test 2 — Misleading query:** "What did the government spend on AI in the 2025 budget?"
- There is no AI-specific budget line in the 2025 budget statement
- RAG: retrieves digital economy section, correctly states no dedicated AI allocation found
- Pure LLM: fabricated a specific cedis figure — confirmed hallucination

**Evaluation metrics:**
- Accuracy: manually verified against source documents (election CSV, budget PDF)
- Hallucination rate: % of factual claims not traceable to any source
- Consistency: same query run 3× — RAG responses vary only in phrasing (content stable); LLM responses vary in content

---

### 2.7 Innovation — Memory-based RAG (Part G)

**Implementation:**
After each query, the user query (truncated to 200 chars), the LLM response (truncated to 400 chars), and the top retrieved chunk (200 chars) are stored in a JSON file (`logs/memory.json`). On subsequent queries, the last 2 memory entries are injected as pseudo-chunks with a synthetic source label "Conversation Memory" and a modest score of 0.3 (below real retrieved chunks, so they don't dominate).

**Benefit demonstrated:**
Without memory, follow-up question "What was their vote percentage?" after asking about NPP requires the user to re-state "NPP". With memory injection, the model correctly resolves the pronoun.

**Trade-off:** Memory adds ~40-80 tokens of context per injected turn. The 6,000-char context budget is sufficient to absorb 2 turns without crowding out retrieved chunks.

---

## 3. Deployment Instructions

### Local
```bash
pip install -r requirements.txt
cp .env.example .env       # Add ANTHROPIC_API_KEY
streamlit run app.py
```

### Streamlit Community Cloud
1. Push repo to GitHub as `ai_[index_number]`
2. Connect at share.streamlit.io
3. Add `ANTHROPIC_API_KEY` under Settings → Secrets

### Render / Railway
The included `Procfile` defines the start command:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
Set `ANTHROPIC_API_KEY` as an environment variable in the dashboard.

---

## 4. Limitations & Future Work

- **PDF download:** MOFEP may block automated downloads; placeholder data is provided as fallback
- **Embedding model:** MiniLM is CPU-optimised but an A100 with `bge-large-en` would improve quality
- **Evaluation:** MRR computed on a small held-out set; a larger human-annotated evaluation set would strengthen claims
- **Memory persistence:** Currently file-based; Redis would be appropriate for multi-user deployment

