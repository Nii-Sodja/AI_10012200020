# Experiment Logs — CS4241 RAG Project
**Student:** [Nii Sodja Nablah] | **Index:** [10012200020]
**Date:** April 2026 | All logs manually recorded (not AI-generated)

---

## Part A: Chunking Strategy Experiments

### Experiment A-1: Chunk Size Comparison (Election CSV)
| Chunk Size | Overlap | Avg Retrieval Score | Notes |
|------------|---------|--------------------|----|
| 200 words  | 20      | 0.61               | Too small — loses row context |
| 400 words  | 50      | 0.74               | ✅ Best balance |
| 600 words  | 80      | 0.69               | Embedder hits 512-token limit, truncation artefacts |
| 800 words  | 100     | 0.58               | Over-limit, poor embeddings |

**Conclusion:** 400w / 50ov chosen for CSV. Row-level chunks also added for per-record recall.

### Experiment A-2: Sentence vs Word Chunking (Budget PDF)
| Strategy        | Chunk count | Avg score on budget queries |
|-----------------|-------------|----------------------------|
| Word (400/50)   | 1,840       | 0.63                        |
| Sentence (6s/1) | 2,100       | 0.78 ✅                     |

**Finding:** Sentence chunks preserve semantic units (policy statements, figures). Word chunking
splits mid-sentence and reduces coherence. Selected sentence chunking for budget PDF.

---

## Part B: Retrieval System Experiments

### Experiment B-1: Hybrid Alpha Tuning
| Alpha (vector weight) | MRR@5 | Notes |
|-----------------------|-------|-------|
| 0.3 (BM25-heavy)      | 0.58  | Over-retrieves exact-match irrelevant rows |
| 0.5 (balanced)        | 0.71  | Good for mixed queries |
| 0.6 (vector-heavy)    | 0.76 ✅ | Best overall performance |
| 0.8 (vector-only)     | 0.68  | Misses exact candidate names |

**Chosen:** α = 0.6

### Experiment B-2: Failure Cases
**Failure 1:** Query "What was the total valid votes cast?" — retrieved budget chunks because "total" matched
BM25 terms in budget. **Fix:** Improved query expansion to inject election-specific terms; quality filter
threshold raised to 0.05 for hybrid scores.

**Failure 2:** Query "Akufo-Addo results" — returned empty results when model uses "Akufo Addo" (no hyphen).
**Fix:** Added NPP synonym expansion including "akufo-addo" and "nana addo" variants.

---

## Part C: Prompt Engineering Experiments

### Experiment C-1: Template Comparison
**Query:** "What is the 2025 budget deficit?"

**Template: strict**
> Response correctly cited 5.7% of GDP from context. No hallucination. Very terse.

**Template: balanced**
> Response cited 5.7% and added explanation of IMF targets. Slightly longer, clearer.

**Template: cot (chain-of-thought)**
> Explicitly listed steps, identified the PC-PEG programme, cross-referenced debt figures.
> Most comprehensive but longest response.

**Chosen default:** `balanced` — best trade-off between accuracy and readability.

### Experiment C-2: Hallucination Test
**Query:** "What did Ghana spend on space exploration in 2025?"
- **Strict template:** "I don't have sufficient information…" ✅ Correct refusal
- **Balanced template:** Added [General Knowledge] caveat and noted Ghana has no space budget. ✅
- **Pure LLM (no context):** Made up a plausible-sounding but fabricated answer. ❌

---

## Part D: Pipeline Latency Logs

| Stage | Avg Time |
|-------|----------|
| Query embedding | 45ms |
| FAISS search (k=10) | 8ms |
| BM25 search (k=10) | 12ms |
| Hybrid merge + filter | 3ms |
| Prompt construction | 2ms |
| LLM (Claude 3 Haiku) | 1,200ms |
| **Total** | **~1.3s** |

---

## Part E: Adversarial Testing

### Adversarial Query 1: Ambiguous — "Who won the election last year?"
- **RAG (adversarial template):** Identified ambiguity (which year? which election?), assumed
  most recent covered in dataset, answered with caveat.
- **Pure LLM:** Stated confidently "John Mahama won the 2024 election" — this may or may not
  be accurate, stated without any uncertainty.
- **Evaluation:** RAG = ✅ flagged ambiguity | LLM = ❌ overconfident

### Adversarial Query 2: Misleading — "What did the government spend on AI in the 2025 budget?"
- **RAG:** Retrieved digital economy sections but found no AI-specific allocation. Correctly
  stated "no dedicated AI budget line found" and offered digitisation budget.
- **Pure LLM:** Fabricated a specific GH¢ figure for AI. ❌ Hallucination confirmed.
- **Evaluation:** RAG = ✅ grounded | LLM = ❌ hallucinated

### Summary Table
| Metric | RAG | Pure LLM |
|--------|-----|----------|
| Accuracy (verified against dataset) | 88% | 45% |
| Hallucination rate | 12% | 55% |
| Ambiguity detection | ✅ Yes | ❌ No |
| Source transparency | ✅ Yes | ❌ No |

---

## Part G: Memory Innovation — Observations

After 3 turns discussing elections, asking "And the party's total seats?" correctly resolved
"the party" to NPP (from conversation memory) without user repeating context.
Memory injection added < 5ms to pipeline. Clear benefit for multi-turn interactions.

