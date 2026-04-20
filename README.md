# GovIntel AI
**Student:** Nii Sodja Nablah | **Index:** 10012200020
**Lecturer:** Godwin N. Danso | **Date:** 20th April 2026

---

## Overview
A fully custom Retrieval-Augmented Generation (RAG) chatbot,
built **without** LangChain, LlamaIndex, or any pre-built RAG pipeline.

All core components — chunking, embedding, vector storage, retrieval, prompt construction —
are implemented from scratch.

**Datasets:**
- Ghana Election Results CSV (GitHub)
- Ghana 2025 Budget Statement PDF (MOFEP)

---

## Project Structure
```
rag_project/
├── app.py                    # Streamlit UI (Final Deliverable)
├── requirements.txt
├── .env.example              # Copy to .env and add your API key
├── Procfile                  # For cloud deployment
├── data/
│   └── data_loader.py        # Part A: Data cleaning & chunking
├── embeddings/
│   └── embedder.py           # Part B: Embedding pipeline
├── retrieval/
│   └── vector_store.py       # Part B: FAISS + BM25 + Hybrid search
├── pipeline/
│   ├── prompt_engine.py      # Part C: Prompt templates & context management
│   └── rag_pipeline.py       # Part D: Full pipeline orchestration
├── logs/
│   ├── experiment_logs.md    # Part A/C/E: Manual experiment notes
│   └── pipeline_log.jsonl    # Auto-generated stage logs
└── docs/
    └── documentation.md      # Detailed technical documentation
```

---

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/[your-username]/ai_[index_number]
cd ai_[index_number]
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```
The app opens at http://localhost:8501

**Note:** On first run the app downloads data, builds embeddings (~2-5 min).
Subsequent runs load from cache (< 10 seconds).

---

## Parts Implemented

| Part | Description | Marks |
|------|-------------|-------|
| A | Data Engineering: cleaning, word+sentence chunking (justified) | 4 |
| B | Embedding (all-MiniLM-L6-v2), FAISS, BM25, Hybrid Search | 6 |
| C | 4 prompt templates, context window management, hallucination control | 4 |
| D | Full pipeline with per-stage logging | 10 |
| E | 2 adversarial queries, RAG vs LLM comparison table | 6 |
| F | Architecture SVG + justification (in app Architecture tab) | 8 |
| G | Memory-based RAG (last-5-turns context injection) | 6 |
| Final | Streamlit UI, experiment logs, documentation | 16 |

---

## Architecture
```
User Query → [Embedder: MiniLM] → [FAISS + BM25 Hybrid Search]
           → [Memory Injection] → [Prompt Template]
           → [Groq Llama 3.1 8B API] → [Response + Stage Logs]
```

## Deployment
Deployed at: **[Add your deployed URL here]**

Cloud options:
- **Streamlit Community Cloud:** Connect GitHub repo, add GROQ_API_KEY as secret.
- **Render:** Use the included Procfile.
- **Railway:** Auto-detects Python, set env var.

---

## Key Design Decisions

### No LangChain — Why?
Building RAG components manually (as required) gives full control over:
- Chunking strategies tuned to each dataset type
- Custom hybrid fusion scoring
- Transparent logging at every pipeline stage
- Memory injection without framework overhead

### Hybrid Search (α=0.6)
Vector search alone misses exact entity names (candidates, parties, regions).
BM25 alone misses semantically similar but lexically different queries.
Hybrid fusion at α=0.6 outperforms either method alone (see experiment_logs.md).

### Memory-based RAG
Stateless RAG fails on follow-up questions. Storing the last 5 turns and
injecting them as pseudo-chunks allows coherent multi-turn conversations.

