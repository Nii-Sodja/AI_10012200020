"""
pipeline/rag_pipeline.py  — Part D: Full RAG Pipeline
Student: [Your Name] | Index: [Your Index Number]

Pipeline: User Query → Query Expansion → Retrieval → Context Selection → Prompt → LLM → Response
Includes: stage-by-stage logging, memory (Part G innovation), adversarial mode.
"""
import os, sys, json, logging, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.data_loader    import load_all_chunks
from embeddings.embedder  import Embedder
from retrieval.vector_store import VectorStore
from pipeline.prompt_engine import construct_prompt, list_templates

import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
PIPELINE_LOG = os.path.join(LOGS_DIR, "pipeline_log.jsonl")
MEMORY_FILE  = os.path.join(LOGS_DIR, "memory.json")


class RAGPipeline:
    """
    Full RAG pipeline with:
    - Stage logging (Part D)
    - Hybrid retrieval (Part B)
    - Prompt engineering (Part C)
    - Memory-based context (Part G — Innovation)
    - Adversarial detection (Part E)
    """

    def __init__(self, top_k: int = 5, template: str = "balanced"):
        self.top_k    = top_k
        self.template = template
        self.embedder  = None
        self.store     = None
        self.client    = None
        self.memory: list[dict] = []   # Part G: conversation memory
        self._ready    = False
        self._load_memory()

    # ── initialisation ──────────────────────────────────────────────

    def initialise(self, force_reload: bool = False, progress_cb=None) -> dict:
        """
        Load data, build embeddings, build index.
        Returns status dict.
        """
        log = {"stage": "init", "timestamp": datetime.utcnow().isoformat()}

        def _cb(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        try:
            _cb("Loading embedding model…")
            self.embedder = Embedder()

            _cb("Checking vector store…")
            self.store = VectorStore(dim=self.embedder.dim)

            if not force_reload and self.store.load():
                _cb("Vector store loaded from cache.")
            else:
                _cb("Loading & chunking documents…")
                chunks = load_all_chunks(force=force_reload)

                _cb(f"Embedding {len(chunks)} chunks…")
                texts   = [c["text"]   for c in chunks]
                sources = [c["source"] for c in chunks]
                embeddings = self.embedder.embed(texts)

                _cb("Building FAISS + BM25 index…")
                self.store.build(embeddings, texts, sources)

            api_key = os.getenv("GROQ_API_KEY", "")
            if api_key and api_key != "your_groq_api_key_here":
                self.client = Groq(api_key=api_key, http_client=httpx.Client())
                _cb("Groq client initialised.")
            else:
                self.client = None
                _cb("WARNING: No GROQ_API_KEY found — running in mock mode.")

            self._ready = True
            log["status"] = "ok"
            log["index_size"] = self.store.index.ntotal if self.store.index else 0
            _cb(f"Pipeline ready. Index size: {log['index_size']}")
        except Exception as e:
            log["status"] = "error"
            log["error"]  = str(e)
            logger.error(f"Init failed: {e}")

        self._write_log(log)
        return log

    # ── query ───────────────────────────────────────────────────────

    def query(self, user_query: str, template: str = None,
               adversarial_mode: bool = False) -> dict:
        """
        Full pipeline execution for a single query.
        Returns rich dict with every stage's data (for display in UI).
        """
        if not self._ready:
            return {"error": "Pipeline not initialised. Call initialise() first."}

        t_start = time.time()
        tmpl = template or self.template
        if adversarial_mode:
            tmpl = "adversarial"

        result = {
            "query":         user_query,
            "template":      tmpl,
            "timestamp":     datetime.utcnow().isoformat(),
            "stages":        {},
        }

        # ── Stage 1: Query Embedding ────────────────────────────────
        logger.info(f"[S1] Embedding query: '{user_query}'")
        query_vec = self.embedder.embed_query(user_query)
        result["stages"]["embedding"] = {
            "vector_shape": list(query_vec.shape),
            "vector_norm":  float((query_vec ** 2).sum() ** 0.5),
        }

        # ── Stage 2: Retrieval ──────────────────────────────────────
        logger.info("[S2] Retrieving documents…")
        retrieval_result = self.store.retrieve(user_query, query_vec, k=self.top_k)
        retrieved = retrieval_result["results"]
        result["stages"]["retrieval"] = {
            "total_in_index":  self.store.index.ntotal,
            "retrieved":       len(retrieved),
            "filtered_out":    len(retrieval_result["filtered_out"]),
            "chunks":          retrieved,
        }
        logger.info(f"[S2] Retrieved {len(retrieved)} chunks.")

        # ── Stage 3: Memory injection (Part G) ─────────────────────
        memory_chunks = self._get_memory_chunks()
        all_chunks = memory_chunks + retrieved
        result["stages"]["memory"] = {
            "memory_chunks_injected": len(memory_chunks),
            "memory_summary": [m["text"][:80] for m in memory_chunks],
        }

        # ── Stage 4: Prompt Construction ───────────────────────────
        logger.info("[S3] Building prompt…")
        prompt, used_chunks, context_str = construct_prompt(user_query, all_chunks, tmpl)
        result["stages"]["prompt"] = {
            "template":      tmpl,
            "prompt_length": len(prompt),
            "chunks_used":   len(used_chunks),
            "final_prompt":  prompt,
        }
        logger.info(f"[S3] Prompt ready: {len(prompt)} chars.")

        # ── Stage 5: LLM Generation ─────────────────────────────────
        logger.info("[S4] Calling LLM…")
        response_text = self._call_llm(prompt)
        result["stages"]["generation"] = {
            "response_length": len(response_text),
        }
        result["response"] = response_text
        result["latency_s"] = round(time.time() - t_start, 2)
        logger.info(f"[S4] Response received ({len(response_text)} chars, {result['latency_s']}s).")

        # ── Stage 6: Save to memory & log ──────────────────────────
        self._update_memory(user_query, response_text, retrieved)
        self._write_log(result)

        return result

    # ── pure LLM (no retrieval) for Part E comparison ───────────────

    def query_pure_llm(self, user_query: str) -> str:
        """Query LLM with NO context (for adversarial comparison)."""
        prompt = f"Answer the following question to the best of your knowledge:\n\n{user_query}"
        return self._call_llm(prompt)

    # ── LLM caller ─────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self.client is None:
            return self._mock_response(prompt)
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM Error: {e}]"

    def _mock_response(self, prompt: str) -> str:
        """Fallback when no API key is set — useful for UI testing."""
        return (
            "⚠️ Mock Mode: No GROQ_API_KEY detected.\n\n"
            "Please add your key to the .env file and restart the app.\n\n"
            f"Your query was received and context was retrieved successfully.\n"
            f"Prompt length: {len(prompt)} characters."
        )

    # ── memory (Part G: Memory-based RAG) ───────────────────────────

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE) as f:
                self.memory = json.load(f)
        else:
            self.memory = []

    def _update_memory(self, query: str, response: str, chunks: list):
        """Store last 5 interactions as memory context."""
        entry = {
            "query":    query[:200],
            "response": response[:400],
            "top_chunk": chunks[0]["text"][:200] if chunks else "",
        }
        self.memory.append(entry)
        self.memory = self.memory[-5:]   # keep last 5
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f)

    def _get_memory_chunks(self) -> list[dict]:
        """Return memory entries as pseudo-chunks for context injection."""
        if not self.memory:
            return []
        chunks = []
        for m in self.memory[-2:]:   # inject last 2 turns
            text = f"[Previous conversation] Q: {m['query']} A: {m['response'][:200]}"
            chunks.append({"text": text, "source": "Conversation Memory", "score": 0.3})
        return chunks

    def clear_memory(self):
        self.memory = []
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        logger.info("Memory cleared.")

    # ── logging ─────────────────────────────────────────────────────

    def _write_log(self, data: dict):
        try:
            with open(PIPELINE_LOG, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Log write failed: {e}")

    def get_templates(self) -> list[str]:
        return list_templates()

    @property
    def is_ready(self) -> bool:
        return self._ready


if __name__ == "__main__":
    pipe = RAGPipeline(top_k=5)
    pipe.initialise()
    result = pipe.query("Who won the 2020 Ghana presidential election?")
    print("\n=== RESPONSE ===")
    print(result["response"])
    print(f"\nLatency: {result['latency_s']}s")
