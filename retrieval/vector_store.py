"""
retrieval/vector_store.py  — Part B: Custom Vector Store + Retrieval
Student: [Your Name] | Index: [Your Index Number]

Implements:
  - FAISS-based vector store
  - Top-k cosine retrieval
  - BM25 keyword retrieval
  - Hybrid search (vector + keyword fusion) — chosen extension
  - Query expansion via synonym injection
  - Failure case detection & filtering
"""
import os, json, logging, re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)
STORE_DIR = os.path.dirname(__file__)
FAISS_IDX = os.path.join(STORE_DIR, "faiss.index")
META_JSON  = os.path.join(STORE_DIR, "store_meta.json")

# ── query expansion synonyms (domain-specific) ──────────────────────
EXPANSION_MAP = {
    "election":   ["vote", "ballot", "polling", "electoral", "presidential"],
    "budget":     ["expenditure", "fiscal", "revenue", "gdp", "appropriation"],
    "npp":        ["new patriotic party", "akufo-addo", "bawumia"],
    "ndc":        ["national democratic congress", "mahama"],
    "economy":    ["economic", "growth", "gdp", "fiscal policy"],
    "winner":     ["won", "victory", "elected", "result"],
    "ghana":      ["ghanaian", "accra"],
    "tax":        ["levy", "vat", "irs", "gra", "revenue"],
    "debt":       ["borrowing", "loan", "imf", "restructuring"],
    "inflation":  ["prices", "cpi", "monetary policy", "rate"],
}


class VectorStore:
    """
    Custom FAISS-backed vector store with hybrid BM25 + vector search.

    Chosen extension: Hybrid Search (keyword + vector).
    Rationale: Structured election data responds well to exact keyword matches
    (party names, candidate names, regions) while budget narrative benefits from
    semantic vector search. Hybrid fusion outperforms either alone.
    """

    def __init__(self, dim: int = 384):
        self.dim   = dim
        self.index = None
        self.texts:   list[str] = []
        self.sources: list[str] = []
        self.bm25 = None

    # ── index building ──────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, texts: list[str], sources: list[str]):
        """Build FAISS index and BM25 index from embeddings + texts."""
        assert len(embeddings) == len(texts) == len(sources)
        self.texts   = texts
        self.sources = sources

        # FAISS — inner product (works as cosine because vectors are L2-normalised)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dim}")

        # BM25 — tokenised word corpus
        tokenised = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built.")

        self._save()

    def _save(self):
        faiss.write_index(self.index, FAISS_IDX)
        with open(META_JSON, "w") as f:
            json.dump({"texts": self.texts, "sources": self.sources}, f)
        logger.info("VectorStore saved to disk.")

    def load(self) -> bool:
        if os.path.exists(FAISS_IDX) and os.path.exists(META_JSON):
            self.index = faiss.read_index(FAISS_IDX)
            with open(META_JSON) as f:
                meta = json.load(f)
            self.texts   = meta["texts"]
            self.sources = meta["sources"]
            tokenised    = [t.lower().split() for t in self.texts]
            self.bm25    = BM25Okapi(tokenised)
            logger.info(f"VectorStore loaded: {self.index.ntotal} vectors.")
            return True
        return False

    @property
    def is_built(self) -> bool:
        return self.index is not None and len(self.texts) > 0

    # ── query expansion ─────────────────────────────────────────────

    def expand_query(self, query: str) -> str:
        """
        Inject domain synonyms for known terms.
        Improves recall for abbreviations and alternate phrasings.
        """
        expanded = query
        for term, synonyms in EXPANSION_MAP.items():
            if re.search(r'\b' + re.escape(term) + r'\b', query, re.IGNORECASE):
                expanded += " " + " ".join(synonyms[:2])
        if expanded != query:
            logger.info(f"Query expanded: '{query}' → '{expanded[:120]}'")
        return expanded

    # ── retrieval ───────────────────────────────────────────────────

    def vector_search(self, query_vec: np.ndarray, k: int = 10) -> list[dict]:
        """Pure FAISS cosine-similarity retrieval."""
        if not self.is_built:
            raise RuntimeError("VectorStore not built.")
        scores, indices = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text":   self.texts[idx],
                "source": self.sources[idx],
                "score":  float(score),
                "method": "vector",
                "rank":   len(results) + 1,
            })
        return results

    def bm25_search(self, query: str, k: int = 10) -> list[dict]:
        """BM25 keyword retrieval."""
        if self.bm25 is None:
            raise RuntimeError("BM25 not built.")
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_k  = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_k:
            results.append({
                "text":   self.texts[idx],
                "source": self.sources[idx],
                "score":  float(scores[idx]),
                "method": "bm25",
                "rank":   len(results) + 1,
            })
        return results

    def hybrid_search(self, query: str, query_vec: np.ndarray,
                       k: int = 5, alpha: float = 0.6) -> list[dict]:
        """
        Hybrid search: alpha * vector_score + (1-alpha) * bm25_score (normalised).
        alpha=0.6 weights semantic search slightly higher than keyword matching —
        tuned empirically (see experiment_logs.md).
        Includes query expansion before retrieval.
        """
        expanded_query = self.expand_query(query)

        vec_results  = self.vector_search(query_vec, k=k * 2)
        bm25_results = self.bm25_search(expanded_query, k=k * 2)

        # normalise scores to [0,1]
        def normalise(results):
            scores = [r["score"] for r in results]
            mn, mx = min(scores, default=0), max(scores, default=1)
            rng = mx - mn or 1
            for r in results:
                r["norm_score"] = (r["score"] - mn) / rng
            return results

        vec_results  = normalise(vec_results)
        bm25_results = normalise(bm25_results)

        # merge by text key
        combined: dict[str, dict] = {}
        for r in vec_results:
            combined[r["text"]] = {**r, "hybrid_score": alpha * r["norm_score"]}
        for r in bm25_results:
            if r["text"] in combined:
                combined[r["text"]]["hybrid_score"] += (1 - alpha) * r["norm_score"]
                combined[r["text"]]["method"] = "hybrid"
            else:
                combined[r["text"]] = {**r, "hybrid_score": (1 - alpha) * r["norm_score"]}

        ranked = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)[:k]
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
            r["score"] = r["hybrid_score"]
        return ranked

    # ── failure detection & fix ─────────────────────────────────────

    def filter_low_quality(self, results: list[dict], threshold: float = 0.15) -> tuple[list, list]:
        """
        Failure case: retrieval returns irrelevant results (low similarity).
        Fix: filter chunks below threshold; log them as failures for analysis.

        Identified failure cases:
          1. Very short or garbage chunks with high BM25 overlap but no semantic content.
          2. Off-topic chunks retrieved due to common stopwords.
        """
        good, bad = [], []
        for r in results:
            if r.get("score", 0) >= threshold:
                good.append(r)
            else:
                bad.append(r)
                logger.warning(f"LOW-QUALITY chunk filtered (score={r['score']:.3f}): {r['text'][:60]}…")
        return good, bad

    def retrieve(self, query: str, query_vec: np.ndarray, k: int = 5) -> dict:
        """
        Full retrieval pipeline:
          1. Hybrid search
          2. Quality filter
          3. Return structured result with metadata
        """
        raw = self.hybrid_search(query, query_vec, k=k + 3)
        good, bad = self.filter_low_quality(raw, threshold=0.05)
        final = good[:k]
        return {
            "query":          query,
            "results":        final,
            "filtered_out":   bad,
            "total_retrieved": len(raw),
        }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from embeddings.embedder import Embedder
    emb = Embedder()
    store = VectorStore(dim=emb.dim)
    if store.load():
        vec = emb.embed_query("Who won the 2020 Ghana election?")
        res = store.retrieve("Who won the 2020 Ghana election?", vec, k=3)
        for r in res["results"]:
            print(f"[{r['score']:.3f}] {r['text'][:100]}")
