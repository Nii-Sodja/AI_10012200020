"""
embeddings/embedder.py  — Part B: Custom Embedding Pipeline
Student: [Your Name] | Index: [Your Index Number]

Uses sentence-transformers (all-MiniLM-L6-v2) locally — no OpenAI calls needed.
Embeddings are cached to disk to avoid recomputing on every run.
"""
import os, json, logging, numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME  = "all-MiniLM-L6-v2"   # 22M params, 384-dim, fast & accurate
EMBED_DIR   = os.path.dirname(__file__)
CACHE_NPY   = os.path.join(EMBED_DIR, "embeddings_cache.npy")
CACHE_META  = os.path.join(EMBED_DIR, "embeddings_meta.json")


class Embedder:
    """Custom embedding pipeline wrapping sentence-transformers."""

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim   = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.dim}")

    def embed(self, texts: list[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """
        Embed a list of texts.
        normalize=True → cosine similarity == dot product (faster at retrieval time).
        """
        logger.info(f"Embedding {len(texts)} texts (batch={batch_size})…")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query string."""
        vec = self.model.encode([query], normalize_embeddings=normalize, convert_to_numpy=True)
        return vec.astype(np.float32)

    # ── cache helpers ──────────────────────────────────────────────

    @staticmethod
    def save_cache(embeddings: np.ndarray, chunks: list):
        np.save(CACHE_NPY, embeddings)
        texts = [c["text"] for c in chunks]
        sources = [c["source"] for c in chunks]
        with open(CACHE_META, "w") as f:
            json.dump({"texts": texts, "sources": sources}, f)
        logger.info(f"Cache saved: {CACHE_NPY}")

    @staticmethod
    def load_cache():
        if os.path.exists(CACHE_NPY) and os.path.exists(CACHE_META):
            embeddings = np.load(CACHE_NPY)
            with open(CACHE_META) as f:
                meta = json.load(f)
            logger.info(f"Cache loaded: {embeddings.shape}")
            return embeddings, meta["texts"], meta["sources"]
        return None, None, None

    @staticmethod
    def cache_exists() -> bool:
        return os.path.exists(CACHE_NPY) and os.path.exists(CACHE_META)


if __name__ == "__main__":
    emb = Embedder()
    v = emb.embed_query("What was Ghana's 2020 election result?")
    print("Query vector shape:", v.shape, "norm:", np.linalg.norm(v))
