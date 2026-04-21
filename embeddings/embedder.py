"""
embeddings/embedder.py  — Part B: Custom Embedding Pipeline
Student: [Your Name] | Index: [Your Index Number]

Uses sentence-transformers (all-MiniLM-L6-v2) locally — no OpenAI calls needed.
Embeddings are cached to disk to avoid recomputing on every run.
"""
import os, json, logging, gc, numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

MODEL_NAME  = "all-MiniLM-L6-v2"   # 22M params, 384-dim, fast & accurate
EMBED_DIR   = os.path.dirname(__file__)
CACHE_NPY   = os.path.join(EMBED_DIR, "embeddings_cache.npy")
CACHE_META  = os.path.join(EMBED_DIR, "embeddings_meta.json")


class Embedder:
    """Custom embedding pipeline wrapping sentence-transformers."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.dim = 384  # known for all-MiniLM-L6-v2
        logger.info(f"Embedder initialized (lazy): {model_name}, dim={self.dim}")

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            self.model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Model loaded.")

    def _unload_model(self):
        if self.model is not None:
            logger.info("Unloading embedding model to free memory.")
            self.model = None
            gc.collect()

    def embed(self, texts: list[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """
        Embed a list of texts.
        normalize=True → cosine similarity == dot product (faster at retrieval time).
        """
        self._load_model()
        logger.info(f"Embedding {len(texts)} texts (batch={batch_size})…")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        result = embeddings.astype(np.float32)
        self._unload_model()
        return result

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query string."""
        self._load_model()
        vec = self.model.encode([query], normalize_embeddings=normalize, convert_to_numpy=True)
        result = vec.astype(np.float32)
        self._unload_model()
        return result

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
