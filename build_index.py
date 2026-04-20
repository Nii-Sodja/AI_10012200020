"""
build_index.py — Runs during Render's buildCommand (before the server starts).

Pre-downloads data, generates embeddings, and builds the FAISS + BM25 index so
cold starts load from disk instead of downloading/processing on first request.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Pre-building RAG index ===")

print("[1/4] Loading data chunks...")
from data.data_loader import load_all_chunks
chunks = load_all_chunks(force=False)
print(f"      {len(chunks)} chunks loaded.")

print("[2/4] Loading embedding model...")
from embeddings.embedder import Embedder
embedder = Embedder()

print("[3/4] Checking vector store cache...")
from retrieval.vector_store import VectorStore
store = VectorStore(dim=embedder.dim)

if store.load():
    print(f"      Vector store already cached: {store.index.ntotal} vectors. Skipping rebuild.")
else:
    print(f"[3/4] Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    sources = [c["source"] for c in chunks]
    embeddings = embedder.embed(texts)
    print("[4/4] Building FAISS + BM25 index...")
    store.build(embeddings, texts, sources)
    print(f"      Index built: {store.index.ntotal} vectors.")

print("=== Pre-build complete! App will start fast on first request. ===")
