"""
setup.py — One-time setup script.
Run: python setup.py
Downloads data, builds index, verifies everything works.
Student: [Your Name] | Index: [Your Index Number]
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("GovIntel AI — Setup")
    print("=" * 60)

    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key == "your_groq_api_key_here":
        print("\n⚠️  WARNING: GROQ_API_KEY not set in .env")
        print("   The app will run in mock mode (no real LLM responses).")
        print("   Copy .env.example to .env and add your key.\n")
    else:
        print("✅ API key found.")

    print("\n[1/3] Loading & chunking data...")
    from data.data_loader import load_all_chunks
    chunks = load_all_chunks(force=False)
    print(f"    Loaded {len(chunks)} chunks.")

    print("\n[2/3] Building embeddings...")
    from embeddings.embedder import Embedder
    emb = Embedder()
    texts   = [c["text"]   for c in chunks]
    sources = [c["source"] for c in chunks]
    embeddings = emb.embed(texts)
    print(f"    Embeddings shape: {embeddings.shape}")

    print("\n[3/3] Building vector index...")
    from retrieval.vector_store import VectorStore
    store = VectorStore(dim=emb.dim)
    store.build(embeddings, texts, sources)
    print(f"    FAISS index: {store.index.ntotal} vectors")

    # Quick smoke test
    print("\n[Smoke test] Running test query...")
    vec = emb.embed_query("Who won the 2020 Ghana presidential election?")
    result = store.retrieve("Who won the 2020 Ghana presidential election?", vec, k=3)
    print(f"    Retrieved {len(result['results'])} chunks. Top result:")
    if result["results"]:
        print(f"    [{result['results'][0]['score']:.3f}] {result['results'][0]['text'][:100]}")

    print("\n" + "=" * 60)
    print("✅ Setup complete! Run:  streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
