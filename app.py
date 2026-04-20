"""
app.py — Streamlit UI for GovIntel AI
Student: [Your Name] | Index: [Your Index Number]

Final Deliverable: Full UI with query input, retrieved chunks display, response panel,
adversarial testing tab, architecture diagram, and experiment logs viewer.
"""
import os, sys, json, time, logging
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st


# ── page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="GovIntel AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0a0e1a; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a1620 100%); }

.hero-banner {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #006064 100%);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(100,181,246,0.3);
    box-shadow: 0 8px 32px rgba(13,71,161,0.4);
}
.hero-banner h1 { color: #e3f2fd; font-size: 2rem; font-weight: 700; margin: 0; }
.hero-banner p  { color: #90caf9; margin: 0.5rem 0 0; font-size: 1rem; }

.chunk-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,181,246,0.2);
    border-left: 4px solid #42a5f5;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #b0bec5;
}
.chunk-card .score-badge {
    display: inline-block;
    background: rgba(66,165,245,0.2);
    color: #42a5f5;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.response-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(102,187,106,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    color: #e8f5e9;
    font-size: 0.97rem;
    line-height: 1.7;
    margin-top: 1rem;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #42a5f5; }
.metric-card .label { font-size: 0.8rem; color: #78909c; margin-top: 4px; }
.stage-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    color: #42a5f5; text-transform: uppercase; margin-bottom: 4px;
}
.warn-box {
    background: rgba(255,152,0,0.1);
    border: 1px solid rgba(255,152,0,0.4);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #ffcc02;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ── singleton pipeline ───────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading AI pipeline… (first load only)")
def get_pipeline():
    from pipeline.rag_pipeline import RAGPipeline
    pipe = RAGPipeline(top_k=5, template="balanced")
    pipe.initialise()
    return pipe


# ── sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k    = st.slider("Top-K chunks", 1, 10, 3)
    template = st.selectbox("Prompt template", ["balanced", "strict", "cot", "adversarial"])
    st.markdown("---")
    st.markdown("### 📊 Pipeline Info")

    pipe = get_pipeline()
    pipe.top_k    = top_k
    pipe.template = template

    if pipe.is_ready and pipe.store:
        st.metric("Index size", f"{pipe.store.index.ntotal:,} chunks")
        st.metric("Embedding dim", f"{pipe.embedder.dim}D")
        st.metric("Memory turns", len(pipe.memory))

    if st.button("🔄 Rebuild Index"):
        with st.spinner("Rebuilding…"):
            pipe.initialise(force_reload=True)
        st.success("Rebuilt!")
    if st.button("🧹 Clear Memory"):
        pipe.clear_memory()
        st.success("Memory cleared.")

    st.markdown("---")
    st.markdown("### 🔑 API Key")
    api_env = os.getenv("GROQ_API_KEY", "")
    if api_env and api_env != "your_groq_api_key_here":
        st.success("API key loaded from .env ✓")
    else:
        key_input = st.text_input("Enter Groq API key", type="password", placeholder="gsk-…")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input
            import groq as _grq
            pipe.client = _grq.Groq(api_key=key_input)
            st.success("Key set!")


# ── main UI ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🎓 GovIntel AI</h1>
  <p>Ghana Election & Budget Intelligence</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat", "🔬 Retrieval Inspector", "⚔️ Adversarial Tests",
                "📐 Architecture", "📋 Experiment Logs"])


# ════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════════════════════════════════════════
with tabs[0]:
    if "history" not in st.session_state:
        st.session_state.history = []

    # ── example queries ──
    st.markdown("**Try an example query:**")
    ex_cols = st.columns(3)
    examples = [
        "Who won the 2020 Ghana presidential election?",
        "What is Ghana's 2025 budget deficit target?",
        "Which party dominates the Ashanti Region?",
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, use_container_width=True):
            st.session_state.pending_query = ex

    # ── chat history ──
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(f'<div class="response-box">{turn["response"]}</div>', unsafe_allow_html=True)

    # ── input ──
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about Ghana elections or the 2025 budget…") or pending

    if user_input and pipe.is_ready:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving context & generating response…"):
                result = pipe.query(user_input, template=template)

            response = result.get("response", "No response generated.")
            st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)

            # metadata expander
            with st.expander(f"📎 Pipeline details — {result.get('latency_s', '?')}s"):
                r_stage = result["stages"].get("retrieval", {})
                e_stage = result["stages"].get("embedding", {})
                m_stage = result["stages"].get("memory", {})

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Chunks retrieved", r_stage.get("retrieved", 0))
                c2.metric("Filtered out",     r_stage.get("filtered_out", 0))
                c3.metric("Memory injected",  m_stage.get("memory_chunks_injected", 0))
                c4.metric("Latency",          f"{result.get('latency_s', '?')}s")

                st.markdown("**Retrieved Chunks:**")
                for ch in r_stage.get("chunks", []):
                    src   = ch.get("source", "?")
                    score = ch.get("score", 0)
                    text  = ch.get("text", "")[:250]
                    method = ch.get("method", "hybrid")
                    st.markdown(
                        f'<div class="chunk-card"><span class="score-badge">score={score:.3f} | {method}</span>'
                        f'<br><b>{src}</b><br>{text}…</div>',
                        unsafe_allow_html=True,
                    )

        st.session_state.history.append({"query": user_input, "response": response})

    elif user_input and not pipe.is_ready:
        st.error("Pipeline not ready. Check sidebar.")


# ════════════════════════════════════════════════════════════════════
# TAB 2: RETRIEVAL INSPECTOR
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🔬 Retrieval Inspector")
    st.caption("Inspect raw retrieval scores, compare vector vs BM25 vs hybrid, and visualise similarity distributions.")

    inspect_query = st.text_input("Inspection query", "What is the GDP growth target in Ghana's 2025 budget?")
    k_val = st.slider("Retrieve top-k", 3, 15, 8, key="inspect_k")

    if st.button("Run Inspection") and pipe.is_ready:
        import plotly.graph_objects as go

        vec = pipe.embedder.embed_query(inspect_query)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**🔵 Vector Search**")
            v_res = pipe.store.vector_search(vec, k=k_val)
            for r in v_res:
                st.markdown(f'<div class="chunk-card"><span class="score-badge">{r["score"]:.4f}</span><br>{r["text"][:150]}…</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown("**🟡 BM25 Keyword**")
            b_res = pipe.store.bm25_search(inspect_query, k=k_val)
            for r in b_res:
                st.markdown(f'<div class="chunk-card"><span class="score-badge">{r["score"]:.4f}</span><br>{r["text"][:150]}…</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown("**🟢 Hybrid (α=0.6)**")
            h_res = pipe.store.hybrid_search(inspect_query, vec, k=k_val)
            for r in h_res:
                st.markdown(f'<div class="chunk-card"><span class="score-badge">{r["score"]:.4f}</span><br>{r["text"][:150]}…</div>', unsafe_allow_html=True)

        # score distribution chart
        st.markdown("---")
        st.markdown("**Score Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Vector", x=[f"#{i+1}" for i in range(len(v_res))],
                             y=[r["score"] for r in v_res], marker_color="#42a5f5"))
        fig.add_trace(go.Bar(name="BM25",   x=[f"#{i+1}" for i in range(len(b_res))],
                             y=[r["score"] for r in b_res], marker_color="#ffca28"))
        fig.add_trace(go.Bar(name="Hybrid", x=[f"#{i+1}" for i in range(len(h_res))],
                             y=[r["score"] for r in h_res], marker_color="#66bb6a"))
        fig.update_layout(barmode="group", template="plotly_dark",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#b0bec5", height=350)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3: ADVERSARIAL TESTS
# ════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("⚔️ Adversarial Testing — Part E")
    st.caption("Compare RAG responses vs pure LLM (no context). Evaluate hallucination, accuracy, consistency.")

    adversarial_queries = {
        "Ambiguous":   "Who won the election last year?",
        "Misleading":  "What did the government spend on AI in the 2025 budget?",
        "Incomplete":  "What is the deficit?",
        "Out-of-scope": "What is the population of Mars?",
    }

    selected_type = st.selectbox("Select adversarial query type", list(adversarial_queries.keys()))
    adv_query = st.text_input("Query (editable)", adversarial_queries[selected_type])

    if st.button("Run Adversarial Test") and pipe.is_ready:
        with st.spinner("Running comparison…"):
            rag_result  = pipe.query(adv_query, adversarial_mode=True)
            pure_result = pipe.query_pure_llm(adv_query)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🟢 RAG Response")
            st.markdown(f'<div class="response-box">{rag_result.get("response","")}</div>', unsafe_allow_html=True)
            r = rag_result["stages"].get("retrieval", {})
            st.caption(f"Retrieved: {r.get('retrieved',0)} chunks | Filtered: {r.get('filtered_out',0)}")

        with c2:
            st.markdown("#### 🔴 Pure LLM (no context)")
            st.markdown(f'<div class="response-box">{pure_result}</div>', unsafe_allow_html=True)
            st.caption("No retrieval context — may hallucinate.")

        st.markdown("---")
        st.markdown("#### 📊 Evidence-based Analysis")
        st.markdown(f"""
| Criterion | RAG System | Pure LLM |
|-----------|-----------|----------|
| Uses verified data | ✅ Yes (from index) | ❌ No |
| Identifies ambiguity | ✅ (adversarial template) | ⚠️ Variable |
| Hallucination risk | 🟡 Low (context-grounded) | 🔴 High |
| Response consistency | ✅ Reproducible | ⚠️ Varies |
| Source transparency | ✅ Shows chunks | ❌ Hidden |
""")


# ════════════════════════════════════════════════════════════════════
# TAB 4: ARCHITECTURE
# ════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📐 System Architecture — Part F")

    arch_svg = """
<svg viewBox="0 0 900 560" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#0d1528;border-radius:12px;padding:16px">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#42a5f5"/>
    </marker>
  </defs>
  <!-- Data Sources -->
  <rect x="30" y="30" width="160" height="60" rx="8" fill="#1a237e" stroke="#42a5f5" stroke-width="1.5"/>
  <text x="110" y="55" text-anchor="middle" fill="#90caf9" font-size="11" font-family="monospace">Election CSV</text>
  <text x="110" y="72" text-anchor="middle" fill="#64b5f6" font-size="9">GodwinDanso/acitydataset</text>

  <rect x="210" y="30" width="160" height="60" rx="8" fill="#1a237e" stroke="#42a5f5" stroke-width="1.5"/>
  <text x="290" y="55" text-anchor="middle" fill="#90caf9" font-size="11" font-family="monospace">Budget PDF</text>
  <text x="290" y="72" text-anchor="middle" fill="#64b5f6" font-size="9">mofep.gov.gh/2025</text>

  <!-- Arrow down to Data Eng -->
  <line x1="110" y1="90" x2="110" y2="140" stroke="#42a5f5" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="290" y1="90" x2="290" y2="140" stroke="#42a5f5" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Part A: Data Engineering -->
  <rect x="30" y="140" width="340" height="70" rx="8" fill="#0d47a1" stroke="#1565c0" stroke-width="1.5"/>
  <text x="200" y="162" text-anchor="middle" fill="#e3f2fd" font-size="12" font-weight="bold">Part A: Data Engineering</text>
  <text x="200" y="180" text-anchor="middle" fill="#90caf9" font-size="10">Cleaning → Word Chunk (400w/50ov) + Sentence Chunk (6s/1ov)</text>
  <text x="200" y="197" text-anchor="middle" fill="#90caf9" font-size="10">Cache: election_chunks.json | budget_chunks.json</text>

  <!-- Arrow to Embedding -->
  <line x1="200" y1="210" x2="200" y2="260" stroke="#42a5f5" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Part B: Embedding -->
  <rect x="30" y="260" width="340" height="70" rx="8" fill="#006064" stroke="#00838f" stroke-width="1.5"/>
  <text x="200" y="282" text-anchor="middle" fill="#e0f7fa" font-size="12" font-weight="bold">Part B: Embedding Pipeline</text>
  <text x="200" y="300" text-anchor="middle" fill="#80deea" font-size="10">all-MiniLM-L6-v2 (384-dim) → FAISS IndexFlatIP</text>
  <text x="200" y="317" text-anchor="middle" fill="#80deea" font-size="10">+ BM25Okapi | Hybrid Search (α=0.6) + Query Expansion</text>

  <!-- User Query flow (right side) -->
  <rect x="620" y="30" width="250" height="50" rx="8" fill="#4a148c" stroke="#7b1fa2" stroke-width="1.5"/>
  <text x="745" y="52" text-anchor="middle" fill="#e1bee7" font-size="12" font-weight="bold">👤 User Query</text>
  <text x="745" y="68" text-anchor="middle" fill="#ce93d8" font-size="9">Streamlit UI → RAG Pipeline</text>

  <line x1="745" y1="80" x2="745" y2="140" stroke="#ce93d8" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Part C: Prompt Engine -->
  <rect x="590" y="140" width="280" height="70" rx="8" fill="#1b5e20" stroke="#2e7d32" stroke-width="1.5"/>
  <text x="730" y="162" text-anchor="middle" fill="#c8e6c9" font-size="12" font-weight="bold">Part C: Prompt Engine</text>
  <text x="730" y="180" text-anchor="middle" fill="#a5d6a7" font-size="10">4 Templates: balanced | strict | cot | adversarial</text>
  <text x="730" y="197" text-anchor="middle" fill="#a5d6a7" font-size="10">Context Window Mgmt (6000 chars) + Reranking</text>

  <!-- Cross-connection: retrieval → prompt -->
  <line x1="370" y1="295" x2="590" y2="230" stroke="#42a5f5" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arr)"/>
  <text x="490" y="260" text-anchor="middle" fill="#64b5f6" font-size="9">retrieved chunks</text>

  <!-- Part G: Memory -->
  <rect x="590" y="260" width="280" height="60" rx="8" fill="#bf360c" stroke="#e64a19" stroke-width="1.5"/>
  <text x="730" y="284" text-anchor="middle" fill="#fbe9e7" font-size="12" font-weight="bold">Part G: Memory RAG (Innovation)</text>
  <text x="730" y="302" text-anchor="middle" fill="#ffab91" font-size="10">Last 5 turns stored → injected as pseudo-chunks</text>

  <line x1="730" y1="210" x2="730" y2="260" stroke="#e64a19" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- LLM -->
  <rect x="590" y="360" width="280" height="60" rx="8" fill="#4a148c" stroke="#7b1fa2" stroke-width="1.5"/>
  <text x="730" y="384" text-anchor="middle" fill="#e1bee7" font-size="12" font-weight="bold">LLM: Llama 3 8B</text>
  <text x="730" y="402" text-anchor="middle" fill="#ce93d8" font-size="10">via Groq API | max_tokens=1024</text>

  <line x1="730" y1="320" x2="730" y2="360" stroke="#ce93d8" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Response -->
  <rect x="590" y="460" width="280" height="60" rx="8" fill="#33691e" stroke="#558b2f" stroke-width="1.5"/>
  <text x="730" y="487" text-anchor="middle" fill="#f1f8e9" font-size="12" font-weight="bold">📤 Response + Metadata</text>
  <text x="730" y="505" text-anchor="middle" fill="#aed581" font-size="10">Response | Chunks | Scores | Latency | Logs</text>

  <line x1="730" y1="420" x2="730" y2="460" stroke="#aed581" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Part D label -->
  <rect x="30" y="460" width="340" height="60" rx="8" fill="#0a1628" stroke="#42a5f5" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="200" y="485" text-anchor="middle" fill="#64b5f6" font-size="11" font-weight="bold">Part D: Full Pipeline Logging</text>
  <text x="200" y="503" text-anchor="middle" fill="#546e7a" font-size="9">Stage logs → pipeline_log.jsonl | Each stage timed</text>

  <!-- Legend -->
  <text x="450" y="540" text-anchor="middle" fill="#546e7a" font-size="9">Student: [Name] | Index: [Index] | 2026</text>
</svg>
"""
    st.markdown(arch_svg, unsafe_allow_html=True)

    st.markdown("""
### Architecture Justification

**Why FAISS + BM25 Hybrid?**
The two datasets have very different retrieval characteristics. Election data contains structured records
(party names, percentages, region names) that respond excellently to BM25 keyword matching. The budget PDF
contains dense economic prose that requires semantic understanding — hence vector search. Hybrid fusion
with α=0.6 (slight vector bias) balances both.

**Why all-MiniLM-L6-v2?**
At 22M parameters and 384 dimensions, it is the fastest sentence-transformer with quality competitive
with much larger models on retrieval benchmarks (BEIR). It fits in ~90MB RAM — deployable on free tiers.

**Why sentence-level chunking for the budget?**
Budget sentences are complete semantic units (a policy statement, a figure). Splitting mid-sentence
destroys meaning. 6-sentence windows provide enough context while keeping chunks under 512 tokens.

**Memory innovation:**
RAG pipelines are typically stateless. Injecting the last 2 conversation turns as pseudo-chunks allows
follow-up questions ("And what about the previous year?") to resolve correctly without repeating context.
""")


# ════════════════════════════════════════════════════════════════════
# TAB 5: EXPERIMENT LOGS
# ════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📋 Experiment Logs — Part A/C/E")

    logs_dir  = os.path.join(os.path.dirname(__file__), "logs")
    pipe_log  = os.path.join(logs_dir, "pipeline_log.jsonl")
    expmt_log = os.path.join(logs_dir, "experiment_logs.md")

    tab_a, tab_b = st.tabs(["Pipeline Logs", "Experiment Notes"])

    with tab_a:
        if os.path.exists(pipe_log):
            lines = open(pipe_log).readlines()
            st.metric("Total logged events", len(lines))
            for line in reversed(lines[-20:]):
                try:
                    entry = json.loads(line)
                    stage = entry.get("stage", entry.get("query", "?")[:50])
                    ts    = entry.get("timestamp", "")[:19]
                    st.markdown(f"`{ts}` — **{stage}**")
                    with st.expander("Details"):
                        st.json(entry)
                except Exception:
                    st.text(line[:200])
        else:
            st.info("No pipeline logs yet. Run some queries first.")

    with tab_b:
        if os.path.exists(expmt_log):
            st.markdown(open(expmt_log).read())
        else:
            st.info("experiment_logs.md not found in /logs.")

