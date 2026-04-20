"""
pipeline/prompt_engine.py  — Part C: Prompt Engineering & Generation
Student: [Your Name] | Index: [Your Index Number]

Implements:
  - Multiple prompt templates (strict, balanced, chain-of-thought)
  - Context window management (truncation + relevance ranking)
  - Hallucination control via explicit instructions
"""
import logging
logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000   # ~1500 tokens — safe context budget


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES (iterations documented in experiment_logs.md)
# ─────────────────────────────────────────────────────────────

TEMPLATES = {

    "strict": """You are an AI assistant for Academic City University with access to Ghana election and budget data.
Answer ONLY using the provided context below. If the context does not contain enough information to answer, say exactly:
"I don't have sufficient information in my knowledge base to answer that."
Do NOT guess, speculate, or use any knowledge outside the context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER (based strictly on context):""",

    "balanced": """You are a knowledgeable AI assistant for Academic City University.
Use the provided context as your primary source. You may use your general knowledge to fill minor gaps,
but clearly label any non-context information with [General Knowledge].
If the context contradicts your general knowledge, trust the context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:""",

    "cot": """You are an analytical AI assistant for Academic City University.
Think step by step before giving your final answer.

CONTEXT:
{context}

QUESTION: {query}

Let me reason through this:
Step 1 — Identify relevant information from the context.
Step 2 — Synthesise the key facts.
Step 3 — Formulate a clear answer.

REASONING:""",

    "adversarial": """You are a careful AI assistant for Academic City University.
The user's query may be ambiguous, misleading, or incomplete. Before answering:
1. Identify any ambiguity in the question.
2. State what assumptions you are making.
3. Answer based on the context, noting if the query has multiple valid interpretations.

CONTEXT:
{context}

QUESTION: {query}

ANALYSIS & ANSWER:""",
}


def rank_chunks_by_relevance(chunks: list[dict], query: str) -> list[dict]:
    """
    Re-rank chunks: prefer chunks whose source matches query keywords.
    Simple heuristic re-ranker (no model needed).
    """
    query_lower = query.lower()
    election_kws = {"election", "vote", "party", "npp", "ndc", "president", "parliament", "constituency", "region"}
    budget_kws   = {"budget", "fiscal", "revenue", "gdp", "tax", "debt", "expenditure", "economy", "inflation", "imf"}

    election_hit = any(k in query_lower for k in election_kws)
    budget_hit   = any(k in query_lower for k in budget_kws)

    def priority(chunk):
        src = chunk.get("source", "").lower()
        score = chunk.get("score", 0)
        if election_hit and "election" in src:
            return score + 0.2
        if budget_hit and "budget" in src:
            return score + 0.2
        return score

    return sorted(chunks, key=priority, reverse=True)


def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict]]:
    """
    Context window management:
    1. Re-rank by relevance
    2. Truncate to max_chars budget
    3. Return formatted context string + used chunks
    """
    ranked = rank_chunks_by_relevance(chunks, "")
    context_parts, used, total_chars = [], [], 0

    for chunk in ranked:
        text = chunk.get("text", "")
        src  = chunk.get("source", "Unknown")
        entry = f"[Source: {src}]\n{text}"
        if total_chars + len(entry) > max_chars:
            # Partial truncation: add as many chars as remain
            remaining = max_chars - total_chars
            if remaining > 200:
                context_parts.append(entry[:remaining] + "…")
                used.append(chunk)
            break
        context_parts.append(entry)
        used.append(chunk)
        total_chars += len(entry)

    context = "\n\n---\n\n".join(context_parts)
    logger.info(f"Context built: {len(used)} chunks, {total_chars} chars.")
    return context, used


def construct_prompt(query: str, chunks: list[dict], template_name: str = "balanced") -> tuple[str, list[dict], str]:
    """
    Returns: (final_prompt, used_chunks, context_string)
    """
    if template_name not in TEMPLATES:
        template_name = "balanced"
    template = TEMPLATES[template_name]

    context, used_chunks = build_context(chunks, MAX_CONTEXT_CHARS)
    prompt = template.format(context=context, query=query)
    logger.info(f"Prompt constructed: template={template_name}, length={len(prompt)} chars.")
    return prompt, used_chunks, context


def list_templates() -> list[str]:
    return list(TEMPLATES.keys())


if __name__ == "__main__":
    sample_chunks = [
        {"text": "NPP won the 2020 election with 51.59% of votes.", "source": "Ghana Election Results", "score": 0.9},
        {"text": "Ghana 2025 budget targets GDP growth of 4%.", "source": "Ghana 2025 Budget", "score": 0.7},
    ]
    for t in list_templates():
        prompt, used, ctx = construct_prompt("Who won the 2020 election?", sample_chunks, t)
        print(f"\n=== Template: {t} ===")
        print(prompt[:300])
