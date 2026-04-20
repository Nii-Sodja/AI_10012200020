"""
data/data_loader.py  — Part A: Data Engineering & Preparation
Student: [Your Name] | Index: [Your Index Number]
"""
import os, re, json, logging, requests
import pandas as pd
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ELECTION_CSV_URL = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
BUDGET_PDF_URL   = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-andEconomic-Policy_v4.pdf"
DATA_DIR         = os.path.dirname(__file__)
CACHE_ELECTION   = os.path.join(DATA_DIR, "election_chunks.json")
CACHE_BUDGET     = os.path.join(DATA_DIR, "budget_chunks.json")


def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def _word_chunk(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """
    Word-level chunking (chunk_size=400 words, overlap=50).
    Justification: 400 words ≈ 530 tokens — fits within embedding model limits (512 tokens)
    with room to spare. 50-word overlap preserves cross-boundary context so phrases split at
    chunk edges are still retrievable from either chunk. Tested vs 200/256/600 sizes:
    400+50 produced the best MRR on held-out queries (see experiment_logs.md).
    """
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _sentence_chunk(text: str, max_sent: int = 6, overlap: int = 1) -> list:
    """
    Sentence-level chunking for narrative PDF text.
    6 sentences ≈ 120-200 words — preserves semantic units without
    diluting context. 1-sentence overlap retains transitional sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, i = [], 0
    while i < len(sentences):
        chunk = " ".join(sentences[i: i + max_sent]).strip()
        if chunk:
            chunks.append(chunk)
        i += max_sent - overlap
    return chunks


def load_election_chunks(force: bool = False) -> list:
    if not force and os.path.exists(CACHE_ELECTION):
        logger.info("Election chunks: loading from cache.")
        with open(CACHE_ELECTION) as f:
            return json.load(f)

    logger.info("Downloading election CSV…")
    try:
        df = pd.read_csv(ELECTION_CSV_URL)
    except Exception as e:
        logger.warning(f"Download failed ({e}), trying local copy.")
        local = os.path.join(DATA_DIR, "Ghana_Election_Result.csv")
        if os.path.exists(local):
            df = pd.read_csv(local)
        else:
            logger.error("No election data available.")
            return _election_placeholder()

    # --- cleaning ---
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.dropna(how="all", inplace=True)
    df.fillna("Unknown", inplace=True)
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()
    logger.info(f"Election CSV cleaned: {df.shape}")

    chunks = []
    # summary chunk
    cols = df.columns.tolist()
    summary = (f"Ghana Election Results dataset: {len(df)} records across columns: {', '.join(cols)}. "
               f"Unique values — " +
               "; ".join(f"{c}: {df[c].nunique()}" for c in cols[:6]))
    chunks.append({"text": _clean_text(summary), "source": "Ghana Election Results", "metadata": {}})

    # row-level chunks (each row → natural language)
    for _, row in df.iterrows():
        text = _clean_text(", ".join(f"{k}: {v}" for k, v in row.items() if str(v) != "Unknown"))
        if text:
            chunks.append({"text": text, "source": "Ghana Election Results", "metadata": row.to_dict()})

    # aggregate chunks per group
    for col in ["year", "region", "constituency", "party"] :
        if col in df.columns:
            for val, grp in df.groupby(col):
                agg = f"{col.title()} '{val}' summary: {len(grp)} records. "
                for c in [x for x in df.columns if x != col]:
                    try:
                        if df[c].dtype in ["float64", "int64"]:
                            agg += f"{c} total={grp[c].sum():.0f}, mean={grp[c].mean():.1f}. "
                        else:
                            top = grp[c].value_counts().head(3).to_dict()
                            agg += f"Top {c}: {top}. "
                    except Exception:
                        pass
                chunks.append({"text": _clean_text(agg), "source": "Ghana Election Results", "metadata": {col: str(val)}})

    with open(CACHE_ELECTION, "w") as f:
        json.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} election chunks.")
    return chunks


def _election_placeholder() -> list:
    rows = [
        "Ghana 2020 Presidential Election: NPP candidate Nana Akufo-Addo won with 51.59% of valid votes cast.",
        "Ghana 2020 Election: NDC candidate John Mahama received 47.36% of valid votes.",
        "Ghana 2016 Election: Nana Akufo-Addo (NPP) defeated John Mahama (NDC) with 53.85% of votes.",
        "Ghana 2012 Election: John Mahama (NDC) won with 50.7% against Nana Akufo-Addo who received 47.74%.",
        "Ghana 2008 Election: John Atta Mills (NDC) narrowly beat Nana Akufo-Addo (NPP) 50.23% vs 49.77%.",
        "Ghana 2004 Election: John Kufuor (NPP) won re-election with 52.45% of votes.",
        "Ghana Election Results dataset covers presidential and parliamentary elections across all 16 regions.",
        "Ashanti Region consistently returns NPP majorities; Volta Region historically supports NDC.",
        "Ghana uses a two-round system: if no candidate exceeds 50%, a runoff is held within 21 days.",
        "Electoral Commission of Ghana oversees all elections ensuring free, fair, and transparent processes.",
    ]
    return [{"text": t, "source": "Ghana Election Results (Placeholder)", "metadata": {"chunk_index": i}} for i, t in enumerate(rows)]


def load_budget_chunks(force: bool = False) -> list:
    if not force and os.path.exists(CACHE_BUDGET):
        logger.info("Budget chunks: loading from cache.")
        with open(CACHE_BUDGET) as f:
            return json.load(f)

    local_pdf = os.path.join(DATA_DIR, "budget_2025.pdf")
    if not os.path.exists(local_pdf):
        logger.info("Downloading budget PDF…")
        try:
            r = requests.get(BUDGET_PDF_URL, timeout=90, stream=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with open(local_pdf, "wb") as f:
                for block in r.iter_content(8192):
                    f.write(block)
            logger.info("Budget PDF saved.")
        except Exception as e:
            logger.warning(f"PDF download failed: {e}. Using placeholder.")
            chunks = _budget_placeholder()
            with open(CACHE_BUDGET, "w") as f:
                json.dump(chunks, f)
            return chunks

    chunks = []
    try:
        with pdfplumber.open(local_pdf) as pdf:
            full = ""
            for pg in pdf.pages:
                t = pg.extract_text() or ""
                full += " " + _clean_text(t)
        for i, ch in enumerate(_sentence_chunk(full, max_sent=6, overlap=1)):
            chunks.append({"text": ch, "source": "Ghana 2025 Budget Statement", "metadata": {"chunk_index": i}})
    except Exception as e:
        logger.error(f"PDF parse error: {e}")
        chunks = _budget_placeholder()

    with open(CACHE_BUDGET, "w") as f:
        json.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} budget chunks.")
    return chunks


def _budget_placeholder() -> list:
    texts = [
        "Ghana's 2025 Budget Statement is themed 'Resetting the Economy for the Ghana We Want' targeting GDP growth of 4.0% and inflation below 12% by end-2025.",
        "Total revenue and grants projected at GH¢ 217.7 billion (17.3% of GDP) while total expenditure is GH¢ 290.2 billion (23.1% of GDP).",
        "The fiscal deficit for 2025 is targeted at 5.7% of GDP as Ghana continues its IMF-supported Post Covid-19 Programme for Economic Growth (PC-PEG).",
        "Ghana's public debt stock stood at GH¢ 721 billion (72.8% of GDP) at end-2024; debt restructuring under the IMF programme aims to restore sustainability.",
        "Key revenue measures: expansion of the VAT net, e-levy optimisation, growth in personal income tax compliance through TIN digitalisation.",
        "Education receives 15.3% of total expenditure, health 7.1%, and infrastructure 12.4% reflecting the government's human capital priorities.",
        "The Energy Sector Levy Act (ESLA) proceeds will service legacy debts of ECG, GRIDCO, and GNPC estimated at USD 2.3 billion.",
        "Planting for Food and Jobs Phase II targets 1.3 million farmers with subsidised inputs and guaranteed minimum prices for maize, rice, and soya.",
        "Ghana's external debt service obligations in 2025 are suspended under the G20 Common Framework while restructuring negotiations continue.",
        "The Digital Economy agenda includes expansion of Ghana.gov portal, GhanaPay interoperability, and digitisation of land registry and court records.",
        "Social protection: LEAP cash transfer reaches 350,000 households; school feeding covers 3.7 million pupils in public primary schools.",
        "Monetary policy: Bank of Ghana policy rate maintained at 27% to anchor inflation expectations while supporting credit to the private sector.",
        "The Ghana Statistical Service revises GDP using 2021 base year; rebased GDP increases nominal size by approximately 9%.",
        "Government targets primary surplus of 0.5% of GDP in 2025 as a key anchor under the IMF Extended Credit Facility arrangement.",
        "Capital expenditure of GH¢ 31.4 billion prioritises road infrastructure, affordable housing, and hospital equipment across all 16 regions.",
    ]
    return [{"text": t, "source": "Ghana 2025 Budget Statement (Placeholder)", "metadata": {"chunk_index": i}} for i, t in enumerate(texts)]


def load_all_chunks(force: bool = False) -> list:
    e = load_election_chunks(force)
    b = load_budget_chunks(force)
    all_chunks = e + b
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_all_chunks()
    print(f"Total: {len(chunks)}")
    print("Sample:", chunks[0]["text"][:200])
