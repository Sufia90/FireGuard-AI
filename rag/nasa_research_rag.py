"""Evidence-grounded Q&A over the BASS-II knowledge base.

answer_with_evidence(query, top_k=3, use_llm=False, use_web=False)
    -> {"summary", "relevant_experiments", "evidence", "web_results"}
"""
import re
from pathlib import Path

from data_processing.data_cleaning import clean_experiments
from data_processing.nasa_data_loader import load_experiments
from rag import prompts

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT / "data" / "documents"
EXPERIMENTS_PATH = ROOT / "data" / "experiments" / "bass2_experimental_table.csv"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "what", "which", "how", "does", "do", "is", "are", "was", "were",
    "about", "under", "between", "from", "that", "this", "these", "those",
    "there", "their", "they", "them", "then", "than", "such", "into",
    "ii", "iii", "iv", "test", "tests", "tested", "bass", "bassi",
}

_store = None
_experiments_df = None


def _get_store():
    global _store
    if _store is None:
        from rag.vector_database import SimpleVectorStore
        from rag.document_ingestion import load_documents_with_pages
        _store = SimpleVectorStore()
        docs = load_documents_with_pages(DOCUMENTS_DIR) if DOCUMENTS_DIR.exists() else []
        _store.add_documents(docs)
    return _store


def _get_experiments():
    global _experiments_df
    if _experiments_df is None:
        _experiments_df = clean_experiments(load_experiments(EXPERIMENTS_PATH))
    return _experiments_df


def _keywords(query):
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", query.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def _word_in(text, word):
    forms = {word}
    if word.endswith("s") and len(word) > 3:
        forms.add(word[:-1])
    return any(re.search(r"\b" + re.escape(f) + r"\b", text) for f in forms)


def rank_experiments(query, top_k=5):
    keywords = _keywords(query)
    df = _get_experiments()
    ranked = []
    for _, row in df.iterrows():
        haystack = " ".join([
            str(row["test_id"]),
            str(row["fuel_material"]),
            str(row["o2_depletion_band"]),
        ]).lower()
        matched = sorted({kw for kw in keywords if _word_in(haystack, kw)})
        if matched:
            ranked.append({
                "test_id": row["test_id"],
                "experiment_id": row["test_id"],
                "fuel": row["fuel_material"],
                "fuel_material": row["fuel_material"],
                "oxygen_pct": round(float(row["initial_o2"]), 1),
                "initial_o2": round(float(row["initial_o2"]), 1),
                "o2_depletion_band": str(row["o2_depletion_band"]),
                "behavior": str(row["o2_depletion_band"]),
                "score": len(matched),
                "why_relevant": (
                    f"Matched: {', '.join(matched)}. "
                    f"BASS-II test {row['test_id']}: {row['fuel_material']} "
                    f"at {row['initial_o2']:.1f}% initial O2, burn-intensity "
                    f"proxy band '{row['o2_depletion_band']}'."
                ),
            })
    ranked.sort(key=lambda r: (-r["score"], str(r["experiment_id"])))
    return ranked[:top_k]


def _direct_answer(query):
    df = _get_experiments()
    q = query.lower()

    m = re.search(r"\b([bfm]\d{1,3})\b", q)
    if m:
        tid = m.group(1).upper()
        hit = df[df["test_id"].str.upper() == tid]
        if not hit.empty:
            r = hit.iloc[0]
            bands = df["o2_depletion_band"].value_counts()
            return (
                f"Test {r['test_id']} burned {r['fuel_material']}. "
                f"It started with {r['initial_o2']:.1f}% oxygen and ended with "
                f"{r['final_o2']:.1f}% — so the flame consumed {r['o2_depletion']:.1f} "
                f"percentage points of oxygen (this consumed amount is called "
                f"O2 depletion). That puts it in the '{r['o2_depletion_band']}' band. "
                f"(Note: the band is our own grouping of the measured numbers, not a "
                f"NASA label.) Airflow during the test: fan {r['fan_display']:.0f}, "
                f"air {r['air_display']:.1f}, restrictor {r['flow_restrictor']:.0f}. "
                f"For context, all {len(df)} tests split "
                f"{int(bands.get('low depletion', 0))} low / "
                f"{int(bands.get('moderate depletion', 0))} moderate / "
                f"{int(bands.get('high depletion', 0))} high."
            )

    if "fuel" in q:
        counts = df["fuel_material"].value_counts()
        top = "; ".join(f"{m} ({c})" for m, c in counts.head(10).items())
        cats = []
        for label, pat in [("PMMA film (acrylic film)", "film"),
                           ("PMMA rod (acrylic rod)", "rod"),
                           ("SIBAL fabric", "sibal"), ("Nomex", "nomex"),
                           ("candle", "candle")]:
            n = int(df["fuel_material"].str.contains(pat, case=False).sum())
            if n:
                cats.append(f"{label}: {n} tests")
        cat_str = f" In plain terms: {'; '.join(cats)}." if cats else ""
        return (
            f"BASS-II burned {df['fuel_material'].nunique()} different fuel materials "
            f"across {len(df)} tests.{cat_str} "
            f"The most-tested materials: {top}."
        )

    if "how many" in q:
        bands = df["o2_depletion_band"].value_counts()
        return (
            f"The BASS-II table holds {len(df)} tests. By how much oxygen their flames "
            f"consumed, they split into {int(bands.get('low depletion', 0))} low, "
            f"{int(bands.get('moderate depletion', 0))} moderate and "
            f"{int(bands.get('high depletion', 0))} high-consumption tests."
        )

    if "extinct" in q:
        fuel_note = ""
        for label, pat in [("PMMA", "pmma"), ("SIBAL", "sibal"), ("Nomex", "nomex")]:
            if pat in q:
                fuel_note = f" for {label}"
                break
        return (
            f"Straight answer: the BASS-II table records oxygen levels and fuel types "
            f"but has no extinction column, so the exact extinction limit{fuel_note} "
            f"can't be read from the test numbers. What the table does show: initial O2 "
            f"across {len(df)} tests ranged {df['initial_o2'].min():.1f}–"
            f"{df['initial_o2'].max():.1f}% by vol. How extinction relates to oxygen "
            f"and airflow lives in the document excerpts below — a full interpretation "
            f"of those needs the LLM layer, which synthesizes across excerpts instead "
            f"of just quoting them."
        )

    if "oxygen" in q or re.search(r"\bo2\b", q):
        g = df.groupby("o2_depletion_band")["o2_depletion"].agg(["mean", "count"])
        hi = df.nlargest(3, "o2_depletion")
        hi_str = "; ".join(
            f"{r['test_id']} ({r['fuel_material']}: consumed {r['o2_depletion']:.1f} "
            f"points from {r['initial_o2']:.1f}% O2)"
            for _, r in hi.iterrows())
        band_str = "; ".join(
            f"{b}: {int(r['count'])} tests, average {r['mean']:.1f} points consumed"
            for b, r in g.iterrows())
        return (
            f"In plain terms: BASS-II measured how much oxygen each flame consumed "
            f"(starting oxygen minus leftover oxygen = O2 depletion). "
            f"Starting oxygen across {len(df)} tests ranged "
            f"{df['initial_o2'].min():.1f}–{df['initial_o2'].max():.1f}% "
            f"(average {df['initial_o2'].mean():.1f}%). Grouped by consumption: "
            f"{band_str}. The highest consumers: {hi_str}."
        )

    return None


def _tavily_search(query):
    try:
        from rag import web_search
        for name in ("search_nasa_web", "tavily_search", "search"):
            fn = getattr(web_search, name, None)
            if callable(fn):
                return fn(query) or []
    except Exception as exc:
        print(f"Web search unavailable: {exc}")
    return []


def ingest_uploaded_pdf(filename, file_bytes):
    store = _get_store()
    safe = "".join(c for c in str(filename)
                   if c.isalnum() or c in "._-").strip("._-") or "upload.pdf"
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    dest = DOCUMENTS_DIR / safe
    dest.write_bytes(file_bytes)
    from data_processing.pdf_processor import extract_pages_from_pdf
    page_texts = extract_pages_from_pdf(dest)
    full_text = "\n\n".join(t for _, t in page_texts)
    before = len(store)
    store.add_documents([{"source": safe, "text": full_text,
                          "page_texts": page_texts}])
    return {
        "filename": safe,
        "pages": len(page_texts),
        "chars": len(full_text),
        "new_chunks": len(store) - before,
        "text": full_text,
    }


def llm_summary_for_upload(upload_info, n_chunks=8):
    from rag.llm_client import generate_grounded_answer
    store = _get_store()
    own = [c for c in store.chunks
           if c["source"] == upload_info["filename"]][:n_chunks]
    evidence = [{"source": c["source"], "excerpt": c["text"][:600]}
                for c in own]
    return generate_grounded_answer(
        "Summarize this research paper: what was studied, the key methods, "
        "and the main findings. Cite the document for each claim.",
        evidence, [])


def answer_with_evidence(query, top_k=3, use_llm=False, use_web=False):
    query = query.strip()
    if not query:
        return {"summary": "Please type a question first.",
                "relevant_experiments": [], "evidence": [], "web_results": []}

    store = _get_store()
    try:
        evidence = store.query(query, top_k=top_k)
    except Exception:
        evidence = []
    relevant = rank_experiments(query, top_k=top_k)

    web_results = _tavily_search(query) if use_web else []

    llm_summary = None
    if use_llm:
        try:
            from rag.llm_client import generate_grounded_answer
            llm_summary = generate_grounded_answer(query, evidence, relevant)
        except Exception as exc:
            print(f"LLM unavailable; using offline template. ({exc})")

    if llm_summary:
        prefix = ""
        if re.search(r"\b[bfm]\d{1,3}\b", query.lower()):
            direct = _direct_answer(query)
            if direct:
                prefix = direct + "\n\n"
        summary = (prefix + llm_summary.strip()
                   + "\n\n(Generated by your configured LLM provider, grounded only "
                     "in the retrieved evidence.)")
    else:
        parts = []
        direct = _direct_answer(query)
        if direct:
            parts.append(direct)
        elif evidence:
            top = evidence[0]
            src = top.get("source", "unknown")
            if top.get("page"):
                src += f", page {top['page']}"
            parts.append(
                f"From the BASS-II documents:\n\n\"{top.get('excerpt', '').strip()}…\"\n\n"
                f"(Source: {src})"
            )
        if relevant:
            ids = ", ".join(r["experiment_id"] for r in relevant)
            best = max(r["score"] for r in relevant)
            if best >= 2:
                parts.append(f"Most relevant BASS-II tests: {ids}.")
            else:
                parts.append(f"Tests mentioning your keywords: {ids}.")
        if evidence and (direct or len(evidence) > 1):
            parts.append("Further excerpts are shown below.")
        if not parts:
            parts.append("No documents or BASS-II tests matched. Try keywords like "
                         "'extinction', 'oxygen', 'SIBAL', 'PMMA', 'ignition', or a test "
                         "ID like 'B12'.")
        parts.append(prompts.OFFLINE_FOOTER)
        summary = "\n\n".join(parts)

    return {
        "summary": summary,
        "relevant_experiments": relevant,
        "evidence": evidence,
        "web_results": web_results,
    }
