"""AI Research Assistant tab: PDF upload + evidence-grounded Q&A.

Features:
  - Upload a PDF → text-extracted, chunked, indexed immediately
  - Extractive summary (TF-IDF, offline) + optional LLM abstractive summary
  - Q&A over BASS-II tests + documents, with ranked experiments and
    page-tagged evidence excerpts
  - Optional Tavily web search for NASA sources (needs TAVILY_API_KEY)
  - LLM provider picker (Google Gemini / HuggingFace — from config.py)
"""
import streamlit as st

import config
from rag import nasa_research_rag, summarizer
from rag import web_search


def _get_nasa_links():
    import pandas as pd
    links = pd.read_csv(config.METADATA_DIR / "nasa_links.csv")
    links = links[links["url"].notna() & (links["url"].str.strip() != "")]
    return links


def _retrieve_only(query):
    """Retrieve evidence + ranked tests without generating an answer."""
    store = nasa_research_rag._get_store()
    return {
        "relevant_experiments": nasa_research_rag.rank_experiments(query),
        "evidence": store.query(query, top_k=3),
        "web_results": [],
    }


def render():
    st.header("🔎 AI Research Assistant")
    st.write("Ask about BASS-II research. The Q&A searches the BASS-II test "
             "table, `SRD_BASS-II.pdf`, and any papers you upload below — "
             "optionally the web too. Every answer lists its sources.")

    # ---- PDF upload ------------------------------------------------------
    st.subheader("Upload research paper (PDF)")
    uploaded = st.file_uploader("Upload research paper (PDF)", type=["pdf"])
    if uploaded is not None:
        if "ingested_uploads" not in st.session_state:
            st.session_state.ingested_uploads = set()
        if uploaded.name not in st.session_state.ingested_uploads:
            with st.spinner("Extracting text and indexing..."):
                info = nasa_research_rag.ingest_uploaded_pdf(
                    uploaded.name, uploaded.getvalue())
            st.session_state.ingested_uploads.add(uploaded.name)
            st.session_state.last_upload_info = info
        info = st.session_state.get("last_upload_info")
        if info:
            st.success(f"Indexed **{info['filename']}**: {info['pages']} pages, "
                       f"{info['new_chunks']} chunks added to the search index.")
            st.subheader("Extractive summary (TF-IDF, offline)")
            st.caption("Top sentences ranked by TF-IDF — no LLM involved.")
            for sentence in summarizer.extractive_summary(info["text"]):
                st.write("• " + sentence)
            if config.configured_llm_providers() and st.checkbox(
                    "Also generate an abstractive summary with the LLM layer"):
                try:
                    abstract = nasa_research_rag.llm_summary_for_upload(info)
                    st.subheader("Abstractive summary (LLM-generated, "
                                 "grounded in the paper)")
                    st.write(abstract)
                except Exception as exc:
                    st.info(f"LLM summary unavailable: {exc}")

    # ---- Q&A -------------------------------------------------------------
    st.subheader("Ask a question")
    query = st.text_input("Ask FireGuard AI",
                          placeholder="What fuels were tested in BASS-II?")

    providers = config.configured_llm_providers()
    c1, c2 = st.columns(2)
    with c1:
        use_llm = st.checkbox(
            "✨ Generate the answer with an LLM",
            value=False,
            help="Grounded strictly in retrieved evidence. "
                 "Without a key, the offline template is used.",
            disabled=not providers,
        )
        if not providers:
            st.caption("No LLM key in .env — offline template mode.")
    with c2:
        use_web = st.checkbox(
            "🌐 Also search the web (Tavily)",
            value=False,
            help="Finds NASA sources online to complement local evidence.",
            disabled=not config.is_configured("tavily"),
        )
        if not config.is_configured("tavily"):
            st.caption("No TAVILY_API_KEY in .env — local evidence only.")

    provider = None
    if use_llm and len(providers) > 1:
        provider = st.selectbox("LLM provider", providers)

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Retrieving evidence..."):
            if use_llm and provider:
                # Explicit provider chosen: retrieve first, then generate.
                from rag.llm_client import generate_grounded_answer
                parts = _retrieve_only(query)
                result = {
                    "summary": generate_grounded_answer(
                        query, parts["evidence"],
                        parts["relevant_experiments"],
                        provider=provider),
                    **parts,
                }
            else:
                result = nasa_research_rag.answer_with_evidence(
                    query, use_llm=use_llm, use_web=use_web)

        st.subheader("Answer")
        st.write(result["summary"])

        if result["relevant_experiments"]:
            st.subheader("Relevant BASS-II tests (ranked)")
            for r in result["relevant_experiments"]:
                with st.expander(
                        f"Test {r['experiment_id']} — {r['behavior']} "
                        f"(relevance: {r['score']})"):
                    st.write(r["why_relevant"])

        if result["evidence"]:
            st.subheader("Evidence excerpts")
            for e in result["evidence"]:
                page = e.get("page")
                location = e["source"]
                if page:
                    location += f" — page {page}"
                location += f", chunk {e.get('chunk_id')}"
                st.markdown(f"**Source:** `{location}` "
                            f"(similarity {e['score']:.2f})")
                st.caption(e["excerpt"] + "…")

        if result.get("web_results"):
            st.markdown(web_search.format_web_results(result["web_results"]))

    with st.expander("NASA sources"):
        st.write("Verified starting points for NASA research (the team adds "
                 "more in `data/metadata/nasa_links.csv`):")
        for _, row in _get_nasa_links().iterrows():
            st.markdown(
                f"- [{row['title']}]({row['url']}) — {row['description']}")
