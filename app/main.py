import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import config
from app import (
    experiment_comparison,
    experiment_explorer,
    fire_safety_insights,
    ml_analysis,
    research_assistant,
)
from data_processing.data_cleaning import clean_experiments
from data_processing.nasa_data_loader import load_experiments
from ml.train_model import load_or_train_model

st.set_page_config(
    page_title="FireGuard AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPACE_CSS = """
<style>
.stApp {
    background: radial-gradient(1200px 600px at 15% -5%, #1b2a4a 0%, transparent 60%),
                radial-gradient(1000px 500px at 90% 10%, #3a1c4d 0%, transparent 55%),
                linear-gradient(180deg, #0b1020 0%, #0d1428 100%);
    color: #e8eefc;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a33 0%, #0b1226 100%);
    border-right: 1px solid rgba(120,160,255,.15);
}
h1, h2, h3 { color: #f2f6ff !important; letter-spacing: -0.01em; }
h1 {
    background: linear-gradient(90deg, #ff9a5c, #ff5c7a, #7aa2ff);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}
div[data-testid="stMetric"] {
    background: rgba(20,30,60,.65);
    border: 1px solid rgba(120,160,255,.18);
    border-radius: 14px;
    padding: 14px 16px;
    backdrop-filter: blur(6px);
}
div[data-testid="stMetric"] label { color: #9fb2dd !important; }
button[data-baseweb="tab"] {
    color: #9fb2dd !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #7aa2ff !important;
}
button[kind="primary"] {
    background: linear-gradient(90deg, #ff7a45, #ff4d6d) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
div[data-testid="stExpander"] { border: 1px solid rgba(120,160,255,.18); border-radius: 12px; }
.stAlert { border-radius: 12px; }
.stCaption, small { color: #9fb2dd !important; }
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)

DERIVED_LABEL_NOTE = (
    "🎯 **About the ML target:** the BASS-II table has no flame-behavior label, "
    "so the model predicts a **burn-intensity proxy** — O₂ depletion "
    "(initial O₂ − final O₂, both measured % by vol) binned into low / moderate / "
    "high tertile bands. This band is derived by us, **not** a NASA-provided label."
)


@st.cache_data
def get_experiments():
    df = load_experiments(config.EXPERIMENTS_CSV)
    return clean_experiments(df)


@st.cache_resource
def get_model_bundle():
    return load_or_train_model(config.MODEL_PKL)


with st.sidebar:
    st.markdown("## 🔥 FireGuard AI")
    st.caption("NASA microgravity combustion research intelligence")
    st.divider()
    st.markdown("### ⚙️ Integrations")
    llm_providers = config.configured_llm_providers()
    st.write(f"🤖 LLM: "
             f"{', '.join(llm_providers) if llm_providers else 'offline mode'}")
    st.write(f"🌐 Web search: "
             f"{'Tavily ✓' if config.is_configured('tavily') else '—'}")
    st.write(f"🔭 Tracing: "
             f"{'LangSmith ✓' if config.is_configured('langsmith') else '—'}")
    st.divider()
    st.caption("Search → Compare → Analyze → Explain → Evidence")

df = get_experiments()
bundle = get_model_bundle()

DATA_BANNER = (
    f"🛰️ **Real BASS-II data:** {len(df)} experiment tests from the NASA PSI "
    "BASS-II experimental table, plus the BASS-II Science Requirements "
    "Document (`SRD_BASS-II.pdf`)."
)

PIPELINE = (
    f"BASS-II table ({len(df)} tests) + SRD_BASS-II.pdf + uploaded papers\n"
    "        │\n"
    " Data processing (cleaning, O2-depletion proxy, PDF extract)\n"
    "        │\n"
    " ┌──────┴──────┐\n"
    " RAG           ML\n"
    " (TF-IDF)      (model comparison → best kept)\n"
    " └──────┬──────┘\n"
    " Evidence-grounded synthesis (optional LLM: Gemini / HuggingFace)\n"
    "        │\n"
    " FireGuard dashboard (this app)"
)

st.title("🔥 FireGuard AI")
st.caption("NASA microgravity combustion research intelligence — BASS-II edition")
st.info(DATA_BANNER)
st.warning(DERIVED_LABEL_NOTE)

tabs = st.tabs([
    "🏠 Home",
    "🔎 AI Research Assistant",
    "📊 Experiment Explorer",
    "🔄 Experiment Comparison",
    "🤖 ML Analysis",
    "🔥 Fire-Safety Insights",
])

with tabs[0]:
    st.header("Mission overview")
    st.write(
        "FireGuard AI turns scattered NASA microgravity combustion research "
        "into one searchable, comparable, analyzable environment: "
        "**Search → Compare → Analyze → Explain → Evidence**."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BASS-II tests", len(df))
    c2.metric("Fuel materials", df["fuel_material"].nunique())
    c3.metric("Initial O₂ range",
              f"{df['initial_o2'].min():.1f}–{df['initial_o2'].max():.1f}%")
    c4.metric("Depletion bands", df["o2_depletion_band"].nunique())
    st.caption(f"Winning model: **{bundle.get('model_name', 'unknown')}** "
               f"({bundle.get('cv_accuracy', 0):.1%} CV accuracy)")

    st.subheader("Pipeline")
    st.code(PIPELINE, language="text")
    st.subheader("Data inventory")
    st.write("The team's research table lives at "
             "`data/metadata/data_inventory.csv`:")
    st.dataframe(pd.read_csv(config.METADATA_DIR / "data_inventory.csv"),
                 use_container_width=True)
    srd_pdf = config.DOCUMENTS_DIR / "SRD_BASS-II.pdf"
    if srd_pdf.exists():
        st.download_button(
            "📄 Download SRD_BASS-II.pdf (source document)",
            data=srd_pdf.read_bytes(),
            file_name="SRD_BASS-II.pdf",
            mime="application/pdf",
        )

with tabs[1]:
    research_assistant.render()

with tabs[2]:
    experiment_explorer.render(df)

with tabs[3]:
    experiment_comparison.render(df)

with tabs[4]:
    ml_analysis.render(df, bundle)

with tabs[5]:
    fire_safety_insights.render(df)
