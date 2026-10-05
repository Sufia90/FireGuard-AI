import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXPERIMENTS_CSV = DATA_DIR / "experiments" / "bass2_experimental_table.csv"
DOCUMENTS_DIR = DATA_DIR / "documents"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = ROOT / "models"
MODEL_PKL = MODELS_DIR / "best_model.pkl"


def _read_secret(name):
    try:
        import streamlit as st
        val = st.secrets.get(name)
        if val not in (None, ""):
            return str(val).strip().strip('"')
    except Exception:
        pass
    return os.environ.get(name, "").strip().strip('"')


def get_key(name):
    return _read_secret(name)


GOOGLE_API_KEY = _read_secret("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = _read_secret("HUGGINGFACE_API_KEY")
TAVILY_API_KEY = _read_secret("TAVILY_API_KEY")
LANGSMITH_API_KEY = _read_secret("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = _read_secret("LANGSMITH_PROJECT") or "FireGuard-AI"
LANGSMITH_TRACING = _read_secret("LANGSMITH_TRACING").lower() == "true"

for _name, _val in {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "HUGGINGFACE_API_KEY": HUGGINGFACE_API_KEY,
    "TAVILY_API_KEY": TAVILY_API_KEY,
    "LANGSMITH_API_KEY": LANGSMITH_API_KEY,
    "LANGSMITH_PROJECT": LANGSMITH_PROJECT,
}.items():
    if _val and not os.environ.get(_name):
        os.environ[_name] = _val
if LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"


def is_configured(name):
    return bool({
        "google": GOOGLE_API_KEY,
        "huggingface": HUGGINGFACE_API_KEY,
        "tavily": TAVILY_API_KEY,
        "langsmith": LANGSMITH_API_KEY,
    }.get(name, ""))


def configured_llm_providers():
    providers = []
    if GOOGLE_API_KEY:
        providers.append("google")
    if HUGGINGFACE_API_KEY:
        providers.append("huggingface")
    return providers
