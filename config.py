"""FireGuard AI — central configuration.

Loads all API keys and settings from the `.env` file in the project root.
Every other module imports from here instead of reading environment
variables directly, so there is exactly one place to check what is
configured.

Expected `.env` keys (see `.env.example`):
    HUGGINGFACE_API_KEY   — Hugging Face Inference API (LLM answers)
    GOOGLE_API_KEY        — Google Gemini (LLM answers)
    TAVILY_API_KEY        — Tavily web search (find NASA sources online)
    LANGSMITH_API_KEY     — LangSmith tracing (optional observability)
    LANGSMITH_PROJECT     — LangSmith project name (default: FireGuard-AI)
    LANGSMITH_TRACING     — "true"/"false": enable LangSmith tracing

Nothing secret is ever printed by this module — status reports show only
whether a key is present, never its value.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load .env (silently skipped if python-dotenv is not installed)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is in requirements.txt; app works without it
    pass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# ---------------------------------------------------------------------------
# API keys (values stay in memory only — never log or print these)
# ---------------------------------------------------------------------------
HUGGINGFACE_API_KEY = _get("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")
TAVILY_API_KEY = _get("TAVILY_API_KEY")
LANGSMITH_API_KEY = _get("LANGSMITH_API_KEY")

# ---------------------------------------------------------------------------
# LangSmith tracing settings
# ---------------------------------------------------------------------------
LANGSMITH_PROJECT = _get("LANGSMITH_PROJECT", "FireGuard-AI")
LANGSMITH_TRACING = _get("LANGSMITH_TRACING", "false").lower() in {
    "1", "true", "yes", "on",
}

# Wire LangSmith into LangChain-style tracing when enabled and keyed.
# (Harmless when langsmith is not installed — tracing calls no-op.)
if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)

# ---------------------------------------------------------------------------
# Model defaults (overridable via .env)
# ---------------------------------------------------------------------------
GEMINI_MODEL = _get("FIREGUARD_GEMINI_MODEL", "gemini-2.0-flash")
HUGGINGFACE_MODEL = _get(
    "FIREGUARD_HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
DOCUMENTS_DIR = DATA_DIR / "documents"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = ROOT / "models"

EXPERIMENTS_CSV = EXPERIMENTS_DIR / "bass2_experimental_table.csv"
MODEL_PKL = MODELS_DIR / "fireguard_model.pkl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LLM_PROVIDERS = ("google", "huggingface")


def is_configured(name: str) -> bool:
    """True if the named setting has a non-empty value.

    Valid names: 'google', 'huggingface', 'tavily', 'langsmith'.
    """
    mapping = {
        "google": GOOGLE_API_KEY,
        "huggingface": HUGGINGFACE_API_KEY,
        "tavily": TAVILY_API_KEY,
        "langsmith": LANGSMITH_API_KEY and LANGSMITH_TRACING,
    }
    if name not in mapping:
        raise ValueError(f"Unknown config name: {name!r}. "
                         f"Choose from {sorted(mapping)}.")
    return bool(mapping[name])


def configured_llm_providers() -> list:
    """LLM providers with a usable API key, in preference order."""
    return [p for p in LLM_PROVIDERS if is_configured(p)]


def status_report() -> str:
    """Human-readable summary of what is configured (no secrets shown)."""
    lines = ["FireGuard AI configuration status:"]
    lines.append(f"  Google Gemini (LLM) : {'✓ configured' if is_configured('google') else '— not set'}")
    lines.append(f"  HuggingFace (LLM)   : {'✓ configured' if is_configured('huggingface') else '— not set'}")
    lines.append(f"  Tavily (web search) : {'✓ configured' if is_configured('tavily') else '— not set'}")
    lines.append(f"  LangSmith (tracing) : {'✓ enabled' if is_configured('langsmith') else '— disabled'}")
    lines.append(f"  Project root        : {ROOT}")
    if not configured_llm_providers():
        lines.append("  Note: no LLM key set — the app runs fully offline "
                     "with template answers.")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick check:  python config.py
    print(status_report())
