import os

import requests
from dotenv import load_dotenv

import config
from rag import prompts

load_dotenv()

try:
    from langsmith import traceable

    def _traced(fn):
        return traceable(name="fireguard-grounded-answer")(fn)
except Exception:
    def _traced(fn):
        return fn


def _audience(query):
    try:
        return prompts.detect_audience(query) or "newcomer"
    except Exception:
        return "newcomer"


def _system_prompt(query):
    base = (
        "You are FireGuard AI, a research assistant for NASA microgravity "
        "combustion (BASS-II). Answer ONLY from the retrieved evidence given "
        "with the question. Cite the source of each claim. Never invent numbers, "
        "findings, conclusions, or citations. If the evidence does not answer "
        "the question, say so plainly."
    )
    if _audience(query) == "technical":
        return base + (" Write for a specialist: precise terminology, quantitative "
                       "detail with units.")
    return base + (" Write for a newcomer: plain language, short sentences. Define "
                   "terms like 'vitiation' or 'extinction' on first use. Lead with a "
                   "direct answer, then supporting detail.")


def _user_prompt(query, evidence, relevant):
    lines = [f"Question: {query}", "",
             "Retrieved evidence (ground every claim in this):"]
    for i, e in enumerate(evidence or [], 1):
        src = str(e.get("source", "unknown"))
        if e.get("page"):
            src += f", page {e['page']}"
        lines.append(f"[{i}] ({src}) {str(e.get('excerpt', ''))[:900]}")
    for r in relevant or []:
        lines.append(f"- Test {r.get('experiment_id')}: {r.get('why_relevant', '')}")
    lines.append("")
    lines.append("Direct answer first, then cited support. No invented facts.")
    return "\n".join(lines)


def _gemini_model_candidates(key, preferred=None):
    cands = []
    if preferred:
        cands.append(preferred)
    try:
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key},
            timeout=20,
        )
        resp.raise_for_status()
        names = [
            m["name"].split("/")[-1]
            for m in resp.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        cands += [n for n in names if "flash" in n] or names
    except Exception:
        pass
    cands += ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    seen = []
    for c in cands:
        if c not in seen:
            seen.append(c)
    return seen


def _gemini_generate(key, model, system, user):
    tried = []
    last_err = None
    for name in _gemini_model_candidates(key, preferred=model):
        tried.append(name)
        try:
            resp = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{name}:generateContent",
                params={"key": key},
                timeout=90,
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": user}]}],
                },
            )
            if resp.status_code in (400, 404):
                last_err = f"{name}: HTTP {resp.status_code}"
                continue
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip(), name
        except (requests.RequestException, KeyError, IndexError) as exc:
            last_err = f"{name}: {exc}"
    raise RuntimeError(
        "Gemini request failed. Tried models: " + ", ".join(tried)
        + (f" (last error: {last_err}). " if last_err else ". ")
        + "Check that GOOGLE_API_KEY is valid and the Generative Language API is enabled."
    )


def _huggingface_generate(key, model, system, user):
    name = model or "mistralai/Mistral-7B-Instruct-v0.3"
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{name}",
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
        json={
            "inputs": f"{system}\n\n{user}",
            "parameters": {"max_new_tokens": 700, "temperature": 0.3},
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return str(data[0].get("generated_text", "")).strip(), name
    return str(data).strip(), name


@_traced
def generate_grounded_answer(query, evidence=None, relevant_experiments=None,
                             provider=None):
    providers = config.configured_llm_providers()
    provider = (provider or (providers[0] if providers else None) or "").lower()
    system = _system_prompt(query)
    user = _user_prompt(query, evidence, relevant_experiments)
    if provider == "google":
        key = os.getenv("GOOGLE_API_KEY", "").strip().strip('"')
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env.")
        text, _ = _gemini_generate(key, None, system, user)
        return text
    if provider in ("huggingface", "hf"):
        key = os.getenv("HUGGINGFACE_API_KEY", "").strip().strip('"')
        if not key:
            raise RuntimeError("HUGGINGFACE_API_KEY is not set in .env.")
        text, _ = _huggingface_generate(key, None, system, user)
        return text
    raise RuntimeError(
        f"No working LLM provider (requested: {provider or 'none'}). "
        "Add GOOGLE_API_KEY or HUGGINGFACE_API_KEY to .env."
    )
