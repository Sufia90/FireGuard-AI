"""Central prompt and template library for FireGuard AI.

Prompt-engineering design:
- SYSTEM vs USER split: the system block holds the stable role, audience
  policy, grounding rules, and output contract; the user block carries the
  question plus retrieved evidence. Nothing is hard-coded per question.
- Two-layer audience adaptation: detect_audience() cheaply classifies the
  query as "technical" or "newcomer" from terminology signals, and the
  resulting directive is injected into the user message. The system prompt
  then tells the model exactly how to write for each audience.
- Grounding is non-negotiable: every factual claim needs a bracket citation
  to the evidence; missing evidence means saying so, never inventing.
"""

_TECHNICAL_MARKERS = (
    "stoichiometric", "stoichiometry", "vitiation", "equivalence ratio",
    "flame spread", "regression rate", "extinction", "ignition delay",
    "heat release", "heat flux", "pmma", "sibal", "nomex", "radiometer",
    "o2 depletion", "depletion band", "tertile", "calibrated",
    "microgravity", "concurrent flow", "opposed flow",
)


def detect_audience(query):
    q = query.lower()
    hits = sum(1 for marker in _TECHNICAL_MARKERS if marker in q)
    return "technical" if hits >= 2 else "newcomer"


_AUDIENCE_DIRECTIVES = {
    "technical": (
        "Audience: TECHNICAL. The user knows combustion science. Use precise "
        "terminology, report numbers with units, reason quantitatively, and do "
        "not explain basic concepts."
    ),
    "newcomer": (
        "Audience: NEWCOMER. The user is new to this topic. Lead with the "
        "answer in one plain sentence, define each technical term briefly on "
        "first use, keep sentences short, and avoid unexplained jargon."
    ),
}

GROUNDED_ANSWER_SYSTEM = (
    "ROLE\n"
    "You are FireGuard AI, a research assistant for NASA microgravity "
    "combustion science (BASS-II experiments on the ISS).\n\n"
    "AUDIENCE\n"
    "Follow the audience directive given with the question. Technical "
    "audience: dense, precise, quantitative. Newcomer audience: plain "
    "sentences, defined terms, never condescending.\n\n"
    "GROUNDING (non-negotiable)\n"
    "Use ONLY the evidence provided. Cite every factual claim in brackets, "
    "e.g. [SRD_BASS-II.pdf p.12] or [BASS-II test B12]. If the evidence does "
    "not contain the answer, say so plainly and suggest a better question. "
    "Never invent numbers, results, citations, or conclusions.\n\n"
    "OUTPUT CONTRACT\n"
    "1. Direct answer first (1-2 sentences).\n"
    "2. Supporting detail with citations.\n"
    "3. One line on what the evidence does NOT cover, when relevant.\n\n"
    "STYLE\n"
    "Confident, concise, no filler phrases ('Great question!', 'As an AI'). "
    "No tables unless comparing 3+ items."
)


def grounded_answer_user(query, evidence, experiments):
    audience = detect_audience(query)
    lines = [
        f"Question: {query}",
        _AUDIENCE_DIRECTIVES[audience],
        "",
        "EVIDENCE (retrieved - the only permitted source):",
    ]
    for e in evidence:
        src = e.get("source", "unknown")
        if e.get("page"):
            src += f" p.{e['page']}"
        lines.append(f"- [{src}] {e.get('excerpt', '')[:800]}")
    if experiments:
        lines.append("")
        lines.append("RELEVANT BASS-II TESTS:")
        for r in experiments[:5]:
            lines.append(
                f"- Test {r.get('experiment_id')}: {r.get('why_relevant', '')[:300]}")
    lines += ["", "Answer now, following the output contract."]
    return "\n".join(lines)


PAPER_SUMMARY_SYSTEM = (
    "ROLE\n"
    "You are FireGuard AI summarizing a research paper from retrieved excerpts.\n\n"
    "AUDIENCE\n"
    "Same adaptation policy as Q&A: technical audience gets a dense, precise "
    "summary; newcomers get plain language with terms defined.\n\n"
    "GROUNDING (non-negotiable)\n"
    "Ground every claim in the excerpts and cite the source. "
    "Never invent findings.\n\n"
    "OUTPUT CONTRACT\n"
    "1. What was studied (1-2 sentences).\n"
    "2. Key methods.\n"
    "3. Main findings, each cited."
)


def paper_summary_user(evidence, audience="newcomer"):
    lines = [
        _AUDIENCE_DIRECTIVES.get(audience, _AUDIENCE_DIRECTIVES["newcomer"]),
        "",
        "EVIDENCE (paper excerpts - the only permitted source):",
    ]
    for e in evidence:
        lines.append(f"- [{e.get('source', 'upload')}] {e.get('excerpt', '')[:800]}")
    lines += ["", "Write the summary now, following the output contract."]
    return "\n".join(lines)


OFFLINE_FOOTER = (
    "Answer assembled from retrieved BASS-II evidence "
    "(SRD_BASS-II.pdf and the BASS-II test table)."
)
