"""Tavily web search for the Research Assistant.

Lets FireGuard AI find NASA sources on the web (papers, experiment pages,
technical reports) to complement the local document index. Requires
TAVILY_API_KEY in .env (loaded via config.py). Without a key, every
function returns [] and the app simply skips web results — nothing breaks.
"""
import config


def is_available() -> bool:
    """True when a Tavily key is configured and the package is installed."""
    if not config.is_configured("tavily"):
        return False
    try:
        import tavily  # noqa: F401
        return True
    except ImportError:
        print("tavily-python is not installed. Run: pip install tavily-python")
        return False


def _get_client():
    from tavily import TavilyClient
    return TavilyClient(api_key=config.TAVILY_API_KEY)


def tavily_search(query: str, max_results: int = 5,
                  search_depth: str = "advanced") -> list:
    """Web search biased toward NASA/space sources.

    Returns [{title, url, snippet, score}] — snippets only, never full
    article text, so results stay clearly labeled as web search hits.
    """
    if not is_available():
        return []
    try:
        client = _get_client()
        response = client.search(
            query=f"NASA microgravity combustion {query}",
            search_depth=search_depth,
            max_results=max_results,
            include_domains=["nasa.gov", "ntrs.nasa.gov"],
        )
    except Exception as exc:
        print(f"Tavily search failed: {exc}")
        return []
    results = []
    for hit in response.get("results", []):
        results.append({
            "title": hit.get("title", "Untitled"),
            "url": hit.get("url", ""),
            "snippet": (hit.get("content", "") or "")[:400].strip(),
            "score": float(hit.get("score", 0.0)),
        })
    return results


def format_web_results(results: list) -> str:
    """Render web hits as markdown for the dashboard."""
    if not results:
        return ""
    lines = ["### 🌐 Web sources (Tavily)"]
    for r in results:
        lines.append(f"- [{r['title']}]({r['url']})")
        if r["snippet"]:
            lines.append(f"  > {r['snippet']}")
    return "\n".join(lines)
