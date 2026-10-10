from langchain_core.tools import tool

from core.config import WEB_SEARCH_MAX_RESULTS


@tool
def web_search(query: str) -> str:
    """Search the live web for current information. Free, no API key (DuckDuckGo)."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))

    if not results:
        return "No results found."

    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('body', '')}\nSource: {r.get('href', '')}"
        for r in results
    )
