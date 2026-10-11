from langchain_core.tools import tool

from core.config import WEB_SEARCH_MAX_RESULTS


@tool
def web_search(query: str) -> str:
    """Search the live web via DuckDuckGo — free, no API key.

    This is the ONLY way to get current information (news, prices, scores,
    schedules, recent events, anything that changes over time). You must call
    this tool for such questions instead of answering from memory, since your
    training data has a cutoff and cannot reflect what's happening now.
    Call it more than once with different, more specific queries if one
    search isn't enough to answer fully (e.g. a request for several items).
    """
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))
    except DDGSException as e:
        # ddgs.text() raises (rather than returning []) when it finds nothing
        # or gets rate-limited — surface that as a tool observation instead of
        # letting it crash the whole turn, so the model can retry/rephrase.
        return (
            f"Search failed or returned no results ({e}). Try a different, "
            "simpler, or more specific query before giving up."
        )

    if not results:
        return "No results found. Try a different, simpler, or more specific query."

    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('body', '')}\nSource: {r.get('href', '')}"
        for r in results
    )
