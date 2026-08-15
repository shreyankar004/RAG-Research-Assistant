"""
Web Search Agent using Tavily for general research mode.
Real web search with source citations.
"""
from dataclasses import dataclass
from typing import List

import streamlit as st

from config import TAVILY_API_KEY
@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float


@dataclass
class WebSearchResult:
    ok: bool
    results: List[SearchResult]
    query: str
    error: str = ""

    @property
    def combined_context(self) -> str:
        parts = []
        for i, r in enumerate(self.results, 1):
            parts.append(f"[Source {i}] {r.title}\nURL: {r.url}\n{r.content}")
        return "\n\n---\n\n".join(parts)

    @property
    def sources_markdown(self) -> str:
        lines = []
        for i, r in enumerate(self.results, 1):
            lines.append(f"{i}. [{r.title}]({r.url})")
        return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def load_tavily_client():
    if not TAVILY_API_KEY:
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=TAVILY_API_KEY)
    except ImportError:
        return None


class WebSearchAgent:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = load_tavily_client()
        return self._client

    def search(self, query: str, max_results: int = 6) -> WebSearchResult:
        if self.client is None:
            return WebSearchResult(
                ok=False,
                results=[],
                query=query,
                error="TAVILY_API_KEY is missing. Add it to .env for web search."
            )
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
            )
            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                )
                for r in response.get("results", [])
            ]
            return WebSearchResult(ok=True, results=results, query=query)
        except Exception as exc:
            return WebSearchResult(ok=False, results=[], query=query, error=str(exc))
