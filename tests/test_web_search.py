"""Tests for normalized web search output and safe failures."""

from __future__ import annotations

from tools.web_search import WebSearchTool


class FakeTavilyClient:
    def search(self, **kwargs: object) -> dict[str, object]:
        return {
            "answer": "unused raw field",
            "results": [
                {
                    "title": "Agent systems",
                    "url": "https://example.com/agents",
                    "content": "A report about AI agent systems.",
                    "score": 0.91,
                    "raw_content": "must not be returned",
                },
                {"title": None, "url": None},
            ],
        }


class FailingTavilyClient:
    def search(self, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("network unavailable")


def test_web_search_normalizes_results() -> None:
    result = WebSearchTool(client=FakeTavilyClient()).execute(query="AI agents", max_results=2)
    assert result == [
        {
            "title": "Agent systems",
            "url": "https://example.com/agents",
            "content": "A report about AI agent systems.",
            "score": 0.91,
        },
        {"title": "", "url": "", "content": "", "score": None},
    ]


def test_web_search_returns_clear_validation_errors() -> None:
    tool = WebSearchTool(client=FakeTavilyClient())
    assert tool.execute(query="")["error"]["type"] == "invalid_query"
    assert tool.execute(query="valid", max_results=0)["error"]["type"] == "invalid_max_results"
    assert tool.execute(query="valid", max_results=True)["error"]["type"] == "invalid_max_results"


def test_web_search_returns_network_error() -> None:
    result = WebSearchTool(client=FailingTavilyClient()).execute(query="AI agents")
    assert result["error"]["type"] == "search_failed"
