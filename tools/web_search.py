"""Public web search tool backed by the Tavily Search API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from dotenv import load_dotenv
import requests
from tavily import TavilyClient

from .base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the public web for current or external information relevant to a user query."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to send to the public web search service.",
                "minLength": 1,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return.",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: TavilyClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> TavilyClient:
        if self._client is not None:
            return self._client

        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path)
        api_key = (self._api_key or os.getenv("TAVILY_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("TAVILY_API_KEY is missing from the project .env file")
        session = requests.Session()
        session.trust_env = False
        self._client = TavilyClient(api_key=api_key, session=session)
        return self._client

    @staticmethod
    def _error(error_type: str, message: str) -> dict[str, Any]:
        return {"error": {"type": error_type, "message": message}}

    def execute(
        self,
        *,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        if kwargs:
            return self._error("invalid_arguments", f"Unexpected arguments: {', '.join(kwargs)}")
        if not isinstance(query, str) or not query.strip():
            return self._error("invalid_query", "query must be a non-empty string")
        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or not 1 <= max_results <= 20
        ):
            return self._error("invalid_max_results", "max_results must be an integer from 1 to 20")

        try:
            response = self._get_client().search(
                query=query.strip(),
                max_results=max_results,
                search_depth="basic",
                timeout=20,
            )
        except ValueError as exc:
            return self._error("configuration_error", str(exc))
        except Exception as exc:
            return self._error("search_failed", f"Tavily search failed: {exc}")

        if not isinstance(response, dict):
            return self._error("invalid_response", "Tavily returned an unexpected response type")
        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            return self._error("invalid_response", "Tavily response did not contain a results list")

        results: list[dict[str, Any]] = []
        for item in raw_results[:max_results]:
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            results.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    # Search pages can contain tens of thousands of characters. The
                    # model only needs a focused excerpt to summarize the result.
                    "content": str(item.get("content") or "")[:1200],
                    "score": float(score) if isinstance(score, (int, float)) else None,
                }
            )
        return results
