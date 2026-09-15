"""Real-LLM checks for follow-up context and session isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent import AgentRuntime  # noqa: E402
from llm import LLMClient  # noqa: E402
from rag.embedding import LocalEmbedder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402
from session import ContextManager, SessionManager, SessionStore  # noqa: E402
from tools import ReadDocsTool, ToolRegistry, WebSearchTool  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "test.faiss"
MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "test_chunks.json"
DATABASE_PATH = PROJECT_ROOT / "data" / "sessions.db"
SYSTEM_MESSAGE = (
    "You are MiniRAGent. Use facts already stated in this session to answer follow-up "
    "questions. Do not use tools for facts the user explicitly supplied. Use project "
    "documents or web tools only when the question actually requires them."
)


def build_manager() -> SessionManager:
    llm = LLMClient()
    embedder = LocalEmbedder(MODEL_PATH)
    store = SessionStore(DATABASE_PATH)
    registry = ToolRegistry()
    registry.register(ReadDocsTool(Retriever(embedder, FaissVectorStore.load(INDEX_PATH, MAPPING_PATH))))
    registry.register(WebSearchTool())
    runtime = AgentRuntime(llm, registry, system_message=SYSTEM_MESSAGE)
    context = ContextManager(store, llm, compact_after_messages=12, recent_user_turns=3)
    return SessionManager(runtime, store, context)


def require_text(label: str, result: object, expected: str) -> None:
    answer = getattr(result, "final_answer", None) or ""
    print(f"{label}: {answer}")
    if expected.casefold() not in answer.casefold():
        raise AssertionError(f"Expected {expected!r} in {label} answer")


def main() -> None:
    manager = build_manager()
    run_id = uuid4().hex[:10]
    user_id = f"context-user-{run_id}"

    followup_session = "followup"
    first = manager.chat(
        user_id,
        followup_session,
        "Remember that my demo project's codename is Blue Lantern. Reply with just the codename.",
    )
    second = manager.chat(user_id, followup_session, "What was it? Reply with just the codename.")
    require_text("same-session first answer", first, "Blue Lantern")
    require_text("same-session pronoun follow-up", second, "Blue Lantern")

    manager.chat(user_id, "window-a", "For this chat, remember that the assigned city is Paris.")
    manager.chat(user_id, "window-b", "For this chat, remember that the assigned city is Tokyo.")
    answer_a = manager.chat(user_id, "window-a", "What city did I assign to this chat?")
    answer_b = manager.chat(user_id, "window-b", "What city did I assign to this chat?")
    require_text("window-a answer", answer_a, "Paris")
    require_text("window-b answer", answer_b, "Tokyo")

    print(
        json.dumps(
            {
                "user_id": user_id,
                "same_session_context": "passed",
                "different_session_isolation": "passed",
                "database": str(DATABASE_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
