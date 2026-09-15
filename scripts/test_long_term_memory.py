"""Real API test for cross-session recall and temporary-search exclusion."""

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
from session import (  # noqa: E402
    ContextManager,
    LongTermMemoryManager,
    LongTermMemoryStore,
    SessionManager,
    SessionStore,
)
from tools import ReadDocsTool, ToolRegistry, WebSearchTool  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "test.faiss"
MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "test_chunks.json"
DATABASE_PATH = PROJECT_ROOT / "data" / "sessions.db"
SYSTEM_MESSAGE = (
    "Use relevant session context and long-term user memories when answering. Use tools "
    "only when required by the request. Do not claim knowledge from another session "
    "unless it appears in the supplied long-term memories."
)


def build_manager() -> tuple[SessionManager, LongTermMemoryStore]:
    llm = LLMClient()
    embedder = LocalEmbedder(MODEL_PATH)
    session_store = SessionStore(DATABASE_PATH)
    memory_store = LongTermMemoryStore(session_store)
    registry = ToolRegistry()
    registry.register(ReadDocsTool(Retriever(embedder, FaissVectorStore.load(INDEX_PATH, MAPPING_PATH))))
    registry.register(WebSearchTool())
    runtime = AgentRuntime(llm, registry, system_message=SYSTEM_MESSAGE)
    context = ContextManager(session_store, llm, compact_after_messages=12, recent_user_turns=3)
    memory = LongTermMemoryManager(memory_store, llm)
    return SessionManager(runtime, session_store, context, memory), memory_store


def main() -> None:
    manager, memory_store = build_manager()
    user_id = f"memory-user-{uuid4().hex[:10]}"

    manager.chat(user_id, "session-a", "My current project is called MiniRAGent.")
    stored = memory_store.all_for_user(user_id)
    print("memory after session-a:")
    print(json.dumps([m.__dict__ for m in stored], ensure_ascii=False, indent=2))
    if not any("MiniRAGent" in memory.content for memory in stored):
        raise AssertionError("The project fact was not saved to long-term memory")

    recalled = manager.chat(user_id, "session-b", "What is the name of my Agent project?")
    print(f"cross-session answer: {recalled.final_answer}")
    if "miniragent" not in (recalled.final_answer or "").casefold():
        raise AssertionError("The project name was not recalled across sessions")

    before_search = len(memory_store.all_for_user(user_id))
    search = manager.chat(
        user_id,
        "session-a",
        "Search the web for today's weather in Shanghai and summarize it.",
    )
    after_search = len(memory_store.all_for_user(user_id))
    print(f"temporary search answer: {search.final_answer}")
    print(f"memory count before search={before_search}, after search={after_search}")
    if before_search != after_search:
        raise AssertionError("Temporary web search was incorrectly saved as long-term memory")

    print(
        json.dumps(
            {
                "user_id": user_id,
                "cross_session_recall": "passed",
                "temporary_search_exclusion": "passed",
                "long_term_memories": [m.content for m in memory_store.all_for_user(user_id)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
