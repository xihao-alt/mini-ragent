"""FastAPI backend and web UI for MiniRAGent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import AgentRuntime
from llm import LLMClient
from rag import KnowledgeBase, LocalEmbedder, load_pdf
from session import (
    ContextManager,
    HandoffStore,
    LongTermMemoryManager,
    LongTermMemoryStore,
    SessionManager,
    SessionStore,
)
from tools import GenerateImageTool, HandoffToHumanTool, OrderLookupTool, ReadDocsTool, ToolRegistry, WebSearchTool


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATIC = ROOT / "static"
SYSTEM_MESSAGE = (
    "You are MiniRAGent. Answer ordinary questions directly. Use read_docs for project "
    "document evidence. Use web_search only when the user explicitly asks to search, "
    "or when the answer needs current, changing, or externally verified information. "
    "Answer stable, widely known facts directly without web_search. Keep ordinary answers "
    "concise unless the user requests detail. Use generate_image for explicit requests to "
    "create an image, poster, diagram, or promotional visual. For images based on project "
    "documents, call read_docs first with one comprehensive query in the document's language "
    "and enough results to cover the requested policy, then pass only the retrieved facts in "
    "the generate_image prompt. Preserve exact policy wording such as whether a deadline starts "
    "at purchase, dispatch, or delivery. For images based on recent or external information, "
    "call web_search first, then "
    "pass the summarized evidence in the generate_image prompt. Never ask generate_image to "
    "guess policies or current facts. "
    "For a question about a specific order, use order_lookup with the order ID first. "
    "The tool uses the current session identity; never ask the model to provide user_id. "
    "Then use read_docs to check the relevant policy before answering or transferring. "
    "For a Packing order, state whether cancellation is allowed according to policy. "
    "For two completed charges on one order, follow the mandatory human escalation policy. "
    "For a potentially human-only support issue, first use read_docs to check the policy. "
    "If the customer only suspects a duplicate charge, do not assume that two completed "
    "charges exist: explain the difference between pending authorizations and completed "
    "charges, ask for the order number and two transaction references, and ask whether "
    "they want human support. Do not claim a handoff was made before calling the tool. "
    "Use handoff_to_human immediately when the user explicitly asks for a person, "
    "clearly refuses further bot service in an angry or distressed message, or the "
    "retrieved policy confirms mandatory immediate escalation (such as two completed "
    "charges for one order or suspected account takeover). If human help is needed but "
    "not mandatory immediately and the customer has not requested it, explain why it "
    "may be needed, request useful case details without asking for full card numbers or "
    "passwords, and ask for consent before creating a handoff. Do not repeatedly ask for "
    "consent after the customer has already asked for a human. Do not transfer solvable "
    "requests unnecessarily. "
    "After a handoff succeeds, clearly tell the user that support is pending and include "
    "the handoff ID. Complete all requested parts before returning a final answer."
)


@dataclass
class ApplicationServices:
    sessions: SessionManager
    store: SessionStore
    handoffs: HandoffStore
    registry: ToolRegistry
    knowledge: KnowledgeBase


def build_services() -> ApplicationServices:
    llm = LLMClient()
    store = SessionStore(DATA / "sessions.db")
    handoffs = HandoffStore(store)
    embedder = LocalEmbedder(ROOT / "models" / "all-MiniLM-L6-v2")
    knowledge_index = DATA / "index" / "knowledge.faiss"
    knowledge_mapping = DATA / "index" / "knowledge_chunks.json"
    if not knowledge_index.exists():
        knowledge_index = DATA / "index" / "novashop_support.faiss"
        knowledge_mapping = DATA / "index" / "novashop_support_chunks.json"
    knowledge = KnowledgeBase(DATA / "docs", knowledge_index, knowledge_mapping, embedder)

    registry = ToolRegistry()
    registry.register(ReadDocsTool(knowledge.retriever))
    registry.register(OrderLookupTool(store))
    registry.register(WebSearchTool())
    registry.register(GenerateImageTool())
    registry.register(HandoffToHumanTool(handoffs))

    runtime = AgentRuntime(llm, registry, system_message=SYSTEM_MESSAGE)
    context = ContextManager(store, llm, compact_after_messages=16, recent_user_turns=4)
    memory = LongTermMemoryManager(LongTermMemoryStore(store), llm)
    sessions = SessionManager(runtime, store, context, memory, handoffs)
    return ApplicationServices(sessions, store, handoffs, registry, knowledge)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str | None = None


class RenameSessionRequest(BaseModel):
    user_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=80)


class HumanMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class CustomerHumanMessageRequest(HumanMessageRequest):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


def _trace_preview(value: Any, *, string_limit: int = 600, list_limit: int = 5) -> Any:
    """Bound UI trace payloads without changing persisted tool results."""
    if isinstance(value, str):
        return value if len(value) <= string_limit else value[:string_limit] + "…"
    if isinstance(value, list):
        preview = [_trace_preview(item) for item in value[:list_limit]]
        if len(value) > list_limit:
            preview.append({"more_results": len(value) - list_limit})
        return preview
    if isinstance(value, dict):
        return {key: _trace_preview(item) for key, item in value.items()}
    return value


def create_app(services: ApplicationServices | None = None) -> FastAPI:
    application = FastAPI(title="MiniRAGent", version="0.1.0")
    application.state.services = services
    application.mount("/static", StaticFiles(directory=STATIC), name="static")

    def current_services() -> ApplicationServices:
        if application.state.services is None:
            application.state.services = build_services()
        return application.state.services

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @application.get("/human", include_in_schema=False)
    def human_console() -> FileResponse:
        return FileResponse(STATIC / "human.html")

    @application.post("/api/chat")
    def chat(payload: ChatRequest) -> dict[str, Any]:
        try:
            result = current_services().sessions.chat(
                payload.user_id, payload.session_id, payload.message
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
        trace = []
        handoff_id = None
        for step in result.steps:
            item = {
                "step": step.number,
                "decision": step.decision.type,
                "tool_name": step.tool_name,
                "tool_arguments": step.tool_arguments,
                "tool_result": _trace_preview(step.tool_result),
                "final_answer": _trace_preview(step.final_answer, string_limit=240),
            }
            trace.append(item)
            if step.tool_name == "handoff_to_human" and isinstance(step.tool_result, dict):
                handoff_id = step.tool_result.get("handoff_id")
        return {
            "answer": result.final_answer,
            "session_id": payload.session_id,
            "trace": trace,
            "handoff": handoff_id is not None,
            "handoff_id": handoff_id,
            "error": result.error,
        }

    @application.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        def lines():
            try:
                for event in current_services().sessions.chat_stream(
                    payload.user_id, payload.session_id, payload.message
                ):
                    event_type = event.get("type")
                    if event_type == "tool_result":
                        event = {**event, "tool_result": _trace_preview(event.get("tool_result"))}
                    elif event_type in {"done", "error"} and event.get("result") is not None:
                        result = event.pop("result")
                        trace = []
                        handoff_id = None
                        for step in result.steps:
                            trace.append({
                                "step": step.number,
                                "decision": step.decision.type,
                                "tool_name": step.tool_name,
                                "tool_arguments": step.tool_arguments,
                                "tool_result": _trace_preview(step.tool_result),
                                "final_answer": _trace_preview(step.final_answer, string_limit=240),
                            })
                            if step.tool_name == "handoff_to_human" and isinstance(step.tool_result, dict):
                                handoff_id = step.tool_result.get("handoff_id")
                        event.update({
                            "answer": result.final_answer,
                            "session_id": payload.session_id,
                            "trace": trace,
                            "handoff": handoff_id is not None,
                            "handoff_id": handoff_id,
                            "error": result.error,
                        })
                    yield json.dumps(event, ensure_ascii=False, default=str) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "message": f"Chat failed: {exc}"}, ensure_ascii=False) + "\n"

        return StreamingResponse(
            lines(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/api/sessions")
    def create_session(payload: CreateSessionRequest) -> dict[str, str]:
        session_id = (payload.session_id or f"chat_{uuid4().hex}").strip()
        try:
            current_services().store.ensure_session(payload.user_id, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"user_id": payload.user_id, "session_id": session_id}

    @application.get("/api/sessions")
    def sessions(user_id: str = Query(min_length=1)) -> list[dict[str, Any]]:
        return [asdict(item) for item in current_services().store.list_sessions(user_id)]

    @application.get("/api/sessions/{session_id}/messages")
    def messages(session_id: str, user_id: str = Query(min_length=1)) -> list[dict[str, Any]]:
        return [asdict(item) for item in current_services().store.messages(user_id, session_id)]

    @application.patch("/api/sessions/{session_id}")
    def rename_session(session_id: str, payload: RenameSessionRequest) -> dict[str, Any]:
        try:
            renamed = current_services().store.rename_session(
                payload.user_id, session_id, payload.title
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not renamed:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "title": payload.title.strip()}

    @application.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str, user_id: str = Query(min_length=1)) -> dict[str, Any]:
        if not current_services().store.delete_session(user_id, session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session_id": session_id, "deleted": True}

    @application.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = Path(file.filename or "").name
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        content = await file.read()
        if not content or len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="PDF must be between 1 byte and 25 MB")
        target = current_services().knowledge.docs_dir / filename
        temporary = target.parent / f".{uuid4().hex}.upload.pdf"
        try:
            temporary.write_bytes(content)
            load_pdf(temporary)
            temporary.replace(target)
            chunks = current_services().knowledge.rebuild()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PDF indexing failed: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return {"filename": filename, "chunk_count": chunks, "documents": current_services().knowledge.documents()}

    @application.get("/api/documents")
    def documents() -> dict[str, list[str]]:
        return {"documents": current_services().knowledge.documents()}

    @application.get("/api/handoffs")
    def handoff_queue(user_id: str | None = None) -> dict[str, Any]:
        handoffs = current_services().handoffs
        items = handoffs.list(user_id=user_id, status="pending")
        return {
            "pending_count": handoffs.pending_count(user_id=user_id),
            "items": [asdict(item) for item in items],
        }

    @application.get("/api/handoffs/current")
    def current_handoff(
        user_id: str = Query(min_length=1),
        session_id: str = Query(min_length=1),
    ) -> dict[str, Any]:
        handoff = current_services().handoffs.current_for_session(user_id, session_id)
        return {"handoff": asdict(handoff) if handoff else None}

    @application.get("/api/handoffs/{handoff_id}/messages")
    def handoff_messages(handoff_id: str, after_id: int = 0) -> dict[str, Any]:
        handoff = current_services().handoffs.get(handoff_id)
        if handoff is None:
            raise HTTPException(status_code=404, detail="Handoff not found")
        return {
            "handoff": asdict(handoff),
            "messages": [
                asdict(item)
                for item in current_services().handoffs.messages(handoff_id, after_id=after_id)
            ],
        }

    @application.post("/api/handoffs/{handoff_id}/customer-message")
    def customer_handoff_message(
        handoff_id: str, payload: CustomerHumanMessageRequest
    ) -> dict[str, Any]:
        services = current_services()
        handoff = services.handoffs.get(handoff_id)
        if handoff is None or handoff.user_id != payload.user_id or handoff.session_id != payload.session_id:
            raise HTTPException(status_code=404, detail="Handoff not found for this session")
        try:
            message = services.handoffs.add_message(handoff_id, "customer", payload.content)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        services.store.append_messages(
            payload.user_id, payload.session_id, [{"role": "user", "content": payload.content}]
        )
        return asdict(message)

    @application.post("/api/handoffs/{handoff_id}/reply")
    def reply_to_handoff(handoff_id: str, payload: HumanMessageRequest) -> dict[str, Any]:
        services = current_services()
        handoff = services.handoffs.get(handoff_id)
        if handoff is None:
            raise HTTPException(status_code=404, detail="Handoff not found")
        try:
            message = services.handoffs.add_message(handoff_id, "human", payload.content)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        services.store.append_messages(
            handoff.user_id, handoff.session_id, [{"role": "human", "content": payload.content}]
        )
        return asdict(message)

    @application.post("/api/handoffs/{handoff_id}/claim")
    def claim_handoff(handoff_id: str) -> dict[str, Any]:
        claimed = current_services().handoffs.claim(handoff_id)
        if claimed is None:
            raise HTTPException(status_code=409, detail="Handoff is missing or is not pending")
        return asdict(claimed)

    @application.post("/api/handoffs/{handoff_id}/close")
    def close_handoff(handoff_id: str) -> dict[str, Any]:
        closed = current_services().handoffs.close(handoff_id)
        if closed is None:
            raise HTTPException(status_code=409, detail="Handoff is missing or is not active")
        return asdict(closed)

    return application


app = create_app()
