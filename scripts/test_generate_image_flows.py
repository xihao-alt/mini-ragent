"""Run real direct and knowledge-grounded image-generation Agent flows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent import AgentRuntime  # noqa: E402
from llm import LLMClient  # noqa: E402
from rag.embedding import LocalEmbedder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402
from tools import GenerateImageTool, ReadDocsTool, ToolRegistry  # noqa: E402

SYSTEM_MESSAGE = (
    "You are MiniRAGent. Complete image requests using generate_image. For a direct creative "
    "request, call generate_image immediately with a polished prompt. If the requested image "
    "depends on the English local knowledge base, call read_docs first, translating "
    "the request into one comprehensive English search query and requesting 5 results. After "
    "receiving the document result, create a factual visual prompt containing only retrieved "
    "rules and call generate_image. Preserve exact policy wording, especially whether deadlines "
    "start at purchase, dispatch, or delivery. Never let the image model guess policies. After "
    "image generation, return the saved image path."
)


def build_registry() -> ToolRegistry:
    embedder = LocalEmbedder(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
    store = FaissVectorStore.load(
        PROJECT_ROOT / "data" / "index" / "novashop_support.faiss",
        PROJECT_ROOT / "data" / "index" / "novashop_support_chunks.json",
    )
    registry = ToolRegistry()
    registry.register(ReadDocsTool(Retriever(embedder, store)))
    registry.register(GenerateImageTool())
    return registry


def summarize(label: str, result: object) -> None:
    steps = getattr(result, "steps")
    print(f"\n=== {label} ===")
    print(json.dumps({
        "succeeded": getattr(result, "succeeded"),
        "final_answer": getattr(result, "final_answer"),
        "error": getattr(result, "error"),
        "tools": [{
            "name": step.tool_name,
            "arguments": step.tool_arguments,
            "result": step.tool_result,
        } for step in steps if step.tool_name],
    }, ensure_ascii=False, indent=2))


def main() -> None:
    runtime = AgentRuntime(LLMClient(), build_registry(), max_steps=6, system_message=SYSTEM_MESSAGE)
    requested_case = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    results = []
    if requested_case in {"A", "ALL"}:
        direct = runtime.run("帮我生成一张科技感的智能客服宣传图。")
        summarize("A_DIRECT", direct)
        direct_tools = [step.tool_name for step in direct.steps if step.tool_name]
        if direct_tools != ["generate_image"]:
            raise AssertionError(f"Unexpected A tool order: {direct_tools}")
        results.append(direct)
    if requested_case in {"B", "ALL"}:
        grounded = runtime.run("根据 NovaShop 的退货政策生成一张退货流程图。")
        summarize("B_KNOWLEDGE_GROUNDED", grounded)
        grounded_tools = [step.tool_name for step in grounded.steps if step.tool_name]
        if (
            len(grounded_tools) < 2
            or grounded_tools[-1] != "generate_image"
            or any(name != "read_docs" for name in grounded_tools[:-1])
        ):
            raise AssertionError(f"Unexpected B tool order: {grounded_tools}")
        image_prompt = next(
            step.tool_arguments["prompt"]
            for step in grounded.steps
            if step.tool_name == "generate_image"
        )
        if "30 calendar days of delivery" not in image_prompt:
            raise AssertionError("B image prompt did not preserve the return deadline")
        results.append(grounded)
    if requested_case not in {"A", "B", "ALL"}:
        raise ValueError("Case must be A, B, or ALL")
    for result in results:
        image_steps = [step for step in result.steps if step.tool_name == "generate_image"]
        if not image_steps or not Path(image_steps[0].tool_result["image_path"]).is_file():
            raise AssertionError("Image was not saved locally")


if __name__ == "__main__":
    main()
