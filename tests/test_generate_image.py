"""Tests for the Agnes-backed generate_image tool."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

from tools.generate_image import GenerateImageTool
from tools.registry import ToolRegistry


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeImages:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> object:
        self.request = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(PNG_1X1).decode(), url=None)]
        )


def test_generate_image_schema_and_local_save(tmp_path) -> None:
    images = FakeImages()
    client = SimpleNamespace(images=images)
    env_file = tmp_path / ".env_image"
    env_file.write_text(
        "OPENAI_API_KEY=test-key\n"
        "AGNES_IMAGE_BASE_URL=https://example.test/v1\n"
        "AGNES_IMAGE_MODEL=agnes-image-test\n",
        encoding="utf-8",
    )
    tool = GenerateImageTool(
        env_path=env_file,
        output_dir=tmp_path / "images",
        client=client,
    )
    registry = ToolRegistry()
    registry.register(tool)

    result = registry.get("generate_image").execute(prompt="A red apple")

    assert registry.schemas()[0] == {
        "name": "generate_image",
        "description": tool.description,
        "parameters": tool.parameters,
    }
    assert images.request == {
        "model": "agnes-image-test",
        "prompt": "A red apple",
        "n": 1,
        "size": "1024x1024",
    }
    assert result["prompt"] == "A red apple"
    assert result["image_url"] is None
    assert Path(str(result["image_path"])).read_bytes() == PNG_1X1
