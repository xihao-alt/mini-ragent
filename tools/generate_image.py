"""Image generation tool backed by the OpenAI-compatible Agnes API."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import dotenv_values
import httpx
from openai import OpenAI

from .base import Tool


class GenerateImageTool(Tool):
    name = "generate_image"
    description = (
        "Generate an image from a complete visual prompt and save it locally. Use this for "
        "explicit image, poster, illustration, diagram, or promotional-art requests. If the "
        "image must contain factual document or web information, retrieve that information "
        "first and include the verified facts in prompt."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Complete image-generation prompt, including verified facts when needed.",
                "minLength": 1,
            },
            "size": {
                "type": "string",
                "description": "Requested image dimensions in WIDTHxHEIGHT form.",
                "default": "1024x1024",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        env_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        client: Any | None = None,
        download_client: httpx.Client | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.env_path = Path(env_path) if env_path else root / ".env_image"
        self.output_dir = Path(output_dir) if output_dir else root / "data" / "generated_images"
        self._client = client
        self._download_client = download_client

    def _configuration(self) -> tuple[str, str, str]:
        values = dotenv_values(self.env_path)

        def get(name: str) -> str:
            return str(values.get(name) or os.getenv(name) or "").strip()

        api_key = get("OPENAI_API_KEY")
        base_url = get("AGNES_IMAGE_BASE_URL")
        model = get("AGNES_IMAGE_MODEL")
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", api_key),
                ("AGNES_IMAGE_BASE_URL", base_url),
                ("AGNES_IMAGE_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing image configuration: {', '.join(missing)}")
        return api_key, base_url.rstrip("/"), model

    def _get_client(self) -> tuple[Any, str]:
        api_key, base_url, model = self._configuration()
        if self._client is None:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=180.0,
                max_retries=2,
                http_client=httpx.Client(trust_env=False),
            )
        return self._client, model

    @staticmethod
    def _extension(image_url: str | None) -> str:
        suffix = Path(urlparse(image_url or "").path).suffix.lower()
        return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"

    def execute(
        self,
        *,
        prompt: str,
        size: str = "1024x1024",
        **kwargs: Any,
    ) -> dict[str, str | None]:
        if kwargs:
            raise TypeError(f"Unexpected generate_image arguments: {', '.join(kwargs)}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(size, str) or not size.strip():
            raise ValueError("size must be a non-empty string")

        client, model = self._get_client()
        response = client.images.generate(
            model=model,
            prompt=prompt.strip(),
            n=1,
            size=size.strip(),
        )
        if not response.data:
            raise RuntimeError("Image API returned no image data")
        image = response.data[0]
        image_url = getattr(image, "url", None)
        encoded = getattr(image, "b64_json", None)
        if not image_url and not encoded:
            raise RuntimeError("Image API returned neither a URL nor base64 image data")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"image-{uuid4().hex}{self._extension(image_url)}"
        if encoded:
            destination.write_bytes(base64.b64decode(encoded))
        else:
            downloader = self._download_client or httpx.Client(trust_env=False, timeout=180.0)
            download = downloader.get(image_url)
            download.raise_for_status()
            destination.write_bytes(download.content)

        return {
            "image_path": str(destination.resolve()),
            "image_url": image_url,
            "prompt": prompt.strip(),
        }
