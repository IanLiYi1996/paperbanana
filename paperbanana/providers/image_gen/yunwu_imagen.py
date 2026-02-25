"""Yunwu image generation provider — Gemini-compatible API proxy via httpx."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import ImageGenProvider

logger = structlog.get_logger()


class YunwuImageGen(ImageGenProvider):
    """Image generation via Yunwu API (Google Gemini-compatible proxy).

    Calls the ``/v1beta/models/{model}:generateContent`` endpoint with
    ``responseModalities: ["image"]``.  Authentication is via a ``key``
    query parameter, matching the upstream Gemini REST API.

    Get an API key from your Yunwu dashboard.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://yunwu.ai",
        model: str = "gemini-3-pro-image-preview",
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = None

    @property
    def name(self) -> str:
        return "yunwu_imagen"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        """Lazy-init an async httpx client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Content-Type": "application/json"},
                timeout=180.0,
            )
        return self._client

    def is_available(self) -> bool:
        return self._api_key is not None

    def _aspect_ratio(self, width: int, height: int) -> str:
        """Map pixel dimensions to a Gemini aspect-ratio string."""
        ratio = width / height
        if ratio > 1.5:
            return "16:9"
        if ratio > 1.2:
            return "3:2"
        if ratio < 0.67:
            return "9:16"
        if ratio < 0.83:
            return "2:3"
        return "1:1"

    def _image_size(self, width: int, height: int) -> str:
        """Map pixel dimensions to a Gemini imageSize string ("1K", "2K", "4K")."""
        max_dim = max(width, height)
        if max_dim <= 1024:
            return "1K"
        if max_dim <= 2048:
            return "2K"
        return "4K"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> Image.Image:
        client = self._get_client()

        if negative_prompt:
            prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["image"],
                "imageConfig": {
                    "aspectRatio": self._aspect_ratio(width, height),
                    "imageSize": self._image_size(width, height),
                },
            },
        }

        endpoint = f"/v1beta/models/{self._model}:generateContent"
        response = await client.post(
            endpoint,
            json=payload,
            params={"key": self._api_key},
        )
        response.raise_for_status()
        data = response.json()

        # Parse the Gemini-style response
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    image_bytes = base64.b64decode(inline["data"])
                    return Image.open(BytesIO(image_bytes))

        logger.error("No image data in Yunwu response", model=self._model)
        raise ValueError(f"Yunwu response for {self._model} did not contain image data.")
