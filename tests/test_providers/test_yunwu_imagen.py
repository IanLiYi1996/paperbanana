"""Tests for the Yunwu image generation provider."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from tenacity import RetryError

from paperbanana.providers.image_gen.yunwu_imagen import YunwuImageGen


@pytest.fixture
def provider():
    return YunwuImageGen(api_key="test-key", base_url="https://yunwu.example.com")


@pytest.fixture
def fake_image_b64() -> str:
    """Create a tiny 1x1 red PNG as base64."""
    img = Image.new("RGB", (1, 1), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestProperties:
    def test_name(self, provider):
        assert provider.name == "yunwu_imagen"

    def test_model_name(self, provider):
        assert provider.model_name == "gemini-3-pro-image-preview"

    def test_custom_model(self):
        p = YunwuImageGen(api_key="k", base_url="https://example.com", model="custom-model")
        assert p.model_name == "custom-model"

    def test_is_available_with_key(self, provider):
        assert provider.is_available() is True

    def test_is_not_available_without_key(self):
        p = YunwuImageGen(api_key=None, base_url="https://example.com")
        assert p.is_available() is False


class TestAspectRatio:
    def test_square(self, provider):
        assert provider._aspect_ratio(1024, 1024) == "1:1"

    def test_landscape_16_9(self, provider):
        assert provider._aspect_ratio(1920, 1080) == "16:9"

    def test_landscape_3_2(self, provider):
        assert provider._aspect_ratio(1500, 1024) == "3:2"

    def test_portrait_9_16(self, provider):
        assert provider._aspect_ratio(1080, 1920) == "9:16"

    def test_portrait_2_3(self, provider):
        assert provider._aspect_ratio(1024, 1500) == "2:3"


class TestImageSize:
    def test_small_maps_to_1k(self, provider):
        assert provider._image_size(1024, 1024) == "1K"

    def test_medium_maps_to_2k(self, provider):
        assert provider._image_size(2048, 1024) == "2K"

    def test_large_maps_to_4k(self, provider):
        assert provider._image_size(4096, 2160) == "4K"

    def test_boundary_1024_is_1k(self, provider):
        assert provider._image_size(1024, 768) == "1K"

    def test_boundary_1025_is_2k(self, provider):
        assert provider._image_size(1025, 768) == "2K"

    def test_boundary_2048_is_2k(self, provider):
        assert provider._image_size(2048, 2048) == "2K"

    def test_boundary_2049_is_4k(self, provider):
        assert provider._image_size(2049, 1024) == "4K"


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_image(self, provider, fake_image_b64):
        """Successful generation returns a PIL Image."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": fake_image_b64,
                                }
                            }
                        ]
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.generate("a red dot")
        assert isinstance(result, Image.Image)

    @pytest.mark.asyncio
    async def test_generate_sends_correct_payload(self, provider, fake_image_b64):
        """Verify the request payload structure matches the yunwu/Gemini API."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"inlineData": {"mimeType": "image/png", "data": fake_image_b64}}]
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        await provider.generate("a cat", width=1920, height=1080)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")

        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"][0]["text"] == "a cat"
        assert payload["generationConfig"]["responseModalities"] == ["image"]
        assert payload["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
        assert payload["generationConfig"]["imageConfig"]["imageSize"] == "2K"

    @pytest.mark.asyncio
    async def test_generate_includes_negative_prompt(self, provider, fake_image_b64):
        """Negative prompt is appended to the text."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"inlineData": {"mimeType": "image/png", "data": fake_image_b64}}]
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        await provider.generate("a cat", negative_prompt="blurry")

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        text = payload["contents"][0]["parts"][0]["text"]
        assert "blurry" in text

    @pytest.mark.asyncio
    async def test_generate_raises_on_empty_response(self, provider):
        """Raises when response has no image data (retried then wrapped by tenacity)."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"candidates": [{"content": {"parts": []}}]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        with pytest.raises(RetryError):
            await provider.generate("a cat")

    @pytest.mark.asyncio
    async def test_generate_uses_key_as_query_param(self, provider, fake_image_b64):
        """API key is passed as query parameter, not header."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"inlineData": {"mimeType": "image/png", "data": fake_image_b64}}]
                    }
                }
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        await provider.generate("a cat")

        call_args = mock_client.post.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params == {"key": "test-key"}
