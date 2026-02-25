"""Tests for the Amazon Bedrock VLM provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Registry / Config integration
# ---------------------------------------------------------------------------


def test_create_bedrock_vlm_via_registry():
    """Bedrock provider is created through the registry without an API key."""
    settings = Settings(
        vlm_provider="bedrock",
        bedrock_vlm_model="anthropic.claude-sonnet-4-20250514-v1:0",
        bedrock_region="us-west-2",
    )
    # boto3 client is created lazily — no need to mock at construction time
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm.name == "bedrock"
    assert vlm.model_name == "anthropic.claude-sonnet-4-20250514-v1:0"


def test_bedrock_model_fallback_to_vlm_model():
    """When BEDROCK_VLM_MODEL is not set, vlm_model is used instead."""
    settings = Settings(
        vlm_provider="bedrock",
        vlm_model="anthropic.claude-haiku-4-5-20251001-v1:0",
        bedrock_vlm_model=None,
    )
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm.model_name == "anthropic.claude-haiku-4-5-20251001-v1:0"


def test_bedrock_default_region():
    """Default region is us-east-1."""
    settings = Settings(vlm_provider="bedrock")
    assert settings.bedrock_region == "us-east-1"


def test_effective_vlm_model_bedrock():
    """effective_vlm_model returns bedrock_vlm_model when set."""
    settings = Settings(
        vlm_provider="bedrock",
        vlm_model="gemini-2.0-flash",
        bedrock_vlm_model="anthropic.claude-sonnet-4-20250514-v1:0",
    )
    assert settings.effective_vlm_model == "anthropic.claude-sonnet-4-20250514-v1:0"


def test_effective_vlm_model_bedrock_fallback():
    """effective_vlm_model falls back to vlm_model when bedrock_vlm_model is None."""
    settings = Settings(
        vlm_provider="bedrock",
        vlm_model="anthropic.claude-haiku-4-5-20251001-v1:0",
        bedrock_vlm_model=None,
    )
    assert settings.effective_vlm_model == "anthropic.claude-haiku-4-5-20251001-v1:0"


# ---------------------------------------------------------------------------
# Provider unit tests (mocked boto3)
# ---------------------------------------------------------------------------


def test_encode_image_small_png():
    """Small images are encoded as PNG without compression."""
    from paperbanana.providers.vlm.bedrock import BedrockVLM

    img = Image.new("RGB", (10, 10), color="red")
    raw, fmt = BedrockVLM._encode_image(img)
    assert fmt == "png"
    assert isinstance(raw, bytes)
    # PNG magic bytes
    assert raw[:4] == b"\x89PNG"


def test_encode_image_large_falls_back_to_jpeg():
    """Images exceeding the PNG size limit fall back to JPEG."""
    import io as _io

    from paperbanana.providers.vlm.bedrock import BedrockVLM

    # Create a 500x500 image. Its PNG is several KB; set the threshold
    # between JPEG size and PNG size to force the JPEG fallback path.
    img = Image.new("RGB", (500, 500), color="red")
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    png_size = buf.tell()

    # Threshold: half of the PNG size — comfortably above JPEG but below PNG
    threshold = png_size // 2
    raw, fmt = BedrockVLM._encode_image(img, max_bytes=threshold)
    assert fmt == "jpeg"
    assert len(raw) <= threshold


def test_best_jpeg_quality_maximises_quality():
    """Binary search picks the highest quality that fits."""
    # Use a noisy image so different quality levels produce different sizes
    import random

    from paperbanana.providers.vlm.bedrock import BedrockVLM

    random.seed(42)
    img = Image.new("RGB", (200, 200))
    pixels = img.load()
    for x in range(200):
        for y in range(200):
            pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Find the result at two different budgets; larger budget → higher quality
    result_small = BedrockVLM._best_jpeg_quality(img, max_bytes=15_000)
    result_large = BedrockVLM._best_jpeg_quality(img, max_bytes=50_000)
    assert result_small is not None
    assert result_large is not None
    _, q_small = result_small
    _, q_large = result_large
    assert q_large >= q_small


def test_encode_image_downscale_preserves_limit():
    """When even JPEG q=20 at full res is too large, downscale kicks in."""
    import random

    from paperbanana.providers.vlm.bedrock import BedrockVLM

    # Create a large noisy image; JPEG at q=20 should still be > 1 KB.
    random.seed(0)
    img = Image.new("RGB", (800, 800))
    pixels = img.load()
    for x in range(800):
        for y in range(800):
            pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Very tight budget forces downscale path
    budget = 10_000
    raw, fmt = BedrockVLM._encode_image(img, max_bytes=budget)
    assert fmt == "jpeg"
    assert len(raw) <= budget


@pytest.mark.asyncio
async def test_generate_text_only():
    """generate() builds the correct Converse payload for text-only requests."""
    from paperbanana.providers.vlm.bedrock import BedrockVLM

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello from Bedrock"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }

    vlm = BedrockVLM(model="anthropic.claude-sonnet-4-20250514-v1:0", region="us-east-1")
    vlm._client = mock_client

    result = await vlm.generate("Hello", system_prompt="Be helpful")

    assert result == "Hello from Bedrock"
    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["modelId"] == "anthropic.claude-sonnet-4-20250514-v1:0"
    assert call_kwargs["system"] == [{"text": "Be helpful"}]
    assert call_kwargs["messages"][0]["content"] == [{"text": "Hello"}]


@pytest.mark.asyncio
async def test_generate_with_images():
    """generate() includes image blocks when images are provided."""
    from paperbanana.providers.vlm.bedrock import BedrockVLM

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "I see a red square"}]}},
        "usage": {},
    }

    vlm = BedrockVLM()
    vlm._client = mock_client

    img = Image.new("RGB", (10, 10), color="red")
    result = await vlm.generate("Describe this image", images=[img])

    assert result == "I see a red square"
    content = mock_client.converse.call_args[1]["messages"][0]["content"]
    # First block should be an image, second should be text
    assert "image" in content[0]
    assert content[0]["image"]["format"] in ("png", "jpeg")
    assert isinstance(content[0]["image"]["source"]["bytes"], bytes)
    assert content[1] == {"text": "Describe this image"}


@pytest.mark.asyncio
async def test_generate_json_mode():
    """response_format='json' injects hint into prompt and system prompt."""
    from paperbanana.providers.vlm.bedrock import BedrockVLM

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": '{"key": "value"}'}]}},
        "usage": {},
    }

    vlm = BedrockVLM()
    vlm._client = mock_client

    await vlm.generate("Return JSON", system_prompt="You are a bot", response_format="json")

    call_kwargs = mock_client.converse.call_args[1]
    user_text = call_kwargs["messages"][0]["content"][0]["text"]
    assert "valid JSON" in user_text
    sys_text = call_kwargs["system"][0]["text"]
    assert "valid JSON" in sys_text


def test_missing_boto3_raises_import_error():
    """ImportError with install hint when boto3 is not available."""
    from paperbanana.providers.vlm.bedrock import BedrockVLM

    vlm = BedrockVLM()
    vlm._client = None
    with patch.dict("sys.modules", {"boto3": None}):
        with pytest.raises(ImportError, match="pip install"):
            vlm._get_client()
