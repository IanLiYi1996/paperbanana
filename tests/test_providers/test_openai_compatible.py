"""Tests for openai_compatible VLM provider integration."""

from __future__ import annotations

import pytest

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry


def test_create_openai_compatible_vlm():
    """Registry creates an OpenAI-compatible VLM with independent config."""
    settings = Settings(
        vlm_provider="openai_compatible",
        openai_compatible_api_key="test-key",
        openai_compatible_base_url="https://my-llm.example.com/v1",
        openai_compatible_model="qwen-vl-plus",
    )
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm.name == "openai"  # reuses OpenAIVLM class
    assert vlm.model_name == "qwen-vl-plus"
    assert vlm._base_url == "https://my-llm.example.com/v1"
    assert vlm._api_key == "test-key"


def test_openai_compatible_uses_vlm_model_as_fallback():
    """When openai_compatible_model is not set, falls back to vlm_model."""
    settings = Settings(
        vlm_provider="openai_compatible",
        vlm_model="some-model",
        openai_compatible_api_key="test-key",
    )
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm.model_name == "some-model"


def test_openai_compatible_model_overrides_vlm_model():
    """openai_compatible_model takes priority over vlm_model."""
    settings = Settings(
        vlm_provider="openai_compatible",
        vlm_model="fallback-model",
        openai_compatible_api_key="test-key",
        openai_compatible_model="preferred-model",
    )
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm.model_name == "preferred-model"


def test_missing_openai_compatible_api_key_raises_helpful_error():
    """Missing OPENAI_COMPATIBLE_API_KEY raises a helpful error."""
    settings = Settings(vlm_provider="openai_compatible", openai_compatible_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_COMPATIBLE_API_KEY not found"):
        ProviderRegistry.create_vlm(settings)


def test_openai_compatible_does_not_touch_openai_config():
    """openai_compatible config is fully independent from openai config."""
    settings = Settings(
        vlm_provider="openai_compatible",
        openai_compatible_api_key="compat-key",
        openai_compatible_base_url="https://compat.example.com/v1",
        openai_api_key="openai-key",
        openai_base_url="https://api.openai.com/v1",
    )
    vlm = ProviderRegistry.create_vlm(settings)
    assert vlm._api_key == "compat-key"
    assert vlm._base_url == "https://compat.example.com/v1"


def test_openai_compatible_config_fields_exist():
    """Settings has openai_compatible fields with correct defaults."""
    settings = Settings(
        openai_compatible_api_key="k",
        openai_compatible_base_url="https://example.com/v1",
        openai_compatible_model="my-model",
    )
    assert settings.openai_compatible_api_key == "k"
    assert settings.openai_compatible_base_url == "https://example.com/v1"
    assert settings.openai_compatible_model == "my-model"


def test_openai_compatible_base_url_has_sensible_default():
    """Default base_url should not be empty."""
    settings = Settings()
    assert settings.openai_compatible_base_url is not None
    assert len(settings.openai_compatible_base_url) > 0
