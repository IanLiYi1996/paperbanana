"""Tests for yunwu_imagen integration with registry and config."""

from __future__ import annotations

import pytest

from paperbanana.core.config import Settings
from paperbanana.providers.registry import ProviderRegistry


def test_create_yunwu_imagen_gen():
    """Test creating a Yunwu image gen provider via the registry."""
    settings = Settings(
        image_provider="yunwu_imagen",
        yunwu_api_key="test-key",
        yunwu_base_url="https://yunwu.example.com",
    )
    gen = ProviderRegistry.create_image_gen(settings)
    assert gen.name == "yunwu_imagen"


def test_missing_yunwu_api_key_raises_helpful_error():
    """Test that missing YUNWU_API_KEY raises a helpful error."""
    settings = Settings(image_provider="yunwu_imagen", yunwu_api_key=None)
    with pytest.raises(ValueError, match="YUNWU_API_KEY not found"):
        ProviderRegistry.create_image_gen(settings)


def test_yunwu_config_fields_exist():
    """Test that Settings has yunwu-specific fields."""
    settings = Settings(
        yunwu_api_key="my-key",
        yunwu_base_url="https://custom.yunwu.com",
    )
    assert settings.yunwu_api_key == "my-key"
    assert settings.yunwu_base_url == "https://custom.yunwu.com"


def test_yunwu_base_url_has_default():
    """Test that yunwu_base_url has a sensible default."""
    settings = Settings()
    assert settings.yunwu_base_url is not None
    assert settings.yunwu_base_url.startswith("https://")
