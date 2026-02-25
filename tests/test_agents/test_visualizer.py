"""Tests for VisualizerAgent code extraction and resolution mapping."""

from __future__ import annotations

import pytest
from PIL import Image

from paperbanana.agents.visualizer import VisualizerAgent


class _DummyImageGen:
    """Captures kwargs passed to generate() for assertion."""

    def __init__(self):
        self.last_kwargs = {}

    async def generate(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return Image.new("RGB", (1, 1), color="white")


class _DummyVLM:
    async def generate(self, *args, **kwargs):
        return ""


def _make_agent(tmp_path, **kwargs):
    return VisualizerAgent(
        image_gen=_DummyImageGen(),
        vlm_provider=_DummyVLM(),
        prompt_dir=str(tmp_path),
        output_dir=str(tmp_path),
        **kwargs,
    )


def test_extract_code_handles_truncated_python_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```python\nimport matplotlib.pyplot as plt\nplt.figure()\n"
    code = agent._extract_code(response)
    assert code == "import matplotlib.pyplot as plt\nplt.figure()"


def test_extract_code_handles_truncated_generic_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```\nprint('hello')\n"
    code = agent._extract_code(response)
    assert code == "print('hello')"


def test_extract_code_handles_complete_python_block(tmp_path):
    agent = _make_agent(tmp_path)
    response = "```python\nprint('ok')\n```\nextra"
    code = agent._extract_code(response)
    assert code == "print('ok')"


def test_extract_code_handles_plain_code_response(tmp_path):
    agent = _make_agent(tmp_path)
    response = "import matplotlib.pyplot as plt\nplt.figure()"
    code = agent._extract_code(response)
    assert code == response


class TestResolutionMapping:
    """output_resolution config should control the pixel dimensions passed to generate()."""

    def test_default_resolution_is_2k(self, tmp_path):
        agent = _make_agent(tmp_path)
        w, h = agent._resolve_dimensions()
        assert max(w, h) <= 2048
        assert max(w, h) > 1024

    def test_1k_resolution(self, tmp_path):
        agent = _make_agent(tmp_path, output_resolution="1k")
        w, h = agent._resolve_dimensions()
        assert max(w, h) <= 1024

    def test_2k_resolution(self, tmp_path):
        agent = _make_agent(tmp_path, output_resolution="2k")
        w, h = agent._resolve_dimensions()
        assert max(w, h) <= 2048
        assert max(w, h) > 1024

    def test_4k_resolution(self, tmp_path):
        agent = _make_agent(tmp_path, output_resolution="4k")
        w, h = agent._resolve_dimensions()
        assert max(w, h) > 2048

    def test_resolution_case_insensitive(self, tmp_path):
        agent = _make_agent(tmp_path, output_resolution="4K")
        w, h = agent._resolve_dimensions()
        assert max(w, h) > 2048

    @pytest.mark.asyncio
    async def test_generate_diagram_uses_configured_resolution(self, tmp_path):
        """_generate_diagram passes resolution-derived dimensions to image_gen."""
        image_gen = _DummyImageGen()
        agent = VisualizerAgent(
            image_gen=image_gen,
            vlm_provider=_DummyVLM(),
            prompt_dir=str(tmp_path),
            output_dir=str(tmp_path),
            output_resolution="4k",
        )
        # Create a minimal prompt file so load_prompt works
        (tmp_path / "diagram").mkdir()
        (tmp_path / "diagram" / "visualizer.txt").write_text("{description}")

        await agent._generate_diagram("test description", None, 0, None)

        w = image_gen.last_kwargs["width"]
        h = image_gen.last_kwargs["height"]
        assert max(w, h) > 2048
