"""Shared utility functions for PaperBanana."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

logger = structlog.get_logger()


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"run_{ts}_{short_uuid}"


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert a base64-encoded string to a PIL Image."""
    data = base64.b64decode(b64_string)
    return Image.open(BytesIO(data))


def load_image(path: str | Path) -> Image.Image:
    """Load an image from a file path."""
    return Image.open(path).convert("RGB")


def save_image(
    image: Image.Image,
    path: str | Path,
    format: str | None = None,
) -> Path:
    """Save a PIL Image to a file path."""
    path = Path(path)
    ensure_dir(path.parent)

    if format is not None:
        if format == "jpeg" and image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(path, format=format.upper())
    else:
        image.save(path)
    return path


def load_text(path: str | Path) -> str:
    """Load text content from a file."""
    return Path(path).read_text(encoding="utf-8")


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON to a file."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON data from a file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_json(text: str) -> str:
    """Extract a JSON object from a VLM response, trying multiple strategies.

    Strategies (in order):
        1. Direct parse — response is already valid JSON
        2. Markdown fences — ```json ... ``` or ``` ... ```
        3. Brace extraction — find outermost { ... } in free text
    """
    import re

    text = text.strip()
    if not text:
        return text

    # Strategy 1: already valid JSON
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    # Strategy 2: markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 3: find outermost { ... } in surrounding text
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

    return text


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to a maximum number of characters."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def hash_content(content: str) -> str:
    """Generate a short hash of content for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def find_prompt_dir() -> str:
    """Locate the prompts directory, handling CWD != project root.

    When PaperBanana is invoked via ``uvx`` or as an MCP server the working
    directory is typically *not* the project root, so the default relative
    ``"prompts"`` path fails.  This helper checks the CWD first, then
    resolves relative to the installed package location.
    """
    candidates = [
        Path("prompts"),
        Path(__file__).resolve().parent.parent.parent / "prompts",
    ]
    for p in candidates:
        if (p / "evaluation").exists() or (p / "diagram").exists():
            return str(p)
    return "prompts"
