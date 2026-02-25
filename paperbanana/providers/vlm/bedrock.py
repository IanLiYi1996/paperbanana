"""Amazon Bedrock VLM provider — uses the Converse API for model-agnostic inference."""

from __future__ import annotations

import asyncio
import io
from typing import Optional

import structlog
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class BedrockVLM(VLMProvider):
    """VLM provider using Amazon Bedrock's Converse API.

    Calls ``bedrock-runtime`` directly via boto3, bypassing proxy payload
    limits (~1 MB) that block image-heavy requests.  The Converse API
    accepts up to 25 MB per request and works with any Bedrock model
    (Claude, Titan, Mistral, etc.).

    Authentication is handled by the standard boto3 credential chain
    (IAM role, environment variables, ``~/.aws/credentials``).
    """

    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
        region: str = "us-east-1",
    ):
        self._model = model
        self._region = region
        self._client = None

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for the Bedrock provider. "
                    "Install with: pip install 'paperbanana[bedrock]'"
                )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
            )
        return self._client

    def is_available(self) -> bool:
        try:
            self._get_client()
            return True
        except Exception:
            return False

    # Converse API hard limit: 5 MB (5,242,880 bytes) per image.
    _MAX_IMAGE_BYTES = 5 * 1024 * 1024

    @staticmethod
    def _jpeg_size(img: Image.Image, quality: int) -> tuple[bytes, int]:
        """Encode *img* as JPEG at *quality* and return ``(bytes, length)``."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), buf.tell()

    @staticmethod
    def _encode_image(img: Image.Image, max_bytes: int = 5 * 1024 * 1024) -> tuple[bytes, str]:
        """Encode a PIL Image to bytes that fit within *max_bytes*,
        preserving as much detail as possible.

        Strategy — maximize quality at each stage, only escalate when needed:

        1. **PNG at original resolution** — lossless, best quality.
        2. **JPEG at original resolution** — binary-search for the highest
           quality (95→20) that fits.  Most 2K diagrams land here.
        3. **Downscale** — only when even JPEG q=20 at full res exceeds
           the limit.  Shrink by the minimum factor, then binary-search
           quality again at the new resolution.

        Returns ``(raw_bytes, format)`` where format is ``"png"`` or ``"jpeg"``.
        """
        # ── Stage 1: lossless PNG ──────────────────────────────────
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        if buf.tell() <= max_bytes:
            return buf.getvalue(), "png"

        # ── Prepare RGB copy for JPEG stages ───────────────────────
        current = img if img.mode == "RGB" else img.convert("RGB")

        # ── Stage 2: JPEG at original resolution, best quality ─────
        best = BedrockVLM._best_jpeg_quality(current, max_bytes)
        if best is not None:
            raw, quality = best
            logger.debug(
                "Image compressed to JPEG",
                quality=quality,
                size=(current.size),
                size_kb=len(raw) // 1024,
            )
            return raw, "jpeg"

        # ── Stage 3: downscale until JPEG fits ─────────────────────
        # Estimate the scale factor from current JPEG-q20 size.
        _, q20_size = BedrockVLM._jpeg_size(current, 20)
        # ratio < 1.0; we need area * ratio ≤ max_bytes
        ratio = max_bytes / q20_size
        # Area scales with scale², so linear scale = sqrt(ratio).
        # Use 0.95× as safety margin.
        scale = max(0.1, ratio**0.5 * 0.95)

        while True:
            w, h = current.size
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if (new_w, new_h) == (w, h):
                # Cannot shrink further — return lowest quality at current size
                raw, _ = BedrockVLM._jpeg_size(current, 20)
                return raw, "jpeg"

            current = img if img.mode == "RGB" else img.convert("RGB")
            current = current.resize((new_w, new_h), Image.LANCZOS)

            best = BedrockVLM._best_jpeg_quality(current, max_bytes)
            if best is not None:
                raw, quality = best
                logger.debug(
                    "Image downscaled to fit",
                    quality=quality,
                    new_size=current.size,
                    size_kb=len(raw) // 1024,
                )
                return raw, "jpeg"

            # Still too large even at q=20 — shrink further
            scale *= 0.75

    @staticmethod
    def _best_jpeg_quality(
        img: Image.Image, max_bytes: int, lo: int = 20, hi: int = 95
    ) -> tuple[bytes, int] | None:
        """Binary-search for the highest JPEG quality that fits in *max_bytes*.

        Returns ``(raw_bytes, quality)`` or ``None`` if even *lo* exceeds the limit.
        """
        raw_lo, size_lo = BedrockVLM._jpeg_size(img, lo)
        if size_lo > max_bytes:
            return None

        # Fast path: best quality already fits
        raw_hi, size_hi = BedrockVLM._jpeg_size(img, hi)
        if size_hi <= max_bytes:
            return raw_hi, hi

        best_raw, best_q = raw_lo, lo
        while lo <= hi:
            mid = (lo + hi) // 2
            raw_mid, size_mid = BedrockVLM._jpeg_size(img, mid)
            if size_mid <= max_bytes:
                best_raw, best_q = raw_mid, mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best_raw, best_q

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
    ) -> str:
        client = self._get_client()

        # Build user content blocks
        content: list[dict] = []
        if images:
            for img in images:
                img_bytes, img_fmt = self._encode_image(img, self._MAX_IMAGE_BYTES)
                content.append(
                    {
                        "image": {
                            "format": img_fmt,
                            "source": {"bytes": img_bytes},
                        }
                    }
                )
        # Inject JSON hint into the prompt when JSON mode is requested
        user_text = prompt
        if response_format == "json":
            user_text = prompt + "\n\nIMPORTANT: Respond with valid JSON only, no other text."
        content.append({"text": user_text})

        messages = [{"role": "user", "content": content}]

        kwargs: dict = {
            "modelId": self._model,
            "messages": messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }

        # System prompt
        if system_prompt:
            sys_text = system_prompt
            if response_format == "json" and "json" not in system_prompt.lower():
                sys_text += "\nAlways respond with valid JSON."
            kwargs["system"] = [{"text": sys_text}]

        # boto3 is synchronous — run in a thread to avoid blocking the event loop
        response = await asyncio.to_thread(client.converse, **kwargs)

        text = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})

        logger.debug(
            "Bedrock response",
            model=self._model,
            region=self._region,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
        )
        return text
