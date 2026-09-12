# Copyright 2026 chatgpt-to-openai-api contributors.
"""Image host client (freeimage.host API: https://freeimage.host/page/api)."""

from __future__ import annotations

import base64
import logging

from curl_cffi.requests import AsyncSession

from . import config

log = logging.getLogger("freeimage")


class FreeimageError(Exception):
    """freeimage.host image-host request failed."""

    def __init__(self, message: str = "freeimage.host request failed") -> None:
        """Store the failure message."""
        super().__init__(message)


async def upload_image(name: str, data: bytes, mime: str) -> str:
    """Upload image bytes, returning the public URL.

    Returns:
        The public URL of the uploaded image.

    Raises:
        FreeimageError: If the API key is missing or the upload fails.

    """
    if not config.FREEIMAGE_API_KEY:
        msg = "FREEIMAGE_API_KEY not configured"
        raise FreeimageError(msg)
    _ = (name, mime)  # freeimage.host takes raw bytes; names kept for signature parity
    source = base64.b64encode(data).decode("ascii")
    session = AsyncSession()
    try:
        response = await session.post(
            f"{config.FREEIMAGE_BASE_URL}/api/1/upload",
            data={
                "key": config.FREEIMAGE_API_KEY,
                "action": "upload",
                "source": source,
                "format": "json",
            },
            timeout=120,
        )
        if response.status_code not in {200, 201}:
            msg = f"upload failed HTTP {response.status_code}: {response.text[:200]}"
            raise FreeimageError(msg)
        payload = response.json()
        image = payload.get("image") if isinstance(payload, dict) else None
        url = image.get("url") if isinstance(image, dict) else None
        if not isinstance(url, str) or not url:
            display = image.get("display_url") if isinstance(image, dict) else None
            if not isinstance(display, str) or not display:
                msg = "no url in response"
                raise FreeimageError(msg)
            return display
        return url
    finally:
        await session.close()
