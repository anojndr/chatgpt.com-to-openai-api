# Copyright 2026 chatgpt-to-openai-api contributors.
"""PixelVault image host client (docs: https://pixelvault.dev/docs/)."""

from __future__ import annotations

import logging

from curl_cffi import CurlMime
from curl_cffi.requests import AsyncSession

from . import config

log = logging.getLogger("pixelvault")


class PixelVaultError(Exception):
    """PixelVault image-host request failed."""

    def __init__(self, message: str = "PixelVault request failed") -> None:
        """Store the failure message."""
        super().__init__(message)


async def upload_image(name: str, data: bytes, mime: str) -> str:
    """Upload image bytes, returning the public CDN URL."""
    if not config.PIXELVAULT_API_KEY:
        msg = "PIXELVAULT_API_KEY not configured"
        raise PixelVaultError(msg)
    session = AsyncSession()
    try:
        part = CurlMime()
        part.addpart(name="file", filename=name, content_type=mime, data=data)
        response = await session.post(
            f"{config.PIXELVAULT_BASE_URL}/v1/images",
            headers={"Authorization": f"Bearer {config.PIXELVAULT_API_KEY}"},
            multipart=part,
            timeout=120,
        )
        if response.status_code not in (200, 201):
            msg = f"upload failed HTTP {response.status_code}: {response.text[:200]}"
            raise PixelVaultError(msg)
        payload = response.json()
        image = payload.get("data") if isinstance(payload, dict) else None
        url = image.get("url") if isinstance(image, dict) else None
        if not isinstance(url, str) or not url:
            msg = "no url in response"
            raise PixelVaultError(msg)
        return url
    finally:
        await session.close()
