"""PixelVault image host client (docs: https://pixelvault.dev/docs/)."""
from __future__ import annotations

import logging

from curl_cffi import CurlMime
from curl_cffi.requests import AsyncSession

from . import config

log = logging.getLogger("pixelvault")


class PixelVaultError(Exception):
    pass


async def upload_image(name: str, data: bytes, mime: str) -> str:
    """Upload image bytes, return public CDN URL."""
    if not config.PIXELVAULT_API_KEY:
        raise PixelVaultError("PIXELVAULT_API_KEY not configured")
    s = AsyncSession()
    try:
        m = CurlMime()
        m.addpart(name="file", filename=name, content_type=mime, data=data)
        r = await s.post(
            f"{config.PIXELVAULT_BASE_URL}/v1/images",
            headers={"Authorization": f"Bearer {config.PIXELVAULT_API_KEY}"},
            multipart=m,
            timeout=120,
        )
        if r.status_code not in (200, 201):
            raise PixelVaultError(f"upload failed HTTP {r.status_code}: {r.text[:200]}")
        url = (r.json().get("data") or {}).get("url")
        if not url:
            raise PixelVaultError("no url in response")
        return url
    finally:
        await s.close()
