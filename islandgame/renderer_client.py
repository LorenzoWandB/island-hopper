"""Best-effort event push to the renderer server. The game must work with no browser open."""
from __future__ import annotations

import os
from typing import Any

import httpx

RENDERER_URL = os.environ.get("ISLAND_RENDERER_URL", "http://127.0.0.1:8765")


def emit(event: dict[str, Any], url: str | None = None) -> None:
    try:
        httpx.post(f"{url or RENDERER_URL}/event", json=event, timeout=1.0)
    except Exception:
        pass
