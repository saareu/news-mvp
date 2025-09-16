#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rss_fetcher.py — minimal, typed fetcher with stdlib fallback
------------------------------------------------------------
Single responsibility: fetch RSS/Atom XML bytes over HTTP(S).

API: fetch_rss_bytes(url: str, *, timeout: float = 20.0, user_agent: Optional[str] = None) -> bytes
No parsing, no transformation.
Uses `httpx` if available, otherwise falls back to `urllib.request`.
"""
from __future__ import annotations

from typing import Optional

try:
    import httpx  # type: ignore
    _HAS_HTTPX = True
except Exception:  # pragma: no cover - best-effort fallback
    httpx = None  # type: ignore
    _HAS_HTTPX = False

DEFAULT_UA = "Mozilla/5.0 (compatible; rss-fetcher/1.0)"


def fetch_rss_bytes(url: str, *, timeout: float = 20.0, user_agent: Optional[str] = None) -> bytes:
    """Fetch an RSS/Atom URL and return the raw response bytes (no changes).

    Uses `httpx` when installed for nicer timeout handling; otherwise falls back
    to `urllib.request` from the standard library.
    """
    ua = user_agent or DEFAULT_UA

    if _HAS_HTTPX and httpx is not None:
        # mypy/static checkers might complain if httpx is None; local binding narrows the type
        httpx_client = httpx
        headers = {"User-Agent": ua}
        with httpx_client.Client(follow_redirects=True, headers=headers, timeout=timeout) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.content

    # Fallback to urllib
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": ua})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()
