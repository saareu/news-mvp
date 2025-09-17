"""Utilities for reversible, CSV-safe id_seed encoding.

We encode the seed using percent-encoding (RFC 3986-style) so the resulting
token is ASCII-only and stable across file encodings (including UTF-8-SIG).

The functions here intentionally avoid inserting BOMs or non-ASCII bytes.
"""

from __future__ import annotations

import urllib.parse
from typing import Optional
import base64
import hashlib
import re
import unicodedata


def encode_id_seed(seed: str) -> str:
    """Return an ASCII-only, reversible token for the given seed string.

    Uses percent-encoding with all characters escaped (safe="") so the
    output is plain ASCII and safe to store in CSV files that may include
    a BOM at the file level.
    """
    return urllib.parse.quote(seed, safe="")


def decode_id_seed(token: str) -> str:
    """Reverse an `encode_id_seed` token back into the original seed.

    Returns the decoded seed string. If `token` is None or empty, returns
    an empty string.
    """
    if not token:
        return ""
    return urllib.parse.unquote(token)


def try_decode_id_seed(token: Optional[str]) -> Optional[str]:
    """Convenience: return decoded seed or None if token is falsy."""
    if not token:
        return None
    return decode_id_seed(token)


# New deterministic ID helpers
_PERSONALIZATION = b"news-id-v1"


def _norm_title(t: str) -> str:
    # Unicode canonical form, strip, collapse whitespace
    t = unicodedata.normalize("NFC", t).strip()
    return re.sub(r"\s+", " ", t)


def make_news_id(title_val: str, pub_val: str, src_val: str, nbytes: int = 12) -> str:
    """
    Returns a short, URL/filename-safe ID.
    nbytes=12 -> 96-bit hash -> 16-char base64url (no padding).
    Increase to 16 for 128-bit if you want even lower collision odds.
    """
    seed = f"{_norm_title(title_val)}|{(pub_val or '').strip()}|{(src_val or '').strip().lower()}"
    h = hashlib.blake2b(
        seed.encode("utf-8"), digest_size=nbytes, person=_PERSONALIZATION
    )
    return base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode("ascii")


def make_news_id_int(title_val: str, pub_val: str, src_val: str, bits: int = 64) -> int:
    """
    Deterministic numeric ID variant (e.g., for SQL BIGINT keys).
    bits must be a multiple of 8 (e.g., 64 or 128).
    """
    seed = f"{_norm_title(title_val)}|{(pub_val or '').strip()}|{(src_val or '').strip().lower()}"
    h = hashlib.blake2b(
        seed.encode("utf-8"), digest_size=bits // 8, person=_PERSONALIZATION
    )
    return int.from_bytes(h.digest(), "big")
