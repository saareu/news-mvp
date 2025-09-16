#!/usr/bin/env python3
# Minimal RSS fetch-and-save script.
"""
Usage: python -m etl.extract.rss_extractor --source <ynet|hayom|haaretz>

This module only:
- fetches the raw RSS/Atom bytes via `etl.extract.rss_fetcher.fetch_rss_bytes`
- saves them to the configured RAW_* directory with filename
  `{source}_{YYYYmmdd_HHMMSS}.xml`
- prints the saved file path (relative) to stdout
- performs no parsing or JSON/CSV output
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from etl.extract.rss_fetcher import fetch_rss_bytes
from etl.config import (
    RAW_YNET_DIR,
    RAW_HAYOM_DIR,
    RAW_HAARETZ_DIR,
    YNET_RSS_URL,
    HAYOM_RSS_URL,
    HAARETZ_RSS_URL,
)

LOG = logging.getLogger("rss_fetch_and_save")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch RSS and save raw XML only")
    p.add_argument("--source", choices=["ynet", "hayom", "haaretz"], required=True)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--user-agent")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _relative_print(p: Path) -> None:
    """Print path relative to cwd; fallback to filename if relative_to fails."""
    try:
        rel = p.relative_to(Path.cwd())
        print(str(rel))
    except Exception:
        # Fallback to filename only
        print(p.name)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.source == "ynet":
        url = YNET_RSS_URL
        out_dir = RAW_YNET_DIR
    elif args.source == "hayom":
        url = HAYOM_RSS_URL
        out_dir = RAW_HAYOM_DIR
    else:
        url = HAARETZ_RSS_URL
        out_dir = RAW_HAARETZ_DIR

    LOG.info("Fetching source %s -> %s", args.source, url)
    raw = fetch_rss_bytes(url, timeout=args.timeout, user_agent=args.user_agent)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.source}_{timestamp}.xml"

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    with open(path, "wb") as fh:
        fh.write(raw)

    # Print the saved filename (relative path) to stdout as the only output
    _relative_print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
