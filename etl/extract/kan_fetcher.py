#!/usr/bin/env python3
"""kan_fetcher.py

Selenium-only fetcher for the Kan newsflash page.

This script uses Selenium + webdriver-manager to launch a headless Chrome
instance, navigate to `https://www.kan.org.il/newsflash/`, and save the
rendered HTML to `data/raw/kan/kan_{timestamp}.html`.

This simplifies the codebase by providing a single reliable fetch method that
behaves like a real browser and avoids many programmatic HTTP blocks.

Requirements:
    pip install selenium webdriver-manager

Usage: python -m etl.extract.kan_fetcher [--headless] [--timeout 30]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


from etl.config import BASE_DIR

LOG = logging.getLogger("kan_fetcher")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selenium fetcher for Kan newsflash")
    p.add_argument("--headless", action="store_true", default=True,
                   help="Run Chrome in headless mode (default: True)")
    p.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Run Chrome with a visible window (useful for debugging)")
    p.add_argument("--timeout", type=int, default=30,
                   help="Page load timeout in seconds")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    url = "https://www.kan.org.il/newsflash/"
    out_dir = BASE_DIR / "data" / "raw" / "kan"
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Starting Selenium Chrome (headless=%s)", args.headless)
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except Exception as e:  # pragma: no cover - dependency error
        LOG.error("Selenium or webdriver-manager not installed: %s", e)
        LOG.error("Install with: pip install selenium webdriver-manager")
        return 2

    options = Options()
    if args.headless:
        # new headless mode if available
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,1024")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        LOG.error("Failed to start ChromeDriver: %s", e)
        return 2

    try:
        driver.set_page_load_timeout(args.timeout)
        driver.get(url)
        html = driver.page_source
    except Exception as e:
        LOG.error("Error fetching page: %s", e)
        driver.quit()
        return 2

    driver.quit()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"kan_{timestamp}.html"
    path.write_text(html, encoding="utf-8")

    try:
        rel = path.relative_to(Path.cwd())
        print(str(rel))
    except Exception:
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
