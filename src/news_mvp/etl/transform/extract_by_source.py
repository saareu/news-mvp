#!/usr/bin/env python3
"""
extract_by_source.py

Unified CLI UI and documentation
--------------------------------

This module runs a conservative fetch -> xml -> json -> csv pipeline for any
RSS feed. It's intentionally source-agnostic: callers provide a `--source` name
which is used only for file naming and an `--rss-url` to fetch.

Primary behavior
 - Fetch RSS bytes from the given `--rss-url`.
 - Save the raw XML to `data/raw/{source}/{source}_{TIMESTAMP}.xml`.
 - Convert XML to JSON (via `etl.transform.xml_to_json.xml_file_to_json`) and
   write `{source}_{TIMESTAMP}.json` next to the XML (UTF-8 with BOM).
 - Extract items and write `{source}_{TIMESTAMP}.csv` next to the JSON.
 - Optionally post-process the CSV using `etl.transform.postprocess_csv.postprocess_csv`.
 - Print the generated CSV path to stdout (one-line, last line) for downstream
   scripts to consume and exit with code 0 on success or non-zero on failure.

Command-line interface (CLI)
---------------------------

All flags are POSIX-style long options. The script returns the generated CSV
file path on stdout (printed as a single line). Use this path in subprocess
pipelines.

Arguments
 - `--source` (required)
     Name used for file naming. Example: `ynet`, `haaretz`, `customblog`.

 - `--rss-url` (required)
     Full URL to the RSS feed to fetch.

 - `--output-dir` (optional)
     A `Path` to use as the base output directory for raw files. If omitted,
     files are written to `data/raw/{source}` (created if missing).

 - `--timeout` (optional, default: 20.0)
     HTTP fetch timeout in seconds (float).

 - `--user-agent` (optional)
     Custom User-Agent string to send with the HTTP request.

 - `--timestamp` (optional)
     Provide a specific timestamp string (format: `YYYYMMDD_HHMMSS`) to be
     embedded in generated filenames. When omitted, the script generates a
     timestamp at runtime.

 - `--no-postprocess` (flag)
     If present, the canonical CSV post-processing step is skipped and the
     raw CSV produced by `json_to_csv` is left unchanged.

 - `--verbose` (flag)
     Enable verbose logging (DEBUG level).

Examples
 - Basic run (uses default output dir):

   python -m etl.transform.extract_by_source \
       --source ynet \
       --rss-url "https://www.ynet.co.il/Integration/StoryRss2.xml"

 - Specify output dir and timestamp:

   python -m etl.transform.extract_by_source \
       --source customblog \
       --rss-url "https://customblog.example/feed.xml" \
       --output-dir data/raw/customblog \
       --timestamp 20250101_120000

Return behavior
 - On success: prints the path to the CSV file (single line) and exits 0.
 - On failure: logs the error and exits with a non-zero status code.

Notes for pipeline integration
 - This script is intended to be used as a component in larger pipelines. The
   printed CSV path is suitable for consumption by `subprocess` calls in later
   stages (canonization, appending, etc.).

"""
from __future__ import annotations

import argparse
import json
import logging
import re
import html as _html
import csv
import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from etl.extract import fetch_rss_bytes
from etl.config import RAW_DIR

from etl.transform import xml_to_json
from etl.transform import json_to_csv
from etl.transform.postprocess_csv import postprocess_csv

LOG = logging.getLogger("extract_by_source")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch source, convert XML->JSON->CSV and return CSV path")
    p.add_argument("--source", help="Source name for file naming (e.g., ynet, hayom, haaretz, custom)", required=True)
    p.add_argument("--rss-url", help="URL of the RSS feed to fetch", required=True)
    p.add_argument("--output-dir", help="Output directory for raw files (default: data/raw/{source})", type=Path)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--user-agent")
    p.add_argument("--timestamp", help="Use specific timestamp in filenames for pipeline consistency")
    p.add_argument("--no-postprocess", action="store_true", help="Skip CSV post-processing (leave raw CSV from json_to_csv)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _relative_print(p: Path) -> None:
    try:
        rel = p.relative_to(Path.cwd())
        print(str(rel))
    except Exception:
        print(p.name)


def run_pipeline_for_source(source: str, rss_url: str, out_dir: Path | None = None, 
                         timeout: float = 20.0, user_agent: str | None = None, 
                         do_postprocess: bool = True, timestamp: str | None = None) -> Path:
    """Run the fetch->xml->json->csv pipeline for any source.

    Args:
        source: Source name for file naming
        rss_url: URL of the RSS feed to fetch
        out_dir: Output directory for raw files (defaults to data/raw/{source})
        timeout: HTTP request timeout
        user_agent: Custom user agent for requests
        do_postprocess: Whether to post-process the CSV
        timestamp: Specific timestamp to use in filenames for pipeline consistency

    Returns the path to the generated CSV file.
    """
    # Create a standard output directory if not provided
    if out_dir is None:
        # Always create a directory under raw/
        from etl.config import RAW_DIR
        out_dir = RAW_DIR / source.lower()
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Fetching %s -> %s", source, rss_url)
    raw_bytes = fetch_rss_bytes(rss_url, timeout=timeout, user_agent=user_agent)

    # Use provided timestamp or generate a new one
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    xml_name = f"{source}_{timestamp}.xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / xml_name
    with open(xml_path, "wb") as fh:
        fh.write(raw_bytes)

    LOG.info("Saved raw XML -> %s", xml_path)

    # Convert XML -> JSON (in-memory), then write JSON file next to XML
    data = xml_to_json.xml_file_to_json(xml_path)
    json_path = xml_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8-sig") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    LOG.info("Wrote JSON -> %s", json_path)

    # Extract items for CSV
    # xml_file_to_json returns {root_tag: {...}} so find_items expects that root mapping's value
    root_value = next(iter(data.values())) if isinstance(data, dict) and data else data
    items = json_to_csv.find_items(root_value if isinstance(root_value, dict) else root_value)

    csv_path = xml_path.with_suffix(".csv")
    json_to_csv.json_items_to_csv(items, csv_path)
    LOG.info("Wrote CSV -> %s", csv_path)

    if do_postprocess:
        # Use the canonical postprocessor (includes parsing heuristics and reporting)
        postprocess_csv(csv_path)
        LOG.info("Post-processed CSV -> %s", csv_path)

    return csv_path


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Pass the timestamp if provided
    csv_path = run_pipeline_for_source(
        args.source,
        rss_url=args.rss_url,
        out_dir=args.output_dir,
        timeout=args.timeout, 
        user_agent=args.user_agent, 
        do_postprocess=not args.no_postprocess,
        timestamp=args.timestamp
    )
    
    # Print the path for downstream processes
    print(str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
