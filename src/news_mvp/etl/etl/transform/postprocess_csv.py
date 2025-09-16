#!/usr/bin/env python3
"""
Post-process an existing CSV: parse JSON-like fields, Python literals, pipe-joined items
and strip HTML. Overwrites the CSV in-place with UTF-8 BOM encoding.

Usage:
    py -m etl.transform.postprocess_csv path/to/file.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import ast
import html as _html
from pathlib import Path
from typing import Any
from html.parser import HTMLParser

# Prefer BeautifulSoup for HTML->text extraction when available
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def get_data(self):
        return "".join(self.fed)


def html_to_text(s: str) -> str:
    """Convert HTML to plain text using BeautifulSoup if available, otherwise
    fall back to a simple HTMLParser-based extractor."""
    if not s:
        return ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(s, "lxml")
        except Exception:
            soup = BeautifulSoup(s, "html.parser")
        return _html.unescape(soup.get_text(separator=" ", strip=True))

    stripper = _HTMLStripper()
    try:
        stripper.feed(s)
        text = stripper.get_data()
    finally:
        try:
            stripper.close()
        except Exception:
            pass
    return _html.unescape(text)


def extract_text_from_obj(obj: Any) -> str:
    if isinstance(obj, dict):
        if "#text" in obj:
            return str(obj["#text"]) or ""
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def parse_cell_value(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if not s:
        return ""

    # Try JSON
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = None

    if parsed is None:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None

    if parsed is not None:
        if isinstance(parsed, list):
            parts = [extract_text_from_obj(x) for x in parsed]
            return "|".join(parts)
        if isinstance(parsed, dict):
            return extract_text_from_obj(parsed)
        return str(parsed)

    if "|" in s:
        parts = []
        for part in s.split("|"):
            part = part.strip()
            if not part:
                continue
            pval = None
            try:
                pval = json.loads(part)
            except Exception:
                try:
                    pval = ast.literal_eval(part)
                except Exception:
                    pval = None
            if pval is not None:
                if isinstance(pval, dict):
                    parts.append(extract_text_from_obj(pval))
                elif isinstance(pval, list):
                    parts.append("|".join(extract_text_from_obj(x) for x in pval))
                else:
                    parts.append(str(pval))
            else:
                parts.append(html_to_text(part))
        return "|".join(parts)

    # Fallback: strip HTML using parser
    return html_to_text(s)


def postprocess_csv(path: Path) -> None:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return
    header = rows[0]
    data_rows = rows[1:]

    # Process each cell
    processed = []
    changed = 0
    total = 0
    for row in data_rows:
        # pad row if shorter
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        row_out = []
        for cell in row[: len(header)]:
            total += 1
            new = parse_cell_value(cell)
            if new != (cell or ""):
                changed += 1
            row_out.append(new)
        processed.append(row_out)

    # Write back
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in processed:
            writer.writerow(r)
    # Print summary
    print(f"Processed {len(processed)} rows, {total} cells, {changed} changed")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Postprocess CSV cells (parse JSON/python literals and strip HTML)")
    p.add_argument("input", help="CSV file path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    postprocess_csv(p)
    print(str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
