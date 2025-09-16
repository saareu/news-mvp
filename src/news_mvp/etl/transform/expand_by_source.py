#!/usr/bin/env python3
"""
expand_by_source.py

Expand HTML/JSON-like fields in a post-processed CSV into separate columns.

Usage:
  python -m etl.transform.expand_by_source --input data/raw/ynet/ynet_20250915_100534.csv \
      --output data/raw/ynet/ynet_20250915_100534_expanded.csv \
      --columns description,title --force

Behavior:
 - Reads the input CSV (UTF-8-SIG expected).
 - For every column listed in `--columns` (or all columns if not provided),
   the script attempts to detect HTML fragments or JSON-like dicts and
   expand them into additional columns.

Notes: see module for full behavior and examples.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Any, Sequence

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception as exc:  # pragma: no cover - hard error path
    raise RuntimeError("beautifulsoup4 is required by expand_by_source.py but is not installed") from exc

LOG = logging.getLogger("expand_by_source")

HTML_DETECT_RE = re.compile(r"<[^>]+>")
JSON_LIKE_RE = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)


def parse_html_fragment(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not text or not HTML_DETECT_RE.search(text):
        return out

    try:
        soup = BeautifulSoup(text, "html.parser")
    except Exception as e:  # pragma: no cover - unlikely but safe
        LOG.debug("BeautifulSoup parse failed: %s", e)
        return out

    visible = soup.get_text(" ", strip=True)
    out["text"] = visible

    a = soup.find("a")
    if a is not None:
        try:
            a_attrs = getattr(a, "attrs", None) or {}
            href = a_attrs.get("href")
        except Exception:
            href = None
        if href:
            out["href"] = href

    img = soup.find("img")
    if img is not None:
        try:
            attrs = getattr(img, "attrs", None) or {}
        except Exception:
            attrs = {}
        src = attrs.get("src") or attrs.get("data-src")
        if src:
            out["img"] = src
        for attr in ("alt", "title", "width", "height", "border"):
            if attr in attrs and attrs.get(attr) is not None:
                out[f"img_{attr}"] = attrs.get(attr)

    tags = [getattr(t, "name", None) for t in soup.find_all()]
    tags = [t for t in tags if t]
    out["_tags"] = ",".join(tags)

    # If the fragment only contains trivial formatting tags and no anchors/images
    # or meaningful text, treat it as not significant.
    trivial_tags = {"br", "p", "span"}
    meaningful = any(t for t in tags if t not in trivial_tags)
    # visible text ignoring whitespace
    if not meaningful and (not visible or visible.strip() == ""):
        return {}
    return out


def parse_json_like(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not text or not JSON_LIKE_RE.search(text):
        return out
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            out[str(k)] = v
    return out


def expand_row(row: Dict[str, str], columns: List[str]) -> Tuple[Dict[str, str], bool]:
    new_row = dict(row)
    changed = False

    for col in columns:
        if col not in row:
            continue
        val = row[col]
        if val is None:
            continue
        html_parts = parse_html_fragment(val)
        if html_parts:
            # Determine if the HTML parts are significant enough to treat as a change.
            trivial_tags = {"br", "p", "span"}
            tags_str = str(html_parts.get("_tags", ""))
            tags_list = [t for t in tags_str.split(",") if t]
            has_nontrivial_tag = any(t for t in tags_list if t not in trivial_tags)
            has_href = "href" in html_parts
            has_img = "img" in html_parts
            text_only = ("text" in html_parts) and (not has_href) and (not has_img) and (not has_nontrivial_tag)

            if "text" in html_parts and not text_only:
                key = f"{col}_text"
                if new_row.get(key, "") != str(html_parts["text"]):
                    new_row[key] = str(html_parts["text"])
                    changed = True
            if "href" in html_parts:
                key = f"{col}_href"
                if new_row.get(key, "") != str(html_parts["href"]):
                    new_row[key] = str(html_parts["href"])
                    changed = True
            if "img" in html_parts:
                key = f"{col}_img"
                if new_row.get(key, "") != str(html_parts["img"]):
                    new_row[key] = str(html_parts["img"])
                    changed = True
            # Add other non-text parts only if the fragment was considered significant
            if not text_only:
                for k, v in html_parts.items():
                    if k in ("text", "href", "img"):
                        continue
                    key = f"{col}_{k}"
                    if new_row.get(key, "") != str(v):
                        new_row[key] = str(v)
                        changed = True
            continue

        json_parts = parse_json_like(val)
        if json_parts:
            for k, v in json_parts.items():
                key = f"{col}_{k}"
                if new_row.get(key, "") != str(v):
                    new_row[key] = str(v)
                    changed = True
            continue

    return new_row, changed


def discover_columns_to_process(
    header: Sequence[str], requested: List[str] | None, rows: Sequence[Dict[str, str]] | None = None
) -> List[str]:
    """
    Decide which columns should be inspected/expanded.

    - If `requested` is provided, return the intersection with `header` preserving order.
    - Otherwise, inspect the header and (if provided) every row to find evidence of
      HTML fragments or JSON-like dicts. This uses `HTML_DETECT_RE`, `JSON_LIKE_RE`
      and simple substring heuristics like 'href=' and 'src=' to avoid missing
      sparse occurrences.

    The returned list preserves the order of `header`.
    """
    if requested:
        return [c for c in requested if c in header]

    # Base candidate names that are commonly expanded
    candidates = {"description", "desc", "content", "enclosure", "image", "guid", "link", "title"}

    def normalize_header(h: str) -> str:
        # strip XML namespace wrappers like '{http://...}content'
        if h and h.startswith("{") and "}" in h:
            try:
                return h.split("}", 1)[1].lower()
            except Exception:
                pass
        return h.lower() if h else ""

    cols: List[str] = []
    # quick header-based checks first
    for h in header:
        nh = normalize_header(h)
        if nh in candidates:
            cols.append(h)
            continue
        if HTML_DETECT_RE.search(h) or JSON_LIKE_RE.search(h) or JSON_LIKE_RE.search(nh):
            cols.append(h)

    # If rows provided, scan all rows to find any column that contains signs of HTML/JSON
    if rows:
        for h in header:
            if h in cols:
                continue
            found = False
            for r in rows:
                try:
                    v = r.get(h, "")
                except Exception:
                    v = ""
                if not v:
                    continue
                # HTML or JSON-like regex
                if HTML_DETECT_RE.search(v) or JSON_LIKE_RE.search(v):
                    found = True
                    break
                # common attribute patterns inside HTML fragments — only count them
                # when angle brackets are present to avoid matching plain URLs.
                low = v.lower()
                if ("<" in v or ">" in v) and ("href=" in low or "src=" in low or "data-src" in low):
                    found = True
                    break
            if found:
                cols.append(h)

    # Ensure we return columns in header order and unique
    seen = set()
    ordered: List[str] = []
    for h in header:
        if h in cols and h not in seen:
            ordered.append(h)
            seen.add(h)
    return ordered


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Expand HTML/JSON-like fields in CSV into multiple columns")
    p.add_argument("--input", required=True, help="Relative path to input CSV")
    p.add_argument("--output", required=True, help="Relative path to output CSV")
    p.add_argument("--columns", nargs="*", help="List of columns to try to expand (defaults: heuristics)")
    p.add_argument("--force", action="store_true", help="Always write output even if no changes")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        LOG.error("Input file does not exist: %s", input_path)
        return 1

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []

    # If no explicit columns provided, inspect the whole CSV rows to detect columns
    cols_to_try = args.columns if args.columns else discover_columns_to_process(header, None, rows)
    LOG.info("Columns to attempt expansion: %s", cols_to_try)

    new_rows = []
    any_changed = False
    new_fieldnames = list(header)

    for row in rows:
        new_row, changed = expand_row(row, cols_to_try)
        if changed:
            any_changed = True
            for k in new_row.keys():
                if k not in new_fieldnames:
                    new_fieldnames.append(k)
        new_rows.append(new_row)

    if not any_changed and not args.force:
        LOG.info("No changes detected but writing expanded CSV anyway (per request)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for r in new_rows:
            out_r = {k: r.get(k, "") for k in new_fieldnames}
            writer.writerow(out_r)

    LOG.info("Wrote expanded CSV: %s (changed=%s)", output_path, any_changed)

    try:
        rel = output_path.relative_to(Path.cwd())
    except Exception:
        rel = output_path

    print(str(rel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
