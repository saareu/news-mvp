#!/usr/bin/env python3
"""canonized_by_source.py

Map an expanded per-source CSV into the canonical schema as described in
`etl/schema/mapping.csv`.

Usage:
    python -m etl.transform.canonized_by_source --input <expanded_csv>

Behavior:
 - Requires an expanded CSV produced by `expand_by_source.py`.
 - Reads `etl/schema/mapping.csv` and for the detected source (derived from
     the input path) maps source columns into canonical columns.
 - Supports concatenation specs in the mapping values. A concat spec uses
     '+' to separate field names and delimiter tokens, e.g.
             description_img_title+-+description_img_alt
     This splits into tokens: ['description_img_title','-','description_img_alt']
     Tokens at even indices are field names; tokens at odd indices are delimiters.
     The script validates the structure and will error on malformed specs.
 - For concatenated fields, if all the joined values for a row are identical,
     the result is that single value (no delimiters). Otherwise the values are
     combined with the provided delimiters (preserving order).
 - After mapping, the script computes a deterministic `id` (SHA1 of guid if
     present else of title), infers `language` from the title (Hebrew -> 'he',
     otherwise 'en'), and strips query params from `image` URLs (remove '?' and
     everything after).
 - Writes canonical CSV into a configurable directory (default: data/canonical/{source}/{basename}_canonical.csv`) and
     prints the relative path as the final stdout line for subprocess chaining.

All paths are configurable via etl/config.py and environment/YAML.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import email.utils
from datetime import datetime, timezone
try:
    from langdetect import detect as langdetect_detect  # type: ignore
except Exception:
    langdetect_detect = None

LOG = logging.getLogger("canonized_by_source")


def detect_source_from_path(p: Path) -> str:
    # Expect path like data/raw/{source}/... or data/canonical/{source}/...
    parts = p.parts
    try:
        idx = parts.index("raw")
        return parts[idx + 1]
    except ValueError:
        # fallback: try canonical
        try:
            idx = parts.index("canonical")
            return parts[idx + 1]
        except ValueError:
            # last resort: parent folder name
            return p.parent.name


def read_mapping(mapping_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(mapping_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # canonical column order = values in the 'canonical' column
        rows = [r for r in reader]
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, rows


def parse_concat_spec(spec: str) -> Tuple[List[str], List[str]]:
    """Parse a concat specification.

    Returns (fields, delimiters). Validation: tokens = spec.split('+') must be
    odd length >= 1 and tokens at even indices are field names.
    Example: 'a+-+b' -> tokens ['a','-','b'] -> fields ['a','b'], delims ['-']
    """
    tokens = [t for t in spec.split("+")]
    if len(tokens) == 0:
        raise ValueError(f"Empty concat spec: {spec!r}")
    if len(tokens) % 2 == 0:
        raise ValueError(f"Malformed concat spec (must have odd number of '+'-separated tokens): {spec!r}")
    fields = [tokens[i].strip() for i in range(0, len(tokens), 2)]
    delims = [tokens[i] for i in range(1, len(tokens), 2)]
    if any(not f for f in fields):
        raise ValueError(f"Malformed concat spec, empty field name: {spec!r}")
    return fields, delims


def combine_concat(spec: str, row: Dict[str, str]) -> str:
    fields, delims = parse_concat_spec(spec)
    values = [row.get(f, "") for f in fields]
    # If all values equal (including empty), return the first value
    if all(v == values[0] for v in values):
        return values[0]
    # Otherwise, interleave values and delimiters
    out_parts: List[str] = []
    for i, v in enumerate(values):
        out_parts.append(v or "")
        if i < len(delims):
            out_parts.append(delims[i])
    return "".join(out_parts)


def sha1_of_text(t: str) -> str:
    return hashlib.sha1((t or "").encode("utf-8")).hexdigest()


def detect_language_from_title(title: str) -> str:
    # Simple heuristic: detect scripts by Unicode ranges
    if not title:
        return "und"
    has_latin = False
    for ch in title:
        # Arabic block
        if "\u0600" <= ch <= "\u06FF":
            return "ar"
        # Hebrew block
        if "\u0590" <= ch <= "\u05FF":
            return "he"
        # Latin letters
        if ('A' <= ch <= 'Z') or ('a' <= ch <= 'z'):
            has_latin = True
    if has_latin:
        # If langdetect is available, use it to detect specific Latin language codes
        if langdetect_detect is not None:
            try:
                code = langdetect_detect(title)
                # langdetect may return things like 'en', 'fr', 'es'
                return code
            except Exception:
                return "en"
        return "en"
    return "und"


def strip_url_query(url: str) -> str:
    if not url:
        return url
    return url.split("?", 1)[0]


def normalize_pubdate(date_str: str, force_tz_offset: int | None = None) -> str:
    """Try to normalize several common date formats into ISO 8601 (UTC offset preserved when available).

    If force_tz_offset is not None, output will always use that offset (e.g., 3 for +03:00, -5 for -05:00).
    """
    if not date_str:
        return ""
    dt = None
    orig_tz = None
    # Try RFC 2822 via email.utils
    try:
        parsed = email.utils.parsedate_tz(date_str)
        if parsed:
            # parsed: (year, month, day, hour, min, sec, wday, yday, isdst, offset)
            y, m, d, H, M, S = parsed[:6]
            offset = parsed[9] if len(parsed) > 9 else None
            if offset is not None:
                from datetime import timezone, timedelta
                orig_tz = timezone(timedelta(seconds=offset))
                dt = datetime(y, m, d, H, M, S, tzinfo=orig_tz)
            else:
                dt = datetime(y, m, d, H, M, S)
    except Exception:
        pass
    if dt is None:
        # Try ISO 8601 parse
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(date_str, fmt)
                # If no tzinfo, leave naive
                break
            except Exception:
                continue
    if dt is not None:
        if force_tz_offset is not None:
            # Replace tzinfo with forced offset, do NOT shift the time
            from datetime import timedelta, timezone as dt_timezone
            tz = dt_timezone(timedelta(hours=force_tz_offset))
            dt = dt.replace(tzinfo=tz)
        # Output in ISO 8601 format
        return dt.isoformat()
    # As a last resort, return the original string
    return date_str


def build_canonical_rows(
    input_rows: List[Dict[str, str]],
    mapping_rows: List[Dict[str, str]],
    fieldnames: List[str],
    source: str,
    source_col_name: str,
) -> Tuple[List[str], List[Dict[str, str]]]:
    # canonical columns order defined by mapping_rows 'canonical' values
    canonical_cols = [r["canonical"] for r in mapping_rows]

    out_rows: List[Dict[str, str]] = []

    # build map canonical -> spec for this source
    spec_for: Dict[str, str] = {}
    for r in mapping_rows:
        spec_for[r["canonical"]] = r.get(source_col_name, "")


    for idx, row in enumerate(input_rows, start=1):
        out: Dict[str, str] = {}
        for canon in canonical_cols:
            spec = (spec_for.get(canon) or "").strip()
            if not spec or spec.lower() == "none":
                out[canon] = ""
                continue
            # If it's a concat spec (contains '+') treat specially
            if "+" in spec:
                try:
                    val = combine_concat(spec, row)
                except Exception as e:
                    raise RuntimeError(f"Invalid concat spec for canonical '{canon}': {spec!r}: {e}") from e
                out[canon] = val
            else:
                # simple mapping: take value from source column name
                out[canon] = row.get(spec, "")

        # Before computing id, ensure canonical 'source' is filled from detected source
        if "source" in out and (not out.get("source")):
            out["source"] = source

        # normalize pubDate if present
        if out.get("pubDate"):
            out["pubDate"] = normalize_pubdate(out["pubDate"], force_tz_offset=build_canonical_rows.force_tz_offset) or out["pubDate"]

        # Enforce non-nullable contract: title, pubDate, and source must be present
        missing = []
        title_val = (out.get("title") or "").strip()
        if not title_val:
            missing.append("title")
        pub_val = (out.get("pubDate") or "").strip()
        if not pub_val:
            missing.append("pubDate")
        src_val = (out.get("source") or "").strip()
        if not src_val:
            missing.append("source")
        if missing:
            guid_preview = out.get("guid") or ""
            raise RuntimeError(
                f"Missing required fields {missing} for input row {idx} (title={title_val!r}, pubDate={pub_val!r}, source={src_val!r}, guid={guid_preview!r})"
            )

        # id hashing: compute from title|pubDate|source (ignore guid) using validated values
        seed = f"{title_val}|{pub_val}|{src_val}"
        out["id"] = sha1_of_text(seed)

        # language detection from title
        if not out.get("language"):
            out["language"] = detect_language_from_title(out.get("title", ""))

        # image cleanup
        if out.get("image"):
            out["image"] = strip_url_query(out["image"])

        out_rows.append(out)

    return canonical_cols, out_rows


def main(argv=None) -> int:


    from etl.config import CANON_DIR
    p = argparse.ArgumentParser(description="Map expanded per-source CSV into canonical CSV using etl/schema/mapping.csv. All paths are configurable via etl/config.py and environment/YAML.")
    p.add_argument("--input", required=True, help="Path to expanded input CSV")
    p.add_argument("--output", help=f"Optional output path; if omitted, writes to <CANON_DIR>/{{source}}/<basename>_canonical.csv (default: {CANON_DIR}/{{source}})")
    p.add_argument("--mapping", default="etl/schema/mapping.csv", help="Path to mapping CSV")
    p.add_argument("--force-tz-offset", type=int, default=None, help="Force output timezone offset in hours (e.g., 3 for +03:00, -5 for -05:00)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = Path(args.input)
    if not input_path.exists():
        LOG.error("Input file does not exist: %s", input_path)
        return 1

    source = detect_source_from_path(input_path)
    LOG.info("Detected source: %s", source)

    # mapping file
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        LOG.error("Mapping file not found: %s", mapping_path)
        return 1

    fieldnames, mapping_rows = read_mapping(mapping_path)
    # mapping_rows is list of dicts with keys equal to CSV header: canonical, haaretz, hayom, ynet

    # Determine the mapping column name that corresponds to the detected source
    # Try exact match, else lowercase match
    candidate_cols = [c for c in fieldnames if c]
    source_col_name = None
    for c in candidate_cols:
        if c.lower() == source.lower():
            source_col_name = c
            break
    if source_col_name is None:
        LOG.error("Source '%s' not found in mapping columns: %s", source, candidate_cols)
        return 1

    # read input CSV
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        input_rows = [r for r in reader]

    # Pass force_tz_offset to build_canonical_rows via function attribute (hacky but avoids changing many signatures)
    build_canonical_rows.force_tz_offset = args.force_tz_offset
    canonical_cols, out_rows = build_canonical_rows(input_rows, mapping_rows, fieldnames, source, source_col_name)

    # build output path
    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = CANON_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        output_path = out_dir / f"{stem}_canonical.csv"

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=canonical_cols)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, "") for k in canonical_cols})

    try:
        rel = output_path.relative_to(Path.cwd())
    except Exception:
        rel = output_path

    print(str(rel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
