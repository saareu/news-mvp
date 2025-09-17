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
from pathlib import Path
from typing import Dict, List, Tuple, cast
from news_mvp.etl.utils.id_seed import make_news_id
from news_mvp.settings import get_runtime_csv_encoding
from news_mvp.schemas import Stage
from news_mvp.schema_io import write_stage_df
import email.utils
from datetime import datetime, timezone
import pandas as pd

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
    # derive CSV encoding from runtime config (configs/<env>.yaml). Allow
    # overriding the environment via NEWS_MVP_ENV; default to 'dev'. This
    # lets users change csv encoding in YAML rather than editing code.
    csv_enc = get_runtime_csv_encoding()
    with open(mapping_path, "r", encoding=csv_enc, newline="") as f:
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
        raise ValueError(
            f"Malformed concat spec (must have odd number of '+'-separated tokens): {spec!r}"
        )
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
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
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
    force_tz_offset: bool = False,
) -> List[Dict[str, str]]:
    """Build canonical rows using mapping for field selection, then add required schema fields."""

    # Use mapping to determine which fields to extract from source
    spec_for: Dict[str, str] = {}
    for r in mapping_rows:
        canonical_key = r.get("canonical") or ""
        spec_for[canonical_key] = r.get(source_col_name, "")

    out_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(input_rows, start=1):
        out: Dict[str, str] = {}

        # Apply mapping transformations
        for canon_field in spec_for:
            spec = (spec_for.get(canon_field) or "").strip()
            if not spec or spec.lower() == "none":
                out[canon_field] = ""
                continue
            # If it's a concat spec (contains '+') treat specially
            if "+" in spec:
                try:
                    val = combine_concat(spec, row)
                except Exception as e:
                    raise RuntimeError(
                        f"Invalid concat spec for canonical '{canon_field}': {spec!r}: {e}"
                    ) from e
                out[canon_field] = val
            else:
                # simple mapping: take value from source column name
                out[canon_field] = row.get(spec, "")

        # Ensure required schema fields are present (beyond mapping)

        # Source must be filled
        if not out.get("source"):
            out["source"] = source

        # Handle pub_date normalization and fallback
        pub_val_raw = out.get("pub_date") or out.get("pubDate") or ""
        if pub_val_raw:
            normalized = normalize_pubdate(pub_val_raw, force_tz_offset=force_tz_offset)
            out["pub_date"] = normalized or pub_val_raw
        else:
            # Fallback to fetching_time
            ft = (
                out.get("fetching_time")
                or out.get("fetchingTime")
                or out.get("fetchingtime")
                or ""
            ).strip()
            if ft:
                normalized = normalize_pubdate(ft, force_tz_offset=force_tz_offset)
                out["pub_date"] = normalized or ft
            else:
                # Last resort: current time
                out["pub_date"] = datetime.now().isoformat()

        # Generate article_id (deterministic ID)
        title_val = (out.get("title") or "").strip()
        pub_val = (out.get("pub_date") or "").strip()
        src_val = (out.get("source") or "").strip()

        if not title_val or not pub_val or not src_val:
            guid_preview = out.get("guid") or ""
            missing = []
            if not title_val:
                missing.append("title")
            if not pub_val:
                missing.append("pub_date")
            if not src_val:
                missing.append("source")
            raise RuntimeError(
                f"Missing required fields {missing} for input row {idx} (title={title_val!r}, pub_date={pub_val!r}, source={src_val!r}, guid={guid_preview!r})"
            )

        out["article_id"] = make_news_id(title_val, pub_val, src_val)

        # Language detection from title
        if not out.get("language"):
            out["language"] = detect_language_from_title(title_val)

        # Image cleanup
        if out.get("image"):
            out["image"] = strip_url_query(out["image"])

        # Set fetching_time if not present (required field)
        if not out.get("fetching_time"):
            out["fetching_time"] = datetime.now(timezone.utc).isoformat()

        out_rows.append(out)

    return out_rows


def main(argv=None) -> int:

    from news_mvp.etl.config import CANON_DIR

    p = argparse.ArgumentParser(
        description="Map expanded per-source CSV into canonical CSV using etl/schema/mapping.csv. All paths are configurable via etl/config.py and environment/YAML."
    )
    p.add_argument("--input", required=True, help="Path to expanded input CSV")
    p.add_argument(
        "--output",
        help=f"Optional output path; if omitted, writes to <CANON_DIR>/{{source}}/<basename>_canonical.csv (default: {CANON_DIR}/{{source}})",
    )
    # Get the base directory for mapping file
    try:
        from news_mvp.paths import Paths

        base_dir = Paths.root()
    except ImportError:
        base_dir = Path(__file__).resolve().parents[2]

    default_mapping_path = (
        base_dir / "src" / "news_mvp" / "etl" / "schema" / "mapping.csv"
    )

    p.add_argument(
        "--mapping", default=str(default_mapping_path), help="Path to mapping CSV"
    )
    p.add_argument(
        "--force-tz-offset",
        type=int,
        default=None,
        help="Force output timezone offset in hours (e.g., 3 for +03:00, -5 for -05:00)",
    )
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
        LOG.error(
            "Source '%s' not found in mapping columns: %s", source, candidate_cols
        )
        return 1

    # read input CSV using pandas
    csv_enc = get_runtime_csv_encoding()
    df_in = pd.read_csv(input_path, encoding=csv_enc, dtype=str)
    input_rows = cast(List[Dict[str, str]], df_in.fillna("").to_dict(orient="records"))

    # read mapping CSV using pandas
    df_mapping = pd.read_csv(mapping_path, encoding=csv_enc, dtype=str)
    mapping_rows = cast(
        List[Dict[str, str]], df_mapping.fillna("").to_dict(orient="records")
    )
    fieldnames = list(df_mapping.columns)

    # Pass force_tz_offset explicitly to the function to make the API explicit
    out_rows = build_canonical_rows(
        input_rows,
        mapping_rows,
        fieldnames,
        source,
        source_col_name,
        args.force_tz_offset,
    )

    # build output path
    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = CANON_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        output_path = out_dir / f"{stem}_canonical.csv"

    # Convert to DataFrame and use schema-aware writer to ensure master-stage compliance
    df_out = pd.DataFrame(out_rows)
    write_stage_df(df_out, str(output_path), Stage.ETL_BEFORE_MERGE, encoding=csv_enc)

    try:
        rel = output_path.relative_to(Path.cwd())
    except Exception:
        rel = output_path

    print(str(rel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
