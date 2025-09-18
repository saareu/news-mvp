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
 - After mapping, the script computes a deterministic `id`
     present else of title), infers `language` from the title (Hebrew -> 'he',
     otherwise 'en'), and strips query params from `image` URLs (remove '?' and
     everything after).
 - Writes canonical CSV into a configurable directory (default: data/canonical/{source}/{basename}_canonical.csv`) and
     prints the relative path as the final stdout line for subprocess chaining.

All paths are configurable via etl/config.py and environment/YAML.
"""
from __future__ import annotations

import argparse
import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, cast
from news_mvp.etl.utils.id_seed import make_news_id
from news_mvp.settings import (
    get_runtime_csv_encoding,
    get_schema_fieldnames,
    get_schema_required,
    get_image_fieldname,
    load_settings,
    get_tags_fieldname,
    get_author_fieldname,
)
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
    force_tz_offset: int | None = None,
) -> List[Dict[str, str]]:
    """Build canonical rows using mapping for field selection, then add required schema fields."""
    # Use mapping to determine which fields to extract from source
    spec_for: Dict[str, str] = {}
    for r in mapping_rows:
        canonical_key = r.get("canonical") or ""
        spec_for[canonical_key] = r.get(source_col_name, "")

    # Schema-driven ordering and required fields
    schema_field_order = get_schema_fieldnames(Stage.ETL_BEFORE_MERGE)

    # Derive required fields from schema (non-nullable) using settings wrapper
    required_fields = get_schema_required(Stage.ETL_BEFORE_MERGE)

    # Assign variables to positions requested by user using the required-fields order
    # Expectation: required_fields order matches [article_id, guid, title, pub_date, source, language, fetching_time]
    ARTICLE_ID, GUID, TITLE, PUB_DATE, SOURCE, LANGUAGE, FETCHING_TIME = required_fields

    out_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(input_rows, start=1):
        # Initialize an ordered output dict with schema fields
        out: Dict[str, str] = {k: "" for k in schema_field_order}

        # Populate values based on mapping specs in schema order.
        # Skip mapping for fields that we compute ourselves: ARTICLE_ID, LANGUAGE, SOURCE, FETCHING_TIME
        computed_fields = {ARTICLE_ID, LANGUAGE, SOURCE, FETCHING_TIME}
        for canon_field in schema_field_order:
            if canon_field in computed_fields:
                # do not take these from the input mapping
                continue
            spec = (spec_for.get(canon_field) or "").strip()
            if not spec or spec.lower() == "none":
                # leave empty
                continue
            # concat spec
            if "+" in spec:
                try:
                    val = combine_concat(spec, row)
                except Exception as e:
                    raise RuntimeError(
                        f"Invalid concat spec for canonical '{canon_field}': {spec!r}: {e}"
                    ) from e
                out[canon_field] = val
            else:
                out[canon_field] = row.get(spec, "")

        out[SOURCE] = source

        # Pub date normalization: pub_date is mandatory and must come from mapping; do NOT fallback
        pub_val_raw = out.get(PUB_DATE)
        if not pub_val_raw or not pub_val_raw.strip():
            raise RuntimeError(
                f"Missing required field '{PUB_DATE}' for input row {idx}"
            )
        out[PUB_DATE] = normalize_pubdate(pub_val_raw, force_tz_offset=force_tz_offset)

        # Title is required (populated from mapping); compute language from title
        title_val = out[TITLE].strip()
        out[LANGUAGE] = detect_language_from_title(title_val)

        # Fetching time: always set by the pipeline (do not accept input value)
        out[FETCHING_TIME] = datetime.now(timezone.utc).isoformat()

        # Compute deterministic article id from title, pub_date, source
        pub_val = out[PUB_DATE].strip()
        src_val = out[SOURCE].strip()
        guid_val = (out.get(GUID) or "").strip()
        if not title_val or not pub_val or not src_val or not guid_val:
            missing = []
            if not title_val:
                missing.append(TITLE)
            if not pub_val:
                missing.append(PUB_DATE)
            if not src_val:
                missing.append(SOURCE)
            if not guid_val:
                missing.append(GUID)
            raise RuntimeError(
                f"Missing required fields {missing} for input row {idx} ({TITLE}={title_val!r}, {PUB_DATE}={pub_val!r}, {SOURCE}={src_val!r}, {GUID}={guid_val!r})"
            )
        out[ARTICLE_ID] = make_news_id(title_val, pub_val, src_val)

        # Image cleanup: remove query strings from image URL (use schema-derived field name)
        IMAGE = get_image_fieldname(Stage.ETL_BEFORE_MERGE)
        if out.get(IMAGE):
            out[IMAGE] = strip_url_query(out[IMAGE])

        out_rows.append(out)

    return out_rows


def normalize_delim_list_column(
    rows: List[Dict[str, str]], column: str, input_sep: str = ",", out_sep: str = "|"
) -> None:
    """Normalize a list-like column in-place.

    For each row, split the value in `column` by `input_sep`, strip whitespace
    from each token and then join non-empty tokens with `out_sep`.

    This mutates `rows` in-place and is intended for fields like `tags`.
    """
    if not rows:
        return
    for r in rows:
        val = r.get(column) or ""
        if not val:
            r[column] = ""
            continue
        parts = [p.strip() for p in val.split(input_sep)]
        parts = [p for p in parts if p]
        r[column] = out_sep.join(parts)


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
        "--env",
        default=None,
        help="Config environment (overrides NEWS_MVP_ENV). e.g. 'dev' or 'prod'.",
    )
    p.add_argument(
        "--force-tz-offset",
        type=int,
        default=None,
        help="Force output timezone offset in hours (e.g., 3 for +03:00, -5 for -05:00)",
    )
    p.add_argument(
        "--strip-tags",
        action="store_true",
        help="Normalize tags column by splitting on ',' trimming and joining with '|' before writing",
    )
    p.add_argument(
        "--strip-authors",
        action="store_true",
        help="Normalize authors column by splitting on ',' trimming and joining with '|' before writing",
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

    # Load env-level settings to allow per-source defaults. Preference order:
    # 1) CLI --env, 2) env var NEWS_MVP_ENV, 3) default 'dev'
    env = args.env or os.environ.get("NEWS_MVP_ENV", "dev")
    cfg = load_settings(env)
    # Per-source config may be under cfg.etl.sources[source]
    src_cfg = {}
    try:
        src_cfg = (cfg.etl or {}).get("sources", {}) or {}
    except Exception:
        # older SimpleNamespace style: cfg.etl.sources is a dict-like
        try:
            src_cfg = cfg.etl.sources
        except Exception:
            src_cfg = {}
    per_source = (
        src_cfg.get(source, {})
        if isinstance(src_cfg, dict)
        else getattr(src_cfg, source, {})
    )

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

    # Defensive check: ensure the mapping column name (source_col_name) corresponds
    # to the detected source. Accept case differences but error on mismatch.
    if source_col_name.lower() != source.lower():
        LOG.error(
            "Detected source '%s' does not match mapping column '%s'",
            source,
            source_col_name,
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
    # Determine effective force_tz_offset: CLI wins, then per-source settings, else None
    effective_force_tz = args.force_tz_offset
    if effective_force_tz is None:
        try:
            effective_force_tz = (
                per_source.get("force_tz_offset")
                if isinstance(per_source, dict)
                else getattr(per_source, "force_tz_offset", None)
            )
        except Exception:
            effective_force_tz = None

    out_rows = build_canonical_rows(
        input_rows,
        mapping_rows,
        fieldnames,
        source,
        source_col_name,
        effective_force_tz,
    )

    # build output path
    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = CANON_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        output_path = out_dir / f"{stem}_canonical.csv"

    # Optionally normalize tags column (split on comma, strip, join with '|')
    # Determine whether to strip tags: CLI flag overrides per-source setting
    strip_tags_flag = args.strip_tags
    if not strip_tags_flag:
        try:
            strip_tags_flag = (
                per_source.get("strip_tags")
                if isinstance(per_source, dict)
                else getattr(per_source, "strip_tags", False)
            )
        except Exception:
            strip_tags_flag = False

    if strip_tags_flag:
        # Use the settings wrapper and allow it to raise if tags field is missing
        tag_field = get_tags_fieldname(Stage.ETL_BEFORE_MERGE)
        normalize_delim_list_column(out_rows, tag_field, input_sep=",", out_sep="|")

    # Optionally normalize authors column similar to tags.
    # Follow same precedence as tags: CLI flag > per-source config > False
    strip_authors_flag = args.strip_authors
    if not strip_authors_flag:
        try:
            strip_authors_flag = (
                per_source.get("strip_authors")
                if isinstance(per_source, dict)
                else getattr(per_source, "strip_authors", False)
            )
        except Exception:
            strip_authors_flag = False

    if strip_authors_flag:
        # Use the settings wrapper and allow it to raise if author field is missing
        author_field = get_author_fieldname(Stage.ETL_BEFORE_MERGE)
        normalize_delim_list_column(out_rows, author_field, input_sep=",", out_sep="|")

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
