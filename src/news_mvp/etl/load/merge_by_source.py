"""Append non-duplicated rows from a source master CSV into a unified master CSV.

This script accepts two relative paths:
- `--source` : the per-source master CSV (input)
- `--master` : the unified master CSV to append into (output)

Behavior:
- Reads both CSVs (master may not exist yet).
- Finds rows in `source` that are not present in `master` (by `id`).
- Appends those new rows to `master`.
- Sorts the resulting master by `pubDate` descending and writes it back.
- Prints the relative path to the unified master as the final stdout line.

Usage:
  py -m etl.load.merge_by_source --source data/canonical/hayom/hayom_..._master.csv --master data/master/master_news.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from news_mvp.etl.config import MASTER_NEWS_CSV
from news_mvp.settings import (
    get_runtime_csv_encoding,
    get_schema_required,
)
from news_mvp.schemas import Stage


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    csv_enc = get_runtime_csv_encoding()
    with open(path, newline="", encoding=csv_enc) as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    csv_enc = get_runtime_csv_encoding()
    with open(path, "w", newline="", encoding=csv_enc) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def parse_pubdate(val: Optional[str]) -> float:
    if not val:
        return 0.0
    try:
        dt = datetime.fromisoformat(val)
        return dt.timestamp()
    except Exception:
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(val, fmt).timestamp()
            except Exception:
                continue
    return 0.0


def merge_by_source(source_paths: List[str], master_path: Optional[str] = None) -> str:
    if master_path is None:
        master_path = str(MASTER_NEWS_CSV)
    else:
        master_path = str(master_path)
    # Ensure output directory exists
    Path(master_path).parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    fieldnames: List[str] = []
    # Determine canonical field names from schema (strict)
    schema_stage = Stage.ETL_BEFORE_MERGE
    required = get_schema_required(schema_stage)
    article_id_field = required[0]
    pub_date_field = required[3]
    # seen_ids removed (unused); dedup handled by dict below

    def safe_print(msg):
        try:
            print(msg)
        except UnicodeEncodeError:
            print(
                msg.encode("ascii", errors="replace").decode("ascii", errors="replace")
            )

    # Read all source CSVs
    for source_path in source_paths:
        rows = read_csv(source_path)
        safe_print(f"[merge_by_source] source: {source_path} rows={len(rows)}")
        if rows:
            safe_print(f"[merge_by_source] source fieldnames: {list(rows[0].keys())}")
            safe_print(f"[merge_by_source] source first row: {repr(rows[0])}")
        # Validate: at this stage every row MUST include a non-empty canonical article_id
        for i, r in enumerate(rows, start=1):
            id_val = (r.get(article_id_field) or "").strip()
            if not id_val:
                raise RuntimeError(
                    f"Missing required '{article_id_field}' in source {source_path} row {i}: {repr(r)}"
                )
            all_rows.append(r)
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    # Read existing master if present
    master_rows = read_csv(master_path)
    safe_print(f"[merge_by_source] master: {master_path} rows={len(master_rows)}")
    if master_rows:
        safe_print(
            f"[merge_by_source] master fieldnames: {list(master_rows[0].keys())}"
        )
        safe_print(f"[merge_by_source] master first row: {repr(master_rows[0])}")
    # Validate master rows also have a canonical article_id
    for i, r in enumerate(master_rows, start=1):
        id_val = (r.get(article_id_field) or "").strip()
        if not id_val:
            raise RuntimeError(
                f"Missing required '{article_id_field}' in existing master {master_path} row {i}: {repr(r)}"
            )
        all_rows.append(r)
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    # Deduplicate by id, preserving last occurrence
    deduped: Dict[str, Dict[str, str]] = {}
    for r in all_rows:
        key = r.get(article_id_field)
        if key is not None:
            deduped[key] = r
    rows = list(deduped.values())
    safe_print(
        f"[merge_by_source] after merge: unified master rows={len(rows)} (added {len(rows)-len(master_rows)})"
    )
    # Sort by canonical pub_date desc
    rows.sort(key=lambda r: parse_pubdate(r.get(pub_date_field)), reverse=True)
    write_csv(master_path, rows, fieldnames)
    return master_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Append non-duplicated rows from one or more source masters into a unified master CSV"
    )
    p.add_argument(
        "--source",
        nargs="+",
        required=True,
        help="Relative path(s) to source master CSV(s)",
    )
    p.add_argument(
        "--master",
        required=False,
        help="Relative path to unified master CSV to append into (default: data/master/master_news.csv)",
    )
    args = p.parse_args(argv)

    for src in args.source:
        if not os.path.exists(src):
            print(f"source not found: {src}")
            return 2

    out = merge_by_source(args.source, args.master)
    print(os.path.relpath(out).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
