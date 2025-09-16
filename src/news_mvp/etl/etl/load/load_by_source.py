"""Merge an input CSV into a per-source project master CSV.

This script expects a relative input CSV (for example the output of
`etl.pipelines.download_images`). It will merge its rows into a master CSV
located at `data/master/master_{source}.csv` by default (where `{source}` is
provided with `--source` or inferred from the input path). It deduplicates by
`id` and `guid`, and sorts the master by `pubDate` (newest first).

Usage examples:
    py -m etl.load.load_by_source --input data/canonical/hayom/hayom_..._master.csv
    py -m etl.load.load_by_source --input data/canonical/ynet/ynet_..._master.csv --source ynet

Options:
    --source  Source name to use in the master filename (default: inferred)
    --master  Optional override for the master CSV path
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional


DEFAULT_MASTER = os.path.join("data", "master")


def infer_source_from_input(path: str) -> str:
    """Try to infer the source name from the input path.

    Examples:
    - data/canonical/hayom/hayom_2025..._master.csv -> hayom
    - data/canonical/ynet/ynet_... -> ynet
    - hayom_2025..._master.csv -> hayom (filename prefix)
    """
    parts = path.replace("\\", "/").split("/")
    # look for a directory under data/canonical/<source>/
    for i in range(len(parts) - 1):
        if parts[i] == "canonical" and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: use filename prefix up to first underscore
    fname = os.path.basename(path)
    if "_" in fname:
        return fname.split("_")[0]
    # last resort
    return "news"


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def parse_pubdate(val: Optional[str]) -> float:
    if not val:
        return 0.0
    try:
        # try ISO parser
        dt = datetime.fromisoformat(val)
        return dt.timestamp()
    except Exception:
        # fallback: try common formats
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(val, fmt).timestamp()
            except Exception:
                continue
    return 0.0


def merge_master(input_path: str, master_path: str) -> str:
    # read input and master
    input_rows = read_csv(input_path)
    master_rows = read_csv(master_path) if os.path.exists(master_path) else []

    # build index by id and guid from master
    seen_ids = {}
    merged: List[Dict[str, str]] = []
    for r in master_rows:
        key = r.get("id") or r.get("guid")
        if key:
            seen_ids[key] = r
        merged.append(r)

    # merge input rows: if id/guid exists replace the master row, else append
    for r in input_rows:
        key = r.get("id") or r.get("guid")
        if key and key in seen_ids:
            # replace fields in-place (prefer incoming values)
            existing = seen_ids[key]
            existing.update(r)
        else:
            merged.append(r)

    # dedupe by (id) or guid preserving the last occurrence (which should be the incoming)
    # then remove `guid` from rows so masters do not include it.
    deduped = {}
    for r in merged:
        key = r.get("id") or r.get("guid") or os.urandom(8).hex()
        # make a shallow copy so we can safely remove guid later
        deduped[key] = dict(r)

    rows = list(deduped.values())

    # remove guid from rows before writing the master
    for r in rows:
        if "guid" in r:
            r.pop("guid", None)

    # sort by pubDate descending
    rows.sort(key=lambda r: parse_pubdate(r.get("pubDate")), reverse=True)

    # determine fieldnames as union of all keys preserving order from master then input
    # but exclude `guid` so the master file never contains it
    fieldnames = []
    for src in (master_rows + input_rows):
        for k in src.keys():
            if k == "guid":
                continue
            if k not in fieldnames:
                fieldnames.append(k)

    write_csv(master_path, rows, fieldnames)
    return master_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Merge an input CSV into the master news CSV")
    p.add_argument("--input", required=True, help="Relative path to input CSV to merge")
    p.add_argument("--source", required=False, help="Source name to use for master file (eg. hayom)")
    p.add_argument("--master", required=False, help="Master CSV path", default=None)
    args = p.parse_args(argv)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"input not found: {input_path}")
        return 2

    # determine master path: explicit override > --source inferred > default
    if args.master:
        master_path = args.master
    else:
        source = args.source or infer_source_from_input(input_path)
        master_path = os.path.join(DEFAULT_MASTER, f"master_{source}.csv")

    out = merge_master(input_path, master_path)
    # print relative master path as final line for CI
    print(os.path.relpath(out).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
