#!/usr/bin/env python3
"""create_csv_to_load_by_source.py

Given a canonical CSV path (relative), write a CSV with rows not present in the per-source master.

Usage:
  python -m etl.load.create_csv_to_load_by_source --input data/canonical/haaretz/haaretz_20250915_100543_expanded_canonical.csv

Behavior:
 - Reads master file from `data/master/master_{source}.csv` where `source` is detected from path (data/canonical/{source}/...)
 - Compares by `id` field (configurable) and writes rows not found to `<basename>_unenhanced.csv` in the same directory as the input.
 - Writes UTF-8-SIG CSV and preserves header order from the input canonical CSV.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Set, Dict

LOG = logging.getLogger("create_csv_to_load_by_source")


def detect_source_from_path(p: Path) -> str:
    parts = p.parts
    try:
        idx = parts.index("canonical")
        return parts[idx + 1]
    except ValueError:
        return p.parent.name


def read_master_ids(master_path: Path, id_field: str = "id") -> Set[str]:
    ids = set()
    if not master_path.exists():
        LOG.info("Master file not found: %s", master_path)
        return ids
    with open(master_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ids.add((r.get(id_field) or "").strip())
    return ids


def create_unenhanced(input_path: Path, master_dir: Path | None = None, id_field: str = "id", output_path: Path | None = None) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    source = detect_source_from_path(input_path)
    if master_dir is None:
        master_dir = Path("data") / "master"
    master_path = Path(master_dir) / f"master_{source}.csv"

    master_ids = read_master_ids(master_path, id_field=id_field)

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        fieldnames = reader.fieldnames or []

    to_write = [r for r in rows if (r.get(id_field) or "").strip() not in master_ids]

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_unenhanced.csv")

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in to_write:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    LOG.info("Wrote unenhanced CSV: %s (rows=%d)", output_path, len(to_write))
    return output_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Create CSV of rows not present in master per-source file")
    p.add_argument("--input", required=True, help="Path to canonical CSV")
    p.add_argument("--master-dir", help="Directory containing per-source master files (default data/master)")
    p.add_argument("--id-field", default="id", help="ID field name to compare (default 'id')")
    p.add_argument("--output", help="Optional output path; default: <input>_unenhanced.csv")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = Path(args.input)
    master_dir = Path(args.master_dir) if args.master_dir else None
    output_path = Path(args.output) if args.output else None

    try:
        out = create_unenhanced(input_path, master_dir=master_dir, id_field=args.id_field, output_path=output_path)
        try:
            rel = out.relative_to(Path.cwd())
        except Exception:
            rel = out
        print(str(rel))
        return 0
    except Exception as e:
        LOG.exception("Failed to create unenhanced CSV: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
