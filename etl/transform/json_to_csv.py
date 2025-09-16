#!/usr/bin/env python3
"""
json_to_csv.py

Convert conservative XML->JSON output to CSV.

Behavior:
- Reads a JSON file (as produced by `xml_to_json.py`). The JSON usually wraps a
  single root element. The converter locates the first list-of-objects inside
  the root and treats that list as the "items" to write as CSV rows.
- Omits the feed-level root itself and writes one CSV row per item.
- Field handling:
  - Scalar fields (str/int/float/bool): written directly
  - List fields: each item coerced to string and concatenated using '|' (pipe)
  - Dict/nested object fields: JSON-serialized as a string to avoid data loss
- Column order: deterministic sorted order of all field names seen across items
- Save location: next to the input JSON file using the same base name and
  a `.csv` suffix. Prints the path of the created CSV file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def find_items(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find the first list of dicts in the given object and return it.

    Traverses one level under the root: if the root maps to a dict, search its
    immediate children for a list-of-dict and return the first match. If the
    root itself is a list-of-dict, return it.
    """
    # If the root is a list of dicts
    if isinstance(obj, list) and obj and all(isinstance(i, dict) for i in obj):
        return obj  # type: ignore[return-value]

    # If root is a dict, check its values
    for v in obj.values():
        if isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            return v

    # Fallback: try to find any list-of-dict nested one level deeper
    for v in obj.values():
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, list) and vv and all(isinstance(i, dict) for i in vv):
                    return vv

    raise ValueError("Could not locate a list of item objects in the JSON input")


def coerce_field(value: Any) -> str:
    """Convert a field to a CSV-safe string per rules:
    - list -> items joined with '|'
    - dict -> JSON string
    - None -> empty string
    - otherwise -> str(value)
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def json_items_to_csv(items: List[Dict[str, Any]], out_path: Path) -> None:
    # Collect all field names
    field_set = set()
    for it in items:
        field_set.update(it.keys())
    fields = sorted(field_set)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for it in items:
            row = [coerce_field(it.get(f)) for f in fields]
            writer.writerow(row)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert JSON (from xml_to_json) to CSV")
    p.add_argument("input", help="Input JSON file")
    p.add_argument("output", nargs="?", help="Optional output CSV file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input JSON not found: {inp}")

    with open(inp, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)

    # If the top-level is a mapping with a single root, drill into it
    if isinstance(data, dict) and len(data) == 1:
        root = next(iter(data.values()))
    else:
        root = data

    items = find_items(root if isinstance(root, dict) else root)

    if args.output:
        outp = Path(args.output)
    else:
        outp = inp.with_suffix(".csv")

    json_items_to_csv(items, outp)
    print(str(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
