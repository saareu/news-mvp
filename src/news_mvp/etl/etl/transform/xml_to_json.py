#!/usr/bin/env python3
"""
xml_to_json.py

A source-agnostic XML -> JSON converter that preserves all field/attribute
names and avoids data loss. Conventions:

- XML attributes become keys prefixed with '@' (e.g. attribute 'id' -> '@id')
- Element text is stored under '#text' if present and non-empty
- Child elements with the same tag are grouped into lists
- Namespaces are preserved in tag names (e.g. '{ns}tag')

This is intentionally conservative: it doesn't attempt to normalize or drop
fields. It is suitable for later transformation steps that rely on full raw
content.

Usage (CLI):
    py -m etl.transform.xml_to_json input.xml output.json
    py -m etl.transform.xml_to_json input.xml  # prints JSON to stdout

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import xml.etree.ElementTree as ET


def element_to_dict(elem: ET.Element) -> Dict[str, Any]:
    """Recursively convert an ElementTree Element into a dictionary.

    Rules:
    - Attributes -> '@' + attr_name
    - Text content -> '#text' if non-empty after stripping
    - Child elements:
        * If tag occurs multiple times, collect into a list.
        * Otherwise keep a single dict value.
    """
    node: Dict[str, Any] = {}

    # Attributes
    for k, v in elem.attrib.items():
        node[f"@{k}"] = v

    # Text
    text = (elem.text or "").strip()
    if text:
        node["#text"] = text

    # Children
    children = list(elem)
    if children:
        child_map: Dict[str, List[Any]] = {}
        for child in children:
            child_repr = element_to_dict(child)
            tag = child.tag
            child_map.setdefault(tag, []).append(child_repr)

        # Flatten singletons
        for tag, items in child_map.items():
            if len(items) == 1:
                node[tag] = items[0]
            else:
                node[tag] = items

    return node


def xml_file_to_json(input_path: Path) -> Dict[str, Any]:
    tree = ET.parse(str(input_path))
    root = tree.getroot()
    return {root.tag: element_to_dict(root)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert XML to JSON preserving all fields")
    p.add_argument("input", help="Input XML file path")
    p.add_argument("output", nargs="?", help="Optional output JSON file path")
    p.add_argument("--indent", type=int, default=2, help="JSON indent spaces (default 2)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    data = xml_file_to_json(inp)
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8-sig") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=args.indent)
        print(str(outp))
    else:
        # Default behavior: write JSON next to the input XML with same base name
        outp = inp.with_suffix(".json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8-sig") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=args.indent)
        print(str(outp))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
