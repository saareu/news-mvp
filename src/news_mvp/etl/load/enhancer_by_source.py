#!/usr/bin/env python3
"""enhancer_by_source.py

Read an `<source>_unenhanced.csv` (output of create_csv_to_load_by_source) and try to scrape missing
`category` and `creator` (and description placeholders) from the article `guid` URL using selectors
defined in `etl/enhance/selectors.csv`.

The selectors CSV has columns: selector, ynet, haaretz, hayom
Values are CSS selector strings or None. For haaretz, some selectors may be attribute-based e.g.
`data-testid="rich-text"` which will be matched by looking for elements with that attribute.

Behavior:
 - For each row in input CSV, if `category` or `creator` is empty, attempt to fetch `guid` and scrape.
 - For `description`, if the cell contains `{}` the code will attempt to replace the `{}` with the first
   element text found under the description selector.
 - Output file: same folder, filename replacing `_unenhanced.csv` with `_enhanced.csv`.

Note: This script requires `beautifulsoup4` and `httpx` (already in requirements). Keep requests lightweight.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import Any

LOG = logging.getLogger("enhancer_by_source")


def load_selectors(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    out: Dict[str, Dict[str, Optional[str]]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sel = row.get("selector")
            if not sel:
                continue
            out[sel] = {}
            for k, v in row.items():
                if k == "selector":
                    continue
                out[sel][k] = (v if v and v.lower() != "none" else None)
    return out


def find_by_attr_or_css(soup: BeautifulSoup, spec: str):
    """Attempt to interpret spec. If spec contains an '=' assume attribute match (e.g. data-testid="rich-text").
    Otherwise use as CSS selector."""
    spec = spec.strip()
    if not spec:
        return []
    # attribute-like pattern: key="value" or key='value'
    m = re.match(r"^(?P<attr>[a-zA-Z0-9_:-]+)\s*=\s*\"(?P<val>.+)\"$", spec)
    if m:
        attr = m.group("attr")
        val = m.group("val")
        # find elements with attribute==val
        return soup.find_all(attrs={attr: val})
    # fallback to CSS selector
    try:
        return soup.select(spec)
    except Exception:
        return []


def parse_dsl(spec: str) -> Dict[str, str]:
    """Parse semicolon-separated key=value pairs into a dict.

    Example: 'aria-label=Breadcrumb;list=ul;item=a;include_parent=FALSE'
    """
    parts = [p.strip() for p in spec.split(';') if p.strip()]
    out: Dict[str, str] = {}
    for p in parts:
        if '=' not in p:
            continue
        k, v = p.split('=', 1)
        out[k.strip().lower()] = v.strip()
    return out


def texts_from_css(soup: BeautifulSoup, selector: str, include_parent: bool = False) -> list[str]:
    try:
        nodes = soup.select(selector)
    except Exception:
        return []
    texts: list[str] = []
    for n in nodes:
        if not isinstance(n, Tag):
            t = str(n).strip()
            if t:
                texts.append(t)
            continue
        # prefer anchors inside
        if not include_parent:
            anchors = n.find_all('a')
            if anchors:
                for a in anchors:
                    at = a.get_text(strip=True)
                    if at:
                        texts.append(at)
                if texts:
                    continue
        t = n.get_text(separator='|', strip=True)
        if t:
            texts.append(t)
    return texts


def texts_from_attr_list_element(el: Any, list_tag: Optional[str], item_sel: Optional[str], include_parent: bool) -> list[str]:
    """Given a labelled element `el`, extract texts from lists under it.

    list_tag may be 'ul' or 'ol' or None to try both. item_sel is a CSS selector applied to each list item.
    """
    list_tags = [list_tag] if list_tag else ['ul', 'ol']
    results: list[str] = []
    for lt in list_tags:
        lists = el.find_all(lt)
        for L in lists:
            items = L.find_all('li') or L.find_all('a')
            for it in items:
                if item_sel:
                    try:
                        sub = it.select(item_sel)
                    except Exception:
                        sub = []
                    for s in sub:
                        if isinstance(s, Tag):
                            txt = s.get_text(strip=True)
                            if txt:
                                results.append(txt)
                else:
                    txt = it.get_text(strip=True)
                    if txt:
                        results.append(txt)
    if include_parent:
        pt = el.get_text(separator='|', strip=True)
        if pt:
            results.insert(0, pt)
    return results


def split_spec_and_range(spec: str) -> tuple[str, Optional[int], Optional[int], Optional[str]]:
    """Split a selector spec into base selector and optional suffixes.

    Supported suffix forms (after the first semicolon):
      - "+N" (e.g. ";+1"): depth/child-index semantics (kept for backward compat)
      - "attr" (e.g. ";title"): extract child element attribute named 'title'
      - "start;end" (e.g. ";1;3"): collect children from start through end (1-based, inclusive)

    Returns (base_spec, start_index, end_index, child_attr). Any of start_index,
    end_index, child_attr may be None when not present.
    """
    if not spec:
        return spec, None, None, None
    spec = spec.strip()
    # Use ';' as the only separator between base selector and suffixes
    if ";" not in spec:
        return spec, None, None, None
    base, rest = spec.split(";", 1)
    rest = rest.strip()
    # +N depth suffix (treat as start=N, end=None)
    m = re.match(r"^\+(?P<n>\d+)$", rest)
    if m:
        return base.strip(), int(m.group("n")), None, None
    # rest may be attribute name or a pair of numbers like "start;end"
    parts = [p.strip() for p in rest.split(";") if p.strip()]
    # two numeric parts: start and end
    if len(parts) >= 2 and all(re.match(r"^\d+$", p) for p in parts[:2]):
        start = int(parts[0])
        end = int(parts[1])
        return base.strip(), start, end, None
    # single numeric part: start only
    if len(parts) == 1 and re.match(r"^\d+$", parts[0]):
        return base.strip(), int(parts[0]), None, None
    # attribute-name suffix (e.g., 'title')
    if len(parts) == 1 and re.match(r"^[A-Za-z0-9_:-]+$", parts[0]):
        return base.strip(), None, None, parts[0]
    # unknown suffix - ignore
    return base.strip(), None, None, None


def parse_selector_and_attr(spec: str) -> tuple[str, Optional[str]]:
    """Parse a selector cell into (selector, attribute).

    The selector cell format is now simple: a CSS selector or an optional
    attribute suffix separated by '@', e.g. '#foo .bar@title' to request the
    'title' attribute. If no '@' is present the attribute is None and the
    scraper should use element text (or prefer anchor text for author blocks).
    """
    if not spec:
        return spec, None
    spec = spec.strip()
    # split at last '@' so selector may contain '@' in CSS escapes
    if "@" in spec:
        sel, attr = spec.rsplit("@", 1)
        sel = sel.strip()
        attr = attr.strip()
        if not sel:
            return spec, None
        return sel, (attr or None)
    return spec, None


def descend_first_child(el, depth: int):
    """Descend `depth` levels by repeatedly taking the first child tag.

    If any level has no child tag, return None.
    """
    cur = el
    for _ in range(depth):
        # find the first child tag (not a string)
        nxt = cur.find(recursive=False)
        if nxt is None:
            return None
        cur = nxt
    return cur


def collect_children_text_from_range(el, start_index: int = 1, end_index: Optional[int] = None) -> Optional[str]:
    """Collect text from child tags from 1-based start_index through end_index (inclusive).

    If end_index is None, collect through the last child. Returns a pipe-concatenated
    string of non-empty child texts (or immediate-grandchild texts) or None if nothing found.
    """
    if not isinstance(el, Tag):
        return None
    children = [c for c in el.find_all(recursive=False) if isinstance(c, Tag)]
    if not children:
        return None
    # convert 1-based start_index to 0-based
    s_idx = max(0, (start_index or 1) - 1)
    if end_index is not None and end_index > 0:
        e_idx = min(len(children), end_index)
    else:
        e_idx = None
    slice_children = children[s_idx:e_idx]
    texts: list[str] = []
    for c in slice_children:
        if not isinstance(c, Tag):
            continue
        # collect texts from this child's immediate child elements if present
        immediate = [gc.get_text(strip=True) for gc in c.find_all(recursive=False) if isinstance(gc, Tag) and gc.get_text(strip=True)]
        if immediate:
            texts.extend(immediate)
        else:
            t = c.get_text(strip=True)
            if t:
                texts.append(t)
    if not texts:
        return None
    return "|".join(texts)


def scrape_article(url: str, selectors_for_source: Dict[str, Optional[str]], timeout: float = 10.0):
    # Return dict with possible keys: category, creator, description_insert
    result: Dict[str, Optional[str]] = {"category": None, "creator": None, "description_insert": None}
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": "news-etl-enhancer/1.0"}) as client:
            r = client.get(url)
            if r.status_code != 200:
                LOG.debug("Failed to fetch %s: %s", url, r.status_code)
                return result
            html = r.text
    except Exception as e:
        LOG.debug("Exception fetching %s: %s", url, e)
        return result

    # Use Python's built-in parser to avoid requiring external parsers like lxml
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        # As a last resort, try the default parser
        soup = BeautifulSoup(html)

    # creator selector might be under 'author' key in selectors.csv
    # New simplified syntax: selector[@attr]
    author_spec_raw = selectors_for_source.get("author")
    if author_spec_raw:
        # support css: prefixed selector or DSL (semicolon-separated) for aria-list extraction
        if author_spec_raw.lower().startswith('css:') or ';' in author_spec_raw:
            # DSL or explicit css
            if author_spec_raw.lower().startswith('css:'):
                sel = author_spec_raw[len('css:'):]
                texts = texts_from_css(soup, sel, include_parent=False)
            else:
                spec = parse_dsl(author_spec_raw)
                if 'css' in spec:
                    sel = spec['css']
                    include_parent = spec.get('include_parent', 'FALSE').upper() == 'TRUE'
                    texts = texts_from_css(soup, sel, include_parent=include_parent)
                else:
                    # look for aria-* or aria/id style labelled element
                    # support aria and aria-* keys
                    aria_val = spec.get('aria') or spec.get('id') or spec.get('aria-labelledby')
                    aria_attr = None
                    aria_attr_val = None
                    if not aria_val:
                        for k, v in spec.items():
                            if k.startswith('aria-'):
                                aria_attr = k
                                aria_attr_val = v
                                break
                    list_tag = spec.get('list')
                    item_sel = spec.get('item')
                    include_parent = spec.get('include_parent', 'FALSE').upper() == 'TRUE'
                    if aria_val:
                        els = soup.find_all(attrs={"aria-labelledby": aria_val}) or soup.find_all(id=aria_val)
                        texts = []
                        for el in els:
                            if isinstance(el, Tag):
                                texts.extend(texts_from_attr_list_element(el, list_tag, item_sel, include_parent))
                    elif aria_attr and aria_attr_val:
                        el = soup.find(attrs={aria_attr: aria_attr_val})
                        if isinstance(el, Tag):
                            texts = texts_from_attr_list_element(el, list_tag, item_sel, include_parent)
                        else:
                            texts = []
                    else:
                        texts = []
            if texts:
                result["creator"] = ", ".join(texts)
        else:
            sel, attr = parse_selector_and_attr(author_spec_raw)
            els = find_by_attr_or_css(soup, sel)
            if els:
                texts = []
                for el in els:
                    if attr:
                        # extract requested attribute from the element (or its first child anchor)
                        if isinstance(el, Tag):
                            if el.has_attr(attr):
                                v = el.get(attr)
                                if v:
                                    texts.append(str(v).strip())
                                    continue
                            # try first descendant with the attribute
                            found = None
                            for d in el.find_all(recursive=True):
                                if isinstance(d, Tag) and d.has_attr(attr):
                                    found = d.get(attr)
                                    break
                            if found:
                                texts.append(str(found).strip())
                                continue
                    # no attr requested: prefer anchor text, then element text
                    if isinstance(el, Tag):
                        a = el.find('a')
                        if a and a.get_text(strip=True):
                            texts.append(a.get_text(strip=True))
                            continue
                    t = el.get_text(strip=True)
                    if t:
                        texts.append(t)
                if texts:
                    result["creator"] = ", ".join(texts)

    # category selector: support css: prefix or semaphore DSL for aria/list extraction, else fallback
    cat_spec_raw = selectors_for_source.get("category")
    if cat_spec_raw:
        if cat_spec_raw.lower().startswith('css:') or ';' in cat_spec_raw:
            if cat_spec_raw.lower().startswith('css:'):
                sel = cat_spec_raw[len('css:'):]
                texts = texts_from_css(soup, sel, include_parent=False)
                if texts:
                    # join multiple category parts with '|' to preserve ordered hierarchy
                    result["category"] = "|".join(texts)
            else:
                spec = parse_dsl(cat_spec_raw)
                if 'css' in spec:
                    sel = spec['css']
                    include_parent = spec.get('include_parent', 'FALSE').upper() == 'TRUE'
                    texts = texts_from_css(soup, sel, include_parent=include_parent)
                    if texts:
                        result['category'] = "|".join(texts)
                else:
                    aria_val = spec.get('aria') or spec.get('id') or spec.get('aria-labelledby')
                    aria_attr = None
                    aria_attr_val = None
                    if not aria_val:
                        for k, v in spec.items():
                            if k.startswith('aria-'):
                                aria_attr = k
                                aria_attr_val = v
                                break
                    list_tag = spec.get('list')
                    item_sel = spec.get('item')
                    include_parent = spec.get('include_parent', 'FALSE').upper() == 'TRUE'
                    texts = []
                    if aria_val:
                        els = soup.find_all(attrs={"aria-labelledby": aria_val}) or soup.find_all(id=aria_val)
                        for el in els:
                            if isinstance(el, Tag):
                                texts.extend(texts_from_attr_list_element(el, list_tag, item_sel, include_parent))
                    elif aria_attr and aria_attr_val:
                        el = soup.find(attrs={aria_attr: aria_attr_val})
                        if isinstance(el, Tag):
                            texts.extend(texts_from_attr_list_element(el, list_tag, item_sel, include_parent))
                    if texts:
                        result['category'] = "|".join(texts)
        else:
            sel, attr = parse_selector_and_attr(cat_spec_raw)
            els = find_by_attr_or_css(soup, sel)
            if els:
                # collect all meaningful texts or attributes and join with '|'
                collected: list[str] = []
                for el in els:
                    if attr and isinstance(el, Tag):
                        if el.has_attr(attr):
                            v = el.get(attr)
                            if v:
                                collected.append(str(v).strip())
                                continue
                        # look for descendant with the attribute
                        found = None
                        for d in el.find_all(recursive=True):
                            if isinstance(d, Tag) and d.has_attr(attr):
                                found = d.get(attr)
                                break
                        if found:
                            collected.append(str(found).strip())
                            continue
                    t = el.get_text(strip=True)
                    if t:
                        collected.append(t)
                if collected:
                    result["category"] = "|".join(collected)

    # description insert: use description selector and take the first child element's text
    desc_spec = selectors_for_source.get("description")
    if desc_spec:
        if desc_spec.lower().startswith('css:') or ';' in desc_spec:
            if desc_spec.lower().startswith('css:'):
                sel = desc_spec[len('css:'):]
                texts = texts_from_css(soup, sel, include_parent=False)
                if texts:
                    result['description_insert'] = texts[0]
            else:
                spec = parse_dsl(desc_spec)
                if 'css' in spec:
                    sel = spec['css']
                    include_parent = spec.get('include_parent', 'FALSE').upper() == 'TRUE'
                    texts = texts_from_css(soup, sel, include_parent=include_parent)
                    if texts:
                        result['description_insert'] = texts[0]
                else:
                    aria_val = spec.get('aria') or spec.get('id') or spec.get('aria-labelledby')
                    aria_attr = None
                    aria_attr_val = None
                    if not aria_val:
                        for k, v in spec.items():
                            if k.startswith('aria-'):
                                aria_attr = k
                                aria_attr_val = v
                                break
                    list_tag = spec.get('list')
                    item_sel = spec.get('item')
                    include_parent = spec.get('include_parent', 'FALSE').upper() == 'TRUE'
                    texts = []
                    if aria_val:
                        els = soup.find_all(attrs={"aria-labelledby": aria_val}) or soup.find_all(id=aria_val)
                        for el in els:
                            texts.extend(texts_from_attr_list_element(el, list_tag, item_sel, include_parent))
                    elif aria_attr and aria_attr_val:
                        el = soup.find(attrs={aria_attr: aria_attr_val})
                        if el:
                            texts.extend(texts_from_attr_list_element(el, list_tag, item_sel, include_parent))
                    if texts:
                        result['description_insert'] = texts[0]
        else:
            sel, attr = parse_selector_and_attr(desc_spec)
            els = find_by_attr_or_css(soup, sel)
            if els:
                first = els[0]
                if attr and isinstance(first, Tag):
                    if first.has_attr(attr):
                        txt = str(first.get(attr)).strip()
                    else:
                        found = None
                        for d in first.find_all(recursive=True):
                            if isinstance(d, Tag) and d.has_attr(attr):
                                found = d.get(attr)
                                break
                        txt = str(found).strip() if found else first.get_text(strip=True)
                else:
                    # prefer first child anchor/text, otherwise element text
                    if isinstance(first, Tag):
                        a = first.find('a')
                        if a and a.get_text(strip=True):
                            txt = a.get_text(strip=True)
                        else:
                            # first child's text if any
                            child = None
                            for ch in first.find_all(recursive=False):
                                if ch.get_text(strip=True):
                                    child = ch
                                    break
                            if child is None:
                                txt = first.get_text(strip=True)
                            else:
                                txt = child.get_text(strip=True)
                    else:
                        txt = str(first)
                if txt:
                    result["description_insert"] = txt

    return result


def enhance_file(input_path: Path, selectors_path: Path, max_rows: Optional[int] = None, delay: float = 0.1) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    selectors = load_selectors(selectors_path)

    # Determine source from path (data/canonical/{source}/...)
    parts = input_path.parts
    source = None
    try:
        idx = parts.index("canonical")
        source = parts[idx + 1]
    except Exception:
        source = input_path.parent.name

    # Build a map of selectors for this source: key -> spec
    selectors_for_source: Dict[str, Optional[str]] = {}
    # The selectors.csv keys are like 'author','category','description' in its 'selector' column
    # But above load_selectors returns mapping keyed by selector column's value; we need to invert
    # We'll use the CSV directly: open and get row entries where the 'selector' header is the logical name
    with open(selectors_path, 'r', encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            key = row.get('selector')
            if not key:
                continue
            val = row.get(source)
            selectors_for_source[key] = (val if val and val.lower() != 'none' else None)

    out_rows = []
    with open(input_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        fieldnames = reader.fieldnames or []

    for i, row in enumerate(rows, start=1):
        if max_rows and i > max_rows:
            break
        # Only attempt scraping if creator or category missing, or description contains '{}'
        need_creator = not (row.get('creator') or '').strip()
        need_category = not (row.get('category') or '').strip()
        desc = row.get('description') or ''
        need_desc_insert = '{}' in desc
        if not (need_creator or need_category or need_desc_insert):
            out_rows.append(row)
            continue

        guid = row.get('guid') or ''
        if not guid:
            LOG.debug('No guid for row %s; skipping', i)
            out_rows.append(row)
            continue

        scraped = scrape_article(guid, selectors_for_source)
        # Apply scraped values
        if need_creator and scraped.get('creator'):
            row['creator'] = scraped['creator']
        if need_category and scraped.get('category'):
            row['category'] = scraped['category']
        if need_desc_insert:
            ins = scraped.get('description_insert')
            if ins:
                row['description'] = desc.replace('{}', ins, 1)

        out_rows.append(row)
        # polite delay
        time.sleep(delay)

    # Output path: replace _unenhanced.csv with _enhanced.csv
    out_path = input_path.with_name(input_path.name.replace('_unenhanced.csv', '_enhanced.csv'))
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    LOG.info('Wrote enhanced CSV: %s (rows=%d)', out_path, len(out_rows))
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Enhance unenhanced CSV by scraping missing fields')
    p.add_argument('--input', required=True, help='Path to unenhanced CSV')
    p.add_argument('--selectors', default='etl/enhance/selectors.csv', help='Path to selectors CSV')
    p.add_argument('--max-rows', type=int, help='Limit rows for testing')
    p.add_argument('--delay', type=float, default=0.1, help='Delay between requests')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        out = enhance_file(Path(args.input), Path(args.selectors), max_rows=args.max_rows, delay=args.delay)
        try:
            rel = out.relative_to(Path.cwd())
        except Exception:
            rel = out
        print(str(rel))
        return 0
    except Exception as e:
        LOG.exception('Enhancement failed: %s', e)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())