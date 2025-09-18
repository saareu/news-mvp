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


# Exported for test_enhancer_selectors.py
def extract_selector_list(selector_string: str) -> list[str]:
    """Exported: Parse fallback selector syntax for tests."""
    return parse_fallback_selectors(selector_string)


def try_selector_for_texts(soup: BeautifulSoup, selector: str) -> list[str]:
    """Exported: Try a selector (css: or attribute) and return texts for tests."""
    if selector.lower().startswith("css:"):
        sel = selector[len("css:") :]
        return texts_from_css(soup, sel)
    else:
        els = find_by_attr_or_css(soup, selector)
        return [el.get_text(strip=True) for el in els if hasattr(el, "get_text")]


# Graceful imports for structured logging
try:
    from news_mvp.logging_setup import get_logger

    LOG = get_logger("enhancer_by_source")
except ImportError:
    import logging

    LOG = logging.getLogger("enhancer_by_source")


def load_selectors(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    out: Dict[str, Dict[str, Optional[str]]] = {}
    from news_mvp.settings import get_runtime_csv_encoding

    csv_enc = get_runtime_csv_encoding()
    with open(path, "r", encoding=csv_enc, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sel = row.get("selector")
            if not sel:
                continue
            out[sel] = {}
            for k, v in row.items():
                if k == "selector":
                    continue
                out[sel][k] = v if v and v.lower() != "none" else None
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


def parse_fallback_selectors(spec: str) -> list[str]:
    """Parse fallback selector syntax with [] brackets.

    Example: '[selector1],[selector2]' -> ['selector1', 'selector2']
    Example: 'selector' -> ['selector']
    """
    if not spec:
        return []

    spec = spec.strip()
    if not (spec.startswith("[") and "]" in spec):
        return [spec]

    # Extract content within brackets and split by ],[
    selectors = []
    current = ""
    in_bracket = False
    i = 0

    while i < len(spec):
        char = spec[i]
        if char == "[":
            in_bracket = True
            current = ""
        elif char == "]":
            if in_bracket and current.strip():
                selectors.append(current.strip())
            in_bracket = False
            current = ""
        elif char == "," and not in_bracket:
            # Skip comma outside brackets
            pass
        elif in_bracket:
            current += char
        i += 1

    return selectors if selectors else [spec]


def parse_dsl(spec: str) -> Dict[str, str]:
    """Parse semicolon-separated key=value pairs into a dict.

    Example: 'aria-label=Breadcrumb;list=ul;item=a;include_parent=FALSE'
    Now also supports: 'h3 class;inst_num=1' to select specific instances
    """
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    out: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            # Handle bare selectors like 'h3 class' -> css='h3 class'
            if p and "css" not in out:
                out["css"] = p
            continue
        k, v = p.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def texts_from_css(
    soup: BeautifulSoup,
    selector: str,
    include_parent: bool = False,
    inst_num: Optional[int] = None,
) -> list[str]:
    try:
        nodes = soup.select(selector)
    except Exception:
        return []

    # If inst_num is specified, select only that instance (1-based)
    if inst_num is not None and inst_num > 0:
        if len(nodes) >= inst_num:
            nodes = [nodes[inst_num - 1]]
        else:
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
            anchors = n.find_all("a")
            if anchors:
                for a in anchors:
                    at = a.get_text(strip=True)
                    if at:
                        texts.append(at)
                if texts:
                    continue
        t = n.get_text(separator="|", strip=True)
        if t:
            texts.append(t)
    return texts


def texts_from_attr_list_element(
    el: Any, list_tag: Optional[str], item_sel: Optional[str], include_parent: bool
) -> list[str]:
    """Given a labelled element `el`, extract texts from lists under it.

    list_tag may be 'ul' or 'ol' or None to try both. item_sel is a CSS selector applied to each list item.
    """
    list_tags = [list_tag] if list_tag else ["ul", "ol"]
    results: list[str] = []
    for lt in list_tags:
        lists = el.find_all(lt)
        for L in lists:
            items = L.find_all("li") or L.find_all("a")
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
        pt = el.get_text(separator="|", strip=True)
        if pt:
            results.insert(0, pt)
    return results


def split_spec_and_range(
    spec: str,
) -> tuple[str, Optional[int], Optional[int], Optional[str]]:
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


def collect_children_text_from_range(
    el, start_index: int = 1, end_index: Optional[int] = None
) -> Optional[str]:
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
        immediate = [
            gc.get_text(strip=True)
            for gc in c.find_all(recursive=False)
            if isinstance(gc, Tag) and gc.get_text(strip=True)
        ]
        if immediate:
            texts.extend(immediate)
        else:
            t = c.get_text(strip=True)
            if t:
                texts.append(t)
    if not texts:
        return None
    return "|".join(texts)


def scrape_article(
    url: str, selectors_for_source: Dict[str, Optional[str]], timeout: float = 10.0
):
    # Return dict with possible keys: category, creator, description_insert, description, image_caption, image_credit
    result: Dict[str, Optional[str]] = {
        "category": None,
        "creator": None,
        "description_insert": None,
        "description": None,
        "image_caption": None,
        "image_credit": None,
    }
    try:
        with httpx.Client(
            timeout=timeout, headers={"User-Agent": "news-etl-enhancer/1.0"}
        ) as client:
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

    # author selector might be under 'author' key in selectors.csv
    # New simplified syntax: selector[@attr] with fallback support [selector1],[selector2]
    author_spec_raw = selectors_for_source.get("author")
    if author_spec_raw:
        # Parse fallback selectors
        fallback_selectors = parse_fallback_selectors(author_spec_raw)
        author_result = None
        used_fallback = False

        for idx, selector_spec in enumerate(fallback_selectors):
            if idx > 0:
                used_fallback = True

            # support css: prefixed selector or DSL (semicolon-separated) for aria-list extraction
            if selector_spec.lower().startswith("css:") or ";" in selector_spec:
                # DSL or explicit css
                if selector_spec.lower().startswith("css:"):
                    sel = selector_spec[len("css:") :]
                    texts = texts_from_css(soup, sel, include_parent=False)
                else:
                    spec = parse_dsl(selector_spec)
                    if "css" in spec:
                        sel = spec["css"]
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        inst_num = None
                        if "inst_num" in spec:
                            try:
                                inst_num = int(spec["inst_num"])
                            except ValueError:
                                pass
                        texts = texts_from_css(
                            soup, sel, include_parent=include_parent, inst_num=inst_num
                        )
                    else:
                        # look for aria-* or aria/id style labelled element
                        # support aria and aria-* keys
                        aria_val = (
                            spec.get("aria")
                            or spec.get("id")
                            or spec.get("aria-labelledby")
                        )
                        aria_attr = None
                        aria_attr_val = None
                        if not aria_val:
                            for k, v in spec.items():
                                if k.startswith("aria-"):
                                    aria_attr = k
                                    aria_attr_val = v
                                    break
                        list_tag = spec.get("list")
                        item_sel = spec.get("item")
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        if aria_val:
                            els = soup.find_all(
                                attrs={"aria-labelledby": aria_val}
                            ) or soup.find_all(id=aria_val)
                            texts = []
                            for el in els:
                                if isinstance(el, Tag):
                                    texts.extend(
                                        texts_from_attr_list_element(
                                            el, list_tag, item_sel, include_parent
                                        )
                                    )
                        elif aria_attr and aria_attr_val:
                            el = soup.find(attrs={aria_attr: aria_attr_val})
                            if isinstance(el, Tag):
                                texts = texts_from_attr_list_element(
                                    el, list_tag, item_sel, include_parent
                                )
                            else:
                                texts = []
                        else:
                            texts = []
                if texts:
                    author_result = "|".join(texts)
                    break
            else:
                sel, attr = parse_selector_and_attr(selector_spec)
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
                            a = el.find("a")
                            if a and a.get_text(strip=True):
                                texts.append(a.get_text(strip=True))
                                continue
                        t = el.get_text(strip=True)
                        if t:
                            texts.append(t)
                    if texts:
                        author_result = "|".join(texts)
                        break

        # Apply fallback normalization for authors
        if author_result:
            if used_fallback and ", " in author_result:
                author_result = author_result.replace(", ", "|")
            # Clean up trailing separators
            author_result = author_result.rstrip("|").rstrip(",").strip()
            if author_result:  # Only set if not empty after cleanup
                result["creator"] = author_result

    # category selector: support css: prefix or semaphore DSL for aria/list extraction, with fallback support
    cat_spec_raw = selectors_for_source.get("category")
    if cat_spec_raw:
        # Parse fallback selectors
        fallback_selectors = parse_fallback_selectors(cat_spec_raw)

        for selector_spec in fallback_selectors:
            if selector_spec.lower().startswith("css:") or ";" in selector_spec:
                if selector_spec.lower().startswith("css:"):
                    sel = selector_spec[len("css:") :]
                    texts = texts_from_css(soup, sel, include_parent=False)
                    if texts:
                        # join multiple category parts with '|' to preserve ordered hierarchy
                        result["category"] = "|".join(texts)
                        break
                else:
                    spec = parse_dsl(selector_spec)
                    if "css" in spec:
                        sel = spec["css"]
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        inst_num = None
                        if "inst_num" in spec:
                            try:
                                inst_num = int(spec["inst_num"])
                            except ValueError:
                                pass
                        texts = texts_from_css(
                            soup, sel, include_parent=include_parent, inst_num=inst_num
                        )
                        if texts:
                            result["category"] = "|".join(texts)
                            break
                    else:
                        aria_val = (
                            spec.get("aria")
                            or spec.get("id")
                            or spec.get("aria-labelledby")
                        )
                        aria_attr = None
                        aria_attr_val = None
                        if not aria_val:
                            for k, v in spec.items():
                                if k.startswith("aria-"):
                                    aria_attr = k
                                    aria_attr_val = v
                                    break
                        list_tag = spec.get("list")
                        item_sel = spec.get("item")
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        texts = []
                        if aria_val:
                            els = soup.find_all(
                                attrs={"aria-labelledby": aria_val}
                            ) or soup.find_all(id=aria_val)
                            for el in els:
                                if isinstance(el, Tag):
                                    texts.extend(
                                        texts_from_attr_list_element(
                                            el, list_tag, item_sel, include_parent
                                        )
                                    )
                        elif aria_attr and aria_attr_val:
                            el = soup.find(attrs={aria_attr: aria_attr_val})
                            if isinstance(el, Tag):
                                texts.extend(
                                    texts_from_attr_list_element(
                                        el, list_tag, item_sel, include_parent
                                    )
                                )
                        if texts:
                            result["category"] = "|".join(texts)
                            break
            else:
                sel, attr = parse_selector_and_attr(selector_spec)
                els = find_by_attr_or_css(soup, sel)
                if els:
                    # collect all meaningful texts or attributes and join with '|'
                    collected: list[str] = []
                    for el in els:
                        if attr and isinstance(el, Tag):
                            if el.has_attr(attr):
                                v = el.get(attr)
                                if v:
                                    content = str(v).strip()
                                    # Handle sub-channel-name formatting: convert "/" to "|"
                                    if (
                                        attr == "content"
                                        and el.get("property") == "sub-channel-name"
                                    ):
                                        content = content.replace("/", "|")
                                    collected.append(content)
                                    continue
                            # look for descendant with the attribute
                            found = None
                            found_element = None
                            for d in el.find_all(recursive=True):
                                if isinstance(d, Tag) and d.has_attr(attr):
                                    found = d.get(attr)
                                    found_element = d
                                    break
                            if found:
                                content = str(found).strip()
                                # Handle sub-channel-name formatting for descendants too
                                if (
                                    attr == "content"
                                    and isinstance(found_element, Tag)
                                    and found_element.get("property")
                                    == "sub-channel-name"
                                ):
                                    content = content.replace("/", "|")
                                collected.append(content)
                                continue
                        t = el.get_text(strip=True)
                        if t:
                            collected.append(t)
                    if collected:
                        result["category"] = "|".join(collected)
                        break

    # description: use description selector for both insert (contains {}) and full replacement (empty)
    desc_spec = selectors_for_source.get("description")
    if desc_spec:
        # Parse fallback selectors for description
        fallback_selectors = parse_fallback_selectors(desc_spec)

        for selector_spec in fallback_selectors:
            if selector_spec.lower().startswith("css:") or ";" in selector_spec:
                if selector_spec.lower().startswith("css:"):
                    sel = selector_spec[len("css:") :]
                    texts = texts_from_css(soup, sel, include_parent=False)
                    if texts:
                        result["description_insert"] = texts[0]
                        result["description"] = texts[0]
                        break
                else:
                    spec = parse_dsl(selector_spec)
                    if "css" in spec:
                        sel = spec["css"]
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        texts = texts_from_css(soup, sel, include_parent=include_parent)
                        if texts:
                            result["description_insert"] = texts[0]
                            result["description"] = texts[0]
                            break
                    else:
                        aria_val = (
                            spec.get("aria")
                            or spec.get("id")
                            or spec.get("aria-labelledby")
                        )
                        aria_attr = None
                        aria_attr_val = None
                        if not aria_val:
                            for k, v in spec.items():
                                if k.startswith("aria-"):
                                    aria_attr = k
                                    aria_attr_val = v
                                    break
                        list_tag = spec.get("list")
                        item_sel = spec.get("item")
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        texts = []
                        if aria_val:
                            els = soup.find_all(
                                attrs={"aria-labelledby": aria_val}
                            ) or soup.find_all(id=aria_val)
                            for el in els:
                                if isinstance(el, Tag):
                                    texts.extend(
                                        texts_from_attr_list_element(
                                            el, list_tag, item_sel, include_parent
                                        )
                                    )
                        elif aria_attr and aria_attr_val:
                            el = soup.find(attrs={aria_attr: aria_attr_val})
                            if isinstance(el, Tag):
                                texts.extend(
                                    texts_from_attr_list_element(
                                        el, list_tag, item_sel, include_parent
                                    )
                                )
                        if texts:
                            result["description_insert"] = texts[0]
                            result["description"] = texts[0]
                            break
            else:
                sel, attr = parse_selector_and_attr(selector_spec)
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
                            txt = (
                                str(found).strip()
                                if found
                                else first.get_text(strip=True)
                            )
                    else:
                        # prefer first child anchor/text, otherwise element text
                        if isinstance(first, Tag):
                            a = first.find("a")
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
                        result["description"] = txt
                        break

    # image_credit selector: extract from meta vr:image_credit
    image_credit_spec = selectors_for_source.get("image_credit")
    if image_credit_spec:
        # Parse fallback selectors
        fallback_selectors = parse_fallback_selectors(image_credit_spec)

        for selector_spec in fallback_selectors:
            if selector_spec.lower().startswith("css:") or ";" in selector_spec:
                if selector_spec.lower().startswith("css:"):
                    sel = selector_spec[len("css:") :]
                    texts = texts_from_css(soup, sel, include_parent=False)
                    if texts:
                        result["image_credit"] = texts[0]
                        break
                else:
                    spec = parse_dsl(selector_spec)
                    if "css" in spec:
                        sel = spec["css"]
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        texts = texts_from_css(soup, sel, include_parent=include_parent)
                        if texts:
                            result["image_credit"] = texts[0]
                            break
                    else:
                        aria_val = (
                            spec.get("aria")
                            or spec.get("id")
                            or spec.get("aria-labelledby")
                        )
                        aria_attr = None
                        aria_attr_val = None
                        if not aria_val:
                            for k, v in spec.items():
                                if k.startswith("aria-"):
                                    aria_attr = k
                                    aria_attr_val = v
                                    break
                        list_tag = spec.get("list")
                        item_sel = spec.get("item")
                        include_parent = (
                            spec.get("include_parent", "FALSE").upper() == "TRUE"
                        )
                        texts = []
                        if aria_val:
                            els = soup.find_all(
                                attrs={"aria-labelledby": aria_val}
                            ) or soup.find_all(id=aria_val)
                            for el in els:
                                if isinstance(el, Tag):
                                    texts.extend(
                                        texts_from_attr_list_element(
                                            el, list_tag, item_sel, include_parent
                                        )
                                    )
                        elif aria_attr and aria_attr_val:
                            el = soup.find(attrs={aria_attr: aria_attr_val})
                            if isinstance(el, Tag):
                                texts.extend(
                                    texts_from_attr_list_element(
                                        el, list_tag, item_sel, include_parent
                                    )
                                )
                        if texts:
                            result["image_credit"] = texts[0]
                            break
            else:
                sel, attr = parse_selector_and_attr(selector_spec)
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
                            txt = (
                                str(found).strip()
                                if found
                                else first.get_text(strip=True)
                            )
                    else:
                        # prefer first child anchor/text, otherwise element text
                        if isinstance(first, Tag):
                            a = first.find("a")
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
                        result["image_credit"] = txt
                        break

    return result


def scrape_article_with_image_caption(
    url: str,
    selectors_for_source: Dict[str, Optional[str]],
    image_url: Optional[str] = None,
    timeout: float = 10.0,
):
    """Extended scraping function that also extracts image captions."""
    # First do the regular scraping
    result = scrape_article(url, selectors_for_source, timeout)

    # If we have an image URL but no image_caption, try to extract it
    if image_url and not result.get("image_caption"):
        try:
            with httpx.Client(
                timeout=timeout, headers={"User-Agent": "news-etl-enhancer/1.0"}
            ) as client:
                r = client.get(url)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    caption = extract_image_caption_if_missing(soup, image_url)
                    if caption:
                        result["image_caption"] = caption
        except Exception:
            pass  # Silently fail image caption extraction

    return result


def extract_image_caption_if_missing(
    soup: BeautifulSoup, image_url: str
) -> Optional[str]:
    """Extract image caption for articles without image_caption.

    Logic:
    1. Find the img tag that matches the image field URL
    2. Extract caption from title attribute, parent figcaption, or nearby caption elements
    """
    if not image_url:
        return None

    try:
        # Extract key parts of the expected URL for matching
        expected_parts = image_url.split("/")
        expected_id = None
        for part in expected_parts:
            if len(part) > 20 and "-" in part:  # Likely the image ID
                expected_id = part.split("/")[0]  # Remove any path after the ID
                break

        # Strategy 1: If we have expected image URL, find that specific image
        if expected_id:
            img_tags = soup.find_all("img")
            for img in img_tags:
                if isinstance(img, Tag):
                    src = img.get("src", "") or ""
                    if isinstance(src, str) and expected_id in src:
                        caption = extract_caption_from_img_tag(img)
                        if caption:
                            return caption

        # Strategy 2: Look for figure elements with images (most reliable)
        figures = soup.find_all("figure")
        for fig in figures:
            if isinstance(fig, Tag):
                img = fig.find("img")
                if img and isinstance(img, Tag) and is_content_image(img):
                    caption = extract_caption_from_img_tag(img)
                    if caption and not is_subscription_text(caption):
                        return caption

        # Strategy 3: Look for large images in the main content area
        content_selectors = [
            "article",
            '[class*="content"]',
            '[class*="article"]',
            '[class*="story"]',
            "main",
        ]

        for sel in content_selectors:
            try:
                content_areas = soup.select(sel)
                for content in content_areas:
                    if isinstance(content, Tag):
                        imgs = content.find_all("img")
                        for img in imgs:
                            if isinstance(img, Tag) and is_content_image(img):
                                src = img.get("src", "") or ""
                                if (
                                    isinstance(src, str) and len(src) > 30
                                ):  # Reasonable URL length
                                    caption = extract_caption_from_img_tag(img)
                                    if caption and not is_subscription_text(caption):
                                        return caption
            except Exception:
                continue

        return None

    except Exception:
        return None


def is_content_image(img_tag) -> bool:
    """Check if an image looks like main content (not ads, icons, etc.)."""
    if not isinstance(img_tag, Tag):
        return False

    src = img_tag.get("src", "") or ""
    if not isinstance(src, str):
        return False

    # Skip obviously non-content images
    skip_patterns = [
        "logo",
        "icon",
        "social",
        "button",
        "avatar",
        "ad",
        "banner",
        "header",
    ]
    if any(pattern in src.lower() for pattern in skip_patterns):
        return False

    # Look for reasonable image dimensions
    width = img_tag.get("width", "") or ""
    height = img_tag.get("height", "") or ""

    # If dimensions are specified and too small, skip
    if width and isinstance(width, str) and width.isdigit() and int(width) < 100:
        return False
    if height and isinstance(height, str) and height.isdigit() and int(height) < 100:
        return False

    # Check if it's in a figure (good sign)
    parent = img_tag.parent
    levels = 0
    while parent and levels < 3:
        if isinstance(parent, Tag) and parent.name == "figure":
            return True
        parent = parent.parent
        levels += 1

    # If it has alt or title text, it's likely content
    alt = img_tag.get("alt", "") or ""
    title = img_tag.get("title", "") or ""
    if (isinstance(alt, str) and len(alt) > 10) or (
        isinstance(title, str) and len(title) > 10
    ):
        return True

    return True  # Default to true for unknown cases


def is_subscription_text(text: str) -> bool:
    """Check if text is subscription/advertising content."""
    if not text:
        return False

    subscription_patterns = [
        "הדפסת כתבה",
        "מנויים בלבד",
        "לרכישת מינוי",
        "ללא פרסומות",
        "subscription",
        "subscribe",
    ]

    return any(pattern in text for pattern in subscription_patterns)


def extract_caption_from_img_tag(img_tag) -> Optional[str]:
    """Extract the best available caption from an image tag."""
    if not isinstance(img_tag, Tag):
        return None

    captions = []

    # Check title attribute (often has full caption in Haaretz)
    title = img_tag.get("title", "") or ""
    if isinstance(title, str) and title.strip() and not is_subscription_text(title):
        captions.append(("title", title.strip()))

    # Check alt attribute
    alt = img_tag.get("alt", "") or ""
    if isinstance(alt, str) and alt.strip() and not is_subscription_text(alt):
        captions.append(("alt", alt.strip()))

    # Look for figcaption in parent elements
    parent = img_tag.parent
    levels = 0
    while parent and levels < 4:
        if isinstance(parent, Tag):
            figcaption = parent.find("figcaption")
            if figcaption:
                caption_text = figcaption.get_text(strip=True)
                if caption_text and not is_subscription_text(caption_text):
                    captions.append(("figcaption", caption_text))
                break
        parent = parent.parent
        levels += 1

    if captions:
        # Priority: title > figcaption > alt
        priority = {"title": 1, "figcaption": 2, "alt": 3}
        captions.sort(key=lambda x: priority.get(x[0], 4))
        return captions[0][1]

    return None


def enhance_file(
    input_path: Path,
    selectors_path: Path,
    max_rows: Optional[int] = None,
    delay: float = 0.1,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    load_selectors(
        selectors_path
    )  # loaded for potential side-effects; mapping built below

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
    from news_mvp.settings import get_runtime_csv_encoding

    csv_enc = get_runtime_csv_encoding()
    with open(selectors_path, "r", encoding=csv_enc, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            key = row.get("selector")
            if not key:
                continue
            val = row.get(source)
            selectors_for_source[key] = val if val and val.lower() != "none" else None

    out_rows = []
    from news_mvp.settings import get_runtime_csv_encoding

    csv_enc = get_runtime_csv_encoding()
    with open(input_path, "r", encoding=csv_enc, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        fieldnames = reader.fieldnames or []

    # Determine canonical fieldnames from schema/settings to avoid hardcoded keys
    from news_mvp.settings import (
        get_author_fieldname,
        get_category_fieldname,
        get_description_fieldname,
        get_schema_required,
    )

    # Use the ETL pre-merge schema stage for canonical fieldnames
    from news_mvp.schemas import Stage

    schema_stage = Stage.ETL_BEFORE_MERGE
    creator_field = get_author_fieldname(schema_stage)
    category_field = get_category_fieldname(schema_stage)
    description_field = get_description_fieldname(schema_stage)
    required_fields = get_schema_required(schema_stage)
    # Per project convention required_fields[1] is the GUID field
    guid_field = required_fields[1]

    for i, row in enumerate(rows, start=1):
        if max_rows and i > max_rows:
            break
        # Only attempt scraping if creator, category, image_caption, image_credit missing, or description contains '{}'
        need_creator = not (row.get(creator_field) or "").strip()
        need_category = not (row.get(category_field) or "").strip()
        need_image_caption = not (row.get("image_caption") or "").strip()
        need_image_credit = not (row.get("image_credit") or "").strip()
        desc = row.get(description_field) or ""
        need_desc_insert = "{}" in desc
        need_description = not desc.strip()  # Check if description is empty

        if not (
            need_creator
            or need_category
            or need_image_caption
            or need_image_credit
            or need_desc_insert
            or need_description
        ):
            out_rows.append(row)
            continue

        # GUID is taken from the schema required fields (index 1)
        guid = row.get(guid_field) or ""
        if not guid:
            LOG.debug("No guid for row %s; skipping", i)
            out_rows.append(row)
            continue

        # Get image URL if we need to extract image caption
        image_url = row.get("image") or "" if need_image_caption else None

        # Use extended scraping function that can also extract image captions
        scraped = scrape_article_with_image_caption(
            guid, selectors_for_source, image_url
        )

        # Apply scraped values into canonical columns
        if need_creator and scraped.get("creator"):
            row[creator_field] = scraped["creator"]
        if need_category and scraped.get("category"):
            row[category_field] = scraped["category"]
        if need_image_caption and scraped.get("image_caption"):
            row["image_caption"] = scraped["image_caption"]
        if need_image_credit and scraped.get("image_credit"):
            row["image_credit"] = scraped["image_credit"]
        if need_desc_insert:
            ins = scraped.get("description_insert")
            if ins:
                row[description_field] = desc.replace("{}", ins, 1)
            else:
                # Remove {} placeholder if no description was found
                row[description_field] = desc.replace("{}", "", 1)
        if need_description and scraped.get("description"):
            row[description_field] = scraped["description"]

        out_rows.append(row)
        # polite delay
        time.sleep(delay)

    # Output path: replace _unenhanced.csv with _enhanced.csv
    out_path = input_path.with_name(
        input_path.name.replace("_unenhanced.csv", "_enhanced.csv")
    )
    from news_mvp.settings import get_runtime_csv_encoding

    csv_enc = get_runtime_csv_encoding()
    with open(out_path, "w", encoding=csv_enc, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    LOG.info("Wrote enhanced CSV: %s (rows=%d)", out_path, len(out_rows))
    return out_path


# A function that runs the scraping for a specific column, that is partially filled
# and fills the missing values from the start/end of the values in the column.
# It gets true or false whether to fill from the start or the end.


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Enhance unenhanced CSV by scraping missing fields"
    )
    p.add_argument("--input", required=True, help="Path to unenhanced CSV")
    # Get the base directory for selectors file
    try:
        from news_mvp.paths import Paths

        base_dir = Paths.root()
    except ImportError:
        base_dir = Path(__file__).resolve().parents[3]

    default_selectors_path = (
        base_dir / "src" / "news_mvp" / "etl" / "schema" / "selectors.csv"
    )

    p.add_argument(
        "--selectors", default=str(default_selectors_path), help="Path to selectors CSV"
    )
    p.add_argument("--max-rows", type=int, help="Limit rows for testing")
    p.add_argument("--delay", type=float, default=0.1, help="Delay between requests")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        out = enhance_file(
            Path(args.input),
            Path(args.selectors),
            max_rows=args.max_rows,
            delay=args.delay,
        )
        try:
            rel = out.relative_to(Path.cwd())
        except Exception:
            rel = out
        print(str(rel))
        return 0
    except Exception as e:
        LOG.exception("Enhancement failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
