"""Download images and produce a master CSV for any source.

Reads an enhanced CSV (produced by the enhancer), sanitizes HTML in text fields,
downloads article images into `pic/`, fills missing `imageCredit` with the
source name and writes a master CSV. Designed to be CI-friendly for GitHub
Actions: deterministic filenames, short timeouts, and limited retries.

Usage (from repo root):
  py -m etl.pipelines.download_images --input data/canonical/hayom/hayom_20250915_100548_expanded_canonical_enhanced.csv

Optional flags:
  --output  Path to write the master CSV. If omitted a name is generated next to the input.
  --source  Source name to use to fill missing imageCredit (defaults to source column or filename).

"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple
import asyncio

import httpx
from bs4 import BeautifulSoup


PIC_DIR = os.path.join(os.getcwd(), "data", "pics")
DEFAULT_TIMEOUT = 10.0  # seconds
MAX_RETRIES = 2


def ensure_pic_dir() -> None:
    os.makedirs(PIC_DIR, exist_ok=True)


def sanitize_html(value: Optional[str]) -> str:
    """Strip HTML tags and collapse whitespace. Returns empty string for None."""
    if not value:
        return ""
    # Use Python's built-in parser to avoid requiring lxml in CI
    soup = BeautifulSoup(value, "html.parser")
    text = soup.get_text(separator=" ")
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deterministic_filename(source: str, idx: int, url: str) -> str:
    """Create a deterministic filename using source, idx and a hash of the url.

    Keeps the original extension when available.
    """
    parsed_ext = os.path.splitext(url.split("?")[0])[-1].lower()
    if parsed_ext and re.match(r"^\.[a-z0-9]{1,6}$", parsed_ext):
        ext = parsed_ext
    else:
        ext = ".jpg"
    # short hash for uniqueness
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    safe_source = re.sub(r"[^0-9A-Za-z_-]", "", source) or "src"
    return f"{safe_source}_{idx:04d}_{h}{ext}"


def filename_from_id_or_fallback(row: Dict[str, str], idx: int, url: str, source: str) -> Tuple[str, str]:
    """Return (filename_with_ext, basename_no_ext).

    Prefer row['id'] as basename. Sanitize it. If missing, fall back to deterministic
    hashed name. Preserve extension when available. If a file with the chosen
    name already exists, append a short hash suffix to avoid accidental overwrite.
    """
    parsed_ext = os.path.splitext(url.split("?")[0])[-1].lower()
    if parsed_ext and re.match(r"^\.[a-z0-9]{1,6}$", parsed_ext):
        ext = parsed_ext
    else:
        ext = ".jpg"

    raw_id = (row.get("id") or "").strip()
    if raw_id:
        # sanitize id for filesystem
        safe_id = re.sub(r"[^0-9A-Za-z_-]", "_", raw_id)
        basename = safe_id
    else:
        # fallback
        basename = f"{re.sub(r'[^0-9A-Za-z_-]', '', source) or 'src'}_{idx:04d}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}"

    filename = f"{basename}{ext}"
    dest = os.path.join(PIC_DIR, filename)
    if os.path.exists(dest):
        # append short url-hash to avoid overwriting unrelated files
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:6]
        filename = f"{basename}_{h}{ext}"
        basename = f"{basename}_{h}"

    return filename, basename


def find_image_url(row: Dict[str, str]) -> Optional[str]:
    """Return the image URL from the `image` column (strict).

    Per the user's direction, we only trust the `image` column as the source of
    the image URL. If it's empty or not a valid URL, return None.
    """
    img = row.get("image")
    if not img:
        return None
    img = img.strip()
    if img.startswith("//"):
        img = "https:" + img
    if img.startswith("http://") or img.startswith("https://"):
        return img
    return None


def download_image(url: str, dest_path: str) -> bool:
    """Synchronous download (fallback)."""
    headers = {"User-Agent": "news-etl/1.0 (+https://example.invalid)"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT, headers=headers, follow_redirects=True) as client:
                r = client.get(url)
                if r.status_code == 200 and r.content:
                    with open(dest_path, "wb") as f:
                        f.write(r.content)
                    return True
                else:
                    print(f"warning: image fetch returned {r.status_code} for {url}")
        except Exception as e:
            print(f"warning: attempt {attempt} failed for {url}: {e}")
    return False


async def download_image_async(url: str, dest_path: str, client: httpx.AsyncClient) -> bool:
    """Asynchronous download with an httpx.AsyncClient instance.

    Returns True on success.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client.get(url)
            if r.status_code == 200 and r.content:
                # write the file synchronously to avoid async filesystem complexity
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                return True
            else:
                print(f"warning: image fetch returned {r.status_code} for {url}")
        except Exception as e:
            print(f"warning: async attempt {attempt} failed for {url}: {e}")
    return False


def process_csv(input_path: str, output_path: str, source_override: Optional[str] = None) -> Tuple[List[Dict[str, str]], List[str], Dict[str, int]]:
    ensure_pic_dir()

    stats = {"rows": 0, "images_downloaded": 0, "images_missing": 0}

    with open(input_path, newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        rows = list(reader)
        if not rows:
            print("input CSV has no rows")
            # return empty structures matching the function's return type
            return [], [], stats

        # determine source name
        if source_override:
            source_name = source_override
        else:
            # look for a 'source' column in the CSV header
            source_name = reader.fieldnames and ("source" if "source" in reader.fieldnames else None)
            if isinstance(source_name, str):
                # if 'source' is a header name we extract first row value
                source_name = rows[0].get("source", "unknown")
            else:
                # fallback to filename-based source
                base = os.path.basename(input_path)
                source_name = base.split("_")[0] if "_" in base else os.path.splitext(base)[0]

        out_fieldnames = list(reader.fieldnames) if reader.fieldnames else list(rows[0].keys())
        # ensure image, imageCredit and imageName columns exist (imageName appended last)
        if "image" not in out_fieldnames:
            out_fieldnames.append("image")
        if "imageCredit" not in out_fieldnames:
            out_fieldnames.append("imageCredit")
        if "imageName" not in out_fieldnames:
            out_fieldnames.append("imageName")

        # sanitize text fields and (optionally) download images
        for idx, row in enumerate(rows, start=1):
            stats["rows"] += 1
            # sanitize several common textual columns
            for col in ["title", "description", "category", "creator", "imageCaption"]:
                if col in row:
                    row[col] = sanitize_html(row.get(col, ""))

            img_url = find_image_url(row)
            # preserve the original remote filename (without ext) in imageNameRemote
            imageName_remote = ""
            if img_url:
                # try to extract original remote basename
                try:
                    remote_basename = os.path.basename(img_url.split("?")[0])
                    imageName_remote = os.path.splitext(remote_basename)[0]
                except Exception:
                    imageName_remote = ""

                filename, basename = filename_from_id_or_fallback(row, idx, img_url, source_name)
                dest = os.path.join(PIC_DIR, filename)
                # for sync path we'll download immediately; for async path the caller will handle
                # set provisional values; actual download may change stats
                row.setdefault("image", "")
                row.setdefault("imageName", "")
                row["_pending_img_url"] = img_url
                row["_pending_dest"] = dest
                row["_pending_basename"] = basename
                row["imageNameRemote"] = imageName_remote
            else:
                row["imageName"] = ""
                row["imageNameRemote"] = ""
                stats["images_missing"] += 1

            # fill missing imageCredit with source_name
            if not row.get("imageCredit"):
                row["imageCredit"] = source_override or row.get("source") or source_name

        # At this point rows have pending download info in _pending_* if they need download.
        # If sync mode (default) perform downloads now; async mode is handled by caller.
        return rows, out_fieldnames, stats


def perform_sync_downloads(rows: List[Dict[str, str]], stats: Dict[str, int]) -> None:
    for row in rows:
        img_url = row.get("_pending_img_url")
        if not img_url:
            continue
        dest = row.get("_pending_dest")
        basename = row.get("_pending_basename")
        if not dest or not isinstance(dest, str):
            row["imageName"] = ""
            stats["images_missing"] += 1
            continue
        # Check if file already exists (avoid redownload)
        if os.path.exists(dest):
            row["image"] = os.path.relpath(dest).replace("\\", "/")
            if row.get("imageNameRemote"):
                row["imageName"] = str(row.get("imageNameRemote"))
            else:
                row["imageName"] = str(basename)
            # Do not increment images_downloaded, as it was not downloaded now
            continue
        succeeded = download_image(img_url, dest)
        if succeeded:
            row["image"] = os.path.relpath(dest).replace("\\", "/")
            if row.get("imageNameRemote"):
                row["imageName"] = str(row.get("imageNameRemote"))
            else:
                row["imageName"] = str(basename)
            stats["images_downloaded"] += 1
        else:
            row["imageName"] = ""
            stats["images_missing"] += 1


async def perform_async_downloads(rows: List[Dict[str, str]], stats: Dict[str, int], concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers={"User-Agent": "news-etl/1.0 (+https://example.invalid)"}) as client:
        async def worker(row: Dict[str, str]) -> None:
            img_url = row.get("_pending_img_url")
            if not img_url:
                return
            dest = row.get("_pending_dest")
            basename = row.get("_pending_basename")
            if not dest or not isinstance(dest, str):
                row["imageName"] = ""
                stats["images_missing"] += 1
                return
            async with sem:
                ok = await download_image_async(img_url, dest, client)
            if ok:
                row["image"] = os.path.relpath(dest).replace("\\", "/")
                if row.get("imageNameRemote"):
                    row["imageName"] = row["imageNameRemote"]
                else:
                    row["imageName"] = str(basename)
                stats["images_downloaded"] += 1
            else:
                row["imageName"] = ""
                stats["images_missing"] += 1

        await asyncio.gather(*(worker(r) for r in rows))


def write_output_csv(output_path: str, rows: List[Dict[str, str]], out_fieldnames: List[str]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {k: row.get(k, "") for k in out_fieldnames}
            writer.writerow(out_row)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download images and produce a master CSV for a source.")
    p.add_argument("--input", required=True, help="Path to enhanced input CSV")
    p.add_argument("--output", required=False, help="Path to output master CSV")
    p.add_argument("--source", required=False, help="Source name to use for imageCredit if missing")
    p.add_argument("--async", dest="async_mode", action="store_true", help="Use async concurrent downloads")
    p.add_argument("--concurrency", type=int, default=6, help="Max concurrent downloads when using --async")
    args = p.parse_args(argv)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"input not found: {input_path}")
        return 2

    if args.output:
        output_path = args.output
    else:
        base = os.path.basename(input_path)
        # timezone-aware UTC now
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = base.split("_")[0] if "_" in base else os.path.splitext(base)[0]
        output_name = f"{prefix}_{now}_master.csv"
        output_path = os.path.join(os.path.dirname(input_path), output_name)

    print(f"reading: {input_path}")
    print(f"writing: {output_path}")
    rows, out_fieldnames, stats = process_csv(input_path, output_path, source_override=args.source)

    # perform downloads
    if args.async_mode:
        import asyncio

        asyncio.run(perform_async_downloads(rows, stats, args.concurrency))
    else:
        perform_sync_downloads(rows, stats)

    # cleanup transient keys
    for r in rows:
        for k in list(r.keys()):
            if k.startswith("_pending_"):
                del r[k]

    # write CSV
    write_output_csv(output_path, rows, out_fieldnames)

    print(f"rows={stats['rows']}, images_downloaded={stats['images_downloaded']}, images_missing={stats['images_missing']}")
    # Print the output CSV relative path as the last line so callers/CI can capture it
    output_rel = os.path.relpath(output_path).replace("\\", "/")
    print(output_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
