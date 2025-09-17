"""Download images and produce a master CSV for any source.

Reads an enhanced CSV (produced by the enhancer), sanitizes HTML in text fields,
downloads article images into `pic/`, fills missing `image_Credit` with the
source name and writes a master CSV. Designed to be CI-friendly for GitHub
Actions: deterministic filenames, short timeouts, and limited retries.

Usage (from repo root):
  py -m etl.pipelines.download_images --input data/canonical/hayom/hayom_20250915_100548_expanded_canonical_enhanced.csv

Optional flags:
  --output  Path to write the master CSV. If omitted a name is generated next to the input.
  --source  Source name to use to fill missing image_Credit (defaults to source column or filename).

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import asyncio

import httpx
from bs4 import BeautifulSoup


# Get the repository root directory relative to this script's location
# This ensures the path works correctly in GitHub Actions and other CI environments
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
)
PIC_DIR = os.path.join(REPO_ROOT, "data", "pics")
DEFAULT_TIMEOUT = 10.0  # seconds
MAX_RETRIES = 2


def get_relative_path_from_repo_root(absolute_path: str) -> str:
    """Get a relative path from the repository root to the given absolute path.

    This ensures consistent path handling in GitHub Actions and other CI environments.
    """
    return os.path.relpath(absolute_path, REPO_ROOT).replace("\\", "/")


def ensure_pic_dir() -> None:
    os.makedirs(PIC_DIR, exist_ok=True)
    # Debug logging for CI troubleshooting (only in debug mode)
    if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
        print(f"DEBUG: PIC_DIR resolved to: {PIC_DIR}")
        print(f"DEBUG: Repository root: {REPO_ROOT}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")


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


def filename_from_id_or_fallback(
    row: Dict[str, str], idx: int, url: str, source: str
) -> Tuple[str, str]:
    """Return (filename_with_ext, basename_no_ext).

    Generate a short, deterministic basename from the article `id` and
    `pubDate` to avoid very long filenames. The format is:

        HHMMSS{first3}{last3}

    where `HHMMSS` comes from the article `pubDate` time (local or with
    timezone info if present), and `{first3}`/`{last3}` are the first 3
    and last 3 characters of the article id. The original file extension
    from the `url` is preserved. If `pubDate` is missing or unparsable
    we fall back to the current UTC time for the HHMMSS portion.
    Article ID is non-nullable - raises ValueError if missing.
    """
    # Get file extension from URL
    parsed_ext = os.path.splitext(url.split("?")[0])[-1].lower()
    if parsed_ext and re.match(r"^\.[a-z0-9]{1,6}$", parsed_ext):
        ext = parsed_ext
    else:
        ext = ".jpg"

    # Article ID is non-nullable - fail if missing
    raw_id = (row.get("article_id") or "").strip()
    if not raw_id:
        raise ValueError(
            f"Article ID is missing or empty for row {idx} in source {source}. Article ID is required for image naming."
        )
    # Use the article id itself as the deterministic basename. This keeps
    # recovery simple: filename = <id><ext>. Return both filename and basename
    # (basename is the id string) so callers can use the basename as an index.
    basename = raw_id
    filename = f"{basename}{ext}"

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
            with httpx.Client(
                timeout=DEFAULT_TIMEOUT,
                headers=headers,
                follow_redirects=True,
            ) as client:
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


async def download_image_async(
    url: str, dest_path: str, client: httpx.AsyncClient
) -> bool:
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


def process_csv(
    input_path: str, output_path: str, source_override: Optional[str] = None
) -> Tuple[List[Dict[str, str]], List[str], Dict[str, int]]:
    ensure_pic_dir()

    stats = {"rows": 0, "images_downloaded": 0, "images_missing": 0}

    # Read CSV using configured encoding. We assume headers are already
    # formatted correctly (no BOM/normalization needed) and that callers
    # use the canonical lower-case column names (e.g., 'id','title','pubDate','image').
    from news_mvp.settings import get_runtime_csv_encoding

    csv_enc = get_runtime_csv_encoding()
    with open(input_path, newline="", encoding=csv_enc) as inf:
        reader = csv.DictReader(inf)
        original_fieldnames = [fn for fn in (reader.fieldnames or [])]
        rows = list(reader)
        if not rows:
            print("input CSV has no rows")
            return [], [], stats

        # determine source name
        if source_override:
            source_name = source_override
        else:
            # look for a 'source' column in provided headers (case-sensitive per assumption)
            source_name = None
            lowered_headers = {f.lower() for f in original_fieldnames}
            if "source" in lowered_headers:
                # assume the 'source' header exists and use the value from the first row
                source_name = rows[0].get("source", "unknown")
            if not source_name:
                base = os.path.basename(input_path)
                source_name = (
                    base.split("_")[0] if "_" in base else os.path.splitext(base)[0]
                )

        # Force output headers to match canonical schema
        from news_mvp.schemas import schema_fieldnames, Stage

        out_fieldnames: List[str] = schema_fieldnames(Stage.ETL_BEFORE_MERGE)

        # ensure per-source pics directory exists (images will be stored in data/pics/{source})
        pic_dir_src = os.path.join(PIC_DIR, source_name)
        os.makedirs(pic_dir_src, exist_ok=True)

        # sanitize text fields and (optionally) download images
        for idx, row in enumerate(rows, start=1):
            stats["rows"] += 1
            # sanitize several common textual columns (normalized to lowercase headers)
            for col in ["title", "description", "category", "creator", "imagecaption"]:
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

                try:
                    filename, basename = filename_from_id_or_fallback(
                        row, idx, img_url, source_name
                    )
                except ValueError as e:
                    print(f"ERROR: {e}")
                    row["image_name"] = ""
                    row["imagenameremote"] = ""
                    stats["images_missing"] += 1
                    continue

                dest = os.path.join(pic_dir_src, filename)

                # Check if image already exists BEFORE setting up download
                if os.path.exists(dest):
                    # Image already exists, use it without downloading
                    # We store only the filename in `image` (no path), so the
                    # image field is recoverable by combining with data/pics/{source}
                    row["image"] = os.path.basename(dest)
                    # internal lowercase key for the downloaded filename index
                    # Only set image_name if a remote original name was found; otherwise keep empty
                    row["imagenameremote"] = imageName_remote
                    row["image_name"] = imageName_remote or ""
                    # Don't increment images_downloaded since we didn't download
                    # Don't set _pending_* fields since no download needed
                    continue

                # Set up for download
                row.setdefault("image", "")
                row.setdefault("image_name", "")
                row["_pending_img_url"] = img_url
                row["_pending_dest"] = dest
                row["_pending_basename"] = basename
                row["imagenameremote"] = imageName_remote
            else:
                row["image_name"] = ""
                row["imagenameremote"] = ""
                stats["images_missing"] += 1

            # fill missing image_Credit with source_name
            if not row.get("image_credit"):
                row["image_credit"] = (
                    source_override or row.get("source") or source_name
                )

        # At this point rows have pending download info in _pending_* if they need download.
        # If sync mode (default) perform downloads now; async mode is handled by caller.
    return rows, out_fieldnames, stats


def perform_sync_downloads(rows: List[Dict[str, str]], stats: Dict[str, int]) -> None:
    for row in rows:
        img_url = row.get("_pending_img_url")
        if not img_url:
            continue
        dest = row.get("_pending_dest")
        if not dest or not isinstance(dest, str):
            row["image_name"] = ""
            stats["images_missing"] += 1
            continue
        # No need to check existence here - already checked in process_csv
        succeeded = download_image(img_url, dest)
        if succeeded:
            # store only the filename in `image`
            row["image"] = os.path.basename(dest)
            # Set image_name only if remote basename is present; otherwise leave empty
            if row.get("imagenameremote"):
                row["image_name"] = str(row.get("imagenameremote"))
            else:
                row["image_name"] = ""
            stats["images_downloaded"] += 1
        else:
            row["image_Name"] = ""
            stats["images_missing"] += 1


async def perform_async_downloads(
    rows: List[Dict[str, str]], stats: Dict[str, int], concurrency: int
) -> None:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "news-etl/1.0 (+https://example.invalid)"},
    ) as client:

        async def worker(row: Dict[str, str]) -> None:
            img_url = row.get("_pending_img_url")
            if not img_url:
                return
            dest = row.get("_pending_dest")
            if not dest or not isinstance(dest, str):
                row["image_Name"] = ""
                stats["images_missing"] += 1
                return
            async with sem:
                ok = await download_image_async(img_url, dest, client)
            if ok:
                # store only the filename in `image`
                row["image"] = os.path.basename(dest)
                if row.get("imagenameremote"):
                    row["image_name"] = row["imagenameremote"]
                else:
                    row["image_name"] = ""
                stats["images_downloaded"] += 1
            else:
                row["image_name"] = ""
                stats["images_missing"] += 1

        await asyncio.gather(*(worker(r) for r in rows))


def write_output_csv(
    output_path: str, rows: List[Dict[str, str]], out_fieldnames: List[str]
) -> None:
    # Write using utf-8-sig so the file is friendly to Windows editors and tools
    from news_mvp.settings import get_runtime_csv_encoding

    csv_enc = get_runtime_csv_encoding()
    with open(output_path, "w", newline="", encoding=csv_enc) as outf:
        writer = csv.DictWriter(outf, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            # Map internal lower-case keys back to preferred camelCase output headers
            out_row: Dict[str, str] = {}
            for header in out_fieldnames:
                h_low = header.lower()
                if h_low == "image_name":
                    out_row[header] = row.get("image_name", "")
                elif h_low == "image_credit":
                    out_row[header] = row.get("image_credit", "")
                elif h_low == "image":
                    out_row[header] = row.get("image", "")
                else:
                    # default mapping: use the lower-case key if present, else empty
                    out_row[header] = row.get(h_low, "")
            writer.writerow(out_row)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Download images and produce a master CSV for a source."
    )
    p.add_argument("--input", required=True, help="Path to enhanced input CSV")
    p.add_argument("--output", required=False, help="Path to output master CSV")
    p.add_argument(
        "--source",
        required=False,
        help="Source name to use for image_credit if missing",
    )
    p.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async concurrent downloads",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Max concurrent downloads when using --async",
    )
    args = p.parse_args(argv)
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
    rows, out_fieldnames, stats = process_csv(
        input_path, output_path, source_override=args.source
    )

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

    print(
        f"rows={stats['rows']}, images_downloaded={stats['images_downloaded']}, images_missing={stats['images_missing']}"
    )
    # Print the output CSV relative path as the last line so callers/CI can capture it
    output_rel = get_relative_path_from_repo_root(output_path)
    print(output_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
