"""
ETL Python API
--------------
Import this module to use the ETL pipeline as a library from other Python code or projects.

Example usage:
    from etl.api import run_etl_for_source, merge_masters, download_images_for_csv
    run_etl_for_source(source="ynet", rss_url="...")
    merge_masters(["data/master/master_ynet.csv", ...])
    download_images_for_csv("data/canonical/ynet/ynet_..._canonical_enhanced.csv")
"""
import subprocess
from pathlib import Path
from typing import List, Optional
import sys


def run_etl_for_source(source: str, rss_url: str, force_tz_offset: Optional[int] = None, timeout: int = 600, retries: int = 1) -> int:
    """Run the full ETL pipeline for a given source."""
    args = [sys.executable, "-m", "etl.pipelines.etl_by_source", "--source", source, "--rss", rss_url, "--timeout", str(timeout), "--retries", str(retries)]
    if force_tz_offset is not None:
        args += ["--force-tz-offset", str(force_tz_offset)]
    return subprocess.call(args)


def merge_masters(source_csvs: List[str], output_csv: Optional[str] = None) -> int:
    """Merge multiple master CSVs into a unified master CSV."""
    args = [sys.executable, "-m", "etl.load.merge_by_source", "--source"] + source_csvs
    if output_csv:
        args += ["--master", output_csv]
    return subprocess.call(args)


def download_images_for_csv(input_csv: str, output_csv: Optional[str] = None, source: Optional[str] = None, async_mode: bool = False, concurrency: int = 6) -> int:
    """Download images for a canonical CSV and produce a master CSV."""
    args = [sys.executable, "-m", "etl.pipelines.download_images", "--input", input_csv]
    if output_csv:
        args += ["--output", output_csv]
    if source:
        args += ["--source", source]
    if async_mode:
        args += ["--async", "--concurrency", str(concurrency)]
    return subprocess.call(args)
