#!/usr/bin/env python3
"""
Convert unified master_news.csv to Parquet files per date/source:
    data/parquet/{date}/{source}/master.parquet

Usage:
    python polars_to_parquets.py --input data/master/master_news.csv
"""
import argparse
import sys
from pathlib import Path
import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polars_validation import load_and_validate_master_csv
import polars as pl


def extract_date(val):
    # Accepts string or datetime, returns YYYYMMDD or None
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.strftime("%Y%m%d")
    if isinstance(val, str):
        if len(val) >= 10:
            return val[:10].replace("-", "")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Split master_news.csv to Parquet files per date/source."
    )
    parser.add_argument("--input", required=True, help="Path to master_news.csv")
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors in validation"
    )
    args = parser.parse_args()

    # Load and validate
    df, result = load_and_validate_master_csv(
        input_path=args.input,
        source=None,  # Accept all sources
        strict=args.strict,
        verbose=True,
    )
    if not result.is_valid:
        print("\n❌ Data is not valid, aborting Parquet export.")
        sys.exit(1)

    if "pub_date" not in df.columns or "source" not in df.columns:
        print("❌ master_news.csv must have 'pub_date' and 'source' columns.")
        sys.exit(1)

    # Add a YYYYMMDD column for grouping
    df = df.with_columns(
        pl.col("pub_date").map_elements(extract_date).alias("date_str")
    )

    # Group by date and source, write each group
    for group in df.partition_by(["date_str", "source"]):
        date = group["date_str"][0]
        source = group["source"][0]
        if not date or not source:
            continue
        out_dir = Path("data/parquet") / date / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "master.parquet"
        print(f"[write] {len(group)} rows -> {out_path}")
        group.write_parquet(str(out_path))
    print("✅ Done.")


if __name__ == "__main__":
    main()
