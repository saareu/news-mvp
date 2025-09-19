#!/usr/bin/env python3
"""
Convert validated master CSV to Parquet, organized as data/parquet/{date}/{source}/master.parquet

Usage:
    python polars_to_parquets.py --input data/master/master_ynet.csv --source ynet [--date 20250919]

- Loads and validates the CSV using polars_validation.py logic
- Writes the DataFrame to data/parquet/{date}/{source}/master.parquet
"""
import argparse
import sys
from pathlib import Path
import datetime

# Add scripts and src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polars_validation import load_and_validate_master_csv


def safe_strftime(val, fmt="%Y-%m-%d"):
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.strftime(fmt)
    return str(val)  # or return "" if you want to skip non-dates


def main():
    parser = argparse.ArgumentParser(
        description="Convert validated master CSV to Parquet, organized by date/source."
    )
    parser.add_argument("--input", required=True, help="Path to master CSV file")
    parser.add_argument(
        "--source",
        required=True,
        choices=["ynet", "haaretz", "hayom"],
        help="Source name",
    )
    parser.add_argument(
        "--date",
        required=False,
        help="Date in YYYYMMDD format (default: min pub_date in data)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors in validation"
    )
    args = parser.parse_args()

    # Load and validate
    df, result = load_and_validate_master_csv(
        input_path=args.input,
        source=args.source,
        strict=args.strict,
        verbose=True,
    )
    if not result.is_valid:
        print("\n❌ Data is not valid, aborting Parquet export.")
        sys.exit(1)

    # Determine date for output path
    if args.date:
        out_date = args.date
    else:
        # Try to get min pub_date from DataFrame
        out_date = "unknown"
        if "pub_date" in df.columns:
            min_date = df["pub_date"].min()
            # Handle polars Date/Datetime, pandas Timestamp, datetime.date, str, etc.
            try:

                if isinstance(min_date, str):
                    # Try to parse as YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
                    if len(min_date) >= 10:
                        out_date = min_date[:10].replace("-", "")
                elif hasattr(min_date, "strftime"):
                    out_date = min_date.strftime("%Y%m%d")
                elif isinstance(min_date, (int, float)):
                    # Could be a timestamp (unlikely for pub_date)
                    out_date = str(int(min_date))
            except Exception:
                pass

    # Output directory
    out_dir = Path("data/parquet") / out_date / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "master.parquet"

    print(f"[write] Writing Parquet to: {out_path}")
    df.write_parquet(str(out_path))
    print(f"✅ Done: {out_path}")


if __name__ == "__main__":
    main()
