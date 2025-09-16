#!/usr/bin/env python3
"""
News MVP Parquet Storage Manager

This script handles:
1. Converting CSV data to Parquet format
2. Creating per-source Parquet files
3. Creating unified Parquet files
4. Managing partitioning by date and hour

Usage:
    python scripts/generate_parquet.py --source ynet --date 2025-09-17
    python scripts/generate_parquet.py --unified --date 2025-09-17
    python scripts/generate_parquet.py --all-sources --date 2025-09-17
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ParquetStorageManager:
    """Manages Parquet file generation and storage."""

    def __init__(self, base_path: str = "data/parquet"):
        self.base_path = Path(base_path)
        self.sources_path = self.base_path / "sources"
        self.unified_path = self.base_path / "unified"
        self.compression = "snappy"  # snappy, gzip, brotli
        self.row_group_size = 50000

        # Create directories
        self.sources_path.mkdir(parents=True, exist_ok=True)
        self.unified_path.mkdir(parents=True, exist_ok=True)

    def get_parquet_schema(self) -> pa.schema:
        """Define the Parquet schema for articles."""
        return pa.schema(
            [
                ("article_id", pa.string()),
                ("title", pa.string()),
                ("description", pa.string()),
                ("category", pa.string()),
                ("pub_date", pa.timestamp("ns")),
                ("tags", pa.list_(pa.string())),  # Array of strings
                ("creator", pa.string()),
                ("source", pa.string()),
                ("language", pa.string()),
                ("image_path", pa.string()),
                ("image_caption", pa.string()),
                ("image_credit", pa.string()),
                ("image_name", pa.string()),
                ("processed_at", pa.timestamp("ns")),
                ("batch_hour", pa.int64()),
            ]
        )

    def read_csv_with_schema(self, csv_path: Path) -> pd.DataFrame:
        """Read CSV file and convert to proper types."""
        try:
            # Read CSV with pandas for easier type conversion
            df = pd.read_csv(csv_path, encoding="utf-8")

            # Convert pubDate to datetime
            if "pubDate" in df.columns:
                df["pubDate"] = pd.to_datetime(df["pubDate"], utc=True)

            # Convert tags string to list
            if "tags" in df.columns:
                df["tags"] = (
                    df["tags"]
                    .fillna("")
                    .str.split(",")
                    .apply(lambda x: [tag.strip() for tag in x if tag.strip()])
                )

            # Add processing metadata
            df["processed_at"] = datetime.now(timezone.utc)
            df["batch_hour"] = datetime.now().hour

            # Rename columns to match schema
            column_mapping = {"id": "article_id", "pubDate": "pub_date"}
            df = df.rename(columns=column_mapping)

            logger.info(f"Read {len(df)} rows from {csv_path}")
            return df

        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            raise

    def save_parquet(
        self,
        df: pd.DataFrame,
        output_path: Path,
        partition_cols: Optional[List[str]] = None,
    ):
        """Save DataFrame to Parquet format."""
        try:
            # Convert to PyArrow table
            table = pa.Table.from_pandas(df, schema=self.get_parquet_schema())

            # Save with partitioning if specified
            if partition_cols:
                pq.write_to_dataset(
                    table,
                    root_path=str(output_path),
                    partition_cols=partition_cols,
                    compression=self.compression,
                    row_group_size=self.row_group_size,
                )
            else:
                pq.write_table(
                    table,
                    output_path,
                    compression=self.compression,
                    row_group_size=self.row_group_size,
                )

            logger.info(f"Saved Parquet file: {output_path} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Error saving Parquet {output_path}: {e}")
            raise

    def generate_source_parquet(
        self, source: str, date: str, hour: Optional[int] = None
    ):
        """Generate Parquet file for a specific source."""
        # Input CSV path
        csv_path = Path("data/master") / f"master_{source}.csv"
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return

        # Output path
        if hour is None:
            hour = datetime.now().hour

        output_dir = self.sources_path / source / date
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{hour:02d}_articles.parquet"

        # Read and process data
        df = self.read_csv_with_schema(csv_path)

        # Filter by date if needed (assuming pub_date is available)
        if "pub_date" in df.columns:
            df_date = pd.to_datetime(date).date()
            df["pub_date_date"] = df["pub_date"].dt.date
            df = df[df["pub_date_date"] == df_date]
            df = df.drop("pub_date_date", axis=1)

        if len(df) == 0:
            logger.info(f"No data for {source} on {date}")
            return

        # Save Parquet
        self.save_parquet(df, output_path)
        logger.info(
            f"Generated source Parquet: {source}/{date}/{hour:02d}_articles.parquet"
        )

    def generate_unified_parquet(self, date: str, hour: Optional[int] = None):
        """Generate unified Parquet file combining all sources."""
        if hour is None:
            hour = datetime.now().hour

        output_dir = self.unified_path / date
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{hour:02d}_unified.parquet"

        # Collect data from all sources
        all_data = []
        sources = ["ynet", "hayom", "haaretz"]

        for source in sources:
            csv_path = Path("data/master") / f"master_{source}.csv"
            if csv_path.exists():
                try:
                    df = self.read_csv_with_schema(csv_path)
                    df["source"] = source  # Ensure source column
                    all_data.append(df)
                    logger.info(f"Added {len(df)} rows from {source}")
                except Exception as e:
                    logger.warning(f"Error processing {source}: {e}")

        if not all_data:
            logger.warning("No data found for unified Parquet")
            return

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Filter by date if needed
        if "pub_date" in combined_df.columns:
            df_date = pd.to_datetime(date).date()
            combined_df["pub_date_date"] = combined_df["pub_date"].dt.date
            combined_df = combined_df[combined_df["pub_date_date"] == df_date]
            combined_df = combined_df.drop("pub_date_date", axis=1)

        if len(combined_df) == 0:
            logger.info(f"No data for unified on {date}")
            return

        # Save unified Parquet
        self.save_parquet(combined_df, output_path)
        logger.info(
            f"Generated unified Parquet: {date}/{hour:02d}_unified.parquet ({len(combined_df)} rows)"
        )

        # Update latest symlink
        self.update_latest_symlink(date, hour)

    def update_latest_symlink(self, date: str, hour: int):
        """Update the latest_unified.parquet symlink."""
        latest_path = self.unified_path / "latest_unified.parquet"
        target_path = self.unified_path / date / f"{hour:02d}_unified.parquet"

        try:
            # Remove existing symlink if it exists
            if latest_path.exists():
                latest_path.unlink()

            # Create new symlink (on Windows, this creates a copy)
            if target_path.exists():
                import shutil

                shutil.copy2(target_path, latest_path)
                logger.info(
                    f"Updated latest_unified.parquet -> {date}/{hour:02d}_unified.parquet"
                )

        except Exception as e:
            logger.warning(f"Error updating latest symlink: {e}")

    def generate_all_sources(self, date: str, hour: Optional[int] = None):
        """Generate Parquet files for all sources."""
        sources = ["ynet", "hayom", "haaretz"]

        for source in sources:
            try:
                self.generate_source_parquet(source, date, hour)
            except Exception as e:
                logger.error(f"Error generating Parquet for {source}: {e}")

        # Generate unified
        try:
            self.generate_unified_parquet(date, hour)
        except Exception as e:
            logger.error(f"Error generating unified Parquet: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate Parquet files for news data")
    parser.add_argument("--source", help="Source name (ynet, hayom, haaretz)")
    parser.add_argument(
        "--unified", action="store_true", help="Generate unified Parquet"
    )
    parser.add_argument(
        "--all-sources", action="store_true", help="Generate Parquet for all sources"
    )
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"), help="Date (YYYY-MM-DD)"
    )
    parser.add_argument("--hour", type=int, help="Hour (0-23)")
    parser.add_argument(
        "--base-path", default="data/parquet", help="Base path for Parquet files"
    )

    args = parser.parse_args()

    # Initialize manager
    manager = ParquetStorageManager(args.base_path)

    try:
        if args.source:
            manager.generate_source_parquet(args.source, args.date, args.hour)
        elif args.unified:
            manager.generate_unified_parquet(args.date, args.hour)
        elif args.all_sources:
            manager.generate_all_sources(args.date, args.hour)
        else:
            logger.error("Must specify --source, --unified, or --all-sources")
            sys.exit(1)

        logger.info("Parquet generation completed successfully")

    except Exception as e:
        logger.error(f"Parquet generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
