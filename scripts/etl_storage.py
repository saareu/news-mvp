#!/usr/bin/env python3
"""
News MVP Storage ETL Pipeline

This script orchestrates the complete storage pipeline:
1. Generate Parquet files (per-source + unified)
2. Organize images by source/date/hour
3. Load data into DuckDB
4. Update image paths and metadata

Usage:
    python scripts/etl_storage.py --source ynet
    python scripts/etl_storage.py --all-sources
    python scripts/etl_storage.py --full-pipeline
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class StorageETLPipeline:
    """Orchestrates the complete storage ETL pipeline."""

    def __init__(self):
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.current_hour = datetime.now().hour

    def run_command(self, cmd: list, description: str) -> bool:
        """Run a command and return success status."""
        try:
            logger.info(f"Running: {description}")
            logger.debug(f"Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                if result.stdout:
                    logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ {description} failed")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            return False

    def generate_parquet_files(self, source: Optional[str] = None):
        """Generate Parquet files for sources."""
        if source:
            # Generate for specific source
            cmd = [
                sys.executable,
                "scripts/generate_parquet.py",
                "--source",
                source,
                "--date",
                self.current_date,
                "--hour",
                str(self.current_hour),
            ]
            return self.run_command(cmd, f"Generate Parquet for {source}")
        else:
            # Generate for all sources + unified
            cmd = [
                sys.executable,
                "scripts/generate_parquet.py",
                "--all-sources",
                "--date",
                self.current_date,
                "--hour",
                str(self.current_hour),
            ]
            return self.run_command(cmd, "Generate Parquet for all sources")

    def organize_images(self, source: Optional[str] = None):
        """Organize images by source/date/hour."""
        if source:
            cmd = [
                sys.executable,
                "scripts/manage_images.py",
                "--organize",
                "--source",
                source,
                "--date",
                self.current_date,
                "--hour",
                str(self.current_hour),
            ]
            return self.run_command(cmd, f"Organize images for {source}")
        else:
            cmd = [
                sys.executable,
                "scripts/manage_images.py",
                "--organize",
                "--all-sources",
                "--date",
                self.current_date,
                "--hour",
                str(self.current_hour),
            ]
            return self.run_command(cmd, "Organize images for all sources")

    def update_image_paths(self, source: Optional[str] = None):
        """Update image paths in CSV files."""
        if source:
            cmd = [
                sys.executable,
                "scripts/manage_images.py",
                "--update-csv",
                "--source",
                source,
                "--date",
                self.current_date,
                "--hour",
                str(self.current_hour),
            ]
            return self.run_command(cmd, f"Update image paths for {source}")
        else:
            cmd = [
                sys.executable,
                "scripts/manage_images.py",
                "--update-csv",
                "--all-sources",
                "--date",
                self.current_date,
                "--hour",
                str(self.current_hour),
            ]
            return self.run_command(cmd, "Update image paths for all sources")

    def load_database(self, source: Optional[str] = None):
        """Load data into DuckDB."""
        if source:
            cmd = [sys.executable, "scripts/load_database.py", "--source", source]
            return self.run_command(cmd, f"Load {source} data into database")
        else:
            cmd = [sys.executable, "scripts/load_database.py", "--all-sources"]
            return self.run_command(cmd, "Load all sources into database")

    def run_source_pipeline(self, source: str):
        """Run complete pipeline for a specific source."""
        logger.info(f"🚀 Starting storage pipeline for {source}")

        steps = [
            ("Generate Parquet", lambda: self.generate_parquet_files(source)),
            ("Organize Images", lambda: self.organize_images(source)),
            ("Update Image Paths", lambda: self.update_image_paths(source)),
            ("Load Database", lambda: self.load_database(source)),
        ]

        success_count = 0
        for step_name, step_func in steps:
            if step_func():
                success_count += 1
            else:
                logger.warning(f"Step '{step_name}' failed, continuing...")

        logger.info(
            f"✅ Pipeline completed for {source}: {success_count}/{len(steps)} steps successful"
        )
        return success_count == len(steps)

    def run_full_pipeline(self):
        """Run complete pipeline for all sources."""
        logger.info("🚀 Starting full storage pipeline for all sources")

        sources = ["ynet", "hayom", "haaretz"]
        success_count = 0

        for source in sources:
            if self.run_source_pipeline(source):
                success_count += 1

        # Generate unified Parquet after all sources
        logger.info("🔄 Generating unified Parquet files")
        if self.generate_parquet_files():
            success_count += 1

        logger.info(
            f"✅ Full pipeline completed: {success_count}/{len(sources) + 1} parts successful"
        )
        return success_count == len(sources) + 1

    def show_stats(self):
        """Show storage statistics."""
        logger.info("📊 Storage Statistics:")

        # Database stats
        try:
            cmd = [sys.executable, "scripts/load_database.py", "--stats"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                print("Database Stats:")
                print(result.stdout)
        except Exception as e:
            logger.warning(f"Could not get database stats: {e}")

        # Image stats
        try:
            cmd = [sys.executable, "scripts/manage_images.py", "--stats"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                print("Image Stats:")
                print(result.stdout)
        except Exception as e:
            logger.warning(f"Could not get image stats: {e}")

        # Parquet files
        parquet_dir = Path("data/parquet")
        if parquet_dir.exists():
            total_parquet = sum(1 for _ in parquet_dir.rglob("*.parquet"))
            print(f"Parquet Files: {total_parquet}")

            # Show recent unified files
            unified_dir = parquet_dir / "unified" / "latest_unified.parquet"
            if unified_dir.exists():
                print(f"Latest Unified: {unified_dir}")


def main():
    parser = argparse.ArgumentParser(description="News MVP Storage ETL Pipeline")
    parser.add_argument(
        "--source", help="Process specific source (ynet, hayom, haaretz)"
    )
    parser.add_argument(
        "--all-sources", action="store_true", help="Process all sources"
    )
    parser.add_argument(
        "--full-pipeline", action="store_true", help="Run complete pipeline"
    )
    parser.add_argument("--stats", action="store_true", help="Show storage statistics")
    parser.add_argument("--date", help="Override date (YYYY-MM-DD)")
    parser.add_argument("--hour", type=int, help="Override hour (0-23)")

    args = parser.parse_args()

    pipeline = StorageETLPipeline()

    # Override date/hour if specified
    if args.date:
        pipeline.current_date = args.date
    if args.hour is not None:
        pipeline.current_hour = args.hour

    try:
        if args.stats:
            pipeline.show_stats()

        elif args.source:
            success = pipeline.run_source_pipeline(args.source)
            sys.exit(0 if success else 1)

        elif args.all_sources:
            success = pipeline.run_full_pipeline()
            sys.exit(0 if success else 1)

        elif args.full_pipeline:
            success = pipeline.run_full_pipeline()
            sys.exit(0 if success else 1)

        else:
            logger.error(
                "Must specify --source, --all-sources, --full-pipeline, or --stats"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
