"""
Directory Bootstrap Script for News MVP ETL Pipeline.

This script ensures all required data directories exist before running the ETL pipeline.
It creates the necessary directory structure for storing raw data, processed data,
master CSV files, and downloaded images.

Usage:
    python scripts/bootstrap_dirs.py

The script will create these directories if they don't exist:
- data/raw/          # Raw RSS feed data
- data/canonical/    # Processed/cleaned data
- data/master/       # Final merged CSV files
- data/pics/         # Downloaded article images
"""

from news_mvp.paths import Paths  # Directory path management
from news_mvp.logging_setup import get_logger  # Logging configuration

log = get_logger(__name__)


def main():
    for p in Paths.ensure_all():  # Creates all data directories
        log.info("ensured: %s", p.resolve())  # Logs each created directory


if __name__ == "__main__":
    main()
