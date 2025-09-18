#!/usr/bin/env python3
"""Load master CSV file and return as clean, validated Polars DataFrame.

This script takes a master CSV file (from ETL pipeline) and returns it as a
Polars DataFrame with proper schema validation, data cleaning, and ID validation.

Usage:
    python master_to_polars.py --input data/master/master_news.csv
    python master_to_polars.py --input data/master/master_ynet.csv --validate-ids
"""

import argparse
import sys
from pathlib import Path

import polars as pl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from news_mvp.schemas import Stage
from news_mvp.schema_io import read_csv_to_stage_df, coerce_to_stage_df


def validate_article_ids(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """Validate and clean article_id column."""
    if verbose:
        print(f"[validation] Original rows: {len(df)}")

    # Remove rows with empty article_id
    df = df.filter(pl.col("article_id").str.strip_chars() != "")
    if verbose:
        print(f"[validation] After removing empty article_id: {len(df)}")

    # Check for duplicates
    duplicate_count = len(df) - df.select("article_id").unique().height
    if duplicate_count > 0 and verbose:
        print(
            f"[validation] Found {duplicate_count} duplicate article_ids, deduplicating..."
        )
        df = df.unique(subset=["article_id"], keep="last")
        print(f"[validation] After deduplication: {len(df)}")
    elif duplicate_count > 0:
        df = df.unique(subset=["article_id"], keep="last")

    return df


def clean_data(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """Clean and normalize data fields."""
    if verbose:
        print("[cleaning] Applying data cleaning...")

    # Strip whitespace from string fields
    string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    for col in string_cols:
        df = df.with_columns(pl.col(col).str.strip_chars())

    # Convert empty strings to null for nullable fields (except required fields)
    required_fields = {
        "article_id",
        "guid",
        "title",
        # "description",  # No longer required
        "pub_date",
        "source",
        "language",
        "fetching_time",
    }
    nullable_fields = set(df.columns) - required_fields

    for col in nullable_fields:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
            )

    return df


def validate_required_fields(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """Validate that required fields are not null/empty."""
    required_fields = {
        "article_id",
        "guid",
        "title",
        # "description",  # No longer required
        "pub_date",
        "source",
        "language",
        "fetching_time",
    }

    for field in required_fields:
        if field in df.columns:
            null_count = df.filter(
                (pl.col(field).is_null()) | (pl.col(field).str.strip_chars() == "")
            ).height
            if null_count > 0 and verbose:
                print(f"[validation] Warning: {null_count} rows have empty {field}")
                # Remove rows with empty required fields
                df = df.filter(
                    (pl.col(field).is_not_null())
                    & (pl.col(field).str.strip_chars() != "")
                )
                print(f"[validation] After removing empty {field}: {len(df)}")
            elif null_count > 0:
                # Remove rows with empty required fields silently
                df = df.filter(
                    (pl.col(field).is_not_null())
                    & (pl.col(field).str.strip_chars() != "")
                )

    return df


def load_master_csv_to_polars(
    input_path: str, validate_ids: bool = True, clean: bool = True, verbose: bool = True
) -> pl.DataFrame:
    """Load master CSV and return as clean, validated Polars DataFrame.

    Args:
        input_path: Path to master CSV file
        validate_ids: Whether to validate and deduplicate article_id
        clean: Whether to apply data cleaning
        verbose: Whether to print progress messages

    Returns:
        Clean, validated Polars DataFrame with master schema
    """

    def print_if_verbose(msg: str):
        if verbose:
            print(msg)

    print_if_verbose(f"[load] Reading CSV: {input_path}")

    # Read CSV using schema-aware function
    df = read_csv_to_stage_df(input_path, Stage.ETL_BEFORE_MERGE)
    print_if_verbose(f"[load] Loaded {len(df)} rows with {len(df.columns)} columns")
    print_if_verbose(f"[load] Columns: {df.columns}")

    # Validate required fields
    df = validate_required_fields(df, verbose=verbose)

    # Validate article IDs if requested
    if validate_ids:
        df = validate_article_ids(df, verbose=verbose)

    # Clean data if requested
    if clean:
        df = clean_data(df, verbose=verbose)

    # --- Transform pub_date to datetime, normalize to +3 timezone, and add merging_time column ---
    import polars as pl
    import pytz
    import datetime

    # Convert pub_date to datetime (assume input is string)
    if "pub_date" in df.columns:
        # Try to parse pub_date as datetime, localize to UTC if not already tz-aware
        try:
            df = df.with_columns(
                pl.col("pub_date")
                .str.strptime(pl.Datetime, strict=False)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Asia/Jerusalem")
                .alias("pub_date")
            )
        except Exception as e:
            print_if_verbose(f"[warning] Could not convert pub_date to datetime: {e}")

    # Normalize fetching_time to Asia/Jerusalem as well
    if "fetching_time" in df.columns:
        try:
            df = df.with_columns(
                pl.col("fetching_time")
                .str.strptime(pl.Datetime, strict=False)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Asia/Jerusalem")
                .alias("fetching_time")
            )
        except Exception as e:
            print_if_verbose(
                f"[warning] Could not convert fetching_time to datetime: {e}"
            )

    # Add merging_time column at the end (current time in Asia/Jerusalem)
    try:
        import pytz

        tz = pytz.timezone("Asia/Jerusalem")
        now = datetime.datetime.now(tz)
        df = df.with_columns(pl.lit(now).alias("merging_time"))
    except Exception as e:
        print_if_verbose(f"[warning] Could not add merging_time: {e}")

    # Final schema coercion to ensure everything is correct
    df = coerce_to_stage_df(df, Stage.ETL_BEFORE_MERGE, strict=True)

    print_if_verbose(
        f"[result] Final DataFrame: {len(df)} rows × {len(df.columns)} columns"
    )
    return df


def get_dataframe_stats(df: pl.DataFrame) -> dict:
    """Get basic statistics about the DataFrame.

    Returns:
        Dictionary with basic stats about the DataFrame
    """
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "sources": df.select("source").unique().to_series().to_list(),
        "source_counts": df.group_by("source")
        .agg(pl.count("article_id").alias("count"))
        .sort("count", descending=True)
        .to_dicts(),
        "date_range": {
            "earliest": df.select(pl.col("pub_date").min()).item(),
            "latest": df.select(pl.col("pub_date").max()).item(),
        },
        "has_images": df.filter(pl.col("image").is_not_null()).height,
        "missing_images": df.filter(pl.col("image").is_null()).height,
        "languages": df.select("language").unique().to_series().to_list(),
        "articles_with_authors": df.filter(
            (pl.col("author").is_not_null()) & (pl.col("author") != "")
        ).height,
        "articles_with_tags": df.filter(
            (pl.col("tags").is_not_null()) & (pl.col("tags") != "")
        ).height,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Load master CSV and return as clean Polars DataFrame"
    )
    parser.add_argument("--input", required=True, help="Path to master CSV file")
    parser.add_argument(
        "--validate-ids",
        action="store_true",
        help="Validate and deduplicate article_id column",
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip data cleaning steps"
    )
    parser.add_argument(
        "--show-sample", action="store_true", help="Show sample of the data"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show detailed statistics about the data"
    )

    args = parser.parse_args()

    # Load the data
    df = load_master_csv_to_polars(
        input_path=args.input, validate_ids=args.validate_ids, clean=not args.no_clean
    )

    # Show basic info
    print(f"\n[info] DataFrame shape: {df.shape}")
    print(f"[info] Schema: {df.schema}")

    # Show detailed stats if requested
    if args.stats:
        stats = get_dataframe_stats(df)
        print("\n[stats] Detailed Statistics:")
        print(f"  Total articles: {stats['total_rows']}")
        print(f"  Sources: {', '.join(stats['sources'])}")
        print("  Source distribution:")
        for source_info in stats["source_counts"]:
            print(f"    {source_info['source']}: {source_info['count']} articles")
        print(
            f"  Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}"
        )
        print(
            f"  Images: {stats['has_images']} have images, {stats['missing_images']} missing"
        )
        print(f"  Languages: {', '.join(stats['languages'])}")
        print(f"  Articles with authors: {stats['articles_with_authors']}")
        print(f"  Articles with tags: {stats['articles_with_tags']}")

    # Show sample if requested
    if args.show_sample:
        print("\n[sample] First 3 rows:")
        print(df.head(3))

        print("\n[sample] Article ID samples:")
        print(df.select("article_id").head(5))

        print("\n[sample] Source distribution:")
        print(
            df.group_by("source")
            .agg(pl.count("article_id").alias("count"))
            .sort("count", descending=True)
        )

    return df


if __name__ == "__main__":
    result_df = main()
