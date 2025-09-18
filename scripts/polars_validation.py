#!/usr/bin/env python3
"""Polars validation for master CSV files.

This script validates the output from master_to_polars.py, ensuring data quality,
schema compliance, and business rules are met.

Usage:
    python polars_validation.py --input data/master/master_ynet.csv
    python polars_validation.py --input data/master/master_ynet.csv --source ynet --strict
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from news_mvp.schemas import Stage
from news_mvp.schema_io import read_csv_to_stage_df, coerce_to_stage_df


class ValidationResult:
    """Container for validation results."""

    def __init__(self, source: str, file_path: str):
        self.source = source
        self.file_path = file_path
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.is_valid = True

    def add_error(self, error: str):
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning."""
        self.warnings.append(warning)

    def summary(self) -> str:
        """Get a summary of validation results."""
        total_issues = len(self.errors) + len(self.warnings)
        if total_issues == 0:
            return f"✅ {self.source}: VALID ({len(self.stats.get('rows', 0))} rows)"
        else:
            return f"❌ {self.source}: {len(self.errors)} errors, {len(self.warnings)} warnings"


def validate_required_fields(df: pl.DataFrame, result: ValidationResult) -> None:
    """Validate that all required fields are present and non-null."""
    required_fields = {
        "ynet": ["article_id", "title", "description", "source", "pub_date"],
        "haaretz": ["article_id", "title", "description", "source", "pub_date"],
        "hayom": ["article_id", "title", "description", "source", "pub_date"],
    }

    source_required = required_fields.get(result.source, required_fields["ynet"])

    for field in source_required:
        if field not in df.columns:
            result.add_error(f"Missing required field: {field}")
        else:
            null_count = df[field].null_count()
            empty_count = df.filter(pl.col(field).str.strip_chars() == "").height

            if null_count > 0:
                result.add_error(f"Field '{field}' has {null_count} null values")
            if empty_count > 0:
                result.add_error(f"Field '{field}' has {empty_count} empty strings")


def validate_data_types(df: pl.DataFrame, result: ValidationResult) -> None:
    """Validate data types and formats."""

    # Date validation
    if "pub_date" in df.columns:
        try:
            # Try to parse dates - should already be validated by master_to_polars.py
            date_series = df["pub_date"]
            if date_series.dtype != pl.Date:
                result.add_warning(
                    f"pub_date column is {date_series.dtype}, expected Date"
                )
        except Exception as e:
            result.add_error(f"Date validation error: {e}")

    # URL validation
    if "url" in df.columns:
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        invalid_urls = df.filter(
            pl.col("url").is_not_null()
            & ~pl.col("url").str.contains(url_pattern, literal=False)
        )
        if len(invalid_urls) > 0:
            result.add_error(f"{len(invalid_urls)} invalid URL formats found")

    # Article ID format validation
    if "article_id" in df.columns:
        # Check for reasonable article ID lengths
        short_ids = df.filter(pl.col("article_id").str.len_chars() < 5)
        long_ids = df.filter(pl.col("article_id").str.len_chars() > 100)

        if len(short_ids) > 0:
            result.add_warning(
                f"{len(short_ids)} article_ids are suspiciously short (< 5 chars)"
            )
        if len(long_ids) > 0:
            result.add_warning(
                f"{len(long_ids)} article_ids are very long (> 100 chars)"
            )


def validate_field_content(df: pl.DataFrame, result: ValidationResult) -> None:
    """Validate field content quality."""

    # Title validation
    if "title" in df.columns:

        # Length checks
        short_titles = df.filter(pl.col("title").str.len_chars() < 5)
        long_titles = df.filter(pl.col("title").str.len_chars() > 200)

        if len(short_titles) > 0:
            result.add_warning(
                f"{len(short_titles)} titles are suspiciously short (< 5 chars)"
            )
        if len(long_titles) > 0:
            result.add_warning(f"{len(long_titles)} titles are very long (> 200 chars)")

        # Check for placeholder text
        placeholders = df.filter(pl.col("title").str.contains(r"\{\}|\[.*?\]"))
        if len(placeholders) > 0:
            result.add_error(f"{len(placeholders)} titles contain placeholder text")

    # Description validation
    if "description" in df.columns:

        # Check for placeholder text
        placeholders = df.filter(pl.col("description").str.contains(r"\{\}"))
        if len(placeholders) > 0:
            result.add_error(
                f"{len(placeholders)} descriptions contain placeholder text '{{}}'"
            )

        # Length validation
        short_descs = df.filter(pl.col("description").str.len_chars() < 20)
        if len(short_descs) > 0:
            result.add_warning(
                f"{len(short_descs)} descriptions are very short (< 20 chars)"
            )

    # Author validation
    if "author" in df.columns:
        # Check for proper author formatting
        bad_authors = df.filter(
            pl.col("author").str.contains(r",\s*,")  # Multiple commas without space
            | pl.col("author").str.contains(r"^,|,$")  # Leading/trailing comma
        )
        if len(bad_authors) > 0:
            result.add_warning(
                f"{len(bad_authors)} author fields have formatting issues"
            )


def validate_cross_field_consistency(
    df: pl.DataFrame, result: ValidationResult
) -> None:
    """Validate relationships between related fields."""

    # Image field consistency
    if "image" in df.columns and "image_credit" in df.columns:
        # If image exists, image_credit should ideally exist
        missing_credits = df.filter(
            pl.col("image").is_not_null()
            & (pl.col("image_credit").is_null() | (pl.col("image_credit") == ""))
        )
        if len(missing_credits) > 0:
            result.add_warning(
                f"{len(missing_credits)} articles have image but missing image_credit"
            )

    # Image caption consistency
    if "image" in df.columns and "image_caption" in df.columns:
        missing_captions = df.filter(
            pl.col("image").is_not_null()
            & (pl.col("image_caption").is_null() | (pl.col("image_caption") == ""))
        )
        if len(missing_captions) > 0:
            result.add_warning(
                f"{len(missing_captions)} articles have image but missing image_caption"
            )

    # Tags consistency
    if "tags" in df.columns:
        # Check for proper tag formatting (should be pipe-separated)
        bad_tags = df.filter(
            pl.col("tags").str.contains(r",[^|]")  # Comma not followed by pipe
            | pl.col("tags").str.contains(r"\|,")  # Pipe followed by comma
        )
        if len(bad_tags) > 0:
            result.add_warning(
                f"{len(bad_tags)} tag fields have inconsistent formatting"
            )


def validate_duplicates(df: pl.DataFrame, result: ValidationResult) -> None:
    """Detect duplicate articles."""

    # Check for duplicate article_ids
    if "article_id" in df.columns:
        duplicate_ids = (
            df.group_by("article_id").agg(pl.len()).filter(pl.col("len") > 1)
        )
        if len(duplicate_ids) > 0:
            result.add_error(f"{len(duplicate_ids)} duplicate article_ids found")

    # Check for duplicate URLs
    if "url" in df.columns:
        duplicate_urls = df.group_by("url").agg(pl.len()).filter(pl.col("len") > 1)
        if len(duplicate_urls) > 0:
            result.add_error(f"{len(duplicate_urls)} duplicate URLs found")

    # Check for duplicate titles (within same source)
    if "title" in df.columns and "source" in df.columns:
        duplicate_titles = (
            df.group_by(["source", "title"]).agg(pl.len()).filter(pl.col("len") > 1)
        )
        if len(duplicate_titles) > 0:
            result.add_warning(
                f"{len(duplicate_titles)} duplicate title+source combinations found"
            )


def validate_source_specific(df: pl.DataFrame, result: ValidationResult) -> None:
    """Source-specific validation rules."""

    if result.source == "ynet":
        # YNet specific validations
        if "author" in df.columns:
            # Check for Hebrew author names (YNet often has Hebrew authors)
            hebrew_authors = df.filter(
                pl.col("author").is_not_null()
                & pl.col("author").str.contains(r"[\u0590-\u05FF]")
            )
            total_authors = df.filter(pl.col("author").is_not_null()).height
            if total_authors > 0 and len(hebrew_authors) == 0:
                result.add_warning(
                    "No Hebrew author names found - possible encoding issue"
                )

        if "category" in df.columns:
            # YNet categories should be in Hebrew
            hebrew_categories = df.filter(
                pl.col("category").is_not_null()
                & pl.col("category").str.contains(r"[\u0590-\u05FF]")
            )
            total_categories = df.filter(pl.col("category").is_not_null()).height
            if total_categories > 0 and len(hebrew_categories) == 0:
                result.add_warning(
                    "No Hebrew categories found - possible encoding issue"
                )

    elif result.source == "haaretz":
        # Haaretz specific validations
        if "category" in df.columns:
            # Haaretz categories should be in Hebrew
            hebrew_categories = df.filter(
                pl.col("category").is_not_null()
                & pl.col("category").str.contains(r"[\u0590-\u05FF]")
            )
            total_categories = df.filter(pl.col("category").is_not_null()).height
            if total_categories > 0 and len(hebrew_categories) == 0:
                result.add_warning(
                    "No Hebrew categories found - possible encoding issue"
                )


def generate_statistics(df: pl.DataFrame, result: ValidationResult) -> None:
    """Generate comprehensive statistics about the DataFrame."""

    result.stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns,
        "null_counts": {col: df[col].null_count() for col in df.columns},
        "unique_counts": {
            col: df[col].n_unique()
            for col in df.columns
            if df[col].dtype in [pl.Utf8, pl.Categorical]
        },
        "data_types": {col: str(df[col].dtype) for col in df.columns},
    }

    # Add field-specific stats
    if "source" in df.columns:
        result.stats["sources"] = df["source"].unique().to_list()
        result.stats["source_distribution"] = (
            df.group_by("source")
            .agg(pl.count("article_id").alias("count"))
            .sort("count", descending=True)
            .to_dicts()
        )

    if "pub_date" in df.columns and df["pub_date"].dtype == pl.Date:
        result.stats["date_range"] = {
            "earliest": df["pub_date"].min(),
            "latest": df["pub_date"].max(),
        }

    # Content quality stats
    if "title" in df.columns:
        result.stats["title_stats"] = {
            "avg_length": df["title"].str.len_chars().mean(),
            "min_length": df["title"].str.len_chars().min(),
            "max_length": df["title"].str.len_chars().max(),
        }

    if "description" in df.columns:
        result.stats["description_stats"] = {
            "avg_length": df["description"].str.len_chars().mean(),
            "min_length": df["description"].str.len_chars().min(),
            "max_length": df["description"].str.len_chars().max(),
        }


def validate_dataframe(
    df: pl.DataFrame, source: str, file_path: str, strict: bool = False
) -> ValidationResult:
    """Complete validation of a Polars DataFrame from master_to_polars.py output.

    Args:
        df: Polars DataFrame from master_to_polars.py
        source: Source name (ynet, haaretz, hayom)
        file_path: Path to the original file
        strict: If True, warnings become errors

    Returns:
        ValidationResult with comprehensive validation results
    """

    result = ValidationResult(source, file_path)

    # Generate statistics first
    generate_statistics(df, result)

    # Run all validations
    validate_required_fields(df, result)
    validate_data_types(df, result)
    validate_field_content(df, result)
    validate_cross_field_consistency(df, result)
    validate_duplicates(df, result)
    validate_source_specific(df, result)

    # In strict mode, warnings become errors
    if strict:
        result.errors.extend(result.warnings)
        result.warnings.clear()
        if result.warnings:  # If there were warnings, mark as invalid
            result.is_valid = False

    return result


def load_and_validate_master_csv(
    input_path: str,
    source: str,
    validate_ids: bool = True,
    clean: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[pl.DataFrame, ValidationResult]:
    """Load master CSV using master_to_polars.py logic and validate it.

    This function replicates the logic from master_to_polars.py and then
    validates the resulting DataFrame.

    Args:
        input_path: Path to master CSV file
        source: Source name (ynet, haaretz, hayom)
        validate_ids: Whether to validate article IDs
        clean: Whether to clean data
        strict: Whether to treat warnings as errors
        verbose: Whether to print progress

    Returns:
        Tuple of (DataFrame, ValidationResult)
    """

    def print_if_verbose(msg: str):
        if verbose:
            print(msg)

    print_if_verbose(f"[load] Reading CSV: {input_path}")

    # Read CSV using schema-aware function
    df = read_csv_to_stage_df(input_path, Stage.ETL_BEFORE_MERGE)
    print_if_verbose(f"[load] Loaded {len(df)} rows with {len(df.columns)} columns")

    # Basic validation and cleaning (from master_to_polars.py)
    if validate_ids:
        # Remove rows with empty article_id
        original_count = len(df)
        df = df.filter(pl.col("article_id").str.strip_chars() != "")
        if len(df) < original_count:
            print_if_verbose(
                f"[validation] Removed {original_count - len(df)} rows with empty article_id"
            )

        # Deduplicate
        original_count = len(df)
        df = df.unique(subset=["article_id"], keep="last")
        if len(df) < original_count:
            print_if_verbose(
                f"[validation] Removed {original_count - len(df)} duplicate article_ids"
            )

    if clean:
        # Strip whitespace from string fields
        string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
        for col in string_cols:
            df = df.with_columns(pl.col(col).str.strip_chars())

        # Convert empty strings to null for nullable fields
        required_fields = {
            "article_id",
            "guid",
            "title",
            "pub_date",
            "source",
            "language",
            "fetching_time",
        }
        nullable_fields = set(df.columns) - required_fields

        for col in nullable_fields:
            if df[col].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.when(pl.col(col) == "")
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

    # Final schema coercion
    df = coerce_to_stage_df(df, Stage.ETL_BEFORE_MERGE, strict=True)
    print_if_verbose(
        f"[result] Final DataFrame: {len(df)} rows × {len(df.columns)} columns"
    )

    # Now validate the cleaned DataFrame
    print_if_verbose("[validation] Running validation checks...")
    validation_result = validate_dataframe(df, source, input_path, strict)

    return df, validation_result


def main():
    parser = argparse.ArgumentParser(
        description="Validate master CSV files with comprehensive schema and data quality checks"
    )
    parser.add_argument("--input", required=True, help="Path to master CSV file")
    parser.add_argument(
        "--source",
        required=True,
        choices=["ynet", "haaretz", "hayom"],
        help="Source name for source-specific validation",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    parser.add_argument(
        "--no-validate-ids",
        action="store_true",
        help="Skip article ID validation and deduplication",
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip data cleaning steps"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages"
    )
    parser.add_argument(
        "--show-errors", action="store_true", help="Show detailed error list"
    )
    parser.add_argument(
        "--show-warnings", action="store_true", help="Show detailed warning list"
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Show detailed statistics"
    )

    args = parser.parse_args()

    # Load and validate
    df, result = load_and_validate_master_csv(
        input_path=args.input,
        source=args.source,
        validate_ids=not args.no_validate_ids,
        clean=not args.no_clean,
        strict=args.strict,
        verbose=not args.quiet,
    )

    # Print summary
    print(f"\n{result.summary()}")

    # Show detailed results if requested
    if args.show_errors and result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for i, error in enumerate(result.errors, 1):
            print(f"  {i}. {error}")

    if args.show_warnings and result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning}")

    if args.show_stats:
        print("\n📊 Statistics:")
        print(f"  Rows: {result.stats.get('rows', 0)}")
        print(f"  Columns: {result.stats.get('columns', 0)}")
        print(f"  Sources: {', '.join(result.stats.get('sources', []))}")

        if "date_range" in result.stats:
            date_range = result.stats["date_range"]
            print(f"  Date range: {date_range['earliest']} to {date_range['latest']}")

        if "source_distribution" in result.stats:
            print("  Source distribution:")
            for source_info in result.stats["source_distribution"]:
                print(f"    {source_info['source']}: {source_info['count']} articles")

    # Exit with appropriate code
    if not result.is_valid:
        print(f"\n❌ Validation FAILED for {args.source}")
        sys.exit(1)
    else:
        print(f"\n✅ Validation PASSED for {args.source}")
        return df


if __name__ == "__main__":
    result_df = main()
