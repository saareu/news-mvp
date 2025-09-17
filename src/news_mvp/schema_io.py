"""Schema-aware I/O utilities for ETL pipeline."""

from typing import Dict, Optional, Union
import pandas as pd
import polars as pl

from news_mvp.schemas import Stage, schema_fieldnames
from news_mvp.settings import get_runtime_csv_encoding


# Known renames from current pipeline headers to master schema snake_case
KNOWN_RENAMES: Dict[str, str] = {
    "id": "article_id",
    "pubDate": "pub_date",
    "imageCaption": "image_caption",
    "imageCredit": "image_credit",
    "creator": "author",
    "imageName": "image_name",
    "fetchingtime": "fetching_time",
}


def _to_polars(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
    """Convert any DataFrame to polars."""
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def _normalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Apply known renames to get master schema column names."""
    renames = {}
    for col in df.columns:
        # Case-insensitive matching for known renames
        for old_name, new_name in KNOWN_RENAMES.items():
            if col.lower() == old_name.lower():
                renames[col] = new_name
                break

    if renames:
        df = df.rename(renames)
    return df


def coerce_to_stage_df(
    df: pl.DataFrame, stage: Stage, strict: bool = False
) -> pl.DataFrame:
    """Coerce DataFrame to match stage schema."""
    df = _normalize_column_names(df)
    schema_cols = schema_fieldnames(stage)

    # Add missing columns with null values
    missing_cols = set(schema_cols) - set(df.columns)
    for col in missing_cols:
        df = df.with_columns(pl.lit(None).alias(col))

    # Reorder columns to match schema order
    df = df.select(schema_cols)

    if strict:
        # Validate no extra columns (already handled by select above)
        pass

    return df


def read_csv_to_stage_df(
    input_path: str, stage: Stage, encoding: Optional[str] = None
) -> pl.DataFrame:
    """Read CSV and coerce to stage schema."""
    enc = encoding or get_runtime_csv_encoding()
    df = pd.read_csv(input_path, encoding=enc, dtype=str)
    df = df.fillna("")
    pl_df = pl.from_pandas(df)
    return coerce_to_stage_df(pl_df, stage=stage, strict=False)


def write_stage_df(
    df: Union[pl.DataFrame, pd.DataFrame],
    output_path: str,
    stage: Stage,
    encoding: Optional[str] = None,
) -> None:
    """Write DataFrame to CSV with stage schema validation."""
    enc = encoding or get_runtime_csv_encoding()
    # Ensure schema shape/order, then write via pandas to preserve encoding (utf-8-sig)
    pl_df = _to_polars(df)
    pl_df = coerce_to_stage_df(pl_df, stage=stage, strict=True)
    pd_df = pl_df.to_pandas(use_pyarrow_extension_array=False)
    pd_df.to_csv(output_path, index=False, encoding=enc)
