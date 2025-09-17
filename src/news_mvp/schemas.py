"""
Centralized schema definitions for the News MVP project.

This module provides:
- PyArrow schemas for Parquet files
- Pydantic models for data validation
- CSV column definitions
- Data type mappings for pandas

Note: Database schemas are defined in schema.sql for comprehensive table definitions,
foreign keys, indexes, and constraints.
"""

import logging
from typing import List, Optional
import pyarrow as pa
import pandas as pd
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger(__name__)


# ===== PYARROW SCHEMAS =====


def get_parquet_schema() -> pa.schema:
    """Get the standard Parquet schema for articles using CSV casing.

    Field naming is unified across CSV → Parquet → DuckDB:
    id, title, description, category, pubDate, tags (list<string>), creator, source,
    language, image, imageCaption, imageCredit, imageName, imageBlob (optional bytes),
    guid, processed_at, batch_hour.
    pubDate stored as timestamp (ns) for analytical queries; original string form can
    be regenerated if needed. Tags become list<string>.
    """
    return pa.schema(
        [
            ("id", pa.string()),
            ("title", pa.string()),
            ("description", pa.string()),
            ("category", pa.string()),
            ("pubDate", pa.timestamp("ns")),
            ("tags", pa.list_(pa.string())),
            ("creator", pa.string()),
            ("source", pa.string()),
            ("language", pa.string()),
            ("image", pa.string()),
            ("imageCaption", pa.string()),
            ("imageCredit", pa.string()),
            ("imageName", pa.string()),
            ("imageBlob", pa.binary()),
            ("guid", pa.string()),
            ("processed_at", pa.timestamp("ns")),
            ("batch_hour", pa.int64()),
        ]
    )


def get_raw_parquet_schema() -> pa.schema:
    """Get schema for raw RSS data."""
    return pa.schema(
        [
            ("id", pa.string()),
            ("title", pa.string()),
            ("description", pa.string()),
            ("link", pa.string()),
            ("pubDate", pa.timestamp("ns")),
            ("source", pa.string()),
            ("raw_xml", pa.string()),
        ]
    )


# ===== PANDAS DATA TYPE MAPPINGS =====

ARTICLE_DTYPES = {
    "id": "string",
    "title": "string",
    "description": "string",
    "category": "string",
    "pubDate": "datetime64[ns, UTC]",
    "tags": "object",  # list of strings
    "creator": "string",
    "source": "string",
    "language": "string",
    "image": "string",
    "imageCaption": "string",
    "imageCredit": "string",
    "imageName": "string",
    "imageBlob": "object",  # will hold bytes when populated
    "guid": "string",
    "processed_at": "datetime64[ns, UTC]",
    "batch_hour": "Int64",
}


RAW_ARTICLE_DTYPES = {
    "id": "string",
    "title": "string",
    "description": "string",
    "link": "string",
    "pubDate": "datetime64[ns]",
    "source": "string",
    "raw_xml": "string",
}


# ===== PYDANTIC MODELS =====


class ArticleModel(BaseModel):
    """Article model with unified CSV casing."""

    id: str = Field(..., description="Article ID")
    title: str = Field(..., description="Article title")
    description: Optional[str] = Field(None, description="Article description")
    category: Optional[str] = Field(None, description="Article category")
    pubDate: datetime = Field(..., description="Publication datetime (UTC)")
    tags: List[str] = Field(default_factory=list, description="Article tags")
    creator: Optional[str] = Field(None, description="Article creator")
    source: str = Field(..., description="Data source")
    language: str = Field("he", description="Article language")
    image: Optional[str] = Field(None, description="Image path")
    imageCaption: Optional[str] = Field(None, description="Image caption")
    imageCredit: Optional[str] = Field(None, description="Image credit")
    imageName: Optional[str] = Field(None, description="Image filename")
    imageBlob: Optional[bytes] = Field(None, description="Raw image bytes (optional)")
    guid: Optional[str] = Field(None, description="RSS feed unique identifier")
    processed_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Processing timestamp UTC",
    )
    batch_hour: int = Field(
        default_factory=lambda: datetime.utcnow().hour,
        description="Processing hour (UTC)",
    )


class RawArticleModel(BaseModel):
    """Pydantic model for raw article data."""

    id: str = Field(..., description="Article ID")
    title: str = Field(..., description="Article title")
    description: Optional[str] = Field(None, description="Article description")
    link: Optional[str] = Field(None, description="Article URL")
    pubDate: datetime = Field(..., description="Publication date")
    source: str = Field(..., description="Data source")
    raw_xml: Optional[str] = Field(None, description="Raw XML data")


class CSVArticleModel(BaseModel):
    """Pydantic model for CSV article data validation."""

    id: str = Field(..., description="Article ID")
    title: str = Field(..., description="Article title")
    description: Optional[str] = Field(None, description="Article description")
    category: Optional[str] = Field(None, description="Article category")
    image: Optional[str] = Field(None, description="Image path")
    imageCaption: Optional[str] = Field(None, description="Image caption")
    imageCredit: Optional[str] = Field(None, description="Image credit")
    pubDate: str = Field(..., description="Publication date string")
    tags: Optional[str] = Field(None, description="Tags as pipe-separated string")
    creator: Optional[str] = Field(None, description="Article creator")
    source: str = Field(..., description="Data source")
    language: Optional[str] = Field("he", description="Article language")
    imageName: Optional[str] = Field(None, description="Image filename")
    guid: Optional[str] = Field(None, description="RSS feed unique identifier")


# ===== CSV COLUMN DEFINITIONS =====

CSV_COLUMNS = {
    "master": [
        "id",
        "title",
        "description",
        "category",
        "image",
        "imageCaption",
        "imageCredit",
        "pubDate",
        "tags",
        "creator",
        "source",
        "language",
        "imageName",
        "guid",
    ],
    # canonical view keeps subset
    "canonical": [
        "id",
        "title",
        "description",
        "category",
        "pubDate",
        "tags",
        "creator",
        "source",
        "language",
    ],
    # enhanced aligns naming; imageBlob may appear after enrichment
    "enhanced": [
        "id",
        "title",
        "description",
        "category",
        "pubDate",
        "tags",
        "creator",
        "source",
        "language",
        "image",
        "imageCaption",
        "imageCredit",
        "imageName",
        "imageBlob",
        "guid",
    ],
}


# ===== UTILITY FUNCTIONS =====


def validate_article_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize article DataFrame using unified schema.

    - Keeps original column names (CSV casing).
    - Normalizes pubDate to UTC and truncates to hour resolution.
    - Splits tags pipe string into list; if already list leaves as-is.
    - Returns DataFrame with validated rows only.
    """
    validated: List[dict] = []
    for _, row in df.iterrows():
        try:
            row_dict: dict = {}
            for col, val in row.items():
                row_dict[col] = None if pd.isna(val) else val

            # Normalize pubDate (accept string or datetime)
            if row_dict.get("pubDate") is not None:
                try:
                    dt = pd.to_datetime(row_dict["pubDate"], utc=True)
                    # Truncate to hour for consistency
                    dt = dt.replace(minute=0, second=0, microsecond=0)
                    row_dict["pubDate"] = dt.to_pydatetime()
                except Exception as e:  # pragma: no cover
                    raise ValueError(
                        f"Invalid pubDate: {row_dict.get('pubDate')} ({e})"
                    )

            # Normalize tags
            tags_val = row_dict.get("tags")
            if isinstance(tags_val, str):
                # Support both pipe and comma separated
                if "|" in tags_val:
                    tags_list = [t.strip() for t in tags_val.split("|") if t.strip()]
                else:
                    tags_list = [t.strip() for t in tags_val.split(",") if t.strip()]
                row_dict["tags"] = tags_list
            elif tags_val is None:
                row_dict["tags"] = []
            elif isinstance(tags_val, list):
                row_dict["tags"] = [str(t).strip() for t in tags_val if str(t).strip()]
            else:
                row_dict["tags"] = [str(tags_val)]

            # Coerce imageBlob if present but not bytes
            if "imageBlob" in row_dict and row_dict["imageBlob"] is not None:
                if not isinstance(row_dict["imageBlob"], (bytes, bytearray)):
                    raise ValueError("imageBlob must be bytes if provided")

            article = ArticleModel(**row_dict)
            # Convert back to dict but ensure pubDate stays datetime (Arrow writer handles)
            validated.append(article.dict())
        except Exception as e:
            logger.warning(
                f"Invalid article data (ID: {row.get('id', 'unknown')}): {e}"
            )
            continue

    if not validated:
        logger.warning("No valid articles found after validation")
        return pd.DataFrame()
    return pd.DataFrame(validated)


def validate_raw_article_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw article DataFrame."""
    validated_articles = []

    for _, row in df.iterrows():
        try:
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                else:
                    row_dict[col] = val

            article = RawArticleModel(**row_dict)
            validated_articles.append(article.dict())
        except Exception as e:
            logger.warning(f"Invalid raw article data: {e}")
            continue

    if not validated_articles:
        logger.warning("No valid raw articles found after validation")
        return pd.DataFrame()

    return pd.DataFrame(validated_articles)


def get_schema_version() -> str:
    """Get current schema version for compatibility checks."""
    return "1.0.0"


def get_required_columns(schema_type: str = "article") -> List[str]:
    """Get required columns for a given schema type (unified naming)."""
    if schema_type == "article":
        return ["id", "title", "pubDate", "source"]
    if schema_type == "raw":
        return ["id", "title", "pubDate", "source"]
    return []


def get_optional_columns(schema_type: str = "article") -> List[str]:
    """Get optional columns for a given schema type."""
    if schema_type == "article":
        all_cols = list(ARTICLE_DTYPES.keys())
        required = get_required_columns("article")
        return [c for c in all_cols if c not in required]
    if schema_type == "raw":
        all_cols = list(RAW_ARTICLE_DTYPES.keys())
        required = get_required_columns("raw")
        return [c for c in all_cols if c not in required]
    return []
