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
    """Get the standard Parquet schema for articles."""
    return pa.schema(
        [
            ("article_id", pa.string()),
            ("title", pa.string()),
            ("description", pa.string()),
            ("category", pa.string()),
            ("pub_date", pa.timestamp("ns")),
            ("tags", pa.list_(pa.string())),
            ("creator", pa.string()),
            ("source", pa.string()),
            ("language", pa.string()),
            ("image_path", pa.string()),
            ("image_caption", pa.string()),
            ("image_credit", pa.string()),
            ("image_name", pa.string()),
            ("guid", pa.string()),  # RSS feed unique identifier
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
    "article_id": "string",
    "title": "string",
    "description": "string",
    "category": "string",
    "pub_date": "datetime64[ns, UTC]",
    "tags": "object",  # List of strings
    "creator": "string",
    "source": "string",
    "language": "string",
    "image_path": "string",
    "image_caption": "string",
    "image_credit": "string",
    "image_name": "string",
    "guid": "string",  # RSS feed unique identifier
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
    """Pydantic model for article validation."""

    article_id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article title")
    description: Optional[str] = Field(None, description="Article description")
    category: Optional[str] = Field(None, description="Article category")
    pub_date: datetime = Field(..., description="Publication date")
    tags: List[str] = Field(default_factory=list, description="Article tags")
    creator: Optional[str] = Field(None, description="Article creator")
    source: str = Field(..., description="Data source")
    language: str = Field("he", description="Article language")
    image_path: Optional[str] = Field(None, description="Path to article image")
    image_caption: Optional[str] = Field(None, description="Image caption")
    image_credit: Optional[str] = Field(None, description="Image credit")
    image_name: Optional[str] = Field(None, description="Image filename")
    guid: Optional[str] = Field(None, description="RSS feed unique identifier")
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Processing timestamp"
    )
    batch_hour: int = Field(
        default_factory=lambda: datetime.now().hour, description="Processing hour"
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
        "guid",  # RSS feed unique identifier
    ],
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
        "image_path",
        "image_caption",
        "image_credit",
        "image_name",
        "guid",  # RSS feed unique identifier
    ],
}


# ===== UTILITY FUNCTIONS =====


def validate_article_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean article DataFrame using Pydantic models."""
    validated_articles = []

    for _, row in df.iterrows():
        try:
            # Convert row to dict and handle NaN values
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                else:
                    row_dict[col] = val

            article = ArticleModel(**row_dict)
            validated_articles.append(article.dict())
        except Exception as e:
            logger.warning(
                f"Invalid article data (ID: {row.get('article_id', 'unknown')}): {e}"
            )
            continue

    if not validated_articles:
        logger.warning("No valid articles found after validation")
        return pd.DataFrame()

    return pd.DataFrame(validated_articles)


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
    """Get required columns for a given schema type."""
    if schema_type == "article":
        return ["article_id", "title", "pub_date", "source"]
    elif schema_type == "raw":
        return ["id", "title", "pubDate", "source"]
    else:
        return []


def get_optional_columns(schema_type: str = "article") -> List[str]:
    """Get optional columns for a given schema type."""
    if schema_type == "article":
        all_cols = list(ARTICLE_DTYPES.keys())
        required = get_required_columns("article")
        return [col for col in all_cols if col not in required]
    elif schema_type == "raw":
        all_cols = list(RAW_ARTICLE_DTYPES.keys())
        required = get_required_columns("raw")
        return [col for col in all_cols if col not in required]
    else:
        return []
