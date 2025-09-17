"""ETL-specific schemas for the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    nullable: bool
    dtype: str
    description: str = ""


class Stage(str, Enum):
    ETL_BEFORE_MERGE = "etl_stage_before_merge"


# Master-level schema (etl_stage_before_merge)
MASTER_SCHEMA_ETL_BEFORE_MERGE: List[ColumnSpec] = [
    ColumnSpec(
        "article_id", False, "str", "Deterministic ID from title|pub_date|source"
    ),
    ColumnSpec("guid", False, "str", "Source GUID/URL"),
    ColumnSpec("title", False, "str", "Headline/title"),
    ColumnSpec("description", True, "str", "Summary/lead text"),
    ColumnSpec("category", True, "str", "Category/section"),
    ColumnSpec("image", True, "str", "Saved image file name (id + ext)"),
    ColumnSpec("image_caption", True, "str", "Image caption"),
    ColumnSpec("image_credit", True, "str", "Image credit/attribution"),
    ColumnSpec(
        "pub_date", False, "str", "Canonical publication time (ISO/RFC as string)"
    ),
    ColumnSpec("tags", True, "str", "Comma- or pipe-separated tags"),
    ColumnSpec("author", True, "str", "Author/Byline"),
    ColumnSpec("source", False, "str", "Source name"),
    ColumnSpec("language", False, "str", "ISO language code"),
    ColumnSpec("image_name", True, "str", "Remote original image basename (no ext)"),
    ColumnSpec("fetching_time", False, "str", "When the feed/item was fetched"),
]


_SCHEMA_REGISTRY: Dict[Stage, List[ColumnSpec]] = {
    Stage.ETL_BEFORE_MERGE: MASTER_SCHEMA_ETL_BEFORE_MERGE,
}


def get_schema(stage: Stage) -> List[ColumnSpec]:
    return _SCHEMA_REGISTRY[stage]


def schema_fieldnames(stage: Stage) -> List[str]:
    return [c.name for c in get_schema(stage)]


def schema_required(stage: Stage) -> Tuple[str, ...]:
    return tuple(c.name for c in get_schema(stage) if not c.nullable)


def schema_dtypes(stage: Stage) -> Dict[str, str]:
    return {c.name: c.dtype for c in get_schema(stage)}
