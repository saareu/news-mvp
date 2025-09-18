# src/news_mvp/settings.py
from functools import lru_cache
import os
from types import SimpleNamespace
import re

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from news_mvp.constants import CSV_ENCODING
from news_mvp.paths import Paths


class RuntimeConfig(BaseModel):
    dry_run: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"  # "json" or "human"
    structured: bool = True


class IngestConfig(BaseModel):
    input_glob: str = "data/raw/source=*/date=*/part-*.parquet"
    batch_size: int = 1000
    parallel_workers: int = 2


# … existing models …
class EtlSource(BaseModel):
    rss: str | None = None
    force_tz_offset: int | None = None
    strip_tags: bool | None = None
    strip_authors: bool | None = None


class EtlSchema(BaseModel):
    mapping_csv: str
    selectors_csv: str


class EtlOutput(BaseModel):
    raw_pattern: str
    canonical_dir: str
    master_dir: str
    images_dir: str


class EtlBehavior(BaseModel):
    download_images: bool = True
    merge_after_all_sources: bool = True


class EtlCfg(BaseModel):
    sources: dict[str, EtlSource]
    etl_schema: EtlSchema
    output: EtlOutput
    behavior: EtlBehavior


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")
    etl: EtlCfg
    runtime: RuntimeConfig = RuntimeConfig()
    logging: LoggingConfig = LoggingConfig()
    ingest: IngestConfig = IngestConfig()
    # CSV encoding used across the project (defaults to utf-8-sig to handle BOM)
    csv_encoding: str = CSV_ENCODING

    @staticmethod
    def _deep_update(d: dict, u: dict) -> dict:
        # Recursively update dict d with values from u
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = Settings._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    @staticmethod
    def load(path: str) -> "Settings":
        base_path = os.path.join(os.path.dirname(path), "base.yaml")
        with open(base_path, "r", encoding=CSV_ENCODING) as f:
            base = yaml.safe_load(f)
        with open(path, "r", encoding=CSV_ENCODING) as f:
            override = yaml.safe_load(f)
        merged = Settings._deep_update(base, override or {})
        # Ensure csv_encoding from the merged config is honoured
        if merged.get("csv_encoding"):
            merged.setdefault("csv_encoding", merged.get("csv_encoding"))
        return Settings(**merged)

    # Convenience instance method to get schema for a given stage.
    def get_schema_for_stage(self, stage) -> list:
        """Return the schema (List[ColumnSpec]) for the given stage.

        Accepts either a `news_mvp.schemas.Stage` enum or a string matching the
        enum value. The actual schema lookup is delegated to
        `news_mvp.schemas.get_schema` to keep definitions in one place and avoid
        duplication. Importing `news_mvp.schemas` is done locally to avoid
        potential import cycles when settings are constructed early in the
        application lifecycle.
        """
        # Local import to avoid import cycles
        from news_mvp.schemas import Stage as SchemaStage
        from news_mvp.schemas import get_schema as _get_schema

        if isinstance(stage, str):
            try:
                stage = SchemaStage(stage)
            except Exception:
                # Try to accept short names like 'etl_before_merge'
                mapping = {s.name: s for s in SchemaStage}
                if stage in mapping:
                    stage = mapping[stage]
                else:
                    raise

        if not isinstance(stage, SchemaStage):
            raise TypeError(
                "stage must be a news_mvp.schemas.Stage or a valid string value"
            )

        return _get_schema(stage)


def load_settings(env: str) -> SimpleNamespace:
    """Compatibility loader used by tests. Returns a simple namespace with
    at least `.app.name` and `.paths.data_dir` to keep older tests working.
    """
    cfg_path = f"configs/{env}.yaml"
    base_path = os.path.join(os.path.dirname(cfg_path), "base.yaml")
    with open(base_path, "r", encoding=CSV_ENCODING) as f:
        base = yaml.safe_load(f)
    with open(cfg_path, "r", encoding=CSV_ENCODING) as f:
        override = yaml.safe_load(f)
    merged = Settings._deep_update(base, override or {})
    # expose csv_encoding on the result for backward compatibility
    csv_enc = merged.get("csv_encoding", CSV_ENCODING)

    app = merged.get("app", {}) or {}
    # ensure a name exists for tests
    app.setdefault("name", "news-mvp")

    paths_ns = SimpleNamespace(data_dir=str(Paths.data_root()))

    result = SimpleNamespace()
    result.app = SimpleNamespace(**app)
    result.paths = paths_ns
    result.csv_encoding = csv_enc
    # copy other top-level keys for convenience
    for k, v in merged.items():
        if k in ("app", "paths"):
            continue
        setattr(result, k, v)

    return result


@lru_cache(maxsize=8)
def get_runtime_csv_encoding(env: str | None = None) -> str:
    """Return the CSV encoding configured for the given env (or default env).

    This is a small cached helper so other modules can cheaply obtain the
    configured CSV encoding without directly importing the constant or
    repeatedly parsing YAML files.
    """
    env = env or os.environ.get("NEWS_MVP_ENV", "dev")
    cfg = load_settings(env)
    return getattr(cfg, "csv_encoding", CSV_ENCODING)


def get_schema_for_stage(stage) -> list:
    """Module-level convenience wrapper to return schema for a given stage.

    This mirrors `Settings.get_schema_for_stage` but is available without
    creating a `Settings` object. `stage` may be a `news_mvp.schemas.Stage` or
    a string matching the enum value or enum name.
    """
    # Local import to avoid import cycles
    from news_mvp.schemas import Stage as SchemaStage
    from news_mvp.schemas import get_schema as _get_schema

    if isinstance(stage, str):
        try:
            stage = SchemaStage(stage)
        except Exception:
            mapping = {s.name: s for s in SchemaStage}
            if stage in mapping:
                stage = mapping[stage]
            else:
                raise

    if not isinstance(stage, SchemaStage):
        raise TypeError(
            "stage must be a news_mvp.schemas.Stage or a valid string value"
        )

    return _get_schema(stage)


def get_schema_fieldnames(stage) -> list[str]:
    """Return the list of fieldnames for the given stage schema."""
    from news_mvp.schemas import schema_fieldnames as _schema_fieldnames

    return _schema_fieldnames(stage)


def get_schema_required(stage) -> tuple[str, ...]:
    """Return the tuple of required (non-nullable) field names for the given stage.

    This is a thin wrapper around `news_mvp.schemas.schema_required` to make the
    API available from `news_mvp.settings` without importing the schemas
    module at program start in callers.
    """
    from news_mvp.schemas import schema_required as _schema_required

    return _schema_required(stage)


def get_image_fieldname(stage) -> str:
    """Return the canonical field name used for images in the given stage.

    Looks up the stage fieldnames and returns the first name matching 'image'
    (case-insensitive). Raises a RuntimeError if no image field is present.
    """
    # Local import to avoid cycles
    names = get_schema_fieldnames(stage)
    for n in names:
        if n.lower() == "image":
            return n
    raise RuntimeError(f"No image field found for schema stage {stage}")


def get_tags_fieldname(stage) -> str:
    """Return the canonical field name used for tags in the given stage.

    Looks up the stage fieldnames and returns the first name matching 'tags'
    (case-insensitive). Raises a RuntimeError if no tags field is present.
    """
    names = get_schema_fieldnames(stage)
    for n in names:
        if n.lower() == "tags":
            return n
    raise RuntimeError(f"No tags field found for schema stage {stage}")


def _normalize_name(s: str) -> str:
    """Normalize a field name for tolerant comparisons: lower-case and strip
    non-alphanumeric characters. This makes matching robust to snake/camel
    differences and small variations in naming in the schema definition.
    """
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def get_canonical_fieldname(logical_name: str, stage) -> str:
    """Return the canonical schema field name that corresponds to the
    logical name provided (case-insensitive and tolerant to separators).

    Example: `get_canonical_fieldname('creator', Stage.ETL_BEFORE_MERGE)` will
    return the actual fieldname defined in the schema such as "creator" or
    "author" if the schema used that name.

    Raises RuntimeError if no matching field is found.
    """
    if not logical_name:
        raise ValueError("logical_name must be provided")

    names = get_schema_fieldnames(stage)
    target = _normalize_name(logical_name)
    # small alias mapping for common synonyms
    aliases: dict[str, list[str]] = {
        "pub_date": ["pubdate", "published", "publicationdate"],
    }
    # First try exact lower-case match
    for n in names:
        if n.lower() == logical_name.lower():
            return n
    # Then try normalized match
    for n in names:
        if _normalize_name(n) == target:
            return n

    # Try aliases (e.g. creator <-> author)
    if logical_name.lower() in aliases:
        for alt in aliases[logical_name.lower()]:
            alt_target = _normalize_name(alt)
            for n in names:
                if _normalize_name(n) == alt_target:
                    return n

    # No match
    raise RuntimeError(
        f"No field matching logical name '{logical_name}' for schema stage {stage}"
    )


def get_author_fieldname(stage) -> str:
    """Return the canonical fieldname used for the author for the
    given schema stage."""
    return get_canonical_fieldname("author", stage)


def get_category_fieldname(stage) -> str:
    """Return the canonical fieldname used for category in the given stage."""
    return get_canonical_fieldname("category", stage)


def get_description_fieldname(stage) -> str:
    """Return the canonical fieldname used for description in the given stage."""
    return get_canonical_fieldname("description", stage)


def get_title_fieldname(stage) -> str:
    """Return the canonical fieldname used for title in the given stage."""
    return get_canonical_fieldname("title", stage)


def get_imagecaption_fieldname(stage) -> str:
    """Return the canonical fieldname used for image caption in the given stage."""
    return get_canonical_fieldname("imagecaption", stage)


def get_image_name_fieldname(stage) -> str:
    """Return the canonical fieldname used for the schema's image name column."""
    return get_canonical_fieldname("image_name", stage)


def get_image_credit_fieldname(stage) -> str:
    """Return the canonical fieldname used for the schema's image credit column."""
    return get_canonical_fieldname("image_credit", stage)
