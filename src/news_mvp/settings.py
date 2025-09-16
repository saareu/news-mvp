# src/news_mvp/settings.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import os
from types import SimpleNamespace
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
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        with open(path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        merged = Settings._deep_update(base, override or {})
        return Settings(**merged)


def load_settings(env: str) -> SimpleNamespace:
    """Compatibility loader used by tests. Returns a simple namespace with
    at least `.app.name` and `.paths.data_dir` to keep older tests working.
    """
    cfg_path = f"configs/{env}.yaml"
    base_path = os.path.join(os.path.dirname(cfg_path), "base.yaml")
    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    merged = Settings._deep_update(base, override or {})

    app = merged.get("app", {}) or {}
    # ensure a name exists for tests
    app.setdefault("name", "news-mvp")

    paths_ns = SimpleNamespace(data_dir=str(Paths.data_root()))

    result = SimpleNamespace()
    result.app = SimpleNamespace(**app)
    result.paths = paths_ns
    # copy other top-level keys for convenience
    for k, v in merged.items():
        if k in ("app", "paths"):
            continue
        setattr(result, k, v)

    return result
