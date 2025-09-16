#!/usr/bin/env python3
"""Print the mapping/selectors paths as computed by the ETL orchestrator.

This mirrors the logic in `src/news_mvp/etl/pipelines/etl_by_source.py` so we can
confirm what absolute paths will be passed to subprocesses in CI.
"""
import os
from pathlib import Path

# Ensure PROD config environment as in the CI run
os.environ.setdefault("NEWS_MVP_CONFIG_ENV", "prod")

from news_mvp.paths import Paths
from news_mvp.settings import Settings

project_root = Paths.root().resolve()
env = os.environ.get("NEWS_MVP_CONFIG_ENV", "dev")
cfg_path = f"configs/{env}.yaml"
print(f"Using config env: {env} (cfg: {cfg_path})")

s = Settings.load(cfg_path)

raw_mapping = (
    str(s.etl.etl_schema.mapping_csv)
    if hasattr(s.etl, "etl_schema")
    else "src/news_mvp/etl/schema/mapping.csv"
)
raw_selectors = (
    str(s.etl.etl_schema.selectors_csv)
    if hasattr(s.etl, "etl_schema")
    else "src/news_mvp/etl/schema/selectors.csv"
)


# Resolve relative paths against project root
def _abs_path(p: str) -> str:
    pth = Path(p)
    return str((project_root / p).resolve()) if not pth.is_absolute() else str(pth)


mapping_path = _abs_path(raw_mapping)
selectors_path = _abs_path(raw_selectors)

print("project_root:", project_root)
print("raw_mapping:", raw_mapping)
print("mapping_path:", mapping_path)
print("mapping exists:", Path(mapping_path).exists())
print("raw_selectors:", raw_selectors)
print("selectors_path:", selectors_path)
print("selectors exists:", Path(selectors_path).exists())
