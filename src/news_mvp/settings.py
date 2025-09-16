from __future__ import annotations
from pathlib import Path
import yaml

class Settings(dict):
    @staticmethod
    def load(path: str | None = None) -> "Settings":
        base = {}
        with open("configs/base.yaml", "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                overlay = yaml.safe_load(f) or {}
            base = deep_merge(base, overlay)
        return Settings(base)

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
