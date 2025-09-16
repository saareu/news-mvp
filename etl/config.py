
"""
ETL Config
----------
All configuration for the ETL pipeline is centralized here.

Override mechanism:
	- By default, config is loaded from this file.
	- If the environment variable NEWS_ETL_CONFIG is set to a YAML file, values from that file override defaults.
	- You can also call `load_config(path)` to override programmatically.

Config options:
	BASE_DIR:         Project root directory
	DATA_DIR:         Data directory
	RAW_DIR:          Raw data directory
	CANON_DIR:        Canonical data directory
	MASTER_DIR:       Master data directory
	MASTER_*_CSV:     Per-source master CSVs
	MASTER_NEWS_CSV:  Unified master CSV
	*_RSS_URL:        RSS URLs for each source
	*_ITEMS_CSV/JSON: Default filenames for each source

Best practice: Do not hardcode paths or URLs elsewhere. Always import from this module.
"""

from __future__ import annotations
from pathlib import Path
import os
import yaml

def _default_base_dir():
	return Path(__file__).resolve().parents[1]

def _load_yaml_config(path):
	try:
		with open(path, 'r', encoding='utf-8') as f:
			return yaml.safe_load(f)
	except Exception:
		return {}

# Load config from YAML if env var is set
_yaml_config = {}
_yaml_path = os.environ.get('NEWS_ETL_CONFIG')
if _yaml_path:
	_yaml_config = _load_yaml_config(_yaml_path)

def _get(key, default):
	return _yaml_config.get(key, default)

BASE_DIR = Path(_get('BASE_DIR', _default_base_dir()))
DATA_DIR = Path(_get('DATA_DIR', BASE_DIR / "data"))
RAW_DIR = Path(_get('RAW_DIR', DATA_DIR / "raw"))
RAW_YNET_DIR = Path(_get('RAW_YNET_DIR', RAW_DIR / "ynet"))
RAW_HAYOM_DIR = Path(_get('RAW_HAYOM_DIR', RAW_DIR / "hayom"))
RAW_HAARETZ_DIR = Path(_get('RAW_HAARETZ_DIR', RAW_DIR / "haaretz"))
CANON_DIR = Path(_get('CANON_DIR', DATA_DIR / "canonical"))
CANON_YNET_DIR = Path(_get('CANON_YNET_DIR', CANON_DIR / "ynet"))
CANON_HAYOM_DIR = Path(_get('CANON_HAYOM_DIR', CANON_DIR / "hayom"))
CANON_HAARETZ_DIR = Path(_get('CANON_HAARETZ_DIR', CANON_DIR / "haaretz"))
MASTER_DIR = Path(_get('MASTER_DIR', DATA_DIR / "master"))

DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
RAW_YNET_DIR    = RAW_DIR / "ynet"
RAW_HAYOM_DIR   = RAW_DIR / "hayom"
RAW_HAARETZ_DIR = RAW_DIR / "haaretz"

CANON_DIR       = DATA_DIR / "canonical"
CANON_YNET_DIR  = CANON_DIR / "ynet"
CANON_HAYOM_DIR = CANON_DIR / "hayom"
CANON_HAARETZ_DIR = CANON_DIR / "haaretz"

MASTER_DIR      = DATA_DIR / "master"

# Source-specific master files
MASTER_YNET_CSV    = Path(_get('MASTER_YNET_CSV', MASTER_DIR / "master_ynet.csv"))
MASTER_YNET_JSON   = Path(_get('MASTER_YNET_JSON', MASTER_DIR / "master_ynet.json"))
MASTER_HAYOM_CSV   = Path(_get('MASTER_HAYOM_CSV', MASTER_DIR / "master_hayom.csv"))
MASTER_HAYOM_JSON  = Path(_get('MASTER_HAYOM_JSON', MASTER_DIR / "master_hayom.json"))
MASTER_HAARETZ_CSV = Path(_get('MASTER_HAARETZ_CSV', MASTER_DIR / "master_haaretz.csv"))
MASTER_HAARETZ_JSON = Path(_get('MASTER_HAARETZ_JSON', MASTER_DIR / "master_haaretz.json"))
MASTER_NEWS_CSV = Path(_get('MASTER_NEWS_CSV', MASTER_DIR / "master_news.csv"))

# Default filenames (override with CLI flags any time)
YNET_ITEMS_JSON    = Path(_get('YNET_ITEMS_JSON', RAW_YNET_DIR / "ynet_items.json"))
YNET_ITEMS_CSV     = Path(_get('YNET_ITEMS_CSV', RAW_YNET_DIR / "ynet_items.csv"))
YNET_CANON_JSON    = Path(_get('YNET_CANON_JSON', CANON_YNET_DIR / "ynet_canonical.json"))
YNET_CANON_CSV     = Path(_get('YNET_CANON_CSV', CANON_YNET_DIR / "ynet_canonical.csv"))

HAYOM_ITEMS_JSON   = Path(_get('HAYOM_ITEMS_JSON', RAW_HAYOM_DIR / "hayom_items.json"))
HAYOM_ITEMS_CSV    = Path(_get('HAYOM_ITEMS_CSV', RAW_HAYOM_DIR / "hayom_items.csv"))
HAYOM_CANON_CSV    = Path(_get('HAYOM_CANON_CSV', CANON_HAYOM_DIR / "hayom_canonical.csv"))
HAYOM_CANON_JSON   = Path(_get('HAYOM_CANON_JSON', CANON_HAYOM_DIR / "hayom_canonical.json"))

HAARETZ_ITEMS_JSON = Path(_get('HAARETZ_ITEMS_JSON', RAW_HAARETZ_DIR / "haaretz_items.json"))
HAARETZ_ITEMS_CSV  = Path(_get('HAARETZ_ITEMS_CSV', RAW_HAARETZ_DIR / "haaretz_items.csv"))
HAARETZ_CANON_CSV  = Path(_get('HAARETZ_CANON_CSV', CANON_HAARETZ_DIR / "haaretz_canonical.csv"))
HAARETZ_CANON_JSON = Path(_get('HAARETZ_CANON_JSON', CANON_HAARETZ_DIR / "haaretz_canonical.json"))

# RSS URLs for each news source
YNET_RSS_URL     = _get('YNET_RSS_URL', "https://www.ynet.co.il/Integration/StoryRss2.xml")
HAYOM_RSS_URL    = _get('HAYOM_RSS_URL', "https://www.israelhayom.co.il/rss.xml")
HAARETZ_RSS_URL  = _get('HAARETZ_RSS_URL', "https://www.haaretz.co.il/srv/htz---all-articles")

# Optional: programmatic override
def load_config(yaml_path):
	global _yaml_config
	_yaml_config = _load_yaml_config(yaml_path)


# Note: Instead of creating directories at import time,
# we now use etl.utils.timestamp_manager.ensure_directories() to create all
# necessary directories when needed
