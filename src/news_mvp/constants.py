"""Central constants for the news_mvp project."""

"""Central constants for the news_mvp project."""
# Default to 'utf-8-sig' for file-level encoding so CSVs opened in Excel
# on Windows show correctly (BOM present). Per-field reversible tokens
# (like `id_seed`) are ASCII-only percent-encoded strings and therefore
# safe regardless of the file-level BOM.
CSV_ENCODING = "utf-8-sig"
