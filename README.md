# news-mvp

Quick start:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
Copy-Item .env.example .env
news-mvp --dry-run
news-mvp bootstrap
python scripts\print_config.py

## Validator and optional dependencies

The project includes a lightweight CSV validator at `scripts/ge_validate.py` which performs a few basic checks (columns present, not-null, unique id, link regex, parseable pubDate).

To avoid installing heavy dependencies by default, Great Expectations is optional. Install the validation extras like this:

```powershell
pip install .[validation]
```

This will install `great_expectations` if you want to use it for more advanced workflows.

### Notes about the validator

- The validator expects columns: `id`, `title`, `link`, `pubDate`.
- If `link` is missing, it will accept `image` or `imageName` as a fallback and treat it as the `link` value (useful because the ETL writes image paths/urls into those columns).
- Exit codes: `0` = validation passed, `2` = validation failed.

### Running the validator

```powershell
py scripts\ge_validate.py data\master\master_news.csv
```

### Running tests

Install test dependencies and run:

```powershell
pip install pytest
py -m pytest -q
```

### Development Setup

For development, install development dependencies and set up pre-commit hooks:

```powershell
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

This will install all necessary development tools:
- pre-commit: For running hooks before commits
- pytest: For running tests
- ruff: For linting
- pyright: For type checking

If a hook modifies files (e.g., formatting or import cleanup), re-stage them:

```powershell
git add .
```

To update hook versions later:

```powershell
pre-commit autoupdate
```

## Scheduled ETL and repository variable

The scheduled ETL workflow runs on GitHub Actions (see `.github/workflows/etl_cron.yml`).
To control whether the scheduled ETL actually executes, create a repository Variable named
`ETL_RUN_ENABLED` (GitHub → Settings → Variables → Actions) and set its value to `true` to enable
scheduled runs, or `false` to keep the schedule present but skip execution.

The workflows perform a runtime check at the start of the ETL job and will exit early when the
variable is not set to `true`, which avoids relying on job-level `if:` expressions that can
trigger editor/CI diagnostics locally.

## Unified Schema & Storage Workflow

End-to-end column naming is unified across CSV → Parquet → DuckDB using the original feed/CSV
style (camel/mixed case) so no renaming layer is needed. Key fields:

Required (non-null): `id`, `title`, `pubDate`, `source`

Optional: `description`, `category`, `tags` (list), `creator`, `language`, `image`,
`imageCaption`, `imageCredit`, `imageName`, `imageBlob` (bytes), `guid`, `processed_at`, `batch_hour`.

`pubDate` is normalized to UTC and truncated to the hour for analytical grouping.
`imageBlob` is only populated if configured to store image bytes in the database.

### Parquet Generation

Parquet files are written with Arrow schema defined in `schemas.get_parquet_schema()`.
They can be ingested directly into DuckDB without column mapping.

### Database Initialization

Run:

```powershell
python scripts\init_database.py --env dev
```

Or via CLI:

```powershell
news-mvp storage init-db --env dev
```

### Loading Parquet Into DuckDB

```powershell
news-mvp storage load-parquet path\to\file.parquet --env dev
```

This performs an upsert on primary key `id`. Tags (if present) are stored as a DuckDB `VARCHAR[]`.

### Inspecting Stats

```powershell
news-mvp storage stats --env dev
```

### Configuration Flags (storage section)

Add to your `configs/dev.yaml` (values shown are defaults):

```yaml
storage:
	store_images_in_db: false       # If true, loads image bytes into imageBlob
	remove_parquet_after_load: false # If true, deletes parquet after successful load
	image_compression: null          # Placeholder for future (e.g., webp)
	db_path: data/db/news_mvp.duckdb # Database location
	schema_file: schema.sql          # Schema file path
```

### End-to-End Smoke Test

Run a lightweight workflow (will attempt to find a parquet file matching ingest pattern):

```powershell
python scripts\e2e_smoke.py --env dev
```

### Future Extensions

Planned additions include retention policies (removing parquet after load), image compression,
and an embeddings pipeline (vector index) building upon the stable unified schema.

