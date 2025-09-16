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

Install test deps and run:

```powershell
pip install pytest
py -m pytest -q
```

## Scheduled ETL and repository variable

The scheduled ETL workflow runs on GitHub Actions (see `.github/workflows/etl_cron.yml`).
To control whether the scheduled ETL actually executes, create a repository Variable named
`ETL_RUN_ENABLED` (GitHub → Settings → Variables → Actions) and set its value to `true` to enable
scheduled runs, or `false` to keep the schedule present but skip execution.

The workflows perform a runtime check at the start of the ETL job and will exit early when the
variable is not set to `true`, which avoids relying on job-level `if:` expressions that can
trigger editor/CI diagnostics locally.

