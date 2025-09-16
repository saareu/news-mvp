.RECIPEPREFIX = >
.PHONY: install run health clean

# Works on macOS/Linux; on Windows, use the PowerShell commands in README.
install:
> python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

run:
> . .venv/bin/activate && news-mvp --dry-run

health:
> . .venv/bin/activate && news-mvp health --dry-run

clean:
> rm -rf .venv __pycache__ .pytest_cache *.log data/* logs/* artifacts || true
