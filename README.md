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

