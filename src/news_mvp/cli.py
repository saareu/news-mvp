import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import typer
from rich.console import Console
from dotenv import load_dotenv
from news_mvp.settings import Settings
from news_mvp.logging_setup import get_logger
from news_mvp.paths import Paths

console = Console()
log = get_logger("news_mvp")

app = typer.Typer()

@app.callback()
def main(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan actions without making changes."),
):
    load_dotenv(override=False)
    ctx.ensure_object(dict)
    ctx.obj["DRY_RUN"] = dry_run
    cfg_path = os.getenv("NEWS_MVP_CONFIG", "configs/dev.yaml")
    ctx.obj["SETTINGS"] = Settings.load(cfg_path)
    if ctx.invoked_subcommand is None:
        console.print(f"[bold cyan]news-mvp[/] loaded config: {cfg_path}, dry_run={dry_run}")
        console.print(ctx.obj["SETTINGS"])

def _effective_dry_run(ctx: typer.Context, subcmd_dry_run):
    # allow either: global --dry-run or subcommand --dry-run
    return bool(subcmd_dry_run if subcmd_dry_run is not None else ctx.obj.get("DRY_RUN", False))

@app.command()
def health(
    ctx: typer.Context,
    dry_run: bool = typer.Option(None, "--dry-run", help="Plan actions without making changes."),
):
    console.print({"ok": True, "dry_run": _effective_dry_run(ctx, dry_run)})

@app.command()
def bootstrap(
    ctx: typer.Context,
    dry_run: bool = typer.Option(None, "--dry-run", help="Plan actions without making changes."),
):
    for p in Paths.ensure_all():
        console.print(f"ensured: {p}")

# --- ETL sub-commands ---
etl_app = typer.Typer(help="ETL commands")
app.add_typer(etl_app, name="etl")

@etl_app.command("run")
def etl_run(
    source: str = typer.Option(..., "--source", help="Source key, e.g., ynet"),
    rss: str | None = typer.Option(None, help="Override RSS URL"),
    env: str = typer.Option("dev", help="Config environment"),
    dry_run: bool = typer.Option(False, help="Do not touch network/filesystem"),
):
    cfg_path = f"configs/{env}.yaml"
    s = Settings.load(cfg_path)
    log = get_logger()
    for p in Paths.ensure_all():
        log.info(f"ensured: {p}")
    if dry_run:
        log.info("etl.skip", extra={"reason": "dry-run", "source": source})
        raise SystemExit(0)
    # Call the merged ETL API
    from news_mvp.etl.api import run_etl_for_source
    rss_url = rss or s.etl.sources[source].rss
    if not rss_url:
        raise typer.BadParameter(f"No RSS configured for source '{source}' and no --rss provided")
    run_etl_for_source(source=source, rss_url=rss_url)

@etl_app.command("merge")
def etl_merge(
    env: str = typer.Option("dev", help="Config environment"),
    dry_run: bool = typer.Option(False, help="Do not write"),
):
    from pathlib import Path
    cfg_path = f"configs/{env}.yaml"
    s = Settings.load(cfg_path)
    log = get_logger()
    for p in Paths.ensure_all():
        log.info(f"ensured: {p}")
    # Use config-driven data dir (from Paths utility)
    masters = sorted(Path(Paths.data_root(), "master").glob("master_*.csv"))
    # Support runtime.dry_run
    if getattr(s, 'runtime', None) and (dry_run or getattr(s.runtime, 'dry_run', False)):
        log.info("merge.skip", extra={"reason": "dry-run", "inputs": [str(p) for p in masters]})
        raise SystemExit(0)
    from news_mvp.etl.api import merge_masters
    output_csv = str(Path(Paths.data_root(), "master", "master_news.csv"))
    merge_masters([str(p) for p in masters], output_csv=output_csv)

@etl_app.command("list-sources")
def etl_list_sources(env: str = "dev"):
    cfg_path = f"configs/{env}.yaml"
    s = Settings.load(cfg_path)
    for name, cfg in s.etl.sources.items():
        print(f"{name}\t{cfg.rss or '-'}\tforce_tz_offset={cfg.force_tz_offset}")

@etl_app.command("run-all")
def etl_run_all(env: str = "dev", dry_run: bool = False):
    cfg_path = f"configs/{env}.yaml"
    s = Settings.load(cfg_path)
    log = get_logger()
    for p in Paths.ensure_all():
        log.info(f"ensured: {p}")
    if dry_run:
        log.info("etl.skip", extra={"reason": "dry-run", "sources": list(s.etl.sources.keys())})
        raise SystemExit(0)
    from news_mvp.etl.api import run_etl_for_source
    for src in s.etl.sources:
        rss = s.etl.sources[src].rss
        if rss is None:
            raise typer.BadParameter(f"No RSS configured for source '{src}'")
        run_etl_for_source(source=src, rss_url=rss)


@etl_app.command("parquetify")
def parquetify(source: str, env: str = "dev"):
    from pathlib import Path
    import polars as pl
    from news_mvp.settings import Settings
    from news_mvp.paths import Paths
    cfg_path = f"configs/{env}.yaml"
    s = Settings.load(cfg_path)
    for _ in Paths.ensure_all():
        pass
    raw_dir = Path("data/raw")/source
    out_base = Path("data/raw") / f"source={source}" / "date="
    for csv in raw_dir.rglob("*.csv"):
        date = csv.stat().st_mtime
        date_str = __import__("datetime").datetime.fromtimestamp(date).strftime("%Y-%m-%d")
        out_dir = Path(f"{out_base}{date_str}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir/f"part-{csv.stem}.parquet"
        pl.read_csv(csv).write_parquet(out)
        print(out)

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
