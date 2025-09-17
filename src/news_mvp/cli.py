import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import typer
from rich.console import Console
from dotenv import load_dotenv
from news_mvp.settings import Settings
from news_mvp.logging_setup import configure_logging, get_logger
from news_mvp.paths import Paths

console = Console()

app = typer.Typer()


@app.callback()
def main(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan actions without making changes."
    ),
):
    load_dotenv(override=False)
    ctx.ensure_object(dict)
    ctx.obj["DRY_RUN"] = dry_run
    cfg_path = os.getenv("NEWS_MVP_CONFIG", "configs/dev.yaml")
    settings = Settings.load(cfg_path)
    ctx.obj["SETTINGS"] = settings

    # Configure structured logging based on settings
    configure_logging(
        level=settings.logging.level,
        format_type=settings.logging.format,
        structured=settings.logging.structured,
    )

    if ctx.invoked_subcommand is None:
        console.print(
            f"[bold cyan]news-mvp[/] loaded config: {cfg_path}, dry_run={dry_run}"
        )
        console.print(ctx.obj["SETTINGS"])


def _effective_dry_run(ctx: typer.Context, subcmd_dry_run):
    # allow either: global --dry-run or subcommand --dry-run
    return bool(
        subcmd_dry_run if subcmd_dry_run is not None else ctx.obj.get("DRY_RUN", False)
    )


@app.command()
def health(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        None, "--dry-run", help="Plan actions without making changes."
    ),
):
    console.print({"ok": True, "dry_run": _effective_dry_run(ctx, dry_run)})


@app.command()
def bootstrap(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        None, "--dry-run", help="Plan actions without making changes."
    ),
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
    s = Settings.load(cfg_path)  # load settings for source/rss lookup

    # Configure logging with settings
    configure_logging(
        level=s.logging.level,
        format_type=s.logging.format,
        structured=s.logging.structured,
    )
    log = get_logger("etl.run")

    for p in Paths.ensure_all():
        log.info("Directory ensured", path=str(p))

    if dry_run:
        log.info("ETL run skipped", reason="dry-run", source=source)
        raise SystemExit(0)

    # Call the merged ETL API
    from news_mvp.etl.api import run_etl_for_source

    rss_url = rss or s.etl.sources[source].rss
    if not rss_url:
        raise typer.BadParameter(
            f"No RSS configured for source '{source}' and no --rss provided"
        )

    log.info("Starting ETL run", source=source, rss_url=rss_url)

    # Set environment variable for ETL subprocess
    import os

    os.environ["NEWS_MVP_CONFIG_ENV"] = env

    run_etl_for_source(source=source, rss_url=rss_url)


@etl_app.command("merge")
def etl_merge(
    env: str = typer.Option("dev", help="Config environment"),
    dry_run: bool = typer.Option(False, help="Do not write"),
):
    from pathlib import Path

    cfg_path = f"configs/{env}.yaml"
    s = Settings.load(cfg_path)

    configure_logging(
        level=s.logging.level,
        format_type=s.logging.format,
        structured=s.logging.structured,
    )
    log = get_logger("etl.merge")

    for p in Paths.ensure_all():
        log.info("Directory ensured", path=str(p))

    # Use config-driven data dir (from Paths utility)
    masters = sorted(Path(Paths.data_root(), "master").glob("master_*.csv"))

    # Support runtime.dry_run
    if getattr(s, "runtime", None) and (
        dry_run or getattr(s.runtime, "dry_run", False)
    ):
        log.info("Merge skipped", reason="dry-run", inputs=[str(p) for p in masters])
        raise SystemExit(0)

    from news_mvp.etl.api import merge_masters

    output_csv = str(Path(Paths.data_root(), "master", "master_news.csv"))
    log.info("Starting merge", inputs=[str(p) for p in masters], output=output_csv)
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

    configure_logging(
        level=s.logging.level,
        format_type=s.logging.format,
        structured=s.logging.structured,
    )
    log = get_logger("etl.run_all")

    for p in Paths.ensure_all():
        log.info("Directory ensured", path=str(p))

    if dry_run:
        log.info(
            "ETL run-all skipped", reason="dry-run", sources=list(s.etl.sources.keys())
        )
        raise SystemExit(0)

    from news_mvp.etl.api import run_etl_for_source

    log.info("Starting ETL for all sources", sources=list(s.etl.sources.keys()))
    for src in s.etl.sources:
        rss = s.etl.sources[src].rss
        if rss is None:
            raise typer.BadParameter(f"No RSS configured for source '{src}'")
        log.info("Processing source", source=src, rss_url=rss)
        run_etl_for_source(source=src, rss_url=rss)


@etl_app.command("parquetify")
def parquetify(source: str, env: str = "dev"):
    from pathlib import Path
    import polars as pl

    cfg_path = f"configs/{env}.yaml"
    s = Settings.load(cfg_path)
    # validate source exists in configuration if available
    if source not in s.etl.sources:
        raise typer.BadParameter(f"Unknown source '{source}'")
    # ensure directories (ignore returned paths)
    for _ in Paths.ensure_all():
        continue
    raw_dir = Path("data/raw") / source
    out_base = Path("data/raw") / f"source={source}" / "date="
    for csv in raw_dir.rglob("*.csv"):
        date = csv.stat().st_mtime
        date_str = (
            __import__("datetime").datetime.fromtimestamp(date).strftime("%Y-%m-%d")
        )
        out_dir = Path(f"{out_base}{date_str}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"part-{csv.stem}.parquet"
        pl.read_csv(csv).write_parquet(out)
        print(out)


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)

# --- Storage / DB sub-commands ---
storage_app = typer.Typer(help="Storage & database commands")
app.add_typer(storage_app, name="storage")


@storage_app.command("init-db")
def storage_init_db(
    ctx: typer.Context,
    env: str = typer.Option("dev", help="Config environment"),
    schema: str = typer.Option("schema.sql", help="Schema SQL file"),
):
    """Initialize DuckDB database using schema file."""
    from news_mvp.db import get_connection, init_schema, fetch_one, DBNotAvailable

    settings = Settings.load(f"configs/{env}.yaml")
    try:
        conn = get_connection(settings.storage.db_path)
        init_schema(conn, schema)
        row = fetch_one(conn, "SELECT COUNT(*) FROM articles") or (0,)
        console.print(
            f"[green]Database initialized[/] path={settings.storage.db_path} articles={row[0]}"
        )
    except DBNotAvailable as e:  # pragma: no cover
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)


@storage_app.command("load-parquet")
def storage_load_parquet(
    ctx: typer.Context,
    parquet: str = typer.Argument(..., help="Path to parquet file"),
    env: str = typer.Option("dev", help="Config environment"),
    store_images: bool = typer.Option(
        None,
        help="Override config: store image bytes in DB (imageBlob)",
    ),
    remove_after: bool = typer.Option(
        None,
        help="Override config: remove parquet file after successful load",
    ),
    compress_images: bool = typer.Option(
        False,
        help="Compress imageBlob bytes before storing (requires Pillow)",
    ),
    image_format: str = typer.Option(
        "webp", help="Image format used when --compress-images is set"
    ),
):
    """Load a parquet file (unified schema) into the articles table."""
    from news_mvp.db import get_connection, load_parquet_into_articles, DBNotAvailable

    settings = Settings.load(f"configs/{env}.yaml")
    if store_images is None:
        store_images = settings.storage.store_images_in_db
    if remove_after is None:
        remove_after = settings.storage.remove_parquet_after_load
    try:
        conn = get_connection(settings.storage.db_path)
        inserted = load_parquet_into_articles(
            conn,
            parquet,
            store_images=store_images,
            remove_after=bool(remove_after),
            compress_images=bool(compress_images),
            image_format=image_format,
        )
        console.print(
            f"[green]Loaded parquet[/] file={parquet} rows={inserted} images={'yes' if store_images else 'no'} removed={'yes' if remove_after else 'no'}"
        )
    except DBNotAvailable as e:  # pragma: no cover
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)


@storage_app.command("stats")
def storage_stats(
    ctx: typer.Context,
    env: str = typer.Option("dev", help="Config environment"),
):
    """Show basic database statistics."""
    from news_mvp.db import get_connection, fetch_one, DBNotAvailable

    settings = Settings.load(f"configs/{env}.yaml")
    try:
        conn = get_connection(settings.storage.db_path)
        total_articles = fetch_one(conn, "SELECT COUNT(*) FROM articles") or (0,)
        sources = fetch_one(conn, "SELECT COUNT(*) FROM sources") or (0,)
        console.print(
            {
                "db_path": settings.storage.db_path,
                "articles": total_articles[0],
                "sources": sources[0],
            }
        )
    except DBNotAvailable as e:  # pragma: no cover
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)


@storage_app.command("retention-cleanup")
def storage_retention_cleanup(
    ctx: typer.Context,
    env: str = typer.Option("dev", help="Config environment"),
    root: str = typer.Option(
        None, help="Root directory to clean (defaults to data root)"
    ),
    pattern: str = typer.Option("**/*.parquet", help="Glob pattern for parquet files"),
    days: int = typer.Option(
        None, help="Retention in days; defaults to settings.storage.retention_days"
    ),
):
    """Delete parquet files older than retention threshold."""
    from news_mvp.db import cleanup_old_parquet

    settings = Settings.load(f"configs/{env}.yaml")
    root_path = root or str(Paths.data_root())
    retention_days = (
        days if days is not None else (settings.storage.retention_days or 0)
    )
    if retention_days <= 0:
        console.print("[yellow]Retention disabled or not set[/]")
        raise typer.Exit(code=0)
    deleted = cleanup_old_parquet(
        root_path, pattern=pattern, retention_days=retention_days
    )
    console.print({"deleted": len(deleted)})
