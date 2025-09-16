import os, sys, click
from rich.console import Console
from dotenv import load_dotenv
from news_mvp.settings import Settings
from news_mvp.logging_setup import get_logger
from news_mvp.paths import Paths

console = Console()
log = get_logger("news_mvp")

@click.group(invoke_without_command=True)
@click.option("--dry-run", is_flag=True, help="Plan actions without making changes.")
@click.pass_context
def app(ctx, dry_run):
    load_dotenv(override=False)
    ctx.ensure_object(dict)
    ctx.obj["DRY_RUN"] = dry_run
    cfg_path = os.getenv("NEWS_MVP_CONFIG", "configs/dev.yaml")
    ctx.obj["SETTINGS"] = Settings.load(cfg_path)
    if ctx.invoked_subcommand is None:
        console.print(f"[bold cyan]news-mvp[/] loaded config: {cfg_path}, dry_run={dry_run}")
        console.print(ctx.obj["SETTINGS"])

def _effective_dry_run(ctx, subcmd_dry_run):
    # allow either: global --dry-run or subcommand --dry-run
    return bool(subcmd_dry_run if subcmd_dry_run is not None else ctx.obj.get("DRY_RUN", False))

@app.command()
@click.option("--dry-run", is_flag=True, default=None, help="Plan actions without making changes.")
@click.pass_context
def health(ctx, dry_run):
    dr = _effective_dry_run(ctx, dry_run)
    console.print({"ok": True, "dry_run": dr})

@app.command()
@click.option("--dry-run", is_flag=True, default=None, help="Plan actions without making changes.")
@click.pass_context
def bootstrap(ctx, dry_run):
    dr = _effective_dry_run(ctx, dry_run)
    # no writes beyond ensuring dirs; respects dry-run conceptually if you add write ops later
    for p in Paths.ensure_all():
        console.print(f"ensured: {p}")

if __name__ == "__main__":
    try:
        app(obj={})
    except KeyboardInterrupt:
        sys.exit(130)
