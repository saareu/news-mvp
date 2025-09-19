"""
Orchestrate the ETL pipeline for a single source.


Workflow per source:
    extract_by_source -> expand_by_source -> canonized_by_source -> create_csv_to_load_by_source
    -> enhancer_by_source -> download_images -> load_by_source

The merge_by_source step is NOT run here; it should be run separately after all sources, as in the CI workflow.

Each step is called as a Python module (e.g. `py -m news_mvp.etl.extract.extract_by_source ...`).
The script captures each step's final printed line (expected to be the output CSV relative path)
and passes it as the input of the next step.

Designed to be non-interactive and CI-friendly (GitHub Actions): prints the final merged master at the end.

for israel hayom, use:
    py -m news_mvp.etl.pipelines.etl_by_source --source hayom --rss https://www.hayom.co.il/rss/news.xml --force-tz-offset 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import structlog

log = structlog.get_logger(__name__)


STEPS = [
    "news_mvp.etl.transform.extract_by_source",
    "news_mvp.etl.transform.expand_by_source",
    "news_mvp.etl.transform.canonized_by_source",
    "news_mvp.etl.load.create_csv_to_load_by_source",
    "news_mvp.etl.load.enhancer_by_source",
    "news_mvp.etl.pipelines.download_images",
    "news_mvp.etl.load.load_by_source",
]


def run_cmd(
    args: List[str], timeout: Optional[int] = None, cwd: Optional[str] = None
) -> subprocess.CompletedProcess:
    log.debug("Running command", args=args, cwd=cwd)

    # Debug: log environment variables
    import os

    env_vars = {
        k: v
        for k, v in os.environ.items()
        if "NEWS_MVP" in k or "CONFIG" in k or "GITHUB" in k
    }
    log.debug("Environment variables", env_vars=env_vars)

    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=os.environ.copy(),  # Ensure child processes inherit environment variables
    )


def capture_output_path(proc: subprocess.CompletedProcess) -> Optional[str]:
    """Return the last non-empty line from stdout if present, else None."""
    if not proc.stdout:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def orchestrate(
    source: str,
    rss: str,
    python_cmd: str = sys.executable,
    dry_run: bool = False,
    per_step_timeout: int = 300,
    per_step_retries: int = 1,
    download_async: bool = False,
    download_concurrency: int = 6,
    force_tz_offset: Optional[int] = None,
) -> int:
    current_input = None
    # steps that shouldn't fail the whole pipeline; we'll continue with previous input
    soft_fail_steps = {
        "news_mvp.etl.load.enhancer_by_source",
        "news_mvp.etl.pipelines.download_images",
    }

    # Load mapping path from config
    from news_mvp.settings import Settings
    from news_mvp.paths import Paths
    import os

    env = os.environ.get("NEWS_MVP_CONFIG_ENV", "dev")
    cfg_path = f"configs/{env}.yaml"
    print(f"DEBUG: Using config environment: {env}, config path: {cfg_path}")
    s = Settings.load(cfg_path)

    # Resolve paths relative to project root
    project_root = Paths.root()
    working_directory = str(
        project_root.resolve()
    )  # Ensure absolute path for subprocess
    # Compute mapping/selectors paths and ensure they are absolute paths
    raw_mapping = (
        str(s.etl.etl_schema.mapping_csv)
        if hasattr(s.etl, "etl_schema")
        else "src/news_mvp/etl/schema/mapping.csv"
    )
    raw_selectors = (
        str(s.etl.etl_schema.selectors_csv)
        if hasattr(s.etl, "etl_schema")
        else "src/news_mvp/etl/schema/selectors.csv"
    )

    # If the configured path is relative, resolve it against project_root
    def _abs_path(p: str) -> str:
        pth = Path(p)
        return str((project_root / p).resolve()) if not pth.is_absolute() else str(pth)

    mapping_path = _abs_path(raw_mapping)
    selectors_path = _abs_path(raw_selectors)

    # Determine the effective timezone offset to force for this source (if any)
    # Priority: explicit function argument > config value for source > None
    effective_force_tz_offset: Optional[int] = force_tz_offset
    try:
        if effective_force_tz_offset is None:
            src_cfg = s.etl.sources.get(source)
            if src_cfg is not None:
                effective_force_tz_offset = src_cfg.force_tz_offset
    except Exception:
        # Be resilient to missing/partial config
        effective_force_tz_offset = force_tz_offset

    # Debug log what will be used
    print(
        f"Using force_tz_offset={effective_force_tz_offset!r} for source={source} (env={env})"
    )

    for idx, step in enumerate(STEPS, start=1):
        print(f"STEP {idx}/{len(STEPS)} -> {step} (current_input={current_input})")
        args: List[str] = [python_cmd, "-m", step]

        out_path = None
        # Build per-step arguments explicitly to avoid quoting issues and to supply required --output
        if step == "news_mvp.etl.transform.extract_by_source":
            args.extend(["--source", source, "--rss-url", rss])
        elif step == "news_mvp.etl.transform.expand_by_source":
            if not current_input:
                raise RuntimeError(
                    "expand_by_source requires an input path from a previous step"
                )
            in_p = Path(current_input)
            out_p = in_p.with_name(f"{in_p.stem}_expanded.csv")
            args.extend(["--input", str(in_p), "--output", str(out_p)])
            out_path = str(out_p)
        elif step == "news_mvp.etl.transform.canonized_by_source":
            if not current_input:
                raise RuntimeError(
                    "canonized_by_source requires an input path from a previous step"
                )
            in_p = Path(current_input)
            from news_mvp.etl.config import CANON_DIR

            can_dir = CANON_DIR / source
            can_dir.mkdir(parents=True, exist_ok=True)
            out_p = can_dir / f"{in_p.stem}_canonical.csv"
            args.extend(
                [
                    "--input",
                    str(in_p),
                    "--output",
                    str(out_p),
                    "--mapping",
                    mapping_path,
                ]
            )
            if effective_force_tz_offset is not None:
                args.extend(["--force-tz-offset", str(effective_force_tz_offset)])
            out_path = str(out_p)
        elif step == "news_mvp.etl.load.create_csv_to_load_by_source":
            if not current_input:
                raise RuntimeError(
                    "create_csv_to_load_by_source requires an input path from a previous step"
                )
            args.extend(["--input", str(current_input)])
        elif step == "news_mvp.etl.load.enhancer_by_source":
            if not current_input:
                raise RuntimeError(
                    "enhancer_by_source requires an input path from a previous step"
                )
            args.extend(["--input", str(current_input), "--selectors", selectors_path])
        elif step == "news_mvp.etl.pipelines.download_images":
            if not current_input:
                raise RuntimeError(
                    "download_images requires an input path from a previous step"
                )
            args.extend(["--input", str(current_input)])
            if download_async:
                args.append("--async")
            if download_concurrency:
                args.extend(["--concurrency", str(download_concurrency)])
        elif step == "news_mvp.etl.load.load_by_source":
            if not current_input:
                raise RuntimeError(
                    "load_by_source requires an input path from a previous step"
                )
            args.extend(["--input", str(current_input)])
        else:
            # Unknown step - pass current input if available
            if current_input:
                args.extend(["--input", str(current_input)])
        # dry-run prints the command and skips execution
        if dry_run:
            print(f"DRY-RUN: {' '.join(args)}")
            # simulate produced path for chaining if this step normally prints one
            # we rely on the convention that the last printed line is the output path; skip simulating
            continue

        attempt = 0
        while attempt < per_step_retries:
            attempt += 1
            print(f"running (attempt {attempt}/{per_step_retries}): {' '.join(args)}")
            try:
                proc = run_cmd(args, timeout=per_step_timeout, cwd=working_directory)
            except subprocess.TimeoutExpired as ex:
                print(
                    f"step timed out after {per_step_timeout}s: {' '.join(args)}",
                    file=sys.stderr,
                )
                proc = subprocess.CompletedProcess(
                    args=args, returncode=124, stdout="", stderr=str(ex)
                )

            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                if attempt >= per_step_retries:
                    if step in soft_fail_steps:
                        print(
                            f"warning: soft-fail step skipped after {attempt} attempts: {' '.join(args)}",
                            file=sys.stderr,
                        )
                        # do not change current_input; proceed to next step
                        break
                    else:
                        print(
                            f"step failed after {attempt} attempts: {' '.join(args)}",
                            file=sys.stderr,
                        )
                        return proc.returncode
                else:
                    print(f"retrying step (attempt {attempt+1}/{per_step_retries})...")
                    continue

            out_path = capture_output_path(proc)
            if out_path:
                current_input = out_path
                print(f"produced: {out_path}")
            else:
                print("no output path captured from step; passing previous input")
            break

    print(f"final unified master: {current_input}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Run full ETL pipeline for a single source. All paths are configurable via etl/config.py and environment/YAML."
    )
    p.add_argument("--source", required=True, help="Source name (eg. hayom)")
    p.add_argument("--rss", required=True, help="RSS feed URL for the source")
    p.add_argument(
        "--python",
        required=False,
        help="Python executable to use (defaults to current)",
        default=sys.executable,
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them"
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-step timeout in seconds (default: 300)",
    )
    p.add_argument(
        "--retries", type=int, default=1, help="Per-step retry count (default: 1)"
    )
    p.add_argument(
        "--download-async",
        action="store_true",
        help="Enable async downloads for download_images step",
    )
    p.add_argument(
        "--download-concurrency",
        type=int,
        default=6,
        help="Concurrency for download_images (default: 6)",
    )
    p.add_argument(
        "--force-tz-offset",
        type=int,
        default=None,
        help="Force output timezone offset in hours for canonized_by_source step (e.g., 3 for +03:00)",
    )
    args = p.parse_args(argv)

    return orchestrate(
        args.source,
        args.rss,
        python_cmd=args.python,
        dry_run=args.dry_run,
        per_step_timeout=args.timeout,
        per_step_retries=args.retries,
        download_async=args.download_async,
        download_concurrency=args.download_concurrency,
        force_tz_offset=args.force_tz_offset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
