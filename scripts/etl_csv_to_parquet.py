# scripts/etl_csv_to_parquet.py
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import polars as pl  # already in deps

def write_parquet(csv_path: Path) -> Path:
    # infer source from parent dir name (e.g., data/raw/ynet/...)
    source = csv_path.parent.name
    date = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("data") / "raw" / f"source={source}" / f"date={date}" / f"part-{ts}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.read_csv(csv_path).write_parquet(out)
    print(str(out))
    return out

def main(arg: str) -> None:
    p = Path(arg)
    csvs: list[Path] = []
    if p.is_dir():
        csvs = sorted(p.rglob("*.csv"))
    elif p.suffix.lower() == ".csv" and p.exists():
        csvs = [p]
    else:
        print(f"No CSVs found at: {p}", file=sys.stderr); sys.exit(1)

    for c in csvs:
        write_parquet(c)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/etl_csv_to_parquet.py <csv-file-or-directory>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
