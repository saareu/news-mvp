import sys
from pathlib import Path
import pandas as pd
from news_mvp.paths import Paths
from news_mvp.logging_setup import get_logger

log = get_logger(__name__)

def csv_to_parquet(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    out = out_dir / (csv_path.stem + ".parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s", out)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/etl_csv_to_parquet.py <csv_path>")
        sys.exit(2)
    csv = Path(sys.argv[1])
    csv_to_parquet(csv, Paths.master())
