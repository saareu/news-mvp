import sys
import re
import pandas as pd
from news_mvp.constants import CSV_ENCODING


def validate(path: str) -> int:
    """Fast pandas-based validator that mirrors the original expectations.

    Exit codes: 0=ok, 2=failure (same as original script expectations).
    """
    # Use the project's canonical CSV encoding so BOMs are handled consistently
    df = pd.read_csv(path, encoding=CSV_ENCODING, engine="python")
    total = len(df)

    # allow fallback columns for link: 'image' or 'imageName'
    # 'link' is optional in current masters; require id, title, pubDate
    required = ["id", "title", "pubDate"]
    wants_link = True
    # If 'link' present or fallback exists, we'll validate it; otherwise skip link checks
    if "link" in df.columns or any(alt in df.columns for alt in ("image", "imageName")):
        wants_link = True
        # ensure we have a 'link' column for validation
        if "link" not in df.columns:
            for alt in ("image", "imageName"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "link"})
                    break
        # include link in required set for not-null checks
        required = ["id", "title", "pubDate", "link"]
    else:
        wants_link = False
    # If 'link' missing, try to use 'image' or 'imageName' as fallback
    if "link" not in df.columns:
        for alt in ("image", "imageName"):
            if alt in df.columns:
                df = df.rename(columns={alt: "link"})
                break
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(
            {
                "ok": False,
                "reason": "missing_columns",
                "missing": missing,
                "stats": {"total": total},
            }
        )
        return 2

    not_null_ok = all(df[c].notna().all() for c in required)
    unique_ok = df["id"].nunique() == total
    url_re = re.compile(r"^https?://")
    links_ok = True
    if wants_link:
        # Use .all() to reduce Series to bool for pyright compatibility
        links_ok = df["link"].astype(str).apply(lambda v: bool(url_re.match(v))).all()
    try:
        pd.to_datetime(df["pubDate"], errors="raise")
        date_ok = True
    except Exception:
        date_ok = False

    all_ok = not_null_ok and unique_ok and links_ok and date_ok
    print(
        {
            "ok": all_ok,
            "stats": {
                "total": total,
                "unique_id": df["id"].nunique(),
                "date_ok": date_ok,
            },
        }
    )
    return 0 if all_ok else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ge_validate.py <csv>", file=sys.stderr)
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))
