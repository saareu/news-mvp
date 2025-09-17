from news_mvp.settings import get_config
from rich import print as rprint

if __name__ == "__main__":
    s = get_config()
    rprint(
        {"logging": s.logging, "etl": s.etl, "runtime": s.runtime, "ingest": s.ingest}
    )
