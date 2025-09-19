from news_mvp.settings import Settings
from rich import print as rprint

if __name__ == "__main__":
    s = Settings.load("configs/dev.yaml")
    rprint({"etl": s.etl, "logging": s.logging, "runtime": s.runtime})
