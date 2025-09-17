from news_mvp.settings import Settings
from rich import print as rprint

if __name__ == "__main__":
    s = Settings.load()
    rprint({"app": s["app"], "logging": s["logging"], "paths": s["paths"]})
