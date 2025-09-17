import os
from pathlib import Path


class Paths:
    @staticmethod
    def root() -> Path:
        """Get the project root directory.

        Can be overridden with NEWS_MVP_ROOT environment variable.
        Defaults to current working directory.
        """
        return Path(os.getenv("NEWS_MVP_ROOT", "."))

    @staticmethod
    def data_root() -> Path:
        """Get the data directory root.

        Can be overridden with NEWS_MVP_DATA_ROOT environment variable.
        Defaults to 'data' relative to project root.
        """
        data_root_env = os.getenv("NEWS_MVP_DATA_ROOT")
        if data_root_env:
            return Path(data_root_env)
        return Paths.root() / "data"

    @staticmethod
    def raw() -> Path:
        return Paths.data_root() / "raw"

    @staticmethod
    def canonical() -> Path:
        return Paths.data_root() / "canonical"

    @staticmethod
    def master() -> Path:
        return Paths.data_root() / "master"

    @staticmethod
    def pics() -> Path:
        return Paths.data_root() / "pics"

    @staticmethod
    def db() -> Path:
        """Get the database directory."""
        return Paths.data_root() / "db"

    @staticmethod
    def database() -> Path:
        """Get the default database file path."""
        return Paths.db() / "news_mvp.duckdb"

    @staticmethod
    def ensure_all():
        for p in [
            Paths.data_root(),
            Paths.raw(),
            Paths.canonical(),
            Paths.master(),
            Paths.pics(),
        ]:
            p.mkdir(parents=True, exist_ok=True)
            yield p


def ensure_dirs(settings):
    """Compatibility helper used by tests: ensures dirs and returns list of created paths."""
    created = []
    for p in Paths.ensure_all():
        created.append(p)
    return created
