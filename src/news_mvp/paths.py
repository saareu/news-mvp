from pathlib import Path


class Paths:
    @staticmethod
    def root() -> Path:
        return Path(".")

    @staticmethod
    def data_root() -> Path:
        return Path("data")

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
