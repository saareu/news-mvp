from news_mvp.paths import Paths
from news_mvp.logging_setup import get_logger

log = get_logger(__name__)


def main():
    for p in Paths.ensure_all():
        log.info("ensured: %s", p.resolve())


if __name__ == "__main__":
    main()
