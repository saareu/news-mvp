from news_mvp.paths import Paths  # Directory path management
from news_mvp.logging_setup import get_logger  # Logging configuration

log = get_logger(__name__)


def main():
    for p in Paths.ensure_all():  # Creates all data directories
        log.info("ensured: %s", p.resolve())  # Logs each created directory


if __name__ == "__main__":
    main()
