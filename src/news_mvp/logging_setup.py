import logging
import sys
import os

import structlog
from structlog.types import FilteringBoundLogger


def configure_logging(
    level: str = "INFO", format_type: str = "json", structured: bool = True
) -> None:
    """Configure structured logging with structlog.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: "json" for machine-readable logs, "human" for dev
        structured: Whether to use structured logging processors
    """
    # Configure standard library logging first
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        format="%(message)s",  # structlog will handle formatting
    )

    # Determine if we're in development mode
    is_dev = format_type == "human" or os.getenv("NEWS_MVP_LOG_HUMAN", "").lower() in (
        "1",
        "true",
        "yes",
    )

    # Configure processors based on environment
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if structured:
        # Add contextual data processors
        processors.extend(
            [
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    ]
                ),
            ]
        )

    if is_dev:
        # Human-readable console output for development
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # JSON output for production/parsing
        processors.append(structlog.processors.JSONRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "news_mvp") -> FilteringBoundLogger:
    """Get a structured logger instance.

    Returns a structlog logger that supports both standard logging methods
    and structured key-value logging.

    Examples:
        log = get_logger(__name__)
        log.info("Starting ETL", source="ynet", rss_url="https://...")
        log.error("Failed to process", error=str(e), source=source)
    """
    return structlog.get_logger(name)


# Legacy compatibility function
def get_standard_logger(name: str = "news_mvp") -> logging.Logger:
    """Get a standard library logger for modules that need it."""
    return logging.getLogger(name)
