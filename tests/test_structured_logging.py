"""Test structured logging configuration."""

import subprocess
import json
from news_mvp.logging_setup import configure_logging, get_logger
from news_mvp.settings import Settings


def test_logging_config_from_settings():
    """Test that logging configuration is read from settings."""
    # Create temporary config for testing
    test_config = {
        "logging": {"level": "DEBUG", "format": "json", "structured": True},
        "etl": {
            "sources": {},
            "etl_schema": {"mapping_csv": "", "selectors_csv": ""},
            "output": {
                "raw_pattern": "",
                "canonical_dir": "",
                "master_dir": "",
                "images_dir": "",
            },
            "behavior": {},
        },
    }

    settings = Settings(**test_config)
    assert settings.logging.level == "DEBUG"
    assert settings.logging.format == "json"
    assert settings.logging.structured is True


def test_cli_json_logging():
    """Test that CLI produces JSON logs in production mode."""
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "news_mvp.cli",
                "etl",
                "run",
                "--source",
                "ynet",
                "--dry-run",
                "--env",
                "prod",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should have JSON logs in stdout
        output_lines = result.stdout.strip().split("\n")
        json_lines = [line for line in output_lines if line.strip().startswith("{")]

        assert len(json_lines) > 0, "Expected JSON log lines"

        # Verify first JSON log is parseable
        first_log = json.loads(json_lines[0])
        assert "timestamp" in first_log
        assert "level" in first_log
        assert "event" in first_log

    except subprocess.TimeoutExpired:
        raise AssertionError("CLI command timed out")


def test_cli_human_logging():
    """Test that CLI produces human-readable logs in dev mode."""
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "news_mvp.cli",
                "etl",
                "run",
                "--source",
                "ynet",
                "--dry-run",
                "--env",
                "dev",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should have human-readable logs in stdout
        output = result.stdout.strip()

        assert "info" in output.lower(), "Expected 'info' in output"
        assert "Directory ensured" in output, "Expected structured log message"
        assert "etl.run" in output, "Expected logger name in output"

    except subprocess.TimeoutExpired:
        raise AssertionError("CLI command timed out")


def test_structured_logging_setup():
    """Test that configure_logging doesn't raise errors."""
    # Should not raise any exceptions
    configure_logging(level="INFO", format_type="json", structured=True)
    configure_logging(level="DEBUG", format_type="human", structured=True)

    # Should be able to get logger
    log = get_logger("test")
    assert log is not None


if __name__ == "__main__":
    test_logging_config_from_settings()
    test_cli_json_logging()
    test_cli_human_logging()
    test_structured_logging_setup()
    print("All logging tests passed!")
