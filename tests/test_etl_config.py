"""Test ETL config path resolution."""

import os
import tempfile
import importlib
from pathlib import Path


def test_etl_config_default_paths():
    """Test that ETL config uses correct default paths."""
    # Clear any environment overrides
    old_config = os.environ.pop("NEWS_ETL_CONFIG", None)
    old_root = os.environ.pop("NEWS_MVP_ROOT", None)
    old_data = os.environ.pop("NEWS_MVP_DATA_ROOT", None)

    try:
        # Re-import to get fresh config
        from news_mvp.etl import config

        importlib.reload(config)

        # Should use current directory (relative by default)
        assert config.BASE_DIR == Path(".")
        assert "data" in str(config.DATA_DIR)
        assert "raw" in str(config.RAW_DIR)
        assert "canonical" in str(config.CANON_DIR)

    finally:
        # Restore environment variables
        if old_config:
            os.environ["NEWS_ETL_CONFIG"] = old_config
        if old_root:
            os.environ["NEWS_MVP_ROOT"] = old_root
        if old_data:
            os.environ["NEWS_MVP_DATA_ROOT"] = old_data


def test_etl_config_environment_override():
    """Test that ETL config respects NEWS_MVP_ROOT environment variable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Set environment override
        os.environ["NEWS_MVP_ROOT"] = str(temp_path)

        try:
            # Re-import to get fresh config with environment override
            from news_mvp.etl import config

            importlib.reload(config)

            # Should use overridden path
            assert config.BASE_DIR == temp_path
            assert config.DATA_DIR == temp_path / "data"
            assert config.RAW_DIR == temp_path / "data" / "raw"

        finally:
            os.environ.pop("NEWS_MVP_ROOT", None)
            # Reload again to restore default behavior
            from news_mvp.etl import config

            importlib.reload(config)


if __name__ == "__main__":
    test_etl_config_default_paths()
    test_etl_config_environment_override()
    print("All ETL config tests passed!")
