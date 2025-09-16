"""Test path resolution with environment variable overrides."""

import os
import tempfile
from pathlib import Path
from news_mvp.paths import Paths


def test_default_paths():
    """Test that default paths work as expected."""
    # Clear any environment variables that might interfere
    old_root = os.environ.pop("NEWS_MVP_ROOT", None)
    old_data = os.environ.pop("NEWS_MVP_DATA_ROOT", None)

    try:
        assert Paths.root() == Path(".")
        assert Paths.data_root() == Path("data")
        assert Paths.raw() == Path("data/raw")
        assert Paths.canonical() == Path("data/canonical")
        assert Paths.master() == Path("data/master")
        assert Paths.pics() == Path("data/pics")
    finally:
        # Restore environment variables
        if old_root:
            os.environ["NEWS_MVP_ROOT"] = old_root
        if old_data:
            os.environ["NEWS_MVP_DATA_ROOT"] = old_data


def test_environment_overrides():
    """Test that environment variables override default paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Clear existing environment variables first
        old_root = os.environ.pop("NEWS_MVP_ROOT", None)
        old_data = os.environ.pop("NEWS_MVP_DATA_ROOT", None)

        try:
            # Test NEWS_MVP_ROOT override
            os.environ["NEWS_MVP_ROOT"] = str(temp_path)
            assert Paths.root() == temp_path
            assert Paths.data_root() == temp_path / "data"

            # Clear root override for next test
            os.environ.pop("NEWS_MVP_ROOT", None)

            # Test NEWS_MVP_DATA_ROOT override
            data_path = temp_path / "custom_data"
            os.environ["NEWS_MVP_DATA_ROOT"] = str(data_path)
            assert Paths.data_root() == data_path
            assert Paths.raw() == data_path / "raw"
            assert Paths.canonical() == data_path / "canonical"
        finally:
            # Restore original environment variables
            os.environ.pop("NEWS_MVP_ROOT", None)
            os.environ.pop("NEWS_MVP_DATA_ROOT", None)
            if old_root:
                os.environ["NEWS_MVP_ROOT"] = old_root
            if old_data:
                os.environ["NEWS_MVP_DATA_ROOT"] = old_data


def test_absolute_paths():
    """Test that absolute paths work correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        abs_path = Path(temp_dir).resolve()

        os.environ["NEWS_MVP_DATA_ROOT"] = str(abs_path)
        try:
            assert Paths.data_root().is_absolute()
            assert Paths.data_root() == abs_path
            assert Paths.raw() == abs_path / "raw"
        finally:
            os.environ.pop("NEWS_MVP_DATA_ROOT", None)


def test_ensure_all_with_custom_paths():
    """Test that ensure_all works with custom paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_data = Path(temp_dir) / "test_data"

        os.environ["NEWS_MVP_DATA_ROOT"] = str(custom_data)
        try:
            # Ensure directories are created
            created_paths = list(Paths.ensure_all())

            # Verify all paths exist
            assert custom_data.exists()
            assert (custom_data / "raw").exists()
            assert (custom_data / "canonical").exists()
            assert (custom_data / "master").exists()
            assert (custom_data / "pics").exists()

            # Verify return values
            assert custom_data in created_paths
            assert custom_data / "raw" in created_paths
        finally:
            os.environ.pop("NEWS_MVP_DATA_ROOT", None)


if __name__ == "__main__":
    test_default_paths()
    test_environment_overrides()
    test_absolute_paths()
    test_ensure_all_with_custom_paths()
    print("All path tests passed!")
