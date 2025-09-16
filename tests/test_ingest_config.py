import os
from pathlib import Path
from news_mvp.settings import Settings

def test_ingest_config_valid():
    """Test that ingest config is properly loaded across environments"""
    for env in ["dev", "staging", "prod"]:
        cfg_path = f"configs/{env}.yaml"
        settings = Settings.load(cfg_path)
        
        # Verify ingest config exists
        assert hasattr(settings, "ingest")
        assert settings.ingest.input_glob
        
        # Test glob pattern matches expected structure
        assert "source=" in settings.ingest.input_glob
        assert "date=" in settings.ingest.input_glob
        assert "part-" in settings.ingest.input_glob

def test_ingest_batch_size_hierarchy():
    """Test that batch size follows expected hierarchy: prod > staging > dev"""
    dev = Settings.load("configs/dev.yaml")
    staging = Settings.load("configs/staging.yaml")
    prod = Settings.load("configs/prod.yaml")
    
    assert prod.ingest.batch_size > staging.ingest.batch_size > dev.ingest.batch_size