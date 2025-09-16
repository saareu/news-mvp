from news_mvp.settings import load_settings
from news_mvp.paths import ensure_dirs
def test_config_dirs_ok():
    s = load_settings("dev")
    ensure_dirs(s)
    assert s.app.name and s.paths.data_dir
