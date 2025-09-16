import subprocess, sys
def test_etl_dry_run_exit0():
    r = subprocess.run([sys.executable, "-m", "news_mvp.cli", "etl", "run", "--source", "ynet", "--dry-run"], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
