import subprocess
import sys


def run_csv(csv_path):
    r = subprocess.run(
        [sys.executable, "scripts/ge_validate.py", str(csv_path)], capture_output=True
    )
    return r


def test_ge_validate_success(tmp_path):
    # create a minimal valid CSV
    csv = tmp_path / "ok.csv"
    csv.write_text(
        "id,title,link,pubDate,source\nid1,hello,http://example.com,2025-01-01,test\nid2,there,http://example.org,2025-01-02,test\n"
    )
    r = run_csv(csv)
    assert r.returncode == 0, r.stderr.decode()


def test_ge_validate_image_fallback(tmp_path):
    csv = tmp_path / "img.csv"
    csv.write_text(
        "id,title,image,pubDate,source\nid1,hello,http://example.com/img.jpg,2025-01-01,test\n"
    )
    r = run_csv(csv)
    assert r.returncode == 0, r.stderr.decode()


def test_ge_validate_missing_column(tmp_path):
    # link is optional; test failure when a required column (title) is missing
    csv = tmp_path / "bad.csv"
    csv.write_text("id,link,pubDate\n1,http://example.com,2025-01-01\n")
    r = run_csv(csv)
    assert r.returncode == 2
