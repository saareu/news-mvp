from etl import api

def test_merge_masters_runs():
    # This is a smoke test; in real use, provide real CSVs or mock subprocess
    assert hasattr(api, 'merge_masters')
    assert callable(api.merge_masters)

def test_run_etl_for_source_exists():
    assert hasattr(api, 'run_etl_for_source')
    assert callable(api.run_etl_for_source)

def test_download_images_for_csv_exists():
    assert hasattr(api, 'download_images_for_csv')
    assert callable(api.download_images_for_csv)
