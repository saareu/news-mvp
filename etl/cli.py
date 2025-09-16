"""
Unified CLI for News ETL
------------------------
Run ETL steps from the command line.

Example:
    python -m etl.cli run-etl --source ynet --rss https://www.ynet.co.il/Integration/StoryRss2.xml
    python -m etl.cli merge-masters --sources data/master/master_ynet.csv data/master/master_hayom.csv
    python -m etl.cli download-images --input data/canonical/ynet/ynet_..._canonical_enhanced.csv
"""
import argparse
import sys
from etl.api import run_etl_for_source, merge_masters, download_images_for_csv

def main():
    parser = argparse.ArgumentParser(description="Unified CLI for News ETL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-etl
    p_etl = subparsers.add_parser("run-etl", help="Run ETL for a source")
    p_etl.add_argument("--source", required=True)
    p_etl.add_argument("--rss", required=True)
    p_etl.add_argument("--force-tz-offset", type=int)
    p_etl.add_argument("--timeout", type=int, default=600)
    p_etl.add_argument("--retries", type=int, default=1)

    # merge-masters
    p_merge = subparsers.add_parser("merge-masters", help="Merge master CSVs")
    p_merge.add_argument("--sources", nargs='+', required=True)
    p_merge.add_argument("--output")

    # download-images
    p_dl = subparsers.add_parser("download-images", help="Download images for a canonical CSV")
    p_dl.add_argument("--input", required=True)
    p_dl.add_argument("--output")
    p_dl.add_argument("--source")
    p_dl.add_argument("--async", dest="async_mode", action="store_true")
    p_dl.add_argument("--concurrency", type=int, default=6)

    args = parser.parse_args()
    if args.command == "run-etl":
        sys.exit(run_etl_for_source(args.source, args.rss, args.force_tz_offset, args.timeout, args.retries))
    elif args.command == "merge-masters":
        sys.exit(merge_masters(args.sources, args.output))
    elif args.command == "download-images":
        sys.exit(download_images_for_csv(args.input, args.output, args.source, args.async_mode, args.concurrency))

if __name__ == "__main__":
    main()
