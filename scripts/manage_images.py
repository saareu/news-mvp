#!/usr/bin/env python3
"""
News MVP Image Manager

This script handles:
1. Organizing images by source/date/hour
2. Managing image storage and cleanup
3. Updating image paths in data
4. Image metadata extraction

Usage:
    python scripts/manage_images.py --organize --source ynet --date 2025-09-17
    python scripts/manage_images.py --cleanup --days 30
    python scripts/manage_images.py --stats
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ImageManager:
    """Manages image storage and organization."""

    def __init__(self, images_base_path: str = "data/images"):
        self.images_base_path = Path(images_base_path)
        self.images_base_path.mkdir(parents=True, exist_ok=True)

    def organize_images_by_source_date(
        self, source: str, date: str, hour: Optional[int] = None
    ):
        """Organize images for a source by date and hour."""
        if hour is None:
            hour = datetime.now().hour

        # Source directory
        source_dir = self.images_base_path / source
        date_dir = source_dir / date
        hour_dir = date_dir / f"{hour:02d}"

        # Create directories
        hour_dir.mkdir(parents=True, exist_ok=True)

        # Find images in pics directory that belong to this source
        pics_dir = Path("data/pics")
        if not pics_dir.exists():
            logger.warning("Pics directory not found")
            return

        # Pattern for source images
        source_pattern = f"{source}_*.jpg"
        moved_count = 0

        for image_file in pics_dir.glob(source_pattern):
            try:
                # Move image to organized location
                target_path = hour_dir / image_file.name
                shutil.move(str(image_file), str(target_path))
                moved_count += 1
                logger.debug(f"Moved {image_file.name} to {target_path}")

            except Exception as e:
                logger.warning(f"Error moving {image_file}: {e}")

        logger.info(f"Organized {moved_count} images for {source}/{date}/{hour:02d}")

    def organize_all_sources(self, date: str, hour: Optional[int] = None):
        """Organize images for all sources."""
        sources = ["ynet", "hayom", "haaretz"]

        for source in sources:
            try:
                self.organize_images_by_source_date(source, date, hour)
            except Exception as e:
                logger.error(f"Error organizing {source}: {e}")

    def cleanup_old_images(self, days: int = 30):
        """Remove images older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0
        total_size = 0

        for source_dir in self.images_base_path.iterdir():
            if not source_dir.is_dir():
                continue

            for date_dir in source_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                try:
                    # Parse date from directory name
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                    if dir_date < cutoff_date.date():
                        # Remove entire date directory
                        size = self.get_directory_size(date_dir)
                        shutil.rmtree(date_dir)
                        removed_count += 1
                        total_size += size
                        logger.info(f"Removed {date_dir} ({size} bytes)")

                except ValueError:
                    # Invalid date format, skip
                    continue
                except Exception as e:
                    logger.warning(f"Error processing {date_dir}: {e}")

        logger.info(
            f"Cleanup complete: removed {removed_count} directories, {total_size} bytes"
        )

    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size

    def update_image_paths_in_csv(
        self, csv_path: Path, source: str, date: str, hour: int
    ):
        """Update image paths in CSV file to point to organized locations."""
        try:
            import pandas as pd

            df = pd.read_csv(csv_path, encoding="utf-8")

            if "image" not in df.columns:
                logger.warning(f"No image column in {csv_path}")
                return

            # Update image paths
            updated_count = 0
            for idx, row in df.iterrows():
                image_path = row.get("image", "")
                if image_path and Path(image_path).name:
                    # Construct new organized path
                    image_name = Path(image_path).name
                    new_path = f"data/images/{source}/{date}/{hour:02d}/{image_name}"

                    # Update if different
                    if image_path != new_path:
                        df.at[idx, "image"] = new_path
                        updated_count += 1

            if updated_count > 0:
                # Save updated CSV
                df.to_csv(csv_path, index=False, encoding="utf-8")
                logger.info(f"Updated {updated_count} image paths in {csv_path}")

        except Exception as e:
            logger.error(f"Error updating image paths in {csv_path}: {e}")

    def get_image_stats(self) -> Dict:
        """Get statistics about stored images."""
        stats = {
            "total_images": 0,
            "total_size": 0,
            "sources": {},
            "oldest_date": None,
            "newest_date": None,
        }

        for source_dir in self.images_base_path.iterdir():
            if not source_dir.is_dir():
                continue

            source_name = source_dir.name
            source_stats = {"images": 0, "size": 0, "dates": []}

            for date_dir in source_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                date_str = date_dir.name
                source_stats["dates"].append(date_str)

                # Count images in this date directory
                for hour_dir in date_dir.iterdir():
                    if hour_dir.is_dir():
                        for image_file in hour_dir.glob("*"):
                            if image_file.is_file() and image_file.suffix.lower() in [
                                ".jpg",
                                ".jpeg",
                                ".png",
                            ]:
                                source_stats["images"] += 1
                                source_stats["size"] += image_file.stat().st_size
                                stats["total_images"] += 1
                                stats["total_size"] += image_file.stat().st_size

            stats["sources"][source_name] = source_stats

            # Update oldest/newest dates
            if source_stats["dates"]:
                dates = sorted(source_stats["dates"])
                if stats["oldest_date"] is None or dates[0] < stats["oldest_date"]:
                    stats["oldest_date"] = dates[0]
                if stats["newest_date"] is None or dates[-1] > stats["newest_date"]:
                    stats["newest_date"] = dates[-1]

        return stats

    def print_stats(self):
        """Print image statistics."""
        stats = self.get_image_stats()

        print("Image Storage Statistics:")
        print(f"Total Images: {stats['total_images']:,}")
        print(
            f"Total Size: {stats['total_size']:,} bytes ({stats['total_size']/1024/1024:.1f} MB)"
        )
        print(f"Date Range: {stats['oldest_date']} to {stats['newest_date']}")
        print("\nBy Source:")

        for source, source_stats in stats["sources"].items():
            print(f"  {source}:")
            print(f"    Images: {source_stats['images']:,}")
            print(f"    Size: {source_stats['size']/1024/1024:.1f} MB")
            print(f"    Dates: {len(source_stats['dates'])}")


def main():
    parser = argparse.ArgumentParser(description="Manage news images")
    parser.add_argument(
        "--organize", action="store_true", help="Organize images by source/date/hour"
    )
    parser.add_argument("--source", help="Source name (ynet, hayom, haaretz)")
    parser.add_argument(
        "--all-sources", action="store_true", help="Organize all sources"
    )
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"), help="Date (YYYY-MM-DD)"
    )
    parser.add_argument("--hour", type=int, help="Hour (0-23)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old images")
    parser.add_argument(
        "--days", type=int, default=30, help="Days to keep images (for cleanup)"
    )
    parser.add_argument("--stats", action="store_true", help="Show image statistics")
    parser.add_argument(
        "--update-csv", action="store_true", help="Update image paths in CSV files"
    )

    args = parser.parse_args()

    manager = ImageManager()

    try:
        if args.stats:
            manager.print_stats()

        elif args.organize:
            if args.source:
                manager.organize_images_by_source_date(
                    args.source, args.date, args.hour
                )
            elif args.all_sources:
                manager.organize_all_sources(args.date, args.hour)
            else:
                logger.error("Must specify --source or --all-sources with --organize")

        elif args.cleanup:
            manager.cleanup_old_images(args.days)

        elif args.update_csv:
            # Update CSV files with new image paths
            sources = (
                ["ynet", "hayom", "haaretz"] if args.all_sources else [args.source]
            )
            hour = args.hour or datetime.now().hour

            for source in sources:
                if source:
                    csv_path = Path("data/master") / f"master_{source}.csv"
                    if csv_path.exists():
                        manager.update_image_paths_in_csv(
                            csv_path, source, args.date, hour
                        )

        else:
            logger.error(
                "Must specify an action: --organize, --cleanup, --stats, or --update-csv"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Image management failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
