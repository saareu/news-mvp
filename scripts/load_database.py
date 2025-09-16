#!/usr/bin/env python3
"""
News MVP Database Loader

This script handles loading news data into DuckDB:
1. Load articles from CSV/Parquet files
2. Load image metadata
3. Maintain referential integrity
4. Handle incremental updates

Usage:
    python scripts/load_database.py --source ynet
    python scripts/load_database.py --all-sources
    python scripts/load_database.py --parquet data/parquet/unified/2025-09-17/12_unified.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
import pyarrow.parquet as pq

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseLoader:
    """Handles loading news data into DuckDB."""

    def __init__(self, db_path: str = "data/db/news_mvp.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = duckdb.connect(str(self.db_path))

        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA wal_autocheckpoint = 1000;")
        self.conn.execute("SET memory_limit = '2GB';")

        # Initialize schema if needed
        self.init_schema()

    def init_schema(self):
        """Initialize database schema if it doesn't exist."""
        schema_path = Path(__file__).parent.parent / "schema.sql"
        if schema_path.exists():
            logger.info("Initializing database schema...")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            # Execute schema (DuckDB handles IF NOT EXISTS)
            self.conn.execute(schema_sql)
            logger.info("Database schema initialized")
        else:
            logger.warning(f"Schema file not found: {schema_path}")

    def get_source_id(self, source_name: str) -> int:
        """Get or create source ID."""
        result = self.conn.execute(
            "SELECT id FROM sources WHERE name = ?", [source_name]
        ).fetchone()

        if result:
            return result[0]

        # Create new source
        self.conn.execute(
            "INSERT INTO sources (name, display_name) VALUES (?, ?)",
            [source_name, source_name.title()],
        )

        result = self.conn.execute("SELECT last_insert_rowid()").fetchone()
        return result[0] if result else 1

    def get_or_create_tag_ids(self, tags: List[str]) -> List[int]:
        """Get or create tag IDs for a list of tags."""
        if not tags:
            return []

        tag_ids = []
        for tag in tags:
            tag = tag.strip()
            if not tag:
                continue

            # Try to get existing tag
            result = self.conn.execute(
                "SELECT id FROM tags WHERE name = ?", [tag]
            ).fetchone()

            if result:
                tag_ids.append(result[0])
            else:
                # Create new tag
                self.conn.execute("INSERT INTO tags (name) VALUES (?)", [tag])
                result = self.conn.execute("SELECT last_insert_rowid()").fetchone()
                tag_id = result[0] if result else 1
                tag_ids.append(tag_id)

        return tag_ids

    def load_articles_from_csv(self, csv_path: Path, source_name: str):
        """Load articles from CSV file."""
        logger.info(f"Loading articles from {csv_path}")

        try:
            # Read CSV
            df = pd.read_csv(csv_path, encoding="utf-8")

            if df.empty:
                logger.warning(f"No data in {csv_path}")
                return

            # Get source ID
            source_id = self.get_source_id(source_name)

            # Process each article
            articles_loaded = 0
            for _, row in df.iterrows():
                try:
                    self.insert_article(row, source_id, source_name)
                    articles_loaded += 1
                except Exception as e:
                    logger.warning(
                        f"Error loading article {row.get('id', 'unknown')}: {e}"
                    )

            logger.info(f"Loaded {articles_loaded} articles from {source_name}")

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            raise

    def load_articles_from_parquet(self, parquet_path: Path):
        """Load articles from Parquet file."""
        logger.info(f"Loading articles from {parquet_path}")

        try:
            # Read Parquet
            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            if df.empty:
                logger.warning(f"No data in {parquet_path}")
                return

            # Process each article
            articles_loaded = 0
            for _, row in df.iterrows():
                try:
                    # Get source from row or filename
                    source_name = row.get("source", "unknown")
                    if source_name == "unknown":
                        # Try to extract from filename
                        filename = parquet_path.name
                        if "unified" in filename:
                            source_name = "unified"
                        else:
                            source_name = "unknown"

                    source_id = self.get_source_id(source_name)
                    self.insert_article(row, source_id, source_name)
                    articles_loaded += 1
                except Exception as e:
                    logger.warning(
                        f"Error loading article {row.get('article_id', 'unknown')}: {e}"
                    )

            logger.info(f"Loaded {articles_loaded} articles from Parquet")

        except Exception as e:
            logger.error(f"Error loading Parquet {parquet_path}: {e}")
            raise

    def insert_article(self, row, source_id: int, source_name: str):
        """Insert a single article into the database."""
        article_id = str(row.get("id") or row.get("article_id", ""))
        if not article_id:
            raise ValueError("Article ID is required")

        # Check if article already exists
        existing = self.conn.execute(
            "SELECT article_id FROM articles WHERE article_id = ?", [article_id]
        ).fetchone()

        if existing:
            # Update existing article
            self.conn.execute(
                """
                UPDATE articles SET
                    title = ?,
                    description = ?,
                    category = ?,
                    pub_date = ?,
                    creator = ?,
                    language = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE article_id = ?
            """,
                [
                    row.get("title", ""),
                    row.get("description", ""),
                    row.get("category", ""),
                    pd.to_datetime(row.get("pubDate") or row.get("pub_date")),
                    row.get("creator", ""),
                    row.get("language", "he"),
                    article_id,
                ],
            )
        else:
            # Insert new article
            self.conn.execute(
                """
                INSERT INTO articles (
                    article_id, title, description, category, pub_date,
                    creator, source_id, language
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    article_id,
                    row.get("title", ""),
                    row.get("description", ""),
                    row.get("category", ""),
                    pd.to_datetime(row.get("pubDate") or row.get("pub_date")),
                    row.get("creator", ""),
                    source_id,
                    row.get("language", "he"),
                ],
            )

        # Handle tags
        tags = row.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        elif not isinstance(tags, list):
            tags = []

        if tags:
            tag_ids = self.get_or_create_tag_ids(tags)
            # Insert article-tag relationships
            for tag_id in tag_ids:
                try:
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO article_tags (article_id, tag_id)
                        VALUES (?, ?)
                    """,
                        [article_id, tag_id],
                    )
                except Exception as e:
                    logger.warning(f"Error inserting tag relationship: {e}")

        # Handle image if present
        image_path = row.get("image") or row.get("image_path")
        if image_path:
            self.insert_image(article_id, row, image_path)

    def insert_image(self, article_id: str, row, image_path: str):
        """Insert image metadata into database."""
        try:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO images (
                    article_id, filename, original_path, stored_path,
                    caption, credit
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    article_id,
                    Path(image_path).name,
                    image_path,
                    image_path,  # For now, stored_path = original_path
                    row.get("imageCaption") or row.get("image_caption", ""),
                    row.get("imageCredit") or row.get("image_credit", ""),
                ],
            )
        except Exception as e:
            logger.warning(f"Error inserting image for {article_id}: {e}")

    def load_source(self, source_name: str):
        """Load data for a specific source."""
        csv_path = Path("data/master") / f"master_{source_name}.csv"
        if csv_path.exists():
            self.load_articles_from_csv(csv_path, source_name)
        else:
            logger.warning(f"CSV file not found: {csv_path}")

    def load_all_sources(self):
        """Load data for all sources."""
        sources = ["ynet", "hayom", "haaretz"]
        for source in sources:
            try:
                self.load_source(source)
            except Exception as e:
                logger.error(f"Error loading {source}: {e}")

    def load_parquet_file(self, parquet_path: str):
        """Load data from a specific Parquet file."""
        path = Path(parquet_path)
        if path.exists():
            self.load_articles_from_parquet(path)
        else:
            logger.error(f"Parquet file not found: {parquet_path}")

    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats = {}

        # Article count
        result = self.conn.execute("SELECT COUNT(*) FROM articles").fetchone()
        stats["total_articles"] = result[0] if result else 0

        # Source count
        result = self.conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        stats["total_sources"] = result[0] if result else 0

        # Image count
        result = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()
        stats["total_images"] = result[0] if result else 0

        # Tag count
        result = self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()
        stats["total_tags"] = result[0] if result else 0

        # Articles by source
        result = self.conn.execute(
            """
            SELECT s.name, COUNT(a.article_id) as count
            FROM sources s
            LEFT JOIN articles a ON s.id = a.source_id
            GROUP BY s.id, s.name
        """
        ).fetchall()
        stats["articles_by_source"] = dict(result) if result else {}

        return stats

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load news data into DuckDB")
    parser.add_argument("--source", help="Source name (ynet, hayom, haaretz)")
    parser.add_argument("--all-sources", action="store_true", help="Load all sources")
    parser.add_argument("--parquet", help="Path to Parquet file to load")
    parser.add_argument(
        "--db-path", default="data/db/news_mvp.duckdb", help="Database path"
    )
    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    loader = DatabaseLoader(args.db_path)

    try:
        if args.stats:
            stats = loader.get_stats()
            print("Database Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        elif args.source:
            loader.load_source(args.source)
            logger.info(f"Loaded data for source: {args.source}")

        elif args.all_sources:
            loader.load_all_sources()
            logger.info("Loaded data for all sources")

        elif args.parquet:
            loader.load_parquet_file(args.parquet)
            logger.info(f"Loaded data from Parquet: {args.parquet}")

        else:
            logger.error("Must specify --source, --all-sources, --parquet, or --stats")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Database loading failed: {e}")
        sys.exit(1)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
