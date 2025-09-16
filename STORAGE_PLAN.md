# News MVP Storage Plan

## Overview
This plan outlines a comprehensive storage strategy for the news ETL pipeline with:
- **Parquet files** per source + unified files (every 3 hours)
- **Image storage** and management
- **Database storage** with normalized schema

## 1. Directory Structure

```
data/
├── parquet/
│   ├── sources/
│   │   ├── ynet/
│   │   │   ├── 2025-09-17/
│   │   │   │   ├── 12_articles.parquet
│   │   │   │   ├── 15_articles.parquet
│   │   │   │   └── 18_articles.parquet
│   │   ├── hayom/
│   │   │   └── 2025-09-17/
│   │   └── haaretz/
│   │       └── 2025-09-17/
│   └── unified/
│       ├── 2025-09-17/
│       │   ├── 12_unified.parquet
│       │   ├── 15_unified.parquet
│       │   └── 18_unified.parquet
│       └── latest_unified.parquet -> 2025-09-17/18_unified.parquet
├── images/
│   ├── ynet/
│   │   └── 2025-09-17/
│   │       ├── 12/
│   │       │   ├── abc123.jpg
│   │       │   └── def456.png
│   │       └── 15/
│   ├── hayom/
│   │   └── 2025-09-17/
│   └── haaretz/
│       └── 2025-09-17/
└── db/
    └── news_mvp.duckdb
```

## 2. Parquet File Strategy

### Per-Source Files
- **Location**: `data/parquet/sources/{source}/{date}/`
- **Naming**: `{hour}_articles.parquet`
- **Partitioning**: By date (YYYY-MM-DD) and hour
- **Content**: All articles from that source for that hour

### Unified Files
- **Location**: `data/parquet/unified/{date}/`
- **Naming**: `{hour}_unified.parquet`
- **Content**: Combined articles from all sources for that hour
- **Latest Link**: `latest_unified.parquet` -> most recent unified file

### Schema
```python
# Parquet schema for articles
schema = {
    'article_id': 'string',      # SHA hash (primary key)
    'title': 'string',
    'description': 'string',
    'category': 'string',
    'pub_date': 'timestamp[ns]',
    'tags': 'string[]',         # Array of tags
    'creator': 'string',
    'source': 'string',
    'language': 'string',
    'image_path': 'string',     # Relative path to image
    'image_caption': 'string',
    'image_credit': 'string',
    'image_name': 'string',
    'processed_at': 'timestamp[ns]',  # When ETL ran
    'batch_hour': 'int64'       # Hour when batch was created
}
```

## 3. Image Storage Strategy

### Organization
- **Base Path**: `data/images/{source}/{date}/{hour}/`
- **Naming**: Keep original filenames or use hash-based names
- **Format**: Preserve original format (JPG, PNG, JPEG)

### Metadata Integration
- Images stored alongside Parquet data
- Image paths referenced in Parquet files
- Image metadata stored in database

### Optimization
- Consider image compression for storage efficiency
- Implement cleanup of old images if needed
- Use relative paths in Parquet files for portability

## 4. Database Schema (DuckDB)

### Tables

#### articles
```sql
CREATE TABLE articles (
    article_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    description TEXT,
    category VARCHAR,
    pub_date TIMESTAMP NOT NULL,
    tags VARCHAR[],           -- DuckDB array type
    creator VARCHAR,
    source_id INTEGER REFERENCES sources(id),
    language VARCHAR(2),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### sources
```sql
CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    display_name VARCHAR,
    rss_url VARCHAR,
    website_url VARCHAR,
    language VARCHAR(2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### categories
```sql
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    display_name VARCHAR,
    source_id INTEGER REFERENCES sources(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### tags
```sql
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### article_tags (junction table)
```sql
CREATE TABLE article_tags (
    article_id VARCHAR REFERENCES articles(article_id),
    tag_id INTEGER REFERENCES tags(id),
    PRIMARY KEY (article_id, tag_id)
);
```

#### images
```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    article_id VARCHAR REFERENCES articles(article_id),
    filename VARCHAR NOT NULL,
    original_path VARCHAR,
    stored_path VARCHAR NOT NULL,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    format VARCHAR,
    caption TEXT,
    credit VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes
```sql
-- Performance indexes
CREATE INDEX idx_articles_pub_date ON articles(pub_date);
CREATE INDEX idx_articles_source ON articles(source_id);
CREATE INDEX idx_articles_category ON articles(category);
CREATE INDEX idx_articles_tags ON article_tags(tag_id);
CREATE INDEX idx_images_article ON images(article_id);

-- Composite indexes for common queries
CREATE INDEX idx_articles_source_date ON articles(source_id, pub_date);
CREATE INDEX idx_articles_date_category ON articles(pub_date, category);
```

## 5. ETL Pipeline Modifications

### New Pipeline Steps

1. **Extract & Transform** (existing)
2. **Generate Parquet Files**
   - Create per-source Parquet files
   - Create unified Parquet file
   - Partition by date/hour
3. **Process Images**
   - Organize images by source/date/hour
   - Update image paths in data
4. **Load Database**
   - Insert articles into DuckDB
   - Insert images metadata
   - Update tags and categories
5. **Cleanup** (optional)
   - Remove old temporary files
   - Update latest symlinks

### Configuration
```yaml
# config file additions
storage:
  parquet:
    base_path: "data/parquet"
    compression: "snappy"  # or "gzip", "brotli"
    row_group_size: 50000

  images:
    base_path: "data/images"
    max_age_days: 30  # cleanup old images

  database:
    path: "data/db/news_mvp.duckdb"
    wal_mode: true
    memory_limit: "2GB"
```

## 6. Benefits of This Approach

### Parquet Advantages
- **Columnar storage**: Fast analytical queries
- **Compression**: Efficient storage
- **Partitioning**: Query only relevant data
- **Schema evolution**: Easy to add new columns
- **Cross-platform**: Readable by multiple tools

### Database Advantages
- **ACID transactions**: Data consistency
- **Rich querying**: Complex joins and aggregations
- **Relationships**: Proper foreign key constraints
- **Indexing**: Fast lookups and searches
- **Concurrent access**: Multiple readers/writers

### Image Management
- **Organized storage**: Easy to find and manage
- **Metadata tracking**: Image info in database
- **Path consistency**: Relative paths for portability
- **Cleanup capability**: Remove old/unused images

## 7. Implementation Priority

1. **Phase 1**: Parquet file generation
2. **Phase 2**: Database schema and loading
3. **Phase 3**: Image management integration
4. **Phase 4**: Query optimization and analytics
5. **Phase 5**: Monitoring and maintenance

This storage plan provides a robust, scalable solution for news data management with both analytical capabilities (Parquet) and transactional consistency (Database).</content>
<parameter name="filePath">c:\code\news-mvp\STORAGE_PLAN.md