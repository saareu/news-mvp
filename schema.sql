-- News MVP Database Schema (DuckDB)
-- This file contains the complete database schema for the news analytics system

-- ===========================================
-- TABLES
-- ===========================================

-- Sources table (ynet, hayom, haaretz, etc.)
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    display_name VARCHAR,
    rss_url VARCHAR,
    website_url VARCHAR,
    language VARCHAR(2) DEFAULT 'he',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    display_name VARCHAR,
    source_id INTEGER REFERENCES sources(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, source_id)
);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main articles table
CREATE TABLE IF NOT EXISTS articles (
    id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    pubDate TIMESTAMP NOT NULL,
    source_id INTEGER REFERENCES sources(id) NOT NULL,
    description TEXT,
    category VARCHAR,
    tags VARCHAR[],  -- DuckDB array type
    creator VARCHAR,
    language VARCHAR(2) DEFAULT 'he',
    image VARCHAR,
    imageCaption TEXT,
    imageCredit VARCHAR,
    imageName VARCHAR,
    imageBlob BLOB,
    guid VARCHAR,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    batch_hour INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Article tags junction table
CREATE TABLE IF NOT EXISTS article_tags (
    article_id VARCHAR REFERENCES articles(id),
    tag_id INTEGER REFERENCES tags(id),
    PRIMARY KEY (article_id, tag_id)
);

-- Images table
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    article_id VARCHAR REFERENCES articles(id),
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

-- ETL batches tracking
CREATE TABLE IF NOT EXISTS etl_batches (
    id INTEGER PRIMARY KEY,
    batch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_name VARCHAR,
    articles_count INTEGER DEFAULT 0,
    images_count INTEGER DEFAULT 0,
    status VARCHAR DEFAULT 'running', -- running, completed, failed
    error_message TEXT,
    processing_time_seconds FLOAT
);

-- Embeddings (draft schema) - one vector per article (latest)
-- vector stored as FLOAT[]; model info for reproducibility
CREATE TABLE IF NOT EXISTS embeddings (
    article_id VARCHAR REFERENCES articles(id),
    model VARCHAR NOT NULL,
    dim INTEGER NOT NULL,
    vector FLOAT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (article_id, model)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

-- ===========================================
-- INDEXES
-- ===========================================

-- Articles indexes
CREATE INDEX IF NOT EXISTS idx_articles_pubDate ON articles(pubDate);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_articles_source_pubDate ON articles(source_id, pubDate DESC);
CREATE INDEX IF NOT EXISTS idx_articles_pubDate_category ON articles(pubDate DESC, category);
CREATE INDEX IF NOT EXISTS idx_articles_source_category ON articles(source_id, category);

-- Tags indexes
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_article_tags_article ON article_tags(article_id);
CREATE INDEX IF NOT EXISTS idx_article_tags_tag ON article_tags(tag_id);

-- Images indexes
CREATE INDEX IF NOT EXISTS idx_images_article ON images(article_id);
CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename);

-- ETL batches indexes
CREATE INDEX IF NOT EXISTS idx_etl_batches_timestamp ON etl_batches(batch_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_etl_batches_source ON etl_batches(source_name);
CREATE INDEX IF NOT EXISTS idx_etl_batches_status ON etl_batches(status);

-- ===========================================
-- VIEWS
-- ===========================================

-- Articles with source information
CREATE VIEW IF NOT EXISTS articles_with_source AS
SELECT
    a.*,
    s.name as source_name,
    s.display_name as source_display_name,
    s.language as source_language
FROM articles a
JOIN sources s ON a.source_id = s.id;

-- Articles with tags as comma-separated string
CREATE VIEW IF NOT EXISTS articles_with_tags AS
SELECT
    a.*,
    CASE
        WHEN a.tags IS NOT NULL AND LENGTH(a.tags) > 0
        THEN array_to_string(a.tags, ', ')
        ELSE ''
    END as tags_string,
    ARRAY_LENGTH(a.tags) as tags_count
FROM articles a;

-- Recent articles view
CREATE VIEW IF NOT EXISTS recent_articles AS
SELECT * FROM articles
WHERE pubDate >= (CURRENT_TIMESTAMP - INTERVAL 24 HOUR)
ORDER BY pubDate DESC;

-- Article statistics by source
CREATE VIEW IF NOT EXISTS article_stats_by_source AS
SELECT
    s.name as source_name,
    COUNT(a.id) as total_articles,
    COUNT(DISTINCT a.category) as categories_count,
    MIN(a.pubDate) as oldest_article,
    MAX(a.pubDate) as newest_article,
    AVG(LENGTH(a.title)) as avg_title_length,
    AVG(LENGTH(a.description)) as avg_description_length
FROM sources s
LEFT JOIN articles a ON s.id = a.source_id
GROUP BY s.id, s.name;

-- ===========================================
-- FUNCTIONS
-- ===========================================

-- Function to get articles by date range
CREATE OR REPLACE MACRO get_articles_by_date(start_date, end_date) AS TABLE (
    SELECT * FROM articles
    WHERE pubDate BETWEEN start_date AND end_date
    ORDER BY pubDate DESC
);

-- Function to get articles by source and date
CREATE OR REPLACE MACRO get_articles_by_source_date(source_name, start_date, end_date) AS TABLE (
    SELECT a.* FROM articles a
    JOIN sources s ON a.source_id = s.id
    WHERE s.name = source_name
    AND a.pubDate BETWEEN start_date AND end_date
    ORDER BY a.pubDate DESC
);

-- Function to search articles by keyword
CREATE OR REPLACE MACRO search_articles(keyword) AS TABLE (
     SELECT * FROM articles
     WHERE LOWER(title) LIKE LOWER('%' || keyword || '%')
         OR LOWER(description) LIKE LOWER('%' || keyword || '%')
     ORDER BY pubDate DESC
);

-- ===========================================
-- INITIAL DATA
-- ===========================================

-- Insert initial sources
INSERT OR IGNORE INTO sources (id, name, display_name, rss_url, website_url, language) VALUES
(1, 'ynet', 'Ynet', 'https://www.ynet.co.il/Integration/StoryRss2.xml', 'https://www.ynet.co.il', 'he'),
(2, 'hayom', 'Israel Hayom', 'https://www.israelhayom.co.il/rss.xml', 'https://www.israelhayom.co.il', 'he'),
(3, 'haaretz', 'Haaretz', 'https://www.haaretz.co.il/rss.xml', 'https://www.haaretz.co.il', 'he');

-- Schema versioning
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings + recent articles view (24 hours)
CREATE VIEW IF NOT EXISTS mv_recent_articles_with_embeddings AS
SELECT a.id, a.title, a.pubDate, a.source_id, e.model, e.dim, e.vector
FROM articles a
JOIN embeddings e ON e.article_id = a.id
WHERE a.pubDate >= (CURRENT_TIMESTAMP - INTERVAL 24 HOUR);

-- ===========================================
-- SAMPLE QUERIES
-- ===========================================

/*
-- Get latest articles from all sources
SELECT * FROM recent_articles LIMIT 10;

-- Get articles by source
SELECT * FROM get_articles_by_source_date('ynet', '2025-09-16', '2025-09-17');

-- Search for articles containing specific keyword
SELECT * FROM search_articles('politics');

-- Get statistics by source
SELECT * FROM article_stats_by_source;

-- Get articles with their source information
SELECT * FROM articles_with_source LIMIT 5;

-- Get articles with tags as string
SELECT id, title, tags_string, tags_count
FROM articles_with_tags
WHERE tags_count > 0
LIMIT 10;
*/