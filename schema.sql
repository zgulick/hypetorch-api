-- schema.sql - Simplified database schema for HypeTorch

-- Set the correct schema
SET search_path TO development;

-- Create entities table
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    type TEXT DEFAULT 'person',
    category TEXT DEFAULT 'Sports',
    subcategory TEXT DEFAULT 'Unrivaled',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create current_metrics table for latest metric values
CREATE TABLE IF NOT EXISTS current_metrics (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id),
    metric_type TEXT NOT NULL,
    value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    time_period TEXT,
    UNIQUE(entity_id, metric_type)
);

-- Create historical_metrics table for historical data
CREATE TABLE IF NOT EXISTS historical_metrics (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id),
    metric_type TEXT NOT NULL,
    value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    time_period TEXT
);

-- Create simplified api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    client_name TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    token_balance INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    permissions TEXT
);

-- Create token_transactions table for tracking usage
CREATE TABLE IF NOT EXISTS token_transactions (
    id SERIAL PRIMARY KEY,
    api_key_id INTEGER NOT NULL REFERENCES api_keys(id),
    amount INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,
    endpoint TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_id TEXT,
    client_ip TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create system_settings table
CREATE TABLE IF NOT EXISTS system_settings (
    id INTEGER PRIMARY KEY,
    dashboardTitle TEXT,
    featuredEntities TEXT,
    defaultTimeframe TEXT,
    enableRodmnScore BOOLEAN,
    enableSentimentAnalysis BOOLEAN,
    enableTalkTimeMetric BOOLEAN,
    enableWikipediaViews BOOLEAN,
    enableRedditMentions BOOLEAN,
    enableGoogleTrends BOOLEAN,
    minEntityDisplayCount INTEGER,
    maxEntityDisplayCount INTEGER,
    refreshInterval INTEGER,
    publicDashboard BOOLEAN,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_category ON entities(category);
CREATE INDEX IF NOT EXISTS idx_entities_subcategory ON entities(subcategory);
CREATE INDEX IF NOT EXISTS idx_current_metrics_entity_id ON current_metrics(entity_id);
CREATE INDEX IF NOT EXISTS idx_current_metrics_metric_type ON current_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_historical_metrics_entity_id ON historical_metrics(entity_id);
CREATE INDEX IF NOT EXISTS idx_historical_metrics_metric_type ON historical_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_historical_metrics_timestamp ON historical_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_token_transactions_api_key_id ON token_transactions(api_key_id);
CREATE INDEX IF NOT EXISTS idx_token_transactions_created_at ON token_transactions(created_at);

-- Podcast Pipeline Tables
-- Create podcast_episodes table for tracking processed episodes
CREATE TABLE IF NOT EXISTS podcast_episodes (
    id SERIAL PRIMARY KEY,
    youtube_video_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    channel_name VARCHAR(255),
    channel_url TEXT,
    episode_url TEXT NOT NULL,
    duration_seconds INTEGER,
    upload_date TIMESTAMP,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_config_id VARCHAR(100),
    audio_file_path TEXT,
    transcript_file_path TEXT,
    processing_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    hype_score_impact DECIMAL(10,2),
    rodmn_score_impact DECIMAL(10,2),
    total_mentions INTEGER DEFAULT 0,
    total_talk_time DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Link podcast episodes to entity mentions
CREATE TABLE IF NOT EXISTS podcast_entity_mentions (
    id SERIAL PRIMARY KEY,
    episode_id INTEGER REFERENCES podcast_episodes(id),
    entity_id INTEGER REFERENCES entities(id),
    mention_count INTEGER DEFAULT 0,
    talk_time_seconds DECIMAL(10,2) DEFAULT 0,
    sentiment_score DECIMAL(5,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Track score changes attributed to specific episodes
CREATE TABLE IF NOT EXISTS podcast_score_attribution (
    id SERIAL PRIMARY KEY,
    episode_id INTEGER REFERENCES podcast_episodes(id),
    entity_id INTEGER REFERENCES entities(id),
    metric_type VARCHAR(50),
    score_before DECIMAL(10,2),
    score_after DECIMAL(10,2),
    score_change DECIMAL(10,2),
    attribution_weight DECIMAL(5,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for podcast tables
CREATE INDEX IF NOT EXISTS idx_podcast_episodes_video_id ON podcast_episodes(youtube_video_id);
CREATE INDEX IF NOT EXISTS idx_podcast_episodes_processed_date ON podcast_episodes(processed_date);
CREATE INDEX IF NOT EXISTS idx_podcast_episodes_status ON podcast_episodes(processing_status);
CREATE INDEX IF NOT EXISTS idx_podcast_episodes_upload_date ON podcast_episodes(upload_date);
CREATE INDEX IF NOT EXISTS idx_podcast_entity_mentions_episode_id ON podcast_entity_mentions(episode_id);
CREATE INDEX IF NOT EXISTS idx_podcast_entity_mentions_entity_id ON podcast_entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_podcast_score_attribution_episode_id ON podcast_score_attribution(episode_id);
CREATE INDEX IF NOT EXISTS idx_podcast_score_attribution_entity_id ON podcast_score_attribution(entity_id);