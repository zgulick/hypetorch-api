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