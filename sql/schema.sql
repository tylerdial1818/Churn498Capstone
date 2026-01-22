-- Retain: Streaming Service Database Schema
-- Generated for PostgreSQL
-- Import CSVs using: \COPY table_name FROM 'file.csv' CSV HEADER

-- Drop tables if they exist (for clean re-import)
DROP TABLE IF EXISTS streaming_events CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS content_catalog CASCADE;
DROP TABLE IF EXISTS accounts CASCADE;

-- Accounts table
CREATE TABLE accounts (
    account_id VARCHAR(12) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    signup_date DATE NOT NULL,
    country VARCHAR(2) NOT NULL DEFAULT 'US',
    age INTEGER CHECK (age >= 18 AND age <= 120),
    gender VARCHAR(10)
);

-- Content catalog table
CREATE TABLE content_catalog (
    content_id VARCHAR(10) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    genre VARCHAR(50) NOT NULL,
    content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('movie', 'series')),
    release_year INTEGER CHECK (release_year >= 1900 AND release_year <= 2030),
    duration_minutes INTEGER CHECK (duration_minutes > 0)
);

-- Subscriptions table
CREATE TABLE subscriptions (
    subscription_id VARCHAR(14) PRIMARY KEY,
    account_id VARCHAR(12) NOT NULL REFERENCES accounts(account_id),
    plan_type VARCHAR(25) NOT NULL CHECK (plan_type IN ('Regular', 'Premium', 'Premium-Multi-Screen')),
    start_date DATE NOT NULL,
    end_date DATE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'cancelled', 'payment_failed')),
    cancel_reason VARCHAR(50),
    previous_plan VARCHAR(25),
    parental_control_enabled BOOLEAN DEFAULT FALSE,
    watchlist_size INTEGER DEFAULT 0,
    app_rating INTEGER CHECK (app_rating >= 1 AND app_rating <= 5),
    app_review_text TEXT
);

-- Payments table
CREATE TABLE payments (
    payment_id VARCHAR(14) PRIMARY KEY,
    account_id VARCHAR(12) NOT NULL REFERENCES accounts(account_id),
    payment_date DATE NOT NULL,
    amount DECIMAL(8,2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    payment_method VARCHAR(20) NOT NULL,
    status VARCHAR(10) NOT NULL CHECK (status IN ('success', 'failed')),
    failure_reason VARCHAR(50)
);

-- Support tickets table
CREATE TABLE support_tickets (
    ticket_id VARCHAR(12) PRIMARY KEY,
    account_id VARCHAR(12) NOT NULL REFERENCES accounts(account_id),
    created_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    category VARCHAR(50) NOT NULL,
    priority VARCHAR(10) NOT NULL CHECK (priority IN ('low', 'medium', 'high'))
);

-- Streaming events table
CREATE TABLE streaming_events (
    event_id VARCHAR(16) PRIMARY KEY,
    account_id VARCHAR(12) NOT NULL REFERENCES accounts(account_id),
    event_timestamp TIMESTAMP NOT NULL,
    content_id VARCHAR(10) NOT NULL REFERENCES content_catalog(content_id),
    device_type VARCHAR(20) NOT NULL,
    watch_duration_minutes DECIMAL(6,1) NOT NULL,
    login_lat DECIMAL(8,4),
    login_long DECIMAL(8,4)
);

-- Indexes for common queries
CREATE INDEX idx_accounts_signup ON accounts(signup_date);
CREATE INDEX idx_subscriptions_account ON subscriptions(account_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);
CREATE INDEX idx_payments_account ON payments(account_id);
CREATE INDEX idx_payments_date ON payments(payment_date);
CREATE INDEX idx_tickets_account ON support_tickets(account_id);
CREATE INDEX idx_events_account ON streaming_events(account_id);
CREATE INDEX idx_events_timestamp ON streaming_events(event_timestamp);
CREATE INDEX idx_events_content ON streaming_events(content_id);
