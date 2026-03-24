-- Centurion Capital LLC - Add market identifier column
-- This migration adds a 'market' column to analysis tables
-- to distinguish between US and Indian stock data.
-- Run this once against an existing database.

-- analysis_runs
ALTER TABLE analysis_runs
    ADD COLUMN IF NOT EXISTS market VARCHAR(10) NOT NULL DEFAULT 'US';
CREATE INDEX IF NOT EXISTS idx_analysis_runs_market ON analysis_runs (market);

-- news_items
ALTER TABLE news_items
    ADD COLUMN IF NOT EXISTS market VARCHAR(10) NOT NULL DEFAULT 'US';
CREATE INDEX IF NOT EXISTS idx_news_items_market ON news_items (market);

-- stock_signals
ALTER TABLE stock_signals
    ADD COLUMN IF NOT EXISTS market VARCHAR(10) NOT NULL DEFAULT 'US';
CREATE INDEX IF NOT EXISTS idx_stock_signals_market ON stock_signals (market);

-- fundamental_metrics
ALTER TABLE fundamental_metrics
    ADD COLUMN IF NOT EXISTS market VARCHAR(10) NOT NULL DEFAULT 'US';
CREATE INDEX IF NOT EXISTS idx_fundamental_metrics_market ON fundamental_metrics (market);

-- backtest_results
ALTER TABLE backtest_results
    ADD COLUMN IF NOT EXISTS market VARCHAR(10) NOT NULL DEFAULT 'US';
CREATE INDEX IF NOT EXISTS idx_backtest_results_market ON backtest_results (market);
