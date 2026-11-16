-- Robustness and load (lot R, plan v4.1). Additive only: no table is rewritten,
-- no row is deleted. Tables and indexes are guarded with IF NOT EXISTS; the two
-- ALTER TABLE statements cannot be guarded in SQLite, so this file is applied once
-- through `wrangler d1 migrations apply` after the baseline (journal/ops/d1-baseline.sql).

-- F10: shared provider cooldown (key-wide or per model), fed by every caller.
CREATE TABLE IF NOT EXISTS provider_state (
 scope TEXT PRIMARY KEY,
 retry_at INTEGER NOT NULL DEFAULT 0,
 reason TEXT,
 updated_at INTEGER NOT NULL
);
-- F12: optional global daily token cap (DAILY_PROVIDER_TOKEN_CAP).
CREATE TABLE IF NOT EXISTS provider_global_budget (
 day TEXT PRIMARY KEY,
 reserved_tokens INTEGER NOT NULL DEFAULT 0,
 calls INTEGER NOT NULL DEFAULT 0
);
-- F13: aggregated counters, fixed code names only, never private text.
CREATE TABLE IF NOT EXISTS service_metrics (
 day TEXT NOT NULL,
 name TEXT NOT NULL,
 value INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(day,name)
);
-- F2: the minute cron only reads live jobs; finished jobs are purged by age.
CREATE INDEX IF NOT EXISTS chat_engine_live ON chat_engine_jobs(lease_until,created_at) WHERE stage NOT IN ('done','failed');
CREATE INDEX IF NOT EXISTS chat_engine_finished ON chat_engine_jobs(updated_at) WHERE stage IN ('done','failed');
-- F3: polling and recent-history reads walk the newest rows of one person only;
-- hourly and daily counters are range reads.
CREATE INDEX IF NOT EXISTS chat_turns_owner_live ON chat_turns(user_id,purged);
CREATE INDEX IF NOT EXISTS chat_turns_owner_time ON chat_turns(user_id,created_at);
-- F9: memory jobs keep their first enqueue time; publication no longer refreshes it.
ALTER TABLE chat_memory_jobs ADD COLUMN queued_at INTEGER NOT NULL DEFAULT 0;
UPDATE chat_memory_jobs SET queued_at=updated_at WHERE queued_at=0;
CREATE INDEX IF NOT EXISTS chat_memory_due ON chat_memory_jobs(lease_until,queued_at);
-- F1: exact stage inputs are kept 30 days (200 detailed traces per person), then
-- replaced by a summary (stages, sizes, tokens, verdicts).
ALTER TABLE chat_turn_traces ADD COLUMN compacted_at INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS chat_traces_detailed ON chat_turn_traces(created_at) WHERE compacted_at=0;
CREATE INDEX IF NOT EXISTS chat_traces_detailed_owner ON chat_turn_traces(user_id,created_at) WHERE compacted_at=0;
-- F11/F14: expired limiter rows and sessions are purged in bounded batches.
CREATE INDEX IF NOT EXISTS rate_limits_expiry ON rate_limits(expires_at);
CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
