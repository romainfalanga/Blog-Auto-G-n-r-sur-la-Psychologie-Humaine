-- Ambre's written reviews (additive): the initial analysis that starts the conversation,
-- then the "points" requested by the member. Generation is asynchronous and resumable:
-- a running review holds a lease; the minute cron retries expired leases up to a cap.
-- sources records what was available when it was written (map, portrait, people,
-- life events, messages taken into account).
CREATE TABLE ambre_reviews (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 kind TEXT NOT NULL CHECK(kind IN ('initial','point')),
 status TEXT NOT NULL CHECK(status IN ('running','ready','failed')),
 content TEXT NOT NULL DEFAULT '',
 error_code TEXT,
 idempotency_key TEXT NOT NULL,
 sources TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(sources) AND json_type(sources)='object'),
 attempts INTEGER NOT NULL DEFAULT 0,
 lease_token TEXT,
 lease_until INTEGER NOT NULL DEFAULT 0,
 created_at INTEGER NOT NULL,
 completed_at INTEGER,
 UNIQUE(user_id,idempotency_key)
);
CREATE INDEX ambre_reviews_owner ON ambre_reviews(user_id,created_at DESC,id);
-- One review in preparation per member; the partial index also serves the cron scan.
CREATE UNIQUE INDEX ambre_reviews_one_running ON ambre_reviews(user_id) WHERE status='running';
CREATE INDEX ambre_reviews_due ON ambre_reviews(lease_until) WHERE status='running';
