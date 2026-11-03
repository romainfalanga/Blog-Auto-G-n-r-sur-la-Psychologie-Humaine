-- Reading suggestions are not psychological evidence and do not bump sources.
-- Keep only the latest 100 public IDs / account and seven days in the Worker.
CREATE TABLE recommendation_exposures (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 article_id TEXT NOT NULL,
 shown_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(user_id,article_id)
);
CREATE INDEX recommendation_exposures_recent ON recommendation_exposures(user_id,shown_at DESC);
