-- Preserve earlier choices, including explanations, while removing the selection cap.
CREATE TABLE IF NOT EXISTS article_likes (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 article_id TEXT NOT NULL,
 reason TEXT NOT NULL DEFAULT '',
 differences TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(user_id,article_id)
);
INSERT OR IGNORE INTO article_likes(user_id,article_id,reason,differences)
 SELECT user_id,article_id,reason,differences FROM inspiration_selections;
CREATE TRIGGER IF NOT EXISTS article_like_insert AFTER INSERT ON article_likes BEGIN
 UPDATE profiles SET version=version+1 WHERE user_id=NEW.user_id;
END;
CREATE TRIGGER IF NOT EXISTS article_like_update AFTER UPDATE OF reason,differences ON article_likes
 WHEN OLD.reason!=NEW.reason OR OLD.differences!=NEW.differences BEGIN
 UPDATE profiles SET version=version+1 WHERE user_id=NEW.user_id;
END;
CREATE TRIGGER IF NOT EXISTS article_like_delete AFTER DELETE ON article_likes BEGIN
 UPDATE profiles SET version=version+1 WHERE user_id=OLD.user_id;
END;
CREATE TABLE IF NOT EXISTS user_portraits (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 portrait TEXT NOT NULL,
 analyzed_version INTEGER NOT NULL,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS journal_jobs (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 kind TEXT NOT NULL CHECK(kind IN ('portrait','story')),
 source_version INTEGER NOT NULL,
 memory_version INTEGER NOT NULL,
 day TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL DEFAULT 'pending',
 attempts INTEGER NOT NULL DEFAULT 0,
 provider_calls INTEGER NOT NULL DEFAULT 0,
 lease_token TEXT,
 lease_until INTEGER NOT NULL DEFAULT 0,
 error_code TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(user_id,kind,source_version,day)
);
CREATE INDEX IF NOT EXISTS journal_jobs_pending ON journal_jobs(kind,status,lease_until);
