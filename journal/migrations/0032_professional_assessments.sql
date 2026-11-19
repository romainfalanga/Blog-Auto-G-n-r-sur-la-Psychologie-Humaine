-- Professional statements are attributed, never generated confirmations.
CREATE TABLE professional_assessments (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 review_id TEXT NOT NULL REFERENCES ambre_reviews(id) ON DELETE CASCADE,
 link_id TEXT REFERENCES care_links(id) ON DELETE SET NULL,
 professional_id TEXT REFERENCES users(id) ON DELETE SET NULL,
 professional_name TEXT NOT NULL,
 conclusion TEXT NOT NULL CHECK(length(conclusion) BETWEEN 1 AND 8000),
 idempotency_key TEXT NOT NULL,
 created_at INTEGER NOT NULL,
 UNIQUE(user_id,professional_id,idempotency_key)
);
CREATE INDEX professional_assessments_owner ON professional_assessments(user_id,created_at,id);
CREATE TRIGGER professional_assessments_insert AFTER INSERT ON professional_assessments BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'diagnoses',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER professional_assessments_delete AFTER DELETE ON professional_assessments
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'diagnoses',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
