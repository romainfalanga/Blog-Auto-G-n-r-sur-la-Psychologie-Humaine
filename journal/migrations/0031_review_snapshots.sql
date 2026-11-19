-- Personal state captured once for a new review. Older reviews have no retroactive snapshot.
CREATE TABLE review_snapshots (
 review_id TEXT PRIMARY KEY REFERENCES ambre_reviews(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 captured_at INTEGER NOT NULL,
 schema_version INTEGER NOT NULL CHECK(schema_version=1),
 payload TEXT NOT NULL CHECK(json_valid(payload) AND json_type(payload)='object'),
 assessment TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(assessment) AND json_type(assessment)='object'),
 source_stamp TEXT NOT NULL,
 UNIQUE(review_id,user_id)
);
CREATE INDEX review_snapshots_owner ON review_snapshots(user_id,captured_at DESC,review_id);
CREATE TRIGGER review_snapshot_immutable BEFORE UPDATE ON review_snapshots BEGIN
 SELECT RAISE(ABORT,'Review snapshots are immutable');
END;

-- Immutability does not override a deliberate withdrawal or forgetting boundary.
CREATE TRIGGER review_snapshots_forget AFTER UPDATE OF context_generation ON change_plans
 WHEN NEW.context_generation<>OLD.context_generation BEGIN
 DELETE FROM review_snapshots WHERE user_id=NEW.user_id;
END;
CREATE TRIGGER review_snapshots_withdraw AFTER UPDATE OF consent ON chat_settings
 WHEN NEW.consent=0 BEGIN
 DELETE FROM review_snapshots WHERE user_id=NEW.user_id;
END;
CREATE TRIGGER review_snapshots_invalidated AFTER UPDATE OF status,error_code ON ambre_reviews
 WHEN NEW.status='failed' AND NEW.error_code IN ('sources_changed','consent_withdrawn') BEGIN
 DELETE FROM review_snapshots WHERE review_id=NEW.id AND user_id=NEW.user_id;
END;
