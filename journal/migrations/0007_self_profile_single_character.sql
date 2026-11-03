-- Preserve identities, histories and consumed weekly credits.
ALTER TABLE profiles ADD COLUMN self_data TEXT NOT NULL DEFAULT '{}';
ALTER TABLE profiles ADD COLUMN self_revision INTEGER NOT NULL DEFAULT 1;
ALTER TABLE personal_characters ADD COLUMN is_primary INTEGER NOT NULL DEFAULT 0 CHECK(is_primary IN (0,1));
UPDATE personal_characters SET is_primary=1 WHERE slot=(SELECT min(c.slot) FROM personal_characters c WHERE c.user_id=personal_characters.user_id);
CREATE UNIQUE INDEX one_primary_character ON personal_characters(user_id) WHERE is_primary=1;
UPDATE story_requests SET status='cancelled',credit_status='released',error_code='character_archived' WHERE status IN ('waiting_portrait','queued','running') AND character_id IN(SELECT id FROM personal_characters WHERE is_primary=0);
UPDATE journal_jobs SET status='obsolete',lease_token=NULL,lease_until=0 WHERE kind='story' AND status!='complete' AND character_id IN(SELECT id FROM personal_characters WHERE is_primary=0);
-- Old character preparations may carry a three-slot response; revoke their leases.
UPDATE journal_jobs SET status='obsolete',lease_token=NULL,lease_until=0 WHERE kind='characters' AND status!='complete';
CREATE TRIGGER invalidate_self_profile AFTER UPDATE OF self_data ON profiles WHEN OLD.self_data!=NEW.self_data BEGIN
 DELETE FROM context_source_derivatives WHERE user_id=NEW.user_id AND source_id='profile:'||NEW.user_id;
END;
-- Obsolete preparations are terminal, including revoked V2 leases. Never
-- resurrect them on a later source edit (the live-job unique index forbids it).
DROP TRIGGER invalidate_personal_context;
CREATE TRIGGER invalidate_personal_context AFTER UPDATE OF version ON profiles WHEN NEW.version!=OLD.version BEGIN
 DELETE FROM user_portraits WHERE user_id=NEW.user_id;
 UPDATE journal_jobs SET status='obsolete',lease_token=NULL,lease_until=0 WHERE user_id=NEW.user_id AND kind='portrait' AND status!='complete';
 UPDATE journal_jobs SET status='pending',source_version=NEW.version,lease_token=NULL,lease_until=0 WHERE user_id=NEW.user_id AND kind='characters' AND status NOT IN ('complete','failed','obsolete');
 UPDATE story_requests SET status='waiting_portrait',source_version=NEW.version,error_code='context_changed',updated_at=CURRENT_TIMESTAMP WHERE user_id=NEW.user_id AND status IN ('waiting_portrait','queued','running');
 UPDATE journal_jobs SET status='pending',source_version=NEW.version,lease_token=NULL,lease_until=0,error_code='context_changed' WHERE user_id=NEW.user_id AND kind='story' AND status NOT IN ('complete','failed','obsolete');
END;
