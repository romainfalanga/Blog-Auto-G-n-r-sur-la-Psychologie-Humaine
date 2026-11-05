-- Release old work without deleting published stories, identities or replies.
UPDATE story_requests
SET status='cancelled',credit_status='released',error_code='feature_retired',updated_at=CURRENT_TIMESTAMP
WHERE status IN ('waiting_portrait','queued','running');

UPDATE journal_jobs
SET status='obsolete',lease_token=NULL,lease_until=0,error_code='feature_retired'
WHERE kind IN ('characters','story') AND status IN ('pending','running');

UPDATE chat_proposals
SET status='rejected',revision=revision+1,apply_token=NULL
WHERE kind='story_request' AND status IN ('pending','applying');

-- Rebuild affected portraits from active sources. Once invalidated, the same
-- rows no longer match, so replaying the migration cannot bump them again.
UPDATE profiles SET version=version+1
WHERE EXISTS(SELECT 1 FROM user_portraits up, json_tree(up.portrait,'$.coverage') source
 WHERE up.user_id=profiles.user_id AND source.value LIKE 'feedback:%')
 OR EXISTS(SELECT 1 FROM journal_jobs j WHERE j.user_id=profiles.user_id AND j.kind='portrait' AND (j.status='running' OR (j.status='pending' AND j.native_candidate IS NOT NULL))
  AND EXISTS(SELECT 1 FROM feedback f WHERE f.user_id=profiles.user_id AND trim(f.why||f.feelings||f.correction||f.differences||f.preference||f.avoid)!=''));

DELETE FROM context_source_derivatives WHERE source_id LIKE 'feedback:%';

-- Source updates now invalidate portraits only.
DROP TRIGGER IF EXISTS invalidate_personal_context;
CREATE TRIGGER invalidate_personal_context AFTER UPDATE OF version ON profiles WHEN NEW.version!=OLD.version BEGIN
 DELETE FROM user_portraits WHERE user_id=NEW.user_id;
 UPDATE journal_jobs SET status='obsolete',lease_token=NULL,lease_until=0 WHERE user_id=NEW.user_id AND kind='portrait' AND status!='complete';
END;
