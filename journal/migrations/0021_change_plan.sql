-- The document contains choices, proposals and reports, not inferred diagnoses.
-- Control state survives document erasure; an old task cannot recreate it.
CREATE TABLE change_plans (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 revision INTEGER NOT NULL DEFAULT 0,
 epoch INTEGER NOT NULL DEFAULT 0,
 context_generation INTEGER NOT NULL DEFAULT 0,
 proposals_enabled INTEGER NOT NULL DEFAULT 1 CHECK(proposals_enabled IN (0,1)),
 reminders_paused INTEGER NOT NULL DEFAULT 0 CHECK(reminders_paused IN (0,1)),
 exploration_reconsent_required INTEGER NOT NULL DEFAULT 0 CHECK(exploration_reconsent_required IN (0,1)),
 past_safety_signal INTEGER NOT NULL DEFAULT 0 CHECK(past_safety_signal IN (0,1)),
 document_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(document_json) AND length(CAST(document_json AS BLOB))<=262144),
 last_mutation TEXT NOT NULL DEFAULT '',
 updated_at INTEGER NOT NULL
);
ALTER TABLE chat_turns ADD COLUMN context_generation INTEGER NOT NULL DEFAULT 0;
CREATE INDEX chat_turns_generation ON chat_turns(user_id,context_generation,created_at);

CREATE TABLE change_requests (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 request_key TEXT NOT NULL,
 epoch INTEGER NOT NULL,
 request_hash TEXT NOT NULL,
 entity_kind TEXT NOT NULL,
 entity_id TEXT NOT NULL,
 deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0,1)),
 created_at INTEGER NOT NULL,
 PRIMARY KEY(user_id,request_key)
);
CREATE INDEX change_requests_entity ON change_requests(user_id,entity_kind,entity_id);

-- Only IDs and revisions: never retain deleted content in a dependency log.
CREATE TABLE change_artifact_refs (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 artifact_kind TEXT NOT NULL,
 artifact_id TEXT NOT NULL,
 entity_kind TEXT NOT NULL,
 entity_id TEXT NOT NULL,
 entity_revision INTEGER NOT NULL,
 context_generation INTEGER NOT NULL,
 PRIMARY KEY(user_id,artifact_kind,artifact_id,entity_kind,entity_id)
);
CREATE INDEX change_refs_entity ON change_artifact_refs(user_id,entity_id);

CREATE TRIGGER care_revision_change_plans_insert AFTER INSERT ON change_plans
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'plan',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_change_plans_update AFTER UPDATE ON change_plans
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) AND (
  COALESCE(json_extract(NEW.document_json,'$.goals'),'[]') != COALESCE(json_extract(OLD.document_json,'$.goals'),'[]') OR
  COALESCE(json_extract(NEW.document_json,'$.steps'),'[]') != COALESCE(json_extract(OLD.document_json,'$.steps'),'[]') OR
  COALESCE(json_extract(NEW.document_json,'$.trials'),'[]') != COALESCE(json_extract(OLD.document_json,'$.trials'),'[]')
 ) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'plan',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_change_plans_delete AFTER DELETE ON change_plans
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'plan',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
