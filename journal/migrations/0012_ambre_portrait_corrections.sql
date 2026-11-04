-- Preserve all conversations and proposals while adding confirmed portrait corrections.
CREATE TABLE chat_proposals_v8 (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,turn_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('self_profile_patch','person_create','person_update','life_event_create','life_event_update','story_request','memory_append')),
 target_id TEXT,expected_revision INTEGER,binding TEXT NOT NULL DEFAULT '{}',
 payload TEXT NOT NULL,before_data TEXT NOT NULL DEFAULT '{}',summary TEXT NOT NULL DEFAULT '',
 revision INTEGER NOT NULL DEFAULT 1,status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','applying','accepted','rejected')),
 apply_token TEXT,result_json TEXT,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(id,user_id),FOREIGN KEY(turn_id,user_id) REFERENCES chat_turns(id,user_id) ON DELETE CASCADE
);
INSERT INTO chat_proposals_v8 SELECT * FROM chat_proposals;
DROP TABLE chat_proposals;
ALTER TABLE chat_proposals_v8 RENAME TO chat_proposals;
CREATE INDEX chat_proposals_owner ON chat_proposals(user_id,created_at,id);
CREATE INDEX chat_provider_cooldown ON chat_turns(status,lease_until) WHERE status='failed' AND error_code LIKE 'provider_http_%';
-- Recompute older generated prose from current declarations; never change source facts.
UPDATE profiles SET version=version+1 WHERE EXISTS (
 SELECT 1 FROM user_portraits up WHERE up.user_id=profiles.user_id
 AND COALESCE(json_extract(up.portrait,'$.analysis_version'),0)<3
);
