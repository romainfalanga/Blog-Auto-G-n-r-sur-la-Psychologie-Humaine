-- Conversations remain separate from canonical personal sources until accepted.
CREATE TABLE chat_settings (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 consent INTEGER NOT NULL DEFAULT 0 CHECK(consent IN (0,1)),
 revision INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE chat_turns (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 idempotency_key TEXT NOT NULL,payload_hash TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
 credit_status TEXT NOT NULL CHECK(credit_status IN ('reserved','consumed','released')),
 week_start TEXT NOT NULL,source_version INTEGER NOT NULL,consent_revision INTEGER NOT NULL,
 content TEXT NOT NULL DEFAULT '',response TEXT NOT NULL DEFAULT '',
 lease_token TEXT,lease_until INTEGER NOT NULL DEFAULT 0,error_code TEXT,
 input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0,observed_cost REAL NOT NULL DEFAULT 0,
 purged INTEGER NOT NULL DEFAULT 0 CHECK(purged IN (0,1)),created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(user_id,idempotency_key),UNIQUE(id,user_id),
 CHECK((status='running' AND credit_status='reserved') OR (status='completed' AND credit_status='consumed') OR (status='failed' AND credit_status='released'))
);
CREATE UNIQUE INDEX chat_one_running ON chat_turns(user_id) WHERE status='running';
CREATE INDEX chat_quota ON chat_turns(user_id,week_start,credit_status);
CREATE TABLE chat_proposals (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,turn_id TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('self_profile_patch','person_create','person_update','life_event_create','life_event_update','story_request')),
 target_id TEXT,expected_revision INTEGER,binding TEXT NOT NULL DEFAULT '{}',
 payload TEXT NOT NULL,before_data TEXT NOT NULL DEFAULT '{}',summary TEXT NOT NULL DEFAULT '',
 revision INTEGER NOT NULL DEFAULT 1,status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','applying','accepted','rejected')),
 apply_token TEXT,result_json TEXT,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(id,user_id),FOREIGN KEY(turn_id,user_id) REFERENCES chat_turns(id,user_id) ON DELETE CASCADE
);
CREATE INDEX chat_proposals_owner ON chat_proposals(user_id,created_at,id);
