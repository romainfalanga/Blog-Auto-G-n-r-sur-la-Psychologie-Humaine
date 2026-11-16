-- Durable message stages and bounded, inspectable conversation memory.
CREATE TABLE chat_memory (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 summary TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(summary)),
 through_turn TEXT, turns_covered INTEGER NOT NULL DEFAULT 0,
 updated_at INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE chat_engine_jobs (
 turn_id TEXT PRIMARY KEY REFERENCES chat_turns(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 stage TEXT NOT NULL DEFAULT 'analysis' CHECK(stage IN ('analysis','writing','review','rewrite','publish','done','failed')),
 analysis TEXT NOT NULL DEFAULT '{}', draft TEXT NOT NULL DEFAULT '{}', review TEXT NOT NULL DEFAULT '{}',
 source_hash TEXT NOT NULL, context TEXT NOT NULL DEFAULT '{}',
 lease_token TEXT, lease_until INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
 input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
 reserved_tokens INTEGER NOT NULL DEFAULT 0, cost REAL NOT NULL DEFAULT 0, duration_ms INTEGER NOT NULL DEFAULT 0,
 created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE INDEX chat_engine_due ON chat_engine_jobs(stage,lease_until,updated_at);
CREATE INDEX chat_engine_owner ON chat_engine_jobs(user_id,created_at DESC);
CREATE TABLE chat_turn_traces (
 turn_id TEXT PRIMARY KEY REFERENCES chat_turns(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 trace TEXT NOT NULL CHECK(json_valid(trace)), created_at INTEGER NOT NULL
);
CREATE INDEX chat_traces_owner ON chat_turn_traces(user_id,created_at DESC);
CREATE TABLE chat_memory_jobs (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 lease_token TEXT,lease_until INTEGER NOT NULL DEFAULT 0,revision INTEGER NOT NULL DEFAULT 0,
 attempts INTEGER NOT NULL DEFAULT 0,updated_at INTEGER NOT NULL
);
