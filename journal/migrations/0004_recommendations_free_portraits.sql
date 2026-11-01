-- Free portraits have an explicit rolling budget; recommendations need no provider call.
CREATE TABLE IF NOT EXISTS portrait_usage (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 window_started INTEGER NOT NULL,
 starts INTEGER NOT NULL DEFAULT 0,
 last_started INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scheduler_state (
 key TEXT PRIMARY KEY,
 counter INTEGER NOT NULL DEFAULT 0
);
