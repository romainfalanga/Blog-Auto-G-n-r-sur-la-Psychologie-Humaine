ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'member' CHECK(role IN ('member','psychologist'));
CREATE TABLE care_settings (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 has_psychologist INTEGER CHECK(has_psychologist IN (0,1)),
 frequency_days INTEGER CHECK(frequency_days BETWEEN 1 AND 365),updated_at INTEGER NOT NULL
);
CREATE TABLE professional_profiles (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 name TEXT NOT NULL,title TEXT NOT NULL,registration_number TEXT NOT NULL,updated_at INTEGER NOT NULL
);
CREATE TABLE care_links (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 psychologist_id TEXT REFERENCES users(id) ON DELETE CASCADE,
 target_email TEXT NOT NULL,code_hash TEXT NOT NULL UNIQUE,
 scopes TEXT NOT NULL CHECK(json_valid(scopes)),
 status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','active','revoked','expired')),
 revision INTEGER NOT NULL DEFAULT 1,created_at INTEGER NOT NULL,expires_at INTEGER NOT NULL,activated_at INTEGER,revoked_at INTEGER
);
CREATE INDEX care_links_member ON care_links(user_id,status,created_at DESC);
CREATE INDEX care_links_psychologist ON care_links(psychologist_id,status,created_at DESC);
CREATE TABLE care_sessions (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 session_at INTEGER NOT NULL,created_by TEXT NOT NULL,created_at INTEGER NOT NULL,
 revision INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX care_sessions_owner ON care_sessions(user_id,session_at DESC,id);
CREATE INDEX care_sessions_due ON care_sessions(session_at,user_id);
CREATE TABLE care_access_log (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 actor_id TEXT REFERENCES users(id) ON DELETE SET NULL,link_id TEXT REFERENCES care_links(id) ON DELETE SET NULL,
 action TEXT NOT NULL,scope TEXT NOT NULL,created_at INTEGER NOT NULL
);
CREATE INDEX care_access_owner ON care_access_log(user_id,created_at DESC,id);
CREATE TABLE care_briefs (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 link_id TEXT NOT NULL REFERENCES care_links(id) ON DELETE CASCADE,
 session_id TEXT NOT NULL REFERENCES care_sessions(id) ON DELETE CASCADE,
 link_revision INTEGER NOT NULL,session_revision INTEGER NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','ready','failed','revoked')),
 content TEXT NOT NULL DEFAULT '',source_hash TEXT NOT NULL DEFAULT '',coverage TEXT NOT NULL DEFAULT '{}',
 lease_token TEXT,lease_until INTEGER NOT NULL DEFAULT 0,attempts INTEGER NOT NULL DEFAULT 0,
 cursor INTEGER NOT NULL DEFAULT 0,notes TEXT NOT NULL DEFAULT '[]',error_code TEXT,
 created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,
 UNIQUE(link_id,session_id,link_revision,session_revision)
);
CREATE INDEX care_briefs_due ON care_briefs(status,lease_until,updated_at);
CREATE INDEX care_briefs_owner ON care_briefs(user_id,created_at DESC,id);
CREATE TABLE care_daily_budget (user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,day TEXT NOT NULL,count INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(user_id,day));

-- Scope-specific revisions make removals, edits and purges invalidate prepared points
-- without incorporating unshared memory or another unshared part of the dossier.
CREATE TABLE care_source_revisions (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 scope TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 1,PRIMARY KEY(user_id,scope)
);
-- Automatic visits rotate independently of generation state and daily budget.
CREATE TABLE care_brief_checks (
 link_id TEXT NOT NULL REFERENCES care_links(id) ON DELETE CASCADE,
 session_id TEXT NOT NULL REFERENCES care_sessions(id) ON DELETE CASCADE,
 checked_at INTEGER NOT NULL,PRIMARY KEY(link_id,session_id)
);
CREATE TRIGGER care_revision_chat_turns_insert AFTER INSERT ON chat_turns
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'conversation',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_chat_turns_update AFTER UPDATE OF content,status,purged,created_at ON chat_turns
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'conversation',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_chat_turns_delete AFTER DELETE ON chat_turns
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'conversation',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_personal_people_insert AFTER INSERT ON personal_people
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'people',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_personal_people_update AFTER UPDATE ON personal_people
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'people',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_personal_people_delete AFTER DELETE ON personal_people
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'people',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_life_events_insert AFTER INSERT ON life_events
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'life_events',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_life_events_update AFTER UPDATE ON life_events
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'life_events',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_life_events_delete AFTER DELETE ON life_events
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'life_events',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_ambre_reviews_insert AFTER INSERT ON ambre_reviews
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'reviews',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_ambre_reviews_update AFTER UPDATE OF content,status,completed_at ON ambre_reviews
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'reviews',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_ambre_reviews_delete AFTER DELETE ON ambre_reviews
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'reviews',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_profiles_insert AFTER INSERT ON profiles
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'portrait',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_profiles_update AFTER UPDATE OF self_data ON profiles
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'portrait',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_profiles_delete AFTER DELETE ON profiles
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'portrait',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_user_portraits_insert AFTER INSERT ON user_portraits
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'portrait',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_user_portraits_update AFTER UPDATE ON user_portraits
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'portrait',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_user_portraits_delete AFTER DELETE ON user_portraits
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'portrait',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_cartographies_insert AFTER INSERT ON cartographies
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'cartography',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_cartographies_update AFTER UPDATE ON cartographies
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'cartography',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_cartographies_delete AFTER DELETE ON cartographies
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'cartography',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;

