-- Additive v2 map; v1 data is retained until the verified deployment (0030).
CREATE TABLE IF NOT EXISTS symptom_maps (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 answers TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(answers) AND json_type(answers)='object'),
 feedback TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(feedback) AND json_type(feedback)='object'),
 notes TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(notes) AND json_type(notes)='object'),
 consent INTEGER NOT NULL DEFAULT 1 CHECK(consent IN (0,1)),
 locked_at INTEGER,
 version INTEGER NOT NULL DEFAULT 1 CHECK(version>=1),
 updated_at INTEGER NOT NULL,
 safety_at INTEGER,
 safety_asked_at INTEGER
);
CREATE TRIGGER care_revision_symptom_maps_insert AFTER INSERT ON symptom_maps
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'symptoms',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_symptom_maps_update AFTER UPDATE OF answers,feedback,notes,consent,locked_at,version,updated_at,safety_at ON symptom_maps
 WHEN EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(NEW.user_id,'symptoms',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER care_revision_symptom_maps_delete AFTER DELETE ON symptom_maps
 WHEN EXISTS(SELECT 1 FROM users WHERE id=OLD.user_id) BEGIN
 INSERT INTO care_source_revisions(user_id,scope,revision) VALUES(OLD.user_id,'symptoms',1)
 ON CONFLICT(user_id,scope) DO UPDATE SET revision=revision+1;
END;
