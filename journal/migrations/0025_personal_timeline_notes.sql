-- Additive timelines: existing people/events and their fields are never rewritten.
ALTER TABLE personal_people ADD COLUMN meeting_date_precision TEXT NOT NULL DEFAULT 'unknown' CHECK(meeting_date_precision IN ('unknown','year','month','day','range','age'));
ALTER TABLE personal_people ADD COLUMN meeting_date_start TEXT NOT NULL DEFAULT '';
ALTER TABLE personal_people ADD COLUMN meeting_date_end TEXT NOT NULL DEFAULT '';
ALTER TABLE personal_people ADD COLUMN meeting_age_text TEXT NOT NULL DEFAULT '';
CREATE TABLE personal_notes (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 event_id TEXT,
 person_id TEXT,
 creation_id TEXT NOT NULL,
 payload_hash TEXT NOT NULL,
 text TEXT NOT NULL CHECK(length(trim(text)) BETWEEN 1 AND 2000),
 revision INTEGER NOT NULL DEFAULT 1,
 created_at INTEGER NOT NULL,
 updated_at INTEGER NOT NULL,
 CHECK((event_id IS NOT NULL AND person_id IS NULL) OR (event_id IS NULL AND person_id IS NOT NULL)),
 UNIQUE(id,user_id),UNIQUE(user_id,creation_id),
 FOREIGN KEY(event_id,user_id) REFERENCES life_events(id,user_id) ON DELETE CASCADE,
 FOREIGN KEY(person_id,user_id) REFERENCES personal_people(id,user_id) ON DELETE CASCADE
);
CREATE INDEX personal_notes_event ON personal_notes(user_id,event_id,created_at DESC,id);
CREATE INDEX personal_notes_person ON personal_notes(user_id,person_id,created_at DESC,id);
CREATE INDEX life_event_people_person ON life_event_people(user_id,person_id,event_id);
-- Parent sources own their aggregate of dated user notes. Updating their revision
-- invalidates the existing extraction cache and care source revisions as well.
CREATE TRIGGER personal_note_insert AFTER INSERT ON personal_notes BEGIN
 UPDATE personal_people SET revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=NEW.person_id AND user_id=NEW.user_id;
 UPDATE life_events SET revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=NEW.event_id AND user_id=NEW.user_id;
 UPDATE profiles SET version=version+1 WHERE user_id=NEW.user_id;
END;
CREATE TRIGGER personal_note_update AFTER UPDATE OF text ON personal_notes WHEN OLD.text!=NEW.text BEGIN
 UPDATE personal_people SET revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=NEW.person_id AND user_id=NEW.user_id;
 UPDATE life_events SET revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=NEW.event_id AND user_id=NEW.user_id;
 UPDATE profiles SET version=version+1 WHERE user_id=NEW.user_id;
END;
CREATE TRIGGER personal_note_delete AFTER DELETE ON personal_notes BEGIN
 UPDATE personal_people SET revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=OLD.person_id AND user_id=OLD.user_id;
 UPDATE life_events SET revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=OLD.event_id AND user_id=OLD.user_id;
 UPDATE profiles SET version=version+1 WHERE user_id=OLD.user_id;
END;
