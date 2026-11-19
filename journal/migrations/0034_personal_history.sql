-- Historique des modifications de Mes proches et de Ma ligne de vie : proches, moments et notes datées.
-- Additive only: one table, one index and triggers. No existing row is rewritten, no existing trigger is
-- replaced, revisions, payload hashes, care_revision_* and derivative caches are untouched.
-- Each modification stores the state it replaces (OLD) as JSON; a creation stores the created state and
-- marks the history as complete since creation. Elements saved before this migration have no earlier
-- history: their first modification records the state they had until then.
-- Only changes of content are recorded: a note added to a person or a moment bumps the parent's revision
-- (personal_note_* triggers) without changing its payload_hash, and records nothing.
-- Deleting an element erases its history (the owner's deletion is an erasure); deleting the account
-- erases everything (ON DELETE CASCADE). At most 50 versions are kept per element, the oldest go first.
-- The history is read by its owner only (GET /api/history) and exported; it is never sent to Ambre.
-- The Worker probes the table (historyReady) and answers « pas encore disponible » until it exists.
CREATE TABLE personal_history (
 id INTEGER PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 entity_type TEXT NOT NULL CHECK(entity_type IN ('person','event','note')),
 entity_id TEXT NOT NULL,
 revision INTEGER NOT NULL,
 action TEXT NOT NULL CHECK(action IN ('create','update')),
 snapshot TEXT NOT NULL CHECK(json_valid(snapshot)),
 changed_at INTEGER NOT NULL
);
CREATE INDEX personal_history_entity ON personal_history(user_id,entity_type,entity_id,id);
CREATE TRIGGER personal_history_cap AFTER INSERT ON personal_history BEGIN
 DELETE FROM personal_history WHERE user_id=NEW.user_id AND entity_type=NEW.entity_type AND entity_id=NEW.entity_id
  AND id<=(SELECT id FROM personal_history WHERE user_id=NEW.user_id AND entity_type=NEW.entity_type AND entity_id=NEW.entity_id ORDER BY id DESC LIMIT 1 OFFSET 50);
END;
CREATE TRIGGER personal_history_person_insert AFTER INSERT ON personal_people BEGIN
 INSERT INTO personal_history(user_id,entity_type,entity_id,revision,action,snapshot,changed_at) VALUES(NEW.user_id,'person',NEW.id,NEW.revision,'create',
  json_object('display_name',NEW.display_name,'role',NEW.role,'custom_role',NEW.custom_role,'relationship_text',NEW.relationship_text,'period_start',NEW.period_start,'period_end',NEW.period_end,
   'bond_quality',NEW.bond_quality,'bond_note',NEW.bond_note,'presence_now',NEW.presence_now,'helps',NEW.helps,'hurts',NEW.hurts,'evolution',NEW.evolution,'feels_safe',NEW.feels_safe,
   'meeting_date_precision',NEW.meeting_date_precision,'meeting_date_start',NEW.meeting_date_start,'meeting_date_end',NEW.meeting_date_end,'meeting_age_text',NEW.meeting_age_text),
  CAST(strftime('%s','now') AS INTEGER));
END;
CREATE TRIGGER personal_history_person_update AFTER UPDATE ON personal_people WHEN OLD.payload_hash IS NOT NEW.payload_hash BEGIN
 INSERT INTO personal_history(user_id,entity_type,entity_id,revision,action,snapshot,changed_at) VALUES(OLD.user_id,'person',OLD.id,OLD.revision,'update',
  json_object('display_name',OLD.display_name,'role',OLD.role,'custom_role',OLD.custom_role,'relationship_text',OLD.relationship_text,'period_start',OLD.period_start,'period_end',OLD.period_end,
   'bond_quality',OLD.bond_quality,'bond_note',OLD.bond_note,'presence_now',OLD.presence_now,'helps',OLD.helps,'hurts',OLD.hurts,'evolution',OLD.evolution,'feels_safe',OLD.feels_safe,
   'meeting_date_precision',OLD.meeting_date_precision,'meeting_date_start',OLD.meeting_date_start,'meeting_date_end',OLD.meeting_date_end,'meeting_age_text',OLD.meeting_age_text),
  CAST(strftime('%s','now') AS INTEGER));
END;
CREATE TRIGGER personal_history_person_delete AFTER DELETE ON personal_people BEGIN
 DELETE FROM personal_history WHERE user_id=OLD.user_id AND entity_type='person' AND entity_id=OLD.id;
END;
-- The links of a moment are written after its row in the same batch: a creation stores no people,
-- a modification stores the people linked until then (the links are replaced after the UPDATE).
CREATE TRIGGER personal_history_event_insert AFTER INSERT ON life_events BEGIN
 INSERT INTO personal_history(user_id,entity_type,entity_id,revision,action,snapshot,changed_at) VALUES(NEW.user_id,'event',NEW.id,NEW.revision,'create',
  json_object('title',NEW.title,'kind',NEW.kind,'description',NEW.description,'meaning_now',NEW.meaning_now,'date_precision',NEW.date_precision,'date_start',NEW.date_start,'date_end',NEW.date_end,'age_text',NEW.age_text,
   'current_importance',NEW.current_importance,'significance',NEW.significance,'felt_then',NEW.felt_then,'felt_now',NEW.felt_now,'impact',NEW.impact,'support_level',NEW.support_level,'support_text',NEW.support_text,'exposure',NEW.exposure),
  CAST(strftime('%s','now') AS INTEGER));
END;
CREATE TRIGGER personal_history_event_update AFTER UPDATE ON life_events WHEN OLD.payload_hash IS NOT NEW.payload_hash BEGIN
 INSERT INTO personal_history(user_id,entity_type,entity_id,revision,action,snapshot,changed_at) VALUES(OLD.user_id,'event',OLD.id,OLD.revision,'update',
  json_object('title',OLD.title,'kind',OLD.kind,'description',OLD.description,'meaning_now',OLD.meaning_now,'date_precision',OLD.date_precision,'date_start',OLD.date_start,'date_end',OLD.date_end,'age_text',OLD.age_text,
   'current_importance',OLD.current_importance,'significance',OLD.significance,'felt_then',OLD.felt_then,'felt_now',OLD.felt_now,'impact',OLD.impact,'support_level',OLD.support_level,'support_text',OLD.support_text,'exposure',OLD.exposure,
   'people_ids',json(COALESCE((SELECT json_group_array(person_id) FROM (SELECT person_id FROM life_event_people WHERE event_id=OLD.id AND user_id=OLD.user_id ORDER BY person_id)),'[]'))),
  CAST(strftime('%s','now') AS INTEGER));
END;
CREATE TRIGGER personal_history_event_delete AFTER DELETE ON life_events BEGIN
 DELETE FROM personal_history WHERE user_id=OLD.user_id AND entity_type='event' AND entity_id=OLD.id;
END;
-- Notes carry their own clock (seconds written by the Worker), the one shown next to them.
CREATE TRIGGER personal_history_note_insert AFTER INSERT ON personal_notes BEGIN
 INSERT INTO personal_history(user_id,entity_type,entity_id,revision,action,snapshot,changed_at) VALUES(NEW.user_id,'note',NEW.id,NEW.revision,'create',json_object('text',NEW.text),NEW.created_at);
END;
CREATE TRIGGER personal_history_note_update AFTER UPDATE OF text ON personal_notes WHEN OLD.text IS NOT NEW.text BEGIN
 INSERT INTO personal_history(user_id,entity_type,entity_id,revision,action,snapshot,changed_at) VALUES(OLD.user_id,'note',OLD.id,OLD.revision,'update',json_object('text',OLD.text),NEW.updated_at);
END;
CREATE TRIGGER personal_history_note_delete AFTER DELETE ON personal_notes BEGIN
 DELETE FROM personal_history WHERE user_id=OLD.user_id AND entity_type='note' AND entity_id=OLD.id;
END;
