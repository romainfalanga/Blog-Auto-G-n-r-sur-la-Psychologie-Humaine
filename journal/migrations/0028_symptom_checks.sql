-- Plan « Mes symptômes » v1.1 (contrat docs/symptomes-contrat-2026-11-19.md, § 3, § 7).
-- Additive only: four new tables, no existing row touched. Every row belongs to one account and
-- disappears with it (ON DELETE CASCADE, like cartographies). The Worker probes these tables
-- (symptomsReady) and answers « bientôt disponible » until this migration is applied.

-- Consent specific to « Mes symptômes » (distinct from Mon enfance and from Ambre).
CREATE TABLE IF NOT EXISTS symptom_consents (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 consent INTEGER NOT NULL CHECK(consent IN (0,1)),
 content_version TEXT NOT NULL CHECK(length(content_version) BETWEEN 1 AND 40),
 granted_at INTEGER,
 withdrawn_at INTEGER,
 updated_at INTEGER NOT NULL,
 CHECK(consent=0 OR granted_at IS NOT NULL)
);

-- One finished passation of one module, never overwritten. `scores` and `result` are recomputed by
-- the Worker from `answers` and the content version; a score sent by the client is never stored.
CREATE TABLE IF NOT EXISTS symptom_checks (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 passation_key TEXT NOT NULL CHECK(length(passation_key) BETWEEN 8 AND 120),
 module TEXT NOT NULL CHECK(length(module) BETWEEN 1 AND 64),
 content_version TEXT NOT NULL CHECK(length(content_version) BETWEEN 1 AND 40),
 instrument_versions TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(instrument_versions) AND json_type(instrument_versions)='object'),
 answers TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(answers) AND json_type(answers)='object' AND length(answers)<=20000),
 scores TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(scores) AND json_type(scores)='object'),
 result TEXT NOT NULL CHECK(json_valid(result) AND json_type(result)='object'),
 state TEXT NOT NULL CHECK(state IN ('au_dessus','en_dessous','repere_present','repere_absent','indetermine','refuse')),
 declined INTEGER NOT NULL DEFAULT 0 CHECK(declined IN (0,1)),
 started_at INTEGER,
 completed_at INTEGER NOT NULL,
 CHECK(declined=0 OR (state='refuse' AND answers='{}')),
 UNIQUE(user_id,passation_key)
);
CREATE INDEX IF NOT EXISTS symptom_checks_user_module ON symptom_checks(user_id,module,completed_at DESC);

-- Passation in progress (autosave, resume): one per module, overwritten, removed when finished.
CREATE TABLE IF NOT EXISTS symptom_drafts (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 module TEXT NOT NULL CHECK(length(module) BETWEEN 1 AND 64),
 passation_key TEXT NOT NULL CHECK(length(passation_key) BETWEEN 8 AND 120),
 content_version TEXT NOT NULL CHECK(length(content_version) BETWEEN 1 AND 40),
 answers TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(answers) AND json_type(answers)='object' AND length(answers)<=20000),
 started_at INTEGER,
 updated_at INTEGER NOT NULL,
 PRIMARY KEY(user_id,module)
);

-- Dated safety alerts (§ 7): which trigger, never a risk score. Written as soon as the answer is
-- saved, even if the passation is abandoned; one row per trigger and passation.
CREATE TABLE IF NOT EXISTS symptom_safety_alerts (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 passation_key TEXT NOT NULL CHECK(length(passation_key) BETWEEN 8 AND 120),
 module TEXT NOT NULL CHECK(length(module) BETWEEN 1 AND 64),
 trigger_id TEXT NOT NULL CHECK(length(trigger_id) BETWEEN 1 AND 64),
 kind TEXT NOT NULL CHECK(kind IN ('suicide','automutilation','violences')),
 screen TEXT NOT NULL CHECK(screen IN ('bloquant','doux')),
 check_id TEXT REFERENCES symptom_checks(id) ON DELETE SET NULL,
 created_at INTEGER NOT NULL,
 UNIQUE(user_id,passation_key,trigger_id)
);
CREATE INDEX IF NOT EXISTS symptom_safety_alerts_user ON symptom_safety_alerts(user_id,created_at DESC);
