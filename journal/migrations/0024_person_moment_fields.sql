-- Lot A5-P2 (plan v5.1): optional form fields for people and moments, and the origin of a note on
-- a person (an observation Ambre proposed and the person accepted, or corrected, before it was saved).
-- Additive only: columns with constant defaults, no table rewritten, no row changed or deleted.
-- ALTER TABLE cannot be guarded in SQLite: applied once through `wrangler d1 migrations apply`.
-- The Worker probes these columns: old notes remain readable, new proposals wait for migration.
-- Every value is optional: '' means "not given". Closed lists are stored as codes, shown in words.
-- People (genogram emotional lines, convoy model): the quality of the bond with a nuance in the
--   person's words, presence today, what helps, what hurts, how the bond changed, feeling safe.
ALTER TABLE personal_people ADD COLUMN bond_quality TEXT NOT NULL DEFAULT '' CHECK(bond_quality IN ('','close','distant','conflictual','cut_off','fusional','supportive','ambivalent','unsure'));
ALTER TABLE personal_people ADD COLUMN bond_note TEXT NOT NULL DEFAULT '' CHECK(length(bond_note)<=300);
ALTER TABLE personal_people ADD COLUMN presence_now TEXT NOT NULL DEFAULT '' CHECK(presence_now IN ('','daily','regular','rare','no_contact','deceased'));
ALTER TABLE personal_people ADD COLUMN helps TEXT NOT NULL DEFAULT '' CHECK(length(helps)<=600);
ALTER TABLE personal_people ADD COLUMN hurts TEXT NOT NULL DEFAULT '' CHECK(length(hurts)<=600);
ALTER TABLE personal_people ADD COLUMN evolution TEXT NOT NULL DEFAULT '' CHECK(length(evolution)<=600);
ALTER TABLE personal_people ADD COLUMN feels_safe TEXT NOT NULL DEFAULT '' CHECK(feels_safe IN ('','yes','not_always','no'));
-- Moments (NIMH life chart, life story interview, LEC-5 exposure without any detail requested):
--   feeling then, feeling now, impact, support received (level and who or how), exposure.
ALTER TABLE life_events ADD COLUMN felt_then TEXT NOT NULL DEFAULT '' CHECK(length(felt_then)<=600);
ALTER TABLE life_events ADD COLUMN felt_now TEXT NOT NULL DEFAULT '' CHECK(length(felt_now)<=600);
ALTER TABLE life_events ADD COLUMN impact TEXT NOT NULL DEFAULT '' CHECK(length(impact)<=600);
ALTER TABLE life_events ADD COLUMN support_level TEXT NOT NULL DEFAULT '' CHECK(support_level IN ('','none','some','much'));
ALTER TABLE life_events ADD COLUMN support_text TEXT NOT NULL DEFAULT '' CHECK(length(support_text)<=300);
ALTER TABLE life_events ADD COLUMN exposure TEXT NOT NULL DEFAULT '' CHECK(exposure IN ('','happened_to_me','witnessed','learned'));
-- Notes on people: 'auto' for historical notes only, 'accepted' for a proposal
-- the person accepted or corrected (observation H). Existing notes are automatic.
ALTER TABLE people_observations ADD COLUMN origin TEXT NOT NULL DEFAULT 'auto' CHECK(origin IN ('auto','accepted'));
