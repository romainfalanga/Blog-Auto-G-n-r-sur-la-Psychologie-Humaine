-- Lot A5-P1 (plan v5.1): structured reviews, their separate check and the person's feedback.
-- Additive only: four columns on ambre_reviews, no table rewritten, no row deleted.
-- ALTER TABLE cannot be guarded in SQLite: applied once through `wrangler d1 migrations apply`.
-- The Worker probes these columns and keeps the previous behaviour until they exist.
-- digest: the structured part of a ready review (formulation, goals with a state in words,
--   trials and effects, alliance, open questions, next step), each element anchored in an
--   exact quote of the person; '{}' for reviews written before this migration.
-- feedback: the person's answer to a review ("resembles", "not_really", "to_correct" and an
--   optional text of at most 600 characters, no score); NULL when none was given.
-- stage and draft: a running review is written, then checked in a separate step (and rewritten
--   once at most); the unchecked draft stays here, never in the conversation, until it passes.
ALTER TABLE ambre_reviews ADD COLUMN digest TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(digest) AND json_type(digest)='object');
ALTER TABLE ambre_reviews ADD COLUMN feedback TEXT CHECK(feedback IS NULL OR (json_valid(feedback) AND json_type(feedback)='object'));
ALTER TABLE ambre_reviews ADD COLUMN stage TEXT NOT NULL DEFAULT 'write' CHECK(stage IN ('write','check','rewrite','done'));
ALTER TABLE ambre_reviews ADD COLUMN draft TEXT NOT NULL DEFAULT '';
