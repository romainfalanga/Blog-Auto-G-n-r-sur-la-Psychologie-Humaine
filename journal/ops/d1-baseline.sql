-- One-time baseline of the migration history (lot R, F22). NOT a migration: it
-- lives outside migrations/ so `wrangler d1 migrations apply` never replays it.
--
-- Production was migrated by hand (0001 to 0021) and has no d1_migrations table,
-- so Wrangler would restart from 0001. This script creates the table in Wrangler's
-- own format and records each migration ONLY if a characteristic object of that
-- migration is present in the live schema. A missing object leaves the migration
-- unrecorded, so `migrations list` shows it as pending instead of hiding it.
--
-- Procedure (see journal/README.md): restore point, then
--   wrangler d1 execute decode-journal --remote --file=ops/d1-baseline.sql
--   wrangler d1 migrations list decode-journal --remote   (expect only 0022)
--   wrangler d1 migrations apply decode-journal --remote
-- Idempotent: running it twice records nothing twice.
CREATE TABLE IF NOT EXISTS d1_migrations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE,
  applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);
-- One INSERT per migration: D1 rejects long compound SELECTs ("too many terms").
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0001.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='users') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='rate_limits');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0002_inspirations.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='public_examples');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0003_likes_portraits.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='article_likes');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0004_recommendations_free_portraits.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='scheduler_state');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0005_personal_context_characters.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='life_events') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='provider_budgets');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0006_recommendation_exposures.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_exposures');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0007_self_profile_single_character.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='profiles' AND sql LIKE '%self_revision%');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0008_billing_trial.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='subscriptions' AND sql LIKE '%trial_used_at%');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0009_private_companion.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_turns') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='chat_one_running');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0010_requested_focus.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%requested_focus%');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0011_native_portrait_candidate.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='journal_jobs' AND sql LIKE '%native_candidate%');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0012_ambre_portrait_corrections.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='chat_provider_cooldown');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0013_billing_cancellation_attempts.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='subscriptions' AND sql LIKE '%cancellation_confirmed%');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0014_retire_personal_stories.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='invalidate_personal_context' AND sql NOT LIKE '%story_requests%');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0015_people_observations.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='people_observations');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0016_cartographies.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='cartographies');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0017_cartographie_notes.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='cartographies' AND sql LIKE '%ambre_notes%');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0018_ambre_reviews.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='ambre_reviews');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0019_ambre_engine.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_engine_jobs');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0020_care.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='care_links');
INSERT OR IGNORE INTO d1_migrations(name) SELECT '0021_change_plan.sql' WHERE EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='change_plans') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_turns' AND sql LIKE '%context_generation%');
