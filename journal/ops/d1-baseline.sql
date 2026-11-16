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
INSERT OR IGNORE INTO d1_migrations(name)
SELECT name FROM (
  SELECT 1 AS n,'0001.sql' AS name,EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='users') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='rate_limits') AS ok
  UNION ALL SELECT 2,'0002_inspirations.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='public_examples')
  UNION ALL SELECT 3,'0003_likes_portraits.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='article_likes')
  UNION ALL SELECT 4,'0004_recommendations_free_portraits.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='scheduler_state')
  UNION ALL SELECT 5,'0005_personal_context_characters.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='life_events') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='provider_budgets')
  UNION ALL SELECT 6,'0006_recommendation_exposures.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='recommendation_exposures')
  UNION ALL SELECT 7,'0007_self_profile_single_character.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='profiles' AND sql LIKE '%self_revision%')
  UNION ALL SELECT 8,'0008_billing_trial.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='subscriptions' AND sql LIKE '%trial_used_at%')
  UNION ALL SELECT 9,'0009_private_companion.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_turns') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='chat_one_running')
  UNION ALL SELECT 10,'0010_requested_focus.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND sql LIKE '%requested_focus%')
  UNION ALL SELECT 11,'0011_native_portrait_candidate.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='journal_jobs' AND sql LIKE '%native_candidate%')
  UNION ALL SELECT 12,'0012_ambre_portrait_corrections.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='chat_provider_cooldown')
  UNION ALL SELECT 13,'0013_billing_cancellation_attempts.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='subscriptions' AND sql LIKE '%cancellation_confirmed%')
  UNION ALL SELECT 14,'0014_retire_personal_stories.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='invalidate_personal_context' AND sql NOT LIKE '%story_requests%')
  UNION ALL SELECT 15,'0015_people_observations.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='people_observations')
  UNION ALL SELECT 16,'0016_cartographies.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='cartographies')
  UNION ALL SELECT 17,'0017_cartographie_notes.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='cartographies' AND sql LIKE '%ambre_notes%')
  UNION ALL SELECT 18,'0018_ambre_reviews.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='ambre_reviews')
  UNION ALL SELECT 19,'0019_ambre_engine.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_engine_jobs')
  UNION ALL SELECT 20,'0020_care.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='care_links')
  UNION ALL SELECT 21,'0021_change_plan.sql',EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='change_plans') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_turns' AND sql LIKE '%context_generation%')
) WHERE ok ORDER BY n;
