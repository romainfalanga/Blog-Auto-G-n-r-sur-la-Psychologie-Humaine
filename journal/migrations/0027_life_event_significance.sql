-- Plan « Mes symptômes » v1.1, Moments: whether a moment is « marquant dans mon histoire ».
-- Distinct from current_importance (1–5, « pèse aujourd'hui »), which is kept and never converted.
-- Additive only: one nullable column without a default. ALTER TABLE ADD COLUMN rewrites no row and
-- fires no trigger, so revisions, payload hashes, care_revision_* and derivative caches stay as they
-- are. NULL means « à préciser »: existing moments keep it until the person chooses.
-- ALTER TABLE cannot be guarded in SQLite: applied once through `wrangler d1 migrations apply`.
-- The Worker probes the column (significanceReady) and ignores the field until it exists.
ALTER TABLE life_events ADD COLUMN significance TEXT CHECK(significance IN ('anecdotal','major'));
