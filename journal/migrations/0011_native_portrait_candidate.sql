-- An unpublished native portrait candidate is always checked against current
-- sources and semantically verified before it can become user_portraits.
ALTER TABLE journal_jobs ADD COLUMN native_candidate TEXT;
CREATE TRIGGER native_portrait_candidate_invalidate
AFTER UPDATE OF version ON profiles WHEN OLD.version!=NEW.version BEGIN
 UPDATE journal_jobs SET native_candidate=NULL WHERE user_id=NEW.user_id;
END;
