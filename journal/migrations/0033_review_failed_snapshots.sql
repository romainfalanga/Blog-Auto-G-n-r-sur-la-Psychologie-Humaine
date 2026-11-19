-- Plan Ambre premier message v1.1 (point 13), additive: a failed review keeps no snapshot.
-- Ready reviews are never touched. The existing sources_changed/consent_withdrawn trigger stays.
CREATE TRIGGER review_snapshots_failed AFTER UPDATE OF status ON ambre_reviews
 WHEN NEW.status='failed' BEGIN
 DELETE FROM review_snapshots WHERE review_id=NEW.id AND user_id=NEW.user_id;
END;
DELETE FROM review_snapshots WHERE EXISTS(SELECT 1 FROM ambre_reviews r WHERE r.id=review_snapshots.review_id AND r.user_id=review_snapshots.user_id AND r.status='failed');
