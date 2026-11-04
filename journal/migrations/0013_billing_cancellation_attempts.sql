-- Keep cancellation retries independent from checkout attempts. A confirmed
-- cancellation followed by a resumed subscription starts a fresh intention.
ALTER TABLE subscriptions ADD COLUMN cancellation_attempt TEXT;
ALTER TABLE subscriptions ADD COLUMN cancellation_subscription_id TEXT;
ALTER TABLE subscriptions ADD COLUMN cancellation_confirmed INTEGER NOT NULL DEFAULT 0 CHECK(cancellation_confirmed IN (0,1));
