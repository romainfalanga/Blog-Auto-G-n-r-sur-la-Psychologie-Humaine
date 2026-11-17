-- Durable commercial credits, separate from provider budgets and deletable content.
-- Reservations follow guarded content INSERTs; triggers commit/refund in the same
-- transaction as completion, failure or expiry. Content deletion never refunds.
ALTER TABLE chat_turns ADD COLUMN quota_period TEXT NOT NULL DEFAULT '';
ALTER TABLE ambre_reviews ADD COLUMN quota_period TEXT NOT NULL DEFAULT '';
CREATE TABLE monthly_usage (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 kind TEXT NOT NULL CHECK(kind IN ('message','review')),
 operation_id TEXT NOT NULL,
 period_id TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('reserved','consumed','released')),
 PRIMARY KEY(user_id,kind,operation_id)
);
CREATE INDEX monthly_usage_period ON monthly_usage(user_id,period_id,kind,state);
-- SQLite lacks IANA zones. EU transitions since1996: last Sundays in March
-- and October at01:00 UTC. Existing application receipts are2025 or later.
UPDATE chat_turns SET quota_period=strftime('%Y-%m',datetime(created_at,CASE WHEN datetime(created_at)>=datetime(strftime('%Y',created_at)||'-04-01','weekday 0','-7 days','+1 hour') AND datetime(created_at)<datetime(strftime('%Y',created_at)||'-11-01','weekday 0','-7 days','+1 hour') THEN '+2 hours' ELSE '+1 hour' END));
UPDATE ambre_reviews SET quota_period=strftime('%Y-%m',datetime(datetime(created_at,'unixepoch'),CASE WHEN datetime(datetime(created_at,'unixepoch'))>=datetime(strftime('%Y',datetime(created_at,'unixepoch'))||'-04-01','weekday 0','-7 days','+1 hour') AND datetime(datetime(created_at,'unixepoch'))<datetime(strftime('%Y',datetime(created_at,'unixepoch'))||'-11-01','weekday 0','-7 days','+1 hour') THEN '+2 hours' ELSE '+1 hour' END));
INSERT OR IGNORE INTO monthly_usage SELECT user_id,'message',id,quota_period,CASE WHEN COALESCE(error_code,'')='uncounted_cautious_reply' THEN 'released' ELSE credit_status END FROM chat_turns WHERE idempotency_key NOT LIKE 'review-%';
INSERT OR IGNORE INTO monthly_usage SELECT user_id,'review',id,quota_period,CASE status WHEN 'running' THEN 'reserved' WHEN 'ready' THEN 'consumed' ELSE 'released' END FROM ambre_reviews;
-- Deleted/cleared reviews retain a purged conversation receipt.
INSERT INTO monthly_usage SELECT user_id,'review',substr(idempotency_key,8),quota_period,'consumed' FROM chat_turns WHERE idempotency_key LIKE 'review-%' AND status='completed'
ON CONFLICT(user_id,kind,operation_id) DO UPDATE SET state='consumed';
CREATE TRIGGER monthly_chat_turns_insert AFTER INSERT ON chat_turns WHEN NEW.idempotency_key NOT LIKE 'review-%' BEGIN
 INSERT INTO monthly_usage(user_id,kind,operation_id,period_id,state) VALUES(NEW.user_id,'message',NEW.id,COALESCE(NULLIF(NEW.quota_period,''),strftime('%Y-%m',datetime(NEW.created_at,CASE WHEN datetime(NEW.created_at)>=datetime(strftime('%Y',NEW.created_at)||'-04-01','weekday 0','-7 days','+1 hour') AND datetime(NEW.created_at)<datetime(strftime('%Y',NEW.created_at)||'-11-01','weekday 0','-7 days','+1 hour') THEN '+2 hours' ELSE '+1 hour' END))),CASE WHEN COALESCE(NEW.error_code,'')='uncounted_cautious_reply' THEN 'released' ELSE NEW.credit_status END)
 ON CONFLICT(user_id,kind,operation_id) DO UPDATE SET state=CASE WHEN monthly_usage.state='consumed' THEN 'consumed' ELSE excluded.state END;
END;
CREATE TRIGGER monthly_chat_turns_update AFTER UPDATE ON chat_turns WHEN NEW.idempotency_key NOT LIKE 'review-%' BEGIN
 INSERT INTO monthly_usage(user_id,kind,operation_id,period_id,state) VALUES(NEW.user_id,'message',NEW.id,COALESCE(NULLIF(NEW.quota_period,''),strftime('%Y-%m',datetime(NEW.created_at,CASE WHEN datetime(NEW.created_at)>=datetime(strftime('%Y',NEW.created_at)||'-04-01','weekday 0','-7 days','+1 hour') AND datetime(NEW.created_at)<datetime(strftime('%Y',NEW.created_at)||'-11-01','weekday 0','-7 days','+1 hour') THEN '+2 hours' ELSE '+1 hour' END))),CASE WHEN COALESCE(NEW.error_code,'')='uncounted_cautious_reply' THEN 'released' ELSE NEW.credit_status END)
 ON CONFLICT(user_id,kind,operation_id) DO UPDATE SET state=CASE WHEN monthly_usage.state='consumed' THEN 'consumed' ELSE excluded.state END;
END;
CREATE TRIGGER monthly_ambre_reviews_insert AFTER INSERT ON ambre_reviews WHEN 1 BEGIN
 INSERT INTO monthly_usage(user_id,kind,operation_id,period_id,state) VALUES(NEW.user_id,'review',NEW.id,COALESCE(NULLIF(NEW.quota_period,''),strftime('%Y-%m',datetime(datetime(NEW.created_at,'unixepoch'),CASE WHEN datetime(datetime(NEW.created_at,'unixepoch'))>=datetime(strftime('%Y',datetime(NEW.created_at,'unixepoch'))||'-04-01','weekday 0','-7 days','+1 hour') AND datetime(datetime(NEW.created_at,'unixepoch'))<datetime(strftime('%Y',datetime(NEW.created_at,'unixepoch'))||'-11-01','weekday 0','-7 days','+1 hour') THEN '+2 hours' ELSE '+1 hour' END))),CASE NEW.status WHEN 'running' THEN 'reserved' WHEN 'ready' THEN 'consumed' ELSE 'released' END)
 ON CONFLICT(user_id,kind,operation_id) DO UPDATE SET state=CASE WHEN monthly_usage.state='consumed' THEN 'consumed' ELSE excluded.state END;
END;
CREATE TRIGGER monthly_ambre_reviews_update AFTER UPDATE ON ambre_reviews WHEN 1 BEGIN
 INSERT INTO monthly_usage(user_id,kind,operation_id,period_id,state) VALUES(NEW.user_id,'review',NEW.id,COALESCE(NULLIF(NEW.quota_period,''),strftime('%Y-%m',datetime(datetime(NEW.created_at,'unixepoch'),CASE WHEN datetime(datetime(NEW.created_at,'unixepoch'))>=datetime(strftime('%Y',datetime(NEW.created_at,'unixepoch'))||'-04-01','weekday 0','-7 days','+1 hour') AND datetime(datetime(NEW.created_at,'unixepoch'))<datetime(strftime('%Y',datetime(NEW.created_at,'unixepoch'))||'-11-01','weekday 0','-7 days','+1 hour') THEN '+2 hours' ELSE '+1 hour' END))),CASE NEW.status WHEN 'running' THEN 'reserved' WHEN 'ready' THEN 'consumed' ELSE 'released' END)
 ON CONFLICT(user_id,kind,operation_id) DO UPDATE SET state=CASE WHEN monthly_usage.state='consumed' THEN 'consumed' ELSE excluded.state END;
END;
-- Cancelling unpublished work releases its hold; published credits stay spent.
CREATE TRIGGER monthly_reviews_delete AFTER DELETE ON ambre_reviews BEGIN
 UPDATE monthly_usage SET state='released' WHERE user_id=OLD.user_id AND kind='review' AND operation_id=OLD.id AND state='reserved';
END;
CREATE TRIGGER monthly_turns_delete AFTER DELETE ON chat_turns BEGIN
 UPDATE monthly_usage SET state='released' WHERE user_id=OLD.user_id AND kind='message' AND operation_id=OLD.id AND state='reserved';
END;
