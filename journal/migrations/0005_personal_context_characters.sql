-- Apply with GENERATION_PAUSED=true and a database backup. D1 executes each
-- migration transactionally. Explicitly save feedback before replacing its parent.
PRAGMA defer_foreign_keys = ON;
ALTER TABLE article_likes ADD COLUMN revision INTEGER NOT NULL DEFAULT 1;
CREATE TRIGGER article_like_revision AFTER UPDATE OF reason,differences ON article_likes
 WHEN OLD.reason!=NEW.reason OR OLD.differences!=NEW.differences BEGIN
 UPDATE article_likes SET revision=OLD.revision+1 WHERE user_id=NEW.user_id AND article_id=NEW.article_id;
END;
CREATE TABLE personal_people (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 creation_id TEXT NOT NULL,payload_hash TEXT NOT NULL,display_name TEXT NOT NULL,role TEXT NOT NULL,custom_role TEXT NOT NULL DEFAULT '',relationship_text TEXT NOT NULL DEFAULT '',period_start TEXT NOT NULL DEFAULT '',period_end TEXT NOT NULL DEFAULT '',revision INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(user_id,creation_id),UNIQUE(id,user_id)
);
CREATE TABLE life_events (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 creation_id TEXT NOT NULL,payload_hash TEXT NOT NULL,title TEXT NOT NULL,kind TEXT NOT NULL,description TEXT NOT NULL,meaning_now TEXT NOT NULL DEFAULT '',date_precision TEXT NOT NULL DEFAULT 'unknown',date_start TEXT NOT NULL DEFAULT '',date_end TEXT NOT NULL DEFAULT '',age_text TEXT NOT NULL DEFAULT '',current_importance INTEGER,fiction_use TEXT NOT NULL DEFAULT 'abstract' CHECK(fiction_use IN ('abstract','understanding_only')),revision INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(user_id,creation_id),UNIQUE(id,user_id)
);
CREATE TABLE life_event_people (
 event_id TEXT NOT NULL,person_id TEXT NOT NULL,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 PRIMARY KEY(event_id,person_id),FOREIGN KEY(event_id,user_id) REFERENCES life_events(id,user_id) ON DELETE CASCADE,FOREIGN KEY(person_id,user_id) REFERENCES personal_people(id,user_id) ON DELETE CASCADE
);
CREATE INDEX personal_people_owner ON personal_people(user_id,id);
CREATE INDEX life_events_owner ON life_events(user_id,id);
CREATE TABLE personal_characters (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,slot INTEGER NOT NULL CHECK(slot BETWEEN 1 AND 3),identity TEXT NOT NULL DEFAULT '{}',facet TEXT NOT NULL DEFAULT '{}',narrative_state TEXT NOT NULL DEFAULT '{}',canon_version INTEGER NOT NULL DEFAULT 0,last_story_id TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,UNIQUE(user_id,slot),UNIQUE(id,user_id)
);
INSERT INTO personal_characters(id,user_id,slot,identity,canon_version,last_story_id)
 SELECT 'legacy-'||u.id,u.id,1,
 CASE WHEN json_valid(up.portrait) AND json_extract(up.portrait,'$.fictional_character.name') IS NOT NULL THEN json_extract(up.portrait,'$.fictional_character') ELSE '{}' END,
 (SELECT count(*) FROM stories s WHERE s.user_id=u.id),
 (SELECT id FROM stories s WHERE s.user_id=u.id ORDER BY day DESC,created_at DESC,id DESC LIMIT 1)
 FROM users u LEFT JOIN user_portraits up ON up.user_id=u.id
 WHERE EXISTS(SELECT 1 FROM stories s WHERE s.user_id=u.id) OR (json_valid(up.portrait) AND json_extract(up.portrait,'$.fictional_character.name') IS NOT NULL);
CREATE TABLE story_requests (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,character_id TEXT NOT NULL,
 idempotency_key TEXT NOT NULL,parameter_hash TEXT NOT NULL,week_start TEXT NOT NULL,expected_parent_id TEXT,
 status TEXT NOT NULL CHECK(status IN ('waiting_portrait','queued','running','completed','failed','cancelled','expired')),
 credit_status TEXT NOT NULL CHECK(credit_status IN ('reserved','consumed','released')),
 source_version INTEGER NOT NULL,canon_version INTEGER NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,provider_calls INTEGER NOT NULL DEFAULT 0,input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0,reserved_tokens INTEGER NOT NULL DEFAULT 0,observed_cost REAL NOT NULL DEFAULT 0,
 expires_at INTEGER NOT NULL,story_id TEXT,error_code TEXT,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(user_id,idempotency_key),UNIQUE(id,user_id),FOREIGN KEY(character_id,user_id) REFERENCES personal_characters(id,user_id) ON DELETE CASCADE,
 CHECK((status IN ('waiting_portrait','queued','running') AND credit_status='reserved') OR (status='completed' AND credit_status='consumed') OR (status IN ('failed','cancelled','expired') AND credit_status='released'))
);
CREATE UNIQUE INDEX story_request_one_active ON story_requests(user_id) WHERE status IN ('waiting_portrait','queued','running');
CREATE INDEX story_requests_quota ON story_requests(user_id,week_start,credit_status);
CREATE INDEX story_requests_queue ON story_requests(status,created_at,id);
INSERT INTO story_requests(id,user_id,character_id,idempotency_key,parameter_hash,week_start,status,credit_status,source_version,canon_version,expires_at,story_id,created_at)
 SELECT 'legacy-'||s.id,s.user_id,'legacy-'||s.user_id,'legacy-'||s.id,'legacy',date(s.day,'-'||((CAST(strftime('%w',s.day) AS INTEGER)+6)%7)||' days'),'completed','consumed',0,0,0,s.id,s.created_at FROM stories s;
CREATE TABLE feedback_saved AS SELECT * FROM feedback;
DROP TABLE feedback;
CREATE TABLE stories_v2 (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,day TEXT NOT NULL,title TEXT NOT NULL,body TEXT NOT NULL,summary TEXT NOT NULL,reflection_questions TEXT NOT NULL,practice TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 character_id TEXT NOT NULL,request_id TEXT,episode_number INTEGER NOT NULL,parent_story_id TEXT,source_version INTEGER NOT NULL DEFAULT 0,canon_version INTEGER NOT NULL DEFAULT 0,narrative_delta TEXT NOT NULL DEFAULT '{}',
 UNIQUE(character_id,episode_number),UNIQUE(request_id),UNIQUE(id,user_id),UNIQUE(id,character_id,user_id),
 FOREIGN KEY(character_id,user_id) REFERENCES personal_characters(id,user_id) ON DELETE CASCADE,
 FOREIGN KEY(request_id,user_id) REFERENCES story_requests(id,user_id) ON DELETE CASCADE,
 FOREIGN KEY(parent_story_id,character_id,user_id) REFERENCES stories_v2(id,character_id,user_id)
);
INSERT INTO stories_v2(id,user_id,day,title,body,summary,reflection_questions,practice,created_at,character_id,request_id,episode_number,parent_story_id)
 SELECT id,user_id,day,title,body,summary,reflection_questions,practice,created_at,'legacy-'||user_id,'legacy-'||id,
 row_number() OVER(PARTITION BY user_id ORDER BY day,created_at,id),lag(id) OVER(PARTITION BY user_id ORDER BY day,created_at,id) FROM stories;
DROP TABLE stories;
ALTER TABLE stories_v2 RENAME TO stories;
CREATE INDEX stories_owner ON stories(user_id,character_id,episode_number DESC,id DESC);
CREATE TABLE feedback (
 story_id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,resonance INTEGER NOT NULL,feelings TEXT NOT NULL,why TEXT NOT NULL,correction TEXT NOT NULL,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 differences TEXT NOT NULL DEFAULT '',scope TEXT NOT NULL DEFAULT 'overall',preference TEXT NOT NULL DEFAULT '',avoid TEXT NOT NULL DEFAULT '',revision INTEGER NOT NULL DEFAULT 1,
 FOREIGN KEY(story_id,user_id) REFERENCES stories(id,user_id) ON DELETE CASCADE
);
INSERT INTO feedback(story_id,user_id,resonance,feelings,why,correction,updated_at) SELECT story_id,user_id,resonance,feelings,why,correction,updated_at FROM feedback_saved;
DROP TABLE feedback_saved;
-- V1 prose and unversioned proofs cannot become V2 evidence.
DELETE FROM user_portraits;
ALTER TABLE user_portraits ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 2;
CREATE TABLE journal_jobs_v2 (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,kind TEXT NOT NULL CHECK(kind IN ('portrait','characters','story')),source_version INTEGER NOT NULL,memory_version INTEGER NOT NULL,day TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,provider_calls INTEGER NOT NULL DEFAULT 0,reserved_tokens INTEGER NOT NULL DEFAULT 0,input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0,observed_cost REAL NOT NULL DEFAULT 0,
 lease_token TEXT,lease_until INTEGER NOT NULL DEFAULT 0,error_code TEXT,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,request_id TEXT UNIQUE,character_id TEXT,
 FOREIGN KEY(request_id,user_id) REFERENCES story_requests(id,user_id) ON DELETE CASCADE,FOREIGN KEY(character_id,user_id) REFERENCES personal_characters(id,user_id) ON DELETE CASCADE
);
INSERT INTO journal_jobs_v2(id,user_id,kind,source_version,memory_version,day,status,attempts,provider_calls,created_at)
 SELECT id,user_id,kind,source_version,memory_version,day,'obsolete',attempts,provider_calls,created_at FROM journal_jobs;
DROP TABLE journal_jobs;
ALTER TABLE journal_jobs_v2 RENAME TO journal_jobs;
CREATE UNIQUE INDEX portrait_version ON journal_jobs(user_id,source_version) WHERE kind='portrait' AND status NOT IN ('complete','obsolete');
CREATE UNIQUE INDEX characters_preparation ON journal_jobs(user_id) WHERE kind='characters' AND status NOT IN ('complete','obsolete','failed');
CREATE INDEX journal_jobs_pending ON journal_jobs(kind,status,lease_until);
UPDATE generation_jobs SET status='obsolete',lease_token=NULL,lease_until=0;
CREATE TABLE context_source_derivatives (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,source_id TEXT NOT NULL,revision INTEGER NOT NULL,analysis_version INTEGER NOT NULL,source_hash TEXT NOT NULL,derivative TEXT NOT NULL,manifest TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(user_id,source_id,revision,analysis_version)
);
CREATE TABLE context_mutations (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,kind TEXT NOT NULL,creation_id TEXT NOT NULL,payload_hash TEXT NOT NULL,entity_id TEXT NOT NULL,
 PRIMARY KEY(user_id,kind,creation_id)
);
-- The global source revision is the invalidation boundary for every write path,
-- including likes and backwards-compatible memory/profile routes.
CREATE TRIGGER invalidate_personal_context AFTER UPDATE OF version ON profiles WHEN NEW.version!=OLD.version BEGIN
 DELETE FROM user_portraits WHERE user_id=NEW.user_id;

 UPDATE journal_jobs SET status='obsolete',lease_token=NULL,lease_until=0 WHERE user_id=NEW.user_id AND kind='portrait' AND status!='complete';
 UPDATE journal_jobs SET status='pending',source_version=NEW.version,lease_token=NULL,lease_until=0 WHERE user_id=NEW.user_id AND kind='characters' AND status NOT IN ('complete','failed');
 UPDATE story_requests SET status='waiting_portrait',source_version=NEW.version,error_code='context_changed',updated_at=CURRENT_TIMESTAMP WHERE user_id=NEW.user_id AND status IN ('waiting_portrait','queued','running');
 UPDATE journal_jobs SET status='pending',source_version=NEW.version,lease_token=NULL,lease_until=0,error_code='context_changed' WHERE user_id=NEW.user_id AND kind='story' AND status NOT IN ('complete','failed','obsolete');
END;

CREATE TABLE provider_budgets (user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,week_start TEXT NOT NULL,provider_calls INTEGER NOT NULL DEFAULT 0,reserved_tokens INTEGER NOT NULL DEFAULT 0,input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0,observed_cost REAL NOT NULL DEFAULT 0,PRIMARY KEY(user_id,week_start));

-- Invalidate only the derivative of an edited/removed source; other source
-- revisions remain reusable. Global portrait invalidation above still applies.
CREATE TRIGGER derivative_like_update AFTER UPDATE OF reason,differences ON article_likes WHEN OLD.reason!=NEW.reason OR OLD.differences!=NEW.differences BEGIN DELETE FROM context_source_derivatives WHERE user_id=NEW.user_id AND source_id='like:'||NEW.article_id; END;
CREATE TRIGGER derivative_like_delete AFTER DELETE ON article_likes BEGIN DELETE FROM context_source_derivatives WHERE user_id=OLD.user_id AND source_id='like:'||OLD.article_id; END;
CREATE TRIGGER derivative_memory_update AFTER UPDATE OF text ON memories WHEN OLD.text!=NEW.text BEGIN DELETE FROM context_source_derivatives WHERE user_id=NEW.user_id AND source_id='memory:'||NEW.user_id; END;
CREATE TRIGGER derivative_person_update AFTER UPDATE ON personal_people BEGIN DELETE FROM context_source_derivatives WHERE user_id=NEW.user_id AND source_id='person:'||NEW.id; END;
CREATE TRIGGER derivative_person_delete AFTER DELETE ON personal_people BEGIN DELETE FROM context_source_derivatives WHERE user_id=OLD.user_id AND source_id='person:'||OLD.id; END;
CREATE TRIGGER derivative_event_update AFTER UPDATE ON life_events BEGIN DELETE FROM context_source_derivatives WHERE user_id=NEW.user_id AND source_id='event:'||NEW.id; END;
CREATE TRIGGER derivative_event_delete AFTER DELETE ON life_events BEGIN DELETE FROM context_source_derivatives WHERE user_id=OLD.user_id AND source_id='event:'||OLD.id; END;
CREATE TRIGGER derivative_feedback_update AFTER UPDATE ON feedback BEGIN DELETE FROM context_source_derivatives WHERE user_id=NEW.user_id AND source_id='feedback:'||NEW.story_id; END;
CREATE TRIGGER derivative_feedback_delete AFTER DELETE ON feedback BEGIN DELETE FROM context_source_derivatives WHERE user_id=OLD.user_id AND source_id='feedback:'||OLD.story_id; END;
