-- Existing accounts opt in explicitly. This revision is independent of chat consent.
ALTER TABLE chat_settings ADD COLUMN people_observations_enabled INTEGER NOT NULL DEFAULT 0 CHECK(people_observations_enabled IN (0,1));
ALTER TABLE chat_settings ADD COLUMN people_observations_revision INTEGER NOT NULL DEFAULT 1;

CREATE TABLE people_observations (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 person_id TEXT NOT NULL,
 turn_id TEXT NOT NULL,
 axis TEXT NOT NULL CHECK(axis IN ('communication','emotions','besoins','limites','appuis','tensions')),
 quote TEXT NOT NULL CHECK(length(quote) BETWEEN 20 AND 280),
 fingerprint TEXT NOT NULL,
 created_at INTEGER NOT NULL,
 UNIQUE(user_id,person_id,fingerprint),
 FOREIGN KEY(person_id,user_id) REFERENCES personal_people(id,user_id) ON DELETE CASCADE,
 FOREIGN KEY(turn_id,user_id) REFERENCES chat_turns(id,user_id) ON DELETE CASCADE
);
CREATE INDEX people_observations_owner ON people_observations(user_id,person_id,created_at DESC,id);
