-- Childhood map saved from /cartographie/ by a signed-in member (additive).
-- answers: {factorId: 1|2} where 2 means "souvent ou gravement";
-- feedback: {outcomeId: "oui"|"non"}. Saving requires explicit consent (health data).
-- A locked map (locked_at set) becomes Ambre's reference; it is not a portrait source.
CREATE TABLE IF NOT EXISTS cartographies (
 user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 answers TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(answers) AND json_type(answers)='object'),
 feedback TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(feedback) AND json_type(feedback)='object'),
 consent INTEGER NOT NULL DEFAULT 0 CHECK(consent IN (0,1)),
 locked_at INTEGER,
 version INTEGER NOT NULL DEFAULT 1 CHECK(version>=1),
 updated_at INTEGER NOT NULL
);
