-- « Mes sujets » : la personne pose ses points de vue sur des sujets (politique, écologie…) et voit
-- comment ses arguments se construisent. Additive only: two tables and two indexes, nothing existing
-- is rewritten. A topic holds a tree: positions at the root, then what supports or contradicts each
-- element (a reply to an objection is an objection to that objection).
-- Deleting a topic or a node erases what hangs below it; deleting the account erases everything.
-- These tables are read by their owner only and exported; they are not sent to Ambre.
CREATE TABLE personal_topics (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 title TEXT NOT NULL,
 created_at INTEGER NOT NULL,
 updated_at INTEGER NOT NULL
);
CREATE INDEX personal_topics_user ON personal_topics(user_id,updated_at);
CREATE TABLE personal_topic_nodes (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 topic_id TEXT NOT NULL REFERENCES personal_topics(id) ON DELETE CASCADE,
 parent_id TEXT REFERENCES personal_topic_nodes(id) ON DELETE CASCADE,
 kind TEXT NOT NULL CHECK(kind IN ('position','support','objection')),
 content TEXT NOT NULL,
 basis TEXT CHECK(basis IS NULL OR basis IN ('fact','experience','value','intuition','hearsay')),
 confidence INTEGER CHECK(confidence IS NULL OR confidence BETWEEN 1 AND 5),
 created_at INTEGER NOT NULL,
 updated_at INTEGER NOT NULL
);
CREATE INDEX personal_topic_nodes_topic ON personal_topic_nodes(topic_id,created_at,id);
