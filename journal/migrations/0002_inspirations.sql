CREATE TABLE IF NOT EXISTS inspiration_selections (
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  article_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  differences TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (user_id, article_id)
);

-- Public canonical source text. This catalog is independent from private accounts.
CREATE TABLE IF NOT EXISTS public_examples (id TEXT PRIMARY KEY, body TEXT NOT NULL);
