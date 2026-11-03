-- A user-approved reading request, not a new psychological trait.
ALTER TABLE story_requests ADD COLUMN requested_focus TEXT NOT NULL DEFAULT '';
