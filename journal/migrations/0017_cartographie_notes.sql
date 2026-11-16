-- The childhood map becomes a long-term working document shared with Ambre (additive).
-- notes: the member's own words, {"factors":{factorId:"texte"},"outcomes":{outcomeId:"texte"}};
-- resources: protective factors checked in "Ce qui m'a aidé", {protectionId:1};
-- ambre_notes: short notes written by Ambre from the conversation,
--   [{id,target:"factor"|"outcome"|"general",target_id,text,created_at}]. They never change version.
ALTER TABLE cartographies ADD COLUMN notes TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(notes) AND json_type(notes)='object');
ALTER TABLE cartographies ADD COLUMN resources TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(resources) AND json_type(resources)='object');
ALTER TABLE cartographies ADD COLUMN ambre_notes TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(ambre_notes) AND json_type(ambre_notes)='array');
