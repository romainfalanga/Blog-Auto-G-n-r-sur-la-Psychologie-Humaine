// A refused written review must never hold the conversation indefinitely
// (independent review of v4: a running review blocks new messages with 409).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,refused} from './v4-R-helper.mjs';

function seedReview(s,id){
 s.raw.prepare("UPDATE ambre_reviews SET status='ready' WHERE id='seed-a'").run();
 s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,idempotency_key,attempts,lease_token,lease_until,created_at) VALUES(?,'a','point','running',?,0,NULL,0,?)").run(id,'rk-'+id,s.now());
 s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES(?,'a',?,'h','Proche','friend','Relation décrite en détail.')").run('pp-'+id,'pp-'+id);
}

test('a rejected key (401) ends the review instead of keeping it running',async t=>{
 const s=setup();await s.ready('a');seedReview(s,'r401');
 provider(t,()=>refused(401,null));
 await s.chat.resumeReviews(s.env);
 const review=s.raw.prepare("SELECT status,error_code,completed_at FROM ambre_reviews WHERE id='r401'").get();
 assert.equal(review.status,'failed');assert.ok(review.completed_at);assert.match(review.error_code,/401|provider/);
 assert.equal(s.raw.prepare("SELECT count(*) n FROM ambre_reviews WHERE user_id='a' AND status='running'").get().n,0,'nothing left to block new messages');
});

test('a refusal whose retry time is more than ten minutes after the request ends the review',async t=>{
 const s=setup();await s.ready('a');seedReview(s,'rlong');
 provider(t,()=>refused(429,'3600'));
 await s.chat.resumeReviews(s.env);
 assert.equal(s.raw.prepare("SELECT status FROM ambre_reviews WHERE id='rlong'").get().status,'failed');
});

test('a short refusal still waits and keeps the review running',async t=>{
 const s=setup();await s.ready('a');seedReview(s,'rshort');
 provider(t,()=>refused(429,'45'));
 await s.chat.resumeReviews(s.env);
 const review=s.raw.prepare("SELECT status,lease_until FROM ambre_reviews WHERE id='rshort'").get();
 assert.equal(review.status,'running');assert.ok(review.lease_until>=s.now()+45);
});
