import test from 'node:test';import assert from 'node:assert/strict';
import {setup,provider} from './v4-R-helper.mjs';
import {readSnapshotSource,snapshotChunks,readReviewSnapshot,exportReviewSnapshots} from '../src/review-snapshot.mjs';
const PASS={criteria:{no_diagnosis:'yes',quotes_faithful:'yes',safety_resources:'na',hypotheses_open:'yes',no_saved_claim:'yes',sources_honest:'yes',no_internal:'yes'},reasons:[]};
const output={content:'Nous pouvons explorer ce que vous avez déclaré, en gardant les nuances et les informations manquantes. '.repeat(18)};
async function ready(t){const s=setup();t.after(()=>s.raw.close());await s.ready();return s;}
const mock=(t,inspect)=>provider(t,(purpose,body,calls)=>{inspect?.(purpose,body,calls);if(purpose==='ambre_snapshot_chunk'){const chunk=JSON.parse(body.messages[1].content);return {summary:'Bloc '+chunk.index+' analysé avec ses déclarations et nuances.',quotes:[]};}if(purpose==='ambre_review')return output;if(purpose==='ambre_review_check')return PASS;});
test('a new review captures an immutable complete state and analyzes the final item beyond 40k',async t=>{
 const s=await ready(t);s.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.self_description',?) WHERE user_id='a'").run('Présentation complète '.repeat(3100)+'DERNIER_ELEMENT_65000');
 const calls=mock(t),r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-all-items'});assert.equal(r.status,202,JSON.stringify(r));
 const review=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);assert.equal(review.status,'ready',JSON.stringify(review));
 const row=await readReviewSnapshot(s.env,'a',r.review_id);assert.equal(row.payload.complete,true);assert.match(row.payload.sections.profile[0].self_data.self_description,/DERNIER_ELEMENT_65000$/);
 const blocks=calls.filter(c=>c.purpose==='ambre_snapshot_chunk');assert.ok(blocks.length>=4);assert.ok(blocks.some(c=>c.body.messages[1].content.includes('DERNIER_ELEMENT_65000')));const chunks=snapshotChunks(row.payload);assert.equal(blocks.length,chunks.length);
 assert.throws(()=>s.raw.prepare("UPDATE review_snapshots SET payload='{}' WHERE review_id=?").run(r.review_id),/immutable/);
 s.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.self_description','Nouveau texte') WHERE user_id='a'").run();assert.match((await readReviewSnapshot(s.env,'a',r.review_id)).payload.sections.profile[0].self_data.self_description,/DERNIER_ELEMENT/);
 const page=await s.call('/api/reviews/'+r.review_id+'/snapshot?section=profile');assert.equal(page.available,true);assert.equal(page.items.length,1);assert.equal(page.next_cursor,null);
 assert.equal((await s.call('/api/reviews/'+r.review_id+'/snapshot','GET',undefined,'b')).status,404);
 assert.equal((await exportReviewSnapshots(s.env,'a')).length,1);await s.call('/api/chat/consent','PUT',{accepted:false});assert.equal(await readReviewSnapshot(s.env,'a',r.review_id),null);
});
// Plan v1.1 (point 1): after the capture a review is a dated document; a modification of the live
// sources is taken into account at the next review, it no longer stops this one.
test('a live source modified after the capture no longer stops the review (dated snapshot)',async t=>{
 const s=await ready(t);let changed=false;const calls=mock(t,(purpose)=>{if(purpose==='ambre_snapshot_chunk'&&!changed){changed=true;s.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.first_name','Changed') WHERE user_id='a'").run();}});
 const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-source-race'});const review=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);assert.equal(review.status,'ready',JSON.stringify(review));assert.ok(calls.length>2);
 assert.notEqual((await readReviewSnapshot(s.env,'a',r.review_id)).payload.sections.profile[0].self_data.first_name,'Changed','the snapshot keeps the state at the capture');
});
test('legacy reviews are explicitly unavailable; oversized sources fail without provider calls',async t=>{
 const s=await ready(t);const old=await s.call('/api/reviews/seed-a/snapshot');assert.equal(old.available,false);assert.equal(old.reason,'legacy_review');
 s.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.self_description',?) WHERE user_id='a'").run('x'.repeat(530000));
 const calls=mock(t),r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-too-large'});const row=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);assert.equal(row.error_code,'snapshot_too_large');assert.equal(row.status,'failed');assert.equal(calls.length,0);
});
test('completed snapshot blocks survive a provider cooldown and are not analyzed twice',async t=>{
 const s=await ready(t);let refused=false;const calls=provider(t,(purpose,body)=>{
  if(purpose==='ambre_snapshot_chunk')return {summary:'Le bloc contient des faits déclarés.',quotes:[]};
  if(purpose==='ambre_review'&&!refused){refused=true;return new Response('{}',{status:429,headers:{'Retry-After':'30'}});}
  if(purpose==='ambre_review')return output;if(purpose==='ambre_review_check')return PASS;
 });
 const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-resume-000'});const first=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);assert.equal(first.status,'running');assert.ok(JSON.parse(first.draft).snapshot_chunks.length>0);
 const count=calls.filter(c=>c.purpose==='ambre_snapshot_chunk').length;s.tick(61);await s.chat.resumeReviews(s.env);
 const last=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);assert.equal(last.status,'ready',JSON.stringify(last));assert.equal(calls.filter(c=>c.purpose==='ambre_snapshot_chunk').length,count);
});
test('snapshot section pagination is complete and deleting a review cascades its archive',async t=>{
 const s=await ready(t);mock(t);const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-pagination'});const record=await readReviewSnapshot(s.env,'a',r.review_id),expected=record.payload.sections.people;
 const collected=[];let cursor=0;do{const page=await s.call(`/api/reviews/${r.review_id}/snapshot?section=people&limit=1&cursor=${cursor}`);assert.equal(page.status,200);collected.push(...page.items);cursor=page.next_cursor;}while(cursor!==null);
 assert.deepEqual(collected,expected);const map=await s.call(`/api/reviews/${r.review_id}/snapshot?section=childhood`);assert.ok(map.items[0].answers.every(x=>x.label));
 assert.equal((await s.call(`/api/reviews/${r.review_id}`,'DELETE')).status,200);assert.equal(await readReviewSnapshot(s.env,'a',r.review_id),null);
});
test('snapshots contain no credentials, billing secrets, provider logs or public catalogue text',async t=>{
 const s=await ready(t);const source=await readSnapshotSource(s.env,'a','new-review');const text=JSON.stringify(source.sections);
 assert.doesNotMatch(text,/password_hash|password_salt|OPENROUTER|checkout_session|lease_token|provider_budgets|public_examples/);assert.equal(Object.hasOwn(source.sections,'subscriptions'),false);
 const raw=source.sections.profile[0];assert.equal(Object.hasOwn(raw,'self_data'),true);
});
test('snapshot safety uses the captured date and keeps historical and current signals distinct',async()=>{
 const {snapshotSafetyContext}=await import('../src/review-snapshot.mjs'),{SYMPTOMES_CARTE}=await import('../src/symptomes-carte.generated.mjs');const at=1800000000,f=SYMPTOMES_CARTE.factors.find(x=>x.safety==='automutilation');
 const make=row=>({captured_at:new Date(at*1000).toISOString(),sections:{symptoms:[row],previous_reviews:[],conversation:[]}});
 assert.equal(snapshotSafetyContext(make({answers:{[f.id]:0},safety_at:at-30*86400})).map_safety,null);
 assert.deepEqual(snapshotSafetyContext(make({answers:{[f.id]:1},safety_at:at-30*86400})).map_safety.kinds,['automutilation']);
 const child={answers:{abus_physique:1},feedback:{ten_idees_suicidaires:'oui'},notes:{},consent:1,locked_at:at,version:1};const result=snapshotSafetyContext({captured_at:new Date(at*1000).toISOString(),sections:{childhood:[child]}});assert.equal(result.childhood_cartography.suicide_recognized,true);
});
test('snapshot review quotes use original strings rather than JSON escape sequences',async t=>{
 const s=await ready(t),quote='Je classe mes notes dans C:\\dossier.';s.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.self_description',?) WHERE user_id='a'").run(quote);
 const calls=provider(t,(purpose)=>purpose==='ambre_snapshot_chunk'?{summary:'La personne classe ses notes.',quotes:[]}:purpose==='ambre_review'?{content:'Vous écrivez « '+quote+' » '+output.content}:purpose==='ambre_review_check'?PASS:undefined);
 const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-quote-escaped'});assert.equal(s.raw.prepare('SELECT status FROM ambre_reviews WHERE id=?').get(r.review_id).status,'ready');const check=JSON.parse(calls.find(c=>c.purpose==='ambre_review_check').body.messages[1].content);assert.deepEqual(check.unfound_quotes,[]);
});
test('owner snapshot read rechecks consent and deletion after loading the immutable row',async t=>{
 const s=await ready(t);mock(t);const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-read-race'});const prepare=s.env.DB.prepare.bind(s.env.DB);let raced=false;
 s.env.DB.prepare=sql=>{const st=prepare(sql);if(sql.startsWith('SELECT * FROM review_snapshots')){const first=st.first.bind(st);st.first=async()=>{const row=await first();if(!raced){raced=true;s.raw.prepare("UPDATE chat_settings SET consent=0 WHERE user_id='a'").run();s.raw.prepare("DELETE FROM review_snapshots WHERE user_id='a'").run();}return row;};}return st;};
 const response=await s.call(`/api/reviews/${r.review_id}/snapshot?section=profile`);assert.equal(response.status,409);assert.equal(response.code,'context_changed');assert.equal(response.items,undefined);
});
test('archive export refuses a concurrently deleted snapshot rather than returning its old copy',async t=>{
 const s=await ready(t);mock(t);await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-export-race'});const prepare=s.env.DB.prepare.bind(s.env.DB);let raced=false;
 s.env.DB.prepare=sql=>{const st=prepare(sql);if(sql.startsWith('SELECT * FROM review_snapshots')){const first=st.first.bind(st);st.first=async()=>{const row=await first();if(!raced){raced=true;s.raw.prepare("DELETE FROM review_snapshots WHERE user_id='a'").run();}return row;};}return st;};
 await assert.rejects(exportReviewSnapshots(s.env,'a'),e=>e.code==='context_changed'&&e.status===409);
});
test('an explicit conversation-generation reset destroys archives that could retain forgotten data',async t=>{
 const s=await ready(t);mock(t);const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-forget-boundary'});assert.ok(await readReviewSnapshot(s.env,'a',r.review_id));
 s.raw.prepare("INSERT INTO change_plans(user_id,document_json,updated_at) VALUES('a','{}',0) ON CONFLICT(user_id) DO NOTHING").run();s.raw.prepare("UPDATE change_plans SET context_generation=context_generation+1 WHERE user_id='a'").run();
 assert.equal(await readReviewSnapshot(s.env,'a',r.review_id),null);const page=await s.call(`/api/reviews/${r.review_id}/snapshot`);assert.equal(page.available,false);assert.equal(page.reason,'snapshot_removed');
});
// Plan v1.1: the final source read before a call is now the presence guard of the snapshot items.
for(const mutation of ['consent','lease'])test(`review provider rechecks ${mutation} after the final snapshot source read and refunds unsent reservations`,async t=>{
 const s=await ready(t),calls=mock(t),prepare=s.env.DB.prepare.bind(s.env.DB);let raced=false;
 s.env.DB.prepare=sql=>{const st=prepare(sql);if(sql.startsWith('SELECT 1 AS ok WHERE')){const first=st.first.bind(st);st.first=async()=>{const row=await first();const reserved=s.raw.prepare('SELECT sum(provider_calls) AS n FROM provider_budgets').get()?.n;if(reserved&&!raced){raced=true;if(mutation==='consent')s.raw.prepare("UPDATE chat_settings SET consent=0 WHERE user_id='a'").run();else s.raw.prepare("UPDATE ambre_reviews SET lease_token=NULL WHERE user_id='a' AND status='running'").run();}return row;};}return st;};
 await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-final-guard-'+mutation});assert.equal(raced,true);assert.equal(calls.length,0);
 const budget=s.raw.prepare('SELECT sum(provider_calls) AS calls,sum(reserved_tokens) AS tokens FROM provider_budgets').get();assert.equal(budget.calls,0);assert.equal(budget.tokens,0);
});
// Plan v1.1 (point 13): a running review keeps its internal snapshot (never exposed); a failed
// review loses it (trigger 0033 and runReview); a ready review keeps it.
test('a failed review keeps no snapshot; a running one keeps it unexposed; a ready one is untouched',async t=>{
 const s=await ready(t);mock(t);const kept=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-ready-kept'});assert.ok(await readReviewSnapshot(s.env,'a',kept.review_id));
 s.tick(32*86400);const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-unpublished'});
 s.raw.prepare('UPDATE ambre_reviews SET status=? WHERE id=?').run('failed',r.review_id);
 assert.equal(await readReviewSnapshot(s.env,'a',r.review_id),null,'failed review: snapshot erased');assert.ok(await readReviewSnapshot(s.env,'a',kept.review_id),'ready review untouched');
 const failedPage=await s.call(`/api/reviews/${r.review_id}/snapshot?section=profile`);assert.equal(failedPage.available,false);assert.equal(failedPage.reason,'review_failed');
});
test('unfinished reviews keep internal snapshots but expose no assessment through owner API or export',async t=>{
 const s=await ready(t);mock(t);const r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-unpublished'});
 for(const status of ['running']){
  s.raw.prepare('UPDATE ambre_reviews SET status=? WHERE id=?').run(status,r.review_id);
  assert.ok(await readReviewSnapshot(s.env,'a',r.review_id));const response=await s.call(`/api/reviews/${r.review_id}/snapshot?section=profile`);
  assert.equal(response.available,false);assert.equal(response.reason,status==='running'?'review_in_progress':'review_failed');assert.equal(response.review_status,status);assert.equal(response.assessment,undefined);assert.equal(response.items,undefined);
  const exported=(await exportReviewSnapshots(s.env,'a'))[0];assert.equal(exported.review_status,status);assert.equal(exported.assessment,null);assert.ok(exported.payload.sections.profile.length);
 }
});
// The public catalogue left the Worker: a new review no longer captures liked articles or the
// retired personal stories, while an archive captured before still opens, pages, exports and
// keeps its labels in « Mes bilans ».
test('new snapshots no longer capture readings or retired stories; an archive that holds them still loads and displays',async t=>{
 const {snapshotPresenceGuard}=await import('../src/review-snapshot.mjs'),{REVIEW_ARCHIVE_SCRIPT}=await import('../src/review-archive-interface.mjs'),{sharedSnapshotPage}=await import('../src/shared-review-snapshot.mjs');
 const s=await ready(t);s.raw.prepare("INSERT INTO article_likes(user_id,article_id,reason,differences) VALUES('a','posts/c/ancien-recit','LECTURE_AIMEE_PRIVEE','')").run();
 const calls=mock(t),r=await s.call('/api/chat/review','POST',{idempotency_key:'snapshot-without-catalogue'});assert.equal(s.raw.prepare('SELECT status FROM ambre_reviews WHERE id=?').get(r.review_id).status,'ready');
 const retired=['reading_choices','personal_characters','personal_stories','story_feedback','inspiration_choices'],fresh=await readReviewSnapshot(s.env,'a',r.review_id);
 for(const section of retired)assert.equal(Object.hasOwn(fresh.payload.sections,section),false,section);
 assert.ok(fresh.payload.sections.profile.length&&fresh.payload.sections.people.length);assert.doesNotMatch(JSON.stringify(calls),/LECTURE_AIMEE_PRIVEE|ancien-recit/);
 // The same archive as it was stored before the retirement (rows are immutable: it is re-inserted).
 const stored=s.raw.prepare('SELECT * FROM review_snapshots WHERE review_id=?').get(r.review_id),payload=JSON.parse(stored.payload);
 payload.sections={...payload.sections,reading_choices:[{article_id:'posts/c/ancien-recit',reason:'LECTURE_AIMEE_PRIVEE',differences:'',revision:1,created_at:'2026-10-01 10:00:00',updated_at:'2026-10-01 10:00:00'}],personal_characters:[],personal_stories:[],story_feedback:[],inspiration_choices:[]};
 s.raw.prepare('DELETE FROM review_snapshots WHERE review_id=?').run(r.review_id);
 s.raw.prepare('INSERT INTO review_snapshots(review_id,user_id,captured_at,schema_version,payload,assessment,source_stamp) VALUES(?,?,?,?,?,?,?)').run(stored.review_id,stored.user_id,stored.captured_at,stored.schema_version,JSON.stringify(payload),stored.assessment,stored.source_stamp);
 const old=await readReviewSnapshot(s.env,'a',r.review_id);assert.equal(old.payload.sections.reading_choices[0].reason,'LECTURE_AIMEE_PRIVEE');
 const overview=await s.call('/api/reviews/'+r.review_id+'/snapshot');assert.equal(overview.available,true);assert.deepEqual(overview.sections.find(x=>x.id==='reading_choices'),{id:'reading_choices',count:1});
 const page=await s.call('/api/reviews/'+r.review_id+'/snapshot?section=reading_choices');assert.equal(page.total,1);assert.equal(page.items[0].article_id,'posts/c/ancien-recit');assert.equal(page.items[0].reason,'LECTURE_AIMEE_PRIVEE');
 for(const section of retired.slice(1))assert.equal((await s.call('/api/reviews/'+r.review_id+'/snapshot?section='+section)).total,0,section);
 assert.ok(snapshotChunks(old.payload).some(chunk=>chunk.items.some(item=>item.path==='reading_choices/0'&&item.text.includes('LECTURE_AIMEE_PRIVEE'))));
 assert.equal((await exportReviewSnapshots(s.env,'a'))[0].payload.sections.reading_choices.length,1);
 // The entry the archive holds is still checked against the table, which stays in the database.
 const guard=snapshotPresenceGuard(old.payload,'a');assert.match(guard.sql,/FROM article_likes WHERE user_id=\? AND article_id IN/);
 const holds=()=>s.raw.prepare('SELECT '+guard.sql+' AS ok').get(...guard.args).ok;assert.equal(holds(),1);
 s.raw.prepare("DELETE FROM article_likes WHERE user_id='a'").run();assert.equal(holds(),0);
 // A professional never receives these sections, and the page still names them.
 const shared=sharedSnapshotPage(old,['portrait','people','life_events','conversation','reviews','plan','memory','cartography','symptoms','diagnoses']);
 assert.equal(shared.sections.some(x=>retired.includes(x.id)),false);assert.ok(shared.sections.some(x=>x.id==='people'));
 const labels=new Function(REVIEW_ARCHIVE_SCRIPT.match(/var ARCHIVE_SECTION_LABELS=\{[^}]*\}/)[0]+';return ARCHIVE_SECTION_LABELS')();
 assert.deepEqual(retired.map(id=>labels[id]),['Articles aimés','Personnages personnels','Histoires personnelles','Retours sur les histoires','Choix d’inspiration']);
});
