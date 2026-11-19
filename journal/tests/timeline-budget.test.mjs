import test from 'node:test';
import assert from 'node:assert/strict';
import {replayTurn,replayReview,MESSAGES} from './v5-A-helper.mjs';
// Stress the new notes against the same frozen A5 reference, not a newly enlarged ceiling.
function fillNotes(s){
 for(const [table,column] of [['personal_people','person_id'],['life_events','event_id']])for(const {id} of s.raw.prepare('SELECT id FROM '+table).all())for(let n=0;n<20;n++){
  const key='note-'+id+'-'+n;s.raw.prepare(`INSERT INTO personal_notes(id,user_id,${column},creation_id,payload_hash,text,created_at,updated_at) VALUES(?,'a',?,?,'fixture',?,1794910000,1794910000)`).run(key,id,key,'NOTEPERSO '+('Je précise le souvenir et ce que je ressens. '.repeat(60)).slice(0,1980));
 }
}
const reference={ideal:{proche:{ambre_analysis:2406,ambre_writing:5953,ambre_review:2460},moment:{ambre_analysis:2438,ambre_writing:5937,ambre_review:2444}},none:{proche:{ambre_analysis:2406,ambre_writing:5939,ambre_review:2451},moment:{ambre_analysis:2438,ambre_writing:6066,ambre_review:2494}}};
for(const strategy of ['ideal','none'])for(const label of ['proche','moment'])test(`dated notes keep ${strategy}/${label} inside the frozen +10% stage budgets`,async t=>{
 const run=await replayTurn(MESSAGES.find(([key])=>key===label)[1],{strategy,digests:true,extend:fillNotes});assert.equal(run.stage,'done');assert.equal(run.calls.length,3);const sizes=Object.fromEntries(run.calls.map(c=>[c.purpose,c.tokens]));
 for(const [purpose,tokens] of Object.entries(reference[strategy][label]))assert.ok(sizes[purpose]<=tokens*1.1,purpose+' '+sizes[purpose]+' > '+tokens+'*1.1');
 t.diagnostic(JSON.stringify(sizes));
});
for(const kind of ['initial','point'])test(`legacy before 0031 dated notes keep ${kind} review inside its frozen context budget`,async t=>{
 const run=await replayReview(kind,{legacy:true,digests:true,extend:fillNotes});assert.equal(run.r.status,202);assert.ok(run.tokens<=(kind==='initial'?12423:12525)*1.1);assert.ok(JSON.stringify(run.data).length<=34000);for(const type of ['person','event']){const source=run.data.sources.find(source=>source.type===type&&source.text.includes('NOTEPERSO'));assert.ok(source,type+' includes dated user notes');assert.match(source.text,/Notes écrites par vous/);assert.equal(source.user_notes_total,20);assert.ok(source.user_notes_omitted>0)}t.diagnostic(JSON.stringify({tokens:run.tokens,chars:JSON.stringify(run.data).length}));
});
for(const kind of ['initial','point'])test(`legacy before 0031 the ${kind} review reads notes when the old relation and event narratives are empty`,async()=>{
 const run=await replayReview(kind,{legacy:true,extend:s=>{fillNotes(s);s.raw.prepare("UPDATE personal_people SET relationship_text='' WHERE user_id='a'").run();s.raw.prepare("UPDATE life_events SET description='',meaning_now='' WHERE user_id='a'").run()}});assert.equal(run.r.status,202);
 for(const type of ['person','event'])assert.ok(run.data.sources.some(source=>source.type===type&&source.text.includes('NOTEPERSO')&&source.text.includes('Notes écrites par vous')),type+' note-only source reaches review');
});

for(const kind of ['initial','point'])test(`full ${kind} snapshot refuses the oversized historical notes fixture without a partial review`,async()=>{
 const run=await replayReview(kind,{extend:fillNotes});assert.equal(run.failure.error_code,'snapshot_too_large');assert.equal(run.failure.status,'failed');assert.equal(run.calls.length,0);assert.equal(run.data,null);assert.equal(run.snapshot,null);
 const row=run.s.raw.prepare('SELECT content FROM ambre_reviews WHERE id=?').get(run.r.review_id);assert.equal(row.content,'');
});
