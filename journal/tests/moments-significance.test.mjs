// Plan « Mes symptômes » v1.1, Moments: « Anecdotique / Marquant » (migration 0027).
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {setup,database} from './v4-R-helper.mjs';
import {replayTurn,replayReview} from './v5-A-helper.mjs';
import {knowledgeIndex,analysisIndex} from '../src/ambre-engine.mjs';
import {byMomentTier,momentTier,significanceReady} from '../src/context-fields.mjs';
import {HTML} from '../src/ui.mjs';

const SQL=readFileSync(new URL('../migrations/0027_life_event_significance.sql',import.meta.url),'utf8');
const moment=(id,extra={})=>({creation_id:'moment-'+id+'-0001',title:'Moment '+id,description:'Récit '+id,kind:'other',date_precision:'year',date_start:'2010',...extra});

test('0027: one nullable column, no row rewritten, no trigger fired, no cache invalidated',()=>{
 const body=SQL.replace(/--[^\n]*/g,'').trim();
 assert.doesNotMatch(body,/\b(DROP|DELETE|UPDATE|RENAME|INSERT|TRIGGER|DEFAULT)\b/i);
 assert.deepEqual(body.split(';').map(s=>s.trim()).filter(Boolean),["ALTER TABLE life_events ADD COLUMN significance TEXT CHECK(significance IN ('anecdotal','major'))"]);
 const db=database('0027');
 db.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','u@example.test','h','s');INSERT INTO profiles(user_id) VALUES('u');INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES('e','u','c-e','hash-e','Titre','other','Récit');INSERT INTO context_source_derivatives(user_id,source_id,revision,analysis_version,source_hash,derivative,manifest) VALUES('u','event:e',1,1,'h','{}','{}')");
 const before={event:db.prepare("SELECT revision,payload_hash,updated_at FROM life_events WHERE id='e'").get(),version:db.prepare("SELECT version FROM profiles WHERE user_id='u'").get().version,care:db.prepare('SELECT * FROM care_source_revisions ORDER BY user_id,scope').all().map(r=>({...r}))};
 db.exec(SQL);
 assert.deepEqual({...db.prepare("SELECT revision,payload_hash,updated_at FROM life_events WHERE id='e'").get()},{...before.event});
 assert.equal(db.prepare("SELECT significance FROM life_events WHERE id='e'").get().significance,null);
 assert.equal(db.prepare("SELECT version FROM profiles WHERE user_id='u'").get().version,before.version);
 assert.equal(db.prepare("SELECT count(*) n FROM context_source_derivatives WHERE source_id='event:e'").get().n,1,'derivative cache kept');
 assert.ok(before.care.length,'care revision seeded by the insert');assert.deepEqual(db.prepare('SELECT * FROM care_source_revisions ORDER BY user_id,scope').all().map(r=>({...r})),before.care,'care_revision_* not fired');
 assert.throws(()=>db.prepare("UPDATE life_events SET significance='important' WHERE id='e'").run(),/CHECK/);
 db.prepare("UPDATE life_events SET significance='major' WHERE id='e'").run();db.prepare("UPDATE life_events SET significance=NULL WHERE id='e'").run();
 assert.throws(()=>db.exec(SQL),/duplicate column/);db.close();
});

test('0027: before the migration the Worker ignores the field without error',async()=>{
 const s=setup({until:'0027'});assert.equal(await significanceReady(s.env),false);
 const created=await s.call('/api/life-events','POST',{...moment('a'),significance:'major',entry:'form'});assert.equal(created.status,201,JSON.stringify(created));
 assert.equal('significance' in created.event,false);
 const list=await s.call('/api/life-events');assert.equal(list.significance_ready,false);
 const patched=await s.call('/api/life-events/'+created.event.id,'PATCH',{revision:1,significance:'anecdotal'});assert.equal(patched.status,200);assert.equal(patched.event.revision,1,'an ignored field is not a change');
 const context=await s.personal.context(s.env,'a');assert.equal(context.events[0].significance,undefined);
});

test('0027: API validation, form requirement, a modification without the field keeps the value',async()=>{
 const s=setup();assert.equal(await significanceReady(s.env),true);
 const missing=await s.call('/api/life-events','POST',{...moment('a'),entry:'form'});assert.equal(missing.status,400);assert.equal(missing.code,'significance_required');
 const empty=await s.call('/api/life-events','POST',{...moment('a'),entry:'form',significance:''});assert.equal(empty.code,'significance_required');
 const wrong=await s.call('/api/life-events','POST',{...moment('a'),entry:'form',significance:'important'});assert.equal(wrong.status,400);assert.equal(wrong.code,'invalid_input');
 assert.equal(s.raw.prepare('SELECT count(*) n FROM life_events').get().n,0);
 const form=await s.call('/api/life-events','POST',{...moment('a'),entry:'form',significance:'major',current_importance:2});assert.equal(form.status,201);assert.equal(form.event.significance,'major');assert.equal(form.event.current_importance,2,'distinct from current_importance');
 const again=await s.call('/api/life-events','POST',{...moment('a'),entry:'form',significance:'major',current_importance:2});assert.equal(again.status,200,'idempotent retry');assert.equal(again.event.id,form.event.id);
 const other=await s.call('/api/life-events','POST',moment('b'));assert.equal(other.status,201,'an API creation outside the form stays « à préciser »');assert.equal(other.event.significance,null);
 const kept=await s.call('/api/life-events/'+form.event.id,'PATCH',{revision:1,description:'Récit précisé'});assert.equal(kept.event.significance,'major');assert.equal(kept.event.revision,2);
 const nulls=await s.call('/api/life-events/'+form.event.id,'PATCH',{revision:2,significance:null,description:'Récit précisé encore'});assert.equal(nulls.event.significance,'major','an empty value keeps the stored one');
 const changed=await s.call('/api/life-events/'+form.event.id,'PATCH',{revision:3,significance:'anecdotal'});assert.equal(changed.event.significance,'anecdotal');assert.equal(changed.event.current_importance,2);
 const same=await s.call('/api/life-events/'+form.event.id,'PATCH',{revision:4,significance:'anecdotal'});assert.equal(same.event.revision,4,'no change, no new revision');
 assert.equal((await s.call('/api/life-events/'+other.event.id,'PATCH',{revision:1,significance:'mineur'})).code,'invalid_input');
 const read=await s.call('/api/life-events/'+other.event.id);assert.equal(read.significance_ready,true);assert.equal(read.event.significance,null);
 // Export and account deletion carry the field.
 const exported=await s.personal.exportData(s.env,'a');assert.deepEqual(exported.events.map(e=>e.significance).sort(),['anecdotal',null].sort());
 s.raw.prepare("DELETE FROM users WHERE id='a'").run();assert.equal(s.raw.prepare("SELECT count(*) n FROM life_events WHERE user_id='a'").get().n,0);
});

test('0027: a moment « à préciser » keeps its source hash; choosing changes it like any edit',async()=>{
 const s=setup(),e=await s.call('/api/life-events','POST',moment('a'));
 const hash=(await s.personal.context(s.env,'a')).source_manifest['event:'+e.event.id].hash;
 const fresh=setup({until:'0027'}),f=await fresh.call('/api/life-events','POST',moment('a'));
 assert.equal((await fresh.personal.context(fresh.env,'a')).source_manifest['event:'+f.event.id].hash,hash,'same hash with or without the column while it is empty');
});

test('Ambre sort: cited first, then marking moments (and every exposure), then the current order',async()=>{
 const events=[{id:'a'},{id:'b',significance:'anecdotal'},{id:'c',significance:'major'},{id:'d',significance:'anecdotal',exposure:'happened_to_me'},{id:'e'}];
 assert.deepEqual(byMomentTier(events).map(e=>e.id),['c','d','a','b','e']);
 assert.equal(momentTier({significance:'anecdotal',exposure:'witnessed'}),0,'never relegated because noted anecdotal');
 const s=setup();
 for(const [id,extra] of [['a',{}],['b',{significance:'anecdotal'}],['c',{significance:'major'}],['d',{significance:'anecdotal'}],['e',{title:'Voyage en Islande'}]]){const r=await s.call('/api/life-events','POST',{...moment(id),...extra});assert.equal(r.status,201)}
 const d=s.raw.prepare("SELECT id,revision FROM life_events WHERE title='Moment d'").get();await s.call('/api/life-events/'+d.id,'PATCH',{revision:d.revision,exposure:'witnessed'});
 const context=await s.personal.context(s.env,'a'),index=knowledgeIndex(context),titles=index.filter(x=>x.kind==='event').map(x=>x.label.split(' · ')[0]);
 assert.deepEqual(titles,['Moment c','Moment d','Moment a','Moment b','Voyage en Islande']);
 const major=index.find(x=>x.label.startsWith('Moment c'));assert.match(major.label,/marquant/);assert.match(major.header,/Dans mon histoire : marquant/);
 assert.match(index.find(x=>x.label.startsWith('Moment b')).header,/Dans mon histoire : anecdotique/);
 assert.doesNotMatch(index.find(x=>x.label.startsWith('Moment a')).header,/Dans mon histoire/,'nothing said for a moment « à préciser »');
 // A moment cited in the message comes before a marking one.
 const order=analysisIndex(index,'Je repense au voyage en Islande').entries.filter(x=>x.kind==='event').map(x=>x.label.split(' · ')[0]);assert.equal(order[0],'Voyage en Islande');assert.equal(order[1],'Moment c');
 // Review / prompt context: same tiers, readable words.
 const review=await replayReview('initial',{extend:async r=>{r.raw.prepare("UPDATE life_events SET significance='anecdotal'").run();r.raw.prepare("UPDATE life_events SET significance='major' WHERE id='ev2'").run();r.raw.prepare("UPDATE life_events SET exposure='witnessed' WHERE id='ev11'").run();}});
 const listed=review.data.events.map(e=>e.id);assert.deepEqual(listed.slice(0,2).sort(),['ev11','ev2'],'marking and exposed moments first, then the current order');
 assert.equal(review.data.events.find(e=>e.id==='ev2').significance,'marquant');assert.equal(review.data.events.find(e=>e.id==='ev1').significance,'anecdotique');
 assert.ok(review.data.events.every(e=>e.current_importance!==undefined),'current_importance still sent');
});

test('Ambre never proposes the value',()=>{
 const source=readFileSync(new URL('../src/chat-backend.mjs',import.meta.url),'utf8');
 assert.doesNotMatch(source.match(/const fields=\{[^\n]*/)[0],/significance/,'absent from the fields Ambre may propose');
});

test('Budget: every moment noted keeps the turn and the review within +10 %',async t=>{
 const all=s=>{s.raw.prepare("UPDATE life_events SET significance=CASE WHEN current_importance>=4 THEN 'major' ELSE 'anecdotal' END").run()};
 for(const strategy of ['ideal','none']){const run=await replayTurn('Je repense au divorce de mes parents et à mon père.',{strategy,extend:all});assert.equal(run.stage,'done');const writing=run.get('ambre_writing').tokens;t.diagnostic(JSON.stringify({strategy,writing,analysis:run.get('ambre_analysis').tokens}));assert.ok(writing<=5953*1.1&&writing<=6321*1.1);assert.ok(run.get('ambre_analysis').tokens<=2478*1.1)}
 for(const [kind,ref] of [['initial',12423],['point',12525]]){const run=await replayReview(kind,{extend:all});t.diagnostic(JSON.stringify({kind,tokens:run.tokens,ratio:+(run.tokens/ref).toFixed(3)}));assert.ok(run.tokens<=ref*1.1)}
});

// Interface: the proposal of a new moment asks for the choice and sends it apart from Ambre's payload.
test('UI: accepting a proposed moment asks « Anecdotique / Marquant » and sends the choice apart',async()=>{
 const script=HTML.match(/<script>([\s\S]*?)<\/script>/)[1];
 assert.match(script,/var SIGNIFICANCE_CHOICES=\[\['anecdotal','Anecdotique'\],\['major','Marquant'\]\];/);
 assert.match(script,/askSignificance\?\{significance:choices\.get\(proposal\.id\)\}/);
 assert.match(script,/significanceField\(grid,values\.significance,\{required:!draft\.id/);
 assert.match(script,/payload\.entry='form'/);
 // Pure helpers, run in a sandbox.
 const code=name=>{const start=script.search(new RegExp('function '+name+'\\('));let depth=0,i=script.indexOf('{',start);for(;i<script.length;i++){if(script[i]==='{')depth++;else if(script[i]==='}'&&!--depth)break}return script.slice(start,i+1)};
 const ctx=vm.createContext({});vm.runInContext(code('significanceClass'),ctx);
 assert.equal(ctx.significanceClass({significance:'major'}),'is-major');assert.equal(ctx.significanceClass({significance:'anecdotal'}),'is-anecdotal');assert.equal(ctx.significanceClass({significance:null}),'is-unset');
});
