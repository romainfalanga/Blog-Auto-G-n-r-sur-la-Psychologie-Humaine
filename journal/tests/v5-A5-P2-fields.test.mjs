import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {setup,database,provider} from './v4-R-helper.mjs';
import {knowledgeIndex} from '../src/ambre-engine.mjs';
import {PERSON_P2_FIELDS,EVENT_P2_FIELDS,contextFieldsReady,unsafeRelation} from '../src/context-fields.mjs';
import {signalGuidance,signalReview} from '../src/clinical-signals.mjs';
const person={creation_id:'person-create-01',display_name:'Alex',role:'partner',bond_quality:'ambivalent',bond_note:'Parfois proche',presence_now:'daily',helps:'Les promenades ensemble',hurts:'Les silences',evolution:'Je prends du recul',feels_safe:'not_always'};
const event={creation_id:'event-create-01',title:'Un départ',description:'Un départ important',kind:'turning_point',felt_then:'De la peur',felt_now:'Du soulagement',impact:'Je vis autrement',support_level:'some',support_text:'Une amie',exposure:'witnessed'};

test('A5-P2: additive 0024 preserves existing data and must not be replayed',()=>{
 const db=database('0024');db.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('a','a@example.test','h','s'); INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES('p','a','key','hash','Alex','friend','Ancien récit');");
 const sql=readFileSync(new URL('../migrations/0024_person_moment_fields.sql',import.meta.url),'utf8');
 assert.doesNotMatch(sql.replace(/--[^\n]*/g,''),/\b(DROP|DELETE|UPDATE|RENAME)\b/i);db.exec(sql);
 assert.equal(db.prepare("SELECT relationship_text FROM personal_people WHERE id='p'").get().relationship_text,'Ancien récit');
 for(const field of PERSON_P2_FIELDS)assert.equal(db.prepare(`SELECT ${field} AS value FROM personal_people WHERE id='p'`).get().value,'');
 assert.throws(()=>db.exec(sql),/duplicate column/);db.close();
});

test('A5-P2: all optional fields persist, reach Ambre without an older narrative, export and clear',async()=>{
 const s=setup(),p=await s.call('/api/people','POST',person);assert.equal(p.status,201,JSON.stringify(p));
 const e=await s.call('/api/life-events','POST',{...event,people_ids:[p.person.id]});assert.equal(e.status,201,JSON.stringify(e));
 for(const k of PERSON_P2_FIELDS)assert.equal(p.person[k],person[k]);
 for(const k of EVENT_P2_FIELDS)assert.equal(e.event[k],event[k]);
 const context=await s.personal.context(s.env,'a'),index=knowledgeIndex(context);
 assert.match(context.sources.find(x=>x.type==='person').text,/Les promenades ensemble/);
 assert.match(context.sources.find(x=>x.type==='event').text,/De la peur/);
 assert.match(index.find(x=>x.kind==='person').text,/pas toujours/);
 assert.match(index.find(x=>x.kind==='event').text,/témoin/);
 const exported=await s.personal.exportData(s.env,'a');assert.equal(exported.people[0].feels_safe,'not_always');assert.equal(exported.events[0].support_text,'Une amie');
 const p2=await s.call('/api/people/'+p.person.id,'PATCH',{revision:1,helps:''});assert.equal(p2.status,200);assert.equal(p2.person.helps,'');assert.equal(p2.person.hurts,person.hurts);
 assert.equal((await s.call('/api/people/'+p.person.id,'PATCH',{revision:1,helps:'Écrasement'})).status,409);
 assert.equal((await s.call('/api/people/'+p.person.id,'PATCH',{revision:2,helps:'Étranger'},'b')).status,404);
 assert.equal((await s.call('/api/life-events/'+e.event.id,'DELETE',{revision:1})).status,200);
 assert.equal((await s.call('/api/people/'+p.person.id,'DELETE',{revision:2})).status,200);
 const empty=await s.personal.exportData(s.env,'a');assert.equal(empty.people.length,0);assert.equal(empty.events.length,0);assert.deepEqual(s.raw.prepare('PRAGMA foreign_key_check').all(),[]);
});

test('A5-P2: choices and limits reject invalid values; empty and omitted values stay optional',async()=>{
 const s=setup();
 for(const data of [{...person,feels_safe:['no']},{...person,feels_safe:'maybe'},{...person,bond_quality:{}},{...person,helps:'a'.repeat(601)},{...person,helps:42}])assert.equal((await s.call('/api/people','POST',data)).status,400);
 for(const data of [{...event,exposure:'victim'},{...event,support_level:'all'},{...event,felt_now:'a'.repeat(601)}])assert.equal((await s.call('/api/life-events','POST',data)).status,400);
 const first=await s.call('/api/people','POST',{creation_id:'minimal-person-01',display_name:'Sam'});assert.equal(first.status,201);assert.equal(first.person.feels_safe,'');
 const repeat=await s.call('/api/people','POST',{creation_id:'minimal-person-01',display_name:'Sam'});assert.equal(repeat.person.id,first.person.id);
});

test('A5-P2: old schema remains usable and filled new fields are never silently lost',async()=>{
 const s=setup({until:'0024'});
 // Required symptom storage is independent of the optional person/moment fields under test.
 s.raw.exec(readFileSync(new URL('../migrations/0029_symptom_maps.sql',import.meta.url),'utf8'));
 assert.equal(await contextFieldsReady(s.env),false);
 assert.equal((await s.call('/api/people')).fields_ready,false);
 const p=await s.call('/api/people','POST',{creation_id:'old-person-0001',display_name:'Alex',helps:''});assert.equal(p.status,201,JSON.stringify(p));
 assert.equal((await s.call('/api/people/'+p.person.id,'PATCH',{revision:1,helps:'Un appui'})).code,'feature_unavailable');
 assert.equal((await s.personal.context(s.env,'a')).people.length,1);
 const self=await s.call('/api/self-profile','PUT',{expected_revision:1,what_brings_me:'Comprendre mes réactions',what_helps_me:'Marcher',life_rhythm:'Travail de nuit'});
 assert.equal(self.status,200,JSON.stringify(self));assert.equal((await s.personal.context(s.env,'a')).has_self_description,true);
 assert.equal(knowledgeIndex(await s.personal.context(s.env,'a')).filter(x=>x.kind==='profile').length,3);
 assert.equal(JSON.parse((await s.personal.exportData(s.env,'a')).self_profile.self_data).life_rhythm,'Travail de nuit');
});

test('A5-P2: unsafe relationship protects named and follow-up turns without labelling other people',async t=>{
 const people=[{...person,id:'p'},{display_name:'Sam',role:'friend',feels_safe:'yes'}];
 const signal=unsafeRelation(people,'Alex veut discuter ce soir.');assert.equal(signal.relation,'partner');
 assert.deepEqual(unsafeRelation(people,'Il insiste encore.','Alex veut discuter ce soir.'),signal);
 assert.deepEqual(unsafeRelation(people,'Comment lui répondre ?','Il insiste encore.',signal),signal);assert.equal(unsafeRelation(people,'Sam aimerait discuter.','Il insiste encore.',signal),null);assert.equal(unsafeRelation(people,'Mon travail me fatigue.'),null);
 assert.match(signalGuidance({unsafe_relation:signal}),/3919/);assert.ok(signalReview('Parlez-lui calmement ce soir.',{unsafe_relation:signal}).includes(2));
 const s=setup();await s.ready();const p=await s.call('/api/people','POST',person);const calls=provider(t);
 await s.call('/api/chat/messages','POST',{content:'Alex veut discuter ce soir, que faire ?',idempotency_key:'unsafe-turn-001'});
 for(let n=0;n<8;n++){const job=s.raw.prepare('SELECT * FROM chat_engine_jobs ORDER BY rowid DESC LIMIT 1').get();if(['done','failed'].includes(job.stage))break;await s.chat.engine.work(s.env,job.turn_id);}
 const writer=calls.find(x=>x.purpose==='ambre_writing');assert.ok(writer,'writer called');assert.match(JSON.stringify(writer.body.messages),/jamais de conseil de confronter/);
 assert.equal(p.person.feels_safe,'not_always');
});
