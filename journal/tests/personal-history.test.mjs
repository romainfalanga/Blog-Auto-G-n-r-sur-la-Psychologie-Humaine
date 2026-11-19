// Migration 0034: dated history of people, moments and notes (Mes proches, Ma ligne de vie).
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {setup,database} from './v4-R-helper.mjs';
import {historyReady,HISTORY_LIMIT} from '../src/personal-history.mjs';

const SQL=readFileSync(new URL('../migrations/0034_personal_history.sql',import.meta.url),'utf8');
let serial=0;
const person=async(s,extra={},user='a')=>(await s.call('/api/people','POST',{creation_id:'person-h-'+(++serial),display_name:'Léa',role:'friend',relationship_text:'On se parle chaque semaine.',...extra},user)).person;
const moment=async(s,extra={},user='a')=>(await s.call('/api/life-events','POST',{creation_id:'event-h-'+(++serial),title:'Déménagement',description:'Un récit.',kind:'turning_point',date_precision:'year',date_start:'2019',...extra},user)).event;
const history=(s,type,id,user='a')=>s.call('/api/history?type='+type+'&id='+encodeURIComponent(id),'GET',undefined,user);
const rows=(s,type,id)=>s.raw.prepare('SELECT * FROM personal_history WHERE entity_type=? AND entity_id=? ORDER BY id').all(type,id);

test('0034: additive only, existing rows and revisions untouched, no history invented for them',()=>{
 const body=SQL.replace(/--[^\n]*/g,'');
 assert.doesNotMatch(body,/\b(DROP|ALTER|RENAME)\b/i);
 assert.doesNotMatch(body,/UPDATE\s+(personal_people|life_events|personal_notes|profiles|care_source_revisions)\s+SET/i,'no existing table is written');
 const db=database('0034');
 db.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','u@example.test','h','s');INSERT INTO profiles(user_id) VALUES('u');INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role) VALUES('p','u','c-p','h-p','Léa','friend');INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES('e','u','c-e','h-e','Titre','other','Récit');INSERT INTO personal_notes(id,user_id,person_id,creation_id,payload_hash,text,created_at,updated_at) VALUES('n','u','p','c-n','h','Note',10,10)");
 const snapshot=()=>JSON.stringify({people:db.prepare('SELECT * FROM personal_people').all(),events:db.prepare('SELECT * FROM life_events').all(),notes:db.prepare('SELECT * FROM personal_notes').all(),profile:db.prepare('SELECT version FROM profiles').get(),care:db.prepare('SELECT * FROM care_source_revisions ORDER BY scope').all()});
 const before=snapshot();db.exec(SQL);assert.equal(snapshot(),before);
 assert.equal(db.prepare('SELECT count(*) n FROM personal_history').get().n,0,'existing elements have no earlier history');
 assert.throws(()=>db.exec(SQL),/already exists/);db.close();
});

test('triggers: creation, content changes only, previous state kept, notes added to a parent record nothing on it',async()=>{
 const s=setup(),p=await person(s);
 assert.deepEqual(rows(s,'person',p.id).map(r=>[r.action,r.revision]),[['create',1]]);
 const n=(await s.call('/api/people/'+p.id+'/notes','POST',{creation_id:'note-h-0001',text:'Première version'})).note;
 assert.equal(rows(s,'person',p.id).length,1,'a note bumps the parent revision without a version of the person');
 const fresh=(await s.call('/api/people/'+p.id)).person;assert.equal(fresh.revision,2);
 const same=await s.call('/api/people/'+p.id,'PATCH',{revision:2,display_name:'Léa'});assert.equal(same.status,200);assert.equal(rows(s,'person',p.id).length,1,'no change, no version');
 await s.call('/api/people/'+p.id,'PATCH',{revision:2,display_name:'Léa B.',relationship_text:'Nous nous voyons moins.'});
 const update=rows(s,'person',p.id)[1];assert.equal(update.action,'update');assert.equal(update.revision,2);
 assert.equal(JSON.parse(update.snapshot).display_name,'Léa');assert.equal(JSON.parse(update.snapshot).relationship_text,'On se parle chaque semaine.');
 s.tick(60);await s.call('/api/people/'+p.id+'/notes/'+n.id,'PATCH',{revision:1,text:'Seconde version'});
 assert.deepEqual(rows(s,'note',n.id).map(r=>[r.action,JSON.parse(r.snapshot).text,r.changed_at]),[['create','Première version',s.now()-60],['update','Première version',s.now()]]);
 assert.ok(s.raw.prepare("SELECT revision FROM care_source_revisions WHERE user_id='a' AND scope='people'").get().revision>=3,'care_revision_* still follow every change');
});

test('GET /api/history: newest first, changed fields with old and new values, the creation last',async()=>{
 const s=setup(),p=await person(s),q=await person(s,{display_name:'Marc',role:'sibling'});
 const e=await moment(s,{people_ids:[p.id]});
 await s.call('/api/life-events/'+e.id,'PATCH',{revision:1,date_precision:'month',date_start:'2019-06',people_ids:[p.id,q.id],significance:'major'});
 await s.call('/api/life-events/'+e.id,'PATCH',{revision:2,description:'Un récit précisé.'});
 const h=await history(s,'event',e.id);assert.equal(h.status,200);assert.equal(h.available,true);assert.equal(h.complete,true);
 assert.deepEqual(h.versions.map(v=>v.action),['update','update','create']);
 assert.deepEqual(h.versions[0].changes,[{field:'description',before:'Un récit.',after:'Un récit précisé.'}]);
 const dates=Object.fromEntries(h.versions[1].changes.map(c=>[c.field,[c.before,c.after]]));
 assert.deepEqual(dates.date_precision,['year','month']);assert.deepEqual(dates.date_start,['2019','2019-06']);assert.ok('date_end' in dates&&'age_text' in dates,'a date is shown as a whole');
 assert.deepEqual(dates.people_ids,[[p.id],[p.id,q.id].sort()]);assert.deepEqual(dates.significance,[null,'major']);
 assert.deepEqual(h.versions[2].values.people_ids,[p.id],'the created state includes its people');assert.equal(h.versions[2].values.description,'Un récit.');
 assert.ok(h.versions.every(v=>!Number.isNaN(Date.parse(v.at))));assert.ok(Date.parse(h.created_at));
 // A deleted person disappears from every state of the moment.
 await s.call('/api/people/'+q.id,'DELETE',{revision:1});
 const after=await history(s,'event',e.id);assert.ok(!JSON.stringify(after.versions).includes(q.id));
 // Notes: the previous text.
 const n=(await s.call('/api/life-events/'+e.id+'/notes','POST',{creation_id:'note-h-0002',text:'Avant'})).note;
 s.tick(5);await s.call('/api/life-events/'+e.id+'/notes/'+n.id,'PATCH',{revision:1,text:'Après'});
 const nh=await history(s,'note',n.id);assert.deepEqual(nh.versions.map(v=>v.action),['update','create']);assert.deepEqual(nh.versions[0].changes,[{field:'text',before:'Avant',after:'Après'}]);assert.equal(nh.versions[1].values.text,'Avant');
});

test('owner only: another account, an unknown element or a bad request get nothing',async()=>{
 const s=setup(),p=await person(s);
 const other=await history(s,'person',p.id,'b');assert.equal(other.status,404);assert.equal(other.versions,undefined);
 assert.equal((await history(s,'event',p.id)).status,404,'the type is part of the identity');
 assert.equal((await history(s,'portrait',p.id)).code,'invalid_input');assert.equal((await history(s,'person','../x')).code,'invalid_input');
 assert.equal((await s.call('/api/history?type=person&id='+p.id,'POST',{})).status,405);
});

test('elements saved before the migration: no invented creation, the first change keeps the state they had',async()=>{
 const s=setup();
 s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role) VALUES('old','a','c-old','legacy','Ancien','friend')").run();
 s.raw.prepare("DELETE FROM personal_history WHERE entity_id='old'").run();
 let h=await history(s,'person','old');assert.equal(h.complete,false);assert.deepEqual(h.versions,[]);
 await s.call('/api/people/old','PATCH',{revision:1,display_name:'Ancien nom changé'});
 h=await history(s,'person','old');assert.equal(h.complete,false);assert.deepEqual(h.versions.map(v=>v.action),['update']);assert.deepEqual(h.versions[0].changes,[{field:'display_name',before:'Ancien',after:'Ancien nom changé'}]);
});

test(`at most ${HISTORY_LIMIT} versions per element, the oldest go first`,async()=>{
 const s=setup(),e=await moment(s);
 for(let i=1;i<=HISTORY_LIMIT+5;i++)await s.call('/api/life-events/'+e.id,'PATCH',{revision:i,description:'Version '+i});
 const kept=rows(s,'event',e.id);assert.equal(kept.length,HISTORY_LIMIT);assert.ok(kept.every(r=>r.action==='update'));
 assert.equal(JSON.parse(kept.at(-1).snapshot).description,'Version '+(HISTORY_LIMIT+4));
 const h=await history(s,'event',e.id);assert.equal(h.complete,false);assert.equal(h.versions.length,HISTORY_LIMIT);assert.equal(h.versions[0].changes[0].after,'Version '+(HISTORY_LIMIT+5));
});

test('deletion erases the history of the element and of its notes; export carries it; account deletion removes it',async()=>{
 const s=setup(),p=await person(s),e=await moment(s);
 const n=(await s.call('/api/people/'+p.id+'/notes','POST',{creation_id:'note-h-0003',text:'Note privée'})).note;
 await s.call('/api/people/'+p.id,'PATCH',{revision:2,display_name:'Léa C.'});
 const exported=await s.personal.exportData(s.env,'a');
 assert.ok(Array.isArray(exported.personal_history));assert.ok(exported.personal_history.some(r=>r.entity_type==='person'&&r.action==='update'&&r.state.display_name==='Léa'));
 assert.ok(exported.personal_history.every(r=>!('user_id' in r)&&!Number.isNaN(Date.parse(r.changed_at))));
 assert.ok(!(await s.personal.exportData(s.env,'b')).personal_history.length,'another account exports nothing of it');
 await s.call('/api/people/'+p.id,'DELETE',{revision:3});
 assert.equal(rows(s,'person',p.id).length,0);assert.equal(rows(s,'note',n.id).length,0,'the cascaded note loses its history');
 assert.equal(rows(s,'event',e.id).length,1);
 s.raw.prepare("DELETE FROM users WHERE id='a'").run();assert.equal(s.raw.prepare("SELECT count(*) n FROM personal_history WHERE user_id='a'").get().n,0);
});

test('the history is never part of what Ambre reads',async()=>{
 const s=setup(),p=await person(s,{relationship_text:'Ancienne description à oublier.'});
 await s.call('/api/people/'+p.id,'PATCH',{revision:1,relationship_text:'Description actuelle.'});
 const context=JSON.stringify(await s.personal.context(s.env,'a'));
 assert.ok(context.includes('Description actuelle.'));assert.ok(!context.includes('Ancienne description à oublier.'));
});

test('before the migration the Worker answers « pas encore disponible » and everything else works',async()=>{
 const s=setup({until:'0034'});assert.equal(await historyReady(s.env),false);
 const p=await person(s);assert.ok(p.id);
 const h=await history(s,'person',p.id);assert.equal(h.status,200);assert.equal(h.available,false);assert.deepEqual(h.versions,[]);assert.ok(h.created_at);
 assert.equal((await history(s,'person',p.id,'b')).status,404);
 assert.deepEqual((await s.personal.exportData(s.env,'a')).personal_history,[]);
});
