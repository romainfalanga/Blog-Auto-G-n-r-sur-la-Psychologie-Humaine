import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {loadContext,validatePortraitV2} from '../src/private-context.mjs';
import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';

function setup({before=false}={}){
 const raw=new DatabaseSync(':memory:');for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(name=>name.endsWith('.sql')&&(!before||name<'0025')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 const prepare=sql=>{let args=[];return {bind(...values){args=values;return this},async first(){return raw.prepare(sql).get(...args)||null},async all(){return {results:raw.prepare(sql).all(...args)}},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}}}}};
 let queue=Promise.resolve(),serial=0,clock=1794910000;
 const env={DB:{prepare,batch(statements){const task=queue.then(async()=>{raw.exec('BEGIN');try{const results=[];for(const statement of statements)results.push(await statement.run());raw.exec('COMMIT');return results}catch(error){raw.exec('ROLLBACK');throw error}});queue=task.catch(()=>{});return task}}};
 const personal=createPersonalBackend({error(status,message,code){throw Object.assign(Error(message),{status,code})},json:(value,status=200)=>new Response(JSON.stringify(value),{status}),body:request=>request.json(),text:(value,max=1000)=>typeof value==='string'?value.trim().slice(0,max):'',now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null}});
 for(const id of ['a','b']){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.com');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO subscriptions(user_id) VALUES(?)').run(id)}
 async function call(path,method='GET',data,user='a'){const request=new Request('https://local.test'+path,{method,...(data?{body:JSON.stringify(data)}:{})}),url=new URL(request.url);try{const response=await personal.routes(request,env,{id:user},url.pathname,url);return {status:response.status,...await response.json()}}catch(error){return {status:error.status||500,code:error.code,message:error.message}}}
 const person=async(name='Sam',user='a',extra={})=>(await call('/api/people','POST',{creation_id:'person_'+(++serial),display_name:name,role:'friend',...extra},user)).person;
 const event=async(people=[],extra={})=>(await call('/api/life-events','POST',{creation_id:'event___'+(++serial),title:'Déménagement',description:'Un récit existant.',kind:'turning_point',people_ids:people,...extra})).event;
 return {raw,env,personal,call,person,event,tick:()=>clock++,version:()=>raw.prepare("SELECT version FROM profiles WHERE user_id='a'").get().version};
}
const notePath=(kind,id)=>'/api/'+(kind==='person'?'people':'life-events')+'/'+id+'/notes';

test('dated user notes preserve author, owner, target and server dates with idempotent creation',async()=>{
 const s=setup(),p=await s.person(),path=notePath('person',p.id),data={creation_id:'same_note_001',text:'Une précision personnelle.'},v=s.version();
 const [a,b]=await Promise.all([s.call(path,'POST',data),s.call(path,'POST',data)]);assert.ok([200,201].includes(a.status));assert.ok([200,201].includes(b.status));assert.ok([a.status,b.status].includes(201));assert.equal(a.note.id,b.note.id);assert.equal(a.note.author,'user');assert.equal(a.note.target_type,'person');assert.equal(a.note.created_at,'2026-11-17T10:06:40.000Z');assert.equal(s.version(),v+1);
 assert.equal((await s.call(path,'POST',{...data,text:'Autre précision.'})).code,'idempotency_conflict');assert.equal((await s.call(path,'POST',{...data,creation_id:'bad_author',author:'ambre'})).status,400);
 assert.equal((await s.call(path,'GET',undefined,'b')).status,404);assert.equal((await s.call(path+'/'+a.note.id,'DELETE',{revision:1},'b')).status,404);
 assert.equal((await s.call(path,'POST',{creation_id:'long_note',text:'x'.repeat(2001)})).status,400);
});

test('note edits and deletes use CAS, replay safely and never resurrect deleted notes',async()=>{
 const s=setup(),p=await s.person(),path=notePath('person',p.id),creation={creation_id:'note_cas_001',text:'Texte initial.'},n=(await s.call(path,'POST',creation)).note,v=s.version();
 const noop=await s.call(path+'/'+n.id,'PATCH',{revision:1,text:n.text});assert.equal(noop.note.revision,1);assert.equal(s.version(),v);
 s.tick();const edits=await Promise.all(['Version A','Version B'].map(text=>s.call(path+'/'+n.id,'PATCH',{revision:1,text})));assert.deepEqual(edits.map(x=>x.status).sort(),[200,409]);const changed=edits.find(x=>x.status===200);assert.equal(changed.note.created_at,n.created_at);assert.notEqual(changed.note.updated_at,n.updated_at);assert.equal(changed.note.revision,2);
 const replay=await s.call(path+'/'+n.id,'PATCH',{revision:1,text:changed.note.text});assert.equal(replay.status,200);assert.equal(replay.note.revision,2);assert.equal(s.version(),v+1);
 assert.equal((await s.call(path+'/'+n.id,'DELETE',{revision:1})).status,409);const removal={revision:2,mutation_id:'delete_note_001'};assert.equal((await s.call(path+'/'+n.id,'DELETE',removal)).deleted,true);assert.equal((await s.call(path+'/'+n.id,'DELETE',removal)).deleted,true);assert.equal(s.version(),v+2);
 assert.equal((await s.call(path,'POST',creation)).code,'note_deleted');assert.equal((await s.call(path)).notes.length,0);
});

test('one shared event and its notes appear in both timelines; removing one person preserves them',async()=>{
 const s=setup(),p=await s.person('Sam'),q=await s.person('Jo'),e=await s.event([p.id,q.id],{date_precision:'year',date_start:'2020',meaning_now:'Ancien sens.',felt_then:'Ancienne émotion.'}),path=notePath('event',e.id),n=(await s.call(path,'POST',{creation_id:'shared_note_001',text:'Une précision partagée.'})).note;
 for(const person of [p,q]){const timeline=await s.call('/api/people/'+person.id+'/timeline');const entry=timeline.items.find(x=>x.kind==='life_event');assert.equal(entry.event.id,e.id);assert.equal(entry.event.notes[0].id,n.id);assert.equal(entry.event.meaning_now,'Ancien sens.')}
 const current=(await s.call('/api/life-events/'+e.id)).event;const changed=await s.call('/api/life-events/'+e.id,'PATCH',{revision:current.revision,title:'Même événement modifié'});assert.equal(changed.event.felt_then,'Ancienne émotion.');assert.equal(changed.event.notes[0].id,n.id);
 const pn=(await s.call(notePath('person',p.id),'POST',{creation_id:'direct_note_001',text:'Une note de relation.'})).note;const fresh=(await s.call('/api/people/'+p.id)).person;assert.equal((await s.call('/api/people/'+p.id,'DELETE',{revision:fresh.revision})).status,200);
 assert.equal(s.raw.prepare('SELECT id FROM personal_notes WHERE id=?').get(pn.id),undefined);const saved=(await s.call('/api/life-events/'+e.id)).event;assert.deepEqual(saved.people_ids,[q.id]);assert.equal(saved.notes[0].id,n.id);
 assert.equal((await s.call('/api/life-events/'+e.id,'DELETE',{revision:saved.revision})).status,200);assert.equal(s.raw.prepare('SELECT count(*) n FROM personal_notes').get().n,0);assert.deepEqual(s.raw.prepare('PRAGMA foreign_key_check').all(),[]);
});

test('event filtering traverses all pages and rejects foreign people and mismatched note parents',async()=>{
 const s=setup(),p=await s.person(),q=await s.person('Other');for(let i=0;i<56;i++)await s.event([i%2?p.id:q.id],{title:'Événement '+i,date_precision:'year',date_start:String(1960+i)});
 let cursor='',ids=[];do{const page=await s.call('/api/life-events?person_id='+p.id+'&limit=7'+(cursor?'&cursor='+cursor:''));assert.equal(page.status,200);assert.ok(page.events.every(e=>e.people_ids.includes(p.id)));ids.push(...page.events.map(e=>e.id));cursor=page.next_cursor}while(cursor);assert.equal(new Set(ids).size,28);
 const foreign=await s.person('Outside','b');assert.equal((await s.call('/api/life-events?person_id='+foreign.id)).status,404);
 const note=(await s.call(notePath('person',p.id),'POST',{creation_id:'parent_check',text:'Cible exacte.'})).note;assert.equal((await s.call(notePath('person',q.id)+'/'+note.id,'PATCH',{revision:1,text:'Déplacement interdit.'})).status,404);
});

test('meeting dates validate partial dates and preserve old period text, while timeline cursors detect changes',async()=>{
 const s=setup(),p=await s.person('Sam','a',{period_start:'Au lycée',period_end:'Encore aujourd’hui',meeting_date_precision:'month',meeting_date_start:'2019-06'});assert.equal(p.period_start,'Au lycée');assert.equal(p.meeting_date_start,'2019-06');
 for(const fields of [{meeting_date_precision:'day',meeting_date_start:'2023-02-29'},{meeting_date_precision:'month',meeting_date_start:'2020-00'},{meeting_date_precision:'range',meeting_date_start:'2022',meeting_date_end:'2020'}])assert.equal((await s.call('/api/people/'+p.id,'PATCH',{revision:p.revision,...fields})).status,400);
 const e=await s.event([p.id],{date_precision:'year',date_start:'2021'});await s.event([p.id]);const first=await s.call('/api/people/'+p.id+'/timeline?limit=1');assert.equal(first.items[0].kind,'relationship_start');assert.equal(first.items[0].date.start,'2019-06');const second=await s.call('/api/people/'+p.id+'/timeline?limit=1&cursor='+encodeURIComponent(first.next_cursor));assert.equal(second.items[0].event.id,e.id);
 await s.call(notePath('person',p.id),'POST',{creation_id:'cursor_change',text:'Nouvelle note.'});assert.equal((await s.call('/api/people/'+p.id+'/timeline?limit=1&cursor='+encodeURIComponent(first.next_cursor))).code,'pagination_changed');
 const fresh=(await s.call('/api/people/'+p.id)).person;const changed=await s.call('/api/people/'+p.id,'PATCH',{revision:fresh.revision,meeting_date_precision:'age',meeting_age_text:'Vers quinze ans'});assert.equal(changed.person.meeting_date_start,'');assert.equal(changed.person.meeting_age_text,'Vers quinze ans');assert.equal(changed.person.period_end,'Encore aujourd’hui');
});

test('context notes are dated, bounded, attributed and change parent source hashes; export is complete',async()=>{
 const s=setup(),p=await s.person(),e=await s.event([p.id]),path=notePath('event',e.id),before=await loadContext(s.env,'a');
 for(let i=0;i<23;i++){s.tick();assert.equal((await s.call(path,'POST',{creation_id:'context_note_'+i,text:'Note '+i+' '+('x'.repeat(600))})).status,201)}
 const c=await loadContext(s.env,'a'),event=c.events[0],source=c.sources.find(x=>x.id==='event:'+e.id);assert.ok(event.user_notes_text.length<=6000);assert.ok(event.user_notes_omitted>0);assert.equal(event.user_notes_total,23);assert.match(event.user_notes_text,/Note de la personne, 2026/);assert.equal(event.user_notes[0].author,'user');assert.notEqual(source.hash,before.sources.find(x=>x.id===source.id).hash);assert.equal(source.revision,e.revision+23);assert.equal(source.fields.user_notes_text,event.user_notes_text);
 const data=await s.personal.exportData(s.env,'a');assert.equal(data.personal_notes.length,23);assert.equal(data.events.length,1);assert.equal(data.events[0].notes.length,3);assert.deepEqual((await s.personal.exportData(s.env,'b')).personal_notes,[]);
 let cursor='',count=0;do{const page=await s.call(path+'?limit=5'+(cursor?'&cursor='+cursor:''));count+=page.notes.length;cursor=page.next_cursor}while(cursor);assert.equal(count,23);
 const listed=await s.call('/api/life-events');assert.equal(listed.events[0].notes_total,23);const remainder=await s.call(path+'?cursor='+listed.events[0].notes_next_cursor);assert.equal(remainder.notes.length,20);
 s.raw.prepare("DELETE FROM users WHERE id='a'").run();assert.equal(s.raw.prepare('SELECT count(*) n FROM personal_notes').get().n,0);assert.equal(s.raw.prepare('SELECT count(*) n FROM context_mutations').get().n,0);
});

test('migration absence keeps old records readable and refuses new notes and meeting fields without data loss',async()=>{
 const s=setup({before:true}),p=await s.person('Sam','a',{period_start:'2018'}),e=await s.event([p.id]);assert.equal((await s.call(notePath('person',p.id))).available,false);assert.equal((await s.call(notePath('event',e.id),'POST',{creation_id:'before_note',text:'À conserver dans le brouillon.'})).code,'feature_unavailable');assert.equal((await s.call('/api/people/'+p.id,'PATCH',{revision:1,meeting_date_precision:'year',meeting_date_start:'2018'})).code,'feature_unavailable');assert.equal((await s.call('/api/people/'+p.id+'/timeline')).items.length,2);assert.deepEqual((await s.personal.exportData(s.env,'a')).personal_notes,[]);
 s.raw.exec(readFileSync(new URL('../migrations/0025_personal_timeline_notes.sql',import.meta.url),'utf8'));assert.equal((await s.call(notePath('person',p.id))).available,true);assert.equal((await s.call('/api/people/'+p.id)).person.period_start,'2018');assert.throws(()=>s.raw.exec(readFileSync(new URL('../migrations/0025_personal_timeline_notes.sql',import.meta.url),'utf8')));
});

test('relation timeline projects only accepted observations with separate provenance and no write',async()=>{
 const s=setup(),p=await s.person();s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision) VALUES('turn','a','turn','hash','completed','consumed','2026-11-16',1,1)").run();
 for(const origin of ['auto','accepted'])s.raw.prepare('INSERT INTO people_observations(id,user_id,person_id,turn_id,axis,quote,fingerprint,created_at,origin) VALUES(?,?,?,\'turn\',\'appuis\',?,?,?,?)').run(origin,'a',p.id,'Nous cuisinons ensemble chaque dimanche.',origin,1794910000,origin);
 const before=s.version(),response=await s.call('/api/people/'+p.id+'/timeline');assert.equal(response.items.length,2);assert.equal(response.items[0].kind,'relationship_start');const observation=response.items.find(x=>x.kind==='ambre_observation');assert.equal(observation.note.proposed_by,'ambre');assert.equal(observation.note.text_origin,'user_report');assert.equal(observation.note.id,'accepted');assert.equal(s.version(),before);assert.equal(s.raw.prepare('SELECT count(*) n FROM personal_notes').get().n,0);
 const prepare=s.env.DB.prepare;let removed=false;s.env.DB.prepare=sql=>{const statement=prepare(sql);if(sql.startsWith('SELECT id,axis,quote,origin,created_at FROM people_observations')){const all=statement.all;statement.all=async()=>{const result=await all();if(!removed){removed=true;s.raw.prepare("DELETE FROM people_observations WHERE id='accepted'").run()}return result}}return statement};
 assert.equal((await s.call('/api/people/'+p.id+'/timeline')).code,'pagination_changed','an observation removed while reading is never returned');
});

test('incomplete prerequisites cannot queue or reserve an automatic portrait',async()=>{
 const s=setup();s.env.OPENROUTER_API_KEY='mock-not-called';const p=await s.person();await s.event([p.id]);await s.personal.sourcesChanged(s.env,'a');const claimed=await s.personal.claimJobs(s.env,'a','portrait');assert.deepEqual((await claimed.json()).jobs,[]);assert.equal(s.raw.prepare('SELECT count(*) n FROM journal_jobs').get().n,0);assert.equal(s.raw.prepare('SELECT count(*) n FROM provider_budgets').get().n,0);
});


test('a title-only event and a relationship without narrative use their user notes as explicit sources',async()=>{
 const s=setup(),p=await s.person(),e=await s.event([p.id]);s.raw.prepare('UPDATE life_events SET description=\'\' WHERE id=?').run(e.id);
 const eventText='Je me suis senti soutenu pendant ce départ.',personText='Nous parlons librement pendant nos promenades.';
 await s.call(notePath('event',e.id),'POST',{creation_id:'event_note_only',text:eventText});
 await s.call(notePath('person',p.id),'POST',{creation_id:'person_note_only',text:personText});
 const c=await loadContext(s.env,'a'),source=c.sources.find(x=>x.id==='event:'+e.id);
 assert.ok(source?.explicit);assert.equal(source.fields.description,'');assert.match(source.fields.user_notes_text,/Je me suis senti soutenu/);assert.ok(c.sources.find(x=>x.id==='person:'+p.id)?.fields.user_notes_text.includes(personText));
 const portrait={schema_version:2,summary:'Un appui au moment du départ.',summary_evidence:[source.id],axes:[{label:'Appui',why:'Un soutien rapporté.',claim_type:'declaration',confidence:.5,evidence:[source.id]}],strengths:[],uncertainties:[],psychological_map:[],citations:[{source_id:source.id,revision:source.revision,quote:eventText}]};
 assert.equal(validatePortraitV2(portrait,c.sources,(status,message)=>{throw Error(message)}).citations[0].quote,eventText);
 seedCompanionPrerequisites(s.raw,'a');s.env.OPENROUTER_API_KEY='mock-never-sent';const jobs=(await (await s.personal.claimJobs(s.env,'a','portrait')).json()).jobs;assert.equal(jobs.length,1,'notes alone qualify after prerequisites');
});

test('losing prerequisites during the last portrait reservation blocks fetch and refunds every reservation',async t=>{
 const s=setup();seedCompanionPrerequisites(s.raw,'a');const e=await s.event();await s.call(notePath('event',e.id),'POST',{creation_id:'race_source_note',text:'Une note personnelle pour cet événement.'});s.env.OPENROUTER_API_KEY='mock-never-sent';s.env.DAILY_PROVIDER_TOKEN_CAP='100000';
 const job=(await (await s.personal.claimJobs(s.env,'a','portrait')).json()).jobs[0];assert.ok(job);
 const prepare=s.env.DB.prepare;let removed=false;s.env.DB.prepare=sql=>{const statement=prepare(sql);if(sql.startsWith('INSERT INTO provider_global_budget')){const run=statement.run;statement.run=async()=>{const result=await run();if(!removed){removed=true;s.raw.prepare("DELETE FROM personal_people WHERE id='ready-person-a-1'").run()}return result}}return statement};
 const original=globalThis.fetch;let calls=0;t.after(()=>{globalThis.fetch=original});globalThis.fetch=async()=>{calls++;return Response.json({choices:[{message:{content:'Never sent'}}]})};
 await assert.rejects(s.personal.proxyJobLLM(s.env,job.id,{lease_token:job.lease_token,source_version:job.source_version,memory_version:job.memory_version,system_prompt:'Portrait',user_prompt:'Contexte',max_tokens:128}),error=>error.code==='context_changed');
 assert.equal(removed,true);assert.equal(calls,0);
 for(const table of ['journal_jobs','provider_budgets']){const row=s.raw.prepare('SELECT provider_calls,reserved_tokens FROM '+table).get();assert.equal(row.provider_calls,0);assert.equal(row.reserved_tokens,0)}
 const daily=s.raw.prepare('SELECT calls,reserved_tokens FROM provider_global_budget').get();assert.equal(daily.calls,0);assert.equal(daily.reserved_tokens,0);
});


test('notes keep their target type when an old event and person share the same id',async()=>{
 const s=setup(),p=await s.person();s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES(?,'a','legacy_collision','hash','Même identifiant','other','')").run(p.id);
 await s.call(notePath('person',p.id),'POST',{creation_id:'collision_person',text:'Texte de la relation.'});await s.call(notePath('event',p.id),'POST',{creation_id:'collision_event',text:'Texte du moment.'});
 const c=await loadContext(s.env,'a'),person=c.people.find(x=>x.id===p.id),event=c.events.find(x=>x.id===p.id);assert.equal(person.user_notes_total,1);assert.equal(event.user_notes_total,1);assert.match(person.user_notes_text,/Texte de la relation/);assert.doesNotMatch(person.user_notes_text,/Texte du moment/);assert.match(event.user_notes_text,/Texte du moment/);assert.doesNotMatch(event.user_notes_text,/Texte de la relation/);
});
