import test from 'node:test';
import assert from 'node:assert/strict';
import {CARE_SCOPES} from '../src/care-backend.mjs';
import {careSetup} from './care-helper.mjs';

function setup(t){const s=careSetup();t.after(()=>s.close());return s}
async function prepare(s,link,id='session'){s.session(id);return s.call('/api/care/briefs','POST',{link_id:link.id,session_id:id})}

test('pending codes disclose no account existence, are random, hashed and bound to a preexisting professional',async t=>{
 const s=setup(t),pending=[];
 for(const email of ['pro@example.test','absent@example.test','b@example.test']){const result=await s.call('/api/care/links','POST',{email,scopes:['conversation'],accepted:true});assert.equal(result.httpStatus,201);assert.match(result.code,/^[a-f0-9]{32}$/);pending.push(result)}
 assert.equal(new Set(pending.map(p=>p.code)).size,3);assert.deepEqual(Object.keys(pending[0]),Object.keys(pending[1]));assert.equal(pending[0].message,pending[1].message);
 const state=await s.call('/api/care');assert.ok(state.links.every(link=>link.professional===null));assert.doesNotMatch(JSON.stringify(state),/Professionnel pro|code_hash/);
 assert.ok(s.raw.prepare('SELECT code_hash FROM care_links WHERE id=?').get(pending[0].id).code_hash!==pending[0].code);
 assert.equal((await s.call('/api/care/activate','POST',{code:pending[0].code},'other')).code,'invalid_code');assert.equal((await s.call('/api/care/activate','POST',{code:'0'.repeat(32)},'pro')).code,'invalid_code');
 s.addUser('absent','psychologist');assert.equal((await s.call('/api/care/activate','POST',{code:pending[1].code},'absent')).code,'invalid_code');
 assert.equal((await s.call('/api/care/activate','POST',{code:pending[0].code},'pro')).httpStatus,200);assert.equal((await s.call('/api/care/activate','POST',{code:pending[0].code},'pro')).code,'invalid_code');
 const linked=(await s.call('/api/care')).links.find(l=>l.id===pending[0].id);assert.equal(linked.professional.verified,false);assert.equal(linked.professional.registration_number,'Déclaré 123');assert.ok(s.limits.some(l=>l[0]==='care-activate:pro'));
});

test('roles, consent, scope validation and settings are enforced',async t=>{
 const s=setup(t);assert.equal((await s.call('/api/care','GET',undefined,'pro')).code,'member_required');assert.equal((await s.call('/api/care/patients')).code,'psychologist_required');
 assert.equal((await s.call('/api/care/links','POST',{email:'pro@example.test',scopes:['people']})).code,'consent_required');assert.equal((await s.call('/api/care/links','POST',{email:'pro@example.test',scopes:['memory'],accepted:true})).code,'invalid_scopes');
 assert.equal((await s.call('/api/care/settings','PUT',{has_psychologist:true,frequency_days:14})).care.frequency_days,14);assert.equal((await s.call('/api/care/settings','PUT',{has_psychologist:null,frequency_days:null})).care.has_psychologist,null);assert.equal((await s.call('/api/care/settings','PUT',{has_psychologist:true,frequency_days:366})).httpStatus,400);
});

test('shared dossier scopes are read only, tenant isolated and every consultation is logged',async t=>{
 const s=setup(t),link=await s.share();s.turn('one');s.turn('secret-b','PRIVATE OTHER MEMBER',s.now()-60,'b');
 const conversation=await s.call(`/api/care/dossiers/${link.id}/conversation`,'GET',undefined,'pro');assert.equal(conversation.httpStatus,200);assert.equal(conversation.data.length,1);assert.doesNotMatch(JSON.stringify(conversation),/PRIVATE OTHER MEMBER/);
 for(const user of ['other','b'])assert.equal((await s.call(`/api/care/dossiers/${link.id}/conversation`,'GET',undefined,user)).httpStatus,404);
 assert.equal((await s.call(`/api/care/dossiers/${link.id}/people`,'GET',undefined,'pro')).httpStatus,404);assert.equal((await s.call(`/api/care/dossiers/${link.id}/conversation`,'PUT',{content:'mutation'},'pro')).httpStatus,404);
 assert.equal((await s.call('/api/care/sessions','POST',{session_at:s.now()},'pro')).httpStatus,403);assert.equal((await s.call(`/api/care/links/${link.id}`,'DELETE',{revision:link.revision},'pro')).httpStatus,403);
 const access=await s.call('/api/care/access-log');assert.ok(access.entries.some(e=>e.action==='read'&&e.scope==='conversation'));assert.ok(access.entries.some(e=>e.action==='denied'));assert.equal((await s.call('/api/care/access-log','GET',undefined,'pro')).httpStatus,403);
 assert.equal((await s.call('/api/care/patients','GET',undefined,'pro')).patients[0].name,'Personne a');assert.equal((await s.call('/api/care/patients','GET',undefined,'other')).patients.length,0);
});

test('session CRUD uses ownership and revisions; pagination reaches older sessions',async t=>{
 const s=setup(t),created=await s.call('/api/care/sessions','POST',{session_at:s.now()+600});assert.equal(created.httpStatus,201);
 assert.equal((await s.call(`/api/care/sessions/${created.id}`,'PUT',{session_at:s.now()+1200,revision:1},'b')).httpStatus,404);
 assert.equal((await s.call(`/api/care/sessions/${created.id}`,'PUT',{session_at:s.now()+1200,revision:1})).httpStatus,200);assert.equal((await s.call(`/api/care/sessions/${created.id}`,'DELETE',{revision:1})).httpStatus,409);
 assert.equal((await s.call(`/api/care/sessions/${created.id}`,'DELETE',{revision:2})).httpStatus,200);
 for(let i=0;i<103;i++)s.session('session-'+i,s.now()-i);
 const initial=await s.call('/api/care'),second=await s.call('/api/care/sessions?offset=100');assert.equal(initial.sessions.length,100);assert.equal(initial.sessions_next_offset,100);assert.equal(second.sessions.length,3);assert.equal(second.next_offset,null);assert.equal(new Set([...initial.sessions,...second.sessions].map(v=>v.id)).size,103);
 assert.equal((await s.call('/api/care/sessions?offset=2.9')).sessions[0].id,'session-2');
 const link=await s.share(),dossier=await s.call(`/api/care/dossiers/${link.id}`,'GET',undefined,'pro');assert.equal(dossier.sessions_next_offset,100);assert.equal((await s.call(`/api/care/dossiers/${link.id}/sessions?offset=100`,'GET',undefined,'pro')).sessions.length,3);
});

test('dossier pagination includes all conversations, events and reviews with stable ties',async t=>{
 const s=setup(t),link=await s.share(['conversation','life_events','reviews']);for(let i=0;i<53;i++){s.turn('turn-'+i);s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES(?,'a',?,'h','Événement','other','Description')").run('event-'+i,'event-'+i);s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,'a','point','ready','Bilan',?,?,?)").run('review-'+i,'review-'+i,s.now(),s.now())}
 for(const [scope,size] of [['conversation',50],['life_events',50],['reviews',20]]){let offset=0,ids=[];do{const result=await s.call(`/api/care/dossiers/${link.id}/${scope}?offset=${offset}`,'GET',undefined,'pro');assert.equal(result.httpStatus,200);assert.ok(result.data.length<=size);ids.push(...result.data.map(v=>v.id));offset=result.next_offset}while(offset!==null);assert.equal(new Set(ids).size,53)}
});

test('prepared point batches only consented user sources, validates exact citations and exposes honest coverage',async t=>{
 const s=setup(t),link=await s.share();for(let i=0;i<25;i++)s.turn('turn-'+i,'Phrase exacte '+i+'. '+(i===0?'x'.repeat(1700):''));s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{\"private\":\"PRIVATE MEMORY\"}',?)").run(s.now());
 s.setGenerator(async(_env,options)=>{const data=JSON.parse(options.user);assert.ok(data.sources.length<=20);assert.ok(data.sources.every(source=>source.text.length<=1600));assert.doesNotMatch(options.user,/PRIVATE MEMORY|RÉPONSE ASSISTANT/);return {text:JSON.stringify({items:[{source_id:data.sources[0].id,quote:data.sources[0].text.slice(0,15),theme:'Évolution'},{source_id:data.sources[0].id,quote:'FAUSSE CITATION',theme:'Faits rapportés'},{source_id:data.sources[0].id,quote:'',theme:'Faits rapportés'},{source_id:data.sources[0].id,quote:data.sources[0].text.slice(0,5),theme:'Diagnostic inventé'}]})}});
 const queued=await prepare(s,link);assert.equal(queued.httpStatus,202);assert.equal(s.raw.prepare('SELECT status FROM care_briefs WHERE id=?').get(queued.id).status,'pending');await s.care.resume(s.env);
 const result=await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro');assert.equal(result.brief.status,'ready');assert.equal(result.brief.coverage.processed,25);assert.equal(result.brief.coverage.excerpted,1);assert.match(result.brief.content,/Évolution/);assert.doesNotMatch(result.brief.content,/FAUSSE CITATION|Diagnostic inventé|NaN/);assert.equal(s.calls.length,2);
 assert.equal((await s.call(`/api/care/briefs/${queued.id}`)).brief.content,result.brief.content);assert.equal((await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'other')).httpStatus,404);
});

test('all authorized source types have readable labels and derived provenance; portrait is an undated snapshot',async t=>{
 const s=setup(t),link=await s.share(CARE_SCOPES);s.session('previous',s.now()-3600);s.turn('turn');
 s.raw.prepare("INSERT INTO change_plans(user_id,document_json,updated_at) VALUES('a',?,?)").run(JSON.stringify({goals:[{id:'goal',revision:1,title:'Mon objectif choisi',status:'active',accepted_at:s.now()-60,updated_at:s.now()-60}]}),s.now());
 s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text,updated_at) VALUES('person','a','person','hash','Camille','friend','Relation apaisante',?)").run(new Date((s.now()-60)*1000).toISOString());
 s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description,updated_at) VALUES('event','a','event','hash','Départ','other','Nouveau travail',?)").run(new Date((s.now()-60)*1000).toISOString());
 s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES('review','a','point','ready','Synthèse antérieure','review',?,?)").run(s.now()-60,s.now()-60);
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,updated_at) VALUES('a','{\"critique\":1}','{}','{\"factors\":{\"critique\":\"Mots personnels\"}}','{}','[]',1,?,?)").run(s.now(),s.now());
 s.raw.prepare("INSERT INTO user_portraits(user_id,portrait,analyzed_version,updated_at) VALUES('a','{\"summary\":\"Synthèse du portrait\"}',1,?)").run(new Date(s.now()*1000).toISOString());
 const queued=await prepare(s,link);const sources=JSON.parse(s.calls[0].user).sources;assert.equal(CARE_SCOPES.length,7);assert.equal(sources.length,8);assert.equal(new Set(sources.map(source=>source.id)).size,8);assert.match(sources.find(source=>source.id==='plan:goals:goal').provenance,/explicitement choisi/);assert.match(sources.find(source=>source.id==='cartography').text,/Critiques et exigences constantes/);assert.match(sources.find(source=>source.id==='portrait').provenance,/date de modification inconnue/);assert.match(sources.find(source=>source.id==='review:review').provenance,/Synthèse IA/);
 assert.equal((await s.call(`/api/care/briefs/${queued.id}`)).brief.coverage.processed,8);const map=await s.call(`/api/care/dossiers/${link.id}/cartography`,'GET',undefined,'pro');assert.equal(map.data.summary.causes[0].label,'Critiques et exigences constantes');
});

test('revocation during generation clears in-flight content and immediately denies later reads',async t=>{
 const s=setup(t),link=await s.share();s.turn('turn');s.setGenerator(async()=>{assert.equal((await s.call(`/api/care/links/${link.id}`,'DELETE',{revision:link.revision})).httpStatus,200);return {text:JSON.stringify({items:[{source_id:'conversation:turn',quote:'Je me sens fatigué.',theme:'Faits rapportés'}]})}});
 const queued=await prepare(s,link),row=s.raw.prepare('SELECT * FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,'revoked');assert.equal(row.content,'');assert.equal(row.notes,'[]');assert.equal(row.lease_token,null);assert.equal((await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro')).httpStatus,404);assert.equal((await s.call(`/api/care/dossiers/${link.id}`,'GET',undefined,'pro')).httpStatus,404);assert.equal((await s.call('/api/care/patients','GET',undefined,'pro')).patients.length,0);
});

test('scope reduction invalidates prepared points and cannot publish a previous scope',async t=>{
 const s=setup(t),link=await s.share(['conversation','people']);s.turn('turn');const queued=await prepare(s,link);assert.equal((await s.call(`/api/care/briefs/${queued.id}`)).brief.status,'ready');
 assert.equal((await s.call(`/api/care/links/${link.id}`,'PUT',{revision:link.revision,scopes:['people'],accepted:true})).httpStatus,200);assert.equal((await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro')).httpStatus,404);assert.equal((await s.call(`/api/care/dossiers/${link.id}/conversation`,'GET',undefined,'pro')).httpStatus,404);assert.equal((await s.call(`/api/care/dossiers/${link.id}/people`,'GET',undefined,'pro')).httpStatus,200);
});

test('revocation racing a read is caught for dossier overview, scope, patients and brief',async t=>{
 for(const route of ['overview','conversation','patients','brief']){const s=careSetup();try{const link=await s.share();s.turn('turn');const queued=await prepare(s,link);s.setHook(async({sql,phase})=>{if(phase==='after'&&((route==='overview'&&sql.includes('SELECT id,link_id,session_id,status'))||(route==='conversation'&&sql.includes('SELECT id,content,response'))||(route==='patients'&&sql.includes('INSERT INTO care_access_log'))||(route==='brief'&&sql.includes('INSERT INTO care_access_log')))){s.setHook(null);await s.call(`/api/care/links/${link.id}`,'DELETE',{revision:link.revision})}});const path=route==='patients'?'/api/care/patients':route==='brief'?`/api/care/briefs/${queued.id}`:`/api/care/dossiers/${link.id}`+(route==='conversation'?'/conversation':'');const result=await s.call(path,'GET',undefined,'pro');if(route==='patients')assert.equal(result.patients.length,0);else assert.equal(result.httpStatus,404,route)}finally{s.close()}}
});

test('source edits, deletion and purge invalidate generation, while unshared memory does not',async t=>{
 for(const mutation of ['edit','delete','purge','unshared']){const s=careSetup();try{const link=await s.share();s.turn('turn');s.setGenerator(async()=>{if(mutation==='edit')s.raw.prepare("UPDATE chat_turns SET content='Nouveau texte' WHERE id='turn'").run();if(mutation==='delete')s.raw.prepare("DELETE FROM chat_turns WHERE id='turn'").run();if(mutation==='purge')s.raw.prepare("UPDATE chat_turns SET purged=1 WHERE id='turn'").run();if(mutation==='unshared')s.raw.prepare("UPDATE memories SET text='Mémoire modifiée' WHERE user_id='a'").run();return {text:JSON.stringify({items:[{source_id:'conversation:turn',quote:'Je me sens fatigué.',theme:'Faits rapportés'}]})}});const queued=await prepare(s,link),row=s.raw.prepare('SELECT * FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,mutation==='unshared'?'ready':'failed',mutation);if(mutation!=='unshared'){assert.equal(row.error_code,'sources_changed');assert.equal(row.content,'');assert.equal(row.notes,'[]')}}finally{s.close()}}
});

test('ready points stop exposing removed source text and are regenerated after edits',async t=>{
 const s=setup(t),link=await s.share();s.turn('turn');const queued=await prepare(s,link);s.raw.prepare("UPDATE chat_turns SET content='Texte corrigé' WHERE id='turn'").run();const stale=await s.call(`/api/care/briefs/${queued.id}`);assert.equal(stale.brief.status,'failed');assert.equal(stale.brief.content,'');assert.equal(stale.brief.error_code,'sources_changed');const retry=await s.call('/api/care/briefs','POST',{link_id:link.id,session_id:'session'});assert.equal(retry.id,queued.id);const ready=await s.call(`/api/care/briefs/${queued.id}`);assert.equal(ready.brief.status,'ready');assert.match(ready.brief.content,/Texte corrigé/);assert.equal(ready.brief.coverage.processed,1);
});

test('leases exclude duplicate work, resume after expiry and reject late publication',async t=>{
 const s=setup(t),link=await s.share();s.turn('turn');s.session('session');const queued=await s.care.queueBrief(s.env,link,s.raw.prepare("SELECT * FROM care_sessions WHERE id='session'").get());s.raw.prepare("UPDATE care_briefs SET status='running',lease_token='previous',lease_until=? WHERE id=?").run(s.now()+100,queued.id);await s.care.processBrief(s.env,queued.id);assert.equal(s.calls.length,0);s.setClock(s.now()+101);await s.care.processBrief(s.env,queued.id);assert.equal(s.calls.length,1);assert.equal((await s.call(`/api/care/briefs/${queued.id}`)).brief.status,'ready');
 s.turn('second');s.setGenerator(async()=>{s.setClock(s.now()+151);return {text:'{"items":[]}'}});await s.call('/api/care/briefs','POST',{link_id:link.id,session_id:'session'});assert.equal(s.raw.prepare('SELECT status FROM care_briefs WHERE id=?').get(queued.id).status,'failed');
});

test('transient provider failures preserve earlier batches and stop after three failed attempts',async t=>{
 const s=setup(t),link=await s.share();for(let i=0;i<21;i++)s.turn('turn-'+i);const queued=await prepare(s,link);let failures=0;s.setGenerator(async()=>{failures++;throw Error('private provider details')});await s.care.processBrief(s.env,queued.id);assert.equal(s.raw.prepare('SELECT cursor FROM care_briefs WHERE id=?').get(queued.id).cursor,20);assert.notEqual(s.raw.prepare('SELECT notes FROM care_briefs WHERE id=?').get(queued.id).notes,'[]');
 for(let i=0;i<2;i++){s.setClock(s.now()+61);await s.care.processBrief(s.env,queued.id)}assert.equal(failures,3);const row=s.raw.prepare('SELECT * FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,'failed');assert.equal(row.error_code,'provider_error');assert.doesNotMatch(JSON.stringify(await s.call(`/api/care/briefs/${queued.id}`)),/private provider/);
});

test('manual and automatic preparation share four daily slots and unchanged ready points cost none',async t=>{
 const s=setup(t),link=await s.share();s.turn('turn');const first=await prepare(s,link,'one');const repeated=await s.call('/api/care/briefs','POST',{link_id:link.id,session_id:'one'},'pro');assert.equal(first.id,repeated.id);assert.equal(s.raw.prepare('SELECT count FROM care_daily_budget').get().count,1);
 for(const id of ['two','three','four'])assert.equal((await prepare(s,link,id)).httpStatus,202);s.session('five');assert.equal((await s.call('/api/care/briefs','POST',{link_id:link.id,session_id:'five'},'pro')).code,'brief_limit');await s.care.resume(s.env);assert.equal(s.raw.prepare('SELECT count FROM care_daily_budget').get().count,4);assert.equal(s.raw.prepare("SELECT count(*) AS n FROM care_briefs WHERE session_id='five'").get().n,0);s.setClock(s.now()+86400);assert.equal((await s.call('/api/care/briefs','POST',{link_id:link.id,session_id:'five'})).httpStatus,202);
});

test('automatic preparation rotates past unchanged ready points and exhausted member budgets',async t=>{
 const s=setup(t);for(let i=0;i<12;i++){const member='member'+i;s.addUser(member);await s.share(['conversation'],member);s.session('session'+i,s.now()+3600+i,member)}
 assert.equal((await s.care.resume(s.env)).scheduled,10);assert.equal((await s.care.resume(s.env)).scheduled,2);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM care_briefs').get().n,12);
 s.setClock(s.now()+901);await s.care.resume(s.env);await s.care.resume(s.env);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM care_brief_checks WHERE checked_at=?').get(s.now()).n,12);
});

test('concurrent queue requests debit once and never reset an already running lease',async t=>{
 const s=setup(t),link=await s.share();s.session('session');const session=s.raw.prepare("SELECT * FROM care_sessions WHERE id='session'").get(),queued=await Promise.all(Array.from({length:8},()=>s.care.queueBrief(s.env,link,session)));assert.equal(new Set(queued.map(job=>job.id)).size,1);assert.equal(s.raw.prepare('SELECT count FROM care_daily_budget').get().count,1);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM care_briefs').get().n,1);
 s.raw.prepare("UPDATE care_briefs SET status='running',lease_token='worker',lease_until=?").run(s.now()+100);await s.care.queueBrief(s.env,link,session);assert.equal(s.raw.prepare('SELECT lease_token FROM care_briefs').get().lease_token,'worker');assert.equal(s.raw.prepare('SELECT count FROM care_daily_budget').get().count,1);
});

test('source revision is guarded atomically at publication and final brief read',async t=>{
 const s=setup(t),link=await s.share();s.turn('turn');s.setHook(({sql,phase})=>{if(phase==='before'&&sql.includes('SET status=?,content=?,notes=?,coverage=?')){s.setHook(null);s.raw.prepare("UPDATE chat_turns SET content='Corrigé juste avant publication' WHERE id='turn'").run()}});const queued=await prepare(s,link);assert.equal(s.raw.prepare('SELECT status FROM care_briefs WHERE id=?').get(queued.id).status,'failed');
 await s.call('/api/care/briefs','POST',{link_id:link.id,session_id:'session'});s.setHook(({sql,phase})=>{if(phase==='before'&&sql.startsWith('SELECT b.* FROM care_briefs')){s.setHook(null);s.raw.prepare("UPDATE chat_turns SET purged=1 WHERE id='turn'").run()}});assert.equal((await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro')).httpStatus,404);
});

test('an expired worker cannot overwrite a resumed worker that has already published',async t=>{
 const s=setup(t),link=await s.share();s.turn('turn','Ancienne citation. Nouvelle citation.');s.session('session');const queued=await s.care.queueBrief(s.env,link,s.raw.prepare("SELECT * FROM care_sessions WHERE id='session'").get());s.setGenerator(async()=>{s.setClock(s.now()+151);s.setGenerator(async()=>({text:JSON.stringify({items:[{source_id:'conversation:turn',quote:'Nouvelle citation.',theme:'Faits rapportés'}]})}));await s.care.processBrief(s.env,queued.id);return {text:JSON.stringify({items:[{source_id:'conversation:turn',quote:'Ancienne citation.',theme:'Faits rapportés'}]})}});await s.care.processBrief(s.env,queued.id);const result=await s.call(`/api/care/briefs/${queued.id}`);assert.equal(result.brief.status,'ready');assert.match(result.brief.content,/Nouvelle citation/);assert.doesNotMatch(result.brief.content,/Ancienne citation/);assert.equal(s.calls.length,2);
});

test('link and brief histories are paginated, owner isolated, and resumable in the shared dossier',async t=>{
 const s=setup(t),link=await s.share();for(let i=0;i<53;i++){s.session('history'+i,s.now()-i);s.raw.prepare("INSERT INTO care_briefs(id,user_id,link_id,session_id,link_revision,session_revision,created_at,updated_at) VALUES(?,'a',?,?,?,1,?,?)").run('brief'+i,link.id,'history'+i,link.revision,s.now(),s.now());s.raw.prepare("INSERT INTO care_links(id,user_id,target_email,code_hash,scopes,status,created_at,expires_at) VALUES(?,'a','pro@example.test',?,'[\"people\"]','revoked',?,?)").run('oldlink'+i,'oldhash'+i,s.now(),s.now())}
 const state=await s.call('/api/care');assert.equal(state.links_next_offset,50);assert.equal(state.briefs_next_offset,50);assert.equal((await s.call('/api/care/links?offset=50')).links.length,4);assert.equal((await s.call('/api/care/briefs?offset=50')).briefs.length,3);assert.equal((await s.call('/api/care/briefs','GET',undefined,'b')).briefs.length,0);assert.equal((await s.call(`/api/care/dossiers/${link.id}/briefs?offset=50`,'GET',undefined,'pro')).briefs.length,3);
});

test('expired codes cannot activate, are marked expired and do not consume live-share slots',async t=>{
 const s=setup(t),pending=[];for(let i=0;i<10;i++)pending.push(await s.call('/api/care/links','POST',{email:'pro@example.test',scopes:['people'],accepted:true}));s.setClock(s.now()+7*86400);assert.equal((await s.call('/api/care/activate','POST',{code:pending[0].code},'pro')).code,'invalid_code');assert.equal((await s.call('/api/care/links','POST',{email:'pro@example.test',scopes:['people'],accepted:true})).httpStatus,201);await s.care.resume(s.env);assert.equal(s.raw.prepare("SELECT count(*) AS n FROM care_links WHERE status='expired'").get().n,10);
});

test('exports include care records without secret code hashes and account deletion cascades cleanly',async t=>{
 const s=setup(t),link=await s.share();s.turn('turn');await prepare(s,link);await s.care.resume(s.env);await s.call('/api/care/settings','PUT',{has_psychologist:true,frequency_days:14});const exported=await s.care.exportData(s.env,'a');assert.equal(exported.care_links.length,1);assert.equal(exported.care_sessions.length,1);assert.equal(exported.care_briefs.length,1);assert.ok(exported.care_access_log.length>0);assert.doesNotMatch(JSON.stringify(exported),/code_hash/);
 s.raw.prepare("DELETE FROM users WHERE id='a'").run();for(const table of ['care_links','care_sessions','care_briefs','care_access_log','care_daily_budget','care_settings','care_source_revisions','care_brief_checks'])assert.equal(s.raw.prepare(`SELECT count(*) AS n FROM ${table}`).get().n,table==='care_source_revisions'?3:0,table);
 assert.equal(s.raw.prepare('PRAGMA foreign_key_check').all().length,0);
});
