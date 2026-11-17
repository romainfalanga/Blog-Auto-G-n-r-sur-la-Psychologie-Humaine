import test from 'node:test';
import assert from 'node:assert/strict';
import {createChangeBackend} from '../src/change-backend.mjs';
import {readChangeState} from '../src/change-plan.mjs';
import {careSetup} from './care-helper.mjs';

function setup(t){
 const s=careSetup();t.after(()=>s.close());let serial=0;
 const change=createChangeBackend({error(status,message,code){throw Object.assign(Error(message),{status,code})},json:(value,status=200)=>Response.json(value,{status}),body:request=>request.json(),now:s.now,random:()=>`review_${++serial}`,limit:async()=>{}});
 async function planCall(path,method='GET',fields={},owner='a'){
  const state=await readChangeState(s.env,owner),payload={revision:state.revision,epoch:state.epoch,...(method==='POST'?{request_key:`request_review_${++serial}`}:{ }),...fields},url=new URL(path,'https://example.test'),request=new Request(url,{method,...(method==='GET'?{}:{body:JSON.stringify(payload),headers:{'Content-Type':'application/json'}})}),user=s.raw.prepare('SELECT id,role FROM users WHERE id=?').get(owner);
  try{const response=await change.routes(request,s.env,user,url.pathname,url);return {httpStatus:response.status,...await response.json()}}catch(error){return {httpStatus:error.status||500,code:error.code,message:error.message}}
 }
 async function choices(owner='a',trials=1){
  const goal=ok(await planCall('/api/plan/goals','POST',{title:'OBJECTIF_CHOISI',why:'Ma raison privée volontairement partagée'},owner));
  const step=ok(await planCall(`/api/plan/goals/${goal.entity_id}/steps`,'POST',{text:'ACTION_CHOISIE',measure:'Mes mots après'},owner));
  const trialIds=[];for(let i=0;i<trials;i++){const trial=ok(await planCall(`/api/plan/steps/${step.entity_id}/results`,'POST',{result_text:'RETOUR_CHOISI_'+i,attempt:'partial',outcome:'neutral'},owner));trialIds.push(trial.entity_id)}
  return {goal:goal.entity_id,step:step.entity_id,trials:trialIds};
 }
 async function prepare(link,id='session'){s.readyGeneration(link.user_id);s.session(id);return ok(await s.call('/api/care/briefs','POST',{link_id:link.id,session_id:id}),202)}
 return {...s,change,planCall,choices,prepare};
}
function ok(result,status=201){assert.equal(result.httpStatus,status,JSON.stringify(result));return result}

test('point provenance records every plan input sent across batches even when none is quoted, and no private preparation',async t=>{
 const s=setup(t),ids=await s.choices('a',23);
 ok(await s.planCall('/api/plan/instructions','POST',{text:'CONSIGNE_JAMAIS_PARTAGÉE'}));ok(await s.planCall('/api/plan/preparations','POST',{content:'POINT_PRIVÉ_JAMAIS_PARTAGÉ'}));
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{\"goals\":[\"MÉMOIRE_JAMAIS_PARTAGÉE\"]}',?)").run(s.now());
 const link=await s.share(['plan']);s.setGenerator(async(_env,options)=>{assert.doesNotMatch(options.user,/JAMAIS_PARTAGÉ/);return {text:'{"items":[]}'}});
 const queued=await s.prepare(link);assert.equal(s.raw.prepare('SELECT status FROM care_briefs WHERE id=?').get(queued.id).status,'pending');assert.equal(s.calls.length,1);
 const sentFirst=JSON.parse(s.calls[0].user).sources.map(row=>row.id.replace(/^plan:[^:]+:/,'')),firstRefs=s.raw.prepare("SELECT entity_id FROM change_artifact_refs WHERE artifact_kind='brief' AND artifact_id=?").all(queued.id).map(row=>row.entity_id);assert.deepEqual(new Set(firstRefs),new Set(sentFirst));assert.equal(firstRefs.length,20);
 await s.care.processBrief(s.env,queued.id);assert.equal(s.calls.length,2);
 const allSent=s.calls.flatMap(call=>JSON.parse(call.user).sources.map(row=>row.id.replace(/^plan:[^:]+:/,''))),refs=s.raw.prepare("SELECT entity_id FROM change_artifact_refs WHERE artifact_kind='brief' AND artifact_id=?").all(queued.id).map(row=>row.entity_id);
 assert.deepEqual(new Set(refs),new Set(allSent));assert.deepEqual(new Set(refs),new Set([ids.goal,ids.step,...ids.trials]));
 const point=ok(await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro'),200);assert.equal(point.brief.status,'ready');assert.equal(point.brief.coverage.processed,25);assert.match(point.brief.content,/Aucune citation retenue/);
});

test('a plan object never sent in the first page does not falsely trigger a continuity reset when removed',async t=>{
 const s=setup(t),ids=await s.choices('a',23),link=await s.share(['plan']);s.setGenerator(async()=>({text:'{"items":[]}'}));
 const queued=await s.prepare(link),sent=new Set(JSON.parse(s.calls[0].user).sources.map(row=>row.id.replace(/^plan:[^:]+:/,''))),unseen=ids.trials.find(id=>!sent.has(id));assert.ok(unseen);
 const removed=ok(await s.planCall(`/api/plan/trials/${unseen}`,'DELETE'),200);assert.equal(removed.context_generation,0);assert.equal(removed.continuity_reset,undefined);
 await s.care.processBrief(s.env,queued.id);const row=s.raw.prepare('SELECT status,content,notes FROM care_briefs WHERE id=?').get(queued.id);assert.ok(['failed','revoked'].includes(row.status));assert.equal(row.content,'');assert.equal(row.notes,'[]');assert.equal(s.calls.length,1);
});

test('a correction through the plan API purges a ready shared point even when the provider did not quote that input',async t=>{
 const s=setup(t),ids=await s.choices(),link=await s.share(['plan']);s.setGenerator(async()=>({text:JSON.stringify({items:[{source_id:'plan:goals:'+ids.goal,quote:'OBJECTIF_CHOISI',theme:'Faits rapportés'}]})}));
 const queued=await s.prepare(link);assert.equal(s.raw.prepare('SELECT status FROM care_briefs WHERE id=?').get(queued.id).status,'ready');
 const corrected=ok(await s.planCall(`/api/plan/trials/${ids.trials[0]}`,'PATCH',{result_text:'RETOUR_CORRIGÉ'}),200);assert.equal(corrected.continuity_reset,true);assert.equal(corrected.context_generation,1);
 const row=s.raw.prepare('SELECT status,content,notes FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,'revoked');assert.equal(row.content,'');assert.equal(row.notes,'[]');assert.equal((await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro')).httpStatus,404);assert.doesNotMatch(JSON.stringify(await s.care.exportData(s.env,'a')),/OBJECTIF_CHOISI|RETOUR_CHOISI/);
});

test('the first provider call already counts as transmission when the member removes its source during generation',async t=>{
 const s=setup(t),ids=await s.choices(),link=await s.share(['plan']);let removed;
 s.setGenerator(async(_env,options)=>{assert.ok(JSON.parse(options.user).sources.some(source=>source.id==='plan:goals:'+ids.goal));removed=ok(await s.planCall(`/api/plan/goals/${ids.goal}`,'DELETE'),200);return {text:JSON.stringify({items:[{source_id:'plan:goals:'+ids.goal,quote:'OBJECTIF_CHOISI',theme:'Faits rapportés'}]})}});
 const queued=await s.prepare(link),row=s.raw.prepare('SELECT status,content,notes FROM care_briefs WHERE id=?').get(queued.id);assert.notEqual(row.status,'ready');assert.equal(row.content,'');assert.equal(row.notes,'[]');assert.equal(removed.context_generation,1,'the plan was already sent to the provider before this deletion');assert.equal(removed.continuity_reset,true);
});

test('editing a private instruction does not invalidate an unchanged shared plan point',async t=>{
 const s=setup(t);await s.choices();const instruction=ok(await s.planCall('/api/plan/instructions','POST',{text:'CONSIGNE_PRIVÉE_AVANT'})),link=await s.share(['plan']),queued=await s.prepare(link),before=await s.planCall('/api/plan/sharing-preview');
 ok(await s.planCall(`/api/plan/instructions/${instruction.entity_id}`,'PATCH',{text:'CONSIGNE_PRIVÉE_APRÈS'}),200);const after=await s.planCall('/api/plan/sharing-preview');
 for(const kind of ['goals','steps','trials'])assert.deepEqual(after[kind],before[kind]);const point=ok(await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro'),200);assert.equal(point.brief.status,'ready','shared inputs did not change');assert.doesNotMatch(point.brief.content,/CONSIGNE_PRIVÉE/);
});

test('a personal preparation with no professional is private even after a later plan opt-in',async t=>{
 const s=setup(t);await s.choices();assert.equal(s.raw.prepare("SELECT count(*) AS n FROM care_links WHERE user_id='a'").get().n,0);
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,revision,updated_at) VALUES('a','{\"goals\":[\"NOTE_PERSONNELLE_PRIVÉE\"]}',1,?)").run(s.now());
 const preview=ok(await s.planCall('/api/plan/preparation'),200);assert.equal(preview.memory_notes[0].origin,'memory_unverified');const point=ok(await s.planCall('/api/plan/preparations','POST',{content:preview.content+' NOTE_PERSONNELLE_PRIVÉE',dependencies:preview.dependencies,include_memory:true,memory_revision:1}));assert.equal(point.preparations[0].private,true);
 const legacy=await s.share(['conversation']);assert.equal((await s.call(`/api/care/dossiers/${legacy.id}/plan`,'GET',undefined,'pro')).httpStatus,404);
 const shared=await s.share(['plan']);const dossier=ok(await s.call(`/api/care/dossiers/${shared.id}/plan`,'GET',undefined,'pro'),200);assert.doesNotMatch(JSON.stringify(dossier),/NOTE_PERSONNELLE_PRIVÉE|preparations|memory_unverified/);const queued=await s.prepare(shared);assert.ok(queued.id);assert.ok(s.calls.every(call=>!call.user.includes('NOTE_PERSONNELLE_PRIVÉE')));
 assert.match(JSON.stringify(await s.change.exportData(s.env,'a')),/NOTE_PERSONNELLE_PRIVÉE/,'the owner export retains their own private document');assert.doesNotMatch(JSON.stringify(await s.care.exportData(s.env,'a')),/NOTE_PERSONNELLE_PRIVÉE/);
});

test('owner account deletion during shared-plan generation cannot publish or preserve plan and care metadata',async t=>{
 const s=setup(t);await s.choices();await s.choices('b');const link=await s.share(['plan']);
 s.setGenerator(async()=>{s.raw.prepare("DELETE FROM users WHERE id='a'").run();return {text:'{"items":[]}'}});await s.prepare(link);
 for(const table of ['change_plans','change_requests','change_artifact_refs','care_links','care_sessions','care_briefs','care_source_revisions','care_access_log','care_daily_budget'])assert.equal(s.raw.prepare(`SELECT count(*) AS n FROM ${table} WHERE user_id='a'`).get().n,0,table);
 assert.equal(s.raw.prepare('PRAGMA foreign_key_check').all().length,0);assert.equal((await readChangeState(s.env,'b')).document.goals.length,1);assert.equal((await s.call(`/api/care/dossiers/${link.id}`,'GET',undefined,'pro')).httpStatus,404);
});

test('owner care export includes all retained link, point and access histories beyond the display limits',async t=>{
 const s=setup(t),link=await s.share(['plan']);s.session('session');
 s.raw.exec('BEGIN');
 try{
  const addLink=s.raw.prepare("INSERT INTO care_links(id,user_id,target_email,code_hash,scopes,status,created_at,expires_at) VALUES(?,'a','pro@example.test',?,'[\"plan\"]','revoked',?,?)"),addBrief=s.raw.prepare("INSERT INTO care_briefs(id,user_id,link_id,session_id,link_revision,session_revision,status,created_at,updated_at) VALUES(?,'a',?,'session',?,1,'revoked',?,?)"),addAccess=s.raw.prepare("INSERT INTO care_access_log(id,user_id,actor_id,link_id,action,scope,created_at) VALUES(?,'a','pro',?,'read','plan',?)");
  for(let i=0;i<1001;i++){addLink.run('old_link_'+i,'old_hash_'+i,s.now()-i,s.now()-i);addBrief.run('old_brief_'+i,link.id,i+1,s.now()-i,s.now()-i)}
  for(let i=0;i<10001;i++)addAccess.run('old_access_'+i,link.id,s.now()-i);
  s.raw.exec('COMMIT');
 }catch(error){s.raw.exec('ROLLBACK');throw error}
 const exported=await s.care.exportData(s.env,'a');
 const tables=['care_links','care_briefs','care_access_log'];assert.deepEqual(Object.fromEntries(tables.map(table=>[table,exported[table].length])),Object.fromEntries(tables.map(table=>[table,s.raw.prepare(`SELECT count(*) AS n FROM ${table} WHERE user_id='a'`).get().n])),'the export must not silently omit retained records');
 assert.doesNotMatch(JSON.stringify(exported),/code_hash|old_hash/);
});

test('a deletion during the export metadata reads cannot return point text checked earlier in the same export',async t=>{
 const s=setup(t),ids=await s.choices(),link=await s.share(['plan']);
 s.setGenerator(async()=>({text:JSON.stringify({items:[{source_id:'plan:goals:'+ids.goal,quote:'OBJECTIF_CHOISI',theme:'Faits rapportés'}]})}));await s.prepare(link);
 let deleted=false;s.setHook(async({sql,phase})=>{if(phase==='before'&&sql==='SELECT * FROM care_settings WHERE user_id=?'){s.setHook(null);deleted=true;ok(await s.planCall(`/api/plan/goals/${ids.goal}`,'DELETE'),200)}});
 try{const exported=await s.care.exportData(s.env,'a');assert.doesNotMatch(JSON.stringify(exported.care_briefs),/OBJECTIF_CHOISI/)}catch(error){if(error.name==='AssertionError')throw error;assert.equal(error.code,'export_sources_changed');assert.equal(error.status,409)}
 assert.equal(deleted,true);
});
