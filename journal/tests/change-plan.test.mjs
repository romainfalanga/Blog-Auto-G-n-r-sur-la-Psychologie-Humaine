import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createChangeBackend} from '../src/change-backend.mjs';
import {PLAN_LIMITS,PLAN_COLLECTIONS,emptyPlan,readChangeState,changeWrite,changeGuard,changeArtifactStatements,serializeSharedPlan,dueChangeTrial,parisDay,clearChatChangeState,clearChangeMemoryState,suspendChangeFollowups} from '../src/change-plan.mjs';

// Execute the complete production migration chain and real SQL transactions.
// This adapter deliberately does not emulate UPDATE/CAS with JavaScript objects.
class SQLiteD1 {
 constructor(){this.raw=new DatabaseSync(':memory:');this.raw.exec('PRAGMA foreign_keys=ON');this.queue=Promise.resolve();this.beforeBatch=null}
 prepare(sql){const db=this;let args=[];return {bind(...values){args=values;return this},async first(){return db.raw.prepare(sql).get(...args)||null},async all(){return {results:db.raw.prepare(sql).all(...args)}},async run(){return this.execute()},execute(){return {meta:{changes:db.raw.prepare(sql).run(...args).changes}}}}}
 async batch(statements){if(this.beforeBatch)await this.beforeBatch();const next=this.queue.then(()=>{this.raw.exec('BEGIN');try{const results=statements.map(statement=>statement.execute());this.raw.exec('COMMIT');return results}catch(error){this.raw.exec('ROLLBACK');throw error}});this.queue=next.catch(()=>{});return next}
}
function setup(t){
 const DB=new SQLiteD1(),raw=DB.raw;
 for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(name=>name.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 for(const [id,role] of [['a','member'],['b','member'],['pro','psychologist']])raw.prepare("INSERT INTO users(id,email,password_hash,password_salt,role) VALUES(?,?,'hash','salt',?)").run(id,id+'@example.test',role);
 let clock=Date.parse('2026-11-17T12:00:00Z')/1000,serial=0;const limits=[],env={DB},random=()=>String(++serial).padStart(10,'0');
 const backend=createChangeBackend({error(status,message,code){throw Object.assign(Error(message),{status,code})},json:(data,status=200)=>Response.json(data,{status}),body:request=>request.json(),now:()=>clock,random,limit:async(...args)=>limits.push(args.slice(1))});
 const state=(id='a')=>readChangeState(env,id);
 async function call(path,method='GET',data,userId='a'){
  const url=new URL(path,'https://example.test'),request=new Request(url,{method,...(data===undefined?{}:{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}})}),user=raw.prepare('SELECT id,role FROM users WHERE id=?').get(userId);
  try{const response=await backend.routes(request,env,user,url.pathname,url);return {httpStatus:response.status,...await response.json()}}catch(error){return {httpStatus:error.status||500,code:error.code,message:error.message}}
 }
 async function mutate(path,method='POST',fields={},userId='a'){
  const current=await state(userId);return call(path,method,{revision:current.revision,epoch:current.epoch,...(method==='POST'?{request_key:'request_'+random()}:{}),...fields},userId);
 }
 function seed(document,userId='a',controls={}){
  raw.prepare("INSERT INTO change_plans(user_id,revision,epoch,document_json,updated_at) VALUES(?,1,0,?,?) ON CONFLICT(user_id) DO UPDATE SET document_json=excluded.document_json,revision=change_plans.revision+1").run(userId,JSON.stringify(document),clock);
  for(const [key,value] of Object.entries(controls)){assert.ok(['context_generation','reminders_paused','proposals_enabled','exploration_reconsent_required','past_safety_signal'].includes(key));raw.prepare(`UPDATE change_plans SET ${key}=? WHERE user_id=?`).run(value,userId)}
 }
 function turn(id,{userId='a',running=false,text='Message original conservé',response='COPIE_SUPPRIMABLE',generation=0}={}){
  raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,lease_token,lease_until,context_generation,created_at) VALUES(?,?,?,'hash',?,?,'2026-11-16',1,1,?,?,?, ?,?,?)").run(id,userId,'key-'+id,running?'running':'completed',running?'reserved':'consumed',text,response,running?'lease':null,running?clock+60:0,generation,new Date(clock*1000).toISOString());
 }
 function artifact(turnId,entityId,{userId='a',kind='goals',revision=1,generation=0}={}){raw.prepare("INSERT INTO change_artifact_refs(user_id,artifact_kind,artifact_id,entity_kind,entity_id,entity_revision,context_generation) VALUES(?,'turn',?,?,?,?,?)").run(userId,turnId,kind,entityId,revision,generation)}
 t.after(()=>raw.close());
 return {DB,raw,env,backend,limits,state,call,mutate,seed,turn,artifact,now:()=>clock,setClock:value=>{clock=value}};
}
const ok=(result,status=201)=>{assert.equal(result.httpStatus,status,JSON.stringify(result));return result};
const goal=(id='goal_a',extra={})=>({id,revision:1,title:'Retrouver de petites promenades',why:'Prendre un moment dehors',priority:2,status:'active',origin:'member',accepted_at:1,created_at:1,updated_at:1,...extra});
const step=(id='step_a',goalId='goal_a',extra={})=>({id,revision:1,goal_id:goalId,text:'Marcher cinq minutes',measure:'Ce que je remarque',strategy_id:null,status:'active',origin:'member',accepted_at:1,created_at:1,updated_at:1,...extra});
const trial=(id='trial_a',stepId='step_a',extra={})=>({id,revision:1,step_id:stepId,step_snapshot:{text:'Marcher cinq minutes',measure:'Ce que je remarque',strategy_id:null},status:'planned',planned_on:null,occurred_on:null,occurred_at:null,timing_note:'',attempt:null,result_text:'',outcome:null,adverse_effect_text:'',followup_opt_in:false,review_mode:'none',review_on:null,reminder_count:0,last_reminder_at:null,last_reminder_turn_id:null,origin:'member',accepted_at:1,created_at:1,updated_at:1,...extra});
const proposal=(id='proposal_a',extra={})=>({id,revision:1,kind:'goal',payload:{title:'Une proposition encore libre'},status:'proposed',source_turn_id:'origin',source_quote:'CITATION_PRIVÉE',created_at:1,...extra});
const plan=(extra={})=>({...emptyPlan(),goals:[goal()],steps:[step()],trials:[trial()],...extra});

test('manual creation is distinct from a proposal and server owns identity, revisions and defaults',async t=>{
 const s=setup(t);s.seed({...emptyPlan(),proposals:[proposal()]});
 const created=ok(await s.mutate('/api/plan/goals','POST',{title:'Une proposition encore libre',id:'attacker',origin:'ambre',accepted_at:0,revision:1,user_id:'b',reminder_count:99}));
 assert.equal(created.schema_version,1);assert.equal((await s.call('/api/plan')).schema_version,1);assert.equal(JSON.parse(s.raw.prepare("SELECT document_json FROM change_plans WHERE user_id='a'").get().document_json).schema_version,1);
 assert.equal(created.proposals[0].status,'proposed');assert.equal(created.steps.length,0);assert.equal(created.trials.length,0);
 const chosen=created.goals[0];assert.notEqual(chosen.id,'attacker');assert.equal(chosen.origin,'member');assert.equal(chosen.accepted_at,s.now());assert.equal(chosen.revision,1);
 const addedStep=ok(await s.mutate(`/api/plan/goals/${chosen.id}/steps`,'POST',{text:'Ma propre action'}));
 const addedTrial=ok(await s.mutate(`/api/plan/steps/${addedStep.entity_id}/trials`,'POST',{reminder_count:0,last_reminder_at:123}));
 assert.equal(addedTrial.trials[0].followup_opt_in,false);assert.equal(addedTrial.trials[0].review_mode,'none');assert.equal(addedTrial.trials[0].occurred_on,null);assert.equal(addedTrial.trials[0].occurred_at,null);
 assert.equal((await s.state('b')).document.goals.length,0);assert.ok(s.limits.some(row=>row[0]==='change-api:a'));
});

test('adoption saves edited choices once, strips resolved payload and does not reapply it after editing',async t=>{
 const s=setup(t);s.seed({...emptyPlan(),proposals:[proposal('proposal_goal',{payload:{title:'Titre proposé',step_text:'Action proposée'}})]});
 const data={revision:1,epoch:0,entity_revision:1,request_key:'adoption_key',edited:{title:'Mon titre choisi',step_text:'Mon action choisie'},create_trial:true};
 const accepted=ok(await s.call('/api/plan/proposals/proposal_goal/accept','POST',data));
 assert.equal(accepted.goals[0].title,'Mon titre choisi');assert.equal(accepted.steps[0].text,'Mon action choisie');assert.equal(accepted.trials[0].status,'planned');assert.equal(accepted.trials[0].followup_opt_in,false);
 assert.equal(accepted.proposals[0].status,'accepted');assert.equal(accepted.proposals[0].payload,null);assert.equal(accepted.proposals[0].source_quote,null);assert.equal(accepted.proposals[0].resolved_id,accepted.entity_id);
 ok(await s.mutate(`/api/plan/goals/${accepted.entity_id}`,'PATCH',{title:'Titre corrigé après adoption'}),200);
 const replay=ok(await s.call('/api/plan/proposals/proposal_goal/accept','POST',data),200);assert.equal(replay.replayed,true);assert.equal(replay.goals.length,1);assert.equal(replay.goals[0].title,'Titre corrigé après adoption');assert.ok(replay.revision>accepted.revision);
 assert.equal((await s.mutate('/api/plan/proposals/proposal_goal/accept','POST',{entity_revision:2})).code,'proposal_resolved');
 ok(await s.mutate('/api/plan/proposals/proposal_goal','DELETE'),200);assert.equal((await s.state()).document.goals.length,1,'deleting a resolved card must retain its independently adopted goal');
});

test('step, trial and result proposals require explicit adoption and dismissed or expired cards cannot be adopted',async t=>{
 const s=setup(t);s.seed(plan({proposals:[proposal('proposal_step',{kind:'step',target_id:'goal_a',payload:{text:'Action choisie ensuite'}}),proposal('proposal_trial',{kind:'trial',target_id:'step_a',payload:{}}),proposal('proposal_result',{kind:'result',target_id:'trial_a',payload:{attempt:'partial',result_text:'Mon retour',outcome:'neutral'}}),proposal('proposal_dismiss'),proposal('proposal_expired',{status:'expired',payload:null})]}));
 assert.equal((await s.state()).document.trials[0].status,'planned');
 ok(await s.mutate('/api/plan/proposals/proposal_step/accept'));ok(await s.mutate('/api/plan/proposals/proposal_trial/accept'));
 const result=ok(await s.mutate('/api/plan/proposals/proposal_result/accept'));assert.equal(result.trials.find(row=>row.id==='trial_a').result_text,'Mon retour');assert.equal(result.trials.length,2);
 const dismissed=ok(await s.mutate('/api/plan/proposals/proposal_dismiss/dismiss'));assert.equal(dismissed.proposals.find(row=>row.id==='proposal_dismiss').payload,null);
 for(const id of ['proposal_dismiss','proposal_expired'])assert.equal((await s.mutate(`/api/plan/proposals/${id}/accept`)).code,'proposal_resolved');
});

test('request keys are global per owner, return current objects and never resurrect deleted objects or an older epoch',async t=>{
 const s=setup(t),request={revision:0,epoch:0,request_key:'creation_key',title:'Ancien texte'};
 const created=ok(await s.call('/api/plan/goals','POST',request));
 ok(await s.mutate(`/api/plan/goals/${created.entity_id}`,'PATCH',{title:'Texte actuel'}),200);
 const replay=ok(await s.call('/api/plan/goals','POST',request),200);assert.equal(replay.goals[0].title,'Texte actuel');assert.equal(replay.replayed,true);
 assert.equal((await s.call('/api/plan/goals','POST',{...request,title:'Autre demande'})).code,'idempotency_conflict');
 assert.equal((await s.mutate('/api/plan/instructions','POST',{request_key:'creation_key',text:'Autre route'})).code,'idempotency_conflict');
 ok(await s.mutate(`/api/plan/goals/${created.entity_id}`,'DELETE'),200);
 assert.equal((await s.call('/api/plan/goals','POST',request)).httpStatus,410);
 const receipt=s.raw.prepare("SELECT * FROM change_requests WHERE user_id='a'").get();assert.equal(receipt.deleted,1);assert.doesNotMatch(JSON.stringify(receipt),/Ancien texte|Texte actuel/);
 const erased=ok(await s.mutate('/api/plan','DELETE'),200);assert.equal(erased.epoch,1);assert.equal(erased.proposals_enabled,false);assert.equal(erased.reminders_paused,true);
 assert.equal((await s.call('/api/plan/goals','POST',request)).code,'stale_epoch');
 assert.equal((await s.mutate('/api/plan/goals','POST',{title:'Nouvelle intention',request_key:'creation_key'})).code,'stale_epoch');
 ok(await s.mutate('/api/plan/goals','POST',{title:'Nouvelle intention'}));ok(await s.mutate('/api/plan','DELETE'),200);assert.equal((await s.state()).epoch,2);
});

test('two concurrent writers at one revision have one CAS winner and one conflict, without a receipt for the loser',async t=>{
 const s=setup(t);let arrivals=0,release;const both=new Promise(resolve=>{release=resolve});
 s.DB.beforeBatch=async()=>{if(++arrivals===2)release();await both};
 const writes=await Promise.all(['first','second'].map(value=>s.call('/api/plan/goals','POST',{revision:0,epoch:0,request_key:'concurrent_'+value,title:value})));
 s.DB.beforeBatch=null;assert.deepEqual(writes.map(row=>row.httpStatus).sort(),[201,409]);assert.equal(writes.find(row=>row.httpStatus===409).code,'revision_conflict');
 const state=await s.state();assert.equal(state.revision,1);assert.equal(state.document.goals.length,1);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_requests').get().n,1);
});

test('simultaneous retries of one request make one object and that same key remains independent between owners',async t=>{
 const s=setup(t),request={revision:0,epoch:0,request_key:'same_click_key',title:'Un seul choix'};let arrivals=0,release;const both=new Promise(resolve=>{release=resolve});s.DB.beforeBatch=async()=>{if(++arrivals===2)release();await both};
 const responses=await Promise.all([s.call('/api/plan/goals','POST',request),s.call('/api/plan/goals','POST',request)]);s.DB.beforeBatch=null;assert.equal(responses.filter(row=>row.httpStatus===201).length,1);assert.ok(responses.every(row=>[200,201,409].includes(row.httpStatus)));assert.equal((await s.state()).document.goals.length,1);
 const retry=ok(await s.call('/api/plan/goals','POST',request),200);assert.equal(retry.replayed,true);assert.equal(retry.goals.length,1);ok(await s.call('/api/plan/goals','POST',request,'b'));assert.equal((await s.state('b')).document.goals.length,1);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_requests').get().n,2);
});

test('a later failing statement rolls back the document, receipt and care revision atomically',async t=>{
 const s=setup(t);s.raw.exec("CREATE TRIGGER reject_receipt BEFORE INSERT ON change_requests BEGIN SELECT RAISE(ABORT,'simulated storage failure'); END");
 const result=await s.mutate('/api/plan/goals','POST',{title:'Ne doit pas rester'});assert.equal(result.httpStatus,500);
 assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_plans').get().n,0);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_requests').get().n,0);assert.equal(s.raw.prepare("SELECT count(*) AS n FROM care_source_revisions WHERE scope='plan'").get().n,0);
});

test('private routes deny professional access and isolate direct IDs and nested references between two members',async t=>{
 const s=setup(t);s.seed(plan({instructions:[{id:'instruction_a',revision:1,text:'Consigne privée',status:'current'}],preparations:[{id:'preparation_a',revision:1,content:'Point privé',private:true}],proposals:[proposal()]}));
 for(const path of ['/api/plan','/api/plan/catalogue','/api/plan/sharing-preview','/api/plan/preparation','/api/plan/goals/goal_a'])assert.equal((await s.call(path,'GET',undefined,'pro')).httpStatus,403,path);
 for(const method of ['POST','PATCH','DELETE','PUT'])assert.equal((await s.call('/api/plan/goals',method,{revision:1,epoch:0},'pro')).httpStatus,403);
 for(const [kind,id] of [['goals','goal_a'],['steps','step_a'],['trials','trial_a'],['proposals','proposal_a'],['instructions','instruction_a'],['preparations','preparation_a']])for(const method of ['GET','PATCH','DELETE']){
  const result=method==='GET'?await s.call(`/api/plan/${kind}/${id}`,'GET',undefined,'b'):await s.mutate(`/api/plan/${kind}/${id}`,method,{},'b');assert.equal(result.httpStatus,404,kind+' '+method);
 }
 for(const [path,data] of [['/api/plan/goals/goal_a/steps',{text:'Vol de relation'}],['/api/plan/steps/step_a/trials',{}],['/api/plan/instructions',{goal_id:'goal_a',text:'Consigne'}],['/api/plan/preparations',{content:'Point',dependencies:[{kind:'goals',id:'goal_a',revision:1}]}],['/api/plan/proposals/proposal_a/accept',{}]])assert.equal((await s.mutate(path,'POST',data,'b')).httpStatus,404,path);
 assert.equal((await s.mutate('/api/plan/followups','PUT',{enabled:false,target_id:'trial_a'},'b')).httpStatus,404);
 s.seed({...emptyPlan(),proposals:[proposal('foreign_target',{kind:'step',target_id:'goal_a',payload:{text:'Ne doit pas être créé'}})]},'b');assert.equal((await s.mutate('/api/plan/proposals/foreign_target/accept','POST',{},'b')).httpStatus,404);
 assert.equal((await s.state()).document.goals[0].title,goal().title);
});

test('deleting a goal cascades dependent objects, instructions, preparations and proposal pointers only within its owner',async t=>{
 const s=setup(t),document=plan({goals:[goal(),goal('kept_goal',{status:'paused'})],steps:[step(),step('kept_step','kept_goal')],instructions:[{id:'instruction_a',revision:1,text:'Liée',goal_id:'goal_a'},{id:'kept_instruction',revision:1,text:'Indépendante'}],preparations:[{id:'preparation_a',revision:1,content:'Copie',dependencies:[{kind:'trials',id:'trial_a',revision:1}]}],proposals:[proposal('linked_proposal',{kind:'result',target_id:'trial_a'}),proposal('resolved_proposal',{status:'accepted',payload:null,resolved_id:'goal_a'}),proposal('kept_proposal')]});
 s.seed(document);s.seed(plan(),'b');ok(await s.mutate('/api/plan/goals/goal_a','DELETE'),200);
 const current=(await s.state()).document;assert.deepEqual(current.goals.map(row=>row.id),['kept_goal']);assert.deepEqual(current.steps.map(row=>row.id),['kept_step']);assert.equal(current.trials.length,0);assert.deepEqual(current.instructions.map(row=>row.id),['kept_instruction']);assert.equal(current.preparations.length,0);assert.deepEqual(current.proposals.map(row=>row.id),['kept_proposal']);assert.equal((await s.state('b')).document.trials.length,1);
});

test('all collections are paginated at fifty while export retains every stored object',async t=>{
 const s=setup(t),document=plan({steps:Array.from({length:60},(_,i)=>step('step_'+i)),trials:Array.from({length:200},(_,i)=>trial('trial_'+i,'step_'+(i%60))),proposals:Array.from({length:55},(_,i)=>proposal('proposal_'+i,{status:'accepted',payload:null,source_quote:null,resolved_id:'goal_a'}))});s.seed(document);
 const first=ok(await s.call('/api/plan'),200);assert.equal(first.steps.length,50);assert.equal(first.trials.length,50);assert.equal(first.proposals.length,50);assert.equal(first.counts.trials,200);
 for(const kind of ['steps','trials','proposals']){const ids=[];let offset=0;do{const page=ok(await s.call(`/api/plan?collection=${kind}&offset=${offset}`),200);assert.ok(page[kind].length<=50);ids.push(...page[kind].map(row=>row.id));offset=page.next_offsets[kind]}while(offset!==null);assert.deepEqual(ids,document[kind].map(row=>row.id))}
 const exported=(await s.backend.exportData(s.env,'a')).change_plan;for(const kind of PLAN_COLLECTIONS)assert.deepEqual(exported[kind],document[kind]);assert.equal(exported.limits.bytes,262144);assert.equal(exported.schema_version,1);
 assert.equal((await s.call('/api/plan?collection=unknown')).httpStatus,400);assert.equal((await s.call('/api/plan?collection=trials&offset=50.9')).trials[0].id,'trial_50');
});

test('three active goals and every storage collection quota reject new writes without trimming older objects',async t=>{
 const s=setup(t);s.seed(plan({goals:[goal(),goal('g2'),goal('g3')]}));assert.equal((await s.mutate('/api/plan/goals','POST',{title:'Quatrième'})).code,'active_goal_limit');
 const cases=[
  ['goals',20,i=>goal('goal_'+i,{status:'paused'}),'/api/plan/goals',{title:'En plus',status:'paused'}],
  ['steps',60,i=>step('step_'+i),'/api/plan/goals/goal_a/steps',{text:'En plus'}],
  ['trials',200,i=>trial('trial_'+i),'/api/plan/steps/step_a/trials',{}],
  ['preparations',20,i=>({id:'preparation_'+i,revision:1,content:'Point',dependencies:[],private:true}),'/api/plan/preparations',{content:'En plus'}],
  ['instructions',10,i=>({id:'instruction_'+i,revision:1,text:'Consigne',status:'current'}),'/api/plan/instructions',{text:'En plus'}]
 ];
 for(const [kind,count,factory,path,body] of cases){const doc=plan({[kind]:Array.from({length:count},(_,i)=>factory(i))});s.seed(doc);const before=await s.state(),result=await s.mutate(path,'POST',body);assert.equal(result.code,'plan_limit',kind);assert.deepEqual((await s.state()).document,before.document);assert.equal((await s.state()).revision,before.revision)}
 const doc=plan({proposals:Array.from({length:41},(_,i)=>proposal('proposal_'+i))});s.seed(doc);assert.equal((await s.mutate('/api/plan/goals/goal_a','PATCH',{priority:1})).code,'plan_limit');
 assert.deepEqual(PLAN_LIMITS,{goals:20,active_goals:3,steps:60,trials:200,proposals:40,preparations:20,instructions:10,bytes:262144});
});

test('accepted text is never silently truncated and invalid calendar dates are refused',async t=>{
 const s=setup(t);s.seed(plan());
 for(const [path,method,field,limit] of [['/api/plan/goals/goal_a','PATCH','title',240],['/api/plan/goals/goal_a','PATCH','why',800],['/api/plan/steps/step_a','PATCH','text',800],['/api/plan/steps/step_a/results','POST','result_text',1200],['/api/plan/instructions','POST','text',1200],['/api/plan/preparations','POST','content',6000]]){
  const exact='é'.repeat(limit),fields={[field]:exact};const good=ok(await s.mutate(path,method,fields),method==='POST'?201:200);assert.equal(good[good.entity_kind].find(row=>row.id===good.entity_id)[field],exact);
  const before=await s.state();assert.equal((await s.mutate(path,method,{[field]:exact+'é'})).httpStatus,400);assert.equal((await s.state()).revision,before.revision);
 }
 for(const day of ['2026-02-29','2026-04-31','2026-13-01','17/11/2026','2026-11-17T00:00:00Z'])assert.equal((await s.mutate('/api/plan/trials/trial_a','PATCH',{planned_on:day})).code,'invalid_date');
 ok(await s.mutate('/api/plan/trials/trial_a','PATCH',{planned_on:'2028-02-29'}),200);
});

test('aggregate cap is 256 KiB in UTF-8 bytes and a rejected write preserves the complete near-limit export',async t=>{
 const s=setup(t),document=plan({preparations:Array.from({length:19},(_,i)=>({id:'preparation_'+i,revision:1,content:'é'.repeat(6000),dependencies:[],private:true})),instructions:Array.from({length:10},(_,i)=>({id:'instruction_'+i,revision:1,text:'é'.repeat(1200),status:'current'}))});
 const bytes=value=>Buffer.byteLength(JSON.stringify(value),'utf8');assert.ok(bytes(document)<262144);assert.ok(bytes(document)+12000>262144);assert.ok(JSON.stringify(document).length+6000<262144);s.seed(document);
 const before=await s.state(),rejected=await s.mutate('/api/plan/preparations','POST',{content:'é'.repeat(6000)});assert.equal(rejected.code,'plan_limit');assert.equal((await s.state()).revision,before.revision);assert.deepEqual((await s.backend.exportData(s.env,'a')).change_plan.preparations,document.preparations);
});

test('undated reports, not tried and worse preserve the member wording without closing a goal or inventing an occurrence date',async t=>{
 const s=setup(t);s.seed(plan());
 const notTried=ok(await s.mutate('/api/plan/steps/step_a/results','POST',{attempt:'not_tried',result_text:"Je n'ai pas eu le temps",timing_note:'cette semaine'}));
 const first=notTried.trials.find(row=>row.id===notTried.entity_id);assert.equal(first.attempt,'not_tried');assert.equal(first.outcome,null);assert.equal(first.occurred_at,null);assert.equal(first.occurred_on,null);assert.equal(first.timing_note,'cette semaine');assert.equal(notTried.goals[0].status,'active');
 const worse=ok(await s.mutate('/api/plan/steps/step_a/results','POST',{attempt:'done',outcome:'worse',result_text:'Je me suis senti plus mal',adverse_effect_text:'Tension pénible'}));
 assert.equal(worse.trials.find(row=>row.id===worse.entity_id).outcome,'worse');assert.equal(worse.goals[0].status,'active');assert.equal(worse.steps[0].status,'active');
 assert.equal((await s.mutate(`/api/plan/trials/${worse.entity_id}`,'PATCH',{status:'planned'})).httpStatus,400);
});

test('editing a step and repeating it keep earlier accepted action and results immutable',async t=>{
 const s=setup(t);s.seed(plan({trials:[trial('trial_a','step_a',{status:'reported',attempt:'partial',result_text:'Ancien retour conservé'})]}));
 ok(await s.mutate('/api/plan/steps/step_a','PATCH',{text:'Marcher deux minutes',measure:'Mon choix suivant'}),200);
 const repeated=ok(await s.mutate('/api/plan/steps/step_a/trials'));assert.equal(repeated.trials.length,2);assert.equal(repeated.trials[0].step_snapshot.text,'Marcher cinq minutes');assert.equal(repeated.trials[0].result_text,'Ancien retour conservé');assert.equal(repeated.trials[1].step_snapshot.text,'Marcher deux minutes');assert.equal(repeated.trials[1].step_snapshot.measure,'Mon choix suivant');
});

test('a stale screen can stop followups, retries a CAS collision and never overwrites unrelated changes',async t=>{
 const s=setup(t);s.seed(plan({trials:[trial('trial_a','step_a',{followup_opt_in:true,review_mode:'next_conversation',reminder_count:1,last_reminder_at:s.now()-604800})]}));
 let raced=false;s.DB.beforeBatch=async()=>{if(raced)return;raced=true;const row=s.raw.prepare("SELECT document_json FROM change_plans WHERE user_id='a'").get(),doc=JSON.parse(row.document_json);doc.goals[0].title='Autre onglet sauvegardé';s.raw.prepare("UPDATE change_plans SET revision=revision+1,document_json=? WHERE user_id='a'").run(JSON.stringify(doc))};
 const stopped=ok(await s.call('/api/plan/followups','PUT',{revision:0,epoch:-1,enabled:false,target_id:'all'}),200);s.DB.beforeBatch=null;assert.equal(stopped.reminders_paused,true);assert.equal(stopped.goals[0].title,'Autre onglet sauvegardé');assert.equal(stopped.trials[0].reminder_count,1);
 assert.equal((await s.call('/api/plan/followups','PUT',{revision:1,epoch:0,enabled:true,confirmed:true,target_id:'all'})).code,'revision_conflict');
 assert.equal((await s.mutate('/api/plan/followups','PUT',{enabled:true,target_id:'all'})).httpStatus,400);
});

test('editing dates or reenabling globally never resets counters or restores a separately revoked trial permission',async t=>{
 const s=setup(t);s.seed(plan({trials:[trial('trial_a','step_a',{followup_opt_in:true,review_mode:'on_date',review_on:'2026-11-17',reminder_count:2,last_reminder_at:s.now()-604800,last_reminder_turn_id:'old'})]}));
 ok(await s.mutate('/api/plan/trials/trial_a','PATCH',{review_on:'2026-11-24',planned_on:'2026-11-20',reminder_count:0,last_reminder_at:null}),200);
 ok(await s.call('/api/plan/followups','PUT',{enabled:false,target_id:'trial_a'}),200);ok(await s.call('/api/plan/followups','PUT',{enabled:false,target_id:'all'}),200);
 const enabled=ok(await s.mutate('/api/plan/followups','PUT',{enabled:true,confirmed:true,target_id:'all'}),200);assert.equal(enabled.reminders_paused,false);assert.equal(enabled.trials[0].followup_opt_in,false);assert.equal(enabled.trials[0].reminder_count,2);assert.equal(enabled.trials[0].last_reminder_turn_id,'old');
});

test('due trial uses the Paris day, seven full UTC days, a later message and no consecutive turns',async t=>{
 const s=setup(t),midnight=Date.parse('2026-03-29T22:00:00Z')/1000;
 s.seed(plan({trials:[trial('trial_a','step_a',{followup_opt_in:true,review_mode:'on_date',review_on:'2026-03-30',created_at:midnight-1000})]}));
 const state=await s.state(),due=(at,options={})=>dueChangeTrial(state,'Mes promenades me préoccupent',at,{turnId:'current',previousTurnId:'previous',...options});
 assert.equal(parisDay(midnight-1),'2026-03-29');assert.equal(parisDay(midnight),'2026-03-30');assert.equal(due(midnight-1),null);assert.equal(due(midnight)?.id,'trial_a');
 const current=state.document.trials[0];Object.assign(current,{reminder_count:1,last_reminder_at:midnight-604799,last_reminder_turn_id:'older'});assert.equal(due(midnight),null);assert.equal(due(midnight+1)?.id,'trial_a');
 current.last_reminder_turn_id='previous';assert.equal(due(midnight+1),null);current.last_reminder_turn_id='older';current.reminder_count=2;assert.equal(due(midnight+1),null);
 Object.assign(current,{reminder_count:0,last_reminder_at:null,last_reminder_turn_id:null,review_mode:'next_conversation',source_turn_id:'adopting',created_at:midnight});assert.equal(due(midnight),null);assert.equal(due(midnight+1,{turnId:'adopting'}),null);assert.equal(due(midnight+1)?.id,'trial_a');
 assert.equal(dueChangeTrial(state,'Quel livre de cuisine choisir ?',midnight+1,{turnId:'current'}),null);state.reminders_paused=1;assert.equal(due(midnight+1),null);
});

test('when dates tie the one due invitation respects goal priority before age',async t=>{
 const s=setup(t);s.seed(plan({goals:[goal('low',{priority:3}),goal('high',{priority:1})],steps:[step('low_step','low'),step('high_step','high')],trials:[trial('older','low_step',{created_at:1,followup_opt_in:true,review_mode:'on_date',review_on:'2026-11-17'}),trial('priority','high_step',{created_at:2,followup_opt_in:true,review_mode:'on_date',review_on:'2026-11-17'})]}));
 assert.equal(dueChangeTrial(await s.state(),'Mes promenades',s.now(),{turnId:'current'}).id,'priority');
});

test('deleting transmitted content during generation cleans controlled copies, preserves only refusal flags and prevents a stale CAS write',async t=>{
 const s=setup(t);s.seed(plan({proposals:[proposal('dependent_proposal',{kind:'result',target_id:'trial_a'})],preparations:[{id:'memory_point',revision:1,content:'COPIE_SUPPRIMABLE',memory_revision:1,dependencies:[],private:true}]}), 'a',{reminders_paused:1});
 s.turn('origin');s.turn('inflight',{running:true});s.turn('other_owner',{userId:'b',response:'AUTRE_COMPTE'});s.artifact('origin','goal_a');s.artifact('inflight','goal_a');
 for(const id of ['origin','inflight']){s.raw.prepare("INSERT INTO chat_turn_traces(turn_id,user_id,trace,created_at) VALUES(?,'a',?,?)").run(id,JSON.stringify({raw:'COPIE_SUPPRIMABLE'}),s.now());s.raw.prepare("INSERT INTO chat_engine_jobs(turn_id,user_id,stage,source_hash,context,draft,created_at,updated_at) VALUES(?,'a',?,'hash',?,?,?,?)").run(id,id==='inflight'?'writing':'done',JSON.stringify({text:'COPIE_SUPPRIMABLE'}),JSON.stringify({text:'COPIE_SUPPRIMABLE'}),s.now(),s.now())}
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a',?,?)").run(JSON.stringify({goals:['COPIE_SUPPRIMABLE'],closed:['SUJET_FERMÉ_PRIVÉ'],safety:['SIGNAL_PRIVÉ'],safety_seen:true}),s.now());
 s.raw.prepare("INSERT INTO chat_memory_jobs(user_id,lease_token,lease_until,updated_at) VALUES('a','lease',?,?)").run(s.now()+60,s.now());s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at) VALUES('review','a','point','ready','COPIE_SUPPRIMABLE','review',?)").run(s.now());
 const stale=await s.state();const removed=ok(await s.mutate('/api/plan/goals/goal_a','DELETE'),200);assert.equal(removed.continuity_reset,true);assert.match(removed.retained,/messages originaux/);
 const current=await s.state();assert.equal(current.context_generation,stale.context_generation+1);assert.equal(current.reminders_paused,1);assert.equal(current.exploration_reconsent_required,1);assert.equal(current.past_safety_signal,1);assert.equal(current.document.proposals.length,0);assert.equal(current.document.preparations.length,0);
 const retained=s.raw.prepare("SELECT * FROM chat_turns WHERE id='origin'").get();assert.equal(retained.response,'');assert.equal(retained.content,'Message original conservé');const stopped=s.raw.prepare("SELECT * FROM chat_turns WHERE id='inflight'").get();assert.equal(stopped.status,'failed');assert.equal(stopped.credit_status,'released');assert.equal(stopped.lease_token,null);assert.equal(stopped.context_generation,0);
 for(const table of ['chat_memory','chat_memory_jobs','chat_turn_traces','chat_engine_jobs','change_artifact_refs'])assert.equal(s.raw.prepare(`SELECT count(*) AS n FROM ${table} WHERE user_id='a'`).get().n,0,table);assert.equal(s.raw.prepare("SELECT content FROM ambre_reviews WHERE id='review'").get().content,'');assert.equal(s.raw.prepare("SELECT response FROM chat_turns WHERE id='other_owner'").get().response,'AUTRE_COMPTE');
 const oldPublish=await changeWrite(s.env,stale,stale.document,{now:s.now(),mutation:'obsolete_publish'}).run();assert.equal(oldPublish.meta.changes,0);assert.doesNotMatch(JSON.stringify(await s.backend.exportData(s.env,'a')),/COPIE_SUPPRIMABLE|CITATION_PRIVÉE|SUJET_FERMÉ_PRIVÉ|SIGNAL_PRIVÉ/);
});

test('administrative edits and edits never transmitted keep continuity; changing transmitted words resets it once',async t=>{
 const s=setup(t);s.seed(plan());s.turn('used');s.artifact('used','goal_a');s.artifact('used','trial_a',{kind:'trials'});s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{\"goals\":[\"Mémoire à préserver\"]}',?)").run(s.now());
 for(const [path,data] of [['/api/plan/goals/goal_a',{priority:1}],['/api/plan/goals/goal_a',{status:'paused'}],['/api/plan/trials/trial_a',{planned_on:'2026-11-24'}],['/api/plan/trials/trial_a',{followup_opt_in:true,review_mode:'on_date',review_on:'2026-11-24'}],['/api/plan/steps/step_a',{text:'Action jamais transmise'}]]){const result=ok(await s.mutate(path,'PATCH',data),200);assert.equal(result.context_generation,0,path);assert.equal(result.continuity_reset,undefined);assert.ok(s.raw.prepare("SELECT 1 FROM chat_memory WHERE user_id='a'").get())}
 const changed=ok(await s.mutate('/api/plan/goals/goal_a','PATCH',{title:'Nouvelle formulation'}),200);assert.equal(changed.context_generation,1);assert.equal(changed.continuity_reset,true);assert.equal(s.raw.prepare("SELECT 1 FROM chat_memory WHERE user_id='a'").get(),undefined);
 const again=ok(await s.mutate('/api/plan/goals/goal_a','PATCH',{title:'Autre correction non transmise'}),200);assert.equal(again.context_generation,1);
});

test('CAS loser cannot purge the winners outputs, traces or memory',async t=>{
 const s=setup(t);s.seed(plan());s.turn('used');s.artifact('used','goal_a');s.raw.prepare("INSERT INTO chat_turn_traces(turn_id,user_id,trace,created_at) VALUES('used','a','{\"text\":\"À conserver\"}',?)").run(s.now());s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{}',?)").run(s.now());
 s.DB.beforeBatch=async()=>{s.DB.beforeBatch=null;s.raw.prepare("UPDATE change_plans SET revision=revision+1,proposals_enabled=0 WHERE user_id='a'").run()};
 const result=await s.mutate('/api/plan/goals/goal_a','DELETE');assert.equal(result.code,'revision_conflict');assert.equal((await s.state()).document.goals.length,1);assert.equal((await s.state()).context_generation,0);assert.equal(s.raw.prepare("SELECT response FROM chat_turns WHERE id='used'").get().response,'COPIE_SUPPRIMABLE');assert.ok(s.raw.prepare("SELECT 1 FROM chat_turn_traces WHERE turn_id='used'").get());assert.ok(s.raw.prepare("SELECT 1 FROM chat_memory WHERE user_id='a'").get());
});

test('personal preparation needs no linked professional, keeps memory separate and saves a private editable document only on request',async t=>{
 const s=setup(t);s.seed(plan({trials:[trial('trial_a','step_a',{status:'reported',attempt:'partial',result_text:'RETOUR_SANS_DATE',adverse_effect_text:'Effet pénible'})]}));
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,revision,updated_at) VALUES('a',?,7,?)").run(JSON.stringify({goals:['NOTE_OBJECTIF'],tried:['NOTE_ESSAI'],session:['NOTE_SÉANCE'],safety:['SÉCURITÉ_EXCLUE'],closed:['REFUS_EXCLU'],hypotheses:['HYPOTHÈSE_EXCLUE']}),s.now());
 const preview=ok(await s.call('/api/plan/preparation'),200);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM care_links').get().n,0);assert.match(preview.content,/date non précisée/);assert.match(preview.content,/RETOUR_SANS_DATE|Effet pénible/);assert.doesNotMatch(preview.content,/NOTE_/);assert.equal(preview.memory_notes.length,3);assert.ok(preview.memory_notes.every(note=>note.origin==='memory_unverified'));assert.doesNotMatch(JSON.stringify(preview),/SÉCURITÉ_EXCLUE|REFUS_EXCLU|HYPOTHÈSE_EXCLUE/);assert.equal((await s.state()).document.preparations.length,0);
 const saved=ok(await s.mutate('/api/plan/preparations','POST',{content:preview.content,dependencies:preview.dependencies}));assert.equal(saved.preparations[0].private,true);assert.equal(saved.preparations[0].memory_revision,null);
 ok(await s.mutate(`/api/plan/preparations/${saved.entity_id}`,'PATCH',{content:'Ma propre reformulation'}),200);assert.doesNotMatch(JSON.stringify(await s.call('/api/plan/sharing-preview')),/Ma propre reformulation|NOTE_/);
 ok(await s.mutate('/api/plan/goals/goal_a','PATCH',{title:'Source corrigée'}),200);assert.equal((await s.state()).document.preparations.length,0);
 assert.equal((await s.mutate('/api/plan/preparations','POST',{content:'Point périmé',dependencies:preview.dependencies})).code,'revision_conflict');
 assert.equal((await s.mutate('/api/plan/preparations','POST',{content:'Note mémoire',include_memory:true,memory_revision:6})).code,'revision_conflict');
 const included=ok(await s.mutate('/api/plan/preparations','POST',{content:'Ajout volontaire NOTE_OBJECTIF',include_memory:true,memory_revision:7}));assert.equal(included.preparations[0].memory_revision,7);assert.equal(included.preparations[0].memory_origin,'memory_unverified');
});

test('preparation accepts only owned sessions, distinguishes undated facts and reports omitted sources honestly',async t=>{
 const s=setup(t),previous=Date.parse('2026-11-10T12:00:00Z')/1000;
 for(const [id,at,user] of [['previous',previous,'a'],['next',s.now()+86400,'a'],['foreign',s.now()+86400,'b']])s.raw.prepare('INSERT INTO care_sessions(id,user_id,session_at,created_by,created_at) VALUES(?,?,?,?,?)').run(id,user,at,user,s.now());
 s.seed(plan({trials:[trial('before','step_a',{status:'reported',result_text:'ANCIEN_RETOUR',occurred_on:'2026-11-09'}),trial('undated','step_a',{status:'reported',result_text:'SANS_DATE'}),...Array.from({length:8},(_,i)=>trial('after_'+i,'step_a',{status:'reported',occurred_on:'2026-11-12',result_text:'x'.repeat(1200)}))]}));
 assert.equal((await s.call('/api/plan/preparation?session_id=foreign')).httpStatus,404);const preview=ok(await s.call('/api/plan/preparation?session_id=next'),200);assert.equal(preview.coverage.since,previous);assert.doesNotMatch(preview.content,/ANCIEN_RETOUR/);assert.match(preview.content,/date non précisée/);assert.ok(preview.coverage.omitted>0);assert.ok(preview.content.length<=6000);
});

test('shared serializer includes adopted related objects only and recursively excludes private or malformed nested data',async()=>{
 const secret={text:'visible',source_quote:'SECRET_CITATION',instructions:[{text:'SECRET_INSTRUCTION'}],proposals:[{payload:'SECRET_PROPOSAL'}]},document=plan({goals:[goal('goal_a',{source_quote:'SECRET_CITATION',instructions:['SECRET_INSTRUCTION']}),goal('unadopted',{accepted_at:null})],steps:[step('step_a','goal_a',{measure:secret,private_care_instruction:'SECRET_INSTRUCTION'}),step('orphan','unadopted')],trials:[trial('trial_a','step_a',{step_snapshot:{text:'Action publique',measure:secret,strategy_id:null,source_quote:'SECRET_CITATION',instructions:'SECRET_INSTRUCTION'},source_quote:'SECRET_CITATION'}),trial('orphan_trial','orphan')],proposals:[proposal()],instructions:[{text:'SECRET_INSTRUCTION'}],preparations:[{content:'SECRET_POINT'}]});
 const shared=serializeSharedPlan({revision:7,updated_at:1,document});assert.equal(shared.goals.length,1);assert.equal(shared.steps.length,1);assert.equal(shared.trials.length,1);assert.equal(shared.trials[0].step_snapshot.text,'Action publique');assert.doesNotMatch(JSON.stringify(shared),/SECRET_|CITATION_PRIVÉE|source_quote|instructions|proposals|preparations/);
});

test('chat-only purge retains adopted choices, erases citations and old proposals and advances epoch and generation',async t=>{
 const s=setup(t);s.seed(plan({goals:[goal('goal_a',{source_turn_id:'source',source_quote:'CITATION_PRIVÉE'})],proposals:[proposal()],preparations:[{id:'memory_preparation',content:'Texte mémoire',memory_revision:1},{id:'personal_preparation',content:'Texte manuel',memory_revision:null}]}),'a',{reminders_paused:1,past_safety_signal:1,exploration_reconsent_required:1});
 const purge=await clearChatChangeState(s.env,'a',{includePlan:false,now:s.now()});const result=await s.DB.batch(purge.statements);assert.equal(result[0].meta.changes,1);const current=await s.state();assert.equal(current.document.goals.length,1);assert.equal(current.document.trials.length,1);assert.equal(current.document.proposals.length,0);assert.equal(current.document.preparations.length,1);assert.equal(current.document.goals[0].source_removed,true);assert.equal(current.epoch,1);assert.equal(current.context_generation,1);assert.equal(current.reminders_paused,1);assert.doesNotMatch(JSON.stringify(current.document),/CITATION_PRIVÉE|source_turn_id/);
 const all=await clearChatChangeState(s.env,'a',{includePlan:true,now:s.now()});await s.DB.batch(all.statements);assert.deepEqual((await s.state()).document,emptyPlan());assert.equal((await s.state()).document.schema_version,1);assert.equal((await s.state()).epoch,2);
});

test('missing migration fails closed and deleting an account cascades all plan metadata',async t=>{
 const s=setup(t);const created=ok(await s.mutate('/api/plan/goals','POST',{title:'À effacer'}));s.turn('source');s.artifact('source',created.entity_id);
 s.raw.prepare("DELETE FROM users WHERE id='a'").run();for(const table of ['change_plans','change_requests','change_artifact_refs'])assert.equal(s.raw.prepare(`SELECT count(*) AS n FROM ${table} WHERE user_id='a'`).get().n,0);
 s.raw.exec('DROP TABLE change_plans');assert.equal((await s.state('b')).available,false);assert.equal((await s.call('/api/plan','GET',undefined,'b')).code,'plan_unavailable');assert.equal((await s.backend.exportData(s.env,'b')).change_plan,null);
});

test('changing a report administrative status cannot silently discard its accepted observation',async t=>{
 const s=setup(t);s.seed(plan());const report=ok(await s.mutate('/api/plan/steps/step_a/results','POST',{attempt:'done',outcome:'worse',result_text:'OBSERVATION_ACCEPTÉE',adverse_effect_text:'EFFET_ACCEPTÉ'}));
 const before=await s.state(),result=await s.mutate(`/api/plan/trials/${report.entity_id}`,'PATCH',{status:'cancelled'}),after=await s.state();
 if(result.httpStatus===200){const retained=after.document.trials.find(row=>row.id===report.entity_id);assert.equal(retained.result_text,'OBSERVATION_ACCEPTÉE');assert.equal(retained.adverse_effect_text,'EFFET_ACCEPTÉ')}else{assert.equal(result.httpStatus,400);assert.deepEqual(after.document,before.document)}
});

test('adopting an identical result is not a content correction and must not reset conversation continuity',async t=>{
 const s=setup(t);s.seed(plan());const report=ok(await s.mutate('/api/plan/steps/step_a/results','POST',{attempt:'done',outcome:'neutral',result_text:'Mes mots déjà acceptés'}));
 const document=(await s.state()).document;document.proposals.push(proposal('same_result',{kind:'result',target_id:report.entity_id,payload:{attempt:'done',outcome:'neutral',result_text:'Mes mots déjà acceptés'}}));s.seed(document);
 s.turn('copy');s.artifact('copy',report.entity_id,{kind:'trials'});s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{}',?)").run(s.now());
 const result=ok(await s.mutate('/api/plan/proposals/same_result/accept'));assert.equal(result.proposals[0].status,'accepted');assert.equal(result.context_generation,0);assert.ok(s.raw.prepare("SELECT 1 FROM chat_memory WHERE user_id='a'").get());
});

test('explicit memory purge compares its own revision and preserves adopted objects and independently chosen followup stops',async t=>{
 const s=setup(t);s.seed(plan({preparations:[{id:'memory_point',content:'Copie mémoire',memory_revision:3},{id:'manual_point',content:'Ma rédaction',memory_revision:null}]}),'a',{reminders_paused:1,exploration_reconsent_required:1,past_safety_signal:1});s.seed(plan(),'b');
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,revision,updated_at) VALUES('a','{\"closed\":[\"Privé\"]}',3,?)").run(s.now());s.raw.prepare("INSERT INTO chat_memory(user_id,summary,revision,updated_at) VALUES('b','{}',1,?)").run(s.now());
 await assert.rejects(()=>clearChangeMemoryState(s.env,'a',{now:s.now()}),{code:'invalid_revision'});
 const stale=await clearChangeMemoryState(s.env,'a',{now:s.now(),expectedMemoryRevision:2});assert.equal((await s.DB.batch(stale.statements))[0].meta.changes,0);assert.equal((await s.state()).context_generation,0);assert.equal(s.raw.prepare("SELECT revision FROM chat_memory WHERE user_id='a'").get().revision,3);
 const clear=await clearChangeMemoryState(s.env,'a',{now:s.now(),expectedMemoryRevision:3});assert.equal((await s.DB.batch(clear.statements))[0].meta.changes,1);
 const current=await s.state();assert.equal(current.epoch,0);assert.equal(current.context_generation,1);assert.equal(current.reminders_paused,1);assert.equal(current.past_safety_signal,0);assert.equal(current.exploration_reconsent_required,0);assert.equal(current.document.goals.length,1);assert.equal(current.document.trials.length,1);assert.deepEqual(current.document.preparations.map(row=>row.id),['manual_point']);assert.equal(s.raw.prepare("SELECT 1 FROM chat_memory WHERE user_id='a'").get(),undefined);assert.ok(s.raw.prepare("SELECT 1 FROM chat_memory WHERE user_id='b'").get());assert.equal((await s.state('b')).context_generation,0);
});

test('a stale composed chat purge cannot erase receipts, dependencies or the callers guarded data',async t=>{
 const s=setup(t);const created=ok(await s.mutate('/api/plan/goals','POST',{title:'À conserver'}));s.turn('original');s.artifact('original',created.entity_id);s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{}',?)").run(s.now());
 const purge=await clearChatChangeState(s.env,'a',{includePlan:true,now:s.now()});ok(await s.mutate('/api/plan/preferences','PUT',{proposals_enabled:false}),200);
 const appended=s.env.DB.prepare(`DELETE FROM chat_memory WHERE user_id=? AND ${purge.guard.sql}`).bind('a',...purge.guard.args);const results=await s.DB.batch([...purge.statements,appended]);assert.ok(results.every(row=>row.meta.changes===0));assert.equal((await s.state()).document.goals.length,1);assert.equal((await s.state()).epoch,0);assert.equal(s.raw.prepare("SELECT deleted FROM change_requests WHERE user_id='a'").get().deleted,0);assert.equal(s.raw.prepare("SELECT count(*) AS n FROM change_artifact_refs WHERE user_id='a'").get().n,1);assert.ok(s.raw.prepare("SELECT 1 FROM chat_memory WHERE user_id='a'").get());
});

test('context guard blocks obsolete dependency publication and stop helper is idempotent and isolated',async t=>{
 const s=setup(t);s.seed(plan());s.seed(plan(),'b');const snapshot=await s.state(),guard=changeGuard('a',snapshot);
 const first=await s.DB.batch(changeArtifactStatements(s.env,'a','turn','transmitted',[{kind:'goals',id:'goal_a',revision:1}],0,guard.sql,guard.args));assert.equal(first[0].meta.changes,1);
 const stopped=await suspendChangeFollowups(s.env,'a',s.now());assert.equal(stopped.reminders_paused,true);assert.equal(stopped.revision,snapshot.revision+1);const again=await suspendChangeFollowups(s.env,'a',s.now());assert.equal(again.revision,stopped.revision);assert.equal((await s.state('b')).reminders_paused,0);
 const stale=await s.DB.batch(changeArtifactStatements(s.env,'a','turn','obsolete',[{kind:'goals',id:'goal_a',revision:1}],0,guard.sql,guard.args));assert.equal(stale[0].meta.changes,0);assert.equal(s.raw.prepare("SELECT count(*) AS n FROM change_artifact_refs WHERE artifact_id='obsolete'").get().n,0);
});
