// User-owned choices and reports. A model may propose; only the member adopts.
import {contextHash} from './private-context.mjs';

export const PLAN_LIMITS=Object.freeze({goals:20,active_goals:3,steps:60,trials:200,proposals:40,preparations:20,instructions:10,bytes:262144});
export const PLAN_COLLECTIONS=['goals','steps','trials','proposals','preparations','instructions'];
const parse=(value,fallback={})=>{try{return JSON.parse(value)}catch{return fallback}};
const clone=value=>JSON.parse(JSON.stringify(value));
const missingTable=error=>/no such table:\s*change_plans/i.test(String(error?.message||''));
export const emptyPlan=()=>Object.fromEntries(PLAN_COLLECTIONS.map(key=>[key,[]]));
export function planDocument(value){
 const document=typeof value==='string'?parse(value):value||{};
 return Object.fromEntries(PLAN_COLLECTIONS.map(key=>[key,Array.isArray(document[key])?document[key]:[]]));
}
export async function readChangeState(env,userId){
 try{
  const row=await env.DB.prepare('SELECT * FROM change_plans WHERE user_id=?').bind(userId).first();
  return {available:true,user_id:userId,revision:0,epoch:0,context_generation:0,proposals_enabled:1,reminders_paused:0,exploration_reconsent_required:0,past_safety_signal:0,updated_at:null,...row,document:planDocument(row?.document_json)};
 }catch(error){if(missingTable(error))return {available:false,user_id:userId,revision:0,epoch:0,context_generation:0,document:emptyPlan()};throw error}
}
export const changeControl=state=>({revision:state.revision,epoch:state.epoch,context_generation:state.context_generation,proposals_enabled:Boolean(state.proposals_enabled),reminders_paused:Boolean(state.reminders_paused),exploration_reconsent_required:Boolean(state.exploration_reconsent_required),past_safety_signal:Boolean(state.past_safety_signal)});
export function changeGuard(userId,state){
 return {sql:'COALESCE((SELECT revision FROM change_plans WHERE user_id=?),0)=? AND COALESCE((SELECT epoch FROM change_plans WHERE user_id=?),0)=? AND COALESCE((SELECT context_generation FROM change_plans WHERE user_id=?),0)=?',args:[userId,state.revision,userId,state.epoch,userId,state.context_generation]};
}
export function changeWrite(env,state,document,{now,mutation,guard='1=1',guardArgs=[],controls={}}){
 const fields={...changeControl(state),...controls},serialized=JSON.stringify(document);
 if(new TextEncoder().encode(serialized).length>PLAN_LIMITS.bytes)throw Object.assign(Error('Votre plan atteint sa limite de stockage. Exportez puis retirez volontairement des éléments avant d’en ajouter.'),{status:422,code:'plan_limit'});
 return env.DB.prepare(`INSERT INTO change_plans(user_id,revision,epoch,context_generation,proposals_enabled,reminders_paused,exploration_reconsent_required,past_safety_signal,document_json,last_mutation,updated_at)
 SELECT ?,?,?,?,?,?,?,?,?,?,? WHERE ${guard}
 ON CONFLICT(user_id) DO UPDATE SET revision=excluded.revision,epoch=excluded.epoch,context_generation=excluded.context_generation,proposals_enabled=excluded.proposals_enabled,reminders_paused=excluded.reminders_paused,exploration_reconsent_required=excluded.exploration_reconsent_required,past_safety_signal=excluded.past_safety_signal,document_json=excluded.document_json,last_mutation=excluded.last_mutation,updated_at=excluded.updated_at
 WHERE change_plans.revision=? AND change_plans.epoch=? AND change_plans.context_generation=?`)
 .bind(state.user_id,state.revision+1,fields.epoch,fields.context_generation,Number(fields.proposals_enabled),Number(fields.reminders_paused),Number(fields.exploration_reconsent_required),Number(fields.past_safety_signal),serialized,mutation,now,...guardArgs,state.revision,state.epoch,state.context_generation);
}
export const mutationGuard=(userId,mutation)=>({sql:'EXISTS(SELECT 1 FROM change_plans WHERE user_id=? AND last_mutation=?)',args:[userId,mutation]});
export function planDependencies(document){
 return PLAN_COLLECTIONS.flatMap(kind=>document[kind].map(item=>({kind,id:item.id,revision:item.revision||1})));
}
export function changeArtifactStatements(env,userId,artifactKind,artifactId,refs,generation,guard='1=1',args=[]){
 return refs.map(ref=>env.DB.prepare(`INSERT OR IGNORE INTO change_artifact_refs(user_id,artifact_kind,artifact_id,entity_kind,entity_id,entity_revision,context_generation) SELECT ?,?,?,?,?,?,? WHERE ${guard}`).bind(userId,artifactKind,artifactId,ref.kind,ref.id,ref.revision||1,generation,...args));
}
export function parisDay(seconds){return new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Paris',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(seconds*1000))}
export function isDay(value){
 if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(value))return false;
 const date=new Date(value+'T12:00:00Z');return Number.isFinite(date.getTime())&&date.toISOString().slice(0,10)===value;
}
const fold=value=>String(value||'').normalize('NFD').replace(/\p{M}/gu,'').toLowerCase().replace(/[’']/g,' ').replace(/[^\p{L}\p{N}\s]/gu,' ').replace(/\s+/g,' ').trim();
export function isStopFollowups(message){
 return ['ne me relance plus','ne me relancez plus','arrete les relances','arretez les relances','ne me rappelle plus les exercices','ne me rappelez plus les exercices'].includes(fold(message));
}
const words=text=>new Set(fold(text).split(' ').filter(word=>word.length>=5&&!['comme','cette','avoir','faire','voudrais','souhaite','essayer','pouvoir','quand','apres','avant','aimerais','parler','aussi','autre','toujours','jamais'].includes(word)));
export function relevantToTrial(message,trial,step,goal){
 if(/\b(mon essai|cet essai|cet exercice|mon exercice|j ai essaye|comme convenu|comme prevu|faire le point)\b/.test(fold(message)))return true;
 const current=words(message),topic=words([trial.step_snapshot?.text,step?.text,goal?.title].filter(Boolean).join(' '));
 return [...current].some(word=>topic.has(word));
}
export function dueChangeTrial(state,message,now,{turnId=null,previousTurnId=null}={}){
 if(!state.available||state.reminders_paused)return null;
 const doc=state.document,today=parisDay(now);
 return doc.trials.filter(trial=>{
  const step=doc.steps.find(x=>x.id===trial.step_id),goal=doc.goals.find(x=>x.id===step?.goal_id);
  return trial.status==='planned'&&trial.followup_opt_in&&trial.review_mode!=='none'&&step?.status==='active'&&goal?.status==='active'&&trial.reminder_count<2&&(!trial.last_reminder_at||now-trial.last_reminder_at>=604800)&&trial.last_reminder_turn_id!==turnId&&(!previousTurnId||trial.last_reminder_turn_id!==previousTurnId)&&trial.source_turn_id!==turnId&&trial.created_at<now&&(trial.review_mode==='next_conversation'||trial.review_mode==='on_date'&&isDay(trial.review_on)&&trial.review_on<=today)&&relevantToTrial(message,trial,step,goal);
 }).sort((a,b)=>{const priority=trial=>doc.goals.find(goal=>goal.id===doc.steps.find(step=>step.id===trial.step_id)?.goal_id)?.priority||2;return (a.review_on||'').localeCompare(b.review_on||'')||priority(a)-priority(b)||a.created_at-b.created_at})[0]||null;
}
export function compactChangeContext(state,message,now,options={}){
 if(!state.available)return {available:false};
 const doc=state.document,terms=words(message),score=item=>[...words([item.title,item.text,item.result_text].filter(Boolean).join(' '))].filter(word=>terms.has(word)).length;
 const goals=doc.goals.filter(x=>x.status==='active').sort((a,b)=>score(b)-score(a)||a.priority-b.priority).slice(0,3);
 const goal=goals[0],steps=doc.steps.filter(x=>x.goal_id===goal?.id&&x.status==='active'),step=steps.sort((a,b)=>score(b)-score(a)||b.updated_at-a.updated_at)[0];
 const trials=doc.trials.filter(x=>x.step_id===step?.id).sort((a,b)=>b.updated_at-a.updated_at),trial=trials[0],due=dueChangeTrial(state,message,now,options);
 const refs=[];
 const pick=(kind,item,fields)=>{if(!item)return null;refs.push({kind,id:item.id,revision:item.revision});return Object.fromEntries(['id','revision',...fields].filter(k=>item[k]!==undefined).map(k=>[k,item[k]]))};
 const context={available:true,...changeControl(state),timezone:'Europe/Paris',goals:goals.map(x=>pick('goals',x,['title','why','priority','status'])),step:pick('steps',step,['goal_id','text','strategy_id','measure','status']),trial:pick('trials',trial,['step_id','status','attempt','result_text','outcome','adverse_effect_text','occurred_on','planned_on']),due_trial:pick('trials',due,['step_id','step_snapshot','review_mode','review_on','reminder_count'])};
 // Remove whole optional blocks. Permissions and restrictions are never clipped.
 for(const key of ['trial','step'])if(JSON.stringify(context).length>1800)context[key]=null;
 while(context.goals.length>1&&JSON.stringify(context).length>1800)context.goals.pop();
 if(JSON.stringify(context).length>1800)context.goals=context.goals.map(({id,revision,title,status})=>({id,revision,title,status}));
 if(context.due_trial&&JSON.stringify(context).length>1800){context.due_trial=null;context.followup_omitted=true}
 const used=new Set([...context.goals.map(x=>x.id),context.step?.id,context.trial?.id,context.due_trial?.id].filter(Boolean));
 return {...context,dependencies:refs.filter((ref,i,list)=>used.has(ref.id)&&list.findIndex(x=>x.id===ref.id)===i)};
}
export function privateCareInstructions(state,compact){
 if(!state.available)return {items:[],omitted:0};
 const selected=state.document.instructions.filter(x=>x.status==='current'&&(!x.goal_id||compact.goals?.some(g=>g.id===x.goal_id)));
 const result=[];let chars=0;
 for(const item of selected){const entry={id:item.id,revision:item.revision,text:item.text,origin:'reported_by_member',updated_at:item.updated_at};const size=JSON.stringify(entry).length;if(chars+size>500)continue;result.push(entry);chars+=size}
 return {items:result,omitted:selected.length-result.length,requires_session_check:selected.length>result.length};
}
export function serializeSharedPlan(state){
 const fields={goals:['id','revision','title','why','status','priority','accepted_at','updated_at'],steps:['id','revision','goal_id','text','strategy_id','measure','status','accepted_at','updated_at'],trials:['id','revision','step_id','step_snapshot','status','attempt','result_text','outcome','adverse_effect_text','planned_on','occurred_on','timing_note','updated_at']};
 const adopted=item=>item&&typeof item==='object'&&!Array.isArray(item)&&typeof item.id==='string'&&Number.isFinite(item.accepted_at)&&item.accepted_at>=0;
 const scalar=(key,value)=>['revision','priority','accepted_at','updated_at'].includes(key)?Number.isFinite(value):typeof value==='string'||value===null;
 const goals=state.document.goals.filter(adopted),goalIds=new Set(goals.map(item=>item.id)),steps=state.document.steps.filter(item=>adopted(item)&&goalIds.has(item.goal_id)),stepIds=new Set(steps.map(item=>item.id)),trials=state.document.trials.filter(item=>adopted(item)&&stepIds.has(item.step_id));
 const chosen={goals,steps,trials};
 return {revision:state.revision,updated_at:state.updated_at,...Object.fromEntries(Object.entries(fields).map(([kind,keys])=>[kind,chosen[kind].map(item=>Object.fromEntries(keys.filter(key=>key==='step_snapshot'||scalar(key,item[key])).map(key=>[key,key==='step_snapshot'?Object.fromEntries(['text','measure','strategy_id','strategy_version'].filter(field=>typeof item.step_snapshot?.[field]==='string'||item.step_snapshot?.[field]===null).map(field=>[field,item.step_snapshot[field]])):item[key]])))]))};
}
export async function changeRequestHash(method,path,body){
 const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().filter(key=>!['request_key','revision'].includes(key)).map(key=>[key,canonical(value[key])])):value;
 return contextHash(JSON.stringify([method,path,canonical(body)]));
}

// Returns statements only. The caller executes these after its guarded CAS in
// the same D1 batch; a failed CAS cannot erase another request's data.
export function invalidateChangeArtifacts(env,userId,ids,mutation,{cutoff=false,now=0}={}){
 const guard=mutationGuard(userId,mutation),selected=JSON.stringify(ids),ref=`SELECT artifact_id FROM change_artifact_refs WHERE user_id=? AND entity_id IN (SELECT value FROM json_each(?)) AND artifact_kind=`,statements=[];
 const stmt=(sql,...args)=>env.DB.prepare(sql).bind(...args);
 statements.push(stmt(`UPDATE chat_turns SET response='' WHERE user_id=? AND id IN (${ref}'turn') AND ${guard.sql}`,userId,userId,selected,...guard.args));
 statements.push(stmt(`DELETE FROM chat_turn_traces WHERE user_id=? AND turn_id IN (${ref}'turn') AND ${guard.sql}`,userId,userId,selected,...guard.args));
 statements.push(stmt(`DELETE FROM chat_engine_jobs WHERE user_id=? AND turn_id IN (${ref}'turn') AND ${guard.sql}`,userId,userId,selected,...guard.args));
 statements.push(stmt(`UPDATE care_briefs SET status='revoked',content='',notes='[]',lease_token=NULL,lease_until=0,updated_at=? WHERE user_id=? AND (id IN (${ref}'brief') OR link_id IN (SELECT id FROM care_links WHERE user_id=? AND EXISTS(SELECT 1 FROM json_each(scopes) WHERE value='plan'))) AND ${guard.sql}`,now,userId,userId,selected,userId,...guard.args));
 if(cutoff){
  statements.push(stmt(`DELETE FROM chat_memory WHERE user_id=? AND ${guard.sql}`,userId,...guard.args));
  statements.push(stmt(`DELETE FROM chat_memory_jobs WHERE user_id=? AND ${guard.sql}`,userId,...guard.args));
  statements.push(stmt(`UPDATE chat_turns SET status='failed',error_code='context_changed',credit_status=CASE WHEN credit_status='reserved' THEN 'released' ELSE credit_status END,lease_token=NULL,lease_until=0 WHERE user_id=? AND status='running' AND ${guard.sql}`,userId,...guard.args));
  statements.push(stmt(`DELETE FROM chat_engine_jobs WHERE user_id=? AND stage NOT IN ('done','failed') AND ${guard.sql}`,userId,...guard.args));
  statements.push(stmt(`UPDATE ambre_reviews SET content='',status='failed',error_code='sources_changed',lease_token=NULL,lease_until=0 WHERE user_id=? AND ${guard.sql}`,userId,...guard.args));
  statements.push(stmt(`UPDATE chat_turns SET response='',purged=1 WHERE user_id=? AND idempotency_key LIKE 'review-%' AND ${guard.sql}`,userId,...guard.args));
  statements.push(stmt(`UPDATE care_briefs SET status='revoked',content='',notes='[]',lease_token=NULL,lease_until=0,updated_at=? WHERE user_id=? AND ${guard.sql}`,now,userId,...guard.args));
 }
 statements.push(stmt(`DELETE FROM change_artifact_refs WHERE user_id=? AND entity_id IN (SELECT value FROM json_each(?)) AND ${guard.sql}`,userId,selected,...guard.args));
 return statements;
}

export async function suspendChangeFollowups(env,userId,now){
 for(let attempt=0;attempt<4;attempt++){
  const state=await readChangeState(env,userId);
  if(!state.available)throw Object.assign(Error('Impossible de suspendre les retours pour le moment.'),{status:503,code:'plan_unavailable'});
  if(state.reminders_paused)return changeControl(state);
  const result=await changeWrite(env,state,state.document,{now,mutation:crypto.randomUUID(),controls:{reminders_paused:true}}).run();
  if(result.meta.changes)return changeControl(await readChangeState(env,userId));
 }
 throw Object.assign(Error('Le plan a changé. Réessayez de suspendre les retours.'),{status:409,code:'revision_conflict'});
}

// Compose with the caller's guarded chat purge in ONE batch. Check result[0]
// before reporting success; every following statement must use returned guard.
export async function clearChatChangeState(env,userId,{includePlan=false,now}={}){
 const state=await readChangeState(env,userId);
 if(!state.available)throw Object.assign(Error('Le plan est momentanément indisponible.'),{status:503,code:'plan_unavailable'});
 const document=includePlan?emptyPlan():clone(state.document),removed=[];
 if(includePlan)removed.push(...PLAN_COLLECTIONS.flatMap(key=>state.document[key].map(item=>item.id)));
 else{
  removed.push(...document.proposals.filter(item=>item.source_turn_id||item.origin==='ambre').map(item=>item.id));
  document.proposals=document.proposals.filter(item=>!removed.includes(item.id));
  document.preparations=document.preparations.filter(item=>!item.memory_revision);
  for(const kind of PLAN_COLLECTIONS)for(const item of document[kind]){
   if(item.source_turn_id||item.source_quote){delete item.source_turn_id;delete item.source_quote;item.source_removed=true}
  }
 }
 const mutation=crypto.randomUUID(),guard=mutationGuard(userId,mutation),controls={epoch:state.epoch+1,context_generation:state.context_generation+1,past_safety_signal:false,exploration_reconsent_required:false,...(includePlan?{reminders_paused:true,proposals_enabled:false}:{})};
 const statements=[changeWrite(env,state,document,{now,mutation,controls})];
 statements.push(env.DB.prepare(`UPDATE change_requests SET deleted=1 WHERE user_id=? AND entity_id IN (SELECT value FROM json_each(?)) AND ${guard.sql}`).bind(userId,JSON.stringify(removed),...guard.args));
 statements.push(env.DB.prepare(`DELETE FROM change_artifact_refs WHERE user_id=? AND ${guard.sql}`).bind(userId,...guard.args));
 return {statements,guard,mutation,previous:changeControl(state),next:{...changeControl(state),...controls,revision:state.revision+1},retained:includePlan?'Le plan a aussi été effacé.':'Les objectifs et essais que vous avez choisis restent dans Mon suivi. Leurs citations de conversation ont été retirées.'};
}

export async function clearChangeMemoryState(env,userId,{now,expectedMemoryRevision}={}){
 const state=await readChangeState(env,userId);
 if(!state.available)throw Object.assign(Error('La mémoire est momentanément indisponible.'),{status:503,code:'plan_unavailable'});
 const document=clone(state.document);document.preparations=document.preparations.filter(item=>!item.memory_revision);
 const mutation=crypto.randomUUID(),guard=mutationGuard(userId,mutation),controls={context_generation:state.context_generation+1,past_safety_signal:false,exploration_reconsent_required:false};
 if(!Number.isInteger(expectedMemoryRevision)||expectedMemoryRevision<0)throw Object.assign(Error('Version de mémoire requise.'),{status:400,code:'invalid_revision'});
 const statements=[changeWrite(env,state,document,{now,mutation,controls,guard:'COALESCE((SELECT revision FROM chat_memory WHERE user_id=?),0)=?',guardArgs:[userId,expectedMemoryRevision]}),...invalidateChangeArtifacts(env,userId,[],mutation,{cutoff:true,now})];
 return {statements,guard,mutation,previous:changeControl(state),next:{...changeControl(state),...controls,revision:state.revision+1}};
}
