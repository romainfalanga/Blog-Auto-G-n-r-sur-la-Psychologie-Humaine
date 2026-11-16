import {CHANGE_STRATEGIES} from './change-strategies.generated.mjs';
import {PLAN_LIMITS,PLAN_COLLECTIONS,emptyPlan,readChangeState,changeControl,changeWrite,mutationGuard,changeRequestHash,invalidateChangeArtifacts,isDay,parisDay,serializeSharedPlan} from './change-plan.mjs';

const parse=(value,fallback={})=>{try{return JSON.parse(value)}catch{return fallback}};
const clone=value=>JSON.parse(JSON.stringify(value));
const strategyIds=new Set(CHANGE_STRATEGIES.strategies.map(item=>item.id));
const statuses=['active','paused','completed','abandoned'];
const attempts=['done','partial','not_tried','stopped','uncertain'];
const outcomes=['helpful','neutral','worse','uncertain'];
const clean=value=>value.replace(/\r\n?/g,'\n').replace(/[\u0000-\u0008\u000B-\u001F\u007F\u200B-\u200F\u202A-\u202E\u2060\uFEFF]/g,'').trim();
export function createChangeBackend({error,json,body,now,random,limit}){
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 const first=(env,sql,...args)=>stmt(env,sql,...args).first();
 const fail=(status,message,code='invalid_plan')=>error(status,message,code);
 const member=user=>{if(user.role==='psychologist')fail(403,'Le plan personnel appartient à la personne accompagnée.','member_required')};
 const text=(value,max,required=false)=>{if(value===undefined||value===null){if(required)fail(400,'Renseignez le texte demandé.');return ''}if(typeof value!=='string'||value.length>max)fail(400,'Ce texte doit contenir au plus '+max+' caractères.');const result=clean(value);if(required&&!result)fail(400,'Renseignez le texte demandé.');return result};
 const choice=(value,values,fallback)=>{const result=value===undefined?fallback:value;if(!values.includes(result))fail(400,'Valeur de plan inconnue.');return result};
 const day=value=>{if(value===undefined||value===null||value==='')return null;if(!isDay(value))fail(400,'Indiquez une date réelle au format AAAA-MM-JJ.','invalid_date');return value};
 const strategy=value=>{if(value===undefined||value===null||value==='')return null;if(!strategyIds.has(value))fail(400,'Stratégie inconnue.');return value};
 const bool=(value,fallback=false)=>{if(value===undefined)return fallback;if(typeof value!=='boolean')fail(400,'Choix invalide.');return value};
 const get=(document,kind,id)=>{const item=document[kind].find(row=>row.id===id);if(!item)fail(404,'Cet élément n’est pas disponible.','not_found');return item};
 const ensure=state=>{if(!state.available)fail(503,'Le plan est momentanément indisponible.','plan_unavailable');return state};
 const revision=(state,data)=>{if(data.epoch!==state.epoch)fail(409,'Votre plan a été effacé ou renouvelé. Rechargez-le.','stale_epoch');if(data.revision!==state.revision)fail(409,'Votre plan a changé. Votre saisie est conservée ; rechargez la version actuelle.','revision_conflict')};
 function objectRevision(item,data){if(data.entity_revision!==undefined&&data.entity_revision!==item.revision)fail(409,'Cet élément a changé. Rechargez sa version actuelle.','revision_conflict')}
 function validateBounds(document){
  for(const kind of PLAN_COLLECTIONS)if(document[kind].filter(item=>kind!=='proposals'||item.status==='proposed').length>PLAN_LIMITS[kind])fail(422,'La limite de '+PLAN_LIMITS[kind]+' éléments est atteinte pour cette rubrique. Exportez puis supprimez volontairement des éléments.','plan_limit');
  if(document.goals.filter(item=>item.status==='active').length>PLAN_LIMITS.active_goals)fail(422,'Trois objectifs peuvent être actifs. Mettez un objectif en pause avant d’en activer un autre.','active_goal_limit');
 }
 function visible(state,url=new URL('https://local/api/plan')){
  const collection=url.searchParams.get('collection'),offset=Math.max(0,Math.floor(Number(url.searchParams.get('offset'))||0)),document=state.document;
  if(collection&&!PLAN_COLLECTIONS.includes(collection))fail(400,'Rubrique inconnue.');
  const selected=collection?[collection]:PLAN_COLLECTIONS;
  const result={available:state.available,...changeControl(state),updated_at:state.updated_at,timezone:'Europe/Paris',limits:PLAN_LIMITS,counts:Object.fromEntries(PLAN_COLLECTIONS.map(key=>[key,document[key].length])),next_offsets:{}};
  for(const key of selected){const start=collection?offset:0;result[key]=document[key].slice(start,start+50);result.next_offsets[key]=document[key].length>start+50?start+50:null}
  return result;
 }
 function goalFields(data,old={}){
  return {title:text(data.title??old.title,240,true),why:text(data.why??old.why,800),status:choice(data.status,statuses,old.status||'active'),priority:choice(data.priority,[1,2,3],old.priority||2)};
 }
 function stepFields(data,old={}){
  return {text:text(data.text??old.text,800,true),measure:text(data.measure??old.measure,300),strategy_id:strategy(data.strategy_id===undefined?old.strategy_id:data.strategy_id),status:choice(data.status,statuses,old.status||'active')};
 }
 function trialFields(data,old={}){
  const status=choice(data.status,['planned','reported','cancelled'],old.status||'planned'),followup=bool(data.followup_opt_in,old.followup_opt_in||false);
  if(old.status==='reported'&&status==='planned')fail(400,'Créez un nouvel essai pour répéter cette étape ; votre résultat précédent reste conservé.');
  const mode=followup?choice(data.review_mode,['none','on_date','next_conversation'],old.review_mode||'none'):'none',reviewOn=day(data.review_on===undefined?old.review_on:data.review_on);
  if(followup&&(mode==='none'||mode==='on_date'&&!reviewOn))fail(400,'Choisissez quand Ambre pourra revenir sur cet essai.');
  const result={status,planned_on:day(data.planned_on===undefined?old.planned_on:data.planned_on),occurred_on:day(data.occurred_on===undefined?old.occurred_on:data.occurred_on),occurred_at:null,timing_note:text(data.timing_note??old.timing_note,240),followup_opt_in:status==='planned'&&followup,review_mode:status==='planned'?mode:'none',review_on:mode==='on_date'?reviewOn:null};
  if(status==='reported')Object.assign(result,{attempt:choice(data.attempt,attempts,old.attempt||'uncertain'),result_text:text(data.result_text??old.result_text,1200,true),outcome:choice(data.outcome,[null,...outcomes],old.outcome??null),adverse_effect_text:text(data.adverse_effect_text??old.adverse_effect_text,1200)});
  else Object.assign(result,{attempt:null,result_text:'',outcome:null,adverse_effect_text:''});
  return result;
 }
 const newItem=(fields,prefix)=>({id:prefix+'_'+random(),revision:1,created_at:now(),updated_at:now(),...fields});
 const stamp=(old,fields)=>Object.assign(old,fields,{revision:old.revision+1,updated_at:now()});
 function addGoal(document,data,origin='member'){
  const goal=newItem({...goalFields(data),origin,accepted_at:now()},'goal');document.goals.push(goal);return goal;
 }
 function addStep(document,goalId,data,origin='member'){
  get(document,'goals',goalId);const step=newItem({goal_id:goalId,...stepFields(data),origin,accepted_at:now()},'step');document.steps.push(step);return step;
 }
 function addTrial(document,stepId,data,origin='member'){
  const step=get(document,'steps',stepId);get(document,'goals',step.goal_id);
  const trial=newItem({step_id:stepId,step_snapshot:{text:step.text,measure:step.measure,strategy_id:step.strategy_id},...trialFields(data),origin,accepted_at:now(),reminder_count:0,last_reminder_at:null,last_reminder_turn_id:null},'trial');document.trials.push(trial);return trial;
 }
 async function receipt(env,userId,data,method,path,state){
  if(typeof data.request_key!=='string'||!/^[\w-]{8,100}$/.test(data.request_key))fail(400,'Une clé de requête est nécessaire.','invalid_request_key');
  const hash=await changeRequestHash(method,path,data),existing=await first(env,'SELECT * FROM change_requests WHERE user_id=? AND request_key=?',userId,data.request_key);
  if(existing){
   if(existing.epoch!==state.epoch||data.epoch!==state.epoch)fail(409,'Cette demande appartient à une ancienne version de votre plan.','stale_epoch');
   if(existing.request_hash!==hash)fail(409,'Cette clé a déjà servi pour un autre contenu.','idempotency_conflict');
   if(existing.deleted||!state.document[existing.entity_kind]?.some(item=>item.id===existing.entity_id))fail(410,'Cet élément a été supprimé. Il ne sera pas recréé.','resource_deleted');
   return {replayed:true,entity_kind:existing.entity_kind,entity_id:existing.entity_id};
  }
  return {hash};
 }
 async function save(env,state,document,{data,controls={},receiptInfo=null,entityKind='',entityId='',removed=[],changedContents=[]}={}){
  validateBounds(document);
  const affected=[...new Set([...removed,...changedContents])],refs=affected.length?await first(env,'SELECT 1 AS found FROM change_artifact_refs WHERE user_id=? AND entity_id IN (SELECT value FROM json_each(?)) LIMIT 1',state.user_id,JSON.stringify(affected)):null;
  const cutoff=Boolean(refs),memory=cutoff?parse((await first(env,'SELECT summary FROM chat_memory WHERE user_id=?',state.user_id))?.summary):{};
  if(cutoff){
   controls={...controls,context_generation:state.context_generation+1,exploration_reconsent_required:Boolean(state.exploration_reconsent_required||memory.closed?.length||memory.closed_overflow),past_safety_signal:Boolean(state.past_safety_signal||memory.safety_seen||memory.safety?.length)};
   // A point with an unproven memory source cannot survive a continuity reset.
   document.preparations=document.preparations.filter(item=>!item.memory_revision);
  }
  if(affected.length)document.preparations=document.preparations.filter(item=>!(item.dependencies||[]).some(ref=>affected.includes(ref.id)));
  const mutation=random(),guard=mutationGuard(state.user_id,mutation),statements=[changeWrite(env,state,document,{now:now(),mutation,controls})];
  if(receiptInfo)statements.push(stmt(env,`INSERT INTO change_requests(user_id,request_key,epoch,request_hash,entity_kind,entity_id,created_at) SELECT ?,?,?,?,?,?,? WHERE ${guard.sql}`,state.user_id,data.request_key,state.epoch,receiptInfo.hash,entityKind,entityId,now(),...guard.args));
  if(removed.length)statements.push(stmt(env,`UPDATE change_requests SET deleted=1 WHERE user_id=? AND entity_id IN (SELECT value FROM json_each(?)) AND ${guard.sql}`,state.user_id,JSON.stringify(removed),...guard.args));
  if(affected.length)statements.push(...invalidateChangeArtifacts(env,state.user_id,affected,mutation,{cutoff,now:now()}));
  const result=await env.DB.batch(statements);
  if(!result[0].meta.changes)fail(409,'Votre plan a changé. Rechargez-le avant d’enregistrer votre saisie.','revision_conflict');
  return {state:ensure(await readChangeState(env,state.user_id)),...(cutoff?{continuity_reset:true}:{}),...(removed.length?{removed:'Les éléments du plan et leurs copies générées ont été retirés.',retained:'Vos messages originaux restent dans votre conversation. Vous pouvez effacer aussi cette conversation.'}:{})};
 }
 function removeObject(document,kind,id){
  get(document,kind,id);const ids=new Set([id]);
  if(kind==='goals')for(const step of document.steps)if(step.goal_id===id)ids.add(step.id);
  if(kind==='goals')for(const instruction of document.instructions)if(instruction.goal_id===id)ids.add(instruction.id);
  if(kind==='goals'||kind==='steps')for(const trial of document.trials)if(ids.has(trial.step_id))ids.add(trial.id);
  for(const proposal of document.proposals)if(ids.has(proposal.target_id)||ids.has(proposal.resolved_id)||ids.has(proposal.payload?.goal_id))ids.add(proposal.id);
  for(const key of PLAN_COLLECTIONS)document[key]=document[key].filter(item=>!ids.has(item.id));
  return [...ids];
 }
 async function preparation(env,state,url){
  const sessionId=url.searchParams.get('session_id');let session=null,since=0;
  if(sessionId){session=await first(env,'SELECT * FROM care_sessions WHERE id=? AND user_id=?',sessionId,state.user_id);if(!session)fail(404,'Séance indisponible.','not_found');since=(await first(env,'SELECT max(session_at) AS at FROM care_sessions WHERE user_id=? AND session_at<?',state.user_id,session.session_at))?.at||0}
  const dependencies=[],parts=['Mon point personnel'+(session?' — séance du '+parisDay(session.session_at):''),'Les éléments ci-dessous sont mes choix et mes retours. Ils ne constituent pas une évaluation clinique.'],coverage={included:0,omitted:0,since:since||null};
  const add=(kind,item,line)=>{if(parts.join('\n\n').length+line.length+2>5400){coverage.omitted++;return}parts.push(line);dependencies.push({kind,id:item.id,revision:item.revision});coverage.included++};
  for(const goal of state.document.goals.filter(item=>['active','paused'].includes(item.status)))add('goals',goal,'Objectif choisi : '+goal.title+(goal.why?'\nCe qui compte pour moi : '+goal.why:''));
  for(const trial of state.document.trials.filter(item=>item.status==='reported'&&(!item.occurred_on||!since||item.occurred_on>=parisDay(since))).sort((a,b)=>b.updated_at-a.updated_at))add('trials',trial,(trial.occurred_on?'Essai du '+trial.occurred_on:'Essai — date non précisée')+'\n'+trial.step_snapshot.text+'\nMon retour : '+trial.result_text+(trial.adverse_effect_text?'\nEffet pénible : '+trial.adverse_effect_text:''));
  parts.push('Ma question pour la séance :');
  const memory=await first(env,'SELECT summary,revision FROM chat_memory WHERE user_id=?',state.user_id),summary=parse(memory?.summary);
  return {content:parts.join('\n\n'),dependencies,coverage,session_id:sessionId||null,session_revision:session?.revision||null,revision:state.revision,epoch:state.epoch,memory_revision:memory?.revision||0,memory_notes:['goals','tried','session'].flatMap(section=>(summary[section]||[]).map(text=>({section,text,origin:'memory_unverified'}))),memory_notice:'Ces notes de conversation restent à vérifier. Elles ne sont pas ajoutées automatiquement au point et ne sont pas partagées.'};
 }
 async function routes(request,env,user,path,url){
  if(!path.startsWith('/api/plan'))return null;member(user);await limit(env,'change-api:'+user.id,400,3600);
  let state=ensure(await readChangeState(env,user.id));const method=request.method;
  if(method==='GET'){
   if(path==='/api/plan')return json(visible(state,url));
   if(path==='/api/plan/catalogue')return json(CHANGE_STRATEGIES);
   if(path==='/api/plan/sharing-preview')return json(serializeSharedPlan(state));
   if(path==='/api/plan/preparation')return json(await preparation(env,state,url));
   const match=path.match(/^\/api\/plan\/(goals|steps|trials|proposals|preparations|instructions)\/([\w-]+)$/);
   if(match)return json({item:get(state.document,match[1],match[2]),...changeControl(state)});
   fail(404,'Route inconnue.','not_found');
  }
  const data=await body(request,20000);
  // A stale screen may still restrict reminders. It may never re-enable them.
  if(path==='/api/plan/followups'&&method==='PUT'&&data.enabled===false){
   for(let attempt=0;attempt<3;attempt++){
    const document=clone(state.document),controls={};
    if(!data.target_id||data.target_id==='all')controls.reminders_paused=true;
    else stamp(get(document,'trials',data.target_id),{followup_opt_in:false,review_mode:'none'});
    try{const result=await save(env,state,document,{controls});return json({...visible(result.state),...result.continuity_reset&&{continuity_reset:true}})}catch(err){if(err.code!=='revision_conflict'||attempt===2)throw err;state=ensure(await readChangeState(env,user.id))}
   }
  }
  let receiptInfo=null;
  if(method==='POST'){receiptInfo=await receipt(env,user.id,data,method,path,state);if(receiptInfo.replayed)return json({...visible(state),replayed:true,entity_kind:receiptInfo.entity_kind,entity_id:receiptInfo.entity_id})}
  revision(state,data);let document=clone(state.document),controls={},entityKind='',entityId='',removed=[],changedContents=[];
  if(path==='/api/plan/preferences'&&method==='PUT'){
   if(typeof data.proposals_enabled!=='boolean')fail(400,'Indiquez votre choix pour les propositions.');controls.proposals_enabled=data.proposals_enabled;
  }else if(path==='/api/plan/followups'&&method==='PUT'){
   if(data.enabled!==true||data.confirmed!==true)fail(400,'La réactivation doit être choisie explicitement.');
   if(!data.target_id||data.target_id==='all')controls.reminders_paused=false;
   else {const trial=get(document,'trials',data.target_id);stamp(trial,trialFields({...data,followup_opt_in:true},trial))}
  }else if(path==='/api/plan'&&method==='DELETE'){
   removed=PLAN_COLLECTIONS.flatMap(key=>document[key].map(item=>item.id));document=emptyPlan();controls={epoch:state.epoch+1,reminders_paused:true,proposals_enabled:false};
  }else if(path==='/api/plan/goals'&&method==='POST'){
   const item=addGoal(document,data);entityKind='goals';entityId=item.id;
  }else if(path==='/api/plan/instructions'&&method==='POST'){
   if(data.goal_id)get(document,'goals',data.goal_id);
   const item=newItem({text:text(data.text,1200,true),status:choice(data.status,['current','to_check','completed'],'current'),goal_id:data.goal_id||null,origin:'reported_by_member'},'instruction');document.instructions.push(item);entityKind='instructions';entityId=item.id;
  }else if(path==='/api/plan/preparations'&&method==='POST'){
   const dependencies=Array.isArray(data.dependencies)?data.dependencies:[];
   if(dependencies.length>100)fail(400,'Trop de références dans le point.');
   for(const ref of dependencies){if(!['goals','steps','trials','instructions'].includes(ref.kind)||get(document,ref.kind,ref.id).revision!==ref.revision)fail(409,'Les éléments du point ont changé. Préparez une nouvelle version.','revision_conflict')}
   let memoryRevision=null;if(data.include_memory===true){const memory=await first(env,'SELECT revision FROM chat_memory WHERE user_id=?',user.id);if(!memory||memory.revision!==data.memory_revision)fail(409,'Les notes de mémoire ont changé.','revision_conflict');memoryRevision=memory.revision}
   const item=newItem({content:text(data.content,6000,true),dependencies,memory_revision:memoryRevision,...(memoryRevision?{memory_origin:'memory_unverified'}:{}),private:true},'preparation');document.preparations.push(item);entityKind='preparations';entityId=item.id;
  }else{
   let match=path.match(/^\/api\/plan\/goals\/([\w-]+)\/steps$/);
   if(match&&method==='POST'){const item=addStep(document,match[1],data);entityKind='steps';entityId=item.id}
   else if((match=path.match(/^\/api\/plan\/steps\/([\w-]+)\/(trials|results)$/))&&method==='POST'){
    const item=addTrial(document,match[1],{...data,...(match[2]==='results'?{status:'reported'}:{})});entityKind='trials';entityId=item.id;
   }else if((match=path.match(/^\/api\/plan\/proposals\/([\w-]+)\/(accept|dismiss)$/))&&method==='POST'){
    const proposal=get(document,'proposals',match[1]);objectRevision(proposal,data);
    if(proposal.status!=='proposed')fail(409,'Cette proposition a déjà été traitée.','proposal_resolved');
    if(match[2]==='dismiss'){stamp(proposal,{status:'dismissed',resolved_at:now(),payload:null,source_quote:null});entityKind='proposals';entityId=proposal.id}
    else{
     const payload=proposal.payload||{},edited={...payload,...(data.edited||{})};let chosen;
     if(proposal.kind==='goal'){
      chosen=addGoal(document,edited,'chosen_proposal');entityKind='goals';
      if(edited.step_text){const step=addStep(document,chosen.id,{text:edited.step_text,strategy_id:edited.strategy_id,measure:edited.measure},'chosen_proposal');if(data.create_trial===true)addTrial(document,step.id,data,'chosen_proposal')}
     }else if(proposal.kind==='step'){chosen=addStep(document,proposal.target_id,edited,'chosen_proposal');entityKind='steps';if(data.create_trial===true)addTrial(document,chosen.id,data,'chosen_proposal')}
     else if(proposal.kind==='trial'){chosen=addTrial(document,proposal.target_id,data,'chosen_proposal');entityKind='trials'}
     else if(proposal.kind==='result'){chosen=get(document,'trials',proposal.target_id);const old=JSON.stringify(chosen);stamp(chosen,trialFields({...edited,...data,status:'reported'},chosen));if(old!==JSON.stringify(chosen))changedContents.push(chosen.id);entityKind='trials'}
     else fail(400,'Proposition inconnue.');
     entityId=chosen.id;stamp(proposal,{status:'accepted',resolved_id:chosen.id,resolved_kind:entityKind,resolved_at:now(),payload:null,source_quote:null});
    }
   }else if((match=path.match(/^\/api\/plan\/(goals|steps|trials|proposals|preparations|instructions)\/([\w-]+)$/))&&['PATCH','DELETE'].includes(method)){
    const kind=match[1],item=get(document,kind,match[2]);objectRevision(item,data);entityKind=kind;entityId=item.id;
    if(method==='DELETE')removed=removeObject(document,kind,item.id);
    else{
     let fields,contentKeys;
     if(kind==='goals'){fields=goalFields(data,item);contentKeys=['title','why']}
     else if(kind==='steps'){fields=stepFields(data,item);contentKeys=['text','measure','strategy_id']}
     else if(kind==='trials'){fields=trialFields(data,item);contentKeys=['attempt','result_text','outcome','adverse_effect_text','occurred_on']}
     else if(kind==='instructions'){fields={text:text(data.text??item.text,1200,true),status:choice(data.status,['current','to_check','completed'],item.status)};contentKeys=['text']}
     else if(kind==='preparations'){fields={content:text(data.content,6000,true)};contentKeys=['content']}
     else fail(400,'Une proposition se modifie au moment de la choisir.');
     if(contentKeys.some(key=>fields[key]!==item[key]))changedContents.push(item.id);stamp(item,fields);
    }
   }else fail(404,'Route inconnue.','not_found');
  }
  const result=await save(env,state,document,{data,controls,receiptInfo,entityKind,entityId,removed,changedContents});
  return json({...visible(result.state),entity_kind:entityKind||undefined,entity_id:entityId||undefined,...(result.continuity_reset?{continuity_reset:true}:{}),...(result.removed?{removed:result.removed,retained:result.retained}:{})},method==='POST'?201:200);
 }
 async function exportData(env,userId){const state=await readChangeState(env,userId);return {change_plan:state.available?{...changeControl(state),updated_at:state.updated_at,limits:PLAN_LIMITS,...state.document}:null}}
 return {routes,exportData,preparation,visible};
}
