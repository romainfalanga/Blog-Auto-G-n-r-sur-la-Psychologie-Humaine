// Conversation integration: proposals are drafts, never user choices.
import {readChangeState,compactChangeContext,privateCareInstructions,dueChangeTrial,changeGuard,changeWrite,mutationGuard,changeArtifactStatements,PLAN_LIMITS} from './change-plan.mjs';
import {CHANGE_STRATEGIES} from './change-strategies.generated.mjs';
import {selectChangeCandidates,loadStrategyDetails,validateStrategySelection,changeReviewCriteria,normalizeChangeReview,deterministicChangeReview,CHANGE_SYSTEM_RULES,CHANGE_REVIEW_CRITERIA} from './change-strategies.mjs';
import {declinesRetention} from './people-observations.mjs';
const object=properties=>({type:'object',properties,required:Object.keys(properties),additionalProperties:false});
const text={type:'string'},nullable={type:['string','null']};
const copy=value=>JSON.parse(JSON.stringify(value));
export {CHANGE_SYSTEM_RULES,CHANGE_REVIEW_CRITERIA};
export const CHANGE_ANALYSIS_FIELDS={strategy_ids:{type:'array',maxItems:2,items:text},change_signal:object({kind:{type:'string',enum:['none','goal_stated','trial_requested','trial_reported','obstacle','stop_requested']},target_id:nullable,quote:text})};
export const CHANGE_DRAFT_FIELDS={change_proposals:{type:'array',maxItems:1,items:object({kind:{type:'string',enum:['goal','step','trial','result']},target_id:nullable,quote:text,payload:object({title:text,why:text,step_text:text,text,measure:text,strategy_id:nullable,result_text:text})})},followup:{...object({trial_id:text,quote:text}),type:['object','null']}};
export const CHANGE_REVIEW_FIELD={type:'array',maxItems:8,items:object({id:{type:'string',enum:Array.from({length:8},(_,i)=>'C-R'+(i+1))},verdict:{type:'string',enum:['pass','fail','not_applicable','not_checked']}})};
export const CHANGE_DRAFT_RULES='change_proposals : zéro ou une proposition, jamais un choix adopté. kind goal (title/why/step_text/strategy_id), step (target_id objectif, text/measure/strategy_id), trial (target_id étape), result (target_id essai, result_text citation exacte du message). quote exacte du message actuel, 240 caractères au plus ; autres champs inutilisés vides. Aucun compteur, état, date, autorisation ou adoption décidés par le modèle. Une ambiguïté ne crée rien. followup=null sauf invitation facultative au seul due_trial admissible ; donner trial_id et quote exacte de cette invitation dans la réponse finale (300 caractères au plus). Une réécriture doit fournir son propre followup. Aucun enregistrement ou engagement annoncé.';
export async function loadChangeContext(env,userId,message,now,options={}){
 const state=await readChangeState(env,userId),plan=compactChangeContext(state,message,now,options),instructions=privateCareInstructions(state,plan);
 if(instructions.requires_session_check)plan.instruction_review_required=true;
 return {state,context:{plan,private_care_instructions:instructions.items||[],instructions_omitted:instructions.omitted||0,strategies:[],due_trial:plan.due_trial||null},catalogue_version:CHANGE_STRATEGIES.version};
}
export function changeBlocked(analysis,plan={}){
 return !plan.available||!plan.proposals_enabled||plan.instruction_review_required||plan.exploration_reconsent_required||analysis.risk!=='none'||['listen','stabilize','repair','safety'].includes(analysis.posture)||analysis.need==='listen'||(analysis.constraints||[]).some(x=>['no_advice','no_exploration','respect_closed_topics'].includes(x));
}
export function changeCandidates(message,context,analysis,care){return changeBlocked(analysis,context.plan)?[]:selectChangeCandidates({message,plan:context.plan,care,analysis})}
export function analyzeChange(ai,{message,analysis,context,candidates}){
 const signal=ai.change_signal,ids=new Set([...(context.plan.goals||[]).map(x=>x.id),context.plan.step?.id,context.plan.trial?.id,context.plan.due_trial?.id].filter(Boolean));
 const valid=signal&&['goal_stated','trial_requested','trial_reported','obstacle','stop_requested'].includes(signal.kind)&&typeof signal.quote==='string'&&signal.quote.trim().length>=2&&signal.quote.length<=240&&message.includes(signal.quote)&&(!signal.target_id||ids.has(signal.target_id));
 return {strategy_ids:changeBlocked(analysis,context.plan)?[]:validateStrategySelection(ai.strategy_ids||[],candidates,{analysis,message,plan:context.plan}),change_signal:valid?{kind:signal.kind,target_id:signal.target_id||null,quote:signal.quote}:{kind:'none'}};
}
export function writingChange(context,analysis){
 const blocked=changeBlocked(analysis,context.plan),plan=copy(context.plan);
 if(blocked||analysis.constraints?.includes('no_questions')||analysis.change_signal?.kind==='trial_reported'||analysis.change_signal?.kind==='obstacle')plan.due_trial=null;
 return {...context,plan,due_trial:plan.due_trial||null,strategies:blocked?[]:loadStrategyDetails(analysis.strategy_ids||[])};
}
export function changeInputRefs(input){
 const context=input.change||{},refs=[...(context.plan?.dependencies||[]),...(context.private_care_instructions||[]).map(x=>({kind:'instructions',id:x.id,revision:x.revision}))];
 return refs.filter((x,i)=>typeof x.id==='string'&&refs.findIndex(y=>y.kind===x.kind&&y.id===x.id)===i);
}
const bounded=(value,max)=>typeof value==='string'&&value.length<=max&&!/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/.test(value)?value.trim():null;
export function prepareChangeDraft(output,{message,analysis,context}){
 const result={change_proposals:[],followup:null,change_errors:[]},blocked=changeBlocked(analysis,context.plan),proposed=output.change_proposals||[];
 if(!Array.isArray(proposed)||proposed.length>1){result.change_errors.push('C-R2');return result}
 for(const proposal of proposed){
  const quote=bounded(proposal?.quote,240),payload=proposal?.payload||{},kind=proposal?.kind,target=proposal?.target_id||null,p={};
  if(blocked||declinesRetention(message)||!quote||!message.includes(quote)){result.change_errors.push('C-R2');continue}
  const limits=kind==='goal'?{title:240,why:800,step_text:800}:kind==='step'?{text:800,measure:300}:kind==='result'?{result_text:1200}:kind==='trial'?{}:null;
  if(!limits){result.change_errors.push('C-R2');continue}
  let invalid=false;for(const [key,max] of Object.entries(limits)){const value=bounded(payload[key]??'',max);if(value===null)invalid=true;else if(value)p[key]=value}
  if(kind==='goal'&&(!p.title||target)||kind==='step'&&(!p.text||!context.plan.goals?.some(x=>x.id===target))||kind==='trial'&&context.plan.step?.id!==target||kind==='result'&&(!p.result_text||!message.includes(p.result_text)||![context.plan.trial?.id,context.plan.due_trial?.id].includes(target)))invalid=true;
  if(payload.strategy_id){if(!context.strategies.some(x=>x.id===payload.strategy_id))invalid=true;else p.strategy_id=payload.strategy_id}
  if(kind==='result')Object.assign(p,{attempt:'uncertain',outcome:null});
  if(invalid){result.change_errors.push('C-R2');continue}
  result.change_proposals.push({kind,target_id:target,quote,payload:p});
 }
 const followup=output.followup;
 if(followup){const quote=bounded(followup.quote,300);if(blocked||analysis.constraints?.includes('no_questions')||!context.due_trial||followup.trial_id!==context.due_trial.id||!quote||!output.content.includes(quote))result.change_errors.push('C-R8');else result.followup={trial_id:followup.trial_id,quote}}
 return result;
}
export function changeReviewInput(analysis,draft,context){return changeReviewCriteria({analysis,draft,changeContext:context})}
export function reviewChange(verdicts,{analysis,draft,message,context,candidates=[]}){
 const criteria=changeReviewInput(analysis,draft,context),base=normalizeChangeReview(verdicts,criteria),deterministic=deterministicChangeReview({analysis,draft,message,plan:context.plan,candidates,changeContext:context});
 const failed=[...new Set([...(base.failed_ids||[]),...(deterministic.failed_ids||[]),...(draft.change_errors||[])])];
 return {...base,failed_ids:failed,passed:failed.length?false:base.passed};
}
export async function changePublication(env,userId,{context,draft,analysis,message,turnId,now,random,guard,guardArgs}){
 const plan=context?.plan;if(!plan?.available)return {statements:[],guard,guardArgs,proposalIds:[]};
 const state=await readChangeState(env,userId),cas=changeGuard(userId,plan);
 if(!state.available||state.revision!==plan.revision||state.epoch!==plan.epoch||state.context_generation!==plan.context_generation)throw Object.assign(Error('Plan changed'),{code:'context_changed'});
 const verified=prepareChangeDraft({...draft,content:draft.content},{message,analysis,context});
 if(verified.change_errors.length)throw Object.assign(Error('Invalid change draft'),{code:'invalid_response'});
 const document=copy(state.document),ids=[],refs=[];
 for(const proposal of verified.change_proposals){
  if(document.proposals.filter(x=>x.status==='proposed').length>=PLAN_LIMITS.proposals)throw Object.assign(Error('Proposal limit'),{code:'plan_limit'});
  const id='proposal_'+random();document.proposals.push({id,revision:1,kind:proposal.kind,target_id:proposal.target_id,payload:proposal.payload,status:'proposed',source_turn_id:turnId,source_quote:proposal.quote,created_at:now,updated_at:now});ids.push(id);refs.push({kind:'proposals',id,revision:1});
 }
 let changed=ids.length>0;
 if(verified.followup){const due=dueChangeTrial(state,message,now,{turnId,previousTurnId:context.previous_turn_id});if(!due||due.id!==verified.followup.trial_id)throw Object.assign(Error('Followup changed'),{code:'context_changed'});const trial=document.trials.find(x=>x.id===due.id);Object.assign(trial,{reminder_count:trial.reminder_count+1,last_reminder_at:now,last_reminder_turn_id:turnId,revision:trial.revision+1,updated_at:now});changed=true}
 const combined=guard+' AND '+cas.sql,args=[...guardArgs,...cas.args];
 if(!changed)return {statements:[],guard:combined,guardArgs:args,proposalIds:[]};
 const mutation='turn:'+turnId,m=mutationGuard(userId,mutation),after=guard+' AND '+m.sql,afterArgs=[...guardArgs,...m.args],statements=[changeWrite(env,state,document,{now,mutation,guard:combined,guardArgs:args})];
 statements.push(...changeArtifactStatements(env,userId,'turn',turnId,refs,state.context_generation,after,afterArgs));
 return {statements,guard:after,guardArgs:afterArgs,proposalIds:ids};
}
export const changeMethods=()=>({version:CHANGE_STRATEGIES.version,common_rules:CHANGE_STRATEGIES.common_rules,methods:CHANGE_STRATEGIES.strategies,review_criteria:CHANGE_REVIEW_CRITERIA});
