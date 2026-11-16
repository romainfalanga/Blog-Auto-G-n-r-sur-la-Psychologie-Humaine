// Private, owner-scoped transparency. This stores supplied inputs, never model reasoning.
const parse=(value,fallback={})=>{try{return JSON.parse(value)}catch{return fallback}};
const copy=value=>JSON.parse(JSON.stringify(value));
const LIMIT=128000;
const size=value=>new TextEncoder().encode(JSON.stringify(value)).length;
export const sourceLocation=(kind,key='')=>{
 if(key.startsWith('profile:'))return {edit_path:'/mon-portrait/',api_path:'/api/self-profile'};
 if(kind==='person')return {edit_path:'/mes-proches/',api_path:'/api/people'};
 if(kind==='event')return {edit_path:'/ma-ligne-de-vie/',api_path:'/api/life-events'};
 if(kind==='profile')return {edit_path:'/mon-portrait/',api_path:'/api/self-profile'};
 if(kind==='memory')return {edit_path:'/mon-compte/',api_path:'/api/memory'};
 if(kind==='review')return {edit_path:'/mes-bilans/',api_path:'/api/reviews'};
 if(kind==='observation')return {edit_path:'/mes-proches/',api_path:'/api/people'};
 if(['cause','outcome','resource','boundary','note'].includes(kind)||key.startsWith('cartography:'))return {edit_path:'/cartographie/',api_path:'/api/cartographie'};
 return {edit_path:'/mon-portrait/',api_path:'/api/portrait'};
};
export const indexLines=index=>index.map(x=>x.id+' '+x.kind+' '+String(x.label).slice(0,100)).join('\n');
export const INDEX_LEGEND='person=proche, portrait=synthèse, profile=présentation, event=moment, cause=vécu, hypothesis=piste, note=note, memory=précision confirmée, outcome=conséquence avec retour, resource=appui, boundary=réponse non, review=bilan IA';
export const USAGE_POLICY={risk_check:'Contrôle conditionnel d’une alerte proposée par le modèle : message actuel sans index, portrait ni mémoire. Un extrait récent limité n’est joint que pour une confirmation actuelle explicite d’un échange de sécurité.',analysis:'Le message, une présélection bornée de l’index (omissions comptées), les limites et antécédents de mémoire, le suivi daté et deux échanges récents orientent la sélection.',writing:'Le message, l’identité déclarée, la date de Paris, le suivi, les éléments récents de mémoire, six échanges et jusqu’à huit sources pertinentes (bilans compris) sont transmis. Les observations autorisées sont rappelées pour le proche concerné. Les réductions nécessaires au budget sont indiquées dans les entrées exactes.',review:'Le relecteur reçoit la réponse avec ses questions, les contraintes du tour, le suivi et des extraits des éléments utilisés.',details_max_items:8,details_max_chars:6500,detail_max_chars:1100,index_entry_max_chars:6000,index_max_chars:2600,index_max_items:48,index_label_max_chars:100,history_turns:6,memory_max_chars:4000,turn_base_tokens:30000,phase_max_attempts:3,memory:'La mémoire est mise à jour séparément, par lots de quatre à huit messages. Une panne laisse le lot en attente, sans inventer de faits ni avancer la couverture. Son extrait utilisé dans chaque tour est conservé dans la trace ; cet appel distinct n’est pas présenté comme un appel de rédaction.'};
export function appendTraceStage(trace,{stage,model,input,system,source_version,started_at,max_tokens,reservation,budget_key}){
 const result={...trace,trace_version:2,details:copy(trace.details||{source_version,values:[],stages:[]})},details=result.details;
 const id='s'+(details.stages.length+1),entry={id,stage,model,status:'running',attempt:details.stages.filter(s=>s.stage===stage).length+1,started_at,max_tokens,...(reservation?{reservation,budget_key}:{}),details_available:true};
 if(details.stages.length>=16){details.omitted_stages=(details.omitted_stages||0)+1;details.truncated=true;return {trace:result,id:null}}
 const before=details.values.length,seen=new Map(details.values.map((value,i)=>[JSON.stringify(value),i]));
 const ref=value=>{const encoded=JSON.stringify(value);if(seen.has(encoded))return seen.get(encoded);const index=details.values.length;details.values.push(copy(value));seen.set(encoded,index);return index};
 entry.system_ref=ref(system);entry.input_refs=Object.fromEntries(Object.entries(input).map(([key,value])=>[key,ref(value)]));details.stages.push(entry);
 if(size(details)>LIMIT){details.values.length=before;delete entry.system_ref;delete entry.input_refs;entry.details_available=false;entry.unavailable_reason='snapshot_limit';details.truncated=true}
 return {trace:result,id};
}
export function finishTraceStage(trace,id,{status,usage,error_code}){
 if(!trace.details||!id)return trace;
 const result={...trace,details:copy(trace.details)},entry=result.details.stages.find(s=>s.id===id);
 if(entry){entry.status=status;if(usage)entry.usage=usage;if(error_code)entry.error_code=error_code}
 return result;
}
export function traceSummary(trace){
 const {details,...summary}=trace;return {...summary,trace_version:trace.trace_version||1,details_available:Boolean(details?.stages?.some(s=>s.details_available)),...(details?.truncated?{details_truncated:true}:{}),...(!details?{unavailable_reason:'legacy_trace'}:{})};
}
export function traceDetail(row){
 const stored=typeof row.trace==='string'?parse(row.trace):row.trace,base={id:row.id||row.turn_id,...(row.status?{status:row.status}:{}),...(row.created_at?{created_at:row.created_at}:{}),...traceSummary(stored)};
 if(!stored.details)return {...base,snapshot:null,stages:[],unavailable_reason:'legacy_trace'};
 const {values=[],stages=[],source_version}=stored.details;
 const expanded=stages.map(({input_refs,system_ref,...stage})=>({...stage,...(stage.details_available?{input:Object.fromEntries(Object.entries(input_refs||{}).map(([key,index])=>[key,values[index]])),system:values[system_ref]}:{input:null,system:null})}));
 const writing=expanded.find(s=>['writing','rewrite'].includes(s.stage)&&s.input),analysis=expanded.find(s=>s.stage==='analysis'&&s.input),input=writing?.input||{};
 return {...base,source_version,stages:expanded,snapshot:{source_version,message:input.message??analysis?.input?.message??null,current_time:input.current_time??analysis?.input?.current_time??null,timezone:input.timezone??analysis?.input?.timezone??null,identity:input.identity??null,self_context:input.self_context??null,care:input.care??analysis?.input?.care??null,memory:input.memory??analysis?.input?.memory??null,history:input.history??null,index:analysis?.input?.index??null,index_total:analysis?.input?.index_total??null,index_omitted:analysis?.input?.index_omitted??null,details:input.details??null,reported_relationship_observations:input.reported_relationship_observations??null,observation_candidates:input.observation_candidates??null,cartography_targets:input.cartography_targets??null,budget:input.budget??null,context_omissions:input.context_omissions??[]},outcome:row.status==='running'?'pending':row.status==='failed'?'failed':stored.review?.fallback?'fallback':stored.review?.passed?'published':stored.error?'failed':'pending'};
}
