// « Mes symptômes » (plan v1.1, contrat docs/symptomes-contrat-2026-11-19.md) : moteur pur partagé.
// Aucune importation, aucune dépendance au Worker ni au DOM : chaque fonction reçoit le contenu
// (`SYMPTOMES`, généré par scripts/build_symptomes.py) en paramètre, pour être utilisée telle quelle
// par le Worker et par la page. Les scores et les états sont toujours recalculés ici à partir des
// réponses et de la version ; un score envoyé par le client n'est jamais lu.
// Ce sont des repères de questionnaire, jamais un diagnostic.

const INDEX=new WeakMap();
const NO_ANSWER=new Set(['nsp','pref']);
export const RESULT_STATES=Object.freeze(['au_dessus','en_dessous','repere_present','repere_absent','indetermine','non_explore','refuse']);
export const CROSSING_STATES=Object.freeze(['signale','non_signale','non_explore','refuse']);
export const PASSATION_KEY=/^[A-Za-z0-9_-]{8,120}$/;
const ITQ_RESULTS=new Set(['tspt','tspt_c','aucun','non_passe']);
// Kinds that make a module state; `contexte`, `grappe` (crossing only) and `entree` do not.
const STATE_KINDS=new Set(['instrument','repere','repere_site','orientation']);
const REPERE_KINDS=new Set(['repere','repere_site','orientation']);
const itemIds=instrument=>(instrument?.items||[]).map(item=>typeof item==='string'?item:item.id);
const isInt=v=>Number.isInteger(v);
const hasOwn=(o,k)=>Boolean(o)&&Object.prototype.hasOwnProperty.call(o,k);

// Identifiers, scales and the questions each published module accepts.
export function symptomsIndex(data){
 if(INDEX.has(data))return INDEX.get(data);
 const instruments=data.instruments||{},scales=data.scales||{},modules=new Map(),allowed=new Map(),instrumentsOf=new Map();
 const scale=(id,extra=[])=>{const s=scales[id];if(!s)throw Error('Échelle inconnue : '+id);const values=s.options.map(o=>o.value);return {id,values:new Set([...values,...extra]),numeric:values.every(isInt)}};
 const base=id=>instruments[id]?.derived_from||id;
 for(const m of data.modules||[]){
  modules.set(m.id,m);
  const accepted=new Map(),used=new Set();
  for(const q of m.questions||[])accepted.set(q.id,scale(q.scale));
  const addInstrument=id=>{
   if(!id)return;const inst=instruments[id];if(!inst)throw Error('Instrument inconnu : '+id);used.add(id);
   const parent=instruments[base(id)];
   for(const item of parent===inst?inst.items:inst.items.map(x=>parent.items.find(p=>(typeof p==='string'?p:p.id)===x)||x)){
    const itemId=typeof item==='string'?item:item.id,na=parent.not_applicable?.item===itemId?['na']:[];
    accepted.set(itemId,scale(item.scale||parent.scale,na));
   }
   if(parent.event?.when)accepted.set(parent.event.when.id,scale(parent.event.when.scale));
  };
  addInstrument(m.entry?.instrument);addInstrument(m.deepen?.instrument);
  for(const branch of m.deepen?.branches||[])addInstrument(branch.instrument);
  for(const id of [...used])if(instruments[id].derived_from)used.add(instruments[id].derived_from);
  // The universal safety question opens every passation of the first step.
  if(m.temps==='temps1'&&data.safety?.universal_question)accepted.set(data.safety.universal_question.id,scale(data.safety.universal_question.scale));
  allowed.set(m.id,accepted);instrumentsOf.set(m.id,[...used].sort());
 }
 const value={data,modules,allowed,instrumentsOf,base};
 INDEX.set(data,value);return value;
}

// ---- Scores ---------------------------------------------------------------------------------
// Sums are computed only when every item is answered (no imputation). PHQ-15: sleep and fatigue use
// the PHQ-9 scale recoded 3→2; item 1d may be « non concerné » ("na"), counted 0 (open point § 9).
function sumScore(inst,parent,answers){
 let total=0,na=false;
 for(const ref of inst.items){
  const id=typeof ref==='string'?ref:ref.id,item=(parent.items||[]).find(p=>(typeof p==='string'?p:p.id)===id);
  let v=answers?.[id];
  if(parent.not_applicable?.item===id&&v==='na'){na=true;v=parent.not_applicable.scored_as??0}
  else if(item&&typeof item==='object'&&item.recode){if(!isInt(v)||!hasOwn(item.recode,String(v)))return {score:null};v=item.recode[String(v)]}
  if(!isInt(v))return {score:null};
  total+=v;
 }
 return {score:total,na};
}
function itqScore(inst,answers){
 const scoring=inst.scoring,positive=scoring.item_positive??2;
 // Prerequisite not met (exposition_evenement ≠ oui, including « pref » or absent): the ITQ is not
 // offered; « non_passe » is a known value, not an unknown one.
 if(inst.prerequisite&&!inst.prerequisite.in.includes(answers?.[inst.prerequisite.q]))return {result:'non_passe',clusters:Object.fromEntries(Object.keys(scoring.clusters).map(c=>[c,false])),sums:null,complete:false};
 const clusters={},sums={};
 for(const [name,items] of Object.entries(scoring.clusters)){
  const values=items.map(id=>answers?.[id]);
  clusters[name]=values.some(v=>isInt(v)&&v>=positive)?true:values.every(isInt)?false:null;
  sums[name]=values.every(isInt)?values.reduce((a,b)=>a+b,0):null;
 }
 const every=names=>names.every(n=>clusters[n]===true)?true:names.some(n=>clusters[n]===false)?false:null;
 const complete=itemIds(inst).every(id=>isInt(answers?.[id]));
 if(!complete)return {result:null,clusters,sums:null,complete:false,tspt:every(scoring.tspt),pos:every(scoring.pos)};
 const tspt=every(scoring.tspt),pos=every(scoring.pos);
 return {result:tspt?(pos?'tspt_c':'tspt'):'aucun',clusters,sums,complete:true,tspt,pos};
}
// {phq2,phq9,…: number|null, itq:{result,clusters,sums}} for the instruments of `only` (all by default).
export function computeScores(answers,data,only=null){
 const instruments=data.instruments||{},out={},details={};
 for(const [id,inst] of Object.entries(instruments)){
  if(only&&!only.includes(id))continue;
  const parent=instruments[inst.derived_from]||inst;
  if(inst.scoring?.type==='algorithme')out[id]=itqScore(inst,answers);
  else{const r=sumScore(inst,parent,answers);out[id]=r.score;if(r.na)details[id]={not_applicable:true}}
 }
 if(Object.keys(details).length)out.details=details;
 return out;
}
export function bandFor(instrumentId,score,data){
 const inst=data.instruments?.[instrumentId];if(!inst?.bands||!isInt(score))return null;
 return inst.bands.find(b=>score>=b.min&&score<=b.max)||null;
}
export function scoreMax(instrumentId,scores,data){
 const inst=data.instruments?.[instrumentId];if(!inst?.scoring)return null;
 return inst.scoring.max-(scores?.details?.[instrumentId]?.not_applicable?2:0);
}

// ---- Expressions (three-valued: true, false, null = inconnu) --------------------------------
export function evaluate(expr,ctx){
 if(!expr||typeof expr!=='object')throw Error('Expression invalide');
 const {answers={},scores={}}=ctx||{};
 if(hasOwn(expr,'all')){let unknown=false;for(const e of expr.all){const v=evaluate(e,ctx);if(v===false)return false;if(v===null)unknown=true}return unknown?null:true}
 if(hasOwn(expr,'any')){let unknown=false;for(const e of expr.any){const v=evaluate(e,ctx);if(v===true)return true;if(v===null)unknown=true}return unknown?null:false}
 if(hasOwn(expr,'not')){const v=evaluate(expr.not,ctx);return v===null?null:!v}
 if(hasOwn(expr,'atleast')){const {n,of}=expr.atleast,values=of.map(e=>evaluate(e,ctx)),yes=values.filter(v=>v===true).length,unknown=values.filter(v=>v===null).length;return yes>=n?true:yes+unknown<n?false:null}
 if(hasOwn(expr,'q')){
  const v=answers[expr.q];if(v===undefined||v===null)return null;
  if(hasOwn(expr,'in'))return expr.in.includes(v)?true:NO_ANSWER.has(v)?null:false;
  if(hasOwn(expr,'gte'))return isInt(v)?v>=expr.gte:null;
 }
 if(hasOwn(expr,'score')){const s=scores[expr.score];return isInt(s)?s>=expr.gte:null}
 if(hasOwn(expr,'itq_result')){const r=scores.itq?.result;return r==null?null:r===expr.itq_result}
 if(hasOwn(expr,'itq_cluster')){const itq=scores.itq;if(!itq)return null;if(itq.result==='non_passe')return false;const c=itq.clusters?.[expr.itq_cluster];return c===undefined?null:c}
 throw Error('Expression inconnue : '+JSON.stringify(expr));
}
// Instruments read by an expression (score, ITQ result or cluster).
const instrumentsIn=(expr,out=new Set())=>{if(!expr||typeof expr!=='object')return out;if(typeof expr.score==='string')out.add(expr.score);if(hasOwn(expr,'itq_result')||hasOwn(expr,'itq_cluster'))out.add('itq');for(const v of Object.values(expr))if(Array.isArray(v))v.forEach(e=>instrumentsIn(e,out));else if(v&&typeof v==='object')instrumentsIn(v,out);return out};
const questionsOf=(expr,out=new Set())=>{if(!expr||typeof expr!=='object')return out;if(typeof expr.q==='string')out.add(expr.q);for(const v of Object.values(expr))if(Array.isArray(v))v.forEach(e=>questionsOf(e,out));else if(v&&typeof v==='object')questionsOf(v,out);return out};

// ---- Validation -----------------------------------------------------------------------------
// fail(status,message,code) must throw. Unknown module or question, value outside the scale,
// unknown content or instrument version: refused. null/absent = no answer (never counted 0).
export function validatePassation(input,data,fail,{partial=false}={}){
 const idx=symptomsIndex(data),invalid=(message,code='invalid_symptoms')=>fail(400,message,code);
 if(!input||typeof input!=='object'||Array.isArray(input))invalid('Passation invalide.');
 const moduleId=input.module;
 if(typeof moduleId!=='string'||!idx.modules.has(moduleId))invalid((data.unpublished_modules||[]).includes(moduleId)?'Ce module n’est pas encore publié.':'Module inconnu.',(data.unpublished_modules||[]).includes(moduleId)?'module_unpublished':'unknown_module');
 if(input.content_version!==data.version)invalid('Version du questionnaire inconnue ou périmée. Rechargez la page.','unknown_version');
 const accepted=idx.allowed.get(moduleId),answers={},raw=input.answers===undefined?{}:input.answers;
 if(!raw||typeof raw!=='object'||Array.isArray(raw))invalid('Réponses invalides.');
 if(Object.keys(raw).length>accepted.size)invalid('Trop de réponses.');
 for(const id of Object.keys(raw).sort()){
  const scale=accepted.get(id);if(!scale)invalid('Question inconnue pour ce module : '+id.slice(0,60)+'.','unknown_question');
  const v=raw[id];if(v===null)continue;
  if(!scale.values.has(v))invalid('Réponse hors de l’échelle pour '+id+'.','invalid_answer');
  answers[id]=v;
 }
 const declined=input.declined===undefined?false:input.declined;
 if(typeof declined!=='boolean')invalid('Indiquez si le module est refusé (true ou false).');
 if(partial&&declined)invalid('Un brouillon ne peut pas être refusé.');
 if(declined&&Object.keys(answers).length)invalid('Un module refusé ne contient pas de réponse.');
 if(!partial&&!declined&&!Object.keys(answers).length)invalid('Aucune réponse : refusez le module ou répondez à au moins une question.','empty_passation');
 // Instrument versions: each instrument with at least one answered item must carry its version.
 const versions=input.instrument_versions===undefined?{}:input.instrument_versions;
 if(!versions||typeof versions!=='object'||Array.isArray(versions))invalid('Versions des instruments invalides.');
 const moduleInstruments=idx.instrumentsOf.get(moduleId),cleanVersions={};
 for(const [id,version] of Object.entries(versions)){
  if(!moduleInstruments.includes(id))invalid('Instrument inconnu pour ce module.','unknown_instrument');
  if(version!==data.instruments[id].version)invalid('Version d’instrument inconnue ou périmée. Rechargez la page.','unknown_version');
  cleanVersions[id]=version;
 }
 for(const id of moduleInstruments){
  const inst=data.instruments[id];if(inst.derived_from)continue;
  const answered=itemIds(inst).some(item=>hasOwn(answers,item));
  const given=hasOwn(cleanVersions,id)||moduleInstruments.some(other=>data.instruments[other].derived_from===id&&hasOwn(cleanVersions,other));
  if(answered&&!given)invalid('Version de l’instrument '+inst.name+' manquante.','unknown_version');
 }
 return {module:moduleId,content_version:data.version,instrument_versions:cleanVersions,answers,declined};
}

// ---- Safety ---------------------------------------------------------------------------------
// Triggers present in a set of answers (blocking or soft), in the order of the content.
export function safetyTriggers(answers,data){
 return (data.safety?.triggers||[]).filter(t=>evaluate(t.when,{answers})===true).map(t=>({id:t.id,kind:t.kind,screen:t.screen}));
}
// What the screen shows: never a risk score. `age` (integer or null) adds Fil Santé Jeunes for 18–25.
export function safetyScreen(triggers,data,{age=null}={}){
 const display=data.safety.display,blocking=(triggers||[]).filter(t=>t.screen==='bloquant'),soft=(triggers||[]).filter(t=>t.screen==='doux');
 if(!blocking.length&&!soft.length)return null;
 const kinds=[...new Set((blocking.length?blocking:soft).map(t=>t.kind))],young=isInt(age)&&age>=18&&age<=25,show=new Set([...kinds,...(young?['jeune_adulte']:[])]);
 const byId=new Map(data.safety.resources.map(r=>[r.id,r])),order=display.blocking_screen.order;
 const resources=order.map(id=>byId.get(id)).filter(r=>r&&r.show.some(k=>show.has(k))).map(({id,number,label})=>({id,number,label}));
 if(!blocking.length){
  const texts=[...new Set(soft.map(t=>t.kind==='suicide'?display.soft_block.text:display.soft_block_by_kind?.[t.kind]).filter(Boolean))];
  return {screen:'doux',blocking:false,kinds,texts,resources};
 }
 const screen=display.blocking_screen,messages=kinds.map(k=>screen.messages?.[k]).filter(Boolean);
 return {screen:'bloquant',blocking:true,kinds,opening:screen.opening,always:screen.always,messages,resources,buttons:screen.buttons.slice(),delay_seconds:0,dismiss_by_outside_click:false,quick_exit:kinds.some(k=>(screen.quick_exit?.kinds||[]).includes(k))?screen.quick_exit.label:null,risk_score:false};
}

// ---- Age gate (date de naissance de Mon compte) --------------------------------------------
export function validBirthDate(value,today){
 return typeof value==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(value)&&value>='0001-01-01'&&Number.isFinite(Date.parse(value))&&new Date(value).toISOString().slice(0,10)===value&&value<=today;
}
export function ageOn(birthDate,today){
 if(!validBirthDate(birthDate,today))return null;
 return Number(today.slice(0,4))-Number(birthDate.slice(0,4))-(today.slice(5)<birthDate.slice(5)?1:0);
}
// {status:'adulte'|'mineur'|'inconnu', age, jeune_adulte}; a minor gets the text and resources of
// the gate, and nothing is recorded.
export function ageGate(birthDate,today,data){
 const gate=data.safety.adult_gate,age=ageOn(birthDate,today);
 if(age===null)return {status:'inconnu',age:null,jeune_adulte:false};
 if(age<gate.min_age){
  const byId=new Map(data.safety.resources.map(r=>[r.id,r]));
  return {status:'mineur',age,jeune_adulte:false,save:false,text:gate.if_minor.text,resources:gate.if_minor.resources.map(id=>byId.get(id)).filter(Boolean).map(({id,number,label})=>({id,number,label}))};
 }
 return {status:'adulte',age,jeune_adulte:age<=25};
}

// ---- Module result --------------------------------------------------------------------------
const instrumentReady=(id,scores)=>id==='itq'?['tspt','tspt_c','aucun'].includes(scores.itq?.result):isInt(scores[id]);
const DISPLAY_ABSENT={si_phq9_absent:'phq9',si_gad7_absent:'gad7'};
// Recomputes a passation of one module from its answers and the content `data` (its version).
// Returns {module, content_version, state, declined, entry, scores, bands, rules[], gene, …}.
export function computeModuleResult(moduleId,answers,data,{declined=false}={}){
 const idx=symptomsIndex(data),m=idx.modules.get(moduleId);if(!m)throw Error('Module inconnu : '+moduleId);
 const base={module:moduleId,content_version:data.version};
 if(declined)return {...base,state:'refuse',declined:true,entry:null,scores:{},bands:{},rules:[],positive_count:0,gene:null,gene_text:null,crossing_suspended:false,safety:[]};
 answers=answers||{};
 const used=idx.instrumentsOf.get(moduleId),scores=computeScores(answers,data,used),ctx={answers,scores};
 const bands={};for(const id of used){const b=bandFor(id,scores[id],data);if(b)bands[id]={id:b.id,label:b.label,max:scoreMax(id,scores,data)}}
 const entry=m.entry?.positive_if?evaluate(m.entry.positive_if,ctx):null;
 // « Je préfère ne pas répondre » to the entry (and no entry answer that opens the module): refused.
 const entryQuestions=[...questionsOf(m.entry?.positive_if)];
 const entryRefused=entry===null&&entryQuestions.some(q=>answers[q]==='pref');
 const entryInstrument=data.instruments[m.entry?.instrument],entryAnswers=new Set([...entryQuestions,...(m.entry?.questions||[]),...(entryInstrument?itemIds(entryInstrument):[])]);
 const rules=m.rules.map(r=>{
  let value=evaluate(r.when,ctx),forced=false;
  // Contract § 5 (deepen_on_request): a rule whose deepening was started (one answer to a question
  // or an instrument item beyond the entry) stays unknown, never forced negative.
  const started=entry===false&&value===null&&([...questionsOf(r.when)].some(q=>!entryAnswers.has(q)&&answers[q]!=null)||[...instrumentsIn(r.when)].some(id=>{const inst=data.instruments[id],parent=data.instruments[inst?.derived_from]||inst;return itemIds(parent).some(q=>!entryAnswers.has(q)&&answers[q]!=null)}));
  // Negative entry: the module stops; what was not asked counts as negative, never as unknown.
  if(entry===false&&value===null&&!started){value=false;forced=true}
  const result=value===true?'positif':value===false?'negatif':'indetermine';
  const out={id:r.id,kind:r.kind,result,subjects:(r.subjects||[]).slice(),...(r.instrument?{instrument:r.instrument}:{}),...(forced?{forced:true}:{})};
  if(r.instrument){const score=r.instrument==='itq'?null:scores[r.instrument];if(isInt(score))out.score=score;if(bands[r.instrument])out.level=bands[r.instrument];if(r.instrument==='itq'&&scores.itq?.result)out.itq_result=scores.itq.result}
  if(result==='positif'){const high=r.high_note_from&&isInt(scores[r.instrument])&&scores[r.instrument]>=r.high_note_from;out.note=high?'high_note':r.positive_note||'aucune'}
  return out;
 });
 // Display flags (the page shows the texts of displayed rules; the state text covers the rest).
 for(const [i,r] of m.rules.entries()){
  const out=rules[i];let show;
  if(r.kind==='grappe')show=false;
  else if(out.result==='indetermine'||out.forced&&out.result==='negatif')show=false;
  else if(r.display&&DISPLAY_ABSENT[r.display])show=!instrumentReady(DISPLAY_ABSENT[r.display],scores)&&(out.result==='positif'||isInt(scores[r.instrument]));
  else if(out.result==='positif')show=Boolean(r.result_positive);
  else if(r.display==='si_positif'||!r.result_negative)show=false;
  else if(r.display_negative==='si_itq_aucun')show=scores.itq?.result==='aucun';
  else if(r.kind==='instrument')show=instrumentReady(r.instrument,scores);
  else show=true;
  out.display=show;
 }
 const relevant=rules.filter(r=>STATE_KINDS.has(r.kind));
 let state;
 if(entryRefused)state='refuse';
 else if(relevant.some(r=>r.kind==='instrument'&&r.result==='positif'))state='au_dessus';
 else if(relevant.some(r=>REPERE_KINDS.has(r.kind)&&r.result==='positif'))state='repere_present';
 else if(rules.some(r=>r.kind==='instrument'&&r.result==='negatif'&&r.display))state='en_dessous';
 // A deepening started after a negative entry and left incomplete: unknown, not « below the entry ».
 else if(relevant.some(r=>r.result==='indetermine')||entry===null||rules.some(r=>r.kind==='entree'&&r.result==='positif'))state='indetermine';
 else if(rules.some(r=>r.kind==='entree'&&r.result==='negatif'&&r.display))state='en_dessous';
 else state='repere_absent';
 // A module with its own overall negative sentence shows it instead of each negative text.
 const moduleNegative=state==='repere_absent'&&Boolean(m.module_negative);
 if(moduleNegative)for(const r of rules)if(r.result==='negatif'&&r.kind!=='instrument'&&r.kind!=='entree')r.display=false;
 const positive_count=rules.filter(r=>r.result==='positif'&&r.display&&r.kind!=='contexte').length;
 const geneQuestion=(m.questions||[]).find(q=>q.role==='gene'),gene=geneQuestion?answers[geneQuestion.id]??null:null;
 const except=data.fixed_texts?.gene_absente_trouble_except||[];
 const gene_text=gene==='nettement'?'gene_nettement':gene==='non'?(except.includes(moduleId)?'gene_absente':'gene_absente_trouble'):null;
 const crossing_suspended=Boolean(m.crossing_suspended_if&&evaluate(m.crossing_suspended_if.when,ctx)===true);
 const publicScores={};for(const id of used){if(id==='itq'){const s=scores.itq;publicScores.itq={result:s.result,clusters:s.clusters,sums:s.sums}}else publicScores[id]=scores[id]}
 return {...base,state,declined:false,entry:entry===true?'positif':entry===false?'negatif':'indetermine',scores:publicScores,bands,rules,positive_count,several:positive_count>=2,module_negative:moduleNegative,gene,gene_text,crossing_suspended,safety:safetyTriggers(answers,data)};
}

// ---- Delays ---------------------------------------------------------------------------------
// retest_days of the module (null = « vie entière » questions, never asked again by default).
export function retestInfo(moduleId,lastCompletedAt,now,data){
 const m=symptomsIndex(data).modules.get(moduleId),days=m?.retest_days??null;
 if(days===null)return {days:null,never:true,next_at:null,due:lastCompletedAt==null};
 if(lastCompletedAt==null)return {days,never:false,next_at:null,due:true};
 const next=lastCompletedAt+days*86400;
 return {days,never:false,next_at:next,due:now>=next};
}
// A new passation before the end of the previous period is allowed and shows overlap_note.
export function overlaps(moduleId,lastCompletedAt,now,data){
 if(lastCompletedAt==null)return false;
 const info=retestInfo(moduleId,lastCompletedAt,now,data);
 return info.never||!info.due;
}

// ---- Crossing with Mon enfance (4 states) ---------------------------------------------------
// Locale-independent order shared by the Worker and the page: accents and case folded, then binary.
export const foldLabel=value=>String(value||'').normalize('NFD').replace(/[̀-ͯ]/g,'').toLowerCase();
export const compareLabels=(a,b)=>{const x=foldLabel(a),y=foldLabel(b);return x<y?-1:x>y?1:a<b?-1:a>b?1:0};
// latest: {moduleId: stored result of the latest passation (computeModuleResult)}.
// childhood: null when Mon enfance is not saved, else {present:[ids], dismissed:[ids]} where
// present = sections top/other, empirical link (associated_count > 0), feedback not « non ».
// labels: {subjectId: label} (Mon enfance wording); crossing.display_labels has priority.
export function crossWithChildhood(latest,childhood,data,{labels={}}={}){
 const idx=symptomsIndex(data),present=new Set(childhood?.present||[]),dismissed=new Set(childhood?.dismissed||[]);
 const suspended=[];
 for(const m of data.modules||[]){const r=latest?.[m.id];if(r&&r.state!=='refuse'&&r.crossing_suspended)suspended.push({module:m.id,text:m.crossing_suspended_if?.text||''})}
 const off=new Set(suspended.map(s=>s.module)),display=data.crossing?.display_labels||{};
 const subjects=[];
 for(const [id,list] of Object.entries(data.subject_modules||{})){
  const mods=list.filter(m=>idx.modules.has(m)&&!off.has(m));if(!mods.length)continue;
  const per=mods.map(m=>{
   const r=latest?.[m];if(!r)return 'absent';
   if(r.state==='refuse'||r.declined)return 'refuse';
   const rules=(r.rules||[]).filter(x=>(x.subjects||[]).includes(id));
   if(rules.some(x=>x.result==='positif'))return 'signale';
   // Contract § 8: a module « indetermine » leaves its subjects unexplored; so does a rule read on an
   // ITQ that was not offered (exposition ≠ oui), e.g. ten_hypervigilance or icd_6b40.
   if(r.state==='non_explore'||r.state==='indetermine'||!rules.length||rules.some(x=>x.result==='indetermine'||x.instrument==='itq'&&x.itq_result==='non_passe'))return 'incomplet';
   return 'negatif';
  });
  const state=per.includes('signale')?'signale':per.every(p=>p==='refuse')?'refuse':per.every(p=>p==='negatif')?'non_signale':'non_explore';
  const signaling=mods.filter((m,i)=>per[i]==='signale'),first=signaling[0]||mods[0];
  const side=!childhood?'inconnu':dismissed.has(id)?'pas_moi':present.has(id)?'present':'absent';
  subjects.push({id,label:display[id]||labels[id]||id,state,childhood:side,modules:signaling,order:idx.modules.get(first).order});
 }
 subjects.sort((a,b)=>a.order-b.order||compareLabels(a.label,b.label)||(a.id<b.id?-1:a.id>b.id?1:0));
 const item=s=>({id:s.id,label:s.label,modules:s.modules});
 return {
  childhood_available:Boolean(childhood),
  lists:{
   deux:subjects.filter(s=>s.state==='signale'&&s.childhood==='present').map(item),
   symptomes:subjects.filter(s=>s.state==='signale'&&s.childhood!=='present').map(item),
   enfance:subjects.filter(s=>s.state==='non_signale'&&s.childhood==='present').map(item)
  },
  subjects:subjects.map(({order,...s})=>s),
  suspended,
  show_counts:false
 };
}
