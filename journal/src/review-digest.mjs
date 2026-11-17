// Lot A5-P1 (plan v5.1, audit v5 F): structured reviews, their separate check, the review
// thread and the person's feedback. Pure functions, except the migration probe.
// A digest is never shown as is: each element is anchored in the person's exact words
// (quote-guard) or dropped, and it is rendered as readable prose for the person.
import {anchorQuote,locateQuote,similarQuotes,foldText} from './quote-guard.mjs';
import {riskSignal} from './chat-safety.mjs';
import {VERIFIED_NUMBERS} from './clinical-signals.mjs';

export const GOAL_STATUS={new:'nouveau',ongoing:'en cours',progressing:'avance',reached:'atteint',paused:'en pause',dropped:'abandonné'};
export const TRIED_RESULT={helped:'a aidé',no_effect:'sans effet',hard:'difficile',not_yet:'pas encore essayé'};
export const FORMULATION_LABELS={holds:'se confirme',revised:'révisé',new:'nouveau'};
export const FEEDBACK_VERDICTS={resembles:'Ce bilan me ressemble',not_really:'Pas vraiment',to_correct:'À corriger'};
export const FEEDBACK_TEXT_MAX=600,DIGEST_MAX_CHARS=2400,THREAD_TURN_CHARS=1200,THREAD_REVIEW_CHARS=3000,THREAD_REVIEWS=12;
const s={type:'string'},list=items=>({type:'array',items});
const object=properties=>({type:'object',properties,required:Object.keys(properties),additionalProperties:false});
const anchored=object({text:s,quote:s});
export const REVIEW_DIGEST_SCHEMA=object({
 formulation:object({holds:list(anchored),revised:list(anchored),new:list(anchored)}),
 goals:list(object({key:s,words:s,status:{type:'string',enum:Object.keys(GOAL_STATUS)},quote:s,next_step:s})),
 tried:list(object({what:s,result:{type:'string',enum:Object.keys(TRIED_RESULT)},quote:s})),
 alliance:object({text:s,quote:s}),
 open_questions:list(s),next_focus:s,safety:s
});
export const REVIEW_OUTPUT_SCHEMA=object({content:s,digest:REVIEW_DIGEST_SCHEMA});
const yesNo={type:'string',enum:['yes','no']},yesNoNa={type:'string',enum:['yes','no','na']};
export const REVIEW_CHECK_CRITERIA={no_diagnosis:'Aucun diagnostic ni nom de trouble présenté comme acquis, pour la personne ou un proche.',quotes_faithful:'Les citations et « vous écrivez que… » reprennent des mots présents dans le contexte, sans invention ni déformation.',safety_resources:'Si safety_signal est présent : le 3114 (ou 17/112 pour une violence) et une aide humaine figurent ; sinon na.',hypotheses_open:'Les interprétations restent des hypothèses révisables, sans score, pourcentage ni certitude.',no_saved_claim:'Aucun enregistrement annoncé, aucune consigne professionnelle inventée.',sources_honest:'Les sources manquantes ou minces sont dites sans reproche.'};
export const REVIEW_CHECK_SCHEMA=object({criteria:object(Object.fromEntries(Object.keys(REVIEW_CHECK_CRITERIA).map(key=>[key,key==='safety_resources'?yesNoNa:yesNo]))),reasons:list(s)});
// Blocking criteria (plan v5.1): no diagnosis, no invented quote, safety resources when a signal is present.
export const BLOCKING_CHECKS=['diagnosis','invented_quote','safety_missing','unverified_number','no_diagnosis','quotes_faithful','safety_resources'];
const clip=(value,max)=>typeof value==='string'?value.replace(/\s+/g,' ').trim().slice(0,max):'';
const parse=(value,fallback)=>{if(value&&typeof value==='object')return value;try{const out=JSON.parse(value);return out&&typeof out==='object'?out:fallback}catch{return fallback}};
const headTail=(value,max)=>{const text=String(value??'');if(text.length<=max)return text;const gap=' […] ',head=Math.ceil((max-gap.length)*.6);return text.slice(0,head)+gap+text.slice(text.length-(max-gap.length-head))};
const slug=value=>foldText(value).replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'').slice(0,40)||'objectif';
export const parseDigest=value=>{const out=parse(value,{});return out&&typeof out==='object'&&!Array.isArray(out)?out:{}};
export const emptyDigest=digest=>!digest||!['holds','revised','new'].some(k=>digest.formulation?.[k]?.length)&&!digest.goals?.length&&!digest.tried?.length&&!digest.alliance?.text&&!digest.open_questions?.length&&!digest.next_focus;
export function parseFeedback(value){const out=parse(value,null);return out&&FEEDBACK_VERDICTS[out.verdict]?{verdict:out.verdict,text:clip(out.text,FEEDBACK_TEXT_MAX),at:Number(out.at)||null}:null}

// The person's own words supplied to the review writer: the only place a digest quote may come from.
export function reviewCorpus(data){
 const parts=[];
 for(const source of data?.sources||[])if(['profile','person','event','memory'].includes(source.type)&&typeof source.text==='string')parts.push(source.text);
 for(const turn of data?.conversation_since||[])if(typeof turn.vous==='string')parts.push(turn.vous);
 for(const note of data?.childhood_cartography?.person_notes||[])if(typeof note?.text==='string')parts.push(note.text);
 for(const item of data?.people_observations||[])if(typeof item?.quote==='string')parts.push(item.quote);
 if(typeof data?.previous_feedback?.text==='string')parts.push(data.previous_feedback.text);
 return parts.join('\n\n');
}
// Every string supplied to the writer: a quotation in the published text must come from it.
export function suppliedText(data){
 const out=[];const walk=v=>{if(typeof v==='string')out.push(v);else if(Array.isArray(v))v.forEach(walk);else if(v&&typeof v==='object')Object.values(v).forEach(walk)};walk(data);return out.join('\n\n');
}
// A deterministic safety signal for the review: the exchange since the last review, the dated
// safety memory, or a recognized consequence about suicide in the map.
export function reviewSafetySignal(data){
 const texts=[...(data?.conversation_since||[]).map(t=>t.vous),...(data?.memory?.safety||[]),...(data?.memory?.safety_history||[])].filter(x=>typeof x==='string');
 let violence=false,self=false;
 for(const text of texts){const signal=riskSignal(text);if(signal?.signal==='violence')violence=true;else if(signal)self=true}
 if(data?.memory?.safety_seen)self=true;
 const map=data?.childhood_cartography;if(map&&(map.suicide_recognized||(map.recognized||[]).some(x=>/suicid/i.test(String(x?.id||'')+' '+String(x?.label||'')))))self=true;
 return self?'self_harm':violence?'violence':null;
}
// Each element keeps an exact, substantial quote of the person, or leaves the digest.
// Negation is allowed where the element itself says something stopped or did not help.
export function validateDigest(raw,corpus,{knownGoals=[]}={}){
 const input=raw&&typeof raw==='object'?raw:{};let dropped=0;
 const anchor=(quote,allowNegated=false)=>{const result=anchorQuote(corpus,clip(quote,300),{allowNegated});return result.ok?clip(result.text,300):null};
 const items=(values,max,make)=>{const out=[];for(const value of Array.isArray(values)?values:[]){if(out.length>=max){dropped++;continue}const item=make(value);if(item)out.push(item);else dropped++}return out};
 const formulation={};
 for(const key of ['holds','revised','new'])formulation[key]=items(input.formulation?.[key],3,v=>{const text=clip(v?.text,300),quote=anchor(v?.quote,key==='revised');return text&&quote?{text,quote}:null});
 const known=knownGoals.filter(g=>g&&typeof g.key==='string');
 const goals=items(input.goals,5,v=>{
  const words=clip(v?.words,200),status=GOAL_STATUS[v?.status]?v.status:null,quote=anchor(v?.quote,['paused','dropped'].includes(v?.status));if(!words||!status||!quote)return null;
  // A goal is either a known one (same key or same words) or explicitly new.
  let key=/^[a-z0-9-]{2,40}$/.test(String(v.key||''))?v.key:slug(words);const match=known.find(g=>g.key===key)||known.find(g=>similarQuotes(g.words,words,0.6));
  if(match)key=match.key;
  return {key,words,status:match||status==='new'?status:'new',quote,next_step:clip(v?.next_step,200)};
 });
 const keys=new Set();for(const goal of goals){while(keys.has(goal.key))goal.key=(goal.key+'-2').slice(-40);keys.add(goal.key)}
 const tried=items(input.tried,5,v=>{const what=clip(v?.what,200),result=TRIED_RESULT[v?.result]?v.result:null,quote=anchor(v?.quote,['no_effect','hard','not_yet'].includes(v?.result));return what&&result&&quote?{what,result,quote}:null});
 const allianceQuote=input.alliance?.text?anchor(input.alliance.quote):null;if(input.alliance?.text&&!allianceQuote)dropped++;
 const digest={v:1,formulation,goals,tried,alliance:allianceQuote?{text:clip(input.alliance.text,300),quote:allianceQuote}:null,open_questions:(Array.isArray(input.open_questions)?input.open_questions:[]).map(q=>clip(q,220)).filter(Boolean).slice(0,3),next_focus:clip(input.next_focus,300),safety:clip(input.safety,300),dropped};
 // Bounded: the least central elements leave first, and are counted.
 const size=()=>JSON.stringify(digest).length,shrink=[()=>digest.formulation.new.pop(),()=>digest.tried.pop(),()=>digest.formulation.holds.length>1&&digest.formulation.holds.pop(),()=>digest.formulation.revised.length>1&&digest.formulation.revised.pop(),()=>digest.goals.length>2&&digest.goals.pop(),()=>digest.open_questions.length>1&&digest.open_questions.pop()];
 while(size()>DIGEST_MAX_CHARS){const step=shrink.find(fn=>fn());if(!step)break;digest.dropped++}
 return digest;
}
const sentence=value=>{const text=String(value||'').trim();return text&&!/[.!?…]$/.test(text)?text+'.':text};
const said=quote=>quote?' Vous avez écrit : « '+quote+' ».':'';
// The digest as readable paragraphs (Mes bilans, Paramètres, export).
export function digestProse(value){
 const d=parseDigest(value);if(emptyDigest(d))return '';
 const out=[],f=d.formulation||{};
 if(f.holds?.length)out.push('Ce qui se confirme. '+f.holds.map(x=>sentence(x.text)+said(x.quote)).join(' '));
 if(f.revised?.length)out.push('Ce qui est révisé. '+f.revised.map(x=>sentence(x.text)+said(x.quote)).join(' '));
 if(f.new?.length)out.push('Nouvelle piste, à vérifier avec vous. '+f.new.map(x=>sentence(x.text)+said(x.quote)).join(' '));
 if(d.goals?.length)out.push('Vos objectifs suivis. '+d.goals.map(g=>'« '+g.words+' » : '+GOAL_STATUS[g.status]+'.'+(g.next_step?' Prochaine étape : '+sentence(g.next_step):'')).join(' '));
 if(d.tried?.length)out.push('Ce que vous avez essayé. '+d.tried.map(t=>sentence(t.what.replace(/[.!?]$/,'')+' : '+TRIED_RESULT[t.result])+said(t.quote)).join(' '));
 if(d.alliance?.text)out.push('Votre retour sur l’accompagnement. '+sentence(d.alliance.text)+said(d.alliance.quote));
 if(d.open_questions?.length)out.push('Questions ouvertes. '+d.open_questions.join(' '));
 if(d.next_focus)out.push('Prochaine étape. '+sentence(d.next_focus));
 if(d.safety)out.push('Sécurité. '+sentence(d.safety));
 return out.join('\n\n');
}
const day=seconds=>Number(seconds)>0?new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Paris',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(Number(seconds)*1000)):null;
// The review thread: a deterministic, bounded memory derived from the ready reviews (newest first).
// Deleting a review or changing a feedback changes it at the next read: nothing is stored.
// Without any digest (reviews written before 0023), the latest review travels as a dated excerpt.
export function reviewThread(rows,{maxChars=THREAD_TURN_CHARS}={}){
 const reviews=(rows||[]).filter(r=>r&&r.id).slice(0,THREAD_REVIEWS);if(!reviews.length)return null;
 const digests=reviews.map(r=>({row:r,d:parseDigest(r.digest),feedback:parseFeedback(r.feedback),date:day(r.completed_at)}));
 const last=digests[0],lastFull=!emptyDigest(last.d);
 const thread={reviews:reviews.length,last:{id:last.row.id,date:last.date,kind:last.row.kind==='initial'?'analyse initiale':'point d’étape'}};
 if(lastFull){if(last.d.next_focus)thread.last.next_focus=last.d.next_focus;const lines=[...(last.d.formulation?.holds||[]).map(x=>'se confirme : '+x.text),...(last.d.formulation?.revised||[]).map(x=>'révisé : '+x.text),...(last.d.formulation?.new||[]).map(x=>'nouveau : '+x.text)];if(lines.length)thread.last.formulation=lines.slice(0,2)}
 else if(typeof last.row.content==='string'&&last.row.content){const text=headTail(last.row.content,900);thread.last.text=text;if(text.length<last.row.content.length)thread.last.excerpt=true}
 // Goals: the dated sequence of states, oldest first; the latest evidence and next step.
 const goals=new Map();
 digests.slice().reverse().forEach(({d,date},rank)=>{for(const g of d.goals||[]){const entry=goals.get(g.key)||{key:g.key,words:g.words,states:[],seen:0};entry.words=g.words;entry.states.push((date||'?')+' '+GOAL_STATUS[g.status]);entry.status=g.status;entry.evidence=g.quote;entry.next_step=g.next_step||'';entry.seen=digests.length-1-rank;goals.set(g.key,entry)}});
 thread.goals=[...goals.values()].sort((a,b)=>a.seen-b.seen).map(g=>({key:g.key,words:g.words,path:g.states.join(' → '),status:g.status,...(g.evidence?{evidence:g.evidence}:{}),...(g.next_step?{next_step:g.next_step}:{}),age:g.seen}));
 const merged=(key)=>{const out=[];for(const {d} of digests)for(const x of d.formulation?.[key]||[]){const same=out.find(y=>similarQuotes(y.text,x.text,0.6));if(same)same.count++;else out.push({text:x.text,count:1})}return out};
 const holds=merged('holds');thread.confirmed=holds.filter(x=>x.count>=2).map(x=>x.text+' (depuis '+x.count+' bilans)').concat(holds.filter(x=>x.count<2).map(x=>x.text)).slice(0,4);
 thread.revised=merged('revised').map(x=>x.text).slice(0,3);
 thread.tried=digests.flatMap(({d,date})=>(d.tried||[]).map(t=>({what:t.what,result:TRIED_RESULT[t.result],...(date?{date}:{})}))).slice(0,5);
 if(last.d.open_questions?.length)thread.open_questions=last.d.open_questions.slice(0,2);
 thread.feedback=digests.filter(x=>x.feedback).slice(0,3).map(x=>({review_date:x.date,verdict:FEEDBACK_VERDICTS[x.feedback.verdict],...(x.feedback.text?{text:x.feedback.text}:{})}));
 for(const key of ['goals','confirmed','revised','tried','feedback'])if(!thread[key].length)delete thread[key];
 // Bounds: goals reached or dropped more than two reviews ago leave first, then older
 // elements, then text is shortened; the latest review and the goals in progress stay longest.
 const size=()=>JSON.stringify(thread).length;
 const steps=[
  ()=>{const i=(thread.goals||[]).findLastIndex(g=>['reached','dropped'].includes(g.status)&&g.age>=2);if(i<0)return false;thread.goals.splice(i,1);return true},
  ()=>thread.tried?.length>2&&thread.tried.pop(),
  ()=>thread.feedback?.some(f=>f.text?.length>160)&&(thread.feedback=thread.feedback.map(f=>f.text?{...f,text:headTail(f.text,160)}:f)),
  ()=>thread.confirmed?.length>2&&thread.confirmed.pop(),
  ()=>thread.revised?.length>1&&thread.revised.pop(),
  ()=>thread.goals?.some(g=>g.path.split(' → ').length>3)&&(thread.goals=thread.goals.map(g=>{const p=g.path.split(' → ');return p.length>3?{...g,path:[p[0],'…',...p.slice(-2)].join(' → ')}:g})),
  ()=>thread.goals?.some(g=>g.evidence?.length>100)&&(thread.goals=thread.goals.map(g=>g.evidence?{...g,evidence:headTail(g.evidence,100)}:g)),
  ()=>thread.open_questions?.length>1&&thread.open_questions.pop(),
  ()=>thread.feedback?.length>1&&thread.feedback.pop(),
  ()=>{const i=(thread.goals||[]).findLastIndex(g=>!['ongoing','progressing','new'].includes(g.status));if(i<0)return false;thread.goals.splice(i,1);return true},
  ()=>thread.goals?.length>2&&thread.goals.pop(),
  ()=>thread.tried?.length&&thread.tried.pop(),
  ()=>thread.last.text?.length>300&&(thread.last.text=headTail(thread.last.text,Math.max(300,thread.last.text.length-300)),thread.last.excerpt=true),
  ()=>thread.revised?.length&&thread.revised.pop(),
  ()=>thread.confirmed?.length&&thread.confirmed.pop(),
  ()=>thread.goals?.some(g=>g.next_step)&&(thread.goals=thread.goals.map(({next_step,...g})=>g)),
  ()=>thread.open_questions?.length&&delete thread.open_questions,
  ()=>thread.feedback?.length&&delete thread.feedback,
  ()=>thread.goals?.length&&thread.goals.pop()
 ];
 while(size()>maxChars){if(!steps.some(step=>step()))break;thread.shortened=true}
 for(const goal of thread.goals||[])delete goal.age;
 for(const key of ['goals','confirmed','revised','tried'])if(thread[key]&&!thread[key].length)delete thread[key];
 return thread;
}
// The thread as readable text for the person (Paramètres, export).
export function threadProse(thread){
 if(!thread)return '';const out=[];
 out.push('Dernier bilan : '+(thread.last.kind||'bilan')+(thread.last.date?' du '+thread.last.date:'')+'.'+(thread.last.next_focus?' Prochaine étape : '+sentence(thread.last.next_focus):''));
 if(thread.last.formulation?.length)out.push('Formulation : '+thread.last.formulation.join(' ; ')+'.');
 if(thread.goals?.length)out.push('Objectifs suivis : '+thread.goals.map(g=>'« '+g.words+' » ('+g.path+')').join(' ; ')+'.');
 if(thread.confirmed?.length)out.push('Ce qui se confirme : '+thread.confirmed.join(' ; ')+'.');
 if(thread.revised?.length)out.push('Ce qui a été révisé : '+thread.revised.join(' ; ')+'.');
 if(thread.tried?.length)out.push('Essais : '+thread.tried.map(t=>t.what+' — '+t.result).join(' ; ')+'.');
 if(thread.feedback?.length)out.push('Vos retours : '+thread.feedback.map(f=>(f.review_date?'bilan du '+f.review_date+' : ':'')+f.verdict+(f.text?' (« '+f.text+' »)':'')).join(' ; ')+'.');
 if(thread.last.text)out.push('Extrait du dernier bilan : '+thread.last.text);
 return out.join('\n');
}
export const threadGoals=thread=>(thread?.goals||[]).map(g=>({key:g.key,words:g.words}));

const DIAGNOSIS=/\bvous (?:etes|souffrez (?:de|d')|avez|presentez) (?:un |une |des |de la |du |d')?(?:trouble|depression|bipolaire|borderline|schizophren|narciss|tdah|autis|tspt|stress post-?traumatique|phobie sociale|personnalite (?:borderline|narciss|dependante|evitante|histrionique))|\bvotre (?:diagnostic|pathologie)\b|\b(?:votre|vos) (?:mere|pere|soeur|frere|conjoint|conjointe|partenaire|compagnon|compagne|mari|femme|parents?) (?:est|sont) (?:un |une |des )?(?:narciss|pervers|bipolaire|borderline|psychopathe|sociopathe)|\bje (?:suis|reste) (?:votre )?(?:psychologue|therapeute|medecin)\b/;
const SAVED=/\bj.ai (?:enregistre|sauvegarde|ajoute (?:a|dans) votre)|\bc.est (?:enregistre|sauvegarde)\b/;
// Quoted passages of the published text («…», “…”), at least twelve characters.
export const quotedPassages=content=>[...String(content||'').matchAll(/«\s*([^»]{12,400}?)\s*»|“([^”]{12,400})”/g)].map(m=>(m[1]||m[2]).trim()).filter(q=>q.length>=12);
// Deterministic blocking checks on the review text before the model check.
export function reviewChecks(content,{supplied='',signal=null}={}){
 const text=foldText(content),failed=[];
 if(DIAGNOSIS.test(text))failed.push('diagnosis');
 const invented=quotedPassages(content).filter(q=>!locateQuote(supplied,q));if(invented.length)failed.push('invented_quote');
 if(signal==='self_harm'&&!/\b3114\b/.test(content)||signal==='violence'&&!/\b(?:17|112)\b/.test(content))failed.push('safety_missing');
 const numbers=[...text.matchAll(/\b(?:appele[rz]|compose[rz]?|numero(?: d.urgence)?(?: est)?|contacte[rz]|joindre)\s+(?:le\s+)?(\d{2,5})\b/g)].map(x=>x[1]);if(numbers.some(n=>!VERIFIED_NUMBERS.includes(n)))failed.push('unverified_number');
 if(SAVED.test(text))failed.push('saved_claim');
 return {failed,invented:invented.slice(0,5).map(q=>q.slice(0,120))};
}
// One verdict from the deterministic checks and the model's criteria.
export function reviewVerdict(output,deterministic){
 const criteria=output?.criteria,valid=criteria&&typeof criteria==='object'&&Object.keys(REVIEW_CHECK_CRITERIA).every(k=>['yes','no','na'].includes(criteria[k]));
 if(!valid)throw Object.assign(Error('Invalid review check'),{code:'invalid_check'});
 const failed=[...new Set([...deterministic.failed,...Object.keys(REVIEW_CHECK_CRITERIA).filter(k=>criteria[k]==='no')])];
 const blocking=failed.filter(code=>BLOCKING_CHECKS.includes(code));
 return {checked:true,passed:!blocking.length&&failed.length<2,blocking,failed,criteria,reasons:(Array.isArray(output.reasons)?output.reasons:[]).filter(x=>typeof x==='string').slice(0,4).map(x=>x.slice(0,200))};
}
// Ambre may suggest a point when a declared session approaches (48 h) or after many exchanges,
// never start one. Only when a point can be requested now.
export function reviewSuggestion({started,running,nextPointAt,care,now,lastReviewAt,turnsSince=0}){
 if(!started||running||nextPointAt)return null;
 const next=Number(care?.next_session_at)||0,last=Number(care?.last_session_at)||0;
 if(next>now&&next-now<=172800&&!(Number(lastReviewAt)>Math.max(last,now-172800)))return {reason:'session_soon',session_at:next};
 if(turnsSince>=15)return {reason:'many_exchanges'};
 return null;
}
// Probe for the 0023 columns. A positive answer is cached per database binding; a negative one
// is re-checked at most every minute, so the Worker follows the migration without a redeploy.
const PROBED=new WeakMap();
export async function reviewColumnsReady(env){
 const db=env?.DB;if(!db)return false;const cached=PROBED.get(db);if(cached===true)return true;if(typeof cached==='number'&&cached>Date.now())return false;
 try{await db.prepare('SELECT digest,feedback,stage,draft FROM ambre_reviews LIMIT 0').all();PROBED.set(db,true);return true}
 catch(error){if(/no such (?:column|table)/i.test(String(error?.message||''))){PROBED.set(db,Date.now()+60000);return false}throw error}
}
