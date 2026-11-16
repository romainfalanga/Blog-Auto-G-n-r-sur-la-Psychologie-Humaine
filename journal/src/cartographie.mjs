import {CARTOGRAPHIE} from './cartographie.generated.mjs';
// Childhood map saved by the member (answers, intensity, "je me reconnais" feedback).
// The ranking mirrors compute() in static/js/cartographie.js: tests compare both.
// These are statistical associations observed in groups, never a diagnosis.
const parse=(v,f={})=>{try{return JSON.parse(v);}catch{return f;}};
const TIER_ORDER={convergence:0,piste:1,theorique:2};
const STRONG={forte:true,moderee:true};
const relationType=l=>l.relation_type||(l.definitional?'definitional':l.strength==='critere'?'classification_context':l.strength==='theorique'?'clinical_model':'empirical_association');
const STRENGTH_RANK={forte:5,critere:4,moderee:3,limitee:2,theorique:1};
export const CARTOGRAPHY_BODY_LIMIT=80000;
export const NOTE_MAX_LENGTH=1000,NOTE_MAX_COUNT=80,AMBRE_NOTE_MAX_LENGTH=400,AMBRE_NOTE_MAX_COUNT=60,AMBRE_NOTES_PER_TURN=3;
const AMBRE_TARGETS=['factor','outcome','general'];
const plainObject=v=>Boolean(v)&&typeof v==='object'&&!Array.isArray(v)&&Object.getPrototypeOf(v)===Object.prototype;
const index=(data=CARTOGRAPHIE)=>{
 if(index.cache?.data===data)return index.cache;
 const value={data,factors:new Map(data.factors.map(f=>[f.id,f])),outcomes:new Map(data.outcomes.map(o=>[o.id,o])),protections:new Map((data.protections||[]).map(p=>[p.id,p]))};
 index.cache=value;return value;
};

// Free text keeps its line breaks; control characters and outer blanks are removed.
const cleanText=value=>String(value).replace(/\r\n?/g,'\n').replace(/[\u0000-\u0009\u000B-\u001F\u007F\u200B-\u200F\u2028-\u202E\u2060\uFEFF]/g,'').replace(/[ \t]+\n/g,'\n').replace(/\n{3,}/g,'\n\n').trim();
const fold=value=>String(value||'').normalize('NFD').replace(/\p{M}/gu,'').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim();
const emptyNotes=()=>({factors:{},outcomes:{}});
const noteCount=notes=>Object.keys(notes?.factors||{}).length+Object.keys(notes?.outcomes||{}).length;

// Strict validation: known identifiers only, 0=no and intensity 1|2, feedback oui|non,
// bounded notes in the member's words and known protective resources.
export function validateCartography(input,fail,data=CARTOGRAPHIE){
 const {factors,outcomes,protections}=index(data),invalid=message=>fail(400,message,'invalid_cartography');
 // The interface always sends the complete state: an absent block is an empty one.
 const rawNotes=input?.notes===undefined?{}:input.notes,rawResources=input?.resources===undefined?{}:input.resources;
 if(!plainObject(rawNotes)||Object.keys(rawNotes).some(k=>k!=='factors'&&k!=='outcomes'))invalid('Notes de cartographie invalides.');
 if(!plainObject(rawResources))invalid('Ressources de cartographie invalides.');
 const notes=emptyNotes();
 for(const [kind,known] of [['factors',factors],['outcomes',outcomes]]){
  const block=rawNotes[kind]===undefined?{}:rawNotes[kind];
  if(!plainObject(block))invalid('Notes de cartographie invalides.');
  for(const id of Object.keys(block).sort()){
   if(!known.has(id))invalid(kind==='factors'?'Note sur un élément de l’enfance inconnu.':'Note sur une conséquence inconnue.');
   if(typeof block[id]!=='string')invalid('Texte de note invalide.');
   const text=cleanText(block[id]);
   if(text.length>NOTE_MAX_LENGTH)invalid('Une note ne peut pas dépasser '+NOTE_MAX_LENGTH+' caractères.');
   if(text)notes[kind][id]=text;
  }
 }
 if(noteCount(notes)>NOTE_MAX_COUNT)invalid('Trop de notes : '+NOTE_MAX_COUNT+' au maximum.');
 const resources={};
 for(const id of Object.keys(rawResources).sort()){
  if(!protections.has(id))invalid('Ressource inconnue.');
  if(rawResources[id]!==1)invalid('Ressource invalide : 1 attendu.');
  resources[id]=1;
 }
 const answers=input?.answers,feedback=input?.feedback===undefined?{}:input.feedback;
 if(!plainObject(answers))invalid('Réponses de cartographie invalides.');
 if(!plainObject(feedback))invalid('Retours de cartographie invalides.');
 const answerKeys=Object.keys(answers),feedbackKeys=Object.keys(feedback);
 if(answerKeys.length>factors.size)invalid('Trop de réponses.');
 if(feedbackKeys.length>outcomes.size)invalid('Trop de retours.');
 const cleanAnswers={},cleanFeedback={};
 for(const id of answerKeys.sort()){
  if(!factors.has(id))invalid('Élément de l’enfance inconnu.');
  if(answers[id]!==0&&answers[id]!==1&&answers[id]!==2)invalid('Réponse invalide : 0, 1 ou 2 attendu.');
  cleanAnswers[id]=answers[id];
 }
 for(const group of data.exclusive||[])if(group.filter(id=>cleanAnswers[id]===1||cleanAnswers[id]===2).length>1)invalid('Ces réponses décrivent des situations alternatives : choisissez celle qui correspond à votre vécu.');
 for(const id of feedbackKeys.sort()){
  if(!outcomes.has(id))invalid('Conséquence inconnue.');
  if(feedback[id]!=='oui'&&feedback[id]!=='non')invalid('Retour invalide : « oui » ou « non » attendu.');
  cleanFeedback[id]=feedback[id];
 }
 return {answers:cleanAnswers,feedback:cleanFeedback,notes,resources};
}

// Same exploratory order as the public compute(); no summed risk or intensity multiplier.
export function computeCartography(answers,data=CARTOGRAPHIE){
 const {factors,outcomes}=index(data),level={};for(const [id,value] of Object.entries(answers||{}))if(factors.has(id)&&(value===1||value===2))level[id]=value;
 const weights=data.weights||{forte:3,moderee:2,limitee:1,theorique:1},byOutcome={};
 for(const link of data.links||[])if(Object.hasOwn(level,link.factor))(byOutcome[link.outcome]=byOutcome[link.outcome]||[]).push(link);
 const results=Object.keys(byOutcome).filter(id=>outcomes.has(id)).map(id=>{
  const links=byOutcome[id].slice().sort((a,b)=>weights[b.strength]-weights[a.strength]);
  const empirical=links.filter(l=>!l.definitional&&relationType(l)==='empirical_association');
  const groups=new Set(empirical.filter(l=>STRONG[l.strength]).map(l=>factors.get(l.factor).overlap_group||l.factor));
  const tier=outcomes.get(id).contextual_only?'theorique':groups.size>=2?'convergence':groups.size?'piste':'theorique';
  const score=empirical.reduce((best,l)=>Math.max(best,weights[l.strength]||0),0);
  return {outcome:outcomes.get(id),links,factors:links.map(l=>l.factor),tier,score,evidence_groups:groups.size,relation_types:[...new Set(links.map(relationType))]};
 });
 results.sort((a,b)=>TIER_ORDER[a.tier]-TIER_ORDER[b.tier]||b.evidence_groups-a.evidence_groups||b.score-a.score||a.outcome.id.localeCompare(b.outcome.id,'en'));
 const aceCount=new Set((data.factors||[]).filter(f=>f.ace&&Object.hasOwn(level,f.id)).map(f=>{const group=(data.exclusive||[]).find(g=>g.includes(f.id));return group?group.join('|'):f.id;})).size;
 return {results,aceCount,selectedCount:Object.keys(level).length};
}

// Stored notes written by Ambre; entries about identifiers that left the dataset are hidden.
function storedAmbreNotes(value,data=CARTOGRAPHIE){
 const {factors,outcomes}=index(data);
 return (Array.isArray(value)?value:[]).filter(n=>n&&typeof n.id==='string'&&typeof n.text==='string'&&AMBRE_TARGETS.includes(n.target)&&(n.target==='general'?n.target_id==null:(n.target==='factor'?factors:outcomes).has(n.target_id))).map(n=>({id:n.id,target:n.target,target_id:n.target==='general'?null:n.target_id,text:n.text,created_at:Number(n.created_at)||0}));
}
// Notes proposed by the model in one turn: known targets, short text, no duplicate
// of a stored note (same target and same normalized text). Invalid items are dropped.
export function chooseAmbreNotes(value,existing=[],data=CARTOGRAPHIE){
 if(!Array.isArray(value))return [];
 const {factors,outcomes}=index(data),key=n=>n.target+'|'+(n.target_id??'')+'|'+fold(n.text),seen=new Set((Array.isArray(existing)?existing:[]).filter(n=>n&&typeof n.text==='string').map(key)),chosen=[];
 for(const item of value.slice(0,AMBRE_NOTES_PER_TURN)){
  if(!item||!AMBRE_TARGETS.includes(item.target)||typeof item.text!=='string')continue;
  const general=item.target==='general',known=item.target==='factor'?factors:outcomes;
  if(general?item.target_id!=null&&item.target_id!=='':typeof item.target_id!=='string'||!known.has(item.target_id))continue;
  const text=cleanText(item.text).replace(/\s+/g,' ');
  if(!fold(text)||text.length>AMBRE_NOTE_MAX_LENGTH)continue;
  const note={target:item.target,target_id:general?null:item.target_id,text};
  if(seen.has(key(note)))continue;
  seen.add(key(note));chosen.push({...note,...(general?{}:{label:known.get(item.target_id).label})});
 }
 return chosen;
}
// Appends to the stored list; beyond the cap the oldest notes leave first.
export function appendAmbreNotes(stored,chosen,{now,random}){
 const list=[...(Array.isArray(stored)?stored:[]),...chosen.map(({target,target_id,text})=>({id:random(),target,target_id,text,created_at:now()}))];
 return list.slice(Math.max(0,list.length-AMBRE_NOTE_MAX_COUNT));
}

// Bounded context should not be filled by one category (for example the 18
// schemas). Evidence families then the member's recognition break ties; the ID
// is only a stable last resort, never a clinical priority.
function contextOrder(feedback){return (a,b)=>b.evidence_groups-a.evidence_groups||Number(feedback[b.outcome.id]==='oui')-Number(feedback[a.outcome.id]==='oui')||b.score-a.score||a.outcome.id.localeCompare(b.outcome.id,'en');}
function diverseResults(items,limit,feedback={}){
 const counts=new Map(),picked=[],perCategory=Math.max(1,Math.ceil(limit/3));
 for(const result of [...items].sort(contextOrder(feedback))){
  const category=result.outcome.category||result.outcome.kind;
  if(picked.length>=limit)break;
  if((counts.get(category)||0)>=perCategory)continue;
  counts.set(category,(counts.get(category)||0)+1);picked.push(result);
 }
 return picked;
}

export function summarizeCartography(answers,feedback={},{data=CARTOGRAPHIE,maxConvergence=15,maxPistes=10,notes=null,resources=null,ambreNotes=null}={}){
 const {factors,outcomes,protections}=index(data),state=computeCartography(answers,data),label=id=>factors.get(id)?.label||id;
 const metadata=o=>({...(o.category?{category:o.category}:{}),...(o.note?{note:o.note}:{}),...(o.age_scope?{age_scope:o.age_scope}:{}),...(o.contextual_only?{contextual_only:true}:{})});
 const item=r=>{
  const empirical=r.links.filter(l=>!l.definitional&&relationType(l)==='empirical_association'),pool=empirical.length?empirical:r.links;
  const max=pool.reduce((best,l)=>STRENGTH_RANK[l.strength]>STRENGTH_RANK[best]?l.strength:best,pool[0].strength);
  const o=r.outcome;
  return {id:o.id,label:o.label,kind:o.kind,...metadata(o),...(o.code?{code:o.code}:{}),tier:r.tier,causes:r.factors.map(label),max_strength:max,relation_types:r.relation_types,evidence_groups:r.evidence_groups,...(o.url?{url:o.url}:{}),...(feedback[o.id]?{feedback:feedback[o.id]}:{})};
 };
 const convergence=state.results.filter(r=>r.tier==='convergence'),pistes=state.results.filter(r=>r.tier==='piste'),tierOf=new Map(state.results.map(r=>[r.outcome.id,r.tier]));
 const reported=value=>Object.keys(feedback||{}).filter(id=>feedback[id]===value&&outcomes.has(id)).sort((a,b)=>(TIER_ORDER[tierOf.get(a)]??3)-(TIER_ORDER[tierOf.get(b)]??3)||outcomes.get(a).label.localeCompare(outcomes.get(b).label,'fr')).map(id=>({id,label:outcomes.get(id).label,tier:tierOf.get(id)||'hors_resultats',...metadata(outcomes.get(id))}));
 const recognized=reported('oui'),dismissed=reported('non');
 const selectedConvergence=diverseResults(convergence,maxConvergence,feedback),selectedPistes=diverseResults(pistes,maxPistes,feedback);
 const exploratory=state.results.filter(r=>r.tier==='theorique'&&!r.outcome.contextual_only&&feedback[r.outcome.id]==='oui'),selectedExploratory=diverseResults(exploratory,maxPistes,feedback);
 return {
  cause_count:state.selectedCount,ace_count:state.aceCount,
  answered_count:Object.keys(answers||{}).filter(id=>factors.has(id)&&[0,1,2].includes(answers[id])).length,
  explicit_negative:Object.keys(answers||{}).filter(id=>factors.has(id)&&answers[id]===0).map(id=>({id,label:label(id)})),
  causes:Object.keys(answers||{}).filter(id=>factors.has(id)&&(answers[id]===1||answers[id]===2)).map(id=>({id,label:label(id),intensity:answers[id]})),
  convergence:selectedConvergence.map(item),
  pistes:selectedPistes.map(item),
  contextual:state.results.filter(r=>r.outcome.contextual_only).map(item),
  exploratory:selectedExploratory.map(item),
  theorique_count:state.results.filter(r=>r.tier==='theorique').length,
  recognized,dismissed,
  suicide_recognized:recognized.some(r=>r.note==='suicide'),
  truncated:convergence.length>selectedConvergence.length||pistes.length>selectedPistes.length||exploratory.length>selectedExploratory.length,
  selection_policy:{order:'evidence_families_then_member_recognition',per_category_convergence:Math.ceil(maxConvergence/3),per_category_pistes:Math.ceil(maxPistes/3),individual_risk:false},
  // The member's own words, attached to the cause or consequence they annotate.
  notes:[...Object.entries(notes?.factors||{}).filter(([id,text])=>factors.has(id)&&typeof text==='string').map(([id,text])=>({target:'factor',id,label:factors.get(id).label,text})),
   ...Object.entries(notes?.outcomes||{}).filter(([id,text])=>outcomes.has(id)&&typeof text==='string').map(([id,text])=>({target:'outcome',id,label:outcomes.get(id).label,text}))],
  resources:Object.keys(resources||{}).filter(id=>protections.has(id)).map(id=>({id,label:protections.get(id).label})),
  ambre_notes:storedAmbreNotes(ambreNotes,data).map(n=>({...n,...(n.target==='general'?{}:{label:(n.target==='factor'?factors:outcomes).get(n.target_id).label})}))
 };
}

// Two or three consequences to start from: recognized first, then convergences
// the person has not set aside. Suicide-related items get a relay, not a topic offer.
export function cartographyHighlights(summary,limit=3){
 if(!summary)return [];
 const dismissed=new Set(summary.dismissed.map(d=>d.id)),seen=new Set(),picked=[];
 const add=x=>{if(picked.length<limit&&!seen.has(x.id)&&!dismissed.has(x.id)&&x.note!=='suicide'&&!x.contextual_only){seen.add(x.id);picked.push(x);}};
 const byId=new Map([...summary.convergence,...summary.pistes].map(x=>[x.id,x]));
 for(const r of summary.recognized)if(byId.has(r.id))add(byId.get(r.id));
 for(const x of summary.convergence)add(x);
 if(!picked.length)for(const x of summary.pistes)add(x);
 return picked;
}

const missingTable=e=>/no such table:\s*cartographies/i.test(String(e?.message||''));
const missingColumn=e=>/no such column:\s*(?:notes|resources|ambre_notes)/i.test(String(e?.message||''));
const COLUMNS='answers,feedback,consent,locked_at,version,updated_at';
// Between the Worker deployment and the 0017 migration the map is read without its notes.
async function selectCartography(env,userId){
 try{return await env.DB.prepare(`SELECT ${COLUMNS},notes,resources,ambre_notes FROM cartographies WHERE user_id=?`).bind(userId).first();}
 catch(e){if(!missingColumn(e))throw e;return env.DB.prepare(`SELECT ${COLUMNS} FROM cartographies WHERE user_id=?`).bind(userId).first();}
}
// Before the 0016 migration is applied the Worker keeps answering without a map.
export async function readCartography(env,userId){
 try{return await selectCartography(env,userId);}
 catch(e){if(missingTable(e))return null;throw e;}
}
const rowNotes=row=>{const value=parse(row.notes),clean=block=>plainObject(block)?Object.fromEntries(Object.entries(block).filter(([,text])=>typeof text==='string')):{};return {factors:clean(value?.factors),outcomes:clean(value?.outcomes)};};
const rowResources=row=>{const value=parse(row.resources);return plainObject(value)?value:{};};
const rowAmbreNotes=row=>storedAmbreNotes(parse(row.ambre_notes,[]));
const rowSummary=row=>summarizeCartography(parse(row.answers),parse(row.feedback),{notes:rowNotes(row),resources:rowResources(row),ambreNotes:rowAmbreNotes(row)});
export function cartographyView(row){
 if(!row)return {cartographie:null,summary:null};
 return {cartographie:{answers:parse(row.answers),feedback:parse(row.feedback),notes:rowNotes(row),resources:rowResources(row),ambre_notes:rowAmbreNotes(row),consent:Boolean(row.consent),locked:row.locked_at!=null,locked_at:row.locked_at??null,version:row.version,updated_at:row.updated_at},summary:rowSummary(row)};
}
// What research says about each recognized or converging consequence, for the checked
// causes only: cause, level of evidence and the sourced summary. Recognized first.
export function cartographyEvidence(answers,feedback={},{data=CARTOGRAPHIE,budget=6000}={}){
 const {factors}=index(data),state=computeCartography(answers,data),evidence=[],sourceIndex=new Map((data.sources||[]).map(s=>[s.id,s])),usedSources=new Map();let used=0,truncated=false;
 const selected=new Set(diverseResults(state.results.filter(r=>r.tier==='convergence'),15,feedback).map(r=>r.outcome.id));
 const wanted=state.results.filter(r=>feedback?.[r.outcome.id]!=='non'&&(feedback?.[r.outcome.id]==='oui'||(!r.outcome.contextual_only&&selected.has(r.outcome.id)))).sort((a,b)=>Number(feedback[b.outcome.id]==='oui')-Number(feedback[a.outcome.id]==='oui')||contextOrder(feedback)(a,b));
 const collected=new Map();
 // One sourced link per subject before adding a second: a long first card must
 // not consume the whole evidence budget and hide the other selected subjects.
 for(let round=0;round<6;round++)for(const r of wanted){
  const l=r.links.filter(l=>l.summary)[round];if(!l)continue;
  const source_ids=l.source_ids||l.sources||[],fresh=source_ids.filter(id=>!usedSources.has(id)&&sourceIndex.has(id)).map(id=>{const {type,url,citation}=sourceIndex.get(id);return {id,type,url,citation};});
  const entry={factor_id:l.factor,cause:factors.get(l.factor)?.label||l.factor,strength:l.strength,relation_type:relationType(l),definitional:Boolean(l.definitional),source_ids,summary:l.summary},size=JSON.stringify(entry).length+JSON.stringify(fresh).length;
  if(used+size>budget){truncated=true;continue;}
  used+=size;if(!collected.has(r.outcome.id))collected.set(r.outcome.id,[]);collected.get(r.outcome.id).push(entry);for(const source of fresh)usedSources.set(source.id,source);
 }
 for(const r of wanted){
  const links=collected.get(r.outcome.id)||[];
  if(links.length<r.links.length)truncated=true;
  if(links.length)evidence.push({id:r.outcome.id,label:r.outcome.label,...(r.outcome.note?{note:r.outcome.note}:{}),...(r.outcome.age_scope?{age_scope:r.outcome.age_scope}:{}),...(r.outcome.contextual_only?{contextual_only:true}:{}),links});
 }
 return {evidence,evidence_sources:[...usedSources.values()],evidence_truncated:truncated};
}
// Only a saved (locked) map with explicit consent becomes Ambre's working document.
export function companionCartography(row){
 if(!row||!row.consent||row.locked_at==null)return null;
 return {version:row.version,locked_at:row.locked_at,...rowSummary(row),...cartographyEvidence(parse(row.answers),parse(row.feedback))};
}
// Ambre's notes never touch answers, version or updated_at. The compare-and-swap
// on the stored list keeps a concurrent removal by the member.
export function ambreNotesWrite(env,userId,before,after,guard='1=1',guardArgs=[]){
 return env.DB.prepare(`UPDATE cartographies SET ambre_notes=? WHERE user_id=? AND consent=1 AND locked_at IS NOT NULL AND ambre_notes=? AND ${guard}`).bind(JSON.stringify(after),userId,before,...guardArgs);
}
export async function addAmbreNotes(env,userId,items,tools,attempts=3){
 for(let attempt=0;attempt<attempts;attempt++){
  const row=await env.DB.prepare('SELECT ambre_notes FROM cartographies WHERE user_id=? AND consent=1 AND locked_at IS NOT NULL').bind(userId).first();
  if(!row)return false;
  const stored=parse(row.ambre_notes,[]),fresh=chooseAmbreNotes(items,stored);
  if(!fresh.length)return true;
  if((await ambreNotesWrite(env,userId,row.ambre_notes,appendAmbreNotes(stored,fresh,tools)).run()).meta.changes)return true;
 }
 return false;
}
export async function loadCompanionCartography(env,userId){return companionCartography(await readCartography(env,userId));}

export function createCartographyBackend({error,json,body,now}){
 const fail=(status,message,code)=>error(status,message,code);
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 const current=(env,userId)=>selectCartography(env,userId);
 async function save(request,env,user){
  const data=await body(request,CARTOGRAPHY_BODY_LIMIT);
  if(data.consent!==true)fail(400,'Votre accord explicite est requis pour enregistrer cette donnée de santé.','consent_required');
  if(typeof data.locked!=='boolean')fail(400,'Indiquez si la cartographie doit être verrouillée.','invalid_cartography');
  // expected_version (or its alias version, null for a first save) is the version last read.
  const expected=data.expected_version!==undefined?data.expected_version:data.version===null?0:data.version;
  if(!Number.isInteger(expected)||expected<0)fail(400,'Version de cartographie invalide (0 pour une première sauvegarde).','invalid_version');
  const {answers,feedback,notes,resources}=validateCartography(data,fail);
  if(data.locked&&!Object.keys(answers).length&&!noteCount(notes)&&!Object.keys(resources).length)fail(400,'Cochez au moins un élément, écrivez une note ou indiquez une ressource avant d’enregistrer la cartographie.','invalid_cartography');
  const before=await current(env,user.id);
  if((before?.version||0)!==expected)fail(409,'Votre cartographie a changé. Rechargez-la avant de réessayer.','revision_conflict');
  const a=JSON.stringify(answers),f=JSON.stringify(feedback),n=JSON.stringify(notes),r=JSON.stringify(resources),t=now();
  const unchanged=before&&before.answers===a&&before.feedback===f&&JSON.stringify(rowNotes(before))===n&&JSON.stringify(rowResources(before))===r&&Boolean(before.consent)&&(before.locked_at!=null)===data.locked;
  if(!unchanged){
   // Relocking unchanged answers keeps the original lock date.
   const lockedAt=data.locked?(before?.locked_at!=null&&before.answers===a?before.locked_at:t):null;
   const result=before
    // ambre_notes is never written here: only Ambre adds to it and only the member removes from it.
    ?await stmt(env,'UPDATE cartographies SET answers=?,feedback=?,notes=?,resources=?,consent=1,locked_at=?,version=version+1,updated_at=? WHERE user_id=? AND version=?',a,f,n,r,lockedAt,t,user.id,expected).run()
    :await stmt(env,'INSERT INTO cartographies(user_id,answers,feedback,notes,resources,consent,locked_at,version,updated_at) VALUES(?,?,?,?,?,1,?,1,?) ON CONFLICT(user_id) DO NOTHING',user.id,a,f,n,r,lockedAt,t).run();
   if(!result.meta.changes)fail(409,'Votre cartographie a changé. Rechargez-la avant de réessayer.','revision_conflict');
  }
  return json({ok:true,...cartographyView(await current(env,user.id))});
 }
 async function removeAmbreNote(env,user,noteId){
  for(let attempt=0;attempt<3;attempt++){
   const row=await current(env,user.id),stored=parse(row?.ambre_notes,[]),kept=(Array.isArray(stored)?stored:[]).filter(note=>note?.id!==noteId);
   if(!row||kept.length===stored.length)fail(404,'Note introuvable.','not_found');
   // Neither version nor updated_at changes: the member's autosave keeps its version.
   const result=await stmt(env,'UPDATE cartographies SET ambre_notes=? WHERE user_id=? AND ambre_notes=?',JSON.stringify(kept),user.id,row.ambre_notes).run();
   if(result.meta.changes)return json({ok:true,...cartographyView(await current(env,user.id))});
  }
  fail(409,'Votre cartographie a changé. Rechargez-la avant de réessayer.','revision_conflict');
 }
 async function routes(request,env,user,path){
  const note=path.match(/^\/api\/cartographie\/ambre-notes\/([\w-]{1,120})$/);
  if(note){if(request.method!=='DELETE')fail(405,'Méthode non autorisée.','method_not_allowed');return removeAmbreNote(env,user,note[1]);}
  if(path!=='/api/cartographie')return null;
  if(request.method==='GET')return json(cartographyView(await current(env,user.id)));
  if(request.method==='PUT')return save(request,env,user);
  if(request.method==='DELETE'){await stmt(env,'DELETE FROM cartographies WHERE user_id=?',user.id).run();return json({ok:true,cartographie:null,summary:null});}
  fail(405,'Méthode non autorisée.','method_not_allowed');
 }
 async function exportData(env,userId){return {cartographie:cartographyView(await readCartography(env,userId)).cartographie};}
 return {routes,exportData};
}
