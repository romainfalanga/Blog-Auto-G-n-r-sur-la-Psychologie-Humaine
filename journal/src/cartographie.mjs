import {CARTOGRAPHIE} from './cartographie.generated.mjs';
// Childhood map saved by the member (answers, intensity, "je me reconnais" feedback).
// The ranking mirrors compute() in static/js/cartographie.js: tests compare both.
// These are statistical associations observed in groups, never a diagnosis.
const parse=(v,f={})=>{try{return JSON.parse(v);}catch{return f;}};
const TIER_ORDER={convergence:0,piste:1,theorique:2};
const STRONG={forte:true,critere:true,moderee:true};
const STRENGTH_RANK={forte:5,critere:4,moderee:3,limitee:2,theorique:1};
export const CARTOGRAPHY_BODY_LIMIT=20000;
const plainObject=v=>Boolean(v)&&typeof v==='object'&&!Array.isArray(v)&&Object.getPrototypeOf(v)===Object.prototype;
const index=(data=CARTOGRAPHIE)=>{
 if(index.cache?.data===data)return index.cache;
 const value={data,factors:new Map(data.factors.map(f=>[f.id,f])),outcomes:new Map(data.outcomes.map(o=>[o.id,o]))};
 index.cache=value;return value;
};

// Strict validation: known identifiers only, intensity 1|2, feedback oui|non.
export function validateCartography(input,fail,data=CARTOGRAPHIE){
 const {factors,outcomes}=index(data),invalid=message=>fail(400,message,'invalid_cartography');
 const answers=input?.answers,feedback=input?.feedback===undefined?{}:input.feedback;
 if(!plainObject(answers))invalid('Réponses de cartographie invalides.');
 if(!plainObject(feedback))invalid('Retours de cartographie invalides.');
 const answerKeys=Object.keys(answers),feedbackKeys=Object.keys(feedback);
 if(answerKeys.length>factors.size)invalid('Trop de réponses.');
 if(feedbackKeys.length>outcomes.size)invalid('Trop de retours.');
 const cleanAnswers={},cleanFeedback={};
 for(const id of answerKeys.sort()){
  if(!factors.has(id))invalid('Élément de l’enfance inconnu.');
  if(answers[id]!==1&&answers[id]!==2)invalid('Intensité invalide : 1 ou 2 attendu.');
  cleanAnswers[id]=answers[id];
 }
 for(const id of feedbackKeys.sort()){
  if(!outcomes.has(id))invalid('Conséquence inconnue.');
  if(feedback[id]!=='oui'&&feedback[id]!=='non')invalid('Retour invalide : « oui » ou « non » attendu.');
  cleanFeedback[id]=feedback[id];
 }
 return {answers:cleanAnswers,feedback:cleanFeedback};
}

// Same tiers, scores and order as the public compute(); intensity 2 weighs 1.5 times.
export function computeCartography(answers,data=CARTOGRAPHIE){
 const level={};for(const [id,value] of Object.entries(answers||{}))if(value===1||value===2)level[id]=value;
 const weights=data.weights||{forte:3,moderee:2,limitee:1,theorique:1},{outcomes}=index(data),byOutcome={};
 for(const link of data.links||[])if(Object.hasOwn(level,link.factor))(byOutcome[link.outcome]=byOutcome[link.outcome]||[]).push(link);
 const results=Object.keys(byOutcome).filter(id=>outcomes.has(id)).map(id=>{
  const links=byOutcome[id].slice().sort((a,b)=>weights[b.strength]-weights[a.strength]);
  const counted=links.filter(l=>!l.definitional),strong=counted.some(l=>STRONG[l.strength]);
  const tier=counted.length>=2&&strong?'convergence':strong?'piste':'theorique';
  const score=links.reduce((sum,l)=>sum+(l.definitional?0.5:(weights[l.strength]||1))*(level[l.factor]===2?1.5:1),0);
  return {outcome:outcomes.get(id),links,factors:links.map(l=>l.factor),tier,score};
 });
 results.sort((a,b)=>TIER_ORDER[a.tier]-TIER_ORDER[b.tier]||b.score-a.score||a.outcome.label.localeCompare(b.outcome.label,'fr'));
 const aceCount=(data.factors||[]).filter(f=>f.ace&&Object.hasOwn(level,f.id)).length;
 return {results,aceCount,selectedCount:Object.keys(level).length};
}

export function summarizeCartography(answers,feedback={},{data=CARTOGRAPHIE,maxConvergence=15,maxPistes=10}={}){
 const {factors,outcomes}=index(data),state=computeCartography(answers,data),label=id=>factors.get(id)?.label||id;
 const item=r=>{
  const counted=r.links.filter(l=>!l.definitional),pool=counted.length?counted:r.links;
  const max=pool.reduce((best,l)=>STRENGTH_RANK[l.strength]>STRENGTH_RANK[best]?l.strength:best,pool[0].strength);
  const o=r.outcome;
  return {id:o.id,label:o.label,kind:o.kind,...(o.code?{code:o.code}:{}),tier:r.tier,causes:r.factors.map(label),max_strength:max,...(o.url?{url:o.url}:{}),...(o.note?{note:o.note}:{}),...(feedback[o.id]?{feedback:feedback[o.id]}:{})};
 };
 const convergence=state.results.filter(r=>r.tier==='convergence'),pistes=state.results.filter(r=>r.tier==='piste'),tierOf=new Map(state.results.map(r=>[r.outcome.id,r.tier]));
 const reported=value=>Object.keys(feedback||{}).filter(id=>feedback[id]===value&&outcomes.has(id)).sort((a,b)=>(TIER_ORDER[tierOf.get(a)]??3)-(TIER_ORDER[tierOf.get(b)]??3)||outcomes.get(a).label.localeCompare(outcomes.get(b).label,'fr')).map(id=>({id,label:outcomes.get(id).label,tier:tierOf.get(id)||'hors_resultats',...(outcomes.get(id).note?{note:outcomes.get(id).note}:{})}));
 const recognized=reported('oui'),dismissed=reported('non');
 return {
  cause_count:state.selectedCount,ace_count:state.aceCount,
  causes:Object.keys(answers||{}).filter(id=>factors.has(id)).map(id=>({id,label:label(id),intensity:answers[id]})),
  convergence:convergence.slice(0,maxConvergence).map(item),
  pistes:pistes.slice(0,maxPistes).map(item),
  theorique_count:state.results.filter(r=>r.tier==='theorique').length,
  recognized,dismissed,
  suicide_recognized:recognized.some(r=>r.note==='suicide'),
  truncated:convergence.length>maxConvergence||pistes.length>maxPistes
 };
}

// Two or three consequences to start from: recognized first, then convergences
// the person has not set aside. Suicide-related items get a relay, not a topic offer.
export function cartographyHighlights(summary,limit=3){
 if(!summary)return [];
 const dismissed=new Set(summary.dismissed.map(d=>d.id)),seen=new Set(),picked=[];
 const add=x=>{if(picked.length<limit&&!seen.has(x.id)&&!dismissed.has(x.id)&&x.note!=='suicide'){seen.add(x.id);picked.push(x);}};
 const byId=new Map([...summary.convergence,...summary.pistes].map(x=>[x.id,x]));
 for(const r of summary.recognized)if(byId.has(r.id))add(byId.get(r.id));
 for(const x of summary.convergence)add(x);
 if(!picked.length)for(const x of summary.pistes)add(x);
 return picked;
}

const missingTable=e=>/no such table:\s*cartographies/i.test(String(e?.message||''));
// Before the 0016 migration is applied the Worker keeps answering without a map.
export async function readCartography(env,userId){
 try{return await env.DB.prepare('SELECT answers,feedback,consent,locked_at,version,updated_at FROM cartographies WHERE user_id=?').bind(userId).first();}
 catch(e){if(missingTable(e))return null;throw e;}
}
export function cartographyView(row){
 if(!row)return {cartographie:null,summary:null};
 const answers=parse(row.answers),feedback=parse(row.feedback);
 return {cartographie:{answers,feedback,consent:Boolean(row.consent),locked:row.locked_at!=null,locked_at:row.locked_at??null,version:row.version,updated_at:row.updated_at},summary:summarizeCartography(answers,feedback)};
}
// Only a locked map saved with explicit consent becomes Ambre's reference.
export function companionCartography(row){
 if(!row||!row.consent||row.locked_at==null)return null;
 return {version:row.version,locked_at:row.locked_at,...summarizeCartography(parse(row.answers),parse(row.feedback))};
}
export async function loadCompanionCartography(env,userId){return companionCartography(await readCartography(env,userId));}

export function createCartographyBackend({error,json,body,now}){
 const fail=(status,message,code)=>error(status,message,code);
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 const current=async(env,userId)=>stmt(env,'SELECT answers,feedback,consent,locked_at,version,updated_at FROM cartographies WHERE user_id=?',userId).first();
 async function save(request,env,user){
  const data=await body(request,CARTOGRAPHY_BODY_LIMIT);
  if(data.consent!==true)fail(400,'Votre accord explicite est requis pour enregistrer cette donnée de santé.','consent_required');
  if(typeof data.locked!=='boolean')fail(400,'Indiquez si la cartographie doit être verrouillée.','invalid_cartography');
  // expected_version (or its alias version, null for a first save) is the version last read.
  const expected=data.expected_version!==undefined?data.expected_version:data.version===null?0:data.version;
  if(!Number.isInteger(expected)||expected<0)fail(400,'Version de cartographie invalide (0 pour une première sauvegarde).','invalid_version');
  const {answers,feedback}=validateCartography(data,fail);
  if(data.locked&&!Object.keys(answers).length)fail(400,'Cochez au moins un élément avant de verrouiller la cartographie.','invalid_cartography');
  const before=await current(env,user.id);
  if((before?.version||0)!==expected)fail(409,'Votre cartographie a changé. Rechargez-la avant de réessayer.','revision_conflict');
  const a=JSON.stringify(answers),f=JSON.stringify(feedback),t=now();
  const unchanged=before&&before.answers===a&&before.feedback===f&&Boolean(before.consent)&&(before.locked_at!=null)===data.locked;
  if(!unchanged){
   // Relocking unchanged answers keeps the original lock date.
   const lockedAt=data.locked?(before?.locked_at!=null&&before.answers===a?before.locked_at:t):null;
   const result=before
    ?await stmt(env,'UPDATE cartographies SET answers=?,feedback=?,consent=1,locked_at=?,version=version+1,updated_at=? WHERE user_id=? AND version=?',a,f,lockedAt,t,user.id,expected).run()
    :await stmt(env,'INSERT INTO cartographies(user_id,answers,feedback,consent,locked_at,version,updated_at) VALUES(?,?,?,1,?,1,?) ON CONFLICT(user_id) DO NOTHING',user.id,a,f,lockedAt,t).run();
   if(!result.meta.changes)fail(409,'Votre cartographie a changé. Rechargez-la avant de réessayer.','revision_conflict');
  }
  return json({ok:true,...cartographyView(await current(env,user.id))});
 }
 async function routes(request,env,user,path){
  if(path!=='/api/cartographie')return null;
  if(request.method==='GET')return json(cartographyView(await current(env,user.id)));
  if(request.method==='PUT')return save(request,env,user);
  if(request.method==='DELETE'){await stmt(env,'DELETE FROM cartographies WHERE user_id=?',user.id).run();return json({ok:true,cartographie:null,summary:null});}
  fail(405,'Méthode non autorisée.','method_not_allowed');
 }
 async function exportData(env,userId){return {cartographie:cartographyView(await readCartography(env,userId)).cartographie};}
 return {routes,exportData};
}
