// « Mes symptômes » (plan v1.1, lot A) : consentement, passations, brouillons, alertes de sécurité,
// croisement avec Mon enfance. Contrat d'API figé : docs/symptomes-contrat-2026-11-19.md, § 10.
// The Worker recomputes every score and state with symptomes-engine.mjs; nothing computed by the
// client is stored. Adults only (date de naissance de Mon compte); a minor's answers are refused.
import {SYMPTOMES} from './symptomes-data.generated.mjs';
import {CARTOGRAPHIE} from './cartographie.generated.mjs';
import {computeCartography,readCartography} from './cartographie.mjs';
import {validatePassation,computeModuleResult,safetyTriggers,safetyScreen,ageGate,retestInfo,overlaps,crossWithChildhood,symptomsIndex,PASSATION_KEY} from './symptomes-engine.mjs';

export const SYMPTOMS_BODY_LIMIT=20000;
export const SYMPTOMS_HISTORY_LIMIT=100;
const parse=(v,f={})=>{try{const x=JSON.parse(v);return x??f}catch{return f}};
const missing=e=>/no such (?:table|column)/i.test(String(e?.message||''));

// Table probe: before migration 0028 the page says « bientôt disponible », never an error 500.
const PROBED=new WeakMap();
export async function symptomsReady(env){
 const db=env?.DB;if(!db)return false;const cached=PROBED.get(db);if(cached===true)return true;if(typeof cached==='number'&&cached>Date.now())return false;
 try{
  for(const table of ['symptom_consents','symptom_checks','symptom_drafts','symptom_safety_alerts'])await db.prepare(`SELECT * FROM ${table} LIMIT 0`).bind().all();
  PROBED.set(db,true);return true;
 }catch(error){if(missing(error)){PROBED.set(db,Date.now()+5000);return false}throw error}
}

const checkView=row=>row&&({id:row.id,module:row.module,passation_key:row.passation_key,content_version:row.content_version,current_version:row.content_version===SYMPTOMES.version,instrument_versions:parse(row.instrument_versions),answers:parse(row.answers),scores:parse(row.scores),result:parse(row.result),state:row.state,declined:Boolean(row.declined),started_at:row.started_at??null,completed_at:row.completed_at});
const draftView=row=>row&&({module:row.module,passation_key:row.passation_key,content_version:row.content_version,current_version:row.content_version===SYMPTOMES.version,answers:parse(row.answers),started_at:row.started_at??null,updated_at:row.updated_at});
const alertView=row=>({id:row.id,module:row.module,trigger_id:row.trigger_id,kind:row.kind,screen:row.screen,check_id:row.check_id??null,created_at:row.created_at});
const today=seconds=>new Date(seconds*1000).toISOString().slice(0,10);

// Mon enfance side of the crossing: subjects of sections top/other with an empirical link, not
// marked « Pas moi »; « Pas moi » subjects are listed apart. null when no map is saved.
export function childhoodSide(row){
 if(!row||!row.consent)return null;
 const feedback=parse(row.feedback),state=computeCartography(parse(row.answers),CARTOGRAPHIE,feedback);
 const present=[...state.sections.top,...state.sections.other].filter(r=>r.associated_count>0&&feedback[r.outcome.id]!=='non').map(r=>r.outcome.id);
 return {present,dismissed:Object.keys(feedback).filter(id=>feedback[id]==='non')};
}
let LABELS=null;
const subjectLabels=()=>LABELS||(LABELS=Object.fromEntries(CARTOGRAPHIE.outcomes.filter(o=>Object.hasOwn(SYMPTOMES.subject_modules||{},o.id)).map(o=>[o.id,o.label])));

async function birthDate(env,userId){
 const row=await env.DB.prepare('SELECT self_data FROM profiles WHERE user_id=?').bind(userId).first();
 const value=parse(row?.self_data).birth_date;return typeof value==='string'?value:null;
}
// Latest dated blocking alert, for the 14-day banner and for Ambre (lot C: safety.reported_at).
export async function latestSafetyReport(env,userId){
 if(!await symptomsReady(env))return null;
 const row=await env.DB.prepare("SELECT created_at,kind FROM symptom_safety_alerts WHERE user_id=? AND screen='bloquant' ORDER BY created_at DESC,id DESC LIMIT 1").bind(userId).first();
 return row?{reported_at:row.created_at,kind:row.kind}:null;
}

export function createSymptomsBackend({error,json,body,now,random,limit}){
 const fail=(status,message,code)=>error(status,message,code);
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 const first=(env,sql,...args)=>stmt(env,sql,...args).first();
 const all=async(env,sql,...args)=>(await stmt(env,sql,...args).all()).results;
 const unavailable=()=>fail(503,'« Mes symptômes » sera bientôt disponible.','feature_unavailable');

 async function readConsent(env,userId){
  const row=await first(env,'SELECT consent,content_version,granted_at,withdrawn_at,updated_at FROM symptom_consents WHERE user_id=?',userId);
  return {given:row?.consent===1,content_version:row?.content_version??null,granted_at:row?.granted_at??null,withdrawn_at:row?.withdrawn_at??null};
 }
 async function age(env,userId){return ageGate(await birthDate(env,userId),today(now()),SYMPTOMES)}
 // Writes need: migration, adult age, consent. Each refusal has its own code (contract § 10).
 async function writable(env,userId,{consent=true}={}){
  if(!await symptomsReady(env))unavailable();
  const gate=await age(env,userId);
  if(gate.status==='mineur')fail(403,gate.text,'minor');
  if(gate.status==='inconnu')fail(403,'Indiquez votre date de naissance dans Mon compte : cette page est réservée aux adultes.','birth_date_required');
  if(consent&&!(await readConsent(env,userId)).given)fail(403,'Votre accord pour « Mes symptômes » est nécessaire avant d’enregistrer vos réponses.','consent_required');
  return gate;
 }
 const alertStatements=(env,userId,passationKey,moduleId,triggers,checkId=null)=>triggers.map(t=>stmt(env,'INSERT INTO symptom_safety_alerts(id,user_id,passation_key,module,trigger_id,kind,screen,check_id,created_at) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(user_id,passation_key,trigger_id) DO UPDATE SET check_id=COALESCE(excluded.check_id,symptom_safety_alerts.check_id)',random(),userId,passationKey,moduleId,t.id,t.kind,t.screen,checkId,now()));
 const passationKey=value=>{if(typeof value!=='string'||!PASSATION_KEY.test(value))fail(400,'Identifiant de passation requis (8 à 120 caractères).','passation_key_required');return value};
 const startedAt=value=>{if(value==null)return null;if(!Number.isInteger(value)||value<0||value>now()+300)fail(400,'Date de début invalide.','invalid_symptoms');return value};

 // Whole page state (GET /api/symptomes).
 async function readState(env,userId){
  if(!await symptomsReady(env))return {available:false,message:'« Mes symptômes » sera bientôt disponible.',content_version:SYMPTOMES.version};
  const t=now(),gate=await age(env,userId),consent=await readConsent(env,userId);
  const latestRows=await all(env,'SELECT c.* FROM symptom_checks c WHERE c.user_id=? AND c.id=(SELECT id FROM symptom_checks x WHERE x.user_id=c.user_id AND x.module=c.module ORDER BY completed_at DESC,rowid DESC LIMIT 1)',userId);
  const answeredRows=await all(env,"SELECT module,max(completed_at) completed_at,count(*) n FROM symptom_checks WHERE user_id=? AND declined=0 GROUP BY module",userId);
  const counts=await all(env,'SELECT module,count(*) n FROM symptom_checks WHERE user_id=? GROUP BY module',userId);
  const drafts=await all(env,'SELECT * FROM symptom_drafts WHERE user_id=?',userId);
  const latest=new Map(latestRows.map(r=>[r.module,checkView(r)])),lastAnswered=new Map(answeredRows.map(r=>[r.module,r.completed_at])),count=new Map(counts.map(r=>[r.module,r.n])),draft=new Map(drafts.map(r=>[r.module,draftView(r)]));
  const modules=SYMPTOMES.modules.map(m=>{const last=latest.get(m.id)||null;return {id:m.id,order:m.order,state:last?last.state:'non_explore',last,count:count.get(m.id)||0,retest:retestInfo(m.id,lastAnswered.get(m.id)??null,t,SYMPTOMES),draft:draft.get(m.id)||null}});
  // « Vie entière » answers are asked once: the latest given answer is offered again (§ 3).
  const lifetime={},lifetimeIds=new Set(SYMPTOMES.modules.flatMap(m=>m.questions.filter(q=>q.period==='vie').map(q=>q.id)));
  const lifetimeRows=await all(env,'SELECT answers FROM symptom_checks WHERE user_id=? AND declined=0 ORDER BY completed_at DESC,rowid DESC LIMIT 500',userId);
  for(const row of lifetimeRows){const a=parse(row.answers);for(const id of lifetimeIds)if(!Object.hasOwn(lifetime,id)&&a[id]!=null)lifetime[id]=a[id]}
  const report=await first(env,"SELECT created_at,kind FROM symptom_safety_alerts WHERE user_id=? AND screen='bloquant' ORDER BY created_at DESC,id DESC LIMIT 1",userId);
  const bannerDays=SYMPTOMES.safety.display.results_banner_days||14,bannerKinds=report&&report.created_at+bannerDays*86400>t?(await all(env,"SELECT DISTINCT kind FROM symptom_safety_alerts WHERE user_id=? AND screen='bloquant' AND created_at>? ORDER BY kind",userId,t-bannerDays*86400)).map(r=>r.kind):[];
  const banner=bannerKinds.length?{until:report.created_at+bannerDays*86400,kinds:bannerKinds,...safetyScreen(bannerKinds.map(kind=>({id:'banner',kind,screen:'bloquant'})),SYMPTOMES,{age:gate.age})}:null;
  const results=Object.fromEntries([...latest].map(([id,c])=>[id,c.result]));
  const crossing=crossWithChildhood(results,childhoodSide(await readCartography(env,userId)),SYMPTOMES,{labels:subjectLabels()});
  return {available:true,content_version:SYMPTOMES.version,consent,age:gate,modules,lifetime_answers:lifetime,safety:{banner,last_reported_at:report?.created_at??null},crossing,unpublished_modules:(SYMPTOMES.unpublished_modules||[]).slice()};
 }

 async function setConsent(request,env,user){
  const data=await body(request,1000);
  if(typeof data.consent!=='boolean')fail(400,'Indiquez votre choix (true ou false).','invalid_consent');
  const t=now();
  if(data.consent){
   await writable(env,user.id,{consent:false});
   await stmt(env,'INSERT INTO symptom_consents(user_id,consent,content_version,granted_at,withdrawn_at,updated_at) VALUES(?,1,?,?,NULL,?) ON CONFLICT(user_id) DO UPDATE SET consent=1,content_version=excluded.content_version,granted_at=CASE WHEN symptom_consents.consent=1 THEN symptom_consents.granted_at ELSE excluded.granted_at END,withdrawn_at=NULL,updated_at=excluded.updated_at',user.id,SYMPTOMES.version,t,t).run();
  }else{
   // Withdrawal is always possible (even for a minor or without birth date). Results are kept until
   // the person deletes them; nothing new is recorded and Ambre stops receiving them (lot C).
   if(!await symptomsReady(env))unavailable();
   await stmt(env,'UPDATE symptom_consents SET consent=0,withdrawn_at=?,updated_at=? WHERE user_id=? AND consent=1',t,t,user.id).run();
  }
  return json({ok:true,consent:await readConsent(env,user.id)});
 }

 async function saveCheck(request,env,user){
  const gate=await writable(env,user.id);
  await limit(env,'symptoms-check:'+user.id,60,3600);
  const data=await body(request,SYMPTOMS_BODY_LIMIT),key=passationKey(data.passation_key),started=startedAt(data.started_at);
  const clean=validatePassation(data,SYMPTOMES,fail);
  const result=computeModuleResult(clean.module,clean.answers,SYMPTOMES,{declined:clean.declined});
  const answers=JSON.stringify(clean.answers),versions=JSON.stringify(clean.instrument_versions);
  const previous=await first(env,'SELECT max(completed_at) at FROM symptom_checks WHERE user_id=? AND module=? AND declined=0',user.id,clean.module);
  const id=random(),t=now(),triggers=result.safety;
  const [inserted]=await env.DB.batch([
   stmt(env,'INSERT INTO symptom_checks(id,user_id,passation_key,module,content_version,instrument_versions,answers,scores,result,state,declined,started_at,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(user_id,passation_key) DO NOTHING',id,user.id,key,clean.module,clean.content_version,versions,answers,JSON.stringify(result.scores),JSON.stringify(result),result.state,Number(clean.declined),started,t),
   ...alertStatements(env,user.id,key,clean.module,triggers),
   stmt(env,'UPDATE symptom_safety_alerts SET check_id=? WHERE user_id=? AND passation_key=? AND EXISTS(SELECT 1 FROM symptom_checks WHERE id=? AND user_id=?)',id,user.id,key,id,user.id),
   stmt(env,'DELETE FROM symptom_drafts WHERE user_id=? AND module=? AND EXISTS(SELECT 1 FROM symptom_checks WHERE id=? AND user_id=?)',user.id,clean.module,id,user.id)
  ]);
  if(!inserted.meta.changes){
   // Same passation sent again (network retry): the first record answers; anything else conflicts.
   const existing=await first(env,'SELECT * FROM symptom_checks WHERE user_id=? AND passation_key=?',user.id,key);
   if(!existing||existing.module!==clean.module||existing.answers!==answers||Boolean(existing.declined)!==clean.declined)fail(409,'Cette passation a déjà été enregistrée avec d’autres réponses.','passation_conflict');
   return json({ok:true,replayed:true,check:checkView(existing),overlap:false,safety:{triggers,screen:safetyScreen(triggers,SYMPTOMES,{age:gate.age})},page:await readState(env,user.id)});
  }
  const overlap=!clean.declined&&overlaps(clean.module,previous?.at??null,t,SYMPTOMES);
  return json({ok:true,check:checkView(await first(env,'SELECT * FROM symptom_checks WHERE id=?',id)),overlap,safety:{triggers,screen:safetyScreen(triggers,SYMPTOMES,{age:gate.age})},page:await readState(env,user.id)},201);
 }

 // Autosave of a passation in progress. Safety answers are recorded here, as soon as they are saved.
 async function saveDraft(request,env,user,moduleId){
  const gate=await writable(env,user.id);
  await limit(env,'symptoms-draft:'+user.id,600,3600);
  const data=await body(request,SYMPTOMS_BODY_LIMIT),key=passationKey(data.passation_key),started=startedAt(data.started_at);
  if(data.module!==undefined&&data.module!==moduleId)fail(400,'Le module ne correspond pas à l’adresse.','invalid_symptoms');
  const clean=validatePassation({...data,module:moduleId,declined:false},SYMPTOMES,fail,{partial:true});
  const triggers=safetyTriggers(clean.answers,SYMPTOMES),t=now();
  await env.DB.batch([
   stmt(env,'INSERT INTO symptom_drafts(user_id,module,passation_key,content_version,answers,started_at,updated_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id,module) DO UPDATE SET passation_key=excluded.passation_key,content_version=excluded.content_version,answers=excluded.answers,started_at=COALESCE(excluded.started_at,symptom_drafts.started_at),updated_at=excluded.updated_at',user.id,moduleId,key,clean.content_version,JSON.stringify(clean.answers),started,t),
   ...alertStatements(env,user.id,key,moduleId,triggers)
  ]);
  return json({ok:true,draft:draftView(await first(env,'SELECT * FROM symptom_drafts WHERE user_id=? AND module=?',user.id,moduleId)),safety:{triggers,screen:safetyScreen(triggers,SYMPTOMES,{age:gate.age})}});
 }

 async function routes(request,env,user,path,url){
  if(path!=='/api/symptomes'&&!path.startsWith('/api/symptomes/'))return null;
  const method=request.method;
  if(path==='/api/symptomes'){if(method!=='GET')fail(405,'Méthode non autorisée.','method_not_allowed');return json(await readState(env,user.id))}
  if(path==='/api/symptomes/consent'){if(method!=='PUT')fail(405,'Méthode non autorisée.','method_not_allowed');return setConsent(request,env,user)}
  if(path==='/api/symptomes/checks'){
   if(method==='POST')return saveCheck(request,env,user);
   if(method==='GET'){
    if(!await symptomsReady(env))unavailable();
    const moduleId=url?.searchParams.get('module')||null;
    if(moduleId!==null&&!symptomsIndex(SYMPTOMES).modules.has(moduleId))fail(400,'Module inconnu.','unknown_module');
    const rows=moduleId?await all(env,'SELECT * FROM symptom_checks WHERE user_id=? AND module=? ORDER BY completed_at DESC,rowid DESC LIMIT ?',user.id,moduleId,SYMPTOMS_HISTORY_LIMIT):await all(env,'SELECT * FROM symptom_checks WHERE user_id=? ORDER BY completed_at DESC,rowid DESC LIMIT ?',user.id,SYMPTOMS_HISTORY_LIMIT);
    return json({checks:rows.map(checkView)});
   }
   if(method==='DELETE'){
    // « Supprimer mes résultats » : passations, brouillons et alertes. The consent stays as chosen.
    if(!await symptomsReady(env))unavailable();
    const [checks,drafts,alerts]=await env.DB.batch([stmt(env,'DELETE FROM symptom_checks WHERE user_id=?',user.id),stmt(env,'DELETE FROM symptom_drafts WHERE user_id=?',user.id),stmt(env,'DELETE FROM symptom_safety_alerts WHERE user_id=?',user.id)]);
    return json({ok:true,deleted:{checks:checks.meta.changes,drafts:drafts.meta.changes,alerts:alerts.meta.changes},page:await readState(env,user.id)});
   }
   fail(405,'Méthode non autorisée.','method_not_allowed');
  }
  const one=path.match(/^\/api\/symptomes\/checks\/([\w-]{1,120})$/);
  if(one){
   if(method!=='DELETE')fail(405,'Méthode non autorisée.','method_not_allowed');
   if(!await symptomsReady(env))unavailable();
   const r=await stmt(env,'DELETE FROM symptom_checks WHERE id=? AND user_id=?',one[1],user.id).run();
   if(!r.meta.changes)fail(404,'Passation introuvable.','not_found');
   return json({ok:true,page:await readState(env,user.id)});
  }
  const draft=path.match(/^\/api\/symptomes\/drafts\/([\w-]{1,64})$/);
  if(draft){
   if(method==='PUT')return saveDraft(request,env,user,draft[1]);
   if(method==='DELETE'){
    if(!await symptomsReady(env))unavailable();
    if(!symptomsIndex(SYMPTOMES).modules.has(draft[1]))fail(400,'Module inconnu.','unknown_module');
    await stmt(env,'DELETE FROM symptom_drafts WHERE user_id=? AND module=?',user.id,draft[1]).run();
    return json({ok:true});
   }
   fail(405,'Méthode non autorisée.','method_not_allowed');
  }
  fail(404,'Route inconnue.','not_found');
 }

 async function exportData(env,userId){
  if(!await symptomsReady(env))return {symptomes:null};
  return {symptomes:{content_version:SYMPTOMES.version,consent:await readConsent(env,userId),
   checks:(await all(env,'SELECT * FROM symptom_checks WHERE user_id=? ORDER BY completed_at,rowid',userId)).map(checkView),
   drafts:(await all(env,'SELECT * FROM symptom_drafts WHERE user_id=? ORDER BY module',userId)).map(draftView),
   safety_alerts:(await all(env,'SELECT * FROM symptom_safety_alerts WHERE user_id=? ORDER BY created_at,id',userId)).map(alertView)}};
 }
 return {routes,readState,exportData};
}
