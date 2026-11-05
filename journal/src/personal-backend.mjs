import { accessSql } from './entitlements.mjs';
import {PSYCHOLOGY_CATALOG,buildPsychologyMap} from './psychology-map.mjs';
import {expandReadingQuery,lookupReadingCatalogue} from './reading-intent.mjs';
import {PORTRAIT_ANALYSIS_VERSION} from './portrait-language.mjs';
import {loadContext,hasPersonalizationSources,validatePortraitV2,contextHash} from './private-context.mjs';
export {hasPersonalizationSources,validatePortraitV2};
const parse=(v,f={})=>{try{return JSON.parse(v);}catch{return f;}};
const dateInParis=date=>Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Paris',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(date).filter(p=>p.type!=='literal').map(p=>[p.type,p.value]));
const dateString=date=>{const p=dateInParis(date);return `${p.year}-${p.month}-${p.day}`;};
function parisMidnight(day){let epoch=Date.parse(day+'T00:00:00Z');for(let i=0;i<3;i++){const parts=Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Paris',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).formatToParts(new Date(epoch)).filter(p=>p.type!=='literal').map(p=>[p.type,p.value]));const displayed=Date.parse(`${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}Z`);epoch+=Date.parse(day+'T00:00:00Z')-displayed;}return new Date(epoch).toISOString();}
export function parisWeek(at=new Date()) {const local=dateString(at instanceof Date?at:new Date(at*1000)),d=new Date(local+'T12:00:00Z');d.setUTCDate(d.getUTCDate()-(d.getUTCDay()+6)%7);const week_start=d.toISOString().slice(0,10);d.setUTCDate(d.getUTCDate()+7);return {week_start,renews_at:parisMidnight(d.toISOString().slice(0,10)),time_zone:'Europe/Paris'};}
export function createPersonalBackend({error,json,text,body,now,random,decorateLikes,recommender,examples=[]}) {
 const first=(env,sql,...args)=>env.DB.prepare(sql).bind(...args).first();
 const all=async(env,sql,...args)=>(await env.DB.prepare(sql).bind(...args).all()).results;
 const run=(env,sql,...args)=>env.DB.prepare(sql).bind(...args).run();
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 const fail=(status,message,code)=>error(status,message,code);
 const paused=env=>['true','1','yes'].includes(String(env.GENERATION_PAUSED||'').toLowerCase());
 const checkPause=env=>{if(paused(env))fail(503,'L’analyse du portrait est momentanément suspendue.','generation_paused');};
 const context=async(env,userId)=>{let c;try{c=await loadContext(env,userId,decorateLikes);}catch(e){if(e.code==='context_changed')fail(409,e.message,e.code);throw e;}if(!c)fail(401,'Compte introuvable.','account_missing');return c;};
 const version=async(env,userId)=>(await first(env,'SELECT version FROM profiles WHERE user_id=?',userId)).version;
 const bump=(env,userId)=>stmt(env,'UPDATE profiles SET version=version+1 WHERE user_id=? AND changes()>0',userId);
 const choices=(value,list,name)=>{if(!list.includes(value))fail(400,`${name} invalide.`,'invalid_input');return value;};
 const string=(v,max=80)=>text(v??'',max);
 const revision=data=>{if(!Number.isInteger(data.revision)||data.revision<1)fail(400,'La révision actuelle est requise.','revision_required');return data.revision;};
 const creation=data=>{const id=string(data.creation_id,120);if(!/^[A-Za-z0-9_-]{8,120}$/.test(id))fail(400,'Identifiant de création requis.','creation_id_required');return id;};
 function personData(data){const p={display_name:string(data.display_name),role:choices(data.role??'other',['father','mother','parent','partner','child','sibling','friend','best_friend','other'],'Rôle'),custom_role:string(data.custom_role),relationship_text:string(data.relationship_text,6000),period_start:string(data.period_start),period_end:string(data.period_end)};if(!p.display_name)fail(400,'Indiquez un prénom ou pseudonyme.','invalid_input');return p;}
 function eventData(data){const e={title:string(data.title,160),kind:choices(data.kind??'other',['difficult','positive','turning_point','ambivalent','other'],'Type'),description:string(data.description,6000),meaning_now:string(data.meaning_now,2000),date_precision:choices(data.date_precision??'unknown',['unknown','year','month','day','range','age'],'Précision de date'),date_start:string(data.date_start,10),date_end:string(data.date_end,10),age_text:string(data.age_text,80),current_importance:data.current_importance??null};
 if(['unknown','age'].includes(e.date_precision)){e.date_start='';e.date_end='';}
 if(e.date_precision!=='range')e.date_end='';
 if(e.date_precision!=='age')e.age_text='';
 if(!e.title||!e.description)fail(400,'Un titre et une description sont requis.','invalid_input');
 if(e.current_importance!==null&&(!Number.isInteger(e.current_importance)||e.current_importance<1||e.current_importance>5))fail(400,'Importance invalide.','invalid_input');
 const validDate=v=>{if(!/^\d{4}(?:-\d{2}(?:-\d{2})?)?$/.test(v))return false;const parts=v.split('-'),[y,m,d]=parts.map(Number),end=new Date(0);end.setUTCFullYear(y,m||1,0);return y>=1&&y<=9999&&(parts.length<2||m>=1&&m<=12)&&(parts.length<3||d>=1&&d<=end.getUTCDate());};
 if(e.date_start&&!validDate(e.date_start)||e.date_end&&!validDate(e.date_end))fail(400,'Date partielle invalide.','invalid_date');
 const lengths={year:4,month:7,day:10};if(lengths[e.date_precision]&&e.date_start.length!==lengths[e.date_precision]||e.date_precision==='range'&&(!e.date_start||!e.date_end||e.date_start>e.date_end)||e.date_precision==='age'&&!e.age_text)fail(400,'La date ne correspond pas à sa précision.','invalid_date');
 if(!Array.isArray(data.people_ids??[])||(data.people_ids??[]).length>10||(data.people_ids??[]).some(id=>typeof id!=='string'||id.length>120))fail(400,'Associations invalides.','invalid_people');
 e.people_ids=[...new Set(data.people_ids??[])].sort();return e;}
 const narrativeKeys=['self_description','self_perception','recurring_patterns','emotions','current_context'];
 const selfFields=['first_name','last_name','birth_date','gender','gender_description',...narrativeKeys,'occupation'];
 const selfSql="EXISTS(SELECT 1 FROM json_each(p.self_data) sp WHERE sp.key IN ('self_description','self_perception','recurring_patterns','emotions','current_context') AND trim(sp.value)!='')";
 function validateSelfPatch(data,existing={}) {
   const profile={...existing};
   for(const key of selfFields)if(Object.hasOwn(data,key)){if(typeof data[key]!=='string')fail(400,'Texte attendu.','invalid_input');const max=key==='self_description'?10000:narrativeKeys.includes(key)||key==='occupation'?4000:key==='birth_date'?10:['first_name','last_name'].includes(key)?100:key==='gender_description'?200:40; if(data[key].length>max)fail(400,'Texte trop long.','invalid_input');profile[key]=data[key].trim();}
   if(profile.gender&&!['female','male','non_binary','self_described','unspecified'].includes(profile.gender))fail(400,'Genre invalide.','invalid_input');
   if(profile.birth_date&&(!/^\d{4}-\d{2}-\d{2}$/.test(profile.birth_date)||!Number.isFinite(Date.parse(profile.birth_date))||new Date(profile.birth_date).toISOString().slice(0,10)!==profile.birth_date||profile.birth_date>new Date().toISOString().slice(0,10)||profile.birth_date<'0001-01-01'))fail(400,'Date de naissance invalide.','invalid_date');
   return profile;
 }
 async function selfProfile(request,env,user){
  const old=await first(env,'SELECT self_data,self_revision FROM profiles WHERE user_id=?',user.id);
  if(request.method==='PUT'){
   const data=await body(request,40000);if(!Number.isInteger(data.expected_revision)||data.expected_revision<1)fail(400,'Révision requise.','revision_required');
   const profile=validateSelfPatch(data,parse(old.self_data));
   const serialized=JSON.stringify(profile),result=await run(env,'UPDATE profiles SET self_data=?,self_revision=self_revision+CASE WHEN self_data!=? THEN 1 ELSE 0 END,version=version+CASE WHEN self_data!=? THEN 1 ELSE 0 END WHERE user_id=? AND self_revision=?',serialized,serialized,serialized,user.id,data.expected_revision);
   if(!result.meta.changes)fail(409,'Le profil a changé. Rechargez-le.','revision_conflict');
  }
  const row=await first(env,'SELECT self_data,self_revision,version FROM profiles WHERE user_id=?',user.id);return json({profile:Object.fromEntries(selfFields.map(k=>[k,parse(row.self_data)[k]||''])),revision:row.self_revision,source_version:row.version});
 }
 const personRow=row=>{if(!row)return null;const {user_id,creation_id,payload_hash,fiction_use,...p}=row;return p;};
 const eventRow=async(env,row)=>row?{...personRow(row),people_ids:(await all(env,'SELECT person_id FROM life_event_people WHERE event_id=? AND user_id=? ORDER BY person_id',row.id,row.user_id)).map(r=>r.person_id)}:null;
 async function crud(request,env,user,path,url){const match=path.match(/^\/api\/(people|life-events)(?:\/([\w-]+))?$/);if(!match)return null;const isPerson=match[1]==='people',table=isPerson?'personal_people':'life_events',kind=isPerson?'person':'event',id=match[2],method=request.method;
 const transform=row=>isPerson?personRow(row):eventRow(env,row);
 if(method==='GET'&&id){const row=await first(env,`SELECT * FROM ${table} WHERE id=? AND user_id=?`,id,user.id);if(!row)fail(404,'Élément introuvable.','not_found');return json({[kind]:await transform(row),source_version:await version(env,user.id)});}
 if(method==='GET'&&!id){
  const cursor=string(url.searchParams.get('cursor')||'',120),limit=Number(url.searchParams.get('limit')||50);
  if(!Number.isInteger(limit)||limit<1||limit>50)fail(400,'Pagination invalide.','invalid_pagination');
  let rows;
  if(isPerson)rows=await all(env,'SELECT * FROM personal_people WHERE user_id=? AND id>? ORDER BY id LIMIT ?',user.id,cursor,limit+1);
  else{
   const dateOrder="CASE WHEN date_precision IN ('unknown','age') THEN '' ELSE date_start END";
   const anchor=cursor?await first(env,`SELECT id,${dateOrder} AS sort_date FROM life_events WHERE id=? AND user_id=?`,cursor,user.id):null;
   if(cursor&&!anchor)fail(409,'Ce repère de pagination a changé. Rechargez votre ligne de vie.','pagination_changed');
   rows=anchor?await all(env,`SELECT * FROM life_events WHERE user_id=? AND (${dateOrder}<? OR (${dateOrder}=? AND id>?)) ORDER BY ${dateOrder} DESC,id LIMIT ?`,user.id,anchor.sort_date,anchor.sort_date,anchor.id,limit+1):await all(env,`SELECT * FROM life_events WHERE user_id=? ORDER BY ${dateOrder} DESC,id LIMIT ?`,user.id,limit+1);
  }
  const more=rows.length>limit,items=await Promise.all(rows.slice(0,limit).map(transform));
  return json({[isPerson?'people':'events']:items,next_cursor:more?items.at(-1).id:null,source_version:await version(env,user.id)});
 }
 if(!['POST','PATCH','DELETE'].includes(method)||method==='POST'&&id||method!=='POST'&&!id)fail(405,'Méthode non autorisée.','method_not_allowed');
 const data=await body(request,45000);let prior=null,expected=null;
 if(id){prior=await first(env,`SELECT * FROM ${table} WHERE id=? AND user_id=?`,id,user.id);if(!prior)fail(404,'Élément introuvable.','not_found');expected=revision(data);if(expected!==prior.revision)fail(409,'Cet élément a été modifié dans un autre onglet.','revision_conflict');}
 if(method==='DELETE'){
 const statements=[stmt(env,`DELETE FROM ${table} WHERE id=? AND user_id=? AND revision=?`,id,user.id,expected),bump(env,user.id)];
 const results=await env.DB.batch(statements);if(!results[0].meta.changes)fail(409,'Cet élément a changé.','revision_conflict');return json({ok:true,[kind]:null,source_version:await version(env,user.id)});}
 const previous=prior?await transform(prior):{};const normalized=(isPerson?personData:eventData)({...previous,...data});const payloadHash=await contextHash(JSON.stringify(normalized));
 if(!isPerson&&normalized.people_ids.length){const found=await all(env,'SELECT id FROM personal_people WHERE user_id=? AND id IN(SELECT value FROM json_each(?))',user.id,JSON.stringify(normalized.people_ids));if(found.length!==normalized.people_ids.length)fail(400,'Un proche associé est introuvable.','invalid_people');}
 let entityId=id;
 if(method==='POST'){
 const key=creation(data),existing=await first(env,'SELECT * FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=?',user.id,kind,key);if(existing){
 const saved=await transform(await first(env,`SELECT * FROM ${table} WHERE user_id=? AND id=?`,user.id,existing.entity_id));
 if(existing.payload_hash!==payloadHash){
  // Previous creation hashes included the retired fiction setting.
  const {people_ids,...fields}=normalized;
  const legacyHashes=isPerson?[]:await Promise.all(['abstract','understanding_only'].map(fiction_use=>contextHash(JSON.stringify({...fields,fiction_use,people_ids}))));
  if(!legacyHashes.includes(existing.payload_hash))fail(409,'Cet identifiant a déjà servi pour un autre contenu.','idempotency_conflict');
 }
 return json({ok:true,[kind]:saved,source_version:await version(env,user.id)});
 }
 entityId=random();const fields=Object.keys(normalized).filter(k=>k!=='people_ids'),values=fields.map(k=>normalized[k]);const statements=[stmt(env,`INSERT OR IGNORE INTO ${table}(id,user_id,creation_id,payload_hash,${fields.join(',')}) SELECT ?,?,?,?,${fields.map(()=>'?').join(',')} WHERE (SELECT count(*) FROM ${table} WHERE user_id=?)<? AND NOT EXISTS(SELECT 1 FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=?)`,entityId,user.id,key,payloadHash,...values,user.id,isPerson?50:500,user.id,kind,key),bump(env,user.id),stmt(env,`INSERT OR IGNORE INTO context_mutations(user_id,kind,creation_id,payload_hash,entity_id) SELECT ?,?,?,?,id FROM ${table} WHERE user_id=? AND creation_id=?`,user.id,kind,key,payloadHash,user.id,key)];
 if(!isPerson)statements.push(stmt(env,`INSERT OR IGNORE INTO life_event_people(event_id,person_id,user_id) SELECT e.id,j.value,e.user_id FROM life_events e,json_each(?) j WHERE e.id=? AND e.user_id=?`,JSON.stringify(normalized.people_ids),entityId,user.id));
 await env.DB.batch(statements);const result=await first(env,'SELECT * FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=?',user.id,kind,key);if(!result)fail(400,'La limite de votre collection est atteinte.','collection_limit');if(result.payload_hash!==payloadHash)fail(409,'Cet identifiant a déjà servi pour un autre contenu.','idempotency_conflict');entityId=result.entity_id;
 }else if(payloadHash!==await contextHash(JSON.stringify((isPerson?personData:eventData)(previous)))){
 const fields=Object.keys(normalized).filter(k=>k!=='people_ids'),values=fields.map(k=>normalized[k]);const statements=[stmt(env,`UPDATE ${table} SET ${fields.map(k=>k+'=?').join(',')},payload_hash=?,revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=? AND revision=?`,...values,payloadHash,entityId,user.id,expected),bump(env,user.id)];
 // Mutation and link replacement share one transaction. The update's change
 // count is inspected before any link write; no external reader sees the gap.
 if(!isPerson){statements.push(stmt(env,'DELETE FROM life_event_people WHERE event_id=? AND user_id=? AND EXISTS(SELECT 1 FROM life_events WHERE id=? AND revision=? AND payload_hash=?)',entityId,user.id,entityId,expected+1,payloadHash));statements.push(stmt(env,'INSERT OR IGNORE INTO life_event_people(event_id,person_id,user_id) SELECT ?,value,? FROM json_each(?) WHERE EXISTS(SELECT 1 FROM life_events WHERE id=? AND user_id=? AND revision=? AND payload_hash=?)',entityId,user.id,JSON.stringify(normalized.people_ids),entityId,user.id,expected+1,payloadHash));}
 const results=await env.DB.batch(statements);if(!results[0].meta.changes)fail(409,'Cet élément a été modifié.','revision_conflict');
 }
 return json({ok:true,[kind]:await transform(await first(env,`SELECT * FROM ${table} WHERE id=? AND user_id=?`,entityId,user.id)),source_version:await version(env,user.id)},method==='POST'?201:200);
 }
 async function companionCapabilities(env,userId){
  const row=await first(env,`SELECT p.version,p.self_data,m.text AS memory_text,
   (SELECT count(*) FROM personal_people WHERE user_id=p.user_id) AS people_count,
   (SELECT count(*) FROM life_events WHERE user_id=p.user_id) AS events_count
   FROM profiles p JOIN memories m ON m.user_id=p.user_id WHERE p.user_id=?`,userId);
  if(!row)fail(401,'Compte introuvable.','account_missing');
  const self=parse(row.self_data),hasValue=key=>typeof self[key]==='string'&&Boolean(self[key].trim());
  const memoryLength=String(row.memory_text||'').length,remaining=Math.max(0,10000-memoryLength-(memoryLength?2:0));
  return {source_version:row.version,profile:{has_information:selfFields.some(hasValue),has_description:narrativeKeys.some(hasValue),first_name:typeof self.first_name==='string'?self.first_name.trim().slice(0,100):''},
   self_profile_patch:{allowed:true},memory_append:{allowed:remaining>0,remaining_characters:remaining,max_characters:2000},
   person_create:{allowed:row.people_count<50,count:row.people_count,limit:50},person_update:{allowed:row.people_count>0,count:row.people_count},
   life_event_create:{allowed:row.events_count<500,count:row.events_count,limit:500},life_event_update:{allowed:row.events_count>0,count:row.events_count},reading_lookup:{allowed:true}};
 }
 async function portrait(env,userId){const c=await context(env,userId),has=hasPersonalizationSources(c),job=await first(env,"SELECT * FROM journal_jobs WHERE user_id=? AND kind='portrait' AND source_version=? AND status NOT IN ('complete','obsolete') ORDER BY created_at DESC",userId,c.source_version),usage=c.access.allowed?null:await first(env,'SELECT * FROM portrait_usage WHERE user_id=?',userId);let status=has?(c.portrait_complete?'ready':job?.status==='running'&&job.lease_until>now()?'updating':job&&(job.status==='failed'||job.attempts>=3||job.provider_calls>=20)?'error':'queued'):'collecting';let retry_after=null,reason=null;if(!c.access.allowed&&usage&&usage.window_started+86400>now()){if(usage.starts>=3){retry_after=usage.window_started+86400;reason='quota';}else if(usage.last_started+600>now()){retry_after=usage.last_started+600;reason='cooldown';}}return {status,retry_after,reason,subject:c.subject,source_version:c.source_version,analyzed_version:c.portrait_version,updated_at:c.updated_at,portrait:c.portrait,outdated:has&&!c.portrait_complete,provisional:has&&status!=='ready',provisional_portrait:has?recommender.provisionalPortrait(c):null,psychology_map:buildPsychologyMap(c,examples)};}
 async function portraitFeedback(request,env,user){
  const data=await body(request,15000),patternId=string(data.pattern_id,100),pattern=PSYCHOLOGY_CATALOG.patterns.find(p=>p.id===patternId);
  if(!pattern)fail(400,'Piste inconnue.','invalid_pattern');
  const kind=choices(data.kind,['nuance','not_concerned'],'Type de retour'),correction=string(data.text,3000),key=creation(data);
  if(kind==='nuance'&&!correction)fail(400,'Précisez votre nuance.','feedback_text_required');
  const payloadHash=await contextHash(JSON.stringify({pattern_id:patternId,kind,text:correction}));
  const appended=`Correction personnelle sur « ${pattern.label} » : ${kind==='not_concerned'?'Je ne me reconnais pas dans cette piste.':'Je souhaite nuancer cette piste.'}${correction?' '+correction:''}`;
  // Registering the idempotency key, concatenating to the latest memory and
  // advancing both revisions happen in one transaction. Concurrent corrections
  // append to one another and never replace the existing user-written memory.
  const result=await env.DB.batch([
   stmt(env,`INSERT OR IGNORE INTO context_mutations(user_id,kind,creation_id,payload_hash,entity_id) SELECT ?,'psychology_feedback',?,?,? FROM memories WHERE user_id=? AND length(text)+length(?)+CASE WHEN trim(text)='' THEN 0 ELSE 2 END<=10000`,user.id,key,payloadHash,patternId,user.id,appended),
   stmt(env,`UPDATE memories SET text=text||CASE WHEN trim(text)='' THEN '' ELSE char(10)||char(10) END||?,proposal='',version=version+1 WHERE user_id=? AND changes()>0 AND EXISTS(SELECT 1 FROM context_mutations WHERE user_id=? AND kind='psychology_feedback' AND creation_id=? AND payload_hash=?)`,appended,user.id,user.id,key,payloadHash),
   bump(env,user.id)
  ]);
  const saved=await first(env,"SELECT payload_hash FROM context_mutations WHERE user_id=? AND kind='psychology_feedback' AND creation_id=?",user.id,key);
  if(!saved)fail(400,'La mémoire de précisions a atteint sa limite. Vous pouvez mettre à jour les informations concernées dans Mon portrait, À propos de moi.','memory_limit');
  if(saved.payload_hash!==payloadHash)fail(409,'Cet identifiant a déjà servi pour un autre retour.','idempotency_conflict');
  const memory=await first(env,'SELECT version FROM memories WHERE user_id=?',user.id);
  return json({ok:true,replayed:!result[0].meta.changes,feedback:{pattern_id:patternId,kind},source_version:await version(env,user.id),memory_version:memory.version});
 }
 async function routePersonal(request,env,user,path,url){
  if(path==='/api/portrait/feedback'&&request.method==='POST')return portraitFeedback(request,env,user);
  if(path==='/api/self-profile'&&['GET','PUT'].includes(request.method))return selfProfile(request,env,user);
  return crud(request,env,user,path,url);
 }
 async function routes(request,env,user,path,url,ctx){
  const response=await routePersonal(request,env,user,path,url);
  if(response&&['PUT','POST','PATCH','DELETE'].includes(request.method)&&(/^\/api\/(self-profile|people|life-events)(?:\/|$)/.test(path)||path==='/api/portrait/feedback'))await sourcesChanged(env,user.id,ctx);
  return response;
 }
 const sourcesSql=`(${selfSql} OR EXISTS(SELECT 1 FROM article_likes l WHERE l.user_id=p.user_id) OR trim(m.text)!='' OR EXISTS(SELECT 1 FROM personal_people pp WHERE pp.user_id=p.user_id AND trim(pp.relationship_text)!='') OR EXISTS(SELECT 1 FROM life_events e WHERE e.user_id=p.user_id AND (trim(e.description)!='' OR trim(e.meaning_now)!='')))`;
 const guard=`EXISTS(SELECT 1 FROM journal_jobs j JOIN profiles p ON p.user_id=j.user_id JOIN memories m ON m.user_id=j.user_id WHERE j.id=? AND j.lease_token=? AND j.kind='portrait' AND j.status='running' AND j.lease_until>? AND j.source_version=p.version AND j.memory_version=m.version AND ${sourcesSql})`;
 const guardArgs=(id,lease)=>[id,lease,now()];
 async function validJob(env,id,data){
  checkPause(env);
  const job=await first(env,"SELECT * FROM journal_jobs WHERE id=? AND kind='portrait'",id);
  if(!job||job.lease_token!==data.lease_token||job.status!=='running'||job.lease_until<=now())fail(409,'Bail expiré ou inconnu.','lease_expired');
  if(data.source_version!==undefined&&data.source_version!==job.source_version||data.memory_version!==undefined&&data.memory_version!==job.memory_version)fail(409,'Le contexte a changé.','context_changed');
  if(!await first(env,`SELECT 1 AS ok WHERE ${guard}`,...guardArgs(id,data.lease_token)))fail(409,'Le contexte a changé.','context_changed');
  return job;
 }
 const portraitBudget=env=>Number(env.PORTRAIT_TOKEN_BUDGET||env.STORY_TOKEN_BUDGET)||300000;
 async function claimJobs(env,userId=null,kind=null,leaseSeconds=1800,maxAttempts=3,native=false,resumeCached=false){
 const attemptLimit=Math.max(1,Math.min(3,Math.floor(Number(maxAttempts)||3)));
 if(kind!==null&&kind!=='portrait')return json({jobs:[]});
 if(paused(env))return json({jobs:[],status:'generation_paused'});
 if(!env.OPENROUTER_API_KEY)return json({jobs:[],status:'provider_missing'});
 const t=now();
 await env.DB.batch([
 stmt(env,"UPDATE journal_jobs SET memory_version=(SELECT version FROM memories m WHERE m.user_id=journal_jobs.user_id) WHERE kind='portrait' AND status='pending'"),
 stmt(env,"UPDATE journal_jobs SET status='pending',attempts=0,provider_calls=0,reserved_tokens=0,lease_token=NULL,lease_until=0,error_code=NULL,created_at=CURRENT_TIMESTAMP WHERE kind='portrait' AND status NOT IN ('complete','obsolete') AND (attempts>=3 OR provider_calls>=20 OR error_code='generation_budget') AND lease_until<=? AND created_at<=datetime('now','-24 hours')",t),
 stmt(env,`INSERT OR IGNORE INTO journal_jobs(id,user_id,kind,source_version,memory_version) SELECT lower(hex(randomblob(16))),p.user_id,'portrait',p.version,m.version FROM profiles p JOIN memories m ON m.user_id=p.user_id LEFT JOIN user_portraits up ON up.user_id=p.user_id WHERE ${sourcesSql} AND (up.schema_version IS NULL OR up.schema_version!=2 OR up.analyzed_version!=p.version OR COALESCE(json_array_length(json_extract(up.portrait,'$.coverage.pending')),0)>0 OR json_type(up.portrait,'$.psychological_map') IS NULL)`) ]);
 const candidates=await all(env,`SELECT j.* FROM journal_jobs j JOIN profiles p ON p.user_id=j.user_id JOIN memories m ON m.user_id=j.user_id JOIN subscriptions s ON s.user_id=j.user_id LEFT JOIN portrait_usage pu ON pu.user_id=j.user_id
 WHERE j.kind='portrait' AND (? IS NULL OR j.user_id=?) AND (?=0 OR COALESCE(j.error_code,'')!='batch_required') AND (?=0 OR j.attempts=0 OR j.native_candidate IS NOT NULL OR j.error_code='validation_error') AND j.status IN ('pending','running') AND j.lease_until<=? AND j.attempts<${attemptLimit} AND j.provider_calls<20 AND j.source_version=p.version AND ${sourcesSql}
 AND (${accessSql(env,'s')} OR pu.user_id IS NULL OR pu.window_started<=? OR pu.starts<3 AND pu.last_started<=?)
 ORDER BY CASE WHEN ${accessSql(env,'s')} THEN 0 ELSE 1 END,j.created_at,j.id LIMIT 30`,userId,userId,Number(native),Number(resumeCached),t,t,t-86400,t-600,t);
 for(const candidate of candidates){try{
 const c=await context(env,candidate.user_id);if(c.source_version!==candidate.source_version||!hasPersonalizationSources(c))continue;
 if(native&&JSON.stringify(c.sources).length>130000){
  // The full Python pipeline extracts large sources in batches. Leave its job
  // available without charging a start, an attempt or a provider reservation.
  await run(env,"UPDATE journal_jobs SET status='pending',lease_token=NULL,lease_until=0,error_code='batch_required' WHERE id=? AND source_version=? AND status IN ('pending','running') AND lease_until<=? AND source_version=(SELECT version FROM profiles WHERE user_id=journal_jobs.user_id)",candidate.id,c.source_version,t);
  continue;
 }
 const lease=random()+random(),paid=c.access.allowed;
 const statements=[stmt(env,`UPDATE journal_jobs SET status='running',attempts=attempts+1,lease_token=?,lease_until=?,memory_version=?,error_code=NULL WHERE id=? AND kind='portrait' AND status IN ('pending','running') AND lease_until<=? AND attempts<${attemptLimit} AND provider_calls<20 AND source_version=(SELECT version FROM profiles WHERE user_id=journal_jobs.user_id) AND (${paid?1:0}=1 OR NOT EXISTS(SELECT 1 FROM portrait_usage pu WHERE pu.user_id=journal_jobs.user_id AND pu.window_started>? AND (pu.starts>=3 OR pu.last_started>?)))`,lease,t+Math.min(1800,Math.max(90,leaseSeconds)),c.memory.version,candidate.id,t,t-86400,t-600)];
 if(!paid)statements.push(stmt(env,`INSERT INTO portrait_usage(user_id,window_started,starts,last_started) SELECT ?,?,1,? WHERE EXISTS(SELECT 1 FROM journal_jobs WHERE id=? AND lease_token=?) ON CONFLICT(user_id) DO UPDATE SET window_started=CASE WHEN window_started<=? THEN excluded.window_started ELSE window_started END,starts=CASE WHEN window_started<=? THEN 1 ELSE starts+1 END,last_started=excluded.last_started`,candidate.user_id,t,t,candidate.id,lease,t-86400,t-86400));
 const result=await env.DB.batch(statements);if(!result[0].meta.changes)continue;
 const base={schema_version:2,id:candidate.id,kind:candidate.kind,provider_calls:candidate.provider_calls,attempts:candidate.attempts+1,reserved_tokens:candidate.reserved_tokens,token_budget:portraitBudget(env),user_id:candidate.user_id,day:dateString(new Date()),lease_token:lease,source_version:c.source_version,profile_version:c.source_version,memory_version:c.memory.version,self_context:c.self_context,memory:c.memory,likes:c.likes,people:c.people,events:c.events,sources:c.sources,source_manifest:c.source_manifest,portrait_previous:null,portrait_previous_version:0,native_candidate:parse(candidate.native_candidate,null)};
 // Deleted/edited sources are removed from active extraction caches before reuse.
 await run(env,`DELETE FROM context_source_derivatives WHERE user_id=? AND NOT EXISTS(SELECT 1 FROM json_each(?) m WHERE m.key=source_id AND json_extract(m.value,'$.revision')=revision AND json_extract(m.value,'$.hash')=source_hash)`,candidate.user_id,JSON.stringify(c.source_manifest));
 base.source_derivatives=(await all(env,'SELECT source_id,revision,source_hash,analysis_version,derivative,manifest FROM context_source_derivatives WHERE user_id=? AND analysis_version=2',candidate.user_id)).map(d=>({...d,derivative:parse(d.derivative),manifest:parse(d.manifest)}));
 return json({jobs:[base]});
 }catch(e){if(e?.status)continue;throw e;}}
 return json({jobs:[]});}
 async function completeJob(env,id,data){
 const sourceVersion=data.source_version??data.profile_version;if(!Number.isInteger(sourceVersion)||!Number.isInteger(data.memory_version))fail(400,'Versions du contexte requises.','versions_required');const job=await validJob(env,id,{...data,source_version:sourceVersion}),args=guardArgs(id,data.lease_token);
 const c=await context(env,job.user_id),p=validatePortraitV2({...data.portrait,analysis_version:PORTRAIT_ANALYSIS_VERSION,identity_context:c.self_context},c.sources,fail),statements=[];
 const derivatives=data.source_derivatives??[];if(!Array.isArray(derivatives)||derivatives.length>1000)fail(400,'Extractions invalides.','invalid_derivative');
 for(const d of derivatives){const s=c.sources.find(s=>s.id===d.source_id);if(!s||d.revision!==s.revision||d.source_hash!==s.hash||d.analysis_version!==2||!Array.isArray(d.derivative?.signals)||d.derivative.signals.length>50)fail(400,'Extraction périmée.','invalid_derivative');for(const signal of d.derivative.signals){if(typeof signal.quote!=='string'||!signal.quote.trim()||signal.quote.length>1200||!s.text.includes(signal.quote)||typeof signal.text!=='string'||signal.text.length>2000)fail(400,'Extraction sans preuve.','invalid_derivative');}const manifest={[s.id]:c.source_manifest[s.id]};statements.push(stmt(env,`INSERT INTO context_source_derivatives(user_id,source_id,revision,analysis_version,source_hash,derivative,manifest) SELECT ?,?,?,?,?,?,? WHERE ${guard} ON CONFLICT(user_id,source_id,revision,analysis_version) DO UPDATE SET derivative=excluded.derivative,manifest=excluded.manifest,source_hash=excluded.source_hash`,job.user_id,s.id,s.revision,2,s.hash,JSON.stringify(d.derivative),JSON.stringify(manifest),...args));}
 statements.push(stmt(env,`INSERT INTO user_portraits(user_id,portrait,analyzed_version,schema_version) SELECT ?,?,?,2 WHERE ${guard} ON CONFLICT(user_id) DO UPDATE SET portrait=excluded.portrait,analyzed_version=excluded.analyzed_version,schema_version=2,updated_at=CURRENT_TIMESTAMP`,job.user_id,JSON.stringify(p),job.source_version,...args));
 statements.push(stmt(env,`UPDATE journal_jobs SET status='complete',lease_until=0,native_candidate=NULL WHERE id=? AND lease_token=? AND ${guard}`,id,data.lease_token,...args));const results=await env.DB.batch(statements);if(!results.at(-1).meta.changes)fail(409,'Le contexte a changé.','context_changed');return json({ok:true,kind:'portrait',source_version:job.source_version,analyzed_version:job.source_version});}
 async function failJob(env,id,data,{retryAfterSeconds=null}={}){
  const job=await first(env,"SELECT * FROM journal_jobs WHERE id=? AND kind='portrait' AND lease_token=? AND status='running' AND lease_until>?",id,string(data.lease_token,100),now());
  if(!job)fail(409,'Bail expiré ou inconnu.','lease_expired');
  const suppliedReason=data.reason==='quota_exceeded'?'generation_budget':data.reason;
  const reason=['provider_error','validation_error','coherence_error','unexpected_error','context_changed','generation_budget'].includes(suppliedReason)?suppliedReason:'unexpected_error';
  const terminal=reason==='generation_budget'||job.attempts>=3||job.provider_calls>=20;
  const wait=now()+(retryAfterSeconds===null?Math.min(3600,300*2**(job.attempts-1)):Math.min(3600,Math.max(1,retryAfterSeconds)));
  const result=await run(env,"UPDATE journal_jobs SET status=?,lease_until=?,lease_token=NULL,error_code=? WHERE id=? AND kind='portrait' AND lease_token=? AND status='running'",terminal?'failed':'pending',terminal?0:wait,reason,id,data.lease_token);
  if(!result.meta.changes)fail(409,'Ce bail n’est plus actif.','context_changed');
  return json({ok:true});
 }
 async function proxyJobLLM(env,id,data,{timeoutMs=180000}={}){if(!env.OPENROUTER_API_KEY)fail(503,'Fournisseur indisponible.','provider_missing');const job=await validJob(env,id,{...data,source_version:data.source_version??data.profile_version});const allowed=[env.OPENROUTER_MODEL||'openai/gpt-5-mini',env.OPENROUTER_ANALYST_MODEL,env.OPENROUTER_PORTRAIT_MODEL].filter(Boolean),model=data.model||env.OPENROUTER_ANALYST_MODEL||allowed[0];if(!allowed.includes(model))fail(400,'Modèle non autorisé.','invalid_model');const system=text(data.system_prompt,50000),prompt=text(data.user_prompt,500000);if(!system||!prompt)fail(400,'Prompts requis.','invalid_prompt');const maxTokens=data.max_tokens??6000,temperature=data.temperature??0.2;if(!Number.isInteger(maxTokens)||maxTokens<128||maxTokens>8000||typeof temperature!=='number'||!Number.isFinite(temperature)||temperature<0||temperature>1)fail(400,'Paramètres de génération invalides.','invalid_parameters');const reservation=Math.ceil((system.length+prompt.length)/3)+maxTokens,requestLimit=portraitBudget(env),weeklyLimit=Number(env.WEEKLY_TOKEN_BUDGET)||3000000,w=parisWeek().week_start;
 const result=await env.DB.batch([
 stmt(env,`UPDATE journal_jobs SET provider_calls=provider_calls+1,reserved_tokens=reserved_tokens+? WHERE id=? AND provider_calls<20 AND reserved_tokens+?<=? AND ${guard} AND COALESCE((SELECT reserved_tokens FROM provider_budgets b WHERE b.user_id=journal_jobs.user_id AND b.week_start=?),0)+?<=?`,reservation,id,reservation,requestLimit,...guardArgs(id,data.lease_token),w,reservation,weeklyLimit),
 stmt(env,`INSERT INTO provider_budgets(user_id,week_start,provider_calls,reserved_tokens) SELECT ?,?,1,? WHERE changes()>0 ON CONFLICT(user_id,week_start) DO UPDATE SET provider_calls=provider_calls+1,reserved_tokens=reserved_tokens+excluded.reserved_tokens`,job.user_id,w,reservation)]);
 if(!result[0].meta.changes)fail(429,'Budget de génération atteint ou contexte expiré.','generation_budget');let response;try{response=await fetch('https://openrouter.ai/api/v1/chat/completions',{method:'POST',headers:{Authorization:`Bearer ${env.OPENROUTER_API_KEY}`,'Content-Type':'application/json','HTTP-Referer':'https://decodetonesprit.fr','X-Title':'Decode ton esprit'},body:JSON.stringify({model,messages:[{role:'system',content:system},{role:'user',content:prompt}],...(model==='anthropic/claude-sonnet-5.5'?{}:{temperature}),reasoning:{effort:'low'},max_tokens:maxTokens,provider:{data_collection:'deny',zdr:true,allow_fallbacks:true}}),signal:AbortSignal.timeout(Math.min(timeoutMs,180000,Math.max(1000,(job.lease_until-now())*1000-5000)))});}catch{fail(502,'Fournisseur momentanément indisponible.','provider_error');}if(!response.ok)fail(502,'Fournisseur momentanément indisponible.','provider_error');let resultBody;try{resultBody=await response.json();}catch{fail(502,'Réponse fournisseur invalide.','provider_error');}const usage=resultBody?.usage||{},input=Number.isFinite(usage.prompt_tokens)?Math.max(0,Math.floor(usage.prompt_tokens)):0,output=Number.isFinite(usage.completion_tokens)?Math.max(0,Math.floor(usage.completion_tokens)):0,cost=Number.isFinite(usage.cost)?Math.max(0,usage.cost):0;
 await env.DB.batch([stmt(env,'UPDATE journal_jobs SET input_tokens=input_tokens+?,output_tokens=output_tokens+?,observed_cost=observed_cost+? WHERE id=?',input,output,cost,id),stmt(env,'UPDATE provider_budgets SET input_tokens=input_tokens+?,output_tokens=output_tokens+?,observed_cost=observed_cost+? WHERE user_id=? AND week_start=?',input,output,cost,job.user_id,w)]);
 const content=resultBody?.choices?.[0]?.message?.content;if(typeof content!=='string'||!content.trim()||content.length>180000||resultBody.choices[0].finish_reason==='length')fail(502,'Réponse fournisseur incomplète.','provider_error');return json({content,usage:{input_tokens:input,output_tokens:output,observed_cost:cost}});}
 // Persist the latest portrait request before starting background work. Source
 // version triggers revoke older leases, so a late model result cannot publish.
 async function sourcesChanged(env,userId,ctx){
  if(paused(env)||!env.OPENROUTER_API_KEY)return;
  const delay=ctx?.waitUntil?3:0,t=now();
  await run(env,`INSERT OR IGNORE INTO journal_jobs(id,user_id,kind,source_version,memory_version,lease_until) SELECT ?,p.user_id,'portrait',p.version,m.version,? FROM profiles p JOIN memories m ON m.user_id=p.user_id LEFT JOIN user_portraits up ON up.user_id=p.user_id WHERE p.user_id=? AND ${sourcesSql} AND (up.schema_version IS NULL OR up.schema_version!=2 OR up.analyzed_version!=p.version OR COALESCE(json_array_length(json_extract(up.portrait,'$.coverage.pending')),0)>0 OR json_type(up.portrait,'$.psychological_map') IS NULL)`,random(),t+delay,userId);
  const job=await first(env,"SELECT source_version,status,lease_until,attempts,provider_calls,native_candidate,error_code FROM journal_jobs WHERE user_id=? AND kind='portrait' AND source_version=(SELECT version FROM profiles WHERE user_id=?) AND status IN ('pending','running') AND COALESCE(error_code,'')!='batch_required' ORDER BY created_at DESC LIMIT 1",userId,userId);
  if(!ctx?.waitUntil||!job||job.attempts>=3||job.attempts>0&&!job.native_candidate&&job.error_code!=='validation_error'||job.provider_calls>=20||job.status==='running'&&job.lease_until>t)return;
  // A queue lease is also a durable debounce deadline; GET recovery never
  // extends it. Concurrent tasks race on the same atomic claim, not provider calls.
  const wait=Math.max(0,job.lease_until-t);
  if(wait>5)return;
  ctx.waitUntil((async()=>{if(wait)await new Promise(resolve=>setTimeout(resolve,wait*1000));await refreshUserPortrait(env,userId);})().catch(()=>{}));
 }
 async function refreshUserPortrait(env,userId=null,{timeoutMs=22000}={}){
  const claimed=await (await claimJobs(env,userId,'portrait',Math.max(120,Math.ceil(timeoutMs/1000)+30),3,true,timeoutMs<=22000)).json(),job=claimed.jobs[0];
  if(!job)return {claimed:0,completed:0,failed:0,status:claimed.status||'idle'};
  try{
   const {analyzePortraitSnapshot}=await import('./portrait-refresh.mjs');
   const deadline=Date.now()+timeoutMs;
   const result=await analyzePortraitSnapshot(job,async payload=>{
    const remaining=deadline-Date.now();if(remaining<(payload.stage==='support'?8000:1000))throw Object.assign(new Error('Provider deadline'),{code:'provider_timeout'});
    const response=await proxyJobLLM(env,job.id,{...payload,model:env.OPENROUTER_PORTRAIT_MODEL||env.OPENROUTER_ANALYST_MODEL,lease_token:job.lease_token,source_version:job.source_version,memory_version:job.memory_version},{timeoutMs:remaining});
    return (await response.json()).content;
   },{saveCandidate:async portrait=>{
    await validJob(env,job.id,{lease_token:job.lease_token,source_version:job.source_version,memory_version:job.memory_version});
    // The candidate is private intermediate data. It is never served as a
    // portrait before semantic verification; both revision and lease must match.
    const saved=await run(env,`UPDATE journal_jobs SET native_candidate=? WHERE id=? AND kind='portrait' AND ${guard}`,JSON.stringify(portrait),job.id,...guardArgs(job.id,job.lease_token));
    if(!saved.meta.changes)fail(409,'Le contexte a changé.','context_changed');
   }});
   await completeJob(env,job.id,{...result,lease_token:job.lease_token,source_version:job.source_version,memory_version:job.memory_version});
   return {claimed:1,completed:1,failed:0,status:'complete'};
  }catch(e){
   if(e?.code==='support_context_budget'){
    // A context-size limit needs the batch analyst, not three identical native retries.
    const released=await run(env,`UPDATE journal_jobs SET status='pending',lease_token=NULL,lease_until=0,error_code='batch_required',native_candidate=NULL WHERE id=? AND ${guard}`,job.id,...guardArgs(job.id,job.lease_token));
    return {claimed:1,completed:0,failed:0,status:released.meta.changes?'batch_required':'context_changed'};
   }
   const reason=['context_changed','lease_expired'].includes(e?.code)?'context_changed':['generation_budget','quota_exceeded'].includes(e?.code)||e?.message==='generation_budget'?'generation_budget':['invalid_portrait','validation_error'].includes(e?.code)||e?.message==='validation_error'?'validation_error':'provider_error';
   // A rejected claim must be redrafted on the next long retry. Transport and
   // deadline failures keep the candidate so GET can resume verification only.
   if(reason==='validation_error')try{const diagnostic=Array.isArray(e.verification_failures)?JSON.stringify({analysis_version:PORTRAIT_ANALYSIS_VERSION,rejected:true,failed_paths:e.verification_failures}):null;await run(env,`UPDATE journal_jobs SET native_candidate=? WHERE id=? AND ${guard}`,diagnostic,job.id,...guardArgs(job.id,job.lease_token));}catch{}
   try{await failJob(env,job.id,{lease_token:job.lease_token,reason},{retryAfterSeconds:reason==='validation_error'||timeoutMs<=22000?30:null});}catch{}
   return {claimed:1,completed:0,failed:1,status:reason};
  }
 }
 async function refreshPendingPortraits(env){
  const result={status:'worker-managed',claimed:0,completed:0,failed:0};
  if(paused(env)||!env.OPENROUTER_API_KEY)return {...result,status:paused(env)?'generation_paused':'provider_missing'};
  // A bounded safety pass recovers jobs if the request's waitUntil was lost.
  for(let index=0;index<3;index++){const next=await refreshUserPortrait(env,null,{timeoutMs:120000});for(const key of ['claimed','completed','failed'])result[key]+=next[key];if(!next.claimed)break;}
  return result;
 }
 async function exportData(env,userId){
  const [self_profile,recommendation_exposures,people,events,characters,story_requests,stories,feedback]=await Promise.all([
   first(env,'SELECT self_data,self_revision FROM profiles WHERE user_id=?',userId),
   all(env,'SELECT article_id,shown_at FROM recommendation_exposures WHERE user_id=? ORDER BY shown_at DESC LIMIT 100',userId),
   all(env,'SELECT * FROM personal_people WHERE user_id=? ORDER BY id',userId),
   all(env,'SELECT * FROM life_events WHERE user_id=? ORDER BY id',userId),
   all(env,'SELECT * FROM personal_characters WHERE user_id=? ORDER BY slot',userId),
   all(env,'SELECT * FROM story_requests WHERE user_id=? ORDER BY created_at,id',userId),
   all(env,'SELECT * FROM stories WHERE user_id=? ORDER BY character_id,episode_number',userId),
   all(env,'SELECT * FROM feedback WHERE user_id=?',userId)
  ]);
  return {self_profile,recommendation_exposures,people:people.map(personRow),
   events:await Promise.all(events.map(async row=>({...await eventRow(env,row),fiction_use:row.fiction_use}))),
   // Retired private material remains available only as an account data export.
   characters:characters.map(({user_id,...row})=>({...row,identity:parse(row.identity),facet:parse(row.facet),narrative_state:parse(row.narrative_state)})),
   story_requests,stories,feedback};
 }
 function readingSuggestionsFromContext(c,{query='',limit=6}={}){
  // This helper accepts only the authenticated, request-local server snapshot.
  // The current question is a reading request, never a new saved profile source.
  const allowed=new Map(examples.map(e=>[e.id,e]));
  return recommender.recommend(c,{limit:Math.max(1,Math.min(6,Number(limit)||6)),includeLiked:true,query:expandReadingQuery(query)}).flatMap(item=>{
   const e=allowed.get(item.id);return e?[{id:e.id,title:e.title,url:e.url,character:e.character,reason:(item.reasons||[]).join(' ').replace('Votre contexte déclaré','Ce que vous partagez')}]:[];
  });
 }
 async function readingSuggestions(env,userId,options={}){return readingSuggestionsFromContext(await context(env,userId),options);}
 const readingLookup=(query,options={})=>lookupReadingCatalogue(examples,query,options);

 return {validateSelfPatch,personData,eventData,context,portrait,companionCapabilities,readingSuggestions,readingSuggestionsFromContext,readingLookup,routes,claimJobs,completeJob,proxyJobLLM,failJob,exportData,sourcesChanged,refreshUserPortrait,refreshPendingPortraits};
}
