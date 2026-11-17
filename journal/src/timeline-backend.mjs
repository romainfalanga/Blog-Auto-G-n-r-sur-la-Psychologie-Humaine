import {contextHash} from './private-context.mjs';
import {timelineReady,noteRow,stripPrivate,eventRows,iso,NOTE_TEXT_MAX} from './timeline-data.mjs';
import {peopleObservationReady} from './people-observations.mjs';
const MAX_NOTES=2000,MAX_PARENT_NOTES=200;
export function createTimelineBackend({error,json,body,now,random}){
 const first=(env,sql,...args)=>env.DB.prepare(sql).bind(...args).first();
 const all=async(env,sql,...args)=>(await env.DB.prepare(sql).bind(...args).all()).results;
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 const version=async(env,userId)=>(await first(env,'SELECT version FROM profiles WHERE user_id=?',userId)).version;
 const fail=(status,message,code)=>error(status,message,code);
 const limit=url=>{const n=Number(url.searchParams.get('limit')||30);if(!Number.isInteger(n)||n<1||n>50)fail(400,'Pagination invalide.','invalid_pagination');return n};
 const key=value=>{if(typeof value!=='string'||!/^[A-Za-z0-9_:-]{8,120}$/.test(value))fail(400,'Identifiant de création ou de modification requis.','creation_id_required');return value};
 const target=(collection,id)=>({type:collection==='people'?'person':'event',table:collection==='people'?'personal_people':'life_events',column:collection==='people'?'person_id':'event_id',id});
 async function parent(env,userId,t){const row=await first(env,`SELECT * FROM ${t.table} WHERE id=? AND user_id=?`,t.id,userId);if(!row)fail(404,'Élément introuvable.','not_found');return row;}
 async function listNotes(env,userId,t,url){
  const sourceVersion=await version(env,userId),size=limit(url),cursor=url.searchParams.get('cursor')||'',available=await timelineReady(env);
  if(!available)return json({notes:[],total:0,next_cursor:null,available:false,source_version:await version(env,userId)});
  const anchor=cursor?await first(env,`SELECT created_at,id FROM personal_notes WHERE user_id=? AND ${t.column}=? AND id=?`,userId,t.id,cursor):null;
  if(cursor&&!anchor)fail(409,'Ce repère a changé. Rechargez les notes.','pagination_changed');
  const rows=await all(env,`SELECT * FROM personal_notes WHERE user_id=? AND ${t.column}=? ${anchor?'AND (created_at<? OR (created_at=? AND id>?))':''} ORDER BY created_at DESC,id LIMIT ?`,userId,t.id,...(anchor?[anchor.created_at,anchor.created_at,anchor.id]:[]),size+1);
  const total=(await first(env,`SELECT count(*) n FROM personal_notes WHERE user_id=? AND ${t.column}=?`,userId,t.id)).n;
  if(await version(env,userId)!==sourceVersion)fail(409,'Les notes ont changé. Rechargez-les.','pagination_changed');
  return json({notes:rows.slice(0,size).map(noteRow),total,next_cursor:rows.length>size?rows[size-1].id:null,available:true,source_version:sourceVersion});
 }
 async function mutate(request,env,userId,t,id){
  if(!await timelineReady(env))fail(503,'Les notes datées ne sont pas encore disponibles.','feature_unavailable');
  const data=await body(request,16000),method=request.method;
  if(!data||typeof data!=='object'||Array.isArray(data))fail(400,'Contenu invalide.','invalid_input');
  const allowed=method==='POST'?['creation_id','text']:method==='PATCH'?['revision','mutation_id','text']:['revision','mutation_id'];
  if(Object.keys(data).some(k=>!allowed.includes(k)))fail(400,'Champ de note non autorisé.','invalid_input');
  let text='';if(method!=='DELETE'){if(typeof data.text!=='string'||!data.text.trim()||data.text.trim().length>NOTE_TEXT_MAX)fail(400,'La note doit contenir entre 1 et 2 000 caractères.','invalid_input');text=data.text.trim();}
  if(method!=='POST'&&(!Number.isInteger(data.revision)||data.revision<1))fail(400,'La révision actuelle est requise.','revision_required');
  const hash=await contextHash(JSON.stringify({target_type:t.type,target_id:t.id,id:id||null,revision:data.revision||null,text}));
  const mutationKey=method==='POST'?key(data.creation_id):key(data.mutation_id||id+':'+data.revision+':'+method.toLowerCase()+':'+hash.slice(0,20));
  const kind='personal_note_'+method.toLowerCase();
  const replay=async receipt=>{
   if(receipt.payload_hash!==hash)fail(409,'Cet identifiant a déjà servi pour un autre contenu.','idempotency_conflict');
   const row=await first(env,`SELECT * FROM personal_notes WHERE user_id=? AND ${t.column}=? AND id=?`,userId,t.id,receipt.entity_id);
   if(method==='POST'&&!row)fail(410,'Cette note a été supprimée.','note_deleted');
   return json({ok:true,...(method==='DELETE'?{deleted:true,id:receipt.entity_id}:{note:row?noteRow(row):null,...(!row?{deleted:true}:{})}),replayed:true,parent_revision:(await parent(env,userId,t)).revision,source_version:await version(env,userId)});
  };
  const oldReceipt=await first(env,'SELECT * FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=?',userId,kind,mutationKey);
  if(oldReceipt)return replay(oldReceipt);
  if(method==='POST'){
   const noteId=random(),at=now();
   await env.DB.batch([
    stmt(env,`INSERT OR IGNORE INTO personal_notes(id,user_id,${t.column},creation_id,payload_hash,text,created_at,updated_at) SELECT ?,?,?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM ${t.table} WHERE id=? AND user_id=?) AND (SELECT count(*) FROM personal_notes WHERE user_id=?)<? AND (SELECT count(*) FROM personal_notes WHERE user_id=? AND ${t.column}=?)<? AND NOT EXISTS(SELECT 1 FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=?)`,noteId,userId,t.id,mutationKey,hash,text,at,at,t.id,userId,userId,MAX_NOTES,userId,t.id,MAX_PARENT_NOTES,userId,kind,mutationKey),
    stmt(env,`INSERT OR IGNORE INTO context_mutations(user_id,kind,creation_id,payload_hash,entity_id) SELECT user_id,?,creation_id,payload_hash,id FROM personal_notes WHERE user_id=? AND creation_id=?`,kind,userId,mutationKey)
   ]);
   const receipt=await first(env,'SELECT * FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=?',userId,kind,mutationKey);
   if(!receipt)fail(409,'La cible a changé ou la limite de notes est atteinte.','note_limit_or_target_changed');
   if(receipt.payload_hash!==hash)fail(409,'Cet identifiant a déjà servi pour un autre contenu.','idempotency_conflict');
   const row=await first(env,`SELECT * FROM personal_notes WHERE id=? AND user_id=? AND ${t.column}=?`,receipt.entity_id,userId,t.id);
   if(!row)fail(410,'Cette note a été supprimée.','note_deleted');
   return json({ok:true,note:noteRow(row),parent_revision:(await parent(env,userId,t)).revision,source_version:await version(env,userId)},201);
  }
  const prior=await first(env,`SELECT * FROM personal_notes WHERE id=? AND user_id=? AND ${t.column}=?`,id,userId,t.id);
  if(!prior)fail(404,'Note introuvable.','not_found');
  if(prior.revision!==data.revision)fail(409,'Cette note a été modifiée. Rechargez-la avant de confirmer.','revision_conflict');
  const guard='EXISTS(SELECT 1 FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=? AND payload_hash=? AND entity_id=?)',args=[userId,kind,mutationKey,hash,id];
  const statements=[stmt(env,`INSERT OR IGNORE INTO context_mutations(user_id,kind,creation_id,payload_hash,entity_id) SELECT ?,?,?,?,? WHERE EXISTS(SELECT 1 FROM personal_notes WHERE id=? AND user_id=? AND ${t.column}=? AND revision=?)`,...args,id,userId,t.id,data.revision)];
  if(method==='PATCH'&&text!==prior.text)statements.push(stmt(env,`UPDATE personal_notes SET text=?,payload_hash=?,revision=revision+1,updated_at=? WHERE id=? AND user_id=? AND ${t.column}=? AND revision=? AND ${guard}`,text,hash,now(),id,userId,t.id,data.revision,...args));
  if(method==='DELETE')statements.push(stmt(env,`DELETE FROM personal_notes WHERE id=? AND user_id=? AND ${t.column}=? AND revision=? AND ${guard}`,id,userId,t.id,data.revision,...args));
  await env.DB.batch(statements);
  const receipt=await first(env,'SELECT * FROM context_mutations WHERE user_id=? AND kind=? AND creation_id=?',userId,kind,mutationKey);
  if(!receipt)fail(409,'Cette note a changé.','revision_conflict');
  return replay(receipt);
 }
 async function timeline(env,userId,id,url){
  const sourceVersion=await version(env,userId);
  const person=await parent(env,userId,target('people',id)),available=await timelineReady(env),size=limit(url),order=url.searchParams.get('order')||'asc';
  if(!['asc','desc'].includes(order))fail(400,'Ordre invalide.','invalid_pagination');
  const events=await all(env,'SELECT e.id,e.date_precision,e.date_start FROM life_events e JOIN life_event_people l ON l.event_id=e.id AND l.user_id=e.user_id WHERE e.user_id=? AND l.person_id=?',userId,id);
  const notes=available?await all(env,'SELECT * FROM personal_notes WHERE user_id=? AND person_id=? ORDER BY created_at,id',userId,id):[];
  const observationsAvailable=await peopleObservationReady(env),observationsSql="SELECT id,axis,quote,origin,created_at FROM people_observations WHERE user_id=? AND person_id=? AND origin='accepted' ORDER BY created_at,id";
  const observations=observationsAvailable?await all(env,observationsSql,userId,id):[];
  const stamp=await contextHash(JSON.stringify([sourceVersion,person.revision,notes.map(n=>[n.id,n.revision]),observations]));
  // Partial dates are lower bounds solely for sorting, never displayed as full dates.
  const lower=value=>value.length===4?value+'-01-01':value.length===7?value+'-01':value;
  const entries=events.map(e=>({kind:'life_event',key:'event:'+e.id,id:e.id,sort:['unknown','age'].includes(e.date_precision)?'':lower(e.date_start)}));
  entries.push({kind:'relationship_start',key:'relationship-start:'+id,sort:person.meeting_date_precision&&!['unknown','age'].includes(person.meeting_date_precision)?lower(person.meeting_date_start):'',date:{precision:person.meeting_date_precision||'unknown',start:person.meeting_date_start||'',end:person.meeting_date_end||'',age_text:person.meeting_age_text||''},legacy_period_start:person.period_start,legacy_period_end:person.period_end});
  for(const row of notes)entries.push({kind:'user_note',key:'user-note:'+row.id,sort:iso(row.created_at),note:noteRow(row)});
  for(const row of observations)entries.push({kind:'ambre_observation',key:'observation:'+row.id,sort:iso(row.created_at),note:{id:row.id,axis:row.axis,text:row.quote,origin:row.origin,proposed_by:'ambre',text_origin:'user_report',created_at:iso(row.created_at)}});
  entries.sort((a,b)=>{if(a.kind==='relationship_start'||b.kind==='relationship_start')return a.kind==='relationship_start'?-1:1;if(!a.sort||!b.sort){if(Boolean(a.sort)!==Boolean(b.sort))return a.sort?-1:1;}return (order==='asc'?1:-1)*a.sort.localeCompare(b.sort)||a.key.localeCompare(b.key)});
  let start=0;const cursor=url.searchParams.get('cursor');
  if(cursor){let parsed;try{if(cursor.length>2000)throw Error();parsed=JSON.parse(atob(cursor));}catch{fail(400,'Pagination invalide.','invalid_pagination')}
   if(parsed.v!==1||parsed.person!==id||parsed.order!==order||parsed.stamp!==stamp)fail(409,'La chronologie a changé. Rechargez-la.','pagination_changed');
   start=entries.findIndex(entry=>entry.key===parsed.key)+1;if(!start)fail(409,'Ce repère a changé.','pagination_changed');
  }
  const selected=entries.slice(start,start+size),ids=selected.filter(e=>e.kind==='life_event').map(e=>e.id);
  const rows=ids.length?await all(env,'SELECT * FROM life_events WHERE user_id=? AND id IN(SELECT value FROM json_each(?))',userId,JSON.stringify(ids)):[];
  const full=await eventRows(env,rows,userId),items=selected.map(({sort,id:entityId,...entry})=>entry.kind==='life_event'?{...entry,event:full.find(e=>e.id===entityId)}:entry);
  if(observationsAvailable&&JSON.stringify(await all(env,observationsSql,userId,id))!==JSON.stringify(observations))fail(409,'La chronologie a changé. Rechargez-la.','pagination_changed');
  if(await version(env,userId)!==sourceVersion)fail(409,'La chronologie a changé. Rechargez-la.','pagination_changed');
  return json({person:stripPrivate(person),items,total:entries.length,next_cursor:start+size<entries.length?btoa(JSON.stringify({v:1,person:id,order,stamp,key:selected.at(-1).key})):null,source_version:sourceVersion,available});
 }
 async function routes(request,env,user,path,url){
  let match=path.match(/^\/api\/people\/([\w-]+)\/timeline$/);
  if(match){if(request.method!=='GET')fail(405,'Méthode non autorisée.','method_not_allowed');return timeline(env,user.id,match[1],url)}
  match=path.match(/^\/api\/(people|life-events)\/([\w-]+)\/notes(?:\/([\w-]+))?$/);if(!match)return null;
  const t=target(match[1],match[2]),id=match[3];await parent(env,user.id,t);
  if(request.method==='GET'&&!id)return listNotes(env,user.id,t,url);
  if(request.method==='POST'&&!id||['PATCH','DELETE'].includes(request.method)&&id)return mutate(request,env,user.id,t,id);
  fail(405,'Méthode non autorisée.','method_not_allowed');
 }
 async function exportData(env,userId){return {personal_notes:await timelineReady(env)?(await all(env,'SELECT * FROM personal_notes WHERE user_id=? ORDER BY created_at,id',userId)).map(noteRow):[]}}
 return {routes,exportData};
}
