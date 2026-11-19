// Historique des modifications (migration 0034) of a person, a moment or a dated note.
// The rows are written by SQLite triggers only: a creation stores the created state, a modification
// stores the state it replaces. This module reads them for their owner and never calls a model;
// the history is not part of the context given to Ambre.
import {iso} from './timeline-data.mjs';
export const HISTORY_LIMIT=50;
export const HISTORY_FIELDS=Object.freeze({
 person:Object.freeze(['display_name','role','custom_role','relationship_text','period_start','period_end','bond_quality','bond_note','presence_now','helps','hurts','evolution','feels_safe','meeting_date_precision','meeting_date_start','meeting_date_end','meeting_age_text']),
 event:Object.freeze(['title','kind','description','meaning_now','date_precision','date_start','date_end','age_text','current_importance','significance','felt_then','felt_now','impact','support_level','support_text','exposure','people_ids']),
 note:Object.freeze(['text'])
});
const TABLES={person:'personal_people',event:'life_events',note:'personal_notes'};
const PROBED=new WeakMap();
// Like significanceReady: a missing table is retried after a few seconds, any other error is raised.
export async function historyReady(env){
 const db=env?.DB;if(!db)return false;const cached=PROBED.get(db);if(cached===true)return true;if(typeof cached==='number'&&cached>Date.now())return false;
 try{await db.prepare('SELECT id FROM personal_history LIMIT 0').bind().all();PROBED.set(db,true);return true}
 catch(error){if(/no such (?:column|table)/i.test(String(error?.message||''))){PROBED.set(db,Date.now()+5000);return false}throw error}
}
// People and moments keep SQLite CURRENT_TIMESTAMP (UTC text); notes keep seconds.
export const textTime=value=>{if(value==null||value==='')return null;if(typeof value==='number')return iso(value);const date=new Date(String(value).replace(' ','T')+(/[zZ]|[+-]\d{2}:?\d{2}$/.test(String(value))?'':'Z'));return Number.isNaN(date.getTime())?null:date.toISOString()};
const parse=text=>{try{const value=JSON.parse(text);return value&&typeof value==='object'?value:{}}catch{return {}}};
export function createHistoryBackend({error,json}){
 const first=(env,sql,...args)=>env.DB.prepare(sql).bind(...args).first();
 const all=async(env,sql,...args)=>(await env.DB.prepare(sql).bind(...args).all()).results;
 const fail=(status,message,code)=>error(status,message,code);
 function project(type,row,known){
  const out={};for(const key of HISTORY_FIELDS[type])if(key!=='people_ids')out[key]=row[key]??(key==='current_importance'||key==='significance'?null:'');
  if(type==='event')out.people_ids=(Array.isArray(row.people_ids)?row.people_ids:[]).filter(id=>known.has(id)).sort();
  return out;
 }
 const same=(a,b)=>JSON.stringify(a??null)===JSON.stringify(b??null);
 // A date is read as a whole: one changed part brings its precision and bounds along.
 const GROUPS=[['date_precision','date_start','date_end','age_text'],['meeting_date_precision','meeting_date_start','meeting_date_end','meeting_age_text']];
 const diff=(type,before,after)=>{const changed=new Set(HISTORY_FIELDS[type].filter(key=>!same(before[key],after[key])));
  for(const group of GROUPS)if(group.some(key=>changed.has(key)))group.forEach(key=>changed.add(key));
  return HISTORY_FIELDS[type].filter(key=>changed.has(key)).map(key=>({field:key,before:before[key]??null,after:after[key]??null}));};
 async function read(env,userId,type,id){
  const row=await first(env,`SELECT * FROM ${TABLES[type]} WHERE id=? AND user_id=?`,id,userId);
  // The same answer covers an unknown identifier and an element of another account.
  if(!row)fail(404,'Élément introuvable.','not_found');
  const created_at=textTime(row.created_at);
  if(!await historyReady(env))return json({available:false,type,id,created_at,versions:[],complete:false,limit:HISTORY_LIMIT});
  const [rows,people,links]=await Promise.all([
   all(env,'SELECT id,revision,action,snapshot,changed_at FROM personal_history WHERE user_id=? AND entity_type=? AND entity_id=? ORDER BY id',userId,type,id),
   type==='event'?all(env,'SELECT id FROM personal_people WHERE user_id=?',userId):[],
   type==='event'?all(env,'SELECT person_id FROM life_event_people WHERE user_id=? AND event_id=? ORDER BY person_id',userId,id):[]
  ]);
  // People deleted since are left out of every state: their deletion erased them.
  const known=new Set(people.map(person=>person.id));
  const current=project(type,type==='event'?{...row,people_ids:links.map(link=>link.person_id)}:row,known);
  const updates=rows.filter(entry=>entry.action==='update'),create=rows.find(entry=>entry.action==='create');
  const states=[...updates.map(entry=>project(type,parse(entry.snapshot),known)),current],versions=[];
  if(create)versions.push({action:'create',at:iso(create.changed_at),values:states[0]});
  updates.forEach((entry,index)=>{const changes=diff(type,states[index],states[index+1]);if(changes.length)versions.push({action:'update',at:iso(entry.changed_at),changes})});
  versions.reverse();
  return json({available:true,type,id,created_at,updated_at:textTime(row.updated_at),complete:Boolean(create),versions,limit:HISTORY_LIMIT});
 }
 async function routes(request,env,user,path,url){
  if(path!=='/api/history')return null;
  if(request.method!=='GET')fail(405,'Méthode non autorisée.','method_not_allowed');
  const type=url.searchParams.get('type')||'',id=url.searchParams.get('id')||'';
  if(!Object.hasOwn(TABLES,type)||!/^[\w-]{1,120}$/.test(id))fail(400,'Élément invalide.','invalid_input');
  return read(env,user.id,type,id);
 }
 async function exportData(env,userId){
  if(!await historyReady(env))return {personal_history:[]};
  const rows=await all(env,'SELECT entity_type,entity_id,revision,action,snapshot,changed_at FROM personal_history WHERE user_id=? ORDER BY id',userId);
  return {personal_history:rows.map(row=>({entity_type:row.entity_type,entity_id:row.entity_id,revision:row.revision,action:row.action,state:parse(row.snapshot),changed_at:iso(row.changed_at)}))};
 }
 return {routes,exportData};
}
