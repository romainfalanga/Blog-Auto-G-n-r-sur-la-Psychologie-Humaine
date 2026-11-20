import {mapSafetyState} from './symptom-map-context.mjs';
import {companionCartography} from './cartographie.mjs';
import {CARTOGRAPHIE} from './cartographie.generated.mjs';
import {SYMPTOMES_CARTE} from './symptomes-carte.generated.mjs';
// Complete, immutable personal review snapshots. No provider traces, credentials,
// billing records or public catalogue contents belong in this allowlist.
export const SNAPSHOT_SCHEMA_VERSION=1;
export const SNAPSHOT_MAX_BYTES=512*1024;
export const SNAPSHOT_MAX_ROWS=10000;
export const SNAPSHOT_ANALYSIS_MAX_CHARS=40000;
const TABLES={profile:'profiles',childhood:'cartographies',symptoms:'symptom_maps',people:'personal_people',events:'life_events',event_people:'life_event_people',people_notes:'personal_notes',event_notes:'personal_notes',observations:'people_observations',plan:'change_plans',memory:'chat_memory',personal_memory:'memories',conversation:'chat_turns',previous_reviews:'ambre_reviews',care:'care_settings',care_sessions:'care_sessions',portrait:'user_portraits',professional_assessments:'professional_assessments'};
const HIDDEN=new Set(['user_id','password_hash','password_salt','creation_id','payload_hash','lease_token','lease_until','checkout_session_id','apply_token','draft','input_tokens','output_tokens','observed_cost','error_code']);
const JSON_FIELDS=new Set(['data','self_data','answers','feedback','notes','resources','ambre_notes','document_json','summary','portrait','digest','sources','response','result_json']);
const failure=(code,message)=>Object.assign(Error(message),{code,status:code==='snapshot_too_large'?413:409});
export async function snapshotStorageAvailable(env){return Boolean(await env.DB.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_snapshots'").first());}
export async function snapshotSourceQuery(env){
 const sections=[];
 for(const [section,table] of Object.entries(TABLES)){
  const info=(await env.DB.prepare(`PRAGMA table_info(${table})`).all()).results||[];
  if(!info.length)continue;
  const names=info.map(x=>x.name);if(!names.includes('user_id'))continue;
  let columns=names.filter(n=>!HIDDEN.has(n)&&/^[a-z_]+$/.test(n));
  if(table==='chat_turns')columns=['id','content','response','created_at'];
  if(table==='ambre_reviews')columns=['id','kind','content','created_at','completed_at',...['digest','feedback'].filter(n=>names.includes(n))];
  const condition=table==='personal_notes'?(section==='people_notes'?' AND person_id IS NOT NULL':' AND event_id IS NOT NULL'):table==='chat_turns'?" AND status='completed' AND purged=0 AND context_generation=COALESCE((SELECT context_generation FROM change_plans WHERE user_id=(SELECT user_id FROM owner)),0) AND idempotency_key!='review-'||(SELECT review_id FROM owner)":table==='ambre_reviews'?" AND status='ready' AND id!=(SELECT review_id FROM owner)":'';
  const order=names.includes('id')?'id':names.includes('article_id')?'article_id':names.includes('event_id')?'event_id,person_id':names.includes('created_at')?'created_at':'rowid';
  const object=columns.map(c=>`'${c}',${c}`).join(',');
  sections.push({section,table,columns,expression:`json((SELECT json_group_array(json_object(${object})) FROM (SELECT ${columns.join(',')} FROM ${table} WHERE user_id=(SELECT user_id FROM owner)${condition} ORDER BY ${order} LIMIT ${SNAPSHOT_MAX_ROWS+1})))`});
 }
 return {sections,sql:`WITH owner AS (SELECT ? AS user_id,? AS review_id) SELECT json_object(${sections.map(x=>`'${x.section}',${x.expression}`).join(',')}) AS payload`};
}
export async function readSnapshotSource(env,userId,reviewId,query){
 query=query||await snapshotSourceQuery(env);
 const row=await env.DB.prepare(query.sql).bind(userId,reviewId).first(),stamp=row.payload;
 if(new TextEncoder().encode(stamp).length>SNAPSHOT_MAX_BYTES)throw failure('snapshot_too_large','Les données dépassent la taille maximale d’un instantané complet. Aucun bilan partiel n’a été généré.');
 const raw=JSON.parse(stamp);
 for(const rows of Object.values(raw))if(rows.length>SNAPSHOT_MAX_ROWS)throw failure('snapshot_too_large','Une rubrique dépasse la limite de lignes d’un instantané complet. Aucun bilan partiel n’a été généré.');
 const sections=Object.fromEntries(Object.entries(raw).map(([key,rows])=>[key,rows.map(row=>Object.fromEntries(Object.entries(row).map(([name,value])=>{if(JSON_FIELDS.has(name)&&typeof value==='string'){try{return [name,JSON.parse(value)]}catch{}}return [name,value]})))]));
 return {query,stamp,sections};
}
export function snapshotGuard(source,userId,reviewId){return {sql:`(${source.query.sql.replace(' AS payload','')})=?`,args:[userId,reviewId,source.stamp]};}
// Plan v1.1 (point 1-2): after the capture, a review works on its frozen snapshot only.
// Changes and additions to the live sources are ignored (the review is a dated document);
// an item the person entered and that the snapshot holds must still exist, otherwise the
// review ends with sources_changed and its snapshot is erased (right to erasure).
// Single-row sections are checked by owner; AI-produced sections (portrait, conversation
// memory, change plan) are not person entries and are left out.
// Snapshots captured before the catalogue was retired may still hold reading_choices,
// inspiration_choices, personal_characters, personal_stories and story_feedback: their
// rows stay checked here (the tables remain) although new captures no longer read them.
const PRESENCE_BY_ID={people:['personal_people','id'],events:['life_events','id'],people_notes:['personal_notes','id'],event_notes:['personal_notes','id'],observations:['people_observations','id'],conversation:['chat_turns','id'," AND status='completed' AND purged=0"],previous_reviews:['ambre_reviews','id'," AND status='ready'"],care_sessions:['care_sessions','id'],personal_characters:['personal_characters','id'],personal_stories:['stories','id'],professional_assessments:['professional_assessments','id'],reading_choices:['article_likes','article_id'],inspiration_choices:['inspiration_selections','article_id'],story_feedback:['feedback','story_id']};
const PRESENCE_SINGLE={childhood:['cartographies',' AND consent=1'],symptoms:['symptom_maps',' AND consent=1'],care:['care_settings','']};
export function snapshotPresenceGuard(payload,userId){
 const parts=[],args=[],sections=payload?.sections||{};
 for(const [section,[table,key,extra='']] of Object.entries(PRESENCE_BY_ID)){
  const ids=[...new Set((sections[section]||[]).map(row=>row?.[key]).filter(id=>id!==null&&id!==undefined))];if(!ids.length)continue;
  parts.push(`(SELECT count(*) FROM ${table} WHERE user_id=? AND ${key} IN (SELECT value FROM json_each(?))${extra})=?`);args.push(userId,JSON.stringify(ids),ids.length);
 }
 for(const [section,[table,extra]] of Object.entries(PRESENCE_SINGLE))if(sections[section]?.length){parts.push(`EXISTS(SELECT 1 FROM ${table} WHERE user_id=?${extra})`);args.push(userId);}
 return {sql:parts.length?'('+parts.join(' AND ')+')':'1=1',args};
}
// Interface labels shown to the person (questions, answers, themes, subjects). A review may
// quote them only as labels (« à la question … vous avez répondu … »), never as her words.
export const ANSWER_LABELS=Object.freeze(['Non','Oui','Souvent','Parfois','Je ne sais pas','Dans le passé','Je me reconnais','Pas moi','Non renseigné','Ces 12 derniers mois','avant ces 12 derniers mois']);
export const INTERFACE_LABELS=Object.freeze([...new Set([...ANSWER_LABELS,...[CARTOGRAPHIE,SYMPTOMES_CARTE].flatMap(data=>['factors','outcomes','retired_outcomes','groups','domains','protections'].flatMap(key=>(Array.isArray(data[key])?data[key]:[]).map(x=>x?.label||x?.title).filter(x=>typeof x==='string'&&x.trim()))),SYMPTOMES_CARTE.period?.label].filter(Boolean))]);
// The person's own words in a snapshot: what she typed (profile, people, moments, notes,
// messages, accepted observations, feedback on reviews). Never a text produced by Ambre or
// an AI (portrait, memory, previous review texts, chunk summaries, replies).
export function snapshotPersonCorpus(captured){
 const out=[],walk=v=>{if(typeof v==='string')out.push(v);else if(Array.isArray(v))v.forEach(walk);else if(v&&typeof v==='object')Object.values(v).forEach(walk)};
 walk(captured?.profile);walk(captured?.people);walk(captured?.events);walk(captured?.people_notes);walk(captured?.event_notes);
 for(const row of captured?.conversation||[])walk(row?.content);
 for(const row of captured?.observations||[])walk(row?.quote);
 // Mes précisions (memories.text) is typed by the person; proposals awaiting her answer are not.
 for(const row of captured?.personal_memory||[])walk(row?.text);
 for(const row of captured?.previous_reviews||[])walk(row?.feedback?.text);
 for(const key of ['childhood','symptoms'])for(const row of captured?.[key]||[]){walk(row?.notes);walk(row?.resources);}
 for(const row of captured?.care_sessions||[])walk(row);
 return out.join('\n\n');
}
export async function readReviewSnapshot(env,userId,reviewId){
 if(!await snapshotStorageAvailable(env))return null;
 const row=await env.DB.prepare('SELECT * FROM review_snapshots WHERE review_id=? AND user_id=?').bind(reviewId,userId).first();
 return row?{...row,payload:JSON.parse(row.payload),assessment:JSON.parse(row.assessment)}:null;
}
export async function storeReviewSnapshot(env,{userId,review,capturedAt,source,payload,assessment={}}){
 const encoded=JSON.stringify(payload);if(new TextEncoder().encode(encoded).length>SNAPSHOT_MAX_BYTES)throw failure('snapshot_too_large','L’instantané complet dépasse 512 Kio, libellés compris. Aucun bilan partiel n’a été généré.');
 const guard=snapshotGuard(source,userId,review.id);
 const stored=await env.DB.prepare(`INSERT INTO review_snapshots(review_id,user_id,captured_at,schema_version,payload,assessment,source_stamp) SELECT ?,?,?,1,?,?,? WHERE EXISTS(SELECT 1 FROM ambre_reviews r JOIN chat_settings cs ON cs.user_id=r.user_id WHERE r.id=? AND r.user_id=? AND r.status='running' AND r.lease_token=? AND cs.consent=1) AND ${guard.sql} ON CONFLICT(review_id) DO NOTHING`).bind(review.id,userId,capturedAt,encoded,JSON.stringify(assessment),source.stamp,review.id,userId,review.lease_token,...guard.args).run();
 if(!stored.meta.changes)throw failure('context_changed','Les données ont changé pendant la capture. Aucun instantané mélangé n’a été enregistré.');
 return readReviewSnapshot(env,userId,review.id);
}
export function snapshotPage(row,{section=null,cursor=0,limit=50}={}){
 const sections=row.payload.sections||{},overview={available:true,review_id:row.review_id,captured_at:row.captured_at,schema_version:row.schema_version,complete:true,assessment:row.assessment,sections:Object.entries(sections).map(([id,items])=>({id,count:items.length}))};
 if(!section)return overview;
 if(!Object.hasOwn(sections,section))throw Object.assign(Error('Rubrique inconnue.'),{status:404,code:'not_found'});
 if(!Number.isInteger(cursor)||cursor<0||!Number.isInteger(limit)||limit<1||limit>100)throw Object.assign(Error('Pagination invalide.'),{status:400,code:'invalid_cursor'});
 let items=sections[section];
 const people=new Map((sections.people||[]).map(x=>[x.id,x.display_name])),events=new Map((sections.events||[]).map(x=>[x.id,x.title]));
 if(['event_people','people_notes','event_notes','observations'].includes(section))items=items.map(x=>({...x,...(x.person_id?{person_label:people.get(x.person_id)||'Proche retiré'}:{}),...(x.event_id?{event_title:events.get(x.event_id)||'Événement retiré'}:{})}));
 if(['childhood','symptoms'].includes(section)){const labels=row.payload.question_labels?.[section]||{};items=items.map(item=>({...item,answers:Object.entries(item.answers||{}).map(([id,value])=>({id,label:labels.factors?.[id]||id,value,note:item.notes?.factors?.[id]||''})),feedback:Object.entries(item.feedback||{}).map(([id,value])=>({id,label:labels.outcomes?.[id]||id,value,note:item.notes?.outcomes?.[id]||''})),notes:{factors:Object.entries(item.notes?.factors||{}).filter(([id])=>!Object.hasOwn(item.answers||{},id)).map(([id,text])=>({id,label:labels.factors?.[id]||id,text})),outcomes:Object.entries(item.notes?.outcomes||{}).filter(([id])=>!Object.hasOwn(item.feedback||{},id)).map(([id,text])=>({id,label:labels.outcomes?.[id]||id,text}))}}));}
 return {...overview,section,items:items.slice(cursor,cursor+limit),next_cursor:cursor+limit<items.length?cursor+limit:null,total:items.length};
}
export async function exportReviewSnapshots(env,userId){
 if(!await snapshotStorageAvailable(env))return [];
 const rows=(await env.DB.prepare('SELECT s.review_id,r.status AS review_status,length(s.payload)+length(s.assessment) AS size FROM review_snapshots s JOIN ambre_reviews r ON r.id=s.review_id AND r.user_id=s.user_id WHERE s.user_id=? ORDER BY s.review_id').bind(userId).all()).results||[];
 if(rows.reduce((n,x)=>n+x.size,0)>16*1024*1024)throw failure('snapshot_too_large','L’export complet des instantanés dépasse 16 Mio. Consultez les archives par bilan ; aucun instantané n’a été omis silencieusement.');
 const out=[];let bytes=0;
 for(const row of rows){
  const item=await readReviewSnapshot(env,userId,row.review_id);if(!item)throw failure('context_changed','Les archives ont changé pendant l’export. Réessayez.');
  const value={review_id:item.review_id,captured_at:item.captured_at,schema_version:item.schema_version,payload:item.payload,review_status:row.review_status,assessment:row.review_status==='ready'?item.assessment:null};bytes+=new TextEncoder().encode(JSON.stringify(value)).length;
  if(bytes>16*1024*1024)throw failure('snapshot_too_large','L’export complet des instantanés dépasse 16 Mio. Consultez les archives par bilan ; aucun instantané n’a été omis silencieusement.');out.push(value);
 }
 const current=(await env.DB.prepare('SELECT s.review_id,r.status AS review_status FROM review_snapshots s JOIN ambre_reviews r ON r.id=s.review_id AND r.user_id=s.user_id JOIN chat_settings cs ON cs.user_id=s.user_id WHERE s.user_id=? AND cs.consent=1 ORDER BY s.review_id').bind(userId).all()).results||[];
 if(JSON.stringify(current.map(x=>[x.review_id,x.review_status]))!==JSON.stringify(rows.map(x=>[x.review_id,x.review_status])))throw failure('context_changed','Les archives ou votre accord ont changé pendant l’export. Réessayez.');
 return out;
}

export function snapshotLabels(){return Object.fromEntries([['childhood',CARTOGRAPHIE],['symptoms',SYMPTOMES_CARTE]].map(([key,data])=>[key,{factors:Object.fromEntries(data.factors.map(f=>[f.id,f.label])),outcomes:Object.fromEntries(data.outcomes.map(o=>[o.id,o.label]))}]));}
// Every character of every personal item is visited; large items are split with explicit offsets.
export function snapshotChunks(payload,maxChars=20000){
 const fragments=[];
 for(const [section,rows] of Object.entries(payload.sections))for(let index=0;index<rows.length;index++){
  const row=rows[index],labels=payload.question_labels?.[section];
  const item=labels?{...row,answer_labels:Object.fromEntries(Object.keys(row.answers||{}).map(id=>[id,labels.factors[id]||id])),feedback_labels:Object.fromEntries(Object.keys(row.feedback||{}).map(id=>[id,labels.outcomes[id]||id]))}:row;
  const text=JSON.stringify(item),step=Math.floor((maxChars-1000)/2);for(let offset=0;offset<text.length;offset+=step)fragments.push({path:section+'/'+index,offset,total_chars:text.length,text:text.slice(offset,offset+step)});
 }
 const chunks=[];let current=[];
 for(const fragment of fragments){if(current.length&&JSON.stringify([...current,fragment]).length>maxChars){chunks.push(current);current=[];}current.push(fragment);}if(current.length)chunks.push(current);
 return chunks.map((items,index)=>({index,items}));
}
// A fragment is a slice of the JSON text of an item: the person's words appear in it with JSON
// escapes (\" for a quote, \n for a line break), and the model sees them escaped a second time in
// the message. A quote is therefore compared on decoded, normalised text on both sides
// (escapes, typographic apostrophes and quotation marks, spaces); its words must still all be there,
// in the same order: a quote that is not in the fragments of the block is always rejected.
const decodeJsonEscapes=text=>String(text).replace(/\\(?:u([0-9a-fA-F]{4})|(["\\/bfnrt]))/g,(m,hex,c)=>hex?String.fromCharCode(parseInt(hex,16)):({b:'\b',f:'\f',n:'\n',r:'\r',t:'\t'}[c]??c));
export function normalizeQuoteText(text){
 return decodeJsonEscapes(text).normalize('NFC').replace(/[\u2018\u2019\u201A\u201B\u2032\u02BC\u0060\u00B4]/g,"'").replace(/[\u201C\u201D\u201E\u201F\u2033\u00AB\u00BB]/g,'"').replace(/[\u2010-\u2015\u2212]/g,'-').replace(/\u2026/g,'...').replace(/\s+/g,' ').replace(/ ?" ?/g,'"').trim();
}
export const SNAPSHOT_SUMMARY_MAX=800,SNAPSHOT_QUOTE_MAX=160,SNAPSHOT_QUOTES_MAX=3;
// A summary over the limit (the schema maxLength is not enforced by every provider in strict mode)
// is cut at the last full sentence, or else at a word, instead of losing the whole block.
export function clipText(text,max,{ellipsis=true}={}){
 const value=String(text).trim();if(value.length<=max)return value;
 const room=ellipsis?max-1:max,head=value.slice(0,room+1),sentence=Math.max(...['. ','! ','? ','… ','.\n','!\n','?\n'].map(m=>head.lastIndexOf(m)));
 if(sentence>=room*0.6)return value.slice(0,sentence+1).trim();
 const space=head.lastIndexOf(' ');return value.slice(0,space>room*0.5?space:room).trim()+(ellipsis?'…':'');
}
const rejected=(reason,message,extra={})=>Object.assign(failure('invalid_response',message),{rejection:{reason,...extra}});
export function validateSnapshotChunk(value,chunk){
 if(!value||typeof value!=='object')throw rejected('not_object','Analyse de bloc invalide.');
 if(typeof value.summary!=='string'||!value.summary.trim())throw rejected('summary_missing','Analyse de bloc invalide.');
 if(!Array.isArray(value.quotes))throw rejected('quotes_missing','Analyse de bloc invalide.',{summary_chars:value.summary.length});
 // The decoded text of each path in this block, contiguous fragments joined (a quote may straddle a cut).
 const texts=new Map();
 for(const item of [...chunk.items].sort((a,b)=>a.path.localeCompare(b.path)||a.offset-b.offset)){const prev=texts.get(item.path);texts.set(item.path,prev?{raw:prev.raw+(prev.end===item.offset?'':'\u0000')+item.text,end:item.offset+item.text.length}:{raw:item.text,end:item.offset+item.text.length});}
 const normalized=new Map([...texts].map(([path,{raw}])=>[path,normalizeQuoteText(raw)]));
 const quotes=[],counts={received:value.quotes.length,empty:0,clipped:0,repathed:0,dropped:0};
 for(const [position,q] of value.quotes.entries()){
  if(!q||typeof q!=='object'||typeof q.quote!=='string')throw rejected('quote_invalid','Citation de bloc non vérifiable.',{quote_index:position});
  const quote=decodeJsonEscapes(q.quote).trim(),needle=normalizeQuoteText(quote);
  if(!needle){counts.empty++;continue;}
  const own=typeof q.path==='string'&&normalized.get(q.path)?.includes(needle)?q.path:null,path=own||[...normalized].find(([,text])=>text.includes(needle))?.[0];
  if(!path)throw rejected(typeof q.path==='string'&&normalized.has(q.path)?'quote_not_found':'quote_not_found_unknown_path','Citation de bloc non vérifiable.',{quote_index:position,quote_chars:q.quote.length,path_known:normalized.has(q.path)});
  if(!own)counts.repathed++;
  if(quotes.length>=SNAPSHOT_QUOTES_MAX){counts.dropped++;continue;}
  // A prefix of an exact quote is still exact: a quote over the limit is cut at a word.
  let kept=quote;if(kept.length>SNAPSHOT_QUOTE_MAX){kept=clipText(kept,SNAPSHOT_QUOTE_MAX,{ellipsis:false});counts.clipped++;if(!normalized.get(path).includes(normalizeQuoteText(kept)))throw rejected('quote_clip_failed','Citation de bloc non vérifiable.',{quote_index:position});}
  quotes.push({path,quote:kept});
 }
 const summary=clipText(value.summary,SNAPSHOT_SUMMARY_MAX),adjusted={...(summary.length!==value.summary.trim().length?{summary_chars:value.summary.length}:{}),...Object.fromEntries(Object.entries(counts).filter(([k,v])=>k!=='received'&&v))};
 return {index:chunk.index,source_refs:chunk.items.map(({path,offset,total_chars,text})=>({path,offset,length:text.length,total_chars})),summary,quotes,...(Object.keys(adjusted).length?{adjusted}:{})};
}
export const SNAPSHOT_CHUNK_SCHEMA={type:'object',properties:{summary:{type:'string',maxLength:800},quotes:{type:'array',maxItems:3,items:{type:'object',properties:{path:{type:'string'},quote:{type:'string',maxLength:160}},required:['path','quote'],additionalProperties:false}}},required:['summary','quotes'],additionalProperties:false};

export function snapshotSafetyContext(payload){
 const captured=payload.sections,at=Date.parse(payload.captured_at)/1000,child=captured.childhood?.[0];
 const childhood=child?companionCartography({...child,...Object.fromEntries(['answers','feedback','notes','resources','ambre_notes'].map(k=>[k,JSON.stringify(child[k]??(k==='ambre_notes'?[]:{}))]))}):null;
 if(childhood&&CARTOGRAPHIE.outcomes.some(o=>o.note==='suicide'&&child.feedback?.[o.id]==='oui')){childhood.suicide_recognized=true;childhood.safety_provenance='Reconnaissance enregistrée dans Mon enfance ; ne prouve pas un danger actuel.';}
 const mapSafety=mapSafetyState(captured.symptoms?.[0],at);
 const since=Math.max(0,...(captured.previous_reviews||[]).map(r=>Number(r.completed_at)||0));
 const conversation=(captured.conversation||[]).filter(r=>{const t=Date.parse(String(r.created_at||'').replace(' ','T')+'Z')/1000;return !Number.isFinite(t)||t>since;});
 return {memory:captured.memory?.[0]?.summary||{},conversation_since:conversation.map(r=>({vous:r.content})),childhood_cartography:childhood,map_safety:mapSafety?{kinds:mapSafety.kinds,reported_at:mapSafety.reported_at}:null};
}
