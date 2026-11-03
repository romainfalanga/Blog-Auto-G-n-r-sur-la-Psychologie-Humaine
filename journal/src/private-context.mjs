// Canonical private sources. Fictional story text is deliberately excluded.
const parse=(v,f={})=>{try{return JSON.parse(v);}catch{return f;}};
const hash=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),b=>b.toString(16).padStart(2,'0')).join('');
export const hasPersonalizationSources=context=>(context.sources||[]).some(s=>s.type==='like'||s.explicit);
export async function loadContext(env,userId,decorateLikes=x=>x) {
 // Read bounded rows rather than aggregating an entire 500-event timeline into
 // one SQLite cell. Retry if any source changed during the snapshot reads.
 let row,likes,feedback,people,events,exposures;
 for(let attempt=0;attempt<4;attempt++){
  row=await env.DB.prepare(`SELECT p.version AS source_version,p.self_data,p.self_revision,m.text AS memory_text,m.version AS memory_version,up.portrait,up.analyzed_version,up.schema_version,up.updated_at,s.status AS subscription_status,s.paid_until FROM profiles p JOIN memories m ON m.user_id=p.user_id JOIN subscriptions s ON s.user_id=p.user_id LEFT JOIN user_portraits up ON up.user_id=p.user_id WHERE p.user_id=?`).bind(userId).first();
  if(!row)return null;
  const queries=[
   'SELECT article_id,reason,differences,revision,created_at,updated_at FROM article_likes WHERE user_id=? ORDER BY created_at,article_id',
   'SELECT story_id,resonance,feelings,why,correction,differences,scope,preference,avoid,revision,updated_at FROM feedback WHERE user_id=? ORDER BY updated_at,story_id',
   'SELECT id,display_name,role,custom_role,relationship_text,period_start,period_end,revision,created_at,updated_at FROM personal_people WHERE user_id=? ORDER BY id',
   `SELECT e.id,e.title,e.kind,e.description,e.meaning_now,e.date_precision,e.date_start,e.date_end,e.age_text,e.current_importance,e.fiction_use,e.revision,e.created_at,e.updated_at,(SELECT json_group_array(person_id) FROM life_event_people WHERE event_id=e.id AND user_id=e.user_id) AS people_ids FROM life_events e WHERE e.user_id=? ORDER BY e.id`,
   "SELECT article_id,shown_at FROM recommendation_exposures WHERE user_id=? AND shown_at>datetime('now','-7 days') ORDER BY shown_at DESC,article_id LIMIT 100"
  ];
  [likes,feedback,people,events,exposures]=await Promise.all(queries.map(async sql=>(await env.DB.prepare(sql).bind(userId).all()).results));
  const after=await env.DB.prepare('SELECT version FROM profiles WHERE user_id=?').bind(userId).first();
  if(!after)return null;
  if(after.version===row.source_version)break;
  if(attempt===3){const error=new Error('Le contexte change trop rapidement. Réessayez.');error.status=409;error.code='context_changed';throw error;}
 }
 const context={source_version:row.source_version,portrait_version:row.schema_version===2?(row.analyzed_version||0):0,updated_at:row.updated_at||null,paid:row.subscription_status==='active'&&Number(row.paid_until)>Date.now()/1000,likes:decorateLikes(likes),feedback,people,exposures,events:events.map(e=>({...e,people_ids:parse(e.people_ids,[])})),memory:{text:row.memory_text,version:row.memory_version},portrait:row.schema_version===2&&row.analyzed_version===row.source_version?parse(row.portrait,null):null};
 const self=parse(row.self_data),narrativeKeys=['self_description','self_perception','recurring_patterns','emotions','current_context'];
 context.has_self_description=narrativeKeys.some(k=>typeof self[k]==='string'&&self[k].trim());
 const today=new Date().toISOString().slice(0,10),birth=self.birth_date;
 context.self_context={age:birth?Number(today.slice(0,4))-Number(birth.slice(0,4))-(today.slice(5)<birth.slice(5)?1:0):null,gender:self.gender||'',gender_description:self.gender_description||'',occupation:self.occupation||''};
 const sources=[];
 const add=(id,type,revision,fields,extra={})=>{const text=Object.entries(fields).filter(([,v])=>typeof v==='string'&&v.trim()).map(([k,v])=>`${k}: ${v}`).join('\n');if(text)sources.push({id,type,revision,text,fields,explicit:true,temporal:null,fiction_use:'abstract',created_at:null,updated_at:null,...extra});};
 if(context.has_self_description)add('profile:'+userId,'profile',row.self_revision,Object.fromEntries([...narrativeKeys,'occupation'].map(k=>[k,self[k]||''])));
 for(const l of context.likes)add('like:'+l.article_id,'like',l.revision,{title:l.title||l.article_id,article_summary:l.summary||'',subject:l.sujet||'',reason:l.reason,differences:l.differences},{explicit:Boolean(l.reason.trim()||l.differences.trim()),created_at:l.created_at,updated_at:l.updated_at});
 if(context.memory.text.trim())add('memory:'+userId,'memory',context.memory.version,{text:context.memory.text});
 for(const p of context.people)if(p.relationship_text.trim())add('person:'+p.id,'person',p.revision,{role:p.role,custom_role:p.custom_role,relationship_text:p.relationship_text},{created_at:p.created_at,updated_at:p.updated_at,temporal:{start:p.period_start,end:p.period_end}});
 for(const e of context.events)if(e.description.trim()||e.meaning_now.trim())add('event:'+e.id,'event',e.revision,{title:e.title,description:e.description,meaning_now:e.meaning_now},{created_at:e.created_at,updated_at:e.updated_at,temporal:{precision:e.date_precision,start:e.date_start,end:e.date_end,age:e.age_text},fiction_use:e.fiction_use,related_people:e.people_ids.map(id=>{const p=context.people.find(p=>p.id===id);return {id,role:p?.role||'',custom_role:p?.custom_role||''};})});
 for(const f of context.feedback)add('feedback:'+f.story_id,'feedback',f.revision,{feelings:f.feelings,why:f.why,differences:f.differences,correction:f.correction,preference:f.preference,avoid:f.avoid},{updated_at:f.updated_at,scope:f.scope,resonance:f.resonance});
 for(const source of sources)source.hash=await hash(JSON.stringify([source.type,source.revision,source.text,source.temporal,source.fiction_use,source.related_people||[]]));
 context.portrait_complete=Boolean(context.portrait&&!(context.portrait.coverage?.pending?.length));
 context.sources=sources;context.source_manifest=Object.fromEntries(sources.map(s=>[s.id,{revision:s.revision,hash:s.hash,type:s.type,explicit:s.explicit}]));
 return context;
}
export function validatePortraitV2(value,sources,fail) {
 const invalid=message=>fail(400,message,'invalid_portrait');
 if(!value||value.schema_version!==2||value.fictional_character!==undefined)invalid('Un portrait V2 sourcé est requis.');
 const clean=(v,max)=>{if(typeof v!=='string'||!v.trim()||v.length>max)invalid('Texte du portrait invalide.');return v.trim();};
 const allowed=new Map(sources.map(s=>[s.id,s])),proofs=new Set(),personalProofs=new Set();
 const personalQuote=(s,quote)=>{const keys={like:['reason','differences'],person:['relationship_text'],event:['description','meaning_now'],feedback:['feelings','why','differences','correction','preference','avoid'],profile:['self_description','self_perception','recurring_patterns','emotions','current_context'],memory:['text']}[s.type]||[];const start=s.text.indexOf(quote),end=start+quote.length;return keys.some(k=>{const value=s.fields?.[k];if(!value?.trim())return false;const pos=s.text.indexOf(k+': '+value)+k.length+2;return Math.min(end,pos+value.length)-Math.max(start,pos)>=Math.min(8,value.trim().length);});};
 if(!Array.isArray(value.citations)||!value.citations.length||value.citations.length>500)invalid('Citations requises.');
 const citations=value.citations.map(p=>{const source=allowed.get(p?.source_id),quote=clean(p?.quote,1200);if(!source||p.revision!==source.revision||!source.text.includes(quote))invalid('Citation absente ou révision périmée.');proofs.add(p.source_id);if(source.explicit&&personalQuote(source,quote))personalProofs.add(p.source_id);return {source_id:p.source_id,revision:p.revision,quote};});
 const evidence=(ids,explicit=false)=>{if(!Array.isArray(ids)||!ids.length||ids.length>50||ids.some(id=>!proofs.has(id)||explicit&&!personalProofs.has(id)))invalid('Preuves du portrait invalides.');return [...new Set(ids)];};
 if(!Array.isArray(value.axes)||value.axes.length<1||value.axes.length>12)invalid('Axes du portrait invalides.');
 const axes=value.axes.map(a=>{const refs=evidence(a.evidence),bare=refs.every(id=>!personalProofs.has(id));if(!['interest','hypothesis','declaration'].includes(a.claim_type)||!Number.isFinite(a.confidence)||a.confidence<0||a.confidence>1||bare&&(a.confidence>0.25||a.claim_type!=='interest'))invalid('Un intérêt de lecture ne prouve pas un fait personnel.');return {label:clean(a.label,160),why:clean(a.why,2400),confidence:a.confidence,claim_type:a.claim_type,evidence:refs};});
 if(!Array.isArray(value.strengths)||value.strengths.length>20||!Array.isArray(value.uncertainties)||value.uncertainties.length>20)invalid('Listes du portrait invalides.');
 const output={schema_version:2,summary:clean(value.summary,6000),summary_evidence:evidence(value.summary_evidence),axes,strengths:value.strengths.map(s=>({text:clean(s.text,1200),evidence:evidence(s.evidence,true)})),uncertainties:value.uncertainties.map(s=>clean(s,1200)),citations};
 for(const field of ['current_state','evolution'])if(value[field])output[field]={summary:clean(value[field].summary,3000),evidence:evidence(value[field].evidence,true)};
 if(value.reading_preferences){output.reading_preferences={};for(const k of ['preferred','avoid']){const list=value.reading_preferences[k];if(!Array.isArray(list)||list.length>12)invalid('Préférences invalides.');output.reading_preferences[k]=list.map(x=>({label:clean(x.label,160),evidence:evidence(x.evidence,true)}));}}
 const processed=value.coverage?.processed??sources.map(s=>s.id),pending=value.coverage?.pending??[];
 if(!Array.isArray(processed)||!Array.isArray(pending)||[...processed,...pending].some(id=>!allowed.has(id))||new Set([...processed,...pending]).size!==sources.length||processed.some(id=>pending.includes(id))||pending.some(id=>proofs.has(id)))invalid('Couverture des sources invalide.');
 output.coverage={processed:[...new Set(processed)],pending:[...new Set(pending)]};
 return output;
}
export {hash as contextHash};
