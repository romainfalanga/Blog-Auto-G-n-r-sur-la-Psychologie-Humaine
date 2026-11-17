import { isEntitled, platformAccess } from './entitlements.mjs';
import {validatePsychologicalMap} from './psychology-map.mjs';
import {declaredLifeContext,assertPortraitLanguage,PORTRAIT_ANALYSIS_VERSION} from './portrait-language.mjs';
import {loadCompanionCartography} from './cartographie.mjs';
import {PERSON_P2_FIELDS,EVENT_P2_FIELDS,SELF_P2_FIELDS,PERSON_PARTS,EVENT_PARTS,personFacts,eventFacts,contextFieldsReady} from './context-fields.mjs';
// Canonical private sources used to understand the member and suggest public readings.
const parse=(v,f={})=>{try{return JSON.parse(v);}catch{return f;}};
const hash=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),b=>b.toString(16).padStart(2,'0')).join('');
export const hasPersonalizationSources=context=>(context.sources||[]).some(s=>s.type==='like'||s.explicit);
export async function loadContext(env,userId,decorateLikes=x=>x) {
 // Read bounded rows rather than aggregating an entire 500-event timeline into
 // one SQLite cell. Retry if any source changed during the snapshot reads.
 let row,likes,people,events,exposures;
 // Lot A5-P2: the optional fields of migration 0024, once it is applied.
 const extended=await contextFieldsReady(env),personExtra=extended?','+PERSON_P2_FIELDS.join(','):'',eventExtra=extended?','+EVENT_P2_FIELDS.map(k=>'e.'+k).join(','):'';
 for(let attempt=0;attempt<4;attempt++){
  row=await env.DB.prepare(`SELECT p.version AS source_version,p.self_data,p.self_revision,m.text AS memory_text,m.version AS memory_version,up.portrait,up.analyzed_version,up.schema_version,up.updated_at,s.status AS subscription_status,s.paid_until,s.trial_end FROM profiles p JOIN memories m ON m.user_id=p.user_id JOIN subscriptions s ON s.user_id=p.user_id LEFT JOIN user_portraits up ON up.user_id=p.user_id WHERE p.user_id=?`).bind(userId).first();
  if(!row)return null;
  const queries=[
   'SELECT article_id,reason,differences,revision,created_at,updated_at FROM article_likes WHERE user_id=? ORDER BY created_at,article_id LIMIT 2000',
   `SELECT id,display_name,role,custom_role,relationship_text,period_start,period_end${personExtra},revision,created_at,updated_at FROM personal_people WHERE user_id=? ORDER BY id LIMIT 50`,
   `SELECT e.id,e.title,e.kind,e.description,e.meaning_now,e.date_precision,e.date_start,e.date_end,e.age_text,e.current_importance${eventExtra},e.fiction_use,e.revision,e.created_at,e.updated_at,(SELECT json_group_array(person_id) FROM life_event_people WHERE event_id=e.id AND user_id=e.user_id) AS people_ids FROM life_events e WHERE e.user_id=? ORDER BY e.id LIMIT 500`,
   "SELECT article_id,shown_at FROM recommendation_exposures WHERE user_id=? AND shown_at>datetime('now','-7 days') ORDER BY shown_at DESC,article_id LIMIT 100"
  ];
  [likes,people,events,exposures]=await Promise.all(queries.map(async sql=>(await env.DB.prepare(sql).bind(userId).all()).results));
  const after=await env.DB.prepare('SELECT version FROM profiles WHERE user_id=?').bind(userId).first();
  if(!after)return null;
  if(after.version===row.source_version)break;
  if(attempt===3){const error=new Error('Le contexte change trop rapidement. Réessayez.');error.status=409;error.code='context_changed';throw error;}
 }
 const context={source_version:row.source_version,portrait_version:row.schema_version===2?(row.analyzed_version||0):0,updated_at:row.updated_at||null,paid:isEntitled({status:row.subscription_status,paid_until:row.paid_until,trial_end:row.trial_end}),access:platformAccess(env,{status:row.subscription_status,paid_until:row.paid_until,trial_end:row.trial_end}),likes:decorateLikes(likes),people,exposures,events:events.map(({fiction_use,...e})=>({...e,people_ids:parse(e.people_ids,[])})),memory:{text:row.memory_text,version:row.memory_version},portrait:row.schema_version===2&&row.analyzed_version===row.source_version?parse(row.portrait,null):null};
 const coverage=context.portrait?.coverage;
 if([...(coverage?.processed||[]),...(coverage?.pending||[])].some(id=>typeof id==='string'&&id.startsWith('feedback:'))){
  context.portrait=null;
  context.portrait_version=0;
 }
 const self=parse(row.self_data),narrativeKeys=['self_description','self_perception','recurring_patterns','emotions','current_context',...SELF_P2_FIELDS];
 context.has_self_description=narrativeKeys.some(k=>typeof self[k]==='string'&&self[k].trim());
 const today=new Date().toISOString().slice(0,10),birth=self.birth_date;
 // Display identity stays separate from psychological evidence. Renaming
 // needs no new inference about the person.
 context.subject={first_name:typeof self.first_name==='string'?self.first_name.trim().slice(0,100):'',gender:self.gender||'unspecified'};
 // G (audit v5): the age at a dated moment is derived from the declared birth date (never stored).
 const born=/^\d{4}-\d{2}-\d{2}$/.test(String(birth||''))?birth:null;
 for(const e of context.events){
  const m=/^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$/.exec(String(e.date_start||''));if(!born||!m||['age','unknown'].includes(e.date_precision))continue;
  const month=m[2]||null,birthMonth=born.slice(5,7),age=Number(m[1])-Number(born.slice(0,4))-(month&&(month<birthMonth||month===birthMonth&&m[3]&&m[3]<born.slice(8))?1:0);
  if(age<0||age>130)continue;e.age_at_event=age;if(!month||month===birthMonth&&!m[3])e.age_at_event_approx=true;
 }
 context.self_context=declaredLifeContext({age:birth?Number(today.slice(0,4))-Number(birth.slice(0,4))-(today.slice(5)<birth.slice(5)?1:0):null,gender:self.gender||'',gender_description:self.gender_description||'',occupation:self.occupation||''});
 const sources=[];
 const add=(id,type,revision,fields,extra={})=>{const text=Object.entries(fields).filter(([,v])=>typeof v==='string'&&v.trim()).map(([k,v])=>`${k}: ${v}`).join('\n');if(text)sources.push({id,type,revision,text,fields,explicit:true,temporal:null,created_at:null,updated_at:null,...extra});};
 if(context.has_self_description)add('profile:'+userId,'profile',row.self_revision,Object.fromEntries([...narrativeKeys,'occupation'].map(k=>[k,self[k]||''])));
 for(const l of context.likes)add('like:'+l.article_id,'like',l.revision,{title:l.title||l.article_id,article_summary:l.summary||'',subject:l.sujet||'',reason:l.reason,differences:l.differences},{explicit:Boolean(l.reason.trim()||l.differences.trim()),created_at:l.created_at,updated_at:l.updated_at});
 if(context.memory.text.trim())add('memory:'+userId,'memory',context.memory.version,{text:context.memory.text});
 // Lot A5-P2: the bond, presence and safety in words, then what helps, hurts and how it changed.
 // Only given fields enter the text, so a person without them keeps the same hash.
 const personFields=p=>({role:p.role,custom_role:p.custom_role,relationship_text:p.relationship_text,...(personFacts(p).length?{bond:personFacts(p).join(' ; ')}:{}),...Object.fromEntries(PERSON_PARTS.filter(([k])=>String(p[k]||'').trim()).map(([k])=>[k,p[k]]))});
 const eventFields=e=>({title:e.title,description:e.description,meaning_now:e.meaning_now,...Object.fromEntries(EVENT_PARTS.filter(([k])=>String(e[k]||'').trim()).map(([k])=>[k,e[k]])),...(eventFacts(e).length?{lived:eventFacts(e).join(' ; ')}:{})});
 for(const p of context.people)if(p.relationship_text.trim()||personFacts(p).length||PERSON_PARTS.some(([k])=>String(p[k]||'').trim()))add('person:'+p.id,'person',p.revision,personFields(p),{created_at:p.created_at,updated_at:p.updated_at,temporal:{start:p.period_start,end:p.period_end}});
 for(const e of context.events)if(e.description.trim()||e.meaning_now.trim()||eventFacts(e).length||EVENT_PARTS.some(([k])=>String(e[k]||'').trim()))add('event:'+e.id,'event',e.revision,eventFields(e),{created_at:e.created_at,updated_at:e.updated_at,temporal:{precision:e.date_precision,start:e.date_start,end:e.date_end,age:e.age_text},current_importance:e.current_importance,related_people:e.people_ids.map(id=>{const p=context.people.find(p=>p.id===id);return {id,role:p?.role||'',custom_role:p?.custom_role||''};})});
 // Retain the historic source hashes to reuse valid portrait extractions.
 const eventHashVersions=new Map(events.map(e=>['event:'+e.id,e.fiction_use]));
 for(const source of sources)source.hash=await hash(JSON.stringify([source.type,source.revision,source.text,source.temporal,eventHashVersions.get(source.id)||'abstract',source.related_people||[]]));
 context.portrait_complete=Boolean(context.portrait&&Array.isArray(context.portrait.psychological_map)&&!(context.portrait.coverage?.pending?.length));
 context.sources=sources;context.source_manifest=Object.fromEntries(sources.map(s=>[s.id,{revision:s.revision,hash:s.hash,type:s.type,explicit:s.explicit}]));
 // A locked childhood map saved with consent is Ambre's starting point. It is
 // kept outside sources: it neither proves a portrait claim nor changes its hashes.
 context.cartography=await loadCompanionCartography(env,userId);
 return context;
}
export function validatePortraitV2(value,sources,fail) {
 const invalid=message=>fail(400,message,'invalid_portrait');
 if(!value||value.schema_version!==2||value.fictional_character!==undefined)invalid('Un portrait V2 sourcé est requis.');
 const clean=(v,max)=>{if(typeof v!=='string'||!v.trim()||v.length>max)invalid('Texte du portrait invalide.');return v.trim();};
 const allowed=new Map(sources.map(s=>[s.id,s])),proofs=new Set(),personalProofs=new Set();
 const personalQuote=(s,quote)=>{const keys={like:['reason','differences'],person:['relationship_text','helps','hurts','evolution'],event:['description','meaning_now','felt_then','felt_now','impact'],profile:['self_description','self_perception','recurring_patterns','emotions','current_context',...SELF_P2_FIELDS],memory:['text']}[s.type]||[];const start=s.text.indexOf(quote),end=start+quote.length;return keys.some(k=>{const value=s.fields?.[k];if(!value?.trim())return false;const pos=s.text.indexOf(k+': '+value)+k.length+2;return Math.min(end,pos+value.length)-Math.max(start,pos)>=Math.min(8,value.trim().length);});};
 if(!Array.isArray(value.citations)||!value.citations.length||value.citations.length>500)invalid('Citations requises.');
 const citations=value.citations.map(p=>{const source=allowed.get(p?.source_id),quote=clean(p?.quote,1200);if(!source||p.revision!==source.revision||!source.text.includes(quote))invalid('Citation absente ou révision périmée.');proofs.add(p.source_id);if(source.explicit&&personalQuote(source,quote))personalProofs.add(p.source_id);return {source_id:p.source_id,revision:p.revision,quote};});
 const evidence=(ids,explicit=false)=>{if(!Array.isArray(ids)||!ids.length||ids.length>50||ids.some(id=>!proofs.has(id)||explicit&&!personalProofs.has(id)))invalid('Preuves du portrait invalides.');return [...new Set(ids)];};
 if(!Array.isArray(value.axes)||value.axes.length<1||value.axes.length>12)invalid('Axes du portrait invalides.');
 const axes=value.axes.map(a=>{const refs=evidence(a.evidence),bare=refs.every(id=>!personalProofs.has(id));if(!['interest','hypothesis','declaration'].includes(a.claim_type)||!Number.isFinite(a.confidence)||a.confidence<0||a.confidence>1||bare&&(a.confidence>0.25||a.claim_type!=='interest'))invalid('Un intérêt de lecture ne prouve pas un fait personnel.');return {label:clean(a.label,160),why:clean(a.why,2400),confidence:a.confidence,claim_type:a.claim_type,evidence:refs};});
 if(!Array.isArray(value.strengths)||value.strengths.length>20||!Array.isArray(value.uncertainties)||value.uncertainties.length>20)invalid('Listes du portrait invalides.');
 const output={schema_version:2,summary:clean(value.summary,6000),summary_evidence:evidence(value.summary_evidence),axes,strengths:value.strengths.map(s=>({text:clean(s.text,1200),evidence:evidence(s.evidence,true)})),uncertainties:value.uncertainties.map(s=>clean(s,1200)),citations};
 for(const field of ['current_state','evolution'])if(value[field])output[field]={summary:clean(value[field].summary,3000),evidence:evidence(value[field].evidence,true)};
 if(value.reading_preferences){output.reading_preferences={};for(const k of ['preferred','avoid']){const list=value.reading_preferences[k];if(!Array.isArray(list)||list.length>12)invalid('Préférences invalides.');output.reading_preferences[k]=list.map(x=>({label:clean(x.label,160),evidence:evidence(x.evidence,true)}));}}
 output.psychological_map=validatePsychologicalMap(value.psychological_map,sources,citations,invalid);
 if(value.analysis_version!==undefined){
  if(value.analysis_version!==PORTRAIT_ANALYSIS_VERSION)invalid('Version d’analyse inconnue.');
  if(!Array.isArray(value.psychological_map))invalid('Une carte explicite est requise pour la nouvelle analyse.');
  output.analysis_version=PORTRAIT_ANALYSIS_VERSION;
  if(value.identity_context)output.identity_context=declaredLifeContext(value.identity_context);
  try{assertPortraitLanguage(output)}catch(error){
   if(error.message==='portrait_language')invalid('La synthèse contient un pronom ambigu. Utilisez {{prenom}} ou un rôle explicite.');
   else if(error.message==='portrait_gender')invalid('La synthèse utilise un accord contraire aux informations déclarées.');
   else throw error;
  }
 }
 const processed=value.coverage?.processed??sources.map(s=>s.id),pending=value.coverage?.pending??[];
 if(!Array.isArray(processed)||!Array.isArray(pending)||[...processed,...pending].some(id=>!allowed.has(id))||new Set([...processed,...pending]).size!==sources.length||processed.some(id=>pending.includes(id))||pending.some(id=>proofs.has(id)))invalid('Couverture des sources invalide.');
 output.coverage={processed:[...new Set(processed)],pending:[...new Set(pending)]};
 return output;
}
export {hash as contextHash};
