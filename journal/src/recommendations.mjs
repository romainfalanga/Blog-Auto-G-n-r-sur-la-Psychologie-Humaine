import { PSYCHOLOGY_CATALOG } from './psychology-catalog.generated.mjs';
import {createRecommender as createBaselineRecommender} from './recommendations-baseline.mjs';
const REVIEWED_STORY_SOURCES=new Set(['editorial_pack_v10_reviewed_story','editorial_pack_v10_and_coverage_reviewed_story','psychology_coverage_reviewed_story']);
export function hasReviewedStoryLinks(article){return Array.isArray(article?.pattern_ids)&&REVIEWED_STORY_SOURCES.has(article.pattern_source)}
// Content similarity is a reading aid, never a psychological diagnosis.
// Only the public catalogue is indexed globally; all user vectors are request-local.
const STOP=new Set('avec dans pour une des les leur leurs cette cet ces est sont etre avoir plus moins tout tous toute toutes sur par qui que quoi quand comme mais donc car pas ne ni sans mon ma mes ton ta tes son sa ses nous vous ils elles elle lui aux du de le la et un en se ce il je tu me te au ou on ses tres aussi fait faire peut dont ses cela ca cet apres avant parce parle parler retrouve retrouver reconnais reconnait reconnais reconnaissant article histoire recit lecture actuellement aujourd hui surtout plutot explorer'.split(' '));
const ALIASES={familiale:'famille',familial:'famille',familiaux:'famille',professionnelle:'travail',professionnel:'travail',professionnelles:'travail',professionnels:'travail',exprime:'exprimer',expression:'exprimer',angoisse:'anxiete',angoisses:'anxiete',anxieux:'anxiete',anxieuse:'anxiete',inquietude:'anxiete',inquiet:'anxiete',inquietant:'anxiete',ruminer:'rumination',rumine:'rumination',ruminations:'rumination',suranalyse:'rumination',overthinking:'rumination',surpensee:'rumination',limites:'limite',culpabiliser:'culpabilite',culpabilise:'culpabilite',coupable:'culpabilite',perfectionniste:'perfectionnisme',perfectionnistes:'perfectionnisme',reconnaissance:'approbation',reconnaitre:'reconnaissance',epuisement:'burnout',epuise:'burnout',epuisee:'burnout',abandonne:'abandon',abandonnee:'abandon',solitaire:'solitude',seul:'solitude',seule:'solitude',colere:'colere',colerique:'colere',aider:'aide',aidant:'aide',aide:'aide',sauveur:'sauver',sauveuse:'sauver',enfants:'enfant',parentale:'parent',parental:'parent',parents:'parent',emotions:'emotion',emotionnel:'emotion',emotionnelle:'emotion',sentiments:'sentiment',confiances:'confiance'};
export function normalize(value){return String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[’']/g,' ').replace(/burn[ -]out/g,'burnout')}
function tokens(value){return normalize(value).match(/[a-z]{3,}/g)?.filter(w=>!STOP.has(w)).map(w=>ALIASES[w]||(w.length>5&&w.endsWith('s')?w.slice(0,-1):w))||[]}
function vectorText(value){const v=new Map();for(const w of tokens(value))v.set(w,(v.get(w)||0)+1);return v}
function add(to,from,weight=1){for(const [k,v] of from)to.set(k,(to.get(k)||0)+v*weight);return to}
function unit(v){const n=Math.sqrt([...v.values()].reduce((a,b)=>a+b*b,0));return n?new Map([...v].map(([k,w])=>[k,w/n])):new Map()}
function cosine(a,b){let n=0;for(const [k,v] of a)n+=v*(b.get(k)||0);return n}
function timestamp(value,fallback){if(typeof value==='number')return value<1e12?value*1000:value;const t=Date.parse(String(value||'').replace(' ','T')+(String(value||'').includes('Z')||String(value||'').includes('+')?'':'Z'));return Number.isFinite(t)?t:fallback}
function ageWeight(value,at,halfLife,floor=0){return floor+(1-floor)*Math.pow(.5,Math.max(0,at-timestamp(value,at))/(86400000*halfLife))}
// A difference qualifies resemblance; only an explicitly confirmed preference filters.
export function classifyText(value, defaultKind='declared') {
 const output=[];
 for(const part of String(value||'').split(/(?<=[.!?;\n])|\bmais\b|\ben revanche\b/i)) {
  const text=part.trim(); if(!text)continue;
  const n=normalize(text).replace(/\bpas a pas\b/g,'progressivement');
  const doubleNeg=/\b(?:ne|n)\b.{0,30}\b(?:pas|plus)\s+(?:vouloir\s+)?(?:eviter|fuir|exclure|refuser)/.test(n)||/\bpas\s+(?:envie de\s+)?(?:eviter|fuir)/.test(n);
  // A difficulty expressing a capacity is a need, not a denial of resemblance.
  // "Je n'arrive pas à poser mes limites" must retain the topic of boundaries.
  const difficulty=/\b(?:ne|n)\s+(?:arriv\w*|parvien\w*|reussi\w*|os\w*|sais|sait|sav\w*|peux|peut|pouv\w*)\s+(?:(?:vraiment|toujours)\s+)?(?:pas|plus|jamais)\b/.test(n);
  const neg=!doubleNeg&&(/\b(?:je veux|je souhaite|je prefere)\s+(?:eviter|exclure|fuir)\b/.test(n)||/\bpas\b|\b(?:ne|n)\b.{0,60}\b(?:plus|jamais)\b|\baucun rapport\b/.test(n));
  const request=/\b(?:je veux|je souhaite|je prefere|j aimerais|envie de|interesse par|explorer)\b/.test(n);
  const explicitPositive=/\b(?:je vis|je ressens|je me reconnais|pour moi|mon vecu|je veux|je souhaite|je prefere|j aimerais)\b/.test(n);
  const kind=doubleNeg?'request':difficulty?(defaultKind==='difference'?'declared':defaultKind):neg?'difference':defaultKind==='difference'&&!explicitPositive?'difference':request?'request':defaultKind==='difference'?'declared':defaultKind;
  output.push({text,kind,ambiguous:doubleNeg});
 }
 return output;
}
export const RECOMMENDATION_VERSION=2;
export const RELEVANCE_THRESHOLD=.105;
export function createRecommender(examples){
 let baseline=null;
 const rows=examples.map(e=>({...e,keywords:e.keywords||[],character_keywords:e.character_keywords||[]}));
 const byId=new Map(rows.map(e=>[e.id,e]));
 const characterNames=new Set(rows.flatMap(e=>tokens(e.character)));
 const documents=rows.map(e=>{
  const v=vectorText(e.title+' '+(e.description||''));
  add(v,vectorText(e.summary||''),.35);
  for(const w of [...e.keywords,e.sujet||'',e.canonical_concept||''])add(v,vectorText(w),3);
  // Editorially reviewed story links improve vocabulary coverage; they never create
  // a personal hypothesis or count as additional evidence about the reader.
  if(hasReviewedStoryLinks(e))for(const id of new Set(e.pattern_ids)){
   const pattern=PSYCHOLOGY_CATALOG.patterns.find(p=>p.id===id);if(pattern)add(v,vectorText(pattern.label),.75);
  }
  // Character identity is not a signal about the reader or the article's topic.
  return v;
 });
 const df=new Map();for(const v of documents)for(const k of v.keys())df.set(k,(df.get(k)||0)+1);
 const weighted=v=>unit(new Map([...v].filter(([k])=>!characterNames.has(k)&&df.has(k)).map(([k,w])=>[k,Math.log1p(Math.max(0,w))*(1+Math.log((rows.length+1)/((df.get(k)||0)+1)))])));
 const vectors=new Map(rows.map((e,i)=>[e.id,weighted(documents[i])]));
 const lex=v=>weighted(vectorText(v));
 function profile(context,at,query=''){
  const signals=[],seen=new Set(),evidence=new Map();
  const likes=(context.likes||[]).filter(l=>byId.has(l.article_id||l.id));
  function signal(id,value,kind,weight,updated,options={}){
   const key=id+':'+normalize(value);if(seen.has(key)||!String(value||'').trim())return;seen.add(key);
   const temporal=kind==='like'?ageWeight(updated,at,120,.4):kind==='request'?ageWeight(updated,at,90,.65):1;
   const v=options.vector||lex(value);if(v.size)signals.push({id,text:value,kind,weight:weight*temporal,v,declared_at:timestamp(updated,0),...options});
  }
  function source(id,value,kind,weight,updated,options={}){for(const part of classifyText(value,kind))signal(id,part.text,part.kind,part.kind==='request'?4:part.kind==='difference'?1.5:weight,updated,{ambiguous:part.ambiguous,...options});}
  for(const l of likes){
   const id=l.article_id||l.id,e=byId.get(id);
   signal('like:'+id,e.title,'like',1,l.created_at,{vector:vectors.get(id)});
   source('like:'+id+':reason',l.reason,'resonance',2,l.updated_at);
   source('like:'+id+':differences',l.differences,'difference',1.5,l.updated_at);
   for(const label of e.keywords){const key=normalize(label),x=evidence.get(key)||{label,ids:[]};if(!x.ids.includes(id))x.ids.push(id);evidence.set(key,x);}
  }
  for(const s of context.sources||[])if(s.type==='profile')for(const key of ['self_description','self_perception','recurring_patterns','emotions','current_context'])source(s.id,s.fields?.[key],'declared',3,s.updated_at);
  source('memory',context.memory?.text,'declared',3,context.memory?.updated_at);
  for(const person of context.people||context.personal_people||[])source('person:'+person.id,person.relationship_text,'declared',3,person.updated_at);
  for(const e of context.life_events||context.events||[]){const importance=Number.isInteger(e.current_importance)&&e.current_importance>=1&&e.current_importance<=5?0.8+(e.current_importance-1)*.1:1;source('event:'+e.id,e.description,'declared',3*importance,e.updated_at);source('event:'+e.id+':meaning',e.meaning_now,'declared',3*importance,e.updated_at);}
  for(const f of context.feedback||[]){
   const id='feedback:'+(f.id||f.story_id);
   source(id+':resemblance',f.resemblance||f.why,Number(f.resonance)>=3?'resonance':'difference',2,f.updated_at);
   source(id+':feelings',f.feelings,'declared',3,f.updated_at);
   source(id+':differences',f.differences,'difference',1.5,f.updated_at);
   source(id+':correction',f.correction,'declared',3,f.updated_at);
   source(id+':wish',f.preference||f.reading_wish||f.wish,'request',4,f.updated_at,{reading_preference:true});
   if(String(f.avoid||'').trim()&&!classifyText(f.avoid).some(p=>p.ambiguous))signal(id+':avoid',f.avoid,'avoid',4,f.updated_at);
   if(f.avoid_confirmed===true&&String(f.avoid_topic||'').trim()&&!classifyText(f.avoid_topic).some(p=>p.ambiguous))signal(id+':avoid',f.avoid_topic,'avoid',4,f.updated_at);
  }
  for(const a of context.avoid_topics||[])if(a.confirmed===true&&!classifyText(a.label||a.topic).some(p=>p.ambiguous))signal('avoid:'+a.id,a.label||a.topic,'avoid',4,a.updated_at);
  // The question is request-local retrieval context, not a saved personal fact.
  for(const part of classifyText(query,'query')){
   const n=normalize(part.text),reading=/\b(?:articles?|lectures?|lire|sujets?|themes?)\b/.test(n),avoid=reading&&/\b(?:je veux|je souhaite|je prefere|j aimerais)\s+(?:eviter|exclure|fuir)\b/.test(n)&&!part.ambiguous;
   // Merely mentioning a reading (or being unable to enjoy it) does not
   // withdraw a previously confirmed avoidance preference.
   const intention=/\b(?:je veux|je souhaite|je prefere|j aimerais|j ai envie de)\s+(?:(?:bien|plutot|maintenant|a nouveau|de nouveau)\s+)*(?:(?:lire|decouvrir|explorer|consulter)\b|(?:un|une|des|de l)\s+(?:articles?|lectures?)\b)/.test(n)||/\b(?:propose|proposez|recommande|recommandez|suggere|suggerez|trouve|trouvez)(?:\s+moi)?\b.{0,40}\b(?:articles?|lectures?)\b/.test(n);
   signal('query:current',part.text,avoid?'avoid':part.kind,avoid?4:part.kind==='difference'?1.5:4,at,{ambiguous:part.ambiguous,explicit_reading_request:reading&&intention&&!n.startsWith('themes de recherche :')&&!avoid&&!part.ambiguous&&part.kind!=='difference'});
  }
  // Typed originals are useful when a caller supplies only the canonical snapshot.
  if(!likes.length&&!context.memory?.text&&!(context.people||context.personal_people||[]).length&&!(context.life_events||context.events||[]).length&&!(context.feedback||[]).length){
   for(const s of context.sources||[]){
    if(s.type==='like'||s.type==='profile')continue;
    source(s.id,s.text,s.kind==='difference'?'difference':'declared',3,s.updated_at);
   }
  }
  // Controlled concepts improve retrieval, but never count a model derivative
  // as a new independent source. User corrections can remove a map hypothesis.
  const catalogue=new Map(PSYCHOLOGY_CATALOG.patterns.map(p=>[p.id,p]));
  const originals=new Map((context.sources||[]).map(s=>[s.id,s]));
  for(const hypothesis of context.portrait?.psychological_map||[]){
   const concept=catalogue.get(hypothesis.pattern_id);
   if(!concept||!['exploring','supported'].includes(hypothesis.status)||!Array.isArray(hypothesis.evidence)||!hypothesis.evidence.length||hypothesis.evidence.some(id=>!originals.get(id)?.explicit))continue;
   for(const id of hypothesis.evidence){const original=originals.get(id);signal(original.type==='memory'?'memory':id,[concept.label,...concept.keywords].join(' '),'hypothesis',(hypothesis.status==='supported'?1:.6)*({low:.8,moderate:1,high:1.2}[hypothesis.influence?.level]||1),original.updated_at);}
  }
  // Never let the same original evidence gain weight by reappearing in a portrait.
  // Copying a personal statement into another collection does not create a
  // second independent reason. Join source groups by exact normalized wording;
  // retain distinct reading likes and distinct accounts of similar situations.
  const parent=new Map(),find=id=>{if(!parent.has(id))parent.set(id,id);if(parent.get(id)!==id)parent.set(id,find(parent.get(id)));return parent.get(id);};
  const baseId=id=>id.replace(/:(?:reason|differences|meaning|resemblance|feelings|correction|wish)$/,'');
  const fingerprints=new Map();for(const s of signals){const id=baseId(s.id);find(id);if(['like','hypothesis','query','avoid'].includes(s.kind)||s.id==='query:current')continue;const key=normalize(s.text).replace(/[^a-z0-9]+/g,' ').trim();if(!key)continue;const prior=fingerprints.get(key);if(prior)parent.set(find(id),find(prior));else fingerprints.set(key,id);}
  for(const s of signals)s.evidence_group=find(baseId(s.id));
  return {signals,likes,evidence};
 }
 function recommend(context={},options={}){
  if(options.algorithm==='baseline'){baseline??=createBaselineRecommender(examples);return baseline.recommend(context,options);}
  const at=options.now??Date.now(),limit=Math.max(1,Math.min(30,Number(options.limit)||12)),p=profile(context,at,String(options.query||'')),excluded=new Set(options.includeLiked?[]:p.likes.map(l=>l.article_id||l.id));
  const positives=p.signals.filter(s=>!['difference','avoid'].includes(s.kind));
  if(!positives.length)return [];
  const negatives=p.signals.filter(s=>s.kind==='difference'),avoids=p.signals.filter(s=>s.kind==='avoid');
  const exposures=new Set((context.exposures||[]).filter(x=>at-timestamp(x.shown_at,0)<7*86400000).slice(-100).map(x=>x.article_id));
  const candidates=rows.filter(e=>!excluded.has(e.id)).map(e=>{
   const v=vectors.get(e.id);if(avoids.some(a=>cosine(a.v,lex(e.keywords.join(' ')+' '+e.sujet))>.28&&!positives.some(s=>(s.reading_preference||s.explicit_reading_request)&&!s.ambiguous&&s.declared_at>a.declared_at&&cosine(s.v,a.v)>.3)))return null;
   // Per-source maximum deduplicates multiple phrases from the same source, while
   // the fixed scale preserves explicit-vs-like intensity before saturation.
   const contributions=positives.map(s=>({...s,similarity:cosine(s.v,v)}));
   const grouped=new Map();for(const s of contributions){const id=s.evidence_group;grouped.set(id,Math.max(grouped.get(id)||0,s.weight*s.similarity));}
   const affinity=[...grouped.values()].sort((a,b)=>b-a).slice(0,12).reduce((a,b)=>a+b,0);
   const correction=Math.max(0,...negatives.map(s=>s.weight*cosine(s.v,v)));
   const score=Math.max(0,(1-Math.exp(-affinity/3))-.24*correction-(exposures.has(e.id)?.035:0));
   const strongest=contributions.sort((a,b)=>b.similarity*b.weight-a.similarity*a.weight)[0];
   const labels={query:'Votre question actuelle',request:'Votre souhait de lecture',declared:'Votre contexte déclaré',resonance:'Ce qui vous parle dans vos lectures',like:'Vos intérêts de lecture',hypothesis:'Une piste à explorer dans votre cartographie'};
   const themes=e.keywords.filter(k=>cosine(lex(k),strongest.v)>.08).slice(0,2);
   const reason=(labels[strongest.kind]||'Votre contexte déclaré')+(themes.length?' rejoint '+themes.join(', '):' rejoint certains thèmes de ce récit')+'.';
   return {e,v,score,reasons:[reason],reason_sources:[{source_id:strongest.id,kind:strongest.kind}],theme:normalize(e.canonical_concept||e.sujet||e.keywords[0])};
  }).filter(c=>c&&c.score>=(options.threshold??RELEVANCE_THRESHOLD)).sort((a,b)=>b.score-a.score||a.e.id.localeCompare(b.e.id));
  const selected=[],counts=new Map(),themes=new Map();
  while(selected.length<limit&&candidates.length){
   let pool=candidates.filter(c=>(counts.get(c.e.character)||0)<(selected.length<6?2:Math.max(2,Math.ceil(limit/3))));
   if(!pool.length&&selected.length<6)break;
   if(!pool.length)pool=candidates;
   let best=null,bestValue=-Infinity;
   for(const c of pool){const duplicate=Math.max(0,...selected.map(s=>cosine(c.v,s.v)));const value=c.score-.06*duplicate-.025*(themes.get(c.theme)||0);if(value>bestValue){bestValue=value;best=c;}}
   selected.push(best);counts.set(best.e.character,(counts.get(best.e.character)||0)+1);themes.set(best.theme,(themes.get(best.theme)||0)+1);candidates.splice(candidates.indexOf(best),1);
  }
  return selected.map(({e,reasons,reason_sources,score})=>({id:e.id,title:e.title,url:e.url,character:e.character,description:e.description||'',keywords:e.keywords.slice(0,3),reasons,reason_sources,relevance:Math.round(score*10000)/10000,recommendation_version:2}));
 }
 function provisionalPortrait(context={}){
  const p=profile(context,Date.now());if(!p.signals.length)return null;
  const axes=[...p.evidence.values()].sort((a,b)=>b.ids.length-a.ids.length||a.label.localeCompare(b.label,'fr')).slice(0,5).map(x=>({label:x.label,why:'Thème de lecture : '+x.ids.length+' article(s) aimé(s), sans conclusion sur votre vécu.',confidence:Math.min(.25,.12+x.ids.length*.02),evidence:x.ids}));
  return {summary:p.likes.length?'Vos lectures donnent des pistes provisoires, à préciser avec votre vécu.':'Votre contexte déclaré permet de préparer votre portrait, même sans article aimé.',axes,strengths:[],uncertainties:['Les récits et les personnages sont fictifs. Un intérêt ne prouve pas un vécu.'],provisional:true};
 }
 return {recommend,provisionalPortrait};
}
