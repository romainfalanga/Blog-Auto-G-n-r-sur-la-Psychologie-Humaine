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
function noteParts(value){
 const positive=[],negative=[];
 for(const sentence of String(value||'').split(/(?<=[.!?;\n])|\bmais\b|\ben revanche\b/i)){
  const n=normalize(sentence).replace(/\bpas a pas\b/g,'progressivement');
  const neg=/\bpas\b|\b(?:ne|n)\b.{0,65}\b(?:pas|plus|jamais)\b|\b(?:pas du tout|ne me ressemble|contraire|a eviter|eviter ce|aucun rapport|pas de|pas ma|pas mon|pas dans|sans rapport)\b/.test(n);
  (neg?negative:positive).push(sentence);
 }
 return {positive:positive.join(' '),negative:negative.join(' ')};
}
function note(value,at,updated){const parts=noteParts(value);return {positive:parts.positive,negative:parts.negative,weight:ageWeight(updated,at,30,.3)}}
export function createRecommender(examples){
 const rows=examples.map(e=>({...e,keywords:e.keywords||[],character_keywords:e.character_keywords||[]}));
 const byId=new Map(rows.map(e=>[e.id,e]));
 const characterNames=new Set(rows.flatMap(e=>tokens(e.character))); 
 const documents=rows.map(e=>{
  const v=vectorText(e.title+' '+(e.description||''));
  for(const w of [...e.keywords,e.sujet||''])add(v,vectorText(w),3);
  // Character traits aid discovery, with less weight than the actual article's subject.
  for(const w of e.character_keywords)add(v,vectorText(w),.35);
  return v;
 });
 const df=new Map();for(const v of documents)for(const k of v.keys())df.set(k,(df.get(k)||0)+1);
 const idf=w=>1+Math.log((rows.length+1)/((df.get(w)||0)+1));
 const weighted=v=>unit(new Map([...v].filter(([k])=>!characterNames.has(k)&&df.has(k)).map(([k,w])=>[k,Math.log1p(w)*idf(k)])));
 const vectors=new Map(rows.map((e,i)=>[e.id,weighted(documents[i])]));
 const lex=v=>weighted(vectorText(v));
 function profile(context,at){
  const base=new Map(),explicit=new Map(),negative=new Map(),evidence=new Map(),characters=new Map();
  const likes=(context.likes||[]).filter(l=>byId.has(l.article_id||l.id));
  for(const l of likes){
   const id=l.article_id||l.id,e=byId.get(id),weight=ageWeight(l.created_at,at,120,.4);
   add(base,vectors.get(id),weight);
   const n=note(l.reason||'',at,l.updated_at);add(explicit,lex(n.positive),2.4*n.weight);add(negative,lex(n.negative),1.7*n.weight);
   // Differences are corrections, not automatic preferences or a biography.
   const differences=note(l.differences||'',at,l.updated_at);add(negative,lex(differences.negative),1.8*differences.weight);add(explicit,lex(differences.positive),.7*differences.weight);
   const traits=lex(e.character_keywords.join(' '));
   const supported=cosine(lex(n.positive),traits)>.08;
   const c=characters.get(e.character)||{count:0,supported:false};c.count++;c.supported||=supported;characters.set(e.character,c);
   for(const label of e.keywords){const k=normalize(label);const x=evidence.get(k)||{label,weight:0,ids:[]};x.weight+=weight;if(!x.ids.includes(id))x.ids.push(id);evidence.set(k,x)}
  }
  const memory=note(context.memory?.text||'',at,context.memory?.updated_at);add(explicit,lex(memory.positive),1.5);add(negative,lex(memory.negative),2);
  for(const f of context.feedback||[]){
   const n=note([f.why,f.feelings].filter(Boolean).join(' '),at,f.updated_at),correction=note(f.correction||'',at,f.updated_at);
   if(Number(f.resonance)>=3)add(explicit,lex(n.positive),1.8*n.weight);
   else add(negative,lex(n.positive),1.8*n.weight);
   add(negative,lex(n.negative),1.8*n.weight);
   add(explicit,lex(correction.positive),1.7*correction.weight);add(negative,lex(correction.negative),2.2*correction.weight);
   // The fiction itself only becomes a weak affinity when positively confirmed.
   if(Number(f.resonance)>=4)add(base,lex(f.fictional_source_summary||''),.35*n.weight);
  }
  const p=context.portrait;
  if(p){
   for(const a of p.axes||[])add(base,lex(a.label),Math.min(.4,Number(a.confidence)||.1));
   for(const a of p.reading_preferences?.preferred||[])add(explicit,lex(a.label),.65);
   for(const a of p.reading_preferences?.avoid||[])add(negative,lex(a.label),.8);
   // Only explicitly sourced current state enters recommendations; never fictional_character.
   if(p.current_state?.evidence?.length)add(explicit,lex(p.current_state.summary),.35);
  }
  return {positive:unit(add(unit(base),unit(explicit),explicit.size?2.2:0)),negative:unit(negative),explicit:unit(explicit),characters,evidence,likes};
 }
 function recommend(context={},options={}){
  const at=options.now??Date.now(),limit=Math.max(1,Math.min(30,Number(options.limit)||12)),p=profile(context,at),excluded=new Set(p.likes.map(l=>l.article_id||l.id));
  if(!p.likes.length)return [];
  const candidates=rows.filter(e=>!excluded.has(e.id)).map(e=>{
   const v=vectors.get(e.id),similarity=cosine(p.positive,v),avoid=cosine(p.negative,v),c=p.characters.get(e.character);
   const continuity=c&&(c.supported||c.count>=2)&&similarity>.12?Math.min(.05,c.count*.015):0;
   const score=Math.max(0,similarity-.8*avoid+continuity);
   const terms=[...e.keywords].map(label=>({label,score:cosine(lex(label),p.positive)-.8*cosine(lex(label),p.negative)})).sort((a,b)=>b.score-a.score);
   const matched=terms.filter(t=>t.score>.07).slice(0,2).map(t=>t.label);
   const reasons=matched.length?['Des thèmes proches de vos lectures : '+matched.join(', ')+'.']:similarity>.08?['Des situations proches de vos lectures et de vos précisions.']:['Une autre piste de lecture à explorer.'];
   if(continuity)reasons.push('Un autre récit de '+e.character+', avec des points communs à explorer.');
   return {e,v,score,reasons};
  }).sort((a,b)=>b.score-a.score||a.e.id.localeCompare(b.e.id));
  const selected=[],counts=new Map();
  while(selected.length<limit&&candidates.length){
   let pool=candidates.filter(x=>(counts.get(x.e.character)||0)<(selected.length<6?2:Math.max(2,Math.ceil(limit/3))));
   if(!pool.length)pool=candidates;
   // The sixth slot explores another character while retaining meaningful similarity.
   if(selected.length===5&&p.likes.length){const other=pool.filter(x=>!p.characters.has(x.e.character)&&x.score>=candidates[0].score*.25);if(other.length)pool=other;}
   let best=null,bestValue=-Infinity;
   for(const c of pool){const duplicate=Math.max(0,...selected.map(s=>cosine(c.v,s.v)));const value=.82*c.score-.18*duplicate;if(value>bestValue){bestValue=value;best=c}}
   selected.push(best);counts.set(best.e.character,(counts.get(best.e.character)||0)+1);candidates.splice(candidates.indexOf(best),1);
  }
  return selected.map(({e,reasons})=>({id:e.id,title:e.title,url:e.url,character:e.character,description:e.description||'',keywords:e.keywords.slice(0,3),reasons}));
 }
 function provisionalPortrait(context={}){
  const p=profile({...context,portrait:null},Date.now());if(!p.likes.length)return null;
  const axes=[...p.evidence.values()].sort((a,b)=>b.weight-a.weight||a.label.localeCompare(b.label,'fr')).slice(0,5).map(x=>({label:x.label,why:'Ce thème apparaît dans '+x.ids.length+' de vos articles aimés. Votre explication peut préciser ce qui vous touche et ce qui vous ressemble moins.',confidence:Math.min(.45,.12+x.ids.length*.04),evidence:x.ids}));
  return {summary:p.likes.length===1?'Un premier article donne des pistes de lecture. Ces repères décrivent votre choix, pas encore votre fonctionnement psychologique.':'Vos lectures font apparaître des thèmes à explorer. Ces repères sont provisoires et évoluent avec vos explications et vos corrections.',axes,strengths:[],uncertainties:['Aimer une histoire ne signifie pas avoir vécu les situations de son personnage.','Les raisons de vos choix et ce qui ne vous ressemble pas permettent de nuancer ces rapprochements.'],provisional:true};
 }
 return {recommend,provisionalPortrait};
}
