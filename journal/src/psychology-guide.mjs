import {PSYCHOLOGY_CATALOG as catalog} from './psychology-catalog.generated.mjs';
const fold=value=>String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
const stop=new Set('avec dans pour une des les leur cette ces est sont etre avoir plus moins tout sur par qui que quoi quand comme mais donc pas sans mon mes son nous vous elle lui aux du de le la et un en se ce il je tu au ou on tres aussi fait faire peut personne article histoire'.split(' '));
const terms=value=>new Set((fold(value).match(/[a-z]{4,}/g)||[]).filter(t=>!stop.has(t)));
// Retrieval broadens the hypotheses available without placing hundreds of full
// interview entries in every turn. Its score is text relevance, never likelihood.
export function selectPsychologyGuide({query='',sources=[],hypotheses=[],limit=24,includeIndex=true}={}){
 const queryTerms=terms(query),sourceTerms=terms(sources.map(s=>s.text||JSON.stringify(s.fields||s.derivative||{})).join(' '));
 // Contradicted or dismissed hypotheses remain available as limits in the
 // conversation, but must not steer the next exploration as active patterns.
 const current=new Set(hypotheses.filter(h=>h.status===undefined||['exploring','supported','mixed'].includes(h.status)).map(h=>h.pattern_id)),related=new Set(catalog.patterns.filter(p=>current.has(p.id)).flatMap(p=>p.related||[]));
 const ranked=catalog.patterns.map((p,index)=>{
  const keys=terms([p.id,p.label,...(p.keywords||[]),...(p.signals||[])].join(' '));
  let queryScore=0,score=current.has(p.id)?100:related.has(p.id)?4:0;
  for(const t of keys){if(queryTerms.has(t))queryScore+=5;score+=sourceTerms.has(t)?1:0;}
  return {p,score:score+queryScore,queryScore,index};
 }).sort((a,b)=>b.score-a.score||a.index-b.index);
 const selected=[],seen=new Set();let activeCount=0;
 const take=(row,direct=false)=>{if(!row||seen.has(row.p.id)||selected.length>=limit||!direct&&current.has(row.p.id)&&activeCount>=Math.floor(limit/2))return false;seen.add(row.p.id);selected.push(row.p);if(current.has(row.p.id))activeCount++;return true;};
 // Reserve two details for the question itself before older hypotheses or
 // catalogue domain order can fill a small companion guide.
 for(const row of ranked.filter(r=>r.queryScore>0).sort((a,b)=>b.queryScore-a.queryScore||b.score-a.score||a.index-b.index).slice(0,2))take(row,true);
 for(const row of ranked.filter(r=>current.has(r.p.id)))take(row);
 // Cover relevant domains in score order, without repeating those represented
 // by the direct question or the bounded active hypotheses above.
 const represented=new Set(selected.map(p=>p.domain));
 for(const row of ranked)if(row.score>0&&!represented.has(row.p.domain)&&take(row))represented.add(row.p.domain);
 for(const row of ranked)take(row);
 return {version:catalog.version,scope:catalog.scope,domains:catalog.domains,
  ...(includeIndex?{index:catalog.patterns.map(p=>[p.id,p.domain,p.label])}:{}),
  patterns:selected.map(p=>({id:p.id,domain:p.domain,subdomain:p.subdomain,kind:p.kind,label:p.label,description:p.description,signals:(p.signals||[]).slice(0,3),alternatives:(p.alternatives||[]).slice(0,2),questions:(p.questions||[]).slice(0,2),related:p.related})),
  selection:{total:catalog.patterns.length,detailed:selected.length,meaning:'Repères pour explorer, pas une évaluation exhaustive. Un repère non retenu reste accessible via son ID dans l’index.'}};
}
