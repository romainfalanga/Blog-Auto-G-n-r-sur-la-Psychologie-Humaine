import {PSYCHOLOGY_CATALOG as catalog} from './psychology-catalog.generated.mjs';
const fold=value=>String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
const stop=new Set('avec dans pour une des les leur cette ces est sont etre avoir plus moins tout sur par qui que quoi quand comme mais donc pas sans mon mes son nous vous elle lui aux du de le la et un en se ce il je tu au ou on tres aussi fait faire peut personne article histoire'.split(' '));
const terms=value=>new Set((fold(value).match(/[a-z]{4,}/g)||[]).filter(t=>!stop.has(t)));
// Retrieval broadens the hypotheses available without placing hundreds of full
// interview entries in every turn. Its score is text relevance, never likelihood.
export function selectPsychologyGuide({query='',sources=[],hypotheses=[],limit=24,includeIndex=true}={}){
 const queryTerms=terms(query),sourceTerms=terms(sources.map(s=>s.text||JSON.stringify(s.fields||s.derivative||{})).join(' '));
 const current=new Set(hypotheses.map(h=>h.pattern_id)),related=new Set(catalog.patterns.filter(p=>current.has(p.id)).flatMap(p=>p.related||[]));
 const ranked=catalog.patterns.map((p,index)=>{
  const keys=terms([p.id,p.label,...(p.keywords||[]),...(p.signals||[])].join(' '));
  let score=current.has(p.id)?100:related.has(p.id)?4:0;
  for(const t of keys)score+=(queryTerms.has(t)?5:0)+(sourceTerms.has(t)?1:0);
  return {p,score,index};
 }).sort((a,b)=>b.score-a.score||a.index-b.index);
 const selected=[],seen=new Set(),take=row=>{if(row&&!seen.has(row.p.id)&&selected.length<limit){seen.add(row.p.id);selected.push(row.p);}};
 for(const row of ranked.filter(r=>current.has(r.p.id)))take(row);
 // A high-scoring domain cannot completely crowd out another relevant domain.
 for(const d of catalog.domains)take(ranked.find(r=>r.p.domain===d.id&&r.score>0));
 for(const row of ranked)take(row);
 return {version:catalog.version,scope:catalog.scope,domains:catalog.domains,
  ...(includeIndex?{index:catalog.patterns.map(p=>[p.id,p.domain,p.label])}:{}),
  patterns:selected.map(p=>({id:p.id,domain:p.domain,subdomain:p.subdomain,kind:p.kind,label:p.label,description:p.description,signals:(p.signals||[]).slice(0,3),alternatives:(p.alternatives||[]).slice(0,2),questions:(p.questions||[]).slice(0,2),related:p.related})),
  selection:{total:catalog.patterns.length,detailed:selected.length,meaning:'Repères pour explorer, pas une évaluation exhaustive. Un repère non retenu reste accessible via son ID dans l’index.'}};
}
