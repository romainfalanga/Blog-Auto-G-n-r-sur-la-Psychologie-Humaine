import { PSYCHOLOGY_CATALOG } from './psychology-catalog.generated.mjs';
export { PSYCHOLOGY_CATALOG };
const patterns=new Map(PSYCHOLOGY_CATALOG.patterns.map(p=>[p.id,p]));
const PERSONAL={profile:['self_description','self_perception','recurring_patterns','emotions','current_context'],like:['reason','differences'],person:['relationship_text'],event:['description','meaning_now'],feedback:['feelings','why','differences','correction','preference','avoid'],memory:['text']};
function personalQuote(source,quote){return !!source?.explicit&&(PERSONAL[source.type]||[]).some(key=>{const value=source.fields?.[key];return typeof value==='string'&&value.trim()&&value.includes(quote)});}
const fingerprint=value=>String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]+/g,' ').trim();
function independentProofs(ids,allowed,citations){const groups=[];for(const id of ids){const source=allowed.get(id),text=fingerprint((PERSONAL[source.type]||[]).map(k=>source.fields?.[k]||'').join(' ')),quotes=citations.filter(c=>c.source_id===id&&personalQuote(source,c.quote)).map(c=>fingerprint(c.quote));if(groups.some(g=>g.text===text||g.quotes.some(q=>quotes.some(v=>q===v||Math.min(q.length,v.length)>=20&&(q.includes(v)||v.includes(q))))))continue;groups.push({text,quotes})}return groups.length;}
export function validatePsychologicalMap(value,sources,citations,invalid=message=>{throw Error(message)}){
 if(value===undefined)return [];
 if(!Array.isArray(value)||value.length>24)return invalid('Cartographie invalide.');
 const allowed=new Map(sources.map(s=>[s.id,s])),proofs=new Set(citations.filter(c=>personalQuote(allowed.get(c.source_id),c.quote)).map(c=>c.source_id)),seen=new Set();
 const clean=(v,max)=>{if(typeof v!=='string'||!v.trim()||v.length>max)return invalid('Texte de cartographie invalide.');return v.trim()};
 const refs=(v,required=false)=>{if(!Array.isArray(v)||v.length>20||required&&!v.length||v.some(id=>!proofs.has(id)))return invalid('Preuves personnelles de cartographie requises.');return [...new Set(v)]};
 const activeIds=new Set(value.map(h=>h?.pattern_id));
 return value.map(h=>{
  if(!h||!patterns.has(h.pattern_id)||seen.has(h.pattern_id)||!['exploring','supported','mixed','contradicted'].includes(h.status))return invalid('Piste psychologique inconnue.');seen.add(h.pattern_id);
  const evidence=refs(h.evidence,true),counter_evidence=refs(h.counter_evidence||[]),types=new Set(evidence.map(id=>allowed.get(id).type));
  if(['mixed','contradicted'].includes(h.status)&&!counter_evidence.length)return invalid('Une piste nuancée doit citer ses contre-indices.');
  if(!Array.isArray(h.alternatives)||!h.alternatives.length||h.alternatives.length>4)return invalid('Explications alternatives requises.');
  // Multiple copies of a source or many likes never constitute independent support.
  const status=h.status==='supported'&&(independentProofs(evidence,allowed,citations)<2||types.size<2)?'exploring':h.status;
  const output={pattern_id:h.pattern_id,status,observation:clean(h.observation,1800),possible_function:clean(h.possible_function,1200),alternatives:h.alternatives.map(x=>clean(x,500)),next_question:clean(h.next_question,500),evidence,counter_evidence};
  const ownProofs=new Set([...evidence,...counter_evidence]);
  if(h.influence!==undefined){
   const x=h.influence;if(!x||!['unknown','low','moderate','high'].includes(x.level)||!Array.isArray(x.contexts)||x.contexts.length>3)return invalid('Influence contextuelle invalide.');
   const ids=refs(x.evidence,x.level!=='unknown');if(ids.some(id=>!ownProofs.has(id)))return invalid('Influence sans preuve de cette piste.');
   if(status==='contradicted'&&x.level!=='unknown'||!ids.length&&x.contexts.length)return invalid('Influence non étayée.');
   output.influence={level:x.level,contexts:x.contexts.map(v=>clean(v,160)),rationale:ids.length?clean(x.rationale,600):'Influence à préciser.',evidence:ids};
  }
  if(h.interactions!==undefined){
   if(!Array.isArray(h.interactions)||h.interactions.length>3||status==='contradicted'&&h.interactions.length)return invalid('Interactions invalides.');
   const targets=new Set();output.interactions=h.interactions.map(x=>{
    if(!x||x.pattern_id===h.pattern_id||!activeIds.has(x.pattern_id)||!patterns.has(x.pattern_id)||targets.has(x.pattern_id)||!['amplifies','buffers','alternative'].includes(x.kind))return invalid('Mécanisme lié inconnu.');
    targets.add(x.pattern_id);const ids=refs(x.evidence,true);
    const target=value.find(v=>v.pattern_id===x.pattern_id),targetProofs=new Set([...(target.evidence||[]),...(target.counter_evidence||[])]);
    if(target.status==='contradicted')return invalid('Une piste contredite ne peut pas amplifier une autre piste.');
    if(!ids.some(id=>ownProofs.has(id))||!ids.some(id=>targetProofs.has(id)))return invalid('Interaction sans preuves des deux pistes.');
    return {pattern_id:x.pattern_id,kind:x.kind,explanation:clean(x.explanation,600),evidence:ids};
   });
  }
  return output;
 });
}
const norm=s=>String(s||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
const words=s=>new Set(norm(s).match(/[a-z]{4,}/g)||[]);
function relatedArticles(pattern,examples){const keys=words([pattern.label,...pattern.keywords].join(' '));return examples.map(e=>{const text=words([e.title,e.sujet,e.canonical_concept,...(e.keywords||[])].join(' '));return {e,n:[...keys].filter(k=>text.has(k)).length}}).filter(x=>x.n>0).sort((a,b)=>b.n-a.n||a.e.id.localeCompare(b.e.id)).slice(0,3).map(({e})=>({id:e.id,title:e.title,url:e.url,character:e.character}));}
export function buildPsychologyMap(context,examples=[]){
 const portrait=context.portrait,allowed=new Map((context.sources||[]).map(s=>[s.id,s]));
 let active=[];
 // Fail closed for stale/legacy maps. Unknown patterns remain available to explore.
 try{active=validatePsychologicalMap(portrait?.psychological_map,context.sources||[],portrait?.citations||[])}catch{active=[]}
 const hypotheses=active.map(h=>{const ids=[...new Set([...h.evidence,...h.counter_evidence])],citations=(portrait.citations||[]).filter(c=>ids.includes(c.source_id));const articles=relatedArticles(patterns.get(h.pattern_id),examples),characters=new Map();for(const a of articles)if(a.character){const slug=norm(a.character).replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');characters.set(a.character,{name:a.character,url:'/personnages/'+slug+'/'})}return {...h,support_level:h.status==='mixed'?'mixed':h.status==='contradicted'?'contradicted':h.status==='supported'?'convergent':'emerging',source_count:independentProofs(h.evidence,allowed,portrait.citations||[]),source_types:[...new Set(h.evidence.map(id=>allowed.get(id)?.type))],citations,related_articles:articles,related_characters:[...characters.values()]};});
 return {catalog_version:PSYCHOLOGY_CATALOG.version,domains:PSYCHOLOGY_CATALOG.domains,subdomains:PSYCHOLOGY_CATALOG.subdomains||[],patterns:PSYCHOLOGY_CATALOG.patterns,hypotheses,source_version:context.source_version,analyzed_version:context.portrait_version,outdated:!context.portrait_complete||!portrait?.psychological_map,scope:PSYCHOLOGY_CATALOG.scope};
}
