import {classifyText} from './recommendations.mjs';
import {PSYCHOLOGY_CATALOG} from './psychology-catalog.generated.mjs';
const fold=value=>String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[’']/g,' ').replace(/[^a-z0-9]+/g,' ').trim();
const saviourPattern=/\b(?:(?:syndrome|complexe) (?:du sauveur|de la sauveuse)|role de (?:sauveur|sauveuse)|surresponsabilite|(?:besoin de )?sauver (?:les autres|tout le monde))\b/g;
const saviourIntent=value=>new RegExp(saviourPattern.source).test(fold(value));
// Ordinary French wording -> editorial search vocabulary. These additions are
// transient retrieval terms, never declarations, portrait evidence or diagnoses.
export function expandReadingQuery(value){
 const query=String(value||'').trim().slice(0,6000);
 // Keep the original request intact. Only affirmative, unambiguous clauses may
 // add editorial vocabulary; an omitted negation must never become a new wish.
 const clauses=classifyText(query.replace(/,\s*(?=je\b|j['’]|surtout quand\b)|\bsurtout quand\b/gi,'\n'));
 const folded=clauses.filter(p=>p.kind!=='difference'&&!p.ambiguous).map(p=>p.text).join('\n').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[’']/g,' '),themes=[];
 if(/\brefus(?:er|e|es|ent|ais|ait|ons)?\b|\bdire non\b|\bpos(?:er|e) (?:mes |ses |des )?limites\b/.test(folded))themes.push('limites affirmation de soi');
 if(/\bdecevoir\b|\bdeplaire\b|\bplaire a tout le monde\b/.test(folded))themes.push('peur de décevoir approbation');
 if(/\brepens(?:e|es|er|ais|ait|ons|ent)\b|\btourn\w* en boucle\b|\btrop (?:penser|reflechir|analyser)\b/.test(folded))themes.push('rumination overthinking');
 if(/\bremet\w* (?:toujours |souvent |tout )?a (?:plus tard|demain)\b/.test(folded))themes.push('procrastination');
 if(/\b(?:n|ne) (?:arrive|parviens) pas a (?:me )?decider\b/.test(folded))themes.push('indécision incertitude');
 if(saviourIntent(folded))themes.push('abnégation sacrifice de soi triangle de Karpman surresponsabilité');
 return query+(themes.length?'\nThèmes de recherche : '+themes.join(' ; '):'');
}

const LOOKUP_STOP=new Set('ambre bonjour salut merci peux peut pourriez pourrais pouvoir proposer propose proposes proposez recommander recommande recommandes recommandez suggerer suggere suggerez donner donne donnez montrer montre montrez trouver trouve trouvez article articles lecture lectures personnage personnages histoire histoires recit recits exemple exemples existant existante existants existantes existe existent catalogue bibliotheque lire lis lu connaissez connais connait connaitre parler parle parlent parlant raconte raconter concerne concernant traite traitent traiter autour propos sujet sujets theme themes syndrome voudrais voudrions souhaite souhaites souhaitez aimerais veux veut eviter exclure fuir conseille conseilles conseillez conseiller aurais auriez saurais savez sais cherche cherches cherchez recherche recherches comment pourquoi quelque chose est ce ca as avez vous nous une des les que qui quoi quel quelle quels quelles cela cette sur dans avec pour'.split(' '));
const LOOKUP_ALIASES=PSYCHOLOGY_CATALOG.patterns.flatMap(p=>{
 const labels=[p.label,...(p.keywords||[])].map(fold);
 return labels.filter(label=>label.split(' ').filter(t=>t.length>=4&&!LOOKUP_STOP.has(t)).length>=2)
  .map(label=>({label,covered:new Set(label.split(' ')),alternatives:labels.filter(other=>other!==label)}));
});

export function createReadingLookup(examples){
 // Build the index from public metadata once. Each lookup still evaluates the
 // full catalogue independently of the reader's profile or preferences.
 const shortTopics=new Set(),rows=examples.map(e=>{
  const labels=[e.sujet,e.canonical_concept,...(e.keywords||[])];
  for(const label of labels){const term=fold(label);if(term.length===3)shortTopics.add(term);}
  const topic=fold([e.title,e.sujet,e.canonical_concept,e.character,...(e.keywords||[])].join(' '));
  return {e,topic:' '+topic+' ',words:new Set(topic.split(' ')),
   editorial:new Set(fold(e.description).split(' ')),searchable:' '+fold(topic+' '+e.description)+' ',
   saviour:saviourIntent(topic),saviourRelated:/\b(?:abnegation|sacrifice de soi|karpman)\b/.test(topic)};
 });
 return (query,{limit=6}={})=>{
  const normalized=fold(query),saviour=saviourIntent(query);
  const terms=[...new Set(normalized.split(' ').filter(t=>(t.length>=4||shortTopics.has(t))&&!LOOKUP_STOP.has(t)))];
  const saviourRemainder=new Set(normalized.replace(saviourPattern,' ').split(' '));
  const extraTerms=terms.filter(t=>saviourRemainder.has(t));
  // A verbatim multiword editorial label can lead to that catalogue entry's
  // other search labels. Keep extra query topics mandatory, including unknown
  // ones, so an alias never turns a partly unmatched request into an exact hit.
  const bridges=LOOKUP_ALIASES.filter(({label})=>(' '+normalized+' ').includes(' '+label+' '));
  const matches=rows.flatMap(({e,topic,words,editorial,searchable,saviour:isSaviour,saviourRelated})=>{
   const hasTerm=t=>words.has(t)||editorial.has(t);
   const exact=saviour?isSaviour&&extraTerms.every(hasTerm):terms.length>0&&terms.every(t=>words.has(t));
   const validBridges=bridges.filter(bridge=>terms.every(t=>bridge.covered.has(t)||hasTerm(t)));
   const bridged=validBridges.some(bridge=>bridge.alternatives.some(label=>searchable.includes(' '+label+' ')));
   const primaryRelated=!saviour&&validBridges.some(bridge=>bridge.alternatives.some(label=>topic.includes(' '+label+' ')));
   const related=saviour?!exact&&saviourRelated&&extraTerms.every(hasTerm):!exact&&terms.length>0&&(terms.every(hasTerm)||bridged);
   return related||exact?[{id:e.id,title:e.title,url:e.url,character:e.character||'',match_kind:related?'related':'metadata_match',primary_related:primaryRelated,reason:saviour&&related?'Ce récit explore le sacrifice de soi ou le rôle de sauveur dans les relations ; ce rapprochement n’est pas un diagnostic.':related?'Le thème apparaît dans la description ou les repères éditoriaux de ce récit ; il s’agit d’un rapprochement, sans diagnostic.':'Le thème est présent dans les métadonnées vérifiées de cette lecture.'}]:[];
  }).sort((a,b)=>Number(b.match_kind==='metadata_match')-Number(a.match_kind==='metadata_match')||Number(b.primary_related)-Number(a.primary_related));
  return {status:matches.length?'matches':'no_match',scope:'complete_public_catalogue',searched_all:true,catalogue_size:rows.length,total_matches:matches.length,absence_proven:false,matches:matches.slice(0,Math.max(1,Math.min(6,Number(limit)||6))).map(({primary_related,...match})=>match)};
 };
}

export function lookupReadingCatalogue(examples,query,options){
 return createReadingLookup(examples)(query,options);
}
