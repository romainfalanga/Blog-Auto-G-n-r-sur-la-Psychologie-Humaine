// Ordinary French wording -> editorial search vocabulary. These additions are
// transient retrieval terms, never declarations, portrait evidence or diagnoses.
export function expandReadingQuery(value){
 const query=String(value||'').trim().slice(0,6000),folded=query.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[’']/g,' '),themes=[];
 if(/\brefus(?:er|e|es|ent|ais|ait|ons)?\b|\bdire non\b|\bpos(?:er|e) (?:mes |ses |des )?limites\b/.test(folded))themes.push('limites affirmation de soi');
 if(/\bdecevoir\b|\bdeplaire\b|\bplaire a tout le monde\b/.test(folded))themes.push('peur de décevoir approbation');
 if(/\brepens(?:e|es|er|ais|ait|ons|ent)\b|\btourn\w* en boucle\b|\btrop (?:penser|reflechir|analyser)\b/.test(folded))themes.push('rumination overthinking');
 if(/\bremet\w* (?:toujours |souvent |tout )?a (?:plus tard|demain)\b/.test(folded))themes.push('procrastination');
 if(/\b(?:n|ne) (?:arrive|parviens) pas a (?:me )?decider\b/.test(folded))themes.push('indécision incertitude');
 return query+(themes.length?'\nThèmes de recherche : '+themes.join(' ; '):'');
}
