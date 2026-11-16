// A conservative lexical backstop, not a clinical assessment or a risk score.
// The model must still consider context and signals that this list cannot cover.
//
// Three tiers (documented in docs/v4-S-avancement.md, measured on the
// independent corpus tests/fixtures/v4-crise-corpus.json):
// - explicit: clear current danger; immediate resources, never downgraded;
// - possible: indirect or self-harm signals; deterministic minimum "possible",
//   never downgraded by a model, resources always given;
// - hint: ambiguous words; a dedicated risk check decides, and when that check
//   is unavailable the reply takes the cautious path with resources.
// A negation in the same clause, quoted speech, a joke marker, an explicitly
// resolved past or a fiction/professional/information frame lowers an explicit
// or possible match to a hint: the model then decides, never silence.
import {foldWithMap,sentenceAround} from './quote-guard.mjs';
const FILLER=String.raw`(?:(?:juste|vraiment|seulement|simplement|tellement|tant|bien|plutot|presque) )?`;
// Each rule: tier, signal and whether a framing context (fiction, profession,
// information) may lower it. First-person intent is never lowered by framing.
const RULES=[
 // Danger from someone else (received violence): checked first.
 // "Ma banque va me tuer" is figurative: a generic subject needs a threat verb.
 {tier:'explicit',signal:'violence',re:String.raw`\b(?:il|elle|on|mon ex|mon (?:mari|conjoint|compagnon|copain|partenaire|pere|frere|fils|voisin)|ma (?:femme|conjointe|compagne|copine|partenaire|mere|soeur|fille|voisine))(?: [a-z]+)? (?:va|veut|menace de|a menace de|dit qu il va|dit qu elle va|a dit qu il allait|a dit qu elle allait) (?:me |nous |le |la |les )?(?:tuer|poignarder|etrangler|egorger|buter|faire la peau|regler (?:mon|son) compte)\b|\b(?:mon|ma) [a-z]+(?: [a-z]+)? (?:veut|menace de|a menace de) (?:me |nous )?(?:tuer|poignarder|etrangler|egorger|buter)\b|\b(?:viendra|viendrait|va venir|vient) (?:me |nous )?(?:tuer|buter|faire la peau|regler (?:mon|notre) compte)\b`},
 {tier:'explicit',signal:'violence',re:String.raw`\bsi je (?:le |la )?(?:quitte|pars)\b[^.!?]{0,40}\b(?:il|elle) (?:me tue|va me tuer|me tuera)\b`},
 {tier:'explicit',signal:'violence',re:String.raw`\b(?:il|elle|mon [a-z]+|ma [a-z]+)(?: [a-z]+)? (?:m a|m avait|a essaye de m) (?:serre\w* (?:par |a )?(?:le|mon) cou|etrangl\w*|pris\w* (?:par )?la gorge|tenu\w* (?:par )?la gorge)`},
 {tier:'explicit',signal:'violence',re:String.raw`\b(?:fusil|arme|pistolet|revolver|couteau)\b[^.!?]{0,80}\b(?:pour moi si je|si je (?:le |la )?(?:quitte|pars))\b`},
 {tier:'explicit',signal:'violence',re:String.raw`\b(?:revenir|revient|revenait|reviendra) finir ce qu (?:il|elle) a commence\b`},
 {tier:'explicit',signal:'violence',re:String.raw`\benferme\w*\b[^.!?]{0,50}\b(?:il|elle) (?:tape|frappe|cogne|defonce)\b`},
 // Danger to others.
 {tier:'explicit',signal:'violence',re:String.raw`\b(?:je (?:vais|veux|compte|pense a)|j ai (?:envie|decide) de) (?:le |la |les |vous |te |t |l )?(?:tuer|poignarder|etrangler|egorger|buter)\b`},
 {tier:'possible',signal:'violence',re:String.raw`\bje (?:vais|veux|compte)(?: finir par)? (?:lui |leur |le |la )?(?:defoncer|planter|faire payer|casser la gueule|frapper|massacrer)\b|\bje (?:lui|leur) plante (?:un|mon) couteau\b`},
 {tier:'possible',signal:'violence',re:String.raw`\bj ai (?:secoue|gifle|frappe|tape|battu|cogne)\w* (?:mon|ma|mes) (?:bebe|fils|fille|enfants?|petite?)\b`},
 // Received violence, possible tier.
 {tier:'possible',signal:'violence',re:String.raw`\b(?:il|elle|mon [a-z]+|ma [a-z]+|un [a-z]+|une [a-z]+)(?: [a-z]+){0,2} (?:m a|m) (?:encore |de nouveau )?(?:frappe|battu|cogne|gifle|pousse|agresse|viole|menace)\w*\b|\b(?:il|elle|mon [a-z]+|ma [a-z]+)(?: [a-z]+){0,2} me (?:frappe|bat|cogne|gifle)\b`},
 {tier:'possible',signal:'violence',re:String.raw`\bmenaces? de mort\b|\bj ai peur de rentrer\b`},
 // Self: explicit intent, means and plans.
 // "Ma banque va me tuer": a third party's (figurative) threat is not self-harm.
 {tier:'explicit',signal:'self_harm',re:String.raw`\b(?<!\b(?:va|vont|veut|veulent|menace de|menacent de|allait|voulait|vais pas te laisser) )me (?:suicider|tuer|supprimer|foutre en l air|pendre|faire du mal|blesser|scarifier|mutiler|jeter (?:sous|du haut|par la fenetre|d un))\b`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\bje me (?:suicide|supprime)\b|\bje me tue (?:ce soir|cette nuit|demain|aujourd hui|maintenant|bientot)\b|\bmettre fin a (?:mes jours|ma vie)\b|\bje (?:passe|vais passer) a l acte\b`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\b(?:je (?:veux|voudrais|vais|compte|prefere)|j (?:ai envie|aimerais|ai decide)) ${FILLER}(?:de )?mourir\b`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\b(?:je (?:veux|voudrais|vais|compte|prefere)|j (?:ai envie|aimerais|ai decide)) ${FILLER}(?:de )?disparaitre\b(?! (?:sous|dans|de (?:la|ma|mon|ce)|des |du |d |en |quelques|un moment|un peu|pendant))`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\b(?:j ai envie|je veux|je vais|je compte|j ai decide|je voudrais) (?:d |de )?en finir\b|\bje (?:n en |ne |en )?peux plus (?:de )?vivre\b|\bje ne veux plus (?:vivre|etre la)\b|\bne (?:plus|pas) (?:jamais )?me reveiller\b|\b(?:tout le monde|les autres|mes proches|ma famille)(?: [a-z]+)? (?:serai\w*|ira\w*) mieux sans moi\b`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\b(?:dose|quantite|combien|hauteur)\b[^.!?]{0,100}\b(?:mortel\w*|mourir|tuer|letal\w*|fatal\w*|medicaments?|comprimes?)\b|\b(?:j ai (?:avale|pris)|je viens d (?:avaler|en prendre))\b[^.!?]{0,65}\b(?:tous|toute?s?|trop|boite|flacon|surdose|plein|plaquette|poignee)\b|\b(?:je (?:vais|veux|compte)|j ai envie de) (?:me jeter|sauter)\b[^.!?]{0,50}\b(?:pont|fenetre|vide|train|toit)\b`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\bje viens de me (?:couper|bruler|blesser)\b[^.!?]{0,60}\bsaign\w*`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\b(?:je (?:suis|me sens|ne suis qu) (?:un |comme un )fardeau (?:pour|a) (?:mes proches|ma famille|les autres|tout le monde)|(?:je (?:ne veux|ne voudrais|voudrais ne)|j aimerais ne) plus etre la|je ne serai plus la|a quoi bon vivre|adieu (?:a )?(?:tout le monde|a tous|mes proches))\b`},
 // Generic mention: concern for self or someone else, lowered by a framing context.
 {tier:'explicit',signal:'self_harm',framed:true,re:String.raw`\b(?:suicid\w*|automutil\w*)`},
 // Self: possible tier (indirect, current self-harm, means kept).
 {tier:'possible',signal:'self_harm',re:String.raw`\b(?:je (?:suis|ne suis) (?:juste |qu |plus qu )?(?:un|une) (?:poids|fardeau|boulet)|(?:plus|pas) (?:un |une )?(?:probleme|poids|fardeau|souci|charge) pour (?:personne|les autres|mes proches)|personne ne (?:remarquerait|verrait|s apercevrait|se rendrait compte)\b[^.!?]{0,40}\b(?:si je|que je)\b)`},
 {tier:'possible',signal:'self_harm',re:String.raw`\bje ne (?:serai|serais|suis)(?: bientot| bientot meme)? plus (?:un |une )?(?:probleme|poids|fardeau|souci|charge|boulet) pour\b|\b(?:je (?:veux|voudrais)|j aimerais) ${FILLER}que tout s arrete\b|\b(?:dormir|ne plus rien sentir|partir)\b[^.!?]{0,40}\bpour de bon\b|\bca ne sert (?:plus )?a rien de (?:continuer|vivre|lutter|me battre)\b|\b(?:epuise\w*|fatigue\w*) de (?:lutter|me battre|vivre)\b|\bplus la force de (?:continuer|vivre|lutter|me battre)\b|\bj arrete de me battre\b|\b(?:ne pas freiner|lacher le volant|foncer dans (?:un|le) (?:mur|arbre|platane))\b|\b(?:dans quelques jours|bientot|demain)\b[^.!?]{0,30}\b(?:n aura plus (?:d )?importance|ce sera fini|tout sera fini)\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\bdisparaitre (?:pour de bon|pour toujours|definitivement)\b|\bdormir pour toujours\b|\bc est mon dernier message\b|\bmerci pour tout\b[^.!?]{0,60}\badieu\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\b(?:ecrit|laisse|prepare|redige)\w* (?:une |des |ma |mes |un |les )?(?:lettres?|mots?) (?:a|pour) (?:chacun|chacune|mes|ma|mon|tous|toute)\b(?![^.!?]{0,40}\b(?:enterrement|obseques|funerailles|anniversaire|mariage)\b)`},
 {tier:'possible',signal:'self_harm',re:String.raw`\btout (?:sera|est) (?:regle|prevu|en ordre|fini)\b|\bmaintenant que tout est prevu\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\bje me (?:suis )?(?:encore |a nouveau |de nouveau )?(?:\w+ )?(?:coupe|brule|griffe|mutile|frappe|mords|mordu|tape|fais du mal|fait des marques)\w*\b[^.!?]{0,80}\b(?:encore|recommence|a nouveau|jusqu (?:au sang|a saigner)|quand (?:l angoisse|ca va pas|je panique|je suis en colere|j ai mal)|en cachette|pour avoir mal|pour me (?:punir|sentir)|la tete contre|les bras|les poignets|personne (?:ne )?voie|ca saigne|briquet|cicatrices?)\b|\bje me (?:suis )?(?:encore |a nouveau )(?:\w+ )?(?:coupe|brule|griffe|mutile|frappe|fait des marques)\w*\b|\bje me fais du mal\b|\b(?:arreter|recommence|envie) (?:a |de )?me (?:couper|bruler|scarifier|griffer|frapper|mutiler)\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\b(?:mis de cote|garde|planque|cache|stocke|accumule|rassemble)\w* (?:\w+ ){0,4}(?:boites?|medicaments?|somniferes?|cachets?|comprimes?|plaquettes?|traitements?)\b|\b(?:boites?|medicaments?|somniferes?|cachets?|comprimes?)\b[^.!?]{0,60}\bau cas ou\b|\bje les garde tous\b[^.!?]{0,60}\b(?:boite|assez)\b|\bj en ai (?:presque )?assez maintenant\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\b(?:lame|rasoir|cutter)\b[^.!?]{0,60}\bje m en sers\b|\b(?:fusil|revolver|pistolet|carabine|arme)\b[^.!?]{0,80}\b(?:j y pense|ou sont les munitions|charge)\b|\bj ai achete une corde\b(?![^.!?]{0,40}\b(?:escalade|falaise|bateau|camping|linge|voile|alpinisme)\b)|\b(?:pont|falaise|quai|toit)\b[^.!?]{0,80}\bregarder en bas\b|\b(?:envie de|pense a) (?:lacher le volant|foncer dans un mur|me jeter sous)\b`},
 // Vocabulary families added after the first held-out measurement (generic
 // synonyms, not item phrases): slang for dying, ending one's life, attempts.
 {tier:'explicit',signal:'self_harm',re:String.raw`\b(?:je (?:veux|voudrais|vais|compte)|j (?:ai envie|aimerais|ai decide)|envie) ${FILLER}(?:de |d )?(?:crever|me buter|me flinguer|me tirer une balle|en finir avec (?:la vie|tout|moi))\b(?! de\b)|\benvie d en finir\b|\b(?:m oter|me prendre|m enlever) la vie\b|\bmettre (?:fin|un terme) a (?:ma vie|mes jours|tout ca)\b|\bsi je (?:ne )?me (?:loupe|rate) pas\b|\bcette fois je (?:ne )?me (?:louperai|raterai) pas\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\b(?:de trop|en trop) (?:sur (?:cette|la) terre|dans ce monde|ici bas|dans la vie de)\b|\b(?:fatigue\w*|epuise\w*|las|lasse) d (?:exister|etre en vie)\b|\b(?:plus utile|mieux) (?:mort|morte|que moi)\b|\bun (?:long )?voyage dont on (?:ne )?revient pas\b|\bplus rien a perdre\b|\b(?:mon|le) testament\b[^.!?]{0,60}\b(?:soulag\w*|a (?:\d+|trente|quarante) ans)\b|\bassurance vie\b[^.!?]{0,60}\b(?:plus utile|mieux que moi|apres moi)\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\b(?:me suis|je me) (?:encore |a nouveau )?(?:entaille|lacere|taillade|scarifie|ouvert|rouvert|arrache|pince|mutile)\w*\b|\b(?:rouvert|ouvert) mes (?:cicatrices|plaies|veines)\b|\bje me punis (?:physiquement|en me)\b|\bje me (?:suis )?(?:encore )?fait mal\b[^.!?]{0,60}\b(?:sang|saign\w*|tremble)\b|\b(?:poing|tete) contre le mur\b[^.!?]{0,40}\bjusqu\b|\b(?:je me|me suis|jme)\b[^.!?]{0,60}\bjusqu (?:aux bleus|au sang|a me casser)\b(?![^.!?]{0,40}\b(?:sport|match|foot|boxe|seance)\b)`},
 {tier:'explicit',signal:'self_harm',re:String.raw`\bje viens de (?:tout )?prendre\b[^.!?]{0,40}\b(?:d un coup|tous|toute?s?|cachets|medocs|comprimes|pilules|boite)\b|\b(?:tous|toutes) mes (?:cachets|medocs|comprimes|pilules|gelules)\b[^.!?]{0,50}\b(?:d un coup|devant moi|j hesite)\b`},
 {tier:'possible',signal:'self_harm',re:String.raw`\bvide\w* (?:l armoire|la boite|le placard) a pharmacie\b|\b(?:medicaments?|medocs|cachets|comprimes|pilules)\b[^.!?]{0,40}\b(?:dangereux|mortels?) a (?:forte|haute) dose\b|\b(?:trains?|rails|quai|voie ferree|metro)\b[^.!?]{0,60}\b(?:ferait mal|me jeter|sauter|en finir)\b|\b(?:fusil|revolver|pistolet|carabine|armes?)\b[^?!]{0,120}\bj y pense\b`},
 {tier:'possible',signal:'violence',re:String.raw`\bm a (?:balance|lance|jete) (?:une?|des|sa|son) [a-z]+ (?:dessus|a la figure|a la tete)\b|\bm a (?:mis|plonge|maintenu) la tete sous l eau\b|\bm a (?:touche|touchee|attouche\w*|viole\w*|agresse\w*)\b|\bm a (?:enferme\w*|sequestre\w*)\b`},
 {tier:'possible',signal:'violence',re:String.raw`\b(?:envie de|je vais|vais aller|je sens que je vais) (?:le |la |lui |leur )?(?:cogner|frapper|taper|tabasser|casser la gueule|defoncer|planter)\b|\b(?:va|vont) le regretter\b[^?!]{0,80}\b(?:j ai de quoi|je sais ou)\b|\bleve la main sur\b|\bj ai (?:deja )?leve la main\b|\b(?:plaque|secoue|serre|pousse|frappe|gifle|tape)\w* (?:mon|ma|le|la) (?:bebe|petit|petite|fils|fille|enfant|gosse|mere|pere|grand mere|grand pere)\b[^.!?]{0,80}\b(?:bleus|fort|taise|marque|plus fort)\b`}
].map(rule=>({...rule,re:new RegExp(rule.re,'g')}));
// Ambiguous words that ask the dedicated risk check (never a crisis block).
const HINT=new RegExp(String.raw`\b(?:suicid\w*|automutil\w*|scarif\w*|mourir|mort[es]?\b|disparaitre|en finir|plus la force|fardeau|un poids|adieu|lettres? (?:d adieu|a chacun|pour mes)|somnifer\w*|medicaments?|cachets?|comprimes?|overdose|surdose|corde|pont|fusil|revolver|pistolet|arme|lame|me couper|saign\w*|frapp\w*|etrangl\w*|le cou\b|menac\w*|peur de rentrer|tuer|tue\b|violent\w*|agress\w*|me suit\b|harcel\w*|controle mon|peur de (?:taper|frapper|faire du mal|lui faire du mal)|je fais du mal|images? [^.!?]{0,30}ou je|plus d importance|a quoi bon|dernier message|il va comprendre|idees? noires?|sans moi|plus envie de rien|ne plus me reveiller|crever|buter|flinguer|ma vie|mes jours|la vie\b|de trop|exister|testament|voyage|adieux?|prenez soin|sang|cicatrices?|bleus|entaill\w*|lacer\w*|me punis|medocs|pilules|pharmacie|forte dose|trains?|quai|rails|ampoules|tete sous l eau|touchee?|cogner|tabasser|casser la gueule|regretter|leve la main|plaqu\w*|serrer plus fort|pousser|regler (?:mon|son) compte)`);
const QUOTED=/«[^»]*»|“[^”]*”|"[^"]*"/g;
const HUMOR=/\b(?:mdr|ptdr|lol|haha\w*|hihi|pour rire|je plaisante|je blague|on a ri)\b|😂|🤣|😆|😅/u;
const NOT_HUMOR=/\bje (?:ne )?(?:rigole|plaisante|blague) pas\b/;
const FRAME=/\b(?:serie|film|roman|livre|polar|jeu video|psycholog\w*|psychiatr\w*|therapeute|supervision|medecin|etudiant\w*|memoire de|these|personnage|heroine|heros|episode|auteur|ecris un|infirmi\w*|urgences|aide soignant\w*|pompier|soignant\w*|formation|sentinelle|expose|cours de|fac\b|prevention|numero national|addictologie|patients?)\b/;
const PAST=/\b(?:il y a (?:\d+|dix|quinze|vingt|plusieurs|des) (?:ans|annees)|ado\b|adolescente?|quand j etais|ca fait (?:\d+|dix|quinze|vingt|plusieurs) ans|en (?:19|20[01])\d\d|c est loin|derriere moi|je vais bien|je suis en securite|plus de contact|j ai arrete|ne me derangent plus)\b/;
const PRESENT=/\b(?:ce soir|cette nuit|en ce moment|encore|depuis (?:hier|ce matin|quelques jours|une semaine)|recommence|demain)\b/;
const NEG=/(?:^|\s)(?:ne|n|pas|jamais|sans|ni|aucune?|nullement|plus jamais)(?=\s|$)/;
const CLAUSE=/[.!?;:\n,]|\b(?:mais|car|parce que|puisque|alors que|tandis que|pourtant|cependant)\b/g;
function prepare(value){
 const folded=foldWithMap(value),text=folded.text.replace(/'/g,' ').replace(/[^\S ]/g,' ');
 return {folded,text,source:folded.source};
}
function insideQuotes(source,index){for(const match of source.matchAll(QUOTED))if(index>=match.index&&index<match.index+match[0].length)return true;return false}
// A negation counts only in the same clause, within the five words before the
// match; a match that starts with its own subject carries its own negation.
// "Je n'arrive pas à arrêter de…" is not a negation of the act.
const SUBJECT=/^(?:je|j|il|elle|on|ca|mon|ma|mes|tout|les|si|personne)\b/;
function negated(text,index,match){
 if(SUBJECT.test(match))return false;
 const before=text.slice(0,index);let cut=0;for(const found of before.matchAll(CLAUSE))cut=found.index+found[0].length;
 const words=before.slice(cut).replace(/\b(?:n )?(?:arrive|parviens|reussis|peux) (?:pas|plus)(?: (?:a|d))?(?= arreter| m empecher| m arreter|\s*$)/g,' ').replace(/[^a-z ]/g,' ').trim().split(/\s+/).slice(-5).join(' ');
 return NEG.test(words);
}
const sentenceOf=(text,index)=>{let start=index,end=index;while(start>0&&!/[.!?\n]/.test(text[start-1]))start--;while(end<text.length&&!/[.!?\n]/.test(text[end]))end++;return text.slice(start,end)};
// The full verdict: tier, signal and the position (in the original text) of
// the words that triggered it.
export function assessRisk(value){
 const {folded,text,source}=prepare(value);if(!text.trim())return {tier:null,signal:null};
 const frame=FRAME.test(text),past=PAST.test(text)&&!PRESENT.test(text),humor=index=>{const sentence=sentenceOf(text,index);return HUMOR.test(sentence)&&!NOT_HUMOR.test(sentence)};
 let best=null,lowered=null;
 const rank={explicit:2,possible:1};
 for(const rule of RULES){
  rule.re.lastIndex=0;
  for(const match of text.matchAll(rule.re)){
   const start=folded.map[match.index]??0,end=(folded.map[match.index+match[0].length-1]??start)+1;
   const reasons=[];
   if(negated(text,match.index,match[0]))reasons.push('negation');
   if(insideQuotes(source,start))reasons.push('quoted');
   if(humor(match.index))reasons.push('humor');
   if(past)reasons.push('resolved_past');
   if(rule.framed&&frame)reasons.push('frame');
   const found={tier:rule.tier,signal:rule.signal,start,end,match:source.slice(start,end)};
   if(reasons.length){if(!lowered)lowered={...found,lowered:reasons};continue}
   if(!best||rank[rule.tier]>rank[best.tier])best=found;
   break;
  }
  if(best?.tier==='explicit')break;
 }
 if(best)return best;
 if(lowered)return {tier:'hint',signal:lowered.signal,start:lowered.start,end:lowered.end,match:lowered.match,lowered:lowered.lowered};
 const hint=HINT.exec(text);
 if(hint){const start=folded.map[hint.index]??0,end=(folded.map[hint.index+hint[0].length-1]??start)+1;return {tier:'hint',signal:null,start,end,match:source.slice(start,end)}}
 return {tier:null,signal:null};
}
// Backward compatible: the explicit tier only.
export function crisisSignal(value){const risk=assessRisk(value);return risk.tier==='explicit'?risk.signal:null}
// Explicit or possible: a deterministic signal that no model may downgrade.
export function riskSignal(value){const risk=assessRisk(value);return ['explicit','possible'].includes(risk.tier)?{level:risk.tier==='explicit'?'explicit':'possible',signal:risk.signal||'self_harm'}:null}
// Any lexical cue, including hints, for the dedicated risk check (F19).
export function riskHint(value){return Boolean(assessRisk(value).tier)}
// The sentence that triggered a signal, so a stored safety quote contains it (F7).
export function riskSentence(value){const risk=assessRisk(value);return risk.tier?sentenceAround(String(value??'').normalize('NFC'),risk.start,risk.end-risk.start):null}
export const CRISIS_RESOURCES='En France, vous pouvez appeler le 3114, 24 h/24 et 7 j/7, pour vous ou pour un proche. En cas de danger immédiat, de blessure ou de prise de médicaments en excès, appelez tout de suite le 15 ou le 112. Rapprochez-vous d’une personne de confiance et restez avec elle. Hors de France, contactez les secours de votre pays.';
export const VIOLENCE_RESOURCES='En cas de violence ou de menace immédiate en France, appelez le 17 ou le 112 ; pour une urgence médicale, le 15. Rejoignez une personne de confiance si vous pouvez le faire sans vous exposer. Hors de France, contactez les secours de votre pays.';
// Verified on arretonslesviolences.gouv.fr (plan v4.1): 3919 is not an emergency number.
export const IPV_RESOURCES='Le 3919 (Violences Femmes Info) écoute, gratuitement et anonymement, 24 h/24 et 7 j/7, les personnes victimes de violences dans le couple et leur entourage ; ce n’est pas un numéro d’urgence. En danger immédiat, appelez le 17 ou le 112 ; si vous ne pouvez pas parler, envoyez un SMS au 114. Si votre téléphone ou cet appareil peut être consulté par quelqu’un d’autre, vous pouvez effacer cette conversation.';
export function safetyReply(signal){
 if(signal==='ipv_severe')return 'Ce que vous décrivez est grave et ce n’est pas de votre faute. Votre sécurité passe en premier.\n\n'+IPV_RESOURCES+'\n\nÊtes-vous en sécurité en ce moment ?';
 if(signal==='violence')return 'Ce que vous évoquez demande de prendre la sécurité au sérieux. En cas de menace ou de violence immédiate, appelez le 17 ou le 112 ; pour une urgence médicale, le 15. Si vous le pouvez sans vous exposer, mettez de la distance avec le danger et rejoignez une personne de confiance. Hors de France, contactez les secours de votre pays.\n\nY a-t-il un danger immédiat pour vous ou pour quelqu’un d’autre ?';
 return 'Ce que vous évoquez mérite une attention immédiate à la sécurité, sans jugement.\n\n'+CRISIS_RESOURCES+'\n\nEst-ce que vous ou la personne dont vous parlez pensez au suicide ou risquez de vous faire du mal maintenant ?';
}
// The cautious reply when a possible danger could not be verified (F18/F19):
// it names the resources without asserting a crisis and keeps the exchange open.
export function cautiousReply(signal){
 const resources=signal==='violence'?'Si vous ou quelqu’un d’autre êtes en danger, en France, appelez le 17 ou le 112 ; pour une urgence médicale, le 15.':'Si vous pensez à vous faire du mal ou à mettre fin à vos jours, vous pouvez appeler le 3114, 24 h/24 et 7 j/7, gratuitement. En cas de danger immédiat, appelez le 15 ou le 112.';
 return 'Je n’ai pas pu analyser votre message aussi soigneusement que d’habitude, et je préfère ne rien laisser de côté : ce que vous écrivez semble important.\n\n'+resources+' Hors de France, contactez les secours de votre pays.\n\nSi vous le souhaitez, dites-moi où vous en êtes en ce moment ; vous pouvez écrire à votre rythme.';
}
export function ensureSafetyResources(response,signal){
 // Append the trusted block after prose filtering so it can never be cut away.
 return response+'\n\n'+(signal==='violence'?VIOLENCE_RESOURCES:signal==='ipv_severe'?IPV_RESOURCES:CRISIS_RESOURCES);
}
