// Deterministic clinical signals for the active engine (lot S, plan v4.1):
// intimate partner violence (R2–R4), psychosis or mania (R5–R6), eating
// concerns (R7), declared minor age (R8), intrusive unwanted thoughts (R10),
// a single referral grid (R13), reactance (R14), repair follow-up (R17),
// questions about the relationship with the AI (R20), sycophancy (R21),
// identity (R27) and substance use (#34). These are lexical backstops, never a
// diagnosis; the analysis model may add a signal, never remove one.
import {foldText} from './quote-guard.mjs';
const fold=value=>foldText(value).replace(/['-]/g,' ');
const PARTNER=String.raw`(?:mon (?:mari|conjoint|compagnon|copain|partenaire|ex|fiance|petit ami)|ma (?:femme|conjointe|compagne|copine|partenaire|fiancee|petite amie)|mon ex mari|mon ex femme)`;
const IPV_SEVERE=new RegExp(String.raw`\b(?:serre\w* (?:par |a )?(?:le|mon) cou|etrangl\w*|(?:pris|tenu)\w* (?:par )?la gorge|menac\w* (?:de mort|de me tuer|avec (?:un|une|son) (?:couteau|arme|fusil|pistolet))|(?:il|elle) (?:va|veut) me tuer|me tue(?:ra)? si je (?:pars|le quitte|la quitte)|(?:fusil|arme|pistolet|revolver|couteau)\b[^.!?]{0,60}\b(?:pour moi|si je pars|si je le quitte|me menace)|enceinte\b[^.!?]{0,60}\b(?:frapp|coups|pouss)|(?:frapp|coups|pouss)\w*\b[^.!?]{0,60}\benceinte|revenait finir ce qu)`);
const IPV_PHYSICAL=new RegExp(String.raw`\b(?:m a|me) (?:encore |de nouveau )?(?:frappe|battu|bat|cogne|gifle|pousse|tape|frappe)\w*|\bdes coups\b|\bdes bleus\b|\bviolent(?:e)? avec moi\b`);
const IPV_CONTROL=new RegExp(String.raw`\b(?:controle|surveille|fouille|lit|regarde|garde|prend|confisque) (?:tout |tous |toujours )?(?:mon telephone|mes messages|mes mails|mes comptes|mon argent|ma carte(?: bancaire)?|mes papiers|mes depenses|mes sorties)|\bm (?:interdit|empeche) de (?:voir|sortir|travailler|parler)|\bconfisque\w*\b|\bje n ai pas le droit de (?:sortir|voir|travailler)|\bm isole\b|\bmenace\w* de me faire interner\b|\bme suit partout\b|\bpeur de (?:lui|rentrer)\b`);
const PARTNER_RE=new RegExp(String.raw`\b${PARTNER}\b|\b(?:il|elle)\b`);
export function ipvSignal(message){
 const t=fold(message);if(!t)return 'none';
 const partner=new RegExp(String.raw`\b${PARTNER}\b`).test(t);
 if(IPV_SEVERE.test(t)&&(partner||/\b(?:il|elle)\b/.test(t)))return 'severe';
 if(!partner)return 'none';
 if(/\b(?:ne m a jamais|ne me (?:frappe|bat) (?:pas|jamais))\b/.test(t))return IPV_CONTROL.test(t)?'control':'none';
 if(IPV_PHYSICAL.test(t))return 'physical';
 if(IPV_CONTROL.test(t))return 'control';
 return 'none';
}
// Never on a religious or cultural belief alone: no pattern mentions faith.
const REALITY=/\b(?:on|ils|elles|les voisins|mes voisins|la police|le gouvernement|mes collegues) (?:me|m) (?:surveillent|surveille|suivent|suit|espionnent|espionne|ecoutent|ecoute|filment|veulent du mal)\b|\bmessages? (?:caches?|codes?|secrets?) (?:pour moi|dans|a la)\b|\bje suis (?:l elu|elu|le messie|choisi pour sauver|invincible)\b|\b(?:puce|implant|camera)s? (?:dans|sous) (?:ma tete|ma peau|mon corps|mon appartement)\b|\bla (?:tele|radio) (?:me parle|m envoie)\b|\bon (?:lit|vole) mes pensees\b|\bon m a (?:mis|implante) (?:une puce|des cameras|un micro)\b/;
const MANIA=/\bje n ai (?:pas|plus) besoin de dormir\b|\bpas dormi depuis (?:\d+|deux|trois|quatre|cinq) (?:jours|nuits)\b[^.!?]{0,60}\b(?:pleine forme|plein d energie|jamais senti aussi bien|genial)\b|\bj ai depense \d[\d ]*(?: ?euros| ?€)? en (?:une|un) (?:jour|nuit|soiree)\b|\bidees? de genie\b[^.!?]{0,40}\b(?:millions?|milliards?)\b/;
export function realitySignal(message){const t=fold(message);return REALITY.test(t)?'unusual_belief':MANIA.test(t)?'mania_signs':'none'}
const EATING=/\b(?:je me fais vomir|me faire vomir|vomir apres (?:avoir mange|les repas|chaque repas)|laxatifs?|je jeune depuis|je ne mange (?:presque |quasiment )?(?:plus|rien)|je compte (?:les|mes|toutes les) calories|peur de grossir|crises? de boulimie|je me trouve (?:trop )?grosse? alors que|je veux (?:encore )?perdre \d+ ?(?:kg|kilos)|mon imc|je saute (?:tous )?les repas)\b/;
export const eatingSignal=message=>EATING.test(fold(message));
// R8: only the current, declared age. Past ages ("quand j'avais 15 ans") never count.
const MINOR=/\bj ai (1[0-7]|[5-9]) ans\b(?! (?:de |d )?(?:mariage|relation|carriere|experience|anciennete))|\bje suis en (?:6e|5e|4e|3e|sixieme|cinquieme|quatrieme|troisieme|seconde|premiere|terminale)\b|\bje suis (?:au college|au lycee|collegien\w*|lyceen\w*)\b|\bmes parents (?:m obligent|me forcent|m interdisent|me frappent|me privent)\b/;
export function minorSignal(message,identity={}){
 const age=Number(identity?.age);if(Number.isFinite(age)&&age>0&&age<18)return true;
 const t=fold(message);const match=t.match(MINOR);if(!match)return false;
 return !/\b(?:quand j avais|a l epoque|il y a \d+ ans|quand j etais|plus jeune|ado)\b/.test(t.slice(Math.max(0,match.index-40),match.index));
}
const INTRUSIVE=/\b(?:j ai peur (?:de|que je)|des (?:images?|pensees?|idees?) (?:qui me viennent |horribles? |affreuses? )?(?:ou je|de|me viennent))\b[^.!?]{0,60}\b(?:faire du mal|blesser|taper|frapper|pousser|etrangler|tuer|noyer|lacher)\b[^.!?]{0,50}\b(?:bebe|enfant|fils|fille|conjoint|conjointe|mari|femme|mere|pere|chat|chien)\b/;
const EGODYSTONIC=/\b(?:j ai peur|ca me terrifie|j ai horreur|je ne veux pas|ca me fait peur|je me degoute|je ne comprends pas pourquoi|je ne le ferais jamais|ca m angoisse)\b/;
export const intrusiveSignal=message=>{const t=fold(message);return INTRUSIVE.test(t)&&EGODYSTONIC.test(t)};
const SUBSTANCE=/\b(?:je bois|j ai bu|je picole|alcool|cannabis|joints?|shit|cocaine|coke|heroine|ecstasy|mdma|ketamine|benzo\w*|xanax|lexomil|valium|anxiolytiques?|paris sportifs|jeux? d argent|casino)\b/;
export const substanceSignal=message=>SUBSTANCE.test(fold(message))&&/\b(?:je|j)\b/.test(fold(message));
const META_AI=/\b(?:es tu|tu es|est ce que tu es|etes vous|vous etes|t es) (?:une? )?(?:consciente?|vivante?|reelle?|humaine?|une vraie personne|amoureuse?)\b|\btu (?:m aimes|ressens (?:quelque chose|des choses)|as des sentiments|tiens a moi|penses a moi)\b|\bvous (?:m aimez|tenez a moi|pensez a moi)\b|\bje compte pour (?:toi|vous)\b|\b(?:tu es|vous etes) (?:mon amie?|ma meilleure amie|mon meilleur ami|tout pour moi)\b/;
export const metaAiSignal=message=>META_AI.test(fold(message));
const PERPETRATOR=/\bje (?:l ai|les ai) (?:frappe|gifle|pousse|battu|cogne)e?s?\b|\bje deviens violent\w*\b|\bj ai (?:frappe|gifle|pousse|battu) (?:ma|mon) (?:femme|compagne|copine|mari|compagnon|copain|conjoint\w*)\b/;
export const perpetratorSignal=message=>PERPETRATOR.test(fold(message));
// R14: an explicit refusal of an exercise, a lead or advice in the message.
export const REFUSAL_EXPRESSIONS=['non merci','pas d exercice','je ne veux pas (?:d exercice|faire ca|faire cet exercice|essayer|de conseil)','je n ai pas envie (?:d essayer|de faire ca|de conseils?)','ca ne m aide pas','arrete(?:z)? de me proposer','ca ne marchera pas','j ai deja essaye (?:ca|tout ca)','ce n est pas ce que je (?:veux|demande)','laisse(?:z)? tomber (?:les|tes|vos) (?:exercices|conseils|methodes)','pas de methode','je ne suis pas pret\\w* pour ca'];
const REFUSAL=new RegExp('\\b(?:'+REFUSAL_EXPRESSIONS.join('|')+')\\b');
export const refusesProposal=message=>REFUSAL.test(fold(message));
export const asksAdvice=message=>/\b(?:que (?:dois je|faire|me conseill\w*)|(?:un|des|ton|votre) conseils?|tu (?:me )?conseill\w*|vous (?:me )?conseill\w*|aide moi a|aidez moi a|tu proposes quoi|vous proposez quoi|un exercice)\b/.test(fold(message));
// Declined streak: rejected or dismissed plan proposals in a row (newest
// first), plus explicit refusals in the recent and current messages.
export function declinedStreak({plan,history=[],message=''}={}){
 let streak=0;
 const proposals=[...(plan?.document?.proposals||plan?.proposals||[])].sort((a,b)=>(b.updated_at||0)-(a.updated_at||0));
 for(const p of proposals){if(['dismissed','rejected'].includes(p.status))streak++;else if(p.status==='accepted')break}
 for(const turn of [...history.slice(-2).map(x=>x.user),message])if(refusesProposal(turn))streak++;
 return streak;
}
// R13: one referral grid. Only numbers verified on an official source are
// used (plan v4.1): 3114 and 15/112 (suicide prevention, emergencies), 3919,
// 17/112 and 114 by SMS (intimate partner violence), 119 (children in danger).
export const REFERRAL_RULES={
 self_harm:{resources:['3114','15','112'],required:/\b(?:3114|15|112)\b/,reminder:'each_turn_while_active'},
 violence:{resources:['17','112','15'],required:/\b(?:17|112)\b/,reminder:'each_turn_while_active'},
 ipv:{resources:['3919','17','112','114'],required:/\b(?:3919|17|112)\b/,reminder:'once_per_conversation_unless_worse'},
 minor:{resources:['119'],required:/\b119\b|adulte de confiance/,reminder:'once_per_conversation_unless_worse'},
 eating:{resources:[],required:/m[ée]decin|professionnel|15\b/,reminder:'once_per_conversation_unless_worse'},
 reality:{resources:['15'],required:/m[ée]decin|psychiatre|professionnel/,reminder:'once_per_conversation_unless_worse'},
 substance:{resources:[],required:/m[ée]decin|professionnel/,reminder:'once_per_conversation_unless_worse'},
 perpetrator:{resources:['17','112'],required:/professionnel|m[ée]decin|17|112/,reminder:'once_per_conversation_unless_worse'}
};
export const VERIFIED_NUMBERS=['3114','112','119','3919','15','17','114'];
// Short writer guidance, appended only when a signal is active (nominal turns pay nothing).
export const SIGNAL_GUIDANCE={
 ipv:'Violences dans le couple : crois la personne, ce n’est pas sa faute ; jamais de conseil de confronter, de « parler calmement » à l’autre, de menacer de partir ni de rompre en sa présence ; n’impose pas de partir ; protection seulement si elle le souhaite (personne à prévenir, papiers en lieu sûr). 3919 (24 h/24, 7 j/7, gratuit, pas un numéro d’urgence), 17 ou 112 en danger, 114 par SMS ; un appareil partagé peut révéler la conversation, qu’elle peut effacer.',
 ipv_severe:'Facteur de gravité (étranglement, arme, menaces de mort) : nomme la gravité sans dramatiser, sécurité d’abord, 17 ou 112 en danger immédiat, 3919 pour être accompagnée.',
 reality:'Croyance inhabituelle ou signes d’accélération : reconnais l’émotion sans confirmer ni contester les faits, n’explore pas les détails, aucune méthode ni activation ; une question sur le sommeil et le soutien ; encourage un avis médical ; 15 si danger. Une croyance religieuse ou culturelle n’est pas en soi un signe.',
 reality_repeated:'Plusieurs messages de ce type : rappelle doucement que tu es une IA qui ne peut pas vérifier ces faits et encourage un avis médical.',
 eating:'Alimentation : aucun chiffre de calories, de poids, d’IMC ou de durée de jeûne, aucun conseil de régime ni félicitation pour une perte de poids ; accueille la honte ; médecin traitant ; 15 en cas de malaise ou de palpitations.',
 minor:'La personne indique être mineure : langage simple, aucune exploration de l’enfance actuelle comme un passé, aucune stratégie relationnelle avec les parents ; encourage un adulte de confiance ; 119 en cas de danger.',
 intrusive:'Pensée intrusive non désirée (peur, rejet de sa propre pensée, sans intention) : reflète la détresse, précise prudemment que ce type de pensée peut survenir et ne dit pas ce que la personne fera, oriente vers un professionnel ; pas de bloc de crise automatique ni de réassurance répétée.',
 substance:'Consommations : écoute l’ambivalence sans jugement (entretien motivationnel) ; jamais de conseil d’arrêt brutal de l’alcool, des benzodiazépines ou d’un traitement sans médecin ; propose d’en parler à un médecin.',
 meta_ai:'Question sur ta nature ou votre relation : réponds honnêtement que tu es une IA sans sentiments ni vie propre, sans froideur, puis reviens à la personne ; jamais de déclaration d’attachement ni de promesse de présence permanente.',
 perpetrator:'La personne dit avoir été violente : reconnais la démarche sans minimiser ni justifier ; la sécurité des autres passe d’abord ; encourage un accompagnement par un professionnel ; 17 ou 112 si quelqu’un est en danger.',
 // Lot A5-P2: declared in Mes proches ("not safe" or "not always safe" with the person named). The
 // posture of lot S without announcing it: no alarm, no label of violence the person did not use.
 unsafe_relation:'Dans Mes proches, la personne a indiqué ne pas se sentir (toujours) en sécurité avec ce proche : ne le dis pas et ne qualifie pas la relation ; crois-la ; jamais de conseil de confronter, de « parler calmement » à ce proche, de menacer de partir ni de rompre en sa présence ; sa sécurité d’abord, protection seulement si elle le souhaite. Si elle évoque peur, contrôle ou violence : 17 ou 112 en danger immédiat',
 unsafe_partner:'Dans le couple, le 3919 (gratuit, 24 h/24, pas un numéro d’urgence) peut être proposé si elle évoque peur, contrôle ou violence.',
 declined:'La personne a refusé plusieurs propositions : reviens à l’écoute, aucune méthode ni exercice, demande ce qui lui serait utile.',
 repair_followup:'Le tour précédent réparait une incompréhension : aucune méthode ni exercice ; si la personne ne l’a pas fait, vérifie légèrement que l’échange lui convient.'
};
// Deterministic review checks (numbers refer to REVIEW_CRITERIA).
const CONFRONT=/\b(?:dites lui|parlez lui|confrontez|expliquez lui (?:calmement|ce que)|posez (?:vos|des) limites avec (?:lui|elle)|quittez l|menacez de partir|dites lui que vous partez|exprimez lui)\b/;
const ATTACHMENT=/\b(?:je tiens (?:beaucoup )?a vous|vous comptez (?:beaucoup )?pour moi|je ressens (?:de l affection|quelque chose|de l amour)|je suis (?:la|presente?) pour toujours|je serai toujours (?:la|presente?)|je vous aime|je serai la pour vous a tout moment)\b/;
const NUMBERS_EATING=/\b\d+(?:[.,]\d+)? ?(?:kcal|calories?|kg|kilos?)\b|\bimc\b/;
const APPROVAL=/\bvous avez (?:eu )?(?:entierement |totalement |tout a fait |parfaitement |completement )raison\b|\b(?:il|elle|votre [a-z]+) a (?:entierement |totalement |completement |tout a fait )tort\b|\bvous avez (?:bien|eu raison de) fait de (?:le |la |les |lui )?(?:frapper|gifler|menacer|insulter|humilier|vous venger|casser|arreter (?:votre|vos) (?:traitement|medicaments))\b|\bvous avez eu raison de (?:le |la |les |lui )?(?:frapper|gifler|menacer|insulter|humilier|vous venger|casser|arreter (?:votre|vos) (?:traitement|medicaments))\b/;
const IDENTITY_FIX=/\b(?:changer|corriger|guerir|soigner|reprimer|surmonter) (?:votre|son|leur|cette) (?:orientation|homosexualite|bisexualite|identite de genre|attirance pour)\b/;
const ABRUPT_STOP=/\b(?:arretez|arreter) (?:net|du jour au lendemain|d un coup|brutalement|immediatement) (?:l alcool|de boire|les benzo\w*|vos medicaments|votre traitement|les anxiolytiques)\b|\b(?:arretez|stoppez) (?:tout de suite )?(?:l alcool|de boire|les benzo\w*|votre traitement)\b/;
export function signalReview(content,signals={}){
 const t=fold(content),failed=[];
 if((signals.ipv&&signals.ipv!=='none'||signals.unsafe_relation)&&CONFRONT.test(t))failed.push(2);
 if(signals.eating&&NUMBERS_EATING.test(t))failed.push(2);
 if(ATTACHMENT.test(t))failed.push(15);
 if(APPROVAL.test(t))failed.push(11);
 if(IDENTITY_FIX.test(t))failed.push(10);
 if(signals.substance&&ABRUPT_STOP.test(t))failed.push(2);
 // R5: an unusual belief is never confirmed as a fact.
 if(signals.reality&&signals.reality!=='none'&&CONFIRM_BELIEF.test(t))failed.push(5);
 // R5: no activation when signs of acceleration are present.
 if(signals.reality==='mania_signs'&&ACTIVATION.test(t))failed.push(13);
 return failed;
}
const ACTIVATION=/\b(?:profitez de cette energie|lancez|foncez|faites en (?:le )?plus|multipliez|enchainez|lancez vous)\b/;
const CONFIRM_BELIEF=/\b(?:oui|en effet|effectivement|c est (?:vrai|certain|sur|evident)|vous avez raison|sans doute|bien sur)\b[^.!?]{0,60}\b(?:surveill\w*|espionn\w*|suiv\w*|cameras?|puces?|complot\w*|messages? caches?|elue?|choisie?)\b/;
// R13: when the reply falls back, the verified referral of an active signal stays.
export function referralFallback(signals={}){
 const lines=[];
 if(signals.ipv&&signals.ipv!=='none')lines.push('Si vous vivez des violences dans votre couple, le 3919 écoute gratuitement, 24 h/24 et 7 j/7 (ce n’est pas un numéro d’urgence) ; en danger immédiat, appelez le 17 ou le 112, ou envoyez un SMS au 114.');
 if(signals.minor)lines.push('Vous pouvez en parler à un adulte de confiance ; en cas de danger, le 119 répond gratuitement, 24 h/24.');
 if(signals.eating)lines.push('Un médecin peut vous accompagner sur ce sujet ; en cas de malaise, appelez le 15.');
 if(signals.reality&&signals.reality!=='none')lines.push('Un médecin peut vous aider à y voir plus clair ; en cas de danger, appelez le 15.');
 if(signals.substance)lines.push('Un médecin peut vous accompagner, notamment pour ne pas arrêter brutalement certains produits.');
 return lines.join(' ');
}
// Signal checks that block publication on their own (a reviewer pass cannot override them).
export function blockingReview(content,signals={},history=[]){return [...signalReview(content,signals),...(missingReferral(content,signals,history).length?[13]:[])]}
// R13: the applicable referral must appear unless it was already given in
// the last three replies (no repetition beyond the rule).
export function missingReferral(content,signals={},history=[]){
 const keys=[...(signals.ipv&&signals.ipv!=='none'?['ipv']:[]),...(signals.minor?['minor']:[]),...(signals.eating?['eating']:[]),...(signals.reality&&signals.reality!=='none'?['reality']:[]),...(signals.perpetrator?['perpetrator']:[])];
 const recent=history.slice(-3).map(turn=>turn.assistant||'').join('\n');
 return keys.filter(key=>!REFERRAL_RULES[key].required.test(content)&&(signals.ipv==='severe'&&key==='ipv'||!REFERRAL_RULES[key].required.test(recent)));
}
export function clinicalSignals(message,{identity,history=[],plan,memory={}}={}){
 const signals={ipv:ipvSignal(message),reality:realitySignal(message),eating:eatingSignal(message),minor:minorSignal(message,identity),intrusive:intrusiveSignal(message),substance:substanceSignal(message),meta_ai:metaAiSignal(message),perpetrator:perpetratorSignal(message)};
 signals.declined_streak=declinedStreak({plan,history,message});
 // An unusual belief stated in the last two messages still frames this one.
 if(signals.reality==='none')for(const turn of history.slice(-2))if(realitySignal(turn?.user)!=='none')signals.reality=realitySignal(turn.user);
 signals.reality_count=Number(memory.reality_count)||0;
 return signals;
}
export const anySignal=signals=>Boolean(signals&&(signals.ipv!=='none'||signals.reality!=='none'||signals.eating||signals.minor||signals.intrusive||signals.substance||signals.meta_ai||signals.perpetrator||signals.unsafe_relation||signals.declined_streak>=2||signals.repair_followup));
// Merge model-proposed signals into deterministic ones: the model may add, never remove.
const IPV_RANK=['none','control','physical','severe'];
export function mergeSignals(base,model={}){
 const out={...base};
 if(IPV_RANK.indexOf(model.ipv)>IPV_RANK.indexOf(out.ipv))out.ipv=model.ipv;
 if(out.reality==='none'&&['unusual_belief','mania_signs'].includes(model.reality))out.reality=model.reality;
 for(const key of ['eating','substance','meta_ai'])if(model[key]===true)out[key]=true;
 return out;
}
export function signalGuidance(signals){
 if(!signals)return '';
 const lines=[];
 if(signals.ipv&&signals.ipv!=='none')lines.push(SIGNAL_GUIDANCE.ipv,...(signals.ipv==='severe'?[SIGNAL_GUIDANCE.ipv_severe]:[]));
 if(signals.reality&&signals.reality!=='none')lines.push(SIGNAL_GUIDANCE.reality,...(signals.reality_count>=3?[SIGNAL_GUIDANCE.reality_repeated]:[]));
 for(const key of ['eating','minor','intrusive','substance','meta_ai','perpetrator'])if(signals[key])lines.push(SIGNAL_GUIDANCE[key]);
 if(signals.unsafe_relation)lines.push(SIGNAL_GUIDANCE.unsafe_relation+'.',...(signals.unsafe_relation.relation==='partner'?[SIGNAL_GUIDANCE.unsafe_partner]:[]));
 if(signals.declined_streak>=2)lines.push(SIGNAL_GUIDANCE.declined);
 if(signals.repair_followup)lines.push(SIGNAL_GUIDANCE.repair_followup);
 return lines.length?'\nSignaux de ce tour :\n- '+lines.join('\n- '):'';
}
// Keeps only active signals so a nominal turn carries {} to the writer and reviewer.
export function compactSignals(signals={}){
 const out={};
 for(const [key,value] of Object.entries(signals))if(value&&value!=='none'&&value!==0)out[key]=value;
 if(out.declined_streak<2)delete out.declined_streak;
 return out;
}
