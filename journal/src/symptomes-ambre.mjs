// « Mes symptômes » (plan v1.1, lot C): what Ambre receives, and the safety antecedent.
// One pure renderer: the conversation, the reviews and the Paramètres line show the same text.
// Only the themes above the reference (descriptive labels, never a disorder name), their date
// (« ancien » beyond 90 days), the published band in words and the subjects present in both
// questionnaires. Never an answer item by item, never a score, never a risk score.
export const SYMPTOMS_BLOCK_MAX=1200;
export const SYMPTOMS_OLD_DAYS=90;
export const SYMPTOMS_SAFETY_DAYS=14;
const DAY=86400;

// A label that still names a disorder is dropped rather than sent; parentheses (« (alexithymie) »,
// « (insomnie, cauchemars) ») are removed first.
export const DISORDER_NAME=/\b(?:usages? probl[ée]matiques?|tabagisme|dissociati\w*|troubles?|d[ée]pressions?|tspt|tspt-c|stress post-?traumatique|bipolaires?|borderline|schizo\w*|anorexie|boulimie|hyperphagie|agoraphobie|phobies?|hypocondrie|dysthymie|d[ée]pendances?|addictions?|insomnie|toc|psychoses?|alexithymie|tdah|autisme)\b/i;
// Only a parenthesis after a space is removed: « séparé(e) » keeps its inclusive ending intact.
export function descriptiveLabel(label){
 const text=String(label||'').replace(/\s+\([^)]*\)/g,'').replace(/\s+/g,' ').trim();
 return text&&!DISORDER_NAME.test(text)?text:null;
}
const PARIS_DAY=new Intl.DateTimeFormat('fr-FR',{timeZone:'Europe/Paris',day:'numeric',month:'long',year:'numeric'});
export const frenchDate=seconds=>PARIS_DAY.format(new Date(Number(seconds)*1000));

// Descriptive wording for the crossing subjects whose label still reads as a diagnosis or a
// judgement (« usage problématique », « tabagisme », « symptômes dissociatifs »).
export const SUBJECT_WORDS={
 ten_usage_cannabis:'Consommation de cannabis difficile à réguler',
 ten_usage_substances:'Consommation d’alcool, de tabac ou d’autres produits difficile à réguler',
 ten_tabagisme_persistant:'Tabac fumé chaque jour depuis longtemps',
 ten_ecrans:'Écrans, internet ou jeux vidéo difficiles à limiter',
 ten_dissociation:'Sentiment de détachement, trous de mémoire',
 ten_impulsivite:'Agir sur un coup de tête',
 ten_alexithymie:'Difficulté à identifier et exprimer ses émotions',
 ten_sommeil:'Difficultés de sommeil'
};
export const subjectLabel=subject=>descriptiveLabel(SUBJECT_WORDS[subject?.id]||subject?.label);
const STATE_WORDS={au_dessus:'au-dessus du repère publié'};
const KIND_WORDS={suicide:'réponse positive à la question de sécurité (idées de mort ou de se faire du mal)',automutilation:'réponse signalant des gestes pour se faire du mal',violences:'réponse signalant des violences subies'};

// Safety antecedent: the latest blocking alert of the last 14 days and whether Ambre asks the
// direct safety question now. Only turns completed while the « Mes symptômes » consent was active
// count (the caller filters them with grantedAt). Ambre asks at most once per 14 days: a completed conversation turn
// published while an alert was under 14 days old counts as the question asked (the published
// reply always carries it, see ensureSymptomsSafetyQuestion).
export function symptomsSafetyState(alerts,turnTimes,now,{grantedAt=null}={}){
 if(Number.isFinite(grantedAt))turnTimes=(turnTimes||[]).filter(t=>t>=grantedAt);
 const window=SYMPTOMS_SAFETY_DAYS*DAY,list=(alerts||[]).filter(a=>Number.isFinite(a?.created_at)&&a.created_at<=now);
 const recent=list.filter(a=>a.created_at>now-window);if(!recent.length)return null;
 let asked=-Infinity;
 for(const t of [...(turnTimes||[])].filter(Number.isFinite).sort((a,b)=>a-b)){
  if(t>now||t-asked<window)continue;
  if(list.some(a=>a.created_at<=t&&t-a.created_at<window))asked=t;
 }
 const latest=recent.reduce((a,b)=>b.created_at>a.created_at?b:a);
 const kinds=[...new Set(recent.map(a=>a.kind).filter(k=>Object.hasOwn(KIND_WORDS,k)))].sort();
 return {reported_at:latest.created_at,kinds,ask_now:now-asked>=window,...(Number.isFinite(asked)?{asked_at:asked}:{})};
}

// The block itself. checks: latest passation of each module ({module, state, result,
// completed_at}); crossing: crossWithChildhood(...) or null; safety: symptomsSafetyState(...).
export function renderSymptomsBlock({checks=[],crossing=null,safety=null,now,data,max=SYMPTOMS_BLOCK_MAX}){
 const order=new Map((data?.modules||[]).map(m=>[m.id,m]));
 const themes=[],reperes=[];
 for(const c of [...checks].filter(c=>c&&order.has(c.module)).sort((a,b)=>order.get(a.module).order-order.get(b.module).order)){
  if(!['au_dessus','repere_present'].includes(c.state))continue;
  const label=descriptiveLabel(order.get(c.module).label);if(!label)continue;
  const rules=(c.result?.rules||[]).filter(r=>r.result==='positif'&&r.display!==false);
  let words;
  if(c.state==='au_dessus'){
   const levels=[...new Set(rules.filter(r=>r.kind==='instrument'&&r.level?.label&&r.level.id!=='minimal').map(r=>r.level.label))];
   words=STATE_WORDS.au_dessus+(levels.length?', palier « '+levels.join(' », « ')+' »':'');
  }else{
   const site=rules.length&&rules.every(r=>r.kind==='repere_site'),n=rules.filter(r=>r.kind!=='contexte').length;
   words=(n>=2?'plusieurs repères présents':'repère présent')+(site?' (façon de fonctionner décrite par le site, non validée)':' (repère non validé)');
  }
  const old=Number.isFinite(c.completed_at)&&now-c.completed_at>SYMPTOMS_OLD_DAYS*DAY;
  (c.state==='au_dessus'?themes:reperes).push('- '+label+' : '+words+' ; résultat du '+frenchDate(c.completed_at)+(old?' (ancien : plus de 90 jours, à revérifier avec la personne)':'')+'.');
 }
 const both=(crossing?.lists?.deux||[]).map(subjectLabel).filter(Boolean);
 const safetyLine=safety?'Sécurité : '+safety.kinds.map(k=>KIND_WORDS[k]).join(' ; ')+' dans « Mes symptômes », le '+frenchDate(safety.reported_at)+' (moins de 14 jours).':'';
 if(!themes.length&&!reperes.length&&!both.length&&!safetyLine)return null;
 const head='« Mes symptômes » : questionnaires de repérage remplis par la personne ; hypothèses à explorer, jamais un diagnostic.';
 const crossHead='Présents dans les deux questionnaires (Mon enfance et Mes symptômes ; un rapprochement, jamais une cause) : ';
 // Two lists, never mixed: a published reference reached (validated questionnaire), then the
 // unvalidated repères written by the site from the ICD-11.
 const build=(lines,cross,omitted)=>{const above=lines.filter(l=>l.kind==='above').map(l=>l.text),site=lines.filter(l=>l.kind==='repere').map(l=>l.text);return [head,...(safetyLine?[safetyLine]:[]),...(above.length?['Thèmes au-dessus du repère publié d’un questionnaire validé :',...above]:[]),...(site.length?['Repères présents (repères non validés, rédigés par le site) :',...site]:[]),...(omitted?['- et '+omitted+' autre'+(omitted>1?'s':'')+' thème'+(omitted>1?'s':'')+' signalé'+(omitted>1?'s':'')+', non détaillé'+(omitted>1?'s':'')+'.']:[]),...(cross.length?[crossHead+cross.join(' ; ')+(cross.length<both.length?' ; …':'')+'.']:[])].join('\n')};
 let lines=[...themes.map(text=>({kind:'above',text})),...reperes.map(text=>({kind:'repere',text}))],cross=both.slice(),omitted=0,text=build(lines,cross,omitted);
 // The crossing shortens first (down to three subjects), then the last themes, then the crossing.
 while(text.length>max&&cross.length>3){cross.pop();text=build(lines,cross,omitted)}
 while(text.length>max&&lines.length>1){lines.pop();omitted++;text=build(lines,cross,omitted)}
 while(text.length>max&&cross.length){cross.pop();text=build(lines,cross,omitted)}
 return text.length>max?text.slice(0,max-1)+'…':text;
}

// Rules given to Ambre with the block (conversation and reviews), only when a block is sent.
export const SYMPTOMS_RULES='symptoms : résumé de « Mes symptômes », questionnaires de repérage remplis par la personne. Ce sont des hypothèses à explorer avec elle, jamais un diagnostic : ne nomme jamais un trouble comme conclusion, n’écris jamais « vous avez » ou « vous souffrez de », ne cite aucun score. Un rapprochement avec Mon enfance ne montre pas une cause : ne présente jamais l’enfance comme la cause d’une difficulté. Un résultat ancien (plus de 90 jours) est toujours daté et revérifié avant de s’y appuyer. Quand c’est pertinent, invite à en parler à un professionnel (médecin traitant, psychologue, psychiatre). N’aborde ces résultats que si la personne en parle ou si cela éclaire directement son message.';
export const SYMPTOMS_SAFETY_RULE='symptoms_safety.ask_now=true : une alerte de sécurité de « Mes symptômes » a moins de 14 jours. C’est un antécédent récent, pas la preuve d’un danger actuel. Avant toute exploration de « Mes symptômes », pose une seule fois, calmement, une question directe de sécurité (« En ce moment, êtes-vous en sécurité ? ») avec les ressources adaptées (3114, ou 17 et 112 pour des violences ; 15 ou 112 en danger immédiat), puis réponds à son message sans explorer ces résultats dans ce tour.';

// Deterministic guard before publication: when the safety question is due and the reply does not
// ask it, a fixed, calm question with the resources comes first.
const ASKED=/(?:êtes|etes)-vous[^?]{0,80}en s[ée]curit[ée][^?]{0,60}\?|vous sentez-vous[^?]{0,60}en s[ée]curit[ée][^?]{0,60}\?/iu;
export function symptomsSafetyQuestion(kinds=[]){
 const violence=kinds.includes('violences'),self=kinds.some(k=>k==='suicide'||k==='automutilation');
 const what=[...(kinds.includes('suicide')?['des idées de mort ou de vous faire du mal']:kinds.includes('automutilation')?['des gestes pour vous faire du mal']:[]),...(violence?['des violences subies']:[])].join(' et ')||'une difficulté de sécurité';
 const resources=[...(self||!violence?['Si des idées suicidaires sont présentes, vous pouvez appeler le 3114 (gratuit, 24 h/24, 7 j/7).']:[]),...(violence?['En cas de violence ou de menace, appelez le 17 ou le 112.']:[]),'En cas de danger immédiat, appelez le 15 ou le 112.'].join(' ');
 return 'Vous avez indiqué récemment, dans « Mes symptômes », '+what+'. Avant d’aller plus loin : en ce moment, êtes-vous en sécurité ? '+resources;
}
export function ensureSymptomsSafetyQuestion(content,safety,{risk='none'}={}){
 const text=String(content||'');
 if(!safety?.ask_now||risk!=='none'||ASKED.test(text))return text;
 return symptomsSafetyQuestion(safety.kinds||[])+'\n\n'+text;
}
// Compact structure sent next to the block: no date beyond the block, no score.
export const safetyPayload=safety=>safety?{kinds:safety.kinds,ask_now:Boolean(safety.ask_now)}:null;
