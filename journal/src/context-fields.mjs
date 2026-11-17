// Lot A5-P2 (plan v5.1): optional form fields of Mes proches, Ma ligne de vie and Mon portrait.
// One place for their codes, limits and French words, shared by the Worker, Ambre and the pages.
// Sources: genogram emotional lines (McGoldrick, Gerson), convoy model (Kahn, Antonucci), NIMH
// life chart (Leverich, Post), life story interview (McAdams), LEC-5 exposure (no detail asked).
export const BOND_QUALITY=Object.freeze({close:'proche',distant:'distant',conflictual:'conflictuel',cut_off:'coupé',fusional:'fusionnel',supportive:'soutenant',ambivalent:'ambivalent',unsure:'je ne sais pas'});
export const PRESENCE_NOW=Object.freeze({daily:'au quotidien',regular:'régulièrement',rare:'rarement',no_contact:'plus de contact',deceased:'personne décédée'});
export const FEELS_SAFE=Object.freeze({yes:'oui',not_always:'pas toujours',no:'non'});
export const SUPPORT_LEVEL=Object.freeze({none:'aucun',some:'un peu',much:'beaucoup'});
export const EXPOSURE=Object.freeze({happened_to_me:'cela m’est arrivé',witnessed:'j’en ai été témoin',learned:'je l’ai appris'});
// Closed lists ('' = not given) and free texts with their limits.
export const PERSON_CHOICES=Object.freeze({bond_quality:BOND_QUALITY,presence_now:PRESENCE_NOW,feels_safe:FEELS_SAFE});
export const PERSON_TEXTS=Object.freeze({bond_note:300,helps:600,hurts:600,evolution:600});
export const EVENT_CHOICES=Object.freeze({support_level:SUPPORT_LEVEL,exposure:EXPOSURE});
export const EVENT_TEXTS=Object.freeze({felt_then:600,felt_now:600,impact:600,support_text:300});
export const PERSON_P2_FIELDS=Object.freeze([...Object.keys(PERSON_CHOICES),...Object.keys(PERSON_TEXTS)]);
export const EVENT_P2_FIELDS=Object.freeze([...Object.keys(EVENT_CHOICES),...Object.keys(EVENT_TEXTS)]);
// Portrait (self_data, no migration): reason for coming, current resources, rhythm of life.
export const SELF_P2_FIELDS=Object.freeze(['what_brings_me','what_helps_me','life_rhythm']);
export const SELF_P2_MAX=2000;
// Readable parts of a person or a moment, in the order a clinician would read them.
export const PERSON_PARTS=Object.freeze([['helps','Ce qui m’aide dans cette relation'],['hurts','Ce qui me blesse ou me pèse'],['evolution','Comment le lien a évolué']]);
export const EVENT_PARTS=Object.freeze([['felt_then','Ce que j’ai ressenti alors'],['felt_now','Ce que je ressens aujourd’hui en y repensant'],['impact','Ce que cela a changé pour moi']]);
const word=(list,value)=>Object.hasOwn(list,value)?list[value]:'';
// Short words for an index label or a header: only what was given.
export function personFacts(p){
 if(!p)return [];
 const out=[];
 const bond=word(BOND_QUALITY,p.bond_quality);if(bond||p.bond_note)out.push('Qualité du lien : '+[bond,String(p.bond_note||'').trim()].filter(Boolean).join(', '));
 const presence=word(PRESENCE_NOW,p.presence_now);if(presence)out.push('Présence aujourd’hui : '+presence);
 const safe=word(FEELS_SAFE,p.feels_safe);if(safe)out.push('En sécurité avec cette personne : '+safe);
 return out;
}
export function eventFacts(e){
 if(!e)return [];
 const out=[];
 const exposure=word(EXPOSURE,e.exposure);if(exposure)out.push('Vécu : '+exposure);
 const support=word(SUPPORT_LEVEL,e.support_level);if(support||e.support_text)out.push('Soutien reçu : '+[support,String(e.support_text||'').trim()].filter(Boolean).join(', '));
 return out;
}
// Index labels stay short (100 characters in the index line): the bond, then the presence.
export const personLabelSuffix=p=>[word(BOND_QUALITY,p?.bond_quality),p?.presence_now==='deceased'?'décès':''].filter(Boolean).join(', ');
export const eventLabelSuffix=e=>({witnessed:'témoin',learned:'appris'})[e?.exposure]||'';
// A person the member does not (always) feel safe with: Ambre takes the partner or family violence
// posture of lot S, without saying so in an alarming way.
export const unsafePerson=p=>p?.feels_safe==='no'||p?.feels_safe==='not_always';
// Column probe: before migration 0024 the forms and Ambre keep their previous fields.
const PROBED=new WeakMap();
export async function contextFieldsReady(env){
 const db=env?.DB;if(!db)return false;const cached=PROBED.get(db);if(cached===true)return true;if(typeof cached==='number'&&cached>Date.now())return false;
 try{
  await db.prepare(`SELECT ${PERSON_P2_FIELDS.join(',')} FROM personal_people LIMIT 0`).bind().all();
  await db.prepare(`SELECT ${EVENT_P2_FIELDS.join(',')} FROM life_events LIMIT 0`).bind().all();
  await db.prepare('SELECT origin FROM people_observations LIMIT 0').bind().all();
  PROBED.set(db,true);return true;
 }catch(error){if(/no such (?:column|table)/i.test(String(error?.message||''))){PROBED.set(db,Date.now()+60000);return false}throw error}
}
// The person named in the message (by first name, or by a role word when that role is unique)
// whom the member does not always feel safe with. Returns {relation, safe} or null; no name kept.
const foldWords=value=>' '+String(value||'').normalize('NFD').replace(/\p{M}/gu,'').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim()+' ';
const ROLE_WORDS={father:['pere','papa'],mother:['mere','maman'],parent:['parent','parents'],partner:['conjoint','conjointe','compagnon','compagne','mari','femme','partenaire','copain','copine'],child:['fils','fille'],sibling:['frere','soeur']};
export function unsafeRelation(people,message,previousMessage='',previousSignal=null){
 const words=foldWords(message),flagged=(people||[]).filter(unsafePerson);if(!flagged.length)return null;
 const named=flagged.filter(p=>{const name=foldWords(p.display_name).trim();return name.length>=2&&words.includes(' '+name+' ')||(ROLE_WORDS[p.role]||[]).some(w=>words.includes(' '+w+' '))&&(people||[]).filter(q=>q.role===p.role).length===1});
 if(!named.length){
  // Keep the protective posture on a pronoun-only follow-up, unless the current message
  // explicitly names another person. Do not infer a new unsafe relationship from a pronoun.
  const explicit=(people||[]).some(p=>{const name=foldWords(p.display_name).trim();return name.length>=2&&words.includes(' '+name+' ')||(ROLE_WORDS[p.role]||[]).some(w=>words.includes(' '+w+' '))});
  return !explicit&&/\b(il|elle|lui|cette personne|ce proche)\b/.test(words)?unsafeRelation(people,previousMessage)||previousSignal:null;
 }
 const partner=named.some(p=>p.role==='partner'),safe=named.some(p=>p.feels_safe==='no')?'no':'not_always';
 return {relation:partner?'partner':['father','mother','parent','child','sibling'].includes(named[0].role)?'family':'other',safe};
}
