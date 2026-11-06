import {contextHash} from './private-context.mjs';

export const PEOPLE_OBSERVATION_AXES=Object.freeze(['communication','emotions','besoins','limites','appuis','tensions']);
const LIMIT_PER_PERSON=60;
const INITIAL_REVISION=1;
// The old database default (0, 1) never recorded an opt-out. Preserve an
// already disabled setting with a higher revision until a fresh agreement
// explicitly includes automatic notes.
export const peopleObservationModeSql=(alias='chat_settings')=>`(${alias}.people_observations_enabled=1 OR ${alias}.people_observations_revision=${INITIAL_REVISION})`;
export const peopleObservationRevision=settings=>settings?.consent===1&&(settings.people_observations_enabled===1||settings.people_observations_revision===INITIAL_REVISION)?settings.people_observations_revision:null;
const fold=value=>String(value||'').normalize('NFD').replace(/\p{M}/gu,'').toLowerCase().replace(/œ/g,'oe').replace(/[^\p{L}\p{N}]+/gu,' ').trim();
const mentions=(words,name)=>name.length>=2&&(' '+words+' ').includes(' '+name+' ');

// Reject the entire message for an opt-out, crisis, sensitive topic or an
// identifier. Do not save a harmless-looking fragment of a sensitive account.
const retention='(?:not\\w*|ret(?:en|ien)\\w*|gard\\w*|memoris\\w*|enregistr\\w*|ajout\\w*|sauvegard\\w*|conserv\\w*|utilis\\w*|consign\\w*|archiv\\w*|collect\\w*|stock\\w*)';
const optOut=new RegExp(`\\b(?:secret\\w*|confidentiel\\w*|entre nous|hors (?:du )?profil|(?:ca|cela) reste ici|(?:c est|ceci est|cela est|ca reste) (?:strictement )?prive\\w*|desactiv\\w* la memoire|pas (?:de trace|dans mon profil)|(?:sans|ne pas|pas(?: besoin de)?|aucune?) (?:(?:l|le|la|les) )?${retention}|(?:n|ne)\\b.{0,65}\\b(?:pas|jamais|plus|rien)\\b.{0,40}\\b${retention}|(?:n|ne)\\b.{0,45}\\b${retention}\\b.{0,20}\\b(?:pas|jamais|plus|rien)|(?:evit|refus)\\w*\\b.{0,35}\\b${retention}|oublie(?:z)? (?:ca|cela|ceci|tout|ce|cette)|(?:garde|gardez) (?:ca |cela |ceci )?pour (?:toi|vous))\\b`,'u');
const danger=/\b(?:suicid\w*|me tuer|se tuer|tuer quelqu\w*|(?:me|se) (?:faire du mal|blesser)|en finir|mourir|pas en securite|danger\w*|urgence\w*|crise\w*|menac\w*|arme[s]?|agress\w*|viol\w*|harcel\w*|abus\w*|emprise|frapp\w*|batt\w*|chantage|trauma\w*)\b/u;
const health=/\b(?:medic\w*|psychiatr\w*|hospital\w*|diagnost\w*|trouble\w*|malad\w*|traitement\w*|therapi\w*|psycho\w*|symptom\w*|prescri\w*|sante|depress\w*|bipol\w*|schizo\w*|autis\w*|tdah|tsp[tc]|toc|handicap\w*|cancer\w*|douleur\w*|hallucin\w*|entend des voix|enceinte|grossesse|fausse couche|avorte\w*|fertilit\w*|suivi par un|comprim\w*|milligramme\w*|antidepresseur\w*|lithium|endometriose|seroposit\w*|vih|fibromyal\w*|anorex\w*|boulim\w*|drogue\w*|addict\w*|overdose)\b/u;
const otherSensitive=/\b(?:sex\w*|intim\w*|orientat\w*|transgenre|lesbien\w*|gay|queer|homosex\w*|bisex\w*|heterosex\w*|asex\w*|religio\w*|croyan\w*|musulman\w*|catholi\w*|juif\w*|juiv\w*|protestan\w*|bouddh\w*|athe\w*|politique\w*|vote\w*|election\w*|syndic\w*|ethni\w*|origine raciale|roms?|genet\w*|biometr\w*|condamn\w*|casier|adresse|numero de telephone|mot de passe|carte bancaire|iban)\b/u;
const fictional=/\b(?:ficti\w*|jeu de role|histoire inventee|fais comme si|supposons que|imagine que)\b/u;

// Automatic notes use a closed vocabulary. An accepted verb alone tells us
// nothing about the privacy of its object or of an attached context clause.
const situation='(?:hier|aujourd hui|ce matin|ce soir|cette semaine|a table|(?:pendant|apres) nos (?:discussions|echanges|desaccords)|(?:quand|lorsque) (?:le ton monte|nous parlons (?:ensemble|de nos desaccords)|nous ne sommes pas d accord))';
const ordinarySituation=new RegExp(`^${situation}$`,'u');
const channel='(?:par (?:ecrit|message|telephone)|a l oral|en personne)';
const choice=`(?:recevoir (?:mes|les|nos) questions ${channel}|(?:parler|discuter|echanger) ${channel}|reprendre (?:la discussion|l echange) plus tard|(?:faire|prendre) une pause|parler (?:moins fort|plus lentement)|attendre avant de repondre)`;
const reportPatterns=[
 'me laisse terminer','ne me laisse pas terminer','m ecoute','me coupe la parole','interrompt la discussion','eleve la voix','quitte la piece',
 `(?:me repond|m a repondu|ne me repond pas) ${channel}`,
 `(?:me demande|m a demande) (?:de ${choice}|d attendre)`,
 `(?:me propose|m a propose) de ${choice}`,
 'm aide a preparer notre discussion','me rappelle notre rendez vous','me propose de m accompagner',
 `(?:me dit|m a dit) qu (?:elle|il|iel) (?:prefere ${choice}|a besoin (?:de temps|d une pause|d espace)|se sent (?:a l aise|mal a l aise|(?:ecoute|entendu|contrarie|rassure)e?))`
];
const ordinaryReport=new RegExp(`^(?:${reportPatterns.join('|')})(?: ${situation})?$`,'u');
const followUp=/\b(?:il|elle|lui|cette personne|ce proche|la meme personne)\b/u;
const namedPeople=(words,people)=>people.filter(person=>mentions(words,fold(person.display_name)));

function isRelationReport(sentence,name){
 if(sentence.length<20||sentence.length>280||!/^[\p{L}\p{M}][\p{L}\p{M}\p{Pd}'’ ,\t]*[.!]?$/u.test(sentence))return false;
 const [first,second,...rest]=sentence.split(',').map(fold);
 if(rest.length||second==='')return false;
 let clause=first;
 if(second&&ordinarySituation.test(first))clause=second;
 else if(second&&!ordinarySituation.test(second))return false;
 return clause.startsWith(name+' ')&&ordinaryReport.test(clause.slice(name.length+1));
}

function reportSentences(message,person){
 if(!person||/["«»“”`]/u.test(message))return [];
 const name=fold(person.display_name);
 // Preserve complete sentences, including negation. Pronouns and role labels
 // alone cannot identify a person, even if the model supplies a valid ID.
 const sentences=message.split(/(?<=[.!?…])\s+/u).map(sentence=>sentence.trim());
 return [...new Set(sentences.filter(sentence=>isRelationReport(sentence,name)))].slice(0,4);
}

export async function peopleObservationContext(env,userId,people,message,previousMessage,revision){
 if(!revision)return null;
 const words=fold(message);
 if([optOut,danger,health,otherSensitive,fictional].some(pattern=>pattern.test(words))||/[@<>]|\d{3,}|https?:/iu.test(message))return null;
 const matches=namedPeople(words,people),person=matches.length===1?matches[0]:null;
 // A previous explicit name can help recall prior notes, but cannot supply
 // the identity for a new note. Ambiguity on the current turn blocks recall.
 const previous=matches.length===0&&previousMessage&&followUp.test(words)?namedPeople(fold(previousMessage),people):[];
 const recall=person||(previous.length===1?previous[0]:null);
 if(!recall)return null;
 const settings=await env.DB.prepare(`SELECT 1 AS active FROM chat_settings WHERE user_id=? AND consent=1 AND ${peopleObservationModeSql()} AND people_observations_revision=?`).bind(userId,revision).first();
 if(!settings)return null;
 const quotes=reportSentences(message,person);
 const saved=(await env.DB.prepare('SELECT axis,quote,created_at FROM people_observations WHERE user_id=? AND person_id=? ORDER BY created_at DESC,id DESC LIMIT 6').bind(userId,recall.id).all()).results.filter(note=>isRelationReport(note.quote,fold(recall.display_name)));
 return {revision,person,quotes,remembered:saved.length?{person_id:recall.id,name:recall.display_name,observations:saved}:null};
}

export function choosePeopleObservations(context,value){
 if(!context?.quotes.length||!Array.isArray(value)||value.length>2)return [];
 const seen=new Set();
 return value.filter(item=>{
  if(item?.person_id!==context.person.id||!PEOPLE_OBSERVATION_AXES.includes(item.axis)||!context.quotes.includes(item.quote)||seen.has(item.quote))return false;
  seen.add(item.quote);
  return true;
 });
}

export function peopleObservationWrites(env,items,context,{userId,turnId,now,random,guard,guardArgs}){
 return Promise.all(items.map(async ({axis,quote})=>{
  const person=context.person;
  return env.DB.prepare(`INSERT INTO people_observations(id,user_id,person_id,turn_id,axis,quote,fingerprint,created_at)
    SELECT ?,?,?,?,?,?,?,? WHERE ${guard}
      AND EXISTS(SELECT 1 FROM chat_settings WHERE user_id=? AND consent=1 AND ${peopleObservationModeSql()} AND people_observations_revision=?)
      AND EXISTS(SELECT 1 FROM personal_people WHERE user_id=? AND id=? AND revision=?)
      AND (SELECT count(*) FROM people_observations WHERE user_id=? AND person_id=?)<?
    ON CONFLICT(user_id,person_id,fingerprint) DO NOTHING`).bind(
      random(),userId,person.id,turnId,axis,quote,await contextHash(fold(quote)),now(),
      ...guardArgs,userId,context.revision,userId,person.id,person.revision,userId,person.id,LIMIT_PER_PERSON
    );
 }));
}

export function createPeopleObservationApi({error,json}){
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 async function enabled(env,userId){
  const settings=await stmt(env,'SELECT consent,people_observations_enabled,people_observations_revision FROM chat_settings WHERE user_id=?',userId).first();
  return peopleObservationRevision(settings)!==null;
 }
 async function routes(request,env,user,path){
  const method=request.method,userId=user.id;
  if(path==='/api/people-observations/preferences')error(410,'Ce réglage séparé a été retiré.','feature_retired');
  const match=path.match(/^\/api\/people\/([\w-]+)\/observations(?:\/([\w-]+))?$/);
  if(!match)return null;
  const [,personId,observationId]=match;
  if(!await stmt(env,'SELECT id FROM personal_people WHERE user_id=? AND id=?',userId,personId).first())error(404,'Proche introuvable.','not_found');
  if(method==='GET'&&!observationId){
   const {results:observations}=await stmt(env,'SELECT id,axis,quote,created_at FROM people_observations WHERE user_id=? AND person_id=? ORDER BY created_at DESC,id DESC LIMIT ?',userId,personId,LIMIT_PER_PERSON).all();
   return json({enabled:await enabled(env,userId),observations});
  }
  if(method==='DELETE'&&observationId){
   // A turn prepared before the deletion cannot silently restore this note.
   // Materialize the automatic default before bumping its initial revision.
   const [removed]=await env.DB.batch([
    stmt(env,'DELETE FROM people_observations WHERE user_id=? AND person_id=? AND id=?',userId,personId,observationId),
    stmt(env,`UPDATE chat_settings SET people_observations_enabled=CASE WHEN people_observations_revision=${INITIAL_REVISION}
      THEN 1 ELSE people_observations_enabled END,people_observations_revision=people_observations_revision+1
      WHERE user_id=? AND changes()>0`,userId)
   ]);
   if(!removed.meta.changes)error(404,'Repère introuvable.','not_found');
   return json({deleted:true});
  }
  error(405,'Méthode non autorisée.','method_not_allowed');
 }
 async function exportData(env,userId){
  const {results}=await stmt(env,'SELECT id,person_id,turn_id,axis,quote,created_at FROM people_observations WHERE user_id=? ORDER BY created_at,id',userId).all();
  return {people_observations_enabled:await enabled(env,userId),people_observations:results};
 }
 return {routes,exportData};
}
