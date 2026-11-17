import {contextHash} from './private-context.mjs';
import {anchorQuote} from './quote-guard.mjs';

export const PEOPLE_OBSERVATION_AXES=Object.freeze(['communication','emotions','besoins','limites','appuis','tensions']);
const LIMIT_PER_PERSON=60;
const INITIAL_REVISION=1;
// The old database default (0, 1) never recorded an opt-out. Preserve an
// already disabled setting with a higher revision until a fresh agreement
// explicitly includes proposals.
export const peopleObservationModeSql=(alias='chat_settings')=>`(${alias}.people_observations_enabled=1 OR ${alias}.people_observations_revision=${INITIAL_REVISION})`;
export const peopleObservationRevision=settings=>settings?.consent===1&&(settings.people_observations_enabled===1||settings.people_observations_revision===INITIAL_REVISION)?settings.people_observations_revision:null;
const fold=value=>String(value||'').normalize('NFD').replace(/\p{M}/gu,'').toLowerCase().replace(/œ/g,'oe').replace(/[^\p{L}\p{N}]+/gu,' ').trim();
const mentions=(words,name)=>name.length>=2&&(' '+words+' ').includes(' '+name+' ');

// Reject the entire message for an opt-out, crisis, sensitive topic or an
// identifier. Do not save a harmless-looking fragment of a sensitive account.
const retention='(?:not\\w*|ret(?:en|ien)\\w*|gard\\w*|memoris\\w*|enregistr\\w*|ajout\\w*|sauvegard\\w*|conserv\\w*|utilis\\w*|consign\\w*|archiv\\w*|collect\\w*|stock\\w*)';
const optOut=new RegExp(`\\b(?:secret\\w*|confidentiel\\w*|entre nous|hors (?:du )?profil|(?:ca|cela) reste ici|(?:c est|ceci est|cela est|ca reste) (?:strictement )?prive\\w*|desactiv\\w* la memoire|pas (?:de trace|dans mon profil)|(?:sans|ne pas|pas(?: besoin de)?|aucune?) (?:(?:l|le|la|les) )?${retention}|(?:n|ne)\\b.{0,65}\\b(?:pas|jamais|plus|rien)\\b.{0,40}\\b${retention}|(?:n|ne)\\b.{0,45}\\b${retention}\\b.{0,20}\\b(?:pas|jamais|plus|rien)|(?:evit|refus)\\w*\\b.{0,35}\\b${retention}|oublie(?:z)? (?:ca|cela|ceci|tout|ce|cette)|(?:garde|gardez) (?:ca |cela |ceci )?pour (?:toi|vous))\\b`,'u');
const danger=/\b(?:suicid\w*|me tuer|se tuer|tuer quelqu\w*|(?:me|se) (?:faire du mal|blesser)|en finir|mourir|pas en securite|danger\w*|urgence\w*|crise\w*|menac\w*|arme[s]?|agress\w*|viol\w*|harcel\w*|abus\w*|emprise|frapp\w*|batt\w*|chantage|trauma\w*)\b/u;
const health=/\b(?:medic\w*|psychiatr\w*|hospital\w*|diagnost\w*|trouble\w*|malad\w*|traitement\w*|therapi\w*|psycho\w*|symptom\w*|prescri\w*|sante|depress\w*|bipol\w*|schizo\w*|autis\w*|tdah|tsp[tc]|toc|handicap\w*|cancer\w*|douleur\w*|hallucin\w*|entend des voix|enceinte|grossesse|fausse couche|avorte\w*|fertilit\w*|suivi par un|comprim\w*|milligramme\w*|antidepresseur\w*|lithium|lamotrigine|mastectomie|dialyse|endometriose|seroposit\w*|vih|fibromyal\w*|anorex\w*|boulim\w*|drogue\w*|addict\w*|overdose)\b/u;
const otherSensitive=/\b(?:sex\w*|intim\w*|orientat\w*|transgenre|lesbien\w*|gay|queer|homosex\w*|bisex\w*|heterosex\w*|asex\w*|religio\w*|croyan\w*|musulman\w*|catholi\w*|juif\w*|juiv\w*|protestan\w*|bouddh\w*|athe\w*|politique\w*|vote\w*|election\w*|syndic\w*|ethni\w*|origine raciale|roms?|genet\w*|biometr\w*|condamn\w*|casier|adresse|numero de telephone|mot de passe|carte bancaire|iban)\b/u;
const fictional=/\b(?:ficti\w*|jeu de role|histoire inventee|fais comme si|supposons que|imagine que)\b/u;

// Shared with other automatic notes: an opt-out or a fictional frame blocks any retention.
export const declinesRetention=message=>{const words=fold(message);return optOut.test(words)||fictional.test(words);};
const followUp=/\b(?:il|elle|lui|cette personne|ce proche|la meme personne)\b/u;
const namedPeople=(words,people)=>people.filter(person=>mentions(words,fold(person.display_name)));

// These are source candidates, never saved observations. The model selects a
// reported experience; the member must accept each resulting proposal.
function reportSentences(message,person){
 if(!person||/["«»“”`]/u.test(message))return [];
 const name=fold(person.display_name);
 return [...new Set(message.split(/(?<=[.!?…])\s+|\n+/u).map(s=>s.trim()).filter(sentence=>
  sentence.length>=20&&sentence.length<=280&&!sentence.includes('?')&&!/\b(?:pour que|sans doute|probablement|peut etre|je suppose|je crois)\b/u.test(fold(sentence))&&mentions(fold(sentence),name)&&anchorQuote(message,sentence).ok
 ))].slice(0,4);
}

// Cache only positive probes so a long-lived Worker sees a newly applied migration.
const readyDatabases=new WeakSet();
export async function peopleObservationReady(env){
 if(readyDatabases.has(env.DB))return true;
 const {results}=await env.DB.prepare('PRAGMA table_info(people_observations)').all();
 const ready=results.some(column=>column.name==='origin');
 if(ready)readyDatabases.add(env.DB);
 return ready;
}
const parse=value=>{try{return JSON.parse(value)}catch{return {}}};
export const isObservationProposal=row=>row.kind==='person_update'&&parse(row.binding).operation==='person_observation';
export const observationProposalRow=row=>({id:row.id,kind:'person_observation',target_id:row.target_id,payload:parse(row.payload),before:{},summary:row.summary,evidence_quote:parse(row.binding).evidence_quote||'',revision:row.revision,status:row.status,result:row.result_json?parse(row.result_json):null,created_at:row.created_at});

export async function peopleObservationContext(env,userId,people,message,previousMessage,revision){
 if(!revision||!await peopleObservationReady(env))return null;
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
 const saved=(await env.DB.prepare("SELECT axis,quote,created_at FROM people_observations WHERE user_id=? AND person_id=? AND origin='accepted' ORDER BY created_at DESC,id DESC LIMIT 6").bind(userId,recall.id).all()).results;
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

// Reuse the existing proposal lifecycle. Its SQL kind remains person_update
// for compatibility with the CHECK constraint; the binding marks the operation.
export async function peopleObservationWrites(env,items,context,{userId,turnId,random,guard,guardArgs}){
 if(!context||!await peopleObservationReady(env))return [];
 return Promise.all(items.map(async ({axis,quote})=>{
  const person=context.person,fingerprint=await contextHash(fold(quote));
  return env.DB.prepare(`INSERT INTO chat_proposals(id,user_id,turn_id,kind,target_id,expected_revision,binding,payload,before_data,summary)
    SELECT ?,?,?,'person_update',?,?,?,?,'{}',? WHERE ${guard}
      AND EXISTS(SELECT 1 FROM chat_settings WHERE user_id=? AND consent=1 AND ${peopleObservationModeSql()} AND people_observations_revision=?)
      AND EXISTS(SELECT 1 FROM personal_people WHERE user_id=? AND id=? AND revision=?)
      AND NOT EXISTS(SELECT 1 FROM people_observations WHERE user_id=? AND person_id=? AND fingerprint=?)
      AND NOT EXISTS(SELECT 1 FROM chat_proposals WHERE user_id=? AND target_id=? AND kind='person_update' AND json_extract(binding,'$.operation')='person_observation' AND json_extract(binding,'$.fingerprint')=? AND (status!='pending' OR (expected_revision=? AND json_extract(binding,'$.observation_revision')=?)))
      AND (SELECT count(*) FROM people_observations WHERE user_id=? AND person_id=?)<?`).bind(
      random(),userId,turnId,person.id,person.revision,JSON.stringify({operation:'person_observation',evidence_quote:quote,fingerprint,observation_revision:context.revision}),JSON.stringify({axis,quote}),'Repère proposé sur '+person.display_name,
      ...guardArgs,userId,context.revision,userId,person.id,person.revision,userId,person.id,fingerprint,userId,person.id,fingerprint,person.revision,context.revision,userId,person.id,LIMIT_PER_PERSON
    );
 }));
}

// Only an explicit accept request calls this writer. Edits are the member's own
// wording; the original exact source stays in binding.evidence_quote for export.
export async function acceptPeopleObservation(env,userId,p,data,{error,random,now}){
 if(!await peopleObservationReady(env))error(503,'Les propositions de repères ne sont pas encore disponibles.','observations_unavailable');
 const payload=data.payload??parse(p.payload);
 if(!payload||typeof payload!=='object'||Array.isArray(payload)||Object.keys(payload).some(key=>!['axis','quote'].includes(key))||!PEOPLE_OBSERVATION_AXES.includes(payload.axis)||typeof payload.quote!=='string'||payload.quote.trim().length<20||payload.quote.trim().length>280)error(400,'Choisissez un axe et un repère entre 20 et 280 caractères.','invalid_input');
 const quote=payload.quote.trim(),fingerprint=await contextHash(fold(quote)),token=random(),noteId=random(),binding=parse(p.binding);
 const target=`EXISTS(SELECT 1 FROM personal_people WHERE id=? AND user_id=? AND revision=?)
  AND EXISTS(SELECT 1 FROM chat_settings WHERE user_id=? AND consent=1 AND ${peopleObservationModeSql()} AND people_observations_revision=?)
  AND EXISTS(SELECT 1 FROM chat_turns WHERE id=? AND user_id=? AND status='completed' AND purged=0)
  AND ((SELECT count(*) FROM people_observations WHERE user_id=? AND person_id=?)<? OR EXISTS(SELECT 1 FROM people_observations WHERE user_id=? AND person_id=? AND fingerprint=?))`;
 const guard="EXISTS(SELECT 1 FROM chat_proposals WHERE id=? AND user_id=? AND status='applying' AND apply_token=?)",args=[p.id,userId,token];
 const statements=[
  env.DB.prepare(`UPDATE chat_proposals SET status='applying',apply_token=?,payload=? WHERE id=? AND user_id=? AND status='pending' AND revision=? AND ${target}`).bind(token,JSON.stringify({axis:payload.axis,quote}),p.id,userId,data.revision,p.target_id,userId,p.expected_revision,userId,binding.observation_revision,p.turn_id,userId,userId,p.target_id,LIMIT_PER_PERSON,userId,p.target_id,fingerprint),
  env.DB.prepare(`INSERT INTO people_observations(id,user_id,person_id,turn_id,axis,quote,fingerprint,created_at,origin) SELECT ?,?,?,?,?,?,?,?,'accepted' WHERE ${guard} ON CONFLICT(user_id,person_id,fingerprint) DO UPDATE SET origin='accepted',axis=excluded.axis,quote=excluded.quote,turn_id=excluded.turn_id`).bind(noteId,userId,p.target_id,p.turn_id,payload.axis,quote,fingerprint,now(),...args),
  env.DB.prepare(`UPDATE chat_proposals SET status='accepted',revision=revision+1,apply_token=NULL,result_json=json_object('kind','person_observation','target_id',target_id,'observation_id',(SELECT id FROM people_observations WHERE user_id=? AND person_id=? AND fingerprint=?)) WHERE id=? AND user_id=? AND ${guard}`).bind(userId,p.target_id,fingerprint,p.id,userId,...args)
 ];
 const results=await env.DB.batch(statements);
 if(!results[0].meta.changes)error(409,'Le proche, votre accord ou les repères ont changé. Demandez une nouvelle proposition à Ambre.','revision_conflict');
}

export function createPeopleObservationApi({error,json}){
 const stmt=(env,sql,...args)=>env.DB.prepare(sql).bind(...args);
 async function enabled(env,userId){
  const settings=await stmt(env,'SELECT consent,people_observations_enabled,people_observations_revision FROM chat_settings WHERE user_id=?',userId).first();
  return await peopleObservationReady(env)&&peopleObservationRevision(settings)!==null;
 }
 async function routes(request,env,user,path){
  const method=request.method,userId=user.id;
  if(path==='/api/people-observations/preferences')error(410,'Ce réglage séparé a été retiré.','feature_retired');
  const match=path.match(/^\/api\/people\/([\w-]+)\/observations(?:\/([\w-]+))?$/);
  if(!match)return null;
  const [,personId,observationId]=match;
  if(!await stmt(env,'SELECT id FROM personal_people WHERE user_id=? AND id=?',userId,personId).first())error(404,'Proche introuvable.','not_found');
  if(method==='GET'&&!observationId){
   const available=await peopleObservationReady(env);
   const {results:observations}=await stmt(env,`SELECT id,axis,quote,${available?"origin":"'auto' AS origin"},created_at FROM people_observations WHERE user_id=? AND person_id=? ORDER BY created_at DESC,id DESC LIMIT ?`,userId,personId,LIMIT_PER_PERSON).all();
   const {results:proposals}=await stmt(env,"SELECT * FROM chat_proposals WHERE user_id=? AND target_id=? AND kind='person_update' AND json_extract(binding,'$.operation')='person_observation' ORDER BY rowid DESC LIMIT 60",userId,personId).all();
   return json({enabled:await enabled(env,userId),available,observations,proposals:proposals.map(observationProposalRow)});
  }
  if(method==='DELETE'&&observationId){
   // A turn prepared before the deletion cannot silently restore this note.
   // Preserve the current proposal preference before bumping its revision.
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
  const available=await peopleObservationReady(env);
  const {results}=await stmt(env,`SELECT id,person_id,turn_id,axis,quote,${available?"origin":"'auto' AS origin"},created_at FROM people_observations WHERE user_id=? ORDER BY created_at,id`,userId).all();
  return {people_observations_enabled:await enabled(env,userId),people_observations:results};
 }
 return {routes,exportData};
}
