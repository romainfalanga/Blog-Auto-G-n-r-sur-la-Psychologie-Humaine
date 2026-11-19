import {CARTOGRAPHIE} from './cartographie.generated.mjs';
import {SYMPTOMES_CARTE} from './symptomes-carte.generated.mjs';
import {platformAccess} from './entitlements.mjs';
const ACTIVE=Object.freeze(CARTOGRAPHIE.factors.filter(f=>!f.historical).map(f=>f.id).sort());
const SYMPTOMS=Object.freeze(SYMPTOMES_CARTE.factors.filter(f=>!f.historical).map(f=>f.id).sort());
export const REQUIRED_SYMPTOMS=SYMPTOMS.length;
const GENDERS=['female','male','non_binary','self_described','unspecified'];
const parse=value=>{try{return JSON.parse(value||'{}')}catch{return {}}};
const q=value=>"'"+String(value).replaceAll("'","''")+"'";
const owner=expression=>{if(!/^(?:\?|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)$/.test(expression))throw Error('Invalid readiness owner');return expression};
export function companionReadinessSql(userExpression='?') {
 const id=owner(userExpression),value=key=>`json_extract(rp.self_data,'$.${key}')`,birth=value('birth_date');
 return `EXISTS(SELECT 1 FROM profiles rp WHERE rp.user_id=${id}
 AND json_type(rp.self_data,'$.first_name')='text' AND length(trim(${value('first_name')}))>0
 AND json_type(rp.self_data,'$.last_name')='text' AND length(trim(${value('last_name')}))>0
 AND json_type(rp.self_data,'$.birth_date')='text' AND ${birth} GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]' AND ${birth}>='0001-01-01' AND date(${birth},'+0 days')=${birth} AND ${birth}<=date('now')
 AND ${value('gender')} IN (${GENDERS.map(q).join(',')})
 AND (SELECT count(*) FROM personal_people WHERE user_id=rp.user_id)>=3
 AND (SELECT count(*) FROM life_events WHERE user_id=rp.user_id)>=3
 AND EXISTS(SELECT 1 FROM cartographies rc WHERE rc.user_id=rp.user_id AND rc.consent=1 AND rc.locked_at IS NOT NULL
 AND (SELECT count(*) FROM json_each(rc.answers) a WHERE a.key IN (${ACTIVE.map(q).join(',')}) AND ((a.type='integer' AND a.value IN (0,1,2)) OR (a.type='text' AND a.value='unknown')))=${ACTIVE.length})
 AND EXISTS(SELECT 1 FROM symptom_maps sm WHERE sm.user_id=rp.user_id AND sm.consent=1 AND sm.locked_at IS NOT NULL
 AND (SELECT count(*) FROM json_each(sm.answers) a WHERE a.key IN (${SYMPTOMS.map(q).join(',')}) AND ((a.type='integer' AND a.value IN (0,1)) OR (a.type='text' AND a.value IN ('unknown','past'))))=${SYMPTOMS.length}))`;
}
export async function readCompanionReadiness(env,userId) {
 const row=await env.DB.prepare(`SELECT p.self_data,(SELECT count(*) FROM personal_people WHERE user_id=p.user_id) people_count,(SELECT count(*) FROM life_events WHERE user_id=p.user_id) events_count,c.answers,c.consent,c.locked_at,s.answers symptoms_answers,s.notes symptoms_notes,s.consent symptoms_consent,s.locked_at symptoms_locked_at FROM profiles p LEFT JOIN cartographies c ON c.user_id=p.user_id LEFT JOIN symptom_maps s ON s.user_id=p.user_id WHERE p.user_id=?`).bind(userId).first();
 const self=parse(row?.self_data),answers=parse(row?.answers),date=self.birth_date;
 const validDate=typeof date==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(date)&&date>='0001-01-01'&&Number.isFinite(Date.parse(date))&&new Date(date).toISOString().slice(0,10)===date&&date<=new Date().toISOString().slice(0,10);
 const missing_fields=['first_name','last_name'].filter(key=>typeof self[key]!=='string'||!self[key].trim());if(!validDate)missing_fields.push('birth_date');if(!GENDERS.includes(self.gender))missing_fields.push('gender');
 const missing_ids=ACTIVE.filter(id=>!Object.hasOwn(answers,id)||![0,1,2,'unknown'].includes(answers[id]));
 const identity={complete:!missing_fields.length,missing_fields},people={complete:Number(row?.people_count)>=3,count:Number(row?.people_count||0),required:3},events={complete:Number(row?.events_count)>=3,count:Number(row?.events_count||0),required:3};
 const childhood={complete:row?.consent===1&&row?.locked_at!=null&&!missing_ids.length,answered:ACTIVE.length-missing_ids.length,required:ACTIVE.length,missing_ids,persisted:row?.answers!=null,consent:row?.consent===1,question_set_version:ACTIVE.join('|')};
 const symptomAnswers=parse(row?.symptoms_answers),symptomNotes=parse(row?.symptoms_notes);
 const symptomMissing=SYMPTOMS.filter(id=>!Object.hasOwn(symptomAnswers,id)||![0,1,'unknown','past'].includes(symptomAnswers[id]));
 const symptoms={complete:row?.symptoms_consent===1&&row?.symptoms_locked_at!=null&&!symptomMissing.length,answered:SYMPTOMS.length-symptomMissing.length,required:SYMPTOMS.length,missing_ids:symptomMissing,persisted:row?.symptoms_answers!=null};
 childhood.saved=row?.consent===1&&row?.locked_at!=null;symptoms.saved=row?.symptoms_consent===1&&row?.symptoms_locked_at!=null;
 const missing=Object.entries({identity,people,events,childhood,symptoms}).filter(([,value])=>!value.complete).map(([key])=>key);
 // Plan v1.1 (lot 3 contract): one counter per page of the first message. complete requires the
 // answers AND the recording of the page (consent and locked_at for Mon enfance and Mes symptômes);
 // saved=false with done=total means « enregistrez ». answered/required repeat done/total. Same verdict as the SQL guard above.
 const pages=[
  {id:'account',label:'Mon compte',href:'/mon-compte/',done:4-missing_fields.length,total:4,saved:true,complete:identity.complete},
  {id:'people',label:'Mes proches',href:'/mes-proches/',done:Math.min(people.count,3),total:3,saved:true,complete:people.complete},
  {id:'timeline',label:'Ma ligne de vie',href:'/ma-ligne-de-vie/',done:Math.min(events.count,3),total:3,saved:true,complete:events.complete},
  {id:'childhood',label:'Mon enfance',href:'/cartographie/',done:childhood.answered,total:childhood.required,saved:childhood.saved,complete:childhood.complete},
  {id:'symptoms',label:'Mes symptômes',href:'/mes-symptomes/',done:symptoms.answered,total:symptoms.required,saved:symptoms.saved,complete:symptoms.complete}];
 for(const page of pages){page.answered=page.done;page.required=page.total;}
 return {version:2,complete:!missing.length,identity,people,events,childhood,symptoms,missing,pages};
}
export async function requireCompanionReadiness(env,userId) {
 const readiness=await readCompanionReadiness(env,userId);
 if(!readiness.complete)throw Object.assign(new Error(`Complétez votre identité, trois proches, trois événements et toutes les questions de Mon enfance et les ${REQUIRED_SYMPTOMS} questions de Mes symptômes avant de discuter avec Ambre.`),{status:403,code:'prerequisites_required',readiness});
 return readiness;
}
export async function requireCompanionGenerationAccess(env,userId,{at=Math.floor(Date.now()/1000),consent=true}={}) {
 await requireCompanionReadiness(env,userId);
 const subscription=await env.DB.prepare('SELECT * FROM subscriptions WHERE user_id=?').bind(userId).first();
 if(!platformAccess(env,subscription,at).allowed)throw Object.assign(new Error('Un accès actif est nécessaire.'),{status:403,code:'access_required'});
 if(consent){const settings=await env.DB.prepare('SELECT consent FROM chat_settings WHERE user_id=?').bind(userId).first();if(!settings?.consent)throw Object.assign(new Error('Votre accord est nécessaire.'),{status:403,code:'consent_withdrawn'})}
}
