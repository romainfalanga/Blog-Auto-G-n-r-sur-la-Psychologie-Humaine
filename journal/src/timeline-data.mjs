// Shared read/validation helpers. These functions never call a model.
export const MEETING_FIELDS=['meeting_date_precision','meeting_date_start','meeting_date_end','meeting_age_text'];
export const NOTE_TEXT_MAX=2000;
const ready=new WeakSet();
export async function timelineReady(env){
 if(ready.has(env.DB))return true;
 const {results}=await env.DB.prepare('PRAGMA table_info(personal_people)').all();
 const available=MEETING_FIELDS.every(key=>results.some(column=>column.name===key));
 if(available)ready.add(env.DB);
 return available;
}
export function meetingData(data,fail){
 if(!MEETING_FIELDS.some(key=>Object.hasOwn(data,key)))return {};
 const out=Object.fromEntries(MEETING_FIELDS.map(key=>[key,data[key]??(key==='meeting_date_precision'?'unknown':'')]));
 for(const [key,value] of Object.entries(out))if(typeof value!=='string'||value.length>(key==='meeting_age_text'?80:20))fail(400,'Repère de rencontre invalide.','invalid_date');
 const precision=out.meeting_date_precision,start=out.meeting_date_start,end=out.meeting_date_end;
 if(!['unknown','year','month','day','range','age'].includes(precision))fail(400,'Précision de rencontre invalide.','invalid_date');
 const valid=value=>{if(!/^\d{4}(?:-\d{2}(?:-\d{2})?)?$/.test(value))return false;const [y,m,d]=value.split('-').map(Number),last=new Date(0);last.setUTCFullYear(y,m||1,0);return y>=1&&y<=9999&&(!m||m>=1&&m<=12)&&(!value.includes('-')||m>=1)&&(!d||d>=1&&d<=last.getUTCDate())&&(value.length<10||d>=1)};
 if(['unknown','age'].includes(precision)){out.meeting_date_start='';out.meeting_date_end='';}
 else if(!valid(start)||precision==='range'&&(!valid(end)||start>end)||({year:4,month:7,day:10}[precision]&&start.length!=={year:4,month:7,day:10}[precision]))fail(400,'La date de rencontre ne correspond pas à sa précision.','invalid_date');
 if(precision!=='range')out.meeting_date_end='';
 if(precision!=='age')out.meeting_age_text='';else if(!out.meeting_age_text.trim())fail(400,'Indiquez l’âge ou choisissez une date inconnue.','invalid_date');
 return out;
}
export const iso=seconds=>new Date(Number(seconds)*1000).toISOString();
export const noteRow=row=>({id:row.id,target_type:row.event_id?'event':'person',target_id:row.event_id||row.person_id,author:'user',text:row.text,revision:row.revision,created_at:iso(row.created_at),updated_at:iso(row.updated_at)});
export const stripPrivate=row=>{const {user_id,creation_id,payload_hash,fiction_use,...value}=row;return value};
const all=async(env,sql,...args)=>(await env.DB.prepare(sql).bind(...args).all()).results;
export async function eventRows(env,rows,userId){
 const ids=rows.map(row=>row.id),available=await timelineReady(env);
 const links=ids.length?await all(env,'SELECT event_id,person_id FROM life_event_people WHERE user_id=? AND event_id IN (SELECT value FROM json_each(?)) ORDER BY person_id',userId,JSON.stringify(ids)):[];
 const notes=available&&ids.length?await all(env,`SELECT * FROM (SELECT *,count(*) OVER(PARTITION BY event_id) AS total,row_number() OVER(PARTITION BY event_id ORDER BY created_at DESC,id) AS rn FROM personal_notes WHERE user_id=? AND event_id IN (SELECT value FROM json_each(?))) WHERE rn<=3 ORDER BY created_at DESC,id`,userId,JSON.stringify(ids)):[];
 return rows.map(row=>{const preview=notes.filter(note=>note.event_id===row.id);return {...stripPrivate(row),people_ids:links.filter(link=>link.event_id===row.id).map(link=>link.person_id),notes:preview.map(noteRow),notes_total:preview[0]?.total||0,notes_next_cursor:preview[0]?.total>3?preview.at(-1).id:null,notes_available:available}});
}
// Notes remain user-authored. A bounded, dated aggregate is part of its existing
// parent source, whose revision changes atomically through the migration triggers.
export async function attachContextNotes(env,userId,people,events){
 if(!await timelineReady(env))return;
 const rows=await all(env,`SELECT * FROM (SELECT *,count(*) OVER(PARTITION BY event_id,person_id) AS total,row_number() OVER(PARTITION BY event_id,person_id ORDER BY created_at DESC,id) AS rn FROM personal_notes WHERE user_id=?) WHERE rn<=20 ORDER BY created_at DESC,id`,userId);
 for(const [parents,column] of [[people,'person_id'],[events,'event_id']])for(const parent of parents){
  const notes=rows.filter(row=>row[column]===parent.id),kept=[];let text='';
  for(const row of notes){const line='Note de la personne, '+iso(row.created_at)+' ['+row.id+'] : '+row.text;if(text.length+line.length+(text?2:0)>6000)break;text+=(text?'\n\n':'')+line;kept.push(noteRow(row));}
  parent.user_notes=kept;parent.user_notes_total=notes[0]?.total||0;parent.user_notes_omitted=parent.user_notes_total-kept.length;parent.user_notes_text=text;
 }
}
