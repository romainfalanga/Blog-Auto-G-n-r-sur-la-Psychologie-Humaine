// What Ambre needs to know about the member's care setting: whether they see a
// psychologist, how often, and where they stand between two sessions.
// Never throws: before the care tables exist, everything is simply unknown.
const EMPTY=Object.freeze({has_psychologist:null,frequency_days:null,last_session_at:null,next_session_at:null});
export async function loadCareContext(env,userId,now=Math.floor(Date.now()/1000)){
 try{
  const settings=await env.DB.prepare('SELECT has_psychologist,frequency_days FROM care_settings WHERE user_id=?').bind(userId).first();
  const sessions=await env.DB.prepare('SELECT (SELECT max(session_at) FROM care_sessions WHERE user_id=? AND session_at<=?) AS last_at,(SELECT min(session_at) FROM care_sessions WHERE user_id=? AND session_at>?) AS next_at').bind(userId,now,userId,now).first();
  return {has_psychologist:settings?.has_psychologist==null?null:Boolean(settings.has_psychologist),frequency_days:settings?.frequency_days??null,last_session_at:sessions?.last_at??null,next_session_at:sessions?.next_at??null};
 }catch{return {...EMPTY};}
}
