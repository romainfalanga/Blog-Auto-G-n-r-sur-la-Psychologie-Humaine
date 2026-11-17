import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
// Explicit fixture data, never a production readiness bypass. Missing narratives
// stay empty: eligibility requires identity, counts and answers, not prose.
export function seedCompanionPrerequisites(raw,user='a',{answer=0}={}) {
 const current=JSON.parse(raw.prepare('SELECT self_data FROM profiles WHERE user_id=?').get(user)?.self_data||'{}');
 const identity={first_name:'Camille',last_name:'Test',birth_date:'1990-01-02',gender:'unspecified',...current};
 raw.prepare('UPDATE profiles SET self_data=? WHERE user_id=?').run(JSON.stringify(identity),user);
 for(let i=1;i<=3;i++){
  raw.prepare("INSERT OR IGNORE INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role) VALUES(?,?,?,'ready',?,'friend')").run('ready-person-'+user+'-'+i,user,'ready-person-'+i,'Proche '+i);
  raw.prepare("INSERT OR IGNORE INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES(?,?,?,'ready',?,'other','')").run('ready-event-'+user+'-'+i,user,'ready-event-'+i,'Moment '+i);
 }
 const map=raw.prepare('SELECT answers FROM cartographies WHERE user_id=?').get(user),answers={...Object.fromEntries(CARTOGRAPHIE.factors.filter(f=>!f.historical).map(f=>[f.id,answer])),...JSON.parse(map?.answers||'{}')};
 raw.prepare('INSERT INTO cartographies(user_id,answers,consent,locked_at,updated_at) VALUES(?,?,1,1,1) ON CONFLICT(user_id) DO UPDATE SET answers=excluded.answers,consent=1,locked_at=COALESCE(cartographies.locked_at,1)').run(user,JSON.stringify(answers));
 return identity;
}
