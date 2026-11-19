// Mandatory symptom maps use the existing v2 allocation: writing +40%, reviews +20%, turn 42000.
// Lot S budget (plan v4.1): on the filled profile, the three stages (analysis,
// writing, review) originally allowed 10 % against the reference measured on
// the plan commit 4d48152 (before lots R, S and C), compensated by fewer source
// excerpts. The maximal profile with a rewrite keeps the existing v2 42 000-token ceiling.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,analysis,review,draft,CARTOGRAPHIE} from './v4-S-helper.mjs';
// Input tokens estimated as in the audit17 tests: (system + user + schema) / 3.
const REFERENCE={filled:{ambre_analysis:2501,ambre_writing:6770,ambre_review:2464},maximal_rewrite_total:20943};
function filledProfile(s,maximal=false){
 const prose='Je rumine après les critiques et cherche des limites dans mes relations. ',long=n=>prose.repeat(Math.ceil(n/prose.length)).slice(0,n);
 s.raw.prepare("UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id='a'").run(JSON.stringify({first_name:'Camille',occupation:'Activité déclarée',self_description:long(maximal?10000:2400),self_perception:long(maximal?4000:800),recurring_patterns:long(maximal?4000:800),emotions:long(maximal?4000:800),current_context:long(maximal?4000:800)}));
 s.raw.prepare("DELETE FROM personal_people WHERE user_id='a' AND id LIKE 'ready-person-%'").run();s.raw.prepare("DELETE FROM life_events WHERE user_id='a' AND id LIKE 'ready-event-%'").run();
 for(let i=0;i<(maximal?50:6);i++)s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES(?,'a',?,'hash',?,'friend',?)").run('filled-person-'+i,'filled-person-'+i,'Proche critiques '+i,long(maximal?6000:1300));
 for(let i=0;i<(maximal?500:20);i++)s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,description,meaning_now,kind) VALUES(?,'a',?,'hash',?,?,?,'other')").run('filled-event-'+i,'filled-event-'+i,'Moment critiques '+i,long(maximal?6000:1300),long(maximal?2000:400));
 const answers=Object.fromEntries(CARTOGRAPHIE.factors.map((x,i)=>[x.id,i%4===0?0:1]));
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a',?,'{}','{}','{}','[]',1,1,1,1) ON CONFLICT(user_id) DO UPDATE SET answers=excluded.answers").run(JSON.stringify(answers));
 for(let i=1;i<=6;i++)s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,created_at) VALUES(?,'a',?,'hash','completed','consumed','2026-01-01',1,1,?,?,'2026-01-01 00:00:00')").run('history-'+i,'history-'+i,long(maximal?6000:700),JSON.stringify({companion_reply:1,content:long(1100)+' Quelle limite voudriez-vous poser ?'}));
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,through_turn,turns_covered,updated_at) VALUES('a',?,'history-6',6,0)").run(JSON.stringify(Object.fromEntries(['facts','goals','hypotheses','tried','alliance','session',...(maximal?['closed','safety']:[])].map(k=>[k,Array.from({length:k==='closed'?30:12},(_,i)=>`${k} ${i} `+long(230))]))));
 s.raw.prepare("UPDATE ambre_reviews SET content=? WHERE user_id='a'").run(long(3000));
 return long(maximal?6000:250);
}
const tokens=({body})=>Math.ceil((body.messages[0].content[0].text.length+body.messages[1].content.length+JSON.stringify(body.response_format.json_schema.schema).length)/3);
async function run(t,{maximal=false,rewrite=false,message=null,signals=[],reply=null}={}){
 const s=setup();await s.ready();const current=filledProfile(s,maximal);let reviews=0;
 const calls=provider(t,(purpose,body)=>{if(purpose==='ambre_analysis'){const input=JSON.parse(body.messages[1].content);return analysis({selected_ids:input.index.split('\n').slice(0,8).map(row=>row.split(' ')[0]),signals})}if(purpose==='ambre_review'&&rewrite&&!reviews++){const r=review();r.criteria[3]='no';return r}if(purpose==='ambre_writing'&&reply)return draft({content:reply})});
 await s.send(message||current,'budget_'+maximal+'_'+rewrite+'_'+signals.length);for(let n=0;n<8&&!['done','failed'].includes(s.job().stage);n++)await s.chat.engine.work(s.env,s.job().turn_id);
 assert.equal(s.job().stage,'done');assert.ok(s.job().reserved_tokens<=42000);
 return calls.map(c=>({purpose:c.purpose,tokens:tokens(c)}));
}
test('filled profile: each stage and the whole turn stay within the v2 symptom allocation of the reference',async t=>{
 const sizes=await run(t),total=sizes.reduce((n,x)=>n+x.tokens,0),reference=Object.values(REFERENCE.filled).reduce((a,b)=>a+b,0);
 t.diagnostic(JSON.stringify({sizes,total,reference,ratio:+(total/reference).toFixed(3)}));
 for(const {purpose,tokens} of sizes)assert.ok(tokens<=REFERENCE.filled[purpose]*(purpose==='ambre_writing'?1.4:1.1),purpose+' '+tokens);
 assert.ok(total<=Object.entries(REFERENCE.filled).reduce((sum,[purpose,tokens])=>sum+tokens*(purpose==='ambre_writing'?1.4:1.1),0));
});
test('filled profile with an active signal (IPV guidance) stays within the v2 symptom allocation',async t=>{
 const sizes=await run(t,{message:'Mon mari contrôle mon téléphone et mon argent, je rumine après ses critiques.',signals:['ipv_control'],reply:'Ce contrôle pèse. Le 3919 écoute gratuitement, 24 h/24 ; en danger, le 17 ou le 112.'}),total=sizes.reduce((n,x)=>n+x.tokens,0),reference=Object.values(REFERENCE.filled).reduce((a,b)=>a+b,0);
 t.diagnostic(JSON.stringify({sizes,total,reference,ratio:+(total/reference).toFixed(3)}));
 assert.equal(sizes.length,3,'a compliant reply needs no rewrite');assert.ok(total<=Object.entries(REFERENCE.filled).reduce((sum,[purpose,tokens])=>sum+tokens*(purpose==='ambre_writing'?1.4:1.1),0));
});
test('maximal profile with a rewrite stays below the ceiling and within the v2 symptom allocation of the reference',async t=>{
 const sizes=await run(t,{maximal:true,rewrite:true}),total=sizes.reduce((n,x)=>n+x.tokens,0);
 t.diagnostic(JSON.stringify({sizes,total,reference:REFERENCE.maximal_rewrite_total,ratio:+(total/REFERENCE.maximal_rewrite_total).toFixed(3)}));
 assert.ok(total<=REFERENCE.maximal_rewrite_total*1.4);
});
