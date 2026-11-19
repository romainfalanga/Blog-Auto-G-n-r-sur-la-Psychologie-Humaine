import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
// Shared harness for lot R tests: real Worker modules, in-memory SQLite with every
// migration, simulated provider, statement counters and query plans.
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend,parisWeek} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';

export const migrationNames=()=>readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort();
export function database(until='9999'){
 const raw=new DatabaseSync(':memory:');raw.exec('PRAGMA foreign_keys=ON');
 for(const name of migrationNames().filter(name=>name<until))raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 return raw;
}
export function setup({users=['a','b'],env:extra={},until='9999'}={}){
 const raw=database(until),stats={statements:0,sql:new Map()};
 const note=sql=>{stats.statements++;stats.sql.set(sql,(stats.sql.get(sql)||0)+1)};
 function prepare(sql){let args=[];return {bind(...a){args=a;return this},async first(){note(sql);return raw.prepare(sql).get(...args)||null},async all(){note(sql);return {results:raw.prepare(sql).all(...args)}},async run(){note(sql);return {meta:{changes:raw.prepare(sql).run(...args).changes}}}}}
 let queue=Promise.resolve();
 const DB={raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());raw.exec('COMMIT');return r}catch(e){raw.exec('ROLLBACK');throw e}});queue=next.catch(()=>{});return next}};
 let clock=1794830400,serial=0;
 const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null}};
 const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue:new Map(),engineEnabled:true});
 const env={DB,OPENROUTER_API_KEY:'mock-only',...extra};
 function addUser(id){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.test');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES(?,'active',?)").run(id,clock+86400*400);}
 for(const id of users)addUser(id);
 async function call(path,method='GET',data,user='a',ctx){const req=new Request('https://example.test'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:user},url.pathname,url,ctx)||await personal.routes(req,env,{id:user},url.pathname,url,ctx);const text=await res.text();return {status:res.status,bytes:text.length,...JSON.parse(text)}}catch(e){return {status:e.status||500,code:e.code,message:e.message}}}
 async function ready(id='a'){seedCompanionPrerequisites(raw,id);raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,'initial','ready','Analyse initiale.',?,0,0)").run('seed-'+id,id,'seed-initial-'+id);const r=await call('/api/chat/consent','PUT',{accepted:true},id);if(r.status!==200)throw Error('consent '+JSON.stringify(r));}
 function measure(){const before=stats.statements,sqlBefore=new Map(stats.sql);return ()=>{const delta=new Map();for(const [k,v] of stats.sql){const d=v-(sqlBefore.get(k)||0);if(d)delta.set(k,d)}return {statements:stats.statements-before,sql:delta}}}
 const plan=sql=>raw.prepare('EXPLAIN QUERY PLAN '+sql).all(...Array((sql.match(/\?/g)||[]).length).fill(null)).map(r=>r.detail);
 return {raw,env,chat,personal,call,ready,addUser,stats,measure,plan,now:()=>clock,setClock:v=>{clock=v},tick:s=>{clock+=s}};
}
export const analysis=()=>({need:'understand',states:[],risk:'none',posture:'explore',sensitive_allowed:false,constraints:[],selected_ids:[],closed_topics:[],strategy_ids:[],change_signal:{kind:'none',target_id:null,quote:''}});
export const draft=()=>({content:'Vous décrivez une rumination après les critiques. Nous pouvons prendre le temps de comprendre ce qui vous pèse.',questions:[],proposals:[],cartography_notes:[],people_observations:[],change_proposals:[],followup:null});
export const review=()=>({criteria:Array(15).fill('yes'),reasons:[],change_criteria:Array.from({length:8},(_,i)=>({id:'C-R'+(i+1),verdict:'not_applicable'}))});
export const response=data=>Response.json({choices:[{message:{content:typeof data==='string'?data:JSON.stringify(data)},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:80,prompt_tokens_details:{cached_tokens:30},cost:0.001}});
// handler(purpose,body,calls) may return an object, a Response or undefined.
export function provider(t,handler){
 const calls=[],original=globalThis.fetch;
 globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;calls.push({purpose,model:body.model,body,chars:init.body.length});const custom=await handler?.(purpose,body,calls);if(custom instanceof Response)return custom;return response(custom??({ambre_snapshot_chunk:{summary:'Faits déclarés dans le bloc, à explorer avec la personne.',quotes:[]},ambre_analysis:analysis(),ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:draft(),ambre_review:review(),ambre_memory:{items:[]}}[purpose]))};
 t?.after(()=>{globalThis.fetch=original});
 return calls;
}
export const refused=(status=429,retryAfter='60',metadata)=>new Response(JSON.stringify({error:{message:'refused',...(metadata?{metadata}:{})}}),{status,headers:{'Content-Type':'application/json',...(retryAfter?{'Retry-After':retryAfter}:{})}});
export function filledProfile(s,user='a',maximal=false){
 const prose='Je rumine après les critiques et cherche des limites dans mes relations. ',long=n=>prose.repeat(Math.ceil(n/prose.length)).slice(0,n);
 s.raw.prepare('UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id=?').run(JSON.stringify({first_name:'Camille',occupation:'Activité déclarée',self_description:long(maximal?10000:2400),self_perception:long(maximal?4000:800),recurring_patterns:long(maximal?4000:800),emotions:long(maximal?4000:800),current_context:long(maximal?4000:800)}),user);
 s.raw.prepare("DELETE FROM personal_people WHERE user_id=? AND id LIKE 'ready-person-%'").run(user);s.raw.prepare("DELETE FROM life_events WHERE user_id=? AND id LIKE 'ready-event-%'").run(user);
 for(let i=0;i<(maximal?50:6);i++)s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES(?,?,?,'hash',?,'friend',?)").run(user+'-person-'+i,user,user+'-person-'+i,'Proche critiques '+i,long(maximal?6000:1300));
 for(let i=0;i<(maximal?500:20);i++)s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,description,meaning_now,kind) VALUES(?,?,?,'hash',?,?,?,'other')").run(user+'-event-'+i,user,user+'-event-'+i,'Moment critiques '+i,long(maximal?6000:1300),long(maximal?2000:400));
 const answers=Object.fromEntries(CARTOGRAPHIE.factors.map((x,i)=>[x.id,i%4===0?0:1]));
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES(?,?,'{}','{}','{}','[]',1,1,1,1) ON CONFLICT(user_id) DO UPDATE SET answers=json_patch(cartographies.answers,excluded.answers),consent=1,locked_at=1").run(user,JSON.stringify(answers));
}
// Completed history rows, optionally with a copied trace and a finished job each.
export function seedTurns(s,user,n,{contentChars=400,replyChars=900,trace=null,traceAt=0,start=0,jobAt=0}={}){
 const week=parisWeek(new Date(s.now()*1000)).week_start;
 const ins=s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,created_at) VALUES(?,?,?,'hash','completed','consumed',?,1,1,?,?,'2026-01-01 00:00:00')");
 const tr=s.raw.prepare('INSERT INTO chat_turn_traces(turn_id,user_id,trace,created_at) VALUES(?,?,?,?)');
 const jb=s.raw.prepare("INSERT INTO chat_engine_jobs(turn_id,user_id,stage,source_hash,created_at,updated_at) VALUES(?,?,'done','h',?,?)");
 s.raw.exec('BEGIN');
 for(let i=start;i<start+n;i++){const id=user+'-h'+i;ins.run(id,user,id,week,('Message '+i+' : je repense à la dispute avec ma sœur. ').repeat(20).slice(0,contentChars),JSON.stringify({companion_reply:1,content:('Réponse '+i+'. Vous dites que cela réveille quelque chose. ').repeat(40).slice(0,replyChars)}));if(trace){tr.run(id,user,typeof trace==='function'?trace(i):trace,typeof traceAt==='function'?traceAt(i):traceAt);jb.run(id,user,jobAt,jobAt)}}
 s.raw.exec('COMMIT');
}
