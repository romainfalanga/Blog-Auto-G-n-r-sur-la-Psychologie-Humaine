import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
// Shared harness for lot S tests: real Worker modules, in-memory SQLite with
// every migration and a simulated provider (no network).
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend,parisWeek} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
export {CARTOGRAPHIE,parisWeek};

export function setup({users=['a','b'],env:extra={}}={}){
 const raw=new DatabaseSync(':memory:');raw.exec('PRAGMA foreign_keys=ON');
 for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 function prepare(sql){let args=[];return {bind(...a){args=a;return this},async first(){return raw.prepare(sql).get(...args)||null},async all(){return {results:raw.prepare(sql).all(...args)}},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}}}}}
 let queue=Promise.resolve();
 const DB={raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());raw.exec('COMMIT');return r}catch(e){raw.exec('ROLLBACK');throw e}});queue=next.catch(()=>{});return next}};
 let clock=1794830400,serial=0;
 const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null}};
 const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue:new Map(),engineEnabled:true});
 const env={DB,OPENROUTER_API_KEY:'mock-only',...extra};
 for(const id of users){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.test');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES(?,'active',?)").run(id,clock+86400*30)}
 async function call(path,method='GET',data,user='a',ctx){const req=new Request('https://example.test'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:user},url.pathname,url,ctx)||await personal.routes(req,env,{id:user},url.pathname,url,ctx);return {status:res.status,...await res.json()}}catch(e){return {status:e.status||500,code:e.code,message:e.message}}}
 async function ready(id='a'){seedCompanionPrerequisites(raw,id);raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,'initial','ready','Analyse initiale.',?,0,0)").run('seed-'+id,id,'seed-initial-'+id);const r=await call('/api/chat/consent','PUT',{accepted:true},id);if(r.status!==200)throw Error('consent '+JSON.stringify(r))}
 const send=(content,key,user='a',ctx)=>call('/api/chat/messages','POST',{content,idempotency_key:key},user,ctx);
 const job=()=>raw.prepare('SELECT * FROM chat_engine_jobs ORDER BY rowid DESC LIMIT 1').get();
 const turn=()=>raw.prepare('SELECT * FROM chat_turns ORDER BY rowid DESC LIMIT 1').get();
 async function settle(n=8){for(let i=0;i<n;i++){clock+=61;await chat.resumeEngine(env)}}
 return {raw,env,chat,personal,call,ready,send,job,turn,settle,now:()=>clock,setClock:v=>{clock=v},tick:s=>{clock+=s}};
}
export const analysis=(extra={})=>({need:'understand',states:[],risk:'none',posture:'explore',sensitive_allowed:false,constraints:[],selected_ids:[],closed_topics:[],signals:[],strategy_ids:[],change_signal:{kind:'none',target_id:null,quote:''},...extra});
export const draft=(extra={})=>({content:'Vous décrivez une situation qui vous pèse. Nous pouvons prendre le temps de la comprendre ensemble.',questions:[],proposals:[],cartography_notes:[],people_observations:[],change_proposals:[],followup:null,...extra});
export const review=(extra={})=>({criteria:Array(15).fill('yes'),reasons:[],change_criteria:Array.from({length:8},(_,i)=>({id:'C-R'+(i+1),verdict:'not_applicable'})),...extra});
export const response=data=>Response.json({choices:[{message:{content:typeof data==='string'?data:JSON.stringify(data)},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:80,prompt_tokens_details:{cached_tokens:30},cost:0.001}});
// handler(purpose, body, calls) may return an object, a Response or undefined (defaults).
export function provider(t,handler){
 const calls=[],original=globalThis.fetch;
 globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;calls.push({purpose,body,system:body.messages[0].content[0].text,input:body.messages[1].content});const custom=await handler?.(purpose,body,calls);if(custom instanceof Response)return custom;return response(custom??({ambre_analysis:analysis(),ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:draft(),ambre_review:review(),ambre_memory:{items:[]}}[purpose]))};
 t?.after?.(()=>{globalThis.fetch=original});
 return calls;
}
export const lastReply=s=>JSON.parse(s.turn().response||'{}').content||'';
export function seedTurns(s,n,{user='a',start=1,content=i=>'Message '+i+'.',created=i=>'2026-01-'+String(1+(i%28)).padStart(2,'0')+' 10:00:00'}={}){
 const week=parisWeek(new Date(s.now()*1000)).week_start;
 const ins=s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,created_at) VALUES(?,?,?,'hash','completed','consumed','2025-01-06',1,1,?,?,?)");
 for(let i=start;i<start+n;i++){const id=user+'-t'+i;ins.run(id,user,id,content(i),JSON.stringify({companion_reply:1,content:'Réponse '+i+'.'}),created(i))}
}
