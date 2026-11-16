import test from 'node:test';
import assert from 'node:assert/strict';
import {createChangeBackend} from '../src/change-backend.mjs';
import {readChangeState,emptyPlan} from '../src/change-plan.mjs';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
function setup(catalogue){const raw=new DatabaseSync(':memory:');for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));function prepare(sql){let args=[];return {bind(...a){args=a;return this;},async first(){return raw.prepare(sql).get(...args)||null;},async all(){return {results:raw.prepare(sql).all(...args)};},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}};}};}let queue=Promise.resolve();const DB={raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const result=[];for(const s of statements)result.push(await s.run());raw.exec('COMMIT');return result;}catch(e){raw.exec('ROLLBACK');throw e;}});queue=next.catch(()=>{});return next;}};let clock=Math.floor(Date.now()/1000),serial=0;const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code});},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null}};const plans=createChangeBackend({...utils,limit:async()=>{}}),personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue,engineEnabled:true}),env={DB,OPENROUTER_API_KEY:'mock-only'};for(const id of ['a','b']){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.com');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES(?,'active',?)").run(id,clock+86400);}async function call(path,method='GET',data,user='a',ctx){const req=new Request('https://example.com'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:user},url.pathname,url,ctx)||await plans.routes(req,env,{id:user},url.pathname,url,ctx)||await personal.routes(req,env,{id:user},url.pathname,url,ctx);return {status:res.status,...await res.json()};}catch(e){return {status:e.status||500,code:e.code,message:e.message};}}return {raw,env,chat,personal,plans,call,setClock:v=>{clock=v;},getClock:()=>clock};}

async function ready(s){for(const id of ['a','b'])s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,'initial','ready','Analyse initiale.',?,0,0)").run('seed-'+id,id,'seed-initial-'+id);assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);}
const message=(s,key='request_001',content='Je rumine quand je me sens critiqué.',ctx)=>s.call('/api/chat/messages','POST',{content,idempotency_key:key},'a',ctx);
const analysis=()=>({need:'understand',states:[],risk:'none',posture:'explore',sensitive_allowed:false,constraints:[],selected_ids:[],closed_topics:[]});
const draft=()=>({content:'Vous décrivez une rumination après les critiques. Nous pouvons prendre le temps de comprendre ce qui vous pèse.',questions:[],proposals:[],cartography_notes:[],people_observations:[]});
const review=()=>({criteria:Array(15).fill('yes'),reasons:[],change_criteria:Array.from({length:8},(_,i)=>({id:'C-R'+(i+1),verdict:'pass'}))});
const response=data=>Response.json({choices:[{message:{content:typeof data==='string'?data:JSON.stringify(data)},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:80,prompt_tokens_details:{cached_tokens:30},cost:0.001}});
function provider(t,handler){const calls=[],original=globalThis.fetch;globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;calls.push({purpose,body});const custom=await handler?.(purpose,body,calls);if(custom instanceof Response)return custom;return response(custom??({ambre_analysis:analysis(),ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:draft(),ambre_review:review(),ambre_memory:{items:[]}}[purpose]));};t.after(()=>{globalThis.fetch=original});return calls;}

// Independent review regression: a destructive owner mutation can win while a
// finished turn is finalizing its trace. No provider or production DB is used.
test('a late trace finalizer cannot restore plan inputs after owner deletion',async t=>{
 const s=setup();await ready(s);
 const document=emptyPlan();
 document.goals.push({id:'g_private',revision:1,title:'Ancien objectif confidentiel',status:'active',priority:1,accepted_at:1,created_at:1,updated_at:1});
 s.raw.prepare("INSERT INTO change_plans(user_id,revision,document_json,updated_at) VALUES('a',1,?,1)").run(JSON.stringify(document));
 provider(t);
 const original=s.env.DB.prepare;
 let deletion=null;
 s.env.DB.prepare=sql=>{
  const statement=original(sql),read=statement.first;
  statement.first=async()=>{
   const row=await read.call(statement);
   const finalized=s.raw.prepare("SELECT id FROM chat_turns WHERE user_id='a' AND idempotency_key='trace_purge_race' AND status='completed'").get();
   if(!deletion&&row?.trace&&sql==='SELECT trace FROM chat_turn_traces WHERE turn_id=? AND user_id=?'&&finalized){
    deletion={pending:true};
    const state=await readChangeState(s.env,'a');
    deletion=await s.call('/api/plan/goals/g_private','DELETE',{revision:state.revision,epoch:state.epoch});
    assert.equal(deletion.status,200,deletion.message);
    assert.equal(s.raw.prepare('SELECT 1 FROM chat_turn_traces WHERE turn_id=?').get(finalized.id),undefined,'the deletion initially removes the trace');
   }
   return row;
  };
  return statement;
 };
 await message(s,'trace_purge_race','Bonjour');
 assert.equal(deletion?.status,200,'the race ran after the turn publication');
 assert.equal((await readChangeState(s.env,'a')).document.goals.length,0);
 const trace=s.raw.prepare('SELECT trace FROM chat_turn_traces').get();
 assert.equal(Boolean(trace),false,'a deleted trace must not be restored by a stale finalizer');
});

test('owner deletion rechecks transmissions committed after its initial provenance read',async t=>{
 const s=setup();await ready(s);
 const document=emptyPlan();
 document.goals.push({id:'g_race',revision:1,title:'Repère confidentiel à effacer',status:'active',priority:1,accepted_at:1,created_at:1,updated_at:1});
 s.raw.prepare("INSERT INTO change_plans(user_id,revision,document_json,updated_at) VALUES('a',1,?,1)").run(JSON.stringify(document));
 const calls=provider(t),original=s.env.DB.prepare;
 let interleaved=false;
 s.env.DB.prepare=sql=>{
  const statement=original(sql),read=statement.first;
  statement.first=async()=>{
   const row=await read.call(statement);
   if(!interleaved&&sql.startsWith('SELECT 1 AS found FROM change_artifact_refs')){
    interleaved=true;
    assert.equal(row,null,'the goal has not been sent at the initial provenance read');
    await message(s,'transmission_during_delete','Repère confidentiel à effacer : je voudrais le comprendre.');
    assert.ok(s.raw.prepare("SELECT 1 FROM change_artifact_refs WHERE entity_id='g_race'").get(),'a transmission really precedes the deletion commit');
   }
   return row;
  };
  return statement;
 };
 const removed=await s.call('/api/plan/goals/g_race','DELETE',{revision:1,epoch:0});
 assert.equal(removed.status,200,removed.message);
 assert.equal(interleaved,true);
 await message(s,'after_transmission_delete','Bonjour');
 const input=JSON.parse(calls.filter(x=>x.purpose==='ambre_writing').at(-1).body.messages[1].content);
 assert.equal(input.history.some(turn=>turn.user.includes('Repère confidentiel à effacer')),false,'an already transmitted deleted source must not reenter through the prior generation');
});

test('an ordinary purge preserves a refusal committed after its memory read',async t=>{
 const s=setup();await ready(s);
 const document=emptyPlan();
 document.goals.push({id:'g_refusal',revision:1,title:'Objectif privé',status:'active',priority:1,accepted_at:1,created_at:1,updated_at:1});
 s.raw.prepare("INSERT INTO change_plans(user_id,revision,document_json,updated_at) VALUES('a',1,?,1)").run(JSON.stringify(document));
 provider(t);
 await message(s,'source_before_refusal','Bonjour');
 const original=s.env.DB.prepare;
 let interleaved=false;
 s.env.DB.prepare=sql=>{
  const statement=original(sql),read=statement.first;
  statement.first=async()=>{
   const row=await read.call(statement);
   if(!interleaved&&sql==='SELECT summary FROM chat_memory WHERE user_id=?'){
    interleaved=true;
    await message(s,'refusal_during_purge','Je ne veux plus parler de ma famille.');
    const memory=JSON.parse(s.raw.prepare("SELECT summary FROM chat_memory WHERE user_id='a'").get().summary);
    assert.ok(memory.closed.length,'the explicit refusal was committed before the purge');
   }
   return row;
  };
  return statement;
 };
 const removed=await s.call('/api/plan/goals/g_refusal','DELETE',{revision:1,epoch:0});
 assert.equal(removed.status,200,removed.message);
 assert.equal(interleaved,true);
 const state=await readChangeState(s.env,'a');
 assert.equal(state.exploration_reconsent_required,1,'ordinary deletion must not clear a concurrently committed refusal');
});
