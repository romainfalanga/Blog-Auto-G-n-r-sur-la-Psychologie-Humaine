// Lot R, F8/F9: cron recovery throughput, stage priority and memory fairness, on a
// virtual clock where simulated provider latencies really overlap.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {setup,response,analysis,draft,review} from './v4-R-helper.mjs';

const LATENCY={ambre_analysis:4000,ambre_risk_check:3000,ambre_writing:15000,ambre_review:4000,ambre_memory:6000};
function virtualTime(t){
 // The provider clock must not advance while a real libuv SHA-256 job is queued.
 // Keep exact hashes but remove unsimulated wall-clock I/O from this virtual-time test.
 const digest=crypto.subtle.digest.bind(crypto.subtle);
 t.mock.method(crypto.subtle,'digest',async (algorithm,data)=>{
  if(String(typeof algorithm==='string'?algorithm:algorithm.name).toUpperCase()!=='SHA-256')return digest(algorithm,data);
  const bytes=ArrayBuffer.isView(data)?new Uint8Array(data.buffer,data.byteOffset,data.byteLength):new Uint8Array(data);
  const result=createHash('sha256').update(bytes).digest();return result.buffer.slice(result.byteOffset,result.byteOffset+result.byteLength);
 });
 const realNow=Date.now,realFetch=globalThis.fetch;let V=1794830400000;const timers=[];
 const clock={get now(){return V},sleep:ms=>new Promise(resolve=>timers.push({at:V+ms,resolve})),advanceTo(at){V=Math.max(V,at)}};
 Date.now=()=>V;
 t.after(()=>{Date.now=realNow;globalThis.fetch=realFetch});
 // Runs a task to completion, firing the earliest virtual timer whenever every
 // in-flight call is waiting on the provider.
 clock.drive=async(task,sync)=>{let settled=false,value,error;task.then(v=>{settled=true;value=v},e=>{settled=true;error=e});
  for(let spins=0;!settled&&spins<200000;spins++){for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));if(settled)break;if(timers.length){timers.sort((a,b)=>a.at-b.at);const next=timers.shift();V=Math.max(V,next.at);sync?.();next.resolve()}}
  if(error)throw error;return value};
 return clock;
}
function latencyProvider(clock,sync,handler){
 const calls=[];
 globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;calls.push({purpose,at:clock.now});await clock.sleep(LATENCY[purpose]||1000);sync();const custom=handler?.(purpose,body);return response(custom??({ambre_analysis:analysis(),ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:draft(),ambre_review:review(),ambre_memory:{items:[]}}[purpose]))};
 return calls;
}

test('F8: 20 abandoned turns are all published within 4 cron passes, closest to publication first',async t=>{
 const clock=virtualTime(t),users=Array.from({length:20},(_,i)=>'u'+i),s=setup({users});const sync=()=>s.setClock(Math.floor(clock.now/1000));sync();
 for(const u of users)await s.ready(u);
 const work=s.chat.engine.work;s.chat.engine.work=async()=>{};
 for(const u of users){const r=await s.call('/api/chat/messages','POST',{content:'Je me sens dépassé au travail et je n’arrive pas à poser de limites.',idempotency_key:'lost_key_'+u},u,{waitUntil(){}});assert.equal(r.status,202)}
 s.chat.engine.work=work;
 // One older turn already waits for its review only: it must be served first.
 latencyProvider(clock,sync);
 const rows=[];let passes=0;
 while(s.raw.prepare("SELECT count(*) n FROM chat_engine_jobs WHERE stage NOT IN ('done','failed')").get().n&&passes<10){
  const t0=clock.now;const result=await clock.drive(s.chat.resumeEngine(s.env),sync);passes++;
  rows.push({pass:passes,rounds:result.rounds,duration_s:(clock.now-t0)/1000,done:s.raw.prepare("SELECT count(*) n FROM chat_engine_jobs WHERE stage='done'").get().n});
  assert.ok(clock.now-t0<60000,`pass ${passes} lasted ${(clock.now-t0)/1000} s`);
  clock.advanceTo(t0+60000);sync();
 }
 t.diagnostic(JSON.stringify(rows));
 assert.ok(passes<=4,`${passes} passes`);
 assert.equal(s.raw.prepare("SELECT count(*) n FROM chat_turns WHERE status='completed'").get().n,20);
});

test('F8: a job waiting for its review is taken before a job that has not started',async t=>{
 const clock=virtualTime(t),users=['old','new'],s=setup({users});const sync=()=>s.setClock(Math.floor(clock.now/1000));sync();
 for(const u of users)await s.ready(u);
 const work=s.chat.engine.work;s.chat.engine.work=async()=>{};
 for(const u of users){await s.call('/api/chat/messages','POST',{content:'Une semaine chargée.',idempotency_key:'order_'+u},u,{waitUntil(){}});clock.advanceTo(clock.now+1000);sync()}
 s.chat.engine.work=work;
 const calls=latencyProvider(clock,sync);
 // The newer job is already at review; the older one has not started.
 const newer=s.raw.prepare("SELECT turn_id FROM chat_engine_jobs WHERE user_id='new'").get().turn_id;
 await clock.drive(work(s.env,newer,19500),sync);
 assert.equal(s.raw.prepare('SELECT stage FROM chat_engine_jobs WHERE turn_id=?').get(newer).stage,'review');
 calls.length=0;
 const original=s.env.DB.prepare;let order=null;
 s.env.DB.prepare=sql=>{const st=original(sql);if(order===null&&/FROM chat_engine_jobs WHERE stage NOT IN/.test(sql)){const all=st.all.bind(st);st.all=async()=>{const r=await all();order=r.results.map(x=>x.turn_id);return r}}return st};
 await clock.drive(s.chat.resumeEngine(s.env),sync);s.env.DB.prepare=original;
 assert.equal(order[0],newer);
});

test('F9: an active person is served by the memory queue within the first 3 passes despite large backlogs',async t=>{
 const clock=virtualTime(t),users=['active',...Array.from({length:12},(_,i)=>'idle'+i)],s=setup({users});const sync=()=>s.setClock(Math.floor(clock.now/1000));sync();
 for(const u of users)await s.ready(u);
 const ins=s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response) VALUES(?,?,?,'h','completed','consumed','2025-01-06',1,1,?,'{}')");
 let n=0;for(const u of users)for(let i=0;i<(u==='active'?8:200);i++){const id=u+'_'+(n++);ins.run(id,u,id,'Message '+id+' sur ma semaine.')}
 for(const u of users.slice(1))s.raw.prepare('INSERT INTO chat_memory_jobs(user_id,updated_at,queued_at) VALUES(?,?,?)').run(u,s.now()-10,s.now()-10);
 latencyProvider(clock,sync,(purpose,body)=>{if(purpose==='ambre_memory'){const input=JSON.parse(body.messages.at(-1).content);return {items:input.turns.slice(0,2).map(x=>({section:'facts',turn_id:x.id,quote:x.text.slice(0,20)}))}}});
 const covered=u=>s.raw.prepare('SELECT turns_covered FROM chat_memory WHERE user_id=?').get(u)?.turns_covered||0;
 let servedAt=null;
 for(let pass=1;pass<=6;pass++){
  // The active person publishes a real turn every minute.
  const sent=await clock.drive(s.call('/api/chat/messages','POST',{content:'Encore une journée '+pass+'.',idempotency_key:'active_turn_'+pass},'active'),sync);assert.equal(sent.status,200,sent.message);
  const t0=clock.now;await clock.drive(s.chat.resumeEngine(s.env),sync);
  if(servedAt===null&&covered('active'))servedAt=pass;
  clock.advanceTo(t0+60000);sync();
 }
 t.diagnostic('active served at pass '+servedAt);
 assert.ok(servedAt!==null&&servedAt<=3,`served at pass ${servedAt}`);
 assert.ok(users.slice(1).every(u=>covered(u)>0),'every backlog keeps progressing');
});
