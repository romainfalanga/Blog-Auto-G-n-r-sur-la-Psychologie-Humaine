import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend,parisWeek} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {deterministicAnalysis,deterministicReview,knowledgeIndex,selectDetails,memoryPrompt,mergeClosed} from '../src/ambre-engine.mjs';
function setup(catalogue){const raw=new DatabaseSync(':memory:');for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));function prepare(sql){let args=[];return {bind(...a){args=a;return this;},async first(){return raw.prepare(sql).get(...args)||null;},async all(){return {results:raw.prepare(sql).all(...args)};},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}};}};}let queue=Promise.resolve();const DB={raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const result=[];for(const s of statements)result.push(await s.run());raw.exec('COMMIT');return result;}catch(e){raw.exec('ROLLBACK');throw e;}});queue=next.catch(()=>{});return next;}};let clock=Math.floor(Date.now()/1000),serial=0;const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code});},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null}};const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue,engineEnabled:true}),env={DB,OPENROUTER_API_KEY:'mock-only'};for(const id of ['a','b']){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.com');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES(?,'active',?)").run(id,clock+86400);}async function call(path,method='GET',data,user='a',ctx){const req=new Request('https://example.com'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:user},path,url,ctx)||await personal.routes(req,env,{id:user},path,url,ctx);return {status:res.status,...await res.json()};}catch(e){return {status:e.status||500,code:e.code,message:e.message};}}return {raw,env,chat,personal,call,setClock:v=>{clock=v;},getClock:()=>clock};}

async function ready(s){for(const id of ['a','b'])s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,'initial','ready','Analyse initiale.',?,0,0)").run('seed-'+id,id,'seed-initial-'+id);assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);}
const message=(s,key='request_001',content='Je rumine quand je me sens critiqué.',ctx)=>s.call('/api/chat/messages','POST',{content,idempotency_key:key},'a',ctx);
const analysis=()=>({need:'understand',states:[],risk:'none',posture:'explore',sensitive_allowed:false,constraints:[],selected_ids:[],closed_topics:[]});
const draft=()=>({content:'Vous décrivez une rumination après les critiques. Nous pouvons prendre le temps de comprendre ce qui vous pèse.',questions:[],proposals:[],cartography_notes:[],people_observations:[]});
const review=()=>({criteria:Array(15).fill('yes'),reasons:[]});
const response=data=>Response.json({choices:[{message:{content:typeof data==='string'?data:JSON.stringify(data)},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:80,prompt_tokens_details:{cached_tokens:30},cost:0.001}});
function provider(t,handler){const calls=[],original=globalThis.fetch;globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;calls.push({purpose,body});const custom=await handler?.(purpose,body,calls);if(custom instanceof Response)return custom;return response(custom??({ambre_analysis:analysis(),ambre_writing:draft(),ambre_review:review(),ambre_memory:{items:[]}}[purpose]));};t.after(()=>{globalThis.fetch=original});return calls;}
function seedTurns(s,n=4){for(let i=1;i<=n;i++)s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response) VALUES(?,'a',?,'hash','completed','consumed',?,1,1,?,?)").run('history-'+i,'history-'+i,parisWeek(new Date()).week_start,'Fait rapporté '+i,JSON.stringify({companion_reply:1,content:'Réponse '+i}));s.raw.prepare("INSERT INTO chat_memory_jobs(user_id,updated_at) VALUES('a',0)").run();}
const job=s=>s.raw.prepare('SELECT * FROM chat_engine_jobs ORDER BY created_at DESC LIMIT 1').get();

test('real enabled backend persists all stages, budget, private trace and no duplicate provider calls',async t=>{
 const s=setup();await ready(s);const calls=provider(t);const result=await message(s);
 assert.equal(result.status,200,result.message);assert.equal(job(s).stage,'done');assert.equal(result.messages.at(-1).content,draft().content);assert.equal(result.quota.used,1);
 assert.deepEqual(calls.map(c=>c.purpose),['ambre_analysis','ambre_writing','ambre_review']);
 assert.equal(s.raw.prepare('SELECT provider_calls FROM provider_budgets').get().provider_calls,3);
 const insight=await s.call('/api/chat/insight');assert.equal(insight.turns.length,1);assert.equal(insight.turns[0].review.passed,true);assert.equal(insight.turns[0].context.input_tokens,300);assert.equal(insight.turns[0].context.cached_tokens,90);
 assert.doesNotMatch(JSON.stringify(insight.turns),/rumination|critiqu|mock-only/);assert.equal(job(s).context,'{}');assert.equal(job(s).draft,'{}');
 await message(s);await s.chat.resumeEngine(s.env);assert.equal(calls.length,3);
 assert.equal((await s.call('/api/chat/insight','GET',undefined,'b')).turns.length,0);
});

test('failed blocking criterion rewrites exactly once and rereviews the full answer',async t=>{
 const s=setup();await ready(s);let reviews=0,writes=0;const calls=provider(t,p=>{if(p==='ambre_writing')return {...draft(),content:++writes===1?'Votre mère est narcissique.':draft().content};if(p==='ambre_review'){const r=review();if(++reviews===1)r.criteria[2]='no';return r}});
 const result=await message(s);assert.equal(result.status,200);assert.equal(result.messages.at(-1).content,draft().content);assert.equal(writes,2);assert.equal(reviews,2);assert.equal(job(s).stage,'done');assert.equal(JSON.parse(job(s).review).rewritten,true);assert.equal(calls.length,5);
});

test('unsafe rewrite is replaced without publishing proposals or cutting sentences',async t=>{
 const s=setup();await ready(s);let writes=0;provider(t,p=>{if(p==='ambre_writing'){writes++;return {...draft(),content:'Vous êtes bipolaire. Pourquoi ? Depuis quand ?'}}});
 const result=await message(s);assert.equal(writes,2);assert.match(result.messages.at(-1).content,/reprendre avec prudence/);assert.doesNotMatch(result.messages.at(-1).content,/bipolaire|Pourquoi/);assert.equal(JSON.parse(job(s).review).fallback,true);
});

test('queue is durable before waitUntil and simultaneous replays never duplicate generation',async t=>{
 const s=setup();await ready(s);let release,started;const began=new Promise(r=>started=r);const calls=provider(t,async p=>{if(p==='ambre_analysis'){started();await new Promise(r=>release=r)}}),tasks=[];
 const result=await message(s,'durable_key','Un message.',{waitUntil:p=>{assert.ok(job(s));tasks.push(p)}});await began;assert.equal(result.status,202);await Promise.all([s.chat.engine.work(s.env,result.request_id),message(s,'durable_key','Un message.')]);assert.equal(calls.length,1);release();await Promise.all(tasks);assert.equal(job(s).stage,'done');assert.equal(calls.length,3);
});

test('provider failure leaves resumable work and sanitized error trace, then resumes from persisted stage',async t=>{
 const s=setup();await ready(s);let failing=true;const calls=provider(t,p=>{if(p==='ambre_writing'&&failing)throw Error('PRIVATE message and mock-only')});
 await message(s);assert.equal(job(s).stage,'writing');assert.equal(job(s).attempts,1);const insight=await s.call('/api/chat/insight');assert.equal(insight.turns[0].error.stage,'writing');assert.doesNotMatch(JSON.stringify(insight.turns),/PRIVATE|mock-only/);
 failing=false;await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');assert.equal(calls.filter(c=>c.purpose==='ambre_analysis').length,1);
});

test('consent withdrawal and source updates discard late provider results',async t=>{
 for(const change of ['consent','source','memory','purge']){
  const s=setup();await ready(s);let release,started;const began=new Promise(r=>started=r);provider(t,async p=>{if(p==='ambre_writing'){started();await new Promise(r=>release=r)}});const tasks=[];
  await message(s,'late_'+change,'Un fait privé.',{waitUntil:p=>tasks.push(p)});await began;
  if(change==='consent')await s.call('/api/chat/consent','PUT',{accepted:false});
  if(change==='source')s.raw.prepare("UPDATE profiles SET version=version+1 WHERE user_id='a'").run();
  if(change==='memory')await s.call('/api/chat/memory','PUT',{revision:0,sections:{facts:['Correction explicite']}});
  if(change==='purge')await s.call('/api/chat','DELETE',{});
  release();await Promise.all(tasks);const row=s.raw.prepare('SELECT * FROM chat_turns').get();assert.notEqual(row.status,'completed',change);assert.equal(row.credit_status,'released',change);assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_proposals').get().n,0);
  if(change==='purge')assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_turn_traces').get().n,0);
 }
});

test('weekly budget rejects calls and immediate crisis safety survives full provider failure',async t=>{
 const s=setup();await ready(s);s.env.WEEKLY_TOKEN_BUDGET=1;const calls=provider(t);const result=await message(s,'risk_budget','Je veux mourir.');assert.equal(calls.length,0);assert.match(result.messages.at(-1).content,/3114/);assert.match(result.messages.at(-1).content,/suicide/);assert.equal(job(s).stage,'failed');assert.equal(result.quota.used,0);
});

test('index preserves canonical names and real proposal targets; prompt memory stays under 4000 chars',()=>{
 const context={sources:[{id:'person:p1',type:'person',fields:{role:'mother'},text:'PRIVATE'}],people:[{id:'p1',display_name:'Alice',role:'mother',relationship_text:'Le récit.'}],events:[]};const index=knowledgeIndex(context),details=selectDetails(index,[index[0].id],'Alice');assert.match(index[0].label,/Alice/);assert.equal(details[0].target_id,'p1');assert.equal(details[0].key,'person:p1');
 const summary=Object.fromEntries(['closed','safety','alliance','goals','session','tried','hypotheses','facts'].map(k=>[k,Array.from({length:30},(_,i)=>k+i+' '+('x'.repeat(290)))]));const memory=memoryPrompt(summary);assert.ok(JSON.stringify(memory).length<=4000);assert.ok(memory.closed.length);assert.equal(memory.closed_omitted,true);assert.ok(deterministicAnalysis('Bonjour',memory).constraints.includes('no_exploration'));
 assert.equal(mergeClosed({closed:Array.from({length:30},(_,i)=>'Refus '+i)},['Refus nouveau']).closed_overflow,true);
});

test('memory batches catch up, preserve all 30 closed topics, expose corrections and erase with watermark',async t=>{
 const s=setup();await ready(s);seedTurns(s,16);const closed=Array.from({length:30},(_,i)=>'Sujet fermé '+i);s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a',?,0)").run(JSON.stringify({closed}));provider(t);
 await s.chat.engine.updateMemory(s.env,'a');let memory=s.raw.prepare('SELECT * FROM chat_memory').get();assert.equal(memory.turns_covered,8);assert.equal(JSON.parse(memory.summary).closed.length,30);assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_memory_jobs').get().n,1);
 await s.chat.engine.updateMemory(s.env,'a');memory=s.raw.prepare('SELECT * FROM chat_memory').get();assert.equal(memory.turns_covered,16);assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_memory_jobs').get().n,0);
 const result=await s.call('/api/chat/memory','PUT',{revision:memory.revision,sections:{}});assert.equal(result.status,200,result.message);memory=s.raw.prepare('SELECT * FROM chat_memory').get();assert.equal(memory.through_turn,'history-16');assert.equal(JSON.parse(memory.summary).closed.length,0);assert.equal((await s.call('/api/chat/memory','PUT',{revision:1,sections:{}})).status,409);
 s.raw.prepare("INSERT INTO chat_memory_jobs(user_id,updated_at) VALUES('a',0)").run();await s.chat.engine.updateMemory(s.env,'a');assert.equal(s.raw.prepare('SELECT revision FROM chat_memory').get().revision,memory.revision);
});

test('a concurrent memory correction wins over an in-flight summarizer',async t=>{
 const s=setup();await ready(s);seedTurns(s);let release,started;const began=new Promise(r=>started=r);provider(t,async p=>{if(p==='ambre_memory'){started();await new Promise(r=>release=r);return {items:[{section:'facts',turn_id:'history-1',quote:'Fait rapporté 1'}]}}});const pending=s.chat.engine.updateMemory(s.env,'a');await began;const result=await s.call('/api/chat/memory','PUT',{revision:0,sections:{facts:['La correction de la personne']}});assert.equal(result.status,200);release();await pending;assert.deepEqual(JSON.parse(s.raw.prepare('SELECT summary FROM chat_memory').get().summary).facts,['La correction de la personne']);
});

test('late results cannot reclaim an expired lease or overwrite a new worker',async t=>{
 const s=setup();await ready(s);let release,started;const began=new Promise(r=>started=r);provider(t,async p=>{if(p==='ambre_writing'){started();await new Promise(r=>release=r)}});const tasks=[];await message(s,'lease_expire','Un fait.',{waitUntil:p=>tasks.push(p)});await began;
 const row=job(s);s.raw.prepare("UPDATE chat_engine_jobs SET lease_token='new_worker',lease_until=? WHERE turn_id=?").run(s.getClock()+150,row.turn_id);release();await Promise.all(tasks);assert.equal(job(s).lease_token,'new_worker');assert.equal(job(s).stage,'writing');assert.equal(s.raw.prepare('SELECT status FROM chat_turns').get().status,'running');
});

test('a crash after publication can finalize trace without a new call or duplicate notes',async t=>{
 const s=setup();await ready(s);const calls=provider(t);const original=s.env.DB.prepare;let interrupted=false;
 s.env.DB.prepare=sql=>{if(!interrupted&&sql.startsWith('INSERT OR REPLACE INTO chat_turn_traces')){interrupted=true;throw Error('Simulated process interruption')}return original(sql)};
 await message(s);assert.equal(s.raw.prepare('SELECT status FROM chat_turns').get().status,'completed');assert.equal(job(s).stage,'publish');assert.equal(calls.length,3);
 await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');assert.equal(calls.length,3);assert.equal((await s.call('/api/chat/insight')).turns[0].review.passed,true);
});

test('cartography note success is announced only when the note CAS actually succeeds',async t=>{
 for(const race of [false,true]){
  const s=setup();await ready(s);s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a','{}','{}','{}','{}','[]',1,1,1,1)").run();const quote='Je rumine quand je me sens critiqué.';
  provider(t,p=>p==='ambre_writing'?{...draft(),cartography_notes:[{target:'general',target_id:null,quote}]}:undefined);
  const original=s.env.DB.prepare;let raced=false;if(race)s.env.DB.prepare=sql=>{if(!raced&&sql.startsWith('UPDATE cartographies SET ambre_notes=?')){raced=true;s.raw.prepare("UPDATE cartographies SET ambre_notes=? WHERE user_id='a'").run(JSON.stringify([{id:'other',target:'general',target_id:null,text:'Une note concurrente.',created_at:1}]))}return original(sql)};
  const result=await message(s);assert.equal(job(s).stage,'done');const notes=JSON.parse(s.raw.prepare('SELECT ambre_notes FROM cartographies').get().ambre_notes);if(race){assert.equal(result.messages.at(-1).cartography_notes,undefined);assert.equal(notes[0].id,'other')}else{assert.equal(result.messages.at(-1).cartography_notes.length,1);assert.match(notes[0].text,/Je rumine/)}
 }
});

test('memory correction compare-and-swap rejects a race even after the initial revision read',async()=>{
 const s=setup();await ready(s);s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{}',0)").run();const original=s.env.DB.prepare;let raced=false;s.env.DB.prepare=sql=>{if(!raced&&sql.startsWith('INSERT INTO chat_memory(user_id,summary,through_turn,turns_covered,updated_at)')){raced=true;s.raw.prepare("UPDATE chat_memory SET summary=?,revision=revision+1 WHERE user_id='a'").run(JSON.stringify({facts:['Correction concurrente']}))}return original(sql)};
 const result=await s.call('/api/chat/memory','PUT',{revision:1,sections:{facts:['Correction dépassée']}});assert.equal(result.status,409);assert.deepEqual(JSON.parse(s.raw.prepare('SELECT summary FROM chat_memory').get().summary).facts,['Correction concurrente']);
});

test('all five blocking clinical criteria trigger rewrite; two nonblocking failures trigger rewrite',async t=>{
 for(const failures of [[1],[2],[3],[4],[5],[6,8],[15]]){
  const s=setup();await ready(s);let writes=0,reviews=0;provider(t,p=>{if(p==='ambre_writing'){writes++;return draft()}if(p==='ambre_review'){const value=review();if(!reviews++)for(const n of failures)value.criteria[n-1]='no';return value}});await message(s);assert.equal(writes,failures[0]===15?1:2,failures.join(','));assert.equal(job(s).stage,'done');
 }
});

test('analysis whitelists traces, enforces rhythm, rejects diagnostic state labels and never follows private instructions',async t=>{
 const s=setup();await ready(s);provider(t,p=>{if(p==='ambre_analysis')return {...analysis(),states:['distress','PRIVATE_SECRET'],constraints:['PRIVATE_SECRET'],posture:'stabilize',need:'listen',sensitive_allowed:true,closed_topics:['PRIVATE_SECRET'],extra:'mock-only'};if(p==='ambre_writing')return {...draft(),content:'Ce moment semble difficile. Vous pouvez avancer à votre rythme.'}});await message(s,'analysis_tags','Je suis débordé.');const trace=(await s.call('/api/chat/insight')).turns[0];assert.deepEqual(trace.analysis.states,['distress']);assert.ok(trace.analysis.constraints.includes('no_questions'));assert.ok(trace.analysis.constraints.includes('no_advice'));assert.equal(trace.analysis.sensitive_allowed,false);assert.doesNotMatch(JSON.stringify(trace),/PRIVATE_SECRET|mock-only|débordé/);
});

test('care text generation retains its public signature and reserves the member budget',async t=>{
 const s=setup();await ready(s);const calls=provider(t,()=>response('Compte rendu de test.'));const result=await s.chat.generateText(s.env,{userId:'a',system:'Préparer la séance.',user:'Faits rapportés.',purpose:'care',maxTokens:300,timeoutMs:1000});assert.equal(result.text,'Compte rendu de test.');assert.equal(result.input_tokens,100);assert.equal(calls[0].purpose,undefined);assert.equal(s.raw.prepare('SELECT provider_calls FROM provider_budgets').get().provider_calls,1);
});

test('rich sources and a 100-turn conversation stay bounded and measure stage inputs',async t=>{
 const s=setup();await ready(s);seedTurns(s,100);s.raw.prepare("UPDATE chat_turns SET week_start='2026-01-01',created_at='2026-01-01 00:00:00'").run();s.raw.prepare("INSERT INTO chat_memory(user_id,summary,through_turn,turns_covered,updated_at) VALUES('a',?,'history-100',100,0)").run(JSON.stringify({facts:Array.from({length:12},(_,i)=>'Fait '+i+' '+('information '.repeat(24))),closed:['Je ne veux pas parler de mon enfance.'],safety:[]}));
 const base=s.personal.context;s.personal.context=async(...args)=>{const c=await base(...args);return {...c,people:Array.from({length:50},(_,i)=>({id:'person-'+i,display_name:'Proche '+i,role:'friend',relationship_text:'Une relation rapportée. '.repeat(200)})),events:Array.from({length:500},(_,i)=>({id:'event-'+i,title:'Événement personnel '+i,description:'Un récit de la personne. '.repeat(200),meaning_now:'Signification déclarée.'})),portrait:{psychological_map:Array.from({length:60},(_,i)=>({id:'h-'+i,label:'Hypothèse incertaine '+i,text:'Piste à vérifier. '.repeat(100)}))}}};
 const calls=provider(t);const result=await message(s,'rich_context','Je rumine quand je me sens critiqué.');assert.equal(job(s).stage,'done',JSON.stringify(job(s)));const inputs=calls.map(({purpose,body})=>({purpose,chars:JSON.stringify(body.messages).length,estimated_tokens:Math.ceil((JSON.stringify(body.messages).length+JSON.stringify(body.response_format).length)/3)}));t.diagnostic(JSON.stringify(inputs));assert.ok(inputs.reduce((n,x)=>n+x.estimated_tokens,0)<=18000);const writing=JSON.parse(calls.find(x=>x.purpose==='ambre_writing').body.messages[1].content);assert.equal(writing.history.length,6);assert.ok(JSON.stringify(writing.memory).length<=4000);assert.equal(result.messages.at(-1).content,draft().content);
});

test('a refused topic commits atomically with the answer before trace finalization',async t=>{
 const s=setup();await ready(s);provider(t);const original=s.env.DB.prepare;let interrupted=false;s.env.DB.prepare=sql=>{if(!interrupted&&sql.startsWith('INSERT OR REPLACE INTO chat_turn_traces')){interrupted=true;throw Error('After commit')}return original(sql)};
 await message(s,'closed_atomic','Je ne veux plus parler de mon enfance.');assert.equal(s.raw.prepare('SELECT status FROM chat_turns').get().status,'completed');assert.deepEqual(JSON.parse(s.raw.prepare('SELECT summary FROM chat_memory').get().summary).closed,['Je ne veux plus parler de mon enfance.']);await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');
});

test('an expired summarizer cannot write after its memory lease is replaced',async t=>{
 const s=setup();await ready(s);seedTurns(s);let release,started;const began=new Promise(r=>started=r);provider(t,async p=>{if(p==='ambre_memory'){started();await new Promise(r=>release=r);return {items:[{section:'facts',turn_id:'history-1',quote:'Fait rapporté 1'}]}}});const pending=s.chat.engine.updateMemory(s.env,'a');await began;s.raw.prepare("UPDATE chat_memory_jobs SET lease_token='new_owner' WHERE user_id='a'").run();release();await pending;assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_memory').get().n,0);assert.equal(s.raw.prepare('SELECT lease_token FROM chat_memory_jobs').get().lease_token,'new_owner');
});

test('past safety history informs context without making every greeting a current crisis',async t=>{
 const history={safety_seen:true,safety:['Je veux mourir.']};assert.equal(deterministicAnalysis('Bonjour',history).risk,'none');assert.equal(deterministicAnalysis('Bonjour',history).posture,'explore');assert.equal(deterministicAnalysis('Je veux mourir.',history).risk,'explicit');
 const s=setup();await ready(s);s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a',?,0)").run(JSON.stringify(history));const calls=provider(t);const result=await message(s,'past_safety','Bonjour');assert.equal(job(s).stage,'done');assert.equal(JSON.parse(job(s).analysis).risk,'none');assert.doesNotMatch(result.messages.at(-1).content,/3114|danger immédiat/);const input=JSON.parse(calls[0].body.messages[1].content);assert.equal(input.memory.safety_seen,true);assert.deepEqual(input.memory.safety,history.safety);
});
