import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync,mkdirSync,writeFileSync} from 'node:fs';
import {createPersonalBackend,parisWeek} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {appendTraceStage,traceDetail} from '../src/ambre-insight.mjs';
import {deterministicAnalysis,deterministicReview,knowledgeIndex,selectDetails,memoryPrompt,mergeClosed,analysisIndex,parisTime} from '../src/ambre-engine.mjs';
function setup(catalogue){const raw=new DatabaseSync(':memory:');for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));function prepare(sql){let args=[];return {bind(...a){args=a;return this;},async first(){return raw.prepare(sql).get(...args)||null;},async all(){return {results:raw.prepare(sql).all(...args)};},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}};}};}let queue=Promise.resolve();const DB={raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const result=[];for(const s of statements)result.push(await s.run());raw.exec('COMMIT');return result;}catch(e){raw.exec('ROLLBACK');throw e;}});queue=next.catch(()=>{});return next;}};let clock=Math.floor(Date.now()/1000),serial=0;const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code});},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null}};const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue,engineEnabled:true}),env={DB,OPENROUTER_API_KEY:'mock-only'};for(const id of ['a','b']){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.com');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES(?,'active',?)").run(id,clock+86400);}async function call(path,method='GET',data,user='a',ctx){const req=new Request('https://example.com'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:user},url.pathname,url,ctx)||await personal.routes(req,env,{id:user},url.pathname,url,ctx);return {status:res.status,...await res.json()};}catch(e){return {status:e.status||500,code:e.code,message:e.message};}}return {raw,env,chat,personal,call,setClock:v=>{clock=v;},getClock:()=>clock};}

async function ready(s){for(const id of ['a','b'])seedCompanionPrerequisites(s.raw,id);for(const id of ['a','b'])s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,'initial','ready','Analyse initiale.',?,0,0)").run('seed-'+id,id,'seed-initial-'+id);assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);}
const message=(s,key='request_001',content='Je rumine quand je me sens critiqué.',ctx)=>s.call('/api/chat/messages','POST',{content,idempotency_key:key},'a',ctx);
const analysis=()=>({need:'understand',states:[],risk:'none',posture:'explore',sensitive_allowed:false,constraints:[],selected_ids:[],closed_topics:[]});
const draft=()=>({content:'Vous décrivez une rumination après les critiques. Nous pouvons prendre le temps de comprendre ce qui vous pèse.',questions:[],proposals:[],cartography_notes:[],people_observations:[]});
const review=()=>({criteria:Array(15).fill('yes'),reasons:[]});
const response=data=>Response.json({choices:[{message:{content:typeof data==='string'?data:JSON.stringify(data)},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:80,prompt_tokens_details:{cached_tokens:30},cost:0.001}});
function provider(t,handler){const calls=[],original=globalThis.fetch;globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;calls.push({purpose,body});const custom=await handler?.(purpose,body,calls);if(custom instanceof Response)return custom;return response(custom??({ambre_analysis:analysis(),ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:draft(),ambre_review:review(),ambre_memory:{items:[]}}[purpose]));};t.after(()=>{globalThis.fetch=original});return calls;}
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
  const s=setup();await ready(s);s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a','{}','{}','{}','{}','[]',1,1,1,1) ON CONFLICT(user_id) DO UPDATE SET answers=json_patch(cartographies.answers,excluded.answers),feedback=excluded.feedback,notes=excluded.notes,resources=excluded.resources,ambre_notes=excluded.ambre_notes,consent=excluded.consent,locked_at=excluded.locked_at,version=excluded.version,updated_at=excluded.updated_at").run();const quote='Je rumine quand je me sens critiqué.';
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
 const s=setup();await ready(s);seedTurns(s,100);s.raw.prepare("UPDATE chat_turns SET week_start='2026-01-01',created_at='2026-01-01 00:00:00'").run();s.raw.prepare("UPDATE monthly_usage SET period_id='2026-01' WHERE kind='message'").run();s.raw.prepare("INSERT INTO chat_memory(user_id,summary,through_turn,turns_covered,updated_at) VALUES('a',?,'history-100',100,0)").run(JSON.stringify({facts:Array.from({length:12},(_,i)=>'Fait '+i+' '+('information '.repeat(24))),closed:['Je ne veux pas parler de mon enfance.'],safety:[]}));
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

test('a reviewer waits for a fresh execution window instead of receiving a few seconds',async t=>{
 const s=setup();await ready(s);const realNow=Date.now,base=realNow();let elapsed=0;Date.now=()=>base+elapsed;t.after(()=>{Date.now=realNow});
 const calls=provider(t,p=>{if(p==='ambre_analysis')elapsed=6000;if(p==='ambre_writing')elapsed=17000});await message(s,'review_window');
 assert.deepEqual(calls.map(c=>c.purpose),['ambre_analysis','ambre_writing']);assert.equal(job(s).stage,'review');assert.equal(job(s).attempts,0);assert.equal(s.raw.prepare('SELECT provider_calls FROM provider_budgets').get().provider_calls,2);assert.equal(JSON.parse(job(s).review).error_code,undefined);assert.equal(s.raw.prepare('SELECT status FROM chat_turns').get().status,'running');
 await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');assert.equal(calls.length,3);assert.equal(JSON.parse(job(s).review).checked,true);assert.equal(JSON.parse(job(s).review).passed,true);
});

test('reviewer transient timeout and HTTP errors resume only the review on later leases',async t=>{
 for(const failure of ['timeout','http']){
  const s=setup();await ready(s);let reviews=0;const calls=provider(t,p=>{if(p==='ambre_review'&&++reviews===1){if(failure==='http')return new Response('PRIVATE_SECRET upstream text',{status:502});throw Object.assign(Error('PRIVATE_SECRET mock-only'),{name:'TimeoutError'})}});
  await message(s,'retry_'+failure);assert.equal(job(s).stage,'review');assert.equal(job(s).attempts,1);assert.equal(JSON.parse(job(s).review).fallback,undefined);assert.equal(s.raw.prepare('SELECT status FROM chat_turns').get().status,'running');
  const trace=(await s.call('/api/chat/insight')).turns[0];assert.equal(trace.review.error_code,failure==='http'?'provider_http_502':'provider_timeout');assert.equal(trace.error.code,trace.review.error_code);assert.doesNotMatch(JSON.stringify(trace),/PRIVATE_SECRET|mock-only/);
  await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');assert.equal(JSON.parse(job(s).review).passed,true);assert.equal(JSON.parse(job(s).review).attempts,2);assert.equal(calls.filter(c=>c.purpose==='ambre_writing').length,1);assert.equal(calls.filter(c=>c.purpose==='ambre_analysis').length,1);assert.equal(reviews,2);
 }
});

test('three unavailable reviewer attempts publish a safe fallback and never retry indefinitely',async t=>{
 const s=setup();await ready(s);const calls=provider(t,p=>{if(p==='ambre_review')return new Response('PRIVATE_SECRET',{status:502})});await message(s,'review_three');assert.equal(job(s).stage,'review');assert.equal(job(s).attempts,1);
 await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'review');assert.equal(job(s).attempts,2);assert.equal(s.raw.prepare('SELECT status FROM chat_turns').get().status,'running');
 await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');const verdict=JSON.parse(job(s).review);assert.equal(verdict.checked,false);assert.equal(verdict.fallback,true);assert.equal(verdict.attempts,3);assert.equal(verdict.error_code,'provider_http_502');assert.match((await s.call('/api/chat')).messages.at(-1).content,/reprendre avec prudence/);assert.equal(calls.filter(c=>c.purpose==='ambre_review').length,3);assert.equal(calls.filter(c=>c.purpose==='ambre_writing').length,1);
 await s.chat.resumeEngine(s.env);assert.equal(calls.length,5);assert.doesNotMatch(JSON.stringify((await s.call('/api/chat/insight')).turns),/PRIVATE_SECRET/);
});

test('malformed reviewer output is classified without persisting provider content',async t=>{
 for(const value of ['PRIVATE_SECRET invalid JSON',{criteria:['PRIVATE_SECRET'],reasons:['mock-only']}]){
  const s=setup();await ready(s);const calls=provider(t,p=>p==='ambre_review'?response(value):undefined);await message(s,'bad_review');assert.equal(job(s).stage,'done');const trace=(await s.call('/api/chat/insight')).turns[0];assert.equal(trace.review.error_code,'review_invalid_response');assert.equal(trace.review.fallback,true);assert.doesNotMatch(JSON.stringify(trace),/PRIVATE_SECRET|mock-only/);assert.equal(calls.filter(c=>c.purpose==='ambre_review').length,1);
 }
});

test('small stages use minimal reasoning and larger reserved output while writing is unchanged',async t=>{
 const s=setup();await ready(s);const calls=provider(t);await message(s,'reasoning_caps');for(const purpose of ['ambre_analysis','ambre_review']){const body=calls.find(c=>c.purpose===purpose).body;assert.equal(body.reasoning.effort,'minimal');assert.equal(body.max_tokens,1500)}const writing=calls.find(c=>c.purpose==='ambre_writing').body;assert.equal(writing.reasoning.effort,'low');assert.equal(writing.max_tokens,1800);assert.equal(writing.model,'anthropic/claude-sonnet-5.5');
 const expected=calls.reduce((sum,{body})=>sum+Math.ceil((body.messages[0].content[0].text.length+body.messages[1].content.length+JSON.stringify(body.response_format.json_schema.schema).length)/3)+body.max_tokens,0);assert.equal(s.raw.prepare('SELECT reserved_tokens FROM provider_budgets').get().reserved_tokens,expected);
 await s.chat.generateText(s.env,{userId:'a',system:'Mémoire.',user:'Citations.',purpose:'memory',maxTokens:1200,schema:{type:'object'}});assert.equal(calls.at(-1).body.max_tokens,2400);assert.equal(calls.at(-1).body.reasoning.effort,'minimal');
});

test('truncated review is distinct from malformed output, retains numeric reasoning usage and retries with room',async t=>{
 const s=setup();await ready(s);let reviews=0;const calls=provider(t,p=>{if(p==='ambre_review'&&++reviews===1)return Response.json({choices:[{message:{content:'PRIVATE_SECRET',reasoning:'PRIVATE_REASONING'},finish_reason:'length'}],usage:{prompt_tokens:100,completion_tokens:1500,completion_tokens_details:{reasoning_tokens:1490}}})});await message(s,'review_truncated');assert.equal(job(s).stage,'review');const trace=(await s.call('/api/chat/insight')).turns[0];assert.equal(trace.review.error_code,'output_limit');assert.equal(trace.review.usage.reasoning_tokens,1490);assert.equal(trace.review.usage.output_tokens,1500);assert.doesNotMatch(JSON.stringify(trace),/PRIVATE_SECRET|PRIVATE_REASONING/);
 await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');assert.equal(JSON.parse(job(s).review).passed,true);const reviewers=calls.filter(c=>c.purpose==='ambre_review');assert.deepEqual(reviewers.map(c=>c.body.max_tokens),[1500,2250]);assert.equal(JSON.parse(job(s).review).usage.reasoning_tokens,1490);assert.equal(JSON.parse(job(s).review).usage.output_tokens,1580);assert.equal(calls.filter(c=>c.purpose==='ambre_writing').length,1);assert.ok(job(s).reserved_tokens<=30000);
});

test('truncated analysis keeps deterministic safety and exposes only numeric usage and output-limit code',async t=>{
 const s=setup();await ready(s);provider(t,p=>p==='ambre_analysis'?Response.json({choices:[{message:{content:null,reasoning:'PRIVATE_REASONING'},finish_reason:'length'}],usage:{prompt_tokens:100,completion_tokens:1500,completion_tokens_details:{reasoning_tokens:1500}}}):undefined);await message(s,'analysis_truncated','Je veux mourir.');assert.equal(job(s).stage,'done');const trace=(await s.call('/api/chat/insight')).turns[0];assert.equal(trace.analysis.fallback,true);assert.equal(trace.analysis.error_code,'output_limit');assert.equal(trace.analysis.risk,'explicit');assert.equal(trace.analysis.usage.reasoning_tokens,1500);assert.match((await s.call('/api/chat')).messages.at(-1).content,/3114/);assert.doesNotMatch(JSON.stringify(trace),/PRIVATE_REASONING/);
});

test('repeated output truncation stops at three bounded review attempts',async t=>{
 const s=setup();await ready(s);const calls=provider(t,(p,body)=>p==='ambre_review'?Response.json({choices:[{message:{content:null},finish_reason:'length'}],usage:{prompt_tokens:100,completion_tokens:body.max_tokens,completion_tokens_details:{reasoning_tokens:body.max_tokens}}}):undefined);await message(s,'limit_three');await s.chat.resumeEngine(s.env);await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');assert.deepEqual(calls.filter(c=>c.purpose==='ambre_review').map(c=>c.body.max_tokens),[1500,2250,3000]);const verdict=JSON.parse(job(s).review);assert.equal(verdict.fallback,true);assert.equal(verdict.error_code,'output_limit');assert.equal(verdict.attempts,3);assert.equal(verdict.usage.reasoning_tokens,6750);assert.ok(job(s).reserved_tokens<=30000);await s.chat.resumeEngine(s.env);assert.equal(calls.length,5);
});

test('explicit listening and question refusals survive a conflicting model analysis',async t=>{
 const listening='Je voudrais simplement être écouté, sans conseil pour le moment.';
 for(const text of [listening,'Je veux uniquement être écoutée.','Sans conseils, merci.']){const a=deterministicAnalysis(text);assert.equal(a.need,'listen');assert.equal(a.posture,'listen');assert.ok(a.constraints.includes('no_advice'))}
 for(const text of ['Pas de questions, merci.','Aucune question pour le moment.','Je préfère parler sans questions.'])assert.ok(deterministicAnalysis(text).constraints.includes('no_questions'));
 for(const [text,constraint] of [[listening,'no_advice'],['Aucune question pour le moment.','no_questions']]){
  const s=setup();await ready(s);const calls=provider(t,p=>p==='ambre_analysis'?{...analysis(),need:'advice',posture:'options',constraints:[]}:undefined);await message(s,'explicit_limit',text);const actual=JSON.parse(calls.find(c=>c.purpose==='ambre_writing').body.messages[1].content).analysis;assert.ok(actual.constraints.includes(constraint));if(constraint==='no_advice'){assert.equal(actual.need,'listen');assert.equal(actual.posture,'listen')}assert.equal(job(s).stage,'done');
 }
});

test('owner polling resumes pending work without blocking GET or duplicating concurrent provider calls',async t=>{
 const s=setup();await ready(s);const realNow=Date.now,base=realNow();let elapsed=0;Date.now=()=>base+elapsed;t.after(()=>{Date.now=realNow});
 let release,started;const began=new Promise(r=>started=r);t.after(()=>release?.());const calls=provider(t,async p=>{if(p==='ambre_analysis')elapsed=6000;if(p==='ambre_writing')elapsed=17000;if(p==='ambre_review'){started();await new Promise(r=>release=r)}});
 const sent=await message(s,'poll_pending');Date.now=realNow;assert.equal(job(s).stage,'review');assert.equal(calls.length,2);
 const otherTasks=[];const other=await s.call('/api/chat','GET',undefined,'b',{waitUntil:p=>otherTasks.push(p)});assert.equal(other.pending_request,null);assert.equal(otherTasks.length,0);assert.equal(calls.length,2);
 await s.call('/api/chat');assert.equal(calls.length,2,'without waitUntil the cron remains responsible');
 const tasks=[],ctx={waitUntil:p=>tasks.push(p)};let timer;const polled=await Promise.race([s.call('/api/chat','GET',undefined,'a',ctx),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('GET awaited the provider')),1500)})]).finally(()=>clearTimeout(timer));assert.equal(polled.status,200);assert.equal(polled.pending_request.id,sent.request_id);await began;
 await Promise.all(Array.from({length:4},()=>s.call('/api/chat','GET',undefined,'a',ctx)));assert.equal(calls.filter(c=>c.purpose==='ambre_review').length,1);assert.equal(job(s).stage,'review');release();await Promise.all(tasks);assert.equal(job(s).stage,'done');assert.equal(calls.length,3);const taskCount=tasks.length;assert.equal((await s.call('/api/chat','GET',undefined,'a',ctx)).pending_request,null);assert.equal(tasks.length,taskCount);
});

test('publication during state construction keeps pending and messages on one consistent snapshot',async()=>{
 const s=setup();await ready(s);s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,lease_until) VALUES('publish-race','a','request_race','hash','running','reserved',?,1,1,'Un message en attente.','',?)").run(parisWeek(new Date()).week_start,s.getClock()+600);
 const original=s.env.DB.prepare;let published=false;s.env.DB.prepare=sql=>{const statement=original(sql);if(sql.startsWith('SELECT id,idempotency_key,content,response,status,error_code,lease_until,created_at FROM chat_turns')){const read=statement.all.bind(statement);statement.all=async()=>{const snapshot=await read();if(!published){published=true;s.raw.prepare("UPDATE chat_turns SET status='completed',credit_status='consumed',response=? WHERE id='publish-race'").run(JSON.stringify({companion_reply:1,content:'Réponse publiée après la lecture.'}))}return snapshot}}return statement};
 const during=await s.call('/api/chat');assert.equal(published,true);assert.deepEqual(during.pending_request,{id:'publish-race',status:'running',client_key:'request_race'});assert.equal(during.last_request.status,'running');assert.equal(during.messages.some(m=>m.role==='assistant'),false);
 const after=await s.call('/api/chat');assert.equal(after.pending_request,null);assert.equal(after.last_request.status,'completed');assert.equal(after.messages.at(-1).content,'Réponse publiée après la lecture.');
});

test('private v2 snapshots exactly reproduce supplied provider inputs, models and instructions',async t=>{
 const s=setup();await ready(s);s.env.OPENROUTER_ANALYSIS_MODEL='test/analyzer';s.env.OPENROUTER_CHAT_MODEL='test/writer';s.env.OPENROUTER_CRITIC_MODEL='test/reviewer';
 s.raw.prepare("UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id='a'").run(JSON.stringify({first_name:'Lara',birth_date:'1990-04-20',self_description:'PRIVATE_SELF Je prends le temps de comprendre.'}));
 for(const [id,user,name,account] of [['pa','a','Alice','PRIVATE_PERSON Je peux parler à Alice.'],['pb','b','Other person','OTHER_USER_SECRET']])s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES(?,?,?,'hash',?,'friend',?)").run(id,user,id,name,account);
 seedTurns(s,8);s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a',?,0)").run(JSON.stringify({facts:['PRIVATE_MEMORY Je rumine après certaines critiques.'],closed:['Je ne souhaite pas parler de mon enfance.']}));
 const calls=provider(t,(purpose)=>purpose==='ambre_review'?Response.json({choices:[{message:{content:JSON.stringify(review()),reasoning:'MODEL_INTERNAL_SECRET'},finish_reason:'stop'}],model:'provider/reviewer-release',usage:{prompt_tokens:100,completion_tokens:80}}):undefined);
 const sent=await message(s,'transparent_turn','PRIVATE_MESSAGE Je veux comprendre ma relation avec Alice.');assert.equal(job(s).stage,'done');
 const overview=await s.call('/api/chat/insight');assert.equal(overview.current.identity.first_name,'Lara');assert.equal(overview.models.memory,'test/analyzer');assert.equal(overview.current.history.length,6);assert.equal(overview.current.self_profile.birth_date,'1990-04-20');assert.equal(overview.turns[0].details_available,true);assert.doesNotMatch(JSON.stringify(overview.turns),/PRIVATE_SELF|PRIVATE_MEMORY|PRIVATE_PERSON|PRIVATE_MESSAGE/);
 const response=await s.call('/api/chat/insight?turn_id='+sent.request_id),detail=response.selected_turn;assert.equal(response.current,undefined,'turn detail avoids rebuilding the current inventory');assert.equal(detail.trace_version,2);assert.equal(detail.stages.length,3);assert.equal(detail.outcome,'published');
 for(let i=0;i<calls.length;i++){assert.deepEqual(detail.stages[i].input,JSON.parse(calls[i].body.messages[1].content));assert.equal(detail.stages[i].system,calls[i].body.messages[0].content[0].text);assert.equal(detail.stages[i].model,calls[i].body.model);assert.equal(detail.stages[i].status,'completed')}
 assert.equal(detail.stages[2].model_received,'provider/reviewer-release');assert.deepEqual(detail.snapshot.details,detail.stages[1].input.details);assert.equal(detail.snapshot.identity.first_name,'Lara');assert.equal(typeof detail.snapshot.self_context.age,'number');assert.match(detail.snapshot.message,/PRIVATE_MESSAGE/);assert.doesNotMatch(JSON.stringify(detail),/OTHER_USER_SECRET|MODEL_INTERNAL_SECRET|mock-only|Bearer /);
 assert.equal((await s.call('/api/chat/insight?turn_id='+sent.request_id,'GET',undefined,'b')).status,404);
 const stored=JSON.parse(s.raw.prepare('SELECT trace FROM chat_turn_traces WHERE turn_id=?').get(sent.request_id).trace);assert.ok(stored.details.values.length>0);assert.equal(stored.details.stages[1].input,undefined,'large common inputs are pooled once');assert.ok(new TextEncoder().encode(JSON.stringify(stored.details)).length<=128000);
 if(process.env.AMBRE_INSIGHT_FIXTURES){const path=process.env.AMBRE_INSIGHT_FIXTURES;mkdirSync(path,{recursive:true});writeFileSync(path+'/insight-current.json',JSON.stringify(overview,null,2));writeFileSync(path+'/insight-turn.json',JSON.stringify(response,null,2))}
 s.raw.prepare("UPDATE profiles SET self_data=json_patch(self_data,?),version=version+1 WHERE user_id='a'").run(JSON.stringify({first_name:'Nouveau prénom'}));assert.equal((await s.call('/api/chat/insight')).current.identity.first_name,'Nouveau prénom');assert.equal((await s.call('/api/chat/insight?turn_id='+sent.request_id)).selected_turn.snapshot.identity.first_name,'Lara');
 await s.call('/api/chat','DELETE',{});assert.equal(s.raw.prepare("SELECT count(*) n FROM chat_turn_traces WHERE user_id='a'").get().n,0);assert.equal((await s.call('/api/chat/insight?turn_id='+sent.request_id)).status,404);
});

test('current source inventory paginates owner data and exposes bounded excerpts honestly',async()=>{
 const s=setup();await ready(s);const long='Présentation '.repeat(750);s.raw.prepare("UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id='a'").run(JSON.stringify({self_description:long}));
 for(let i=0;i<105;i++)s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,description,kind) VALUES(?,'a',?,'hash',?,?,'other')").run('ev'+i,'ev'+i,'Événement '+i,'Récit personnel '+i);
 s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,description,kind) VALUES('other-event','b','other-event','hash','OTHER_USER_SECRET','Autre personne','other')").run();
 s.raw.prepare("INSERT INTO article_likes(user_id,article_id,reason,differences) VALUES('a','article-test','Un récit aimé.','Des différences.')").run();
 const first=await s.call('/api/chat/insight');assert.equal(first.current.sources.items.length,100);assert.equal(first.current.sources.next_offset,100);assert.equal(first.knowledge.next_offset,100);const second=await s.call('/api/chat/insight?offset=100');assert.equal(second.current.sources.items.length,73);assert.equal(second.current.sources.next_offset,null);const profile=second.current.sources.items.find(x=>x.kind==='profile');assert.equal(profile.content.length,6000);assert.equal(profile.truncated,true);assert.ok(profile.original_chars>6000);assert.equal(profile.api_path,'/api/self-profile');assert.equal(second.current.self_profile.self_description,long);
 assert.doesNotMatch(JSON.stringify([first,second]),/OTHER_USER_SECRET/);assert.equal(first.current.stored_elsewhere.find(x=>x.key==='likes').usage,'not_sent_to_chat');assert.equal(first.current.stored_elsewhere.find(x=>x.key==='accepted_memory').api_path,'/api/memory');assert.equal(first.current.stored_elsewhere.find(x=>x.key==='older_history').edit_path,'/mon-accompagnement/');assert.equal(first.current.usage_policy.details_max_chars,5600);assert.equal(first.current.usage_policy.last_review_max_chars,900);
});

test('legacy traces remain explicitly unavailable and turn summaries are paginated without private snapshots',async()=>{
 const s=setup();await ready(s);seedTurns(s,25);for(let i=1;i<=25;i++)s.raw.prepare("INSERT INTO chat_turn_traces(turn_id,user_id,trace,created_at) VALUES(?,'a',?,?)").run('history-'+i,JSON.stringify({analysis:{need:'listen'},review:{passed:true}}),i);
 const first=await s.call('/api/chat/insight');assert.equal(first.turns.length,20);assert.equal(first.turns_next_offset,20);assert.equal(first.turns[0].details_available,false);assert.equal(first.turns[0].unavailable_reason,'legacy_trace');const next=await s.call('/api/chat/insight?turns_offset=20');assert.equal(next.turns.length,5);assert.equal(next.turns_next_offset,null);
 const detail=(await s.call('/api/chat/insight?turn_id=history-1')).selected_turn;assert.equal(detail.details_available,false);assert.equal(detail.unavailable_reason,'legacy_trace');assert.equal(detail.snapshot,null);assert.deepEqual(detail.stages,[]);
 s.raw.prepare("DELETE FROM users WHERE id='a'").run();assert.equal(s.raw.prepare("SELECT count(*) n FROM chat_turn_traces WHERE user_id='a'").get().n,0,'account deletion cascades through private snapshots');
});

test('cartography sources retain explicit feedback, negative answers, evidence and child-only limits',()=>{
 const context={people:[],events:[],sources:[],cartography:{locked_at:1,causes:[{id:'f1',label:'Vécu rapporté',intensity:1}],explicit_negative:[{id:'f2',label:'Vécu écarté'}],convergence:[{id:'o1',label:'Piste empirique',category:'tendency',relation_types:['empirical_association']}],recognized:[{id:'o1',label:'Piste empirique'},{id:'child',label:'Contexte du jeune enfant',contextual_only:true,note:'jeune_enfant',age_scope:'young_child_only',category:'diagnosis'}],dismissed:[{id:'o2',label:'Piste écartée',category:'schema'}],contextual:[{id:'hidden',label:'Repère contextuel masqué',contextual_only:true}],resources:[{id:'r1',label:'Adulte soutenant'}],evidence:[{id:'o1',links:[{factor_id:'f1',relation_type:'empirical_association',source_ids:['ref1'],summary:'Association observée dans des groupes.'}]}],evidence_sources:[{id:'ref1',url:'https://example.org/study',citation:'Étude vérifiable'}]}};
 const index=knowledgeIndex(context),outcome=index.find(x=>x.key==='cartography:outcome:o1'),child=index.find(x=>x.key==='cartography:outcome:child');assert.equal(outcome.metadata.feedback,'oui');assert.equal(index.find(x=>x.key==='cartography:outcome:o2').metadata.feedback,'non');assert.ok(index.some(x=>x.metadata?.explicit_negative));assert.ok(index.some(x=>x.kind==='resource'));assert.match(outcome.text,/empirical_association|Étude vérifiable/);assert.equal(child,undefined,'v5 final review: an older answer on a hidden contextual subject stays stored but is no longer indexed');
 // Plan v5.1: contextual subjects are no longer indexed.
 assert.equal(index.some(x=>x.key==='cartography:outcome:hidden'),false);
 assert.equal(selectDetails(index,[],'Bonjour').some(x=>x.key==='cartography:outcome:child'),false);
});

test('snapshot storage shares identical inputs and marks omissions instead of inventing exact content',()=>{
 const args={stage:'review',model:'test/model',input:{message:'Un message privé',memory:{facts:['Un fait']},history:[]},system:'Instructions publiques',source_version:1,started_at:1,max_tokens:1500};const first=appendTraceStage({},args);const second=appendTraceStage(first.trace,args);assert.equal(second.trace.details.values.length,first.trace.details.values.length);assert.equal(second.trace.details.stages.length,2);
 const oversized=appendTraceStage({}, {...args,input:{message:'x'.repeat(150000)}});const detail=traceDetail({id:'test',trace:oversized.trace});assert.equal(detail.stages[0].details_available,false);assert.equal(detail.stages[0].unavailable_reason,'snapshot_limit');assert.equal(detail.stages[0].input,null);assert.equal(detail.details_truncated,true);assert.ok(JSON.stringify(oversized.trace).length<128000);
});

test('dense selected outcome retains a proof summary and source URL before long cause lists',()=>{
 const context={sources:[],people:[],events:[],cartography:{convergence:[{id:'dense',label:'Piste à vérifier',causes:Array.from({length:58},(_,i)=>'Vécu très long numéro '+i),relation_types:['empirical_association']}],recognized:[{id:'dense',label:'Piste à vérifier'}],evidence:[{id:'dense',links:[{relation_type:'empirical_association',source_ids:['source-one'],summary:'Une association a été observée dans des groupes, sans démontrer une causalité individuelle.'}]}],evidence_sources:[{id:'source-one',url:'https://example.org/publication',citation:'Référence de la publication.'}]}};
 const index=knowledgeIndex(context),source=index.find(x=>x.kind==='outcome'),selected=selectDetails(index,[source.id],'Cette piste')[0];assert.ok(selected.text.length<=1100);assert.match(selected.text,/Une association a été observée/);assert.match(selected.text,/https:\/\/example.org\/publication/);assert.match(selected.text,/empirical_association/);assert.equal(selected.metadata.feedback,'oui');assert.equal(selected.truncated,true);
});

test('private stage snapshots preserve every rewrite and retry while distinguishing pending from failure',async t=>{
 const s=setup();await ready(s);let writes=0,reviews=0;const calls=provider(t,p=>{if(p==='ambre_writing')return {...draft(),content:++writes===1?'Vous êtes bipolaire.':draft().content};if(p==='ambre_review'&&++reviews===2)throw Object.assign(Error('Internal private provider message'),{name:'TimeoutError'})});const sent=await message(s,'snapshot_rewrite');assert.equal(job(s).stage,'review');assert.equal((await s.call('/api/chat/insight?turn_id='+sent.request_id)).selected_turn.outcome,'pending');await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');
 const detail=(await s.call('/api/chat/insight?turn_id='+sent.request_id)).selected_turn;assert.deepEqual(detail.stages.map(x=>x.stage),['analysis','writing','review','rewrite','review','review']);assert.equal(detail.stages[4].status,'error');assert.equal(detail.stages[4].error_code,'provider_timeout');assert.equal(detail.outcome,'published');for(let i=0;i<calls.length;i++)assert.deepEqual(detail.stages[i].input,JSON.parse(calls[i].body.messages[1].content));assert.doesNotMatch(JSON.stringify(detail),/Internal private provider message/);
});

test('primary clinical evidence retains its complete source and theoretical status in the actual writing request',async t=>{
 const s=setup();await ready(s);const link=CARTOGRAPHIE.links.find(x=>x.factor==='emprise_religieuse'&&x.outcome==='sch_assujettissement');assert.equal(link.relation_type,'clinical_model');assert.equal(link.strength,'theorique');
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a',?,?,'{}','{}','[]',1,1,1,1) ON CONFLICT(user_id) DO UPDATE SET answers=json_patch(cartographies.answers,excluded.answers),feedback=excluded.feedback,notes=excluded.notes,resources=excluded.resources,ambre_notes=excluded.ambre_notes,consent=excluded.consent,locked_at=excluded.locked_at,version=excluded.version,updated_at=excluded.updated_at").run(JSON.stringify({emprise_religieuse:1}),JSON.stringify({sch_assujettissement:'oui'}));
 const index=knowledgeIndex(await s.personal.context(s.env,'a')),id=index.find(x=>x.key==='cartography:outcome:sch_assujettissement').id;const calls=provider(t,p=>p==='ambre_analysis'?{...analysis(),selected_ids:[id]}:undefined);await message(s,'whole_evidence','Je voudrais comprendre cette emprise religieuse et mon assujettissement.');assert.equal(job(s).stage,'done');
 const writing=JSON.parse(calls.find(c=>c.purpose==='ambre_writing').body.messages[1].content),entry=writing.details.find(x=>x.key==='cartography:outcome:sch_assujettissement');assert.equal(entry.metadata.evidence_summary,link.summary);assert.ok(link.source_ids.includes(entry.metadata.evidence_source.id));assert.equal(entry.metadata.evidence_source.url,CARTOGRAPHIE.sources.find(x=>x.id===entry.metadata.evidence_source.id).url);assert.ok(entry.metadata.relation_types.includes('clinical_model'));assert.ok(entry.text.length<=1100);assert.ok(writing.details.reduce((sum,x)=>sum+x.text.length,0)<=6500);
 const references=new Map(CARTOGRAPHIE.sources.map(x=>[x.id,x]));const overhead=Math.max(...CARTOGRAPHIE.links.map(x=>{const ref=references.get(x.source_ids?.[0]);return JSON.stringify({evidence_summary:x.summary||'',evidence_source:ref?{id:ref.id,url:ref.url}:null}).length}));assert.ok(overhead<=600,'current primary-evidence metadata remains bounded to 600 characters per source');assert.ok(writing.details.length<=8);
});


test('joined refusal phrases preserve the no-question constraint without suppressing crisis safety',()=>{
 for(const text of ['Je voudrais seulement être écouté, sans conseil ni question pour le moment.','Sans conseils ni questions, écoutez-moi.','Je ne veux ni conseil ni question.']){const result=deterministicAnalysis(text);assert.ok(result.constraints.includes('no_questions'),text);assert.ok(result.constraints.includes('no_advice'),text);assert.equal(result.need,'listen',text);assert.ok(deterministicReview({content:'Que ressentez-vous ?'},result).includes(12));}
 assert.equal(deterministicAnalysis('Je veux mourir, sans conseil ni question.').constraints.includes('no_questions'),false);
});
test('every validated portrait pattern has its own stable labelled index entry',()=>{
 const hypotheses=[{pattern_id:'rumination',status:'exploring',observation:'Je repense aux critiques.',alternatives:['Un souci ponctuel.']},{pattern_id:'perfectionism',status:'mixed',observation:'Je vise un résultat parfait.',counter_evidence:['profile:a']}];
 const index=knowledgeIndex({people:[],events:[],sources:[],portrait:{psychological_map:hypotheses}}),entries=index.filter(x=>x.kind==='hypothesis');
 assert.deepEqual(entries.map(x=>x.key),['hypothesis:rumination','hypothesis:perfectionism']);assert.ok(entries.every(x=>x.label&&!x.label.includes('undefined')));assert.match(entries[0].label,/Pensées/);
 const selected=selectDetails(index,entries.map(x=>x.id),'Ces deux pistes');assert.equal(selected.length,2);assert.ok(selected[1].text.includes('counter_evidence'));assert.equal(selected[1].metadata.pattern_id,'perfectionism');
});

const flaggedAnalysis=()=>({...analysis(),risk:'possible',posture:'safety'});
const noCurrentRisk=()=>({risk:'none',category:'none',evidence_quote:''});

test('a diagnostic concern falsely flagged by analysis is independently checked without catalogue or historical danger',async t=>{
 const s=setup();await ready(s);seedTurns(s,2);s.raw.prepare("UPDATE chat_turns SET content='Je veux mourir.',response=? WHERE id='history-2'").run(JSON.stringify({companion_reply:1,content:'Pensez-vous au suicide maintenant ?'}));
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a',?,0)").run(JSON.stringify({safety_seen:true,safety:['Je veux mourir.']}));s.raw.prepare("UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id='a'").run(JSON.stringify({self_description:'Antécédents de dépression et de crises, cartographie et critiques.'}));
 const current='Le rapprochement avec les critiques de ma cartographie suffit-il à conclure que j’ai un trouble ?',calls=provider(t,p=>p==='ambre_analysis'?flaggedAnalysis():undefined),sent=await message(s,'risk_diagnostic',current);
 assert.equal(job(s).stage,'done');const a=JSON.parse(job(s).analysis);assert.equal(a.risk,'none');assert.notEqual(a.posture,'safety');assert.equal(a.risk_check.status,'rejected');assert.doesNotMatch(sent.messages.at(-1).content,/3114|112|danger immédiat/);
 assert.deepEqual(calls.map(x=>x.purpose),['ambre_analysis','ambre_risk_check','ambre_writing','ambre_review']);const check=calls[1].body;assert.deepEqual(JSON.parse(check.messages[1].content),{message:current});assert.equal(check.model,'openai/gpt-5-nano');assert.equal(check.reasoning.effort,'minimal');assert.equal(check.max_tokens,1500);
 const detail=(await s.call('/api/chat/insight?turn_id='+sent.request_id)).selected_turn;assert.equal(detail.stages[1].stage,'risk_check');assert.deepEqual(detail.stages[1].input,JSON.parse(check.messages[1].content));assert.equal(detail.stages[1].system,check.messages[0].content[0].text);assert.equal(detail.stages[1].status,'completed');assert.equal((await s.call('/api/chat/insight?turn_id='+sent.request_id,'GET',undefined,'b')).status,404);
});

test('the exact fictional live message and ordinary greetings do not inherit danger from an index',async t=>{
 for(const current of ['Toujours pour cet exemple fictif, le rapprochement avec les critiques de ma cartographie suffit-il à conclure que j’ai un trouble ?','Bonjour']){
  const s=setup();await ready(s);seedTurns(s,2);const calls=provider(t,p=>p==='ambre_analysis'?flaggedAnalysis():undefined);const sent=await message(s,'risk_fiction',current);assert.equal(job(s).stage,'done');assert.deepEqual(JSON.parse(calls[1].body.messages[1].content),{message:current});assert.equal(JSON.parse(job(s).analysis).risk,'none');assert.doesNotMatch(sent.messages.at(-1).content,/3114|112|suicide/);
 }
});

test('a current indirect warning survives independent verification and overrides refusal of questions',async t=>{
 const s=setup();await ready(s);const quote='Ce soir, tout sera terminé.',current='J’ai pris mes dispositions pour que chacun soit tranquille. '+quote+' Sans conseil ni question.';assert.equal(deterministicAnalysis(current).risk,'none');
 const calls=provider(t,p=>p==='ambre_analysis'?flaggedAnalysis():p==='ambre_risk_check'?{risk:'possible',category:'indirect_warning',evidence_quote:quote}:undefined),sent=await message(s,'risk_indirect',current);assert.equal(job(s).stage,'done');assert.match(sent.messages.at(-1).content,/3114/);const a=JSON.parse(job(s).analysis);assert.equal(a.risk,'possible');assert.equal(a.safety_signal,'self_harm');assert.equal(a.risk_check.status,'confirmed');assert.equal(a.constraints.includes('no_questions'),false);
 const writing=JSON.parse(calls.find(x=>x.purpose==='ambre_writing').body.messages[1].content);assert.equal(writing.analysis.risk_evidence,quote);assert.doesNotMatch(JSON.stringify((await s.call('/api/chat/insight')).turns),/Ce soir, tout sera terminé/);
});

test('deterministic current danger never waits for or depends on the extra model check',async t=>{
 for(const current of ['Je veux mourir, sans conseil ni question.','Ma famille serait mieux sans moi.','J’ai pris toute la boîte de comprimés.']){
  const s=setup();await ready(s);const calls=provider(t,p=>p==='ambre_risk_check'?new Response('Unavailable',{status:503}):undefined),sent=await message(s,'risk_backstop',current);assert.equal(job(s).stage,'done');assert.match(sent.messages.at(-1).content,/3114/);assert.equal(calls.some(x=>x.purpose==='ambre_risk_check'),false);assert.equal(JSON.parse(job(s).analysis).constraints.includes('no_questions'),false);
 }
});

test('only a current confirmation receives the immediately preceding safety exchange',async t=>{
 const s=setup();await ready(s);seedTurns(s,2);s.raw.prepare("UPDATE chat_turns SET content='Je veux mourir.',response=? WHERE id='history-2'").run(JSON.stringify({companion_reply:1,content:'Pensez-vous au suicide ce soir ?'}));const current='Oui, ce soir.',calls=provider(t,p=>p==='ambre_analysis'?flaggedAnalysis():p==='ambre_risk_check'?{risk:'possible',category:'current_confirmation',evidence_quote:current}:undefined);await message(s,'risk_confirmation',current);assert.equal(job(s).stage,'done');assert.deepEqual(JSON.parse(calls[1].body.messages[1].content),{message:current,recent_confirmation_context:{id:'history-2',at:JSON.parse(calls[1].body.messages[1].content).recent_confirmation_context.at,user:'Je veux mourir.',assistant:'Pensez-vous au suicide ce soir ?'}});assert.equal(JSON.parse(job(s).analysis).risk,'possible');
});

test('a short execution window persists the pending risk check and never repeats initial analysis',async t=>{
 const s=setup();await ready(s);const realNow=Date.now,base=realNow();let elapsed=0;Date.now=()=>base+elapsed;t.after(()=>{Date.now=realNow});const calls=provider(t,p=>{if(p==='ambre_analysis'){elapsed=13000;return flaggedAnalysis()}});await message(s,'risk_window','Une question de diagnostic.');assert.equal(job(s).stage,'analysis');assert.equal(job(s).attempts,0);assert.equal(JSON.parse(job(s).analysis).risk_check.status,'pending');assert.equal(JSON.parse(job(s).analysis).risk,'none');assert.equal(calls.length,1);
 await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');assert.deepEqual(calls.map(x=>x.purpose),['ambre_analysis','ambre_risk_check','ambre_writing','ambre_review']);
});

test('a transient risk verification failure resumes the same substep under an exclusive lease',async t=>{
 const s=setup();await ready(s);let checks=0;const calls=provider(t,p=>{if(p==='ambre_analysis')return flaggedAnalysis();if(p==='ambre_risk_check'&&++checks===1)throw Object.assign(Error('PRIVATE_PROVIDER_SECRET'),{name:'TimeoutError'})}),sent=await message(s,'risk_retry','Une question de diagnostic.');assert.equal(job(s).stage,'analysis');assert.equal(job(s).attempts,1);assert.equal(JSON.parse(job(s).analysis).risk,'none');assert.equal(s.raw.prepare('SELECT response FROM chat_turns').get().response,'');
 await Promise.all([s.chat.engine.work(s.env,sent.request_id),s.chat.engine.work(s.env,sent.request_id)]);assert.equal(job(s).stage,'done');assert.equal(calls.filter(x=>x.purpose==='ambre_analysis').length,1);assert.equal(checks,2);const detail=(await s.call('/api/chat/insight?turn_id='+sent.request_id)).selected_turn;assert.equal(detail.stages[1].error_code,'provider_timeout');assert.equal(detail.stages[2].stage,'risk_check');assert.doesNotMatch(JSON.stringify(detail),/PRIVATE_PROVIDER_SECRET/);
});

// Lot S (F18): an unavailable check never ends in a bare failure; the fixed cautious
// reply names the resources without asserting a crisis. A published reply counts
// as one message (the chat_turns CHECK ties completed to consumed).
test('three unavailable risk checks publish a cautious reply with resources, never an invented crisis',async t=>{
 const s=setup();await ready(s);const calls=provider(t,p=>p==='ambre_analysis'?flaggedAnalysis():p==='ambre_risk_check'?new Response('PRIVATE_PROVIDER_SECRET',{status:502}):undefined),sent=await message(s,'risk_unavailable','Une question de diagnostic.');await s.chat.resumeEngine(s.env);await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');const turn=s.raw.prepare('SELECT * FROM chat_turns').get();assert.equal(turn.credit_status,'consumed');const reply=JSON.parse(turn.response).content;assert.match(reply,/3114/);assert.match(reply,/n’ai pas pu analyser/);assert.doesNotMatch(reply,/pensez au suicide ou risquez/);assert.equal(calls.filter(x=>x.purpose==='ambre_risk_check').length,3);assert.equal(calls.filter(x=>x.purpose==='ambre_analysis').length,1);assert.equal(calls.some(x=>x.purpose==='ambre_writing'),false);
 const detail=(await s.call('/api/chat/insight?turn_id='+sent.request_id)).selected_turn;assert.equal(detail.analysis.risk_check.status,'unavailable');assert.equal(detail.stages.filter(x=>x.stage==='risk_check').length,3);assert.doesNotMatch(JSON.stringify(detail),/PRIVATE_PROVIDER_SECRET/);await s.chat.resumeEngine(s.env);assert.equal(calls.length,4);
});

test('risk checks reject malformed, contradictory or unanchored verdicts without quoting private provider text',async t=>{
 for(const [verdict,code] of [[{risk:'none',category:'self_harm',evidence_quote:''},'risk_check_invalid_response'],[{risk:'none',category:'none',evidence_quote:'PRIVATE_PROVIDER_SECRET'},'risk_check_invalid_response'],[{risk:'possible',category:'self_harm',evidence_quote:'A quote from old history'},'risk_check_unanchored'],['PRIVATE_PROVIDER_SECRET invalid JSON','risk_check_invalid_response']]){
  const s=setup();await ready(s);let checks=0;provider(t,p=>p==='ambre_analysis'?flaggedAnalysis():p==='ambre_risk_check'?(++checks===1?verdict:noCurrentRisk()):undefined);const sent=await message(s,'risk_invalid','Une question de diagnostic.');assert.equal(job(s).stage,'analysis');assert.equal(job(s).attempts,1);assert.equal(s.raw.prepare('SELECT response FROM chat_turns').get().response,'');const detail=(await s.call('/api/chat/insight?turn_id='+sent.request_id)).selected_turn;assert.equal(detail.stages[1].status,'error');assert.equal(detail.stages[1].error_code,code);assert.doesNotMatch(JSON.stringify(detail),/PRIVATE_PROVIDER_SECRET|A quote from old history/);await s.chat.resumeEngine(s.env);assert.equal(job(s).stage,'done');
 }
});

// Lot S (F18): the exhausted turn budget buys no extra call; the possible danger gets the fixed cautious reply.
test('the conditional verification obeys the existing total turn budget',async t=>{
 const s=setup();await ready(s);const calls=provider(t,p=>{if(p==='ambre_analysis'){s.raw.prepare('UPDATE chat_engine_jobs SET reserved_tokens=29000').run();return flaggedAnalysis()}});const sent=await message(s,'risk_budget_extra','Une question de diagnostic.');assert.equal(sent.last_error,null);assert.match(sent.messages.at(-1).content,/3114/);assert.equal(job(s).stage,'done');assert.equal(JSON.parse(job(s).analysis).risk_check.error_code,'turn_budget');assert.equal(job(s).reserved_tokens,29000);assert.equal(calls.length,1);assert.equal(s.raw.prepare('SELECT credit_status FROM chat_turns').get().credit_status,'consumed');
});

test('confirmed violence selects violence resources in the published answer and fallback',async t=>{
 for(const fallback of [false,true]){
  const s=setup();await ready(s);const current='Il bloque la porte, brandit son arme et me dit que je ne sortirai pas vivante.';assert.equal(deterministicAnalysis(current).risk,'none');provider(t,p=>p==='ambre_analysis'?flaggedAnalysis():p==='ambre_risk_check'?{risk:'explicit',category:'violence',evidence_quote:'brandit son arme'}:p==='ambre_review'&&fallback?response('invalid JSON'):undefined);const sent=await message(s,'risk_violence',current);assert.equal(job(s).stage,'done');assert.equal(JSON.parse(job(s).analysis).safety_signal,'violence');assert.match(sent.messages.at(-1).content,/17 ou le 112/);assert.doesNotMatch(sent.messages.at(-1).content,/3114|suicide/);
 }
});

test('consent, source changes and purge invalidate late risk-check results',async t=>{
 for(const change of ['consent','source','purge']){
  const s=setup();await ready(s);let release,started;const began=new Promise(r=>started=r),tasks=[];provider(t,async p=>{if(p==='ambre_analysis')return flaggedAnalysis();if(p==='ambre_risk_check'){started();await new Promise(r=>release=r);return noCurrentRisk()}});await message(s,'risk_late_'+change,'Une question de diagnostic.',{waitUntil:p=>tasks.push(p)});await began;
  if(change==='consent')await s.call('/api/chat/consent','PUT',{accepted:false});if(change==='source')s.raw.prepare("UPDATE profiles SET version=version+1 WHERE user_id='a'").run();if(change==='purge')await s.call('/api/chat','DELETE',{});release();await Promise.all(tasks);assert.notEqual(s.raw.prepare('SELECT status FROM chat_turns').get().status,'completed');if(change==='purge')assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_turn_traces').get().n,0);
 }
});

function filledProfile(s,maximal=false){
 s.raw.exec("DELETE FROM personal_people WHERE id LIKE 'ready-person-a-%';DELETE FROM life_events WHERE id LIKE 'ready-event-a-%';");
 const prose='Je rumine après les critiques et cherche des limites dans mes relations. ',long=n=>prose.repeat(Math.ceil(n/prose.length)).slice(0,n);
 s.raw.prepare("UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id='a'").run(JSON.stringify({first_name:'Camille',occupation:'Activité déclarée',self_description:long(maximal?10000:2400),self_perception:long(maximal?4000:800),recurring_patterns:long(maximal?4000:800),emotions:long(maximal?4000:800),current_context:long(maximal?4000:800)}));
 for(let i=0;i<(maximal?50:6);i++)s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES(?,'a',?,'hash',?,'friend',?)").run('filled-person-'+i,'filled-person-'+i,'Proche critiques '+i,long(maximal?6000:1300));
 for(let i=0;i<(maximal?500:20);i++)s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,description,meaning_now,kind) VALUES(?,'a',?,'hash',?,?,?,'other')").run('filled-event-'+i,'filled-event-'+i,'Moment critiques '+i,long(maximal?6000:1300),long(maximal?2000:400));
 const answers=Object.fromEntries(CARTOGRAPHIE.factors.map((x,i)=>[x.id,i%4===0?0:1]));
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a',?,'{}','{}','{}','[]',1,1,1,1) ON CONFLICT(user_id) DO UPDATE SET answers=json_patch(cartographies.answers,excluded.answers),feedback=excluded.feedback,notes=excluded.notes,resources=excluded.resources,ambre_notes=excluded.ambre_notes,consent=excluded.consent,locked_at=excluded.locked_at,version=excluded.version,updated_at=excluded.updated_at").run(JSON.stringify(answers));
 seedTurns(s,6);s.raw.prepare("UPDATE chat_turns SET content=?,response=?,week_start='2026-01-01',created_at='2026-01-01 00:00:00'").run(long(maximal?6000:700),JSON.stringify({companion_reply:1,content:long(1100)+' Quelle limite voudriez-vous poser ?'}));
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,through_turn,turns_covered,updated_at) VALUES('a',?,'history-6',6,0)").run(JSON.stringify(Object.fromEntries(['facts','goals','hypotheses','tried','alliance','session',...(maximal?['closed','safety']:[])].map(k=>[k,Array.from({length:k==='closed'?30:12},(_,i)=>`${k} ${i} `+long(230))]))));
 s.raw.prepare("UPDATE ambre_reviews SET content=? WHERE user_id='a'").run(long(3000));
 return long(maximal?6000:250);
}

test('audit17 filled SQLite profiles complete ordinary turns and one rewrite within the planned budget',async t=>{
 for(const maximal of [false,true])for(const rewrite of [false,true]){
  const s=setup();await ready(s);const current=filledProfile(s,maximal);let reviews=0;
  const calls=provider(t,(purpose,body)=>{if(purpose==='ambre_analysis'){const input=JSON.parse(body.messages[1].content);return {...analysis(),selected_ids:input.index.split('\n').slice(0,8).map(row=>row.split(' ')[0])}}if(purpose==='ambre_review'&&rewrite&&!reviews++){const r=review();r.criteria[3]='no';return r}});
  const sent=await message(s,'filled_'+maximal+'_'+rewrite,current);for(let n=0;n<8&&!['done','failed'].includes(job(s).stage);n++)await s.chat.engine.work(s.env,sent.request_id);
  const sizes=calls.map(({purpose,body})=>({purpose,input:Math.ceil((body.messages[0].content[0].text.length+body.messages[1].content.length+JSON.stringify(body.response_format.json_schema.schema).length)/3),reserved:Math.ceil((body.messages[0].content[0].text.length+body.messages[1].content.length+JSON.stringify(body.response_format.json_schema.schema).length)/3)+body.max_tokens}));
  t.diagnostic(JSON.stringify({maximal,rewrite,sizes,total_input:sizes.reduce((n,x)=>n+x.input,0),reserved:job(s).reserved_tokens}));
  assert.equal(job(s).stage,'done',JSON.stringify({job:job(s),trace:JSON.parse(s.raw.prepare('SELECT trace FROM chat_turn_traces').get().trace).error}));assert.equal(JSON.parse(job(s).review).passed,true);assert.equal(calls.filter(x=>x.purpose==='ambre_writing').length,rewrite?2:1);assert.ok(job(s).reserved_tokens<=30000);
 }
});

test('audit17 recovery reserves every failed attempt, keeps analysis once, and completes the maximal rewrite',async t=>{
 const s=setup();await ready(s);const current=filledProfile(s,true);let writingPhase=0,writeAttempts=0,reviewPhase=0,reviewAttempts=0;
 const calls=provider(t,(purpose,body)=>{
  if(purpose==='ambre_analysis')return {...analysis(),selected_ids:JSON.parse(body.messages[1].content).index.split('\n').slice(0,8).map(x=>x.split(' ')[0])};
  if(purpose==='ambre_writing'){if(++writeAttempts%3!==0)return new Response('Unavailable',{status:502});writingPhase++;return draft()}
  if(purpose==='ambre_review'){if(++reviewAttempts%3!==0)return Response.json({choices:[{message:{content:'{}'},finish_reason:'length'}]});reviewPhase++;const r=review();if(reviewPhase===1)r.criteria[3]='no';return r}
 });
 const sent=await message(s,'maximal_recovery',current);for(let n=0;n<12&&!['done','failed'].includes(job(s).stage);n++)await s.chat.engine.work(s.env,sent.request_id);
 assert.equal(job(s).stage,'done',JSON.stringify({job:job(s),trace:JSON.parse(s.raw.prepare('SELECT trace FROM chat_turn_traces').get().trace).error}));assert.equal(JSON.parse(job(s).review).passed,true);assert.equal(writingPhase,2);assert.equal(reviewPhase,2);assert.equal(calls.filter(c=>c.purpose==='ambre_analysis').length,1);assert.equal(calls.length,13);
 const total=calls.reduce((n,{body})=>n+Math.ceil((body.messages[0].content[0].text.length+body.messages[1].content.length+JSON.stringify(body.response_format.json_schema.schema).length)/3)+body.max_tokens,0);
 assert.equal(job(s).reserved_tokens,total);assert.equal(s.raw.prepare('SELECT reserved_tokens FROM provider_budgets').get().reserved_tokens,total);assert.ok(total<94500);assert.ok(total>30000);
 t.diagnostic(JSON.stringify({scenario:'maximal_rewrite_all_retries',calls:calls.length,reserved:total,stages:calls.map(({purpose,body})=>({purpose,reserved:Math.ceil((body.messages[0].content[0].text.length+body.messages[1].content.length+JSON.stringify(body.response_format.json_schema.schema).length)/3)+body.max_tokens}))}));
});

test('audit17 relevant excerpts, complete index labels and more than 1100 sources stay bounded',()=>{
 const context={people:[],events:[],sources:Array.from({length:560},(_,i)=>({id:'source:'+i,type:'memory',text:'Détail ancien '+i})),cartography:{explicit_negative:[{id:'long',label:'Parent incohérent et attentes contradictoires clairement refusés'}]}};
 context.people=Array.from({length:50},(_,i)=>({id:'p'+i,display_name:'Personne '+i,role:'friend',relationship_text:'Une relation'}));context.events=Array.from({length:500},(_,i)=>({id:'e'+i,title:'Moment '+i,description:'Un moment'}));
 const index=knowledgeIndex(context);assert.ok(index.length>1100);const shortlist=analysisIndex(index,'Parent incohérent et attentes contradictoires');assert.ok(shortlist.text.length<=2600);assert.ok(shortlist.omitted>1000);assert.match(shortlist.text,/Parent incohérent et attentes contradictoires clairement refusés/);assert.deepEqual(selectDetails(index,[],'Bonjour'),[]);assert.ok(selectDetails(index,[],'attentes contradictoires').some(x=>x.key==='cartography:negative:long'));
 const mapped=knowledgeIndex({cartography:{convergence:[{id:'piste',label:'Une piste',url:'/personnages/lecture/'}]}});assert.doesNotMatch(JSON.stringify(mapped),/personnages/);
});

test('audit17 ordinary shame, habit language and acknowledgement do not impose a crisis or blocked review',async t=>{
 for(const text of ['Je veux faire disparaître cette habitude.','Mon travail est un fardeau.','Ce n’est pas moi qui décide.','J’ai honte de ne pas avoir fait l’exercice.','Nous sommes ravis.']){
  const result=deterministicAnalysis(text);assert.equal(result.risk,'none',text);assert.notEqual(result.posture,'stabilize',text);assert.equal(result.constraints.includes('no_questions'),false,text);
 }
 assert.equal(deterministicAnalysis('Nous sommes ravis.').need,'understand');
 for(const content of ['J’ai noté que vous avez besoin de temps.','C’est noté.','Contactez votre médecin dans les 48 heures.'])assert.deepEqual(deterministicReview({content},analysis()),[],content);
 assert.ok(deterministicReview({content:'J’ai enregistré cela dans votre profil.'},analysis()).includes(4));assert.ok(deterministicReview({content:'Appelez le 9999.'},analysis()).includes(2));
 const s=setup();await ready(s);const calls=provider(t);const sent=await message(s,'ordinary_habit','Je veux faire disparaître cette habitude.');assert.equal(job(s).stage,'done');assert.doesNotMatch(sent.messages.at(-1).content,/3114/);assert.equal(calls.length,3);
});

test('audit17 recent memory, final question, Paris dates and reviewer care remain available',async t=>{
 const summary={goals:Array.from({length:12},(_,i)=>'Objectif '+i+' '+('x'.repeat(210)))};assert.match(memoryPrompt(summary).goals[0],/^Objectif 11/);assert.doesNotMatch(JSON.stringify(memoryPrompt(summary)),/Objectif 0 /);
 assert.equal(parisTime(Date.parse('2026-11-16T23:30:00Z')/1000),'2026-11-17 00:30');assert.equal(parisTime(Date.parse('2026-06-16T22:30:00Z')/1000),'2026-06-17 00:30');
 const s=setup();await ready(s);seedTurns(s,1);s.setClock(Date.parse('2026-11-16T23:30:00Z')/1000);s.raw.prepare('UPDATE subscriptions SET paid_until=?').run(s.getClock()+86400);s.raw.prepare("INSERT INTO care_settings(user_id,has_psychologist,frequency_days,updated_at) VALUES('a',1,7,0)").run();s.raw.prepare("INSERT INTO care_sessions(id,user_id,session_at,created_by,created_at) VALUES('next','a',?,'a',0)").run(s.getClock()+3600);
 s.raw.prepare('UPDATE chat_turns SET response=?').run(JSON.stringify({companion_reply:1,content:'Une longue réponse. '.repeat(80)+' Quelle limite compte pour vous ?'}));
 const calls=provider(t,p=>p==='ambre_writing'?{...draft(),questions:['Quelle option préférez-vous ?']}:undefined);await message(s,'dated_question','La limite avec ma famille.');assert.equal(job(s).stage,'done');
 const writer=JSON.parse(calls.find(c=>c.purpose==='ambre_writing').body.messages[1].content),critic=JSON.parse(calls.find(c=>c.purpose==='ambre_review').body.messages[1].content);
 assert.match(writer.history.at(-1).assistant,/Quelle limite compte pour vous \?$/);assert.equal(writer.current_time,'2026-11-17 00:30');assert.equal(writer.care.next_session_local,'2026-11-17 01:30');assert.deepEqual(critic.questions,['Quelle option préférez-vous ?']);assert.equal(critic.care.has_psychologist,true);
});

test('audit17 summarizer failure keeps coverage and summary intact, exposes retry and recovers later',async t=>{
 const s=setup();await ready(s);seedTurns(s);let failure=true;provider(t,p=>p==='ambre_memory'?(failure?new Response('Unavailable',{status:502}):{items:[{section:'facts',turn_id:'history-4',quote:'Fait rapporté 4'}]}):undefined);
 // Lot S: a 503 is a provider refusal (lot R) and no longer a failed attempt; 502 is a consumed failure.
 await s.chat.engine.updateMemory(s.env,'a');assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_memory').get().n,0);assert.equal(s.raw.prepare('SELECT attempts FROM chat_memory_jobs').get().attempts,1);assert.equal((await s.call('/api/chat/insight')).memory.status,'retry_pending');
 failure=false;s.setClock(s.getClock()+61);await s.chat.engine.updateMemory(s.env,'a');const memory=s.raw.prepare('SELECT * FROM chat_memory').get();assert.equal(memory.turns_covered,4);assert.deepEqual(JSON.parse(memory.summary).facts,['Fait rapporté 4']);
 s.raw.prepare('UPDATE chat_memory SET summary=?').run(JSON.stringify({facts:['Un fait'],safety:[],safety_seen:true}));await s.call('/api/chat/memory','PUT',{revision:memory.revision,sections:{facts:['Fait corrigé']}});assert.equal(JSON.parse(s.raw.prepare('SELECT summary FROM chat_memory').get().summary).safety_seen,true);
 await s.call('/api/chat/memory','PUT',{revision:memory.revision+1,sections:{}});assert.equal(JSON.parse(s.raw.prepare('SELECT summary FROM chat_memory').get().summary).safety_seen,undefined);
});

test('audit17 bilans and authorized relationship candidates reach the writer and saved observations are recalled',async t=>{
 const s=setup();await ready(s);const quote='Alice me coupe la parole.';s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES('alice','a','alice','hash','Alice','friend','Une proche.')").run();s.raw.prepare("UPDATE ambre_reviews SET content='Le bilan évoque les échanges avec Alice.' WHERE user_id='a'").run();
 const calls=provider(t,p=>p==='ambre_writing'?{...draft(),people_observations:[{person_id:'alice',axis:'communication',quote}]}:undefined);await message(s,'observation_one',quote);assert.equal(job(s).stage,'done');assert.equal(s.raw.prepare('SELECT count(*) n FROM people_observations').get().n,0);const proposal=(await s.call('/api/chat')).proposals.find(p=>p.kind==='person_observation');assert.ok(proposal);assert.equal((await s.call('/api/chat/proposals/'+proposal.id+'/accept','POST',{revision:proposal.revision})).status,200);assert.equal(s.raw.prepare('SELECT count(*) n FROM people_observations').get().n,1);
 const input=JSON.parse(calls.find(c=>c.purpose==='ambre_writing').body.messages[1].content);assert.deepEqual(input.observation_candidates,{person_id:'alice',quotes:[quote]});assert.match(input.review_thread.last.text,/Alice/);assert.ok(!input.details.some(x=>x.key===`review:${input.review_thread.last.id}`),'the last review is not sent twice (A5-P1: inside the review thread)');
 await message(s,'observation_two','Je repense à Alice.');const input2=JSON.parse(calls.filter(c=>c.purpose==='ambre_writing').at(-1).body.messages[1].content);assert.equal(input2.reported_relationship_observations.observations[0].quote,quote);
});

test('audit17 revoked observation revision blocks a late engine answer',async t=>{
 const s=setup();await ready(s);provider(t,p=>{if(p==='ambre_review')s.raw.prepare("UPDATE chat_settings SET people_observations_revision=people_observations_revision+1 WHERE user_id='a'").run()});await message(s,'observation_revision','Bonjour');assert.equal(job(s).stage,'failed');assert.equal(s.raw.prepare('SELECT error_code FROM chat_turns').get().error_code,'context_changed');
});

test('audit17 maximal safety verification and all recovery phases fit without replaying analysis',async t=>{
 const s=setup();await ready(s);const quote='Ce soir, tout sera terminé.',current=filledProfile(s,true).slice(0,5972)+quote;let risks=0,writes=0,reviews=0;
 const calls=provider(t,p=>{
  if(p==='ambre_analysis')return {...analysis(),risk:'possible',posture:'safety'};
  if(p==='ambre_risk_check'){if(++risks<3)return new Response('Unavailable',{status:502});return {risk:'possible',category:'indirect_warning',evidence_quote:quote}}
  if(p==='ambre_writing'){if(++writes%3)return new Response('Unavailable',{status:502});return draft()}
  if(p==='ambre_review'){if(++reviews%3)return Response.json({choices:[{message:{content:'{}'},finish_reason:'length'}]});const r=review();if(reviews===3)r.criteria[3]='no';return r}
 });
 const sent=await message(s,'maximal_safety_recovery',current);for(let n=0;n<16&&!['done','failed'].includes(job(s).stage);n++)await s.chat.engine.work(s.env,sent.request_id);
 assert.equal(job(s).stage,'done',JSON.stringify({job:job(s),trace:JSON.parse(s.raw.prepare('SELECT trace FROM chat_turn_traces').get().trace).error}));assert.equal(JSON.parse(job(s).review).passed,true);assert.equal(calls.filter(c=>c.purpose==='ambre_analysis').length,1);assert.equal(calls.length,16);assert.ok(job(s).reserved_tokens<=94500);assert.match(s.raw.prepare('SELECT response FROM chat_turns WHERE id=?').get(sent.request_id).response,/3114/);
 t.diagnostic(JSON.stringify({scenario:'maximal_safety_rewrite_all_retries',calls:calls.length,reserved:job(s).reserved_tokens}));
});


test('a long primary nuance remains whole in the source serializer without asserting a catalogue association',()=>{
 const summary="Une formulation de test suffisamment longue pour vérifier la transmission complète d’une précaution importante. ".repeat(3)+"Ce n'est pas la foi elle-même qui est décrite, mais l'emprise et la contrainte exercées dans un contexte particulier.";
 const index=knowledgeIndex({sources:[],people:[],events:[],cartography:{convergence:[{id:'synthetic_only',label:'Fixture technique',relation_types:['clinical_model']}],evidence:[{id:'synthetic_only',links:[{relation_type:'clinical_model',source_ids:['fixture'],summary}]}],evidence_sources:[{id:'fixture',url:'https://example.org/fixture'}]}}),entry=selectDetails(index,[index[0].id],'Fixture')[0];
 assert.equal(entry.metadata.evidence_summary,summary);assert.match(entry.metadata.evidence_summary,/foi elle-même.*emprise/);
});

test('hidden contextual subjects never reach the index, even with an older answer',()=>{
 // v5 final review: contextual subjects leave the index; an older answer on one stays stored but is not sent.
 assert.equal(knowledgeIndex({sources:[],people:[],events:[],cartography:{contextual:[{id:'adult_context',label:'Repère somatique',contextual_only:true,age_scope:'adult'}]}}).length,0);
 const index=knowledgeIndex({sources:[],people:[],events:[],cartography:{contextual:[],recognized:[{id:'adult_context',label:'Repère somatique',contextual_only:true,age_scope:'adult'}],dismissed:[{id:'child_context',label:'Attachement du jeune enfant',contextual_only:true,age_scope:'young_child_only'}]}});
 assert.equal(index.some(x=>/adult_context|child_context/.test(x.key)),false);
});
