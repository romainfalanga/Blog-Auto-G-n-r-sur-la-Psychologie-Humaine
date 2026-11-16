// Lot R, F10/F12/F21: shared provider guard, refunds, cooldowns and daily cap.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,refused} from './v4-R-helper.mjs';

const say=(s,u,key,content='J’ai besoin de parler de ma journée, elle a été lourde.',ctx)=>s.call('/api/chat/messages','POST',{content,idempotency_key:key},u,ctx);

test('F10: a 429 with Retry-After defers every turn, refunds every reservation and is called at most once per model',async t=>{
 const users=['p0','p1','p2','p3','p4'],s=setup({users});for(const u of users)await s.ready(u);
 let refusing=true;const calls=provider(t,()=>refusing?refused(429,'60'):undefined);
 for(const u of users)await say(s,u,'storm_'+u);
 // Clients keep polling for 60 virtual seconds.
 for(let tick=0;tick<40;tick++){s.tick(1);for(const u of users)await s.call('/api/chat',undefined,undefined,u,{waitUntil:p=>p})}
 const byModel=Object.groupBy(calls,c=>c.model);
 for(const [model,list] of Object.entries(byModel))assert.ok(list.length<=1,`${model}: ${list.length} calls during the cooldown`);
 assert.deepEqual(s.raw.prepare("SELECT status,count(*) n FROM chat_turns WHERE idempotency_key LIKE 'storm_%' GROUP BY status").all().map(r=>({...r})),[{status:'running',n:5}]);
 assert.equal(s.raw.prepare('SELECT coalesce(sum(reserved_tokens),0) n FROM provider_budgets').get().n,0,'no token debited for refused calls');
 assert.equal(s.raw.prepare('SELECT coalesce(sum(reserved_tokens),0) n FROM chat_engine_jobs').get().n,0,'job reservations refunded');
 assert.equal(s.raw.prepare('SELECT coalesce(max(attempts),0) n FROM chat_engine_jobs').get().n,0,'a refusal is not an attempt');
 // The provider recovers: the same turns are published without being resent.
 refusing=false;s.tick(61);const pending=[];
 for(const u of users)await s.call('/api/chat',undefined,undefined,u,{waitUntil:p=>pending.push(p)});
 await Promise.all(pending);
 assert.equal(s.raw.prepare("SELECT count(*) n FROM chat_turns WHERE status='completed' AND idempotency_key LIKE 'storm_%'").get().n,5);
 const trace=JSON.parse(s.raw.prepare("SELECT trace FROM chat_turn_traces WHERE turn_id=(SELECT id FROM chat_turns WHERE user_id='p0')").get().trace);
 const refusedStage=trace.details.stages.find(stage=>stage.status==='refused');
 assert.ok(refusedStage,'the refused call stays visible in the private trace');assert.equal(refusedStage.error_code,'provider_http_429');assert.equal(refusedStage.reservation,0);
});

test('F10: a key-wide budget refusal blocks every model and surfaces the dedicated 429 message',async t=>{
 const s=setup();await s.ready('a');await s.ready('b');
 const calls=provider(t,()=>refused(429,'30',{limit_source:'openrouter_key_limit'}));
 await say(s,'a','key_limit_1');
 assert.equal(calls.length,1);
 const view=await s.call('/api/chat',undefined,undefined,'b');
 assert.equal(view.access.reason,'provider_cooldown');assert.ok(view.access.retry_after_seconds>0&&view.access.retry_after_seconds<=30);
 // The key cooldown lasts longer than a turn can wait: the turn ends with the typed code, uncounted.
 s.tick(31);const longCalls=provider(t,()=>refused(429,'3600',{limit_source:'openrouter_key_limit'}));
 await s.chat.engine.work(s.env,s.raw.prepare("SELECT id FROM chat_turns WHERE user_id='a'").get().id);
 assert.equal(longCalls.length,1);
 const after=await s.call('/api/chat');
 assert.equal(after.last_request.status,'failed');assert.equal(after.last_error.code,'provider_http_429_key_budget');
 assert.match(after.last_error.message,/budget fournisseur/);assert.match(after.last_error.message,/Aucun message n’a été décompté/);
 assert.equal(after.quota.used,0);
});

test('F10: a refusal during review keeps the checked draft and never publishes the cautious fallback',async t=>{
 const s=setup();await s.ready('a');let reviewRefused=true;
 provider(t,purpose=>purpose==='ambre_review'&&reviewRefused?refused(503,'20'):undefined);
 await say(s,'a','review_503');
 const job=s.raw.prepare('SELECT stage,attempts FROM chat_engine_jobs').get();
 assert.equal(job.stage,'review');assert.equal(job.attempts,0);
 reviewRefused=false;s.tick(21);
 await s.chat.engine.work(s.env,s.raw.prepare('SELECT turn_id FROM chat_engine_jobs').get().turn_id);
 const turn=s.raw.prepare('SELECT status,response FROM chat_turns').get();
 assert.equal(turn.status,'completed');assert.match(JSON.parse(turn.response).content,/rumination/);
});

test('F10: written reviews share the guard and refund a refused call',async t=>{
 const s=setup();await s.ready('a');
 s.raw.prepare("INSERT INTO provider_state(scope,retry_at,reason,updated_at) VALUES('key',?,'provider_http_429_key_budget',?)").run(s.now()+120,s.now());
 const calls=provider(t);
 const summary=await s.chat.resumeReviews(s.env);
 assert.equal(calls.length,0);assert.equal(summary.claimed,0);
 // A review that is refused after its reservation gets its tokens back.
 s.raw.prepare('DELETE FROM provider_state').run();
 s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,idempotency_key,attempts,lease_token,lease_until,created_at) VALUES('r1','a','point','running','rk-1',0,NULL,0,?)").run(s.now());
 s.raw.prepare("UPDATE ambre_reviews SET status='ready' WHERE id='seed-a'").run();
 provider(t,()=>refused(429,'45'));
 s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES('pp','a','pp','h','Proche','friend','Relation décrite en détail.')").run();
 await s.chat.resumeReviews(s.env);
 assert.equal(s.raw.prepare('SELECT coalesce(sum(reserved_tokens),0) n FROM provider_budgets').get().n,0);
 const review=s.raw.prepare("SELECT status,error_code,lease_until FROM ambre_reviews WHERE id='r1'").get();
 assert.equal(review.status,'running');assert.equal(review.error_code,'provider_http_429');assert.ok(review.lease_until>=s.now()+45);
 assert.ok(s.raw.prepare("SELECT retry_at FROM provider_state WHERE scope LIKE 'model:%'").get().retry_at>=s.now()+45);
});

test('F12: beyond the global daily cap no provider call is made, the turn ends with a clear uncounted message',async t=>{
 const s=setup({env:{DAILY_PROVIDER_TOKEN_CAP:'20000'}});await s.ready('a');await s.ready('b');
 const calls=provider(t);
 const first=await say(s,'a','cap_first');assert.equal(first.status,200);
 const used=s.raw.prepare('SELECT reserved_tokens FROM provider_global_budget').get().reserved_tokens;assert.ok(used>0);
 // The next turn exceeds the remaining daily allowance during its stages.
 s.raw.prepare('UPDATE provider_global_budget SET reserved_tokens=?').run(20000-1000);
 const before=calls.length;await say(s,'b','cap_second','Je veux mourir ce soir, je n’en peux plus.');
 const turn=s.raw.prepare("SELECT status,error_code,credit_status FROM chat_turns WHERE idempotency_key='cap_second'").get();
 assert.equal(turn.status,'failed');assert.equal(turn.error_code,'daily_cap');assert.equal(turn.credit_status,'released');
 const view=await s.call('/api/chat',undefined,undefined,'b');
 assert.match(view.last_error.message,/limite d’utilisation pour aujourd’hui/);assert.match(view.last_error.message,/3114/);
 assert.equal(calls.length,before,'no provider call beyond the cap');
 // Background work stops at 80 % of the cap so the conversation keeps priority.
 s.raw.prepare('UPDATE provider_global_budget SET reserved_tokens=?').run(16500);
 assert.equal((await s.call('/api/chat',undefined,undefined,'a')).access.reason,null);
 s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response) SELECT 'm'||value,'a','m'||value,'h','completed','consumed','2025-01-06',1,1,'Fait '||value,'{}' FROM json_each('[1,2,3,4,5]')").run();
 s.raw.prepare("INSERT OR REPLACE INTO chat_memory_jobs(user_id,updated_at) VALUES('a',0)").run();
 const memoryCalls=calls.length;await s.chat.engine.updateMemory(s.env,'a');assert.equal(calls.length,memoryCalls,'memory waits below the conversation reserve');
 s.raw.prepare('UPDATE provider_global_budget SET reserved_tokens=20000').run();
 assert.equal((await s.call('/api/chat',undefined,undefined,'a')).access.reason,'daily_cap');
});

test('F21: a cooling analysis model switches small calls to the configured fallback model',async t=>{
 const s=setup({env:{OPENROUTER_ANALYSIS_FALLBACK_MODEL:'mistralai/small-fallback'}});await s.ready('a');
 let primaryDown=true;const calls=provider(t,(purpose,body)=>body.model==='openai/gpt-5-nano'&&primaryDown?new Response('down',{status:500}):undefined);
 const r=await say(s,'a','fallback_1');assert.equal(r.status,200);
 const analysisCalls=calls.filter(c=>c.purpose==='ambre_analysis').map(c=>c.model),reviewCalls=calls.filter(c=>c.purpose==='ambre_review').map(c=>c.model);
 assert.deepEqual(analysisCalls,['openai/gpt-5-nano']);
 assert.deepEqual(reviewCalls,['mistralai/small-fallback'],'the next small call uses the fallback while the primary cools down');
 assert.ok(calls.filter(c=>c.purpose==='ambre_writing').every(c=>c.model!=='mistralai/small-fallback'),'the writer never uses the small fallback');
 s.tick(61);primaryDown=false;await say(s,'a','fallback_2','Une autre journée.');
 assert.equal(calls.filter(c=>c.purpose==='ambre_analysis').at(-1).model,'openai/gpt-5-nano');
});
