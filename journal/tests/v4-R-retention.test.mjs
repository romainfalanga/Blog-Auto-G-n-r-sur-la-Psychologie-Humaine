// Lot R, F1/F2: trace retention, bounded purge of finished jobs, no full scans.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,filledProfile,seedTurns} from './v4-R-helper.mjs';
import {traceSummary,COMPACT_TRACE_SQL} from '../src/ambre-insight.mjs';

const realTrace=async t=>{const s=setup();filledProfile(s);await s.ready();provider(t);const r=await s.call('/api/chat/messages','POST',{content:'Je rumine quand ma sœur me critique et je voudrais comprendre pourquoi.',idempotency_key:'real_turn_0001'});assert.equal(r.status,200);return s.raw.prepare('SELECT trace FROM chat_turn_traces').get().trace};
const sizes=s=>s.raw.prepare("SELECT (SELECT sum(length(trace)) FROM chat_turn_traces) traces,(SELECT sum(length(content)+length(response)) FROM chat_turns) turns").get();

test('F1: after 31 simulated days, 3 000 traces weigh at most twice the conversation and keep verdicts, models and tokens',async t=>{
 const trace=await realTrace(t),s=setup();filledProfile(s);await s.ready();provider(t);
 seedTurns(s,'a',3000,{trace,traceAt:i=>s.now()-3000+i,jobAt:s.now()});
 const before=sizes(s);assert.ok(before.traces>10*before.turns,'precondition: detailed traces dominate the database');
 s.tick(31*86400);let passes=0;
 while(s.raw.prepare('SELECT count(*) n FROM chat_turn_traces WHERE compacted_at=0').get().n&&passes<40){await s.chat.resumeEngine(s.env);passes++}
 const after=sizes(s);
 t.diagnostic(`traces ${(before.traces/1e6).toFixed(1)} Mo -> ${(after.traces/1e6).toFixed(2)} Mo ; conversation ${(after.turns/1e6).toFixed(2)} Mo ; ${passes} passages`);
 assert.ok(after.traces<=2*after.turns,`${after.traces} > 2 x ${after.turns}`);
 const kept=JSON.parse(s.raw.prepare("SELECT trace FROM chat_turn_traces WHERE turn_id='a-h10'").get().trace),original=JSON.parse(trace);
 assert.deepEqual(traceSummary(kept).review,original.review,'verdicts (and their labels, restored on read) are kept');assert.deepEqual(kept.analysis,original.analysis);assert.deepEqual(kept.models,original.models);
 assert.equal(kept.details.compacted,1);assert.deepEqual(kept.details.values,[]);
 assert.deepEqual(kept.details.stages.map(x=>[x.stage,x.model,x.status,x.usage?.input_tokens]),original.details.stages.map(x=>[x.stage,x.model,x.status,x.usage?.input_tokens]));
 assert.ok(kept.details.stages.every(x=>x.input_bytes>0&&x.system_bytes>0&&x.input_refs===undefined));
 const list=await s.call('/api/chat/insight');
 assert.ok(list.turns.length);assert.ok(list.turns.every(x=>x.details_available===false&&x.unavailable_reason==='retention'));
 const detail=(await s.call('/api/chat/insight?turn_id=a-h10')).selected_turn;
 assert.equal(detail.unavailable_reason,'retention');assert.ok(detail.stages.every(x=>x.input===null));
 assert.doesNotMatch(JSON.stringify(detail),/rumine quand ma sœur/,'exact inputs are gone');
});

test('F1: a person keeps at most 200 detailed traces; live turns are never compacted',async t=>{
 const trace=await realTrace(t),s=setup();await s.ready();provider(t);
 seedTurns(s,'a',230,{trace,traceAt:i=>s.now()-1000+i,jobAt:s.now()});
 s.raw.prepare("UPDATE chat_turns SET week_start='2025-01-06' WHERE id LIKE 'a-h%'").run();
 // A finished turn whose job is still live (being resumed) keeps its exact inputs.
 s.raw.prepare("UPDATE chat_engine_jobs SET stage='analysis' WHERE turn_id='a-h3'").run();
 for(let n=0;n<3;n++){const sent=await s.call('/api/chat/messages','POST',{content:'Encore un message '+n+' sur ma semaine.',idempotency_key:'cap_turn_a_'+n});assert.equal(sent.status,200,sent.message)}
 const detailed=s.raw.prepare("SELECT count(*) n FROM chat_turn_traces WHERE user_id='a' AND compacted_at=0").get().n;
 assert.ok(detailed<=200+1,`${detailed} detailed traces`);
 assert.equal(s.raw.prepare("SELECT compacted_at FROM chat_turn_traces WHERE turn_id='a-h3'").get().compacted_at,0,'a live job keeps its exact inputs');
 assert.ok(s.raw.prepare("SELECT compacted_at FROM chat_turn_traces WHERE turn_id='a-h4'").get().compacted_at>0);
});

test('F2: the cron reads live jobs through the partial index and purges finished jobs after 7 days',async t=>{
 const s=setup();await s.ready();provider(t);
 seedTurns(s,'a',50,{trace:'{}',traceAt:s.now(),jobAt:s.now()-8*86400});
 seedTurns(s,'a',5,{trace:'{}',traceAt:s.now(),jobAt:s.now()-86400,start:100});
 const plan=s.plan("SELECT turn_id FROM chat_engine_jobs WHERE stage NOT IN ('done','failed') AND lease_until<=? ORDER BY created_at LIMIT 5").join(' | ');
 assert.doesNotMatch(plan,/SCAN chat_engine_jobs/);assert.match(plan,/chat_engine_live/);
 const used=s.measure();await s.chat.resumeEngine(s.env);
 for(const sql of used().sql.keys())if(/chat_engine_jobs/.test(sql)&&/^\s*(SELECT|DELETE|UPDATE)/i.test(sql))assert.doesNotMatch(s.plan(sql).join(' | '),/SCAN chat_engine_jobs\b(?! USING)/,sql);
 assert.equal(s.raw.prepare("SELECT count(*) n FROM chat_engine_jobs WHERE updated_at<?").get(s.now()-7*86400).n,0);
 assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_engine_jobs').get().n,5);
});

// SQLite may return a scalar-subquery aggregate as plain TEXT rather than preserving
// its JSON subtype. substr reproduces that boundary deterministically on every version.
function withoutAggregateSubtype(sql){
 const openings=[...sql.matchAll(/\(SELECT json_group_array\(/g)].map(m=>m.index);
 for(const start of openings.reverse()){
  let depth=0,quoted=false,end=start;
  for(;end<sql.length;end++){
   const c=sql[end];
   if(c==="'"){if(quoted&&sql[end+1]==="'"){end++;continue;}quoted=!quoted;continue;}
   if(quoted)continue;
   if(c==='(')depth++;else if(c===')'&&--depth===0)break;
  }
  assert.equal(depth,0,'aggregate subquery must be balanced');
  sql=sql.slice(0,start)+'substr('+sql.slice(start,end+1)+',1)'+sql.slice(end+1);
 }
 assert.ok(openings.length>=2,'exercise both aggregate embedding boundaries');
 return sql;
}
test('F1: scalar aggregate subtype loss never turns retained stages or criteria into JSON strings',async t=>{
 const original=await realTrace(t),s=setup();await s.ready();
 seedTurns(s,'a',1,{trace:original,traceAt:s.now()});
 const portable=withoutAggregateSubtype(COMPACT_TRACE_SQL);
 s.raw.prepare('UPDATE chat_turn_traces SET trace='+portable).run();
 const kept=JSON.parse(s.raw.prepare('SELECT trace FROM chat_turn_traces').get().trace),source=JSON.parse(original);
 assert.ok(Array.isArray(kept.details.stages));assert.ok(Array.isArray(kept.review.change.criteria));
 assert.deepEqual(traceSummary(kept).review,source.review);
 assert.deepEqual(kept.details.stages.map(x=>[x.stage,x.model,x.status,x.usage]),source.details.stages.map(x=>[x.stage,x.model,x.status,x.usage]));
 assert.deepEqual(kept.details.values,[]);
 const retained=sizes(s);assert.ok(retained.traces<=2*retained.turns,`${retained.traces} > 2 x ${retained.turns}`);
});
