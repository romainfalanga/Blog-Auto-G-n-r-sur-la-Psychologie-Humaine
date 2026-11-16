// Lot R, F3: polling cost independent of the conversation length.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,seedTurns} from './v4-R-helper.mjs';

async function pendingTurn(t,n){
 const s=setup();await s.ready();seedTurns(s,'a',n);provider(t);
 s.raw.prepare("UPDATE chat_turns SET week_start='2025-01-06'").run();
 const sent=await s.call('/api/chat/messages','POST',{content:'Une journée lourde.',idempotency_key:'poll_turn_1'},'a',{waitUntil(){}});
 assert.equal(sent.status,202);return {s,id:sent.request_id};
}
const rowsRead=(s,sql)=>s.plan(sql).join(' | ');

test('F3: a status poll without news reads a bounded number of rows and stays under 2 KB, whatever the history',async t=>{
 const measures=[];
 for(const n of [0,3000]){
  const {s,id}=await pendingTurn(t,n),m=s.measure();
  const poll=await s.call('/api/chat/requests/'+id+'?view=status','GET',undefined,'a',{waitUntil(){}});
  const used=m();
  assert.equal(poll.status,200);assert.equal(poll.request.status,'running');assert.ok(poll.bytes<2048,`${poll.bytes} bytes`);
  for(const sql of used.sql.keys())assert.doesNotMatch(rowsRead(s,sql),/SCAN chat_turns|TEMP B-TREE/,sql);
  measures.push({n,statements:used.statements,bytes:poll.bytes});
 }
 t.diagnostic(JSON.stringify(measures));
 assert.equal(measures[0].statements,measures[1].statements);assert.ok(Math.abs(measures[0].bytes-measures[1].bytes)<16);
});

test('F3: the full conversation read and hourly counters use indexes without SCAN or TEMP B-TREE',async t=>{
 const {s}=await pendingTurn(t,300),m=s.measure();
 await s.call('/api/chat');
 for(const sql of m().sql.keys())if(/chat_turns/.test(sql))assert.doesNotMatch(rowsRead(s,sql),/SCAN chat_turns\b(?! USING)|USE TEMP B-TREE/,sql);
 for(const sql of ["SELECT count(*) FROM chat_turns WHERE user_id=? AND created_at>datetime('now','-1 hour')","SELECT count(*) AS n FROM chat_turns WHERE user_id=? AND idempotency_key LIKE 'review-%' AND created_at>datetime('now','-1 day')"])assert.match(rowsRead(s,sql),/chat_turns_owner_time/);
});

test('F3: the status poll reports a provider wait and a failure without the full state',async t=>{
 const s=setup();await s.ready();
 provider(t,()=>new Response('{}',{status:429,headers:{'Retry-After':'40'}}));
 const sent=await s.call('/api/chat/messages','POST',{content:'Une journée lourde.',idempotency_key:'poll_wait_1'});
 const poll=await s.call('/api/chat/requests/'+sent.request_id+'?view=status');
 assert.equal(poll.request.status,'running');assert.ok(poll.request.waiting.retry_after_seconds>0&&poll.request.waiting.retry_after_seconds<=40);
 assert.equal(poll.messages,undefined);
 const other=await s.call('/api/chat/requests/'+sent.request_id+'?view=status','GET',undefined,'b');assert.equal(other.status,404);
});
