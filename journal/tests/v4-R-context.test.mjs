// Lot R, F23: the personal context is read once per turn version, not at every stage.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,filledProfile} from './v4-R-helper.mjs';

const lifeEventReads=used=>[...used.sql.entries()].filter(([sql])=>/FROM life_events/.test(sql)).reduce((n,[,count])=>n+count,0);

test('F23: a maximal profile reads life_events at most twice per turn and uses far less CPU',async t=>{
 const rows=[];
 for(const maximal of [false,true]){
  const s=setup();filledProfile(s,'a',maximal);await s.ready();provider(t);
  const m=s.measure(),cpu=process.cpuUsage();
  const r=await s.call('/api/chat/messages','POST',{content:'Je rumine quand ma sœur me critique.',idempotency_key:'cpu_turn_00001'});
  const spent=process.cpuUsage(cpu),used=m();assert.equal(r.status,200,r.message);
  rows.push({profile:maximal?'maximal':'rempli',life_event_reads:lifeEventReads(used),statements:used.statements,cpu_ms:Math.round((spent.user+spent.system)/1000)});
  assert.ok(lifeEventReads(used)<=2,`${lifeEventReads(used)} reads of life_events`);
 }
 t.diagnostic(JSON.stringify(rows));
});

test('F23: a source edited between two stages is still detected (no stale cache)',async t=>{
 const s=setup();filledProfile(s,'a');await s.ready();let edited=false;
 provider(t,purpose=>{if(purpose==='ambre_writing'&&!edited){edited=true;s.raw.prepare("UPDATE life_events SET description='Changé',revision=revision+1 WHERE id='a-event-0'").run();s.raw.prepare("UPDATE profiles SET version=version+1 WHERE user_id='a'").run()}});
 await s.call('/api/chat/messages','POST',{content:'Je rumine quand ma sœur me critique.',idempotency_key:'stale_turn_001'});
 const turn=s.raw.prepare('SELECT status,error_code FROM chat_turns').get();
 assert.equal(turn.status,'failed');assert.equal(turn.error_code,'context_changed');
 // A cartography note written by another tab is also part of the validated stamp.
 const s2=setup();filledProfile(s2,'a');await s2.ready();let noted=false;
 provider(t,purpose=>{if(purpose==='ambre_writing'&&!noted){noted=true;s2.raw.prepare("UPDATE cartographies SET locked_at=2 WHERE user_id='a'").run()}});
 await s2.call('/api/chat/messages','POST',{content:'Je rumine quand ma sœur me critique.',idempotency_key:'stale_turn_002'});
 assert.equal(s2.raw.prepare('SELECT error_code FROM chat_turns').get().error_code,'context_changed');
});
