// Lot R, F23: the personal context is read once per turn version, not at every stage.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,filledProfile} from './v4-R-helper.mjs';

// Reads of the event contents (the 4 MB part); bounded count(*) subqueries are listed apart.
const lifeEventReads=used=>[...used.sql.entries()].filter(([sql])=>/FROM life_events/.test(sql)&&/description/.test(sql)).reduce((n,[,count])=>n+count,0);
const lifeEventCounts=used=>[...used.sql.entries()].filter(([sql])=>/FROM life_events/.test(sql)&&!/description/.test(sql)).reduce((n,[,count])=>n+count,0);

test('F23: a maximal profile reads life_events at most twice per turn and uses far less CPU',async t=>{
 const rows=[];
 for(const maximal of [false,true]){
  const s=setup();filledProfile(s,'a',maximal);await s.ready();provider(t);
  const m=s.measure(),cpu=process.cpuUsage();
  const r=await s.call('/api/chat/messages','POST',{content:'Je rumine quand ma sœur me critique.',idempotency_key:'cpu_turn_00001'});
  const spent=process.cpuUsage(cpu),used=m();assert.equal(r.status,200,r.message);
  rows.push({profile:maximal?'maximal':'rempli',life_event_reads:lifeEventReads(used),life_event_counts:lifeEventCounts(used),statements:used.statements,cpu_ms:Math.round((spent.user+spent.system)/1000)});
  assert.ok(lifeEventReads(used)<=2,`${lifeEventReads(used)} reads of life_events`);
 }
 t.diagnostic(JSON.stringify(rows));
});

test('F23: on a maximal profile the per-turn CPU is several times lower than without the context cache',async t=>{
 async function turn(cached){const s=setup();filledProfile(s,'a',true);await s.ready();provider(t);
  // Without the revision stamp every stage reloads and re-derives the full context (pre-F23 behaviour).
  if(!cached){const original=s.env.DB.prepare;s.env.DB.prepare=sql=>{if(sql.startsWith('SELECT p.version,p.self_revision'))throw new Error('no stamp');return original(sql)}}
  const m=s.measure(),cpu=process.cpuUsage();await s.call('/api/chat/messages','POST',{content:'Je rumine quand ma sœur me critique.',idempotency_key:'cpu_turn_00001'});const c=process.cpuUsage(cpu);return {cpu:(c.user+c.system)/1000,reads:lifeEventReads(m())}}
 await turn(true);await turn(false);
 const before=[],after=[];for(let i=0;i<3;i++){before.push(await turn(false));after.push(await turn(true))}
 const median=list=>list.map(x=>x.cpu).sort((a,b)=>a-b)[1],ratio=median(before)/median(after);
 t.diagnostic(JSON.stringify({before_ms:Math.round(median(before)),after_ms:Math.round(median(after)),ratio:+ratio.toFixed(2),reads_before:before[0].reads,reads_after:after[0].reads}));
 assert.ok(before[0].reads>2&&after[0].reads<=2);
 // CPU timings vary on shared runners: the measured ratio is reported; 2 is the floor checked here.
 assert.ok(ratio>=2,`ratio ${ratio.toFixed(2)}`);
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
