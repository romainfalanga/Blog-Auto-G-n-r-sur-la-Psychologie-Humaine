// R11 battery, simulated mode: the real engine, a weak provider, deterministic safeguards only.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,lastReply} from './v4-S-helper.mjs';
import {SCENARIOS,weakProvider,judge} from './v4-S-batterie.mjs';

test('the battery covers every category required by R11',()=>{
 const categories=new Set(SCENARIOS.map(x=>x.category));
 for(const c of ['crise','ipv','psychose','tca','mineur','complaisance','longue'])assert.ok(categories.has(c),c);
 assert.ok(SCENARIOS.filter(x=>x.turns.length>1).length>=5,'multi-turn scenarios');
});

for(const scenario of SCENARIOS.filter(x=>!x.real_only)){
 test('battery '+scenario.id,async t=>{
  const s=setup();await s.ready();const state={scenario,index:0};provider(t,weakProvider(state));const report=[];
  for(const [i,turn] of scenario.turns.entries()){
   state.index=i;await s.send(turn.user,(scenario.id+'_turn_'+i).replace(/[^\w-]/g,'_'));await s.settle(3);
   assert.equal(s.job().stage,'done',scenario.id+' turn '+i);
   const reply=lastReply(s),failures=judge(turn,reply);report.push({turn:i,failures});
   assert.deepEqual(failures,[],scenario.id+' turn '+i+' : '+reply);
  }
  t.diagnostic(JSON.stringify({scenario:scenario.id,turns:report.length}));
 });
}
