// Final review of the v5.1 publication: the cautious reply is left out of the weekly quota
// only when the provider failed on both stages (analysis and risk check).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,response} from './v5-A-helper.mjs';
import {cautiousReply} from '../src/chat-safety.mjs';

const defaults={ambre_writing:{content:'Je vous écoute.',questions:[],proposals:[],cartography_notes:[],people_observations:[],change_proposals:[],followup:null},ambre_review:{criteria:Array(15).fill('yes'),reasons:[],change_criteria:Array.from({length:8},(_,i)=>({id:'C-R'+(i+1),verdict:'not_applicable'}))},ambre_memory:{items:[]},ambre_risk_check:{risk:'none',category:'none',evidence_quote:''}};
async function play(handler,key){
 const s=setup();await s.ready();
 globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;return handler(purpose,body)||response(defaults[purpose]);};
 await s.send('Ces temps-ci je pense souvent à la mort.',key);await s.settle(4);return s;
}

test('an invalid risk-check answer after an analysis outage is not a double provider outage: the cautious reply is counted',async t=>{
 const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original});
 const s=await play(purpose=>purpose==='ambre_analysis'?new Response('{}',{status:500}):purpose==='ambre_risk_check'?response({unexpected:true}):null,'invalid_check');
 assert.equal(s.job().stage,'done');assert.equal(JSON.parse(s.turn().response).content,cautiousReply('self_harm'));
 assert.equal(s.turn().error_code,null,'counted: the verification answered, even if unusable');
 assert.equal((await s.call('/api/chat')).quota.used,1);
});

test('a provider outage on both stages still leaves the cautious reply uncounted',async t=>{
 const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original});
 const s=await play(purpose=>purpose==='ambre_analysis'?new Response('{}',{status:500}):purpose==='ambre_risk_check'?new Response('{}',{status:500}):null,'double_outage_c5a');
 assert.equal(s.turn().error_code,'uncounted_cautious_reply');
 assert.equal((await s.call('/api/chat')).quota.used,0);
});
