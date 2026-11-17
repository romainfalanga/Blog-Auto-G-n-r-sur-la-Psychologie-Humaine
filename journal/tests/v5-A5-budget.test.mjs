// Lot A5-P0 budget (plan v5.1): reference frozen BEFORE any A5-P0 change, on the filled
// profile of the v5 audit (portrait, three ready reviews, memory, eight exchanges), measured
// on tree 9bef77c. Input tokens are estimated as in v4-S-budget: (system + user + schema) / 3.
// Every stage of every message type, and each review, may grow by at most 10 %.
import test from 'node:test';
import assert from 'node:assert/strict';
import {replayTurn,replayReview,MESSAGES} from './v5-A-helper.mjs';
const REFERENCE={
 ideal:{proche:{ambre_analysis:2406,ambre_writing:5953,ambre_review:2460},suite_bilan:{ambre_analysis:2395,ambre_writing:5944,ambre_review:2451},moment:{ambre_analysis:2438,ambre_writing:5937,ambre_review:2444},portrait:{ambre_analysis:2478,ambre_writing:5945,ambre_review:2452}},
 none:{proche:{ambre_analysis:2406,ambre_writing:5939,ambre_review:2451},suite_bilan:{ambre_analysis:2395,ambre_writing:6321,ambre_review:2463},moment:{ambre_analysis:2438,ambre_writing:6066,ambre_review:2494},portrait:{ambre_analysis:2478,ambre_writing:6236,ambre_review:2496}},
 review:{initial:12423,point:12525}
};
const sum=o=>Object.values(o).reduce((a,b)=>a+b,0);
for(const strategy of ['ideal','none'])for(const [label,message] of MESSAGES)test(`turn budget ${strategy}/${label}: each stage within +10 % of the frozen reference`,async t=>{
 const run=await replayTurn(message,{strategy});assert.equal(run.stage,'done');
 const sizes=Object.fromEntries(run.calls.map(c=>[c.purpose,c.tokens])),ref=REFERENCE[strategy][label];
 t.diagnostic(JSON.stringify({sizes,ref,ratio:+(sum(sizes)/sum(ref)).toFixed(3)}));
 for(const [purpose,tokens] of Object.entries(ref))assert.ok(sizes[purpose]<=tokens*1.1,`${purpose} ${sizes[purpose]} > ${tokens}*1.1`);
 assert.ok(sum(sizes)<=sum(ref)*1.1);
});
for(const kind of ['initial','point'])test(`review budget ${kind}: within +10 % of the frozen reference`,async t=>{
 const run=await replayReview(kind);assert.equal(run.r.status,202);
 t.diagnostic(JSON.stringify({tokens:run.tokens,ref:REFERENCE.review[kind],ratio:+(run.tokens/REFERENCE.review[kind]).toFixed(3)}));
 assert.ok(run.tokens<=REFERENCE.review[kind]*1.1);
});
