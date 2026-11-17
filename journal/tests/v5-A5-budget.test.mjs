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

// Lot A5-P1: the review thread replaces the last-review excerpt. Measured with the three reviews
// carrying full digests (three goals followed, trials, feedback): every stage stays within +10 %
// of the same frozen reference, and the thread keeps its fixed share (1 200 characters).
// Measured on this tree: writing +9.0 % (ideal), +0 % to +7 % (none) against the frozen reference,
// i.e. +1.4 % against P0 (6 397 → 6 487 tokens, system +206 characters); turn total +5.1 % at most.
for(const strategy of ['ideal','none'])for(const [label,message] of MESSAGES)test(`A5-P1 turn budget with a full review thread ${strategy}/${label}: within +10 %`,async t=>{
 const run=await replayTurn(message,{strategy,digests:true});assert.equal(run.stage,'done');
 const sizes=Object.fromEntries(run.calls.map(c=>[c.purpose,c.tokens])),ref=REFERENCE[strategy][label],input=JSON.parse(run.get('ambre_writing').input);
 t.diagnostic(JSON.stringify({sizes,ref,ratio:+(sum(sizes)/sum(ref)).toFixed(3),thread_chars:JSON.stringify(input.review_thread).length}));
 assert.ok(input.review_thread.goals.length>=2&&input.review_thread.feedback.length>=1,'goals and feedback reach the writer');
 assert.ok(JSON.stringify(input.review_thread).length<=1200);
 for(const [purpose,tokens] of Object.entries(ref))assert.ok(sizes[purpose]<=tokens*1.1,`${purpose} ${sizes[purpose]} > ${tokens}*1.1`);
 assert.ok(sum(sizes)<=sum(ref)*1.1);
});
// The review may cost more (plan v5.1): its writing input stays within +10 % (the context ceiling
// of 34 000 characters is unchanged), and the separate check adds one small call. Measured:
// initial 13 047 + check 2 629 tokens (+26 % in all), point 13 622 + 2 628 (+30 %); output cap
// 3 200 → 4 800 (text and digest) + 1 500 (check). A rewrite at most doubles both calls.
for(const kind of ['initial','point'])test(`A5-P1 review cost ${kind}: writing within +10 %, check under 3 500 tokens, both published together`,async t=>{
 const content='BILAN '+'Vous écrivez que la critique réveille une vigilance ancienne ; cela peut se relier à votre histoire. '.repeat(55);
 const run=await replayReview(kind,{digests:true,write:()=>({content,digest:{formulation:{holds:[],revised:[],new:[]},goals:[],tried:[],alliance:{text:'',quote:''},open_questions:['Qu’aimeriez-vous travailler ?'],next_focus:'La suite',safety:''}})});
 const ref=REFERENCE.review[kind],total=run.tokens+run.checkTokens;
 t.diagnostic(JSON.stringify({write:run.tokens,check:run.checkTokens,total,ref,ratio:+(total/ref).toFixed(3),output_cap:run.outputCap}));
 assert.equal(run.r.status,202);assert.ok(run.check,'a separate check call');
 assert.ok(run.tokens<=ref*1.1);assert.ok(run.checkTokens<=3500);assert.ok(total<=ref*1.35);
 if(kind==='point')assert.ok(JSON.stringify(run.data.review_thread).length<=3000);
});

// A5-P2: every new text filled to its maximum; no larger context allocation or extra call.
function fillP2(s){
 const p=s.raw.prepare("SELECT self_data FROM profiles WHERE user_id='a'").get();
 s.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({...JSON.parse(p.self_data),what_brings_me:'Je cherche à comprendre. '.repeat(100).slice(0,2000),what_helps_me:'La marche me ressource. '.repeat(100).slice(0,2000),life_rhythm:'Je travaille de nuit. '.repeat(110).slice(0,2000)}));
 s.raw.prepare("UPDATE personal_people SET bond_quality='ambivalent',presence_now='daily',feels_safe='yes',bond_note=?,helps=?,hurts=?,evolution=?").run('Nuance du lien '.repeat(30).slice(0,300),...Array(3).fill('Ce lien change selon les périodes. '.repeat(30).slice(0,600)));
 s.raw.prepare("UPDATE life_events SET felt_then=?,felt_now=?,impact=?,support_level='some',support_text=?,exposure='witnessed'").run(...Array(3).fill('Ce moment compte dans ma vie. '.repeat(30).slice(0,600)),'Une personne a écouté. '.repeat(20).slice(0,300));
}
for(const strategy of ['ideal','none'])for(const [label,message] of MESSAGES)test(`A5-P2 full optional fields ${strategy}/${label}: turn within +10 %`,async t=>{
 const run=await replayTurn(message,{strategy,digests:true,extend:fillP2});assert.equal(run.stage,'done');
 const sizes=Object.fromEntries(run.calls.map(c=>[c.purpose,c.tokens])),ref=REFERENCE[strategy][label];
 t.diagnostic(JSON.stringify({sizes,total:sum(sizes),reference:sum(ref),ratio:+(sum(sizes)/sum(ref)).toFixed(3)}));
 assert.equal(run.calls.length,3,'no extra provider call for forms');
 for(const [purpose,tokens] of Object.entries(ref))assert.ok(sizes[purpose]<=tokens*1.1,`${purpose} ${sizes[purpose]} > ${tokens}*1.1`);
 assert.ok(sum(sizes)<=sum(ref)*1.1);
});
