const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),{readFileSync}=require('node:fs'),{join}=require('node:path');
const browser=require('../static/js/cartographie.js'),childhood=require('../data/cartographie.json');
test('Mon enfance ranking remains identical to the published implementation for 200 mixed profiles',()=>{
 const original=readFileSync(join(__dirname,'fixtures/cartographie-before-symptom-v2.cjs'),'utf8'),context={module:{exports:{}},Set};vm.runInNewContext(original,context);let seed=17;const random=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32);
 for(let n=0;n<200;n++){const answers={},feedback={};for(const f of childhood.factors)if(random()<.7)answers[f.id]=[0,1,2,'unknown'][Math.floor(random()*4)];for(const o of childhood.outcomes)if(random()<.2)feedback[o.id]=random()<.5?'oui':'non';assert.equal(JSON.stringify(browser.compute(childhood,answers,feedback)),JSON.stringify(context.module.exports.compute(childhood,answers,feedback)));}
});
test('symptom ranking groups overlapping signs, gates products and ignores unknown or limited-only answers',()=>{
 const factors=['a','b','b2','c','d','product'].map(id=>({id,overlap_group:id==='b2'?'b':id})),outcomes=[{id:'x'},{id:'y'}],links=[['a','x','critere'],['b','x','critere'],['b2','x','critere'],['c','x','critere'],['d','x','limitee'],['a','y','critere'],['b','y','critere']].map(([factor,outcome,strength])=>({factor,outcome,strength})),data={factors,outcomes,links,requires:{y:['product']},ranking:{version:'essential_share_desc'}};
 assert.equal(browser.compute(data,{a:1,b:'unknown',d:1}).results.length,0);
 let result=browser.compute(data,{a:1,b:1,b2:1,d:1});assert.deepEqual(result.results.map(r=>r.outcome.id),['x']);assert.equal(result.results[0].essential_count,2);assert.equal(result.results[0].essential_total,3);assert.equal(result.results[0].essential_score,500);assert.equal(result.results[0].uncertain_links.length,1);
 result=browser.compute(data,{a:1,b:1,product:1});assert.deepEqual(result.results.map(r=>r.outcome.id),['y','x']);
});
test('symptom ranking is identical in the browser and Worker for complete and sparse real-bank profiles',async()=>{
 const {computeCartography}=await import('../journal/src/cartographie.mjs'),data=require('../data/symptomes_carte.json');let seed=19;const random=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32);
 for(let n=0;n<60;n++){const answers={},feedback={};for(const f of data.factors)if(n===0||random()<.7)answers[f.id]=n===0?1:[0,1,'unknown'][Math.floor(random()*3)];for(const o of data.outcomes)if(random()<.15)feedback[o.id]=random()<.5?'oui':'non';assert.deepEqual(browser.compute(data,answers,feedback),computeCartography(answers,data,feedback));}
});
test('past never contributes to symptom ranking, family count, requirements or current uncertainty',async()=>{
 const {computeCartography,validateCartography}=await import('../journal/src/cartographie.mjs'),data=require('../data/symptomes_carte.json');const answers=Object.fromEntries(data.factors.map(f=>[f.id,'past']));
 assert.equal(browser.compute(data,answers).selectedCount,0);assert.deepEqual(browser.compute(data,answers).results,[]);assert.deepEqual(browser.compute(data,answers),computeCartography(answers,data));
 assert.throws(()=>validateCartography({answers:{abus_physique:'past'},notes:{factors:{abus_physique:'Historique'}}},(status,message)=>{throw Error(message)}),/Choisissez/,'childhood does not acquire the new response');
});
