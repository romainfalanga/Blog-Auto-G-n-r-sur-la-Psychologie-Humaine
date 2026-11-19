import test from 'node:test';
import assert from 'node:assert/strict';
import {SYMPTOMES_CARTE as data} from '../src/symptomes-carte.generated.mjs';
import {computeCartography} from '../src/cartographie.mjs';
const contexts = ['ten_symptomes_depressifs_perinataux','ten_violences_amoureuses_jeunes','ten_violences_couple','ten_automutilations_non_suicidaires','ten_procrastination'];
for (const outcome of contexts) test(`${outcome}: shared signs never infer a missing context`,()=>{
 const answers=Object.fromEntries(data.links.filter(l=>l.outcome===outcome).map(l=>[l.factor,1]));
 const required=data.requires[outcome];assert.ok(required?.length);
 const visible=()=>computeCartography(answers,data).results.some(r=>r.outcome.id===outcome);
 for(const value of [undefined,0,'unknown']){for(const id of required){if(value===undefined)delete answers[id];else answers[id]=value;}assert.equal(visible(),false);}
 for(const id of required)answers[id]=1;assert.equal(visible(),true);
});
test('retaining anger and masking it count as the same essential family',()=>{
 const a=data.factors.find(f=>f.id==='emo_ravaler_sa_colere');
 const b=data.factors.find(f=>f.id==='emo_cacher_sa_colere');
 assert.equal(a.overlap_group,b.overlap_group);
});
test('context-only questions have no artificial clinical links',()=>{
 for(const id of ['contexte_perinatal','contexte_relation_adolescence','contexte_relation_adulte'])assert.equal(data.links.some(l=>l.factor===id),false);
 assert.ok(data.factors.some(f=>f.safety==='automutilation'),'classification gate does not remove safety detection');
});
