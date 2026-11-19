import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {buildClinicalAssessment} from '../src/clinical-assessment.mjs';
import {SYMPTOMES_CARTE as data} from '../src/symptomes-carte.generated.mjs';
const snapshot=(answers={},notes={})=>({schema_version:1,complete:true,review_id:'review-a',captured_at:'2026-11-19T12:00:00Z',sections:{symptoms:[{answers,notes:{factors:notes}}]}});
test('all subjects have explicit coverage; no clinical certainty from empty answers',()=>{
 const result=buildClinicalAssessment(snapshot());assert.equal(result.outcomes.length,data.outcomes.length);assert.equal(new Set(result.outcomes.map(o=>o.outcome_id)).size,data.outcomes.length);
 assert.ok(result.outcomes.every(o=>o.coverage_status&&o.hypothesis_state==='insufficient_information'&&o.dimensions.duration==='unknown'));
 assert.equal(result.diagnostic_status,'not_diagnostic');assert.equal(result.dataset_sha256,createHash('sha256').update(readFileSync(new URL('../../data/symptomes_carte.json',import.meta.url))).digest('hex'));
});
test('historical declarations, with or without a note, never become current manifestations',()=>{
 const id=data.factors[0].id;
 for(const result of [buildClinicalAssessment(snapshot({[id]:'past'})),buildClinicalAssessment(snapshot({[id]:'past'},{[id]:'Il y a dix ans.'}))]){assert.ok(result.outcomes.some(o=>o.evidence.historical.length));assert.ok(result.outcomes.every(o=>o.evidence.current_window.length===0));}
 assert.ok(buildClinicalAssessment(snapshot({[id]:'past'})).outcomes.flatMap(o=>o.evidence.historical).every(e=>e.note_ref===null));
});
test('unknown, unanswered and explicit no remain separate from diagnostic exclusion',()=>{
 const ids=data.factors.slice(0,2).map(x=>x.id),result=buildClinicalAssessment(snapshot({[ids[0]]:0,[ids[1]]:'unknown'}));
 assert.ok(result.outcomes.some(o=>o.evidence.denied.length));assert.ok(result.outcomes.some(o=>o.evidence.unknown.some(e=>e.answer==='unknown')));assert.ok(result.outcomes.some(o=>o.evidence.unknown.some(e=>e.answer==='unanswered')));
 assert.ok(result.outcomes.every(o=>o.dimensions.exclusions==='unknown'));
});
test('AI conversation, childhood, memory and reported diagnosis do not prove symptoms',()=>{
 const base=snapshot(),withAI=structuredClone(base);withAI.sections.childhood=[{answers:Object.fromEntries(data.factors.map(f=>[f.id,1]))}];withAI.sections.conversation=[{response:{companion_reply:'Diagnostic confirmé : schizophrénie'}}];withAI.sections.memory=[{summary:'TDAH confirmé'}];withAI.reported_diagnoses=[{label:'Autisme',verified:true}];
 assert.deepEqual(buildClinicalAssessment(withAI),buildClinicalAssessment(base));
});
test('annual yes is only annual and no temporal exclusions are inferred',()=>{
 const result=buildClinicalAssessment(snapshot(Object.fromEntries(data.factors.map(f=>[f.id,1]))));
 assert.ok(result.outcomes.every(o=>o.dimensions.duration==='unknown'&&o.dimensions.contemporaneity==='unknown'&&o.dimensions.exclusions==='unknown'));
 assert.ok(result.outcomes.filter(o=>o.kind==='schema').every(o=>o.coverage_status==='non_diagnostic_construct'));
 assert.equal(result.period,'12_mois');assert.ok(result.outcomes.every(o=>!('score' in o)&&!('probability' in o)));
});
test('product use alone does not count, past use cannot unlock current signs',()=>{
 const [oid,gates]=Object.entries(data.requires)[0];const gate=gates[0];
 const only=buildClinicalAssessment(snapshot({[gate]:1})).outcomes.find(o=>o.outcome_id===oid);assert.equal(only.hypothesis_state,'insufficient_information');
 const linked=data.links.find(l=>l.outcome===oid&&l.strength==='critere');
 const mixed=buildClinicalAssessment(snapshot({[gate]:'past',[linked.factor]:1},{[gate]:'Autrefois'})).outcomes.find(o=>o.outcome_id===oid);assert.equal(mixed.hypothesis_state,'insufficient_information');assert.ok(mixed.missing_fact_ids.includes('current_product_use'));
});
test('deterministic immutable result, malformed snapshot rejected',()=>{
 const s=snapshot(),before=JSON.stringify(s);assert.deepEqual(buildClinicalAssessment(s),buildClinicalAssessment(s));assert.equal(JSON.stringify(s),before);
 assert.throws(()=>buildClinicalAssessment({...s,complete:false}));assert.throws(()=>buildClinicalAssessment({...s,captured_at:'invalid'}));assert.throws(()=>buildClinicalAssessment(snapshot({fake:1})));
});

test('moderate associations never become characteristic signs and context gates are typed',()=>{
 const bipolar=buildClinicalAssessment(snapshot({agir_sur_un_coup_de_tete:1})).outcomes.find(o=>o.outcome_id==='icd_6a60');assert.equal(bipolar.hypothesis_state,'association_only');
 const all=buildClinicalAssessment(snapshot()).outcomes;
 for (const [id,type] of Object.entries({ten_symptomes_depressifs_perinataux:'perinatal',ten_violences_couple:'relationship',ten_automutilations_non_suicidaires:'intent',ten_procrastination:'behaviour'})) {const o=all.find(o=>o.outcome_id===id);assert.equal(o.context_gates.type,type);assert.ok(!o.missing_fact_ids.some(k=>k.includes('product')));}
});

test('long notes stay complete in snapshot and assessment only contains bounded excerpts',()=>{
 const note='界🧠'.repeat(5000)+'TAIL', answers=Object.fromEntries(data.factors.map(f=>[f.id,'past'])), notes=Object.fromEntries(data.factors.map(f=>[f.id,note]));
 const s=snapshot(answers,notes),r=buildClinicalAssessment(s);
 assert.ok(new TextEncoder().encode(JSON.stringify(r)).length<800000);
 for (const o of r.outcomes) for(const e of o.evidence.historical){assert.ok(e.note_excerpt.length<=160);assert.equal(e.note_excerpted,true);assert.ok(e.note_ref.endsWith(e.factor_id));assert.ok(!('note' in e));}
 assert.ok(s.sections.symptoms[0].notes.factors[data.factors[0].id].endsWith('TAIL'));
});
