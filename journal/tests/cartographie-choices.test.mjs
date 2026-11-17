import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {validateCartography,computeCartography,summarizeCartography,cartographyPromptBlock,cartographyIndexEntries} from '../src/cartographie.mjs';
const require=createRequire(import.meta.url),publicPage=require('../../static/js/cartographie.js');
const data=JSON.parse(readFileSync(new URL('../../data/cartographie.json',import.meta.url)));
const fail=(status,message,code)=>{throw Object.assign(new Error(message),{status,code})};

test('uncertainty is a strict fourth answer, distinct from a negative or missing answer',()=>{
 const answers={abus_physique:'unknown',abus_sexuel:0,abus_emotionnel:1};assert.deepEqual(validateCartography({answers},fail).answers,answers);
 for(const invalid of [-1,3,null,false,'Unknown','je ne sais pas','0'])assert.throws(()=>validateCartography({answers:{abus_physique:invalid}},fail),{code:'invalid_cartography'});
 const summary=summarizeCartography(answers);assert.equal(summary.answered_count,3);assert.equal(summary.cause_count,1);assert.equal(summary.ace_count,1);assert.deepEqual(summary.explicit_uncertain,[{id:'abus_physique',label:'Violences physiques',answer:'Je ne sais pas'}]);assert.deepEqual(summary.explicit_negative.map(x=>x.id),['abus_sexuel']);assert.deepEqual(summary.causes.map(x=>x.id),['abus_emotionnel']);
});

test('uncertain answers never create or strengthen associations and do not conflict with alternatives',()=>{
 const all=Object.fromEntries(CARTOGRAPHIE.factors.map(f=>[f.id,'unknown']));const result=computeCartography(all);assert.equal(result.selectedCount,0);assert.equal(result.aceCount,0);assert.deepEqual(result.results,[]);
 assert.deepEqual(validateCartography({answers:{divorce:'unknown',separation_conflit:1}},fail).answers,{divorce:'unknown',separation_conflit:1});
 for(const factor of CARTOGRAPHIE.factors){const base={abus_physique:1};if(factor.id==='abus_physique')continue;assert.deepEqual(computeCartography({...base,[factor.id]:'unknown'}),computeCartography(base),factor.id)}
});

test('mixed unknown, negative and historical answers retain strict browser/Worker parity',()=>{
 let seed=8421;const rand=()=>((seed=(seed*1664525+1013904223)>>>0)/4294967296);
 for(let n=0;n<300;n++){const answers={};for(const f of data.factors){const v=rand();if(v<.6)answers[f.id]=[0,1,2,'unknown'][Math.floor(rand()*4)]}const a=publicPage.compute(data,answers),b=computeCartography(answers);assert.deepEqual(a.results.map(x=>[x.outcome.id,x.section,x.relevance,x.score,x.associated_count]),b.results.map(x=>[x.outcome.id,x.section,x.relevance,x.score,x.associated_count]));assert.equal(a.aceCount,b.aceCount);assert.equal(a.selectedCount,b.selectedCount)}
});

test('Ambre receives uncertainty as uncertainty, never as a lived experience or a no',()=>{
 const summary=summarizeCartography({abus_physique:'unknown'}),block=cartographyPromptBlock(summary);assert.deepEqual(block.explicit_uncertain,summary.explicit_uncertain);assert.deepEqual(block.causes,[]);assert.deepEqual(block.explicit_negative,[]);
 const entry=cartographyIndexEntries(summary).find(x=>x.metadata.explicit_uncertain);assert.match(entry.label,/Je ne sais pas/);assert.equal(entry.metadata.answer,'unknown');assert.match(entry.content.interpretation,/ni oui, ni non/);assert.notEqual(entry.kind,'cause');
 const many=summarizeCartography(Object.fromEntries(CARTOGRAPHIE.factors.map(f=>[f.id,'unknown']))),tight=cartographyPromptBlock(many,1200);assert.ok(JSON.stringify(tight).length<=1200);assert.ok(tight.explicit_uncertain_omitted>0);assert.deepEqual(tight.causes,[]);
});

test('binary questions keep old Souvent values without silently converting them',()=>{
 const ids=['parent_incarcere','deces_proche','migration','traumatisme_ponctuel','divorce','adoption'];assert.deepEqual(data.factors.filter(f=>f.response_mode==='binary').map(f=>f.id),ids);
 const answers=Object.fromEntries(ids.map(id=>[id,2])),validated=validateCartography({answers},fail);assert.deepEqual(validated.answers,answers);const summary=summarizeCartography(answers);assert.ok(summary.causes.every(c=>c.intensity===2&&c.historical_frequency));assert.ok(cartographyPromptBlock(summary).causes.every(c=>c.intensity===2&&c.label.includes('ancien Souvent')));
 for(const id of ids)assert.equal(CARTOGRAPHIE.factors.find(f=>f.id===id).response_mode,'binary');
});

test('the combined historical control retains its meaning and links without populating either new dimension',()=>{
 const old=data.factors.find(f=>f.id==='controle');assert.equal(old.label,'Contrôle et obéissance stricte');assert.equal(old.historical,true);assert.deepEqual(old.replaced_by,['parents_stricts','controle_psychologique']);assert.equal(data.links.filter(l=>l.factor==='controle').length,9);
 for(const value of [0,1,2,'unknown']){const answer=validateCartography({answers:{controle:value}},fail).answers;assert.deepEqual(answer,{controle:value});assert.equal(Object.hasOwn(answer,'parents_stricts'),false);assert.equal(Object.hasOwn(answer,'controle_psychologique'),false)}
 const summary=summarizeCartography({controle:2});assert.equal(summary.causes[0].historical,true);assert.match(summary.causes[0].interpretation,/ne pas attribuer/);assert.match(cartographyPromptBlock(summary).causes[0].label,/ancienne question combinée/);
 const separated={parents_stricts:2,controle_psychologique:1};assert.equal(computeCartography(separated).results.length,0);assert.ok(summarizeCartography(separated).causes.every(c=>c.interpretation.includes('aucune association scientifique')));assert.ok(data.links.every(l=>!old.replaced_by.includes(l.factor)));assert.equal(data.factors.filter(f=>f.group==='climat'&&!f.historical).length,14);
});
