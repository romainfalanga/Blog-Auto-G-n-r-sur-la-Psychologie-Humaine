import test from 'node:test';
import assert from 'node:assert/strict';
import {selectPsychologyGuide} from '../src/psychology-guide.mjs';
import {PSYCHOLOGY_CATALOG as catalog} from '../src/psychology-catalog.generated.mjs';
import {buildCompanionSystemPrompt} from '../src/psychology-framework.mjs';

test('portrait guide keeps every concept accessible and bounds detailed entries',()=>{
 const guide=selectPsychologyGuide({sources:[{text:'Difficultés de sommeil, concentration et rumination ; soutien de mes amis.'}]});
 assert.equal(guide.index.length,catalog.patterns.length);
 assert.deepEqual(new Set(guide.index.map(p=>p[0])),new Set(catalog.patterns.map(p=>p.id)));
 assert.equal(guide.patterns.length,24);
 assert.ok(guide.patterns.some(p=>p.id==='rumination'));
 assert.ok(guide.patterns.some(p=>(p.keywords||[]).includes('sommeil')||/sommeil|dormir/i.test(p.label)));
 assert.ok(new Set(guide.patterns.map(p=>p.domain)).size>4);
 assert.ok(JSON.stringify(guide).length<36000);
});

test('existing hypotheses stay in the guide even when the latest query changes subject',()=>{
 const guide=selectPsychologyGuide({query:'Mon travail',hypotheses:[{pattern_id:'parentification'}],limit:8,includeIndex:false});
 assert.ok(guide.patterns.some(p=>p.id==='parentification'));
 assert.equal(guide.patterns.length,8);
 assert.equal(guide.index,undefined);
});

test('companion receives a small exploration guide instead of the whole atlas',()=>{
 const prompt=buildCompanionSystemPrompt({query:'Je rumine et cherche la certitude.'});
 assert.match(prompt,/rumination/);
 assert.ok(prompt.length<18000);
 assert.ok(!prompt.includes(JSON.stringify(catalog.patterns)));
});
