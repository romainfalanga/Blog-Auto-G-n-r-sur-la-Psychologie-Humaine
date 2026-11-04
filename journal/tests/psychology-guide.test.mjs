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

test('contradicted and dismissed hypotheses do not receive active-pattern or related-pattern priority',()=>{
 const options={query:'Mon sommeil se décale et je peine à me réveiller.',limit:8,includeIndex:false};
 const baseline=selectPsychologyGuide(options).patterns.map(p=>p.id);
 for(const status of ['contradicted','dismissed'])assert.deepEqual(selectPsychologyGuide({...options,hypotheses:[{pattern_id:'rumination',status}]}).patterns.map(p=>p.id),baseline);
 for(const status of ['exploring','supported','mixed'])assert.ok(selectPsychologyGuide({...options,hypotheses:[{pattern_id:'parentification',status}]}).patterns.some(p=>p.id==='parentification'));
});

const broadSources=catalog.patterns.map(p=>({text:[p.label,...p.keywords,...p.signals].join(' ')}));
test('the current sleep or executive question retains the first two slots despite positive context in all sixteen domains',()=>{
 assert.equal(new Set(catalog.patterns.map(p=>p.domain)).size,16);
 for(const [query,domain] of [['Je ne parviens pas à dormir : je veux parler de mon sommeil et des réveils nocturnes.','body'],['Je perds ma concentration et ma mémoire de travail quand les distractions s’accumulent.','executive']]){
  const guide=selectPsychologyGuide({query,sources:broadSources,limit:8,includeIndex:false});
  assert.equal(guide.patterns.length,8);assert.equal(guide.patterns[0].domain,domain);assert.equal(guide.patterns[1].domain,domain);
 }
 const wide=selectPsychologyGuide({query:'Mon sommeil et mes réveils nocturnes.',sources:broadSources});assert.equal(wide.patterns.length,24);assert.equal(wide.index.length,catalog.patterns.length);assert.equal(new Set(wide.patterns.map(p=>p.domain)).size,16);
});

test('active hypotheses use at most half the companion budget before score-ordered domain diversity',()=>{
 const hypotheses=catalog.patterns.filter(p=>p.domain==='thinking').slice(0,8).map(p=>({pattern_id:p.id,status:'supported'})),active=new Set(hypotheses.map(h=>h.pattern_id));
 const query='Je souhaite comprendre mon sommeil et mon endormissement.';
 const guide=selectPsychologyGuide({query,sources:broadSources,hypotheses,limit:8,includeIndex:false});
 assert.equal(guide.patterns.length,8);assert.equal(guide.patterns[0].domain,'body');assert.equal(guide.patterns.filter(p=>active.has(p.id)).length,4);assert.ok(new Set(guide.patterns.map(p=>p.domain)).size>=4);
 // Contradicted hypotheses add neither a reservation nor a related-pattern
 // bonus; their counter-evidence remains in the separate conversation context.
 const contradicted=hypotheses.map(h=>({...h,status:'contradicted'}));assert.deepEqual(selectPsychologyGuide({query,sources:broadSources,hypotheses:contradicted,limit:8,includeIndex:false}).patterns,selectPsychologyGuide({query,sources:broadSources,limit:8,includeIndex:false}).patterns);
});
