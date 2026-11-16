const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { compute } = require('../static/js/cartographie.js');

const fixture = {
  weights: { forte: 3, moderee: 2, limitee: 1, theorique: 1 },
  factors: [
    { id: 'a', ace: true }, { id: 'b', ace: true }, { id: 'c', ace: false }, { id: 'd', ace: true }, { id: 'e', ace: true }
  ],
  outcomes: [
    { id: 'o1', label: 'Un', kind: 'trouble' }, { id: 'o2', label: 'Deux', kind: 'schema' }, { id: 'o3', label: 'Trois', kind: 'tendance' }
  ],
  links: [
    { factor: 'a', outcome: 'o1', strength: 'forte' },
    { factor: 'b', outcome: 'o1', strength: 'limitee' },
    { factor: 'a', outcome: 'o2', strength: 'moderee' },
    { factor: 'c', outcome: 'o3', strength: 'theorique' },
    { factor: 'b', outcome: 'o3', strength: 'limitee' }
  ]
};

test('nothing selected gives no result', () => {
  const r = compute(fixture, []);
  assert.strictEqual(r.results.length, 0);
  assert.strictEqual(r.selectedCount, 0);
});

test('a single strong link is a lead, not a convergence', () => {
  const r = compute(fixture, ['a']);
  const o1 = r.results.find(x => x.outcome.id === 'o1');
  assert.strictEqual(o1.tier, 'piste');
});

test('a limited link does not promote a documented association to convergence', () => {
  const r = compute(fixture, ['a', 'b']);
  assert.strictEqual(r.results[0].outcome.id, 'o1');
  assert.strictEqual(r.results[0].tier, 'piste');
  assert.deepStrictEqual(r.results[0].factors, ['a', 'b']);
  assert.strictEqual(r.results[0].score, 3);
});

test('several weak or theoretical links never count as a convergence', () => {
  const r = compute(fixture, ['b', 'c']);
  const o3 = r.results.find(x => x.outcome.id === 'o3');
  assert.strictEqual(o3.tier, 'theorique');
});

test('adverse childhood experiences are counted', () => {
  assert.strictEqual(compute(fixture, ['a', 'b', 'c', 'd', 'e']).aceCount, 4);
});

test('published dataset is consistent', () => {
  const file = path.join(__dirname, '..', 'data', 'cartographie.json');
  const data = JSON.parse(fs.readFileSync(file, 'utf8'));
  const factors = new Set(data.factors.map(f => f.id));
  const outcomes = new Set(data.outcomes.map(o => o.id));
  const sources = new Set(data.sources.map(s => s.id));
  assert.ok(data.links.length > 0);
  for (const l of data.links) {
    assert.ok(factors.has(l.factor), l.factor);
    assert.ok(outcomes.has(l.outcome), l.outcome);
    assert.ok(l.sources.length > 0 && l.sources.every(s => sources.has(s)), `${l.factor}->${l.outcome}`);
    assert.ok(!/\d+\s?%/.test(l.summary), 'no individual percentages');
  }
  for (const o of data.outcomes) assert.ok(!/^icd_6a0[0-5]/.test(o.id), `${o.id} must not be linked to upbringing`);
  const all = compute(data, data.factors.map(f => f.id));
  assert.ok(all.results.length === outcomes.size);
});

test('definitional feelings do not create a convergence on their own', () => {
  const data = {
    factors: [{ id: 'r', ace: false }, { id: 's', ace: false }],
    outcomes: [{ id: 'sch', label: 'Schéma', kind: 'schema' }],
    links: [
      { factor: 'r', outcome: 'sch', strength: 'moderee', definitional: true },
      { factor: 's', outcome: 'sch', strength: 'theorique' }
    ]
  };
  const r = compute(data, ['r', 's']);
  assert.notStrictEqual(r.results[0].tier, 'convergence');
});

test('a classification context remains a contextual reference, not empirical evidence', () => {
  const data = { factors: [{ id: 'a' }], outcomes: [{ id: 'o', label: 'O', kind: 'trouble' }], links: [{ factor: 'a', outcome: 'o', strength: 'critere' }] };
  assert.strictEqual(compute(data, ['a']).results[0].tier, 'theorique');
});

test('reported intensity does not multiply uncalibrated evidence or risk', () => {
  const list = compute(fixture, ['a', 'b']);
  const intensity = compute(fixture, { a: 2, b: 1 });
  assert.strictEqual(intensity.results[0].outcome.id, 'o1');
  assert.strictEqual(intensity.results[0].tier, list.results[0].tier);
  assert.strictEqual(intensity.results[0].score, 3);
  assert.strictEqual(intensity.selectedCount, 2);
  assert.strictEqual(compute(fixture, { a: 1, b: 1 }).results[0].score, list.results[0].score);
  assert.strictEqual(compute(fixture, { a: 0, b: 'x' }).selectedCount, 0);
  assert.strictEqual(compute(fixture, new Set(['a'])).results.length, compute(fixture, ['a']).results.length);
});

test('three clinical-model or limited links never become an empirical convergence', () => {
  const data = { factors: [{ id: 'a' }, { id: 'b' }, { id: 'c' }], outcomes: [{ id: 'o', label: 'O', kind: 'tendance' }],
    links: [{ factor: 'a', outcome: 'o', strength: 'theorique' }, { factor: 'b', outcome: 'o', strength: 'limitee' }, { factor: 'c', outcome: 'o', strength: 'theorique' }] };
  assert.strictEqual(compute(data, ['a', 'b']).results[0].tier, 'theorique');
  assert.strictEqual(compute(data, ['a', 'b', 'c']).results[0].tier, 'theorique');
});

test('overlapping experiences count as one family and never increase the strongest evidence', () => {
  const data = { factors: [{id:'a',overlap_group:'same'},{id:'b',overlap_group:'same'},{id:'c'}], outcomes:[{id:'o',label:'O'}], links:['a','b','c'].map(factor=>({factor,outcome:'o',strength:'moderee',relation_type:'empirical_association'})) };
  assert.strictEqual(compute(data,{a:1,b:2}).results[0].tier,'piste');
  const result=compute(data,{a:1,b:2,c:1}).results[0];
  assert.strictEqual(result.tier,'convergence');
  assert.strictEqual(result.score,2);
  assert.strictEqual(result.evidence_groups,2);
});

test('coverage reports exact denominators without turning reference presence into diagnosis coverage', () => {
  const data=JSON.parse(fs.readFileSync(path.join(__dirname,'..','data','cartographie.json'),'utf8'));
  assert.strictEqual(data.outcomes.length,82);
  assert.strictEqual(data.coverage.young_schemas.represented_count,18);
  assert.strictEqual(data.coverage.icd11.catalog_codes,858);
  assert.strictEqual(data.coverage.icd11.exact_code_count,19);
  assert.strictEqual(data.coverage.dsm5tr.complete,false);
  assert.strictEqual(data.coverage.psychology_catalog.catalog_patterns,161);
  const review=data.coverage.psychology_catalog.semantic_review;
  assert.strictEqual(review.reviewed_patterns,161);
  assert.strictEqual(Object.values(review.status_counts).reduce((sum,n)=>sum+n,0),161);
  assert.strictEqual(data.coverage.dsm5tr.chapters_reviewed,22);
  const audit=JSON.parse(fs.readFileSync(path.join(__dirname,'..',review.report),'utf8'));
  assert.strictEqual(new Set(audit.psychology_patterns.map(p=>p.id)).size,161);
  assert.strictEqual(audit.icd11_codes.length,858);
  const targets=new Set([...data.factors,...data.outcomes,...data.protections].map(x=>x.id));
  for(const p of audit.psychology_patterns){assert.ok(p.reason.length>30);for(const t of p.targets)assert.ok(targets.has(t.id),t.id);}
  assert.strictEqual(data.coverage.icd11.not_exactly_referenced_count,839);
  assert.ok(data.links.filter(l=>l.outcome==='sch_negativite').every(l=>l.relation_type==='clinical_model'&&l.strength==='theorique'));
  const definitions=new Set(data.links.filter(l=>l.definitional).map(l=>l.relation_type));
  assert.deepStrictEqual([...definitions],['definitional']);
});
