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

test('two factors including a strong link converge and rank first', () => {
  const r = compute(fixture, ['a', 'b']);
  assert.strictEqual(r.results[0].outcome.id, 'o1');
  assert.strictEqual(r.results[0].tier, 'convergence');
  assert.deepStrictEqual(r.results[0].factors, ['a', 'b']);
  assert.strictEqual(r.results[0].score, 4);
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
