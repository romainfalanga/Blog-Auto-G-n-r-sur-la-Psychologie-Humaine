// Plan v4.1, lot C3–C6: link strengths, wording, added links and the out-of-scope list.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'data/cartographie.json'), 'utf8'));
const source = id => data.sources.find(s => s.id === id);
const link = (factor, outcome) => data.links.find(l => l.factor === factor && l.outcome === outcome);
const factor = id => data.factors.find(f => f.id === id);

test('C3: no moderate or strong association rests on a single primary study', () => {
  const offenders = data.links.filter(l => l.relation_type === 'empirical_association' && ['forte', 'moderee'].includes(l.strength) && l.sources.length === 1 && source(l.sources[0]).type === 'etude');
  assert.deepStrictEqual(offenders.map(l => l.factor + '→' + l.outcome), []);
  assert.strictEqual(link('parent_addiction', 'icd_6c40_2').strength, 'limitee');
  assert.strictEqual(link('harcelement', 'ten_difficultes_professionnelles').strength, 'limitee', 'two papers, one 1958 cohort');
  assert.strictEqual(link('adoption', 'ten_conduites_suicidaires').strength, 'moderee', 'two distinct cohorts are not lowered');
});

test('C3: the same source and result never carry two strengths for one experience', () => {
  for (const [aggregate, specific] of [['ten_conduites_suicidaires', 'ten_tentatives_suicide'], ['ten_sante_physique', 'ten_maladies_cardiovasculaires']]) {
    for (const a of data.links.filter(l => l.outcome === aggregate)) {
      const b = link(a.factor, specific);
      if (b && a.sources.slice().sort().join() === b.sources.slice().sort().join()) assert.strictEqual(a.strength, b.strength, a.factor + ' ' + aggregate);
    }
  }
  assert.strictEqual(link('negligence_emotionnelle', 'ten_conduites_suicidaires').strength, 'limitee');
  assert.strictEqual(link('abus_emotionnel', 'ten_conduites_suicidaires').strength, 'moderee', 'an aggregate is not stronger than its components');
  assert.strictEqual(link('abus_sexuel', 'anx_generale').strength, 'moderee');
  assert.match(link('abus_sexuel', 'anx_generale').summary, /qualité/);
});

test('C3: isolation is no longer linked to problematic substance use (Lacey 2014 measures CRP)', () => {
  assert.ok(!link('isolement_social_enfant', 'ten_usage_substances'));
  assert.ok(!data.links.some(l => l.sources.includes('lacey2014')));
  assert.ok(data.links.some(l => l.factor === 'isolement_social_enfant'), 'other isolation links remain');
});

test('C3: wording — before 18, neutral examples, no gendered parent, no duplicate identity example', () => {
  assert.ok(data.groups.every(g => /18 ans|enfance/.test(g.description)), 'every theme situates the period');
  assert.strictEqual(factor('punitions_corporelles').label, 'Fessées, tapes ou gifles données pour éduquer');
  assert.ok(!JSON.stringify(data.factors).includes('« ordinaires »'));
  assert.ok(!factor('parent_absent').description.includes('père'));
  assert.ok(factor('traumatisme_ponctuel').examples.length >= 3);
  assert.ok(!factor('discrimination').examples.some(e => /orientation/.test(e)));
  assert.ok(factor('abus_emotionnel').examples.some(e => /orientation/.test(e)), 'the identity example stays in one place');
});

test('C4: added links cite a source reopened for this review, at a capped strength', () => {
  const added = [['temoin_violences', 'ten_externalisation', 'vu2016_temoin'], ['precarite', 'dep_generale', 'gilman2003_precarite']];
  for (const [f, o, s] of added) {
    const l = link(f, o);
    assert.ok(l, f + '→' + o);
    assert.strictEqual(l.strength, 'limitee', 'abstract-only reading caps the strength');
    assert.strictEqual(l.relation_type, 'empirical_association');
    assert.deepStrictEqual(l.sources, [s]);
    assert.match(source(s).read, /relu le 18\/11\/2026/);
  }
  assert.deepStrictEqual(link('abus_sexuel', 'icd_6b60').sources, ['ludwig2018'], 'Hailes 2019 re-reports Ludwig 2018: no double counting');
  assert.strictEqual(link('abus_sexuel', 'icd_6b60').strength, 'limitee');
});

test('C6: the out-of-scope list is published with existing sources', () => {
  const scope = data.notes.hors_perimetre;
  assert.ok(scope && scope.title === 'Ce que la carte ne couvre pas');
  assert.ok(scope.items.length >= 10);
  for (const item of scope.items) { assert.ok(item.scope && item.reason); assert.ok(!/\d+\s?%/.test(item.reason)); }
  for (const id of ['Autisme', 'Mécanismes', 'Dépendance affective', 'Exil', 'Secrets', 'risque personnel']) assert.ok(scope.items.some(i => i.scope.includes(id)), id);
  for (const id of scope.sources) assert.ok(source(id), id);
});

test('C5: experiences without any link remain valid questions', () => {
  const linked = new Set(data.links.map(l => l.factor));
  const unlinked = data.factors.filter(f => !linked.has(f.id)).map(f => f.id).sort();
  assert.deepStrictEqual(unlinked, ['controle_psychologique', 'migration', 'parents_stricts', 'secret_famille']);
});
