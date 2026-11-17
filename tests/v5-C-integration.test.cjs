// Plan v5.1, lot C5b: the reviewed v5 fragment is integrated exactly as accepted by
// docs/relecture-cartographie-v5-2026-11-19.md §8; refused proposals stay out.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const data = require(path.join(root, 'data/cartographie.json'));
const fragment = require(path.join(root, 'editorial/cartographie/liens-exhaustivite-v5-2026-11-19.json'));
const outcomes = new Map(data.outcomes.map(o => [o.id, o]));
const link = (f, o) => data.links.find(l => l.factor === f && l.outcome === o);

test('accepted subjects and links are present, all limited; refused ones are absent', () => {
  for (const id of ['ten_difficultes_personnalite', 'ten_impulsivite', 'ten_preoccupation_apparence', 'ten_violences_amoureuses_jeunes']) assert.ok(outcomes.has(id), id);
  assert.equal(outcomes.get('ten_difficultes_personnalite').contextual_only, true);
  for (const id of ['ten_traits_psychopathiques', 'ten_defenses_peu_adaptees']) assert.ok(!outcomes.has(id), id);
  assert.ok(!link('exil', 'icd_6a20') && !link('abus_emotionnel', 'pat_self_criticism') && !link('abus_physique', 'ten_violences_amoureuses_jeunes'));
  assert.equal(fragment.links.length, 22);
  for (const l of fragment.links) { const p = link(l.factor, l.outcome); assert.ok(p, l.factor + '>' + l.outcome); assert.equal(p.strength, 'limitee'); assert.equal(p.relation_type, 'empirical_association'); }
  assert.ok(!Object.keys(fragment).some(k => k.startsWith('proposed_') || k === 'out_of_scope'), 'no proposal keys in the definitive fragment');
  assert.ok(!data.sources.some(s => ['deruiter2022_psychopathie', 'martinjoy2017_defenses'].includes(s.id)));
});

test('corrections of existing links and descriptions follow the review', () => {
  assert.equal(link('negligence_physique', 'icd_6a60').strength, 'limitee');
  assert.equal(link('negligence_emotionnelle', 'icd_6a60'), undefined, 'the owner declined displaying this association, including as limited evidence');
  assert.equal(link('parent_trouble', 'icd_6b05').strength, 'moderee');
  assert.deepEqual(link('harcelement', 'icd_6b04').sources.slice().sort(), ['deng2024_anxiete_sociale', 'gonzalez2017']);
  for (const o of ['anx_generale', 'dep_generale']) assert.ok(link('traumatisme_ponctuel', o).sources.includes('rubens2018_catastrophes'));
  const mentions = data.outcomes.filter(o => /dépendance affective/.test(o.description || ''));
  assert.deepEqual(mentions.map(o => o.id), ['att_anxieux'], 'one mention only, on anxious attachment');
  assert.equal(outcomes.get('pat_social_isolation').label, 'Solitude ou isolement subis');
  const scope = data.notes.hors_perimetre.items.map(i => i.scope);
  assert.ok(scope.includes('Traits psychopathiques') && scope.includes('Mécanismes de défense'));
  assert.match(data.notes.hors_perimetre.items.find(i => i.scope.startsWith('Exil')).reason, /troubles psychotiques/);
});
