/* Shared exploratory calculation. Saving and consent are managed by the UI
   (cartographie-ui.js); this module never estimates an individual risk or
   establishes a diagnosis. The same algorithm, in integers only, is implemented in
   journal/src/cartographie.mjs (computeCartography): tests require strict parity. */
(function (root) {
  'use strict';

  var STRONG = { forte: true, moderee: true };
  // Only empirical associations (strong, moderate or limited) count as an associated
  // experience. Clinical models, definitions and classification contexts weigh little.
  var ASSOCIATED = { forte: true, moderee: true, limitee: true };
  // v5 ranking (variant L with strength guard). Defaults mirror data.ranking.
  var POINTS = { forte: 12, moderee: 8, limitee: 4 };
  var OTHER_POINTS = 1;
  var INTENSITY = { 1: 2, 2: 3 };
  var SECTION_TOP = 5;
  var SECTION_ORDER = ['recognized', 'top', 'frequent', 'other', 'dismissed'];

  // 0 is an explicit no; 1 and 2 are reported experiences.
  function intensities(selected) {
    var map = {};
    if (selected instanceof Set || Array.isArray(selected)) {
      selected.forEach(function (id) { map[id] = 1; });
    } else if (selected && typeof selected === 'object') {
      Object.keys(selected).forEach(function (id) {
        if (selected[id] === 1 || selected[id] === 2) map[id] = selected[id];
      });
    }
    return map;
  }
  function relation(l) { return l.relation_type || (l.definitional ? 'definitional' : l.strength === 'critere' ? 'classification_context' : l.strength === 'theorique' ? 'clinical_model' : 'empirical_association'); }
  function isEmpirical(l) { return !l.definitional && relation(l) === 'empirical_association'; }
  function measured(l) { return isEmpirical(l) && POINTS[l.strength] > 0; }
  function binary(a, b) { return a < b ? -1 : a > b ? 1 : 0; }
  // Relevance order: subjects with a measured association first; strength guard: a
  // subject whose best measured link is limited never precedes a subject with a moderate
  // or strong link; then relevance, evidence families, strongest link, identifier
  // (binary order, no locale).
  // A subject whose measured links all come from a parent's illness or addiction
  // (familial transmission) is treated as limited by the guard (clinical counter-review).
  function evidenceClass(r) { return r.max_points > POINTS.limitee && !r.familial ? 0 : 1; }
  function compareRank(a, b) {
    return Number(b.has_empirical) - Number(a.has_empirical) || evidenceClass(a) - evidenceClass(b) || b.relevance - a.relevance || b.evidence_families - a.evidence_families || b.max_points - a.max_points || binary(a.outcome.id, b.outcome.id);
  }
  // Specificity is capped at 1 (1000) for serious disorders and for links that
  // largely reflect familial transmission (independent clinical review, lot C5a).
  var UNIT = 1000;

  function compute(data, selected, feedback) {
    var level = intensities(selected);
    feedback = feedback || {};
    var factorIndex = {};
    (data.factors || []).forEach(function (f) { factorIndex[f.id] = f; });
    Object.keys(level).forEach(function (id) { if (!factorIndex[id]) delete level[id]; });
    var chosen = new Set(Object.keys(level));
    var weights = data.weights || { forte: 3, moderee: 2, limitee: 1, theorique: 1 };
    var ranking = data.ranking || {};
    var specificity = ranking.specificity || {};
    var frequentSet = new Set(ranking.frequent || []);
    var top = ranking.section_top || SECTION_TOP;
    var familial = new Set(ranking.familial_factors || []), serious = new Set(ranking.serious_outcomes || []);
    var traumaOutcomes = new Set(ranking.trauma_outcomes || []), traumaFactors = new Set(ranking.trauma_factors || []);
    var outcomes = {};
    (data.outcomes || []).forEach(function (o) { outcomes[o.id] = o; });
    var byOutcome = {};
    (data.links || []).forEach(function (link) {
      var o = outcomes[link.outcome];
      // Retired and contextual subjects are never displayed (their stored feedback stays valid).
      if (!chosen.has(link.factor) || !o || o.retired || o.contextual_only) return;
      (byOutcome[link.outcome] = byOutcome[link.outcome] || []).push(link);
    });
    var rows = Object.keys(byOutcome).map(function (id) {
      // Measured associations first, by strength; the other references follow.
      var links = byOutcome[id].slice().sort(function (a, b) { return Number(isEmpirical(b)) - Number(isEmpirical(a)) || weights[b.strength] - weights[a.strength] || binary(a.factor, b.factor); });
      var empirical = links.filter(isEmpirical);
      var associated = [];
      empirical.forEach(function (l) { if (ASSOCIATED[l.strength] && associated.indexOf(l.factor) === -1) associated.push(l.factor); });
      var groups = new Set(empirical.filter(function (l) { return STRONG[l.strength]; }).map(function (l) { return factorIndex[l.factor].overlap_group || l.factor; }));
      // Editorial families reduce overlapping checkboxes. They do not establish
      // statistical independence, diagnostic convergence or increased certainty.
      var tier = groups.size >= 2 ? 'convergence' : groups.size ? 'piste' : 'theorique';
      var score = empirical.reduce(function (best, l) { return Math.max(best, weights[l.strength] || 0); }, 0);
      var families = {}, measuredFamilies = {}, maxPoints = 0, spec = specificity[id] || UNIT, trauma = false, familialOnly = null;
      if (serious.has(id)) spec = Math.min(spec, UNIT);
      links.forEach(function (l) {
        var m = measured(l);
        var pts = (m ? POINTS[l.strength] : OTHER_POINTS) * INTENSITY[level[l.factor]] * (familial.has(l.factor) ? Math.min(spec, UNIT) : spec);
        var fam = factorIndex[l.factor].overlap_group || l.factor;
        if (!(families[fam] >= pts)) families[fam] = pts;
        if (m) { measuredFamilies[fam] = true; maxPoints = Math.max(maxPoints, POINTS[l.strength]); }
        if (m) familialOnly = (familialOnly !== false) && familial.has(l.factor);
        if (m && POINTS[l.strength] >= POINTS.moderee && traumaOutcomes.has(id) && traumaFactors.has(l.factor) && level[l.factor] === 2) trauma = true;
      });
      var points = Object.keys(families).reduce(function (s, k) { return s + families[k]; }, 0);
      return {
        outcome: outcomes[id], links: links, factors: links.map(function (l) { return l.factor; }),
        associated_factors: associated, associated_count: associated.length, other_links: links.length - empirical.length,
        tier: tier, score: score, evidence_groups: groups.size, relation_types: Array.from(new Set(links.map(relation))),
        relevance: points, has_empirical: maxPoints > 0, evidence_families: Object.keys(measuredFamilies).length,
        max_points: maxPoints, frequent: frequentSet.has(id), familial: familialOnly === true, trauma_first: trauma, serious: serious.has(id)
      };
    });
    rows.sort(compareRank);
    // The five first: trauma subjects first chosen, then in rank order; a serious
    // disorder needs a strong link. Kept in rank order inside the section.
    var open = rows.filter(function (r) { return !feedback[r.outcome.id]; });
    var picked = open.filter(function (r) { return r.trauma_first; }).slice(0, top);
    open.forEach(function (r) { if (picked.length < top && picked.indexOf(r) === -1 && !(r.serious && r.max_points < POINTS.forte)) picked.push(r); });
    var sections = { recognized: [], top: [], frequent: [], other: [], dismissed: [] };
    rows.forEach(function (r) {
      var fb = feedback[r.outcome.id];
      r.section = fb === 'oui' ? 'recognized' : fb === 'non' ? 'dismissed' : picked.indexOf(r) !== -1 ? 'top' : r.frequent && r.max_points >= POINTS.moderee ? 'frequent' : 'other';
      sections[r.section].push(r);
    });
    var results = [];
    SECTION_ORDER.forEach(function (s) { results = results.concat(sections[s]); });
    var aceCount = new Set((data.factors || []).filter(function (f) { return f.ace && chosen.has(f.id); }).map(function (f) {
      var exclusive = (data.exclusive || []).find(function (group) { return group.indexOf(f.id) !== -1; });
      return exclusive ? exclusive.join('|') : f.id;
    })).size;
    return { results: results, sections: sections, aceCount: aceCount, selectedCount: chosen.size };
  }

  var api = { compute: compute, intensities: intensities, relation: relation, isEmpirical: isEmpirical, SECTION_ORDER: SECTION_ORDER };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Cartographie = api;
})(this);
