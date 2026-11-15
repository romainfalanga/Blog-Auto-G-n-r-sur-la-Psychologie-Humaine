/* Cartographie : relie des vécus de l'enfance aux conséquences psychologiques
   que la recherche leur associe. Tout est calculé dans le navigateur ;
   aucune réponse n'est envoyée ni enregistrée. */
(function (root) {
  'use strict';

  var TIER_ORDER = { convergence: 0, piste: 1, theorique: 2 };
  var STRONG = { forte: true, moderee: true };
  var STRENGTH_LABEL = { forte: 'Preuves solides', moderee: 'Preuves modérées', limitee: 'Preuves limitées', theorique: 'Modèle clinique' };
  var KIND_LABEL = { trouble: 'Trouble (CIM-11)', schema: 'Schéma précoce', attachement: "Style d'attachement", tendance: 'Tendance psychologique' };

  function compute(data, selected) {
    var chosen = selected instanceof Set ? selected : new Set(selected || []);
    var weights = data.weights || { forte: 3, moderee: 2, limitee: 1, theorique: 1 };
    var byOutcome = {};
    (data.links || []).forEach(function (link) {
      if (!chosen.has(link.factor)) return;
      (byOutcome[link.outcome] = byOutcome[link.outcome] || []).push(link);
    });
    var outcomes = {};
    (data.outcomes || []).forEach(function (o) { outcomes[o.id] = o; });
    var results = Object.keys(byOutcome).filter(function (id) { return outcomes[id]; }).map(function (id) {
      var links = byOutcome[id].slice().sort(function (a, b) { return weights[b.strength] - weights[a.strength]; });
      var factors = links.map(function (l) { return l.factor; });
      var strong = links.some(function (l) { return STRONG[l.strength]; });
      var tier = links.length >= 2 && strong ? 'convergence' : (strong ? 'piste' : 'theorique');
      var score = links.reduce(function (sum, l) { return sum + (weights[l.strength] || 1); }, 0);
      return { outcome: outcomes[id], links: links, factors: factors, tier: tier, score: score };
    });
    results.sort(function (a, b) {
      return TIER_ORDER[a.tier] - TIER_ORDER[b.tier] || b.score - a.score || a.outcome.label.localeCompare(b.outcome.label, 'fr');
    });
    var aceCount = (data.factors || []).filter(function (f) { return f.ace && chosen.has(f.id); }).length;
    return { results: results, aceCount: aceCount, selectedCount: chosen.size };
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'className') node.className = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }

  function mount(container) {
    var data = JSON.parse(document.getElementById('cartographie-data').textContent);
    var factorsById = {};
    data.factors.forEach(function (f) { factorsById[f.id] = f; });
    var sourcesById = {};
    data.sources.forEach(function (s) { sourcesById[s.id] = s; });
    var selected = new Set();
    var kindFilter = 'all';
    var form = container.querySelector('[data-carto-form]');
    var output = container.querySelector('[data-carto-results]');
    var counter = container.querySelector('[data-carto-count]');
    var reset = container.querySelector('[data-carto-reset]');
    var filters = container.querySelector('[data-carto-filters]');

    data.groups.forEach(function (group) {
      var fieldset = el('fieldset', { className: 'carto-group' }, [
        el('legend', { text: group.label }),
        el('p', { className: 'carto-group-desc', text: group.description })
      ]);
      data.factors.filter(function (f) { return f.group === group.id; }).forEach(function (f) {
        var input = el('input', { type: 'checkbox', value: f.id, id: 'carto-' + f.id });
        input.addEventListener('change', function () {
          if (input.checked) selected.add(f.id); else selected.delete(f.id);
          render();
        });
        var hint = f.description + (f.examples && f.examples.length ? ' Par exemple : ' + f.examples.join(' ; ') + '.' : '');
        fieldset.appendChild(el('label', { className: 'carto-factor', for: 'carto-' + f.id }, [
          input,
          el('span', { className: 'carto-factor-text' }, [
            el('strong', { text: f.label }),
            el('span', { className: 'carto-factor-hint', text: hint })
          ])
        ]));
      });
      form.appendChild(fieldset);
    });

    if (filters) {
      [['all', 'Tout']].concat(Object.keys(KIND_LABEL).map(function (k) { return [k, KIND_LABEL[k]]; })).forEach(function (pair) {
        var button = el('button', { type: 'button', className: 'carto-filter', 'data-kind': pair[0], 'aria-pressed': pair[0] === 'all' ? 'true' : 'false', text: pair[1] });
        button.addEventListener('click', function () {
          kindFilter = pair[0];
          filters.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', b === button ? 'true' : 'false'); });
          render();
        });
        filters.appendChild(button);
      });
    }

    reset.addEventListener('click', function () {
      selected.clear();
      form.querySelectorAll('input[type=checkbox]').forEach(function (i) { i.checked = false; });
      render();
    });

    function card(result) {
      var o = result.outcome;
      var head = el('div', { className: 'carto-card-head' }, [
        el('span', { className: 'carto-kind carto-kind-' + o.kind, text: KIND_LABEL[o.kind] + (o.code ? ' · ' + o.code : '') }),
        el('h3', { text: o.label })
      ]);
      var list = el('ul', { className: 'carto-links' });
      result.links.forEach(function (l) {
        var src = (l.sources || []).map(function (id) { return sourcesById[id]; }).filter(Boolean);
        var srcList = el('span', { className: 'carto-sources' });
        src.forEach(function (s, i) {
          if (i) srcList.appendChild(document.createTextNode(' · '));
          srcList.appendChild(el('a', { href: s.url, rel: 'noopener', target: '_blank', text: s.citation }));
        });
        list.appendChild(el('li', {}, [
          el('span', { className: 'carto-from' }, [
            el('strong', { text: factorsById[l.factor].label }),
            el('span', { className: 'carto-strength carto-strength-' + l.strength, text: STRENGTH_LABEL[l.strength] })
          ]),
          el('span', { className: 'carto-summary', text: l.summary }),
          srcList
        ]));
      });
      var body = [head, o.description ? el('p', { className: 'carto-desc', text: o.description }) : null, list];
      if (o.url) body.push(el('a', { className: 'carto-read', href: o.url, text: o.url.indexOf('/personnages/') === 0 ? 'Découvrir le parcours qui l’illustre' : 'Lire l’article qui l’illustre' }));
      return el('article', { className: 'carto-card carto-tier-' + result.tier }, body);
    }

    function render() {
      var state = compute(data, selected);
      counter.textContent = state.selectedCount ? state.selectedCount + ' élément' + (state.selectedCount > 1 ? 's' : '') + ' coché' + (state.selectedCount > 1 ? 's' : '') : 'Aucun élément coché';
      output.innerHTML = '';
      if (!state.selectedCount) {
        output.appendChild(el('p', { className: 'carto-empty', text: 'Cochez ce qui correspond à votre enfance : les conséquences associées apparaîtront ici, avec les éléments qui y convergent et les sources.' }));
        return;
      }
      if (state.aceCount >= 4 && data.notes && data.notes.cumul) {
        output.appendChild(el('div', { className: 'carto-note carto-note-cumul' }, [
          el('strong', { text: state.aceCount + ' expériences adverses de l’enfance cochées. ' }),
          el('span', { text: data.notes.cumul.text || data.notes.cumul })
        ]));
      }
      var shown = state.results.filter(function (r) { return kindFilter === 'all' || r.outcome.kind === kindFilter; });
      var tiers = [
        ['convergence', 'Plusieurs éléments convergent', 'Au moins deux éléments cochés sont associés à cette conséquence, dont un avec des preuves solides ou modérées.'],
        ['piste', 'Pistes à explorer', 'Un élément coché est associé à cette conséquence par des preuves solides ou modérées.'],
        ['theorique', 'Liens issus de modèles cliniques ou de preuves limitées', 'Ces liens viennent de modèles thérapeutiques ou d’études encore peu nombreuses.']
      ];
      tiers.forEach(function (t) {
        var items = shown.filter(function (r) { return r.tier === t[0]; });
        if (!items.length) return;
        var section = el('section', { className: 'carto-tier' }, [el('h2', { text: t[1] + ' (' + items.length + ')' }), el('p', { className: 'carto-tier-desc', text: t[2] })]);
        var grid = el('div', { className: 'carto-grid' });
        items.forEach(function (r) { grid.appendChild(card(r)); });
        section.appendChild(grid);
        output.appendChild(section);
      });
      if (!shown.length) output.appendChild(el('p', { className: 'carto-empty', text: 'Aucune conséquence de ce type n’est associée aux éléments cochés.' }));
    }
    render();
  }

  var api = { compute: compute, mount: mount };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Cartographie = api;
  if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', function () {
      var c = document.querySelector('[data-cartographie]');
      if (c) mount(c);
    });
  }
})(this);
