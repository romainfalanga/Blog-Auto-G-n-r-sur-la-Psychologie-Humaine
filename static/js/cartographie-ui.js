/* Interface de la Cartographie : parcours par étapes, conséquences de chaque
   cause, synthèse par domaine, enregistrement facultatif dans le profil.
   Le calcul vient de cartographie.js (Cartographie.compute). Rien n'est envoyé
   tant que la personne ne choisit pas d'enregistrer sa cartographie. */
(function () {
  'use strict';

  var STRENGTH_LABEL = { forte: 'Preuves solides', critere: 'Condition du diagnostic (CIM-11)', moderee: 'Preuves modérées', limitee: 'Preuves limitées', theorique: 'Modèle clinique' };
  var STRENGTH_ORDER = ['forte', 'critere', 'moderee', 'limitee', 'theorique'];
  var KIND_LABEL = { trouble: 'Trouble étudié', schema: 'Schéma précoce', attachement: 'Attachement', tendance: 'Tendance' };
  var NOTE_TEXT = { jeune_enfant: 'Diagnostic posé chez le jeune enfant après des soins gravement insuffisants : il ne s’applique pas à un adulte.' };
  var TIERS = {
    convergence: ['Plusieurs de vos réponses y mènent', 'Au moins deux causes cochées y sont associées, dont une avec des preuves solides ou modérées.'],
    piste: ['Une de vos réponses y mène', 'Une cause cochée y est associée par des preuves solides ou modérées, ou en est une condition dans la classification.'],
    theorique: ['Pistes issues de modèles cliniques ou de preuves limitées', 'Des liens à garder en tête, moins établis.']
  };

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === 'text') node.textContent = v;
      else if (k === 'className') node.className = v;
      else if (k.indexOf('on') === 0) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return node;
  }

  function app(root) {
    var data = JSON.parse(document.getElementById('cartographie-data').textContent);
    var factors = {}, outcomes = {}, sources = {};
    data.factors.forEach(function (f) { factors[f.id] = f; });
    data.outcomes.forEach(function (o) { outcomes[o.id] = o; });
    data.sources.forEach(function (s) { sources[s.id] = s; });
    var linksByFactor = {};
    data.links.forEach(function (l) { (linksByFactor[l.factor] = linksByFactor[l.factor] || []).push(l); });

    var state = { answers: {}, feedback: {}, step: 0, view: 'form', kind: 'all', open: {}, account: null, saved: null, locked: false, version: null, dirty: false };
    var groups = data.groups.map(function (g) {
      return { group: g, items: data.factors.filter(function (f) { return f.group === g.id; }) };
    }).filter(function (g) { return g.items.length; });

    var $ = function (sel) { return root.querySelector(sel); };
    var stepsNav = $('[data-steps]'), stepBody = $('[data-step-body]'), results = $('[data-results]'), summary = $('[data-summary]');
    var countEls = root.querySelectorAll('[data-count]'), saveBox = $('[data-save]'), dialog = $('[data-dialog]');

    function compute() { return window.Cartographie.compute(data, state.answers); }
    function answeredIn(g) { return g.items.filter(function (f) { return state.answers[f.id]; }).length; }

    /* ---- Formulaire par étapes ---- */
    function renderSteps() {
      stepsNav.innerHTML = '';
      groups.forEach(function (g, i) {
        var n = answeredIn(g);
        stepsNav.appendChild(el('button', {
          type: 'button', className: 'carto-step' + (i === state.step ? ' is-current' : ''), 'aria-current': i === state.step ? 'step' : null,
          onclick: function () { state.step = i; renderForm(); }
        }, [el('span', { className: 'carto-step-num', text: String(i + 1) }), el('span', { className: 'carto-step-label', text: g.group.label }), n ? el('span', { className: 'carto-step-count', text: String(n) }) : null]));
      });
      var current = stepsNav.children[state.step];
      if (current && current.scrollIntoView) current.scrollIntoView({ block: 'nearest', inline: 'center' });
    }

    function choice(f, level, label) {
      var on = (state.answers[f.id] || 0) === level;
      return el('button', {
        type: 'button', className: 'carto-choice' + (on ? ' is-on' : ''), 'aria-pressed': on ? 'true' : 'false',
        onclick: function () {
          if (state.locked) return;
          if (level) state.answers[f.id] = level; else delete state.answers[f.id];
          state.dirty = true; renderForm(); renderResults();
        }, disabled: state.locked
      }, [label]);
    }

    function renderForm() {
      renderSteps();
      var g = groups[state.step];
      stepBody.innerHTML = '';
      stepBody.appendChild(el('div', { className: 'carto-step-head' }, [
        el('h2', { text: g.group.label }),
        el('p', { text: g.group.description })
      ]));
      var list = el('ul', { className: 'carto-factors' });
      g.items.forEach(function (f) {
        var nLinks = (linksByFactor[f.id] || []).length;
        var detailsId = 'carto-hint-' + f.id;
        list.appendChild(el('li', { className: 'carto-factor-row' + (state.answers[f.id] ? ' is-checked' : '') }, [
          el('div', { className: 'carto-factor-main' }, [
            el('strong', { className: 'carto-factor-label', text: f.label }),
            el('p', { className: 'carto-factor-hint', id: detailsId, onclick: function (e) { e.currentTarget.classList.toggle('is-open'); }, text: f.description + (f.examples && f.examples.length ? ' Par exemple : ' + f.examples.join(' ; ') + '.' : '') })
          ]),
          el('div', { className: 'carto-factor-actions' }, [
            el('div', { className: 'carto-choices', role: 'group', 'aria-label': f.label, 'aria-describedby': detailsId }, [choice(f, 0, 'Non'), choice(f, 1, 'Oui'), choice(f, 2, 'Souvent')]),
            nLinks ? el('button', { type: 'button', className: 'carto-see', 'aria-label': 'Voir les ' + nLinks + ' conséquences associées : ' + f.label, onclick: function () { openFactor(f); } }, ['Conséquences (' + nLinks + ')']) : null
          ])
        ]));
      });
      stepBody.appendChild(list);
      stepBody.appendChild(el('div', { className: 'carto-step-nav' }, [
        state.step > 0 ? el('button', { type: 'button', className: 'carto-btn', onclick: function () { state.step--; renderForm(); stepBody.scrollIntoView({ block: 'start' }); } }, ['← Précédent']) : el('span'),
        state.step < groups.length - 1
          ? el('button', { type: 'button', className: 'carto-btn carto-btn-primary', onclick: function () { state.step++; renderForm(); stepBody.scrollIntoView({ block: 'start' }); } }, ['Suivant →'])
          : el('button', { type: 'button', className: 'carto-btn carto-btn-primary', onclick: function () { setView('results'); } }, ['Voir ce qui en ressort →'])
      ]));
      updateCounts();
    }

    /* ---- Conséquences d'une cause ---- */
    function openFactor(f) {
      var body = dialog.querySelector('[data-dialog-body]');
      body.innerHTML = '';
      dialog.querySelector('[data-dialog-title]').textContent = f.label;
      body.appendChild(el('p', { className: 'carto-dialog-intro', text: 'Conséquences que la recherche associe à ce vécu. Elles ne surviennent pas chez tout le monde : ce sont des fréquences plus élevées dans des groupes de personnes.' }));
      var links = (linksByFactor[f.id] || []).slice();
      STRENGTH_ORDER.forEach(function (s) {
        var items = links.filter(function (l) { return l.strength === s && outcomes[l.outcome]; });
        if (!items.length) return;
        body.appendChild(el('h3', { className: 'carto-dialog-h', text: STRENGTH_LABEL[s] + ' (' + items.length + ')' }));
        var ul = el('ul', { className: 'carto-dialog-list' });
        items.forEach(function (l) {
          var o = outcomes[l.outcome];
          ul.appendChild(el('li', {}, [
            el('span', { className: 'carto-kind', text: KIND_LABEL[o.kind] + (o.code ? ' · ' + o.code : '') }),
            el('strong', { text: o.label }),
            el('p', { text: l.summary }),
            sourceList(l.sources)
          ]));
        });
        body.appendChild(ul);
      });
      if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    }

    function sourceList(ids) {
      var src = (ids || []).map(function (id) { return sources[id]; }).filter(Boolean);
      if (!src.length) return null;
      return el('details', { className: 'carto-sources' }, [
        el('summary', { text: 'Sources (' + src.length + ')' }),
        el('ul', {}, src.map(function (s) { return el('li', {}, [el('a', { href: s.url, rel: 'noopener', target: '_blank', text: s.citation })]); }))
      ]);
    }

    /* ---- Résultats ---- */
    function renderSummary(state2) {
      summary.innerHTML = '';
      if (!state2.results.length) return;
      var byDomain = {};
      state2.results.forEach(function (r) {
        var d = r.outcome.domain || 'soi';
        byDomain[d] = byDomain[d] || { score: 0, n: 0, conv: 0 };
        byDomain[d].score += r.score; byDomain[d].n++; if (r.tier === 'convergence') byDomain[d].conv++;
      });
      var max = Math.max.apply(null, Object.keys(byDomain).map(function (k) { return byDomain[k].score; }));
      var list = el('ul', { className: 'carto-domains' });
      (data.domains || []).forEach(function (d) {
        var v = byDomain[d.id];
        if (!v) return;
        list.appendChild(el('li', {}, [
          el('span', { className: 'carto-domain-label', text: d.label }),
          el('span', { className: 'carto-domain-bar', 'aria-hidden': 'true' }, [el('span', { style: 'width:' + Math.max(6, Math.round(100 * v.score / max)) + '%' })]),
          el('span', { className: 'carto-domain-meta', text: v.n + ' piste' + (v.n > 1 ? 's' : '') + (v.conv ? ', dont ' + v.conv + ' en convergence' : '') })
        ]));
      });
      summary.appendChild(el('h3', { className: 'carto-summary-title', text: 'Vue d’ensemble' }));
      summary.appendChild(list);
      summary.appendChild(el('p', { className: 'carto-summary-note', text: 'La longueur des barres compare les domaines entre eux pour vos réponses ; elle ne mesure pas un risque personnel.' }));
    }

    function card(r) {
      var o = r.outcome, open = !!state.open[o.id];
      var fb = state.feedback[o.id];
      var head = el('button', { type: 'button', className: 'carto-card-toggle', 'aria-expanded': open ? 'true' : 'false', onclick: function () { state.open[o.id] = !open; renderResults(); } }, [
        el('span', { className: 'carto-kind', text: KIND_LABEL[o.kind] + (o.code ? ' · ' + o.code : '') }),
        el('span', { className: 'carto-card-title', text: o.label }),
        el('span', { className: 'carto-card-meta', text: r.factors.length + ' cause' + (r.factors.length > 1 ? 's' : '') + ' · ' + STRENGTH_LABEL[r.links[0].strength] })
      ]);
      var children = [head];
      if (open) {
        var body = [];
        if (o.description) body.push(el('p', { className: 'carto-desc', text: o.description }));
        if (o.note && NOTE_TEXT[o.note]) body.push(el('p', { className: 'carto-caution', text: NOTE_TEXT[o.note] }));
        var ul = el('ul', { className: 'carto-links' });
        r.links.forEach(function (l) {
          ul.appendChild(el('li', {}, [
            el('span', { className: 'carto-from' }, [el('strong', { text: factors[l.factor].label }), el('span', { className: 'carto-strength carto-strength-' + l.strength, text: STRENGTH_LABEL[l.strength] })]),
            el('span', { className: 'carto-summary', text: l.summary }),
            sourceList(l.sources)
          ]));
        });
        body.push(ul);
        if (o.kind === 'trouble') body.push(el('p', { className: 'carto-caution', text: 'Risque moyen observé dans des groupes de personnes, pas votre situation : seul un professionnel peut évaluer un trouble.' }));
        if (o.url) body.push(el('a', { className: 'carto-read', href: o.url, text: o.url.indexOf('/personnages/') === 0 ? 'Découvrir le parcours qui l’illustre →' : 'Lire l’article qui l’illustre →' }));
        children.push(el('div', { className: 'carto-card-body' }, body));
      }
      children.push(el('div', { className: 'carto-feedback', role: 'group', 'aria-label': 'Vous reconnaissez-vous ?' }, [
        el('button', { type: 'button', className: 'carto-fb' + (fb === 'oui' ? ' is-on' : ''), 'aria-pressed': fb === 'oui' ? 'true' : 'false', disabled: state.locked, onclick: function () { setFeedback(o.id, 'oui'); } }, ['Je me reconnais']),
        el('button', { type: 'button', className: 'carto-fb' + (fb === 'non' ? ' is-on' : ''), 'aria-pressed': fb === 'non' ? 'true' : 'false', disabled: state.locked, onclick: function () { setFeedback(o.id, 'non'); } }, ['Pas moi'])
      ]));
      return el('article', { className: 'carto-card carto-tier-' + r.tier + (fb === 'non' ? ' is-dismissed' : '') }, children);
    }

    function setFeedback(id, value) {
      if (state.locked) return;
      if (state.feedback[id] === value) delete state.feedback[id]; else state.feedback[id] = value;
      state.dirty = true; renderResults();
    }

    function renderResults() {
      var s = compute();
      renderSummary(s);
      results.innerHTML = '';
      updateCounts(s);
      renderSave();
      if (!s.selectedCount) {
        results.appendChild(el('p', { className: 'carto-empty', text: 'Répondez « Oui » ou « Souvent » aux situations qui correspondent à votre enfance : les conséquences associées apparaîtront ici.' }));
        return;
      }
      if (s.results.some(function (r) { return r.outcome.note === 'suicide'; })) {
        results.appendChild(el('div', { className: 'carto-note-urgent', role: 'note' }, [el('strong', { text: 'Si vous avez des idées suicidaires en ce moment, ' }), 'appelez le 3114 (gratuit, 24 h/24). Danger immédiat : 15 ou 112.']));
      }
      if (s.aceCount >= 4 && data.notes && data.notes.cumul) {
        results.appendChild(el('div', { className: 'carto-note-cumul' }, [el('strong', { text: s.aceCount + ' expériences adverses de l’enfance. ' }), (data.notes.cumul.text || String(data.notes.cumul))]));
      }
      var filters = el('div', { className: 'carto-filters', role: 'group', 'aria-label': 'Filtrer par type' });
      [['all', 'Tout']].concat(Object.keys(KIND_LABEL).map(function (k) { return [k, KIND_LABEL[k]]; })).forEach(function (p) {
        var n = p[0] === 'all' ? s.results.length : s.results.filter(function (r) { return r.outcome.kind === p[0]; }).length;
        if (!n) return;
        filters.appendChild(el('button', { type: 'button', className: 'carto-filter', 'aria-pressed': state.kind === p[0] ? 'true' : 'false', onclick: function () { state.kind = p[0]; renderResults(); } }, [p[1] + ' (' + n + ')']));
      });
      results.appendChild(filters);
      var shown = s.results.filter(function (r) { return state.kind === 'all' || r.outcome.kind === state.kind; });
      ['convergence', 'piste', 'theorique'].forEach(function (t) {
        var items = shown.filter(function (r) { return r.tier === t; });
        if (!items.length) return;
        var section = el(t === 'theorique' ? 'details' : 'section', { className: 'carto-tier' }, [
          el(t === 'theorique' ? 'summary' : 'h3', { className: 'carto-tier-title', text: TIERS[t][0] + ' (' + items.length + ')' }),
          el('p', { className: 'carto-tier-desc', text: TIERS[t][1] })
        ]);
        var grid = el('div', { className: 'carto-grid' });
        items.forEach(function (r) { grid.appendChild(card(r)); });
        section.appendChild(grid);
        results.appendChild(section);
      });
    }

    function updateCounts(s) {
      var n = Object.keys(state.answers).length;
      var label = n ? n + ' réponse' + (n > 1 ? 's' : '') : 'Aucune réponse';
      var r = s ? s.results.length : null;
      countEls.forEach(function (c) { c.textContent = label + (r !== null && n ? ' · ' + r + ' piste' + (r > 1 ? 's' : '') : ''); });
    }

    /* ---- Vue mobile ---- */
    function setView(v) {
      state.view = v;
      root.setAttribute('data-view', v);
      root.querySelectorAll('[data-tab]').forEach(function (t) { t.setAttribute('aria-selected', t.getAttribute('data-tab') === v ? 'true' : 'false'); });
      var target = v === 'results' ? $('[data-pane-results]') : $('[data-pane-form]');
      if (target && window.matchMedia('(max-width: 900px)').matches) target.scrollIntoView({ block: 'start' });
    }
    root.querySelectorAll('[data-tab]').forEach(function (t) { t.addEventListener('click', function () { setView(t.getAttribute('data-tab')); }); });
    root.querySelectorAll('[data-go-results]').forEach(function (b) { b.addEventListener('click', function () { setView('results'); }); });

    $('[data-reset]').addEventListener('click', function () {
      if (state.locked) return;
      state.answers = {}; state.feedback = {}; state.open = {}; state.dirty = true; state.step = 0;
      renderForm(); renderResults();
    });
    dialog.querySelector('[data-dialog-close]').addEventListener('click', function () { if (dialog.close) dialog.close(); else dialog.removeAttribute('open'); });
    dialog.addEventListener('click', function (e) { if (e.target === dialog && dialog.close) dialog.close(); });

    /* ---- Enregistrement dans le profil ---- */
    function renderSave() {
      if (!saveBox) return;
      saveBox.innerHTML = '';
      if (!state.account) {
        saveBox.appendChild(el('p', {}, ['Vos réponses restent sur cet appareil. ', el('a', { href: '/mon-compte/?retour=/cartographie/', text: 'Connectez-vous' }), ' pour les enregistrer dans votre profil : Ambre s’appuiera alors sur votre cartographie.']));
        return;
      }
      if (state.locked) {
        saveBox.appendChild(el('p', { className: 'carto-save-status', text: 'Cartographie enregistrée et verrouillée dans votre profil. Ambre s’en sert comme base de vos échanges.' }));
        saveBox.appendChild(el('div', { className: 'carto-save-actions' }, [
          el('a', { className: 'carto-btn carto-btn-primary', href: '/mon-accompagnement/', text: 'Parler avec Ambre' }),
          el('button', { type: 'button', className: 'carto-btn', onclick: function () { save(false); } }, ['Modifier']),
          el('button', { type: 'button', className: 'carto-btn carto-btn-quiet', onclick: removeSaved }, ['Supprimer de mon profil'])
        ]));
        return;
      }
      var consent = el('input', { type: 'checkbox', id: 'carto-consent' });
      saveBox.appendChild(el('label', { className: 'carto-consent', for: 'carto-consent' }, [consent, ' J’accepte qu’Ambre utilise ces réponses, qui concernent ma santé et mon histoire, pour personnaliser l’accompagnement. Je peux les modifier ou les supprimer à tout moment.']));
      var btn = el('button', { type: 'button', className: 'carto-btn carto-btn-primary', disabled: !Object.keys(state.answers).length, onclick: function () {
        if (!consent.checked) { msg.textContent = 'Cochez l’accord pour enregistrer.'; return; }
        save(true);
      } }, ['Enregistrer et verrouiller dans mon profil']);
      var msg = el('p', { className: 'carto-save-msg', role: 'status' });
      saveBox.appendChild(el('div', { className: 'carto-save-actions' }, [btn]));
      saveBox.appendChild(msg);
    }

    function api(method, body) {
      return fetch('/api/cartographie', { method: method, credentials: 'same-origin', headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined })
        .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { if (!r.ok) throw new Error(j.error || 'Enregistrement impossible pour le moment.'); return j; }); });
    }

    function applySaved(j) {
      var c = j.cartographie || j;
      if (!c || !c.answers) return;
      state.answers = c.answers || {}; state.feedback = c.feedback || {};
      state.locked = Boolean(c.locked_at); state.version = c.version || null; state.dirty = false;
    }

    function save(lock) {
      api('PUT', { answers: state.answers, feedback: state.feedback, locked: lock, consent: true, version: state.version })
        .then(function (j) { applySaved(j); if (!lock) state.locked = false; renderForm(); renderResults(); })
        .catch(function (e) { var m = saveBox.querySelector('.carto-save-msg'); if (m) m.textContent = e.message; else alert(e.message); });
    }

    function removeSaved() {
      if (!window.confirm('Supprimer votre cartographie de votre profil ? Ambre ne pourra plus s’en servir.')) return;
      api('DELETE').then(function () { state.locked = false; state.version = null; renderForm(); renderResults(); }).catch(function (e) { alert(e.message); });
    }

    fetch('/api/me', { credentials: 'same-origin' }).then(function (r) { return r.ok ? r.json() : null; }).then(function (me) {
      if (!me || !me.user) return null;
      state.account = me.user;
      return fetch('/api/cartographie', { credentials: 'same-origin' }).then(function (r) { return r.ok ? r.json() : null; }).then(function (j) { if (j) applySaved(j); });
    }).catch(function () { /* hors ligne ou aperçu : la page reste utilisable sans compte */ }).then(function () { renderForm(); renderResults(); });

    window.addEventListener('pageshow', function (e) { if (e.persisted && !state.account) { state.answers = {}; state.feedback = {}; renderForm(); renderResults(); } });
    setView('form');
    renderForm(); renderResults();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var root = document.querySelector('[data-cartographie-app]');
    if (root && window.Cartographie) app(root);
  });
})();
