/* Interface de la Cartographie : sommaire des thèmes, réponses et précisions
   écrites, conséquences de chaque cause, synthèse par domaine, notes d'Ambre,
   enregistrement automatique dans le profil après accord.
   Le calcul vient de cartographie.js (Cartographie.compute). Rien n'est envoyé
   tant que la personne n'a pas choisi d'enregistrer sa cartographie. */
(function () {
  'use strict';

  var STRENGTH_LABEL = { forte: 'Preuves solides', critere: 'Condition du diagnostic (CIM-11)', moderee: 'Preuves modérées', limitee: 'Preuves limitées', theorique: 'Modèle clinique' };
  var STRENGTH_ORDER = ['forte', 'critere', 'moderee', 'limitee', 'theorique'];
  var KIND_LABEL = { trouble: 'Trouble étudié', schema: 'Schéma précoce', attachement: 'Attachement', tendance: 'Tendance' };
  var NOTE_TEXT = { jeune_enfant: 'Diagnostic posé chez le jeune enfant après des soins gravement insuffisants : il ne s’applique pas à un adulte.' };
  var TIERS = {
    convergence: ['Plusieurs de vos réponses y mènent', 'Au moins deux causes y sont associées avec des preuves solides ou modérées, ou au moins trois causes y convergent.'],
    piste: ['Une de vos réponses y mène', 'Une cause y est associée par des preuves solides ou modérées, ou en est une condition dans la classification.'],
    theorique: ['Pistes issues de modèles cliniques ou de preuves limitées', 'Des liens à garder en tête, moins établis.']
  };
  var NOTE_MAX = 1000;

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === 'text') node.textContent = v;
      else if (k === 'className') node.className = v;
      else if (k === 'value') node.value = v;
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

    var state = {
      answers: {}, feedback: {}, notes: { factors: {}, outcomes: {} }, resources: {}, ambreNotes: [],
      theme: null, view: 'form', kind: 'all', open: {}, noteOpen: {},
      account: null, saved: false, version: null, saving: false, saveError: '', savedAt: null
    };
    var themes = data.groups.map(function (g) {
      return { id: g.id, label: g.label, description: g.description, kind: 'causes', items: data.factors.filter(function (f) { return f.group === g.id; }) };
    }).filter(function (t) { return t.items.length; });
    if ((data.protections || []).length) {
      themes.push({ id: 'ressources', label: 'Ce qui m’a aidé', description: 'Les personnes, les lieux et les appuis qui ont compté. Ils éclairent autant votre parcours que ce qui a manqué.', kind: 'ressources', items: data.protections });
    }

    var $ = function (sel) { return root.querySelector(sel); };
    var formBody = $('[data-form-body]'), results = $('[data-results]'), summary = $('[data-summary]');
    var countEls = root.querySelectorAll('[data-count]'), saveBox = $('[data-save]'), dialog = $('[data-dialog]');

    function compute() { return window.Cartographie.compute(data, state.answers); }
    function answered(t) {
      return t.items.filter(function (i) { return t.kind === 'ressources' ? state.resources[i.id] : (state.answers[i.id] || state.notes.factors[i.id]); }).length;
    }
    function ambreFor(target, id) {
      return state.ambreNotes.filter(function (n) { return n.target === target && n.target_id === id; });
    }

    /* ---- Enregistrement ---- */
    var saveTimer = null;
    function changed() {
      state.saveError = '';
      if (!state.saved) return;
      clearTimeout(saveTimer);
      saveTimer = setTimeout(function () { save(false); }, 1200);
    }
    function hasContent() {
      return Object.keys(state.answers).length || Object.keys(state.resources).length || Object.keys(state.notes.factors).length || Object.keys(state.notes.outcomes).length;
    }
    function api(method, path, body) {
      return fetch('/api/cartographie' + (path || ''), { method: method, credentials: 'same-origin', headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined })
        .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { if (!r.ok) { var e = new Error((j.error && j.error.message) || j.message || (typeof j.error === 'string' ? j.error : '') || 'Enregistrement impossible pour le moment.'); e.status = r.status; throw e; } return j; }); });
    }
    function applyServer(j, keepLocal) {
      var c = j && j.cartographie;
      if (!c) return;
      state.saved = true; state.version = c.version || null; state.savedAt = new Date();
      state.ambreNotes = c.ambre_notes || [];
      if (keepLocal) return;
      state.answers = c.answers || {}; state.feedback = c.feedback || {}; state.resources = c.resources || {};
      state.notes = { factors: (c.notes && c.notes.factors) || {}, outcomes: (c.notes && c.notes.outcomes) || {} };
    }
    function save(retried) {
      if (!state.account || !hasContent()) { renderSave(); return; }
      state.saving = true; renderSave();
      api('PUT', '', { answers: state.answers, feedback: state.feedback, notes: state.notes, resources: state.resources, locked: true, consent: true, version: state.version })
        .then(function (j) {
          var before = JSON.stringify(state.ambreNotes), wasSaved = state.saved;
          state.saving = false; applyServer(j, true);
          var typing = document.activeElement && document.activeElement.tagName === 'TEXTAREA';
          if ((before !== JSON.stringify(state.ambreNotes) || !wasSaved) && !typing) renderAll(); else renderSave();
        })
        .catch(function (e) {
          if (e.status === 409 && !retried) {
            return api('GET').then(function (j) { applyServer(j, true); save(true); });
          }
          state.saving = false; state.saveError = e.message; renderSave();
        });
    }
    function removeSaved() {
      if (!window.confirm('Supprimer votre cartographie de votre profil ? Ambre ne pourra plus s’en servir, et ses notes seront effacées.')) return;
      api('DELETE').then(function () { state.saved = false; state.version = null; state.ambreNotes = []; renderAll(); }).catch(function (e) { state.saveError = e.message; renderSave(); });
    }
    function removeAmbreNote(id) {
      api('DELETE', '/ambre-notes/' + encodeURIComponent(id)).then(function (j) { applyServer(j, true); renderAll(); }).catch(function (e) { state.saveError = e.message; renderSave(); });
    }

    function renderSave() {
      if (!saveBox) return;
      saveBox.innerHTML = '';
      if (!state.account) {
        saveBox.appendChild(el('p', {}, ['Vos réponses restent sur cet appareil. ', el('a', { href: '/mon-compte/', text: 'Connectez-vous' }), ' pour les enregistrer : Ambre s’appuiera sur votre cartographie et pourra y noter ce qui ressort de vos échanges.']));
        return;
      }
      if (state.saved) {
        var status = state.saving ? 'Enregistrement…' : (state.saveError ? state.saveError : 'Enregistrée dans votre profil. Chaque modification est sauvegardée automatiquement.');
        saveBox.appendChild(el('p', { className: 'carto-save-status' + (state.saveError ? ' is-error' : ''), role: 'status', text: status }));
        saveBox.appendChild(el('div', { className: 'carto-save-actions' }, [
          el('a', { className: 'carto-btn carto-btn-primary', href: '/mon-accompagnement/', text: 'En parler avec Ambre' }),
          el('button', { type: 'button', className: 'carto-btn carto-btn-quiet', onclick: removeSaved }, ['Supprimer de mon profil'])
        ]));
        return;
      }
      var consent = el('input', { type: 'checkbox', id: 'carto-consent' });
      var msg = el('p', { className: 'carto-save-status is-error', role: 'status', text: state.saveError });
      saveBox.appendChild(el('label', { className: 'carto-consent', for: 'carto-consent' }, [consent, ' J’accepte que mes réponses et mes textes, qui concernent ma santé et mon histoire, soient enregistrés dans mon profil et utilisés par Ambre. Je peux les modifier ou les supprimer à tout moment.']));
      saveBox.appendChild(el('div', { className: 'carto-save-actions' }, [
        el('button', { type: 'button', className: 'carto-btn carto-btn-primary', disabled: !hasContent(), onclick: function () {
          if (!consent.checked) { msg.textContent = 'Cochez l’accord pour enregistrer.'; return; }
          save(false);
        } }, ['Enregistrer ma cartographie'])
      ]));
      saveBox.appendChild(msg);
    }

    /* ---- Notes ---- */
    function ambreNotes(target, id) {
      var notes = ambreFor(target, id);
      if (!notes.length) return null;
      return el('ul', { className: 'carto-ambre-notes' }, notes.map(function (n) {
        return el('li', {}, [
          el('span', { className: 'carto-ambre-tag', text: 'Ambre a noté' }),
          el('span', { className: 'carto-ambre-text', text: n.text }),
          el('button', { type: 'button', className: 'carto-ambre-remove', 'aria-label': 'Retirer cette note d’Ambre', onclick: function () { removeAmbreNote(n.id); } }, ['Retirer'])
        ]);
      }));
    }
    function noteField(kind, id, label, placeholder, fieldLabel) {
      var store = state.notes[kind], key = kind + ':' + id;
      var has = Boolean(store[id]);
      if (!has && !state.noteOpen[key]) {
        return el('button', { type: 'button', className: 'carto-note-add', onclick: function () { state.noteOpen[key] = true; renderAll(); var t = root.querySelector('[data-note="' + key + '"]'); if (t) t.focus(); } }, ['+ ' + label]);
      }
      var area = el('textarea', { className: 'carto-note', 'data-note': key, rows: '3', maxlength: String(NOTE_MAX), placeholder: placeholder, 'aria-label': label, value: store[id] || '' });
      area.addEventListener('input', function () {
        var v = area.value.trim();
        if (v) store[id] = area.value.slice(0, NOTE_MAX); else delete store[id];
        changed(); updateCounts();
      });
      return el('div', { className: 'carto-note-wrap' }, [el('label', { className: 'carto-note-label', text: fieldLabel || label }), area]);
    }

    /* ---- Mon enfance ---- */
    function choice(current, level, label, onPick) {
      var on = (current || 0) === level;
      return el('button', { type: 'button', className: 'carto-choice' + (on ? ' is-on' : ''), 'aria-pressed': on ? 'true' : 'false', onclick: function () { onPick(level); } }, [label]);
    }
    function setAnswer(f, level) {
      if (level) state.answers[f.id] = level; else delete state.answers[f.id];
      if (level) (data.exclusive || []).forEach(function (pair) {
        var i = pair.indexOf(f.id);
        if (i !== -1) delete state.answers[pair[1 - i]];
      });
      changed(); renderAll();
    }

    function renderHome() {
      formBody.innerHTML = '';
      formBody.appendChild(el('p', { className: 'carto-home-intro', text: 'Choisissez un thème. Vous pouvez répondre dans l’ordre que vous voulez, vous arrêter et revenir plus tard.' }));
      var list = el('ul', { className: 'carto-themes' });
      themes.forEach(function (t, i) {
        var n = answered(t);
        list.appendChild(el('li', {}, [el('button', { type: 'button', className: 'carto-theme' + (n ? ' has-answers' : ''), onclick: function () { state.theme = i; renderForm(); scrollToForm(); } }, [
          el('span', { className: 'carto-theme-num', text: String(i + 1) }),
          el('span', { className: 'carto-theme-text' }, [
            el('strong', { text: t.label }),
            el('span', { className: 'carto-theme-desc', text: t.description })
          ]),
          el('span', { className: 'carto-theme-progress', text: n ? n + ' sur ' + t.items.length : t.items.length + ' questions' }),
          el('span', { className: 'carto-theme-arrow', 'aria-hidden': 'true', text: '›' })
        ])]));
      });
      formBody.appendChild(list);
      formBody.appendChild(el('div', { className: 'carto-form-foot' }, [
        hasContent() ? el('button', { type: 'button', className: 'carto-btn carto-btn-quiet', onclick: resetAll }, ['Tout effacer']) : null
      ]));
    }

    function factorRow(f) {
      var nLinks = (linksByFactor[f.id] || []).length;
      var hint = f.description + (f.examples && f.examples.length ? ' Par exemple : ' + f.examples.join(' ; ') + '.' : '');
      return el('li', { className: 'carto-factor-row' + (state.answers[f.id] ? ' is-checked' : '') }, [
        el('strong', { className: 'carto-factor-label', text: f.label }),
        el('p', { className: 'carto-factor-hint', text: hint }),
        el('div', { className: 'carto-factor-actions' }, [
          el('div', { className: 'carto-choices', role: 'group', 'aria-label': f.label }, [
            choice(state.answers[f.id], 0, 'Non', function (l) { setAnswer(f, l); }),
            choice(state.answers[f.id], 1, 'Oui', function (l) { setAnswer(f, l); }),
            choice(state.answers[f.id], 2, 'Souvent', function (l) { setAnswer(f, l); })
          ]),
          nLinks ? el('button', { type: 'button', className: 'carto-see', 'aria-label': 'Voir les ' + nLinks + ' conséquences associées : ' + f.label, onclick: function () { openFactor(f); } }, ['Conséquences (' + nLinks + ')']) : null
        ]),
        noteField('factors', f.id, 'Ajouter une précision', 'Ce dont vous vous souvenez, votre âge, les personnes concernées, ce que vous ressentiez…', 'Ma précision'),
        ambreNotes('factor', f.id)
      ]);
    }
    function resourceRow(r) {
      return el('li', { className: 'carto-factor-row' + (state.resources[r.id] ? ' is-checked' : '') }, [
        el('strong', { className: 'carto-factor-label', text: r.label }),
        el('p', { className: 'carto-factor-hint', text: r.description }),
        el('div', { className: 'carto-factor-actions' }, [
          el('div', { className: 'carto-choices', role: 'group', 'aria-label': r.label }, [
            choice(state.resources[r.id], 0, 'Non', function () { delete state.resources[r.id]; changed(); renderAll(); }),
            choice(state.resources[r.id], 1, 'Oui', function () { state.resources[r.id] = 1; changed(); renderAll(); })
          ])
        ])
      ]);
    }

    function renderTheme() {
      var t = themes[state.theme];
      formBody.innerHTML = '';
      formBody.appendChild(el('button', { type: 'button', className: 'carto-back', onclick: function () { state.theme = null; renderForm(); scrollToForm(); } }, ['‹ Tous les thèmes']));
      formBody.appendChild(el('div', { className: 'carto-step-head' }, [
        el('p', { className: 'carto-step-kicker', text: 'Thème ' + (state.theme + 1) + ' sur ' + themes.length }),
        el('h2', { text: t.label }),
        el('p', { text: t.description })
      ]));
      var list = el('ul', { className: 'carto-factors' });
      t.items.forEach(function (i) { list.appendChild(t.kind === 'ressources' ? resourceRow(i) : factorRow(i)); });
      formBody.appendChild(list);
      var last = state.theme === themes.length - 1;
      formBody.appendChild(el('div', { className: 'carto-step-nav' }, [
        el('button', { type: 'button', className: 'carto-btn', onclick: function () { state.theme = null; renderForm(); scrollToForm(); } }, ['Tous les thèmes']),
        last
          ? el('button', { type: 'button', className: 'carto-btn carto-btn-primary', onclick: function () { setView('results'); } }, ['Voir ce qui en ressort'])
          : el('button', { type: 'button', className: 'carto-btn carto-btn-primary', onclick: function () { state.theme++; renderForm(); scrollToForm(); } }, ['Thème suivant : ' + themes[state.theme + 1].label])
      ]));
    }
    function renderForm() { if (state.theme === null) renderHome(); else renderTheme(); updateCounts(); }
    function scrollToForm() { var p = $('[data-pane-form]'); if (p && p.scrollIntoView) p.scrollIntoView({ block: 'start' }); }
    function resetAll() {
      if (!window.confirm('Effacer toutes vos réponses et vos précisions ?')) return;
      state.answers = {}; state.feedback = {}; state.resources = {}; state.notes = { factors: {}, outcomes: {} }; state.open = {}; state.noteOpen = {}; state.theme = null;
      if (state.saved) { api('DELETE').catch(function () {}); state.saved = false; state.version = null; state.ambreNotes = []; }
      renderAll();
    }

    /* ---- Conséquences d'une cause ---- */
    function sourceList(ids) {
      var src = (ids || []).map(function (id) { return sources[id]; }).filter(Boolean);
      if (!src.length) return null;
      return el('details', { className: 'carto-sources' }, [
        el('summary', { text: 'Sources (' + src.length + ')' }),
        el('ul', {}, src.map(function (s) { return el('li', {}, [el('a', { href: s.url, rel: 'noopener', target: '_blank', text: s.citation })]); }))
      ]);
    }
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
        body.appendChild(el('ul', { className: 'carto-dialog-list' }, items.map(function (l) {
          var o = outcomes[l.outcome];
          return el('li', {}, [
            el('span', { className: 'carto-kind', text: KIND_LABEL[o.kind] + (o.code ? ' · ' + o.code : '') }),
            el('strong', { text: o.label }),
            el('p', { text: l.summary }),
            sourceList(l.sources)
          ]);
        })));
      });
      if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    }
    dialog.querySelector('[data-dialog-close]').addEventListener('click', function () { if (dialog.close) dialog.close(); else dialog.removeAttribute('open'); });
    dialog.addEventListener('click', function (e) { if (e.target === dialog && dialog.close) dialog.close(); });

    /* ---- Ce qui en ressort ---- */
    function renderSummary(s) {
      summary.innerHTML = '';
      if (!s.results.length) return;
      var byDomain = {};
      s.results.forEach(function (r) {
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
      var o = r.outcome, fb = state.feedback[o.id];
      var open = Boolean(state.open[o.id]);
      var children = [el('button', { type: 'button', className: 'carto-card-toggle', 'aria-expanded': open ? 'true' : 'false', onclick: function () { state.open[o.id] = !open; renderResults(); } }, [
        el('span', { className: 'carto-kind', text: KIND_LABEL[o.kind] + (o.code ? ' · ' + o.code : '') }),
        el('span', { className: 'carto-card-title', text: o.label }),
        el('span', { className: 'carto-card-meta', text: r.factors.length + ' cause' + (r.factors.length > 1 ? 's' : '') + ' · ' + STRENGTH_LABEL[r.links[0].strength] + (open ? ' · Replier' : ' · Voir le détail') })
      ])];
      if (open) {
        var body = [];
        if (o.description) body.push(el('p', { className: 'carto-desc', text: o.description }));
        if (o.note && NOTE_TEXT[o.note]) body.push(el('p', { className: 'carto-caution', text: NOTE_TEXT[o.note] }));
        body.push(el('ul', { className: 'carto-links' }, r.links.map(function (l) {
          return el('li', {}, [
            el('div', { className: 'carto-from' }, [el('strong', { text: factors[l.factor].label }), el('span', { className: 'carto-strength carto-strength-' + l.strength, text: STRENGTH_LABEL[l.strength] })]),
            el('p', { className: 'carto-link-text', text: l.summary }),
            sourceList(l.sources)
          ]);
        })));
        if (o.kind === 'trouble') body.push(el('p', { className: 'carto-caution', text: 'Risque moyen observé dans des groupes de personnes, pas votre situation : seul un professionnel peut évaluer un trouble.' }));
        if (o.url) body.push(el('a', { className: 'carto-read', href: o.url, text: o.url.indexOf('/personnages/') === 0 ? 'Découvrir le parcours qui l’illustre →' : 'Lire l’article qui l’illustre →' }));
        children.push(el('div', { className: 'carto-card-body' }, body));
      }
      children.push(el('div', { className: 'carto-feedback', role: 'group', 'aria-label': 'Vous reconnaissez-vous ?' }, [
        el('button', { type: 'button', className: 'carto-fb' + (fb === 'oui' ? ' is-on' : ''), 'aria-pressed': fb === 'oui' ? 'true' : 'false', onclick: function () { setFeedback(o.id, 'oui'); } }, ['Je me reconnais']),
        el('button', { type: 'button', className: 'carto-fb' + (fb === 'non' ? ' is-on' : ''), 'aria-pressed': fb === 'non' ? 'true' : 'false', onclick: function () { setFeedback(o.id, 'non'); } }, ['Pas moi'])
      ]));
      if (fb === 'oui' || open || state.notes.outcomes[o.id]) {
        children.push(el('div', { className: 'carto-card-notes' }, [noteField('outcomes', o.id, 'Décrire comment cela se manifeste aujourd’hui', 'Dans quelles situations, avec qui, ce que vous faites ou ressentez…', 'Comment cela se manifeste aujourd’hui')]));
      }
      var an = ambreNotes('outcome', o.id);
      if (an) children.push(el('div', { className: 'carto-card-notes' }, [an]));
      return el('article', { className: 'carto-card carto-tier-' + r.tier + (fb === 'non' ? ' is-dismissed' : '') }, children);
    }
    function setFeedback(id, value) {
      if (state.feedback[id] === value) delete state.feedback[id]; else state.feedback[id] = value;
      changed(); renderResults();
    }

    function renderResults() {
      var s = compute();
      renderSummary(s);
      results.innerHTML = '';
      updateCounts(s);
      renderSave();
      var general = ambreNotes('general', null);
      if (general) results.appendChild(el('section', { className: 'carto-ambre-general' }, [el('h3', { className: 'carto-tier-title', text: 'Notes d’Ambre' }), general]));
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
        section.appendChild(el('div', { className: 'carto-grid' }, items.map(card)));
        results.appendChild(section);
      });
    }

    function updateCounts(s) {
      s = s || compute();
      var n = Object.keys(state.answers).length;
      var label = n ? n + ' réponse' + (n > 1 ? 's' : '') + ' · ' + s.results.length + ' piste' + (s.results.length > 1 ? 's' : '') : 'Aucune réponse';
      countEls.forEach(function (c) { c.textContent = label; });
    }
    function renderAll() { renderForm(); renderResults(); }

    /* ---- Onglets (mobile) ---- */
    function setView(v) {
      state.view = v;
      root.setAttribute('data-view', v);
      root.querySelectorAll('[data-tab]').forEach(function (t) { t.setAttribute('aria-selected', t.getAttribute('data-tab') === v ? 'true' : 'false'); });
      var target = v === 'results' ? $('[data-pane-results]') : $('[data-pane-form]');
      if (target && target.scrollIntoView && window.matchMedia('(max-width: 900px)').matches) target.scrollIntoView({ block: 'start' });
    }
    root.querySelectorAll('[data-tab]').forEach(function (t) { t.addEventListener('click', function () { setView(t.getAttribute('data-tab')); }); });
    root.querySelectorAll('[data-go-results]').forEach(function (b) { b.addEventListener('click', function () { setView('results'); }); });

    fetch('/api/me', { credentials: 'same-origin' }).then(function (r) { return r.ok ? r.json() : null; }).then(function (me) {
      if (!me || !me.user) return null;
      state.account = me.user;
      return api('GET').then(function (j) { applyServer(j, false); });
    }).catch(function () { /* hors ligne ou aperçu : la page reste utilisable sans compte */ }).then(renderAll);

    window.addEventListener('pageshow', function (e) {
      if (e.persisted && !state.account) { state.answers = {}; state.feedback = {}; state.resources = {}; state.notes = { factors: {}, outcomes: {} }; renderAll(); }
    });
    setView('form');
    renderAll();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var root = document.querySelector('[data-cartographie-app]');
    if (root && window.Cartographie) app(root);
  });
})();
