/* Cartographie : six thèmes indépendants et sauvegarde automatique du compte
   connecté. Sans connexion, les réponses restent dans le brouillon local.
   Le calcul vient de cartographie.js (Cartographie.compute). */
(function () {
  'use strict';

  var STRENGTH_LABEL = { forte: 'Preuves solides', critere: 'Contexte décrit par la CIM-11', moderee: 'Preuves modérées', limitee: 'Preuves limitées', theorique: 'Modèle clinique' };
  var STRENGTH_ORDER = ['forte', 'critere', 'moderee', 'limitee', 'theorique'];
  var KIND_LABEL = { trouble: 'Trouble étudié', schema: 'Schéma précoce', attachement: 'Attachement', tendance: 'Tendance' };
  function kindLabel(outcome) { return ({symptom_group: 'Symptômes étudiés', clinical_group: 'Groupe de troubles', pattern_qualifier: 'Qualificatif clinique', personality_trait: 'Trait de personnalité (dimension)', defense_mechanism: 'Mécanisme de défense', schema_mode: 'Mode de schéma (modèle clinique)', schema_coping_style: 'Style d’adaptation aux schémas (modèle clinique)', health_outcome: 'Repère de santé', cognitive_process: 'Processus cognitif', categorical_personality_diagnosis: 'Repère clinique de personnalité'})[outcome.category] || KIND_LABEL[outcome.kind]; }
  var NOTE_TEXT = { jeune_enfant: 'Repère clinique du jeune enfant : ces réponses ne permettent pas de l’attribuer à votre situation adulte.' };
  var NOTE_MAX = 1000;
  var DRAFT_KEY = 'cartographie-brouillon-v1';

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
    (data.retired_outcomes || []).concat(data.outcomes).forEach(function (o) { outcomes[o.id] = o; });
    data.sources.forEach(function (s) { sources[s.id] = s; });
    var linksByFactor = {};
    data.links.forEach(function (l) { (linksByFactor[l.factor] = linksByFactor[l.factor] || []).push(l); });

    var state = {
      answers: {}, feedback: {}, notes: { factors: {}, outcomes: {} }, resources: {}, ambreNotes: [],
      view: 'form', themesOpen: {}, open: {}, noteOpen: {},
      account: null, accountReady: false, loading: true, loadError: false, saved: false, version: 0,
      saving: false, deleting: false, deleted: false, dirty: false, editRevision: 0, conflict: null,
      saveError: '', savedAt: null
    };
    var themes = data.groups.map(function (g) {
      return { id: g.id, label: g.label, description: g.description, kind: 'causes', items: data.factors.filter(function (f) { return f.group === g.id; }) };
    }).filter(function (t) { return t.items.length; });


    var $ = function (sel) { return root.querySelector(sel); };
    var formBody = $('[data-form-body]'), results = $('[data-results]');
    var countEls = root.querySelectorAll('[data-count]'), saveBox = $('[data-save]'), dialog = $('[data-dialog]');

    function compute() { return window.Cartographie.compute(data, state.answers); }
    function answered(t) {
      return t.items.filter(function (i) { return Object.prototype.hasOwnProperty.call(state.answers, i.id) || state.notes.factors[i.id]; }).length;
    }
    function ambreFor(target, id) {
      return state.ambreNotes.filter(function (n) { return n.target === target && n.target_id === id; });
    }

    /* ---- Brouillon local : rien n'est perdu au rechargement, même sans compte ---- */
    function draftKey() { return DRAFT_KEY + (state.account ? ':' + state.account.id : ''); }
    function readDraft() {
      try {
        var d = JSON.parse(window.localStorage.getItem(draftKey()) || 'null');
        if (!d || typeof d !== 'object') return false;
        var known = function (obj, index) { var out = {}; Object.keys(obj || {}).forEach(function (k) { if (index[k]) out[k] = obj[k]; }); return out; };
        var prot = {}; (data.protections || []).forEach(function (x) { prot[x.id] = true; });
        state.answers = known(d.answers, factors); state.feedback = known(d.feedback, outcomes); state.resources = known(d.resources, prot);
        state.notes = { factors: known(d.notes && d.notes.factors, factors), outcomes: known(d.notes && d.notes.outcomes, outcomes) };
        return true;
      } catch (e) { return false; }
    }
    function writeDraft() {
      try {
        if (hasContent() || state.account && state.dirty) window.localStorage.setItem(draftKey(), JSON.stringify({ answers: state.answers, feedback: state.feedback, notes: state.notes, resources: state.resources, at: Date.now() }));
        else window.localStorage.removeItem(draftKey());
      } catch (e) { /* stockage indisponible : la page reste utilisable */ }
    }
    function clearDraft() { try { window.localStorage.removeItem(draftKey()); if (state.account) window.localStorage.removeItem(DRAFT_KEY); } catch (e) { /* idem */ } }

    /* ---- Enregistrement sérialisé et conflits explicites ---- */
    var saveTimer = null, savePromise = Promise.resolve();
    function changed() {
      state.editRevision++; state.dirty = true; state.deleted = false;
      if (!state.conflict && !state.loadError) state.saveError = '';
      writeDraft(); renderSave();
      clearTimeout(saveTimer);
      if (state.accountReady && !state.conflict && !state.loadError && !state.deleting) saveTimer = setTimeout(save, 700);
    }
    function hasContent() {
      return Object.keys(state.answers).length || Object.keys(state.feedback).length || Object.keys(state.resources).length || Object.keys(state.notes.factors).length || Object.keys(state.notes.outcomes).length;
    }
    function api(method, path, body) {
      return fetch('/api/cartographie' + (path || ''), { method: method, credentials: 'same-origin', headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined })
        .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { if (!r.ok) { var e = new Error((j.error && j.error.message) || j.message || (typeof j.error === 'string' ? j.error : '') || 'Enregistrement impossible pour le moment.'); e.status = r.status; throw e; } return j; }); });
    }
    function applyServer(j, keepLocal) {
      var c = j && j.cartographie;
      state.saved = Boolean(c); state.version = c ? c.version : 0;
      state.savedAt = c && c.updated_at ? new Date(c.updated_at * 1000) : null;
      state.ambreNotes = c && c.ambre_notes || [];
      if (keepLocal) return;
      state.answers = c && c.answers || {}; state.feedback = c && c.feedback || {}; state.resources = c && c.resources || {};
      state.notes = { factors: c && c.notes && c.notes.factors || {}, outcomes: c && c.notes && c.notes.outcomes || {} };
    }
    function fetchConflict() {
      return api('GET').then(function (j) { state.conflict = { remote: j }; renderSave(); }).catch(function () {
        state.conflict = { remote: null }; state.saveError = 'La version enregistrée ne peut pas être chargée pour le moment. Vos réponses ici sont conservées.'; renderSave();
      });
    }
    function save() {
      clearTimeout(saveTimer);
      if (!state.accountReady || !state.dirty || state.saving || state.deleting || state.conflict || state.loadError) return savePromise;
      var revision = state.editRevision;
      state.saving = true; state.saveError = ''; renderSave();
      savePromise = api('PUT', '', { answers: state.answers, feedback: state.feedback, notes: state.notes, resources: state.resources, locked: Boolean(hasContent()), consent: true, version: state.version })
        .then(function (j) {
          applyServer(j, true); state.dirty = state.editRevision !== revision;
          if (state.dirty) writeDraft(); else clearDraft();
        })
        .catch(function (e) {
          state.saveError = e.status === 409 ? 'Votre cartographie a changé ailleurs. Vos réponses ici sont conservées. Choisissez la version à garder avant de reprendre l’enregistrement.' : e.message + ' Vos modifications restent sur cet appareil.';
          if (e.status === 409) { state.conflict = { remote: null }; return fetchConflict(); }
        })
        .then(function () {
          state.saving = false; renderSave();
          // A response acknowledges only the snapshot sent, never edits made while it was in flight.
          if (state.dirty && !state.saveError && !state.conflict && !state.deleting) saveTimer = setTimeout(save, 0);
        });
      return savePromise;
    }
    function resolveConflict(keepLocal) {
      var remote = state.conflict && state.conflict.remote;
      if (!remote) { fetchConflict(); return; }
      if (!window.confirm(keepLocal ? 'Remplacer la version enregistrée par les réponses et précisions affichées ici ? Les ressources déjà enregistrées seront conservées.' : 'Charger la version enregistrée et remplacer vos modifications affichées ici ?')) return;
      if (keepLocal) {
        // Resources no longer have controls here: always retain the current stored values.
        if (remote.cartographie) state.resources = remote.cartographie.resources || {};
        applyServer(remote, true); state.dirty = true; state.editRevision++;
      } else { applyServer(remote, false); state.dirty = false; state.deleted = !remote.cartographie; clearDraft(); }
      state.conflict = null; state.saveError = ''; renderAll();
      if (keepLocal) { writeDraft(); save(); }
    }
    function clearAnswers() {
      state.answers = {}; state.feedback = {}; state.resources = {}; state.notes = { factors: {}, outcomes: {} };
      state.ambreNotes = []; state.open = {}; state.noteOpen = {}; state.saved = false; state.version = 0;
      state.dirty = false; state.deleted = true; state.conflict = null; state.saveError = ''; clearDraft();
    }
    function removeSaved() {
      if (state.deleting || !window.confirm('Effacer votre cartographie et vos précisions' + (state.account ? ' de votre profil et de cet appareil ? Ambre ne pourra plus les utiliser.' : ' de cet appareil ?'))) return;
      clearTimeout(saveTimer); state.deleting = true; state.saveError = ''; renderSave();
      formBody.inert = true; results.inert = true;
      // DELETE follows any in-flight PUT. The latter cannot schedule a new save while deleting.
      savePromise.then(function () { return state.account ? api('DELETE') : null; }).then(function () { clearAnswers(); })
        .catch(function (e) { state.saveError = e.message + ' La suppression n’a pas pu être confirmée. Réessayez.'; })
        .then(function () { state.deleting = false; formBody.inert = false; results.inert = false; renderAll(); });
    }
    function removeAmbreNote(id) {
      if (state.deleting) return;
      savePromise.then(function () { return api('DELETE', '/ambre-notes/' + encodeURIComponent(id)); }).then(function (j) {
        state.ambreNotes = j.cartographie && j.cartographie.ambre_notes || [];
        renderAll();
      }).catch(function (e) { state.saveError = e.message; renderSave(); });
    }
    function renderSave() {
      if (!saveBox) return;
      saveBox.innerHTML = '';
      if (!state.account) {
        saveBox.dataset.state = 'anonymous';
        saveBox.appendChild(el('p', {}, [state.loading ? 'Vérification de votre connexion… Vos réponses restent sur cet appareil.' : 'Sans connexion, vos réponses restent sur cet appareil et ne sont pas envoyées à Ambre. ', state.loading ? null : el('a', { href: '/mon-compte/', text: 'Se connecter' })]));
        if (hasContent()) saveBox.appendChild(el('button', { type: 'button', className: 'carto-btn carto-btn-quiet', disabled: state.deleting, onclick: removeSaved }, ['Effacer mon brouillon']));
        return;
      }
      // Connected members get no standing notice: the box only reports a problem, a pending choice or a deletion.
      saveBox.dataset.state = state.deleting ? 'deleting' : state.saving ? 'saving' : state.saveError ? 'error' : state.loading ? 'loading' : state.conflict ? 'conflict' : state.deleted ? 'deleted' : state.dirty ? 'dirty' : state.saved ? 'saved' : 'idle';
      var status = state.deleting ? 'Suppression…' : state.saveError || (state.deleted ? 'Cartographie supprimée. Elle ne sera enregistrée de nouveau que si vous ajoutez une réponse ou une précision.' : '');
      if (status) saveBox.appendChild(el('p', { className: 'carto-save-status' + (state.saveError ? ' is-error' : ''), role: 'status', text: status }));
      var actions = el('div', { className: 'carto-save-actions' });
      if (state.conflict) {
        var remote = state.conflict.remote && state.conflict.remote.cartographie;
        if (state.conflict.remote) {
          saveBox.appendChild(el('p', { className: 'carto-conflict-detail', text: remote ? 'Version enregistrée : ' + Object.keys(remote.answers || {}).length + ' situations sélectionnées ; version affichée ici : ' + Object.keys(state.answers).length + '.' : 'La cartographie a été supprimée du profil depuis votre dernière lecture.' }));
          actions.appendChild(el('button', { type: 'button', className: 'carto-btn', onclick: function () { resolveConflict(false); } }, ['Charger la version enregistrée']));
          actions.appendChild(el('button', { type: 'button', className: 'carto-btn', onclick: function () { resolveConflict(true); } }, ['Garder mes réponses affichées']));
        } else actions.appendChild(el('button', { type: 'button', className: 'carto-btn', onclick: fetchConflict }, ['Recharger la version enregistrée']));
      } else if (state.loadError) actions.appendChild(el('button', { type: 'button', className: 'carto-btn', onclick: loadAccountMap }, ['Réessayer le chargement']));
      else if (state.saveError && state.dirty) actions.appendChild(el('button', { type: 'button', className: 'carto-btn', onclick: save }, ['Réessayer l’enregistrement']));
      if (actions.children.length) saveBox.appendChild(actions);
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
      var fieldId = 'carto-note-' + kind + '-' + id;
      var area = el('textarea', { id: fieldId, className: 'carto-note', 'data-note': key, 'data-focus': 'note:' + key, rows: '3', maxlength: String(NOTE_MAX), placeholder: placeholder, 'aria-label': label + ' : ' + (kind === 'factors' ? factors[id].label : outcomes[id].label), value: store[id] || '' });
      area.addEventListener('input', function () {
        var v = area.value.trim();
        if (v) store[id] = area.value.slice(0, NOTE_MAX); else delete store[id];
        changed(); refreshChoices();
      });
      return el('div', { className: 'carto-note-wrap' }, [el('label', { className: 'carto-note-label', for: fieldId, text: fieldLabel || label }), area]);
    }

    /* ---- Mon enfance ---- */
    function choice(current, level, label, onPick) {
      var on = current === level;
      return el('button', { type: 'button', className: 'carto-choice' + (on ? ' is-on' : ''), 'data-level': String(level), 'aria-pressed': on ? 'true' : 'false', onclick: function () { onPick(level); } }, [label]);
    }
    function setAnswer(f, level) {
      if (state.answers[f.id] === level) return;
      state.answers[f.id] = level;
      if (level) (data.exclusive || []).forEach(function (pair) {
        var i = pair.indexOf(f.id);
        if (i !== -1) delete state.answers[pair[1 - i]];
      });
      changed(); refreshChoices(); renderResults();
    }

    function refreshChoices() {
      formBody.querySelectorAll('[data-factor]').forEach(function (row) {
        var id = row.getAttribute('data-factor'), value = state.answers[id];
        row.classList.toggle('is-checked', Boolean(value));
        row.querySelectorAll('[data-level]').forEach(function (button) {
          var selected = value === Number(button.getAttribute('data-level'));
          button.classList.toggle('is-on', selected); button.setAttribute('aria-pressed', selected ? 'true' : 'false');
        });
      });
      themes.forEach(function (theme) {
        var count = root.querySelector('[data-theme-count="' + theme.id + '"]');
        if (count) count.textContent = answered(theme) + ' / ' + theme.items.length + ' renseignés';
      });
      updateCounts();
    }
    function renderForm() {
      formBody.innerHTML = '';
      var list = el('div', { className: 'carto-themes' });
      themes.forEach(function (theme) {
        var details = el('details', { className: 'carto-theme-panel', 'data-theme': theme.id, open: Boolean(state.themesOpen[theme.id]) });
        details.appendChild(el('summary', { className: 'carto-theme', 'data-focus': 'theme:' + theme.id }, [
          el('span', { className: 'carto-theme-text' }, [el('strong', { text: theme.label }), el('span', { className: 'carto-theme-desc', text: theme.description })]),
          el('span', { className: 'carto-theme-progress', 'data-theme-count': theme.id, text: answered(theme) + ' / ' + theme.items.length + ' renseignés' }),
          el('span', { className: 'carto-theme-arrow', 'aria-hidden': 'true', text: '+' })
        ]));
        details.appendChild(el('ul', { className: 'carto-factors' }, theme.items.map(factorRow)));
        details.addEventListener('toggle', function () { if (details.isConnected) state.themesOpen[theme.id] = details.open; });
        list.appendChild(details);
      });
      formBody.appendChild(list);
      formBody.querySelectorAll('[data-factor]').forEach(function (row) { row.querySelectorAll('[data-level]').forEach(function (button) { button.setAttribute('data-focus', 'answer:' + row.getAttribute('data-factor') + ':' + button.getAttribute('data-level')); }); });
      updateCounts();
    }

    function factorRow(f) {
      var nLinks = (linksByFactor[f.id] || []).length;
      var hint = f.description + (f.examples && f.examples.length ? ' Par exemple : ' + f.examples.join(' ; ') + '.' : '');
      return el('li', { 'data-factor': f.id, className: 'carto-factor-row' + (state.answers[f.id] ? ' is-checked' : '') }, [
        el('strong', { className: 'carto-factor-label', text: f.label }),
        el('p', { className: 'carto-factor-hint', text: hint }),
        el('div', { className: 'carto-factor-actions' }, [
          el('div', { className: 'carto-choices', role: 'group', 'aria-label': f.label }, [
            choice(state.answers[f.id], 0, 'Non', function (l) { setAnswer(f, l); }),
            choice(state.answers[f.id], 1, 'Oui', function (l) { setAnswer(f, l); }),
            choice(state.answers[f.id], 2, 'Souvent', function (l) { setAnswer(f, l); })
          ]),
          nLinks ? el('button', { type: 'button', className: 'carto-see', 'aria-label': 'Voir les ' + nLinks + ' associations étudiées : ' + f.label, onclick: function () { openFactor(f); } }, ['Liens étudiés (' + nLinks + ')']) : null
        ]),
        noteField('factors', f.id, 'Ajouter une précision', 'Ce dont vous vous souvenez, votre âge, les personnes concernées, ce que vous ressentiez…', 'Ma précision'),
        ambreNotes('factor', f.id)
      ]);
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
      body.appendChild(el('p', { className: 'carto-dialog-intro', text: 'Associations étudiées et repères cliniques concernant ce vécu. Une association observée dans un groupe ne démontre ni une cause ni un trouble chez vous.' }));
      var links = (linksByFactor[f.id] || []).slice();
      STRENGTH_ORDER.forEach(function (s) {
        var items = links.filter(function (l) { return l.strength === s && outcomes[l.outcome]; });
        if (!items.length) return;
        body.appendChild(el('h3', { className: 'carto-dialog-h', text: STRENGTH_LABEL[s] + ' (' + items.length + ')' }));
        body.appendChild(el('ul', { className: 'carto-dialog-list' }, items.map(function (l) {
          var o = outcomes[l.outcome];
          return el('li', {}, [
            el('span', { className: 'carto-kind', text: kindLabel(o) + (o.code ? ' · ' + o.code : '') }),
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
    function card(r) {
      var o = r.outcome, fb = state.feedback[o.id];
      var open = Boolean(state.open[o.id]);
      var children = [el('button', { type: 'button', className: 'carto-card-toggle', 'data-focus': 'outcome:' + o.id, 'aria-expanded': open ? 'true' : 'false', onclick: function () { state.open[o.id] = !open; renderResults(); } }, [
        el('span', { className: 'carto-kind', text: kindLabel(o) + (o.code ? ' · ' + o.code : '') }),
        el('span', { className: 'carto-card-title', text: o.label }),
        el('span', { className: 'carto-card-meta', text: r.factors.length + ' vécu' + (r.factors.length > 1 ? 's' : '') + ' associé' + (r.factors.length > 1 ? 's' : '') + ' · ' + STRENGTH_LABEL[r.links[0].strength] + (open ? ' · Replier' : ' · Voir le détail') })
      ])];
      if (o.contextual_only && !(o.note && NOTE_TEXT[o.note])) children.push(el('p', { className: 'carto-caution carto-card-caution', text: 'Repère de contexte : Ambre ne vous le suggère pas à partir de ces réponses.' }));
      if (o.note && NOTE_TEXT[o.note]) children.push(el('p', { className: 'carto-caution carto-card-caution', text: NOTE_TEXT[o.note] }));
      if (open) {
        var body = [];
        if (o.description) body.push(el('p', { className: 'carto-desc', text: o.description }));
        body.push(el('ul', { className: 'carto-links' }, r.links.map(function (l) {
          return el('li', {}, [
            el('div', { className: 'carto-from' }, [el('strong', { text: factors[l.factor].label }), el('span', { className: 'carto-strength carto-strength-' + l.strength, text: STRENGTH_LABEL[l.strength] })]),
            el('p', { className: 'carto-link-text', text: l.summary }),
            sourceList(l.sources)
          ]);
        })));
        if (o.kind === 'trouble') body.push(el('p', { className: 'carto-caution', text: 'Ces rapprochements ne décrivent pas votre situation clinique et ne mesurent pas votre risque personnel. Seul un professionnel peut évaluer un trouble.' }));
        children.push(el('div', { className: 'carto-card-body' }, body));
      }
      children.push(el('div', { className: 'carto-feedback', role: 'group', 'aria-label': 'Vous reconnaissez-vous ?' }, [
        el('button', { type: 'button', className: 'carto-fb' + (fb === 'oui' ? ' is-on' : ''), 'data-focus': 'feedback:' + o.id + ':oui', 'aria-pressed': fb === 'oui' ? 'true' : 'false', onclick: function () { setFeedback(o.id, 'oui'); } }, ['Je me reconnais']),
        el('button', { type: 'button', className: 'carto-fb' + (fb === 'non' ? ' is-on' : ''), 'data-focus': 'feedback:' + o.id + ':non', 'aria-pressed': fb === 'non' ? 'true' : 'false', onclick: function () { setFeedback(o.id, 'non'); } }, ['Pas moi'])
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
      var focused = captureFocus();
      var s = compute();
      results.innerHTML = '';
      updateCounts(s);
      renderSave();
      var general = ambreNotes('general', null);
      if (general) results.appendChild(el('section', { className: 'carto-ambre-general' }, [el('h3', { className: 'carto-tier-title', text: 'Notes d’Ambre' }), general]));
      if (!s.selectedCount) {
        results.appendChild(el('p', { className: 'carto-empty', text: 'Les associations étudiées apparaîtront ici si certaines situations correspondent à votre enfance. Une réponse « Non » reste enregistrée et ne crée pas d’association.' }));
        return;
      }
      if (!s.results.length) {
        results.appendChild(el('p', { className: 'carto-empty', text: 'Vos réponses restent enregistrées. Aucun rapprochement suffisamment documenté n’est actuellement proposé pour ces vécus.' }));
        return;
      }
      // One list, from the most associated experiences to the fewest; the engine's order breaks ties.
      var shown = s.results.map(function (r, index) { return { r: r, index: index }; })
        .sort(function (a, b) { return b.r.factors.length - a.r.factors.length || a.index - b.index; })
        .map(function (item) { return item.r; });
      results.appendChild(el('div', { className: 'carto-grid' }, shown.map(card)));
      restoreFocus(focused);
    }

    function updateCounts(s) {
      s = s || compute();
      var n = Object.keys(state.answers).length;
      var label = n ? n + ' réponse' + (n > 1 ? 's' : '') + ' · ' + s.results.length + ' association' + (s.results.length > 1 ? 's' : '') : 'Aucune réponse';
      countEls.forEach(function (c) { c.textContent = label; });
    }
    function captureFocus() {
      var active = document.activeElement;
      if (!active || !root.contains(active)) return null;
      var key = active.getAttribute('data-focus');
      if (!key) return null;
      return { key: key, start: active.selectionStart, end: active.selectionEnd };
    }
    function restoreFocus(saved) {
      if (!saved) return;
      var active = Array.from(root.querySelectorAll('[data-focus]')).find(function (node) { return node.getAttribute('data-focus') === saved.key; });
      if (active) { active.focus({ preventScroll: true }); if (typeof active.setSelectionRange === 'function' && saved.start != null) active.setSelectionRange(saved.start, saved.end); }
    }
    function renderAll() { var focused = captureFocus(); renderForm(); renderResults(); restoreFocus(focused); }

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

    function loadAccountMap() {
      state.loading = true; state.loadError = false; state.saveError = ''; renderSave();
      return api('GET').then(function (j) {
        if (state.dirty && j.cartographie) {
          state.conflict = { remote: j };
          state.saveError = 'Une cartographie est déjà enregistrée. Vos réponses ici sont conservées : choisissez la version à garder.';
        } else if (j.cartographie) { applyServer(j, false); clearDraft(); }
        else if (hasContent()) { state.dirty = true; state.editRevision++; }
        state.accountReady = true;
      }).catch(function () { state.loadError = true; state.saveError = 'Impossible de charger votre cartographie. Vos modifications restent sur cet appareil ; réessayez avant de les enregistrer.'; })
        .then(function () { state.loading = false; renderAll(); if (state.dirty && !state.loadError && !state.conflict) save(); });
    }
    fetch('/api/me', { credentials: 'same-origin' }).then(function (r) { return r.ok ? r.json() : null; }).then(function (me) {
      if (!me || !me.user) { state.loading = false; renderSave(); return; }
      state.account = me.user;
      if (!state.dirty && readDraft()) { state.dirty = true; state.editRevision++; }
      return loadAccountMap();
    }).catch(function () { state.loading = false; renderSave(); });

    window.addEventListener('beforeunload', function (event) {
      if (state.account && (state.dirty || state.saving || state.deleting)) { event.preventDefault(); event.returnValue = ''; }
    });

    readDraft();
    setView('form');
    renderAll();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var root = document.querySelector('[data-cartographie-app]');
    if (root && window.Cartographie) app(root);
  });
})();
