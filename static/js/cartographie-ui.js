/* Cartographie : six thèmes indépendants et sauvegarde automatique du compte
   connecté. Sans connexion, les réponses restent dans le brouillon local.
   Le calcul vient de cartographie.js (Cartographie.compute). */
(function () {
  'use strict';

  // Links that are not measured associations: explanations only, never counted.
  var OTHER_RELATIONS = [['clinical_model', 'Repères cliniques']];
  function relationOf(l) { return l.definitional ? 'definitional' : l.relation_type || (l.strength === 'critere' ? 'classification_context' : l.strength === 'theorique' ? 'clinical_model' : 'empirical_association'); }
  function isEmpirical(l) { return relationOf(l) === 'empirical_association'; }
  function plural(n, one, many) { return n + ' ' + (n > 1 ? many : one); }
  var FAMILIAL_TEXT = 'Ce lien reflète en grande partie une transmission familiale (hérédité et environnement partagé), pas seulement l’effet de ce vécu.';
  // Presentation only: the source descriptions, classification and evidence stay intact.
  var PLAIN_DESCRIPTIONS = {
    anx_generale: 'Inquiétudes, peurs ou angoisses durables qui gênent la vie quotidienne.',
    ten_externalisation: 'Opposition, colères, agressivité ou conduites antisociales, observées dans l’enfance, l’adolescence ou plus tard.',
    dep_generale: 'Tristesse, perte d’intérêt, fatigue, pensées de dévalorisation.',
    ten_symptomes_depressifs_perinataux: 'Une souffrance dépressive pendant la grossesse ou l’année qui suit la naissance.',
    icd_6a60: 'Des épisodes maniaques ou hypomaniaques et dépressifs dont la présentation varie selon les personnes.',
    ten_usage_cannabis: 'Un usage de cannabis difficile à maîtriser ou associé à des difficultés dans la vie quotidienne.'
  };
  function outcomeLabel(o) { return o.label.replace(' (preuves agrégées)', ''); }
  function explanation(text) {
    return text.replace(/ selon le DSM-IV/g, '')
      .replace(/^La CIM-11 retient les violences physiques comme événement pouvant précéder un stress post-traumatique ; chez/, 'Chez')
      .replace('le diagnostic CIM-11', 'un diagnostic')
      .replace('La CIM-11 décrit la personnalité par dimensions. ', '')
      .replace('; elle indique les repères de la CIM-11 sur les événements extrêmement menaçants', '');
  }
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
    var factors = {}, outcomes = {};
    data.factors.forEach(function (f) { factors[f.id] = f; });
    (data.retired_outcomes || []).concat(data.outcomes).forEach(function (o) { outcomes[o.id] = o; });
    // Contextual and retired subjects are not displayed, here or in « Liens étudiés ».
    function visible(id) { var o = outcomes[id]; return Boolean(o && !o.retired && !o.contextual_only); }
    var linksByFactor = {};
    data.links.forEach(function (l) { if (visible(l.outcome)) (linksByFactor[l.factor] = linksByFactor[l.factor] || []).push(l); });

    var state = {
      answers: {}, feedback: {}, notes: { factors: {}, outcomes: {} }, resources: {}, ambreNotes: [],
      view: 'form', themesOpen: {}, open: {}, noteOpen: {}, rowOpen: {}, foldOpen: {},
      account: null, accountReady: false, initialized: false, sessionUnavailable: false, mayImportAnonymous: false, loading: true, loadError: false, saved: false, version: 0,
      saving: false, deleting: false, deleted: false, dirty: false, editRevision: 0, conflict: null,
      saveError: '', savedAt: null
    };
    var themes = data.groups.map(function (g) {
      return { id: g.id, label: g.label, description: g.description, kind: 'causes', items: data.factors.filter(function (f) { return f.group === g.id; }) };
    }).filter(function (t) { return t.items.length; });


    var $ = function (sel) { return root.querySelector(sel); };
    var formBody = $('[data-form-body]'), results = $('[data-results]');
    var countEls = root.querySelectorAll('[data-count]'), saveBox = $('[data-save]'), dialog = $('[data-dialog]');
    var noteDialog = $('[data-note-dialog]'), noteDraft = null, previousOverflow = '';
    var noteArea = noteDialog.querySelector('textarea');

    function compute() { return window.Cartographie.compute(data, state.answers, state.feedback); }
    function factorVisible(f) { return !f.historical || Object.prototype.hasOwnProperty.call(state.answers, f.id) || state.notes.factors[f.id] || ambreFor('factor', f.id).length || state.noteOpen['legacy:' + f.id]; }
    function themeItems(t) { return t.items.filter(factorVisible); }
    function answered(t) {
      return themeItems(t).filter(function (i) { return Object.prototype.hasOwnProperty.call(state.answers, i.id) || state.notes.factors[i.id]; }).length;
    }
    function ambreFor(target, id) {
      return state.ambreNotes.filter(function (n) { return n.target === target && n.target_id === id; });
    }

    /* ---- Brouillon local : rien n'est perdu au rechargement, même sans compte ---- */
    function draftKey() { return DRAFT_KEY + (state.account ? ':' + state.account.id : ''); }
    function readDraft(key) {
      try {
        var d = JSON.parse(window.localStorage.getItem(key || draftKey()) || 'null');
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
    // Navigation may carry the first GET responses. Consume each once so retries
    // and conflict refreshes always read the current server state.
    var initialResponses = {};
    try {
      var initialNode = document.getElementById('cartographie-initial');
      var initial = initialNode && JSON.parse(initialNode.textContent);
      if (initial && initial.page === '/cartographie/' && initial.responses && typeof initial.responses === 'object') initialResponses = initial.responses;
    } catch (e) { /* Missing or invalid bootstrap uses the ordinary API path. */ }
    function request(path, options) {
      options = options || {};
      if ((!options.method || options.method === 'GET') && Object.prototype.hasOwnProperty.call(initialResponses, path)) {
        var entry = initialResponses[path];
        delete initialResponses[path];
        if (entry && Number.isInteger(entry.status) && entry.status >= 100 && entry.status <= 599 && Object.prototype.hasOwnProperty.call(entry, 'body')) {
          return Promise.resolve({ ok: entry.status >= 200 && entry.status < 300, status: entry.status, json: function () { return Promise.resolve(entry.body); } });
        }
      }
      return fetch(path, Object.assign({}, options, { cache: 'no-store' }));
    }
    function api(method, path, body) {
      return request('/api/cartographie' + (path || ''), { method: method, credentials: 'same-origin', headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined })
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
          state.saveError = e.status === 409 ? 'Vos réponses enregistrées ont changé ailleurs. Vos réponses ici sont conservées. Choisissez la version à garder avant de reprendre l’enregistrement.' : e.message + ' Vos modifications restent sur cet appareil.';
          if (e.status === 409) { state.conflict = { remote: null }; return fetchConflict(); }
        })
        .then(function () {
          state.saving = false; renderSave();
          // A response acknowledges only the snapshot sent, never edits made while it was in flight.
          if (state.dirty && !state.saveError && !state.conflict && !state.deleting) saveTimer = setTimeout(save, 0);
        });
      return savePromise;
    }
    // One site window replaces the browser's confirm(): it resolves true only on the confirm button.
    function confirmDialog(title, text, confirmLabel, danger) {
      return new Promise(function (resolve) {
        var answer = false, opener = document.activeElement;
        var cancel = el('button', { type: 'button', className: 'carto-btn', onclick: function () { box.close(); } }, ['Annuler']);
        var confirm = el('button', { type: 'button', className: 'carto-btn carto-btn-primary' + (danger ? ' danger' : ''), 'data-confirm': 'true', onclick: function () { answer = true; box.close(); } }, [confirmLabel]);
        var box = el('dialog', { className: 'carto-dialog carto-confirm', 'aria-labelledby': 'carto-confirm-title' }, [
          el('div', { className: 'carto-dialog-head' }, [el('h2', { id: 'carto-confirm-title', text: title })]),
          el('div', { className: 'carto-dialog-body' }, [el('p', { text: text })]),
          el('div', { className: 'carto-note-actions' }, [cancel, confirm])
        ]);
        box.addEventListener('close', function () { box.remove(); if (opener && opener.isConnected && opener.focus) opener.focus(); resolve(answer); });
        root.appendChild(box); box.showModal(); cancel.focus();
      });
    }
    function resolveConflict(keepLocal) {
      var remote = state.conflict && state.conflict.remote;
      if (!remote) { fetchConflict(); return; }
      confirmDialog(keepLocal ? 'Garder les réponses affichées ici ?' : 'Charger la version enregistrée ?', keepLocal ? 'La version enregistrée sera remplacée par les réponses et précisions affichées ici. Les ressources déjà enregistrées seront conservées.' : 'Vos modifications affichées ici seront remplacées par la version enregistrée.', keepLocal ? 'Garder mes réponses' : 'Charger').then(function (yes) { if (yes) applyConflictChoice(keepLocal); });
    }
    function applyConflictChoice(keepLocal) {
      var remote = state.conflict && state.conflict.remote;
      if (!remote) return;
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
      if (state.deleting) return;
      confirmDialog('Effacer vos réponses et vos précisions ?', state.account ? 'Elles seront retirées de votre profil et de cet appareil. Ambre ne pourra plus les utiliser.' : 'Elles seront retirées de cet appareil.', 'Effacer', true).then(function (yes) { if (yes) applyRemoval(); });
    }
    function applyRemoval() {
      if (state.deleting) return;
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
        // While the session check runs, show nothing: a connected member would otherwise
        // see a notice flash and vanish.
        if (state.loading) { saveBox.dataset.state = 'loading'; return; }
        saveBox.dataset.state = 'anonymous';
        saveBox.appendChild(el('p', {}, [state.sessionUnavailable ? 'Connexion non vérifiée. Vos réponses restent sur cet appareil et ne sont pas envoyées à Ambre. ' : 'Sans connexion, vos réponses restent sur cet appareil et ne sont pas envoyées à Ambre. ', state.loading ? null : el('a', { href: '/mon-compte/', text: 'Se connecter' })]));
        if (hasContent()) saveBox.appendChild(el('button', { type: 'button', className: 'carto-btn carto-btn-quiet', disabled: state.deleting, onclick: removeSaved }, ['Effacer mon brouillon']));
        return;
      }
      // Connected members get no standing notice: the box only reports a problem, a pending choice or a deletion.
      saveBox.dataset.state = state.deleting ? 'deleting' : state.saving ? 'saving' : state.saveError ? 'error' : state.loading ? 'loading' : state.conflict ? 'conflict' : state.deleted ? 'deleted' : state.dirty ? 'dirty' : state.saved ? 'saved' : 'idle';
      var status = state.deleting ? 'Suppression…' : state.saveError || (state.deleted ? 'Réponses supprimées. Elles ne seront enregistrées de nouveau que si vous ajoutez une réponse ou une précision.' : '');
      if (status) saveBox.appendChild(el('p', { className: 'carto-save-status' + (state.saveError ? ' is-error' : ''), role: 'status', text: status }));
      var actions = el('div', { className: 'carto-save-actions' });
      if (state.conflict) {
        var remote = state.conflict.remote && state.conflict.remote.cartographie;
        if (state.conflict.remote) {
          saveBox.appendChild(el('p', { className: 'carto-conflict-detail', text: remote ? 'Version enregistrée : ' + Object.keys(remote.answers || {}).length + ' situations sélectionnées ; version affichée ici : ' + Object.keys(state.answers).length + '.' : 'Vos réponses ont été supprimées du profil depuis votre dernière lecture.' }));
          var differences = [];
          ['factors', 'outcomes'].forEach(function (kind) {
            var local = state.notes[kind], stored = remote && remote.notes && remote.notes[kind] || {};
            Object.keys(Object.assign({}, local, stored)).forEach(function (id) {
              if ((local[id] || '') === (stored[id] || '')) return;
              var item = (kind === 'factors' ? factors : outcomes)[id];
              differences.push(el('li', {}, [
                el('strong', { text: item ? item.label : 'Précision' }),
                el('p', { text: 'Ici : ' + (local[id] || 'Aucune précision') }),
                el('p', { text: 'Version enregistrée : ' + (stored[id] || 'Aucune précision') })
              ]));
            });
          });
          if (differences.length) saveBox.appendChild(el('details', { className: 'carto-note-conflicts' }, [
            el('summary', { text: 'Comparer les précisions (' + differences.length + ')' }), el('ul', {}, differences)
          ]));
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
    // Editing a note never mutates the saved document. Only the explicit save
    // joins the existing serialized autosave and account-scoped recovery draft.
    function noteField(kind, id, label, placeholder, fieldLabel) {
      var text = state.notes[kind][id] || '', key = kind + ':' + id;
      var button = el('button', {
        type: 'button', className: 'carto-note-add', 'data-focus': 'note:' + key,
        'aria-haspopup': 'dialog', onclick: function () { openNote(kind, id, label, placeholder, fieldLabel); }
      }, [text ? (kind === 'factors' ? 'Modifier ma précision' : 'Modifier ma description') : kind === 'factors' ? 'Préciser' : label]);
      if (!text) return button;
      return el('div', { className: 'carto-note-wrap' }, [
        el('p', { className: 'carto-note-preview', text: text }), button
      ]);
    }
    function finePointer() {
      try { return typeof window.matchMedia === 'function' && window.matchMedia('(hover: hover) and (pointer: fine)').matches; } catch (error) { return false; }
    }
    function sizeNoteDialog() {
      var viewport = window.visualViewport;
      noteDialog.style.setProperty('--note-viewport-height', (viewport ? viewport.height : window.innerHeight) + 'px');
      noteDialog.style.setProperty('--note-viewport-top', (viewport ? viewport.offsetTop : 0) + 'px');
    }
    function openNote(kind, id, label, placeholder, fieldLabel) {
      noteDraft = { kind: kind, id: id, key: 'note:' + kind + ':' + id, fallback: kind === 'factors' ? 'theme:' + factors[id].group : 'outcome:' + id, initial: state.notes[kind][id] || '' };
      noteDialog.querySelector('[data-note-title]').textContent = (kind === 'factors' ? factors[id] : outcomes[id]).label;
      noteDialog.querySelector('label').textContent = fieldLabel || label;
      noteArea.value = noteDraft.initial;
      noteArea.placeholder = placeholder;
      updateNoteCount(); sizeNoteDialog();
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      noteDialog.showModal();
      // A touch screen gets the heading, so opening a note never summons the keyboard;
      // a mouse and keyboard user can type straight away.
      if (finePointer()) noteArea.focus({ preventScroll: true });
      else noteDialog.querySelector('[data-note-title]').focus({ preventScroll: true });
    }
    function closeNote() { noteDialog.close(); }
    function updateNoteCount() {
      noteDialog.querySelector('[data-note-count]').textContent = noteArea.value.length + ' / ' + NOTE_MAX + ' caractères';
    }
    noteArea.addEventListener('input', updateNoteCount);
    noteDialog.querySelector('[data-note-cancel]').addEventListener('click', closeNote);
    noteDialog.addEventListener('cancel', function (event) { event.preventDefault(); closeNote(); });
    noteDialog.addEventListener('keydown', function (event) {
      if (event.key !== 'Tab') return;
      var controls = Array.from(noteDialog.querySelectorAll('textarea, button')).filter(function (node) { return !node.disabled; });
      var first = controls[0], last = controls[controls.length - 1], active = document.activeElement;
      if (event.shiftKey && (active === first || active === noteDialog.querySelector('[data-note-title]'))) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && active === last) { event.preventDefault(); first.focus(); }
    });
    noteDialog.addEventListener('close', function () {
      var key = noteDraft && noteDraft.key, fallback = noteDraft && noteDraft.fallback;
      noteDraft = null; noteArea.value = '';
      document.body.style.overflow = previousOverflow;
      if (key && !restoreFocus({ key: key })) restoreFocus({ key: fallback });
    });
    noteDialog.querySelector('form').addEventListener('submit', function (event) {
      event.preventDefault();
      if (!noteDraft) return;
      if (noteArea.value === noteDraft.initial) { closeNote(); return; }
      var value = noteArea.value.slice(0, NOTE_MAX), store = state.notes[noteDraft.kind];
      if (!value.trim()) value = '';
      if (value !== noteDraft.initial) {
        if (value) store[noteDraft.id] = value; else delete store[noteDraft.id];
        changed(); renderAll();
      }
      closeNote();
    });
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', sizeNoteDialog);
      window.visualViewport.addEventListener('scroll', sizeNoteDialog);
    }
    window.addEventListener('resize', sizeNoteDialog);

    /* ---- Mon enfance ---- */
    function choice(current, level, label, onPick) {
      // Historical frequency stays stored as 2; the three-choice UI presents it as Yes.
      var on = current === level || current === 2 && level === 1;
      return el('button', { type: 'button', className: 'carto-choice' + (on ? ' is-on' : ''), 'data-level': String(level), 'aria-pressed': on ? 'true' : 'false', onclick: function () { onPick(level); } }, [label]);
    }
    function setAnswer(f, level) {
      if (state.answers[f.id] === level) return;
      state.answers[f.id] = level;
      if (level === 1 || level === 2) (data.exclusive || []).forEach(function (pair) {
        var i = pair.indexOf(f.id);
        if (i !== -1 && (state.answers[pair[1 - i]] === 1 || state.answers[pair[1 - i]] === 2)) delete state.answers[pair[1 - i]];
      });
      changed(); refreshChoices(); renderResults();
    }

    function refreshChoices() {
      formBody.querySelectorAll('[data-factor]').forEach(function (row) {
        var id = row.getAttribute('data-factor'), value = state.answers[id];
        row.classList.toggle('is-checked', value === 1 || value === 2);
        row.querySelectorAll('[data-level]').forEach(function (button) {
          var level = button.getAttribute('data-level');
          var selected = value === (level === 'unknown' ? 'unknown' : Number(level)) || value === 2 && level === '1';
          button.classList.toggle('is-on', selected); button.setAttribute('aria-pressed', selected ? 'true' : 'false');
        });
      });
      themes.forEach(function (theme) {
        var count = root.querySelector('[data-theme-count="' + theme.id + '"]');
        if (count) count.textContent = answered(theme) + ' / ' + themeItems(theme).length + ' renseignés';
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
          el('span', { className: 'carto-theme-progress', 'data-theme-count': theme.id, text: answered(theme) + ' / ' + themeItems(theme).length + ' renseignés' }),
          el('span', { className: 'carto-theme-arrow', 'aria-hidden': 'true', text: '›' })
        ]));
        details.appendChild(el('ul', { className: 'carto-factors' }, themeItems(theme).map(factorRow)));
        details.addEventListener('toggle', function () { if (details.isConnected) state.themesOpen[theme.id] = details.open; });
        list.appendChild(details);
      });
      formBody.appendChild(list);
      formBody.querySelectorAll('[data-factor]').forEach(function (row) { row.querySelectorAll('[data-level]').forEach(function (button) { button.setAttribute('data-focus', 'answer:' + row.getAttribute('data-factor') + ':' + button.getAttribute('data-level')); }); });
      updateCounts();
    }

    function factorRow(f) {
      var nLinks = (linksByFactor[f.id] || []).filter(function (l) { return isEmpirical(l) || relationOf(l) === 'clinical_model'; }).length, value = state.answers[f.id];
      var hint = f.description + (f.examples && f.examples.length ? ' Par exemple : ' + f.examples.join(' ; ') + '.' : '');
      var controls;
      if (f.historical) {
        var answer = value === 'unknown' ? 'Je ne sais pas' : value === 0 ? 'Non' : value === 1 || value === 2 ? 'Oui' : 'Non renseignée';
        controls = el('div', { className: 'carto-legacy-answer' }, [
          el('p', { text: 'Ancienne question — réponse conservée : ' + answer + '.' }),
          el('p', { text: 'Cette question réunissait des règles strictes et du contrôle psychologique. Vous pouvez répondre séparément aux deux questions suivantes ; votre ancienne réponse ne permet pas de choisir à votre place.' }),
          Object.prototype.hasOwnProperty.call(state.answers, f.id) ? el('button', { type: 'button', className: 'carto-note-add', onclick: function () { delete state.answers[f.id]; state.noteOpen['legacy:' + f.id] = true; changed(); renderAll(); var next = root.querySelector('[data-factor="' + f.replaced_by[0] + '"] button'); if (next) next.focus({ preventScroll: true }); } }, ['Effacer cette ancienne réponse']) : null
        ]);
      } else {
        var choices = [choice(value, 0, 'Non', function (l) { setAnswer(f, l); }), choice(value, 1, 'Oui', function (l) { setAnswer(f, l); })];
        choices.push(choice(value, 'unknown', 'Je ne sais pas', function (l) { setAnswer(f, l); }));
        controls = el('div', { className: 'carto-choices', role: 'group', 'aria-label': f.label }, choices);
      }
      var note = noteField('factors', f.id, 'Ajouter une précision', 'Ce dont vous vous souvenez, votre âge, les personnes concernées, ce que vous ressentiez…', 'Votre précision');
      return el('li', { 'data-factor': f.id, className: 'carto-factor-row' + (value === 1 || value === 2 ? ' is-checked' : '') }, [
        el('strong', { className: 'carto-factor-label', text: f.label }),
        el('p', { className: 'carto-factor-hint', text: hint }),
        el('div', { className: 'carto-factor-actions' }, [controls]),
        el('div', { className: 'carto-factor-links' }, [note.tagName === 'BUTTON' ? note : null,
          nLinks ? el('button', { type: 'button', className: 'carto-see', 'aria-label': 'Voir les ' + nLinks + ' associations étudiées : ' + f.label, onclick: function () { openFactor(f); } }, ['Liens étudiés (' + nLinks + ')']) : null
        ]),
        nLinks ? null : el('p', { className: 'carto-no-link', text: f.association_status === 'unreviewed' ? 'Votre réponse peut préciser le contexte pour Ambre. Aucun rapprochement scientifique n’est proposé pour cette nouvelle question.' : 'Aucune association suffisamment documentée n’est retenue pour l’instant.' }),
        note.tagName === 'BUTTON' ? null : note,
        ambreNotes('factor', f.id)
      ]);
    }
    /* ---- Conséquences d'une cause ---- */
    function openFactor(f) {
      var body = dialog.querySelector('[data-dialog-body]');
      body.innerHTML = '';
      dialog.querySelector('[data-dialog-title]').textContent = f.label;
      body.appendChild(el('p', { className: 'carto-dialog-intro', text: 'Associations étudiées et repères cliniques concernant ce vécu. Une association observée dans un groupe ne démontre ni une cause ni un trouble chez vous.' }));
      var links = linksByFactor[f.id] || [];
      var sections = [['Associations étudiées', links.filter(isEmpirical)]]
        .concat(OTHER_RELATIONS.map(function (r) { return [r[1], links.filter(function (l) { return !isEmpirical(l) && relationOf(l) === r[0]; })]; }));
      sections.forEach(function (section) {
        var items = section[1];
        if (!items.length) return;
        body.appendChild(el('h3', { className: 'carto-dialog-h', text: section[0] + ' (' + items.length + ')' }));
        body.appendChild(el('ul', { className: 'carto-dialog-list' }, items.map(function (l) {
          var o = outcomes[l.outcome];
          return el('li', {}, [el('details', { className: 'carto-dialog-item' }, [
            el('summary', { text: outcomeLabel(o) }),
            el('p', { text: explanation(l.summary) })]),
          ]);
        })));
      });
      if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    }
    function openScope() {
      var scope = data.notes && data.notes.hors_perimetre;
      if (!scope) return;
      var body = dialog.querySelector('[data-dialog-body]');
      body.innerHTML = '';
      dialog.querySelector('[data-dialog-title]').textContent = scope.title || 'Ce que la carte ne couvre pas';
      if (scope.text) body.appendChild(el('p', { className: 'carto-dialog-intro', text: scope.text }));
      body.appendChild(el('ul', { className: 'carto-dialog-list carto-scope-list' }, (scope.items || []).filter(function (item) { return item.scope !== 'Modèles de classement HiTOP et AMPD'; }).map(function (item) {
        return el('li', {}, [el('strong', { text: item.scope }), el('p', { text: explanation(item.reason) })]);
      })));
      if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    }
    root.querySelectorAll('[data-scope-open]').forEach(function (b) { if (data.notes && data.notes.hors_perimetre) b.addEventListener('click', openScope); else b.hidden = true; });
    dialog.querySelector('[data-dialog-close]').addEventListener('click', function () { if (dialog.close) dialog.close(); else dialog.removeAttribute('open'); });
    dialog.addEventListener('click', function (e) { if (e.target === dialog && dialog.close) dialog.close(); });

    /* ---- Ce qui en ressort ---- */
    // Checked experiences behind a subject: measured associations first by strength,
    // then experiences linked only by a model. Stored frequency does not change the label.
    function fromLine(r) {
      var seen = {}, names = [];
      r.links.forEach(function (l) {
        if (seen[l.factor]) return;
        seen[l.factor] = true;
        names.push(factors[l.factor].label + (factors[l.factor].historical ? ' (ancienne question)' : ''));
      });
      if (names.length <= 3) return names.join(', ');
      var rest = names.length - 3;
      return names.slice(0, 3).join(', ') + ' et ' + rest + (rest > 1 ? ' autres' : ' autre');
    }
    var familialFactors = {};
    ((data.ranking && data.ranking.familial_factors) || []).forEach(function (id) { familialFactors[id] = true; });
    function card(r) {
      var o = r.outcome, fb = state.feedback[o.id];
      var open = Boolean(state.open[o.id]);
      var children = [
        el('h4', { className: 'carto-card-title', text: outcomeLabel(o) }),
        o.description ? el('p', { className: 'carto-desc', text: PLAIN_DESCRIPTIONS[o.id] || o.description }) : null,
        el('p', { className: 'carto-from-line' }, ['Relié à : ', el('b', { text: fromLine(r) })]),
        r.familial ? el('p', { className: 'carto-familial', text: FAMILIAL_TEXT }) : null,
        el('div', { className: 'carto-feedback', role: 'group', 'aria-label': 'Vous reconnaissez-vous ?' }, [
          el('button', { type: 'button', className: 'carto-fb' + (fb === 'oui' ? ' is-on' : ''), 'data-focus': 'feedback:' + o.id + ':oui', 'aria-pressed': fb === 'oui' ? 'true' : 'false', onclick: function () { setFeedback(o.id, 'oui'); } }, ['Je me reconnais']),
          el('button', { type: 'button', className: 'carto-fb' + (fb === 'non' ? ' is-on' : ''), 'data-focus': 'feedback:' + o.id + ':non', 'aria-pressed': fb === 'non' ? 'true' : 'false', onclick: function () { setFeedback(o.id, 'non'); } }, ['Pas moi']),
          el('button', { type: 'button', className: 'carto-detail', 'data-focus': 'outcome:' + o.id, 'aria-expanded': open ? 'true' : 'false', onclick: function () { state.open[o.id] = !open; renderResults(); } }, [open ? 'Replier' : 'Voir le détail'])
        ])
      ];
      if (open) {
        var body = [];
        var linkItem = function (label) { return function (l) {
          return el('li', {}, [
            el('div', { className: 'carto-from' }, [el('strong', { text: factors[l.factor].label }), label ? el('span', { className: 'carto-relation', text: label(l) }) : null]),
            el('p', { className: 'carto-link-text', text: explanation(l.summary) }),
            familialFactors[l.factor] && isEmpirical(l) ? el('p', { className: 'carto-familial', text: FAMILIAL_TEXT }) : null,
          ]);
        }; };
        var measured = r.links.filter(isEmpirical), others = r.links.filter(function (l) { return relationOf(l) === 'clinical_model'; });
        if (measured.length) body.push(el('ul', { className: 'carto-links' }, measured.map(linkItem(null))));
        if (others.length) {
          body.push(el('p', { className: 'carto-other-title', text: 'Autres repères' }));
          body.push(el('ul', { className: 'carto-links carto-links-other' }, others.map(linkItem(function (l) { return OTHER_RELATIONS.filter(function (x) { return x[0] === relationOf(l); }).map(function (x) { return x[1]; })[0] || 'Repère clinique'; }))));
        }
        children.push(el('div', { className: 'carto-card-body' }, body));
      }
      if (fb === 'oui' || open || state.notes.outcomes[o.id]) {
        children.push(el('div', { className: 'carto-card-notes' }, [noteField('outcomes', o.id, 'Décrire comment cela se manifeste aujourd’hui', 'Dans quelles situations, avec qui, ce que vous faites ou ressentez…', 'Comment cela se manifeste aujourd’hui')]));
      }
      var an = ambreNotes('outcome', o.id);
      if (an) children.push(el('div', { className: 'carto-card-notes' }, [an]));
      return el('article', { className: 'carto-card', 'data-section': r.section, 'data-outcome': o.id, 'data-associated': String(r.associated_count) }, children);
    }
    // A folded subject: its title turns into the full card in place.
    function row(r) {
      var o = r.outcome, open = Boolean(state.rowOpen[o.id]);
      var line = el('button', { type: 'button', className: 'carto-row', 'data-focus': 'row:' + o.id, 'data-outcome': o.id, 'data-section': r.section, 'aria-expanded': open ? 'true' : 'false', onclick: function () { state.rowOpen[o.id] = !open; renderResults(); } }, [
        el('span', { className: 'carto-row-title', text: outcomeLabel(o) })
      ]);
      return el('li', { className: 'carto-row-item' }, open ? [line, card(r)] : [line, o.description ? el('p', { className: 'carto-row-description', text: PLAIN_DESCRIPTIONS[o.id] || o.description }) : null]);
    }
    function fold(name, title, items) {
      var details = el('details', { className: 'carto-fold', 'data-fold': name }, [
        el('summary', { 'data-focus': 'fold:' + name, text: title + ' (' + items.length + ')' }),
        el('ul', { className: 'carto-rows' }, items.map(row))
      ]);
      if (state.foldOpen[name]) details.open = true;
      details.addEventListener('toggle', function () { if (details.isConnected) state.foldOpen[name] = details.open; });
      return details;
    }
    function setFeedback(id, value) {
      if (state.feedback[id] === value) delete state.feedback[id]; else state.feedback[id] = value;
      // The card moves to its new section; a folded section opens so focus can follow it.
      var moved = compute().results.filter(function (r) { return r.outcome.id === id; })[0];
      if (moved && moved.section === 'dismissed') { state.foldOpen[moved.section] = true; state.rowOpen[id] = true; }
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
        results.appendChild(el('p', { className: 'carto-empty', text: 'Les associations étudiées apparaîtront ici si certaines situations correspondent à votre enfance. Les réponses « Non » et « Je ne sais pas » restent enregistrées et ne créent pas d’association.' }));
        return;
      }
      if (!s.results.length) {
        results.appendChild(el('p', { className: 'carto-empty', text: 'Vos réponses restent enregistrées. Aucun rapprochement suffisamment documenté n’est actuellement proposé pour ces vécus.' }));
        return;
      }
      // One list in the engine's order; only « Pas moi » subjects are folded.
      results.appendChild(el('p', { className: 'carto-rank-note', text: 'Classés du plus grand au plus petit nombre de vécus associés ; ce n’est pas une mesure de votre risque.' }));
      results.appendChild(el('div', { className: 'carto-grid' }, s.results.filter(function (r) { return r.section !== 'dismissed'; }).map(card)));
      if (s.sections.dismissed.length) results.appendChild(fold('dismissed', 'Écartés : « Pas moi »', s.sections.dismissed));
      restoreFocus(focused);
    }

    function updateCounts(s) {
      s = s || compute();
      var n = Object.keys(state.answers).length;
      // Only this counter is announced to screen readers, never the rebuilt list.
      var label = n ? plural(n, 'réponse', 'réponses') + ' · ' + plural(s.results.length, 'sujet affiché', 'sujets affichés') : 'Aucune réponse';
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
      if (active) { active.focus({ preventScroll: true }); if (typeof active.setSelectionRange === 'function' && saved.start != null) active.setSelectionRange(saved.start, saved.end); return true; }
    }
    function renderReadiness() {
      var waiting = !state.initialized;
      root.setAttribute('data-loading', waiting ? 'true' : 'false');
      root.setAttribute('aria-busy', waiting ? 'true' : 'false');
      var message = $('[data-loading-status]'); if (message) message.hidden = !waiting;
      root.querySelectorAll('[data-tab]').forEach(function (tab) { tab.disabled = waiting; });
    }
    function renderAll() {
      // Neither anonymous storage nor an empty map is a member's saved map.
      // Build the first view only after session and map loading have settled.
      if (!state.initialized) { renderReadiness(); return; }
      var focused = captureFocus(); renderForm(); renderResults(); restoreFocus(focused); renderReadiness();
    }

    /* ---- Onglets (mobile) ---- */
    function setView(v, scroll) {
      state.view = v;
      root.setAttribute('data-view', v);
      root.querySelectorAll('[data-tab]').forEach(function (t) { t.setAttribute('aria-selected', t.getAttribute('data-tab') === v ? 'true' : 'false'); });
      // Never move the page on load or when switching tabs; only come back up to the
      // tabs when the person had scrolled past them (otherwise the new pane opens mid-way).
      var tabs = $('.carto-tabs');
      if (!scroll || !tabs || !window.matchMedia('(max-width: 900px)').matches) return;
      var top = root.getBoundingClientRect().top + window.pageYOffset;
      if (window.pageYOffset > top) window.scrollTo(0, top);
    }
    root.querySelectorAll('[data-tab]').forEach(function (t) { t.addEventListener('click', function () { setView(t.getAttribute('data-tab'), true); }); });
    root.querySelectorAll('[data-go-results]').forEach(function (b) { b.addEventListener('click', function () { setView('results', true); }); });

    function loadAccountMap() {
      state.loading = true; state.loadError = false; state.saveError = ''; renderSave();
      return api('GET').then(function (j) {
        // Import the local anonymous draft only after an empty account is confirmed.
        // A failed GET must never present another storage scope as this account's map.
        if (!j.cartographie && state.mayImportAnonymous && !state.dirty && !hasContent()) readDraft(DRAFT_KEY);
        state.mayImportAnonymous = false;
        if (state.dirty && j.cartographie) {
          state.conflict = { remote: j };
          state.saveError = 'Des réponses sont déjà enregistrées. Vos réponses ici sont conservées : choisissez la version à garder.';
        } else if (j.cartographie) { applyServer(j, false); clearDraft(); }
        else if (hasContent()) { state.dirty = true; state.editRevision++; }
        state.accountReady = true;
      }).catch(function () { state.loadError = true; state.saveError = 'Impossible de charger vos réponses. Vos modifications restent sur cet appareil ; réessayez avant de les enregistrer.'; })
        .then(function () { state.loading = false; state.initialized = true; renderAll(); if (state.dirty && !state.loadError && !state.conflict) save(); });
    }
    function openAnonymous(unavailable) {
      state.sessionUnavailable = Boolean(unavailable);
      readDraft(DRAFT_KEY);
      state.loading = false; state.initialized = true; renderAll();
    }
    request('/api/me', { credentials: 'same-origin' }).then(function (r) {
      if (!r.ok) throw new Error('Session unavailable');
      return r.json();
    }).then(function (me) {
      if (!me || !me.user) { openAnonymous(false); return; }
      state.account = me.user;
      // An account draft has priority. An anonymous draft may still seed an
      // empty account, as before, but is never painted during the session check.
      if (readDraft()) { state.dirty = true; state.editRevision++; }
      else state.mayImportAnonymous = true;
      return loadAccountMap();
    }).catch(function () { openAnonymous(true); });

    window.addEventListener('beforeunload', function (event) {
      if ((noteDraft && noteArea.value !== noteDraft.initial) || (state.account && (state.dirty || state.saving || state.deleting))) { event.preventDefault(); event.returnValue = ''; }
    });

    setView('form');
    renderAll();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var root = document.querySelector('[data-cartographie-app]');
    if (root && window.Cartographie) app(root);
  });
})();
