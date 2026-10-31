(function () {
  'use strict';
  var widgets = Array.from(document.querySelectorAll('[data-affinity-widget]'));
  var likes = new Map();
  var drafts = new Map();
  var user = null;
  var loading;
  var busyIds = new Set();

  async function api(path, method, data) {
    var options = {method: method || 'GET', credentials: 'same-origin', headers: {}};
    if (data !== undefined) { options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(data); }
    var response = await fetch(path, options);
    var result = await response.json();
    if (!response.ok) throw new Error(result.error || 'L’enregistrement n’a pas abouti. Réessayez.');
    return result;
  }
  function group(id) { return widgets.filter(function (w) { return w.dataset.articleId === id; }); }
  function status(id, message, error) {
    group(id).forEach(function (w) {
      var node = w.querySelector('[data-affinity-status]');
      node.textContent = message;
      node.classList.toggle('affinity-error', !!error);
    });
  }
  function render(id) {
    var entry = likes.get(id);
    group(id).forEach(function (w) {
      var b = w.querySelector('[data-like-article]');
      b.setAttribute('aria-pressed', entry ? 'true' : 'false');
      b.replaceChildren(document.createTextNode(entry ? '♥ Cette histoire me parle' : '♡ Ça me parle'));
      b.disabled = busyIds.has(id);
      var panel = w.querySelector('[data-affinity-note]');
      if (panel) {
        panel.hidden = !entry;
        var input = panel.querySelector('[data-like-reason]');
        var value = drafts.has(id) ? drafts.get(id) : entry ? (entry.reason || '') : '';
        if (input.value !== value) input.value = value;
        input.disabled = busyIds.has(id);
        panel.querySelector('[data-save-like-reason]').disabled = busyIds.has(id);
      }
    });
  }
  async function load() {
    var me = await api('/api/me');
    user = me.user;
    if (user) {
      var result = await api('/api/likes');
      (result.likes || []).forEach(function (item) { likes.set(item.article_id, item); });
    }
    new Set(widgets.map(function (w) { return w.dataset.articleId; })).forEach(render);
  }
  if (widgets.length) {
    loading = load();
    loading.catch(function () { /* Retry explicitly when a reader acts. */ });
    widgets.forEach(function (widget) {
      var id = widget.dataset.articleId;
      widget.querySelector('[data-like-article]').addEventListener('click', async function () {
        if (busyIds.has(id)) return;
        busyIds.add(id); render(id);
        try {
          try { await loading; } catch (e) { loading = load(); await loading; }
          if (!user) {
            status(id, 'Connectez-vous pour conserver cette histoire dans vos articles aimés.');
            window.location.assign('/journal/?article=' + encodeURIComponent(id) + '&like=1');
            return;
          }
          var liked = !likes.has(id);
          await api('/api/likes/' + encodeURIComponent(id), 'PUT', {liked: liked});
          if (liked) likes.set(id, {article_id: id, reason: ''});
          else { likes.delete(id); drafts.delete(id); }
          status(id, liked ? 'Article ajouté à votre collection. Vous pouvez expliquer pourquoi, si vous le souhaitez.' : 'Article retiré de votre collection.');
        } catch (e) { status(id, e.message, true); }
        finally { busyIds.delete(id); render(id); }
      });
      var reasonInput = widget.querySelector('[data-like-reason]');
      if (reasonInput) reasonInput.addEventListener('input', function () { drafts.set(id, reasonInput.value); });
      var save = widget.querySelector('[data-save-like-reason]');
      if (save) save.addEventListener('click', async function () {
        if (busyIds.has(id)) return;
        var reason = widget.querySelector('[data-like-reason]').value;
        drafts.set(id, reason);
        busyIds.add(id); render(id);
        try {
          await api('/api/likes/' + encodeURIComponent(id), 'PUT', {liked: true, reason: reason});
          likes.set(id, {article_id: id, reason: reason});
          // Preserve a newer draft if another editor changed while this request was pending.
          if (drafts.get(id) === reason) drafts.delete(id);
          status(id, 'Votre explication est enregistrée. Elle sera prise en compte dans les prochaines analyses de votre portrait.');
        } catch (e) { status(id, e.message, true); }
        finally { busyIds.delete(id); render(id); }
      });
    });
  }

  var search = document.getElementById('character-search');
  var filter = document.getElementById('character-keyword');
  if (search && filter) {
    var cards = Array.from(document.querySelectorAll('[data-character-search]'));
    function normalized(value) { return String(value).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase(); }
    function filterCharacters() {
      var query = normalized(search.value), keyword = normalized(filter.value), count = 0;
      cards.forEach(function (card) {
        var text = normalized(card.dataset.characterSearch);
        var show = (!query || text.includes(query)) && (!keyword || text.includes(keyword));
        card.hidden = !show;
        if (show) { count++; card.classList.add('is-visible'); }
      });
      document.getElementById('character-count').textContent = count + ' personnage' + (count > 1 ? 's' : '') + ' à explorer';
    }
    search.addEventListener('input', filterCharacters);
    filter.addEventListener('change', filterCharacters);
    filterCharacters();
  }
})();
