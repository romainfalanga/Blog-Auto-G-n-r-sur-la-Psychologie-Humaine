(function () {
  'use strict';
  if (window.parent !== window || document.getElementById('ambre-launcher')) return;
  function init() {
    if (document.getElementById('ambre-launcher')) return;
    var launcher = document.createElement('button');
    launcher.id = 'ambre-launcher'; launcher.type = 'button'; launcher.className = 'ambre-launcher';
    launcher.setAttribute('aria-label', 'Ouvrir la conversation avec Ambre');
    launcher.setAttribute('aria-haspopup', 'dialog'); launcher.setAttribute('aria-controls', 'ambre-panel');
    launcher.innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z"/><path d="M8 11h8M8 14h5"/></svg><span>Ambre</span>';
    var panel = document.createElement('dialog'); panel.id = 'ambre-panel'; panel.className = 'ambre-panel';
    panel.setAttribute('aria-labelledby', 'ambre-panel-title');
    panel.innerHTML = '<div class="ambre-panel-bar"><h2 id="ambre-panel-title" tabindex="-1">Ambre</h2><a href="/mon-accompagnement/" aria-label="Ouvrir Ambre en pleine page" title="Ouvrir en pleine page">↗</a><button type="button" class="ambre-panel-close" aria-label="Fermer la conversation">×</button></div><div class="ambre-panel-status" role="status"><p>Ouverture de la conversation…</p><button type="button" hidden>Réessayer</button></div><div class="ambre-panel-content"></div>';
    document.body.append(launcher, panel);
    var content = panel.querySelector('.ambre-panel-content'), status = panel.querySelector('.ambre-panel-status');
    var retry = status.querySelector('button'), frame = null, timeout = null, opening = false, previousOverflow = '', channel = null, verifiedAt = 0;
    var allowed = ['/mon-accompagnement/', '/mon-compte/', '/mon-portrait/', '/mes-proches/', '/ma-ligne-de-vie/', '/cartographie/', '/mes-bilans/', '/parametres/', '/mon-suivi/', '/espace-psychologue/'];
    function send(type) { if (frame && frame.contentWindow) frame.contentWindow.postMessage({channel:'dte-ambre',version:1,type:type}, location.origin); }
    function busy() { if (frame) frame.hidden = true; status.hidden = false; retry.hidden = true; status.querySelector('p').textContent = 'Ouverture de la conversation…'; launcher.setAttribute('aria-busy', 'true'); }
    function failure() { clearTimeout(timeout); opening = false; launcher.removeAttribute('aria-busy'); status.hidden = false; retry.hidden = false; status.querySelector('p').textContent = 'La conversation n’a pas pu s’ouvrir. Vous pouvez réessayer.'; }
    function reset() { clearTimeout(timeout); if (frame) frame.remove(); frame = null; opening = false; if (panel.open) open(); }
    function create() {
      frame = document.createElement('iframe'); frame.title = 'Conversation avec Ambre'; frame.hidden = true;
      frame.src = '/mon-accompagnement/?embedded=1'; frame.setAttribute('allow', 'microphone');
      // Only our same-origin conversation route permits embedding. No arbitrary URL is accepted.
      content.append(frame);
    }
    function open() {
      if (opening) return; opening = true; busy();
      if (!panel.open) { previousOverflow = document.documentElement.style.overflow; document.documentElement.style.overflow = 'hidden'; panel.showModal(); panel.querySelector('h2').focus(); }
      clearTimeout(timeout); timeout = setTimeout(failure, 30000);
      if (!frame) create(); else send('open');
    }
    function close() { clearTimeout(timeout); opening = false; launcher.removeAttribute('aria-busy'); send('close'); if (panel.open) panel.close(); document.documentElement.style.overflow = previousOverflow; if (!launcher.hidden) launcher.focus(); }
    function visibility() { launcher.hidden = /^\/mon-accompagnement\/?$/.test(location.pathname); if (launcher.hidden && panel.open) close(); }
    panel.addEventListener('keydown', function (event) {
      if (event.key !== 'Tab') return;
      var first = panel.querySelector('.ambre-panel-bar a'), last = panel.querySelector('.ambre-panel-close');
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); if (frame && !frame.hidden) send('focus-last'); else last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); if (frame && !frame.hidden) send('focus-first'); else if (!retry.hidden) retry.focus(); else first.focus(); }
      else if (!event.shiftKey && document.activeElement === retry) { event.preventDefault(); first.focus(); }
    });
    launcher.addEventListener('click', open);
    retry.addEventListener('click', reset);
    panel.querySelector('.ambre-panel-close').addEventListener('click', close);
    panel.addEventListener('cancel', function (event) { event.preventDefault(); close(); });
    panel.addEventListener('click', function (event) { if (event.target === panel) { var box = panel.getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) close(); } });
    window.addEventListener('message', function (event) {
      var data = event.data;
      if (!frame || event.origin !== location.origin || event.source !== frame.contentWindow || !data || data.channel !== 'dte-ambre' || data.version !== 1) return;
      if (data.type === 'ready' && panel.open && opening) { verifiedAt = Date.now(); clearTimeout(timeout); opening = false; launcher.removeAttribute('aria-busy'); status.hidden = true; frame.hidden = false; }
      if (data.type === 'close') close();
      if (data.type === 'focus-header' && panel.open) panel.querySelector('.ambre-panel-bar a').focus();
      if (data.type === 'focus-close' && panel.open) panel.querySelector('.ambre-panel-close').focus();
      if (data.type === 'session-changed') { document.dispatchEvent(new CustomEvent('dte:session-invalidated')); if (channel) channel.postMessage({type:'session-changed'}); }
      if (data.type === 'navigate' && typeof data.path === 'string') {
        var target; try { target = new URL(data.path, location.origin); } catch (_) { return; }
        if (target.origin !== location.origin || allowed.indexOf(target.pathname.replace(/\/+$/, '') + '/') === -1) return;
        location.assign(target.pathname + target.search + target.hash);
      }
    });
    document.addEventListener('dte:session-changed', function () { reset(); if (channel) channel.postMessage({type:'session-changed'}); });
    try { channel = new BroadcastChannel('dte-session'); channel.onmessage = function (event) { if (event.data && event.data.type === 'session-changed') { reset(); document.dispatchEvent(new CustomEvent('dte:session-invalidated')); } }; } catch (_) {}
    // Returning to a tab revalidates the session before exposing its old frame.
    document.addEventListener('visibilitychange', function () { if (document.visibilityState === 'hidden' && panel.open) { if (frame) frame.hidden = true; } else if (document.visibilityState === 'visible' && panel.open) open(); });
    window.addEventListener('focus', function () { if (panel.open && Date.now() - verifiedAt > 30000) open(); });
    document.addEventListener('dte:page-changed', visibility); window.addEventListener('popstate', visibility); visibility();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once:true}); else init();
})();
