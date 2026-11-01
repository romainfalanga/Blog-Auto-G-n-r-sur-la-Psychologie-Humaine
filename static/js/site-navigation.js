(function () {
  'use strict';
  var toggle = document.getElementById('site-menu-toggle');
  var navigation = document.getElementById('site-navigation');
  if (!toggle || !navigation) return;
  function close() { navigation.classList.remove('is-open'); toggle.setAttribute('aria-expanded', 'false'); }
  toggle.addEventListener('click', function () {
    var open = toggle.getAttribute('aria-expanded') !== 'true';
    navigation.classList.toggle('is-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  });
  navigation.addEventListener('click', function (event) { if (event.target.closest('a,button')) close(); });
  document.addEventListener('click', function (event) { if (!event.target.closest('.site-header')) close(); });
  document.addEventListener('keydown', function (event) { if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') { close(); toggle.focus(); } });
  window.matchMedia('(max-width: 1100px)').addEventListener('change', close);
})();
