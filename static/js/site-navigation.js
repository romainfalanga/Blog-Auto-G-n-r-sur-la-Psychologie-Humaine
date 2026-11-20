(function () {
  'use strict';
  // Safari applies :active to a touched link or button only when a touch listener exists: the pressed
  // state of cards and buttons then answers the finger at once (passive, no effect on scrolling).
  document.addEventListener('touchstart', function () {}, { passive: true });
})();
