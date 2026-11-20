const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');

test('legacy page redirects drop retired identifiers and retain public article or payment intent', async () => {
  const { default: redirect } = await import('../netlify/edge-functions/legacy-private-pages.js');
  const base = 'https://decodetonesprit.fr';
  const config = readFileSync(join(__dirname, '../netlify.toml'), 'utf8');
  // "/" now opens Ambre (302); that rule is its only redirect and it never carries legacy identifiers.
  const home = config.split('[[redirects]]').slice(1).filter(block => /^\s*from\s*=\s*"\/"\s*$/m.test(block));
  assert.equal(home.length, 1);
  assert.match(home[0], /to = "\/accueil\/"\s+status = 302/);
  const [, pattern] = config.match(/pattern = "([^"]+)"\s+function = "legacy-private-pages"/);
  const route = new RegExp('^(?:' + pattern + ')$');
  assert.equal(route.test('/personnages/camille/'), false);
  assert.equal(route.test('/'), false);
  for (const [path, location] of [
    ['/mon-personnage/?story=old', '/ambre/'],
    ['/mes-personnages?character=old', '/ambre/'],
    ['/mes-histoires/old?story=old', '/ambre/'],
    ['/journal/?story=old', '/ambre/'],
    ['/journal/?article=public&like=1&story=old', '/mon-compte/'],
    ['/journal/?tab=likes', '/mon-compte/'],
    ['/journal/?payment=success&character=old', '/mon-compte/?payment=success'],
    ['/journal/?tab=portrait&character=old', '/mon-compte/'],
  ]) {
    const url = new URL(base + path);
    assert.equal(route.test(url.pathname), true, path);
    const response = redirect(new Request(url));
    assert.equal(response.status, 302, path);
    assert.equal(response.headers.get('Location'), location, path);
    assert.equal(response.headers.get('Cache-Control'), 'private, no-store', path);
  }
  for (const path of ['/', '/cartographie/', '/personnages/camille/', '/mon-compte/', '/posts/example/', '/journalisme/']) {
    assert.equal(redirect(new Request(base + path)), undefined, 'Unrelated pages must never redirect: ' + path);
  }
  const request = new Request(base + '/mon-personnage', { method: 'POST' });
  assert.equal(redirect(request), undefined);
});
