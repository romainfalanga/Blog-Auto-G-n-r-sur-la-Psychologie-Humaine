const test = require('node:test');
const assert = require('node:assert/strict');

test('legacy page redirects drop retired identifiers and retain public article or payment intent', async () => {
  const { default: redirect } = await import('../netlify/edge-functions/legacy-private-pages.js');
  const base = 'https://decodetonesprit.fr';
  for (const [path, location] of [
    ['/mon-personnage/?story=old', '/mon-accompagnement/'],
    ['/mes-personnages?character=old', '/mon-accompagnement/'],
    ['/mes-histoires/old?story=old', '/mon-accompagnement/'],
    ['/journal/?story=old', '/mon-accompagnement/'],
    ['/journal/?article=public&like=1&story=old', '/mon-portrait/?article=public&like=1#mes-articles'],
    ['/journal/?payment=success&character=old', '/mon-compte/?payment=success'],
    ['/journal/?tab=portrait&character=old', '/mon-portrait/'],
  ]) {
    const response = redirect(new Request(base + path));
    assert.equal(response.status, 302, path);
    assert.equal(response.headers.get('Location'), location, path);
    assert.equal(response.headers.get('Cache-Control'), 'private, no-store', path);
  }
  const request = new Request(base + '/mon-personnage', { method: 'POST' });
  assert.equal(redirect(request), undefined);
});
