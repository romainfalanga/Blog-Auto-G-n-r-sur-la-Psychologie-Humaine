// Lot P (page speed): immutable caching is only safe when every URL changes with its content.
const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync, readdirSync, statSync } = require('node:fs');
const { join } = require('node:path');

const root = join(__dirname, '..');
const read = path => readFileSync(join(root, path), 'utf8');
const blocks = () => read('netlify.toml').split(/\n(?=\[\[)/);
const walk = (directory, files = []) => { for (const name of readdirSync(directory)) { const path = join(directory, name); if (statSync(path).isDirectory()) walk(path, files); else files.push(path); } return files; };

test('/js/* is cached immutably and every template reference to /js/ carries a content version', () => {
  const js = blocks().filter(block => block.startsWith('[[headers]]') && /for = "\/js\/\*"/.test(block));
  assert.equal(js.length, 1);
  assert.match(js[0], /Cache-Control = "public, max-age=31536000, immutable"/);
  let references = 0;
  for (const file of walk(join(root, 'layouts'))) {
    for (const match of readFileSync(file, 'utf8').matchAll(/(?:src|href)="\/js\/([\w.-]+)(\?v=\{\{ readFile "static\/js\/([\w.-]+)" \| md5 \}\})?"/g)) {
      references++;
      assert.ok(match[2], file + ' /js/' + match[1] + ' has no ?v= version');
      assert.equal(match[3], match[1], 'the version is the digest of the file it names');
    }
    assert.doesNotMatch(readFileSync(file, 'utf8'), /(?:src|href)="\/js\/[^"?{]*"/, file);
  }
  assert.ok(references >= 1);
});

test('the application files are proxied to the Worker, which alone decides their cache', () => {
  const app = blocks().filter(block => /from = "\/app\/\*"/.test(block));
  assert.equal(app.length, 1);
  assert.match(app[0], /to = "https:\/\/decode-journal\.romainfalanga83\.workers\.dev\/app\/:splat"\s+status = 200\s+force = true/);
  // A header rule here would also mark a 404 for an unknown name as immutable.
  assert.equal(blocks().filter(block => block.startsWith('[[headers]]') && /for = "\/app\//.test(block)).length, 0);
});

test('the header logo is a small WebP with a PNG fallback, in Hugo and in the Worker shell', async () => {
  const brand = read('layouts/partials/header-brand.html');
  assert.match(brand, /<picture><source srcset="\/images\/logo-mark-128\.webp" type="image\/webp"><img src="\/images\/logo-mark-128\.png" alt="" width="64" height="64"><\/picture>/);
  assert.doesNotMatch(brand, /favicon-192x192/);
  for (const file of ['static/images/logo-mark-128.webp', 'static/images/logo-mark-128.png']) assert.ok(statSync(join(root, file)).size < 4096, file);
  const webp = readFileSync(join(root, 'static/images/logo-mark-128.webp'));
  assert.equal(webp.subarray(0, 4).toString(), 'RIFF'); assert.equal(webp.subarray(8, 12).toString(), 'WEBP');
  const generated = await import('../journal/src/navigation.generated.mjs');
  assert.ok(generated.HEADER_BRAND_HTML.includes('/images/logo-mark-128.webp'));
});
