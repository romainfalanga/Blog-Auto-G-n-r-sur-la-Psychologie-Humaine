const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync, existsSync, readdirSync, statSync } = require('node:fs');
const { join } = require('node:path');

const root = join(__dirname, '..');
const read = path => readFileSync(join(root, path), 'utf8');

test('Ambre opens the shared menu and leads to the conversation page', async () => {
  const navigation = JSON.parse(read('data/navigation.json'));
  assert.deepEqual(navigation[0], { label: 'Ambre', href: '/ambre/' });
  assert.deepEqual(navigation.map(item => item.label), ['Ambre', 'Ma ligne de vie', 'Mes proches', 'Mon enfance', 'Mes symptômes', 'Mes bilans', 'Mes diagnostics', 'Mon compte']);
  // The Worker menu is generated from the same file as the Hugo header.
  const generated = await import('../journal/src/navigation.generated.mjs');
  const links = [...generated.NAVIGATION_HTML.matchAll(/<a href="([^"]+)">([^<]+)<\/a>/g)].map(match => ({ label: match[2], href: match[1] }));
  assert.deepEqual(links, navigation);
  assert.match(read('layouts/partials/header.html'), /range \.Site\.Data\.navigation/);
  assert.deepEqual(Object.keys(generated).sort(), ['HEADER_BRAND_HTML', 'NAVIGATION_HTML', 'SITE_FOOTER_HTML', 'SITE_NAVIGATION_SCRIPT', 'SITE_SYSTEM_CSS']);
});

test('the Ambre bubble is removed from the public site: no asset, no loader, no framing exception', () => {
  for (const file of ['static/js/ambre-widget.js', 'static/css/ambre-widget.css']) assert.equal(existsSync(join(root, file)), false, file);
  const files = [];
  const walk = directory => { for (const name of readdirSync(directory)) { const path = join(directory, name); if (statSync(path).isDirectory()) walk(path); else files.push(path); } };
  for (const directory of ['layouts', 'static/js', 'static/css', 'netlify']) walk(join(root, directory));
  for (const file of files) assert.doesNotMatch(readFileSync(file, 'utf8'), /ambre-widget|ambre-launcher|ambre-panel|embedded=1/, file);
  const config = read('netlify.toml');
  assert.doesNotMatch(config, /bulle/);
  assert.match(config, /X-Frame-Options = "DENY"/);
});

test('opening the site lands on Ambre: "/" redirects (302) to /ambre/, the catalogue stays at /personnages/', () => {
  const config = read('netlify.toml');
  const blocks = config.split(/\n(?=\[\[)/).filter(block => block.startsWith('[[redirects]]'));
  const home = blocks.filter(block => /^\s*from\s*=\s*"\/"\s*$/m.test(block));
  assert.equal(home.length, 1);
  // Temporary, and forced: Hugo still builds an index.html at "/", which Netlify would otherwise serve.
  assert.match(home[0], /to = "\/ambre\/"\s+status = 302\s+force = true/);
  // The proxied Ambre page itself is never redirected back to "/", and nothing redirects the catalogue.
  for (const from of ['/ambre', '/ambre/*']) assert.doesNotMatch(blocks.find(block => block.includes('from = "' + from + '"')), /to = "\/"/);
  assert.ok(!blocks.some(block => /from\s*=\s*"\/personnages\/?"\s*$/m.test(block)));
  assert.match(read('layouts/personnages/list.html'), /character-directory/);
  // The logo and the footer lead straight to Ambre (no redirect hop); the footer keeps a link to the catalogue.
  assert.match(read('layouts/partials/header-brand.html'), /^<a href="\/ambre\/" class="site-logo"/);
  const footer = read('layouts/partials/footer.html');
  assert.match(footer, /href="\/ambre\/" class="site-logo"/);
  assert.doesNotMatch(footer, /Personnages/);
  // No internal link of the site or of the personal space points at "/" any more.
  for (const file of ['layouts/partials/header-brand.html', 'layouts/partials/footer.html', 'layouts/partials/breadcrumb.html', 'journal/src/ui.mjs', 'journal/src/navigation.generated.mjs']) {
    assert.doesNotMatch(read(file), /href=\\?"\/\\?"|\.href='\/'/, file);
  }
});

test('Netlify proxies the list of close people and every person page to the Worker, without caching', () => {
  const config = read('netlify.toml');
  const blocks = config.split(/\n(?=\[\[)/);
  const redirect = from => blocks.find(block => block.startsWith('[[redirects]]') && block.includes('from = "' + from + '"'));
  const headers = path => blocks.find(block => block.startsWith('[[headers]]') && block.includes('for = "' + path + '"'));
  assert.match(redirect('/mes-proches'), /to = "https:\/\/decode-journal\.romainfalanga83\.workers\.dev\/mes-proches\/"\s+status = 200\s+force = true/);
  // "/mes-proches/*" covers "/mes-proches/<identifier>/": the splat carries the identifier.
  assert.match(redirect('/mes-proches/*'), /to = "https:\/\/decode-journal\.romainfalanga83\.workers\.dev\/mes-proches\/:splat"\s+status = 200\s+force = true/);
  for (const path of ['/mes-proches', '/mes-proches/*', '/ambre', '/ambre/*']) assert.match(headers(path), /Cache-Control = "private, no-store"/, path);
  assert.match(redirect('/ambre'), /workers\.dev\/ambre\/"/);
});

test('Mes symptômes follows Mon enfance in the menu and uses the shared Hugo map and private bootstrap', () => {
  const navigation = JSON.parse(read('data/navigation.json'));
  const index = navigation.findIndex(item => item.label === 'Mon enfance');
  assert.deepEqual(navigation[index + 1], { label: 'Mes symptômes', href: '/mes-symptomes/' });
  const config = read('netlify.toml');
  const blocks = config.split(/\n(?=\[\[)/);
  const redirect = from => blocks.find(block => block.startsWith('[[redirects]]') && block.includes('from = "' + from + '"'));
  const headers = path => blocks.find(block => block.startsWith('[[headers]]') && block.includes('for = "' + path + '"'));
  assert.equal(redirect('/mes-symptomes'), undefined);
  assert.equal(redirect('/mes-symptomes/*'), undefined);
  assert.match(read('content/mes-symptomes.md'), /layout: "cartographie"/);
  assert.match(read('content/mes-symptomes.md'), /dataset: "symptomes"/);
  for (const path of ['/mes-symptomes', '/mes-symptomes/*']) assert.match(headers(path), /Cache-Control = "private, no-store"/, path);
  const worker = read('journal/src/worker.mjs');
  assert.doesNotMatch(worker, /'\/mes-symptomes':'Mes symptômes'/);
  assert.match(worker, /symptomes:\{path:'\/mes-symptomes\/',routes:\['\/api\/symptomes'\]\}/);
});
