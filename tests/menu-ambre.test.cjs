const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync, existsSync, readdirSync, statSync } = require('node:fs');
const { join } = require('node:path');

const root = join(__dirname, '..');
const read = path => readFileSync(join(root, path), 'utf8');

test('Ambre opens the shared menu and leads to the conversation page', async () => {
  const navigation = JSON.parse(read('data/navigation.json'));
  assert.deepEqual(navigation[0], { label: 'Ambre', href: '/mon-accompagnement/' });
  assert.deepEqual(navigation.map(item => item.label), ['Ambre', 'Mon enfance', 'Mes proches', 'Ma ligne de vie', 'Mes bilans', 'Mon compte']);
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
  // The home page stays public: Ambre is reached from the menu, never by a redirect of "/".
  assert.ok(!config.split('[[redirects]]').slice(1).some(block => /^\s*from\s*=\s*"\/"\s*$/m.test(block)));
});

test('Netlify proxies the list of close people and every person page to the Worker, without caching', () => {
  const config = read('netlify.toml');
  const blocks = config.split(/\n(?=\[\[)/);
  const redirect = from => blocks.find(block => block.startsWith('[[redirects]]') && block.includes('from = "' + from + '"'));
  const headers = path => blocks.find(block => block.startsWith('[[headers]]') && block.includes('for = "' + path + '"'));
  assert.match(redirect('/mes-proches'), /to = "https:\/\/decode-journal\.romainfalanga83\.workers\.dev\/mes-proches\/"\s+status = 200\s+force = true/);
  // "/mes-proches/*" covers "/mes-proches/<identifier>/": the splat carries the identifier.
  assert.match(redirect('/mes-proches/*'), /to = "https:\/\/decode-journal\.romainfalanga83\.workers\.dev\/mes-proches\/:splat"\s+status = 200\s+force = true/);
  for (const path of ['/mes-proches', '/mes-proches/*', '/mon-accompagnement', '/mon-accompagnement/*']) assert.match(headers(path), /Cache-Control = "private, no-store"/, path);
  assert.match(redirect('/mon-accompagnement'), /workers\.dev\/mon-accompagnement\/"/);
});
