// Run with: PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/v5-C-cartographie-browser.cjs
// Mon enfance: one count-based order, full answer labels and preserved feedback.
// Every API response is synthetic and local.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'data/cartographie.json'),'utf8'));
const Carto=require(path.join(root,'static/js/cartographie.js'));
const template=fs.readFileSync(path.join(root,'layouts/_default/cartographie.html'),'utf8');
const renderTemplate=input=>input.replace(/{{ if \$symptoms }}[\s\S]*?{{ else }}([\s\S]*?){{ end }}/g,'$1').replace(/{{ if \$symptoms }}[\s\S]*?{{ end }}/g,'').replace(/{{ \$title }}/g,'Mon enfance').replace(/{{[^}]*}}/g,match=>match.includes('jsonify')?JSON.stringify(data):match.includes('safeJS')&&match.includes('readFile')?fs.readFileSync(path.join(root,match.match(/readFile "([^"]+)"/)[1]),'utf8'):match.includes('readFile')?'test':'');
// Match the real page: data and both inline scripts execute in the head, before the app exists.
const head=renderTemplate(template.slice(0,template.indexOf('{{ define "main" }}')));
const markup=renderTemplate(template.slice(template.indexOf('<div class="container-wide carto" data-cartographie-app'),template.lastIndexOf('{{ end }}')));
const html='<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/css/site-system.css"><link rel="stylesheet" href="/css/style.css">'+head+'</head><body><main id="main-content">'+markup+'</main></body></html>';
const artifacts=path.join(root,'work','mon-enfance');fs.mkdirSync(artifacts,{recursive:true});
const VIOLENCES={abus_physique:2,abus_emotionnel:2,temoin_violences:1,ressenti_peur:1,pas_protege:1};

(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});let passed=0;
 async function pageFor(width,answers,feedback={}){
  const context=await browser.newContext({viewport:{width,height:1000}}),page=await context.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(draft=>localStorage.setItem('cartographie-brouillon-v1',JSON.stringify(draft)),{answers,feedback,notes:{factors:{},outcomes:{}},resources:{}});
  await page.route('https://carto.test/**',route=>{
   const url=new URL(route.request().url());
   if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:html});
   if(url.pathname.startsWith('/css/'))return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'static',url.pathname),'utf8')});
   if(url.pathname.startsWith('/js/'))return route.fulfill({contentType:'application/javascript',body:fs.readFileSync(path.join(root,'static',url.pathname),'utf8')});
   if(url.pathname==='/api/me')return route.fulfill({contentType:'application/json',body:JSON.stringify({user:null})});
   return route.fulfill({status:404,body:''});
  });
  await page.goto('https://carto.test/');await page.locator('article.carto-card').first().waitFor({state:'attached'});
  if(width<901)await page.locator('[data-tab="results"]').click();
  return {page,context,errors};
 }
 async function check(name,fn){await fn();passed++;console.log('PASS '+name);}
 const ids=sel=>sel.evaluateAll(nodes=>nodes.map(n=>n.dataset.outcome));
 try{
  for(const width of [320,390,768,1440])await check('Mon enfance, descending counts, explanations without categories or bibliography at '+width+'px',async()=>{
   const feedback={att_anxieux:'oui'};
   const {page,context,errors}=await pageFor(width,VIOLENCES,feedback);
   const expected=Carto.compute(data,VIOLENCES,feedback).results;
   assert.equal(await page.locator('h1').innerText(),'Mon enfance');
   assert.deepEqual(await ids(page.locator('article.carto-card')),expected.map(r=>r.outcome.id));
   const counts=await page.locator('article.carto-card').evaluateAll(nodes=>nodes.map(n=>Number(n.dataset.associated)));
   assert.deepEqual(counts,[...counts].sort((a,b)=>b-a));
   assert.equal(await page.locator('.carto-section-title,.carto-kind,.carto-sources,details.carto-fold').count(),0);
   assert.equal(await page.locator('.carto-rank-note').count(),1);
   await page.locator('.carto-detail').first().click();
   assert.ok(await page.locator('.carto-link-text').count()>0);
   assert.equal(await page.locator('[data-results] a,.carto-sources,.carto-kind').count(),0);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
   await page.evaluate(()=>window.scrollTo(0,0));
   await page.screenshot({path:path.join(artifacts,'results-'+width+'.png')});
   if(width<901)await page.locator('[data-tab="form"]').click();
   await page.locator('[data-theme="violences"] > summary').click();
   const factor=page.locator('[data-factor="abus_physique"]');
   const checks=await factor.locator('.carto-choice').evaluateAll(nodes=>nodes.map(n=>{const r=n.getBoundingClientRect();return {label:n.textContent,width:r.width,height:r.height,clipped:n.scrollWidth>n.clientWidth,wrap:getComputedStyle(n).whiteSpace}}));
   assert.deepEqual(checks.map(c=>c.label),['Non','Oui','Je ne sais pas']);
   assert.ok(checks.every(c=>c.height>=44&&c.width>=44&&!c.clipped&&c.wrap==='nowrap'),JSON.stringify(checks));
   assert.equal(await factor.locator('[data-level="1"]').getAttribute('aria-pressed'),'true','stored 2 is displayed as Yes');assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')).answers.abus_physique),2,'viewing preserves the original frequency');
   const unknown=factor.locator('[data-level="unknown"]');await factor.locator('[data-level="1"]').focus();await page.keyboard.press('Tab');
   assert.equal(await unknown.evaluate(n=>n===document.activeElement),true);
   assert.equal(await unknown.evaluate(n=>n.matches(':focus-visible')),true);
   assert.ok(await unknown.evaluate(n=>parseFloat(getComputedStyle(n).outlineWidth)>=2));
   await page.keyboard.press('Space');assert.equal(await unknown.getAttribute('aria-pressed'),'true');
   assert.ok(await unknown.evaluate(n=>Number(getComputedStyle(n).fontWeight)>Number(getComputedStyle(n.parentElement.querySelector('[data-level="0"]')).fontWeight)&&getComputedStyle(n).boxShadow!=='none'&&n.scrollWidth<=n.clientWidth),'selection has a visible shape and weight, not color alone');
   await page.keyboard.press('Space');assert.equal(await unknown.getAttribute('aria-pressed'),'true','reselecting preserves the answer');
   await page.locator('[data-theme="famille"] > summary').click();
   const binary=page.locator('[data-factor="adoption"]');
   assert.deepEqual(await binary.locator('.carto-choice').allInnerTexts(),['Non','Oui','Je ne sais pas']);
   assert.ok(await binary.locator('.carto-choice').evaluateAll(nodes=>nodes.every(n=>n.getBoundingClientRect().height>=44&&n.scrollWidth<=n.clientWidth)));
   await factor.scrollIntoViewIfNeeded();await factor.locator('[data-level="1"]').focus();await page.keyboard.press('Tab');
   await page.screenshot({path:path.join(artifacts,'choices-'+width+'.png')});
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
   // Factor details also retain explanations without visible references.
   await factor.locator('.carto-see').click();
   assert.ok(await page.locator('[data-dialog-body] p').count()>0);
   assert.equal(await page.locator('[data-dialog-body] a,.carto-sources,.carto-kind').count(),0);
   assert.deepEqual(errors,[]);await context.close();
  });
  for(const width of [320,390,768,1440])await check('recognition keeps rank; Pas moi folds reversibly with focus at '+width+'px',async()=>{
   const {page,context,errors}=await pageFor(width,VIOLENCES);
   const before=await ids(page.locator('article.carto-card'));
   const card=page.locator('article.carto-card').nth(8),id=await card.getAttribute('data-outcome');
   await card.getByRole('button',{name:'Je me reconnais'}).click();
   assert.deepEqual(await ids(page.locator('article.carto-card')),before);
   assert.equal(await page.evaluate(()=>document.activeElement.getAttribute('data-focus')),'feedback:'+id+':oui');
   assert.equal(await card.getByRole('button',{name:'Je me reconnais'}).getAttribute('aria-pressed'),'true');
   const first=page.locator('article.carto-card').first(),firstId=await first.getAttribute('data-outcome');
   await first.getByRole('button',{name:'Pas moi'}).click();
   const dismissed=page.locator('details.carto-fold[data-fold="dismissed"]');
   assert.equal(await dismissed.evaluate(n=>n.open),true);
   assert.equal(await page.evaluate(()=>document.activeElement.getAttribute('data-focus')),'feedback:'+firstId+':non');
   assert.deepEqual(await ids(page.locator('article.carto-card:not([data-section="dismissed"])')),before.slice(1));
   await dismissed.locator('article.carto-card').getByRole('button',{name:'Pas moi'}).click();
   assert.deepEqual(await ids(page.locator('article.carto-card')),before);
   assert.equal(await page.evaluate(()=>document.activeElement.getAttribute('data-focus')),'feedback:'+firstId+':non');
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
   assert.deepEqual(errors,[]);await context.close();
  });
  console.log(passed+' Mon enfance browser scenarios passed.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
