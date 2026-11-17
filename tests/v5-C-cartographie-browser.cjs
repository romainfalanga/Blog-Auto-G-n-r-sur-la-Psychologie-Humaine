// Run with: PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/v5-C-cartographie-browser.cjs
// Plan v5.1, lot C5a: sections of « Ce qui en ressort » (recognized, five closest,
// frequent subjects as cards, folded others and dismissed), focus following a moved
// card, no horizontal scroll at 390 px. Every API response is synthetic and local.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'data/cartographie.json'),'utf8'));
const Carto=require(path.join(root,'static/js/cartographie.js'));
const template=fs.readFileSync(path.join(root,'layouts/_default/cartographie.html'),'utf8');
const markup=template.slice(template.indexOf('<div class="container-wide carto" data-cartographie-app'),template.lastIndexOf('{{ end }}')).replace(/{{[^}]*}}/g,match=>match.includes('$data | jsonify')?JSON.stringify(data):match.includes('readFile')?'test':'');
const html='<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/css/style.css"></head><body><main>'+markup+'</main></body></html>';
const VIOLENCES={abus_physique:2,abus_emotionnel:2,temoin_violences:1,ressenti_peur:1,pas_protege:1};

(async()=>{
 const browser=await chromium.launch({headless:true});let passed=0;
 async function pageFor(width,answers,feedback={}){
  const context=await browser.newContext({viewport:{width,height:1000}}),page=await context.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(draft=>localStorage.setItem('cartographie-brouillon-v1',JSON.stringify(draft)),{answers,feedback,notes:{factors:{},outcomes:{}},resources:{}});
  await page.route('https://carto.test/**',route=>{
   const url=new URL(route.request().url());
   if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:html});
   if(url.pathname==='/css/style.css')return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'static/css/style.css'),'utf8')});
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
  for(const width of [390,1440])await check('sections, one ranking sentence, frequent cards and closed folds at '+width+'px',async()=>{
   const feedback={att_anxieux:'oui'};
   const {page,context,errors}=await pageFor(width,VIOLENCES,feedback);
   const expected=Carto.compute(data,VIOLENCES,feedback).sections;
   assert.equal(await page.locator('.carto-rank-note').count(),1);
   assert.deepEqual(await page.locator('.carto-section-title').allInnerTexts(),['Vous vous y reconnaissez (1)','Le plus en lien avec vos réponses','Aussi associés à la plupart des vécus difficiles']);
   assert.deepEqual(await ids(page.locator('article.carto-card[data-section="recognized"]')),expected.recognized.map(r=>r.outcome.id));
   assert.deepEqual(await ids(page.locator('article.carto-card[data-section="top"]')),expected.top.map(r=>r.outcome.id));
   assert.equal(expected.top.length,5);
   assert.deepEqual(await ids(page.locator('article.carto-card[data-section="frequent"]')),expected.frequent.map(r=>r.outcome.id));
   assert.ok(expected.frequent.some(r=>r.outcome.id==='dep_generale'),'depression with a strong link stays visible as a card');
   assert.match(await page.locator('article.carto-card[data-section="frequent"]').first().innerText(),/Associé à de nombreux vécus/);
   const fold=page.locator('details.carto-fold[data-fold="other"]');
   assert.equal(await fold.evaluate(n=>n.open),false);
   assert.equal(await fold.locator('summary').innerText(),'Autres sujets reliés à vos réponses ('+expected.other.length+')');
   assert.deepEqual(await ids(fold.locator('.carto-row')),expected.other.map(r=>r.outcome.id));
   assert.equal(await page.locator('details.carto-fold[data-fold="dismissed"]').count(),0);
   const text=await page.locator('[data-results]').innerText();
   assert.doesNotMatch(text,/3114|Repère de contexte|Tendance psychologique\n/);
   assert.equal(await page.locator('.carto-filters,.carto-progress,.carto-note-urgent,[data-delete]').count(),0);
   const contextual=new Set(data.outcomes.filter(o=>o.contextual_only).map(o=>o.id));
   assert.ok(!(await ids(page.locator('[data-results] [data-outcome]'))).some(id=>contextual.has(id)));
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true,'no horizontal scroll');
   await page.locator('article.carto-card[data-section="frequent"] .carto-detail').first().click();
   assert.equal(await page.locator('article.carto-card[data-section="frequent"] .carto-detail').first().getAttribute('aria-expanded'),'true');
   assert.ok(await page.locator('article.carto-card[data-section="frequent"] .carto-card-body').count()>0);
   assert.deepEqual(errors,[]);await context.close();
  });
  await check('a folded row opens in place; « Je me reconnais » moves it first and keeps focus; the fold stays open',async()=>{
   const {page,context,errors}=await pageFor(1440,VIOLENCES);
   const fold=page.locator('details.carto-fold[data-fold="other"]');
   await fold.locator('summary').click();
   const row=fold.locator('.carto-row').first(),id=await row.getAttribute('data-outcome');
   await row.click();
   assert.equal(await fold.locator('.carto-row').first().getAttribute('aria-expanded'),'true');
   const inner=fold.locator('article.carto-card[data-outcome="'+id+'"]');assert.equal(await inner.count(),1);
   await inner.getByRole('button',{name:'Je me reconnais'}).click();
   assert.deepEqual(await ids(page.locator('article.carto-card[data-section="recognized"]')),[id]);
   assert.equal(await page.evaluate(()=>document.activeElement&&document.activeElement.getAttribute('data-focus')),'feedback:'+id+':oui','focus follows the moved card');
   assert.equal(await page.locator('details.carto-fold[data-fold="other"]').evaluate(n=>n.open),true,'the fold stays open after re-rendering');
   assert.deepEqual(errors,[]);await context.close();
  });
  for(const width of [390,1440])await check('« Pas moi » moves a card to the dismissed fold, reversibly, focus kept at '+width+'px',async()=>{
   const {page,context,errors}=await pageFor(width,VIOLENCES);
   const first=page.locator('article.carto-card[data-section="top"]').first(),id=await first.getAttribute('data-outcome');
   await first.getByRole('button',{name:'Pas moi'}).click();
   const dismissed=page.locator('details.carto-fold[data-fold="dismissed"]');
   assert.equal(await dismissed.evaluate(n=>n.open),true);
   assert.equal(await dismissed.locator('summary').innerText(),'Écartés : « Pas moi » (1)');
   assert.equal(await page.evaluate(()=>document.activeElement&&document.activeElement.getAttribute('data-focus')),'feedback:'+id+':non');
   assert.ok(!(await ids(page.locator('article.carto-card[data-section="top"]'))).includes(id));
   await dismissed.locator('article.carto-card[data-outcome="'+id+'"]').getByRole('button',{name:'Pas moi'}).click();
   assert.equal(await page.locator('article.carto-card[data-section="top"]').first().getAttribute('data-outcome'),id,'undoing restores the rank');
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
   assert.deepEqual(errors,[]);await context.close();
  });
  await check('few subjects: no fold; only models: the five first are model cards with an honest label',async()=>{
   const {page,context,errors}=await pageFor(1440,{ressenti_responsable:1});
   assert.equal(await page.locator('details.carto-fold').count(),Carto.compute(data,{ressenti_responsable:1}).sections.other.length?1:0);
   const badges=await page.locator('article.carto-card .carto-badge').allInnerTexts();
   assert.ok(badges.length>0);
   assert.deepEqual(errors,[]);await context.close();
  });
  console.log(passed+' lot C5a browser scenarios passed.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
