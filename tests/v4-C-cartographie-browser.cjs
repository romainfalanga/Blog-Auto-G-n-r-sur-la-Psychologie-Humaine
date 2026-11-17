// Run with: PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/v4-C-cartographie-browser.cjs
// Plan v4.1, lot C: shared order page/Ambre, explanations without visible bibliography, unlinked questions,
// screen-reader counter and out-of-scope list. Every API response is synthetic and local.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'data/cartographie.json'),'utf8'));
const template=fs.readFileSync(path.join(root,'layouts/_default/cartographie.html'),'utf8');
const markup=template.slice(template.indexOf('<div class="container-wide carto" data-cartographie-app'),template.lastIndexOf('{{ end }}')).replace(/{{[^}]*}}/g,match=>match.includes('$data | jsonify')?JSON.stringify(data):match.includes('readFile')?'test':'');
const html='<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/css/style.css"><link rel="stylesheet" href="/css/cartographie.css"></head><body><main>'+markup+'</main></body></html>';

(async()=>{
 const {summarizeCartography}=await import(pathToFileURL(path.join(root,'journal/src/cartographie.mjs')).href);
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});let passed=0;
 async function pageFor(width=1440){
  const context=await browser.newContext({viewport:{width,height:1000}}),page=await context.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('https://carto.test/**',route=>{
   const url=new URL(route.request().url());
   if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:html});
   if(url.pathname.startsWith('/css/'))return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'static',url.pathname),'utf8')});
   if(url.pathname.startsWith('/js/'))return route.fulfill({contentType:'application/javascript',body:fs.readFileSync(path.join(root,'static',url.pathname),'utf8')});
   if(url.pathname==='/api/me')return route.fulfill({contentType:'application/json',body:JSON.stringify({user:null})});
   return route.fulfill({status:404,body:''});
  });
  await page.goto('https://carto.test/');await page.locator('.carto-theme-panel').first().waitFor();
  await page.waitForFunction(()=>!document.querySelector('[data-save]').textContent.includes('Vérification de votre connexion'));
  return {page,context,errors};
 }
 async function check(name,fn){await fn();passed++;console.log('PASS '+name);}
 async function select(page,id,value=1){const f=data.factors.find(x=>x.id===id);const panel=page.locator('[data-theme="'+f.group+'"]');if(!await panel.evaluate(n=>n.open))await panel.locator('summary').click();await page.locator('[data-factor="'+id+'"] [data-level="'+value+'"]').click();}
 try{
  await check('the five first cards shown are the five first subjects handed to Ambre',async()=>{
   const {page,context,errors}=await pageFor();
   const answers={parentification:1,ressenti_responsable:1,enfant_sage:1,debrouiller_seul:1,parent_trouble:1,abus_sexuel:1};
   for(const id of Object.keys(answers))await select(page,id);
   const titles=await page.locator('article.carto-card .carto-card-title').allInnerTexts();
   const ambre=summarizeCartography(answers).display_order.slice(0,5).map(x=>x.label);
   assert.deepEqual(titles.slice(0,5),ambre);
   // v5 (lot C5a): contextual subjects are no longer displayed, even in the folded rows.
   const contextual=new Set(data.outcomes.filter(o=>o.contextual_only).map(o=>o.id));
   const shown=await page.locator('[data-results] [data-outcome]').evaluateAll(nodes=>nodes.map(n=>n.dataset.outcome));
   assert.ok(shown.length>5);assert.ok(!shown.some(id=>contextual.has(id)),'no contextual subject');
   assert.deepEqual(errors,[]);await context.close();
  });
  await check('no contextual subject anywhere, in the list or in « Liens étudiés »; other references add no associated experience',async()=>{
   const {page,context,errors}=await pageFor();
   await select(page,'abus_sexuel');await select(page,'abus_physique');await select(page,'critique');
   assert.equal(await page.getByText('Repère de contexte').count(),0);
   const contextual=new Set(data.outcomes.filter(o=>o.contextual_only).map(o=>o.label));
   await page.locator('[data-factor="abus_sexuel"] .carto-see').click();
   const listed=await page.locator('[data-dialog-body] .carto-dialog-list strong').allInnerTexts();
   assert.ok(listed.length>0);assert.ok(!listed.some(t=>contextual.has(t)),'the dialog hides contextual subjects');
   assert.doesNotMatch(await page.locator('[data-dialog-body]').innerText(),/non compté/);
   await page.locator('[data-dialog-close]').click();
   const mixed=page.locator('article.carto-card').filter({has:page.locator('.carto-from-line')}).first();
   await mixed.locator('.carto-detail').click();
   const card=page.locator('article.carto-card').filter({has:page.locator('.carto-other-title')}).first();
   if(await card.count()){assert.equal(await card.locator('.carto-other-title').innerText(),'Autres repères (modèles cliniques, définitions)');assert.ok(await card.locator('.carto-links-other > li').count()>0);}
   const counted=Number(await mixed.getAttribute('data-associated'));
   assert.equal(await mixed.locator('.carto-links:not(.carto-links-other) > li').count()>=counted,true);
   assert.deepEqual(errors,[]);await context.close();
  });
  await check('questions without any link say so; the dialog separates measured links from other references',async()=>{
   const {page,context,errors}=await pageFor(390);
   for(const group of new Set(['secret_famille','migration','exil'].map(id=>data.factors.find(f=>f.id===id).group))){const panel=page.locator('[data-theme="'+group+'"]');if(!await panel.evaluate(n=>n.open))await panel.locator('summary').click();assert.equal(await panel.evaluate(n=>n.open),true);}
   for(const id of ['secret_famille','migration']){assert.equal(await page.locator('[data-factor="'+id+'"] .carto-no-link').innerText(),'Aucune association suffisamment documentée n’est retenue pour l’instant.');assert.equal(await page.locator('[data-factor="'+id+'"] .carto-see').count(),0);}
   assert.equal(await page.locator('[data-factor="exil"] .carto-see').count(),1,'exile now has its ICD-11 context');
   await page.locator('[data-factor="exil"] .carto-see').click();
   const headings=await page.locator('[data-dialog-body] .carto-dialog-h').allInnerTexts();
   assert.deepEqual(headings,['Contexte décrit par la CIM-11 ('+data.links.filter(l=>l.factor==='exil'&&!data.outcomes.find(o=>o.id===l.outcome).contextual_only).length+')']);
   await page.locator('[data-dialog-close]').click();
   assert.deepEqual(errors,[]);await context.close();
  });
  await check('only the counter is announced to screen readers',async()=>{
   const {page,context,errors}=await pageFor();
   assert.equal(await page.locator('[data-results][aria-live]').count(),0);
   const live=page.locator('[data-count][role="status"][aria-live="polite"]');assert.equal(await live.count(),1);
   assert.equal(await live.innerText(),'Aucune réponse');
   await select(page,'abus_physique');
   const n=await page.locator('[data-results] article.carto-card,[data-results] .carto-row').count();
   assert.equal(await live.textContent(),'1 réponse · '+n+' sujet'+(n>1?'s affichés':' affiché'));
   assert.equal(await page.locator('[aria-live]').count(),1);
   assert.deepEqual(errors,[]);await context.close();
  });
  for(const width of [1440,320])await check('a discreet link opens what the map does not cover, without intro or banner at '+width+'px',async()=>{
   const {page,context,errors}=await pageFor(width);
   if(width<901)await page.locator('[data-tab="results"]').click();
   const link=page.locator('[data-scope-open]');assert.equal(await link.count(),1);assert.equal(await link.innerText(),'Ce que la carte ne couvre pas');
   assert.equal(await page.locator('.carto-home-intro,.carto-lead,.carto-reading-guide,.carto-coverage,.carto-note-urgent,.carto-mobile-bar').count(),0);
   await link.click();
   assert.equal(await page.locator('[data-dialog-title]').innerText(),'Ce que la carte ne couvre pas');
   assert.equal(await page.locator('.carto-scope-list > li').count(),data.notes.hors_perimetre.items.length);
   assert.equal(await page.locator('[data-dialog-body] .carto-sources a').count(),0);
   await page.locator('[data-dialog-close]').click();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);
   assert.deepEqual(errors,[]);await context.close();
  });
  console.log(passed+' lot C browser scenarios passed.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
