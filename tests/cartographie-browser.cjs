// Run with: PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/cartographie-browser.cjs
// Browser-only integration checks; every API response is synthetic and local.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'data/cartographie.json'),'utf8'));
const source=fs.readFileSync(path.join(root,'static/js/cartographie-ui.js'),'utf8');
const template=fs.readFileSync(path.join(root,'layouts/_default/cartographie.html'),'utf8');
const markup=template.slice(template.indexOf('<div class="container-wide carto" data-cartographie-app'),template.lastIndexOf('{{ end }}')).replace(/{{[^}]*}}/g,match=>match.includes('$data | jsonify')?JSON.stringify(data):match.includes('readFile')?'test':'');
const document='<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/css/style.css"></head><body><main>'+markup+'</main></body></html>';
const fixture=(answers={},version=1)=>({answers,version,feedback:{},notes:{factors:{},outcomes:{}},resources:{prot_amities:1},ambre_notes:[],updated_at:1794830400,consent:true,locked:true});
const pause=()=>{let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}};

(async()=>{
 const browser=await chromium.launch({headless:true});let passed=0;
 async function pageFor(options={}){
  const context=await browser.newContext({viewport:{width:options.width||1440,height:1000}}),page=await context.newPage();
  const server={account:options.anonymous?null:{id:'synthetic-member'},map:options.map||null,puts:[],requests:[],errors:[],beforePut:null,beforeGet:options.beforeGet||null,getFailure:Boolean(options.getFailure),deleteFailure:false,putFailure:false,inflight:0,maxInflight:0};
  page.on('pageerror',error=>server.errors.push(error.message));page.on('dialog',dialog=>dialog.accept());
  if(options.draft)await page.addInitScript(draft=>{if(!sessionStorage.getItem('draft-seeded')){localStorage.setItem('cartographie-brouillon-v1',JSON.stringify(draft));sessionStorage.setItem('draft-seeded','1')}},options.draft);
  await page.route('https://carto.test/**',async route=>{
   const request=route.request(),url=new URL(request.url()),method=request.method();
   if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:document});
   if(url.pathname==='/css/style.css')return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'static/css/style.css'),'utf8')});
   if(url.pathname==='/js/cartographie.js')return route.fulfill({contentType:'application/javascript',body:fs.readFileSync(path.join(root,'static/js/cartographie.js'),'utf8')});
   if(url.pathname==='/js/cartographie-ui.js')return route.fulfill({contentType:'application/javascript',body:source});
   if(url.pathname==='/api/me')return route.fulfill({contentType:'application/json',body:JSON.stringify({user:server.account})});
   if(url.pathname==='/api/cartographie'){
    server.requests.push(method);
    if(method==='GET'){if(server.beforeGet)await server.beforeGet();if(server.getFailure)return route.fulfill({status:503,json:{error:'Chargement indisponible.'}});return route.fulfill({json:{cartographie:server.map}})}
    if(method==='PUT'){
     const body=request.postDataJSON();server.puts.push(body);server.inflight++;server.maxInflight=Math.max(server.maxInflight,server.inflight);
     if(server.beforePut)await server.beforePut(body);
     server.inflight--;
     if(server.putFailure)return route.fulfill({status:503,json:{error:'Connexion momentanément indisponible.'}});
     if(body.version!==(server.map?.version||0))return route.fulfill({status:409,json:{error:'Version différente.'}});
     server.map={...fixture({},(server.map?.version||0)+1),...body,version:(server.map?.version||0)+1};
     return route.fulfill({json:{cartographie:server.map}});
    }
    if(method==='DELETE'){if(server.deleteFailure)return route.fulfill({status:503,json:{error:'Suppression indisponible.'}});server.map=null;return route.fulfill({json:{cartographie:null}})}
   }
   return route.fulfill({status:404,body:''});
  });
  await page.goto('https://carto.test/');await page.locator('.carto-theme-panel').first().waitFor();
  if(options.waitForLoad!==false)await page.waitForFunction(()=>!document.querySelector('[data-save]').textContent.includes('Chargement de votre cartographie')&&!document.querySelector('[data-save]').textContent.includes('Vérification de votre connexion'));
  return {page,context,server};
 }
 async function check(name,fn){if(process.env.CARTOGRAPHY_BROWSER_CASE&&!name.includes(process.env.CARTOGRAPHY_BROWSER_CASE))return;await fn();passed++;console.log('PASS '+name)}
 async function opened(page,id='violences'){const panel=page.locator('[data-theme="'+id+'"]');if(!await panel.evaluate(node=>node.open))await panel.locator('summary').click();return panel}
 async function select(page,id,value){const factor=data.factors.find(f=>f.id===id);await opened(page,factor.group);const button=page.locator('[data-factor="'+id+'"] [data-level="'+value+'"]');await button.click();return button}
 async function saved(page){try{await page.waitForFunction(()=>document.querySelector('[data-save]').dataset.state==='saved',{},{timeout:7000})}catch(error){console.error('Save UI:',await page.locator('[data-save]').innerText());throw error}}
 try{
  await check('six independent closed accordions, anonymous local-only answers and stable focus',async()=>{
   const {page,context,server}=await pageFor({anonymous:true});assert.equal(await page.locator('.carto-theme-panel').count(),6);assert.equal(await page.locator('.carto-theme-panel[open]').count(),0);assert.equal(await page.locator('[data-theme="ressources"]').count(),0);assert.equal(await page.locator('input[type=checkbox]').count(),0);
   const button=await select(page,'abus_physique',1);assert.equal(await button.evaluate(node=>node===document.activeElement),true);await opened(page,'negligence');assert.equal(await page.locator('.carto-theme-panel[open]').count(),2);assert.equal(await button.getAttribute('aria-pressed'),'true');
   const row=page.locator('[data-factor="abus_physique"]');await row.locator('.carto-note-add').click();const note=row.locator('textarea');await note.fill('Une précision personnelle synthétique.');await page.waitForTimeout(900);assert.equal(await note.evaluate(node=>node===document.activeElement),true);assert.equal(server.puts.length,0);assert.equal(server.requests.length,0);assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')).answers.abus_physique),1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('connected first answer autosaves with consent and explicit Non survives a reload',async()=>{
   const {page,context,server}=await pageFor();await select(page,'abus_physique',0);await saved(page);assert.equal(server.puts.length,1);assert.equal(server.puts[0].consent,true);assert.equal(server.map.answers.abus_physique,0);assert.equal(await page.locator('.carto-card').count(),0);await page.reload();await page.locator('.carto-theme-panel').first().waitFor();await opened(page);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="0"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('.carto-theme-panel[open]').count(),1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('reselecting a saved answer creates no pending draft or conflict after reload',async()=>{
   const {page,context,server}=await pageFor({map:fixture({critique:2})});await select(page,'critique',2);assert.equal(await page.evaluate(()=>localStorage.getItem('cartographie-brouillon-v1:synthetic-member')),null);await page.reload();await saved(page);await page.waitForTimeout(900);assert.equal(server.puts.length,0);assert.equal(await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).count(),0);assert.equal(server.map.answers.critique,2);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('edits made during PUT are serialized and existing hidden resources retained',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});const first=pause(),release=pause();server.beforePut=async()=>{if(server.puts.length===1){first.resolve();await release.promise}};await select(page,'abus_physique',2);await first.promise;await select(page,'abus_sexuel',1);assert.equal(server.puts.length,1);release.resolve();await saved(page);assert.equal(server.puts.length,2);assert.equal(server.maxInflight,1);assert.equal(server.puts[1].version,2);assert.equal(server.map.answers.abus_sexuel,1);assert.deepEqual(server.map.resources,{prot_amities:1});assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('409 never overwrites silently and explicit local choice keeps current stored resources',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});server.map=fixture({critique:2},2);server.map.resources={prot_famille:1};await select(page,'abus_physique',2);await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).waitFor();await page.waitForTimeout(900);assert.equal(server.puts.length,1);assert.equal(server.map.answers.critique,2);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="2"]').getAttribute('aria-pressed'),'true');await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).click();await saved(page);assert.equal(server.map.answers.abus_physique,2);assert.deepEqual(server.map.resources,{prot_famille:1});assert.equal(server.puts.length,2);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('loading the remote conflict version changes the view without another PUT',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});server.map=fixture({critique:2},2);await select(page,'abus_physique',2);await page.getByRole('button',{name:'Charger la version enregistrée',exact:true}).click();await saved(page);assert.equal(server.puts.length,1);assert.equal(await page.locator('[data-factor="abus_physique"] [aria-pressed="true"]').count(),0);await opened(page,'climat');assert.equal(await page.locator('[data-factor="critique"] [data-level="2"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('save errors retain input and retry succeeds without losing focus while typing',async()=>{
   const {page,context,server}=await pageFor();server.putFailure=true;await select(page,'abus_physique',1);await page.getByRole('button',{name:'Réessayer l’enregistrement',exact:true}).waitFor();assert.equal(server.puts.length,1);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="1"]').getAttribute('aria-pressed'),'true');server.putFailure=false;await page.getByRole('button',{name:'Réessayer l’enregistrement',exact:true}).click();await saved(page);assert.equal(server.puts.length,2);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('DELETE follows in-flight PUT and neither timers nor stale anonymous draft recreate the map',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1}),draft:{answers:{abus_sexuel:1},notes:{factors:{},outcomes:{}}}});const first=pause(),release=pause();server.beforePut=async()=>{first.resolve();await release.promise};await select(page,'abus_physique',2);await first.promise;await page.getByRole('button',{name:'Supprimer ma cartographie',exact:true}).click();assert.equal(server.requests.includes('DELETE'),false);release.resolve();await page.waitForFunction(()=>document.querySelector('[data-save]').textContent.includes('Cartographie supprimée.'));await page.waitForTimeout(1000);assert.equal(server.map,null);assert.deepEqual(server.requests.slice(-2),['PUT','DELETE']);await page.reload();await page.waitForTimeout(1000);assert.equal(server.puts.length,1);assert.equal(server.map,null);assert.equal(await page.locator('[data-level][aria-pressed="true"]').count(),0);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('failed deletion is visible and can be retried',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});server.deleteFailure=true;await page.getByRole('button',{name:'Supprimer ma cartographie',exact:true}).click();await page.waitForFunction(()=>document.querySelector('[data-save]').textContent.includes('suppression n’a pas pu être confirmée'));assert.ok(server.map);server.deleteFailure=false;await page.getByRole('button',{name:'Supprimer ma cartographie',exact:true}).click();await page.waitForFunction(()=>document.querySelector('[data-save]').textContent.includes('Cartographie supprimée.'));assert.equal(server.map,null);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('typing during initial GET is preserved and requires an explicit conflict choice',async()=>{
   const pending=pause();const {page,context,server}=await pageFor({map:fixture({critique:1}),beforeGet:()=>pending.promise,waitForLoad:false});await select(page,'abus_physique',1);pending.resolve();await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).waitFor();assert.equal(server.puts.length,0);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="1"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('.carto-theme-panel[open]').count(),1);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="1"]').evaluate(node=>node===document.activeElement),true);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('failed initial GET blocks upload and exposes a working reload action',async()=>{
   const {page,context,server}=await pageFor({getFailure:true});await select(page,'abus_physique',1);await page.waitForTimeout(900);assert.equal(server.puts.length,0);server.getFailure=false;await page.getByRole('button',{name:'Réessayer le chargement',exact:true}).click();await saved(page);assert.equal(server.puts.length,1);assert.equal(server.map.answers.abus_physique,1);assert.deepEqual(server.errors,[]);await context.close();
  });
  for(const width of [1440,390,320])await check('centered choices, associations ordered by experiences and sources at '+width+'px',async()=>{
   const {page,context,server}=await pageFor({width,anonymous:true});await select(page,'critique',1);await select(page,'abus_sexuel',1);await select(page,'abus_physique',1);const bounds=await page.locator('[data-factor="abus_physique"] .carto-choices').evaluate(node=>{const box=node.getBoundingClientRect(),parent=node.parentElement.getBoundingClientRect();return {center:box.x+box.width/2,parent:parent.x+parent.width/2}});assert.ok(Math.abs(bounds.center-bounds.parent)<2);if(width<901){const noteCenter=await page.locator('[data-factor="abus_physique"] > .carto-note-add').evaluate(node=>{const box=node.getBoundingClientRect();return box.x+box.width/2});assert.ok(Math.abs(noteCenter-bounds.center)<2,'explanation button follows answer alignment')}assert.equal(await page.locator('.carto-filters').count(),0);assert.equal(await page.locator('.carto-domain-bar').count(),0);assert.equal(await page.locator('.carto-answer-help,.carto-home-intro,.carto-lead,.carto-reading-guide,.carto-coverage,.carto-mobile-bar').count(),0);if(width<901)await page.locator('[data-tab="results"]').click();assert.equal(await page.locator('.carto-summary-title').count(),0);assert.equal(await page.locator('[data-tab="results"]').innerText(),'Ce qui en ressort');assert.ok(await page.locator('.carto-card').count()>0);const counts=await page.locator('.carto-card-meta').evaluateAll(nodes=>nodes.map(node=>parseInt(node.textContent,10)));assert.ok(counts[0]>1,'several experiences can be associated with one subject');assert.deepEqual(counts,counts.slice().sort((x,y)=>y-x),'subjects are ordered by number of associated experiences');assert.equal(await page.locator('.carto-tier-title').count(),0);assert.equal(await page.locator('details.carto-tier').count(),0);await page.locator('.carto-card-toggle').first().click();await page.locator('[data-results] .carto-sources summary').first().click();assert.ok(await page.locator('[data-results] .carto-sources[open] a').count()>0);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.resolve(root,'../cartographie-review-'+width+'.png'),fullPage:true});assert.deepEqual(server.errors,[]);await context.close();
  });
  console.log(passed+' browser scenarios passed.');
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
