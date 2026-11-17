// Run with: PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/cartographie-browser.cjs
// Browser-only integration checks; every API response is synthetic and local.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'data/cartographie.json'),'utf8'));
const artifacts=path.join(root,'work','cartographie-choix');fs.mkdirSync(artifacts,{recursive:true});
const source=fs.readFileSync(path.join(root,'static/js/cartographie-ui.js'),'utf8');
const template=fs.readFileSync(path.join(root,'layouts/_default/cartographie.html'),'utf8');
const markup=template.slice(template.indexOf('<div class="container-wide carto" data-cartographie-app'),template.lastIndexOf('{{ end }}')).replace(/{{[^}]*}}/g,match=>match.includes('$data | jsonify')?JSON.stringify(data):match.includes('readFile')?'test':'');
const document='<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/css/style.css"></head><body><header style="height:130px">Décode ton esprit</header><main>'+markup+'</main></body></html>';
const fixture=(answers={},version=1)=>({answers,version,feedback:{},notes:{factors:{},outcomes:{}},resources:{prot_amities:1},ambre_notes:[],updated_at:1794830400,consent:true,locked:true});
const pause=()=>{let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}};

(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});let passed=0;
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
  for(const width of [390,1440])await check('uncertainty and new dimensions remain distinct at '+width+'px',async()=>{
   const {page,context,server}=await pageFor({width,anonymous:true});
   assert.equal(await page.locator('[data-factor="controle"]').count(),0);
   const unknown=await select(page,'abus_physique','unknown');assert.equal(await unknown.getAttribute('aria-pressed'),'true');assert.equal(await page.locator('[data-factor="abus_physique"]').evaluate(n=>n.classList.contains('is-checked')),false);assert.equal(await page.locator('article.carto-card').count(),0);
   await opened(page,'climat');assert.equal(await page.locator('[data-factor="parents_stricts"] [aria-pressed="true"]').count(),0);assert.equal(await page.locator('[data-factor="controle_psychologique"] [aria-pressed="true"]').count(),0);
   await select(page,'parents_stricts',1);await select(page,'controle_psychologique','unknown');assert.equal(await page.locator('article.carto-card').count(),0);assert.match(await page.locator('[data-factor="parents_stricts"]').innerText(),/Aucun rapprochement scientifique/);
   const draft=await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')));assert.deepEqual(draft.answers,{abus_physique:'unknown',parents_stricts:1,controle_psychologique:'unknown'});assert.equal(server.puts.length,0);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:path.join(artifacts,'nouvelles-questions-'+width+'.png'),fullPage:true});
   await page.reload();await opened(page,'climat');assert.equal(await page.locator('[data-factor="controle_psychologique"] [data-level="unknown"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  for(const width of [390,1440])await check('old Souvent and combined-control history survive unrelated edits at '+width+'px',async()=>{
   const map=fixture({adoption:2,controle:2});map.notes.factors.controle='Les deux sens étaient mêlés dans mon ancienne réponse.';const {page,context,server}=await pageFor({width,map});await opened(page,'famille');
   for(const factor of data.factors.filter(f=>f.response_mode==='binary'))assert.equal(await page.locator('[data-factor="'+factor.id+'"] [data-level="2"]').count(),0);
   assert.match(await page.locator('[data-factor="adoption"]').innerText(),/Ancienne réponse : Souvent/);assert.equal(await page.locator('[data-factor="adoption"] [aria-pressed="true"]').count(),0);
   await opened(page,'climat');const old=page.locator('[data-factor="controle"]');assert.match(await old.innerText(),/Ancienne question.*réponse conservée : Souvent/);assert.equal(await old.locator('[data-level]').count(),0);assert.equal(await old.locator('textarea').inputValue(),map.notes.factors.controle);
   await select(page,'parents_stricts',0);await saved(page);assert.equal(server.map.answers.controle,2);assert.equal(server.map.answers.adoption,2);assert.equal(Object.hasOwn(server.map.answers,'controle_psychologique'),false);assert.equal(server.map.notes.factors.controle,map.notes.factors.controle);
   await page.screenshot({path:path.join(artifacts,'historique-'+width+'.png'),fullPage:true});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   await page.reload();await saved(page);await opened(page,'famille');assert.match(await page.locator('[data-factor="adoption"]').innerText(),/Ancienne réponse : Souvent/);
   await select(page,'adoption','unknown');await saved(page);assert.equal(server.map.answers.adoption,'unknown');assert.equal(server.map.answers.controle,2);assert.equal(await page.locator('[data-factor="adoption"] .carto-legacy-frequency').isVisible(),false);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('an uncertain alternative never erases another answer, including when a yes is selected',async()=>{
   const {page,context,server}=await pageFor({map:fixture({separation_conflit:1})});await select(page,'divorce','unknown');await saved(page);assert.deepEqual(server.map.answers,{separation_conflit:1,divorce:'unknown'});await select(page,'separation_conflit',2);await saved(page);assert.deepEqual(server.map.answers,{separation_conflit:2,divorce:'unknown'});assert.equal(await page.locator('[data-factor="divorce"] [data-level="unknown"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('409 keeps uncertainty, old answers and typed notes until an explicit conflict choice',async()=>{
   const map=fixture({controle:2,adoption:2,abus_physique:0});map.notes.factors.controle='Historique conservé.';const {page,context,server}=await pageFor({map});server.map=fixture({controle:2,adoption:2,abus_physique:1},2);server.map.notes=map.notes;
   await select(page,'abus_physique','unknown');await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).waitFor();assert.equal(server.map.answers.abus_physique,1);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="unknown"]').getAttribute('aria-pressed'),'true');await opened(page,'climat');await page.locator('[data-factor="controle"] textarea').fill('Ma précision en conflit.');
   await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).click();await saved(page);assert.equal(server.map.answers.abus_physique,'unknown');assert.equal(server.map.answers.controle,2);assert.equal(server.map.answers.adoption,2);assert.equal(server.map.notes.factors.controle,'Ma précision en conflit.');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('erasing the old combined answer never chooses the replacement fields or erases its note',async()=>{
   const map=fixture({controle:2});map.notes.factors.controle='Une précision à conserver.';const {page,context,server}=await pageFor({map});await opened(page,'climat');await page.getByRole('button',{name:'Effacer cette ancienne réponse',exact:true}).click();await saved(page);assert.deepEqual(server.map.answers,{});assert.equal(server.map.notes.factors.controle,map.notes.factors.controle);assert.equal(await page.locator('[data-factor="parents_stricts"] [aria-pressed="true"]').count(),0);assert.equal(await page.locator('[data-factor="controle_psychologique"] [aria-pressed="true"]').count(),0);assert.equal(await page.evaluate(()=>document.activeElement.closest('[data-factor]').dataset.factor),'parents_stricts');assert.deepEqual(server.errors,[]);await context.close();
  });
  console.log(passed+' parcours nouveaux choix validés.');
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
