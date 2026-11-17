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
const renderTemplate=input=>input.replace(/{{[^}]*}}/g,match=>match.includes('jsonify')?JSON.stringify(data):match.includes('safeJS')&&match.includes('readFile')?fs.readFileSync(path.join(root,match.match(/readFile "([^"]+)"/)[1]),'utf8'):match.includes('readFile')?'test':'');
// Match the real page: data and both inline scripts execute in the head, before the app exists.
const head=renderTemplate(template.slice(0,template.indexOf('{{ define "main" }}')));
const markup=renderTemplate(template.slice(template.indexOf('<div class="container-wide carto" data-cartographie-app'),template.lastIndexOf('{{ end }}')));
const document='<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/css/style.css">'+head+'</head><body><header style="height:130px">Décode ton esprit</header><main>'+markup+'</main></body></html>';
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
   if(url.pathname.startsWith('/css/'))return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'static',url.pathname),'utf8')});
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
  for(const width of [320,390,1440])await check('uncertainty and new dimensions remain distinct at '+width+'px',async()=>{
   const {page,context,server}=await pageFor({width,anonymous:true});
   assert.equal(await page.locator('[data-factor="controle"]').count(),0);
   await opened(page,'violences');
   const dividers=await page.locator('[data-factor="abus_physique"] .carto-choice').evaluateAll(nodes=>nodes.map(n=>({left:getComputedStyle(n).borderLeftWidth,top:getComputedStyle(n).borderTopWidth,rect:{x:n.getBoundingClientRect().x,y:n.getBoundingClientRect().y,width:n.getBoundingClientRect().width},clipped:n.scrollWidth>n.clientWidth})));
   assert.equal(dividers.length,3);assert.deepEqual(dividers.map(x=>x.left),width<601?['0px','1px','0px']:['0px','1px','1px']);assert.deepEqual(dividers.map(x=>x.top),width<601?['0px','0px','1px']:['0px','0px','0px']);assert.ok(dividers.every(x=>!x.clipped));
   if(width<601){assert.equal(dividers[0].rect.x,dividers[2].rect.x);assert.ok(Math.abs(dividers[2].rect.width-dividers[0].rect.width-dividers[1].rect.width)<2);assert.ok(dividers[2].rect.y>dividers[0].rect.y)}
   for(const factor of data.factors.filter(f=>!f.historical)){assert.deepEqual(await page.locator('[data-factor="'+factor.id+'"] .carto-choice').allTextContents(),['Non','Oui','Je ne sais pas']);}
   await opened(page,'famille');const binary=page.locator('[data-factor="adoption"] .carto-choice');assert.equal(await binary.count(),3);assert.equal(await binary.filter({hasText:'Souvent'}).count(),0);
   await opened(page,'violences');await page.locator('[data-factor="abus_physique"] .carto-choices').screenshot({path:path.join(artifacts,'separateurs-'+width+'.png')});
   const unknown=await select(page,'abus_physique','unknown');assert.equal(await unknown.getAttribute('aria-pressed'),'true');assert.equal(await page.locator('[data-factor="abus_physique"]').evaluate(n=>n.classList.contains('is-checked')),false);assert.equal(await page.locator('article.carto-card').count(),0);
   await opened(page,'climat');assert.equal(await page.locator('[data-factor="parents_stricts"] [aria-pressed="true"]').count(),0);assert.equal(await page.locator('[data-factor="controle_psychologique"] [aria-pressed="true"]').count(),0);
   await select(page,'parents_stricts',1);await select(page,'controle_psychologique','unknown');assert.equal(await page.locator('article.carto-card').count(),0);assert.match(await page.locator('[data-factor="parents_stricts"]').innerText(),/Aucun rapprochement scientifique/);
   const draft=await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')));assert.deepEqual(draft.answers,{abus_physique:'unknown',parents_stricts:1,controle_psychologique:'unknown'});assert.equal(server.puts.length,0);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:path.join(artifacts,'nouvelles-questions-'+width+'.png'),fullPage:true});
   await page.reload();await opened(page,'climat');assert.equal(await page.locator('[data-factor="controle_psychologique"] [data-level="unknown"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  for(const width of [390,1440])await check('old Souvent and combined-control history survive unrelated edits at '+width+'px',async()=>{
   const map=fixture({adoption:2,controle:2,abus_physique:2});map.notes.factors.controle='Les deux sens étaient mêlés dans mon ancienne réponse.';const {page,context,server}=await pageFor({width,map});await opened(page,'famille');
   for(const factor of data.factors)assert.equal(await page.locator('[data-factor="'+factor.id+'"] [data-level="2"]').count(),0);
   assert.equal(await page.locator('[data-factor="adoption"] [data-level="1"]').getAttribute('aria-pressed'),'true');assert.equal(server.puts.length,0);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="1"]').getAttribute('aria-pressed'),'true');
   await opened(page,'climat');const old=page.locator('[data-factor="controle"]');assert.match(await old.innerText(),/Ancienne question.*réponse conservée : Oui/);assert.equal(await old.locator('[data-level]').count(),0);assert.equal(await old.locator('.carto-note-preview').innerText(),map.notes.factors.controle);
   await select(page,'parents_stricts',0);await saved(page);assert.equal(server.map.answers.controle,2);assert.equal(server.map.answers.adoption,2);assert.equal(server.map.answers.abus_physique,2);assert.equal(Object.hasOwn(server.map.answers,'controle_psychologique'),false);assert.equal(server.map.notes.factors.controle,map.notes.factors.controle);
   await page.screenshot({path:path.join(artifacts,'historique-'+width+'.png'),fullPage:true});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   await page.reload();await saved(page);await opened(page,'famille');assert.equal(await page.locator('[data-factor="adoption"] [data-level="1"]').getAttribute('aria-pressed'),'true');await select(page,'abus_physique',1);await saved(page);assert.equal(server.map.answers.abus_physique,1);assert.equal(server.map.answers.adoption,2);
   await select(page,'adoption','unknown');await saved(page);assert.equal(server.map.answers.adoption,'unknown');assert.equal(server.map.answers.controle,2);assert.equal(await page.locator('.carto-legacy-frequency').count(),0);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('an uncertain alternative never erases another answer, including when a yes is selected',async()=>{
   const {page,context,server}=await pageFor({map:fixture({separation_conflit:2})});await select(page,'divorce','unknown');await saved(page);assert.deepEqual(server.map.answers,{separation_conflit:2,divorce:'unknown'});await select(page,'separation_conflit',1);await saved(page);assert.deepEqual(server.map.answers,{separation_conflit:1,divorce:'unknown'});assert.equal(await page.locator('[data-factor="divorce"] [data-level="unknown"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('409 keeps uncertainty, old answers and typed notes until an explicit conflict choice',async()=>{
   const map=fixture({controle:2,adoption:2,abus_physique:0});map.notes.factors.controle='Historique conservé.';const {page,context,server}=await pageFor({map});server.map=fixture({controle:2,adoption:2,abus_physique:1},2);server.map.notes=map.notes;
   await select(page,'abus_physique','unknown');await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).waitFor();assert.equal(server.map.answers.abus_physique,1);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="unknown"]').getAttribute('aria-pressed'),'true');await opened(page,'climat');await page.locator('[data-focus="note:factors:controle"]').click();await page.locator('[data-note-dialog] textarea').fill('Ma précision en conflit.');await page.locator('[data-note-save]').click();
   await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).click();await saved(page);assert.equal(server.map.answers.abus_physique,'unknown');assert.equal(server.map.answers.controle,2);assert.equal(server.map.answers.adoption,2);assert.equal(server.map.notes.factors.controle,'Ma précision en conflit.');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('erasing the old combined answer never chooses the replacement fields or erases its note',async()=>{
   const map=fixture({controle:2});map.notes.factors.controle='Une précision à conserver.';const {page,context,server}=await pageFor({map});await opened(page,'climat');await page.getByRole('button',{name:'Effacer cette ancienne réponse',exact:true}).click();await saved(page);assert.deepEqual(server.map.answers,{});assert.equal(server.map.notes.factors.controle,map.notes.factors.controle);assert.equal(await page.locator('[data-factor="parents_stricts"] [aria-pressed="true"]').count(),0);assert.equal(await page.locator('[data-factor="controle_psychologique"] [aria-pressed="true"]').count(),0);assert.equal(await page.evaluate(()=>document.activeElement.closest('[data-factor]').dataset.factor),'parents_stricts');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('all result surfaces hide evidence and classifications while retaining explanations and internal metadata',async()=>{
   const answers=Object.fromEntries(data.factors.map(f=>[f.id,2]));const {page,context,server}=await pageFor({map:fixture(answers)});
   if(await page.locator('[data-tab="results"]').isVisible())await page.locator('[data-tab="results"]').click();
   while(await page.getByRole('button',{name:'Voir le détail',exact:true}).count())await page.getByRole('button',{name:'Voir le détail',exact:true}).first().click();
   assert.ok(await page.locator('.carto-card-body').count()>30);
   const text=await page.locator('[data-results]').innerText();assert.doesNotMatch(text,/Preuves (solides|modérées|limitées)|preuves agrégées|CIM[-‑ ]?11|DSM[-‑ ]?(IV|5)|Correspondance de définition|Symptômes dépressifs durables ou épisodes dépressifs/);
   assert.equal(await page.locator('.carto-badge,.carto-strength,.carto-card-meta').count(),0);
   assert.match(await page.locator('[data-outcome="dep_generale"] .carto-desc').innerText(),/Tristesse, perte d’intérêt, fatigue/);
   assert.match(text,/Dans une cohorte australienne/);assert.match(text,/chez les enfants exposés|Chez les enfants exposés/);assert.match(text,/ne vaut pas diagnostic individuel/);
   await page.locator('[data-focus="feedback:dep_generale:non"]').click();
   assert.equal(await page.locator('.carto-row small').count(),0);assert.doesNotMatch(await page.locator('[data-results]').innerText(),/Preuves (solides|modérées|limitées)|CIM[-‑ ]?11|DSM[-‑ ]?(IV|5)/);
   if(await page.locator('[data-tab="form"]').isVisible())await page.locator('[data-tab="form"]').click();await opened(page,'violences');await page.locator('[data-factor="abus_physique"] .carto-see').click();
   const dialogText=await page.locator('[data-dialog-body]').innerText();assert.match(dialogText,/Associations étudiées/);assert.match(dialogText,/Chez les enfants exposés/);assert.doesNotMatch(dialogText,/Preuves (solides|modérées|limitées)|CIM[-‑ ]?11|DSM[-‑ ]?(IV|5)|Correspondance de définition/);
   await page.locator('[data-dialog-close]').click();await page.locator('[data-scope-open]').click();assert.doesNotMatch(await page.locator('[data-dialog-body]').innerText(),/CIM[-‑ ]?11|DSM[-‑ ]?(IV|5)|HiTOP|AMPD/);
   for(const factor of data.factors){const links=data.links.filter(l=>l.factor===factor.id);if(links.length&&links.every(l=>l.relation_type==='classification_context'||l.relation_type==='definitional'||l.definitional))assert.equal(await page.locator('[data-factor="'+factor.id+'"] .carto-see').count(),0);}
   const internal=await page.locator('#cartographie-data').textContent();assert.deepEqual(JSON.parse(internal),data);assert.ok(data.links.some(l=>l.strength==='forte'));assert.ok(data.outcomes.some(o=>o.classification_codes));assert.ok(data.links.some(l=>l.relation_type==='classification_context'));
   assert.deepEqual(server.errors,[]);await context.close();
  });
  console.log(passed+' parcours nouveaux choix validés.');
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
