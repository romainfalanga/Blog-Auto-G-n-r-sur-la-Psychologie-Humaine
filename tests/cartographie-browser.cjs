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
  const server={account:options.anonymous?null:{id:'synthetic-member'},map:options.map||null,puts:[],requests:[],meRequests:0,errors:[],beforePut:null,beforeMe:options.beforeMe||null,meFailure:Boolean(options.meFailure),beforeGet:options.beforeGet||null,getFailure:Boolean(options.getFailure),deleteFailure:false,putFailure:false,inflight:0,maxInflight:0};
  page.on('pageerror',error=>server.errors.push(error.message));page.on('dialog',dialog=>dialog.accept());
  if(options.accountDraft)await page.addInitScript(draft=>localStorage.setItem('cartographie-brouillon-v1:synthetic-member',JSON.stringify(draft)),options.accountDraft);
  if(options.draft)await page.addInitScript(draft=>{if(!sessionStorage.getItem('draft-seeded')){localStorage.setItem('cartographie-brouillon-v1',JSON.stringify(draft));sessionStorage.setItem('draft-seeded','1')}},options.draft);
  const initial=options.initial;const initialText=typeof initial==='string'?initial:JSON.stringify(initial);
  const snapshotScript='<script>document.addEventListener("DOMContentLoaded",function(){function snapshot(){return {loading:document.querySelector("[data-cartographie-app]").dataset.loading,loaderVisible:!document.querySelector("[data-loading-status]").hidden,selected:Array.from(document.querySelectorAll("[data-level][aria-pressed=true]")).map(n=>[n.closest("[data-factor]").dataset.factor,n.dataset.level])};}window.cartoAtReady=snapshot();requestAnimationFrame(function(){window.cartoAtFrame=snapshot();});});</script>';
  const html=document.replace('</head>',(initialText?'<script id="cartographie-initial" type="application/json">'+initialText.replace(/</g,'\\u003c')+'</script>':'')+'</head>').replace('</body>',snapshotScript+'</body>');
  await page.route('https://carto.test/**',async route=>{
   const request=route.request(),url=new URL(request.url()),method=request.method();
   if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:html});
   if(url.pathname.startsWith('/css/'))return route.fulfill({contentType:'text/css',body:fs.readFileSync(path.join(root,'static',url.pathname),'utf8')});
   if(url.pathname==='/js/cartographie.js')return route.fulfill({contentType:'application/javascript',body:fs.readFileSync(path.join(root,'static/js/cartographie.js'),'utf8')});
   if(url.pathname==='/js/cartographie-ui.js')return route.fulfill({contentType:'application/javascript',body:source});
   if(url.pathname==='/api/me'){server.meRequests++;if(server.beforeMe)await server.beforeMe();if(server.meFailure)return route.abort('failed');return route.fulfill({contentType:'application/json',body:JSON.stringify({user:server.account})});}
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
  await page.goto('https://carto.test/');
  if(options.waitForLoad!==false){await page.waitForFunction(()=>document.querySelector('[data-cartographie-app]').dataset.loading==='false');await page.locator('.carto-theme-panel').first().waitFor();}
  return {page,context,server};
 }
 async function check(name,fn){if(process.env.CARTOGRAPHY_BROWSER_CASE&&!name.includes(process.env.CARTOGRAPHY_BROWSER_CASE))return;await fn();passed++;console.log('PASS '+name)}
 async function opened(page,id='violences'){const panel=page.locator('[data-theme="'+id+'"]');if(!await panel.evaluate(node=>node.open))await panel.locator('summary').click();return panel}
 async function select(page,id,value){const factor=data.factors.find(f=>f.id===id);await opened(page,factor.group);const button=page.locator('[data-factor="'+id+'"] [data-level="'+value+'"]');await button.click();return button}
 async function saved(page){try{await page.waitForFunction(()=>document.querySelector('[data-save]').dataset.state==='saved',{},{timeout:7000})}catch(error){console.error('Save UI:',await page.locator('[data-save]').innerText());throw error}}
 try{
  for(const anonymous of [false,true])await check('bootstrap '+(anonymous?'anonymous':'member')+' is ready at DOMContentLoaded without initial API roundtrips',async()=>{
   const map=fixture({critique:2});const initial={page:'/cartographie/',responses:{'/api/me':{status:200,body:{user:anonymous?null:{id:'synthetic-member'}}},'/api/cartographie':{status:200,body:{cartographie:map}}}};
   const {page,context,server}=await pageFor({anonymous,map,initial,draft:{answers:{abus_physique:1}}});
   await page.waitForFunction(()=>window.cartoAtFrame);const ready=await page.evaluate(()=>window.cartoAtReady);assert.equal(ready.loading,'false');assert.equal(ready.loaderVisible,false);assert.deepEqual(await page.evaluate(()=>window.cartoAtFrame),ready);assert.deepEqual(ready.selected,anonymous?[['abus_physique','1']]:[['critique','1']]);assert.equal(server.meRequests,0);assert.deepEqual(server.requests,[]);
   if(!anonymous){server.map=fixture({abus_sexuel:1},2);await select(page,'critique',0);await page.getByRole('button',{name:'Charger la version enregistrée',exact:true}).click();await saved(page);assert.deepEqual(server.requests,['PUT','GET']);assert.equal(await page.locator('[data-factor="abus_sexuel"] [data-level="1"]').getAttribute('aria-pressed'),'true');}
   assert.deepEqual(server.errors,[]);await context.close();
  });
  for(const initial of ['{broken',{page:'/wrong/',responses:{'/api/me':{status:200,body:{user:null}}}}])await check('invalid bootstrap falls back to ordinary session and map requests '+(typeof initial),async()=>{
   const {page,context,server}=await pageFor({initial,map:fixture({critique:2})});assert.equal(server.meRequests,1);assert.deepEqual(server.requests,['GET']);assert.equal(await page.locator('[data-factor="critique"] [data-level="1"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('bootstrap map errors remain errors and a retry consumes no stale bootstrap',async()=>{
   const initial={page:'/cartographie/',responses:{'/api/me':{status:200,body:{user:{id:'synthetic-member'}}},'/api/cartographie':{status:503,body:{error:'indisponible'}}}};
   const {page,context,server}=await pageFor({initial,map:fixture({critique:2})});assert.equal(server.meRequests,0);assert.deepEqual(server.requests,[]);await page.getByRole('button',{name:'Réessayer le chargement',exact:true}).click();await saved(page);assert.deepEqual(server.requests,['GET']);assert.equal(await page.locator('[data-factor="critique"] [data-level="1"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('bootstrap session errors do not pretend the visitor is verified anonymous',async()=>{
   const initial={page:'/cartographie/',responses:{'/api/me':{status:503,body:{error:'indisponible'}}}};
   const {page,context,server}=await pageFor({initial});assert.match(await page.locator('[data-save]').innerText(),/Connexion non vérifiée/);assert.equal(server.meRequests,0);assert.deepEqual(server.requests,[]);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('six independent closed accordions, anonymous local-only answers and stable focus',async()=>{
   const {page,context,server}=await pageFor({anonymous:true});assert.equal(await page.locator('.carto-theme-panel').count(),6);assert.equal(await page.locator('.carto-theme-panel[open]').count(),0);assert.equal(await page.locator('[data-theme="ressources"]').count(),0);assert.equal(await page.locator('input[type=checkbox]').count(),0);
   const button=await select(page,'abus_physique',1);assert.equal(await button.evaluate(node=>node===document.activeElement),true);await opened(page,'negligence');assert.equal(await page.locator('.carto-theme-panel[open]').count(),2);assert.equal(await button.getAttribute('aria-pressed'),'true');
   const row=page.locator('[data-factor="abus_physique"]');await row.locator('.carto-note-add').click();const note=row.locator('textarea');await note.fill('Une précision personnelle synthétique.');await page.waitForTimeout(900);assert.equal(await note.evaluate(node=>node===document.activeElement),true);assert.equal(server.puts.length,0);assert.equal(server.requests.length,0);assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')).answers.abus_physique),1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('connected first answer autosaves with consent and explicit Non survives a reload',async()=>{
   const {page,context,server}=await pageFor();await select(page,'abus_physique',0);await saved(page);assert.equal(server.puts.length,1);assert.equal(server.puts[0].consent,true);assert.equal(server.map.answers.abus_physique,0);assert.equal(await page.locator('.carto-card').count(),0);await page.reload();await page.locator('.carto-theme-panel').first().waitFor();await opened(page);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="0"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('.carto-theme-panel[open]').count(),1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('stored retired feedback and notes survive editing without moving to a new card',async()=>{
   const map=fixture({abus_physique:1});map.feedback={att_desorganise:'oui',icd_6b81:'oui',pat_migration_adjustment:'non'};map.notes.outcomes={att_desorganise:'Mon vécu adulte enregistré.'};
   const {page,context,server}=await pageFor({map});await select(page,'abus_physique',0);await saved(page);
   assert.deepEqual(server.map.feedback,map.feedback);assert.deepEqual(server.map.notes,map.notes);
   assert.equal(Object.hasOwn(server.map.feedback,'att_desorganisation_enfant'),false);assert.equal(Object.hasOwn(server.map.feedback,'icd_6b81_boulimie'),false);
   assert.equal(await page.locator('[data-focus="feedback:att_desorganise:oui"]').count(),0);
   await page.reload();await saved(page);assert.deepEqual(server.map.feedback,map.feedback);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('anonymous drafts retain historical identifiers and unlinked experiences show an honest empty state',async()=>{
   const draft={answers:{},feedback:{att_desorganise:'oui'},notes:{factors:{},outcomes:{att_desorganise:'Ancienne note.'}},resources:{}};
   const {page,context,server}=await pageFor({anonymous:true,draft});await select(page,'migration',1);
   const stored=await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')));
   assert.deepEqual(stored.feedback,draft.feedback);assert.deepEqual(stored.notes,draft.notes);assert.equal(stored.answers.migration,1);
   assert.equal(await page.locator('.carto-card').count(),0);assert.match(await page.locator('[data-results]').innerText(),/Aucun rapprochement suffisamment documenté/);
   assert.equal(server.puts.length,0);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('reselecting a saved answer creates no pending draft or conflict after reload',async()=>{
   const {page,context,server}=await pageFor({map:fixture({critique:1})});await select(page,'critique',1);assert.equal(await page.evaluate(()=>localStorage.getItem('cartographie-brouillon-v1:synthetic-member')),null);await page.reload();await saved(page);await page.waitForTimeout(900);assert.equal(server.puts.length,0);assert.equal(await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).count(),0);assert.equal(server.map.answers.critique,1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('edits made during PUT are serialized and existing hidden resources retained',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});const first=pause(),release=pause();server.beforePut=async()=>{if(server.puts.length===1){first.resolve();await release.promise}};await select(page,'abus_physique',0);await first.promise;await select(page,'abus_sexuel',1);assert.equal(server.puts.length,1);release.resolve();await saved(page);assert.equal(server.puts.length,2);assert.equal(server.maxInflight,1);assert.equal(server.puts[1].version,2);assert.equal(server.map.answers.abus_sexuel,1);assert.deepEqual(server.map.resources,{prot_amities:1});assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('409 never overwrites silently and explicit local choice keeps current stored resources',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});server.map=fixture({critique:2},2);server.map.resources={prot_famille:1};await select(page,'abus_physique',0);await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).waitFor();await page.waitForTimeout(900);assert.equal(server.puts.length,1);assert.equal(server.map.answers.critique,2);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="0"]').getAttribute('aria-pressed'),'true');await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).click();await saved(page);assert.equal(server.map.answers.abus_physique,0);assert.deepEqual(server.map.resources,{prot_famille:1});assert.equal(server.puts.length,2);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('loading the remote conflict version changes the view without another PUT',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});server.map=fixture({critique:2},2);await select(page,'abus_physique',0);await page.getByRole('button',{name:'Charger la version enregistrée',exact:true}).click();await saved(page);assert.equal(server.puts.length,1);assert.equal(await page.locator('[data-factor="abus_physique"] [aria-pressed="true"]').count(),0);await opened(page,'climat');assert.equal(await page.locator('[data-factor="critique"] [data-level="1"]').getAttribute('aria-pressed'),'true');assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('save errors retain input and retry succeeds without losing focus while typing',async()=>{
   const {page,context,server}=await pageFor();server.putFailure=true;await select(page,'abus_physique',1);await page.getByRole('button',{name:'Réessayer l’enregistrement',exact:true}).waitFor();assert.equal(server.puts.length,1);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="1"]').getAttribute('aria-pressed'),'true');server.putFailure=false;await page.getByRole('button',{name:'Réessayer l’enregistrement',exact:true}).click();await saved(page);assert.equal(server.puts.length,2);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('a remote deletion during editing requires a deliberate conflict choice',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1})});server.map=null;await select(page,'abus_physique',0);await page.getByRole('button',{name:'Charger la version enregistrée',exact:true}).waitFor();assert.equal(server.puts.length,1);assert.equal(server.map,null);await page.getByRole('button',{name:'Charger la version enregistrée',exact:true}).click();await page.waitForFunction(()=>document.querySelector('[data-save]').dataset.state==='deleted');assert.equal(await page.locator('[data-level][aria-pressed="true"]').count(),0);assert.equal(server.map,null);await page.waitForTimeout(900);assert.equal(server.puts.length,1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('the results page has no crisis banner or saved-map deletion button',async()=>{
   const {page,context,server}=await pageFor({map:fixture({abus_physique:1,abus_sexuel:1})});assert.ok(await page.locator('.carto-card').count()>0);assert.equal(await page.locator('.carto-note-urgent,[data-delete]').count(),0);assert.equal(await page.getByRole('button',{name:'Supprimer ma cartographie',exact:true}).count(),0);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('slow session and map never paint the anonymous draft or an empty member map',async()=>{
   const session=pause(),map=pause();let mapRequested=false;const draft={answers:{abus_physique:1},notes:{factors:{abus_physique:'ANONYMOUS DRAFT MUST NOT FLASH'},outcomes:{}}};
   const {page,context,server}=await pageFor({width:390,draft,map:fixture({critique:2}),beforeMe:()=>session.promise,beforeGet:()=>{mapRequested=true;return map.promise},waitForLoad:false});
   for(let i=0;i<2;i++){assert.equal(await page.locator('[data-cartographie-app]').getAttribute('aria-busy'),'true');assert.equal(await page.locator('.carto-theme-panel').count(),0);assert.equal(await page.locator('[data-tab="results"]').isDisabled(),true);assert.equal(await page.locator('[data-loading-status]').isVisible(),true);assert.equal(await page.getByText('ANONYMOUS DRAFT MUST NOT FLASH').count(),0);if(!i){session.resolve();for(let n=0;n<100&&!mapRequested;n++)await page.waitForTimeout(10);assert.ok(mapRequested)}}
   const artifacts=path.join(root,'work','cartographie-chargement');fs.mkdirSync(artifacts,{recursive:true});await page.screenshot({path:path.join(artifacts,'attente-390.png'),fullPage:true});
   map.resolve();await saved(page);assert.equal(await page.locator('[data-cartographie-app]').getAttribute('aria-busy'),'false');await opened(page,'climat');assert.equal(await page.locator('[data-factor="critique"] [data-level="1"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('[data-factor="abus_physique"] [aria-pressed="true"]').count(),0);assert.equal(server.puts.length,0);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('account drafts are revealed only with their account and require an explicit conflict choice',async()=>{
   const map=pause(),draft={answers:{abus_physique:1},notes:{factors:{abus_physique:'ACCOUNT DRAFT'},outcomes:{}}};const {page,context,server}=await pageFor({accountDraft:draft,map:fixture({critique:1}),beforeGet:()=>map.promise,waitForLoad:false});assert.equal(await page.getByText('ACCOUNT DRAFT').count(),0);assert.equal(await page.locator('.carto-theme-panel').count(),0);map.resolve();await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).waitFor();await opened(page);assert.equal(await page.locator('[data-factor="abus_physique"] textarea').inputValue(),'ACCOUNT DRAFT');assert.equal(server.puts.length,0);await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).click();await saved(page);assert.equal(server.map.answers.abus_physique,1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('offline session checking releases the anonymous draft without loading or saving an account',async()=>{
   const draft={answers:{critique:'unknown'},notes:{factors:{},outcomes:{}}};const {page,context,server}=await pageFor({meFailure:true,draft});assert.match(await page.locator('[data-save]').innerText(),/Connexion non vérifiée/);await select(page,'critique',1);assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')).answers.critique),1);assert.equal(server.requests.length,0);assert.equal(server.puts.length,0);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('typing during a retry GET keeps local edits and requires an explicit conflict choice',async()=>{
   const {page,context,server}=await pageFor({getFailure:true,map:fixture({critique:1})});const pending=pause();server.beforeGet=()=>pending.promise;server.getFailure=false;await page.getByRole('button',{name:'Réessayer le chargement',exact:true}).click();await select(page,'abus_physique',1);pending.resolve();await page.getByRole('button',{name:'Garder mes réponses affichées',exact:true}).waitFor();assert.equal(server.puts.length,0);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="1"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('.carto-theme-panel[open]').count(),1);assert.equal(await page.locator('[data-factor="abus_physique"] [data-level="1"]').evaluate(node=>node===document.activeElement),true);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('an anonymous draft seeds only a confirmed empty account and survives an initial map error',async()=>{
   const draft={answers:{critique:2},notes:{factors:{critique:'LOCAL DRAFT'},outcomes:{}}};const {page,context,server}=await pageFor({draft,getFailure:true});assert.equal(await page.locator('[data-level][aria-pressed="true"]').count(),0);assert.equal(await page.getByText('LOCAL DRAFT').count(),0);assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('cartographie-brouillon-v1')).answers.critique),2);server.getFailure=false;await page.getByRole('button',{name:'Réessayer le chargement',exact:true}).click();await saved(page);assert.equal(server.map.answers.critique,2);assert.equal(server.map.notes.factors.critique,'LOCAL DRAFT');assert.equal(server.puts.length,1);assert.deepEqual(server.errors,[]);await context.close();
  });
  await check('failed initial GET blocks upload and exposes a working reload action',async()=>{
   const {page,context,server}=await pageFor({getFailure:true});await select(page,'abus_physique',1);await page.waitForTimeout(900);assert.equal(server.puts.length,0);server.getFailure=false;await page.getByRole('button',{name:'Réessayer le chargement',exact:true}).click();await saved(page);assert.equal(server.puts.length,1);assert.equal(server.map.answers.abus_physique,1);assert.deepEqual(server.errors,[]);await context.close();
  });
  for(const width of [1440,390,320])await check('centered choices, associations ordered by experiences and sources at '+width+'px',async()=>{
   const {page,context,server}=await pageFor({width,anonymous:true});assert.equal(await page.evaluate(()=>window.scrollY),0,'the page does not scroll on load');if(width<901){await page.locator('[data-tab="results"]').click();assert.equal(await page.evaluate(()=>window.scrollY),0,'switching tab at the top does not scroll');await page.locator('[data-tab="form"]').click();assert.equal(await page.evaluate(()=>window.scrollY),0)}await select(page,'critique',1);await select(page,'abus_sexuel',1);await select(page,'abus_physique',1);const bounds=await page.locator('[data-factor="abus_physique"] .carto-choices').evaluate(node=>{const box=node.getBoundingClientRect(),parent=node.parentElement.getBoundingClientRect();return {center:box.x+box.width/2,parent:parent.x+parent.width/2}});assert.ok(Math.abs(bounds.center-bounds.parent)<2);if(width<901){const noteCenter=await page.locator('[data-factor="abus_physique"] > .carto-note-add').evaluate(node=>{const box=node.getBoundingClientRect();return box.x+box.width/2});assert.ok(Math.abs(noteCenter-bounds.center)<2,'explanation button follows answer alignment')}assert.equal(await page.locator('.carto-filters').count(),0);assert.equal(await page.locator('.carto-domain-bar').count(),0);assert.equal(await page.locator('.carto-answer-help,.carto-home-intro,.carto-lead,.carto-reading-guide,.carto-coverage,.carto-mobile-bar').count(),0);if(width<901)await page.locator('[data-tab="results"]').click();assert.equal(await page.locator('.carto-summary-title').count(),0);assert.equal(await page.locator('[data-tab="results"]').innerText(),'Ce qui en ressort');assert.ok(await page.locator('.carto-card').count()>0);const counts=await page.locator('article.carto-card').evaluateAll(nodes=>nodes.map(node=>Number(node.dataset.associated)));assert.ok(Math.max(...counts)>1,'several experiences can be associated with one subject');const contextual=new Set(data.outcomes.filter(o=>o.contextual_only).map(o=>o.label));const titles=await page.locator('.carto-card-title').allInnerTexts();assert.ok(!titles.some(t=>contextual.has(t)),'contextual subjects are not displayed');const shown=await page.locator('article.carto-card[data-section="top"]').evaluateAll(nodes=>nodes.map(node=>node.dataset.outcome));const expected=await page.evaluate(()=>{const d=JSON.parse(document.getElementById('cartographie-data').textContent);return window.Cartographie.compute(d,{critique:1,abus_sexuel:1,abus_physique:1},{}).sections.top.map(r=>r.outcome.id)});assert.deepEqual(shown,expected,'the five first cards follow the shared ranking');assert.equal(await page.locator('.carto-rank-note').count(),1);assert.equal(await page.getByText('Repère de contexte').count(),0);assert.equal(await page.locator('.carto-tier-title').count(),0);assert.equal(await page.locator('details.carto-tier').count(),0);await page.locator('.carto-detail').first().click();assert.equal(await page.locator('[data-results] .carto-sources,[data-results] .carto-kind').count(),0);assert.ok(await page.locator('.carto-link-text').count()>0);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true);await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.resolve(root,'../cartographie-review-'+width+'.png'),fullPage:true});assert.deepEqual(server.errors,[]);await context.close();
  });
  console.log(passed+' browser scenarios passed.');
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
