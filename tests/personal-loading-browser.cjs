// Real Worker + in-memory D1: initial page ready at first frame, prepared navigation and persistence.
// CHROMIUM_PATH=/usr/bin/chromium node tests/personal-loading-browser.cjs
const {chromium}=require('playwright');
const {DatabaseSync}=require('node:sqlite');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),out=process.env.TEST_ARTIFACTS_DIR||path.join(root,'work/chargement-pages/browser');fs.mkdirSync(out,{recursive:true});
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve}}
(async()=>{
 const {default:worker}=await import(require('node:url').pathToFileURL(path.join(root,'journal/src/worker.mjs')).href);
 const db=new DatabaseSync(':memory:');for(const f of fs.readdirSync(path.join(root,'journal/migrations')).filter(f=>f.endsWith('.sql')).sort())db.exec(fs.readFileSync(path.join(root,'journal/migrations',f),'utf8'));
 function prepare(sql){let params=[];return {bind(...p){params=p;return this},async first(){return db.prepare(sql).get(...params)||null},async all(){return {results:db.prepare(sql).all(...params)}},async run(){return {meta:{changes:db.prepare(sql).run(...params).changes}}}}}
 let queue=Promise.resolve();const env={PUBLIC_ORIGIN:'https://loading.test',FREE_ACCESS_ENABLED:'true',DB:{prepare,batch(items){const task=queue.then(async()=>{db.exec('BEGIN');try{const result=[];for(const item of items)result.push(await item.run());db.exec('COMMIT');return result}catch(e){db.exec('ROLLBACK');throw e}});queue=task.catch(()=>{});return task}}};
 const tasks=[],ctx={waitUntil(p){tasks.push(p)}},errors=[],gates=new Map(),requests=[];let failure=false;
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined}),context=await browser.newContext({viewport:{width:1440,height:900}}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 function hold(url,method='GET'){const gate={arrived:deferred(),release:deferred()},key=method+' '+url;gates.set(key,gate);return {arrived:gate.arrived.promise,release(){gates.delete(key);gate.release.resolve()}}}
 await page.route('https://loading.test/**',async route=>{
  const r=route.request(),url=new URL(r.url());requests.push({path:url.pathname,method:r.method(),type:r.resourceType()});
  if(!url.pathname.startsWith('/api/')&&r.resourceType()!=='document')return route.fulfill({status:204,body:''});
  const gate=gates.get(r.method()+' '+url.pathname);if(gate){gate.arrived.resolve();await gate.release.promise}
  if(failure&&url.pathname==='/api/self-profile'&&r.method()==='GET'){failure=false;return route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Indisponibilité de test'})})}
  const result=await worker.fetch(new Request(r.url(),{method:r.method(),headers:r.headers(),body:r.postData()||undefined}),env,ctx);await route.fulfill({status:result.status,headers:Object.fromEntries(result.headers),body:await result.text()});
 });
 async function api(url,method='GET',data){return page.evaluate(async({url,method,data})=>{const r=await fetch(url,{method,headers:data?{'Content-Type':'application/json'}:{},body:data?JSON.stringify(data):undefined});return {status:r.status,...await r.json()}},{url,method,data})}
 async function capture(name){assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:path.join(out,name+'.png'),fullPage:true})}
 await page.addInitScript(()=>{window.initialSnapshots=[];document.addEventListener('DOMContentLoaded',()=>{const snapshot=phase=>{const main=document.querySelector('main');window.initialSnapshots.push({phase,title:main?.querySelector('h1')?.textContent,ready:Boolean(document.querySelector('#self-profile-form,#people-editor,#events-editor')),loading:[...document.querySelectorAll('main .page-loading,main .loading,main [aria-busy=true]')].some(n=>n.getClientRects().length),presentation:document.querySelector('#self-description')?.value,text:main?.innerText})};setTimeout(()=>snapshot('DOMContentLoaded-settled'),0);requestAnimationFrame(()=>snapshot('first-frame'))})});
 async function noFlash(){assert.equal(await page.getByText('Connectez-vous pour ouvrir votre espace.',{exact:true}).count(),0);assert.equal(await page.getByRole('heading',{name:'Articles aimés',exact:true}).count(),0)}
 let checks=0;const firstFrames=[];
 try{
  await page.goto('https://loading.test/mon-portrait/');await page.getByRole('button',{name:'Créer mon compte',exact:true}).waitFor();
  assert.equal((await api('/api/auth/register','POST',{email:'loading@example.test',password:'Temporary test password 123',role:'member'})).status,200);
  let profile=await api('/api/self-profile');const legacy={first_name:'Camille',occupation:'Activité conservée',self_perception:'Ancien texte',recurring_patterns:'Ancien schéma',emotions:'Calme',current_context:'Contexte historique',what_brings_me:'Comprendre',what_helps_me:'Marcher',life_rhythm:'Régulier'};
  profile=await api('/api/self-profile','PUT',{...legacy,self_description:'Présentation enregistrée',expected_revision:profile.revision});assert.equal(profile.status,200);
  await api('/api/people','POST',{creation_id:'ready-person',display_name:'Camille proche',role:'friend',relationship_text:'Une relation déjà enregistrée.'});
  await api('/api/life-events','POST',{creation_id:'ready-event',title:'Un moment déjà enregistré',kind:'positive',description:'Un souvenir personnel.',date_precision:'year',date_start:'2020',people_ids:[]});
  for(const width of [320,390,768,1440]){
   await page.setViewportSize({width,height:900});
   for(const [route,title,ready,content]of [['mon-portrait','Mon portrait','#self-profile-form','Présentation enregistrée'],['mes-proches','Mes proches','#people-editor','Camille proche'],['ma-ligne-de-vie','Ma ligne de vie','#events-editor','Un moment déjà enregistré']]){
    requests.length=0;const response=await page.goto('https://loading.test/'+route+'/',{waitUntil:'domcontentloaded'});
    assert.match(response.headers()['cache-control'],/private/);assert.match(response.headers()['cache-control'],/no-store/);
    await page.waitForFunction(()=>initialSnapshots.length===2);const snapshots=await page.evaluate(()=>initialSnapshots);firstFrames.push({route,width,snapshots});
    assert.equal(snapshots.length,2);for(const frame of snapshots){assert.equal(frame.title,title);assert.equal(frame.ready,true,route+' '+frame.phase);assert.equal(frame.loading,false);assert.ok(route==='mon-portrait'?frame.presentation===content:frame.text.includes(content),route+' real content '+frame.phase)}
    assert.deepEqual(requests.filter(r=>r.path.startsWith('/api/')),[],'bootstrap removes redundant initial API requests');
    assert.equal(await page.locator(ready).count(),1);await noFlash();await capture(route+'-'+width+'-ready');checks++;
   }
  }
  await page.setViewportSize({width:1440,height:900});await page.goto('https://loading.test/mes-proches/');await page.locator('#people-editor').waitFor();
  const oldView=await page.locator('main').innerText(),held=hold('/api/self-profile');await page.locator('#site-navigation a[href="/mon-portrait/"]').click();await held.arrived;
  assert.equal(new URL(page.url()).pathname,'/mes-proches/');assert.equal(await page.locator('main').innerText(),oldView);await noFlash();await capture('navigation-pending-keeps-old-page');
  held.release();await page.locator('#self-profile-form').waitFor();assert.equal(new URL(page.url()).pathname,'/mon-portrait/');assert.equal(await page.locator('#self-description').inputValue(),'Présentation enregistrée');await noFlash();checks++;
  await page.locator('#site-navigation a[href="/mes-proches/"]').click();await page.locator('#people-editor').waitFor();
  const older=hold('/api/self-profile'),latest=hold('/api/life-events');await page.locator('#site-navigation a[href="/mon-portrait/"]').click();await older.arrived;
  await page.locator('#site-navigation a[href="/ma-ligne-de-vie/"]').click();await latest.arrived;assert.equal(new URL(page.url()).pathname,'/mes-proches/');latest.release();await page.locator('#events-editor').waitFor();
  const oldResponse=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/self-profile');older.release();await oldResponse;await page.evaluate(()=>new Promise(requestAnimationFrame));assert.equal(await page.locator('h1').innerText(),'Ma ligne de vie');assert.equal(new URL(page.url()).pathname,'/ma-ligne-de-vie/');assert.equal(await page.locator('#self-profile-form').count(),0);checks++;
  await page.locator('#site-navigation a[href="/mon-portrait/"]').click();await page.locator('#self-profile-form').waitFor();
  assert.equal(await page.locator('#view textarea').count(),1);assert.deepEqual(await page.locator('#view input,#view select,#view textarea').evaluateAll(nodes=>nodes.map(n=>n.name)),['first_name','last_name','birth_date','gender','gender_description','occupation','self_description']);assert.equal(await page.locator('#view h3,#view h4,#view #mes-articles').count(),0);
  const input=page.locator('#self-description');assert.equal(await input.inputValue(),'Présentation enregistrée');await input.fill('Ma présentation\nAvec mes mots.');
  await page.getByRole('button',{name:'Enregistrer mes informations',exact:true}).click();await page.getByText('Informations enregistrées.',{exact:true}).waitFor();
  profile=await api('/api/self-profile');for(const [key,value]of Object.entries(legacy))assert.equal(profile.profile[key],value);assert.equal(profile.profile.self_description,'Ma présentation\nAvec mes mots.');checks++;
  await input.fill('Brouillon local');profile=await api('/api/self-profile','PUT',{self_description:'Changement autre onglet',occupation:'Activité modifiée ailleurs',expected_revision:profile.revision});
  await page.getByRole('button',{name:'Enregistrer mes informations',exact:true}).click();await page.getByRole('button',{name:'Comparer la version enregistrée',exact:true}).click();
  assert.equal(await input.inputValue(),'Brouillon local');await page.locator('.self-profile-saved').filter({hasText:'Ma présentation : Changement autre onglet'}).waitFor();assert.ok((await page.locator('.self-profile-saved').allInnerTexts()).includes('Ma présentation : Changement autre onglet'));
  await page.getByRole('button',{name:'Conserver mon brouillon pour un nouvel enregistrement',exact:true}).click();await page.getByRole('button',{name:'Enregistrer mes informations',exact:true}).click();await page.getByText('Informations enregistrées.',{exact:true}).waitFor();profile=await api('/api/self-profile');assert.equal(profile.profile.occupation,'Activité modifiée ailleurs');assert.equal(profile.profile.self_description,'Brouillon local');checks++;
  const save=hold('/api/self-profile','PUT');await input.fill('Premier texte');await page.getByRole('button',{name:'Enregistrer mes informations',exact:true}).click();await save.arrived;await input.fill('Premier texte et la suite');save.release();await page.getByText('Enregistré. Les modifications saisies pendant l’enregistrement restent à sauvegarder.',{exact:true}).waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');
  await page.locator('#site-navigation a[href="/mes-proches/"]').click();await page.locator('#people-editor').waitFor();await page.goBack();await page.locator('#self-profile-form').waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');await page.getByRole('button',{name:'Enregistrer mes informations',exact:true}).click();await page.getByText('Informations enregistrées.',{exact:true}).waitFor();await page.reload();await page.locator('#self-profile-form').waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');checks++;
  const identity={first_name:'Alex',last_name:'Martin',birth_date:'1990-05-02',gender:'self_described',gender_description:'Mes mots personnels',occupation:'Nouvelle activité'};
  for(const [name,value]of Object.entries(identity)){const control=page.locator('#self-profile-form [name="'+name+'"]');if(name==='gender')await control.selectOption(value);else await control.fill(value)}
  const identityRequest=page.waitForRequest(r=>new URL(r.url()).pathname==='/api/self-profile'&&r.method()==='PUT');await page.getByRole('button',{name:'Enregistrer mes informations',exact:true}).click();
  const identityPayload=(await identityRequest).postDataJSON();assert.deepEqual(Object.keys(identityPayload).sort(),[...Object.keys(identity),'expected_revision'].sort());
  await page.getByText('Informations enregistrées.',{exact:true}).waitFor();profile=await api('/api/self-profile');for(const [name,value]of Object.entries(identity))assert.equal(profile.profile[name],value);
  for(const key of ['self_perception','recurring_patterns','emotions','current_context','what_brings_me','what_helps_me','life_rhythm'])assert.equal(profile.profile[key],legacy[key]);assert.equal(profile.profile.self_description,'Premier texte et la suite');checks++;
  await page.reload();await page.locator('#self-profile-form').waitFor();for(const [name,value]of Object.entries(identity))assert.equal(await page.locator('#self-profile-form [name="'+name+'"]').inputValue(),value);
  await page.locator('#site-navigation a[href="/mes-proches/"]').click();await page.locator('#people-editor').waitFor();failure=true;await page.locator('#site-navigation a[href="/mon-portrait/"]').click();await page.getByRole('button',{name:'Recharger mes informations',exact:true}).click();await page.locator('#self-profile-form').waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');checks++;
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'first-frames.json'),JSON.stringify(firstFrames,null,2));console.log(checks+' ready-page/navigation/persistence scenarios passed; no JavaScript errors or overflow.');
 }finally{for(const gate of gates.values())gate.release.resolve();await browser.close();await Promise.allSettled(tasks);db.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
