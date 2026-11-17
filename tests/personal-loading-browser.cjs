// Real Worker + in-memory D1: progressive loading, navigation races and presentation persistence.
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
 const tasks=[],ctx={waitUntil(p){tasks.push(p)}},errors=[],gates=new Map();let failure=false;
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined}),context=await browser.newContext({viewport:{width:1440,height:900}}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 function hold(url,method='GET'){const gate={arrived:deferred(),release:deferred()},key=method+' '+url;gates.set(key,gate);return {arrived:gate.arrived.promise,release(){gates.delete(key);gate.release.resolve()}}}
 await page.route('https://loading.test/**',async route=>{
  const r=route.request(),url=new URL(r.url());
  if(!url.pathname.startsWith('/api/')&&r.resourceType()!=='document')return route.fulfill({status:204,body:''});
  const gate=gates.get(r.method()+' '+url.pathname);if(gate){gate.arrived.resolve();await gate.release.promise}
  if(failure&&url.pathname==='/api/self-profile'&&r.method()==='GET'){failure=false;return route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Indisponibilité de test'})})}
  const result=await worker.fetch(new Request(r.url(),{method:r.method(),headers:r.headers(),body:r.postData()||undefined}),env,ctx);await route.fulfill({status:result.status,headers:Object.fromEntries(result.headers),body:await result.text()});
 });
 async function api(url,method='GET',data){return page.evaluate(async({url,method,data})=>{const r=await fetch(url,{method,headers:data?{'Content-Type':'application/json'}:{},body:data?JSON.stringify(data):undefined});return {status:r.status,...await r.json()}},{url,method,data})}
 async function capture(name){assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:path.join(out,name+'.png'),fullPage:true})}
 async function noFlash(){assert.equal(await page.getByText('Connectez-vous pour ouvrir votre espace.',{exact:true}).count(),0);assert.equal(await page.getByRole('heading',{name:'Articles aimés',exact:true}).count(),0)}
 let checks=0;
 try{
  await page.goto('https://loading.test/mon-portrait/');await page.getByRole('button',{name:'Créer mon compte',exact:true}).waitFor();
  assert.equal((await api('/api/auth/register','POST',{email:'loading@example.test',password:'Temporary test password 123',role:'member'})).status,200);
  let profile=await api('/api/self-profile');const legacy={first_name:'Camille',occupation:'Activité conservée',self_perception:'Ancien texte',recurring_patterns:'Ancien schéma',emotions:'Calme',current_context:'Contexte historique',what_brings_me:'Comprendre',what_helps_me:'Marcher',life_rhythm:'Régulier'};
  profile=await api('/api/self-profile','PUT',{...legacy,self_description:'Présentation enregistrée',expected_revision:profile.revision});assert.equal(profile.status,200);
  for(const width of [320,390,768,1440]){
   await page.setViewportSize({width,height:900});
   for(const [route,title,business,ready]of [['mon-portrait','Mon portrait','/api/self-profile','#self-profile-form'],['mes-proches','Mes proches','/api/people','#people-editor'],['ma-ligne-de-vie','Ma ligne de vie','/api/life-events','#events-editor']]){
    const session=hold('/api/me'),data=hold(business);await page.goto('https://loading.test/'+route+'/',{waitUntil:'domcontentloaded'});await session.arrived;
    assert.equal(await page.locator('h1').innerText(),title);await noFlash();assert.equal(await page.locator('main [aria-busy=true]').count(),1);await capture(route+'-'+width+'-session');
    session.release();await data.arrived;assert.equal(await page.locator('h1').innerText(),title);await noFlash();assert.equal(await page.locator(ready).count(),0);await capture(route+'-'+width+'-data');
    data.release();await page.locator(ready).waitFor();await noFlash();await capture(route+'-'+width+'-ready');checks++;
   }
  }
  await page.setViewportSize({width:1440,height:900});
  const session=hold('/api/me');await page.goto('https://loading.test/mes-proches/',{waitUntil:'domcontentloaded'});await session.arrived;
  await page.locator('#site-navigation a[href="/mon-portrait/"]').click();await noFlash();assert.equal(await page.locator('h1').innerText(),'Mon portrait');
  await page.locator('#site-navigation a[href="/ma-ligne-de-vie/"]').click();await noFlash();session.release();await page.locator('#events-editor').waitFor();assert.equal(await page.locator('h1').innerText(),'Ma ligne de vie');checks++;
  const people=hold('/api/people');await page.locator('#site-navigation a[href="/mes-proches/"]').click();await people.arrived;
  await page.locator('#site-navigation a[href="/mon-portrait/"]').click();await page.locator('#self-profile-form').waitFor();people.release();await page.waitForResponse(r=>new URL(r.url()).pathname==='/api/people');
  assert.equal(await page.locator('#people-editor').count(),0);assert.equal(await page.locator('#self-profile-form').count(),1);checks++;
  assert.equal(await page.locator('#view textarea').count(),1);assert.equal(await page.locator('#view input,#view select,#view h3,#view h4,#view #mes-articles').count(),0);
  const input=page.locator('#self-description');assert.equal(await input.inputValue(),'Présentation enregistrée');await input.fill('Ma présentation\nAvec mes mots.');
  await page.getByRole('button',{name:'Enregistrer ma présentation',exact:true}).click();await page.getByText('Présentation enregistrée.',{exact:true}).waitFor();
  profile=await api('/api/self-profile');for(const [key,value]of Object.entries(legacy))assert.equal(profile.profile[key],value);assert.equal(profile.profile.self_description,'Ma présentation\nAvec mes mots.');checks++;
  await input.fill('Brouillon local');profile=await api('/api/self-profile','PUT',{self_description:'Changement autre onglet',occupation:'Activité modifiée ailleurs',expected_revision:profile.revision});
  await page.getByRole('button',{name:'Enregistrer ma présentation',exact:true}).click();await page.getByRole('button',{name:'Comparer la version enregistrée',exact:true}).click();
  assert.equal(await input.inputValue(),'Brouillon local');assert.equal(await page.locator('.self-profile-saved').innerText(),'Changement autre onglet');
  await page.getByRole('button',{name:'Conserver mon brouillon pour un nouvel enregistrement',exact:true}).click();await page.getByRole('button',{name:'Enregistrer ma présentation',exact:true}).click();await page.getByText('Présentation enregistrée.',{exact:true}).waitFor();profile=await api('/api/self-profile');assert.equal(profile.profile.occupation,'Activité modifiée ailleurs');assert.equal(profile.profile.self_description,'Brouillon local');checks++;
  const save=hold('/api/self-profile','PUT');await input.fill('Premier texte');await page.getByRole('button',{name:'Enregistrer ma présentation',exact:true}).click();await save.arrived;await input.fill('Premier texte et la suite');save.release();await page.getByText('Enregistré. Le texte saisi pendant l’enregistrement reste à sauvegarder.',{exact:true}).waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');
  await page.locator('#site-navigation a[href="/mes-proches/"]').click();await page.locator('#people-editor').waitFor();await page.goBack();await page.locator('#self-profile-form').waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');await page.getByRole('button',{name:'Enregistrer ma présentation',exact:true}).click();await page.getByText('Présentation enregistrée.',{exact:true}).waitFor();await page.reload();await page.locator('#self-profile-form').waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');checks++;
  failure=true;await page.reload();await page.getByRole('button',{name:'Recharger ma présentation',exact:true}).click();await page.locator('#self-profile-form').waitFor();assert.equal(await input.inputValue(),'Premier texte et la suite');checks++;
  assert.deepEqual(errors,[]);console.log(checks+' loading/presentation scenarios passed; 36 screenshots; no JavaScript errors or overflow.');
 }finally{for(const gate of gates.values())gate.release.resolve();await browser.close();await Promise.allSettled(tasks);db.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
