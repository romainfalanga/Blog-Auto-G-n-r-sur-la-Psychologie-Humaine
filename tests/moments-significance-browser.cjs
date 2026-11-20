// Migration 0027 « Anecdotique / Marquant »: local browser bench with the real Worker and in-memory SQLite.
// 390×844 touch, 1280×700 and 1440×900; screenshots in TEST_ARTIFACTS_DIR.
// Run with PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/moments-significance-browser.cjs
const {chromium}=require('playwright');
const {DatabaseSync}=require('node:sqlite');
const fs=require('node:fs'),path=require('node:path');const root=path.resolve(__dirname,'..'),artifacts=process.env.TEST_ARTIFACTS_DIR||path.join(root,'work','moments-browser');fs.mkdirSync(artifacts,{recursive:true});const assert=require('node:assert/strict');
function database(until){
 const raw=new DatabaseSync(':memory:');for(const file of fs.readdirSync(path.join(root,'journal/migrations')).filter(x=>x.endsWith('.sql')&&(!until||x<until)).sort())raw.exec(fs.readFileSync(path.join(root,'journal/migrations',file),'utf8'));
 function prepare(sql){let params=[];return {bind(...p){params=p;return this},async first(){return raw.prepare(sql).get(...params)||null},async all(){return {results:raw.prepare(sql).all(...params)}},async run(){return {meta:{changes:raw.prepare(sql).run(...params).changes}}}}}
 let queue=Promise.resolve();
 return {raw,env:{DB:{prepare,batch(statements){const task=queue.then(async()=>{raw.exec('BEGIN');try{const out=[];for(const s of statements)out.push(await s.run());raw.exec('COMMIT');return out}catch(e){raw.exec('ROLLBACK');throw e}});queue=task.catch(()=>{});return task}},PUBLIC_ORIGIN:'https://ambre.test',FREE_ACCESS_ENABLED:'true'}};
}
(async()=>{
 const {default:worker}=await import(require('node:url').pathToFileURL(path.join(root,'journal/src/worker.mjs')).href);
 const tasks=[],ctx={waitUntil(task){tasks.push(task)}},errors=[],problems=[],checks=[];
 const check=(name,ok,detail)=>{checks.push(name);if(!ok)problems.push(name+' '+JSON.stringify(detail??''))};
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});
 async function page(db,kind){
  const opts=kind==='m'?{viewport:{width:390,height:844},hasTouch:true,isMobile:true,deviceScaleFactor:2}:kind==='d1280'?{viewport:{width:1280,height:700}}:{viewport:{width:1440,height:900}};
  const c=await browser.newContext({...opts,locale:'fr-FR'});const p=await c.newPage();p.on('pageerror',e=>errors.push(kind+': '+e.message));
  await p.addInitScript(()=>{try{Object.defineProperty(window,'SpeechRecognition',{value:undefined,configurable:true});Object.defineProperty(window,'webkitSpeechRecognition',{value:undefined,configurable:true})}catch(e){}});
  await p.route('https://ambre.test/**',async route=>{const req=route.request();if(!/^\/(?:api|app)\//.test(new URL(req.url()).pathname)&&req.resourceType()!=='document')return route.fulfill({status:204,body:''});const r=await worker.fetch(new Request(req.url(),{method:req.method(),headers:req.headers(),body:req.postData()||undefined}),db.env,ctx);await route.fulfill({status:r.status,headers:Object.fromEntries(r.headers),body:await r.text()})});
  return p;
 }
 async function register(p,email){await p.goto('https://ambre.test/mon-suivi/');await p.getByRole('button',{name:'Créer mon compte',exact:true}).click();await p.locator('#auth-email').fill(email);await p.locator('#auth-password').fill('Temporary test password 123');await p.locator('#auth-submit').click();await p.locator('#auth-dialog').waitFor({state:'hidden'});await p.getByRole('heading',{name:'Mon suivi',exact:true}).waitFor()}
 async function api(p,url,method='GET',data){return p.evaluate(async({url,method,data})=>{const r=await fetch(url,{method,headers:data?{'Content-Type':'application/json'}:{},body:data?JSON.stringify(data):undefined});return {status:r.status,...await r.json()}},{url,method,data})}
 const overflow=p=>p.evaluate(()=>document.documentElement.scrollWidth-innerWidth);
 const smallTargets=(p,selector)=>p.evaluate(sel=>[...document.querySelectorAll(sel)].filter(n=>n.offsetParent).map(n=>{const r=n.getBoundingClientRect();return {text:n.textContent.trim(),w:Math.round(r.width),h:Math.round(r.height)}}).filter(r=>r.h<44||r.w<44),selector);
 try{
  for(const kind of ['m','d1280','d1440']){
   const db=database(),member=await page(db,kind),press=loc=>kind==='m'?loc.tap():loc.click();
   const shot=async name=>{const over=await overflow(member);check(kind+' '+name+': no horizontal overflow',over<=0,over);await member.screenshot({path:path.join(artifacts,kind+'-'+name+'.png'),fullPage:true})};
   await register(member,'moments-'+kind+'@example.test');
   const person=(await api(member,'/api/people','POST',{creation_id:'person-'+kind+'-1',display_name:'Léa',role:'sibling',relationship_text:'Ma sœur.'})).person;
   // Older moments (created before the choice existed): no significance, one with an exposure.
   const old=[];for(const [i,title,date,people] of [[1,'Déménagement',  '2015',[person.id]],[2,'Premier concert','2018',[]],[3,'Accident de voiture','2012',[person.id]]]){const r=await api(member,'/api/life-events','POST',{creation_id:'old-event-'+kind+'-'+i,title,kind:'other',description:'Un souvenir ancien : '+title.toLowerCase()+'.',date_precision:'year',date_start:date,people_ids:people});if(!r.event)throw new Error(JSON.stringify(r));old.push(r.event)}
   check(kind+' an API creation without the form marker stays « à préciser »',old.every(e=>e.significance===null),old.map(e=>e.significance));
   const exposed=await api(member,'/api/life-events/'+old[2].id,'PATCH',{revision:old[2].revision,exposure:'happened_to_me'});check(kind+' exposure saved, significance kept empty',exposed.status===200&&exposed.event.significance===null,exposed);
   await member.goto('https://ambre.test/ma-ligne-de-vie/');await member.locator('#event-'+old[0].id).waitFor();
   check(kind+' old moments show « À préciser » with two choices',await member.locator('.significance-choice').count()===3,await member.locator('.significance-choice').count());
   await shot('frise-a-preciser');
   // The form: a segmented choice without default, required at creation.
   await press(member.locator('#events-editor summary'));await member.locator('[name=title]').fill('Naissance de ma fille');await member.locator('[name=description]').fill('Le jour où tout a changé.');await member.locator('[name=date_precision]').selectOption('year');await member.locator('[name=date_start]').fill('2021');
   check(kind+' no default choice',await member.locator('#events-form [name=significance]:checked').count()===0);
   await member.locator('#events-form').getByRole('button',{name:'Enregistrer',exact:true}).click();await member.waitForTimeout(150);
   const invalid=await member.evaluate(()=>{const i=document.querySelector('#events-form [name=significance]');return i?{valid:i.validity.valid,message:i.validationMessage}:null});
   check(kind+' saving without a choice is blocked with the French message',invalid&&!invalid.valid&&/Anecdotique.*Marquant/.test(invalid.message),invalid);
   check(kind+' nothing was created without the choice',(await api(member,'/api/life-events')).events.length===3);
   check(kind+' segment targets ≥ 44 px',(await smallTargets(member,'#events-form .segment')).length===0,await smallTargets(member,'#events-form .segment'));
   await member.locator('#events-form .significance-field').scrollIntoViewIfNeeded();await shot('formulaire');
   await press(member.locator('#events-form .segment',{hasText:'Marquant'}));
   check(kind+' the chosen segment is filled',await member.evaluate(()=>getComputedStyle(document.querySelector('#events-form .segment:has(input:checked)')).backgroundColor)==='rgb(32, 94, 84)');
   await member.locator('#events-form').getByRole('button',{name:'Enregistrer',exact:true}).click();await member.locator('#events-form').waitFor({state:'detached'});
   let events=(await api(member,'/api/life-events')).events;const born=events.find(e=>e.title==='Naissance de ma fille');
   check(kind+' form creation saved as « major »',born&&born.significance==='major',born);
   check(kind+' current_importance untouched',born&&born.current_importance===null,born&&born.current_importance);
   const card=member.locator('#event-'+born.id);check(kind+' marking card: class, full dot and strong title',await card.evaluate(n=>n.classList.contains('is-major')&&getComputedStyle(n,'::before').backgroundColor==='rgb(32, 94, 84)'&&Number(getComputedStyle(n.querySelector('h4')).fontWeight)>=700));
   // A simple way to choose on an old moment.
   const second=member.locator('#event-'+old[1].id);check(kind+' quick-choice buttons ≥ 44 px',(await smallTargets(member,'#event-'+old[1].id+' .segment-button')).length===0,await smallTargets(member,'#event-'+old[1].id+' .segment-button'));
   await press(second.getByRole('button',{name:'Anecdotique',exact:true}));await member.locator('#event-'+old[1].id+'.is-anecdotal').waitFor();
   events=(await api(member,'/api/life-events')).events;check(kind+' quick choice saved',events.find(e=>e.id===old[1].id).significance==='anecdotal');
   check(kind+' anecdotal card: hollow dot, lighter title',await member.locator('#event-'+old[1].id).evaluate(n=>getComputedStyle(n,'::before').backgroundColor!=='rgb(32, 94, 84)'&&Number(getComputedStyle(n.querySelector('h4')).fontWeight)<600));
   await shot('frise-tous');
   // « Tous / Marquants »: the anecdotal moment is hidden, the moments « à préciser » are listed apart.
   await press(member.locator('.moment-filter').getByRole('button',{name:'Marquants',exact:true}));await member.locator('.moment-filter button[aria-pressed="true"]',{hasText:'Marquants'}).waitFor();
   check(kind+' filter: pressed state',await member.locator('.moment-filter button[aria-pressed="true"]').textContent()==='Marquants');
   check(kind+' filter: marking moment in the line',await member.locator('.event-timeline #event-'+born.id).count()===1);
   check(kind+' filter: anecdotal hidden',await member.locator('#event-'+old[1].id).count()===0);
   check(kind+' filter: « à préciser » listed apart',await member.locator('.pending-moments .event-card').count()===2,await member.locator('.pending-moments .event-card').count());
   check(kind+' filter targets ≥ 44 px',(await smallTargets(member,'.moment-filter button')).length===0,await smallTargets(member,'.moment-filter button'));
   await shot('filtre-marquants');
   await press(member.locator('#event-'+old[2].id).getByRole('button',{name:'Marquant',exact:true}));await member.locator('.event-timeline #event-'+old[2].id).waitFor();
   check(kind+' a moment chosen in the filter joins the line',await member.locator('.pending-moments .event-card').count()===1);
   // Person page: same display, filter, quick choice; then « Ajouter un moment » with this person.
   await member.goto('https://ambre.test/mes-proches/'+person.id+'/');await member.locator('.person-frise').first().waitFor();
   check(kind+' person frise: marking moment strong',await member.locator('[data-event-id="'+old[2].id+'"]').evaluate(n=>n.classList.contains('is-major')&&Number(getComputedStyle(n.querySelector('.frise-title')).fontWeight)>=700));
   check(kind+' person frise: « à préciser » choice',await member.locator('[data-event-id="'+old[0].id+'"] .significance-choice').count()===1);
   await shot('proche-tous');
   await press(member.locator('.person-timeline .moment-filter').getByRole('button',{name:'Marquants',exact:true}));await member.locator('.person-timeline .moment-filter button[aria-pressed="true"]',{hasText:'Marquants'}).waitFor();
   check(kind+' person filter: « à préciser » apart',await member.locator('.person-timeline .pending-moments [data-event-id="'+old[0].id+'"]').count()===1);
   await shot('proche-marquants');
   await press(member.locator('[data-event-id="'+old[0].id+'"]').getByRole('button',{name:'Anecdotique',exact:true}));await member.locator('.moment-filter-hidden').waitFor();
   check(kind+' person quick choice saved',(await api(member,'/api/life-events/'+old[0].id)).event.significance==='anecdotal');
   await press(member.locator('.person-timeline .moment-filter').getByRole('button',{name:'Tous',exact:true}));
   await press(member.locator('.person-moment-add'));await member.locator('#events-form').waitFor();
   check(kind+' « ajouter-avec »: person chosen and no default choice',await member.locator('[name=people_ids][value="'+person.id+'"]').isChecked()&&await member.locator('#events-form [name=significance]:checked').count()===0);
   await member.locator('[name=title]').fill('Fou rire avec Léa');await member.locator('[name=description]').fill('Un dîner léger.');await press(member.locator('#events-form .segment',{hasText:'Anecdotique'}));
   await member.locator('#events-form .significance-field').scrollIntoViewIfNeeded();await shot('ajouter-avec');
   await member.locator('#events-form').getByRole('button',{name:'Enregistrer',exact:true}).click();await member.waitForURL('**/mes-proches/'+person.id+'/');await member.locator('.person-frise').first().waitFor();
   events=(await api(member,'/api/life-events')).events;const laugh=events.find(e=>e.title==='Fou rire avec Léa');check(kind+' « ajouter-avec » saved as anecdotal with the person',laugh&&laugh.significance==='anecdotal'&&laugh.people_ids.includes(person.id),laugh);
   await shot('proche-apres-ajout');
   // A modification without the field keeps the stored value.
   const kept=await api(member,'/api/life-events/'+born.id,'PATCH',{revision:born.revision,description:'Le jour où tout a changé, vraiment.'});check(kind+' PATCH without the field keeps « major »',kept.event.significance==='major',kept.event);
   // Export.
   const exported=await api(member,'/api/export');const rows=JSON.stringify(exported);check(kind+' export carries the field',rows.includes('"significance":"major"')&&rows.includes('"significance":"anecdotal"'));
   if(kind==='d1440'){
    // Modifier can bring a moment back to « À préciser » (explicit null).
    await member.goto('https://ambre.test/ma-ligne-de-vie/');let row=(await api(member,'/api/life-events/'+old[1].id)).event;
    await member.locator('#event-'+old[1].id).getByRole('button',{name:'Modifier',exact:true}).click();
    check(kind+' edit form: three choices, current one checked',await member.locator('#events-form [name=significance]').count()===3&&await member.locator('#events-form [name=significance][value=anecdotal]').isChecked());
    check(kind+' radiogroup labelled by its legend',await member.locator('#events-form [role=radiogroup]').evaluate(n=>document.getElementById(n.getAttribute('aria-labelledby'))?.textContent)==='Place dans votre histoire');
    await member.locator('#events-form .segment',{hasText:'À préciser'}).click();await member.locator('#events-form').getByRole('button',{name:'Enregistrer',exact:true}).click();await member.locator('#events-form').waitFor({state:'detached'});
    row=(await api(member,'/api/life-events/'+old[1].id)).event;check(kind+' back to « à préciser »',row.significance===null,row);
    // Quick choice on the card while its edit form is open: the draft follows, no conflict on the next save.
    await member.locator('#event-'+old[1].id).getByRole('button',{name:'Modifier',exact:true}).click();await member.locator('#events-form [name=description]').fill('Un souvenir ancien, précisé.');
    await member.locator('#event-'+old[1].id).getByRole('button',{name:'Marquant',exact:true}).click();await member.locator('#event-'+old[1].id+'.is-major').waitFor();
    check(kind+' open draft follows the quick choice',await member.locator('#events-form [name=significance][value=major]').isChecked()&&await member.locator('#events-form [name=description]').inputValue()==='Un souvenir ancien, précisé.');
    await member.locator('#events-form').getByRole('button',{name:'Enregistrer',exact:true}).click();await member.locator('#events-form').waitFor({state:'detached'});
    row=(await api(member,'/api/life-events/'+old[1].id)).event;check(kind+' saved after the quick choice without conflict',row.significance==='major'&&row.description==='Un souvenir ancien, précisé.',row);
    // A long, paginated line: « Marquants » is filtered by the API, so moments beyond the first page appear.
    for(let i=0;i<55;i++){const r=await api(member,'/api/life-events','POST',{creation_id:'bulk-moment-'+i,title:'Petit moment '+i,kind:'other',description:'Un détail.',date_precision:'year',date_start:String(2030+i),significance:'anecdotal',people_ids:[]});if(r.status!==201)throw new Error(JSON.stringify(r))}
    await member.goto('https://ambre.test/ma-ligne-de-vie/');await member.locator('.moment-filter').waitFor();
    check(kind+' all moments: the marking one is beyond the first page',await member.locator('#event-'+born.id).count()===0);
    await member.locator('.moment-filter').getByRole('button',{name:'Marquants',exact:true}).click();await member.locator('#event-'+born.id).waitFor();
    check(kind+' « Marquants » shows every marking moment of the line',await member.locator('.event-timeline .event-card.is-major').count()===3,await member.locator('.event-timeline .event-card.is-major').count());
    check(kind+' « Marquants » loads no anecdotal moment',await member.locator('.event-card.is-anecdotal').count()===0);
    await shot('filtre-marquants-pagine');
   }
   await member.context().close();db.raw.close();
  }
  // Before migration 0027: no choice in the form, creation works, nothing about « à préciser ».
  const before=database('0027'),member=await page(before,'d1440');await register(member,'avant-0027@example.test');
  await member.goto('https://ambre.test/ma-ligne-de-vie/');await member.locator('#events-editor summary').click();await member.locator('[name=title]').fill('Avant la migration');await member.locator('[name=description]').fill('Un récit.');
  check('before 0027: no choice in the form',await member.locator('[name=significance]').count()===0);
  await member.locator('#events-form').getByRole('button',{name:'Enregistrer',exact:true}).click();await member.locator('#events-form').waitFor({state:'detached'});
  const listed=await api(member,'/api/life-events');check('before 0027: saved, no flag, no filter',listed.events.length===1&&listed.significance_ready===false&&await member.locator('.moment-filter,.significance-view').count()===0,listed);
  const ignored=await api(member,'/api/life-events','POST',{creation_id:'avant-ignore',title:'Ignoré',description:'Choix ignoré.',significance:'major',entry:'form'});check('before 0027: the field is ignored without error',ignored.status===201&&!('significance' in ignored.event),ignored);
  await member.screenshot({path:path.join(artifacts,'d1440-avant-migration.png'),fullPage:true});
  await member.context().close();before.raw.close();
  if(errors.length)problems.push('JavaScript: '+errors.join(' | '));
  for(const name of checks)if(!problems.some(p=>p.startsWith(name)))console.log('OK  ',name);
  if(problems.length){console.error(problems.map(p=>'FAIL '+p).join('\n'));process.exitCode=1}else console.log(checks.length+' vérifications « Anecdotique / Marquant » à 390×844 tactile, 1280×700 et 1440×900, sans erreur JavaScript ni débordement.');
 }finally{await browser.close();await Promise.allSettled(tasks)}
})().catch(e=>{console.error(e);process.exitCode=1});
