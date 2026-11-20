// Migration 0034 « Historique »: dates and history of people, moments and notes, with the real Worker
// and in-memory SQLite. 390×844 touch and 1440×900; screenshots in TEST_ARTIFACTS_DIR.
// Run with PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/personal-history-browser.cjs
const {chromium}=require('playwright');
const {DatabaseSync}=require('node:sqlite');
const fs=require('node:fs'),path=require('node:path');const root=path.resolve(__dirname,'..'),artifacts=process.env.TEST_ARTIFACTS_DIR||path.join(root,'work','history-browser');fs.mkdirSync(artifacts,{recursive:true});
function database(until){
 const raw=new DatabaseSync(':memory:');raw.exec('PRAGMA foreign_keys=ON');for(const file of fs.readdirSync(path.join(root,'journal/migrations')).filter(x=>x.endsWith('.sql')&&(!until||x<until)).sort())raw.exec(fs.readFileSync(path.join(root,'journal/migrations',file),'utf8'));
 function prepare(sql){let params=[];return {bind(...p){params=p;return this},async first(){return raw.prepare(sql).get(...params)||null},async all(){return {results:raw.prepare(sql).all(...params)}},async run(){return {meta:{changes:raw.prepare(sql).run(...params).changes}}}}}
 let queue=Promise.resolve();
 return {raw,env:{DB:{prepare,batch(statements){const task=queue.then(async()=>{raw.exec('BEGIN');try{const out=[];for(const s of statements)out.push(await s.run());raw.exec('COMMIT');return out}catch(e){raw.exec('ROLLBACK');throw e}});queue=task.catch(()=>{});return task}},PUBLIC_ORIGIN:'https://ambre.test',FREE_ACCESS_ENABLED:'true'}};
}
const DAY=/\b\d{1,2} [a-zéû]+\.? \d{4}\b/;
(async()=>{
 const {default:worker}=await import(require('node:url').pathToFileURL(path.join(root,'journal/src/worker.mjs')).href);
 const tasks=[],ctx={waitUntil(task){tasks.push(task)}},errors=[],problems=[],checks=[];
 const check=(name,ok,detail)=>{checks.push(name);if(!ok)problems.push(name+' '+JSON.stringify(detail??''))};
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});
 async function page(db,kind){
  const opts=kind==='m'?{viewport:{width:390,height:844},hasTouch:true,isMobile:true,deviceScaleFactor:2}:{viewport:{width:1440,height:900}};
  const c=await browser.newContext({...opts,locale:'fr-FR',timezoneId:'Europe/Paris'});const p=await c.newPage();p.on('pageerror',e=>errors.push(kind+': '+e.message));
  await p.addInitScript(()=>{try{Object.defineProperty(window,'SpeechRecognition',{value:undefined,configurable:true});Object.defineProperty(window,'webkitSpeechRecognition',{value:undefined,configurable:true})}catch(e){}});
  await p.route('https://ambre.test/**',async route=>{const req=route.request();if(!/^\/(?:api|app)\//.test(new URL(req.url()).pathname)&&req.resourceType()!=='document')return route.fulfill({status:204,body:''});const r=await worker.fetch(new Request(req.url(),{method:req.method(),headers:req.headers(),body:req.postData()||undefined}),db.env,ctx);await route.fulfill({status:r.status,headers:Object.fromEntries(r.headers),body:await r.text()})});
  return p;
 }
 async function register(p,email){await p.goto('https://ambre.test/mon-suivi/');await p.getByRole('button',{name:'Créer mon compte',exact:true}).click();await p.locator('#auth-email').fill(email);await p.locator('#auth-password').fill('Temporary test password 123');await p.locator('#auth-submit').click();await p.locator('#auth-dialog').waitFor({state:'hidden'});await p.getByRole('heading',{name:'Mon suivi',exact:true}).waitFor()}
 async function api(p,url,method='GET',data){return p.evaluate(async({url,method,data})=>{const r=await fetch(url,{method,headers:data?{'Content-Type':'application/json'}:{},body:data?JSON.stringify(data):undefined});return {status:r.status,...await r.json()}},{url,method,data})}
 const overflow=p=>p.evaluate(()=>document.documentElement.scrollWidth-innerWidth);
 const smallTargets=(p,selector)=>p.evaluate(sel=>[...document.querySelectorAll(sel)].filter(n=>n.offsetParent).map(n=>{const r=n.getBoundingClientRect();return {text:n.textContent.trim(),w:Math.round(r.width),h:Math.round(r.height)}}).filter(r=>r.h<44||r.w<44),selector);
 const typing=p=>p.evaluate(()=>{const a=document.activeElement;return Boolean(a&&(a.matches('input,textarea,select')||a.isContentEditable))});
 try{
  for(const kind of ['m','d1440']){
   const db=database(),member=await page(db,kind),press=loc=>kind==='m'?loc.tap():loc.click();
   const shot=async(name,dialog)=>{if(dialog)await member.waitForTimeout(300);const over=await overflow(member);check(kind+' '+name+': no horizontal overflow',over<=0,over);await member.screenshot({path:path.join(artifacts,kind+'-'+name+'.png'),fullPage:!dialog})};
   await register(member,'historique-'+kind+'@example.test');
   let lea=(await api(member,'/api/people','POST',{creation_id:'person-'+kind+'-1',display_name:'Léa',role:'sibling',relationship_text:'Ma sœur, très présente.'})).person;
   lea=(await api(member,'/api/people/'+lea.id,'PATCH',{revision:lea.revision,relationship_text:'Ma sœur, nous nous voyons moins depuis son départ.',bond_quality:'close'})).person;
   let note=(await api(member,'/api/people/'+lea.id+'/notes','POST',{creation_id:'note-'+kind+'-1',text:'Elle m’a appelé pour mon anniversaire.'})).note;
   note=(await api(member,'/api/people/'+lea.id+'/notes/'+note.id,'PATCH',{revision:note.revision,text:'Elle m’a appelé pour mon anniversaire, longtemps.'})).note;
   let moment=(await api(member,'/api/life-events','POST',{creation_id:'event-'+kind+'-1',title:'Déménagement à Lyon',kind:'turning_point',description:'Nous avons quitté Paris.',date_precision:'year',date_start:'2019',people_ids:[lea.id],significance:'major'})).event;
   moment=(await api(member,'/api/life-events/'+moment.id,'PATCH',{revision:moment.revision,date_precision:'month',date_start:'2019-09',description:'Nous avons quitté Paris pour Lyon, à la rentrée.'})).event;
   const legacy=(await api(member,'/api/life-events','POST',{creation_id:'event-'+kind+'-2',title:'Premier emploi',kind:'positive',description:'Un poste à mi-temps.',date_precision:'year',date_start:'2016',significance:'anecdotal'})).event;
   // An element saved before the migration has no creation row.
   db.raw.prepare("DELETE FROM personal_history WHERE entity_type='event' AND entity_id=?").run(legacy.id);
   // Mes proches: the date the person was added on the card.
   await member.goto('https://ambre.test/mes-proches/');await member.locator('#person-'+lea.id).waitFor();
   const card=await member.locator('#person-'+lea.id+' .added-on').textContent();check(kind+' directory card: « Ajouté le … »',/^Ajouté le /.test(card)&&DAY.test(card),card);
   await shot('proches');
   // Person page: date, discreet link, dialog with the previous state.
   await member.goto('https://ambre.test/mes-proches/'+lea.id+'/');await member.locator('.person-info .history-link').waitFor();
   check(kind+' person page: « Ajouté le … »',/^Ajouté le /.test(await member.locator('.person-info .added-on').textContent()));
   check(kind+' history links ≥ 44 px',(await smallTargets(member,'.history-link')).length===0,await smallTargets(member,'.history-link'));
   const noteMeta=await member.locator('[data-note-id="'+note.id+'"] .note-dates').textContent();check(kind+' note: creation date and « modifiée le … »',DAY.test(noteMeta)&&/modifiée le /.test(noteMeta),noteMeta);
   await shot('proche');
   await press(member.locator('.person-info .history-link'));const dialog=member.locator('dialog.history-dialog');await dialog.locator('.history-entry').first().waitFor();
   const text=await dialog.textContent();
   check(kind+' person history: newest change first, before and after',(await dialog.locator('.history-entry').count())===2&&/Modifié le/.test(await dialog.locator('.history-entry').first().textContent())&&text.includes('Ma sœur, très présente.')&&text.includes('Ma sœur, nous nous voyons moins depuis son départ.')&&text.includes('Qualité du lien'),text);
   check(kind+' person history: the creation closes the list',/Ajouté le/.test(await dialog.locator('.history-entry').last().textContent()));
   check(kind+' dialog: focus on the close button, no keyboard',!(await typing(member))&&await member.evaluate(()=>document.activeElement.classList.contains('close')));
   check(kind+' dialog close ≥ 44 px',(await smallTargets(member,'dialog.history-dialog .close')).length===0);
   await shot('historique-proche',true);
   await press(dialog.locator('.close'));await dialog.waitFor({state:'detached'});
   check(kind+' focus returns to the link',await member.evaluate(()=>document.activeElement.classList.contains('history-link')));
   await press(member.locator('[data-note-id="'+note.id+'"] .history-link'));await dialog.locator('.history-entry').first().waitFor();
   const noteText=await dialog.textContent();check(kind+' note history: previous and new text',noteText.includes('Texte précédent')&&noteText.includes('Elle m’a appelé pour mon anniversaire.')&&noteText.includes('Nouveau texte')&&/Ajoutée le/.test(noteText),noteText);
   await shot('historique-note',true);
   await member.keyboard.press('Escape');await dialog.waitFor({state:'detached'});
   // Ma ligne de vie: the date of the moment and the date it was added, then its history.
   await member.goto('https://ambre.test/ma-ligne-de-vie/');await member.locator('#event-'+moment.id).waitFor();
   const meta=await member.locator('#event-'+moment.id).textContent();check(kind+' moment: its own date and « Ajouté le … »',/septembre/.test(meta)&&/Ajouté le /.test(meta),meta);
   check(kind+' moment actions ≥ 44 px',(await smallTargets(member,'#event-'+moment.id+' .source-actions button')).length===0,await smallTargets(member,'#event-'+moment.id+' .source-actions button'));
   await shot('ligne-de-vie');
   await press(member.locator('#event-'+moment.id+' .history-link'));await dialog.locator('.history-entry').first().waitFor();
   const eventText=await dialog.textContent();check(kind+' moment history: date read as a whole, story before and after',eventText.includes('2019')&&eventText.includes('septembre 2019')&&eventText.includes('Nous avons quitté Paris.')&&eventText.includes('Récit'),eventText);
   check(kind+' moment history: people in the created state',(await dialog.locator('.history-entry').last().textContent()).includes('Léa'));
   await shot('historique-moment',true);
   await press(dialog.locator('.close'));await dialog.waitFor({state:'detached'});
   await press(member.locator('#event-'+legacy.id+' .history-link'));await dialog.locator('.history-gap').waitFor();
   check(kind+' element without earlier history: said soberly',/ne sont pas connues/.test(await dialog.textContent())&&/Aucune modification enregistrée/.test(await dialog.textContent()));
   await shot('historique-ancien',true);
   await press(dialog.locator('.close'));
   await member.context().close();db.raw.close();
  }
  // Before migration 0034: the page works, the dialog says the history is not available yet.
  const before=database('0034'),member=await page(before,'m');await register(member,'avant-0034@example.test');
  const p=(await api(member,'/api/people','POST',{creation_id:'person-avant-1',display_name:'Marc',role:'friend'})).person;
  await member.goto('https://ambre.test/mes-proches/'+p.id+'/');await member.locator('.person-info .history-link').tap();
  await member.locator('dialog.history-dialog .helper').waitFor();check('before 0034: « pas encore disponible »',/pas encore disponible/.test(await member.locator('dialog.history-dialog').textContent()));
  await member.screenshot({path:path.join(artifacts,'m-avant-migration.png')});
  await member.context().close();before.raw.close();
  if(errors.length)problems.push('JavaScript: '+errors.join(' | '));
  for(const name of checks)if(!problems.some(p=>p.startsWith(name)))console.log('OK  ',name);
  if(problems.length){console.error(problems.map(p=>'FAIL '+p).join('\n'));process.exitCode=1}else console.log(checks.length+' vérifications « Historique » à 390×844 tactile et 1440×900, sans erreur JavaScript ni débordement.');
 }finally{await browser.close();await Promise.allSettled(tasks)}
})().catch(e=>{console.error(e);process.exitCode=1});
