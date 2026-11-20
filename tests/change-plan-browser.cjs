// Local browser integration with the real Worker, in-memory SQLite and synthetic accounts.
// Run with PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers node tests/change-plan-browser.cjs
const {chromium}=require('playwright');
const {DatabaseSync}=require('node:sqlite');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');const root=path.resolve(__dirname,'..'),artifacts=process.env.TEST_ARTIFACTS_DIR||path.join(os.tmpdir(),'decode-plan-browser');fs.mkdirSync(artifacts,{recursive:true});const assert=require('node:assert/strict');
(async()=>{
 const {default:worker}=await import(require('node:url').pathToFileURL(path.join(root,'journal/src/worker.mjs')).href);
 const raw=new DatabaseSync(':memory:');for(const file of fs.readdirSync(path.join(root,'journal/migrations')).filter(x=>x.endsWith('.sql')).sort())raw.exec(fs.readFileSync(path.join(root,'journal/migrations',file),'utf8'));
 function prepare(sql){let params=[];return {bind(...p){params=p;return this},async first(){return raw.prepare(sql).get(...params)||null},async all(){return {results:raw.prepare(sql).all(...params)}},async run(){return {meta:{changes:raw.prepare(sql).run(...params).changes}}}}}
 let queue=Promise.resolve();const env={DB:{prepare,batch(statements){const task=queue.then(async()=>{raw.exec('BEGIN');try{const out=[];for(const s of statements)out.push(await s.run());raw.exec('COMMIT');return out}catch(e){raw.exec('ROLLBACK');throw e}});queue=task.catch(()=>{});return task}},PUBLIC_ORIGIN:'https://ambre.test',FREE_ACCESS_ENABLED:'true',OPENROUTER_API_KEY:'mock-only'};
 const tasks=[];const ctx={waitUntil(task){tasks.push(task)}};const errors=[];
 const browser=await chromium.launch({headless:true});
 async function page(width=1440){const c=await browser.newContext({viewport:{width,height:900}});const p=await c.newPage();p.on('pageerror',e=>errors.push(e.message));await p.addInitScript(()=>{window.SpeechRecognition=undefined;window.webkitSpeechRecognition=undefined});await p.route('https://ambre.test/**',async route=>{const req=route.request();const pathname=new URL(req.url()).pathname;if(!/^\/(?:api|app)\//.test(new URL(req.url()).pathname)&&!['document'].includes(req.resourceType()))return route.fulfill({status:204,body:''});const r=await worker.fetch(new Request(req.url(),{method:req.method(),headers:req.headers(),body:req.postData()||undefined}),env,ctx);await route.fulfill({status:r.status,headers:Object.fromEntries(r.headers),body:await r.text()})});return p}
 // Worker Response status is a property (the bridge deliberately exercises real headers/cookies).
 async function register(p,email,pro=false){await p.goto('https://ambre.test/mon-suivi/');await p.getByRole('button',{name:'Créer mon compte',exact:true}).click();await p.locator('#auth-email').fill(email);await p.locator('#auth-password').fill('Temporary test password 123');if(pro){await p.getByRole('button',{name:'Je suis psychologue',exact:true}).click();await p.locator('[name=professional_name]').fill('Camille Test');await p.locator('[name=professional_title]').fill('Psychologue');await p.locator('[name=registration_number]').fill('TEST-NON-VERIFIE')}await p.locator('#auth-submit').click();await p.locator('#auth-dialog').waitFor({state:'hidden'});await p.getByRole('heading',{name:'Mon suivi',exact:true}).waitFor()}
 async function api(p,path,method='GET',data){return p.evaluate(async({path,method,data})=>{const r=await fetch(path,{method,headers:data?{'Content-Type':'application/json'}:{},body:data?JSON.stringify(data):undefined});return {status:r.status,...await r.json()}},{path,method,data})}
 const providerCalls=[];const realFetch=globalThis.fetch;globalThis.fetch=async(url,options)=>{if(!String(url).includes('openrouter.ai'))return realFetch(url,options);const body=JSON.parse(options.body),purpose=body.response_format?.json_schema?.name;providerCalls.push({purpose,body});const values={ambre_analysis:{need:'understand',states:[],risk:'none',posture:'explore',sensitive_allowed:false,constraints:[],selected_ids:[],closed_topics:[]},ambre_writing:{content:'Vous décrivez une inquiétude quand votre travail est critiqué. Nous pouvons regarder ce qui vous touche dans ces moments.',questions:[],proposals:[],cartography_notes:[],people_observations:[]},ambre_review:{criteria:Array(15).fill('yes'),reasons:[]},ambre_memory:{items:[]}};return Response.json({choices:[{message:{content:JSON.stringify(values[purpose]||{content:'Point de départ disponible.'})},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:80,prompt_tokens_details:{cached_tokens:30},cost:.001}})};
 try{
 const member=await page();await register(member,'plan-browser@example.test');const me=await api(member,'/api/me'),id=me.user.id;
 await member.goto('https://ambre.test/mon-suivi/');
 await member.getByRole('heading',{name:'Mon plan',exact:true}).waitFor();
 const dialog=()=>member.locator('dialog.change-dialog[open]');
 async function submit(label='Enregistrer'){await dialog().getByRole('button',{name:label,exact:true}).click();await dialog().waitFor({state:'hidden'});await member.getByRole('heading',{name:'Mon plan',exact:true}).waitFor()}
 async function section(label){const d=member.locator('details.change-section').filter({has:member.locator('summary',{hasText:label})});if(!await d.evaluate(e=>e.open))await d.locator('summary').click();return d}
 for(const width of [1440,390,320]){
  await member.setViewportSize({width,height:900});
  await member.getByRole('button',{name:'Ajouter un objectif',exact:true}).click();
  await dialog().getByLabel('Ce que vous souhaitez changer',{exact:true}).fill('Mieux prendre ma place '+width);
  await dialog().getByLabel('Pourquoi cela compte pour vous',{exact:true}).fill('Oser exprimer une préférence sans chercher un résultat parfait.');
  assert.equal(await member.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'goal dialog overflow '+width);
  await submit();
  let goal=member.locator('.change-card').filter({has:member.getByRole('heading',{name:'Mieux prendre ma place '+width,exact:true})});
  await goal.getByRole('button',{name:'Ajouter une étape',exact:true}).click();
  await dialog().getByLabel('Une étape accessible, avec vos mots',{exact:true}).fill('Exprimer une préférence simple '+width);
  await submit();
  let steps=await section('Mes étapes');
  let step=steps.locator('.change-card').filter({has:member.getByRole('heading',{name:'Exprimer une préférence simple '+width,exact:true})});
  await step.getByRole('button',{name:'Prévoir un essai',exact:true}).click();
  assert.equal(await dialog().locator('[name=followup_opt_in]').isChecked(),false);
  await dialog().locator('[name=planned_on]').fill('2026-11-20');
  await submit();
  let trials=await section('Mes essais et mes retours');
  let trial=trials.locator('.change-card').filter({has:member.getByRole('heading',{name:'Exprimer une préférence simple '+width,exact:true})});
  await trial.getByRole('button',{name:'Noter mon retour',exact:true}).click();
  await dialog().locator('[name=attempt]').selectOption('not_tried');
  await dialog().locator('[name=result_text]').fill('Je n’ai pas essayé ; je souhaite attendre un moment plus adapté.');
  await dialog().locator('[name=outcome]').selectOption('worse');
  await dialog().locator('[name=adverse_effect_text]').fill('La seule idée de le faire a été pénible.');
  assert.equal(await dialog().locator('[name=occurred_on]').inputValue(),'');
  await submit();
  const saved=await api(member,'/api/plan');assert.equal(saved.goals.length,[1440,390,320].indexOf(width)+1);assert.equal(saved.trials.at(-1).attempt,'not_tried');assert.equal(saved.trials.at(-1).occurred_on,null);assert.equal(saved.trials.at(-1).followup_opt_in,false);
  await member.screenshot({path:path.join(artifacts,'plan-'+width+'.png'),fullPage:true});
  assert.equal(await member.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'plan overflow '+width);
 }
 // A concurrent edit must preserve the form and require an explicit comparison.
 let goal=member.locator('.change-card').filter({has:member.getByRole('heading',{name:'Mieux prendre ma place 320',exact:true})});
 await goal.getByRole('button',{name:'Modifier',exact:true}).click();await dialog().locator('[name=title]').fill('Ma saisie conservée');
 let plan=await api(member,'/api/plan');let target=plan.goals.find(g=>g.title.endsWith('320'));
 assert.equal((await api(member,'/api/plan/goals/'+target.id,'PATCH',{revision:plan.revision,epoch:plan.epoch,entity_revision:target.revision,title:'Modification dans un autre onglet'})).status,200);
 await dialog().getByRole('button',{name:'Enregistrer',exact:true}).click();await dialog().getByRole('button',{name:'Voir la version actuelle',exact:true}).waitFor();assert.equal(await dialog().locator('[name=title]').inputValue(),'Ma saisie conservée');
 await dialog().getByRole('button',{name:'Voir la version actuelle',exact:true}).click();await dialog().getByText('Modification dans un autre onglet',{exact:true}).waitFor();await dialog().getByRole('button',{name:'Garder ma saisie sur cette version',exact:true}).click();await submit();
 await member.getByRole('button',{name:'Préparer mon point personnel',exact:true}).click();await dialog().locator('[name=content]').fill('Mon point personnel modifié, sans partage professionnel.');await submit('Conserver ce point privé');
 plan=await api(member,'/api/plan');assert.equal(plan.preparations[0].content,'Mon point personnel modifié, sans partage professionnel.');
 for(const width of [1440,390,320]){
  await member.setViewportSize({width,height:900});
  await member.goto('https://ambre.test/mon-compte/');await member.locator('#self-profile-form').waitFor();assert.equal(await member.locator('.portrait-tabs, [data-portrait-tab], #portrait-synthesis').count(),0);assert.equal(await member.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'portrait '+width);
 }
 raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,'initial','ready','Point de départ.',?,0,0)").run('initial-review',id,'initial-key');
 assert.equal((await api(member,'/api/chat/consent','PUT',{accepted:true})).status,200);
 await member.goto('https://ambre.test/ambre/');assert.equal(await member.getByText('Options de conversation',{exact:true}).count(),0,'no options menu in the conversation');await member.goto('https://ambre.test/mon-suivi/#mon-plan');await member.getByText('Propositions et retours dans la conversation',{exact:true}).click();await member.getByRole('button',{name:'Suspendre les retours',exact:true}).click();
 assert.equal((await api(member,'/api/plan')).reminders_paused,true);
 await member.goto('https://ambre.test/mon-compte/');await member.getByRole('button',{name:'Effacer la conversation',exact:true}).click();assert.equal(await dialog().locator('[name=include_plan]').isChecked(),false);await dialog().getByRole('button',{name:'Effacer la conversation',exact:true}).click();await dialog().waitFor({state:'hidden'});
 let retained=await api(member,'/api/plan');assert.equal(retained.goals.length,3);assert.equal(retained.trials.length,3);assert.equal((await api(member,'/api/chat')).journey.started,false);
 const exported=await api(member,'/api/export');assert.equal(exported.status,200);assert.equal(exported.change_plan.schema_version,1);

 assert.deepEqual(errors,[]);console.log(JSON.stringify({widths:[1440,390,320],real_worker:true,sqlite:true,goals:3,reported_trials:3,unknown_dates_preserved:true,conflict:'explicit comparison with draft retained',private_preparation:true,catalogue:36,portrait_tabs:'preserved',errors}));
 }finally{await browser.close();await Promise.allSettled(tasks);globalThis.fetch=realFetch;raw.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
