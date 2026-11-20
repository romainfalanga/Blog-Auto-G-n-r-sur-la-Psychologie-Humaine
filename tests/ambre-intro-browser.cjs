// Plan 19/11 lot 3: Ambre's first message, launch, wait, failure and post-review message, in a real
// browser against a simulated /api/chat (no Worker, no provider). 390 touch and 1440.
// Run: node tests/ambre-intro-browser.cjs  (AMBRE_SHOTS=<dir> to choose where screenshots go).
const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=process.env.AMBRE_SHOTS||path.join(root,'work/ambre-intro-browser');fs.mkdirSync(out,{recursive:true});
 const {HTML,APP_ASSETS,INITIAL_PAGE_DATA_MARKER}=await import('../journal/src/ui.mjs');
 const page$=HTML.replace(INITIAL_PAGE_DATA_MARKER,'null'),now=()=>Math.floor(Date.now()/1000);
 const full={version:2,complete:true,identity:{complete:true,missing_fields:[]},people:{complete:true,count:4,required:3},events:{complete:true,count:3,required:3},childhood:{complete:true,answered:60,required:60},symptoms:{complete:true,answered:516,required:516},missing:[]};
 const partial={version:2,complete:false,identity:{complete:false,missing_fields:['gender']},people:{complete:false,count:1,required:3},events:{complete:true,count:3,required:3},childhood:{complete:false,answered:60,required:60,consent:false},symptoms:{complete:false,answered:212,required:516},missing:['identity','people','childhood','symptoms']};
 const quotas={reviews:{remaining:1,limit:1,renews_at:'2026-12-01'}};
 const base=(journey,extra={})=>({consent:false,access:{allowed:false,reason:'journey_not_started'},quota:{used:0,remaining:100,limit:100},messages:[],proposals:[],readiness:journey.readiness,quotas,journey:{sources:{},quotas,can_start:false,started:false,...journey,review:{running_id:null,last_at:null,next_point_at:null,last_failure:null,...journey.review}},...extra});
 const longReview='Bilan complet.\n\n'+'Ce que vous avez décrit montre une manière de vous protéger en vous taisant. '.repeat(30);
 const chatMessage='Voici ce qui ressort de votre bilan :\n• Vous vous taisez souvent pour éviter les conflits avec vos proches.\n• Votre enfance donne des repères sur cette prudence.\n• Hypothèse : une difficulté à nommer ce que vous ressentez, appuyée par vos réponses ; il manque des exemples récents.\n\nQu’aimeriez-vous creuser en premier ? Et qu’en pensez-vous : est-ce que cela vous ressemble ?';
 const scenarios={
  incomplete:()=>base({readiness:partial}),
  complete:()=>base({readiness:full,can_start:true}),
  running:()=>base({readiness:full,started:true,review:{running_id:'r1',stage:'snapshot',blocks_done:5,blocks_total:12,started_at:now()-140,estimate:{p50_seconds:540,p90_seconds:780,basis:'initial'}}},{consent:true,access:{allowed:true,reason:null}}),
  writing:()=>base({readiness:full,started:true,review:{running_id:'r1',stage:'write',blocks_done:12,blocks_total:12,started_at:now()-1000,estimate:{p50_seconds:540,p90_seconds:780,basis:'measured'}}},{consent:true,access:{allowed:true,reason:null}}),
  failed:()=>base({readiness:full,can_start:true,review:{last_failure:{id:'r1',error_code:'provider_timeout',retryable:true,retry_at:null}}},{consent:true,access:{allowed:false,reason:'journey_not_started'}}),
  blocked:()=>base({readiness:full,review:{last_failure:{id:'r1',error_code:'request_limit',retryable:false,retry_at:now()+6*3600}}},{consent:true}),
  ready:()=>base({readiness:full,started:true,review:{last_at:now()-60}},{consent:true,access:{allowed:true,reason:null},messages:[{id:'t1-assistant',role:'assistant',review_id:'r1',review_kind:'initial',content:longReview,chat_message:chatMessage}]}),
  withdrawn:()=>base({readiness:full,started:true,review:{last_at:now()-60}},{consent:false,access:{allowed:false,reason:'consent_required'},messages:[{id:'t1-assistant',role:'assistant',review_id:'r1',review_kind:'initial',content:longReview,chat_message:chatMessage}]}),
  legacy:()=>base({readiness:full,started:true,review:{last_at:now()-60}},{consent:true,access:{allowed:true,reason:null},messages:[{id:'t1-assistant',role:'assistant',review_id:'r1',review_kind:'initial',content:'Votre analyse initiale, rédigée avant le message court.'}]})
 };
 let scenario='incomplete',guest=false;const posts=[];
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||(fs.existsSync('/usr/bin/google-chrome')?'/usr/bin/google-chrome':undefined)});
 const shots=[],errors=[];
 async function context(width,touch){
  const ctx=await browser.newContext({viewport:{width,height:touch?844:900},hasTouch:touch,isMobile:touch,deviceScaleFactor:touch?2:1,locale:'fr-FR'});
  await ctx.route('https://archive.test/**',async route=>{
   const r=route.request(),url=new URL(r.url());
   if(url.pathname===APP_ASSETS.script.path)return route.fulfill({status:200,contentType:'application/javascript',body:APP_ASSETS.script.body});
   if(url.pathname===APP_ASSETS.style.path)return route.fulfill({status:200,contentType:'text/css',body:APP_ASSETS.style.body});
   if(url.pathname==='/api/me')return route.fulfill({json:guest?{user:null}:{user:{id:'u1',email:'test@example.test',role:'member'},configuration:{ai:true}}});
   if(url.pathname==='/api/chat')return route.fulfill({json:scenarios[scenario]()});
   if(url.pathname==='/api/chat/consent'&&r.method()==='PUT'){posts.push(JSON.parse(r.postData()));scenario='ready';return route.fulfill({json:scenarios.ready()})}
   if(url.pathname==='/api/chat/start'&&r.method()==='POST'){posts.push(JSON.parse(r.postData()));scenario='running';return route.fulfill({status:202,json:{...scenarios.running(),review_id:'r1'}})}
   if(r.resourceType()==='document')return route.fulfill({status:200,contentType:'text/html',body:page$});
   return route.fulfill({status:204,body:''});
  });
  const page=await ctx.newPage();page.on('pageerror',e=>errors.push(e.message));return {ctx,page};
 }
 // composer: true when the input bar must be shown, false when only the sentence replaces it.
 async function check(page,name,composer=false){
  const report=await page.evaluate(()=>{
   const small=[...document.querySelectorAll('.conversation button:not([hidden]), .ambre-need, .chat-review-link')].filter(n=>n.offsetParent).map(n=>{const box=n.closest('.check')||n,r=box.getBoundingClientRect();return {t:(n.textContent||n.id||n.className).trim().slice(0,30),h:Math.round(r.height),w:Math.round(r.width)}}).filter(x=>x.h<44||x.w<44);
   const input=document.getElementById('chat-content'),mic=document.querySelector('.chat-microphone'),send=document.querySelector('#chat-form button[type=submit]'),note=document.getElementById('chat-locked-note');
   return {overflow:document.documentElement.scrollWidth>innerWidth,small,focusInput:document.activeElement===input,inputVisible:!!(input&&input.offsetParent),micVisible:!!(mic&&mic.offsetParent),sendVisible:!!(send&&send.offsetParent),note:note&&note.offsetParent?note.innerText:'',checkboxes:document.querySelectorAll('.conversation input[type=checkbox]').length,gate:!!document.getElementById('chat-consent-gate')};
  });
  assert.equal(report.overflow,false,name+': horizontal scroll');assert.deepEqual(report.small,[],name+': targets under 44 px');assert.equal(report.focusInput,false,name+': keyboard opened');assert.equal(report.checkboxes,0,name+': no agreement box');assert.equal(report.gate,false,name+': no agreement gate');
  assert.deepEqual([report.inputVisible,report.micVisible,report.sendVisible],[composer,composer,composer],name+': input bar '+(composer?'shown':'hidden'));
  if(!composer)assert.match(report.note,/Vous pourrez écrire à Ambre après votre premier bilan\.|Connectez-vous pour écrire à Ambre\./,name+': sentence instead of the bar');
 }
 async function shot(page,name){const file=path.join(out,name+'.png');await page.screenshot({path:file,fullPage:false});shots.push(file)}
 try{
  for(const [width,touch] of [[390,true],[1440,false]]){
   const {ctx,page}=await context(width,touch),tag=String(width);
   guest=true;scenario='incomplete';await page.goto('https://archive.test/ambre/');await page.locator('#ambre-intro').waitFor();
   assert.equal(await page.locator('.ambre-need-count').count(),0);assert.equal(await page.locator('[data-auth=login]').innerText(),'Se connecter');assert.equal(await page.locator('[data-auth=register]').innerText(),'Créer un compte');await check(page,'guest '+tag);await shot(page,'guest-'+tag);
   guest=false;
   await page.goto('https://archive.test/ambre/');await page.locator('#ambre-intro').waitFor();
   assert.match(await page.locator('#ambre-intro').innerText(),/3\/4/);assert.match(await page.locator('#ambre-intro').innerText(),/Enregistrez la page/);assert.equal(await page.locator('#ambre-start').isDisabled(),true);
   assert.equal(await page.locator('#ambre-intro-title').innerText(),'Bonjour, je suis Ambre.');assert.match(await page.locator('.ambre-intro-lead').innerText(),/assistante psychologue IA/);
   assert.equal(await page.locator('#chat-locked-note').innerText(),'Vous pourrez écrire à Ambre après votre premier bilan.');await check(page,'incomplete '+tag);await shot(page,'incomplete-'+tag);
   if(touch){await page.locator('#ambre-intro').evaluate(n=>n.scrollIntoView({block:'end'}));await shot(page,'incomplete-'+tag+'-bottom')}
   scenario='complete';await page.reload();await page.locator('#ambre-start').waitFor();assert.equal(await page.locator('#ambre-start').isDisabled(),false,'the launch button alone is enough');assert.equal(await page.locator('#ambre-start-reason').isVisible(),false);
   await page.locator('#ambre-start').scrollIntoViewIfNeeded();await check(page,'complete '+tag);await shot(page,'complete-'+tag);
   await page.locator('#ambre-start').click();await page.locator('.ambre-progress').waitFor();assert.equal(posts.at(-1).accepted,true);
   assert.equal(await page.evaluate(()=>document.activeElement&&document.activeElement.id),'ambre-progress-title');
   const bar=page.locator('[role=progressbar]');assert.match(await bar.getAttribute('aria-valuetext'),/Lecture de vos informations : 5 sur 12/);
   assert.match(await page.locator('.ambre-progress').innerText(),/entre 9 et 13 minutes/);assert.match(await page.locator('.ambre-progress').innerText(),/Vous pouvez fermer cette page/);
   assert.equal(await page.locator('[aria-live=polite], [role=status]').evaluateAll(n=>n.filter(x=>x.closest('.ambre-progress')).length),0,'no second live region in the wait');
   await page.locator('.ambre-progress').scrollIntoViewIfNeeded();await check(page,'running '+tag);await shot(page,'running-'+tag);
   scenario='writing';await page.reload();await page.locator('.ambre-progress').waitFor();assert.match(await page.locator('.ambre-progress').innerText(),/plus long que prévu/);await page.locator('.ambre-progress').scrollIntoViewIfNeeded();await shot(page,'late-'+tag);
   scenario='failed';await page.reload();await page.locator('#ambre-retry').waitFor();assert.match(await page.locator('.ambre-failure').innerText(),/Rien n’a été perdu/);assert.match(await page.locator('.ambre-failure').innerText(),/8 bilans par 24 heures/);await page.locator('.ambre-failure').scrollIntoViewIfNeeded();await check(page,'failed '+tag);await shot(page,'failed-'+tag);
   scenario='blocked';await page.reload();await page.locator('.ambre-failure').waitFor();assert.equal(await page.locator('#ambre-retry').count(),0);await page.locator('.ambre-failure').scrollIntoViewIfNeeded();await shot(page,'failed-final-'+tag);
   scenario='ready';await page.reload();await page.locator('.chat-review-links').waitFor();assert.deepEqual(await page.locator('.chat-review-links a').evaluateAll(a=>a.map(x=>[x.textContent,x.getAttribute('href')])),[['Lire le bilan complet','/mes-bilans/'],['Voir Mes diagnostics','/mes-diagnostics/']]);
   assert.equal(await page.locator('#chat-content').getAttribute('aria-disabled'),null);assert.equal(await page.locator('#chat-content').evaluate(n=>n.readOnly),false);await check(page,'ready '+tag,true);await shot(page,'ready-'+tag);
   scenario='withdrawn';await page.reload();await page.locator('#ambre-start').waitFor();assert.equal(await page.locator('#ambre-start').innerText(),'Reprendre la conversation');assert.equal(await page.locator('#ambre-start').isDisabled(),false);await check(page,'withdrawn '+tag,true);await page.locator('#ambre-start').scrollIntoViewIfNeeded();await shot(page,'withdrawn-'+tag);
   await page.locator('#ambre-start').click();await page.locator('#ambre-start').waitFor({state:'detached'});await page.locator('.chat-review-links').waitFor();assert.deepEqual(posts.at(-1),{accepted:true});assert.equal(await page.locator('#chat-content').isEditable(),true);
   scenario='legacy';await page.reload();await page.getByText('Retrouver ce bilan dans Mes bilans').waitFor();await shot(page,'legacy-'+tag);
   await ctx.close();
  }
  assert.deepEqual(errors,[]);console.log('Ambre intro bench passed:',shots.length,'captures in',out);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
