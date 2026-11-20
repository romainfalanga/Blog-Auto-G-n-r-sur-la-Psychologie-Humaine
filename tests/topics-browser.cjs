// « Mes sujets » in a real browser against a simulated /api/topics (no Worker): list, creation,
// positions, reasons, objections, replies, overview, edition, deletion. 390 touch and 1440.
// Run: node tests/topics-browser.cjs  (TOPICS_SHOTS=<dir> to choose where screenshots go).
const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=process.env.TOPICS_SHOTS||path.join(root,'work/topics-browser');fs.mkdirSync(out,{recursive:true});
 const {HTML,APP_ASSETS,INITIAL_PAGE_DATA_MARKER}=await import('../journal/src/ui.mjs');
 const page$=HTML.replace(INITIAL_PAGE_DATA_MARKER,'null');
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||(fs.existsSync('/usr/bin/google-chrome')?'/usr/bin/google-chrome':undefined)});
 const errors=[],shots=[];
 try{
  for(const [width,touch] of [[390,true],[1440,false]]){
   const topics=new Map();let serial=0;const now=()=>Math.floor(Date.now()/1000);
   const ctx=await browser.newContext({viewport:{width,height:touch?844:900},hasTouch:touch,isMobile:touch,deviceScaleFactor:touch?2:1,locale:'fr-FR'});
   await ctx.route('https://archive.test/**',async route=>{
    const r=route.request(),url=new URL(r.url()),m=r.method(),parts=url.pathname.split('/').slice(3),data=r.postData()?JSON.parse(r.postData()):{};
    if(url.pathname===APP_ASSETS.script.path)return route.fulfill({status:200,contentType:'application/javascript',body:APP_ASSETS.script.body});
    if(url.pathname===APP_ASSETS.style.path)return route.fulfill({status:200,contentType:'text/css',body:APP_ASSETS.style.body});
    if(url.pathname==='/api/me')return route.fulfill({json:{user:{id:'u1',email:'test@example.test',role:'member'},configuration:{ai:true}}});
    if(url.pathname==='/api/topics'&&m==='GET')return route.fulfill({json:{topics:[...topics.values()].map(t=>({...t.topic,positions:t.nodes.filter(n=>n.kind==='position').length,arguments:t.nodes.filter(n=>n.kind!=='position').length}))}});
    if(url.pathname==='/api/topics'&&m==='POST'){const id='s'+(++serial),topic={id,title:data.title.trim(),created_at:now(),updated_at:now()};topics.set(id,{topic,nodes:[]});return route.fulfill({status:201,json:{topic:{...topic,positions:0,arguments:0}}})}
    if(url.pathname.startsWith('/api/topics/')){
     const t=topics.get(parts[0]);if(!t)return route.fulfill({status:404,json:{error:'Ce sujet est introuvable.',code:'not_found'}});
     if(parts.length===1&&m==='GET')return route.fulfill({json:t});
     if(parts.length===1&&m==='PUT'){t.topic.title=data.title;return route.fulfill({json:{topic:t.topic}})}
     if(parts.length===1&&m==='DELETE'){topics.delete(parts[0]);return route.fulfill({json:{ok:true}})}
     if(parts.length===2&&m==='POST'){const node={id:'n'+(++serial),parent_id:data.parent_id||null,kind:data.parent_id?data.kind:'position',content:data.content,basis:data.parent_id?data.basis||null:null,confidence:data.parent_id?null:data.confidence||null,created_at:now(),updated_at:now()};t.nodes.push(node);return route.fulfill({status:201,json:{node}})}
     const node=t.nodes.find(n=>n.id===parts[2]);
     if(m==='PUT'){Object.assign(node,{content:data.content},node.kind==='position'?{confidence:data.confidence||null}:{basis:data.basis||null});return route.fulfill({json:{node}})}
     if(m==='DELETE'){const gone=new Set([node.id]);let grew=true;while(grew){grew=false;for(const n of t.nodes)if(n.parent_id&&gone.has(n.parent_id)&&!gone.has(n.id)){gone.add(n.id);grew=true}}t.nodes=t.nodes.filter(n=>!gone.has(n.id));return route.fulfill({json:{ok:true}})}
    }
    if(r.resourceType()==='document')return route.fulfill({status:200,contentType:'text/html',body:page$});
    return route.fulfill({status:204,body:''});
   });
   const page=await ctx.newPage(),tag=String(width);page.on('pageerror',e=>errors.push(e.message));
   const shot=async name=>{const file=path.join(out,name+'-'+tag+'.png');await page.screenshot({path:file,fullPage:true});shots.push(file)};
   const check=async name=>{const report=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth,small:[...document.querySelectorAll('#view button, #view a')].filter(n=>n.offsetParent).map(n=>{const r=n.getBoundingClientRect();return {t:n.textContent.trim().slice(0,30),h:Math.round(r.height)}}).filter(x=>x.h<44),arrows:/[←→‹›]/.test(document.getElementById('view').innerText)}));
    assert.equal(report.overflow,false,name+': horizontal scroll');assert.deepEqual(report.small,[],name+': targets under 44 px');assert.equal(report.arrows,false,name+': no arrow')};
   const fill=async(text,choice)=>{const form=page.locator('.topic-form').last();await form.locator('textarea').fill(text);if(choice)await form.locator('select').selectOption(choice);await form.locator('button[type=submit]').click()};
   await page.goto('https://archive.test/mes-sujets/');await page.locator('.topic-new').waitFor();
   assert.equal(await page.locator('h1').innerText(),'Mes sujets');assert.match(await page.locator('#view').innerText(),/aucun sujet/);await check('empty');await shot('empty');
   await page.locator('.topic-new input').fill('Écologie');await page.locator('.topic-new button').click();
   await page.locator('.topic-head h2').waitFor();assert.equal(await page.locator('.topic-head h2').innerText(),'Écologie');assert.match(page.url(),/\/mes-sujets\/s1\/$/);
   await fill('Il faut réduire la place de la voiture en ville.','4');
   await page.locator('.topic-position').waitFor();assert.match(await page.locator('.topic-position').innerText(),/Bonne certitude/);
   assert.match(await page.locator('.topic-gaps').innerText(),/sans raison écrite/);
   await page.getByRole('button',{name:'Ajouter une raison'}).click();await fill('La pollution de l’air baisse quand le trafic baisse.','hearsay');
   await page.locator('.topic-node').waitFor();assert.match(await page.locator('.topic-gaps').innerText(),/sans objection notée[\s\S]*entendue ou lue/);
   await page.getByRole('button',{name:'Ajouter une objection'}).click();await fill('Certaines personnes n’ont pas d’autre moyen de se déplacer.','experience');
   await page.locator('.topic-node.is-against').waitFor();assert.match(await page.locator('.topic-gaps').innerText(),/objection sans réponse/);await check('gaps');await shot('gaps');
   await page.getByRole('button',{name:'Y répondre'}).click();await fill('C’est pourquoi il faut d’abord développer les transports en commun.','value');
   await page.locator('.topic-children .topic-node').waitFor();assert.match(await page.locator('.topic-children').innerText(),/Ma réponse/);
   await page.getByRole('button',{name:'Ce qui l’appuie'}).first().click();await fill('Des mesures de qualité de l’air publiées par ma ville.','fact');
   await page.waitForFunction(()=>!document.querySelector('.topic-gaps'));assert.match(await page.locator('.topic-overview').innerText(),/1 position · 4 arguments[\s\S]*Chaque position a ses raisons/);
   await page.getByRole('button',{name:'Ajouter une position'}).click();await fill('Le nucléaire me met mal à l’aise, sans que je sache bien pourquoi.','2');
   await page.waitForFunction(()=>document.querySelectorAll('.topic-position').length===2);await check('full');await shot('full');
   // Edit, rename, delete an element, then come back to the list.
   await page.locator('.topic-node').first().getByRole('button',{name:'Modifier'}).first().click();await fill('La pollution de l’air baisse nettement quand le trafic baisse.','fact');
   await page.waitForFunction(()=>/nettement/.test(document.querySelector('.topic-node').innerText));
   await page.getByRole('button',{name:'Renommer'}).click();await page.locator('.topic-form input').fill('Écologie en ville');await page.locator('.topic-form button[type=submit]').click();
   await page.waitForFunction(()=>document.querySelector('.topic-head h2').innerText==='Écologie en ville');
   await page.locator('.topic-position').nth(1).getByRole('button',{name:'Supprimer'}).first().click();await page.locator('.confirm-dialog button').last().click();
   await page.waitForFunction(()=>document.querySelectorAll('.topic-position').length===1);
   await page.locator('.topic-all').click();await page.locator('.topic-card').waitFor();assert.match(await page.locator('.topic-card').innerText(),/Écologie en ville[\s\S]*1 position · 4 arguments/);await check('list');await shot('list');
   await page.goBack();await page.locator('.topic-head h2').waitFor();
   await page.getByRole('button',{name:'Supprimer ce sujet'}).click();await page.locator('.confirm-dialog button').last().click();await page.locator('.topic-new').waitFor();assert.equal(await page.locator('.topic-card').count(),0);
   await page.goto('https://archive.test/mes-sujets/inconnu/');await page.locator('.notice.warn').waitFor();assert.match(await page.locator('#view').innerText(),/introuvable[\s\S]*|Tous mes sujets/);
   await ctx.close();
  }
  assert.deepEqual(errors,[]);
  console.log('Topics bench passed: '+shots.length+' captures in '+out);
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
