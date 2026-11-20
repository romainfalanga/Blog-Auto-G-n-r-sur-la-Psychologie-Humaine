// « Ma cartographie » in a real browser against a simulated /api/graph (no Worker): drawing,
// selection, linked elements, search, filters by kind, zoom. 390 touch and 1440.
// Run: node tests/graph-browser.cjs  (GRAPH_SHOTS=<dir> to choose where screenshots go).
const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=process.env.GRAPH_SHOTS||path.join(root,'work/graph-browser');fs.mkdirSync(out,{recursive:true});
 const {HTML,APP_ASSETS,INITIAL_PAGE_DATA_MARKER}=await import('../journal/src/ui.mjs');
 const page$=HTML.replace(INITIAL_PAGE_DATA_MARKER,'null');
 const nodes=[{id:'me',type:'me',label:'Alex',href:'/mon-compte/',detail:'Le centre.'}],edges=[];
 const add=(id,type,label,parent,extra={})=>{nodes.push({id,type,label,detail:'Détail fictif de '+label+'.',href:'/mes-proches/',...extra});if(parent)edges.push({source:parent,target:id})};
 for(const [key,label] of [['people','Mes proches'],['events','Ma ligne de vie'],['childhood','Mon enfance'],['symptoms','Mes symptômes'],['topics','Mes sujets'],['reviews','Mes bilans']])add('page:'+key,'page',label,'me');
 ['Léa','Camille','Jonas','Maman','Papa','Inès'].forEach((name,i)=>add('person:'+i,'person',name,'page:people'));
 for(let i=0;i<14;i++){add('event:'+i,'event','Moment fictif '+(i+1),'page:events');edges.push({source:'event:'+i,target:'person:'+(i%6)});if(i%3===0)edges.push({source:'event:'+i,target:'person:'+((i+2)%6)});if(i%4===0)add('note:'+i,'note','Une note sur ce moment','event:'+i)}
 for(let i=0;i<9;i++)add('childhood:'+i,'childhood','Vécu d’enfance '+(i+1),'page:childhood');
 for(let t=0;t<6;t++){add('theme:'+t,'theme','Thème '+(t+1),'page:symptoms');for(let i=0;i<7;i++)add('symptom:'+t+'-'+i,i%5===0?'past':'symptom','Signe '+(t+1)+'.'+(i+1),'theme:'+t)}
 for(let t=0;t<3;t++){add('topic:'+t,'topic',['Écologie','Travail','Éducation'][t],'page:topics');for(let p=0;p<2;p++){add('idea:'+t+p,'position','Position '+(p+1),'topic:'+t);add('idea:'+t+p+'s','support','Une raison','idea:'+t+p);add('idea:'+t+p+'o','objection','Une objection','idea:'+t+p)}}
 add('review:1','review','Bilan initial du 3 novembre 2026','page:reviews');
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||(fs.existsSync('/usr/bin/google-chrome')?'/usr/bin/google-chrome':undefined)});
 const errors=[],shots=[];
 try{
  for(const [width,touch] of [[390,true],[1440,false]]){
   const ctx=await browser.newContext({viewport:{width,height:touch?844:900},hasTouch:touch,isMobile:touch,deviceScaleFactor:touch?2:1,locale:'fr-FR'});
   await ctx.route('https://archive.test/**',async route=>{
    const r=route.request(),url=new URL(r.url());
    if(url.pathname===APP_ASSETS.script.path)return route.fulfill({status:200,contentType:'application/javascript',body:APP_ASSETS.script.body});
    if(url.pathname===APP_ASSETS.style.path)return route.fulfill({status:200,contentType:'text/css',body:APP_ASSETS.style.body});
    if(url.pathname==='/api/me')return route.fulfill({json:{user:{id:'u1',email:'test@example.test',role:'member'},configuration:{ai:true}}});
    if(url.pathname==='/api/graph')return route.fulfill({json:{nodes,edges,truncated:false}});
    if(r.resourceType()==='document')return route.fulfill({status:200,contentType:'text/html',body:page$});
    return route.fulfill({status:204,body:''});
   });
   const page=await ctx.newPage(),tag=String(width);page.on('pageerror',e=>errors.push(e.message));
   const shot=async name=>{const file=path.join(out,name+'-'+tag+'.png');await page.screenshot({path:file,fullPage:false});shots.push(file)};
   const overflow=async name=>assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,name+': horizontal scroll');
   await page.goto('https://archive.test/ma-cartographie/');await page.locator('.graph-stage canvas').waitFor();await page.waitForTimeout(1500);
   assert.equal(await page.locator('h1').innerText(),'Ma cartographie');assert.match(await page.locator('.graph-panel').innerText(),new RegExp(nodes.length+' éléments et '+edges.length+' liens'));
   const painted=await page.evaluate(()=>{const c=document.querySelector('.graph-stage canvas'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<d.length;i+=4)if(d[i])n++;return n});assert.ok(painted>2000,'the graph is drawn ('+painted+' pixels)');
   await overflow('graph');await shot('graph');
   // Search, then open a result: the panel shows what it is linked to.
   await page.locator('.graph-tools input').fill('lea');await page.locator('.graph-results button').first().waitFor();assert.match(await page.locator('.graph-results').innerText(),/Léa/);
   await page.locator('.graph-results button').first().click();
   assert.match(await page.locator('.graph-panel').innerText(),/Proches[\s\S]*Léa[\s\S]*Ouvrir[\s\S]*Relié à \(\d+\)[\s\S]*Mes proches[\s\S]*Moment fictif/);await shot('selected');
   await page.locator('.graph-panel .graph-links button').filter({hasText:'Moment fictif 1'}).first().click();assert.match(await page.locator('.graph-panel h2').innerText(),/Moment fictif 1/);
   await page.locator('.graph-tools input').fill('');
   // A kind can be hidden: its elements leave the lists.
   await page.locator('.graph-filters summary').click();
   await page.locator('.graph-kind').filter({hasText:'Proches'}).click();assert.equal(await page.locator('.graph-kind').filter({hasText:'Proches'}).getAttribute('aria-pressed'),'false');
   assert.doesNotMatch(await page.locator('.graph-panel .graph-links').innerText(),/Léa/);
   await page.locator('.graph-kind').filter({hasText:'Proches'}).click();
   // Tapping the centre of the stage selects or clears without error; every control is large enough.
   const box=await page.locator('.graph-stage').boundingBox();if(touch)await page.touchscreen.tap(box.x+box.width/2,box.y+box.height/2);else{await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.wheel(0,-300);await page.mouse.click(box.x+box.width/2,box.y+box.height/2)}
   const small=await page.evaluate(()=>[...document.querySelectorAll('#view button, #view a')].filter(n=>n.offsetParent).map(n=>({t:n.textContent.trim().slice(0,20),h:Math.round(n.getBoundingClientRect().height)})).filter(x=>x.h<44));assert.deepEqual(small,[],'targets under 44 px');
   await overflow('end');await page.locator('.graph-tools button').click();await shot('recentred');
   await ctx.close();
  }
  assert.deepEqual(errors,[]);console.log('Graph bench passed: '+shots.length+' captures in '+out);
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
