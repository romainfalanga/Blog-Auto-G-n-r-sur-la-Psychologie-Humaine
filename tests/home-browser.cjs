// The dashboard and the navigation without a menu, in a real browser against a simulated API:
// ten tiles on one phone screen without scrolling, richer cards on a wide screen, a button back
// to the dashboard on every other page. Run: node tests/home-browser.cjs (HOME_SHOTS=<dir>).
const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=process.env.HOME_SHOTS||path.join(root,'work/home-browser');fs.mkdirSync(out,{recursive:true});
 const {HTML,APP_ASSETS,INITIAL_PAGE_DATA_MARKER}=await import('../journal/src/ui.mjs');
 const page$=HTML.replace(INITIAL_PAGE_DATA_MARKER,'null'),navigation=JSON.parse(fs.readFileSync(path.join(root,'data/navigation.json'),'utf8'));
 const readiness={complete:false,identity:{complete:true,missing_fields:[]},people:{count:4,required:3,complete:true},events:{count:12,required:3,complete:true},childhood:{answered:60,required:60,complete:true},symptoms:{answered:212,required:516,complete:false},missing:['symptoms']};
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||(fs.existsSync('/usr/bin/google-chrome')?'/usr/bin/google-chrome':undefined)});
 const errors=[],shots=[];
 try{
  for(const [width,height,touch] of [[390,844,true],[360,640,true],[320,568,true],[1440,900,false],[1024,768,false]]){
   for(const guest of [false,true]){
    const ctx=await browser.newContext({viewport:{width,height},hasTouch:touch,isMobile:touch,deviceScaleFactor:touch?2:1,locale:'fr-FR'});
    await ctx.route('https://archive.test/**',async route=>{
     const r=route.request(),url=new URL(r.url());
     if(url.pathname===APP_ASSETS.script.path)return route.fulfill({status:200,contentType:'application/javascript',body:APP_ASSETS.script.body});
     if(url.pathname===APP_ASSETS.style.path)return route.fulfill({status:200,contentType:'text/css',body:APP_ASSETS.style.body});
     if(url.pathname==='/api/me')return route.fulfill({json:guest?{user:null,configuration:{ai:true}}:{user:{id:'u1',email:'test@example.test',role:'member'},profile:{first_name:'Alex'},companion_readiness:readiness,configuration:{ai:true}}});
     if(url.pathname==='/api/dashboard')return route.fulfill({json:{topics:3,reviews:1,review_running:false,elements:87}});
     if(url.pathname==='/api/people')return route.fulfill({json:{people:[],fields_ready:true}});
     if(r.resourceType()==='document')return route.fulfill({status:200,contentType:'text/html',body:page$});
     return route.fulfill({status:204,body:''});
    });
    const page=await ctx.newPage(),tag=width+'x'+height+(guest?'-guest':'');page.on('pageerror',e=>errors.push(e.message));
    await page.goto('https://archive.test/accueil/');await page.locator('.home-grid').waitFor();await page.waitForTimeout(300);
    const links=await page.locator('.home-tile').evaluateAll(nodes=>nodes.map(n=>({label:n.querySelector('strong').textContent,href:n.getAttribute('href')})));
    assert.deepEqual(links,navigation,tag+': every page of the former menu has its tile, in order');
    const report=await page.evaluate(()=>({scrollY:document.documentElement.scrollHeight-innerHeight,scrollX:document.documentElement.scrollWidth-innerWidth,nav:document.querySelectorAll('.site-navigation, .nav-links, .site-menu-toggle').length,back:getComputedStyle(document.querySelector('.site-back')).visibility,tiles:[...document.querySelectorAll('.home-tile')].map(n=>{const r=n.getBoundingClientRect();return {b:Math.round(r.bottom),h:Math.round(r.height),w:Math.round(r.width),clipped:n.scrollHeight>n.clientHeight+1||n.scrollWidth>n.clientWidth+1}})}));
    assert.equal(report.nav,0,tag+': no menu');assert.equal(report.back,'hidden',tag+': no back button on the dashboard');assert.ok(report.scrollX<=0,tag+': no horizontal scroll');
    assert.ok(report.tiles.every(t=>t.h>=44&&t.w>=44&&!t.clipped),tag+': tiles large enough and not clipped '+JSON.stringify(report.tiles.filter(t=>t.h<44||t.clipped)));
    if(width<900){assert.ok(report.scrollY<=0,tag+': everything fits on one screen ('+report.scrollY+' px over)');assert.ok(report.tiles.every(t=>t.b<=height),tag+': every tile is on screen')}
    if(!guest){assert.match(await page.locator('.home-head').innerText(),/Bonjour Alex/);if(width>=900)assert.match(await page.locator('.home-grid').innerText(),/4 proches[\s\S]*212 \/ 516 réponses[\s\S]*3 sujets[\s\S]*87 éléments reliés[\s\S]*1 bilan/)}
    else{assert.equal(await page.locator('.home-auth button').count(),2)}
    const file=path.join(out,'home-'+tag+'.png');await page.screenshot({path:file});shots.push(file);
    if(!guest&&(width===390||width===1440)){
     // A tile opens its page; the button in the header comes back to the dashboard.
     await page.locator('.home-tile').filter({hasText:'Mes proches'}).click();await page.waitForFunction(()=>location.pathname==='/mes-proches/');
     assert.equal(await page.evaluate(()=>getComputedStyle(document.querySelector('.site-back')).visibility),'visible');assert.equal(await page.evaluate(()=>document.querySelectorAll('.site-navigation, .nav-links, .site-menu-toggle').length),0);
     const back=await page.locator('.site-back').boundingBox();assert.ok(back.height>=44&&back.width>=44,'back button large enough');
     const shot=path.join(out,'page-'+tag+'.png');await page.screenshot({path:shot});shots.push(shot);
     await page.locator('.site-back').click();await page.waitForFunction(()=>location.pathname==='/accueil/');await page.locator('.home-grid').waitFor();
     await page.locator('.home-tile').filter({hasText:'Ma ligne de vie'}).click();await page.waitForFunction(()=>location.pathname==='/ma-ligne-de-vie/');await page.locator('.site-header .site-logo').click();await page.waitForFunction(()=>location.pathname==='/accueil/');
    }
    await ctx.close();
   }
  }
  assert.deepEqual(errors,[]);console.log('Home bench passed: '+shots.length+' captures in '+out);
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
