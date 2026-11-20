// Ma ligne de vie: dates chosen in lists (day, month, year, period) and the grouped life line,
// in a real browser against a simulated API (no Worker). 390 touch and 1440.
// Run: node tests/timeline-dates-browser.cjs  (TIMELINE_SHOTS=<dir> to choose where screenshots go).
const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=process.env.TIMELINE_SHOTS||path.join(root,'work/timeline-dates-browser');fs.mkdirSync(out,{recursive:true});
 const {HTML,APP_ASSETS,INITIAL_PAGE_DATA_MARKER}=await import('../journal/src/ui.mjs');
 const page$=HTML.replace(INITIAL_PAGE_DATA_MARKER,'null');
 const moment=(id,title,date_precision,date_start,date_end='',extra={})=>({id,title,kind:'other',description:'Un récit fictif pour ce moment.',meaning_now:'',date_precision,date_start,date_end,age_text:'',current_importance:null,significance:null,people_ids:[],revision:1,created_at:'2026-11-01 10:00:00',updated_at:'2026-11-01 10:00:00',...extra});
 const seed=()=>[moment('e1','Début de mon premier emploi','range','2019-03','2021-06-15'),moment('e2','Déménagement à Lyon','day','2021-09-04'),moment('e3','Une année de doutes','year','2021'),moment('e4','Rencontre de Camille','month','2021-09'),moment('e5','Cours du soir','range','2022-01',''),moment('e6','Voyage en Italie','range','2022-07-10','2022-07-24'),moment('e7','Naissance de ma nièce','day','2022-11-30'),moment('e8','Un souvenir flou','unknown',''),moment('e9','Ancienne saisie','age','','',{age_text:'Vers 15 ans'})];
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||(fs.existsSync('/usr/bin/google-chrome')?'/usr/bin/google-chrome':undefined)});
 const errors=[],shots=[];
 try{
  for(const [width,touch] of [[390,true],[1440,false]]){
   let events=seed(),sent=[];
   const ctx=await browser.newContext({viewport:{width,height:touch?844:900},hasTouch:touch,isMobile:touch,deviceScaleFactor:touch?2:1,locale:'fr-FR'});
   await ctx.route('https://archive.test/**',async route=>{
    const r=route.request(),url=new URL(r.url()),m=r.method();
    if(url.pathname===APP_ASSETS.script.path)return route.fulfill({status:200,contentType:'application/javascript',body:APP_ASSETS.script.body});
    if(url.pathname===APP_ASSETS.style.path)return route.fulfill({status:200,contentType:'text/css',body:APP_ASSETS.style.body});
    if(url.pathname==='/api/me')return route.fulfill({json:{user:{id:'u1',email:'test@example.test',role:'member'},configuration:{ai:true}}});
    if(url.pathname==='/api/people')return route.fulfill({json:{people:[],fields_ready:true}});
    if(url.pathname==='/api/life-events'&&m==='GET'){const sorted=[...events].sort((a,b)=>(b.date_start||'').localeCompare(a.date_start||''));return route.fulfill({json:{events:sorted,fields_ready:true,significance_ready:false,next_cursor:null}})}
    if(url.pathname==='/api/life-events'&&m==='POST'){const data=JSON.parse(r.postData());sent.push(data);const event=moment('n'+sent.length,data.title,data.date_precision,data.date_start,data.date_end);events.push(event);return route.fulfill({status:201,json:{event,source_version:2}})}
    if(r.resourceType()==='document')return route.fulfill({status:200,contentType:'text/html',body:page$});
    return route.fulfill({status:204,body:''});
   });
   const page=await ctx.newPage(),tag=String(width);page.on('pageerror',e=>errors.push(e.message));
   const shot=async name=>{const file=path.join(out,name+'-'+tag+'.png');await page.screenshot({path:file,fullPage:true});shots.push(file)};
   const overflow=async name=>assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,name+': horizontal scroll');
   await page.goto('https://archive.test/ma-ligne-de-vie/');await page.locator('.event-timeline').waitFor();
   const text=await page.locator('.event-timeline').innerText();
   assert.match(text,/2022[\s\S]*Se poursuit cette année : Début de mon premier emploi[\s\S]*2019/,'a period is recalled in the years it runs through... ');
   assert.match(text,/2021[\s\S]*Au cours de l’année[\s\S]*Une année de doutes[\s\S]*Septembre[\s\S]*Déménagement à Lyon[\s\S]*Rencontre de Camille/);
   assert.match(text,/Depuis janvier 2022 · /);assert.match(text,/10 juillet 2022 – 24 juillet 2022 · 15 jours/);assert.match(text,/mars 2019 – 15 juin 2021 · 2 ans et 3 mois/);
   assert.match(await page.locator('.undated-events').innerText(),/Un souvenir flou[\s\S]*Ancienne saisie/);
   await overflow('timeline');await shot('timeline');
   // A new moment: a day, then a period with its own precisions, then a period still running.
   const open=()=>page.locator('#events-editor summary').click(),form=page.locator('#events-form');
   const pick=async(box,values)=>{const selects=form.locator('fieldset.date-picker:not([hidden])').nth(box).locator('select:not([hidden])');for(const [index,value] of values)await selects.nth(index).selectOption(value)};
   await open();await form.waitFor();
   assert.equal(await form.locator('input[type=text][name^=date], input[type=month], input[type=date], input[name=age_text]:not([type=hidden])').count(),0,'no typed date');
   assert.equal(await form.locator('select[name=date_precision] option[value=age]').count(),0,'no free-text age for a new moment');
   await form.locator('[name=title]').fill('Mon mariage');await form.locator('[name=description]').fill('Un jour fictif.');
   await form.locator('[name=date_precision]').selectOption('day');await pick(0,[[0,'29'],[1,'02'],[2,'2020']]);
   assert.equal(await form.locator('[name=date_start]').inputValue(),'2020-02-29');
   await pick(0,[[2,'2021']]);assert.equal(await form.locator('[name=date_start]').inputValue(),'','29 February does not exist in 2021: the day is asked again');
   await pick(0,[[0,'28']]);await overflow('form day');await shot('form-day');
   await form.locator('button[type=submit]').click();await page.waitForFunction(()=>/Mon mariage/.test((document.querySelector('.event-timeline')||{}).innerText||''));
   assert.deepEqual([sent[0].date_precision,sent[0].date_start,sent[0].date_end],['day','2021-02-28','']);
   await open();await form.waitFor();await form.locator('[name=title]').fill('Mes études');await form.locator('[name=description]').fill('Une période fictive.');
   await form.locator('[name=date_precision]').selectOption('range');
   await pick(0,[[0,'year'],[1,'2010']]);await pick(1,[[0,'month'],[1,'06'],[2,'2015']]);await overflow('form range');await shot('form-range');
   await form.locator('button[type=submit]').click();await page.waitForFunction(()=>/Mes études/.test((document.querySelector('.event-timeline')||{}).innerText||''));
   assert.deepEqual([sent[1].date_precision,sent[1].date_start,sent[1].date_end],['range','2010','2015-06']);
   await open();await form.waitFor();await form.locator('[name=title]').fill('Mon travail actuel');await form.locator('[name=description]').fill('Une période en cours.');
   await form.locator('[name=date_precision]').selectOption('range');await pick(0,[[0,'month'],[1,'04'],[2,'2024']]);await form.locator('.date-ongoing input').check();
   assert.equal(await form.locator('fieldset.date-picker:not([hidden])').count(),1,'no end to choose for a running period');
   await form.locator('button[type=submit]').click();await page.waitForFunction(()=>/Mon travail actuel/.test((document.querySelector('.event-timeline')||{}).innerText||''));
   assert.deepEqual([sent[2].date_precision,sent[2].date_start,sent[2].date_end],['range','2024-04','']);
   assert.match(await page.locator('.event-timeline').innerText(),/Depuis avril 2024/);await overflow('after');await shot('after');
   await ctx.close();
  }
  assert.deepEqual(errors,[]);console.log('Timeline dates bench passed: '+shots.length+' captures in '+out);
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
