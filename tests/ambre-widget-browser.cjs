// Shared public/private launcher, real browser, fake same-origin chat to isolate the bridge.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const script=fs.readFileSync(path.join(root,'static/js/ambre-widget.js'),'utf8');
const css=fs.readFileSync(path.join(root,'static/css/ambre-widget.css'),'utf8');
const mock=`<!doctype html><html><body><h1>Conversation</h1><textarea aria-label="Message"></textarea><button id="account">Mon compte</button><script>
function send(type,extra){parent.postMessage(Object.assign({channel:'dte-ambre',version:1,type:type},extra),location.origin)}
window.addEventListener('message',function(e){if(e.source!==parent||e.origin!==location.origin)return;if(e.data.type==='open')send('ready');if(e.data.type==='focus-first')document.querySelector('textarea').focus();if(e.data.type==='focus-last')document.querySelector('#account').focus();if(e.data.type==='close')window.closedCount=(window.closedCount||0)+1});
document.querySelector('#account').onclick=function(){send('navigate',{path:'/mon-compte/'})};
document.addEventListener('keydown',function(e){if(e.key==='Escape')send('close');if(e.key==='Tab'&&e.shiftKey&&document.activeElement.tagName==='TEXTAREA'){e.preventDefault();send('focus-close')}else if(e.key==='Tab'&&!e.shiftKey&&document.activeElement.id==='account'){e.preventDefault();send('focus-header')}});send('ready');
</script></body></html>`;
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});
 const context=await browser.newContext(),page=await context.newPage();let loads=0;const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await context.route('https://widget.test/**',route=>{
  if(new URL(route.request().url()).searchParams.get('embedded')==='1'){loads++;return route.fulfill({contentType:'text/html',body:mock})}
  return route.fulfill({contentType:'text/html',body:'<!doctype html><html><head><style>'+css+'</style></head><body><h1>Ma page</h1><a href="/mes-proches/">Mes proches</a><script>'+script+'</script></body></html>'});
 });
 try{
  for(const width of [320,390,768,1440]){
   await page.setViewportSize({width,height:820});await page.goto('https://widget.test/cartographie/');
   const before=loads;assert.equal(await page.locator('iframe').count(),0);
   await page.getByRole('button',{name:'Ouvrir la conversation avec Ambre'}).click();
   await page.locator('#ambre-panel iframe').waitFor({state:'visible'});assert.equal(loads,before+1);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   const box=await page.locator('#ambre-panel').boundingBox();assert.ok(box.x>=0&&box.x+box.width<=width+1);
   const chat=page.frameLocator('#ambre-panel iframe');await chat.getByRole('textbox',{name:'Message'}).fill('Mon brouillon préservé');
   await page.getByRole('button',{name:'Fermer la conversation'}).click();await page.locator('#ambre-panel').waitFor({state:'hidden'});
   assert.equal(await page.locator('#ambre-launcher').evaluate(n=>n===document.activeElement),true);
   await page.locator('#ambre-launcher').click();await page.locator('#ambre-panel iframe').waitFor({state:'visible'});
   assert.equal(await chat.getByRole('textbox',{name:'Message'}).inputValue(),'Mon brouillon préservé');assert.equal(loads,before+1);
   // A same-origin message from the parent itself must not pass the source guard.
   await page.evaluate(()=>window.postMessage({channel:'dte-ambre',version:1,type:'navigate',path:'/mon-compte/'},location.origin));
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(resolve)));assert.equal(new URL(page.url()).pathname,'/cartographie/');
   await chat.getByRole('button',{name:'Mon compte'}).focus();await page.keyboard.press('Tab');await page.waitForFunction(()=>document.activeElement===document.querySelector('.ambre-panel-bar a'));
   await page.keyboard.press('Shift+Tab');await page.waitForFunction(()=>{var d=document.querySelector('iframe').contentDocument;return d.activeElement===d.querySelector('#account')});
   await chat.getByRole('textbox').press('Shift+Tab');await page.waitForFunction(()=>document.activeElement===document.querySelector('.ambre-panel-close'));
   await page.keyboard.press('Tab');await page.waitForFunction(()=>{var d=document.querySelector('iframe').contentDocument;return d.activeElement===d.querySelector('textarea')});
   await chat.getByRole('textbox').press('Escape');await page.locator('#ambre-panel').waitFor({state:'hidden'});
   await page.locator('#ambre-launcher').click();await page.locator('#ambre-panel iframe').waitFor({state:'visible'});
   await chat.getByRole('button',{name:'Mon compte'}).click();await page.waitForURL('https://widget.test/mon-compte/');
  }
  await page.goto('https://widget.test/mon-accompagnement/');assert.equal(await page.locator('#ambre-launcher').isVisible(),false);
  assert.deepEqual(errors,[]);console.log('Widget: 4 widths, lazy frame, draft persistence, focus, Escape, strict bridge and account navigation passed.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
