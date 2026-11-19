const test=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const {join}=require('node:path');
const base='https://decodetonesprit.fr';
const origin='https://decode-journal.romainfalanga83.workers.dev/api/initial-state?page=cartographie';
const makeData=text=>({page:'/cartographie/',responses:{'/api/me':{status:200,body:{user:{id:text}}},'/api/cartographie':{status:200,body:{cartographie:{notes:text}}}}});
const html=()=>new Response('<!doctype html><html><head><title>Mon enfance</title></head><body>Ordinary UI</body></html>',{headers:{'Content-Type':'text/html; charset=utf-8','Cache-Control':'public,max-age=3600','ETag':'public-etag','Last-Modified':'Tue, 17 Nov 2026 10:00:00 GMT','Content-Length':'100','Content-Encoding':'gzip','Vary':'Accept-Encoding'}});
function privateHeaders(response){for(const name of ['Cache-Control','CDN-Cache-Control','Netlify-CDN-Cache-Control'])assert.equal(response.headers.get(name),'private, no-store');assert.match(response.headers.get('Vary'),/Cookie/);assert.equal(response.headers.get('ETag'),null);assert.equal(response.headers.get('Last-Modified'),null);assert.equal(response.headers.get('Content-Length'),null)}
test('edge routes only exact Mon enfance GETs and does not opt into caching',async t=>{
  const {default:handler}=await import('../netlify/edge-functions/personal-page-data.js');
  const config=readFileSync(join(__dirname,'../netlify.toml'),'utf8');
  const blocks=config.split('[[edge_functions]]').slice(1).map(s=>s.split('\n[[')[0]).filter(s=>s.includes('function = "personal-page-data"'));
  assert.equal(blocks.length,4);assert.deepEqual(blocks.map(s=>s.match(/path = "([^"]+)"/)[1]),['/cartographie','/cartographie/','/mes-symptomes','/mes-symptomes/']);assert.ok(blocks.every(s=>!s.includes('cache =')));
  t.mock.method(globalThis,'fetch',()=>assert.fail('No network outside the exact route'));
  const context={next(){assert.fail('No transform outside the exact route')}};
  for(const path of ['/','/cartographie/other','/css/cartographie.css'])assert.equal(await handler(new Request(base+path),context),undefined);
  for(const method of ['POST','HEAD','OPTIONS'])assert.equal(await handler(new Request(base+'/cartographie/',{method}),context),undefined);
});
test('edge forwards only Cookie to the fixed Worker and safely injects per-user JSON with no shared cache',async t=>{
  const {default:handler}=await import('../netlify/edge-functions/personal-page-data.js');
  const attack='</script><script>evil()</script><>&\u2028\u2029';
  const calls=[];t.mock.method(globalThis,'fetch',async(url,options)=>{calls.push([url,options]);return Response.json(makeData(options.headers.get('Cookie')==='session=B'?attack:'session=A'))});
  for(const cookie of ['session=A','session=B']){
    const response=await handler(new Request(base+'/cartographie/?page=https://evil.test',{headers:{Cookie:cookie,'Authorization':'Do not forward','X-Forwarded-Host':'evil.test'}}),{next:async()=>html()});
    privateHeaders(response);assert.equal(response.headers.get('Content-Encoding'),null);const text=await response.text(),match=text.match(/<script id="cartographie-initial" type="application\/json">(.*?)<\/script>/s);assert.ok(match);
    assert.deepEqual(JSON.parse(match[1]),makeData(cookie==='session=B'?attack:cookie));assert.doesNotMatch(match[1],/[<>&\u2028\u2029]/);assert.equal((text.match(/<script\b/g)||[]).length,1);assert.ok(text.indexOf('cartographie-initial')<text.indexOf('</head>'));
  }
  for(const [url,options] of calls){assert.equal(url,origin);assert.equal(options.method,'GET');assert.equal(options.cache,'no-store');assert.equal(options.redirect,'error');assert.deepEqual([...options.headers.keys()],['cookie'])}
});
test('edge failures and non-HTML responses preserve the ordinary response without inventing a session',async t=>{
  const {default:handler}=await import('../netlify/edge-functions/personal-page-data.js');
  for(const failure of [()=>{throw new Error('SECRET')},()=>new Response('bad',{status:503}),()=>new Response('invalid json'),()=>Response.json({page:'/other/',responses:{}})]){
    t.mock.method(globalThis,'fetch',failure);const response=await handler(new Request(base+'/cartographie'),{next:async()=>html()});privateHeaders(response);assert.equal(response.headers.get('Content-Encoding'),'gzip');assert.equal(await response.text(),await html().text());
  }
  t.mock.method(globalThis,'fetch',async()=>Response.json(makeData('a')));
  const response=await handler(new Request(base+'/cartographie/'),{next:async()=>new Response('plain',{headers:{'Content-Type':'text/plain'}})});assert.equal(await response.text(),'plain');privateHeaders(response);
});

test('symptoms bootstrap uses its own endpoint and falls back within a bounded wait',async t=>{
 const {default:handler}=await import('../netlify/edge-functions/personal-page-data.js');
 t.mock.method(globalThis,'fetch',async(url)=>{assert.match(url,/page=symptomes$/);return Response.json({page:'/mes-symptomes/',responses:{'/api/me':{status:200,body:{user:null}},'/api/symptomes':{status:200,body:{cartographie:null}}}})});
 let response=await handler(new Request(base+'/mes-symptomes/'),{next:async()=>html()});privateHeaders(response);assert.match(await response.text(),/cartographie-initial/);
 t.mock.method(globalThis,'fetch',()=>new Promise(()=>{}));const start=Date.now();
 response=await handler(new Request(base+'/mes-symptomes'),{next:async()=>html()});assert.ok(Date.now()-start<1500);privateHeaders(response);assert.doesNotMatch(await response.text(),/cartographie-initial/);
});
