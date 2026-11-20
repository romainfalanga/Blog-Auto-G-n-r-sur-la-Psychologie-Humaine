// Lot P (page speed): content-addressed application files, Ambre prepared with its page,
// concurrent bootstrap reads and a catalogue imported only when a route needs it.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import worker from '../src/worker.mjs';
import {HTML,APP_ASSETS,APP_SCRIPT,APP_STYLE} from '../src/ui.mjs';
import {setup} from './v4-S-helper.mjs';

function fixture(t){
  const s=setup();t.after(()=>s.raw.close());
  for(const id of ['a','b']){
    s.raw.prepare('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)').run(createHash('sha256').update('token-'+id).digest('hex'),id,Math.floor(Date.now()/1000)+3600);
    s.raw.prepare('UPDATE profiles SET self_data=? WHERE user_id=?').run(JSON.stringify({first_name:'Prénom '+id,self_description:'Secret '+id}),id);
  }
  const ctx={waitUntil(){}};
  s.request=(path,user='a',method='GET')=>worker.fetch(new Request('https://example.test'+path,{method,headers:user?{Cookie:'__Host-dte_session=token-'+user}:undefined}),s.env,ctx);
  t.mock.method(globalThis,'fetch',()=>{assert.fail('No HTTP or model call')});
  return s;
}
const initialData=html=>JSON.parse(html.match(/<script type="application\/json" id="initial-page-data">(.*?)<\/script>/s)[1]);

test('the application script and style are content-addressed public files, identical for every visitor',async t=>{
  const s=fixture(t);
  assert.match(APP_ASSETS.script.path,/^\/app\/[0-9a-f]{16,}\.js$/);assert.match(APP_ASSETS.style.path,/^\/app\/[0-9a-f]{16,}\.css$/);
  assert.ok(HTML.includes('<script defer src="'+APP_ASSETS.script.path+'"></script>'));assert.ok(HTML.includes('<link rel="stylesheet" href="'+APP_ASSETS.style.path+'">'));
  assert.ok(HTML.length<20000,'the private shell no longer carries the application: '+HTML.length);
  for(const [asset,type,body] of [[APP_ASSETS.script,/^application\/javascript/,APP_SCRIPT],[APP_ASSETS.style,/^text\/css/,APP_STYLE]]){
    const bodies=[];
    for(const user of ['a','b',null]){
      const response=await s.request(asset.path,user),text=await response.text();
      assert.equal(response.status,200);assert.equal(response.headers.get('Cache-Control'),'public, max-age=31536000, immutable');
      assert.match(response.headers.get('Content-Type'),type);assert.equal(response.headers.get('X-Content-Type-Options'),'nosniff');
      assert.equal(response.headers.get('Set-Cookie'),null);bodies.push(text);
      assert.doesNotMatch(text,/Prénom [ab]|Secret [ab]|a@example\.test|b@example\.test/);
    }
    assert.equal(bodies[0],body);assert.equal(bodies[1],body);assert.equal(bodies[2],body);
    const head=await s.request(asset.path,'a','HEAD');assert.equal(head.status,200);assert.equal(await head.text(),'');
  }
  // The script parses on its own and reads its prepared data from the JSON block, not from a nonce.
  assert.doesNotThrow(()=>new Function(APP_SCRIPT));assert.match(APP_SCRIPT,/getElementById\('initial-page-data'\)/);assert.doesNotMatch(APP_SCRIPT,/INITIAL_PAGE_DATA\*\//);
  for(const path of ['/app/0000000000000000.js','/app/'+APP_ASSETS.script.path.slice(5,-3)+'.css','/app/','/app/../worker.mjs']){
    const response=await s.request(path);assert.equal(response.status,404,path);assert.equal(response.headers.get('Cache-Control'),'private, no-store',path);
  }
  for(const method of ['POST','PUT','DELETE'])assert.notEqual((await s.request(APP_ASSETS.script.path,'a',method)).headers.get('Cache-Control'),'public, max-age=31536000, immutable');
});

test('personal shells keep safe JSON bootstrap, with review lists now prepared and no inline script or nonce',async t=>{
  const s=fixture(t);
  for(const [path,user] of [['/mon-suivi/','a'],['/mes-bilans/','a'],['/mes-diagnostics/','a'],['/mon-compte/',null]]){
    const response=await s.request(path,user),html=await response.text();
    assert.equal(response.headers.get('Cache-Control'),'private, no-store');assert.doesNotMatch(html,/INITIAL_PAGE_DATA/);
    assert.doesNotMatch(html,/<script(?: nonce="[^"]*")?>/);assert.doesNotMatch(response.headers.get('Content-Security-Policy'),/nonce-/);
    const data=initialData(html);if(path==='/mon-compte/')assert.deepEqual(Object.keys(data.responses),['/api/me']);else if(['/mes-bilans/','/mes-diagnostics/'].includes(path)){assert.equal(data.page,path);assert.deepEqual(Object.keys(data.responses),['/api/me','/api/reviews']);assert.equal(data.responses['/api/me'].body.user.id,user);}else assert.equal(data,null);
  }
});

test('Ambre is prepared with its page: /api/me and /api/chat arrive in the HTML, for the owner only',async t=>{
  const s=fixture(t);
  for(const path of ['/ambre/','/ambre']){
    const response=await s.request(path,'a'),html=await response.text(),data=initialData(html);
    assert.equal(response.status,200);assert.equal(response.headers.get('Cache-Control'),'private, no-store');
    assert.equal(data.page,'/ambre/');assert.deepEqual(Object.keys(data.responses),['/api/me','/api/chat']);
    assert.equal(data.responses['/api/me'].body.user.id,'a');
    const direct=await s.request('/api/chat','a');assert.deepEqual(data.responses['/api/chat'],{status:direct.status,body:await direct.json()});
    assert.doesNotMatch(html,/Prénom b|Secret b/);
  }
  for(const user of [null,'expired']){
    const data=initialData(await (await s.request('/ambre/',user)).text());
    assert.deepEqual(Object.keys(data.responses),['/api/me']);assert.equal(data.responses['/api/me'].body.user,null);
  }
  const data=await (await s.request('/api/initial-state?page=chat','b')).json();
  assert.deepEqual(Object.keys(data.responses),['/api/me','/api/chat']);assert.equal(data.responses['/api/me'].body.user.id,'b');
});

test('the page routes start before /api/me has answered, and /api/me starts its own reads together',{timeout:3000},async t=>{
  const s=fixture(t),prepare=s.env.DB.prepare.bind(s.env.DB),started=new Set();let release;
  const gate=new Promise(resolve=>{release=resolve});
  const watched={'SELECT * FROM profiles WHERE user_id=?':'profile','SELECT * FROM subscriptions WHERE user_id=?':'subscription','SELECT text,proposal,version FROM memories WHERE user_id=?':'memory'};
  s.env.DB.prepare=sql=>{
    const statement=prepare(sql),name=watched[sql]||(sql.startsWith('SELECT * FROM personal_people WHERE')?'people':null);
    if(name){const first=statement.first.bind(statement),all=statement.all.bind(statement);const wait=async()=>{started.add(name);if(started.size===4)release();await gate};statement.first=async()=>{await wait();return first()};statement.all=async()=>{await wait();return all()}}
    return statement;
  };
  const data=await (await s.request('/api/initial-state?page=people')).json();
  assert.deepEqual([...started].sort(),['memory','people','profile','subscription']);
  assert.equal(data.responses['/api/me'].status,200);assert.equal(data.responses['/api/people'].status,200);
});

test('the catalogue module is imported only by routes that read it, once per isolate',async t=>{
  let imports;
  const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8')
    .replace("const importExamples=()=>import('./examples.mjs');","const importExamples=async()=>{globalThis.__lotPImports=(globalThis.__lotPImports||0)+1;return import("+JSON.stringify(new URL('../src/examples.mjs',import.meta.url).href)+");};")
    .replace(/from '(\.\/[^']+)'/g,(_,specifier)=>'from '+JSON.stringify(new URL('../src/'+specifier.slice(2),import.meta.url).href));
  assert.ok(source.includes('__lotPImports'),'the lazy import is still the single entry point');
  const isolated=(await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'))).default;
  const s=fixture(t),call=(path,user='a')=>isolated.fetch(new Request('https://example.test'+path,{headers:user?{Cookie:'__Host-dte_session=token-'+user}:undefined}),s.env,{waitUntil(){}});
  imports=()=>globalThis.__lotPImports||0;
  for(const [path,user] of [[APP_ASSETS.script.path,null],[APP_ASSETS.style.path,'a'],['/api/health',null],['/mon-compte/',null],['/api/me',null],['/mon-suivi/','a']])assert.equal((await call(path,user)).status,200,path);
  assert.equal(imports(),0,'static files, shells and anonymous pages do not import the catalogue');
  const me=await (await call('/api/me')).json();assert.equal(me.user.id,'a');assert.equal(imports(),1);
  const list=await (await call('/api/examples',null)).json();assert.ok(list.examples.length>100);
  await call('/api/me','b');assert.equal(imports(),1,'the module is imported once per isolate');
  delete globalThis.__lotPImports;
});
