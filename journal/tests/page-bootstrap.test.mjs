import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import worker from '../src/worker.mjs';
import {setup} from './v4-S-helper.mjs';

const paths={account:'/mon-compte/',people:'/mes-proches/',events:'/ma-ligne-de-vie/',cartographie:'/cartographie/'};
const routes={account:['/api/self-profile'],people:['/api/people'],events:['/api/people','/api/life-events'],cartographie:['/api/cartographie']};
const malicious='</script><script>alert("x")</script><>&\u2028\u2029';
function fixture(t){
  const s=setup();t.after(()=>s.raw.close());
  for(const id of ['a','b']){
    s.raw.prepare('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)').run(createHash('sha256').update('token-'+id).digest('hex'),id,Math.floor(Date.now()/1000)+3600);
    s.raw.prepare('UPDATE profiles SET self_data=? WHERE user_id=?').run(JSON.stringify({self_description:id==='a'?malicious:'Presentation B',first_name:'Name '+id}),id);
    s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role) VALUES(?,?,?,'hash',?,'friend')").run('person-'+id,id,'person-'+id,'Person '+id);
    s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES(?,?,?,'hash',?,'other',?)").run('event-'+id,id,'event-'+id,'Event '+id,'Private '+id);
    s.raw.prepare('INSERT INTO cartographies(user_id,answers,consent,version,updated_at) VALUES(?,?,1,1,1700000000)').run(id,JSON.stringify({abus_physique:id==='a'?'unknown':2}));
  }
  const ctx={waitUntil(){throw new Error('Bootstrap must not schedule work')}};
  s.request=(path,user='a',method='GET')=>worker.fetch(new Request('https://example.test'+path,{method,headers:user?{Cookie:'__Host-dte_session=token-'+user}:undefined}),s.env,ctx);
  t.mock.method(globalThis,'fetch',()=>{assert.fail('Bootstrap must not call HTTP or a model')});
  return s;
}
test('initial state whitelists each page, reuses the authenticated GET data and never mutates storage',async t=>{
  const s=fixture(t),before=s.raw.prepare('SELECT total_changes() n').get().n;
  for(const user of ['a','b'])for(const page of Object.keys(paths)){
    const response=await s.request('/api/initial-state?page='+page,user),data=await response.json();
    assert.equal(response.status,200);assert.equal(response.headers.get('Cache-Control'),'private, no-store');
    assert.equal(data.page,paths[page]);assert.deepEqual(Object.keys(data.responses),['/api/me',...routes[page]]);
    assert.equal(data.responses['/api/me'].body.user.id,user);
    for(const path of routes[page]){const direct=await s.request(path,user);assert.deepEqual(data.responses[path],{status:direct.status,body:await direct.json()})}
    if(page==='people')assert.deepEqual(data.responses['/api/people'].body.people.map(p=>p.id),['person-'+user]);
    if(page==='events')assert.deepEqual(data.responses['/api/life-events'].body.events.map(e=>e.id),['event-'+user]);
  }
  assert.equal(s.raw.prepare('SELECT total_changes() n').get().n,before);
});
test('anonymous and expired sessions contain only me; methods and unknown page keys are rejected without database reads',async t=>{
  const s=fixture(t);
  for(const user of [null,'expired'])for(const page of Object.keys(paths)){
    const data=await (await s.request('/api/initial-state?page='+page,user)).json();assert.deepEqual(Object.keys(data.responses),['/api/me']);assert.equal(data.responses['/api/me'].body.user,null);
  }
  s.env.DB.prepare=()=>assert.fail('Invalid page/method or HEAD must not access D1');
  for(const key of ['','other','__proto__','constructor','https://evil.test/'])assert.equal((await s.request('/api/initial-state?page='+encodeURIComponent(key))).status,400);
  for(const method of ['POST','PUT','DELETE','HEAD','OPTIONS'])assert.equal((await s.request('/api/initial-state?page=account','a',method)).status,405);
  for(const page of ['account','people','events']){const response=await s.request(paths[page],'a','HEAD');assert.equal(response.status,200);assert.equal(await response.text(),'')}
  for(const path of ['/mon-suivi/','/mes-bilans/'])assert.equal((await s.request(path)).status,200,'Other HTML routes remain shells without database reads');
});
test('event bootstrap starts both independent business reads before either has completed',{timeout:3000},async t=>{
  const s=fixture(t),prepare=s.env.DB.prepare.bind(s.env.DB),started=new Set();let release;
  const bothStarted=new Promise(resolve=>{release=resolve});
  s.env.DB.prepare=sql=>{
    const statement=prepare(sql),table=sql.startsWith('SELECT * FROM personal_people WHERE')?'people':sql.startsWith('SELECT * FROM life_events WHERE')?'events':null;
    if(table){const all=statement.all.bind(statement);statement.all=async()=>{started.add(table);if(started.size===2)release();await bothStarted;return all()}}
    return statement;
  };
  const data=await (await s.request('/api/initial-state?page=events')).json();
  assert.equal(started.size,2);assert.equal(data.responses['/api/people'].status,200);assert.equal(data.responses['/api/life-events'].status,200);
});
test('initial entry errors are safe and isolated; a failed session check does not invent an anonymous user',async t=>{
  const s=fixture(t),prepare=s.env.DB.prepare.bind(s.env.DB);
  s.env.DB.prepare=sql=>{if(sql.startsWith('SELECT * FROM personal_people'))throw new Error('SECRET_DB_CONNECTION');return prepare(sql)};
  let response=await s.request('/api/initial-state?page=events'),data=await response.json();
  assert.equal(data.responses['/api/people'].status,500);assert.equal(data.responses['/api/life-events'].status,200);assert.doesNotMatch(JSON.stringify(data),/SECRET_DB_CONNECTION/);
  s.env.DB.prepare=()=>{throw new Error('SECRET_SESSION')};
  data=await (await s.request('/api/initial-state?page=account')).json();
  assert.deepEqual(Object.keys(data.responses),['/api/me']);assert.equal(data.responses['/api/me'].status,500);assert.equal(data.responses['/api/me'].body.user,undefined);assert.doesNotMatch(JSON.stringify(data),/SECRET_SESSION/);
});
test('personal HTML embeds escaped data in a non-executable JSON block beside the external application file',async t=>{
  const s=fixture(t);
  for(const key of ['account','people','events']){
    const response=await s.request(paths[key]),html=await response.text();assert.equal(response.status,200);
    assert.doesNotMatch(html,/\/\*INITIAL_PAGE_DATA\*\/null/);assert.equal((html.match(/<script\b/g)||[]).length,2);assert.doesNotMatch(html,/ambre-launcher|ambre-panel|ambre-widget|<iframe/);
    assert.match(html,new RegExp('"page":"'+paths[key]+'"'));
    // Lot P: no inline executable script remains, so the policy carries no nonce and no unsafe-inline script.
    assert.doesNotMatch(html,/<script(?: nonce="[^"]*")?>/);assert.match(html,/<script defer src="\/app\/[0-9a-f]+\.js"><\/script>/);assert.match(html,/<link rel="stylesheet" href="\/app\/[0-9a-f]+\.css">/);
    const csp=response.headers.get('Content-Security-Policy');assert.match(csp,/script-src 'self';/);assert.doesNotMatch(csp,/nonce-|script-src[^;]*unsafe-inline/);
    assert.equal(initialData(html).page,paths[key]);
    if(key==='account'){
      assert.ok(html.includes('\\u003c/script\\u003e'));assert.ok(html.includes('\\u0026\\u2028\\u2029'));assert.ok(!html.includes(malicious));
      assert.ok(html.includes('"first_name":"Name a"'));assert.ok(!html.includes('Presentation B'));
    }
  }
});
test('no HTML may be framed any more, even with the retired embedded parameter; old portrait redirects to account',async t=>{
  const s=fixture(t);
  for(const path of ['/ambre/?embedded=1','/ambre?embedded=1','/ambre/','/mon-compte/?embedded=1','/mes-proches/person-a/?embedded=1','/api/me?embedded=1']){
    const response=await s.request(path);
    assert.equal(response.status,200);
    assert.equal(response.headers.get('X-Frame-Options'),'DENY');
    assert.ok(response.headers.get('Content-Security-Policy').includes("frame-ancestors 'none'"));
    assert.doesNotMatch(await response.text(),/ambre-launcher|ambre-embedded|dte-ambre|postMessage/);
  }
  const response=await s.request('/mon-portrait/?article=example&like=1');
  assert.equal(response.status,302);assert.equal(response.headers.get('Location'),'/mon-compte/?article=example&like=1');
});
const initialData=html=>JSON.parse(html.match(/<script type="application\/json" id="initial-page-data">(.*?)<\/script>/s)[1]);
test('a person page is prepared for its owner only; another account or an unknown identifier reads as not found',async t=>{
  const s=fixture(t),before=s.raw.prepare('SELECT total_changes() n').get().n;
  for(const path of ['/mes-proches/person-a/','/mes-proches/person-a']){
    const response=await s.request(path,'a'),html=await response.text(),data=initialData(html);
    assert.equal(response.status,200);assert.equal(response.headers.get('Cache-Control'),'private, no-store');assert.equal((html.match(/<script\b/g)||[]).length,2);
    assert.equal(data.page,'/mes-proches/person-a/');assert.deepEqual(Object.keys(data.responses),['/api/me','/api/people/person-a','/api/people/person-a/timeline']);
    assert.equal(data.responses['/api/people/person-a'].body.person.display_name,'Person a');assert.equal(typeof data.responses['/api/people/person-a'].body.fields_ready,'boolean');
    assert.equal(data.responses['/api/people/person-a/timeline'].body.items[0].kind,'relationship_start');
    for(const route of ['/api/people/person-a','/api/people/person-a/timeline']){const direct=await s.request(route,'a');assert.deepEqual(data.responses[route],{status:direct.status,body:await direct.json()})}
  }
  for(const [path,user] of [['/mes-proches/person-a/','b'],['/mes-proches/missing/','a'],['/mes-proches/person-b/','a']]){
    const response=await s.request(path,user),html=await response.text(),data=initialData(html),id=path.split('/')[2];
    assert.equal(response.status,200,'the shell renders the message');assert.equal(data.responses['/api/me'].body.user.id,user);
    for(const route of ['/api/people/'+id,'/api/people/'+id+'/timeline']){assert.equal(data.responses[route].status,404);assert.equal(data.responses[route].body.code,'not_found')}
    assert.doesNotMatch(html,/Person [ab]"/,'no name of another account reaches the page');
  }
  const anonymous=initialData(await (await s.request('/mes-proches/person-a/',null)).text());
  assert.deepEqual(Object.keys(anonymous.responses),['/api/me']);assert.equal(anonymous.responses['/api/me'].body.user,null);
  assert.equal(s.raw.prepare('SELECT total_changes() n').get().n,before,'opening a person page never writes');
  let response=await s.request('/mes-proches/?person=person-a','a');assert.equal(response.status,302);assert.equal(response.headers.get('Location'),'/mes-proches/person-a/');
  assert.equal((await s.request('/mes-proches/?person=..%2Fapi','a')).status,200,'an invalid legacy identifier stays on the list');
  for(const path of ['/mes-proches/person-a/extra/','/mes-proches/person-a/timeline','/mes-proches/a%2Fb/'])assert.equal((await s.request(path,'a')).status,404,path);
  assert.notEqual((await s.request('/mes-proches/person-a/','a','POST')).status,200,'the page only answers GET and HEAD');
  s.env.DB.prepare=()=>assert.fail('HEAD must not access D1');
  response=await s.request('/mes-proches/person-a/','a','HEAD');assert.equal(response.status,200);assert.equal(await response.text(),'');
});
