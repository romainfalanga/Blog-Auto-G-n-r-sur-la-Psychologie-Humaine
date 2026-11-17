import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import worker from '../src/worker.mjs';
import {setup} from './v4-S-helper.mjs';

const paths={portrait:'/mon-portrait/',people:'/mes-proches/',events:'/ma-ligne-de-vie/',cartographie:'/cartographie/'};
const routes={portrait:['/api/self-profile'],people:['/api/people'],events:['/api/people','/api/life-events'],cartographie:['/api/cartographie']};
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
  for(const method of ['POST','PUT','DELETE','HEAD','OPTIONS'])assert.equal((await s.request('/api/initial-state?page=portrait','a',method)).status,405);
  for(const page of ['portrait','people','events']){const response=await s.request(paths[page],'a','HEAD');assert.equal(response.status,200);assert.equal(await response.text(),'')}
  for(const path of ['/mon-compte/','/parametres/','/mes-bilans/','/mon-accompagnement/'])assert.equal((await s.request(path)).status,200,'Other HTML routes remain shells without database reads');
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
  data=await (await s.request('/api/initial-state?page=portrait')).json();
  assert.deepEqual(Object.keys(data.responses),['/api/me']);assert.equal(data.responses['/api/me'].status,500);assert.equal(data.responses['/api/me'].body.user,undefined);assert.doesNotMatch(JSON.stringify(data),/SECRET_SESSION/);
});
test('personal HTML embeds escaped data in its existing nonce-bearing script',async t=>{
  const s=fixture(t);
  for(const key of ['portrait','people','events']){
    const response=await s.request(paths[key]),html=await response.text();assert.equal(response.status,200);
    assert.doesNotMatch(html,/\/\*INITIAL_PAGE_DATA\*\/null/);assert.equal((html.match(/<script\b/g)||[]).length,1);
    assert.match(html,new RegExp('"page":"'+paths[key]+'"'));
    const nonce=html.match(/<script nonce="([^"]+)"/)[1];assert.ok(response.headers.get('Content-Security-Policy').includes("'nonce-"+nonce+"'"));
    if(key==='portrait'){
      assert.ok(html.includes('\\u003c/script\\u003e'));assert.ok(html.includes('\\u0026\\u2028\\u2029'));assert.ok(!html.includes(malicious));
      assert.ok(html.includes('"first_name":"Name a"'));assert.ok(!html.includes('Presentation B'));
    }
  }
});
