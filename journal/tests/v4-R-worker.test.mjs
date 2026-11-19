// Lot R, F11/F12/F13/F14/F16: real Worker routes (limiter, account caps, cron isolation,
// internal metrics, sessions) on an in-memory database with every migration.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {database,setup,provider,refused} from './v4-R-helper.mjs';

const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<title>T</title><main>App shell</main><script>window.started=1</script>';").replace("import { EXAMPLES } from './examples.mjs';",'const EXAMPLES=[];')
 .replace("import { createRecommender } from './recommendations.mjs';",'const createRecommender=()=>({recommend:()=>[],provisionalPortrait:()=>null});');
const rewritten=['change-backend','care-backend','chat-backend','entitlements','personal-backend','cartographie','symptomes-backend','navigation.generated','companion-readiness','monthly-quotas'].reduce((text,name)=>text.replace(`'./${name}.mjs'`,JSON.stringify(new URL(`../src/${name}.mjs`,import.meta.url).href)),source);
const worker=await import('data:text/javascript;base64,'+Buffer.from(rewritten).toString('base64'));
const origin='https://journal.example';
function environment(extra={}){
 const raw=database();
 function prepare(sql){let args=[];return {bind(...a){args=a;return this},async first(){return raw.prepare(sql).get(...args)||null},async all(){return {results:raw.prepare(sql).all(...args)}},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}}}}}
 let queue=Promise.resolve();
 return {DB:{raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());raw.exec('COMMIT');return r}catch(e){raw.exec('ROLLBACK');throw e}});queue=next.catch(()=>{});return next}},PUBLIC_ORIGIN:origin,...extra};
}
// Every visitor reaches the Worker through the same Netlify egress address.
const PROXY='198.51.100.7';
const request=(env,path,{method='GET',data,cookie,client}={})=>worker.default.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),Origin:origin,'CF-Connecting-IP':PROXY,...(client?{'x-nf-client-connection-ip':client}:{})},body:data?JSON.stringify(data):undefined}),env,{waitUntil(){}});
const password='long secure password 123';

test('F11: 31 failed logins from one visitor do not block another visitor behind the same proxy',async()=>{
 const env=environment();
 assert.equal((await request(env,'/api/auth/register',{method:'POST',data:{email:'victim@example.test',password},client:'203.0.113.20'})).status,200);
 for(let i=0;i<31;i++){const r=await request(env,'/api/auth/login',{method:'POST',data:{email:`target${i%3}@example.test`,password:'wrong password 1234'},client:'203.0.113.66'});if(i===30)assert.equal(r.status,429)}
 const ok=await request(env,'/api/auth/login',{method:'POST',data:{email:'victim@example.test',password},client:'203.0.113.20'});
 assert.equal(ok.status,200,'another visitor still logs in');
 // The per-email limiter is not bypassed by rotating addresses.
 let last;for(let i=0;i<13;i++)last=await request(env,'/api/auth/login',{method:'POST',data:{email:'victim@example.test',password:'wrong password 1234'},client:'203.0.113.'+(100+i)});
 assert.equal(last.status,429);
});

test('F11: IPv6 visitors are grouped by /64 and the client address helper is stable',()=>{
 const at=ip=>worker.clientAddress(new Request(origin,{headers:{'x-nf-client-connection-ip':ip,'CF-Connecting-IP':PROXY}}));
 assert.equal(at('2001:db8:1:2:aaaa::1'),at('2001:0db8:0001:0002:ffff:1:2:3'));
 assert.notEqual(at('2001:db8:1:2::1'),at('2001:db8:1:3::1'));
 assert.equal(at('203.0.113.9'),'203.0.113.9');
 assert.equal(worker.clientAddress(new Request(origin,{headers:{'CF-Connecting-IP':'192.0.2.1'}})),'192.0.2.1');
});

test('F12: account creation is capped per address and per day',async()=>{
 const env=environment({ACCOUNT_CREATIONS_PER_IP_DAY:'3'});
 for(let i=0;i<3;i++)assert.equal((await request(env,'/api/auth/register',{method:'POST',data:{email:`n${i}@example.test`,password},client:'203.0.113.30'})).status,200);
 assert.equal((await request(env,'/api/auth/register',{method:'POST',data:{email:'n4@example.test',password},client:'203.0.113.30'})).status,429);
 assert.equal((await request(env,'/api/auth/register',{method:'POST',data:{email:'n5@example.test',password},client:'203.0.113.31'})).status,200);
});

test('F11/F14/F13: the cron purges expired limiter rows and sessions, isolates a failing chain and records counters',async t=>{
 const env=environment();const now=Math.floor(Date.now()/1000);
 env.DB.raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','u@example.test','h','s')").run();
 for(let i=0;i<20;i++){env.DB.raw.prepare('INSERT INTO rate_limits(key,count,expires_at) VALUES(?,1,?)').run('k'+i,i<15?now-1:now+600);env.DB.raw.prepare("INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,'u',?)").run('s'+i,i<12?now-1:now+600)}
 provider(t,()=>{throw new Error('no provider in this test')});
 // A portrait chain that rejects no longer hides the other chains.
 env.OPENROUTER_API_KEY='mock-only';
 const original=env.DB.prepare;env.DB.prepare=sql=>{if(/FROM journal_jobs/.test(sql))throw new Error('portrait chain down');return original(sql)};
 const result=await worker.default.scheduled({},env,{waitUntil(){}});
 assert.equal(result.portraits.error,true);assert.equal(result.engine.error,undefined);assert.ok(result.reviews);
 assert.deepEqual(result.housekeeping,{rate_limits_purged:15,sessions_purged:12});
 env.DB.prepare=original;
 const counters=env.DB.raw.prepare('SELECT name,value FROM service_metrics').all().map(r=>r.name);
 assert.ok(counters.includes('cron_runs'));assert.ok(counters.includes('cron_failed_portraits'));
});

test('F13: internal metrics are protected like other internal routes and carry no private text',async t=>{
 const env=environment();
 const r=await worker.default.fetch(new Request(origin+'/api/internal/metrics',{method:'POST'}),env,{});
 assert.equal(r.status,401);
 const s=setup();await s.ready('a');let refuse=true;provider(t,p=>p==='ambre_analysis'&&refuse?refused(429,'5'):undefined);
 await s.call('/api/chat/messages','POST',{content:'Mon secret très personnel : Zéphyrine me manque.',idempotency_key:'metrics_turn_1'});
 refuse=false;s.tick(6);await s.chat.engine.work(s.env,s.raw.prepare('SELECT id FROM chat_turns').get().id);
 const m=await s.chat.metrics(s.env),names=m.counters.map(x=>x.name);
 for(const name of ['provider_refusal_429','turn_waiting','turn_published','provider_calls','provider_input_tokens'])assert.ok(names.includes(name),name);
 assert.equal(m.backlog.turns_live,0);
 assert.doesNotMatch(JSON.stringify(m),/Zéphyrine|secret|a@example|id_0/);
});

test('F14: « Déconnecter tous les appareils » ends every session of the account only',async()=>{
 const env=environment(),login=async(email,client)=>{const r=await request(env,'/api/auth/login',{method:'POST',data:{email,password},client});assert.equal(r.status,200);return r.headers.get('set-cookie').split(';')[0]};
 const first=await request(env,'/api/auth/register',{method:'POST',data:{email:'me@example.test',password},client:'203.0.113.40'}),phone=first.headers.get('set-cookie').split(';')[0];
 await request(env,'/api/auth/register',{method:'POST',data:{email:'other@example.test',password},client:'203.0.113.41'});
 const laptop=await login('me@example.test','203.0.113.42'),other=await login('other@example.test','203.0.113.43');
 const out=await request(env,'/api/auth/logout-all',{method:'POST',data:{},cookie:laptop});assert.equal(out.status,200);assert.match(out.headers.get('set-cookie'),/Max-Age=0/);
 assert.equal((await request(env,'/api/me',{cookie:phone})).status,200);
 assert.equal((await (await request(env,'/api/me',{cookie:phone})).json()).user,null);
 assert.ok((await (await request(env,'/api/me',{cookie:other})).json()).user,'another account keeps its session');
});

test('F16: the application shell runs under a per-response nonce, never unsafe-inline scripts',async()=>{
 const env=environment();
 const page=await worker.default.fetch(new Request(origin+'/mon-compte/'),env,{}),csp=page.headers.get('content-security-policy'),html=await page.text();
 assert.doesNotMatch(csp,/script-src[^;]*unsafe-inline/);
 const nonce=csp.match(/'nonce-([^']+)'/)[1];assert.ok(nonce.length>=20);
 const tags=html.match(/<script[^>]*>/g)||[];assert.ok(tags.length);for(const tag of tags)assert.equal(tag,`<script nonce="${nonce}">`);
 const again=await worker.default.fetch(new Request(origin+'/mon-compte/'),env,{});assert.notEqual(again.headers.get('content-security-policy'),csp);
 const api=await worker.default.fetch(new Request(origin+'/api/health'),env,{});assert.match(api.headers.get('content-security-policy'),/script-src 'self';/);
});

test('F16: an unexpected database error on the memory route never reaches the client',async()=>{
 const s=setup();await s.ready('a');
 const original=s.env.DB.prepare;s.env.DB.prepare=sql=>{if(/FROM chat_memory WHERE user_id/.test(sql))throw new Error('SQLITE_ERROR: near "SELECT summary FROM chat_memory": secret schema');return original(sql)};
 const r=await s.call('/api/chat/memory','PUT',{revision:0,sections:{}});
 s.env.DB.prepare=original;
 assert.equal(r.status,500);assert.doesNotMatch(r.message,/SQLITE|schema|SELECT/);
});
