import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
const examples=[{id:'example-a',title:'Un exemple',url:'https://decodetonesprit.fr/posts/example-a/',character:'Alice',description:'Description',body:'Canonical public article A'},{id:'example-b',title:'Autre exemple',url:'https://decodetonesprit.fr/posts/example-b/',character:'Benoît',description:'Description B',body:'Canonical public article B'}];
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<h1>Journal</h1>';").replace("import { EXAMPLES } from './examples.mjs';", "const EXAMPLES="+JSON.stringify(examples)+";").replace("import { generateJournal } from './generator.mjs';", "const generateJournal=(job,env)=>env.__generator(job);");
const {default:worker,passwordHash,constantEqual,verifyStripeSignature,validateClaims,activeSubscription,claimJobs,completeJob,runGenerationBatch}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
function database() {
 const db=new DatabaseSync(':memory:'); db.exec(readFileSync(new URL('../migrations/0001.sql',import.meta.url),'utf8')); db.exec(readFileSync(new URL('../migrations/0002_inspirations.sql',import.meta.url),'utf8')); for(const example of examples) db.prepare('INSERT INTO public_examples(id,body) VALUES(?,?)').run(example.id,example.body);
 function prepare(sql){ let params=[]; return {bind(...v){params=v;return this;},async first(){return db.prepare(sql).get(...params)||null;},async all(){return {results:db.prepare(sql).all(...params)};},async run(){const r=db.prepare(sql).run(...params);return {meta:{changes:r.changes}};}};}
 let queue=Promise.resolve(); return {raw:db,prepare,batch(statements){const task=queue.then(async()=>{db.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());db.exec('COMMIT');return r;}catch(e){db.exec('ROLLBACK');throw e;}});queue=task.catch(()=>{});return task;}};
}
const origin='https://journal.example';
function request(env,path,{method='GET',data,cookie,originHeader=origin}={}) {return worker.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),...(originHeader?{Origin:originHeader}:{}),'CF-Connecting-IP':'192.0.2.8'},body:data?JSON.stringify(data):undefined}),env);}
async function register(env,email){const response=await request(env,'/api/auth/register',{method:'POST',data:{email,password:'long secure password 123'}});assert.equal(response.status,200);return {cookie:response.headers.get('set-cookie').split(';')[0],user:(await response.json()).user};}
test('password derivation is salted and constant comparison rejects prefix',async()=>{const a=await passwordHash('secret secret','salt-a'),b=await passwordHash('secret secret','salt-b');assert.notEqual(a,b);assert.equal(a.length,64);assert.equal(constantEqual(a,a),true);assert.equal(constantEqual(a,a+'x'),false);});
test('strict GitHub claims bind repository branch workflow and audience',()=>{const c={iss:'https://token.actions.githubusercontent.com',aud:'decode-ton-esprit-journal',repository:'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine',ref:'refs/heads/main',workflow_ref:'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/.github/workflows/journal-daily.yml@refs/heads/main',event_name:'schedule',iat:1000,nbf:1000,exp:1600};assert.equal(validateClaims(c,c.repository,1100),true);for(const change of [{ref:'refs/heads/evil'},{aud:'other'},{workflow_ref:c.workflow_ref.replace('journal-daily','evil')},{exp:1050},{event_name:'pull_request'}])assert.equal(validateClaims({...c,...change},c.repository,1100),false);});
test('Stripe signatures check timestamp, signature and secret',async()=>{const secret='test-key',raw='{"id":"evt_test"}',ts=1000;const key=await crypto.subtle.importKey('raw',new TextEncoder().encode(secret),{name:'HMAC',hash:'SHA-256'},false,['sign']);const sig=Buffer.from(await crypto.subtle.sign('HMAC',key,new TextEncoder().encode(ts+'.'+raw))).toString('hex');assert.equal(await verifyStripeSignature(raw,`t=${ts},v1=${sig}`,secret,1100),true);assert.equal(await verifyStripeSignature(raw+' ',`t=${ts},v1=${sig}`,secret,1100),false);assert.equal(await verifyStripeSignature(raw,`t=${ts},v1=${sig}`,secret,1500),false);assert.equal(activeSubscription({status:'active',paid_until:1200},1100),true);assert.equal(activeSubscription({status:'past_due',paid_until:1200},1100),false);});
test('account lifecycle, cookies, origin protection, feedback isolation and deletion',async()=>{
 const env={DB:database()}; const a=await register(env,'a@example.com'),b=await register(env,'b@example.com');
 let response=await request(env,'/api/me',{cookie:a.cookie}); let me=await response.json();assert.equal(me.user.id,a.user.id);assert.equal(me.subscription.active,false);assert.equal(me.configuration.billing,false);
 response=await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Alice',consent_ai:true}});assert.equal(response.status,200);
 response=await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,originHeader:'https://evil.test',data:{display_name:'Hijacked',consent_ai:true}});assert.equal(response.status,403);
 env.DB.raw.prepare('INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice) VALUES(?,?,?,?,?,?,?,?)').run('story-a',a.user.id,'2026-10-30','Private story','Private body','Summary','["Question?"]','Exercise');
 response=await request(env,'/api/stories/story-a',{cookie:b.cookie});assert.equal(response.status,404);
 response=await request(env,'/api/stories/story-a/feedback',{method:'PUT',cookie:b.cookie,data:{resonance:5,feelings:'mine'}});assert.equal(response.status,404);
 response=await request(env,'/api/stories/story-a/feedback',{method:'PUT',cookie:a.cookie,data:{resonance:4,feelings:'Intéressant',why:'Cela me ressemble',correction:'Je ne travaille plus ici'}});assert.equal(response.status,200);
 const p=env.DB.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id);assert.equal(p.version,3);
 response=await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'Je travaille maintenant à domicile.'}});assert.equal(response.status,200);
 response=await request(env,'/api/export',{cookie:a.cookie});const exported=await response.json();assert.equal(exported.feedback.length,1);assert.equal(exported.memory.text,'Je travaille maintenant à domicile.');
 response=await request(env,'/api/billing/checkout',{method:'POST',cookie:a.cookie,data:{}});assert.equal(response.status,503);
 response=await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'wrong'}});assert.equal(response.status,401);
 response=await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});assert.equal(response.status,200);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM stories WHERE user_id=?').get(a.user.id).n,0);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM feedback').get().n,0);
 response=await request(env,'/api/me',{cookie:a.cookie});assert.equal((await response.json()).user,null);
 response=await request(env,'/api/me',{cookie:b.cookie});assert.equal((await response.json()).user.id,b.user.id);
});
test('internal generation denies missing or unsigned tokens',async()=>{const env={DB:database()};const r=await request(env,'/api/internal/jobs/claim',{method:'POST',data:{}});assert.equal(r.status,401);});

test('daily jobs enforce payment, consent, exclusive leases and context versions',async()=>{
 const env={DB:database()}; const a=await register(env,'jobs@example.com');
 await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections:[{article_id:'example-a',reason:'Ce sujet me parle',differences:''}]}});
 await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Camille',consent_ai:true}});
 assert.equal((await (await claimJobs(env)).json()).jobs.length,0);
 env.DB.raw.prepare("UPDATE subscriptions SET status='active',paid_until=? WHERE user_id=?").run(Math.floor(Date.now()/1000)+86400,a.user.id);
 let jobs=(await (await claimJobs(env)).json()).jobs;assert.equal(jobs.length,1); const j=jobs[0];
 assert.equal((await (await claimJobs(env)).json()).jobs.length,0);
 const completion={lease_token:j.lease_token,profile_version:j.profile_version,memory_version:j.memory.version,title:'Une autre possibilité',body:'Un récit fictif réaliste. '.repeat(40),summary:'Un petit changement possible.',reflection_questions:['Que ressentez-vous ?'],practice:'Observer sans conclure.',memory_proposal:'Hypothèse à confirmer.'};
 await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'Correction importante'}});
 await assert.rejects(completeJob(env,j.id,completion),e=>e.status===409);
 jobs=(await (await claimJobs(env)).json()).jobs;assert.equal(jobs.length,1);const j2=jobs[0];
 assert.notEqual(j2.lease_token,j.lease_token);assert.equal(j2.memory.text,'Correction importante');
 const done=await completeJob(env,j2.id,{...completion,lease_token:j2.lease_token,profile_version:j2.profile_version,memory_version:j2.memory.version});assert.equal(done.status,200);
 assert.equal((await (await claimJobs(env)).json()).jobs.length,0);
 const memory=env.DB.raw.prepare('SELECT * FROM memories WHERE user_id=?').get(a.user.id);assert.equal(memory.text,'Correction importante');assert.equal(memory.proposal,'Hypothèse à confirmer.');
 assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM stories WHERE user_id=?').get(a.user.id).n,1);
 await assert.rejects(completeJob(env,j2.id,{...completion,lease_token:j2.lease_token}),e=>e.status===409);
});

test('Checkout reuses durable attempts, rotates expired sessions, reconciles subscriptions and deletes Stripe customer first',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_WEBHOOK_SECRET:'webhook-test',STRIPE_PRICE_ID:'price_10',OPENROUTER_API_KEY:'test-only'};
 const a=await register(env,'billing@example.com');await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Moi',consent_ai:true}});await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections:[{article_id:'example-a',reason:'Ce sujet me parle',differences:''}]}});const originalFetch=globalThis.fetch;
 const sessions=new Map(),idempotency=new Map();let serial=0,live=null,deleteFail=false,customerDeleted=false;
 const keys=[];
 globalThis.fetch=async(url,options={})=>{
   const u=new URL(url),p=u.pathname,method=options.method||'GET';
   const result=x=>new Response(JSON.stringify(x),{status:200});
   if(p==='/v1/prices/price_10')return result({active:true,currency:'eur',unit_amount:1000,recurring:{interval:'month',interval_count:1}});
   if(p==='/v1/customers'&&method==='POST')return result({id:'cus_a'});
   if(p==='/v1/customers/cus_a'&&method==='GET')return result({id:'cus_a',deleted:customerDeleted});
   if(p==='/v1/customers/cus_a'&&method==='DELETE'){if(deleteFail)return new Response('{}',{status:503});customerDeleted=true;return result({id:'cus_a',deleted:true});}
   if(p==='/v1/subscriptions')return result({data:live?[live]:[]});
   if(p==='/v1/subscriptions/sub_a')return result(live);
   if(p==='/v1/checkout/sessions'&&method==='GET')return result({data:[...sessions.values()].filter(s=>s.status==='open')});
   if(p==='/v1/checkout/sessions'&&method==='POST'){
     const k=options.headers['Idempotency-Key'];keys.push(k);if(idempotency.has(k))return result(idempotency.get(k));
     const s={id:'cs_'+(++serial),status:'open',mode:'subscription',url:'https://checkout.stripe.com/'+serial};sessions.set(s.id,s);idempotency.set(k,s);return result(s);
   }
   if(p.startsWith('/v1/checkout/sessions/'))return result(sessions.get(p.split('/').at(-1)));
   throw new Error('Unexpected Stripe route '+method+' '+p);
 };
 try {
   const call=()=>request(env,'/api/billing/checkout',{method:'POST',cookie:a.cookie,data:{}});
   const responses=await Promise.all([call(),call()]);for(const r of responses)assert.equal(r.status,200);
   assert.equal(serial,1);assert.equal(new Set(keys).size,1);
   const firstState=env.DB.raw.prepare('SELECT * FROM subscriptions WHERE user_id=?').get(a.user.id);assert.equal(firstState.checkout_session_id,'cs_1');assert.ok(firstState.checkout_attempt);
   assert.equal((await request(env,'/api/me',{cookie:a.cookie})).status,200);
   const me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.subscription.manageable,true);
   sessions.get('cs_1').status='expired';let r=await call();assert.equal(r.status,200);assert.equal(serial,2);assert.equal(new Set(keys).size,2);
   live={id:'sub_a',customer:'cus_a',status:'active',current_period_end:Math.floor(Date.now()/1000)+86400,items:{data:[{quantity:1,price:{id:'price_10'}}]}};
   sessions.get('cs_2').status='complete';sessions.get('cs_2').subscription='sub_a';r=await call();assert.equal(r.status,409);assert.equal(serial,2);
   deleteFail=true;r=await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});assert.equal(r.status,502);assert.ok(env.DB.raw.prepare('SELECT id FROM users WHERE id=?').get(a.user.id));
   deleteFail=false;r=await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});assert.equal(r.status,200);assert.equal(customerDeleted,true);assert.equal(env.DB.raw.prepare('SELECT id FROM users WHERE id=?').get(a.user.id),undefined);
 } finally {globalThis.fetch=originalFetch;}
});

test('journal routes preserve public blog and examples do not expose full bodies',async()=>{
 const env={DB:database()};
 let r=await request(env,'/');assert.equal(r.status,302);assert.equal(r.headers.get('location'),'https://decodetonesprit.fr/');
 for(const path of ['/journal','/journal/']) {r=await request(env,path);assert.equal(r.status,200);assert.match(await r.text(),/Journal/);}
 r=await request(env,'/api/examples');const catalog=(await r.json()).examples;assert.equal(catalog.length,2);assert.equal(catalog[0].id,'example-a');assert.equal(catalog[0].body,undefined);
 r=await request(env,'/api/inspirations');assert.equal(r.status,401);
 r=await request(env,'/api/health');assert.equal((await r.json()).ai,false);
});
test('inspirations validate canonical IDs, stay private, invalidate generation and export/delete',async()=>{
 const env={DB:database()};const a=await register(env,'inspire-a@example.com'),b=await register(env,'inspire-b@example.com');
 const put=selections=>request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections}});
 for(const selections of [[{article_id:'made-up',reason:'why'}],[{article_id:'example-a',reason:' '}],[{article_id:'example-a',reason:'yes'},{article_id:'example-a',reason:'again'}],[{article_id:'example-a',reason:'x'.repeat(2001)}]]) assert.equal((await put(selections)).status,400);
 let r=await put([{article_id:'example-a',reason:'Cette réaction me parle.',differences:'Je ne suis pas Alice.'}]);assert.equal(r.status,200);
 assert.equal(env.DB.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id).version,2);
 r=await request(env,'/api/inspirations',{cookie:b.cookie});assert.deepEqual((await r.json()).selections,[]);
 r=await request(env,'/api/export',{cookie:a.cookie});assert.equal((await r.json()).selections[0].differences,'Je ne suis pas Alice.');
 await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Moi',consent_ai:true}});
 env.DB.raw.prepare("UPDATE subscriptions SET status='active',paid_until=? WHERE user_id=?").run(Math.floor(Date.now()/1000)+86400,a.user.id);
 const job=(await (await claimJobs(env)).json()).jobs[0];assert.equal(job.inspirations[0].body,'Canonical public article A');assert.equal(job.inspirations[0].reason,'Cette réaction me parle.');
 await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});
 assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM inspiration_selections').get().n,0);
});
test('scheduled generation skips without OpenRouter key and processes three distinct leases with sanitized failures',async()=>{
 const skipped=await worker.scheduled({}, {DB:{prepare(){throw new Error('Must not reserve without key');}}},{});assert.equal(skipped.status,'provider_missing');assert.equal(skipped.claimed,0);
 const env={DB:database(),OPENROUTER_API_KEY:'test-only'};
 for(let i=0;i<4;i++){const a=await register(env,'cron-'+i+'@example.com');await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections:[{article_id:'example-a',reason:'Ce sujet me parle',differences:''}]}});await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Personne '+i,consent_ai:true}});env.DB.raw.prepare("UPDATE subscriptions SET status='active',paid_until=? WHERE user_id=?").run(Math.floor(Date.now()/1000)+86400,a.user.id);}
 const ids=[];env.__generator=async job=>{ids.push(job.user_id);if(ids.length===2){const e=new Error('secret private provider failure');e.code='provider_error';throw e;}return {title:'Un récit',body:'Une fiction possible, pas une biographie. '.repeat(40),summary:'Une situation possible',reflection_questions:['Que souhaitez-vous en retenir ?'],practice:'Observer un détail.',memory_proposal:'Hypothèse provisoire'};};
 const r=await runGenerationBatch(env);assert.deepEqual(r,{status:'done',claimed:3,completed:2,failed:1});assert.equal(new Set(ids).size,3);
 const failed=env.DB.raw.prepare("SELECT * FROM generation_jobs WHERE status='pending'").get();assert.equal(failed.error_code,'provider_error');assert.ok(failed.lease_until>Math.floor(Date.now()/1000));assert.equal(JSON.stringify(failed).includes('secret private'),false);
 assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM stories').get().n,2);
});

test('generation requires inspiration and canonical D1 body; missing catalog never becomes generic story',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test-only'};const a=await register(env,'catalog@example.com');
 await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Moi',consent_ai:true}});
 env.DB.raw.prepare("UPDATE subscriptions SET status='active',paid_until=? WHERE user_id=?").run(Math.floor(Date.now()/1000)+86400,a.user.id);
 assert.equal((await (await claimJobs(env)).json()).jobs.length,0);
 await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections:[{article_id:'example-a',reason:'Je me reconnais dans ce passage.',differences:'Le reste est différent.'}]}});
 env.DB.raw.prepare('DELETE FROM public_examples WHERE id=?').run('example-a');env.__generator=()=>{throw new Error('Must not generate without canonical source');};
 const status=await runGenerationBatch(env);assert.equal(status.status,'catalog_missing');assert.equal(status.claimed,0);
 const blocked=env.DB.raw.prepare('SELECT * FROM generation_jobs WHERE user_id=?').get(a.user.id);assert.equal(blocked.error_code,'catalog_missing');assert.equal(blocked.attempts,0);
 env.DB.raw.prepare('INSERT INTO public_examples(id,body) VALUES(?,?)').run('example-a','Fresh canonical article from D1');env.DB.raw.prepare('UPDATE generation_jobs SET lease_until=0 WHERE user_id=?').run(a.user.id);
 const job=(await (await claimJobs(env)).json()).jobs[0];assert.equal(job.inspirations[0].body,'Fresh canonical article from D1');
 await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});assert.equal(env.DB.raw.prepare('SELECT body FROM public_examples WHERE id=?').get('example-a').body,'Fresh canonical article from D1');
});

test('Checkout cannot charge before AI availability, profile consent and a canonical inspiration',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_WEBHOOK_SECRET:'test',STRIPE_PRICE_ID:'price_10'};const a=await register(env,'readiness@example.com');
 const originalFetch=globalThis.fetch;let remoteCalls=0;globalThis.fetch=async()=>{remoteCalls++;throw new Error('No payment call should happen');};
 try {
   const checkout=()=>request(env,'/api/billing/checkout',{method:'POST',cookie:a.cookie,data:{}});
   assert.equal((await checkout()).status,503);env.OPENROUTER_API_KEY='test-only';assert.equal((await checkout()).status,400);
   await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Moi',consent_ai:false}});assert.equal((await checkout()).status,400);
   await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{display_name:'Moi',consent_ai:true}});assert.equal((await checkout()).status,400);
   let me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.inspirations_count,0);
   await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections:[{article_id:'example-a',reason:'Ce passage me parle.',differences:''}]}});
   me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.inspirations_count,1);
   env.DB.raw.prepare('DELETE FROM public_examples WHERE id=?').run('example-a');assert.equal((await checkout()).status,503);assert.equal(remoteCalls,0);
 } finally {globalThis.fetch=originalFetch;}
});
