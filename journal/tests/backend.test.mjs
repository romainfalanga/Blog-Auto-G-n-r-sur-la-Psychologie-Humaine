import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
const examples=Array.from({length:14},(_,i)=>({id:`category/example-${i+1}`,title:`Exemple ${i+1}`,url:`https://decodetonesprit.fr/posts/example-${i+1}/`,character:'Personnage '+i,keywords:['confiance','relations'],description:`Résumé canonique ${i+1}`}));
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<h1>Journal</h1>';").replace("import { EXAMPLES } from './examples.mjs';",'const EXAMPLES='+JSON.stringify(examples)+';');
const {default:worker,passwordHash,constantEqual,verifyStripeSignature,validateClaims,activeSubscription,claimJobs,completeJob,proxyJobLLM,runGenerationBatch}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
function database({legacy=false}={}) {
 const db=new DatabaseSync(':memory:');
 for(const migration of ['0001.sql','0002_inspirations.sql',...(!legacy?['0003_likes_portraits.sql']:[])]) db.exec(readFileSync(new URL('../migrations/'+migration,import.meta.url),'utf8'));
 for(const example of examples) db.prepare('INSERT INTO public_examples(id,body) VALUES(?,?)').run(example.id,'Canonical body: '+example.title);
 function prepare(sql){let params=[];return {bind(...v){params=v;return this;},async first(){return db.prepare(sql).get(...params)||null;},async all(){return {results:db.prepare(sql).all(...params)};},async run(){const r=db.prepare(sql).run(...params);return {meta:{changes:r.changes}};}};}
 let queue=Promise.resolve();return {raw:db,prepare,batch(statements){const task=queue.then(async()=>{db.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());db.exec('COMMIT');return r;}catch(e){db.exec('ROLLBACK');throw e;}});queue=task.catch(()=>{});return task;}};
}
const origin='https://journal.example';
const request=(env,path,{method='GET',data,cookie,originHeader=origin}={})=>worker.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),...(originHeader?{Origin:originHeader}:{}),'CF-Connecting-IP':'192.0.2.8'},body:data?JSON.stringify(data):undefined}),env);
async function register(env,email='a@example.com'){const r=await request(env,'/api/auth/register',{method:'POST',data:{email,password:'long secure password 123'}});assert.equal(r.status,200);assert.match(r.headers.get('set-cookie'),/HttpOnly; Secure; SameSite=Lax/);return {cookie:r.headers.get('set-cookie').split(';')[0],user:(await r.json()).user};}
const like=(env,a,index,reason)=>request(env,'/api/likes/'+encodeURIComponent(examples[index].id),{method:'PUT',cookie:a.cookie,data:{liked:true,...(reason!==undefined?{reason}:{})}});
async function prepareUser(env,a,count=10){await request(env,'/api/consent',{method:'PUT',cookie:a.cookie,data:{consent_ai:true}});for(let i=0;i<count;i++)assert.equal((await like(env,a,i)).status,200);}
function pay(env,a){env.DB.raw.prepare("UPDATE subscriptions SET status='active',paid_until=? WHERE user_id=?").run(Math.floor(Date.now()/1000)+86400,a.user.id);}
const claim=async env=>(await (await claimJobs(env)).json()).jobs[0];
const portrait=()=>({summary:'Des affinités à explorer, sans certitude biographique.',axes:[{label:'Se faire confiance',why:'Plusieurs thèmes appréciés concernent les choix.',confidence:0.65,evidence:[examples[0].id,examples[1].id]}],strengths:['Curiosité'],uncertainties:['Le contexte réel reste inconnu.'],fictional_character:{name:'Camille',description:'Un personnage fictif qui explore des choix possibles.'}});
const complete=(env,j,data)=>completeJob(env,j.id,{...data,lease_token:j.lease_token,source_version:j.source_version,memory_version:j.memory.version});
const story=()=>({title:'Une autre possibilité',body:'Une fiction possible, pas une biographie. '.repeat(40),summary:'Le personnage essaie une autre façon de répondre.',reflection_questions:['Qu’en pensez-vous ?'],practice:'Observer un détail.'});
async function makeStory(env,a){const p=await claim(env);assert.equal(p.kind,'portrait');await complete(env,p,{portrait:portrait()});const s=await claim(env);assert.equal(s.kind,'story');await complete(env,s,story());return s;}

test('salted password derivation and constant comparison',async()=>{const a=await passwordHash('secret secret','salt-a'),b=await passwordHash('secret secret','salt-b');assert.notEqual(a,b);assert.equal(a.length,64);assert.equal(constantEqual(a,a),true);assert.equal(constantEqual(a,a+'x'),false);});
test('OIDC exact repository, branch, workflow and allowed events',()=>{const c={iss:'https://token.actions.githubusercontent.com',aud:'decode-ton-esprit-journal',repository:'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine',ref:'refs/heads/main',workflow_ref:'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/.github/workflows/generate-articles.yml@refs/heads/main',event_name:'schedule',iat:1000,nbf:1000,exp:1600};for(const event_name of ['schedule','workflow_dispatch','push'])assert.equal(validateClaims({...c,event_name},c.repository,1100),true);for(const change of [{ref:'refs/heads/evil'},{aud:'other'},{workflow_ref:c.workflow_ref.replace('generate-articles','evil')},{exp:1050},{event_name:'pull_request'}])assert.equal(validateClaims({...c,...change},c.repository,1100),false);});
test('Stripe signature and paid entitlement verification',async()=>{const secret='test-key',raw='{"id":"evt_test"}',ts=1000;const key=await crypto.subtle.importKey('raw',new TextEncoder().encode(secret),{name:'HMAC',hash:'SHA-256'},false,['sign']);const sig=Buffer.from(await crypto.subtle.sign('HMAC',key,new TextEncoder().encode(ts+'.'+raw))).toString('hex');assert.equal(await verifyStripeSignature(raw,`t=${ts},v1=${sig}`,secret,1100),true);assert.equal(await verifyStripeSignature(raw+' ',`t=${ts},v1=${sig}`,secret,1100),false);assert.equal(await verifyStripeSignature(raw,`t=${ts},v1=${sig}`,secret,1500),false);assert.equal(activeSubscription({status:'active',paid_until:1200},1100),true);assert.equal(activeSubscription({status:'past_due',paid_until:1200},1100),false);});
test('legacy inspirations migrate losslessly without deleting source rows',()=>{const db=database({legacy:true});db.raw.prepare('INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,?,?)').run('u','legacy@example.com','hash','salt');db.raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run('u');db.raw.prepare('INSERT INTO inspiration_selections(user_id,article_id,reason,differences) VALUES(?,?,?,?)').run('u',examples[0].id,'Cela me parle','Mais ma vie diffère');db.raw.exec(readFileSync(new URL('../migrations/0003_likes_portraits.sql',import.meta.url),'utf8'));const row=db.raw.prepare('SELECT * FROM article_likes').get();assert.equal(row.reason,'Cela me parle');assert.equal(row.differences,'Mais ma vie diffère');assert.equal(db.raw.prepare('SELECT count(*) n FROM inspiration_selections').get().n,1);});
test('public routes and authentication/origin boundaries',async()=>{const env={DB:database()};let r=await request(env,'/');assert.equal(r.status,302);assert.equal(r.headers.get('location'),'https://decodetonesprit.fr/');for(const path of ['/journal','/journal/'])assert.equal((await request(env,path)).status,200);const catalog=(await (await request(env,'/api/examples')).json()).examples;assert.equal(catalog.length,14);assert.equal(catalog[0].body,undefined);assert.equal((await request(env,'/api/likes')).status,401);const a=await register(env);r=await request(env,'/api/consent',{method:'PUT',cookie:a.cookie,originHeader:'https://evil.test',data:{consent_ai:true}});assert.equal(r.status,403);assert.equal((await request(env,'/api/internal/jobs/claim',{method:'POST',data:{}})).status,401);});
test('unlimited likes and optional explanations invalidate source only on actual changes',async()=>{
 const env={DB:database()};const a=await register(env),b=await register(env,'b@example.com');await prepareUser(env,a,14);
 let me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.deepEqual(me.readiness,{likes_count:14,explained_count:0,target:10,ready:true});assert.equal(me.profile.display_name,undefined);const source=me.source_version;
 await like(env,a,0);me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.source_version,source);
 let r=await like(env,a,0,'Je me reconnais dans cette réaction.');let likes=await r.json();assert.equal(likes.count,14);assert.equal(likes.explained_count,1);assert.equal(likes.source_version,source+1);
 await like(env,a,0,'Je me reconnais dans cette réaction.');me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.source_version,source+1);
 assert.deepEqual((await (await request(env,'/api/likes',{cookie:b.cookie})).json()).likes,[]);
 assert.equal((await request(env,'/api/likes/made-up',{method:'PUT',cookie:a.cookie,data:{liked:true}})).status,404);
 r=await request(env,'/api/likes/'+encodeURIComponent(examples[0].id),{method:'PUT',cookie:a.cookie,data:{liked:false}});likes=await r.json();assert.equal(likes.count,13);assert.equal(likes.source_version,source+2);
 const p=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(p.status,'collecting');assert.equal(p.portrait,null);
});
test('backward inspiration API has no three-item cap or explanation requirement',async()=>{const env={DB:database()};const a=await register(env);const selections=examples.slice(0,12).map(e=>({article_id:e.id}));let r=await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections}});assert.equal(r.status,200);const result=await r.json();assert.equal(result.count,12);const v=result.source_version;r=await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections}});assert.equal((await r.json()).source_version,v);assert.equal((await (await request(env,'/api/inspirations',{cookie:a.cookie})).json()).selections.length,12);});
test('ten likes and no questionnaire produce portrait first, then one daily story',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test-only'};const a=await register(env);await prepareUser(env,a,9);assert.equal(await claim(env),undefined);await like(env,a,9);pay(env,a);
 env.DB.raw.prepare('UPDATE profiles SET data=? WHERE user_id=?').run(JSON.stringify({display_name:'DO NOT SEND OLD PROFILE',occupation:'PRIVATE LEGACY'}),a.user.id);
 const p=await claim(env);assert.equal(p.kind,'portrait');assert.equal(p.likes.length,10);assert.equal(p.likes[0].summary,'Résumé canonique 1');assert.equal(p.profile,undefined);assert.equal(p.history,undefined);assert.equal(JSON.stringify(p).includes('PRIVATE LEGACY'),false);assert.equal(await claim(env),undefined);
 assert.equal((await (await request(env,'/api/portrait',{cookie:a.cookie})).json()).status,'updating');await complete(env,p,{portrait:portrait()});
 const ready=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(ready.status,'ready');assert.equal(ready.source_version,ready.analyzed_version);
 const s=await claim(env);assert.equal(s.kind,'story');assert.equal(s.inspirations.length,3);assert.match(s.inspirations[0].body,/Canonical body/);assert.equal(s.portrait.fictional_character.name,'Camille');await complete(env,s,{...story(),memory_proposal:'Fiction must never become biography.'});
 assert.equal(await claim(env),undefined);assert.equal(env.DB.raw.prepare('SELECT proposal FROM memories WHERE user_id=?').get(a.user.id).proposal,'');assert.equal(env.DB.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id).version,p.source_version);
});
test('portrait completion rejects stale versions and evidence from another source',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test-only'};const a=await register(env);await prepareUser(env,a);pay(env,a);const stale=await claim(env);await like(env,a,0,'Nouveau commentaire');await assert.rejects(complete(env,stale,{portrait:portrait()}),e=>e.status===409);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM user_portraits').get().n,0);
 const fresh=await claim(env);assert.ok(fresh.source_version>stale.source_version);const invalid=portrait();invalid.axes[0].evidence=[examples[13].id];await assert.rejects(complete(env,fresh,{portrait:invalid}),e=>e.status===400);await complete(env,fresh,{portrait:portrait()});
 const s=await claim(env);await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'Je souhaite corriger cette interprétation.'}});await assert.rejects(complete(env,s,story()),e=>e.status===409);const next=await claim(env);assert.equal(next.kind,'portrait');assert.equal(next.memory.text,'Je souhaite corriger cette interprétation.');assert.equal(next.portrait_previous.summary,portrait().summary);
});
test('private negative feedback changes portrait source, never leaks between accounts',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test-only'};const a=await register(env),b=await register(env,'b@example.com');await prepareUser(env,a);pay(env,a);const s=await makeStory(env,a);let r=await request(env,'/api/stories/'+s.id,{cookie:b.cookie});assert.equal(r.status,404);
 r=await request(env,'/api/stories/'+s.id+'/feedback',{method:'PUT',cookie:b.cookie,data:{resonance:1}});assert.equal(r.status,404);
 r=await request(env,'/api/stories/'+s.id+'/feedback',{method:'PUT',cookie:a.cookie,data:{resonance:1,why:'Cela ne me ressemble pas.',correction:'Je ne vis pas cette situation.'}});assert.equal(r.status,200);const updated=await r.json();assert.ok(updated.source_version>s.source_version);
 const job=await claim(env);assert.equal(job.kind,'portrait');assert.equal(job.feedback[0].resonance,1);assert.equal(job.feedback[0].correction,'Je ne vis pas cette situation.');assert.equal(job.history,undefined);assert.equal(job.feedback[0].body,undefined);assert.equal(job.feedback[0].fictional_source_summary,story().summary);
 const feedbackPortrait=portrait();feedbackPortrait.axes[0].evidence=['story:'+s.id];await complete(env,job,{portrait:feedbackPortrait});
 const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();assert.equal(exported.likes.length,10);assert.equal(exported.feedback.length,1);assert.ok(exported.portrait.portrait);
 await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});for(const table of ['article_likes','stories','feedback','user_portraits','journal_jobs'])assert.equal(env.DB.raw.prepare(`SELECT count(*) n FROM ${table}`).get().n,0);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM public_examples').get().n,14);
});
test('LLM proxy gates model, entitlement, stale source and twenty-call budget without exposing key',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'private-test-secret',OPENROUTER_MODEL:'approved/model'};const a=await register(env);await prepareUser(env,a);pay(env,a);const j=await claim(env),data={lease_token:j.lease_token,purpose:'analyst',system_prompt:'System',user_prompt:'Prompt',max_tokens:1000};let calls=0;const originalFetch=globalThis.fetch;
 globalThis.fetch=async(url,opts)=>{calls++;assert.equal(url,'https://openrouter.ai/api/v1/chat/completions');assert.equal(opts.headers.Authorization,'Bearer private-test-secret');assert.equal(JSON.parse(opts.body).model,'approved/model');return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:'Result'}}]}),{status:200});};
 try {
  await assert.rejects(proxyJobLLM(env,j.id,{...data,model:'expensive/other'}),e=>e.status===400);await assert.rejects(proxyJobLLM(env,j.id,{...data,max_tokens:9000}),e=>e.status===400);assert.equal(calls,0);
  for(let i=0;i<20;i++){const r=await proxyJobLLM(env,j.id,data);assert.deepEqual(await r.json(),{content:'Result'});}await assert.rejects(proxyJobLLM(env,j.id,data),e=>e.status===429);assert.equal(calls,20);
  await like(env,a,0,'Nouvel élément');await assert.rejects(proxyJobLLM(env,j.id,data),e=>e.status===409);assert.equal(calls,20);
 } finally {globalThis.fetch=originalFetch;}
});
test('Cloudflare scheduler is inert and GitHub claims skip absent provider',async()=>{const env={DB:{prepare(){throw new Error('No database work');}}};assert.deepEqual(await runGenerationBatch(env),{status:'workflow-managed',claimed:0,completed:0,failed:0});assert.equal((await worker.scheduled({},env,{})).status,'workflow-managed');assert.deepEqual(await (await claimJobs(env)).json(),{jobs:[],status:'provider_missing'});});

test('Checkout reuses durable attempts, rotates expired sessions, reconciles subscriptions and deletes Stripe customer first',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_WEBHOOK_SECRET:'webhook-test',STRIPE_PRICE_ID:'price_10',OPENROUTER_API_KEY:'test-only'};
 const a=await register(env,'billing@example.com');await prepareUser(env,a);const originalFetch=globalThis.fetch;
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


test('Checkout requires ten likes, consent and AI but no name or written explanation',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_WEBHOOK_SECRET:'test',STRIPE_PRICE_ID:'price_10'};const a=await register(env,'readiness@example.com');let calls=0;const originalFetch=globalThis.fetch;
 globalThis.fetch=async()=>{calls++;return new Response('{}',{status:503});};
 try {const checkout=()=>request(env,'/api/billing/checkout',{method:'POST',cookie:a.cookie,data:{}});assert.equal((await checkout()).status,503);env.OPENROUTER_API_KEY='test';assert.equal((await checkout()).status,400);await prepareUser(env,a,9);assert.equal((await checkout()).status,400);assert.equal(calls,0);await like(env,a,9);assert.equal((await checkout()).status,502);assert.equal(calls,1);const me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.readiness.ready,true);assert.equal(me.readiness.explained_count,0);assert.equal(me.profile.display_name,undefined);} finally {globalThis.fetch=originalFetch;}
});

test('paid users keep generation after reducing likes below ten, but zero likes pauses it',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'};const a=await register(env,'paid-nine@example.com');await prepareUser(env,a);pay(env,a);
 const initial=await claim(env);await complete(env,initial,{portrait:portrait()});
 await request(env,'/api/likes/'+encodeURIComponent(examples[9].id),{method:'PUT',cookie:a.cookie,data:{liked:false}});
 const state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.status,'queued');
 const updated=await claim(env);assert.equal(updated.kind,'portrait');assert.equal(updated.likes.length,9);const revised=portrait();revised.fictional_character={name:'Un autre prénom',description:'Le personnage explore désormais une nouvelle situation.'};await complete(env,updated,{portrait:revised});const stable=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(stable.portrait.fictional_character.name,'Camille');assert.equal(stable.portrait.fictional_character.description,revised.fictional_character.description);
 const daily=await claim(env);assert.equal(daily.kind,'story');await complete(env,daily,story());
 await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections:[]}});
 assert.equal((await (await request(env,'/api/portrait',{cookie:a.cookie})).json()).status,'collecting');assert.equal(await claim(env),undefined);
});

test('likes response collection and source_version belong to one snapshot during concurrent mutation',async()=>{
 const db=database(),env={DB:db};const a=await register(env,'snapshot@example.com');await like(env,a,0);
 const before=db.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id).version;
 const prepare=db.prepare.bind(db);let injected=false;
 db.prepare=sql=>{
  const statement=prepare(sql);
  if(sql.includes('AS likes_json')) {
   const first=statement.first.bind(statement);statement.first=async()=>{const snapshot=await first();if(!injected){injected=true;db.raw.prepare('INSERT INTO article_likes(user_id,article_id,reason) VALUES(?,?,?)').run(a.user.id,examples[1].id,'A concurrent explanation');}return snapshot;};
  }
  return statement;
 };
 const response=await request(env,'/api/likes',{cookie:a.cookie});const snapshot=await response.json();assert.equal(snapshot.count,1);assert.equal(snapshot.source_version,before);
 assert.equal(db.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id).version,before+1);
 const newer=await (await request(env,'/api/likes',{cookie:a.cookie})).json();assert.equal(newer.count,2);assert.equal(newer.source_version,before+1);
});

test('exhausted portrait versions retry automatically after a bounded 24-hour cooldown',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'};const a=await register(env,'retry@example.com');await prepareUser(env,a);pay(env,a);const initial=await claim(env);
 env.DB.raw.prepare("UPDATE journal_jobs SET status='pending',attempts=3,provider_calls=20,lease_until=0 WHERE id=?").run(initial.id);
 assert.equal(await claim(env),undefined);let state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.status,'error');assert.ok(state.retry_after>Math.floor(Date.now()/1000));
 env.DB.raw.prepare("UPDATE journal_jobs SET created_at=datetime('now','-25 hours') WHERE id=?").run(initial.id);
 const retry=await claim(env);assert.equal(retry.id,initial.id);assert.equal(retry.source_version,initial.source_version);assert.notEqual(retry.lease_token,initial.lease_token);
 const budget=env.DB.raw.prepare('SELECT attempts,provider_calls,created_at FROM journal_jobs WHERE id=?').get(initial.id);assert.equal(budget.attempts,1);assert.equal(budget.provider_calls,0);assert.ok(Date.parse(budget.created_at+'Z')>Date.now()-60000);
 state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.status,'updating');assert.equal(state.retry_after,null);await complete(env,retry,{portrait:portrait()});assert.equal((await (await request(env,'/api/portrait',{cookie:a.cookie})).json()).status,'ready');
});

test('internal endpoints verify a real RSA GitHub OIDC signature and reject forged tokens',async()=>{
 const pair=await crypto.subtle.generateKey({name:'RSASSA-PKCS1-v1_5',modulusLength:2048,publicExponent:new Uint8Array([1,0,1]),hash:'SHA-256'},true,['sign','verify']);
 const jwk=await crypto.subtle.exportKey('jwk',pair.publicKey);jwk.kid='test-github-key';jwk.alg='RS256';
 const t=Math.floor(Date.now()/1000),repo='romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine';
 const b64=value=>Buffer.from(JSON.stringify(value)).toString('base64url');
 const signing=b64({alg:'RS256',kid:jwk.kid})+'.'+b64({iss:'https://token.actions.githubusercontent.com',aud:'decode-ton-esprit-journal',repository:repo,ref:'refs/heads/main',workflow_ref:repo+'/.github/workflows/generate-articles.yml@refs/heads/main',event_name:'push',iat:t,nbf:t,exp:t+600});
 const signature=Buffer.from(await crypto.subtle.sign('RSASSA-PKCS1-v1_5',pair.privateKey,new TextEncoder().encode(signing))).toString('base64url');
 const originalFetch=globalThis.fetch;globalThis.fetch=async url=>{assert.equal(url,'https://token.actions.githubusercontent.com/.well-known/jwks');return new Response(JSON.stringify({keys:[jwk]}),{status:200});};
 try {
  const call=token=>worker.fetch(new Request(origin+'/api/internal/jobs/claim',{method:'POST',headers:{Authorization:'Bearer '+token}}),{DB:database()});
  let r=await call(signing+'.'+signature);assert.equal(r.status,200);assert.deepEqual(await r.json(),{jobs:[],status:'provider_missing'});
  r=await call(signing+'.'+(signature[0]==='A'?'B':'A')+signature.slice(1));assert.equal(r.status,401);
 } finally {globalThis.fetch=originalFetch;}
});


test('four thousand accented characters fit the documented optional reason with real UTF-8 content length',async()=>{
 const env={DB:database()}; const a=await register(env,'unicode@example.com'); const payload=JSON.stringify({liked:true,reason:'é'.repeat(4000)});
 const response=await worker.fetch(new Request(origin+'/api/likes/'+encodeURIComponent(examples[0].id),{method:'PUT',headers:{'Content-Type':'application/json','Content-Length':String(Buffer.byteLength(payload)),'Origin':origin,'Cookie':a.cookie},body:payload}),env);
 assert.equal(response.status,200);assert.equal((await response.json()).likes[0].reason.length,4000);
});
