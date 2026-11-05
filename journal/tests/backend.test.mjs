import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { createRecommender } from '../src/recommendations.mjs';
import {loadContext} from '../src/private-context.mjs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
const examples=Array.from({length:14},(_,i)=>({id:`category/example-${i+1}`,title:`Exemple ${i+1}`,url:`https://decodetonesprit.fr/posts/example-${i+1}/`,character:'Personnage '+i,keywords:['confiance','relations'],description:`Résumé canonique ${i+1}`}));
const recommendationContexts=[];
// Use the real recommendation engine while recording its authenticated input.
globalThis.__backendRecommendationFactory=items=>{
 const engine=createRecommender(items);
 return {recommend(context,options){recommendationContexts.push(structuredClone(context));return engine.recommend(context,options);},provisionalPortrait(context){recommendationContexts.push(structuredClone(context));return engine.provisionalPortrait(context);}};
};
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<title>Mes histoires</title><main>App shell</main>';").replace("import { EXAMPLES } from './examples.mjs';",'const EXAMPLES='+JSON.stringify(examples)+';').replace("import { createRecommender } from './recommendations.mjs';",'const createRecommender=globalThis.__backendRecommendationFactory;').replace("'./chat-backend.mjs'",JSON.stringify(new URL('../src/chat-backend.mjs',import.meta.url).href)).replace("'./entitlements.mjs'",JSON.stringify(new URL('../src/entitlements.mjs',import.meta.url).href)).replace("'./personal-backend.mjs'",JSON.stringify(new URL('../src/personal-backend.mjs',import.meta.url).href));
const {default:worker,passwordHash,constantEqual,verifyStripeSignature,validateClaims,activeSubscription,claimJobs,completeJob,proxyJobLLM,failJob,runGenerationBatch,refreshUserPortrait}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
function database({legacy=false,v2=false,beforeRetirement=false}={}) {
 const db=new DatabaseSync(':memory:');
 for(const migration of ['0001.sql','0002_inspirations.sql',...(!legacy?['0003_likes_portraits.sql','0004_recommendations_free_portraits.sql','0005_personal_context_characters.sql','0006_recommendation_exposures.sql',...(!v2?['0007_self_profile_single_character.sql']:[]),'0008_billing_trial.sql','0009_private_companion.sql','0010_requested_focus.sql','0011_native_portrait_candidate.sql','0012_ambre_portrait_corrections.sql','0013_billing_cancellation_attempts.sql',...(!v2&&!beforeRetirement?['0014_retire_personal_stories.sql']:[])]:[])]) db.exec(readFileSync(new URL('../migrations/'+migration,import.meta.url),'utf8'));
 for(const example of examples) db.prepare('INSERT INTO public_examples(id,body) VALUES(?,?)').run(example.id,'Canonical body: '+example.title);
 function prepare(sql){let params=[];return {bind(...v){params=v;return this;},async first(){return db.prepare(sql).get(...params)||null;},async all(){return {results:db.prepare(sql).all(...params)};},async run(){const r=db.prepare(sql).run(...params);return {meta:{changes:r.changes}};}};}
 let queue=Promise.resolve();return {raw:db,prepare,batch(statements){const task=queue.then(async()=>{db.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());db.exec('COMMIT');return r;}catch(e){db.exec('ROLLBACK');throw e;}});queue=task.catch(()=>{});return task;}};
}
const origin='https://journal.example';
const request=(env,path,{method='GET',data,cookie,originHeader=origin,context}={})=>worker.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),...(originHeader?{Origin:originHeader}:{}),'CF-Connecting-IP':'192.0.2.8'},body:data?JSON.stringify(data):undefined}),env,context);
async function register(env,email='a@example.com'){const r=await request(env,'/api/auth/register',{method:'POST',data:{email,password:'long secure password 123'}});assert.equal(r.status,200);assert.match(r.headers.get('set-cookie'),/HttpOnly; Secure; SameSite=Lax/);return {cookie:r.headers.get('set-cookie').split(';')[0],user:(await r.json()).user};}
const like=(env,a,index,reason)=>request(env,'/api/likes/'+encodeURIComponent(examples[index].id),{method:'PUT',cookie:a.cookie,data:{liked:true,...(reason!==undefined?{reason}:{})}});
async function prepareUser(env,a,count=10){await request(env,'/api/consent',{method:'PUT',cookie:a.cookie,data:{consent_ai:true}});for(let i=0;i<count;i++)assert.equal((await like(env,a,i)).status,200);}
function pay(env,a){env.DB.raw.prepare("UPDATE subscriptions SET status='active',paid_until=? WHERE user_id=?").run(Math.floor(Date.now()/1000)+86400,a.user.id);}
const claim=async env=>(await (await claimJobs(env)).json()).jobs[0];
const complete=(env,j,data)=>completeJob(env,j.id,{...data,lease_token:j.lease_token,source_version:j.source_version,memory_version:j.memory.version});

test('salted password derivation and constant comparison',async()=>{const a=await passwordHash('secret secret','salt-a'),b=await passwordHash('secret secret','salt-b');assert.notEqual(a,b);assert.equal(a.length,64);assert.equal(constantEqual(a,a),true);assert.equal(constantEqual(a,a+'x'),false);});

test('OIDC exact repository, branch, workflow and allowed events',()=>{const c={iss:'https://token.actions.githubusercontent.com',aud:'decode-ton-esprit-journal',repository:'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine',ref:'refs/heads/main',workflow_ref:'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine/.github/workflows/generate-articles.yml@refs/heads/main',event_name:'schedule',iat:1000,nbf:1000,exp:1600};for(const event_name of ['schedule','workflow_dispatch','push'])assert.equal(validateClaims({...c,event_name},c.repository,1100),true);for(const change of [{ref:'refs/heads/evil'},{aud:'other'},{workflow_ref:c.workflow_ref.replace('generate-articles','evil')},{exp:1050},{event_name:'pull_request'}])assert.equal(validateClaims({...c,...change},c.repository,1100),false);});

test('Stripe signature and paid entitlement verification',async()=>{const secret='test-key',raw='{"id":"evt_test"}',ts=1000;const key=await crypto.subtle.importKey('raw',new TextEncoder().encode(secret),{name:'HMAC',hash:'SHA-256'},false,['sign']);const sig=Buffer.from(await crypto.subtle.sign('HMAC',key,new TextEncoder().encode(ts+'.'+raw))).toString('hex');assert.equal(await verifyStripeSignature(raw,`t=${ts},v1=${sig}`,secret,1100),true);assert.equal(await verifyStripeSignature(raw+' ',`t=${ts},v1=${sig}`,secret,1100),false);assert.equal(await verifyStripeSignature(raw,`t=${ts},v1=${sig}`,secret,1500),false);assert.equal(activeSubscription({status:'active',paid_until:1200},1100),true);assert.equal(activeSubscription({status:'past_due',paid_until:1200},1100),false);});

test('legacy inspirations migrate losslessly without deleting source rows',()=>{const db=database({legacy:true});db.raw.prepare('INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,?,?)').run('u','legacy@example.com','hash','salt');db.raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run('u');db.raw.prepare('INSERT INTO inspiration_selections(user_id,article_id,reason,differences) VALUES(?,?,?,?)').run('u',examples[0].id,'Cela me parle','Mais ma vie diffère');db.raw.exec(readFileSync(new URL('../migrations/0003_likes_portraits.sql',import.meta.url),'utf8'));const row=db.raw.prepare('SELECT * FROM article_likes').get();assert.equal(row.reason,'Cela me parle');assert.equal(row.differences,'Mais ma vie diffère');assert.equal(db.raw.prepare('SELECT count(*) n FROM inspiration_selections').get().n,1);});

test('unlimited likes and optional explanations invalidate source only on actual changes',async()=>{
 const env={DB:database()};const a=await register(env),b=await register(env,'b@example.com');await prepareUser(env,a,14);
 let me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.deepEqual(me.readiness,{likes_count:14,explained_count:0});assert.equal(me.profile.display_name,undefined);const source=me.source_version;
 await like(env,a,0);me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.source_version,source);
 let r=await like(env,a,0,'Je me reconnais dans cette réaction.');let likes=await r.json();assert.equal(likes.count,14);assert.equal(likes.explained_count,1);assert.equal(likes.source_version,source+1);
 await like(env,a,0,'Je me reconnais dans cette réaction.');me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.source_version,source+1);
 assert.deepEqual((await (await request(env,'/api/likes',{cookie:b.cookie})).json()).likes,[]);
 assert.equal((await request(env,'/api/likes/made-up',{method:'PUT',cookie:a.cookie,data:{liked:true}})).status,404);
 r=await request(env,'/api/likes/'+encodeURIComponent(examples[0].id),{method:'PUT',cookie:a.cookie,data:{liked:false}});likes=await r.json();assert.equal(likes.count,13);assert.equal(likes.source_version,source+2);
 const p=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(p.status,'queued');assert.equal(p.portrait,null);assert.ok(p.provisional_portrait);
});

test('backward inspiration API has no three-item cap or explanation requirement',async()=>{const env={DB:database()};const a=await register(env);const selections=examples.slice(0,12).map(e=>({article_id:e.id}));let r=await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections}});assert.equal(r.status,200);const result=await r.json();assert.equal(result.count,12);const v=result.source_version;r=await request(env,'/api/inspirations',{method:'PUT',cookie:a.cookie,data:{selections}});assert.equal((await r.json()).source_version,v);assert.equal((await (await request(env,'/api/inspirations',{cookie:a.cookie})).json()).selections.length,12);});

test('Cloudflare scheduler is inert and GitHub claims skip absent provider',async()=>{const env={DB:{prepare(){throw new Error('No database work');}}};assert.deepEqual(await runGenerationBatch(env),{status:'workflow-managed',claimed:0,completed:0,failed:0});assert.equal((await worker.scheduled({},env,{})).status,'workflow-managed');assert.deepEqual(await (await claimJobs(env)).json(),{jobs:[],status:'provider_missing'});});


test('Checkout reuses durable attempts, rotates expired sessions, reconciles subscriptions and deletes Stripe customer first',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_WEBHOOK_SECRET:'webhook-test',STRIPE_PRICE_ID:'price_29',OPENROUTER_API_KEY:'test-only'};
 const a=await register(env,'billing@example.com');await prepareUser(env,a);const originalFetch=globalThis.fetch;
 const sessions=new Map(),idempotency=new Map();let serial=0,live=null,deleteFail=false,customerDeleted=false;
 const keys=[];
 globalThis.fetch=async(url,options={})=>{
   const u=new URL(url),p=u.pathname,method=options.method||'GET';
   const result=x=>new Response(JSON.stringify(x),{status:200});
   if(p==='/v1/prices/price_29')return result({active:true,currency:'eur',unit_amount:2900,recurring:{interval:'month',interval_count:1}});
   if(p==='/v1/customers'&&method==='POST')return result({id:'cus_a'});
   if(p==='/v1/customers/cus_a'&&method==='GET')return result({id:'cus_a',deleted:customerDeleted});
   if(p==='/v1/customers/cus_a'&&method==='DELETE'){if(deleteFail)return new Response('{}',{status:503});customerDeleted=true;return result({id:'cus_a',deleted:true});}
   if(p==='/v1/subscriptions')return result({data:live?[live]:[]});
   if(p==='/v1/subscriptions/sub_a')return result(live);
   if(p==='/v1/checkout/sessions'&&method==='GET')return result({data:[...sessions.values()].filter(s=>s.status==='open')});
   if(p==='/v1/checkout/sessions'&&method==='POST'){
     const fields=new URLSearchParams(options.body);assert.equal(fields.get('payment_method_collection'),'always');assert.equal(fields.get('subscription_data[trial_period_days]'),'7');assert.equal(fields.get('subscription_data[trial_settings][end_behavior][missing_payment_method]'),'cancel');assert.equal(fields.get('success_url'),origin+'/mon-compte/?payment=success');assert.equal(fields.get('cancel_url'),origin+'/mon-compte/?payment=cancelled');
     const k=options.headers['Idempotency-Key'];keys.push(k);if(idempotency.has(k))return result(idempotency.get(k));
     const s={id:'cs_'+(++serial),status:'open',mode:'subscription',metadata:{offer:'v4-29-trial7'},url:'https://checkout.stripe.com/'+serial};sessions.set(s.id,s);idempotency.set(k,s);return result(s);
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
   live={id:'sub_a',customer:'cus_a',status:'active',current_period_end:Math.floor(Date.now()/1000)+86400,items:{data:[{quantity:1,price:{id:'price_29'}}]}};
   sessions.get('cs_2').status='complete';sessions.get('cs_2').subscription='sub_a';r=await call();assert.equal(r.status,409);assert.equal(serial,2);
   deleteFail=true;r=await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});assert.equal(r.status,502);assert.ok(env.DB.raw.prepare('SELECT id FROM users WHERE id=?').get(a.user.id));
   deleteFail=false;r=await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});assert.equal(r.status,200);assert.equal(customerDeleted,true);assert.equal(env.DB.raw.prepare('SELECT id FROM users WHERE id=?').get(a.user.id),undefined);
 } finally {globalThis.fetch=originalFetch;}
});



test('Checkout needs configured AI and billing but no likes or profile gate',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_WEBHOOK_SECRET:'test',STRIPE_PRICE_ID:'price_29'};const a=await register(env,'readiness@example.com');let calls=0;const originalFetch=globalThis.fetch;
 globalThis.fetch=async()=>{calls++;return new Response('{}',{status:503});};
 try {const checkout=()=>request(env,'/api/billing/checkout',{method:'POST',cookie:a.cookie,data:{}});assert.equal((await checkout()).status,503);assert.equal(calls,0);env.OPENROUTER_API_KEY='test';assert.equal((await checkout()).status,502);assert.equal(calls,1);} finally {globalThis.fetch=originalFetch;}
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


test('former collection page redirects to portrait collection and preserves all intent',async()=>{
 const env={PUBLIC_ORIGIN:origin};
 for(const path of ['/mes-articles','/mes-articles/'])for(const method of ['GET','HEAD']){
 const query='?article=posts%2Fexample&character=Camille&like=1&tab=likes';
 const response=await request(env,path+query,{method});assert.equal(response.status,302);
 const target=new URL(response.headers.get('location'));assert.equal(target.pathname,'/mon-portrait/');assert.equal(target.search,query);assert.equal(target.hash,'#mes-articles');
 assert.equal((await request(env,target.pathname+target.search)).status,200);
 }
});


test('like explanations preserve independent reason and differences, version actual edits and isolate users',async()=>{
 const env={DB:database()},a=await register(env),b=await register(env,'different@example.com');
 const path='/api/likes/'+encodeURIComponent(examples[0].id);
 const put=async(data,cookie=a.cookie)=>request(env,path,{method:'PUT',cookie,data:{liked:true,...data}});
 let r=await (await put({reason:'Une ressemblance',differences:'Une différence'})).json();const version=r.source_version;
 assert.equal(r.likes[0].reason,'Une ressemblance');assert.equal(r.likes[0].differences,'Une différence');
 r=await (await put({reason:'Une ressemblance'})).json();assert.equal(r.source_version,version);assert.equal(r.likes[0].differences,'Une différence');
 r=await (await put({differences:'Autre différence'})).json();assert.equal(r.source_version,version+1);assert.equal(r.likes[0].reason,'Une ressemblance');
 r=await (await put({})).json();assert.equal(r.source_version,version+1);assert.equal(r.likes[0].differences,'Autre différence');
 r=await (await put({differences:''})).json();assert.equal(r.likes[0].differences,'');assert.equal(r.likes[0].reason,'Une ressemblance');assert.equal(r.source_version,version+2);
 assert.equal((await put({differences:'é'.repeat(4001)})).status,400);
 r=await (await put({reason:'à'.repeat(4000),differences:'é'.repeat(4000)})).json();assert.equal(r.likes[0].reason.length,4000);assert.equal(r.likes[0].differences.length,4000);
 assert.deepEqual((await (await request(env,'/api/likes',{cookie:b.cookie})).json()).likes,[]);
 await put({reason:'Compte B',differences:'Autre vécu'},b.cookie);
 assert.equal((await (await request(env,'/api/likes',{cookie:a.cookie})).json()).likes[0].differences,'é'.repeat(4000));
});



test('differences alone count as an explanation in likes and account readiness',async()=>{
 const env={DB:database()},a=await register(env);
 const path='/api/likes/'+encodeURIComponent(examples[0].id);
 const put=async differences=>(await (await request(env,path,{method:'PUT',cookie:a.cookie,data:{liked:true,differences}})).json());
 let result=await put('Ma situation est différente.');assert.equal(result.likes[0].reason,'');assert.equal(result.explained_count,1);
 let me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.readiness.explained_count,1);
 result=await put('   ');assert.equal(result.explained_count,0);
 me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.readiness.explained_count,0);
});

const {parisWeek}=await import('../src/personal-backend.mjs');
const versionOf=(env,a)=>env.DB.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id).version;
async function addEvent(env,a,extra={}){const r=await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),title:'Un nouveau départ',kind:'positive',description:'J’ai repris confiance après avoir retrouvé mes amis.',date_precision:'year',date_start:'2020',...extra}});assert.equal(r.status,201,JSON.stringify(await r.clone().json()));return (await r.json()).event;}
async function addPerson(env,a,extra={}){const r=await request(env,'/api/people',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),display_name:'Mon proche',role:'friend',relationship_text:'Cette amitié me rassure quand je doute.',...extra}});assert.equal(r.status,201,JSON.stringify(await r.clone().json()));return (await r.json()).person;}
function portraitV2(job,{pending=[]}={}){const sources=job.sources.filter(s=>!pending.includes(s.id)),refs=sources.slice(0,1).map(s=>s.id),explicit=sources.find(s=>s.explicit);return {schema_version:2,psychological_map:[],summary:'Des pistes à explorer depuis les éléments exprimés.',summary_evidence:refs,axes:[{label:'Confiance et liens',why:'Une piste exprimée ou un intérêt de lecture.',claim_type:sources[0].explicit?'hypothesis':'interest',confidence:sources[0].explicit?0.5:0.2,evidence:refs}],strengths:explicit?[{text:'Une ressource évoquée dans les liens.',evidence:[explicit.id]}]:[],uncertainties:['Ces éléments ne définissent pas toute votre vie.'],citations:sources.map(s=>({source_id:s.id,revision:s.revision,quote:s.text.slice(0,Math.min(120,s.text.length))})),coverage:{processed:sources.map(s=>s.id),pending}};}
async function readyInputs(env,a,likes=10){await prepareUser(env,a,likes);const current=await (await request(env,'/api/self-profile',{cookie:a.cookie})).json();assert.equal((await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:current.revision,self_description:'Je souhaite apprendre à poser mes limites.',recurring_patterns:'Je réponds oui trop vite quand un proche me sollicite.'}})).status,200);}

test('V2 people/events CRUD is private, idempotent, bounded, versioned and keeps events when person deleted',async()=>{
 const env={DB:database()},a=await register(env),b=await register(env,'b@example.com');const key=crypto.randomUUID(),data={creation_id:key,display_name:'Pseudonyme',role:'father',relationship_text:'Une relation apaisée aujourd’hui.'};let r=await request(env,'/api/people',{method:'POST',cookie:a.cookie,data});assert.equal(r.status,201);const p=(await r.json()).person,v=versionOf(env,a);r=await request(env,'/api/people',{method:'POST',cookie:a.cookie,data});assert.equal((await r.json()).person.id,p.id);assert.equal(versionOf(env,a),v);
 r=await request(env,'/api/people',{method:'POST',cookie:a.cookie,data:{...data,display_name:'Autre'}});assert.equal(r.status,409);assert.equal((await r.json()).code,'idempotency_conflict');
 r=await request(env,'/api/people/'+p.id,{method:'PATCH',cookie:a.cookie,data:{revision:1,relationship_text:data.relationship_text}});assert.equal(r.status,200);assert.equal(versionOf(env,a),v);
 r=await request(env,'/api/people/'+p.id,{method:'PATCH',cookie:a.cookie,data:{revision:1,relationship_text:'Un désaccord récent mais situé.'}});assert.equal((await r.json()).person.revision,2);assert.equal(versionOf(env,a),v+1);
 assert.equal((await request(env,'/api/people/'+p.id,{method:'PATCH',cookie:a.cookie,data:{revision:1,display_name:'Stale'}})).status,409);
 assert.equal((await request(env,'/api/people/'+p.id,{cookie:b.cookie})).status,404);
 const foreign=await addPerson(env,b);r=await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),title:'Un souvenir',description:'La relation était difficile.',people_ids:[foreign.id]}});assert.equal(r.status,400);
 const e=await addEvent(env,a,{people_ids:[p.id],date_precision:'range',date_start:'2003',date_end:'2005-04'});assert.deepEqual(e.people_ids,[p.id]);r=await request(env,'/api/people/'+p.id,{method:'DELETE',cookie:a.cookie,data:{revision:2}});assert.equal(r.status,200);r=await request(env,'/api/life-events/'+e.id,{cookie:a.cookie});assert.equal(r.status,200);assert.deepEqual((await r.json()).event.people_ids,[]);const afterDelete=versionOf(env,a);r=await request(env,'/api/life-events/'+e.id,{method:'PATCH',cookie:a.cookie,data:{revision:1}});assert.equal(r.status,200);assert.equal(versionOf(env,a),afterDelete,'Removing an association must not turn a later identical save into a mutation');
 assert.equal((await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),title:'Date impossible',description:'Test',date_precision:'day',date_start:'2024-02-31'}})).status,400);
 assert.equal((await request(env,'/api/people',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),display_name:'x'.repeat(81)}})).status,400);
 r=await request(env,'/api/people',{method:'POST',cookie:a.cookie,data});assert.equal(r.status,200);assert.equal((await r.json()).person,null,'Deleted creation cannot be resurrected by stale retry');
});

test('two editors and double creation cannot overwrite revisions or increment versions twice',async()=>{const env={DB:database()},a=await register(env);const key=crypto.randomUUID(),data={creation_id:key,display_name:'Alias',role:'other',relationship_text:'Un contexte.'},v=versionOf(env,a);const results=await Promise.all([1,2].map(()=>request(env,'/api/people',{method:'POST',cookie:a.cookie,data})));const rows=await Promise.all(results.map(r=>r.json()));assert.equal(rows[0].person.id,rows[1].person.id);assert.equal(versionOf(env,a),v+1);const id=rows[0].person.id,edits=await Promise.all(['A','B'].map(relationship_text=>request(env,'/api/people/'+id,{method:'PATCH',cookie:a.cookie,data:{revision:1,relationship_text}})));assert.deepEqual(edits.map(r=>r.status).sort(),[200,409]);assert.equal(versionOf(env,a),v+2);});

test('a declared event with zero likes produces source-backed portrait; a role alone does not',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);await addPerson(env,a,{relationship_text:'',role:'mother'});assert.equal(await claim(env),undefined);assert.equal((await (await request(env,'/api/portrait',{cookie:a.cookie})).json()).status,'collecting');const e=await addEvent(env,a);const j=await claim(env);assert.equal(j.kind,'portrait');assert.equal(j.likes.length,0);assert.equal(j.sources.length,1);assert.equal(j.sources[0].id,'event:'+e.id);assert.equal(j.sources[0].revision,1);assert.equal(j.sources[0].hash.length,64);await complete(env,j,{portrait:portraitV2(j)});const p=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(p.status,'ready');assert.equal(p.portrait.schema_version,2);assert.equal(p.portrait.fictional_character,undefined);});

test('portrait rejects missing summary evidence, forged citations, stale revisions and bare-like biography',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);await like(env,a,0);const j=await claim(env);const valid=portraitV2(j);for(const mutate of [p=>p.summary_evidence=[],p=>p.citations[0].quote='INVENTED EVIDENCE',p=>p.citations[0].revision++,p=>p.axes[0].confidence=0.26,p=>p.axes[0].claim_type='declaration',p=>p.strengths=[{text:'Un trait inventé',evidence:[j.sources.find(s=>s.type==='event')?.id||j.sources[0].id]}]]){const p=structuredClone(valid);mutate(p);await assert.rejects(complete(env,j,{portrait:p}),e=>e.status===400);}await complete(env,j,{portrait:valid});});

test('Paris calendar weeks handle midnight and both DST changes independently from 168 hours',()=>{assert.equal(parisWeek(new Date('2026-11-01T22:59:59Z')).week_start,'2026-10-26');assert.equal(parisWeek(new Date('2026-11-01T23:00:00Z')).week_start,'2026-11-02');assert.equal(parisWeek(new Date('2026-03-23T10:00:00Z')).renews_at,'2026-03-29T22:00:00.000Z');assert.equal(parisWeek(new Date('2026-10-19T10:00:00Z')).renews_at,'2026-10-25T23:00:00.000Z');});

test('incremental derivatives persist only exact revisions and partial portraits continue',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);pay(env,a);await addEvent(env,a);const e2=await addEvent(env,a);let j=await claim(env);const source=j.sources[0],cached={source_id:source.id,revision:source.revision,source_hash:source.hash,analysis_version:2,derivative:{signals:[{kind:'declaration',text:'Un élément déclaré.',quote:source.text.slice(0,60)}]},manifest:{}};await complete(env,j,{portrait:portraitV2(j,{pending:[j.sources[1].id]}),source_derivatives:[cached]});let p=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.notEqual(p.status,'ready');j=await claim(env);assert.equal(j.kind,'portrait');assert.equal(j.source_derivatives.length,1);await complete(env,j,{portrait:portraitV2(j)});p=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(p.status,'ready');const target=source.id.slice(6);await request(env,'/api/life-events/'+target,{method:'PATCH',cookie:a.cookie,data:{revision:1,description:'Une nouvelle interprétation.'}});assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM context_source_derivatives').get().n,0);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM user_portraits').get().n,0);});

test('legacy migration preserves exact story bodies, IDs, feedback and slot identity but invalidates V1 portrait',()=>{const db=new DatabaseSync(':memory:');for(const name of ['0001.sql','0002_inspirations.sql','0003_likes_portraits.sql','0004_recommendations_free_portraits.sql'])db.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));db.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','old@test','h','s'),('v','empty@test','h','s');INSERT INTO profiles(user_id) VALUES('u'),('v');INSERT INTO memories(user_id) VALUES('u'),('v');INSERT INTO subscriptions(user_id) VALUES('u'),('v');");db.prepare('INSERT INTO user_portraits(user_id,portrait,analyzed_version) VALUES(?,?,?)').run('u',JSON.stringify({fictional_character:{name:'Camille'}}),1);const day=parisWeek().week_start;for(let i=0;i<3;i++)db.prepare('INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice) VALUES(?,?,?,?,?,?,?,?)').run('story-'+i,'u',i===0?day:`2020-01-0${i+1}`,'Titre '+i,'Body exact à conserver '+i,'Fin '+i,'[]','');db.exec("INSERT INTO feedback(story_id,user_id,resonance,feelings,why,correction) VALUES('story-0','u',2,'émotion','texte à conserver','correction');");db.exec(readFileSync(new URL('../migrations/0005_personal_context_characters.sql',import.meta.url),'utf8'));assert.equal(db.prepare('SELECT count(*) n FROM stories').get().n,3);assert.equal(db.prepare("SELECT body FROM stories WHERE id='story-0'").get().body,'Body exact à conserver 0');assert.equal(db.prepare('SELECT correction FROM feedback').get().correction,'correction');assert.equal(db.prepare('SELECT count(*) n FROM user_portraits').get().n,0);assert.equal(JSON.parse(db.prepare('SELECT identity FROM personal_characters').get().identity).name,'Camille');assert.equal(db.prepare('SELECT count(*) n FROM story_requests WHERE week_start=?').get(day).n,1);assert.equal(db.prepare('SELECT count(*) n FROM users').get().n,2);assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(),[]);db.prepare("DELETE FROM users WHERE id='u'").run();assert.equal(db.prepare('SELECT count(*) n FROM feedback').get().n,0);});

test('free portrait starts remain bounded after personal form edits and do not depend on likes',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);await addEvent(env,a);for(let i=0;i<3;i++){const j=await claim(env);assert.equal(j.kind,'portrait');await complete(env,j,{portrait:portraitV2(j)});await addEvent(env,a);if(i<2){assert.equal(await claim(env),undefined);env.DB.raw.prepare('UPDATE portrait_usage SET last_started=last_started-601 WHERE user_id=?').run(a.user.id);}}assert.equal(await claim(env),undefined);const state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.reason,'quota');assert.equal(state.status,'queued');assert.ok(state.retry_after>Math.floor(Date.now()/1000));});

test('partial dates reject zero segments without inventing missing day or month',async()=>{const env={DB:database()},a=await register(env);for(const [date_precision,date_start]of [['month','2020-00'],['day','2020-06-00'],['day','2021-02-29']]){const r=await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),title:'Date',description:'Contexte',date_precision,date_start}});assert.equal(r.status,400,date_start);}const e=await addEvent(env,a,{date_precision:'month',date_start:'2020-06'});assert.equal(e.date_start,'2020-06');const undated=await addEvent(env,a,{date_precision:'unknown',date_start:''});assert.equal(undated.date_start,'');});

test('reading-context snapshots retry concurrent source edits instead of mixing source revisions',async()=>{const {loadContext}=await import('../src/private-context.mjs');const env={DB:database()},a=await register(env);await addEvent(env,a);const before=versionOf(env,a),prepare=env.DB.prepare.bind(env.DB);let injected=false;env.DB.prepare=sql=>{const statement=prepare(sql);if(sql.includes('FROM life_events e')){const all=statement.all.bind(statement);statement.all=async()=>{const result=await all();if(!injected){injected=true;env.DB.raw.prepare('UPDATE memories SET text=?,version=version+1 WHERE user_id=?').run('A new source arriving during reading.',a.user.id);env.DB.raw.prepare('UPDATE profiles SET version=version+1 WHERE user_id=?').run(a.user.id);}return result;};}return statement;};const c=await loadContext(env,a.user.id);assert.equal(c.source_version,before+1);assert.equal(c.memory.text,'A new source arriving during reading.');assert.ok(c.sources.some(s=>s.type==='memory'&&s.revision===2));});

test('the bounded five-hundred-event collection remains complete without a giant aggregate cell',async()=>{const {loadContext}=await import('../src/private-context.mjs');const env={DB:database()},a=await register(env);const insert=env.DB.raw.prepare('INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES(?,?,?,?,?,?,?)');for(let i=0;i<500;i++)insert.run('event-'+i,a.user.id,'creation-'+i,'hash','Event '+i,'other','x'.repeat(5970)+' END MARKER '+i);const c=await loadContext(env,a.user.id);assert.equal(c.events.length,500);assert.equal(c.sources.length,500);assert.ok(c.sources.find(s=>s.id==='event:event-499').text.endsWith('END MARKER 499'));const r=await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),title:'Too many',description:'No silent loss'}});assert.equal(r.status,400);assert.equal((await r.json()).code,'collection_limit');});

test('recommendation exposure stores bounded public IDs without changing personal source versions',async()=>{const env={DB:database()},a=await register(env);await like(env,a,0,'Je souhaite explorer la confiance et les relations.');const v=versionOf(env,a);for(let i=0;i<150;i++)env.DB.raw.prepare('INSERT INTO recommendation_exposures(user_id,article_id,shown_at) VALUES(?,?,?)').run(a.user.id,'old-'+i,i<20?'2020-01-01 00:00:00':'2026-11-03 00:00:00');const r=await request(env,'/api/recommendations',{cookie:a.cookie});assert.equal(r.status,200);assert.equal(versionOf(env,a),v);assert.ok(env.DB.raw.prepare('SELECT count(*) n FROM recommendation_exposures WHERE user_id=?').get(a.user.id).n<=100);assert.equal(env.DB.raw.prepare("SELECT count(*) n FROM recommendation_exposures WHERE shown_at<'2021'").get().n,0);});

test('self profile validates optional fields, isolates identity from analysis and uses compare-and-swap revisions',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),b=await register(env,'other-profile@example.com');
 const initial=await (await request(env,'/api/self-profile',{cookie:a.cookie})).json();assert.equal(initial.revision,1);assert.equal(initial.profile.gender,'');
 for(const data of [{birth_date:'2025-02-29'},{birth_date:'2000-00-10'},{birth_date:'2999-01-01'},{gender:'anything'},{self_description:'x'.repeat(10001)},{emotions:4}])assert.equal((await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:1,...data}})).status,400);
 const update={expected_revision:1,first_name:'CIVIL_SECRET',last_name:'FAMILY_SECRET',birth_date:'1990-02-20',gender:'non_binary',self_description:'Je souhaite poser mes limites.',occupation:'Un métier déclaré'};
 const replies=await Promise.all([request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:update}),request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{...update,self_description:'Une autre déclaration.'}})]);assert.deepEqual(replies.map(r=>r.status).sort(),[200,409]);
 const saved=await (await request(env,'/api/self-profile',{cookie:a.cookie})).json();assert.equal(saved.revision,2);assert.equal(saved.source_version,initial.source_version+1);
 assert.equal((await (await request(env,'/api/self-profile',{cookie:b.cookie})).json()).profile.first_name,'');
 const j=await claim(env);assert.equal(j.kind,'portrait');assert.equal(j.sources[0].type,'profile');assert.equal(j.sources[0].id,'profile:'+a.user.id);assert.equal(j.sources[0].revision,2);assert.equal(j.self_context.gender,'non_binary');for(const secret of ['CIVIL_SECRET','FAMILY_SECRET','1990-02-20'])assert.equal(JSON.stringify(j).includes(secret),false);
 const current=await (await request(env,'/api/self-profile',{cookie:a.cookie})).json();await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:2,self_description:current.profile.self_description}});assert.equal((await (await request(env,'/api/self-profile',{cookie:a.cookie})).json()).revision,2);
 const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();assert.equal(JSON.parse(exported.self_profile.self_data).first_name,'CIVIL_SECRET');
});

test('demographics and occupation alone provide no psychological evidence',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);pay(env,a);await prepareUser(env,a);
 await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:1,occupation:'Psychologue',gender:'female',birth_date:'1980-01-01'}});
 const j=await claim(env);assert.equal(j.sources.some(s=>s.type==='profile'),false);await complete(env,j,{portrait:portraitV2(j)});
 assert.equal(await claim(env),undefined);
});

test('legacy questionnaire edits preserve the self profile and invalidate an in-flight portrait',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);await readyInputs(env,a);const before=await (await request(env,'/api/self-profile',{cookie:a.cookie})).json(),j=await claim(env);
 const r=await request(env,'/api/profile',{method:'PUT',cookie:a.cookie,data:{consent_ai:true,display_name:'Pseudonyme historique'}});assert.equal(r.status,200);const after=await (await request(env,'/api/self-profile',{cookie:a.cookie})).json();assert.deepEqual(after.profile,before.profile);assert.equal(after.revision,before.revision);assert.ok(after.source_version>before.source_version);await assert.rejects(complete(env,j,{portrait:portraitV2(j)}),e=>e.status===409);
});

test('self-profile removal deletes its extraction cache and invalidates the portrait',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);await readyInputs(env,a);const j=await claim(env),source=j.sources.find(s=>s.type==='profile');
 await complete(env,j,{portrait:portraitV2(j),source_derivatives:[{source_id:source.id,revision:source.revision,source_hash:source.hash,analysis_version:2,derivative:{signals:[{kind:'declaration',text:'Une déclaration personnelle.',quote:source.text.slice(0,60)}]},manifest:{}}]});assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM context_source_derivatives').get().n,1);
 await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:2,self_description:''}});assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM context_source_derivatives').get().n,0);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM user_portraits').get().n,0);
});

test('trial entitlement uses trial end exclusively and keeps portrait access',async()=>{
 const t=Math.floor(Date.now()/1000);
 assert.equal(activeSubscription({status:'trialing',trial_end:t+100,paid_until:0}),true);
 assert.equal(activeSubscription({status:'trialing',trial_end:t,paid_until:t+999}),false);
 for(const status of ['none','canceled','past_due','unpaid','paused','incomplete'])assert.equal(activeSubscription({status,trial_end:t+999,paid_until:t+999}),false);
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);
 env.DB.raw.prepare("UPDATE subscriptions SET status='trialing',trial_end=?,paid_until=0,trial_used_at=? WHERE user_id=?").run(t+604800,t,a.user.id);
 const me=await (await request(env,'/api/me',{cookie:a.cookie})).json();
 assert.equal(me.subscription.active,true);assert.equal(me.subscription.trial_eligible,false);assert.equal(me.subscription.access_until,t+604800);
 await addEvent(env,a);
 const job=await claim(env);assert.equal(job.kind,'portrait');
 await complete(env,job,{portrait:portraitV2(job)});
 assert.equal((await (await request(env,'/api/portrait',{cookie:a.cookie})).json()).status,'ready');

});

test('trial cancellation works without AI and historical trials never receive a second offer',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_PRICE_ID:'price_29',STRIPE_WEBHOOK_SECRET:'test'},a=await register(env,'trial@example.com'),t=Math.floor(Date.now()/1000),originalFetch=globalThis.fetch;
 env.DB.raw.prepare("UPDATE subscriptions SET customer_id='cus_trial',subscription_id='sub_trial',status='trialing',trial_end=? WHERE user_id=?").run(t+604800,a.user.id);
 let sub={id:'sub_trial',customer:'cus_trial',status:'trialing',trial_start:t,trial_end:t+604800,current_period_end:t+604800,cancel_at_period_end:false,items:{data:[{quantity:1,price:{id:'price_29'}}]}};let created;
 globalThis.fetch=async(url,options={})=>{const u=new URL(url),p=u.pathname,fields=new URLSearchParams(options.body);const ok=x=>new Response(JSON.stringify(x));
  if(p==='/v1/subscriptions/sub_trial'){if(options.method==='POST'){assert.equal(fields.get('cancel_at_period_end'),'true');sub.cancel_at_period_end=true;}return ok(sub);}
  if(p==='/v1/prices/price_29')return ok({active:true,currency:'eur',unit_amount:2900,recurring:{interval:'month',interval_count:1}});
  if(p==='/v1/subscriptions')return u.searchParams.has('starting_after')?ok({data:[sub],has_more:false}):ok({data:[{id:'sub_other',status:'canceled'}],has_more:true});
  if(p==='/v1/checkout/sessions'){if(options.method==='POST'){created=fields;return ok({id:'cs_new',url:'https://checkout.stripe.com/new'});}return ok({data:[]});}
  throw new Error('Unexpected route '+p);
 };
 try {
  let r=await request(env,'/api/billing/cancel',{method:'POST',cookie:a.cookie,data:{}});assert.equal(r.status,200);assert.deepEqual(await r.json(),{ok:true,cancel_at_period_end:true,access_until:t+604800});
  let me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.subscription.active,true);assert.equal(me.subscription.cancel_at_period_end,true);
  sub={...sub,status:'canceled'};env.OPENROUTER_API_KEY='test';env.DB.raw.prepare("UPDATE subscriptions SET status='canceled',trial_used_at=0 WHERE user_id=?").run(a.user.id);
  r=await request(env,'/api/billing/checkout',{method:'POST',cookie:a.cookie,data:{}});assert.equal(r.status,200);assert.equal(created.get('payment_method_collection'),'always');assert.equal(created.has('subscription_data[trial_period_days]'),false);
  me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.subscription.trial_eligible,false);assert.equal(me.subscription.active,false);
 } finally {globalThis.fetch=originalFetch;}
});

test('signed Stripe webhooks reconcile canonical trial, paid, and canceled states despite delayed events',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_PRICE_ID:'price_29',STRIPE_WEBHOOK_SECRET:'webhook-secret'},a=await register(env,'webhooks@example.com'),t=Math.floor(Date.now()/1000),originalFetch=globalThis.fetch;
 env.DB.raw.prepare("UPDATE subscriptions SET customer_id='cus_hook' WHERE user_id=?").run(a.user.id);
 let sub={id:'sub_hook',customer:'cus_hook',status:'trialing',trial_start:t,trial_end:t+604800,current_period_end:t+604800,items:{data:[{quantity:1,price:{id:'price_29'}}]}};
 globalThis.fetch=async()=>new Response(JSON.stringify(sub));
 const deliver=async(id,created)=>{const raw=JSON.stringify({id,type:'customer.subscription.updated',created,data:{object:{id:'sub_hook'}}});const key=await crypto.subtle.importKey('raw',new TextEncoder().encode(env.STRIPE_WEBHOOK_SECRET),{name:'HMAC',hash:'SHA-256'},false,['sign']);const signature=Buffer.from(await crypto.subtle.sign('HMAC',key,new TextEncoder().encode(t+'.'+raw))).toString('hex');return worker.fetch(new Request(origin+'/api/billing/webhook',{method:'POST',headers:{'stripe-signature':`t=${t},v1=${signature}`},body:raw}),env);};
 try {
  assert.equal((await deliver('evt_trial',t)).status,200);let s=env.DB.raw.prepare('SELECT * FROM subscriptions').get();assert.equal(activeSubscription(s),true);assert.equal(s.trial_used_at,t);
  sub={...sub,status:'active',current_period_end:t+2592000};assert.equal((await deliver('evt_delayed',t-100)).status,200);s=env.DB.raw.prepare('SELECT * FROM subscriptions').get();assert.equal(s.status,'active');assert.equal(activeSubscription(s),true);
  sub={...sub,status:'canceled'};assert.equal((await deliver('evt_cancel',t-200)).status,200);s=env.DB.raw.prepare('SELECT * FROM subscriptions').get();assert.equal(activeSubscription(s),false);assert.equal(s.trial_used_at,t);
  assert.equal((await deliver('evt_cancel',t-200)).status,200);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM webhook_events').get().n,3);
 }finally{globalThis.fetch=originalFetch;}
});

test('checkout rejects old ten-euro price before customer or session creation',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'test',STRIPE_PRICE_ID:'price_old',STRIPE_WEBHOOK_SECRET:'test',OPENROUTER_API_KEY:'test'},a=await register(env,'price@example.com'),originalFetch=globalThis.fetch;let calls=0;
 globalThis.fetch=async()=>{calls++;return new Response(JSON.stringify({active:true,currency:'eur',unit_amount:1000,recurring:{interval:'month',interval_count:1}}));};
 try{assert.equal((await request(env,'/api/billing/checkout',{method:'POST',cookie:a.cookie,data:{}})).status,503);assert.equal(calls,1);}finally{globalThis.fetch=originalFetch;}
});

test('free preview bypasses free portrait start/cooldown limits and each source edit revokes the earlier lease',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test',FREE_ACCESS_ENABLED:'1'},a=await register(env),t=Math.floor(Date.now()/1000);
 const event=await addEvent(env,a);env.DB.raw.prepare('INSERT INTO portrait_usage(user_id,window_started,starts,last_started) VALUES(?,?,3,?)').run(a.user.id,t,t);
 let state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.retry_after,null);assert.equal(state.reason,null);
 const previous=await claim(env);assert.equal(previous.kind,'portrait');
 await request(env,'/api/life-events/'+event.id,{method:'PATCH',cookie:a.cookie,data:{revision:1,description:'Je vois ce souvenir autrement aujourd’hui.'}});
 assert.equal(env.DB.raw.prepare('SELECT status FROM journal_jobs WHERE id=?').get(previous.id).status,'obsolete');
 await assert.rejects(complete(env,previous,{portrait:portraitV2(previous)}),e=>e.status===409);
 const current=await claim(env);assert.equal(current.source_version,previous.source_version+1);await complete(env,current,{portrait:portraitV2(current)});
 for(let index=0;index<3;index++){await addEvent(env,a,{description:'Un souvenir distinct numéro '+index});const job=await claim(env);assert.equal(job.kind,'portrait');await complete(env,job,{portrait:portraitV2(job)});}
 assert.equal(env.DB.raw.prepare('SELECT starts FROM portrait_usage WHERE user_id=?').get(a.user.id).starts,3);
 state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.status,'ready');assert.equal(state.retry_after,null);
});

function internalPortraitFeedback(env,account,data){
 const backend=createPersonalBackend({error(status,message,code){throw Object.assign(new Error(message),{status,code});},json:value=>value,text:value=>String(value||''),body:request=>request.json(),now:()=>Math.floor(Date.now()/1000),random:()=>crypto.randomUUID(),decorateLikes:value=>value,recommender:createRecommender(examples),examples});
 const url=new URL('/api/portrait/feedback',origin);
 return backend.routes(new Request(url,{method:'POST',body:JSON.stringify(data)}),env,account.user,url.pathname,url);
}

test('the former cartography feedback endpoint rejects account requests without changing the portrait',async()=>{
 const env={DB:database()},a=await register(env),version=versionOf(env,a);
 const response=await request(env,'/api/portrait/feedback',{method:'POST',cookie:a.cookie,data:{pattern_id:'rumination',kind:'nuance',text:'Une précision.',creation_id:crypto.randomUUID()}});
 assert.equal(response.status,404);assert.equal(versionOf(env,a),version);
});

test('internal portrait corrections append atomically, remain private and replay without duplicate versions',async()=>{
 const env={DB:database(),FREE_ACCESS_ENABLED:'true'},a=await register(env),b=await register(env,'feedback-other@example.com');
 await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'Une correction déjà écrite par moi.'}});const before=versionOf(env,a);
 const first={pattern_id:'rumination',kind:'nuance',text:'Cela arrive surtout quand je manque de sommeil.',creation_id:crypto.randomUUID()};
 const second={pattern_id:'rumination',kind:'not_concerned',text:'En vacances cette piste ne me correspond pas.',creation_id:crypto.randomUUID()};
 const responses=await Promise.all([first,second].map(data=>internalPortraitFeedback(env,a,data)));for(const response of responses)assert.equal(response.ok,true);
 let memory=await (await request(env,'/api/memory',{cookie:a.cookie})).json();assert.ok(memory.text.startsWith('Une correction déjà écrite par moi.'));assert.ok(memory.text.includes(first.text));assert.ok(memory.text.includes(second.text));assert.equal(versionOf(env,a),before+2);
 const after=versionOf(env,a),replay=await internalPortraitFeedback(env,a,first);assert.equal(replay.replayed,true);assert.equal(versionOf(env,a),after);
 await assert.rejects(internalPortraitFeedback(env,a,{...first,text:'Un autre retour.'}),error=>error.status===409);
 await assert.rejects(internalPortraitFeedback(env,a,{...first,creation_id:crypto.randomUUID(),pattern_id:'invented'}),error=>error.status===400);
 assert.equal((await (await request(env,'/api/memory',{cookie:b.cookie})).json()).text,'');
 memory=await (await request(env,'/api/memory',{cookie:a.cookie})).json();assert.equal(memory.text.split(first.text).length-1,1);
 await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'x'.repeat(9950)}});const fullVersion=versionOf(env,a);
 await assert.rejects(internalPortraitFeedback(env,a,{...first,creation_id:crypto.randomUUID()}),error=>error.code==='memory_limit');assert.equal(versionOf(env,a),fullVersion);assert.equal((await (await request(env,'/api/memory',{cookie:a.cookie})).json()).text.length,9950);
});

test('source writes debounce a durable native refresh and latest source version wins without duplicate provider calls',async t=>{
 const env={DB:database(),OPENROUTER_API_KEY:'mock-provider',FREE_ACCESS_ENABLED:'true'},a=await register(env),tasks=[],context={waitUntil:task=>tasks.push(task)};let calls=0;
 const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original;});
 globalThis.fetch=async(_url,options)=>{calls++;const input=JSON.parse(JSON.parse(options.body).messages.at(-1).content);if(input.claims)return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify({checks:input.claims.map(c=>({path:c.path,supported:true}))})}}]}));const snapshot=await loadContext(env,a.user.id);return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify({...portraitV2(snapshot),psychological_map:[]})}}],usage:{prompt_tokens:20,completion_tokens:20}}));};
 const response=await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,context,data:{creation_id:crypto.randomUUID(),title:'Un changement',description:'J’ai appris à demander de l’aide à mes amis.'}});assert.equal(response.status,201);
 const old=env.DB.raw.prepare("SELECT id,lease_until FROM journal_jobs WHERE kind='portrait' AND status='pending'").get();assert.ok(old.lease_until>=Math.floor(Date.now()/1000));
 await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,context,data:{text:'Je souhaite nuancer ce récit avec mes mots.'}});
 assert.equal(env.DB.raw.prepare('SELECT status FROM journal_jobs WHERE id=?').get(old.id).status,'obsolete');
 assert.equal(calls,0);assert.ok(tasks.length>=2);await Promise.all(tasks);
 const state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.status,'ready');assert.equal(state.analyzed_version,versionOf(env,a));assert.equal(calls,2);assert.equal(state.psychology_map,undefined);assert.equal(state.portrait.psychological_map,undefined);
 const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();assert.equal(exported.portrait.psychology_map.outdated,false);assert.ok(Array.isArray(exported.portrait.portrait.psychological_map));
});

test('cron refresh recovers a persisted portrait queue and rejects stale model results after an intervening source write',async t=>{
 const env={DB:database(),OPENROUTER_API_KEY:'mock-provider',FREE_ACCESS_ENABLED:'true'},a=await register(env);await addEvent(env,a);let calls=0;const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original;});
 globalThis.fetch=async(_url,options)=>{calls++;const input=JSON.parse(JSON.parse(options.body).messages.at(-1).content);if(input.claims)return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify({checks:input.claims.map(c=>({path:c.path,supported:true}))})}}]}));const snapshot=await loadContext(env,a.user.id);if(calls===1)await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'Une correction ajoutée pendant le calcul.'}});return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify({...portraitV2(snapshot),psychological_map:[]})}}]}));};
 const result=await runGenerationBatch(env);assert.equal(result.status,'worker-managed');assert.equal(result.failed,1);assert.equal(result.completed,1);assert.equal(calls,3);
 const saved=env.DB.raw.prepare('SELECT analyzed_version,portrait FROM user_portraits WHERE user_id=?').get(a.user.id);assert.equal(saved.analyzed_version,versionOf(env,a));assert.ok(JSON.parse(saved.portrait).citations.some(c=>c.source_id==='memory:'+a.user.id));
});

test('current-version legacy portraits are queued for map upgrade and claims can be scoped to one user',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'mock-provider',FREE_ACCESS_ENABLED:'true'},a=await register(env),b=await register(env,'scoped@example.com');await addEvent(env,a);await addEvent(env,b);
 const scoped=(await (await claimJobs(env,b.user.id,'portrait',90)).json()).jobs[0];assert.equal(scoped.user_id,b.user.id);await complete(env,scoped,{portrait:portraitV2(scoped)});
 const legacy=JSON.parse(env.DB.raw.prepare('SELECT portrait FROM user_portraits WHERE user_id=?').get(b.user.id).portrait);delete legacy.psychological_map;env.DB.raw.prepare('UPDATE user_portraits SET portrait=? WHERE user_id=?').run(JSON.stringify(legacy),b.user.id);
 const state=await (await request(env,'/api/portrait',{cookie:b.cookie})).json();assert.equal(state.status,'queued');assert.equal(state.outdated,true);assert.equal(state.psychology_map,undefined);
 const upgrade=(await (await claimJobs(env,b.user.id,'portrait',90)).json()).jobs[0];assert.equal(upgrade.user_id,b.user.id);assert.equal(upgrade.source_version,scoped.source_version);assert.notEqual(upgrade.id,scoped.id);
 assert.equal(env.DB.raw.prepare("SELECT status FROM journal_jobs WHERE user_id=? AND kind='portrait' ORDER BY created_at LIMIT 1").get(a.user.id).status,'pending');
});

test('versioned memory editors cannot overwrite a concurrently appended portrait correction',async()=>{
 const env={DB:database()},a=await register(env);const memory=await (await request(env,'/api/memory',{cookie:a.cookie})).json();
 const data={pattern_id:'rumination',kind:'nuance',text:'Un contexte précis à conserver.',creation_id:crypto.randomUUID()};assert.equal((await internalPortraitFeedback(env,a,data)).ok,true);
 const version=versionOf(env,a);let response=await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'Mon ancien brouillon.',expected_version:memory.version}});assert.equal(response.status,409);assert.equal((await response.json()).code,'revision_conflict');assert.equal(versionOf(env,a),version);
 const latest=await (await request(env,'/api/memory',{cookie:a.cookie})).json();assert.ok(latest.text.includes(data.text));response=await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:latest.text+' Une précision ajoutée.',expected_version:latest.version}});assert.equal(response.status,200);
});

test('native refresh defers large source sets without attempts, free-start quota or provider cost and Python can claim them',async t=>{
 const env={DB:database(),OPENROUTER_API_KEY:'mock-provider'},a=await register(env);await addEvent(env,a);
 const insert=env.DB.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES(?,?,?,'hash','Un souvenir','other',?)");
 for(let index=0;index<50;index++)insert.run('large-'+index,a.user.id,'large-key-'+index,'Une description détaillée. '.repeat(130));env.DB.raw.prepare('UPDATE profiles SET version=version+1 WHERE user_id=?').run(a.user.id);
 const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original;});let calls=0;globalThis.fetch=async()=>{calls++;throw new Error('Native provider must not run');};
 const result=await runGenerationBatch(env);assert.equal(result.claimed,0);assert.equal(calls,0);
 const pending=env.DB.raw.prepare("SELECT * FROM journal_jobs WHERE user_id=? AND source_version=? AND kind='portrait'").get(a.user.id,versionOf(env,a));assert.equal(pending.error_code,'batch_required');assert.equal(pending.status,'pending');assert.equal(pending.attempts,0);assert.equal(pending.provider_calls,0);assert.equal(pending.reserved_tokens,0);
 assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM portrait_usage WHERE user_id=?').get(a.user.id).n,0);
 const pythonJob=await claim(env);assert.equal(pythonJob.id,pending.id);assert.equal(pythonJob.sources.length,51);assert.equal(pythonJob.attempts,1);
});

test('native HTTP budget persists a candidate and resumes only its verification after a timeout',async t=>{
 const env={DB:database(),OPENROUTER_API_KEY:'mock-provider',FREE_ACCESS_ENABLED:'true'},a=await register(env);await addEvent(env,a);
 const originalFetch=globalThis.fetch,originalNow=Date.now;let elapsed=0,calls=0;Date.now=()=>originalNow()+elapsed;t.after(()=>{globalThis.fetch=originalFetch;Date.now=originalNow;});
 globalThis.fetch=async(_url,options)=>{calls++;const input=JSON.parse(JSON.parse(options.body).messages.at(-1).content);if(input.claims)return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify({checks:input.claims.map(c=>({path:c.path,supported:true}))})}}]}));const snapshot=await loadContext(env,a.user.id);if(calls===1)elapsed=23000;return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify({...portraitV2(snapshot),psychological_map:[]})}}]}));};
 const failed=await refreshUserPortrait(env,a.user.id);assert.equal(failed.failed,1);assert.equal(calls,1);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM user_portraits').get().n,0);
 const pending=env.DB.raw.prepare("SELECT * FROM journal_jobs WHERE user_id=? AND status='pending'").get(a.user.id);assert.equal(pending.attempts,1);assert.ok(pending.lease_until<=Math.floor(Date.now()/1000)+30);
 elapsed=0;env.DB.raw.prepare('UPDATE journal_jobs SET lease_until=0 WHERE id=?').run(pending.id);
 assert.ok(pending.native_candidate);const fast=await refreshUserPortrait(env,a.user.id);assert.equal(fast.completed,1);assert.equal(calls,2);assert.equal(env.DB.raw.prepare('SELECT native_candidate FROM journal_jobs WHERE id=?').get(pending.id).native_candidate,null);
 const recovered=await runGenerationBatch(env);assert.equal(recovered.completed,0);assert.equal(calls,2);assert.equal(env.DB.raw.prepare('SELECT analyzed_version FROM user_portraits WHERE user_id=?').get(a.user.id).analyzed_version,versionOf(env,a));
});


test('native portrait model uses its own binding and omits unsupported temperature',async t=>{
 const env={DB:database(),OPENROUTER_API_KEY:'mock-provider',FREE_ACCESS_ENABLED:'true',OPENROUTER_PORTRAIT_MODEL:'anthropic/claude-sonnet-5.5'},a=await register(env);await addEvent(env,a);
 const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original});let calls=0;
 globalThis.fetch=async(_url,options)=>{calls++;const body=JSON.parse(options.body);assert.equal(body.model,'anthropic/claude-sonnet-5.5');assert.equal(Object.hasOwn(body,'temperature'),false);const input=JSON.parse(body.messages.at(-1).content),result=input.claims?{checks:input.claims.map(c=>({path:c.path,supported:true}))}:{...portraitV2(await loadContext(env,a.user.id)),psychological_map:[]};return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify(result)}}]}));};
 const result=await runGenerationBatch(env);assert.equal(result.completed,1);assert.equal(calls,2);
});

test('source edits clear unpublished native portrait candidates',async()=>{
 const env={DB:database()},a=await register(env);await addEvent(env,a);const job=await claimJobs({...env,OPENROUTER_API_KEY:'mock'});const id=(await job.json()).jobs[0].id;
 env.DB.raw.prepare('UPDATE journal_jobs SET native_candidate=? WHERE id=?').run('{"private":"candidate"}',id);
 await request(env,'/api/memory',{method:'PUT',cookie:a.cookie,data:{text:'Une nouvelle précision.'}});
 assert.equal(env.DB.raw.prepare('SELECT native_candidate FROM journal_jobs WHERE id=?').get(id).native_candidate,null);
});


test('a semantic rejection can retry automatically with a new guarded synthesis, within three attempts',async t=>{
 const env={DB:database(),OPENROUTER_API_KEY:'mock-provider',FREE_ACCESS_ENABLED:'true'},a=await register(env);await addEvent(env,a);let checks=0,calls=0;const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original});
 globalThis.fetch=async(_url,options)=>{calls++;const input=JSON.parse(JSON.parse(options.body).messages.at(-1).content);let result;if(input.claims){checks++;result={checks:input.claims.map(c=>({path:c.path,supported:checks>1}))}}else result={...portraitV2(await loadContext(env,a.user.id)),psychological_map:[]};return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify(result)}}]}));};
 const first=await refreshUserPortrait(env,a.user.id);assert.equal(first.failed,1);const job=env.DB.raw.prepare("SELECT * FROM journal_jobs WHERE user_id=? AND status='pending'").get(a.user.id);assert.equal(job.error_code,'validation_error');const diagnostic=JSON.parse(job.native_candidate);assert.equal(diagnostic.rejected,true);assert.ok(diagnostic.failed_paths.includes('summary'));assert.deepEqual(Object.keys(diagnostic).sort(),['analysis_version','failed_paths','rejected']);env.DB.raw.prepare('UPDATE journal_jobs SET lease_until=0 WHERE id=?').run(job.id);
 const retry=await refreshUserPortrait(env,a.user.id);assert.equal(retry.completed,1);assert.equal(calls,4);
});

test('portrait display identity is current and separate from evidence; chat readings stay canonical and transient',async()=>{
 const env={DB:database(),PUBLIC_ORIGIN:origin,FREE_ACCESS_ENABLED:'true'},a=await register(env);
 env.DB.raw.prepare('UPDATE profiles SET self_data=? WHERE user_id=?').run(JSON.stringify({first_name:'Romain',last_name:'Nom privé',gender:'male',self_description:'Je cherche à mieux comprendre mes relations et ma confiance.'}),a.user.id);
 const c=await loadContext(env,a.user.id);
 assert.deepEqual(c.subject,{first_name:'Romain',gender:'male'});
 assert.equal(c.self_context.first_name,undefined);
 assert.ok(!JSON.stringify(c.sources).includes('Romain'));
 assert.ok(!JSON.stringify(c.sources).includes('Nom privé'));
 const result=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();
 assert.deepEqual(result.subject,c.subject);
 const personal=createPersonalBackend({error(status,message,code){throw Object.assign(new Error(message),{status,code});},json:v=>v,text:v=>String(v||''),body:r=>r.json(),now:()=>Math.floor(Date.now()/1000),random:()=>crypto.randomUUID(),decorateLikes:v=>v,recommender:createRecommender(examples),examples});
 const before=env.DB.raw.prepare('SELECT version,self_revision,self_data FROM profiles WHERE user_id=?').get(a.user.id);
 const readings=await personal.readingSuggestions(env,a.user.id,{query:'Un article pour explorer la confiance dans les relations',limit:30});
 assert.ok(readings.length>0&&readings.length<=6);
 for(const r of readings){const original=examples.find(e=>e.id===r.id);assert.equal(r.url,original.url);assert.equal(r.title,original.title);assert.ok(r.reason);}
 assert.deepEqual(env.DB.raw.prepare('SELECT version,self_revision,self_data FROM profiles WHERE user_id=?').get(a.user.id),before);
 env.DB.raw.prepare('UPDATE profiles SET self_data=? WHERE user_id=?').run(JSON.stringify({first_name:'Autre prénom',gender:'non_binary'}),a.user.id);
 assert.equal((await (await request(env,'/api/portrait',{cookie:a.cookie})).json()).subject.first_name,'Autre prénom');
});

test('portrait publication enforces current gender and V3 language even when a legacy caller omits analysis metadata',async()=>{
 const env={DB:database(),PUBLIC_ORIGIN:origin,FREE_ACCESS_ENABLED:'true'},a=await register(env);
 const saved=await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:1,first_name:'Romain',gender:'male',self_description:'Je me décris comme créatif.'}});assert.equal(saved.status,200);
 env.OPENROUTER_API_KEY='mock-only';const j=await claim(env);assert.equal(j.kind,'portrait');const s=j.sources.find(s=>s.type==='profile');
 const p={schema_version:2,summary:'{{prenom}} se décrit comme créatif.',summary_evidence:[s.id],axes:[{label:'Créativité',why:'{{prenom}} évoque la créativité.',confidence:.3,claim_type:'declaration',evidence:[s.id]}],strengths:[],uncertainties:[],citations:[{source_id:s.id,revision:s.revision,quote:s.fields.self_description}],psychological_map:[]};
 await assert.rejects(()=>complete(env,j,{portrait:{...p,summary:'Elle se décrit comme créative.'}}),e=>e.code==='invalid_portrait');
 await assert.rejects(()=>complete(env,j,{portrait:{...p,summary:'{{prenom}} est créative.',identity_context:{gender:'female'}}}),e=>e.code==='invalid_portrait');
 await complete(env,j,{portrait:p});const stored=JSON.parse(env.DB.raw.prepare('SELECT portrait FROM user_portraits WHERE user_id=?').get(a.user.id).portrait);
 assert.equal(stored.analysis_version,3);assert.equal(stored.identity_context.gender,'male');assert.equal(stored.identity_context.first_name,undefined);
});

test('cancellation retries share a durable key and a resumed subscription opens one fresh attempt under concurrency',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'fake',STRIPE_PRICE_ID:'price_29'},a=await register(env,'cancellation-retry@example.com'),db=env.DB.raw,t=Math.floor(Date.now()/1000),originalFetch=globalThis.fetch;
 db.prepare("UPDATE subscriptions SET customer_id='cus_cancel',subscription_id='sub_cancel',status='active',paid_until=?,checkout_attempt='checkout-preserved' WHERE user_id=?").run(t+86400,a.user.id);
 const remote={id:'sub_cancel',customer:'cus_cancel',status:'active',current_period_end:t+86400,cancel_at_period_end:false,items:{data:[{quantity:1,price:{id:'price_29'}}]}},cache=new Map(),keys=[];let fail=true,mutations=0;
 globalThis.fetch=async(url,options={})=>{assert.equal(new URL(url).pathname,'/v1/subscriptions/sub_cancel');if(options.method==='POST'){const key=options.headers['Idempotency-Key'];keys.push(key);if(fail){fail=false;return new Response('{}',{status:503});}if(!cache.has(key)){remote.cancel_at_period_end=true;mutations++;cache.set(key,structuredClone(remote));}return Response.json(cache.get(key));}return Response.json(remote);};
 try{
  const cancel=()=>request(env,'/api/billing/cancel',{method:'POST',cookie:a.cookie,data:{}});
  assert.equal((await cancel()).status,502);const originalAttempt=db.prepare('SELECT cancellation_attempt FROM subscriptions WHERE user_id=?').get(a.user.id).cancellation_attempt;assert.ok(originalAttempt);
  for(const response of await Promise.all([cancel(),cancel()])){assert.equal(response.status,200);assert.equal((await response.json()).cancel_at_period_end,true);}
  assert.equal(new Set(keys).size,1);assert.equal(mutations,1);assert.equal(db.prepare('SELECT cancellation_attempt FROM subscriptions WHERE user_id=?').get(a.user.id).cancellation_attempt,originalAttempt);
  remote.cancel_at_period_end=false; // The customer resumes in the Stripe portal.
  const previousKeys=keys.length;
  for(const response of await Promise.all([cancel(),cancel()])){assert.equal(response.status,200);assert.equal((await response.json()).cancel_at_period_end,true);}
  assert.equal(new Set(keys.slice(previousKeys)).size,1);assert.notEqual(keys.at(-1),keys[0]);assert.equal(mutations,2);
  const saved=db.prepare('SELECT * FROM subscriptions WHERE user_id=?').get(a.user.id);assert.equal(saved.checkout_attempt,'checkout-preserved');assert.equal(saved.cancellation_confirmed,1);assert.equal(saved.cancel_at_period_end,1);
 }finally{globalThis.fetch=originalFetch;}
});

test('a cached cancellation response cannot report success while canonical Stripe state remains resumed',async()=>{
 const env={DB:database(),STRIPE_SECRET_KEY:'fake',STRIPE_PRICE_ID:'price_29'},a=await register(env,'cancellation-canonical@example.com'),db=env.DB.raw,t=Math.floor(Date.now()/1000),originalFetch=globalThis.fetch;
 db.prepare("UPDATE subscriptions SET customer_id='cus_replay',subscription_id='sub_replay',status='active',paid_until=?,cancellation_attempt='lost-response',cancellation_subscription_id='sub_replay' WHERE user_id=?").run(t+86400,a.user.id);
 const remote={id:'sub_replay',customer:'cus_replay',status:'active',current_period_end:t+86400,cancel_at_period_end:false,items:{data:[{quantity:1,price:{id:'price_29'}}]}},keys=[];
 globalThis.fetch=async(url,options={})=>{assert.equal(new URL(url).pathname,'/v1/subscriptions/sub_replay');if(options.method==='POST'){const key=options.headers['Idempotency-Key'];keys.push(key);if(!key.endsWith('-lost-response'))remote.cancel_at_period_end=true;return Response.json({...remote,cancel_at_period_end:true});}return Response.json(remote);};
 try{
  const cancel=()=>request(env,'/api/billing/cancel',{method:'POST',cookie:a.cookie,data:{}});let response=await cancel();assert.equal(response.status,502);assert.equal((await response.json()).code,'cancellation_not_confirmed');assert.equal(db.prepare('SELECT cancel_at_period_end FROM subscriptions WHERE user_id=?').get(a.user.id).cancel_at_period_end,0);
  response=await cancel();assert.equal(response.status,200);assert.equal((await response.json()).cancel_at_period_end,true);assert.notEqual(keys[0],keys[1]);
 }finally{globalThis.fetch=originalFetch;}
});

test('retired personal story links go to Ambre and discard private query parameters',async()=>{
 for(const path of ['/mon-personnage','/mon-personnage/old-story','/mes-personnages/','/mes-histoires/','/journal/']){
  for(const method of ['GET','HEAD']){
   const response=await request({},path+'?story=old-story&personnage=private-character',{method});
   assert.equal(response.status,302,path);
   assert.equal(response.headers.get('location'),'/mon-accompagnement/');
   assert.match(response.headers.get('cache-control'),/private, no-store/);
  }
 }
 const likes=await request({},'/journal/?tab=likes&article=example&like=1&story=private');
 assert.equal(likes.headers.get('location'),'/mon-portrait/?article=example&like=1#mes-articles');
 const billing=await request({},'/journal/?payment=success&personnage=private');
 assert.equal(billing.headers.get('location'),'/mon-compte/?payment=success');
 assert.equal((await request({},'/mon-accompagnement/')).status,200);
});

function seedRetiredStory(db,userId){
 const t=Math.floor(Date.now()/1000)+1200,week=parisWeek().week_start;
 db.prepare("INSERT INTO personal_characters(id,user_id,slot,identity,is_primary) VALUES('old-character',?,1,'{\"name\":\"Camille\"}',1)").run(userId);
 db.prepare(`INSERT INTO story_requests(id,user_id,character_id,idempotency_key,parameter_hash,week_start,status,credit_status,source_version,canon_version,expires_at)
  VALUES('published',?,'old-character','published-key','hash',?,'completed','consumed',1,0,?),
   ('old-request',?,'old-character','old-request-key','hash',?,'running','reserved',1,0,?)`).run(userId,week,t,userId,week,t);
 db.prepare(`INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice,character_id,request_id,episode_number)
  VALUES('old-story',?,'2020-01-01','Titre archivé','Texte exact à conserver','Résumé','[]','','old-character','published',1)`).run(userId);
 db.prepare("INSERT INTO feedback(story_id,user_id,resonance,feelings,why,correction) VALUES('old-story',?,3,'','Retour historique à conserver','')").run(userId);
 db.prepare(`INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,request_id,character_id,status,lease_token,lease_until)
  VALUES('old-lease',?,'story',1,1,'old-request','old-character','running','old-token',?),
   ('old-preparation',?,'characters',1,1,NULL,NULL,'pending',NULL,0)`).run(userId,t,userId);
}

test('stale personal story clients and leased jobs cannot change or publish legacy content',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},account=await register(env);
 const {cookie,user}=account,db=env.DB.raw;
 seedRetiredStory(db,user.id);
 for(const [path,method] of [['/api/characters','GET'],['/api/characters/prepare','POST'],['/api/characters/old-character/continue','POST'],['/api/stories','GET'],['/api/stories/old-story','GET'],['/api/stories/old-story/feedback','PUT'],['/api/story-requests/old-request','GET'],['/api/story-requests/old-request/cancel','POST']]){
  const response=await request(env,path,{method,cookie,...(method!=='GET'?{data:{}}:{})});
  assert.equal(response.status,410,path);assert.equal((await response.json()).code,'feature_retired');
 }
 const data={lease_token:'old-token',source_version:1,memory_version:1};
 for(const call of [completeJob,proxyJobLLM,failJob])await assert.rejects(call(env,'old-lease',data),e=>e.code==='lease_expired');
 assert.equal(await claim(env),undefined);
 const person=await addPerson(env,account),event=await addEvent(env,account,{people_ids:[person.id],fiction_use:'understanding_only'});
 assert.equal(Object.hasOwn(event,'fiction_use'),false);
 const portraitJob=await claim(env);
 assert.equal(portraitJob.kind,'portrait');
 assert.ok(portraitJob.sources.some(s=>s.id==='event:'+event.id));
 assert.ok(portraitJob.sources.every(s=>s.type!=='feedback'&&!Object.hasOwn(s,'fiction_use')));
 await complete(env,portraitJob,{portrait:portraitV2(portraitJob)});
 const exported=await (await request(env,'/api/export',{cookie})).json();
 assert.equal(exported.stories[0].body,'Texte exact à conserver');
 assert.equal(exported.feedback[0].why,'Retour historique à conserver');
 assert.equal(exported.characters[0].identity.name,'Camille');
 assert.deepEqual(exported.events[0].people_ids,[person.id]);
 const other=await register(env,'archive-other@example.test');
 const otherExport=await (await request(env,'/api/export',{cookie:other.cookie})).json();
 assert.deepEqual(otherExport.stories,[]);
 assert.equal((await request(env,'/api/account',{method:'DELETE',cookie,data:{password:'long secure password 123'}})).status,200);
 for(const table of ['stories','feedback','personal_people','life_events','life_event_people','personal_characters','story_requests','journal_jobs'])assert.equal(db.prepare(`SELECT count(*) n FROM ${table}`).get().n,0,table);
 assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(),[]);
});

test('retirement migration releases open work, invalidates affected portraits and preserves historical data',async()=>{
 const env={DB:database({beforeRetirement:true})},account=await register(env),db=env.DB.raw,userId=account.user.id;
 seedRetiredStory(db,userId);
 db.prepare("INSERT INTO user_portraits(user_id,portrait,analyzed_version,schema_version) VALUES(?,?,1,2)").run(userId,JSON.stringify({coverage:{processed:['feedback:old-story'],pending:[]}}));
 const originalVersion=versionOf(env,account);
 const migration=readFileSync(new URL('../migrations/0014_retire_personal_stories.sql',import.meta.url),'utf8');
 db.exec(migration);db.exec(migration);
 assert.equal(versionOf(env,account),originalVersion+1);
 assert.equal(db.prepare('SELECT count(*) n FROM user_portraits').get().n,0);
 assert.deepEqual(db.prepare("SELECT id,status,credit_status FROM story_requests ORDER BY id").all().map(row=>({...row})),[
  {id:'old-request',status:'cancelled',credit_status:'released'},{id:'published',status:'completed',credit_status:'consumed'}
 ]);
 assert.ok(db.prepare("SELECT status,lease_token,lease_until FROM journal_jobs WHERE kind!='portrait'").all().every(row=>row.status==='obsolete'&&row.lease_token===null&&row.lease_until===0));
 assert.equal(db.prepare("SELECT body FROM stories WHERE id='old-story'").get().body,'Texte exact à conserver');
 assert.equal(db.prepare("SELECT why FROM feedback WHERE story_id='old-story'").get().why,'Retour historique à conserver');
 assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(),[]);
});

test('portrait pause blocks provider work while leaving event edits available',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},account=await register(env);
 await addEvent(env,account);const job=await claim(env);
 env.GENERATION_PAUSED='true';
 assert.equal((await (await claimJobs(env)).json()).status,'generation_paused');
 await assert.rejects(complete(env,job,{portrait:portraitV2(job)}),e=>e.code==='generation_paused');
 const event=await addEvent(env,account,{description:'Un moment dont je veux garder la trace.'});
 assert.ok(event.id);
});

test('portrait proxy keeps the approved model, twenty call limit and observed usage ledger',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test',OPENROUTER_MODEL:'approved/model'},account=await register(env);
 await addEvent(env,account);const job=await claim(env),db=env.DB.raw;
 const data={lease_token:job.lease_token,source_version:job.source_version,memory_version:job.memory_version,
  system_prompt:'Trusted instructions',user_prompt:'Test source',max_tokens:128,temperature:0.15};
 const original=globalThis.fetch;let calls=0;
 globalThis.fetch=async(url,options)=>{
  calls++;assert.equal(url,'https://openrouter.ai/api/v1/chat/completions');
  const payload=JSON.parse(options.body);assert.equal(payload.model,'approved/model');assert.equal(payload.temperature,0.15);
  return Response.json({choices:[{finish_reason:'stop',message:{content:'A result'}}],usage:{prompt_tokens:12,completion_tokens:8,cost:0.001}});
 };
 try{
  await assert.rejects(proxyJobLLM(env,job.id,{...data,model:'not-allowed'}),e=>e.code==='invalid_model');
  await assert.rejects(proxyJobLLM(env,job.id,{...data,max_tokens:9000}),e=>e.code==='invalid_parameters');
  for(let index=0;index<20;index++)assert.equal((await (await proxyJobLLM(env,job.id,data)).json()).content,'A result');
  await assert.rejects(proxyJobLLM(env,job.id,data),e=>e.code==='generation_budget');
  assert.equal(calls,20);
  const ledger=db.prepare('SELECT * FROM provider_budgets WHERE user_id=?').get(account.user.id);
  assert.equal(ledger.provider_calls,20);assert.equal(ledger.input_tokens,240);assert.equal(ledger.output_tokens,160);
  await failJob(env,job.id,{lease_token:job.lease_token,reason:'provider_error'});
  assert.equal(db.prepare('SELECT status FROM journal_jobs WHERE id=?').get(job.id).status,'failed');
 }finally{globalThis.fetch=original;}
});
