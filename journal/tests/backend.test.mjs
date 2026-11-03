import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { DatabaseSync } from 'node:sqlite';
import { createRecommender } from '../src/recommendations.mjs';
const examples=Array.from({length:14},(_,i)=>({id:`category/example-${i+1}`,title:`Exemple ${i+1}`,url:`https://decodetonesprit.fr/posts/example-${i+1}/`,character:'Personnage '+i,keywords:['confiance','relations'],description:`Résumé canonique ${i+1}`}));
const recommendationContexts=[];
// Use the real recommendation engine while recording its authenticated input.
globalThis.__backendRecommendationFactory=items=>{
 const engine=createRecommender(items);
 return {recommend(context,options){recommendationContexts.push(structuredClone(context));return engine.recommend(context,options);},provisionalPortrait(context){recommendationContexts.push(structuredClone(context));return engine.provisionalPortrait(context);}};
};
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<title>Mes histoires</title><main>App shell</main>';").replace("import { EXAMPLES } from './examples.mjs';",'const EXAMPLES='+JSON.stringify(examples)+';').replace("import { createRecommender } from './recommendations.mjs';",'const createRecommender=globalThis.__backendRecommendationFactory;').replace("'./chat-backend.mjs'",JSON.stringify(new URL('../src/chat-backend.mjs',import.meta.url).href)).replace("'./entitlements.mjs'",JSON.stringify(new URL('../src/entitlements.mjs',import.meta.url).href)).replace("'./personal-backend.mjs'",JSON.stringify(new URL('../src/personal-backend.mjs',import.meta.url).href));
const {default:worker,passwordHash,constantEqual,verifyStripeSignature,validateClaims,activeSubscription,claimJobs,completeJob,proxyJobLLM,failJob,runGenerationBatch}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
function database({legacy=false,v2=false}={}) {
 const db=new DatabaseSync(':memory:');
 for(const migration of ['0001.sql','0002_inspirations.sql',...(!legacy?['0003_likes_portraits.sql','0004_recommendations_free_portraits.sql','0005_personal_context_characters.sql','0006_recommendation_exposures.sql',...(!v2?['0007_self_profile_single_character.sql']:[]),'0008_billing_trial.sql','0009_private_companion.sql','0010_requested_focus.sql']:[])]) db.exec(readFileSync(new URL('../migrations/'+migration,import.meta.url),'utf8'));
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

test('unlimited likes and optional explanations invalidate source only on actual changes',async()=>{
 const env={DB:database()};const a=await register(env),b=await register(env,'b@example.com');await prepareUser(env,a,14);
 let me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.deepEqual(me.readiness,{likes_count:14,explained_count:0,target:10,ready:true});assert.equal(me.profile.display_name,undefined);const source=me.source_version;
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


test('private story previews use only existing definition prose and stay short',()=>{
 const ui=readFileSync(new URL('../src/ui.mjs',import.meta.url),'utf8');
 const start=ui.indexOf('function storyDefinitionPreview('),end=ui.indexOf('async function loadStory(',start);
 const preview=runInNewContext(ui.slice(start,end)+';storyDefinitionPreview');
 assert.equal(preview('Ouverture narrative.\n\n## Définition\n\nPremière phrase. Deuxième phrase ! Troisième phrase ? Quatrième phrase.\n\n## Autre partie\nContenu ignoré.'),'Première phrase. Deuxième phrase ! Troisième phrase ?');
 assert.equal(preview('Seulement une histoire sans définition.'),'');
 const long=preview('## Définition\n\n'+('Un mot très long dans une phrase. ').repeat(30));assert.ok(long.length<=450);assert.ok(long.split('.').length<=4);
 assert.equal(preview('## Définition\n\n**Une piste** avec [un exemple](https://example.test).'),'Une piste avec un exemple.');
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
function portraitV2(job,{pending=[]}={}){const sources=job.sources.filter(s=>!pending.includes(s.id)),refs=sources.slice(0,1).map(s=>s.id),explicit=sources.find(s=>s.explicit);return {schema_version:2,summary:'Des pistes à explorer depuis les éléments exprimés.',summary_evidence:refs,axes:[{label:'Confiance et liens',why:'Une piste exprimée ou un intérêt de lecture.',claim_type:sources[0].explicit?'hypothesis':'interest',confidence:sources[0].explicit?0.5:0.2,evidence:refs}],strengths:explicit?[{text:'Une ressource évoquée dans les liens.',evidence:[explicit.id]}]:[],uncertainties:['Ces éléments ne définissent pas toute votre vie.'],citations:sources.map(s=>({source_id:s.id,revision:s.revision,quote:s.text.slice(0,Math.min(120,s.text.length))})),coverage:{processed:sources.map(s=>s.id),pending}};}
const storyV2=()=>{const s=story();return {...s,narrative_delta:{events:[{text:'Le personnage tente une action.',quote:'Une fiction possible, pas une biographie.'}],relationships:[],open_questions:[],resolution:{text:'Une perspective reste ouverte.',quote:'Une fiction possible, pas une biographie.'}}};};
async function readyInputs(env,a){await prepareUser(env,a);const current=await (await request(env,'/api/self-profile',{cookie:a.cookie})).json();assert.equal((await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:current.revision,self_description:'Je souhaite apprendre à poser mes limites.'}})).status,200);}
async function setupV2(env,a){pay(env,a);await readyInputs(env,a);await addEvent(env,a);let j=await claim(env);assert.equal(j?.kind,'portrait');await complete(env,j,{portrait:portraitV2(j)});assert.equal(await claim(env),undefined,'No automatically generated stories');let r=await request(env,'/api/characters/prepare',{method:'POST',cookie:a.cookie,data:{}});assert.equal(r.status,202,JSON.stringify(await r.clone().json()));j=await claim(env);assert.equal(j?.kind,'characters');await complete(env,j,{characters:j.missing_slots.map(slot=>({slot,identity:{name:'Personnage '+slot,description:'Une identité fictive distincte '+slot},facet:{label:'Une perspective '+slot,description:'Explorer une piste avec prudence.',evidence:[j.sources.find(s=>s.type==='event')?.id||j.sources[0].id]},narrative_state:{questions:[]}}))});return (await (await request(env,'/api/characters',{cookie:a.cookie})).json()).characters;}
async function continueStory(env,a,ch,key=crypto.randomUUID()){return request(env,'/api/characters/'+ch.id+'/continue',{method:'POST',cookie:a.cookie,data:{idempotency_key:key,expected_parent_id:ch.last_story_id}});}
async function published(env,a,ch){const r=await continueStory(env,a,ch);assert.equal(r.status,202,JSON.stringify(await r.clone().json()));const j=await claim(env);assert.equal(j?.kind,'story');const out=await (await complete(env,j,storyV2())).json();return {j,id:out.story_id};}

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

test('one character identity is stable and only explicit continuations create stories',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);const chars=await setupV2(env,a);assert.equal(chars.length,1);assert.equal(new Set(chars.map(c=>c.id)).size,1);assert.equal(await claim(env),undefined);await request(env,'/api/characters/prepare',{method:'POST',cookie:a.cookie,data:{}});assert.equal(await claim(env),undefined);await addEvent(env,a);const j=await claim(env);await complete(env,j,{portrait:portraitV2(j)});const again=(await (await request(env,'/api/characters',{cookie:a.cookie})).json()).characters;assert.deepEqual(again.map(c=>[c.id,c.identity]),chars.map(c=>[c.id,c.identity]));assert.equal(await claim(env),undefined);});

test('seven shared weekly publications across characters reject an eighth with replay-safe completion',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);const chars=await setupV2(env,a);for(let i=0;i<7;i++){const ch=chars[0],{j,id}=await published(env,a,ch);const replay=await (await complete(env,j,storyV2())).json();assert.equal(replay.story_id,id);ch.last_story_id=id;}const r=await continueStory(env,a,chars[0]);assert.equal(r.status,429);assert.equal((await r.json()).code,'quota_exhausted');const state=await (await request(env,'/api/characters',{cookie:a.cookie})).json();assert.equal(state.quota.used,7);assert.equal(state.quota.remaining,0);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM stories').get().n,7);assert.equal(env.DB.raw.prepare("SELECT count(*) n FROM story_requests WHERE credit_status='consumed'").get().n,7);});

test('concurrent characters reserve one credit; same key replays and conflicting params fail',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);const chars=await setupV2(env,a),key=crypto.randomUUID();const first=await continueStory(env,a,chars[0],key),saved=(await first.json()).request;let r=await continueStory(env,a,chars[0],key);assert.equal((await r.json()).request.id,saved.id);r=await continueStory(env,a,{...chars[0],last_story_id:'conflicting-parent'},key);assert.equal(r.status,409);assert.equal((await r.json()).code,'idempotency_conflict');await request(env,'/api/story-requests/'+saved.id+'/cancel',{method:'POST',cookie:a.cookie,data:{}});const races=await Promise.all([1,2,3].map(()=>continueStory(env,a,chars[0])));assert.equal(races.filter(r=>r.status===202).length,1);assert.equal(races.filter(r=>r.status===409).length,2);const state=await (await request(env,'/api/characters',{cookie:a.cookie})).json();assert.equal(state.quota.reserved,1);assert.equal(state.quota.used,0);});

test('cancellation wins before publication and cannot undo an already published episode',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);const r=await continueStory(env,a,chars[0]),id=(await r.json()).request.id,j=await claim(env);let cancelled=await request(env,'/api/story-requests/'+id+'/cancel',{method:'POST',cookie:a.cookie,data:{}});assert.equal((await cancelled.json()).request.credit_status,'released');await assert.rejects(complete(env,j,storyV2()),e=>e.status===409);const next=await published(env,a,chars[0]);cancelled=await request(env,'/api/story-requests/'+next.id+'/cancel',{method:'POST',cookie:a.cookie,data:{}});const result=await cancelled.json();assert.equal(result.request.story_id,next.id);assert.equal(result.request.status,'completed');assert.equal(result.quota.used,1);});

test('source rebase keeps request, reserved week, attempts and provider budget and invalidates old lease',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);const r=await continueStory(env,a,chars[0]),saved=(await r.json()).request,j=await claim(env);env.DB.raw.prepare('UPDATE story_requests SET provider_calls=6,reserved_tokens=20000 WHERE id=?').run(saved.id);env.DB.raw.prepare('UPDATE journal_jobs SET provider_calls=6,reserved_tokens=20000 WHERE id=?').run(j.id);await addEvent(env,a);await assert.rejects(complete(env,j,storyV2()),e=>e.status===409);let row=env.DB.raw.prepare('SELECT * FROM story_requests WHERE id=?').get(saved.id);assert.equal(row.status,'waiting_portrait');assert.equal(row.provider_calls,6);assert.equal(row.attempts,1);assert.equal(row.week_start,saved.week_start);const p=await claim(env);assert.equal(p.kind,'portrait');await complete(env,p,{portrait:portraitV2(p)});const retry=await claim(env);assert.equal(retry.kind,'story');assert.equal(retry.request_id,saved.id);assert.equal(retry.request.provider_calls,6);assert.equal(retry.request.attempts,2);assert.notEqual(retry.lease_token,j.lease_token);await complete(env,retry,storyV2());});

test('subscription expiry, request expiry and expired leases cannot publish or retain credits',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);let r=await continueStory(env,a,chars[0]),id=(await r.json()).request.id,j=await claim(env);env.DB.raw.prepare('UPDATE journal_jobs SET lease_until=0 WHERE id=?').run(j.id);await assert.rejects(complete(env,j,storyV2()),e=>e.status===409);env.DB.raw.prepare('UPDATE story_requests SET expires_at=0 WHERE id=?').run(id);r=await request(env,'/api/story-requests/'+id,{cookie:a.cookie});assert.equal((await r.json()).request.status,'expired');r=await continueStory(env,a,chars[0]);id=(await r.json()).request.id;j=await claim(env);env.DB.raw.prepare('UPDATE subscriptions SET paid_until=0 WHERE user_id=?').run(a.user.id);await assert.rejects(complete(env,j,storyV2()),e=>e.status===409);r=await request(env,'/api/story-requests/'+id,{cookie:a.cookie});assert.equal((await r.json()).request.credit_status,'released');});

test('Paris calendar weeks handle midnight and both DST changes independently from 168 hours',()=>{assert.equal(parisWeek(new Date('2026-11-01T22:59:59Z')).week_start,'2026-10-26');assert.equal(parisWeek(new Date('2026-11-01T23:00:00Z')).week_start,'2026-11-02');assert.equal(parisWeek(new Date('2026-03-23T10:00:00Z')).renews_at,'2026-03-29T22:00:00.000Z');assert.equal(parisWeek(new Date('2026-10-19T10:00:00Z')).renews_at,'2026-10-25T23:00:00.000Z');});

test('an old-week reservation completing now consumes its original week',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);const r=await continueStory(env,a,chars[0]),saved=(await r.json()).request;env.DB.raw.prepare("UPDATE story_requests SET week_start='2020-01-06' WHERE id=?").run(saved.id);const j=await claim(env);await complete(env,j,storyV2());assert.equal(env.DB.raw.prepare('SELECT week_start FROM story_requests WHERE id=?').get(saved.id).week_start,'2020-01-06');const state=await (await request(env,'/api/characters',{cookie:a.cookie})).json();assert.equal(state.quota.used,0);});

test('understanding-only scene text cannot reach characters or story prompts through portrait or derivatives',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);pay(env,a);await readyInputs(env,a);await addEvent(env,a,{description:'PRIVATE FORBIDDEN SCENE 123456',fiction_use:'understanding_only'});await addEvent(env,a,{description:'Une ressource admissible.'});let j=await claim(env);await complete(env,j,{portrait:portraitV2(j)});await request(env,'/api/characters/prepare',{method:'POST',cookie:a.cookie,data:{}});j=await claim(env);assert.equal(j.kind,'characters');assert.equal(JSON.stringify(j).includes('PRIVATE FORBIDDEN SCENE'),false);assert.ok(j.sources.some(s=>s.restricted&&s.text===''));});

test('personal feedback is complete and separately scoped; no-op updates retain revision and privacy',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),b=await register(env,'b@example.com'),chars=await setupV2(env,a),{id}=await published(env,a,chars[0]);const data={resonance:2,why:'Ce qui me ressemble.',differences:'Une différence sans négation.',correction:'x'.repeat(150)+' Correction importante à la fin.',scope:'reaction',preference:'Plus de moments ordinaires.',avoid:'Éviter le deuil.'};let r=await request(env,'/api/stories/'+id+'/feedback',{method:'PUT',cookie:a.cookie,data});const saved=await r.json(),v=versionOf(env,a);assert.equal(saved.feedback.revision,1);r=await request(env,'/api/stories/'+id+'/feedback',{method:'PUT',cookie:a.cookie,data:{...data,revision:1}});assert.equal((await r.json()).feedback.revision,1);assert.equal(versionOf(env,a),v);assert.equal((await request(env,'/api/stories/'+id+'/feedback',{method:'PUT',cookie:b.cookie,data})).status,404);const j=await claim(env),f=j.sources.find(s=>s.type==='feedback');assert.ok(f.text.endsWith('avoid: Éviter le deuil.'));assert.ok(f.fields.correction.includes('Correction importante à la fin.'));assert.equal(f.fields.differences,data.differences);assert.equal(j.portrait_previous,null);});

test('incremental derivatives persist only exact revisions and partial portraits continue without unlocking fiction',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);pay(env,a);await addEvent(env,a);const e2=await addEvent(env,a);let j=await claim(env);const source=j.sources[0],cached={source_id:source.id,revision:source.revision,source_hash:source.hash,analysis_version:2,derivative:{signals:[{kind:'declaration',text:'Un élément déclaré.',quote:source.text.slice(0,60)}]},manifest:{}};await complete(env,j,{portrait:portraitV2(j,{pending:[j.sources[1].id]}),source_derivatives:[cached]});let p=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.notEqual(p.status,'ready');await request(env,'/api/characters/prepare',{method:'POST',cookie:a.cookie,data:{}});j=await claim(env);assert.equal(j.kind,'portrait');assert.equal(j.source_derivatives.length,1);await complete(env,j,{portrait:portraitV2(j)});p=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(p.status,'ready');const target=source.id.slice(6);await request(env,'/api/life-events/'+target,{method:'PATCH',cookie:a.cookie,data:{revision:1,description:'Une nouvelle interprétation.'}});assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM context_source_derivatives').get().n,0);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM user_portraits').get().n,0);});

test('exports include story IDs and relationships; deletion cascades every private V2 collection',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a),{id}=await published(env,a,chars[0]);const p=await addPerson(env,a);await addEvent(env,a,{people_ids:[p.id]});const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();assert.equal(exported.stories[0].id,id);assert.equal(exported.characters.length,1);assert.equal(exported.people.length,1);assert.ok(exported.events.some(e=>e.people_ids.includes(p.id)));const r=await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}});assert.equal(r.status,200);for(const table of ['stories','feedback','personal_people','life_events','life_event_people','personal_characters','story_requests','journal_jobs','context_mutations','provider_budgets'])assert.equal(env.DB.raw.prepare(`SELECT count(*) n FROM ${table}`).get().n,0,table);assert.deepEqual(env.DB.raw.prepare('PRAGMA foreign_key_check').all(),[]);});

test('legacy migration preserves exact story bodies, IDs, feedback and slot identity but invalidates V1 portrait',()=>{const db=new DatabaseSync(':memory:');for(const name of ['0001.sql','0002_inspirations.sql','0003_likes_portraits.sql','0004_recommendations_free_portraits.sql'])db.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));db.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','old@test','h','s'),('v','empty@test','h','s');INSERT INTO profiles(user_id) VALUES('u'),('v');INSERT INTO memories(user_id) VALUES('u'),('v');INSERT INTO subscriptions(user_id) VALUES('u'),('v');");db.prepare('INSERT INTO user_portraits(user_id,portrait,analyzed_version) VALUES(?,?,?)').run('u',JSON.stringify(portrait()),1);const day=parisWeek().week_start;for(let i=0;i<3;i++)db.prepare('INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice) VALUES(?,?,?,?,?,?,?,?)').run('story-'+i,'u',i===0?day:`2020-01-0${i+1}`,'Titre '+i,'Body exact à conserver '+i,'Fin '+i,'[]','');db.exec("INSERT INTO feedback(story_id,user_id,resonance,feelings,why,correction) VALUES('story-0','u',2,'émotion','texte à conserver','correction');");db.exec(readFileSync(new URL('../migrations/0005_personal_context_characters.sql',import.meta.url),'utf8'));assert.equal(db.prepare('SELECT count(*) n FROM stories').get().n,3);assert.equal(db.prepare("SELECT body FROM stories WHERE id='story-0'").get().body,'Body exact à conserver 0');assert.equal(db.prepare('SELECT correction FROM feedback').get().correction,'correction');assert.equal(db.prepare('SELECT count(*) n FROM user_portraits').get().n,0);assert.equal(JSON.parse(db.prepare('SELECT identity FROM personal_characters').get().identity).name,'Camille');assert.equal(db.prepare('SELECT count(*) n FROM story_requests WHERE week_start=?').get(day).n,1);assert.equal(db.prepare('SELECT count(*) n FROM users').get().n,2);assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(),[]);db.prepare("DELETE FROM users WHERE id='u'").run();assert.equal(db.prepare('SELECT count(*) n FROM feedback').get().n,0);});

test('generation pause blocks all new provider work while preserving private reading and edits',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);const r=await continueStory(env,a,chars[0]),j=await claim(env);env.GENERATION_PAUSED='true';assert.equal((await (await claimJobs(env)).json()).status,'generation_paused');assert.equal((await continueStory(env,a,chars[0])).status,503);await assert.rejects(complete(env,j,storyV2()),e=>e.status===503);await addEvent(env,a);assert.equal((await request(env,'/api/characters',{cookie:a.cookie})).status,200);});

test('canonical personal routes retain legacy query intent and private cache controls',async()=>{for(const base of ['/mes-histoires','/mes-histoires/','/journal','/journal/']){const r=await request({PUBLIC_ORIGIN:origin},base+'?story=old-id&personnage=stable');assert.equal(r.status,302);const location=new URL(r.headers.get('location'));assert.equal(location.pathname,'/mon-personnage/');assert.equal(location.search,'?story=old-id&personnage=stable');}const r=await request({},'/mon-personnage/');assert.equal(r.status,200);assert.match(r.headers.get('cache-control'),/private, no-store/);});

test('LLM reservations enforce allowed models, temperatures, cumulative twenty calls and uncertain failures',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'private-test-secret',OPENROUTER_MODEL:'approved/model'},a=await register(env),chars=await setupV2(env,a);const r=await continueStory(env,a,chars[0]),rid=(await r.json()).request.id,j=await claim(env),data={lease_token:j.lease_token,source_version:j.source_version,memory_version:j.memory_version,purpose:'analyst',system_prompt:'Trusted instructions',user_prompt:'Private sources',max_tokens:128,temperature:0.15};let calls=0;const original=globalThis.fetch;globalThis.fetch=async(url,options)=>{calls++;assert.equal(url,'https://openrouter.ai/api/v1/chat/completions');const sent=JSON.parse(options.body);assert.equal(sent.model,'approved/model');assert.equal(sent.temperature,0.15);return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:'A result'}}],usage:{prompt_tokens:12,completion_tokens:8,cost:0.001}}));};try{await assert.rejects(proxyJobLLM(env,j.id,{...data,model:'not-allowed'}),e=>e.status===400);await assert.rejects(proxyJobLLM(env,j.id,{...data,max_tokens:9000}),e=>e.status===400);assert.equal(calls,0);for(let i=0;i<20;i++)assert.equal((await (await proxyJobLLM(env,j.id,data)).json()).content,'A result');await assert.rejects(proxyJobLLM(env,j.id,data),e=>e.status===429);assert.equal(calls,20);const ledger=env.DB.raw.prepare('SELECT * FROM story_requests WHERE id=?').get(rid);assert.equal(ledger.provider_calls,20);assert.equal(ledger.input_tokens,240);assert.equal(ledger.output_tokens,160);assert.ok(ledger.reserved_tokens>2560);await failJob(env,j.id,{lease_token:j.lease_token,reason:'provider_error'});assert.equal(env.DB.raw.prepare('SELECT credit_status FROM story_requests WHERE id=?').get(rid).credit_status,'released');
 const next=await continueStory(env,a,chars[0]);assert.equal(next.status,202);const retry=await claim(env);globalThis.fetch=async()=>{calls++;throw new Error('lost reply');};await assert.rejects(proxyJobLLM(env,retry.id,{...data,lease_token:retry.lease_token}),e=>e.status===502);assert.equal(env.DB.raw.prepare('SELECT provider_calls FROM story_requests WHERE id=?').get(retry.request_id).provider_calls,1);assert.equal(env.DB.raw.prepare('SELECT provider_calls FROM provider_budgets WHERE user_id=?').get(a.user.id).provider_calls,21);
 }finally{globalThis.fetch=original;}});

test('temporary failure retries the same request at most three times and then frees its credit',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);const r=await continueStory(env,a,chars[0]),id=(await r.json()).request.id;for(let i=0;i<3;i++){const j=await claim(env);assert.equal(j.request_id,id);assert.equal(j.request.attempts,i+1);await failJob(env,j.id,{lease_token:j.lease_token,reason:'provider_error'});env.DB.raw.prepare('UPDATE journal_jobs SET lease_until=0 WHERE id=?').run(j.id);}assert.equal(await claim(env),undefined);const row=env.DB.raw.prepare('SELECT status,credit_status,attempts FROM story_requests WHERE id=?').get(id);assert.deepEqual({...row},{status:'failed',credit_status:'released',attempts:3});});

test('a profile mutation between preflight and publication cannot insert even one stale story',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);await continueStory(env,a,chars[0]);const j=await claim(env),batch=env.DB.batch.bind(env.DB);let inject=true;env.DB.batch=statements=>{if(inject){inject=false;env.DB.raw.prepare('UPDATE profiles SET version=version+1 WHERE user_id=?').run(a.user.id);}return batch(statements);};await assert.rejects(complete(env,j,storyV2()),e=>e.status===409);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM stories').get().n,0);const row=env.DB.raw.prepare('SELECT status,credit_status FROM story_requests WHERE id=?').get(j.request_id);assert.equal(row.status,'waiting_portrait');assert.equal(row.credit_status,'reserved');assert.equal(env.DB.raw.prepare('SELECT canon_version FROM personal_characters WHERE id=?').get(chars[0].id).canon_version,0);});

test('two simultaneous complete calls return one story and one debit',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);await continueStory(env,a,chars[0]);const j=await claim(env);const results=await Promise.all([complete(env,j,storyV2()),complete(env,j,storyV2())]);const bodies=await Promise.all(results.map(r=>r.json()));assert.equal(bodies[0].story_id,bodies[1].story_id);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM stories').get().n,1);assert.equal(env.DB.raw.prepare("SELECT count(*) n FROM story_requests WHERE credit_status='consumed'").get().n,1);});

test('foreign character, story, request and event links are rejected at both API and SQLite boundaries',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a),b=await register(env,'private-b@example.com');let r=await continueStory(env,b,chars[0]);assert.equal(r.status,404);r=await continueStory(env,a,chars[0]);const id=(await r.json()).request.id;assert.equal((await request(env,'/api/story-requests/'+id,{cookie:b.cookie})).status,404);assert.equal((await request(env,'/api/story-requests/'+id+'/cancel',{method:'POST',cookie:b.cookie,data:{}})).status,404);const p=await addPerson(env,a),e=await addEvent(env,b);assert.throws(()=>env.DB.raw.prepare('INSERT INTO life_event_people(event_id,person_id,user_id) VALUES(?,?,?)').run(e.id,p.id,b.user.id),/FOREIGN KEY/);assert.equal(env.DB.raw.prepare('SELECT status FROM story_requests WHERE id=?').get(id).status,'waiting_portrait');});

test('free portrait starts remain bounded after personal form edits and do not depend on likes',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);await addEvent(env,a);for(let i=0;i<3;i++){const j=await claim(env);assert.equal(j.kind,'portrait');await complete(env,j,{portrait:portraitV2(j)});await addEvent(env,a);if(i<2){assert.equal(await claim(env),undefined);env.DB.raw.prepare('UPDATE portrait_usage SET last_started=last_started-601 WHERE user_id=?').run(a.user.id);}}assert.equal(await claim(env),undefined);const state=await (await request(env,'/api/portrait',{cookie:a.cookie})).json();assert.equal(state.reason,'quota');assert.equal(state.status,'queued');assert.ok(state.retry_after>Math.floor(Date.now()/1000));});

test('character history remains separate after more than thirty episodes and no catalogue entry blocks the queue',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);const db=env.DB.raw;for(let i=1;i<=35;i++){db.prepare('INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice,character_id,episode_number,parent_story_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)').run('historic-'+i,a.user.id,'2020-01-01','History '+i,'Body','Resolution '+i,'[]','',chars[0].id,i,i===1?null:'historic-'+(i-1));}db.prepare('UPDATE personal_characters SET canon_version=35,last_story_id=? WHERE id=?').run('historic-35',chars[0].id);chars[0].last_story_id='historic-35';db.exec('DELETE FROM public_examples');await continueStory(env,a,chars[0]);const j=await claim(env);assert.equal(j.kind,'story');assert.equal(j.history.length,35);assert.equal(j.history[0].summary,'Resolution 1');assert.equal(j.history.at(-1).summary,'Resolution 35');assert.deepEqual(j.inspirations,[]);const publishedStory=await (await complete(env,j,storyV2())).json();chars[0].last_story_id=publishedStory.story_id;await continueStory(env,a,chars[0]);const j2=await claim(env);assert.equal(j2.history.length,36);assert.equal(j2.character.id,chars[0].id);});

test('partial dates reject zero segments without inventing missing day or month',async()=>{const env={DB:database()},a=await register(env);for(const [date_precision,date_start]of [['month','2020-00'],['day','2020-06-00'],['day','2021-02-29']]){const r=await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),title:'Date',description:'Contexte',date_precision,date_start}});assert.equal(r.status,400,date_start);}const e=await addEvent(env,a,{date_precision:'month',date_start:'2020-06'});assert.equal(e.date_start,'2020-06');const undated=await addEvent(env,a,{date_precision:'unknown',date_start:''});assert.equal(undated.date_start,'');});

test('a late failure cannot requeue a rebased request or touch a replacement lease',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);await continueStory(env,a,chars[0]);const j=await claim(env),batch=env.DB.batch.bind(env.DB);let inject=true;env.DB.batch=statements=>{if(inject){inject=false;env.DB.raw.prepare('UPDATE profiles SET version=version+1 WHERE user_id=?').run(a.user.id);}return batch(statements);};await assert.rejects(failJob(env,j.id,{lease_token:j.lease_token,reason:'provider_error'}),e=>e.status===409);const r=env.DB.raw.prepare('SELECT status,credit_status,error_code,attempts FROM story_requests WHERE id=?').get(j.request_id);assert.equal(r.status,'waiting_portrait');assert.equal(r.error_code,'context_changed');assert.equal(r.credit_status,'reserved');assert.equal(r.attempts,1);});

test('source deletion neutralizes obsolete facet prose in cards and future generation payloads while keeping identities',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);const ch=env.DB.raw.prepare('SELECT facet FROM personal_characters WHERE id=?').get(chars[0].id),facet=JSON.parse(ch.facet);facet.description='PRIVATE OLD FACET WORDING';env.DB.raw.prepare('UPDATE personal_characters SET facet=? WHERE id=?').run(JSON.stringify(facet),chars[0].id);const eventId=facet.evidence[0].slice(6);await addEvent(env,a,{description:'Une nouvelle source indépendante.'});await request(env,'/api/life-events/'+eventId,{method:'DELETE',cookie:a.cookie,data:{revision:1}});const state=await (await request(env,'/api/characters',{cookie:a.cookie})).json();assert.equal(JSON.stringify(state).includes('PRIVATE OLD FACET WORDING'),false);assert.deepEqual(state.characters.map(c=>c.identity),chars.map(c=>c.identity));const p=await claim(env);await complete(env,p,{portrait:portraitV2(p)});await continueStory(env,a,chars[0]);const j=await claim(env);assert.equal(j.kind,'story');assert.equal(JSON.stringify(j).includes('PRIVATE OLD FACET WORDING'),false);assert.equal(j.character.facet.outdated,true);});

test('reading-context snapshots retry concurrent source edits instead of mixing source revisions',async()=>{const {loadContext}=await import('../src/private-context.mjs');const env={DB:database()},a=await register(env);await addEvent(env,a);const before=versionOf(env,a),prepare=env.DB.prepare.bind(env.DB);let injected=false;env.DB.prepare=sql=>{const statement=prepare(sql);if(sql.includes('FROM life_events e')){const all=statement.all.bind(statement);statement.all=async()=>{const result=await all();if(!injected){injected=true;env.DB.raw.prepare('UPDATE memories SET text=?,version=version+1 WHERE user_id=?').run('A new source arriving during reading.',a.user.id);env.DB.raw.prepare('UPDATE profiles SET version=version+1 WHERE user_id=?').run(a.user.id);}return result;};}return statement;};const c=await loadContext(env,a.user.id);assert.equal(c.source_version,before+1);assert.equal(c.memory.text,'A new source arriving during reading.');assert.ok(c.sources.some(s=>s.type==='memory'&&s.revision===2));});

test('the bounded five-hundred-event collection remains complete without a giant aggregate cell',async()=>{const {loadContext}=await import('../src/private-context.mjs');const env={DB:database()},a=await register(env);const insert=env.DB.raw.prepare('INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description) VALUES(?,?,?,?,?,?,?)');for(let i=0;i<500;i++)insert.run('event-'+i,a.user.id,'creation-'+i,'hash','Event '+i,'other','x'.repeat(5970)+' END MARKER '+i);const c=await loadContext(env,a.user.id);assert.equal(c.events.length,500);assert.equal(c.sources.length,500);assert.ok(c.sources.find(s=>s.id==='event:event-499').text.endsWith('END MARKER 499'));const r=await request(env,'/api/life-events',{method:'POST',cookie:a.cookie,data:{creation_id:crypto.randomUUID(),title:'Too many',description:'No silent loss'}});assert.equal(r.status,400);assert.equal((await r.json()).code,'collection_limit');});

test('recommendation exposure stores bounded public IDs without changing personal source versions',async()=>{const env={DB:database()},a=await register(env);await like(env,a,0,'Je souhaite explorer la confiance et les relations.');const v=versionOf(env,a);for(let i=0;i<150;i++)env.DB.raw.prepare('INSERT INTO recommendation_exposures(user_id,article_id,shown_at) VALUES(?,?,?)').run(a.user.id,'old-'+i,i<20?'2020-01-01 00:00:00':'2026-11-03 00:00:00');const r=await request(env,'/api/recommendations',{cookie:a.cookie});assert.equal(r.status,200);assert.equal(versionOf(env,a),v);assert.ok(env.DB.raw.prepare('SELECT count(*) n FROM recommendation_exposures WHERE user_id=?').get(a.user.id).n<=100);assert.equal(env.DB.raw.prepare("SELECT count(*) n FROM recommendation_exposures WHERE shown_at<'2021'").get().n,0);});

test('an understanding-only event can build a portrait but cannot bypass V3 character requirements',async()=>{const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);pay(env,a);await addEvent(env,a,{fiction_use:'understanding_only'});const j=await claim(env);await complete(env,j,{portrait:portraitV2(j)});const r=await request(env,'/api/characters/prepare',{method:'POST',cookie:a.cookie,data:{}});assert.equal(r.status,409);assert.equal((await r.json()).code,'likes_required');assert.equal(await claim(env),undefined);});

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

test('demographics and occupation alone provide no psychological evidence or character eligibility',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env);pay(env,a);await prepareUser(env,a);
 await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:1,occupation:'Psychologue',gender:'female',birth_date:'1980-01-01'}});
 const j=await claim(env);assert.equal(j.sources.some(s=>s.type==='profile'),false);await complete(env,j,{portrait:portraitV2(j)});
 const state=await (await request(env,'/api/characters',{cookie:a.cookie})).json();assert.equal(state.readiness.likes_count,10);assert.equal(state.readiness.has_self_description,false);assert.deepEqual(state.readiness.reasons,['self_description_required']);assert.equal((await request(env,'/api/characters/prepare',{method:'POST',cookie:a.cookie,data:{}})).status,409);
});

test('V3 migration preserves three identities and histories, archives secondary jobs and retains consumed quotas',async()=>{
 const env={DB:database({v2:true})},a=await register(env),db=env.DB.raw;
 for(let slot=1;slot<=3;slot++)db.prepare('INSERT INTO personal_characters(id,user_id,slot,identity) VALUES(?,?,?,?)').run('char-'+slot,a.user.id,slot,JSON.stringify({name:'Identity '+slot}));
 const week=parisWeek().week_start;
 db.prepare("INSERT INTO story_requests(id,user_id,character_id,idempotency_key,parameter_hash,week_start,status,credit_status,source_version,canon_version,expires_at) VALUES('spent',?,'char-2','spent-key','hash',?,'completed','consumed',1,0,0),('pending',?,'char-3','pending-key','hash',?,'running','reserved',1,0,9999999999)").run(a.user.id,week,a.user.id,week);
 db.prepare("INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice,character_id,request_id,episode_number) VALUES('historic',?,'2020-01-01','Titre','Exact archived body','Résumé','[]','','char-2','spent',1)").run(a.user.id);
 db.prepare("INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,request_id,character_id,status,lease_token,lease_until) VALUES('secondary-job',?,'story',1,1,'pending','char-3','running','old-token',9999999999)").run(a.user.id);
 db.exec(readFileSync(new URL('../migrations/0007_self_profile_single_character.sql',import.meta.url),'utf8'));
 const state=await (await request(env,'/api/characters',{cookie:a.cookie})).json();assert.deepEqual(state.characters.map(c=>c.id),['char-1']);assert.equal(state.archived_characters.length,2);assert.ok(state.archived_characters.every(c=>c.read_only));assert.equal(state.quota.used,1);assert.equal(state.quota.reserved,0);assert.equal(db.prepare("SELECT body FROM stories WHERE id='historic'").get().body,'Exact archived body');assert.equal(db.prepare("SELECT status FROM journal_jobs WHERE id='secondary-job'").get().status,'obsolete');
 assert.equal((await request(env,'/api/characters/char-2/continue',{method:'POST',cookie:a.cookie,data:{idempotency_key:'archive-attempt',expected_parent_id:null}})).status,409);assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(),[]);
});

test('loss of eligibility between claim and completion releases reservation and prevents publication',async()=>{
 for(const remove of ['likes','description']){const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);await continueStory(env,a,chars[0]);const j=await claim(env);
 if(remove==='likes')await request(env,'/api/likes/'+encodeURIComponent(examples[0].id),{method:'PUT',cookie:a.cookie,data:{liked:false}});else await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:2,self_description:''}});
 await assert.rejects(complete(env,j,storyV2()),e=>e.status===409);const state=await (await request(env,'/api/characters',{cookie:a.cookie})).json();assert.equal(state.readiness.eligible,false);assert.equal(state.quota.reserved,0);assert.equal(state.quota.used,0);assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM stories').get().n,0);
 }
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

test('V3 source edits never resurrect multiple terminal historical character jobs',async()=>{
 const env={DB:database({v2:true})},a=await register(env),db=env.DB.raw;
 for(const id of ['failed-one','failed-two'])db.prepare("INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,status) VALUES(?,?,'characters',1,1,'failed')").run(id,a.user.id);
 db.exec(readFileSync(new URL('../migrations/0007_self_profile_single_character.sql',import.meta.url),'utf8'));
 const r=await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:1,self_description:'Je me présente après la migration.'}});assert.equal(r.status,200);assert.equal((await r.json()).revision,2);
 assert.deepEqual(db.prepare('SELECT status FROM journal_jobs ORDER BY id').all().map(x=>x.status),['obsolete','obsolete']);
 db.prepare("INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,status) VALUES('new-live',?,'characters',2,1,'pending')").run(a.user.id);
 const second=await request(env,'/api/self-profile',{method:'PUT',cookie:a.cookie,data:{expected_revision:2,self_description:'Une description actualisée.'}});assert.equal(second.status,200);
 assert.equal(db.prepare("SELECT count(*) n FROM journal_jobs WHERE kind='characters' AND status='pending'").get().n,1);assert.equal(db.prepare("SELECT count(*) n FROM journal_jobs WHERE status='obsolete'").get().n,2);
});

test('trial entitlement uses trial end exclusively and preserves full story lifecycle',async()=>{
 const t=Math.floor(Date.now()/1000);
 assert.equal(activeSubscription({status:'trialing',trial_end:t+100,paid_until:0}),true);
 assert.equal(activeSubscription({status:'trialing',trial_end:t,paid_until:t+999}),false);
 for(const status of ['none','canceled','past_due','unpaid','paused','incomplete'])assert.equal(activeSubscription({status,trial_end:t+999,paid_until:t+999}),false);
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),chars=await setupV2(env,a);
 env.DB.raw.prepare("UPDATE subscriptions SET status='trialing',trial_end=?,paid_until=0,trial_used_at=? WHERE user_id=?").run(t+604800,t,a.user.id);
 const me=await (await request(env,'/api/me',{cookie:a.cookie})).json();assert.equal(me.subscription.active,true);assert.equal(me.subscription.trial_eligible,false);assert.equal(me.subscription.access_until,t+604800);
 let r=await continueStory(env,a,chars[0]);assert.equal(r.status,202);const id=(await r.json()).request.id,j=await claim(env);assert.equal(j.kind,'story');await complete(env,j,storyV2());
 assert.equal(env.DB.raw.prepare('SELECT credit_status FROM story_requests WHERE id=?').get(id).credit_status,'consumed');
 const latest=env.DB.raw.prepare('SELECT last_story_id FROM personal_characters WHERE id=?').get(chars[0].id);r=await continueStory(env,a,{...chars[0],last_story_id:latest.last_story_id});assert.equal(r.status,202);const pending=(await r.json()).request.id;
 env.DB.raw.prepare('UPDATE subscriptions SET trial_end=0 WHERE user_id=?').run(a.user.id);
 r=await request(env,'/api/story-requests/'+pending,{cookie:a.cookie});assert.equal((await r.json()).request.credit_status,'released');
 assert.equal(env.DB.raw.prepare("SELECT count(*) AS n FROM story_requests WHERE user_id=? AND credit_status='consumed'").get(a.user.id).n,1);
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

test('approved reading focus reaches story jobs and is part of idempotency',async()=>{
 const env={DB:database(),OPENROUTER_API_KEY:'test'},a=await register(env),[ch]=await setupV2(env,a);
 const payload={idempotency_key:'focus-specific-request',expected_parent_id:ch.last_story_id,requested_focus:'Comprendre la peur de décevoir et poser une limite.'};
 let r=await request(env,'/api/characters/'+ch.id+'/continue',{method:'POST',cookie:a.cookie,data:payload});assert.equal(r.status,202);
 const original=(await r.json()).request.id;
 r=await request(env,'/api/characters/'+ch.id+'/continue',{method:'POST',cookie:a.cookie,data:payload});assert.equal((await r.json()).request.id,original);
 r=await request(env,'/api/characters/'+ch.id+'/continue',{method:'POST',cookie:a.cookie,data:{...payload,requested_focus:'Autre angle'}});assert.equal(r.status,409);
 const j=await claim(env);assert.equal(j.kind,'story');assert.equal(j.requested_focus,payload.requested_focus);assert.equal((await (await request(env,'/api/characters',{cookie:a.cookie})).json()).quota.reserved,1);
});
