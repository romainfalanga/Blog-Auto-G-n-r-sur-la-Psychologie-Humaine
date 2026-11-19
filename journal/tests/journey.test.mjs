import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {buildReviewSystemPrompt,buildCompanionSystemPrompt} from '../src/psychology-framework.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
import {parisMonth,readMonthlyQuotas} from '../src/monthly-quotas.mjs';

function migrated(){
 const raw=new DatabaseSync(':memory:');
 for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 return raw;
}
function d1(raw){
 const prepare=sql=>{let args=[];return {bind(...a){args=a;return this;},async first(){return raw.prepare(sql).get(...args)||null;},async all(){return {results:raw.prepare(sql).all(...args)};},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}};}};};
 let queue=Promise.resolve();
 return {raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());raw.exec('COMMIT');return r;}catch(e){raw.exec('ROLLBACK');throw e;}});queue=next.catch(()=>{});return next;}};
}
// engineEnabled:true is the production path (worker.mjs); the default replays the legacy generate() path.
function setup(users=['a','b'],envExtra={},{engineEnabled=false}={}){
 const raw=migrated(),DB=d1(raw);let clock=Date.parse('2026-11-17T12:00:00Z')/1000,serial=0;
 const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code});},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null,recommend:()=>[]}};
 const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,engineEnabled}),env={DB,OPENROUTER_API_KEY:'mock-only',...envExtra};
 for(const id of users){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.com');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES(?,'active',?)").run(id,clock+366*86400);}
 async function call(path,method='GET',data,user='a',ctx){const req=new Request('https://example.com'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:user},url.pathname,url,ctx)||await personal.routes(req,env,{id:user},url.pathname,url,ctx);return {status:res.status,...await res.json()};}catch(e){return {status:e.status||500,code:e.code,message:e.message};}}
 return {raw,env,chat,personal,call,advance:seconds=>{clock+=seconds;},now:()=>clock};
}
const addPersonOnly=(s,user='a')=>s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES(?,?,?,'hash','Mila','friend','Nous nous parlons souvent, elle m’écoute.')").run('person-'+user,user,'creation-'+user);
function seedReady(s,user='a'){
 seedCompanionPrerequisites(s.raw,user);
 // Keep the narrative-bearing person and exactly the required three people.
 if(s.raw.prepare('SELECT id FROM personal_people WHERE id=?').get('person-'+user))s.raw.prepare('DELETE FROM personal_people WHERE id=?').run('ready-person-'+user+'-1');
}
function addPerson(s,user='a'){addPersonOnly(s,user);seedReady(s,user)}
const renewal=s=>Date.parse(parisMonth(s.now()).end)/1000;
const reviewText=(label='Ce qui semble ressortir.')=>label+' '+'Vous écrivez que vous vous taisez souvent pour éviter un conflit, et cela peut se relier à ce que vous avez coché. '.repeat(22)+'Qu’avez-vous envie de comprendre ou de changer en priorité ?';
const reviewReply=(content=reviewText())=>new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({content})},finish_reason:'stop'}],usage:{prompt_tokens:900,completion_tokens:1100,cost:0.02}}));
const chatReply=(content='Je vous écoute. Dans quelle situation cela arrive-t-il ?')=>new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({content,questions:[],proposals:[],cartography_notes:[]})},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:100,cost:0.001}}));
// Lot A5-P1: every review is checked by a separate call before publication. These tests count the
// writing calls; the check passes here and is tested on its own in v5-A5-P1-reviews.test.mjs.
const CHECK_PASS={criteria:{no_diagnosis:'yes',quotes_faithful:'yes',safety_resources:'na',hypotheses_open:'yes',no_saved_claim:'yes',sources_honest:'yes'},reasons:[]};
const isCheck=options=>{try{return JSON.parse(options.body).response_format?.json_schema?.name==='ambre_review_check'}catch{return false}};
const checkReply=()=>new Response(JSON.stringify({choices:[{message:{content:JSON.stringify(CHECK_PASS)},finish_reason:'stop'}],usage:{prompt_tokens:0,completion_tokens:0,cost:0}}));
const checked=fn=>async(url,options)=>isCheck(options)?checkReply():fn(url,options);
function withFetch(t,fn){const previous=globalThis.fetch;globalThis.fetch=checked(fn);t.after(()=>{globalThis.fetch=previous;});}
const contextOf=request=>JSON.parse(request.messages[1].content.slice(request.messages[1].content.indexOf('{')));
const review=(s,id)=>s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(id);
const start=(s,key='start_key_001',user='a',ctx)=>s.call('/api/chat/start','POST',{idempotency_key:key,accepted:true},user,ctx);

test('the journey (legacy generate() path, inactive in production): prerequisites, initial and point share a Paris calendar month, history and export',async t=>{
 const s=setup(),requests=[];let reply=reviewReply;
 withFetch(t,async(_url,options)=>{requests.push(JSON.parse(options.body));return reply();});
 let state=await s.call('/api/chat');
 const {readiness,quotas,...emptyJourney}=state.journey;
 assert.deepEqual(emptyJourney,{started:false,sources:{cartography:{done:false,count:0},portrait:{done:false},people:{done:false,count:0},life_events:{done:false,count:0}},can_start:false,review:{running_id:null,last_at:null,next_point_at:null}});
 assert.deepEqual(readiness.missing,['identity','people','events','childhood']);
 assert.deepEqual(readiness.identity.missing_fields,['first_name','last_name','birth_date','gender']);assert.equal(readiness.childhood.required,60);
 assert.equal(quotas.messages.limit,100);assert.equal(quotas.reviews.limit,1);assert.equal(quotas.reviews.remaining,1);
 assert.equal(state.access.reason,'prerequisites_required');
 let r=await start(s);
 assert.equal(r.status,403);assert.equal(r.code,'prerequisites_required');
 assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_settings').get().n,0,'a refused launch records no agreement');
 assert.equal((await s.call('/api/chat/start','POST',{idempotency_key:'start_key_001'})).code,'invalid_consent');
 assert.equal((await s.call('/api/chat/start','POST',{idempotency_key:'short',accepted:true})).code,'idempotency_required');
 addPersonOnly(s);
 s.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({first_name:'Alix',self_description:'Je me tais souvent pour éviter les conflits.'}));
 const [f1,f2]=[...new Set(CARTOGRAPHIE.links.filter(l=>l.strength==='forte'&&!l.definitional).map(l=>l.factor))];
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,consent,locked_at,version,updated_at) VALUES('a',?,'{}',?,?,1,1,3,1)").run(JSON.stringify({[f1]:2,[f2]:1}),JSON.stringify({factors:{[f1]:'Ce que j’en retiens.'},outcomes:{}}),JSON.stringify({[CARTOGRAPHIE.protections[0].id]:1}));
 state=await s.call('/api/chat');
 assert.deepEqual(state.journey.sources,{cartography:{done:true,count:2},portrait:{done:true},people:{done:true,count:1},life_events:{done:false,count:0}});
 assert.equal(state.journey.can_start,false);assert.equal(state.journey.started,false);
 r=await start(s);assert.equal(r.status,403);assert.equal(r.code,'prerequisites_required');assert.equal(requests.length,0);
 assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_settings').get().n,0,'partial sources must not record agreement');
 seedReady(s);state=await s.call('/api/chat');
 assert.equal(state.journey.can_start,true);assert.equal(state.journey.readiness.complete,true);
 assert.deepEqual(state.journey.readiness.identity,{complete:true,missing_fields:[]});
 assert.deepEqual(state.journey.readiness.people,{complete:true,count:3,required:3});assert.deepEqual(state.journey.readiness.events,{complete:true,count:3,required:3});
 assert.equal(state.journey.readiness.childhood.answered,60);assert.equal(state.journey.readiness.childhood.consent,true);

 assert.equal((await s.call('/api/chat/review','POST',{idempotency_key:'point_key_000'})).code,'journey_not_started');
 await s.call('/api/chat/consent','PUT',{accepted:true});
 r=await s.call('/api/chat/messages','POST',{content:'Bonjour Ambre.',idempotency_key:'early_message_1'});
 assert.equal(r.status,403);assert.equal(r.code,'journey_not_started');assert.equal((await s.call('/api/chat')).access.reason,'journey_not_started');
 await s.call('/api/chat/consent','PUT',{accepted:false});
 assert.equal(requests.length,0);

 r=await start(s);
 assert.equal(r.status,202);assert.equal(typeof r.review_id,'string');
 const initialId=r.review_id;
 assert.equal(r.consent,true,'launching records the conversation agreement');
 assert.equal(r.journey.started,true);assert.equal(r.journey.can_start,false);assert.equal(r.journey.review.running_id,null);
 assert.equal(r.journey.review.last_at,s.now());assert.equal(r.journey.review.next_point_at,Date.parse('2026-11-30T23:00:00Z')/1000);
 assert.equal(r.access.allowed,true);assert.equal(r.quota.used,0,'a review is not a message of the monthly quota');
 assert.equal(r.quotas.reviews.used,1);assert.equal(r.quotas.reviews.reserved,0);assert.equal(r.quotas.reviews.remaining,0);
 assert.deepEqual(r.messages.map(m=>[m.role,m.review_id,m.review_kind]),[['assistant',initialId,'initial']]);assert.equal(r.messages[0].content,reviewText());
 assert.equal(requests.length,1);
 const initialRequest=requests[0],initialContext=contextOf(initialRequest);
 assert.match(initialRequest.messages[0].content,/ANALYSE INITIALE/);assert.doesNotMatch(initialRequest.messages[0].content,/POINT D’ÉTAPE/);
 assert.equal(initialRequest.max_tokens,4800,'A5-P1: the text and its digest');assert.equal(initialRequest.provider.zdr,true);assert.equal(initialRequest.provider.data_collection,'deny');assert.equal(initialRequest.response_format.json_schema.strict,true);
 assert.deepEqual(initialContext.available,{cartography:{done:true,count:2},portrait:{done:true},people:{done:true,count:3},life_events:{done:true,count:3}});
 assert.equal(initialContext.kind,'initial');assert.equal(initialContext.childhood_cartography.cause_count,2);assert.equal(initialContext.childhood_cartography.person_notes[0].text,'Ce que j’en retiens.');assert.equal(initialContext.childhood_cartography.resources.length,1);
 assert.ok(initialContext.sources.some(x=>x.type==='profile'));assert.deepEqual(initialContext.people.map(p=>p.name).sort(),['Mila','Proche 2','Proche 3']);assert.equal(initialContext.current_identity.first_name,'Alix');
 for(const key of ['reading_candidates','catalogue_lookup','conversation_since','previous_review'])assert.equal(Object.hasOwn(initialContext,key),false,key);
 assert.ok(JSON.stringify(initialContext).length<=34000);
 let row=review(s,initialId);
 assert.equal(row.status,'ready');assert.equal(row.kind,'initial');assert.equal(row.attempts,1);assert.equal(row.lease_token,null);assert.equal(row.completed_at,s.now());
 assert.deepEqual(JSON.parse(row.sources),{cartography:{present:true,version:3,causes:2},portrait:true,people:3,life_events:3,messages:0,previous_review_id:null,check:{passed:true,rewritten:false,failed:[]}});
 const budget=s.raw.prepare("SELECT * FROM provider_budgets WHERE user_id='a'").get();
 assert.equal(budget.provider_calls,2,'the review and its separate check');assert.equal(budget.input_tokens,900);assert.equal(budget.output_tokens,1100);assert.ok(budget.reserved_tokens>3500);

 r=await start(s);assert.equal(r.status,200);assert.equal(r.review_id,initialId);assert.equal(requests.length,1,'the same key never generates twice');
 r=await start(s,'start_key_002');assert.equal(r.status,409);assert.equal(r.code,'journey_started');
 assert.equal((await s.call('/api/chat','GET',undefined,'b')).journey.started,false);

 reply=chatReply;
 r=await s.call('/api/chat/messages','POST',{content:'J’aimerais comprendre pourquoi je me tais en réunion.',idempotency_key:'first_message_1'});
 assert.equal(r.status,200);assert.equal(r.quota.used,1);assert.equal(r.messages.length,3);
 const chatRequest=requests.at(-1),chatContext=contextOf(chatRequest);
 assert.match(chatRequest.messages[0].content,/PARCOURS\./);assert.match(chatRequest.messages[0].content,/une seule à la fois/);
 assert.equal(chatContext.last_review.kind,'initial');assert.ok(chatContext.last_review.text.length<=3900);assert.ok(chatContext.last_review.text.startsWith('Ce qui semble ressortir.'));
 assert.ok(chatRequest.messages.slice(2).every(m=>m.content!==''),'a review is not replayed as an empty dialogue turn');
 assert.equal(chatRequest.messages.length,3);

 r=await s.call('/api/chat/review','POST',{idempotency_key:'point_key_001'});
 assert.equal(r.status,429);assert.equal(r.code,'review_quota_exhausted');assert.equal(r.retry_at,renewal(s));assert.equal(typeof r.error,'string');
 assert.equal(s.raw.prepare('SELECT count(*) n FROM ambre_reviews').get().n,1);
 const monthEnd=renewal(s),callsBeforeRenewal=requests.length;
 s.advance(monthEnd-s.now()-1);
 r=await s.call('/api/chat/review','POST',{idempotency_key:'point_last_second'});assert.equal(r.status,429);assert.equal(r.code,'review_quota_exhausted');assert.equal(requests.length,callsBeforeRenewal);
 s.advance(1);
 state=await s.call('/api/chat');assert.equal(state.journey.review.next_point_at,null);assert.equal(state.quotas.reviews.remaining,1);assert.equal(state.quota.used,0);assert.equal(state.quotas.reviews.period.id,'2026-12');

 reply=()=>reviewReply(reviewText('Ce qui a évolué depuis le dernier bilan.'));
 r=await s.call('/api/chat/review','POST',{idempotency_key:'point_key_001'});
 assert.equal(r.status,202);const pointId=r.review_id;assert.notEqual(pointId,initialId);
 assert.deepEqual(r.messages.map(m=>m.review_kind||m.role),['initial','user','assistant','point']);
 assert.equal(r.journey.review.last_at,s.now());assert.equal(r.journey.review.next_point_at,Date.parse('2026-12-31T23:00:00Z')/1000);assert.equal(r.quota.used,0);assert.equal(r.quotas.reviews.used,1);
 const pointRequest=requests.at(-1),pointContext=contextOf(pointRequest);
 assert.match(pointRequest.messages[0].content,/POINT D’ÉTAPE/);
 assert.equal(pointContext.previous_review.kind,'initial');assert.ok(pointContext.previous_review.text.startsWith('Ce qui semble ressortir.'));
 assert.deepEqual(pointContext.conversation_since,[{vous:'J’aimerais comprendre pourquoi je me tais en réunion.',ambre:'Je vous écoute.'}]);
 assert.deepEqual(pointContext.conversation_coverage,{included_turns:1,total_turns:1,truncated:false});
 assert.deepEqual(JSON.parse(review(s,pointId).sources),{cartography:{present:true,version:3,causes:2},portrait:true,people:3,life_events:3,messages:1,previous_review_id:initialId,thread_reviews:1,check:{passed:true,rewritten:false,failed:[]}});
 assert.equal((await s.call('/api/chat/review','POST',{idempotency_key:'point_key_002'})).code,'review_quota_exhausted');
 const beforeReplay=requests.length;const replay=await s.call('/api/chat/review','POST',{idempotency_key:'point_key_001'});assert.equal(replay.status,200);assert.equal(replay.review_id,pointId);assert.equal(requests.length,beforeReplay,'replaying the point reuses the consumed monthly credit');

 r=await s.call('/api/reviews');
 assert.equal(r.status,200);assert.deepEqual(r.reviews.map(x=>[x.id,x.kind,x.status]),[[pointId,'point','ready'],[initialId,'initial','ready']]);assert.equal(r.next_cursor,null);
 assert.ok(r.reviews.every(x=>!Object.hasOwn(x,'content')&&!Object.hasOwn(x,'sources')&&x.excerpt.length<=220&&x.length>1200));
 r=await s.call('/api/reviews?limit=1');assert.deepEqual(r.reviews.map(x=>x.id),[pointId]);assert.ok(r.next_cursor);
 r=await s.call('/api/reviews?limit=1&cursor='+encodeURIComponent(r.next_cursor));assert.deepEqual(r.reviews.map(x=>x.id),[initialId]);assert.equal(r.next_cursor,null);
 assert.equal((await s.call('/api/reviews?limit=500')).reviews.length,2);assert.equal((await s.call('/api/reviews?cursor=bad')).code,'invalid_cursor');
 assert.deepEqual((await s.call('/api/reviews','GET',undefined,'b')).reviews,[]);
 r=await s.call('/api/reviews/'+initialId);assert.equal(r.review.content,reviewText());assert.equal(r.review.sources.people,3);assert.equal(r.review.status,'ready');
 assert.equal((await s.call('/api/reviews/'+initialId,'GET',undefined,'b')).status,404);
 assert.equal((await s.call('/api/reviews/'+initialId,'DELETE',undefined,'b')).status,404);
 const exported=await s.chat.exportData(s.env,'a');
 assert.deepEqual(exported.ambre_reviews.map(x=>[x.id,x.kind,x.content.slice(0,10)]),[[pointId,'point','Ce qui a é'],[initialId,'initial','Ce qui sem']]);
 assert.deepEqual((await s.chat.exportData(s.env,'b')).ambre_reviews,[]);

 assert.deepEqual(await s.call('/api/reviews/'+pointId,'DELETE'),{status:200,deleted:true});
 state=await s.call('/api/chat');
 assert.deepEqual(state.messages.map(m=>m.review_kind||m.role),['initial','user','assistant'],'the deleted review leaves the conversation too');
 assert.equal(state.journey.started,true);
 assert.equal((await s.call('/api/reviews/'+pointId)).status,404);
 await s.call('/api/reviews/'+initialId,'DELETE');
 state=await s.call('/api/chat');
 assert.equal(state.journey.started,false);assert.equal(state.journey.can_start,false);assert.equal(state.access.reason,'journey_not_started');
 const beforeDeletedRetry=requests.length;
 r=await start(s,'deleted_cannot_reset');assert.equal(r.status,429);assert.equal(r.code,'review_quota_exhausted');assert.equal(requests.length,beforeDeletedRetry);
 const afterDeletion=await readMonthlyQuotas(s.env,'a',s.now());assert.equal(afterDeletion.reviews.used,1);assert.equal(afterDeletion.reviews.remaining,0);
 assert.equal(s.raw.prepare("SELECT count(*) n FROM monthly_usage WHERE kind='review' AND state='consumed'").get().n,2,'both monthly receipts survive content deletion');
 const purged=s.raw.prepare("SELECT purged,response FROM chat_turns WHERE idempotency_key LIKE 'review-%'").all();assert.equal(purged.length,2);assert.ok(purged.every(turn=>turn.purged===1&&turn.response===''),'review content is purged while quota receipts remain');
 const retainedNovember=await readMonthlyQuotas(s.env,'a',Date.parse('2026-11-17T12:00:00Z')/1000);assert.equal(retainedNovember.reviews.used,1);assert.equal(retainedNovember.messages.used,1);
 assert.equal((await s.call('/api/chat/messages','POST',{content:'Encore là ?',idempotency_key:'after_delete_01'})).code,'journey_not_started');
});

test('a review is prepared in the background: one at a time, blocking messages, then published once',async t=>{
 const s=setup();addPerson(s);let release,calls=0;const begun=new Promise(resolve=>{withFetch(t,async()=>{calls++;resolve();return new Promise(done=>{release=()=>done(reviewReply());});});});
 const tasks=[],ctx={waitUntil:p=>tasks.push(p)};
 let r=await start(s,'start_key_001','a',ctx);
 assert.equal(r.status,202);const id=r.review_id;
 assert.equal(r.journey.started,true);assert.equal(r.journey.review.running_id,id);assert.equal(r.journey.review.last_at,null);assert.equal(r.messages.length,0);
 await begun;
 assert.equal((await start(s,'start_key_001','a',ctx)).status,202,'the same key reports the running review');
 assert.equal((await start(s,'start_key_002','a',ctx)).code,'review_in_progress');
 assert.equal((await s.call('/api/chat/review','POST',{idempotency_key:'point_key_001'})).code,'review_in_progress');
 r=await s.call('/api/chat/messages','POST',{content:'Bonjour.',idempotency_key:'during_review_1'});
 assert.equal(r.status,409);assert.equal(r.code,'review_in_progress');
 r=await s.call('/api/reviews/'+id);assert.equal(r.review.status,'running');assert.equal(r.review.content,'');
 assert.equal((await s.chat.resumeReviews(s.env)).claimed,0,'a held lease is not taken by the cron');
 release();await Promise.all(tasks);
 r=await s.call('/api/chat');
 assert.equal(r.journey.review.running_id,null);assert.equal(r.messages.length,1);assert.equal(r.messages[0].review_id,id);assert.equal(calls,1);
 assert.equal(s.raw.prepare("SELECT count(*) n FROM chat_turns WHERE idempotency_key LIKE 'review-%'").get().n,1);
 assert.deepEqual(await s.chat.resumeReviews(s.env),{claimed:0,failed:0});
});

test('an interrupted review is resumed by the cron, with a cap on attempts',async t=>{
 const s=setup();addPerson(s);let mode='timeout',calls=0;
 withFetch(t,async()=>{calls++;if(mode==='timeout')throw Object.assign(new Error('timeout'),{name:'TimeoutError'});if(mode==='error')return new Response('{}',{status:500});if(mode==='short')return reviewReply('Trop court.');return reviewReply();});
 let r=await start(s);
 assert.equal(r.status,202);const id=r.review_id;
 let row=review(s,id);
 assert.equal(row.status,'running');assert.equal(row.attempts,1);assert.equal(row.lease_token,null);assert.ok(row.lease_until<=s.now());assert.equal(row.error_code,'provider_timeout');
 assert.equal(r.journey.review.running_id,id);assert.equal(r.messages.length,0);
 mode='ok';
 assert.deepEqual(await s.chat.resumeReviews(s.env),{claimed:1,failed:0});
 row=review(s,id);assert.equal(row.status,'ready');assert.equal(row.attempts,2);assert.equal(row.error_code,null);
 assert.equal((await s.call('/api/chat')).messages[0].review_id,id);assert.equal(calls,2);
 assert.equal(s.raw.prepare("SELECT provider_calls FROM provider_budgets WHERE user_id='a'").get().provider_calls,3,'each attempt is counted in the provider budget (the timed-out write, then the write and its check)');

 const f=setup();addPerson(f);mode='error';
 r=await start(f);const failing=r.review_id;
 row=review(f,failing);assert.equal(row.status,'running');assert.equal(row.error_code,'provider_http_500');assert.ok(row.lease_until>f.now(),'a provider error waits before the next attempt');
 assert.deepEqual(await f.chat.resumeReviews(f.env),{claimed:0,failed:0});
 f.advance(31);mode='short';
 assert.equal((await f.chat.resumeReviews(f.env)).claimed,1);
 row=review(f,failing);assert.equal(row.status,'running');assert.equal(row.attempts,2);assert.equal(row.error_code,'invalid_response');
 f.advance(31);mode='error';
 assert.equal((await f.chat.resumeReviews(f.env)).claimed,1);
 row=review(f,failing);assert.equal(row.status,'failed');assert.equal(row.attempts,3);assert.equal(row.error_code,'provider_http_500');assert.equal(row.completed_at,f.now());
 const state=await f.call('/api/chat');
 assert.equal(state.journey.started,false);assert.equal(state.journey.can_start,true);assert.equal(state.messages.length,0);
 assert.equal(f.raw.prepare('SELECT count(*) n FROM chat_turns').get().n,0);
 assert.equal((await f.call('/api/reviews')).reviews[0].status,'failed');
 mode='ok';r=await start(f,'start_key_retry');assert.equal(r.status,202);assert.equal(review(f,r.review_id).status,'ready');
});

test('the cron works in bounded batches, fails exhausted leases and skips without a provider',async t=>{
 const users=Array.from({length:8},(_,i)=>'u'+i),s=setup(users);let calls=0;withFetch(t,async()=>{calls++;return reviewReply();});
 for(const id of users){addPerson(s,id);s.raw.prepare('INSERT INTO chat_settings(user_id,consent) VALUES(?,1)').run(id);s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,idempotency_key,attempts,lease_until,created_at) VALUES(?,?,'initial','running',?,?,0,?)").run('r-'+id,id,'key-'+id,id==='u7'?3:1,s.now());}
 assert.deepEqual(await s.chat.resumeReviews({...s.env,OPENROUTER_API_KEY:''}),{claimed:0,failed:0});
 assert.deepEqual(await s.chat.resumeReviews({...s.env,GENERATION_PAUSED:'true'}),{claimed:0,failed:0});
 assert.deepEqual(await s.chat.resumeReviews(s.env,{limit:3}),{claimed:3,failed:1});
 assert.equal(calls,3);
 assert.deepEqual({...review(s,'r-u7')},{...review(s,'r-u7'),status:'failed',error_code:'attempts_exhausted'});
 const count=status=>s.raw.prepare('SELECT count(*) n FROM ambre_reviews WHERE status=?').get(status).n;
 assert.equal(count('ready'),3);assert.equal(count('running'),4);
 assert.deepEqual(await s.chat.resumeReviews(s.env),{claimed:4,failed:0});
 assert.equal(count('ready'),7);assert.equal(calls,7);
 assert.throws(()=>s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,idempotency_key,created_at) VALUES('x','u0','point','running','k1',1),('y','u0','point','running','k2',1)").run(),/UNIQUE/,'one running review per member');
});

test('no inline attempt when it is disabled, and final causes end a review at once',async t=>{
 const s=setup(['a','b'],{REVIEW_INLINE_TIMEOUT_MS:'0'});addPerson(s);let calls=0;withFetch(t,async()=>{calls++;return reviewReply();});
 let r=await start(s);
 assert.equal(r.status,202);assert.equal(calls,0);assert.equal(review(s,r.review_id).attempts,0);assert.equal(r.journey.review.running_id,r.review_id);
 assert.equal((await s.chat.resumeReviews(s.env)).claimed,1);assert.equal(review(s,r.review_id).status,'ready');assert.equal(calls,1);

 const w=setup();addPerson(w);
 globalThis.fetch=checked(async()=>{await w.call('/api/chat/consent','PUT',{accepted:false});return reviewReply();});
 r=await start(w);
 let row=review(w,r.review_id);assert.equal(row.status,'failed');assert.equal(row.error_code,'consent_withdrawn');
 assert.equal(w.raw.prepare('SELECT count(*) n FROM chat_turns').get().n,0,'nothing is published after the agreement is withdrawn');
 assert.equal(row.content,'');

 const b=setup(['a'],{WEEKLY_TOKEN_BUDGET:'10'});addPerson(b);calls=0;globalThis.fetch=checked(async()=>{calls++;return reviewReply();});
 r=await start(b);row=review(b,r.review_id);
 assert.equal(row.status,'failed');assert.equal(row.error_code,'generation_budget');assert.equal(calls,0);

 const p=setup(['a'],{GENERATION_PAUSED:'true'});addPerson(p);
 r=await start(p);assert.equal(r.status,503);assert.equal(r.code,'generation_paused');assert.equal(p.raw.prepare('SELECT count(*) n FROM ambre_reviews').get().n,0);
 const x=setup();addPerson(x);x.raw.prepare("UPDATE subscriptions SET status='canceled',paid_until=0 WHERE user_id='a'").run();
 r=await start(x);assert.equal(r.status,403);assert.equal(r.code,'subscription_required');
});

test('review prompts keep hypotheses, sources honesty and safety; the conversation prompt drops readings',()=>{
 const initial=buildReviewSystemPrompt('initial'),point=buildReviewSystemPrompt('point');
 for(const prompt of [initial,point]){
  assert.match(prompt,/Jamais de diagnostic/);assert.match(prompt,/jamais de pourcentage/);assert.match(prompt,/3114/);assert.match(prompt,/ne suggère jamais qu’un événement a eu lieu/);
  assert.match(prompt,/quelles sources manquent/);assert.match(prompt,/Entre 600 et 900 mots/);assert.match(prompt,/Ne propose ni article, ni lecture, ni personnage/);
 }
 assert.match(initial,/ce qu’elle a envie de comprendre ou de changer en priorité/);assert.match(initial,/forces et ses appuis/);assert.match(initial,/zones sensibles/);
 assert.match(point,/ce qui se confirme et ce qui s’infirme/);assert.match(point,/objectifs qu’elle a exprimés/);assert.match(point,/réellement essayé/);assert.match(point,/pistes concrètes/);
 const conversation=buildCompanionSystemPrompt({query:'Je rumine.',cartography:true,journey:true});
 assert.doesNotMatch(conversation,/reading_|catalogue|story_excerpt|relative_support/);
 assert.match(conversation,/Tu ne proposes ni article, ni lecture, ni personnage/);assert.match(conversation,/ce que la personne veut comprendre et de ce qu’elle veut changer/);
 assert.ok(!buildCompanionSystemPrompt({query:'Je rumine.'}).includes('PARCOURS.'));
});

// Worker level: page route, cron entry point, export and account deletion.
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<title>T</title><main>App shell</main>';").replace("import { APP_ASSETS, INITIAL_PAGE_DATA_MARKER } from './ui.mjs';","const APP_ASSETS={},INITIAL_PAGE_DATA_MARKER='/*INITIAL_PAGE_DATA*/null';").replace("const importExamples=()=>import('./examples.mjs');",'const importExamples=async()=>({EXAMPLES:[]});')
 .replace("import { createRecommender } from './recommendations.mjs';",'const createRecommender=()=>({recommend:()=>[],provisionalPortrait:()=>null});');
const rewritten=source.replace(/from '(\.\/[^']+)'/g,(_,specifier)=>'from '+JSON.stringify(new URL('../src/'+specifier.slice(2),import.meta.url).href));
const {default:worker}=await import('data:text/javascript;base64,'+Buffer.from(rewritten).toString('base64'));
const origin='https://journal.example';
const request=(env,path,{method='GET',data,cookie}={})=>worker.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),Origin:origin,'CF-Connecting-IP':'192.0.2.10'},body:data?JSON.stringify(data):undefined}),env);
async function register(env,email){const r=await request(env,'/api/auth/register',{method:'POST',data:{email,password:'long secure password 123'}});assert.equal(r.status,200);return {cookie:r.headers.get('set-cookie').split(';')[0],user:(await r.json()).user};}

test('Worker: “Mes bilans” page, cron resumption, export and account deletion cover the reviews',async t=>{
 const env={DB:d1(migrated()),OPENROUTER_API_KEY:'mock-only',FREE_ACCESS_ENABLED:'true',REVIEW_INLINE_TIMEOUT_MS:'0'},a=await register(env,'journey@example.test');
 for(const path of ['/mes-bilans','/mes-bilans/']){const page=await request(env,path);assert.equal(page.status,200);assert.match(page.headers.get('content-type'),/text\/html/);assert.match(await page.text(),/App shell/);}
 assert.equal((await request(env,'/api/reviews')).status,401);assert.equal((await request(env,'/api/chat/start',{method:'POST',data:{idempotency_key:'start_key_001',accepted:true}})).status,401);
 env.DB.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id=?").run(JSON.stringify({self_description:'Je me tais souvent.'}),a.user.id);
 seedCompanionPrerequisites(env.DB.raw,a.user.id);
 // The same cron also refreshes portraits: only review calls are counted here.
 let calls=0;withFetch(t,async(url,options)=>{assert.equal(String(url),'https://openrouter.ai/api/v1/chat/completions');if(JSON.parse(options.body).response_format?.json_schema?.name!=='ambre_review')return new Response('{}',{status:503});calls++;return reviewReply();});
 let r=await request(env,'/api/chat/start',{method:'POST',cookie:a.cookie,data:{idempotency_key:'start_key_001',accepted:true}});
 assert.equal(r.status,202);const body=await r.json(),id=body.review_id;assert.equal(body.journey.review.running_id,id);assert.equal(calls,0);
 r=await (await request(env,'/api/reviews/'+id,{cookie:a.cookie})).json();assert.equal(r.review.status,'running');
 const pending=[];const result=await worker.scheduled({},env,{waitUntil:p=>pending.push(p)});await Promise.all(pending);
 assert.deepEqual(result.reviews,{claimed:1,failed:0});assert.equal(calls,1);
 r=await (await request(env,'/api/chat',{cookie:a.cookie})).json();
 assert.equal(r.journey.started,true);assert.equal(r.messages[0].review_id,id);assert.equal(r.messages[0].review_kind,'initial');
 r=await (await request(env,'/api/reviews',{cookie:a.cookie})).json();assert.equal(r.reviews.length,1);
 const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();
 assert.equal(exported.ambre_reviews.length,1);assert.equal(exported.ambre_reviews[0].content,reviewText());assert.equal(exported.ambre_reviews[0].sources.portrait,true);
 assert.equal((await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}})).status,200);
 assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM ambre_reviews').get().n,0,'account deletion removes the reviews');
});

// Lot A5-P0: the same journey on the production engine. The last review reaches every
// writing request (dated, beginning and end kept); a point reads each rubric of the self
// description and the accounts of people.
const engineReply=name=>({ambre_analysis:{need:'understand',states:[],risk:'none',posture:'explore',sensitive_allowed:false,constraints:[],selected_ids:[],closed_topics:[],signals:[],strategy_ids:[],change_signal:{kind:'none',target_id:null,quote:''}},ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:{content:'Je vous écoute. Vous vous taisez pour éviter un conflit.',questions:[],proposals:[],cartography_notes:[],people_observations:[],change_proposals:[],followup:null},ambre_review:{criteria:Array(15).fill('yes'),reasons:[],change_criteria:Array.from({length:8},(_,i)=>({id:'C-R'+(i+1),verdict:'not_applicable'}))},ambre_memory:{items:[]}})[name];
test('the journey on the production engine: last review in every writing request, rubrics and accounts in the point',async t=>{
 const s=setup(['a','b'],{},{engineEnabled:true}),requests=[];
 withFetch(t,async(_url,options)=>{const body=JSON.parse(options.body),schema=body.response_format?.json_schema,data=schema?.schema?.properties?.criteria?engineReply(schema.name):schema?.name==='ambre_review'?null:engineReply(schema?.name);requests.push(body);return data?new Response(JSON.stringify({choices:[{message:{content:JSON.stringify(data)},finish_reason:'stop'}],usage:{prompt_tokens:100,completion_tokens:80,cost:0.001}})):reviewReply()});
 addPerson(s);
 s.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({first_name:'Alix',birth_date:'1990-05-04',self_description:'Je me tais souvent pour éviter les conflits.',self_perception:'PERCEPTION_MARK je me trouve trop prudente.',current_context:'CONTEXTE_MARK un nouveau poste depuis septembre.'}));
 seedReady(s);
 let r=await start(s);assert.equal(r.status,202);const initialId=r.review_id,initialRequest=requests.at(-1);
 assert.ok(contextOf(initialRequest).sources.some(x=>x.rubric==='current_context'&&x.text.includes('CONTEXTE_MARK')),'each rubric is its own source in a review');
 const work=async()=>{for(let n=0;n<10;n++){const job=s.raw.prepare('SELECT * FROM chat_engine_jobs ORDER BY rowid DESC LIMIT 1').get();if(!job||['done','failed'].includes(job.stage))return job;await s.chat.engine.work(s.env,job.turn_id)}};
 for(const [i,content] of ['J’aimerais comprendre pourquoi je me tais en réunion.','Et avec Mila aussi, je me tais.'].entries()){
  r=await s.call('/api/chat/messages','POST',{content,idempotency_key:'engine_message_'+i});assert.equal(r.status,200,JSON.stringify(r).slice(0,300));assert.equal((await work()).stage,'done');
  const writing=requests.filter(x=>x.response_format?.json_schema?.name==='ambre_writing').at(-1),input=JSON.parse(writing.messages[1].content);
  // A5-P1: the review thread replaces last_review; a review without digest travels as a dated excerpt.
  const thread=input.review_thread;assert.equal(input.last_review,undefined);
  assert.equal(thread.last.id,initialId);assert.equal(thread.last.kind,'analyse initiale');assert.match(thread.last.date,/^\d{4}-\d{2}-\d{2}$/);
  assert.ok(thread.last.text.length<=900);assert.ok(thread.last.text.startsWith('Ce qui semble ressortir.'));assert.ok(thread.last.text.endsWith('Qu’avez-vous envie de comprendre ou de changer en priorité ?'),'the end of the review (its question) is kept');assert.ok(JSON.stringify(thread).length<=1200);
  assert.ok(!input.details.some(x=>x.key==='review:'+initialId),'the last review is not sent twice');
  assert.match(writing.messages[0].content[0].text,/review_thread/);
 }
 assert.equal((await s.call('/api/chat')).quota.used,2);
 assert.equal((await s.call('/api/chat/review','POST',{idempotency_key:'engine_point_same_month'})).code,'review_quota_exhausted');
 s.advance(renewal(s)-s.now());
 r=await s.call('/api/chat/review','POST',{idempotency_key:'engine_point_001'});assert.equal(r.status,202);
 const point=contextOf(requests.at(-1));
 assert.equal(point.previous_review.kind,'initial');assert.equal(point.conversation_since.length,2);
 assert.ok(point.sources.some(x=>x.rubric==='self_perception'&&x.text.includes('PERCEPTION_MARK')));assert.ok(point.sources.some(x=>x.type==='person'&&x.text.includes('elle m’écoute')));
 assert.ok(JSON.stringify(point).length<=34000);
});
