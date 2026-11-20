import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {loadContext,hasPersonalizationSources} from '../src/private-context.mjs';

const at=Date.parse('2026-11-04T12:00:00Z');

function fixture(){
 const raw=new DatabaseSync(':memory:');for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 raw.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','test@example.invalid','h','s'); INSERT INTO profiles(user_id,self_data) VALUES('u','{\"self_description\":\"Je décris mes limites.\"}'); INSERT INTO memories(user_id) VALUES('u'); INSERT INTO subscriptions(user_id) VALUES('u'); INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description,current_importance) VALUES('e','u','created-event','h','Un événement','turning_point','La rumination me pèse.',5);");
 let reads=0;
 const DB={raw,prepare(sql){let args=[];return {
  bind(...x){args=x;return this},
  async first(){reads++;return raw.prepare(sql).get(...args)||null},
  async all(){reads++;return {results:raw.prepare(sql).all(...args)}},
  async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}}}
 }}};
 DB.batch=async statements=>{raw.exec('BEGIN');try{const results=[];for(const statement of statements)results.push(await statement.run());raw.exec('COMMIT');return results}catch(e){raw.exec('ROLLBACK');throw e}};
 const env={DB,FREE_ACCESS_ENABLED:'true'};
 const personal=createPersonalBackend({error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:v=>String(v||''),body:r=>r.json(),now:()=>Math.floor(at/1000),random:()=>crypto.randomUUID()});
 return {raw,DB,env,personal,reads:()=>reads};
}
const manifestDigest=context=>createHash('sha256').update(JSON.stringify(context.sources.map(s=>[s.id,s.type,s.revision,s.hash,s.explicit]))).digest('hex');

test('the authenticated snapshot carries the declared importance of a moment',async()=>{
 const f=fixture(),context=await loadContext(f.env,'u');assert.equal(context.events[0].current_importance,5);assert.equal(context.sources.find(s=>s.id==='event:e').current_importance,5);
 assert.equal('likes' in context,false);assert.equal('exposures' in context,false);
});

// Reference computed with the code that still read article_likes (before the catalogue left the
// Worker), on this exact fixture: a member who never liked an article keeps the same sources,
// revisions and hashes, so nothing of the saved portrait or its extraction cache is invalidated.
test('without any liked article, the sources and their hashes are the ones computed before the catalogue was retired',async()=>{
 const f=fixture();
 f.raw.exec("UPDATE memories SET text='Je préfère avancer pas à pas.' WHERE user_id='u'; INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text,period_start) VALUES('p','u','created-person','h','Sacha','friend','Sacha m’écoute sans me juger.','2019');");
 const context=await loadContext(f.env,'u');
 assert.deepEqual(context.sources.map(s=>s.id),['profile:u','memory:u','person:p','event:e']);
 assert.equal(manifestDigest(context),'6c1dd0c04dd5101d525d10da8cbe798b2d35cebf61c4aad02f2aa05f59e0dc0b');
});

test('a liked article left in the database is no longer a source, and a saved portrait that cited it stays current without any new job',async t=>{
 const f=fixture();seedCompanionPrerequisites(f.raw,'u');f.env.OPENROUTER_API_KEY='mock-only';
 const previous=globalThis.fetch;let calls=0;t.after(()=>{globalThis.fetch=previous});globalThis.fetch=async()=>{calls++;throw Error('Unexpected provider call')};
 const before=manifestDigest(await loadContext(f.env,'u'));
 f.raw.prepare("INSERT INTO article_likes(user_id,article_id,reason,differences) VALUES('u','posts/c/lu','Je me reconnais dans ce récit.','')").run();
 const version=f.raw.prepare("SELECT version FROM profiles WHERE user_id='u'").get().version;
 const portrait={schema_version:2,summary:'Synthèse enregistrée.',summary_evidence:['like:posts/c/lu'],axes:[],strengths:[],uncertainties:[],citations:[{source_id:'like:posts/c/lu',revision:1,quote:'Je me reconnais dans ce récit.'}],psychological_map:[],coverage:{processed:['profile:u','like:posts/c/lu','event:e'],pending:[]}};
 f.raw.prepare("INSERT INTO user_portraits(user_id,portrait,analyzed_version,schema_version) VALUES('u',?,?,2)").run(JSON.stringify(portrait),version);
 const context=await loadContext(f.env,'u');
 assert.equal(context.sources.some(s=>s.type==='like'),false);assert.equal(manifestDigest(context),before);assert.equal(hasPersonalizationSources(context),true);
 assert.equal(context.source_version,version);assert.equal(context.portrait_complete,true);assert.equal(context.portrait.summary,'Synthèse enregistrée.');
 // Neither a page view, nor the scheduled pass, nor a claim queues a regeneration: nothing loops.
 for(let pass=0;pass<3;pass++){
  await f.personal.sourcesChanged(f.env,'u',{waitUntil(){assert.fail('No background refresh')}});
  const shown=await f.personal.portrait(f.env,'u');assert.equal(shown.status,'ready');assert.equal(shown.outdated,false);assert.equal(shown.provisional_portrait,null);
  assert.deepEqual((await (await f.personal.claimJobs(f.env,'u','portrait')).json()).jobs,[]);
  assert.deepEqual(await f.personal.refreshPendingPortraits(f.env),{status:'worker-managed',claimed:0,completed:0,failed:0});
 }
 assert.equal(f.raw.prepare("SELECT count(*) AS n FROM journal_jobs").get().n,0);assert.equal(calls,0);
 assert.equal(f.raw.prepare("SELECT version FROM profiles WHERE user_id='u'").get().version,version);
 assert.equal(f.raw.prepare("SELECT count(*) AS n FROM article_likes").get().n,1,'the liked article stays stored');
});

test('liked articles alone are not a source of the portrait any more',async()=>{
 const f=fixture();seedCompanionPrerequisites(f.raw,'u');f.env.OPENROUTER_API_KEY='mock-only';
 f.raw.exec("UPDATE profiles SET self_data=json_remove(self_data,'$.self_description') WHERE user_id='u'; DELETE FROM life_events WHERE id='e'; INSERT INTO article_likes(user_id,article_id,reason) VALUES('u','posts/c/lu','Je me reconnais dans ce récit.');");
 const context=await loadContext(f.env,'u');assert.deepEqual(context.sources,[]);assert.equal(hasPersonalizationSources(context),false);
 assert.equal((await f.personal.portrait(f.env,'u')).status,'collecting');
 await f.personal.sourcesChanged(f.env,'u');assert.deepEqual((await (await f.personal.claimJobs(f.env,'u','portrait')).json()).jobs,[]);
 assert.equal(f.raw.prepare("SELECT count(*) AS n FROM journal_jobs").get().n,0);
});

test('the private job proxy keeps the same data-retention constraints as Ambre',async t=>{
 const f=fixture(),now=Math.floor(at/1000);seedCompanionPrerequisites(f.raw,'u');f.env.OPENROUTER_API_KEY='mock-only';f.raw.prepare("INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,status,lease_token,lease_until) VALUES('j','u','portrait',1,1,'running','lease',?)").run(now+120);
 const previous=globalThis.fetch;t.after(()=>{globalThis.fetch=previous});globalThis.fetch=async(_url,options)=>{const body=JSON.parse(options.body);assert.deepEqual(body.provider,{data_collection:'deny',zdr:true,allow_fallbacks:true});return Response.json({choices:[{message:{content:'{}'},finish_reason:'stop'}]})};
 const response=await f.personal.proxyJobLLM(f.env,'j',{lease_token:'lease',source_version:1,memory_version:1,system_prompt:'Return JSON.',user_prompt:'Synthetic test.',max_tokens:128});assert.equal(response.status,200);
});

test('an impossible native token budget is terminal and never calls the provider',async t=>{
 const f=fixture();seedCompanionPrerequisites(f.raw,'u');f.env.OPENROUTER_API_KEY='mock-only';f.env.STORY_TOKEN_BUDGET='1';
 const previous=globalThis.fetch;let calls=0;t.after(()=>{globalThis.fetch=previous});globalThis.fetch=async()=>{calls++;throw Error('Unexpected provider call')};
 const result=await f.personal.refreshUserPortrait(f.env,'u');assert.equal(result.status,'generation_budget');
 const job=f.raw.prepare("SELECT status,error_code,provider_calls FROM journal_jobs WHERE kind='portrait'").get();assert.equal(job.status,'failed');assert.equal(job.error_code,'generation_budget');assert.equal(job.provider_calls,0);assert.equal(calls,0);
 const retry=await f.personal.refreshUserPortrait(f.env,'u');assert.equal(retry.claimed,0);assert.equal(calls,0);
 f.raw.prepare("UPDATE journal_jobs SET created_at=datetime('now','-25 hours') WHERE kind='portrait'").run();f.env.STORY_TOKEN_BUDGET='300000';
 const renewed=await (await f.personal.claimJobs(f.env,'u','portrait')).json();assert.equal(renewed.jobs.length,1);assert.equal(renewed.jobs[0].attempts,1);assert.equal(calls,0);
});
