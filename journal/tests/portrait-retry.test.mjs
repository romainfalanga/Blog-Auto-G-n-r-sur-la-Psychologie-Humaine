import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
import {DatabaseSync} from 'node:sqlite';
import {createPersonalBackend} from '../src/personal-backend.mjs';

function setup(t){
 const db=new DatabaseSync(':memory:');t.after(()=>db.close());
 const dir=new URL('../migrations/',import.meta.url);
 for(const file of readdirSync(dir).filter(file=>file.endsWith('.sql')).sort())db.exec(readFileSync(new URL(file,dir),'utf8'));
 const DB={prepare(query){return {bind(...args){return {
  first:()=>db.prepare(query).get(...args)||null,
  all:()=>({results:db.prepare(query).all(...args)}),
  run:()=>({meta:{changes:db.prepare(query).run(...args).changes}})
 };}};},batch(statements){db.exec('BEGIN');try{const result=statements.map(s=>s.run());db.exec('COMMIT');return result;}catch(e){db.exec('ROLLBACK');throw e;}}};
 const env={DB,OPENROUTER_API_KEY:'test-only',FREE_ACCESS_ENABLED:'true'};
 const now=()=>Math.floor(Date.now()/1000);
 const backend=createPersonalBackend({error:(status,message,code)=>{throw Object.assign(Error(message),{status,code});},json:v=>Response.json(v),text:v=>String(v||'').trim(),now,random:()=>crypto.randomUUID(),decorateLikes:v=>v,recommender:{provisionalPortrait:()=>null}});
 const failed=(id,{paths=['contract.language'],sourceVersion=2,providerCalls=3,old=false}={})=>{
  db.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.test');
  db.prepare('INSERT INTO profiles(user_id,version,self_data) VALUES(?,2,?)').run(id,JSON.stringify({self_description:'Je souhaite mieux comprendre mes réactions au travail.'}));
  db.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);
  db.prepare('INSERT INTO subscriptions(user_id) VALUES(?)').run(id);
  const diagnostic=JSON.stringify({analysis_version:3,rejected:true,failed_paths:paths});
  db.prepare("INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,status,attempts,provider_calls,reserved_tokens,error_code,native_candidate,created_at) VALUES(?,?,'portrait',?,1,'failed',3,?,1200,'validation_error',?,datetime('now',?))").run(id,id,sourceVersion,providerCalls,diagnostic,old?'-2 days':'-0 days');
 };
 const job=id=>db.prepare('SELECT * FROM journal_jobs WHERE id=?').get(id);
 return {db,env,backend,failed,job,now};
}

test('an old language rejection gets one long recovery with cumulative usage and both checks',async t=>{
 const {env,backend,failed,job}=setup(t);failed('legacy',{old:true});
 assert.equal((await backend.portrait(env,'legacy')).status,'queued');
 assert.equal((await backend.refreshUserPortrait(env,'legacy')).claimed,0);
 assert.equal(job('legacy').provider_calls,3);assert.equal(job('legacy').attempts,3);
 const oldFetch=globalThis.fetch,stages=[];t.after(()=>{globalThis.fetch=oldFetch;});
 globalThis.fetch=async(url,request)=>{
  assert.equal(url,'https://openrouter.ai/api/v1/chat/completions');
  const {messages}=JSON.parse(request.body),data=JSON.parse(messages.at(-1).content);
  let result;
  if(data.claims){stages.push('support');result={checks:data.claims.map(c=>({path:c.path,supported:true}))};}
  else{
   stages.push('portrait');assert.match(messages[0].content,/Une vérification a refusé/);
   const [s]=data.sources;
   result={schema_version:2,summary:'{{prenom}} dit vouloir comprendre ses réactions au travail. Il reste une question à explorer.',summary_evidence:[s.id],axes:[{label:'Travail',why:'{{prenom}} souhaite comprendre ses réactions au travail.',confidence:.3,claim_type:'declaration',evidence:[s.id]}],strengths:[],uncertainties:[],psychological_map:[],citations:[{source_id:s.id,revision:s.revision,quote:s.fields.self_description}]};
  }
  return Response.json({choices:[{finish_reason:'stop',message:{content:JSON.stringify(result)}}]});
 };
 const results=await Promise.all([backend.refreshPendingPortraits(env),backend.refreshPendingPortraits(env)]);
 assert.equal(results.reduce((n,r)=>n+r.claimed,0),1);
 assert.deepEqual(stages,['portrait','support']);
 const complete=job('legacy');assert.equal(complete.status,'complete');assert.equal(complete.attempts,4);
 assert.equal(complete.provider_calls,5);assert.ok(complete.reserved_tokens>1200);
 assert.equal((await backend.portrait(env,'legacy')).status,'ready');
});

test('a second failure, unrelated diagnostics and stale source versions do not get the recovery attempt',async t=>{
 const {env,backend,failed,job}=setup(t);
 failed('legacy');failed('fresh',{paths:['contract.pronoun']});failed('semantic',{paths:['summary']});
 failed('spent',{providerCalls:19});failed('stale',{sourceVersion:1});
 const claim=async id=>(await (await backend.claimJobs(env,id)).json()).jobs[0];
 for(const id of ['fresh','semantic','spent']){
  assert.equal(await claim(id),undefined);assert.equal((await backend.portrait(env,id)).status,'error');
 }
 assert.notEqual((await claim('stale')).id,'stale');assert.equal(job('stale').attempts,3);
 const attempt=await claim('legacy');assert.equal(attempt.attempts,4);assert.equal(attempt.provider_calls,3);
 await backend.failJob(env,attempt.id,{lease_token:attempt.lease_token,reason:'validation_error'});
 assert.equal(await claim('legacy'),undefined);assert.equal((await backend.portrait(env,'legacy')).status,'error');
});
