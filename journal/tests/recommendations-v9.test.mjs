import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createRecommender,classifyText} from '../src/recommendations.mjs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {loadContext} from '../src/private-context.mjs';

const rows=[['limits','Limites','Affirmation'],['rumination','Rumination','Pensées'],['grief','Deuil','Perte'],['joy','Gratitude','Joie']].map(([id,...keywords],i)=>({id,title:keywords.join(' '),keywords,sujet:keywords[0],character:'Personnage '+i,url:'/articles/'+id+'/'}));
const engine=createRecommender(rows),at=Date.parse('2026-11-04T12:00:00Z');
const profile=text=>({sources:[{id:'profile:u',type:'profile',explicit:true,fields:{self_description:text}}]});

test('reported inability is a reading need, while denial of a pattern stays negative',()=>{
 for(const text of ["Je n'arrive pas à poser mes limites.",'Je ne sais pas exprimer mes limites.',"Je n'ose pas exprimer mes limites."]){assert.equal(classifyText(text)[0].kind,'declared');assert.equal(engine.recommend(profile(text),{now:at})[0].id,'limits');}
 assert.equal(classifyText('Je ne rumine plus.')[0].kind,'difference');
 assert.equal(engine.recommend(profile('Je ne rumine plus.'),{now:at}).length,0);
});

test('copying one personal statement across collections cannot multiply its relevance',()=>{
 const text='Je ressens de la rumination et des pensées répétées.',base=profile(text),copies={...base,memory:{text},people:[{id:'p',relationship_text:text}],events:[{id:'e',description:text}]};
 assert.deepEqual(engine.recommend(copies,{now:at}),engine.recommend(base,{now:at}));
 const different={...base,people:[{id:'p',relationship_text:'Avec mes parents, la rumination prend la forme de pensées répétées.'}]};
 assert.ok(engine.recommend(different,{now:at})[0].relevance>engine.recommend(base,{now:at})[0].relevance);
});

test('a current reading exclusion is respected despite past interest and stays request-local',()=>{
 const context={likes:[{article_id:'rumination',reason:'Je me reconnais dans la rumination.'}],memory:{text:'Je souhaite explorer le deuil et la rumination.'}},before=structuredClone(context);
 const items=engine.recommend(context,{now:at,includeLiked:true,query:'Je veux éviter les articles sur la rumination. Je veux lire sur le deuil.'});
 assert.ok(items.some(x=>x.id==='grief'));assert.ok(!items.some(x=>x.id==='rumination'));assert.deepEqual(context,before);
 assert.ok(engine.recommend(context,{now:at,includeLiked:true}).some(x=>x.id==='rumination'));
});

test('only a newer unambiguous reading choice overrides an old avoidance preference',()=>{
 const old={story_id:'old',avoid:'Deuil',updated_at:'2026-10-01 12:00:00'},context={memory:{text:'Je souhaite explorer le deuil.'},feedback:[old]};
 assert.ok(!engine.recommend(context,{now:at}).some(x=>x.id==='grief'));
 assert.ok(engine.recommend(context,{now:at,query:'Je veux lire un article sur le deuil.'}).some(x=>x.id==='grief'));
 assert.ok(!engine.recommend(context,{now:at,query:'Je veux aider un proche pendant son deuil.'}).some(x=>x.id==='grief'));
 for(const query of ['Une lecture de mon fil me rappelle mon deuil.',"Je n'arrive pas à apprécier les articles sur le deuil."])assert.ok(!engine.recommend(context,{now:at,query}).some(x=>x.id==='grief'));
 assert.ok(engine.recommend({...context,feedback:[old,{story_id:'new',preference:'Je veux lire sur le deuil.',updated_at:'2026-11-03 12:00:00'}]},{now:at}).some(x=>x.id==='grief'));
});

test('declared event importance influences reading priority, without creating a new proof',()=>{
 const event={id:'e',description:'La rumination prend la forme de pensées répétées.'},low=engine.recommend({events:[{...event,current_importance:1}]},{now:at}),high=engine.recommend({events:[{...event,current_importance:5}]},{now:at});
 assert.equal(high[0].id,low[0].id);assert.ok(high[0].relevance>low[0].relevance);assert.equal(high[0].reason_sources[0].source_id,'event:e');
});

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
 const env={DB,FREE_ACCESS_ENABLED:'true'};
 const personal=createPersonalBackend({error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:v=>String(v||''),body:r=>r.json(),now:()=>Math.floor(at/1000),random:()=>crypto.randomUUID(),decorateLikes:v=>v,recommender:engine,examples:rows});
 return {raw,DB,env,personal,reads:()=>reads};
}

test('the authenticated snapshot carries event importance and serves readings without another SQL read',async()=>{
 const f=fixture(),context=await loadContext(f.env,'u'),before=structuredClone(context);assert.equal(context.events[0].current_importance,5);assert.equal(context.sources.find(s=>s.id==='event:e').current_importance,5);
 const reads=f.reads(),items=f.personal.readingSuggestionsFromContext(context,{query:'Un article sur les limites.'});assert.ok(items.length);assert.equal(f.reads(),reads);assert.deepEqual(context,before);
});

test('the private job proxy keeps the same data-retention constraints as Ambre',async t=>{
 const f=fixture(),now=Math.floor(at/1000);f.env.OPENROUTER_API_KEY='mock-only';f.raw.prepare("INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,status,lease_token,lease_until) VALUES('j','u','portrait',1,1,'running','lease',?)").run(now+120);
 const previous=globalThis.fetch;t.after(()=>{globalThis.fetch=previous});globalThis.fetch=async(_url,options)=>{const body=JSON.parse(options.body);assert.deepEqual(body.provider,{data_collection:'deny',zdr:true,allow_fallbacks:true});return Response.json({choices:[{message:{content:'{}'},finish_reason:'stop'}]})};
 f.DB.batch=async statements=>{f.raw.exec('BEGIN');try{const results=[];for(const statement of statements)results.push(await statement.run());f.raw.exec('COMMIT');return results}catch(e){f.raw.exec('ROLLBACK');throw e}};
 const response=await f.personal.proxyJobLLM(f.env,'j',{lease_token:'lease',source_version:1,memory_version:1,system_prompt:'Return JSON.',user_prompt:'Synthetic test.',max_tokens:128});assert.equal(response.status,200);
});

test('an impossible native token budget is terminal and never calls the provider',async t=>{
 const f=fixture();f.env.OPENROUTER_API_KEY='mock-only';f.env.STORY_TOKEN_BUDGET='1';
 f.DB.batch=async statements=>{f.raw.exec('BEGIN');try{const results=[];for(const statement of statements)results.push(await statement.run());f.raw.exec('COMMIT');return results}catch(e){f.raw.exec('ROLLBACK');throw e}};
 const previous=globalThis.fetch;let calls=0;t.after(()=>{globalThis.fetch=previous});globalThis.fetch=async()=>{calls++;throw Error('Unexpected provider call')};
 const result=await f.personal.refreshUserPortrait(f.env,'u');assert.equal(result.status,'generation_budget');
 const job=f.raw.prepare("SELECT status,error_code,provider_calls FROM journal_jobs WHERE kind='portrait'").get();assert.equal(job.status,'failed');assert.equal(job.error_code,'generation_budget');assert.equal(job.provider_calls,0);assert.equal(calls,0);
 const retry=await f.personal.refreshUserPortrait(f.env,'u');assert.equal(retry.claimed,0);assert.equal(calls,0);
 f.raw.prepare("UPDATE journal_jobs SET created_at=datetime('now','-25 hours') WHERE kind='portrait'").run();f.env.STORY_TOKEN_BUDGET='300000';
 const renewed=await (await f.personal.claimJobs(f.env,'u','portrait')).json();assert.equal(renewed.jobs.length,1);assert.equal(renewed.jobs[0].attempts,1);assert.equal(calls,0);
});
