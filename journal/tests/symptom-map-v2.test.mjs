import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createCartographyBackend,computeCartography,readSymptomMap} from '../src/cartographie.mjs';
const data={ranking:{version:'essential_share_desc'},factors:[{id:'a',overlap_group:'a',safety:'suicide'},{id:'a2',overlap_group:'a'},{id:'b'},{id:'c'},{id:'use'}],outcomes:[{id:'x'},{id:'y'}],links:[{factor:'a',outcome:'x',strength:'critere'},{factor:'a2',outcome:'x',strength:'critere'},{factor:'b',outcome:'x',strength:'critere'},{factor:'c',outcome:'x',strength:'moderee'},{factor:'a',outcome:'y',strength:'critere'}],requires:{y:['use']}};
test('essential families, requirements, uncertainty and dismissal determine symptom rank',()=>{
 let s=computeCartography({a:1,a2:1,b:'unknown',c:1},data);assert.equal(s.results.length,1);assert.equal(s.results[0].essential_count,1);assert.equal(s.results[0].essential_total,2);assert.equal(s.results[0].essential_score,333);assert.equal(s.results[0].moderate_count,1);
 s=computeCartography({a:1,b:1,use:1},data,{x:'non'});assert.deepEqual(s.results.map(x=>x.outcome.id),['y','x']);assert.equal(s.results[1].essential_score,666);assert.equal(computeCartography({a:'unknown',b:0,use:1},data).results.length,0);
});
function setup(){const raw=new DatabaseSync(':memory:');raw.exec('PRAGMA foreign_keys=ON');for(const f of readdirSync(new URL('../migrations/',import.meta.url)).filter(x=>x.endsWith('.sql')).sort())raw.exec(readFileSync(new URL('../migrations/'+f,import.meta.url),'utf8'));raw.prepare("INSERT INTO users(id,email,password_hash,password_salt,created_at) VALUES('u','u@test','x','s','now')").run();const DB={prepare(sql){let args=[];return {bind(...a){args=a;return this},async first(){return raw.prepare(sql).get(...args)||null},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}}}}}};let time=100;const backend=createCartographyBackend({error(status,message,code){throw Object.assign(Error(message),{status,code})},json:x=>x,body:r=>r.json(),now:()=>time++,table:'symptom_maps',data,route:'/api/symptomes',body_limit:120000});return {raw,env:{DB},backend};}
test('symptom map saves without consent prompt, CAS protects changes, safety timestamps are server owned',async()=>{
 const {raw,env,backend}=setup(),save=input=>backend.routes(new Request('https://site/api/symptomes',{method:'PUT',body:JSON.stringify(input)}),env,{id:'u'},'/api/symptomes');
 const first=await save({answers:{a:1},version:0,safety_at:999,safety_asked_at:999});assert.equal(first.cartographie.locked,true);assert.equal(first.cartographie.safety_at,100);assert.equal((await readSymptomMap(env,'u')).safety_asked_at,null);
 await assert.rejects(save({answers:{b:1},version:0}),e=>e.status===409);
 const next=await save({answers:{a:0},version:1});assert.equal(next.cartographie.safety_at,100);assert.equal(next.cartographie.version,2);
 await assert.rejects(save({answers:{a:2},version:2}),e=>e.status===400);
 assert.equal((await backend.exportData(env,'u')).symptom_map.answers.a,0);assert.ok(raw.prepare("SELECT revision FROM care_source_revisions WHERE user_id='u' AND scope='symptoms'").get().revision>=2);
 raw.prepare("DELETE FROM users WHERE id='u'").run();assert.equal(await readSymptomMap(env,'u'),null);
});
test('symptom map is isolated by owner, rejects unknown identifiers and supports deliberate erase',async()=>{
 const {env,backend}=setup();const request=(method,value)=>new Request('https://site/api/symptomes',{method,...(value?{body:JSON.stringify(value)}:{})});
 await backend.routes(request('PUT',{answers:{a:1,b:'unknown'},notes:{factors:{a:'mes mots'}},feedback:{x:'oui'},expected_version:0}),env,{id:'u'},'/api/symptomes');
 assert.equal((await backend.routes(request('GET'),env,{id:'other'},'/api/symptomes')).cartographie,null);
 await assert.rejects(backend.routes(request('PUT',{answers:{forged:1},version:1}),env,{id:'u'},'/api/symptomes'),e=>e.status===400);
 assert.equal(await backend.routes(request('GET'),env,{id:'u'},'/api/symptomes/history'),null,'v1 subroutes are not retained');
 await backend.routes(request('DELETE'),env,{id:'u'},'/api/symptomes');assert.equal(await readSymptomMap(env,'u'),null);
});
test('past symptoms save without a note, keep an optional note, remain out of rank and safety, and export unchanged',async()=>{
 const {env,backend}=setup(),save=input=>backend.routes(new Request('https://site/api/symptomes',{method:'PUT',body:JSON.stringify(input)}),env,{id:'u'},'/api/symptomes');
 const bare=await save({answers:{a:'past'},notes:{},version:0});
 assert.equal(bare.cartographie.answers.a,'past');assert.equal(bare.cartographie.safety_at,null);assert.equal(bare.summary.results.length,0);
 const first=await save({answers:{a:'past'},notes:{factors:{a:'Avant 2020, puis cela a cessé.'}},version:1});
 assert.equal(first.cartographie.answers.a,'past');assert.equal(first.cartographie.notes.factors.a,'Avant 2020, puis cela a cessé.');
 const exported=(await backend.exportData(env,'u')).symptom_map;assert.equal(exported.answers.a,'past');assert.equal(exported.notes.factors.a,'Avant 2020, puis cela a cessé.');
 const removed=await save({answers:{a:'past'},notes:{},version:2});assert.equal(removed.cartographie.answers.a,'past');assert.equal(removed.cartographie.notes.factors?.a,undefined);
 const after=await readSymptomMap(env,'u');assert.equal(after.version,3);assert.equal((await backend.exportData(env,'u')).symptom_map.answers.a,'past');
});
