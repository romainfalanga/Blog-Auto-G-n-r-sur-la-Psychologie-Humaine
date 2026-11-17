// Plan v4.1, lot C1: one public order shared by the page and Ambre, counting only
// experiences linked by an empirical association. Criteria checked by simulation.
import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createRequire} from 'node:module';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {computeCartography,summarizeCartography,cartographyHighlights,cartographyEvidence,DISPLAY_ORDER_MAX} from '../src/cartographie.mjs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
const require=createRequire(import.meta.url);
const publicPage=require('../../static/js/cartographie.js');
const fullData=JSON.parse(readFileSync(new URL('../../data/cartographie.json',import.meta.url),'utf8'));
const ids=CARTOGRAPHIE.factors.map(f=>f.id);
const EMPIRICAL=new Set(['forte','moderee','limitee']);
const relation=l=>l.definitional?'definitional':l.relation_type;

function profiles(size,count,seed){
 let state=seed;const rand=()=>(state=(state*1103515245+12345)%2147483648)/2147483648;
 const out=[];
 for(let n=0;n<count;n++){
  const answers={},pool=ids.slice();
  while(Object.keys(answers).length<size&&pool.length){const id=pool.splice(Math.floor(rand()*pool.length),1)[0];answers[id]=rand()<0.5?1:2;}
  for(const group of CARTOGRAPHIE.exclusive||[])if(group.filter(id=>answers[id]).length>1)delete answers[group[1]];
  out.push(answers);
 }
 return out;
}
const displayed=answers=>publicPage.compute(fullData,answers).results;

test('simulation: top 5 displayed equals top 5 handed to Ambre, no contextual card, page/Worker parity',()=>{
 let checked=0;
 for(const size of [1,3,6,10,15,30])for(const answers of profiles(size,size>10?150:400,1000+size)){
  const page=displayed(answers),worker=computeCartography(answers).results,summary=summarizeCartography(answers);
  const pageIds=page.map(r=>r.outcome.id);
  assert.deepEqual(worker.map(r=>r.outcome.id),pageIds,'Worker order = page order (full dataset vs compact module)');
  assert.deepEqual(summary.display_order.slice(0,5).map(x=>x.id),pageIds.slice(0,5),'top 5 shown = top 5 for Ambre');
  assert.deepEqual(summary.display_order.map(x=>x.id),pageIds.slice(0,DISPLAY_ORDER_MAX));
  assert.equal(summary.display_count,page.length);
  // v5 (lot C5a): contextual subjects are never displayed; subjects with a measured
  // association come before model-only subjects; strength guard (see v5-C5a tests).
  assert.ok(!page.some(r=>r.outcome.contextual_only),'no contextual card');
  for(let i=1;i<page.length;i++)if(page[i-1].section===page[i].section)assert.ok(Number(page[i-1].has_empirical)>=Number(page[i].has_empirical),'measured associations before models only');
  // Ambre's bounded lists keep the same order (subsequences of the display order).
  for(const list of [summary.convergence,summary.pistes,summary.exploratory]){const positions=list.map(x=>pageIds.indexOf(x.id));assert.deepEqual(positions,positions.slice().sort((a,b)=>a-b));}
  checked++;
 }
 assert.ok(checked>=1900);
});

test('the displayed count only includes empirical associations; other references are never counted',()=>{
 for(const answers of profiles(12,200,77))for(const r of displayed(answers)){
  const expected=[...new Set(r.links.filter(l=>relation(l)==='empirical_association'&&EMPIRICAL.has(l.strength)).map(l=>l.factor))];
  assert.deepEqual(r.associated_factors.slice().sort(),expected.sort());
  assert.equal(r.associated_count,expected.length);
  assert.equal(r.other_links,r.links.filter(l=>relation(l)!=='empirical_association').length);
  // Empirical links come first so the header shows the strongest measured evidence.
  const firstOther=r.links.findIndex(l=>relation(l)!=='empirical_association');
  if(firstOther>=0)assert.ok(r.links.slice(firstOther).every(l=>relation(l)!=='empirical_association'));
 }
});

test('theoretical, definitional and classification links no longer push a subject up the list',()=>{
 const data={weights:{forte:3,critere:2,moderee:2,limitee:1,theorique:1},factors:['a','b','c','d'].map(id=>({id,label:id.toUpperCase()})),
  outcomes:[{id:'model',label:'Modèle',kind:'schema'},{id:'measured',label:'Mesuré',kind:'trouble'},{id:'tie',label:'Égalité',kind:'trouble'},{id:'ctx',label:'Contexte',kind:'trouble',contextual_only:true}],
  links:[{factor:'a',outcome:'model',strength:'theorique',relation_type:'clinical_model'},{factor:'b',outcome:'model',strength:'theorique',relation_type:'definitional',definitional:true},{factor:'c',outcome:'model',strength:'critere',relation_type:'classification_context'},
   {factor:'d',outcome:'measured',strength:'limitee',relation_type:'empirical_association'},
   {factor:'d',outcome:'tie',strength:'limitee',relation_type:'empirical_association'},{factor:'a',outcome:'tie',strength:'theorique',relation_type:'clinical_model'},
   {factor:'a',outcome:'ctx',strength:'forte',relation_type:'empirical_association'},{factor:'b',outcome:'ctx',strength:'forte',relation_type:'empirical_association'}]};
 const answers={a:1,b:1,c:1,d:1},page=publicPage.compute(data,answers).results,worker=computeCartography(answers,data).results;
 // v5: a model never passes a measured subject, but it breaks a tie between two
 // subjects of equal measured weight; contextual subjects are not displayed.
 assert.deepEqual(page.map(r=>r.outcome.id),['tie','measured','model']);
 assert.deepEqual(worker.map(r=>r.outcome.id),['tie','measured','model']);
 assert.equal(page[2].associated_count,0);assert.equal(page[2].other_links,3);
 assert.ok(page[2].relevance<page[1].relevance);
});

test('interpretation.ranking describes the order actually displayed',()=>{
 for(const text of [fullData.interpretation.ranking,CARTOGRAPHIE.interpretation.ranking]){
  assert.match(text,/spécificité/);assert.match(text,/Souvent/);assert.match(text,/reconnaissez/);assert.match(text,/ni une probabilité ni un risque/);assert.match(text,/Ambre/);
  assert.doesNotMatch(text,/contextuels viennent en dernier/);assert.doesNotMatch(text,/ne comptent pas/);
  assert.doesNotMatch(text,/Niveau maximal du lien/);
 }
});

test('highlights and evidence follow the shared order after recognition',()=>{
 for(const answers of profiles(8,200,4242)){
  const summary=summarizeCartography(answers),order=summary.display_order.map(x=>x.id),all=computeCartography(answers).results.map(r=>r.outcome.id);
  const positions=cartographyHighlights(summary,3).map(x=>all.indexOf(x.id));
  assert.deepEqual(positions,positions.slice().sort((a,b)=>a-b));
  const evidence=cartographyEvidence(answers).evidence.map(e=>all.indexOf(e.id));
  assert.deepEqual(evidence,evidence.slice().sort((a,b)=>a-b));
  assert.ok(order.every(id=>all.includes(id)));
 }
});

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
const fail=(status,message,code)=>{throw Object.assign(new Error(message),{status,code});};

test('Ambre’s prompt carries the same display order as the page',async t=>{
 const raw=migrated(),DB=d1(raw);let serial=0;const clock=Math.floor(Date.now()/1000);
 const utils={error:fail,json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null,recommend:()=>[]}};
 const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue:new Map()}),env={DB,OPENROUTER_API_KEY:'mock-only'};
 raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES('a','a@example.com','hash','salt')").run();raw.prepare("INSERT INTO profiles(user_id) VALUES('a')").run();raw.prepare("INSERT INTO memories(user_id) VALUES('a')").run();raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES('a','active',?)").run(clock+86400);
 raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES('seed','a','initial','ready','Analyse initiale de test.','seed-initial',0,0)").run();
 const answers={parentification:1,ressenti_responsable:1,enfant_sage:2,debrouiller_seul:1,parent_trouble:1};
 raw.prepare('INSERT INTO cartographies(user_id,answers,feedback,consent,locked_at,version,updated_at) VALUES(?,?,?,?,?,?,?)').run('a',JSON.stringify(answers),'{}',1,1700000000,1,1700000000);
 const call=async(path,method='GET',data)=>{const req=new Request('https://example.com'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:'a'},path,url)||await personal.routes(req,env,{id:'a'},path,url);return {status:res.status,...await res.json()};}catch(e){return {status:e.status||500,code:e.code,message:e.message};}};
 let request;const previous=globalThis.fetch;t.after(()=>{globalThis.fetch=previous;});
 globalThis.fetch=async(_url,options)=>{request=request||JSON.parse(options.body);return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({content:'Partons de votre carte.',questions:[],proposals:[],cartography_notes:[]})},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:10,cost:0}}));};
 assert.equal((await call('/api/chat/consent','PUT',{accepted:true})).status,200);
 const result=await call('/api/chat/messages','POST',{content:'Par où commencer ?',idempotency_key:'carto_v4c_order_01'});
 assert.ok([200,202].includes(result.status),JSON.stringify(result));
 assert.ok(request,'the model was called');
 const message=request.messages.find(m=>m.role==='user'&&m.content.includes('childhood_cartography'));
 const data=JSON.parse(message.content.slice(message.content.indexOf('{')));
 const map=data.childhood_cartography,page=displayed(answers).map(r=>r.outcome.id);
 assert.deepEqual(map.display_order.slice(0,5).map(x=>x.id),page.slice(0,5));
 assert.equal(map.display_count,page.length);
 // The parentification profile used to put four theoretical cards first; now measured associations lead.
 const first=displayed(answers)[0];
 assert.ok(first.associated_count>0,'the first subject has at least one empirical association');
});
