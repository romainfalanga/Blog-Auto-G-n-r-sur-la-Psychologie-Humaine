import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createRequire} from 'node:module';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {validateCartography,computeCartography,summarizeCartography,cartographyHighlights,companionCartography} from '../src/cartographie.mjs';
import {loadContext} from '../src/private-context.mjs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {buildCompanionSystemPrompt} from '../src/psychology-framework.mjs';
const require=createRequire(import.meta.url);
const publicPage=require('../../static/js/cartographie.js');
const fail=(status,message,code)=>{throw Object.assign(new Error(message),{status,code});};
const suicide=CARTOGRAPHIE.outcomes.find(o=>o.note==='suicide');
const strongFactors=[...new Set(CARTOGRAPHIE.links.filter(l=>l.strength==='forte'&&!l.definitional).map(l=>l.factor))];

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
const saveRow=(raw,user,{answers,feedback={},locked=true,consent=1,version=1})=>raw.prepare('INSERT INTO cartographies(user_id,answers,feedback,consent,locked_at,version,updated_at) VALUES(?,?,?,?,?,?,?)').run(user,JSON.stringify(answers),JSON.stringify(feedback),consent,locked?1700000000:null,version,1700000000);

test('the generated module is a compact, consistent copy of the public dataset',()=>{
 const data=JSON.parse(readFileSync(new URL('../../data/cartographie.json',import.meta.url),'utf8'));
 assert.equal(CARTOGRAPHIE.factors.length,data.factors.length);
 assert.equal(CARTOGRAPHIE.outcomes.length,data.outcomes.length);
 assert.equal(CARTOGRAPHIE.links.length,data.links.length);
 assert.deepEqual(CARTOGRAPHIE.weights,data.weights);
 assert.ok(CARTOGRAPHIE.links.every(l=>!Object.hasOwn(l,'sources')&&!Object.hasOwn(l,'summary')));
 assert.ok(suicide,'a suicide-related consequence carries its note');
});

test('validation accepts known identifiers and normalizes their order',()=>{
 const [f1,f2]=CARTOGRAPHIE.factors.map(f=>f.id),o1=CARTOGRAPHIE.outcomes[0].id;
 const result=validateCartography({answers:{[f2]:2,[f1]:1},feedback:{[o1]:'non'}},fail);
 assert.deepEqual(Object.keys(result.answers),[f1,f2].sort());
 assert.deepEqual(result.feedback,{[o1]:'non'});
 assert.deepEqual(validateCartography({answers:{}},fail),{answers:{},feedback:{}});
});

test('validation rejects unknown ids, invalid intensities, invalid feedback and wrong shapes',()=>{
 const f=CARTOGRAPHIE.factors[0].id,o=CARTOGRAPHIE.outcomes[0].id;
 for(const input of [{answers:{inconnu:1}},{answers:{[f]:3}},{answers:{[f]:'2'}},{answers:{[f]:true}},{answers:{[f]:0}},{answers:[f]},{answers:null},{},
  {answers:{[f]:1},feedback:{[o]:'peut-être'}},{answers:{[f]:1},feedback:{inconnue:'oui'}},{answers:{[f]:1},feedback:[o]},{answers:JSON.parse('{"__proto__":1}')},{answers:{[f]:1},feedback:{[o]:1}}])
  assert.throws(()=>validateCartography(input,fail),e=>e.status===400&&e.code==='invalid_cartography',JSON.stringify(input));
});

test('the Worker ranking matches the public page compute on the real dataset, with intensities',()=>{
 let seed=7;const rand=()=>(seed=(seed*1103515245+12345)%2147483648)/2147483648;
 const ids=CARTOGRAPHIE.factors.map(f=>f.id);
 for(let n=0;n<60;n++){
  const answers={};for(const id of ids)if(rand()<0.15)answers[id]=rand()<0.5?2:1;
  const worker=computeCartography(answers),page=publicPage.compute(CARTOGRAPHIE,answers);
  assert.deepEqual(worker.results.map(r=>[r.outcome.id,r.tier,r.score]),page.results.map(r=>[r.outcome.id,r.tier,r.score]));
  assert.equal(worker.aceCount,page.aceCount);assert.equal(worker.selectedCount,page.selectedCount);
 }
});

test('the public compute keeps accepting lists and Sets as intensity 1 and weighs intensity 2 by 1.5',()=>{
 const data={weights:{forte:3,moderee:2,limitee:1,theorique:1},factors:[{id:'a'},{id:'b'}],outcomes:[{id:'o',label:'O',kind:'schema'}],links:[{factor:'a',outcome:'o',strength:'forte'},{factor:'b',outcome:'o',strength:'limitee',definitional:true}]};
 assert.equal(publicPage.compute(data,['a']).results[0].score,3);
 assert.equal(publicPage.compute(data,new Set(['a'])).results[0].score,3);
 assert.equal(publicPage.compute(data,{a:1}).results[0].score,3);
 assert.equal(publicPage.compute(data,{a:2}).results[0].score,4.5);
 assert.equal(publicPage.compute(data,{a:2,b:2}).results[0].score,5.25);
 assert.equal(publicPage.compute(data,{a:2,b:2}).results[0].tier,'piste');
 assert.equal(publicPage.compute(data,{a:3}).selectedCount,0);
 assert.equal(computeCartography({a:2,b:2},data).results[0].score,5.25);
});

test('the summary lists causes, convergences, leads and the person’s feedback; definitional links do not converge',()=>{
 const data={weights:{forte:3,critere:2,moderee:2,limitee:1,theorique:1},factors:[{id:'a',label:'A',ace:true},{id:'b',label:'B'},{id:'c',label:'C'}],
  outcomes:[{id:'conv',label:'Convergente',kind:'trouble',code:'6B40',url:'/posts/x/y/'},{id:'def',label:'Définitionnelle',kind:'schema'},{id:'solo',label:'Seule',kind:'tendance'},{id:'theo',label:'Théorique',kind:'tendance'},{id:'sui',label:'Idées suicidaires',kind:'tendance',note:'suicide'}],
  links:[{factor:'a',outcome:'conv',strength:'forte'},{factor:'b',outcome:'conv',strength:'limitee'},{factor:'a',outcome:'def',strength:'moderee',definitional:true},{factor:'b',outcome:'def',strength:'moderee'},{factor:'c',outcome:'solo',strength:'moderee'},{factor:'c',outcome:'theo',strength:'theorique'},{factor:'a',outcome:'sui',strength:'moderee'}]};
 const s=summarizeCartography({a:2,b:1,c:1},{solo:'oui',theo:'non',sui:'oui'},{data});
 assert.equal(s.cause_count,3);assert.equal(s.ace_count,1);
 assert.deepEqual(s.causes,[{id:'a',label:'A',intensity:2},{id:'b',label:'B',intensity:1},{id:'c',label:'C',intensity:1}]);
 assert.deepEqual(s.convergence.map(x=>x.id),['conv']);
 assert.deepEqual(s.convergence[0],{id:'conv',label:'Convergente',kind:'trouble',code:'6B40',tier:'convergence',causes:['A','B'],max_strength:'forte',url:'/posts/x/y/'});
 assert.ok(s.pistes.some(x=>x.id==='def'),'a definitional link alone leaves a lead');
 assert.equal(s.pistes.find(x=>x.id==='solo').feedback,'oui');
 assert.equal(s.theorique_count,1);
 assert.deepEqual(s.dismissed,[{id:'theo',label:'Théorique',tier:'theorique'}]);
 assert.equal(s.suicide_recognized,true);
 const highlights=cartographyHighlights(s).map(x=>x.id);
 assert.equal(highlights[0],'solo');assert.ok(!highlights.includes('sui'));assert.ok(!highlights.includes('theo'));
});

test('only a locked map saved with consent reaches Ambre',()=>{
 const answers={[strongFactors[0]]:1};
 assert.equal(companionCartography(null),null);
 assert.equal(companionCartography({answers:JSON.stringify(answers),feedback:'{}',consent:1,locked_at:null,version:1}),null);
 assert.equal(companionCartography({answers:JSON.stringify(answers),feedback:'{}',consent:0,locked_at:5,version:1}),null);
 assert.equal(companionCartography({answers:JSON.stringify(answers),feedback:'{}',consent:1,locked_at:5,version:3}).version,3);
});

// Worker harness: same in-memory D1 and import rewriting as backend.test.mjs.
const examples=Array.from({length:3},(_,i)=>({id:`posts/c/example-${i}`,title:`Exemple ${i}`,url:`/posts/c/example-${i}/`,character:'P',keywords:[],description:'R'}));
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<title>T</title>';").replace("import { EXAMPLES } from './examples.mjs';",'const EXAMPLES='+JSON.stringify(examples)+';')
 .replace("import { createRecommender } from './recommendations.mjs';",'const createRecommender=()=>({recommend:()=>[],provisionalPortrait:()=>null});');
const rewritten=['chat-backend','entitlements','personal-backend','cartographie'].reduce((s,name)=>s.replace(`'./${name}.mjs'`,JSON.stringify(new URL(`../src/${name}.mjs`,import.meta.url).href)),source);
const {default:worker}=await import('data:text/javascript;base64,'+Buffer.from(rewritten).toString('base64'));
const origin='https://journal.example';
const request=(env,path,{method='GET',data,cookie,originHeader=origin}={})=>worker.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),...(originHeader?{Origin:originHeader}:{}),'CF-Connecting-IP':'192.0.2.9'},body:data?JSON.stringify(data):undefined}),env);
async function register(env,email){const r=await request(env,'/api/auth/register',{method:'POST',data:{email,password:'long secure password 123'}});assert.equal(r.status,200);return {cookie:r.headers.get('set-cookie').split(';')[0],user:(await r.json()).user};}
const send=async(...args)=>{const r=await request(...args);return {status:r.status,...await r.json()};};

test('cartography routes: consent, optimistic version, lock, summary, isolation and deletion',async()=>{
 const env={DB:d1(migrated())},a=await register(env,'carto-a@example.test'),b=await register(env,'carto-b@example.test');
 const [f1,f2]=strongFactors,o=CARTOGRAPHIE.outcomes[0].id;
 assert.equal((await request(env,'/api/cartographie')).status,401);
 let r=await send(env,'/api/cartographie',{cookie:a.cookie});
 assert.deepEqual(r,{status:200,cartographie:null,summary:null});
 const answers={[f1]:2,[f2]:1};
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers,locked:true,expected_version:0}});
 assert.equal(r.status,400);assert.equal(r.code,'consent_required');
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers,consent:'true',locked:true,expected_version:0}});
 assert.equal(r.code,'consent_required');
 assert.equal((await request(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,originHeader:'https://evil.example',data:{answers,consent:true,locked:true,expected_version:0}})).status,403);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers,consent:true,locked:true}});
 assert.equal(r.code,'invalid_version');
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{},consent:true,locked:true,expected_version:0}});
 assert.equal(r.status,400);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{inconnu:1},consent:true,locked:false,expected_version:0}});
 assert.equal(r.code,'invalid_cartography');
 const version=env.DB.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id).version;
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers,feedback:{[o]:'oui'},consent:true,locked:true,expected_version:0}});
 assert.equal(r.status,200);assert.equal(r.ok,true);
 assert.equal(r.cartographie.version,1);assert.equal(r.cartographie.locked,true);assert.equal(typeof r.cartographie.locked_at,'number');assert.equal(r.cartographie.consent,true);
 assert.deepEqual(r.cartographie.answers,answers);assert.deepEqual(r.cartographie.feedback,{[o]:'oui'});
 assert.equal(r.summary.cause_count,2);assert.ok(Array.isArray(r.summary.convergence)&&Array.isArray(r.summary.pistes));
 assert.equal(env.DB.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(a.user.id).version,version,'the map is not a portrait source');
 const lockedAt=r.cartographie.locked_at;
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers,feedback:{[o]:'oui'},consent:true,locked:true,expected_version:0}});
 assert.equal(r.status,409);assert.equal(r.code,'revision_conflict');
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers,feedback:{[o]:'oui'},consent:true,locked:true,expected_version:1}});
 assert.equal(r.cartographie.version,1,'an identical save is idempotent');
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers,feedback:{[o]:'oui'},consent:true,locked:false,expected_version:1}});
 assert.equal(r.cartographie.version,2);assert.equal(r.cartographie.locked,false);assert.equal(r.cartographie.locked_at,null);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{[f1]:1},consent:true,locked:true,expected_version:2}});
 assert.equal(r.cartographie.version,3);assert.equal(r.cartographie.locked,true);assert.deepEqual(r.cartographie.feedback,{});
 assert.ok(r.cartographie.locked_at>=lockedAt);
 r=await send(env,'/api/cartographie',{cookie:a.cookie});
 assert.equal(r.cartographie.version,3);assert.equal(r.summary.cause_count,1);
 assert.deepEqual(await send(env,'/api/cartographie',{cookie:b.cookie}),{status:200,cartographie:null,summary:null});
 const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();
 assert.deepEqual(exported.cartographie.answers,{[f1]:1});assert.equal(exported.cartographie.version,3);
 assert.equal((await (await request(env,'/api/export',{cookie:b.cookie})).json()).cartographie,null);
 assert.deepEqual(await send(env,'/api/cartographie',{method:'DELETE',cookie:a.cookie}),{status:200,ok:true,cartographie:null,summary:null});
 assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM cartographies WHERE user_id=?').get(a.user.id).n,0);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:b.cookie,data:{answers:{[f2]:2},consent:true,locked:true,version:null}});
 assert.equal(r.status,200);assert.equal(r.cartographie.version,1,'version: null is accepted for a first save');
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:b.cookie,data:{answers:{[f2]:1},consent:true,locked:true,version:1}});
 assert.equal(r.cartographie.version,2);
 assert.equal((await request(env,'/api/cartographie',{method:'POST',cookie:b.cookie,data:{}})).status,405);
 assert.equal((await request(env,'/api/account',{method:'DELETE',cookie:b.cookie,data:{password:'long secure password 123'}})).status,200);
 assert.equal(env.DB.raw.prepare('SELECT count(*) n FROM cartographies').get().n,0,'account deletion removes the map');
});

test('the private context carries only a locked, consented map and tolerates a database before 0016',async()=>{
 const raw=migrated(),env={DB:d1(raw)};
 for(const id of ['a','b','c']){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'h','s')").run(id,id+'@example.test');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO subscriptions(user_id) VALUES(?)').run(id);}
 saveRow(raw,'a',{answers:{[strongFactors[0]]:2,[strongFactors[1]]:1},feedback:{[suicide.id]:'oui'},version:4});
 saveRow(raw,'b',{answers:{[strongFactors[0]]:1},locked:false});
 const context=await loadContext(env,'a');
 assert.equal(context.cartography.version,4);assert.equal(context.cartography.cause_count,2);assert.equal(context.cartography.suicide_recognized,true);
 assert.ok(context.sources.every(s=>!String(s.id).startsWith('cartograph')),'the map is not a portrait source');
 assert.equal((await loadContext(env,'b')).cartography,null);
 assert.equal((await loadContext(env,'c')).cartography,null);
 raw.exec('DROP TABLE cartographies');
 assert.equal((await loadContext(env,'a')).cartography,null);
});

function chatSetup(catalogue=new Map()){
 const raw=migrated(),DB=d1(raw);let clock=Math.floor(Date.now()/1000),serial=0;
 const utils={error:fail,json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null,recommend:()=>[]}};
 const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue}),env={DB,OPENROUTER_API_KEY:'mock-only'};
 raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES('a','a@example.com','hash','salt')").run();raw.prepare("INSERT INTO profiles(user_id) VALUES('a')").run();raw.prepare("INSERT INTO memories(user_id) VALUES('a')").run();raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES('a','active',?)").run(clock+86400);
 async function call(path,method='GET',data){const req=new Request('https://example.com'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:'a'},path,url)||await personal.routes(req,env,{id:'a'},path,url);return {status:res.status,...await res.json()};}catch(e){return {status:e.status||500,code:e.code,message:e.message};}}
 return {raw,env,call};
}
function withFetch(t,fn){const previous=globalThis.fetch;globalThis.fetch=fn;t.after(()=>{globalThis.fetch=previous;});}

test('the welcome message invites to fill the map, then starts from a locked one',async()=>{
 const s=chatSetup();
 let state=await s.call('/api/chat');
 assert.equal(state.welcome.id,'ambre-welcome-v7');assert.match(state.welcome.content,/\/cartographie\//);
 saveRow(s.raw,'a',{answers:{[strongFactors[0]]:1},locked:false});
 state=await s.call('/api/chat');
 assert.equal(state.welcome.id,'ambre-welcome-v7-draft');assert.match(state.welcome.content,/verrouillerez/);
 const answers=Object.fromEntries(strongFactors.slice(0,4).map(id=>[id,2]));
 const summary=summarizeCartography(answers,{});
 const recognized=summary.convergence.find(x=>x.note!=='suicide')||summary.pistes[0];
 s.raw.prepare('UPDATE cartographies SET answers=?,feedback=?,locked_at=1700000000,version=5 WHERE user_id=?').run(JSON.stringify(answers),JSON.stringify({[recognized.id]:'oui',[suicide.id]:'oui'}),'a');
 state=await s.call('/api/chat');
 assert.equal(state.welcome.id,'ambre-welcome-v7-c5');
 assert.match(state.welcome.content,/4 éléments de votre enfance sont cochés/);
 assert.ok(state.welcome.content.includes('« '+recognized.label+' »'));
 assert.match(state.welcome.content,/Voulez-vous que nous commencions par/);
 assert.match(state.welcome.content,/pas un diagnostic/);
 assert.match(state.welcome.content,/3114/);
 assert.ok(!state.welcome.content.includes('commencions par « '+suicide.label));
});

test('Ambre receives the locked map, its guidance and matching catalogue readings from the first message',async t=>{
 const answers=Object.fromEntries(strongFactors.slice(0,5).map(id=>[id,1]));
 const summary=summarizeCartography(answers,{});
 const illustrated=[...summary.convergence,...summary.pistes].find(x=>x.url?.startsWith('/posts/'));
 assert.ok(illustrated,'the dataset illustrates a consequence with a catalogue article');
 const articleId=illustrated.url.slice(1,-1),dismissed=[...summary.convergence,...summary.pistes].find(x=>x.id!==illustrated.id);
 const s=chatSetup(new Map([[articleId,{id:articleId,title:'Article illustratif',url:illustrated.url,character:'Léa'}]]));
 saveRow(s.raw,'a',{answers,feedback:{[illustrated.id]:'oui',...(dismissed?{[dismissed.id]:'non'}:{})},version:2});
 assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);
 let request;
 withFetch(t,async(_url,options)=>{request=JSON.parse(options.body);return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({content:'Partons de ce que vous avez reconnu.',questions:[],reading_suggestions:[{id:articleId,reason:'Elle illustre ce que vous avez reconnu.'}],proposals:[]})},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:10,cost:0}}));});
 const result=await s.call('/api/chat/messages','POST',{content:'Par où commencer ?',idempotency_key:'carto_first_001'});
 assert.ok([200,202].includes(result.status),JSON.stringify(result));
 const system=request.messages[0].content,data=JSON.parse(request.messages[1].content.slice(request.messages[1].content.indexOf('{')));
 assert.match(system,/CARTOGRAPHIE DE L’ENFANCE/);assert.match(system,/jamais un diagnostic/);assert.match(system,/3114/);
 const map=data.childhood_cartography;
 assert.equal(map.version,2);assert.equal(map.cause_count,5);
 assert.ok(map.recognized.some(r=>r.id===illustrated.id));
 if(dismissed)assert.ok(map.dismissed.some(r=>r.id===dismissed.id));
 assert.ok(!JSON.stringify(map).includes('"url"'),'no public URL is handed to the model');
 assert.ok([...map.convergence,...map.pistes].some(x=>x.reading_id===articleId));
 assert.ok(data.reading_candidates.some(c=>c.id===articleId));
 const final=await s.call('/api/chat');
 assert.equal(final.messages.find(m=>m.role==='assistant').reading_suggestions[0].id,articleId);
});

test('without a locked map the system prompt has no cartography section',()=>{
 assert.ok(!buildCompanionSystemPrompt({query:'Bonjour'}).includes('CARTOGRAPHIE DE L’ENFANCE'));
 assert.ok(buildCompanionSystemPrompt({query:'Bonjour',cartography:true}).length<18000);
});
