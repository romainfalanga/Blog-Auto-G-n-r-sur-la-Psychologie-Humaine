import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createRequire} from 'node:module';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {validateCartography,computeCartography,summarizeCartography,cartographyHighlights,companionCartography,cartographyEvidence,chooseAmbreNotes,appendAmbreNotes} from '../src/cartographie.mjs';
import {loadContext} from '../src/private-context.mjs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {buildCompanionSystemPrompt} from '../src/psychology-framework.mjs';
import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
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
 assert.deepEqual(CARTOGRAPHIE.retired_outcomes.map(o=>o.id),data.retired_outcomes.map(o=>o.id));
 assert.equal(CARTOGRAPHIE.links.length,data.links.length);
 assert.deepEqual(CARTOGRAPHIE.weights,data.weights);
 assert.ok(CARTOGRAPHIE.links.every(l=>!Object.hasOwn(l,'sources')));
 assert.deepEqual(CARTOGRAPHIE.links.map(l=>l.summary),data.links.map(l=>l.summary.trim()),'link summaries let Ambre explain a link without the articles');
 assert.ok(suicide,'a suicide-related consequence carries its note');
});

test('validation accepts known identifiers and normalizes their order',()=>{
 const [f1,f2]=CARTOGRAPHIE.factors.map(f=>f.id),o1=CARTOGRAPHIE.outcomes[0].id;
 const result=validateCartography({answers:{[f2]:2,[f1]:1},feedback:{[o1]:'non'}},fail);
 assert.deepEqual(Object.keys(result.answers),[f1,f2].sort());
 assert.deepEqual(result.feedback,{[o1]:'non'});
 assert.deepEqual(validateCartography({answers:{}},fail),{answers:{},feedback:{},notes:{factors:{},outcomes:{}},resources:{}});
});

test('validation rejects unknown ids, invalid intensities, invalid feedback and wrong shapes',()=>{
 const f=CARTOGRAPHIE.factors[0].id,o=CARTOGRAPHIE.outcomes[0].id;
 for(const input of [{answers:{inconnu:1}},{answers:{[f]:3}},{answers:{[f]:'2'}},{answers:{[f]:true}},{answers:[f]},{answers:null},{},
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

test('the public compute keeps accepting lists and Sets without an intensity multiplier',()=>{
 const data={weights:{forte:3,moderee:2,limitee:1,theorique:1},factors:[{id:'a'},{id:'b'}],outcomes:[{id:'o',label:'O',kind:'schema'}],links:[{factor:'a',outcome:'o',strength:'forte'},{factor:'b',outcome:'o',strength:'limitee',definitional:true}]};
 assert.equal(publicPage.compute(data,['a']).results[0].score,3);
 assert.equal(publicPage.compute(data,new Set(['a'])).results[0].score,3);
 assert.equal(publicPage.compute(data,{a:1}).results[0].score,3);
 assert.equal(publicPage.compute(data,{a:2}).results[0].score,3);
 assert.equal(publicPage.compute(data,{a:2,b:2}).results[0].score,3);
 assert.equal(publicPage.compute(data,{a:2,b:2}).results[0].tier,'piste');
 assert.equal(publicPage.compute(data,{a:3}).selectedCount,0);
 assert.equal(computeCartography({a:2,b:2},data).results[0].score,3);
});

test('the summary lists causes, convergences, leads and the person’s feedback; definitional links do not converge',()=>{
 const data={weights:{forte:3,critere:2,moderee:2,limitee:1,theorique:1},factors:[{id:'a',label:'A',ace:true},{id:'b',label:'B'},{id:'c',label:'C'}],
  outcomes:[{id:'conv',label:'Convergente',kind:'trouble',code:'6B40',url:'/posts/x/y/'},{id:'def',label:'Définitionnelle',kind:'schema'},{id:'solo',label:'Seule',kind:'tendance'},{id:'theo',label:'Théorique',kind:'tendance'},{id:'sui',label:'Idées suicidaires',kind:'tendance',note:'suicide'}],
  links:[{factor:'a',outcome:'conv',strength:'forte'},{factor:'b',outcome:'conv',strength:'moderee'},{factor:'a',outcome:'def',strength:'moderee',definitional:true},{factor:'b',outcome:'def',strength:'moderee'},{factor:'c',outcome:'solo',strength:'moderee'},{factor:'c',outcome:'theo',strength:'theorique'},{factor:'a',outcome:'sui',strength:'moderee'}]};
 const s=summarizeCartography({a:2,b:1,c:1},{solo:'oui',theo:'non',sui:'oui'},{data});
 assert.equal(s.cause_count,3);assert.equal(s.ace_count,1);
 assert.deepEqual(s.causes,[{id:'a',label:'A',intensity:2},{id:'b',label:'B',intensity:1},{id:'c',label:'C',intensity:1}]);
 assert.deepEqual(s.convergence.map(x=>x.id),['conv']);
 assert.deepEqual(s.convergence[0],{id:'conv',label:'Convergente',kind:'trouble',associated_count:2,code:'6B40',tier:'convergence',causes:['A','B'],max_strength:'forte',relation_types:['empirical_association'],evidence_groups:2,url:'/posts/x/y/'});
 assert.ok(s.pistes.some(x=>x.id==='def'),'a definitional link alone leaves a lead');
 assert.equal(s.pistes.find(x=>x.id==='solo').feedback,'oui');
 assert.equal(s.theorique_count,1);
 assert.deepEqual(s.dismissed,[{id:'theo',label:'Théorique',tier:'theorique',associated_count:0}]);
 assert.equal(s.suicide_recognized,true);
 const highlights=cartographyHighlights(s).map(x=>x.id);
 assert.deepEqual(highlights,['conv','def','solo']);assert.ok(!highlights.includes('sui'));assert.ok(!highlights.includes('theo'));
});

test('only a locked map saved with consent reaches Ambre',()=>{
 const answers={[strongFactors[0]]:1};
 assert.equal(companionCartography(null),null);
 assert.equal(companionCartography({answers:JSON.stringify(answers),feedback:'{}',consent:1,locked_at:null,version:1}),null);
 assert.equal(companionCartography({answers:JSON.stringify(answers),feedback:'{}',consent:0,locked_at:5,version:1}),null);
 assert.equal(companionCartography({answers:JSON.stringify(answers),feedback:'{}',consent:1,locked_at:5,version:3}).version,3);
});

test('the bounded summary leaves room for different categories and records omissions',()=>{
 const outcomes=Array.from({length:20},(_,i)=>({id:'schema_'+i,label:'Schéma '+i,kind:'schema',category:'schema'}));
 outcomes.push({id:'z_recognized',label:'Reconnu',kind:'schema',category:'schema'},{id:'other',label:'Autre',kind:'tendance',category:'tendency'},{id:'context',label:'Contexte',kind:'trouble',category:'diagnosis',contextual_only:true});
 const data={factors:[{id:'a',label:'A'},{id:'b',label:'B'}],outcomes,links:outcomes.flatMap(o=>['a','b'].map(factor=>({factor,outcome:o.id,strength:'moderee',relation_type:'empirical_association',summary:'Association documentée.',sources:[]})))};
 const summary=summarizeCartography({a:1,b:1},{z_recognized:'oui'},{data});
 assert.equal(summary.convergence.filter(x=>x.category==='schema').length,5);
 // v4.1 (lot C): recognition still wins a place under the category cap, but the list
 // keeps the public display order shared with the page.
 assert.ok(summary.convergence.some(x=>x.id==='z_recognized'),'a recognized subject is selected despite the category cap');
 assert.deepEqual(summary.convergence.map(x=>x.id),computeCartography({a:1,b:1},data,{z_recognized:'oui'}).results.map(r=>r.outcome.id).filter(id=>summary.convergence.some(x=>x.id===id)),'selection keeps the display order');
 assert.ok(summary.convergence.some(x=>x.id==='other'));
 assert.equal(summary.truncated,true);
 assert.ok(!cartographyHighlights(summary).some(x=>x.id==='context'));
 assert.ok(!cartographyEvidence({a:1,b:1},{},{data}).evidence.some(x=>x.id==='context'));
 // v5 (final review): an older recognition of a hidden (contextual) subject stays stored but is no longer sent to Ambre.
 assert.ok(!cartographyEvidence({a:1,b:1},{context:'oui'},{data}).evidence.some(x=>x.id==='context'));
 assert.ok(!summarizeCartography({a:1,b:1},{context:'oui'},{data}).recognized.some(x=>x.id==='context'));
});

test('explicit negative answers are preserved without becoming reported experiences',()=>{
 const answers={abus_physique:0,abus_sexuel:2};
 assert.deepEqual(validateCartography({answers},fail).answers,answers);
 const summary=summarizeCartography(answers);
 assert.equal(summary.cause_count,1);assert.equal(summary.answered_count,2);
 assert.deepEqual(summary.explicit_negative,[{id:'abus_physique',label:'Violences physiques'}]);
 assert.deepEqual(summary.causes.map(x=>x.id),['abus_sexuel']);
 assert.equal(computeCartography({abus_physique:0}).results.length,0);
 assert.throws(()=>validateCartography({answers:{divorce:1,separation_conflit:2}},fail),e=>e.code==='invalid_cartography');
 assert.equal(computeCartography({divorce:1,separation_conflit:2}).aceCount,1,'old contradictory maps do not double count parental separation');
});

test('childhood diagnoses remain context and age restrictions survive recognition and evidence',()=>{
 const answers={negligence_emotionnelle:1,negligence_physique:1,soins_instables:2};
 const summary=summarizeCartography(answers,{icd_6b44:'oui'});
 // v5 (final review): contextual subjects are neither displayed nor sent to Ambre;
 // an older recognition stays stored (validated) but leaves the summary and the evidence.
 assert.deepEqual(summary.contextual,[]);
 assert.ok(!summary.display_order.some(o=>o.id==='icd_6b44'));
 assert.ok(!summary.recognized.some(o=>o.id==='icd_6b44'));
 assert.deepEqual(validateCartography({answers,feedback:{icd_6b44:'oui'}},fail).feedback,{icd_6b44:'oui'});
 assert.ok(!cartographyHighlights(summary,100).some(o=>['icd_6b44','icd_6b45'].includes(o.id)));
 assert.ok(!cartographyEvidence(answers,{icd_6b44:'oui'}).evidence.some(o=>o.id==='icd_6b44'));
});

test('public and backend agree for every factor pair, including explicit negatives',()=>{
 const fullData=JSON.parse(readFileSync(new URL('../../data/cartographie.json',import.meta.url),'utf8'));
 const ids=CARTOGRAPHIE.factors.map(f=>f.id);
 const row=({outcome,links,...rest})=>({id:outcome.id,...rest});
 const resultShape=result=>({...result,results:result.results.map(row),sections:Object.fromEntries(Object.entries(result.sections).map(([k,v])=>[k,v.map(row)]))});
 for(let a=0;a<ids.length;a++)for(let b=a;b<ids.length;b++){
  const negative=ids.find(id=>id!==ids[a]&&id!==ids[b]);
  const answers={[ids[a]]:1,[ids[b]]:2,[negative]:0};
  const outs=CARTOGRAPHIE.outcomes,feedback=a%3===0?{[outs[(a*7+b)%outs.length].id]:b%2?'oui':'non',[outs[(a+b*5)%outs.length].id]:'non'}:{};
  const page=publicPage.compute(CARTOGRAPHIE,answers,feedback),worker=computeCartography(answers,CARTOGRAPHIE,feedback),fullPage=publicPage.compute(fullData,answers,feedback);
  assert.deepEqual(worker,page);
  assert.deepEqual(resultShape(worker),resultShape(fullPage));
 }
});

test('retired identifiers survive validation and preserve meaning without suggestions or migration',()=>{
 const feedback=Object.fromEntries(CARTOGRAPHIE.retired_outcomes.map(o=>[o.id,'oui']));
 feedback.icd_6b81='oui';
 const notes={factors:{},outcomes:Object.fromEntries(CARTOGRAPHIE.retired_outcomes.map(o=>[o.id,'Mes mots enregistrés auparavant.']))};
 const answers={abus_physique:1,negligence_physique:2,migration:1,exil:1};
 const cleaned=validateCartography({answers,feedback,notes},fail);
 assert.deepEqual(cleaned.feedback,feedback);assert.deepEqual(cleaned.notes,notes);
 assert.ok(!Object.hasOwn(cleaned.feedback,'att_desorganisation_enfant'));
 assert.ok(!Object.hasOwn(cleaned.feedback,'icd_6b81_boulimie'));
 const summary=summarizeCartography(answers,feedback,{notes});
 const automatic=[...summary.convergence,...summary.pistes,...summary.contextual,...summary.exploratory,...cartographyHighlights(summary,100)];
 const evidence=cartographyEvidence(answers,feedback).evidence;
 for(const old of CARTOGRAPHIE.retired_outcomes){
  // v5 (final review): stored and validated, but no longer sent to Ambre.
  assert.ok(!summary.recognized.some(o=>o.id===old.id));
  assert.ok(!summary.notes.some(n=>n.id===old.id));
  assert.ok(!automatic.some(o=>o.id===old.id));assert.ok(!evidence.some(o=>o.id===old.id));
  assert.deepEqual(chooseAmbreNotes([{target:'outcome',target_id:old.id,text:'Inférence nouvelle.'}]),[]);
 }
 assert.ok(!summary.recognized.some(o=>['att_desorganisation_enfant','icd_6b81_boulimie'].includes(o.id)));
});

test('the complete serialized evidence, including wrappers and sources, respects its budget',()=>{
 const answers=Object.fromEntries(CARTOGRAPHIE.factors.filter(f=>f.id!=='separation_conflit').map(f=>[f.id,1]));
 for(const budget of [1500,3000,6000]){
  const result=cartographyEvidence(answers,{}, {budget});
  assert.ok(JSON.stringify(result).length<=budget,`${JSON.stringify(result).length} > ${budget}`);
  assert.equal(result.evidence_truncated,true);
  assert.ok(result.evidence.length>0);
  const references=new Set(result.evidence.flatMap(o=>o.links.flatMap(l=>l.source_ids)));
  assert.deepEqual(new Set(result.evidence_sources.map(s=>s.id)),references);
 }
});

// Worker harness: same in-memory D1 and import rewriting as backend.test.mjs.
const examples=Array.from({length:3},(_,i)=>({id:`posts/c/example-${i}`,title:`Exemple ${i}`,url:`/posts/c/example-${i}/`,character:'P',keywords:[],description:'R'}));
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<title>T</title>';").replace("import { EXAMPLES } from './examples.mjs';",'const EXAMPLES='+JSON.stringify(examples)+';')
 .replace("import { createRecommender } from './recommendations.mjs';",'const createRecommender=()=>({recommend:()=>[],provisionalPortrait:()=>null});');
const rewritten=source.replace(/from '(\.\/[^']+)'/g,(_,specifier)=>'from '+JSON.stringify(new URL('../src/'+specifier.slice(2),import.meta.url).href));
const {default:worker}=await import('data:text/javascript;base64,'+Buffer.from(rewritten).toString('base64'));
const origin='https://journal.example';
const request=(env,path,{method='GET',data,cookie,originHeader=origin}={})=>worker.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),...(originHeader?{Origin:originHeader}:{}),'CF-Connecting-IP':'192.0.2.9'},body:data?JSON.stringify(data):undefined}),env);
async function register(env,email){const r=await request(env,'/api/auth/register',{method:'POST',data:{email,password:'long secure password 123'}});assert.equal(r.status,200);return {cookie:r.headers.get('set-cookie').split(';')[0],user:(await r.json()).user};}
const send=async(...args)=>{const r=await request(...args);return {status:r.status,...await r.json()};};

test('autosave keeps historical feedback and notes across real API validation',async()=>{
 const env={DB:d1(migrated())},a=await register(env,'carto-history@example.test');
 const feedback={att_desorganise:'oui',pat_migration_adjustment:'non',icd_6d10:'oui',icd_6b81:'oui'};
 const notes={factors:{migration:'Un souvenir.'},outcomes:{att_desorganise:'Je parlais de ma relation adulte.',pat_migration_adjustment:'Un ancien repère.'}};
 let r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{migration:1},feedback,notes,consent:true,locked:true,version:0}});
 assert.equal(r.status,200);assert.deepEqual(r.cartographie.feedback,feedback);assert.deepEqual(r.cartographie.notes,notes);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{...r.cartographie,answers:{migration:1,abus_physique:2},version:r.cartographie.version}});
 assert.equal(r.status,200);assert.deepEqual(r.cartographie.feedback,feedback);assert.deepEqual(r.cartographie.notes,notes);
 r=await send(env,'/api/cartographie',{cookie:a.cookie});
 assert.equal(r.status,200);assert.deepEqual(r.cartographie.feedback,feedback);
 assert.ok(!r.summary.recognized.some(o=>['att_desorganisation_enfant','icd_6b81_boulimie'].includes(o.id)));
 assert.ok(!r.summary.recognized.some(o=>o.id==='att_desorganise'),'a retired recognition stays stored but is not sent to Ambre');assert.deepEqual(r.cartographie.notes,notes);
 env.DB.raw.close();
});

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
 assert.equal(context.cartography.version,4);assert.equal(context.cartography.cause_count,2);assert.equal(context.cartography.suicide_recognized,false,'an older recognition of a hidden suicide card is no longer sent');
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
 raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES('seed','a','initial','ready','Analyse initiale de test.','seed-initial',0,0)").run();
 async function call(path,method='GET',data){const req=new Request('https://example.com'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:'a'},path,url)||await personal.routes(req,env,{id:'a'},path,url);return {status:res.status,...await res.json()};}catch(e){return {status:e.status||500,code:e.code,message:e.message};}}
 return {raw,env,call};
}
function withFetch(t,fn){const previous=globalThis.fetch;globalThis.fetch=fn;t.after(()=>{globalThis.fetch=previous;});}
// Complete only the prerequisites unrelated to each test's map-access condition.
// The common fixture opts in and locks maps, so restore both original flags (or
// remove its synthetic map) before exercising missing/unconsented/unlocked maps.
function seedChatPrerequisites(raw){
 const map=raw.prepare("SELECT consent,locked_at FROM cartographies WHERE user_id='a'").get();
 seedCompanionPrerequisites(raw,'a');
 if(map)raw.prepare("UPDATE cartographies SET consent=?,locked_at=? WHERE user_id='a'").run(map.consent,map.locked_at);
 else raw.prepare("DELETE FROM cartographies WHERE user_id='a'").run();
}


test('the welcome message invites to fill the map, then starts from a locked one',async()=>{
 const s=chatSetup();
 let state=await s.call('/api/chat');
 assert.equal(state.welcome.id,'ambre-welcome-v9');assert.match(state.welcome.content,/\/cartographie\//);assert.ok(!state.welcome.content.split('\n\n')[1].includes('/cartographie/'),'an invitation stays near the end');
 saveRow(s.raw,'a',{answers:{[strongFactors[0]]:1},locked:false});
 state=await s.call('/api/chat');
 assert.equal(state.welcome.id,'ambre-welcome-v9-draft');assert.match(state.welcome.content,/enregistrez-la/);
 const answers=Object.fromEntries(strongFactors.slice(0,4).map(id=>[id,2]));
 const summary=summarizeCartography(answers,{});
 const recognized=summary.convergence.find(x=>x.note!=='suicide')||summary.pistes[0];
 s.raw.prepare('UPDATE cartographies SET answers=?,feedback=?,locked_at=1700000000,version=5 WHERE user_id=?').run(JSON.stringify(answers),JSON.stringify({[recognized.id]:'oui',[suicide.id]:'oui'}),'a');
 state=await s.call('/api/chat');
 assert.equal(state.welcome.id,'ambre-welcome-v9-c5');
 const second=state.welcome.content.split('\n\n')[1];
 assert.match(second,/carte de travail commune : 4 éléments de votre enfance y sont cochés/,'a saved map is introduced in the second paragraph');
 assert.match(second,/je peux y ajouter une courte note/);
 assert.ok(state.welcome.content.includes('« '+recognized.label+' »'));
 assert.match(state.welcome.content,/Voulez-vous que nous commencions par/);
 assert.match(state.welcome.content,/pas un diagnostic/);
 // v5 (final review): the hidden suicide card no longer reaches Ambre; safety relies on the conversation.
 assert.ok(!state.welcome.content.includes(suicide.label));
});

test('Ambre receives the locked map, its guidance and the research summaries of recognized or converging consequences',async t=>{
 const answers=Object.fromEntries(strongFactors.slice(0,5).map(id=>[id,1]));
 const summary=summarizeCartography(answers,{}),all=[...summary.convergence,...summary.pistes];
 const recognized=summary.pistes[0]||summary.convergence[0],dismissed=summary.convergence.find(x=>x.id!==recognized.id);
 const s=chatSetup(new Map(all.filter(x=>x.url?.startsWith('/posts/')).map(x=>[x.url.slice(1,-1),{id:x.url.slice(1,-1),title:'Article',url:x.url}])));
 saveRow(s.raw,'a',{answers,feedback:{[recognized.id]:'oui',...(dismissed?{[dismissed.id]:'non'}:{})},version:2});
 seedChatPrerequisites(s.raw);
 assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);
 let request;
 withFetch(t,async(_url,options)=>{request=JSON.parse(options.body);return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({content:'Partons de ce que vous avez reconnu.',questions:[],proposals:[],cartography_notes:[]})},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:10,cost:0}}));});
 const result=await s.call('/api/chat/messages','POST',{content:'Par où commencer ?',idempotency_key:'carto_first_001'});
 assert.ok([200,202].includes(result.status),JSON.stringify(result));
 const system=request.messages[0].content,data=JSON.parse(request.messages[1].content.slice(request.messages[1].content.indexOf('{')));
 assert.match(system,/CARTOGRAPHIE DE L’ENFANCE/);assert.match(system,/jamais un diagnostic/);assert.match(system,/3114/);assert.match(system,/evidence donne/);
 const map=data.childhood_cartography;
 assert.equal(map.version,2);assert.equal(map.cause_count,Object.keys(answers).length);
 assert.ok(map.recognized.some(r=>r.id===recognized.id));
 if(dismissed)assert.ok(map.dismissed.some(r=>r.id===dismissed.id));
 assert.ok(!JSON.stringify(map).includes('"url"')&&!JSON.stringify(map).includes('reading_id'),'no article reference is handed to the model');
 assert.equal(map.evidence[0].id,recognized.id,'recognized consequences come first');
 if(dismissed)assert.ok(!map.evidence.some(e=>e.id===dismissed.id),'a dismissed consequence brings no research summary');
 const expected=new Set([recognized.id,...summary.convergence.filter(x=>x.id!==dismissed?.id).map(x=>x.id)]);
 assert.ok(map.evidence.every(e=>expected.has(e.id)));
 for(const e of map.evidence)for(const link of e.links){assert.ok(Object.values(Object.fromEntries(CARTOGRAPHIE.factors.map(f=>[f.id,f.label]))).includes(link.cause));assert.ok(['forte','critere','moderee','limitee','theorique'].includes(link.strength));assert.ok(link.summary.length>10);}
 assert.ok(JSON.stringify(map.evidence).length<=7500);
 assert.equal(Object.hasOwn(data,'reading_candidates'),false);
});
test('without a locked map the system prompt has no cartography section',()=>{
 assert.ok(!buildCompanionSystemPrompt({query:'Bonjour'}).includes('CARTOGRAPHIE DE L’ENFANCE'));
 assert.ok(buildCompanionSystemPrompt({query:'Bonjour',cartography:true}).length<18000);
});

// --- V3: the member's notes, resources and Ambre's notes ---
const [fa,fb]=CARTOGRAPHIE.factors.map(f=>f.id),[oa,ob]=CARTOGRAPHIE.outcomes.map(o=>o.id),[pa,pb]=CARTOGRAPHIE.protections.map(p=>p.id);
const fullRow=(raw,user,{answers={},feedback={},notes={},resources={},ambre=[],locked=true,consent=1,version=1})=>raw.prepare('INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)').run(user,JSON.stringify(answers),JSON.stringify(feedback),JSON.stringify(notes),JSON.stringify(resources),JSON.stringify(ambre),consent,locked?1700000000:null,version,1700000000);
const stored=(raw,user='a')=>JSON.parse(raw.prepare('SELECT ambre_notes FROM cartographies WHERE user_id=?').get(user).ambre_notes);

test('the generated module carries the protective resources',()=>{
 const data=JSON.parse(readFileSync(new URL('../../data/cartographie.json',import.meta.url),'utf8'));
 assert.deepEqual(CARTOGRAPHIE.protections,data.protections.map(p=>({id:p.id,label:p.label})));
});

test('notes and resources are validated: known ids, cleaned text, bounded length and count',()=>{
 const r=validateCartography({answers:{},notes:{factors:{[fa]:'  Mon père criait\r\nsouvent.\u0000  ',[fb]:'   '},outcomes:{[oa]:'Je me tais en réunion.'}},resources:{[pa]:1}},fail);
 assert.deepEqual(r.notes,{factors:{[fa]:'Mon père criait\nsouvent.'},outcomes:{[oa]:'Je me tais en réunion.'}});
 assert.deepEqual(r.resources,{[pa]:1});
 assert.equal(validateCartography({answers:{},notes:{factors:{[fa]:'x'.repeat(1000)}}},fail).notes.factors[fa].length,1000);
 const many=Object.fromEntries(CARTOGRAPHIE.factors.slice(0,50).map(f=>[f.id,'note'])),manyOutcomes=Object.fromEntries(CARTOGRAPHIE.outcomes.slice(0,31).map(o=>[o.id,'note']));
 assert.equal(Object.keys(validateCartography({answers:{},notes:{factors:many,outcomes:Object.fromEntries(Object.entries(manyOutcomes).slice(0,30))}},fail).notes.outcomes).length,30);
 for(const input of [{answers:{},notes:{factors:{inconnu:'x'}}},{answers:{},notes:{outcomes:{[fa]:'x'}}},{answers:{},notes:{factors:{[fa]:'x'.repeat(1001)}}},{answers:{},notes:{factors:{[fa]:3}}},{answers:{},notes:{general:{a:'x'}}},{answers:{},notes:[]},{answers:{},notes:{factors:[]}},
  {answers:{},notes:{factors:many,outcomes:manyOutcomes}},{answers:{},resources:{inconnue:1}},{answers:{},resources:{[pa]:2}},{answers:{},resources:{[pa]:true}},{answers:{},resources:[pa]},{answers:{},resources:{[fa]:1}}])
  assert.throws(()=>validateCartography(input,fail),e=>e.status===400&&e.code==='invalid_cartography',JSON.stringify(input).slice(0,120));
});

test('the summary attaches the member’s notes, resources and Ambre’s notes to their labels',()=>{
 const s=summarizeCartography({[fa]:1},{},{notes:{factors:{[fa]:'Mes mots.'},outcomes:{[oa]:'Aujourd’hui.'}},resources:{[pb]:1},ambreNotes:[{id:'n1',target:'outcome',target_id:ob,text:'Elle dit se taire.',created_at:5},{id:'n2',target:'general',target_id:null,text:'Note libre.',created_at:6},{id:'n3',target:'factor',target_id:'disparu',text:'x',created_at:7}]});
 assert.deepEqual(s.notes,[{target:'factor',id:fa,label:CARTOGRAPHIE.factors[0].label,text:'Mes mots.'},{target:'outcome',id:oa,label:CARTOGRAPHIE.outcomes[0].label,text:'Aujourd’hui.'}]);
 assert.deepEqual(s.resources,[{id:pb,label:CARTOGRAPHIE.protections[1].label}]);
 assert.deepEqual(s.ambre_notes,[{id:'n1',target:'outcome',target_id:ob,text:'Elle dit se taire.',created_at:5,label:CARTOGRAPHIE.outcomes[1].label},{id:'n2',target:'general',target_id:null,text:'Note libre.',created_at:6}]);
 const empty=summarizeCartography({[fa]:1},{});
 assert.deepEqual([empty.notes,empty.resources,empty.ambre_notes],[[],[],[]]);
});

test('Ambre’s notes are validated, deduplicated and capped at sixty, oldest first out',()=>{
 const existing=[{id:'n1',target:'factor',target_id:fa,text:'Mon père criait souvent.',created_at:1}];
 const chosen=chooseAmbreNotes([{target:'factor',target_id:fa,text:'mon pere criait  souvent'},{target:'outcome',target_id:oa,text:' Je me tais\nen réunion. '},{target:'general',target_id:null,text:'Note générale.'},{target:'factor',target_id:fb,text:'quatrième'}],existing);
 assert.deepEqual(chosen,[{target:'outcome',target_id:oa,text:'Je me tais en réunion.',label:CARTOGRAPHIE.outcomes[0].label},{target:'general',target_id:null,text:'Note générale.'}],'three per turn, duplicate dropped');
 for(const bad of [{target:'factor',target_id:'inconnu',text:'x y'},{target:'factor',target_id:oa,text:'x y'},{target:'outcome',target_id:null,text:'x y'},{target:'general',target_id:fa,text:'x y'},{target:'autre',target_id:null,text:'x y'},{target:'general',target_id:null,text:'x'.repeat(401)},{target:'general',target_id:null,text:' … '},{target:'general',target_id:null,text:4},null])
  assert.deepEqual(chooseAmbreNotes([bad]),[],JSON.stringify(bad)?.slice(0,80));
 assert.deepEqual(chooseAmbreNotes([{target:'general',target_id:null,text:'Même'},{target:'general',target_id:null,text:'même.'}]).length,1);
 assert.deepEqual(chooseAmbreNotes('non'),[]);
 let n=0;const list=appendAmbreNotes(Array.from({length:59},(_,i)=>({id:'o'+i,target:'general',target_id:null,text:'t'+i,created_at:i})),[{target:'general',target_id:null,text:'a',label:'ignored'},{target:'factor',target_id:fa,text:'b'}],{now:()=>99,random:()=>'new'+(++n)});
 assert.equal(list.length,60);assert.equal(list[0].id,'o1');assert.deepEqual(list.at(-1),{id:'new2',target:'factor',target_id:fa,text:'b',created_at:99});
});

test('V3 routes: notes and resources round-trip, autosave lock rule, Ambre’s notes survive PUT and can be removed',async()=>{
 const env={DB:d1(migrated())},a=await register(env,'v3-a@example.test'),b=await register(env,'v3-b@example.test'),raw=env.DB.raw;
 let r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{},notes:{factors:{},outcomes:{}},resources:{},consent:true,locked:true,version:null}});
 assert.equal(r.status,400);assert.equal(r.code,'invalid_cartography');
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{},notes:{factors:{[fa]:' Ce que j’en retiens. '}},resources:{},consent:true,locked:true,version:null}});
 assert.equal(r.status,200,'a note alone is enough to save');
 assert.deepEqual(r.cartographie.notes,{factors:{[fa]:'Ce que j’en retiens.'},outcomes:{}});
 assert.deepEqual(r.cartographie.resources,{});assert.deepEqual(r.cartographie.ambre_notes,[]);assert.equal(r.cartographie.version,1);assert.equal(r.cartographie.locked,true);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:b.cookie,data:{answers:{},resources:{[pa]:1},consent:true,locked:true,version:null}});
 assert.equal(r.status,200,'a resource alone is enough to save');assert.deepEqual(r.summary.resources,[{id:pa,label:CARTOGRAPHIE.protections[0].label}]);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{},notes:{factors:{inconnu:'x'}},consent:true,locked:true,version:1}});
 assert.equal(r.code,'invalid_cartography');
 raw.prepare('UPDATE cartographies SET ambre_notes=? WHERE user_id=?').run(JSON.stringify([{id:'n1',target:'factor',target_id:fa,text:'Première note.',created_at:10},{id:'n2',target:'general',target_id:null,text:'Deuxième note.',created_at:11}]),a.user.id);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{[fa]:2},notes:{factors:{[fa]:'Ce que j’en retiens.'},outcomes:{[oa]:'Aujourd’hui.'}},resources:{[pa]:1,[pb]:1},ambre_notes:[],consent:true,locked:true,version:1}});
 assert.equal(r.status,200);assert.equal(r.cartographie.version,2);
 assert.deepEqual(r.cartographie.ambre_notes.map(n=>n.id),['n1','n2'],'PUT never modifies Ambre’s notes');
 assert.deepEqual(r.cartographie.resources,{[pa]:1,[pb]:1});assert.equal(r.summary.notes.length,2);assert.equal(r.summary.ambre_notes[0].label,CARTOGRAPHIE.factors[0].label);
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{[fa]:2},notes:{factors:{[fa]:'Ce que j’en retiens.'},outcomes:{[oa]:'Aujourd’hui.'}},resources:{[pb]:1,[pa]:1},consent:true,locked:true,version:2}});
 assert.equal(r.cartographie.version,2,'an identical complete state is idempotent');
 r=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{[fa]:2},consent:true,locked:true,version:2}});
 assert.equal(r.cartographie.version,3);assert.deepEqual(r.cartographie.notes,{factors:{},outcomes:{}},'absent notes mean no notes');assert.deepEqual(r.cartographie.resources,{});
 assert.equal((await request(env,'/api/cartographie/ambre-notes/n1',{method:'DELETE',cookie:a.cookie,originHeader:'https://evil.example'})).status,403);
 assert.equal((await request(env,'/api/cartographie/ambre-notes/n1',{method:'DELETE'})).status,401);
 assert.equal((await request(env,'/api/cartographie/ambre-notes/n1',{method:'DELETE',cookie:b.cookie})).status,404,'another account cannot remove it');
 assert.equal((await request(env,'/api/cartographie/ambre-notes/n1',{cookie:a.cookie})).status,405);
 r=await send(env,'/api/cartographie/ambre-notes/n1',{method:'DELETE',cookie:a.cookie});
 assert.equal(r.status,200);assert.equal(r.ok,true);assert.deepEqual(r.cartographie.ambre_notes.map(n=>n.id),['n2']);assert.equal(r.cartographie.version,3,'removing a note keeps the version');assert.ok(r.summary);
 assert.equal((await request(env,'/api/cartographie/ambre-notes/n1',{method:'DELETE',cookie:a.cookie})).status,404);
 r=await send(env,'/api/cartographie',{cookie:a.cookie});
 assert.deepEqual(Object.keys(r.cartographie).sort(),['ambre_notes','answers','consent','feedback','locked','locked_at','notes','resources','updated_at','version']);
 const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();
 assert.deepEqual(exported.cartographie.ambre_notes.map(n=>n.id),['n2']);assert.deepEqual(exported.cartographie.notes,{factors:{},outcomes:{}});assert.deepEqual(exported.cartographie.resources,{});
 assert.equal((await request(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{answers:{[fa]:1},notes:{factors:{[fa]:'é'.repeat(1000)}},consent:true,locked:true,version:3,padding:'x'.repeat(80000)}})).status,413);
});

const reply=(notes,content='Je l’ai noté dans votre cartographie.')=>new Response(JSON.stringify({choices:[{message:{content:JSON.stringify({content,questions:[],reading_suggestions:[],proposals:[],cartography_notes:notes})},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:10,cost:0}}));
const say=(s,key,content)=>s.call('/api/chat/messages','POST',{content,idempotency_key:key});

test('Ambre reads the notes and resources and writes validated notes without touching answers or version',async t=>{
 const s=chatSetup();
 fullRow(s.raw,'a',{answers:{[fa]:2},notes:{factors:{[fa]:'Ce que j’en retiens.'}},resources:{[pa]:1},ambre:[{id:'old',target:'general',target_id:null,text:'Ancienne note.',created_at:1}],version:7});
 seedChatPrerequisites(s.raw);
 assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);
 let request;
 withFetch(t,async(_url,options)=>{request=JSON.parse(options.body);return reply([{target:'factor',target_id:fa,text:'Dit que son père criait presque tous les soirs.'},{target:'general',target_id:null,text:'ancienne note'},{target:'outcome',target_id:'inconnue',text:'À écarter.'}]);});
 const before=s.raw.prepare('SELECT answers,feedback,notes,resources,version,updated_at,locked_at FROM cartographies WHERE user_id=?').get('a');
 const result=await say(s,'carto_write_001','Mon père criait presque tous les soirs.');
 assert.ok([200,202].includes(result.status),JSON.stringify(result));
 const data=JSON.parse(request.messages[1].content.slice(request.messages[1].content.indexOf('{'))),map=data.childhood_cartography;
 assert.deepEqual(map.person_notes,[{target:'factor',id:fa,label:CARTOGRAPHIE.factors[0].label,text:'Ce que j’en retiens.'}]);
 assert.deepEqual(map.resources,[CARTOGRAPHIE.protections[0].label]);
 assert.deepEqual(map.ambre_notes,[{target:'general',text:'Ancienne note.'}]);
 const schema=request.response_format.json_schema.schema;
 assert.equal(schema.properties.cartography_notes.maxItems,3);assert.ok(schema.required.includes('cartography_notes'));
 assert.match(request.messages[0].content,/cartography_notes/);assert.match(request.messages[0].content,/faux souvenirs/);assert.match(request.messages[0].content,/Tu ne coches jamais une cause/);
 const notes=stored(s.raw);
 assert.deepEqual(notes.map(n=>[n.target,n.target_id,n.text]),[['general',null,'Ancienne note.'],['factor',fa,'Dit que son père criait presque tous les soirs.']]);
 assert.equal(typeof notes[1].id,'string');assert.equal(typeof notes[1].created_at,'number');
 assert.deepEqual(s.raw.prepare('SELECT answers,feedback,notes,resources,version,updated_at,locked_at FROM cartographies WHERE user_id=?').get('a'),before);
 const message=(await s.call('/api/chat')).messages.find(m=>m.role==='assistant');
 assert.deepEqual(message.cartography_notes,[{target:'factor',target_id:fa,text:'Dit que son père criait presque tous les soirs.',label:CARTOGRAPHIE.factors[0].label}]);
 globalThis.fetch=async()=>reply([{target:'factor',target_id:fa,text:'dit que son père criait presque tous les soirs'}]);
 await say(s,'carto_write_002','Oui, presque tous les soirs.');
 assert.equal(stored(s.raw).length,2,'a repeated note is not stored twice');
 assert.equal((await s.call('/api/chat')).messages.at(-1).cartography_notes,undefined);
});

test('prerequisites block absent, unconsented and unlocked maps; eligible opt-outs and immediate danger write no notes',async t=>{
 const attempt=async(prepare,content='Mon père criait presque tous les soirs.',blocked=false)=>{
  const s=chatSetup();prepare(s);seedChatPrerequisites(s.raw);assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);
  let request;globalThis.fetch=async(_url,options)=>{request=JSON.parse(options.body);return reply([{target:'general',target_id:null,text:'Dit que son père criait.'}],'Je vous écoute.');};
  const before=s.raw.prepare("SELECT * FROM cartographies WHERE user_id='a'").get();
  const result=await say(s,'carto_none_001',content);
  if(blocked){
   assert.equal(result.status,403,JSON.stringify(result));assert.equal(result.code,'prerequisites_required');
   assert.equal(request,undefined,'ineligible maps must never reach the provider');
   assert.deepEqual(s.raw.prepare("SELECT * FROM cartographies WHERE user_id='a'").get(),before,'rejection must not alter the map or append a note');
  }else assert.ok([200,202].includes(result.status),JSON.stringify(result));
  return {s,request,result,message:(await s.call('/api/chat')).messages.find(m=>m.role==='assistant')};
 };
 const previous=globalThis.fetch;t.after(()=>{globalThis.fetch=previous;});
 let run=await attempt(()=>{},undefined,true);
 assert.equal(run.s.raw.prepare('SELECT count(*) n FROM cartographies').get().n,0);assert.equal(run.message,undefined);
 run=await attempt(s=>fullRow(s.raw,'a',{answers:{[fa]:1},consent:0}),undefined,true);
 assert.deepEqual(stored(run.s.raw),[],'no consent, no note');assert.equal(run.message,undefined);
 run=await attempt(s=>fullRow(s.raw,'a',{answers:{[fa]:1},locked:false}),undefined,true);
 assert.deepEqual(stored(run.s.raw),[],'a map that is not saved for Ambre receives no note');
 run=await attempt(s=>fullRow(s.raw,'a',{answers:{[fa]:1}}),'Mon père criait souvent, mais ne garde pas ça dans mon profil.');
 assert.deepEqual(stored(run.s.raw),[],'an opt-out blocks the note');assert.equal(run.request.response_format.json_schema.schema.properties.cartography_notes,undefined);
 run=await attempt(s=>fullRow(s.raw,'a',{answers:{[fa]:1}}),'Je vais me suicider ce soir.');
 assert.deepEqual(stored(run.s.raw),[],'no note in immediate danger');
 run=await attempt(s=>fullRow(s.raw,'a',{answers:{[fa]:1}}));
 assert.equal(stored(run.s.raw).length,1,'the same reply is stored when the map is saved with consent');
});

test('a note removed by the member during the turn stays removed and the new note is still kept',async t=>{
 const s=chatSetup();fullRow(s.raw,'a',{answers:{[fa]:1},ambre:[{id:'old',target:'general',target_id:null,text:'Ancienne note.',created_at:1}]});
 seedChatPrerequisites(s.raw);
 assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);
 const cas=s.env.DB.prepare;let raced=false;
 // Simulate the member removing a note between Ambre's read and her guarded write.
 s.env.DB.prepare=sql=>{if(!raced&&sql.startsWith('UPDATE cartographies SET ambre_notes=?')&&sql.includes('chat_turns')){raced=true;s.raw.prepare("UPDATE cartographies SET ambre_notes='[]' WHERE user_id='a'").run();}return cas(sql);};
 withFetch(t,async()=>reply([{target:'factor',target_id:fa,text:'Dit que son père criait.'}]));
 await say(s,'carto_race_001','Mon père criait.');
 assert.equal(raced,true);
 assert.deepEqual(stored(s.raw).map(n=>n.text),['Dit que son père criait.']);
});

test('an oversized context drops Ambre’s oldest notes first and keeps the member’s words bounded',async t=>{
 const s=chatSetup();
 const notes={factors:Object.fromEntries(CARTOGRAPHIE.factors.slice(0,40).map(f=>[f.id,'n'.repeat(1000)])),outcomes:{}};
 const ambre=Array.from({length:60},(_,i)=>({id:'n'+i,target:'general',target_id:null,text:'Note numéro '+i+' '+'x'.repeat(380),created_at:i}));
 fullRow(s.raw,'a',{answers:Object.fromEntries(CARTOGRAPHIE.factors.map(f=>[f.id,2])),notes,ambre});
 s.raw.prepare("UPDATE memories SET text=? WHERE user_id='a'").run('m'.repeat(9000));
 seedChatPrerequisites(s.raw);
 assert.equal((await s.call('/api/chat/consent','PUT',{accepted:true})).status,200);
 let request;withFetch(t,async(_url,options)=>{request=JSON.parse(options.body);return reply([]);});
 await say(s,'carto_big_001','Bonjour Ambre.');
 const raw=request.messages[1].content,map=JSON.parse(raw.slice(raw.indexOf('{'))).childhood_cartography;
 assert.ok(raw.length<17000,String(raw.length));
 assert.equal(map.notes_truncated,true);
 assert.ok(map.person_notes.length>=1&&map.person_notes.every(n=>n.text.length<=440));
 assert.ok(map.ambre_notes.length<60);
 if(map.ambre_notes.length)assert.match(map.ambre_notes.at(-1).text,/^Note numéro 59 /,'the most recent note is kept last');
});

test('a database migrated to 0016 only still serves the map to Ambre without notes',async()=>{
 const raw=new DatabaseSync(':memory:');
 for(const name of readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')&&n<'0017').sort())raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES('a','a@example.test','h','s')").run();raw.prepare("INSERT INTO profiles(user_id) VALUES('a')").run();raw.prepare("INSERT INTO memories(user_id) VALUES('a')").run();raw.prepare("INSERT INTO subscriptions(user_id) VALUES('a')").run();
 saveRow(raw,'a',{answers:{[fa]:1}});
 const context=await loadContext({DB:d1(raw)},'a');
 assert.equal(context.cartography.cause_count,1);assert.deepEqual(context.cartography.ambre_notes,[]);assert.deepEqual(context.cartography.notes,[]);
});

test('new choices survive API reload and export without migrating historical answers or losing notes on conflict',async()=>{
 const env={DB:d1(migrated())},a=await register(env,'carto-explicit-choices@example.test');
 const original={answers:{controle:2,adoption:2,abus_physique:'unknown',divorce:'unknown',separation_conflit:1},notes:{factors:{controle:'Ma précision historique',adoption:'Mon récit conservé'},outcomes:{}},consent:true,locked:true,version:0};
 let saved=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:original});assert.equal(saved.status,200);assert.deepEqual(saved.cartographie.answers,original.answers);assert.equal(saved.summary.explicit_uncertain.length,2);
 const stale={...original,answers:{...original.answers,parents_stricts:0,controle_psychologique:'unknown'},version:0};assert.equal((await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:stale})).status,409);
 let read=await send(env,'/api/cartographie',{cookie:a.cookie});assert.deepEqual(read.cartographie.answers,original.answers);assert.deepEqual(read.cartographie.notes,original.notes);assert.equal(Object.hasOwn(read.cartographie.answers,'parents_stricts'),false);
 saved=await send(env,'/api/cartographie',{method:'PUT',cookie:a.cookie,data:{...stale,version:1}});assert.equal(saved.status,200);assert.equal(saved.cartographie.answers.controle,2);assert.equal(saved.cartographie.answers.adoption,2);assert.equal(saved.cartographie.answers.parents_stricts,0);assert.equal(saved.cartographie.answers.controle_psychologique,'unknown');
 const exported=await send(env,'/api/export',{cookie:a.cookie});assert.deepEqual(exported.cartographie.answers,saved.cartographie.answers);assert.deepEqual(exported.cartographie.notes,original.notes);env.DB.raw.close();
});

test('the declined association stays absent from real saved-map endpoints and Ambre context',async()=>{
 const env={DB:d1(migrated())},member=await register(env,'carto-declined-association@example.test');
 try{
  const answers={negligence_emotionnelle:2},notes={factors:{negligence_emotionnelle:'Précision conservée après retrait du lien.'},outcomes:{}};
  const saved=await send(env,'/api/cartographie',{method:'PUT',cookie:member.cookie,data:{answers,notes,consent:true,locked:true,version:0}});assert.equal(saved.status,200);assert.deepEqual(saved.cartographie.answers,answers);
  const read=await send(env,'/api/cartographie',{cookie:member.cookie});assert.equal(read.status,200);assert.deepEqual(read.cartographie.notes,notes);assert.ok(read.summary.display_count>0);assert.ok(!JSON.stringify(read.summary).includes('icd_6a60'));
  const context=await loadContext(env,member.user.id);assert.deepEqual(context.cartography.causes.map(c=>c.id),['negligence_emotionnelle']);assert.ok(!JSON.stringify(context.cartography).includes('icd_6a60'),'no result, score, evidence or source selection transmits the declined association');
  const exported=await send(env,'/api/export',{cookie:member.cookie});assert.deepEqual(exported.cartographie.answers,answers);assert.deepEqual(exported.cartographie.notes,notes);
 }finally{env.DB.raw.close()}
});
