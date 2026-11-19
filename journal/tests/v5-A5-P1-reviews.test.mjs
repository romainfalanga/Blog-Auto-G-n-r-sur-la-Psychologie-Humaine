import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
// Lot A5-P1 (plan v5.1, audit v5 F): structured reviews anchored in the person's words, a
// separate check before publication, the review thread at every turn, the person's feedback,
// a suggested (never started) point, and a Worker that tolerates the missing 0023 columns.
import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createPersonalBackend} from '../src/personal-backend.mjs';
import {createChatBackend} from '../src/chat-backend.mjs';
import {parseFeedback as parseFeedbackValue,validateDigest,reviewThread,digestProse,reviewChecks,reviewVerdict,reviewSuggestion,reviewCorpus,REVIEW_OUTPUT_SCHEMA,REVIEW_CHECK_SCHEMA,THREAD_TURN_CHARS,THREAD_REVIEW_CHARS,DIGEST_MAX_CHARS} from '../src/review-digest.mjs';
import {buildReviewSystemPrompt} from '../src/psychology-framework.mjs';
import {analysis,draft,review as critic,response} from './v4-R-helper.mjs';

const migrations=readdirSync(new URL('../migrations/',import.meta.url)).filter(n=>n.endsWith('.sql')).sort();
// Existing digest/check/thread assertions retain the pre-0031 bounded-context contract.
function setup({before='0031'}={}){
 const raw=new DatabaseSync(':memory:');raw.exec('PRAGMA foreign_keys=ON');
 // Keep required symptoms storage while independently withholding the optional digest migration.
 for(const name of migrations.filter(n=>!before||n<before||n.startsWith('0026_')||n.startsWith('0029_')))raw.exec(readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8'));
 const prepare=sql=>{let args=[];return {bind(...a){args=a;return this},async first(){return raw.prepare(sql).get(...args)||null},async all(){return {results:raw.prepare(sql).all(...args)}},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}}}}};
 let queue=Promise.resolve();
 const DB={raw,prepare,batch(statements){const next=queue.then(async()=>{raw.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());raw.exec('COMMIT');return r}catch(e){raw.exec('ROLLBACK');throw e}});queue=next.catch(()=>{});return next}};
 let clock=1794830400,serial=0;
 const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),text:(v,max=1000)=>typeof v==='string'?v.slice(0,max).trim():'',body:async req=>req.json(),now:()=>clock,random:()=>`id_${String(++serial).padStart(8,'0')}`,decorateLikes:x=>x,recommender:{provisionalPortrait:()=>null}};
 const personal=createPersonalBackend(utils),chat=createChatBackend({...utils,personal,catalogue:new Map(),engineEnabled:true}),env={DB,OPENROUTER_API_KEY:'mock-only'};
 for(const id of ['a','b']){raw.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'hash','salt')").run(id,id+'@example.test');raw.prepare('INSERT INTO profiles(user_id) VALUES(?)').run(id);raw.prepare('INSERT INTO memories(user_id) VALUES(?)').run(id);raw.prepare("INSERT INTO subscriptions(user_id,status,paid_until) VALUES(?,'active',?)").run(id,clock+86400*400)}
 raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({first_name:'Alix',self_description:'Je me tais souvent pour éviter les conflits avec ma sœur.'}));
 seedCompanionPrerequisites(raw);
 async function call(path,method='GET',data,user='a'){const req=new Request('https://example.test'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});try{const url=new URL(req.url),res=await chat.routes(req,env,{id:user},url.pathname,url)||await personal.routes(req,env,{id:user},url.pathname,url);return {status:res.status,...await res.json()}}catch(e){return {status:e.status||500,code:e.code,message:e.message}}}
 const job=()=>raw.prepare('SELECT * FROM chat_engine_jobs ORDER BY rowid DESC LIMIT 1').get();
 async function message(content,key){const r=await call('/api/chat/messages','POST',{content,idempotency_key:key});for(let n=0;n<8&&job()&&!['done','failed'].includes(job().stage);n++)await chat.engine.work(env,job().turn_id);return r}
 return {raw,env,chat,call,message,job,row:id=>raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(id),now:()=>clock,tick:s=>{clock+=s}};
}
const PASS={criteria:{no_diagnosis:'yes',quotes_faithful:'yes',safety_resources:'na',hypotheses_open:'yes',no_saved_claim:'yes',sources_honest:'yes',no_internal:'yes'},reasons:[]};
const text=(extra='')=>extra+'Vous écrivez que vous vous taisez souvent pour éviter un conflit, et cela peut se relier à ce que vous avez renseigné. '.repeat(14)+'Ce bilan vous ressemble-t-il ? Vous pouvez le corriger.';
const initialDigest=()=>({formulation:{holds:[{text:'Le silence protège du conflit',quote:'pour éviter les conflits avec ma sœur'}],revised:[],new:[{text:'Une colère ancienne envers la mère',quote:'je déteste ma mère depuis toujours'}]},goals:[{key:'parler-reunion',words:'oser parler en réunion',status:'new',quote:'Je me tais souvent pour éviter les conflits',next_step:'repérer un moment'}],tried:[],alliance:{text:'',quote:''},open_questions:['Dans quelles situations le silence vous coûte-t-il le plus ?'],next_focus:'Les moments où vous vous taisez',safety:''});
const pointDigest=()=>({formulation:{holds:[{text:'Le silence protège du conflit',quote:'pour éviter les conflits avec ma sœur'}],revised:[{text:'Parler est parfois possible',quote:'J’ai osé prendre la parole mardi en réunion'}],new:[]},goals:[{key:'parler-reunion',words:'oser parler en réunion',status:'progressing',quote:'J’ai osé prendre la parole mardi en réunion',next_step:'recommencer une fois'}],tried:[{what:'Prendre la parole en réunion',result:'helped',quote:'cela m’a soulagée'}],alliance:{text:'Les questions précises aident',quote:'vos questions précises m’aident'},open_questions:['Qu’est-ce qui a rendu mardi possible ?'],next_focus:'Ce qui rend la parole possible',safety:''});
// Simulated provider: review writing and check answers are configurable, the turn stages use the lot R defaults.
function provider(t,{write=()=>({content:text(),digest:initialDigest()}),check=()=>PASS}={}){
 const calls=[],original=globalThis.fetch;
 globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;calls.push({purpose,body});
  if(purpose==='ambre_snapshot_chunk'){const chunk=JSON.parse(body.messages[1].content),quote='pour éviter les conflits avec ma sœur',item=chunk.items?.find(x=>x.text.includes(quote));return response({summary:'Déclarations personnelles à explorer.',quotes:item?[{path:item.path,quote}]:[]})}
  if(purpose==='ambre_review'&&body.response_format.json_schema.schema.properties.digest){const out=await write(body,calls);return out instanceof Response?out:response(out)}
  if(purpose==='ambre_review_check'){const out=await check(body,calls);return out instanceof Response?out:response(out)}
  return response({ambre_analysis:analysis(),ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:draft(),ambre_review:critic(),ambre_memory:{items:[]}}[purpose])};
 t.after(()=>{globalThis.fetch=original});return calls;
}
const start=s=>s.call('/api/chat/start','POST',{idempotency_key:'start_key_001',accepted:true});
const writingInput=calls=>JSON.parse(calls.filter(c=>c.purpose==='ambre_writing').at(-1).body.messages[1].content);
const reviewInput=call=>{const c=call.body.messages[1].content;return JSON.parse(c.slice(c.indexOf('{'),c.lastIndexOf('}')+1).split('\n\nRelecture')[0])};

test('digest: every element keeps an exact quote of the person or leaves, with closed states and bounds',()=>{
 const corpus='Je me tais souvent pour éviter les conflits avec ma sœur.\n\nJe ne veux plus me taire en réunion, ça n’a pas marché.';
 const d=validateDigest({formulation:{holds:[{text:'Silence protecteur',quote:'pour éviter les conflits avec ma sœur'},{text:'Inventé',quote:'je déteste ma mère'}],revised:[{text:'Changement',quote:'Je ne veux plus me taire en réunion'}],new:[{text:'Trop court',quote:'ma sœur'}]},goals:[{key:'Bad Key!',words:'parler en réunion',status:'on_fire',quote:'Je me tais souvent pour éviter'},{key:'x',words:'parler en réunion',status:'ongoing',quote:'Je me tais souvent pour éviter'}],tried:[{what:'Se taire',result:'no_effect',quote:'ça n’a pas marché.'}],alliance:{text:'Retour',quote:'phrase absente du corpus'},open_questions:['Une ?','Deux ?','Trois ?','Quatre ?'],next_focus:'Suite',safety:''},corpus);
 assert.deepEqual(d.formulation.holds.map(x=>x.quote),['pour éviter les conflits avec ma sœur']);
 assert.equal(d.formulation.revised.length,1,'a revision may quote a negation');assert.equal(d.formulation.new.length,0,'a quote under twelve characters is not an anchor');
 assert.equal(d.goals.length,1);assert.equal(d.goals[0].status,'new','an unknown goal is explicitly new');assert.equal(d.goals[0].key,'parler-en-reunion','an invalid key becomes a slug of the words');
 assert.equal(d.tried.length,1);assert.equal(d.alliance,null);assert.equal(d.open_questions.length,3);assert.equal(d.dropped,4);
 const known=validateDigest({goals:[{key:'autre',words:'parler en réunion',status:'progressing',quote:'Je me tais souvent pour éviter'}]},corpus,{knownGoals:[{key:'parler-reunion',words:'parler en reunion'}]});
 assert.equal(known.goals[0].key,'parler-reunion','a known goal keeps its stable key');assert.equal(known.goals[0].status,'progressing');
 const long='Je me tais souvent pour éviter les conflits avec ma sœur. '.repeat(3);
 const big=validateDigest({formulation:{holds:Array(3).fill({text:'x'.repeat(300),quote:long.slice(0,280)}),revised:Array(3).fill({text:'y'.repeat(300),quote:long.slice(0,280)}),new:Array(3).fill({text:'z'.repeat(300),quote:long.slice(0,280)})},goals:Array(5).fill(0).map((_,i)=>({key:'g'+i,words:'objectif '+i,status:'new',quote:long.slice(0,280),next_step:'n'.repeat(200)})),tried:[],alliance:{text:'',quote:''},open_questions:[],next_focus:'',safety:''},long);
 assert.ok(JSON.stringify(big).length<=DIGEST_MAX_CHARS);assert.ok(big.dropped>0);
 assert.match(digestProse(d),/Ce qui se confirme\. Silence protecteur\. Vous avez écrit : « pour éviter les conflits avec ma sœur »\./);assert.match(digestProse(d),/Vos objectifs suivis\. « parler en réunion » : nouveau\./);
 assert.doesNotMatch(digestProse(d),/[0-9]+ ?%|score/);
 for(const schema of [REVIEW_OUTPUT_SCHEMA,REVIEW_CHECK_SCHEMA]){const walk=v=>{if(v?.type==='object'){assert.deepEqual(v.required,Object.keys(v.properties));assert.equal(v.additionalProperties,false);Object.values(v.properties).forEach(walk)}if(v?.type==='array')walk(v.items)};walk(schema)}
});

test('thread: dated goal states, confirmed elements, feedback; bounded at 1 200 / 3 000 characters; legacy excerpt',()=>{
 const rows=[{id:'r3',kind:'point',completed_at:1794830400+86400*14,digest:JSON.stringify(validateDigest({goals:[{key:'g1',words:'oser parler',status:'progressing',quote:'osé prendre la parole mardi'}],formulation:{holds:[{text:'Le silence protège',quote:'osé prendre la parole mardi'}]},next_focus:'La suite'},'osé prendre la parole mardi',{knownGoals:[{key:'g1',words:'oser parler'}]})),feedback:JSON.stringify({verdict:'to_correct',text:'Ce n’était pas ma sœur mais mon frère.',at:1})},
  {id:'r2',kind:'point',completed_at:1794830400+86400*7,digest:JSON.stringify(validateDigest({goals:[{key:'g1',words:'oser parler',status:'ongoing',quote:'osé prendre la parole mardi'}],formulation:{holds:[{text:'Le silence protège',quote:'osé prendre la parole mardi'}]}},'osé prendre la parole mardi',{knownGoals:[{key:'g1',words:'oser parler'}]})),feedback:null},
  {id:'r1',kind:'initial',completed_at:1794830400,digest:JSON.stringify(validateDigest({goals:[{key:'g1',words:'oser parler',status:'new',quote:'osé prendre la parole mardi'}]},'osé prendre la parole mardi')),feedback:null}];
 const thread=reviewThread(rows);
 assert.equal(thread.last.id,'r3');assert.equal(thread.last.next_focus,'La suite');assert.equal(thread.reviews,3);
 assert.match(thread.goals[0].path,/^\d{4}-\d{2}-\d{2} nouveau → \d{4}-\d{2}-\d{2} en cours → \d{4}-\d{2}-\d{2} avance$/);
 assert.deepEqual(thread.confirmed,['Le silence protège (depuis 2 bilans)']);assert.equal(thread.feedback[0].verdict,'À corriger');assert.match(thread.feedback[0].text,/mon frère/);
 // Many reviews and goals: within both bounds; reached goals of older reviews leave first.
 const many=Array.from({length:12},(_,i)=>({id:'m'+i,kind:'point',completed_at:2e9-i*86400,feedback:JSON.stringify({verdict:'not_really',text:'Précision '.repeat(50),at:1}),digest:JSON.stringify(validateDigest({goals:Array.from({length:5},(_,g)=>({key:'goal-'+(i*5+g),words:'Objectif numéro '+(i*5+g)+' avec des mots assez longs pour peser',status:g?'ongoing':'reached',quote:'une citation exacte et assez longue de la personne'})),tried:[{what:'Essai '+i,result:'helped',quote:'une citation exacte et assez longue de la personne'}],formulation:{holds:[{text:'Formulation '+i+' qui tient',quote:'une citation exacte et assez longue de la personne'}]},open_questions:['Question ouverte numéro '+i+' ?'],next_focus:'Prochaine étape '+i},'une citation exacte et assez longue de la personne'))}));
 for(const max of [THREAD_TURN_CHARS,THREAD_REVIEW_CHARS]){const bounded=reviewThread(many,{maxChars:max});assert.ok(JSON.stringify(bounded).length<=max,max+': '+JSON.stringify(bounded).length);assert.equal(bounded.last.id,'m0');assert.ok(bounded.goals?.length>=1);assert.ok(!bounded.goals.some(g=>g.status==='reached'&&!g.key.startsWith('goal-0')),'old reached goals leave first')}
 const legacy=reviewThread([{id:'old',kind:'initial',completed_at:1,digest:'{}',feedback:null,content:'DEBUT '+'texte '.repeat(400)+' FIN'}]);
 assert.ok(legacy.last.text.startsWith('DEBUT')&&legacy.last.text.endsWith('FIN')&&legacy.last.excerpt);assert.ok(JSON.stringify(legacy).length<=THREAD_TURN_CHARS);
 assert.equal(reviewThread([]),null);
});

test('checks: diagnosis, invented quote, missing safety resources and unverified numbers block; the verdict needs every criterion',()=>{
 const supplied='Je me tais souvent pour éviter les conflits avec ma sœur.';
 assert.deepEqual(reviewChecks(text(),{supplied}).failed,[]);
 assert.deepEqual(reviewChecks('Vous êtes bipolaire, sans doute.',{supplied}).failed,['diagnosis']);
 assert.deepEqual(reviewChecks('Votre mère est narcissique.',{supplied}).failed,['diagnosis']);
 const invented=reviewChecks('Vous dites « je déteste ma mère depuis toujours » et « pour éviter les conflits avec ma sœur ».',{supplied});
 assert.deepEqual(invented.failed,['invented_quote']);assert.deepEqual(invented.invented,['je déteste ma mère depuis toujours']);
 assert.deepEqual(reviewChecks(text(),{supplied,signal:'self_harm'}).failed,['safety_missing']);
 assert.deepEqual(reviewChecks(text('Le 3114 répond 24 h/24. '),{supplied,signal:'self_harm'}).failed,[]);
 assert.deepEqual(reviewChecks(text('Appelez le 0800. '),{supplied}).failed,['unverified_number']);
 assert.throws(()=>reviewVerdict({criteria:{no_diagnosis:'yes'}},{failed:[]}),/Invalid review check/);
 assert.equal(reviewVerdict(PASS,{failed:[]}).passed,true);
 assert.equal(reviewVerdict({...PASS,criteria:{...PASS.criteria,quotes_faithful:'no'}},{failed:[]}).passed,false);
 assert.equal(reviewVerdict({...PASS,criteria:{...PASS.criteria,sources_honest:'no'}},{failed:[]}).passed,true,'one non-blocking criterion does not block');
 assert.equal(reviewVerdict({...PASS,criteria:{...PASS.criteria,sources_honest:'no',hypotheses_open:'no'}},{failed:[]}).passed,false);
});

test('legacy before 0031: a review is written with its digest, checked separately, then published; the thread reaches the next turns and the point',async t=>{
 const s=setup();let phase='initial';
 s.raw.prepare("INSERT INTO care_settings(user_id,has_psychologist,frequency_days,updated_at) VALUES('a',1,14,0)").run();
 const calls=provider(t,{write:()=>phase==='initial'?{content:text(),digest:initialDigest()}:{content:text('Ce qui a évolué. '),digest:pointDigest()}});
 let r=await start(s);assert.equal(r.status,202);const initialId=r.review_id;
 assert.deepEqual(calls.map(c=>c.purpose),['ambre_review','ambre_review_check']);
 const write=calls[0].body,check=calls[1].body;
 assert.equal(write.max_tokens,5600,'plan v1.1 point 8: chat_message (≤ 2 500 characters) raises the writing cap');assert.ok(write.response_format.json_schema.schema.properties.chat_message);assert.ok(write.response_format.json_schema.schema.properties.digest);assert.match(write.messages[0].content,/citation exacte et continue/);
 assert.equal(check.max_tokens,1500);assert.equal(check.model,'openai/gpt-5-nano');assert.equal(check.reasoning.effort,'minimal');
 const checkInput=JSON.parse(check.messages[1].content);assert.equal(checkInput.review,text());assert.equal(checkInput.safety_signal,null);assert.equal(checkInput.digest.goals.length,1);
 let row=s.row(initialId);assert.equal(row.status,'ready');assert.equal(row.stage,'done');assert.equal(row.draft,'');
 const stored=JSON.parse(row.digest);assert.equal(stored.goals[0].key,'parler-reunion');assert.equal(stored.formulation.new.length,0,'the invented quote left the digest');assert.equal(stored.dropped,1);
 assert.deepEqual(JSON.parse(row.sources).check,{passed:true,rewritten:false,failed:[],digest_dropped:1,chat_fallback:true},'the mock writer returns no chat_message: deterministic fallback');
 const budget=s.raw.prepare("SELECT provider_calls FROM provider_budgets WHERE user_id='a'").get();assert.equal(budget.provider_calls,2);
 const read=await s.call('/api/reviews/'+initialId);assert.match(read.review.digest_text,/Vos objectifs suivis\. « oser parler en réunion » : nouveau\./);assert.equal(read.review.feedback,null);
 // The next turn receives the thread, outside the details budget.
 await s.message('J’ai osé prendre la parole mardi en réunion et cela m’a soulagée. Vos questions précises m’aident.','turn_one_0001');
 assert.equal(s.job().stage,'done');let input=writingInput(calls);
 assert.equal(input.review_thread.last.id,initialId);assert.equal(input.review_thread.goals[0].words,'oser parler en réunion');assert.match(input.review_thread.goals[0].path,/nouveau$/);assert.equal(input.review_thread.last.next_focus,'Les moments où vous vous taisez');
 assert.ok(JSON.stringify(input.review_thread).length<=THREAD_TURN_CHARS);assert.equal(input.last_review,undefined);
 // Feedback on the initial analysis, then a point: memory, plan, care, thread and feedback are in its context.
 assert.equal((await s.call('/api/reviews/'+initialId+'/feedback','PUT',{verdict:'to_correct',text:'Ce n’est pas ma sœur, c’est mon frère.'})).status,200);
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,through_turn,turns_covered,updated_at) VALUES('a',?,NULL,1,0) ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary").run(JSON.stringify({goals:['MEM_GOAL oser parler en réunion'],tried:['MEM_TRIED prendre la parole mardi'],alliance:['MEM_ALLIANCE questions précises'],facts:['MEM_FACT hors bilan']}));
 const now=s.now();s.raw.prepare("INSERT INTO change_plans(user_id,revision,document_json,updated_at) VALUES('a',1,?,?)").run(JSON.stringify({schema_version:1,goals:[{id:'g1',revision:1,title:'PLAN_GOAL Oser parler',why:'Être entendue',status:'active',priority:1,accepted_at:now,updated_at:now}],steps:[{id:'s1',revision:1,goal_id:'g1',text:'PLAN_STEP Dire une phrase en réunion',strategy_id:null,measure:'Mon ressenti',status:'active',accepted_at:now,updated_at:now}],trials:[],proposals:[],preparations:[],instructions:[]}),now);
 s.tick(32*86400);s.raw.prepare("INSERT INTO care_sessions(id,user_id,session_at,created_by,created_at) VALUES('cs1','a',?,'member',0)").run(s.now()+86400);
 phase='point';r=await s.call('/api/chat/review','POST',{idempotency_key:'point_key_0001'});assert.equal(r.status,202,r.message);
 const pointWrite=calls.filter(c=>c.purpose==='ambre_review').at(-1),point=reviewInput(pointWrite);
 assert.equal(point.review_thread.last.id,initialId);assert.equal(point.review_thread.goals[0].key,'parler-reunion');assert.equal(point.review_thread.feedback[0].verdict,'À corriger');
 assert.deepEqual(point.previous_feedback,{verdict:'À corriger',text:'Ce n’est pas ma sœur, c’est mon frère.'});
 assert.deepEqual(point.memory.goals,['MEM_GOAL oser parler en réunion']);assert.equal(point.memory.facts,undefined);assert.ok(point.memory.tried&&point.memory.alliance);
 assert.match(point.change_plan.goals[0].title,/PLAN_GOAL/);assert.match(point.change_plan.steps[0].text,/PLAN_STEP/);
 assert.equal(point.care.has_psychologist,true);assert.match(point.care.next_session,/^\d{4}-\d{2}-\d{2} /);
 assert.ok(JSON.stringify(point).length<=34000);assert.match(pointWrite.body.messages[0].content,/review_thread/);
 const pointId=r.review_id,pointRow=s.row(pointId);assert.equal(pointRow.status,'ready');
 const pointStored=JSON.parse(pointRow.digest);assert.equal(pointStored.goals[0].key,'parler-reunion');assert.equal(pointStored.goals[0].status,'progressing');assert.equal(pointStored.tried[0].result,'helped');assert.equal(pointStored.alliance.quote,'Vos questions précises m’aident','the exact words of the person, not the model’s');
 assert.equal(JSON.parse(pointRow.sources).previous_feedback,'to_correct');
 // The next turn sees the goal's path across both reviews; deleting the point recalculates the thread.
 await s.message('Je voudrais continuer sur ce point.','turn_two_0001');input=writingInput(calls);
 assert.match(input.review_thread.goals[0].path,/nouveau → \d{4}-\d{2}-\d{2} avance$/);assert.equal(input.review_thread.last.id,pointId);
 assert.equal((await s.call('/api/reviews/'+pointId,'DELETE')).deleted,true);
 await s.message('Encore une précision.','turn_three_001');input=writingInput(calls);assert.equal(input.review_thread.last.id,initialId);assert.match(input.review_thread.goals[0].path,/nouveau$/);
 // Paramètres show the thread and the feedback; the export carries digests, feedback and the thread.
 const insight=await s.call('/api/chat/insight');assert.equal(insight.current.review_thread.value.last.id,initialId);assert.match(insight.current.review_thread.text,/À corriger/);
 assert.match(insight.current.stored_elsewhere.find(x=>x.key==='reviews').description,/fil des bilans/i);
});

test('legacy before 0031: a failed check asks one rewrite; a second failure publishes nothing; an unavailable check is retried at the check stage',async t=>{
 // One rewrite: the invented quote is reported to the writer, the rewrite passes.
 let s=setup(),writes=0;let calls=provider(t,{write:()=>++writes===1?{content:text('Vous dites « je déteste ma mère depuis toujours ». '),digest:initialDigest()}:{content:text(),digest:initialDigest()}});
 let r=await start(s);let row=s.row(r.review_id);
 assert.deepEqual(calls.map(c=>c.purpose),['ambre_review','ambre_review_check','ambre_review','ambre_review_check']);
 assert.match(calls[2].body.messages[1].content,/Relecture du bilan précédent[^]*invented_quote[^]*je déteste ma mère/);
 assert.equal(row.status,'ready');assert.equal(row.content,text());assert.deepEqual(JSON.parse(row.sources).check.rewritten,true);
 // Two failures: nothing is published, the unchecked draft is erased, the person may ask again.
 s=setup();calls=provider(t,{check:()=>({...PASS,criteria:{...PASS.criteria,no_diagnosis:'no'},reasons:['diagnostic']})});
 r=await start(s);row=s.row(r.review_id);
 assert.equal(row.status,'failed');assert.equal(row.error_code,'review_not_approved');assert.equal(row.content,'');assert.equal(row.draft,'');
 assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_turns').get().n,0,'never a review published without a passed check');
 assert.equal(calls.filter(c=>c.purpose==='ambre_review').length,2);
 assert.equal((await s.call('/api/chat')).journey.started,false);
 // The check provider fails: the draft waits at the check stage; the cron checks without writing again.
 s=setup();let down=true;calls=provider(t,{check:()=>down?new Response('{}',{status:500}):PASS});
 r=await start(s);row=s.row(r.review_id);
 assert.equal(row.status,'running');assert.equal(row.stage,'check');assert.match(row.draft,/Vous écrivez que vous vous taisez/);assert.equal(row.error_code,'provider_http_500');
 assert.equal((await s.call('/api/chat')).messages.length,0,'the unchecked draft never reaches the conversation');
 down=false;s.tick(31);assert.equal((await s.chat.resumeReviews(s.env)).claimed,1);
 row=s.row(r.review_id);assert.equal(row.status,'ready');assert.equal(row.draft,'');
 assert.deepEqual(calls.map(c=>c.purpose),['ambre_review','ambre_review_check','ambre_review_check']);
 // An invalid check answer is not a pass either.
 s=setup();calls=provider(t,{check:()=>({criteria:{no_diagnosis:'yes'}})});
 r=await start(s);assert.equal(s.row(r.review_id).status,'running');assert.equal(s.row(r.review_id).error_code,'invalid_check');
});

test('legacy before 0031: a safety signal since the last review makes the resources blocking in the check',async t=>{
 const s=setup();provider(t);await start(s);
 await s.message('Je pense au suicide depuis quelques jours.','risk_turn_0001');
 s.tick(32*86400);let writes=0;const calls=provider(t,{write:()=>++writes===1?{content:text(),digest:initialDigest()}:{content:text('Le 3114 répond 24 h/24, 7 j/7 ; parlez-en aussi à une personne de confiance. '),digest:initialDigest()}});
 const r=await s.call('/api/chat/review','POST',{idempotency_key:'point_key_0001'});
 const checks=calls.filter(c=>c.purpose==='ambre_review_check').map(c=>JSON.parse(c.body.messages[1].content));
 assert.equal(checks[0].safety_signal,'self_harm');assert.deepEqual(checks[0].deterministic_failures,['safety_missing']);
 assert.equal(writes,2);assert.equal(s.row(r.review_id).status,'ready');assert.match(s.row(r.review_id).content,/3114/);
});

test('legacy before 0031: feedback: three answers without score, an optional text, the owner only; it changes the thread and the source hash',async t=>{
 const s=setup();const calls=provider(t);const r=await start(s),id=r.review_id;
 const put=(data,user='a')=>s.call('/api/reviews/'+id+'/feedback','PUT',data,user);
 assert.equal((await put({verdict:'great'})).code,'invalid_input');assert.equal((await put({verdict:'resembles',text:'x'.repeat(601)})).code,'invalid_input');
 assert.equal((await put({verdict:'resembles'},'b')).status,404);assert.equal((await s.call('/api/reviews/'+id+'/feedback','POST',{verdict:'resembles'})).status,405);
 let saved=await put({verdict:'resembles',text:'  Oui, c’est juste.  '});assert.deepEqual(saved.feedback,{verdict:'resembles',text:'Oui, c’est juste.',at:s.now()});
 assert.deepEqual((await s.call('/api/reviews/'+id)).review.feedback,saved.feedback);assert.equal((await s.call('/api/reviews')).reviews[0].feedback.verdict,'resembles');
 await s.message('Bonjour Ambre.','feedback_turn_01');assert.equal(writingInput(calls).review_thread.feedback[0].verdict,'Ce bilan me ressemble');
 const before=s.job().source_hash;saved=await put({verdict:'not_really'});
 await s.message('Encore moi.','feedback_turn_02');assert.notEqual(s.job().source_hash,before,'the digest and the feedback are part of the source hash and the index cache key');assert.equal(writingInput(calls).review_thread.feedback[0].verdict,'Pas vraiment');
 assert.equal((await put({verdict:null})).feedback,null);assert.equal(s.row(id).feedback,null);
 const exported=await s.chat.exportData(s.env,'a');assert.ok(exported.ambre_reviews[0].digest.goals);assert.match(exported.ambre_reviews[0].digest_text,/objectifs suivis/);assert.ok(exported.ambre_review_thread.thread.last.id);
 // Erasing the conversation empties the digest and the feedback of every review.
 await put({verdict:'to_correct'});assert.equal((await s.call('/api/chat','DELETE',{})).status,200);
 assert.deepEqual({...s.raw.prepare('SELECT digest,feedback,draft,content FROM ambre_reviews WHERE id=?').get(id)},{digest:'{}',feedback:null,draft:'',content:''});
});

test('legacy before 0031: Ambre may suggest a point before a declared session, never start one',async t=>{
 const s=setup();const calls=provider(t);await start(s);
 assert.equal((await s.call('/api/chat')).journey.review.suggestion,undefined,'too early for a point');
 s.tick(32*86400);assert.equal((await s.call('/api/chat')).journey.review.suggestion,undefined,'no session, few exchanges');
 s.raw.prepare("INSERT INTO care_sessions(id,user_id,session_at,created_by,created_at) VALUES('cs1','a',?,'member',0)").run(s.now()+86400);
 const state=await s.call('/api/chat');assert.deepEqual(state.journey.review.suggestion,{reason:'session_soon',session_at:s.now()+86400});
 const reviews=s.raw.prepare('SELECT count(*) n FROM ambre_reviews').get().n;
 await s.message('Je prépare ma séance de demain.','suggest_turn_01');const input=writingInput(calls);
 assert.equal(input.review_suggestion.reason,'session_soon');assert.match(calls.filter(c=>c.purpose==='ambre_writing').at(-1).body.messages[0].content[0].text,/ne jamais lancer de bilan/);
 assert.equal(s.raw.prepare('SELECT count(*) n FROM ambre_reviews').get().n,reviews,'a suggestion never creates a review');
 assert.deepEqual(reviewSuggestion({started:true,running:false,nextPointAt:null,care:{},now:0,lastReviewAt:0,turnsSince:15}),{reason:'many_exchanges'});
 assert.equal(reviewSuggestion({started:true,running:false,nextPointAt:123,care:{next_session_at:10},now:0,turnsSince:30}),null);
});

test('legacy before 0031: before the 0023 migration the Worker keeps working: checked reviews without digest, the legacy excerpt as thread, no feedback',async t=>{
 const s=setup({before:'0023'});const calls=provider(t);
 let r=await start(s);assert.equal(r.status,202);const id=r.review_id;
 assert.deepEqual(calls.map(c=>c.purpose),['ambre_review','ambre_review_check']);
 assert.equal(s.row(id).status,'ready');assert.equal(s.row(id).digest,undefined);
 assert.equal((await s.call('/api/reviews/'+id+'/feedback','PUT',{verdict:'resembles'})).code,'feature_unavailable');
 assert.equal((await s.call('/api/reviews')).reviews.length,1);assert.equal((await s.call('/api/reviews/'+id)).review.content,text());
 await s.message('Bonjour.','legacy_turn_0001');assert.equal(s.job().stage,'done');const input=writingInput(calls);
 assert.equal(input.review_thread.last.id,id);assert.ok(input.review_thread.last.text.startsWith('Vous écrivez'));
 s.tick(32*86400);r=await s.call('/api/chat/review','POST',{idempotency_key:'point_key_0001'});assert.equal(s.row(r.review_id).status,'ready');
 const exported=await s.chat.exportData(s.env,'a');assert.equal(exported.ambre_reviews.length,2);assert.equal(exported.ambre_review_thread,null);
 assert.equal((await s.call('/api/chat','DELETE',{})).status,200);
});

test('review prompts ask for the anchored digest, the thread and the feedback, without scores',()=>{
 const initial=buildReviewSystemPrompt('initial'),point=buildReviewSystemPrompt('point');
 for(const prompt of [initial,point]){assert.match(prompt,/holds = ce qui se confirme, revised = ce qui est nuancé ou contredit, new = nouvelle hypothèse/);assert.match(prompt,/au moins douze caractères/);assert.match(prompt,/Aucun score, aucun diagnostic/);assert.match(prompt,/corriger ce bilan/)}
 assert.match(point,/reprends chaque objectif suivi avec son état/);assert.match(point,/previous_feedback/);assert.match(point,/change_plan/);assert.match(point,/prochaine séance/);
 assert.equal(reviewCorpus({sources:[{type:'profile',text:'A'},{type:'portrait',text:'B'}],conversation_since:[{vous:'C',ambre:'D'}]}),'A\n\nC','only the person’s words anchor a digest');
});

// Mes bilans: the digest as prose and the three answers (no score), sent once chosen.
import vm from 'node:vm';
import {HTML as SHELL,APP_SCRIPT,APP_STYLE} from '../src/ui.mjs';
// Lot P: the private shell plus its two content-addressed files, as the browser assembles them.
const HTML=SHELL.replace('</head>',()=>'<style>'+APP_STYLE+'</style></head>')+'<script>'+APP_SCRIPT+'</script>';
test('Mes bilans: “Ce qu’Ambre retient de ce bilan” and the answer “Ce bilan me ressemble / Pas vraiment / À corriger”',async()=>{
 const script=HTML.match(/<script>([\s\S]*?)<\/script>/)[1],code=name=>{const start=script.indexOf('function '+name+'(');const rest=script.slice(start+1),end=rest.search(/\n(?:async )?function |\nvar /);return script.slice(start,start+1+end)};
 class Node{constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.listeners={};this.value='';this._text=''}append(...items){this.children.push(...items)}set textContent(v){this._text=String(v);this.children=[]}get textContent(){return this._text+this.children.map(c=>c.textContent).join(' ')}setAttribute(k,v){this.attributes[k]=v}getAttribute(k){return this.attributes[k]}addEventListener(k,fn){(this.listeners[k]??=[]).push(fn)}async click(){for(const fn of this.listeners.click||[])await fn({})}all(tag){return this.children.flatMap(c=>[...(c.tagName===tag?[c]:[]),...c.all(tag)])}}
 const sent=[],ctx=vm.createContext({document:{createElement:tag=>new Node(tag)},api:async(path,method,body)=>{sent.push({path,method,body});return {feedback:{verdict:body.verdict,text:body.text,at:1}}}});
 vm.runInContext([code('el'),code('actionButton'),"var REVIEW_FEEDBACK=[['resembles','Ce bilan me ressemble'],['not_really','Pas vraiment'],['to_correct','À corriger']];",code('drawReviewDigest'),code('drawReviewFeedback')].join('\n'),ctx);
 const digest=ctx.drawReviewDigest('Ce qui se confirme. Le silence protège.\n\nVos objectifs suivis. « oser parler » : en cours.');
 assert.match(digest.textContent,/Ce qu’Ambre retient de ce bilan/);assert.equal(digest.all('P').length,3);
 const box=ctx.drawReviewFeedback({id:'r1',feedback:null},()=>true),buttons=box.all('BUTTON');
 assert.deepEqual(buttons.map(b=>b._text),['Ce bilan me ressemble','Pas vraiment','À corriger','Enregistrer mon retour']);
 assert.equal(buttons[3].disabled,true,'nothing is sent before a choice');assert.doesNotMatch(box.textContent,/\d+ ?\/ ?\d+|score|note sur/i);
 await buttons[2].click();assert.equal(buttons[2].attributes['aria-pressed'],'true');assert.equal(buttons[0].attributes['aria-pressed'],'false');
 box.all('TEXTAREA')[0].value='  Ce n’était pas ma sœur.  ';assert.equal(box.all('TEXTAREA')[0].maxLength,600);
 await buttons[3].click();assert.deepEqual(JSON.parse(JSON.stringify(sent)),[{path:'/api/reviews/r1/feedback',method:'PUT',body:{verdict:'to_correct',text:'Ce n’était pas ma sœur.'}}]);
 assert.match(box.textContent,/Ambre en tiendra compte au prochain point/);
});

// Independent review fixes (A5-P1): cooldown, exhausted attempts, closed lists, daily creations.
test('legacy before 0031: fix 1: during a model cooldown the cron ends a review running for more than ten minutes',async t=>{
 const s=setup();provider(t,{write:()=>{throw Object.assign(new Error('timeout'),{name:'TimeoutError'})}});
 const r=await start(s);assert.equal(s.row(r.review_id).status,'running');
 s.raw.prepare("INSERT INTO provider_state(scope,retry_at,reason,updated_at) VALUES('key',?,'provider_http_429',?)").run(s.now()+3600,s.now());
 s.tick(300);assert.deepEqual(await s.chat.resumeReviews(s.env),{claimed:0,failed:0},'not yet ten minutes');assert.equal(s.row(r.review_id).status,'running');
 s.tick(301);assert.deepEqual(await s.chat.resumeReviews(s.env),{claimed:0,failed:1});
 const row=s.row(r.review_id);assert.equal(row.status,'failed');assert.equal(row.error_code,'provider_cooldown');assert.equal(row.draft,'');
 assert.equal((await s.call('/api/chat')).journey.review.running_id,null,'the conversation is no longer blocked');
});
test('legacy before 0031: fix 2: a review failed by exhausted attempts loses its unchecked draft',async t=>{
 const s=setup();provider(t,{check:()=>new Response('{}',{status:500})});
 const r=await start(s);assert.equal(s.row(r.review_id).stage,'check');assert.notEqual(s.row(r.review_id).draft,'');
 s.raw.prepare('UPDATE ambre_reviews SET attempts=3,lease_until=0 WHERE id=?').run(r.review_id);
 assert.equal((await s.chat.resumeReviews(s.env)).failed,1);
 const row=s.row(r.review_id);assert.equal(row.status,'failed');assert.equal(row.error_code,'provider_http_500');assert.equal(row.draft,'');
});
test('legacy before 0031: fix 3: closed lists accept their own keys only',async t=>{
 const corpus='Je me tais souvent pour éviter les conflits avec ma sœur.';
 for(const bad of ['constructor','__proto__','toString','hasOwnProperty']){
  const d=validateDigest({goals:[{key:'g1',words:'parler',status:bad,quote:'Je me tais souvent pour éviter'}],tried:[{what:'Se taire',result:bad,quote:'Je me tais souvent pour éviter'}]},corpus);
  assert.equal(d.goals.length,0,bad);assert.equal(d.tried.length,0,bad);
  assert.equal(parseFeedbackValue({verdict:bad}),null,bad);
 }
 const s=setup();provider(t);const r=await start(s);
 for(const bad of ['constructor','__proto__','toString'])assert.equal((await s.call('/api/reviews/'+r.review_id+'/feedback','PUT',{verdict:bad})).code,'invalid_input',bad);
 assert.equal(s.row(r.review_id).feedback,null);
});
test('legacy before 0031: fix 4: eight review creations per day, whatever was deleted since',async t=>{
 const s=setup();provider(t,{check:()=>({...PASS,criteria:{...PASS.criteria,no_diagnosis:'no'}})});
 for(let i=0;i<8;i++){const r=await s.call('/api/chat/start','POST',{idempotency_key:'start_key_00'+i,accepted:true});assert.equal(r.status,202,i+' '+r.message);assert.equal(s.row(r.review_id).status,'failed');assert.equal((await s.call('/api/reviews/'+r.review_id,'DELETE')).deleted,true)}
 assert.equal(s.raw.prepare('SELECT count(*) n FROM ambre_reviews').get().n,0,'every attempt was deleted');
 const blocked=await s.call('/api/chat/start','POST',{idempotency_key:'start_key_009',accepted:true});assert.equal(blocked.status,429);assert.equal(blocked.code,'request_limit');
 s.tick(86401);assert.equal((await s.call('/api/chat/start','POST',{idempotency_key:'start_key_010',accepted:true})).status,202,'a new day reopens the counter');
});

test('snapshot reviews retain separate checking and faithful digest anchors after every block is analyzed',async t=>{
 const s=setup({before:null}),calls=provider(t),r=await start(s);assert.equal(r.status,202);const row=s.row(r.review_id);assert.equal(row.status,'ready',JSON.stringify(row));
 const blocks=calls.filter(c=>c.purpose==='ambre_snapshot_chunk');assert.ok(blocks.length>0);assert.deepEqual(calls.slice(-2).map(c=>c.purpose),['ambre_review','ambre_review_check']);
 const input=reviewInput(calls.at(-2));assert.equal(input.coverage.complete,true);assert.equal(input.coverage.analyzed_blocks,blocks.length);assert.equal(input.coverage.total_blocks,blocks.length);
 const check=JSON.parse(calls.at(-1).body.messages[1].content);assert.equal(check.review,text());assert.equal(check.digest.formulation.new.length,0);assert.equal(check.digest.goals.length,1);
 assert.equal(s.raw.prepare("SELECT provider_calls FROM provider_budgets WHERE user_id='a'").get().provider_calls,calls.length);
 assert.equal(JSON.parse(row.sources).check.passed,true);
});
