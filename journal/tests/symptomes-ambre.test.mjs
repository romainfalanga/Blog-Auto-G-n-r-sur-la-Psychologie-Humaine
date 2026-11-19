import test from 'node:test';
import assert from 'node:assert/strict';
import {SYMPTOMES_CARTE as data} from '../src/symptomes-carte.generated.mjs';
import {symptomMapView,symptomMapBlock,symptomReviewBlock,symptomMapIndexEntries,childhoodMapBlock,mapSafetyState,ensureMapSafetyQuestion,MAP_HYPOTHESIS_RULES,SAFETY_WINDOW} from '../src/symptom-map-context.mjs';
import {reviewSafetySignal,reviewCorpus} from '../src/review-digest.mjs';
import {replayTurn,replayReview} from './v5-A-helper.mjs';
const now=1800000000;
function fullRow(){return {version:1,updated_at:now,answers:Object.fromEntries(data.factors.map((f,i)=>[f.id,i%7===0?'unknown':1])),feedback:{},notes:{factors:{},outcomes:{}}};}
test('full symptom map has a bounded overview and theme / subject detail index',()=>{
 const row=fullRow(),view=symptomMapView(row),block=symptomMapBlock(view),index=symptomMapIndexEntries(view);
 assert.ok(block.length<=3500);assert.match(block,/12 derniers mois/);assert.ok(index.some(e=>e.key.startsWith('symptom_map:theme:')));assert.ok(index.some(e=>e.key.startsWith('symptom_map:subject:')));
 assert.equal(view.themes.reduce((n,t)=>n+t.answers.length,0),data.factors.length);
 assert.ok(view.themes.some(t=>t.answers.some(a=>a.answer==='Je ne sais pas')));
 assert.ok(symptomMapBlock(view,6000).length<=6000);
});
test('No and personal notes remain in theme details; feedback is not inferred from answers',()=>{
 const row=fullRow(),id=data.factors[0].id;row.answers[id]=0;row.notes.factors[id]='Ces mots sont les miens.';
 const view=symptomMapView(row),answer=view.themes.flatMap(t=>t.answers).find(a=>a.id===id);
 assert.equal(answer.answer,'Non');assert.equal(answer.note,row.notes.factors[id]);assert.deepEqual(view.feedback,{});
});
test('childhood fixed overview fits its separate 3000 character allowance',()=>{
 assert.ok(childhoodMapBlock({causes:Array.from({length:80},()=>({label:'Un vécu déclaré '.repeat(10)}))}).length<=3000);
});
test('recent safety declaration uses server dates and an actual-question cooldown',()=>{
 assert.equal(mapSafetyState({safety_at:null},now),null);assert.equal(mapSafetyState({safety_at:now+1},now),null);
 assert.equal(mapSafetyState({safety_at:now-SAFETY_WINDOW},now),null);
 assert.equal(mapSafetyState({safety_at:now-10,safety_asked_at:null},now).ask_now,true);
 assert.equal(mapSafetyState({safety_at:now-10,safety_asked_at:now-20},now).ask_now,false);
 assert.equal(mapSafetyState({safety_at:now-10,safety_asked_at:now-SAFETY_WINDOW},now).ask_now,true);
});
test('security guard preserves current-risk handling and does not duplicate question',()=>{
 const safety={ask_now:true,kinds:['suicide']},plain='Je vous écoute.';
 assert.equal(ensureMapSafetyQuestion(plain,safety,{risk:'explicit'}),plain);
 const guarded=ensureMapSafetyQuestion(plain,safety);assert.match(guarded,/3114/);assert.match(guarded,/15 ou le 112/);assert.equal(ensureMapSafetyQuestion(guarded,safety),guarded);
 assert.equal(ensureMapSafetyQuestion(plain,{...safety,ask_now:false}),plain);
 assert.equal(reviewSafetySignal({map_safety:safety}),'self_harm');
});
test('hypothesis contract allows named hypotheses but not diagnoses or automatic referral',()=>{
 assert.match(MAP_HYPOTHESIS_RULES,/hypothèse : X/);assert.match(MAP_HYPOTHESIS_RULES,/il manque C/);assert.match(MAP_HYPOTHESIS_RULES,/pas systématiquement/);
});
function seed(s,{safety=false}={}){
 const row=fullRow();s.raw.prepare("INSERT INTO symptom_maps(user_id,answers,feedback,notes,consent,locked_at,version,updated_at,safety_at) VALUES('a',?,?,?,1,?,1,?,?)").run(JSON.stringify(row.answers),'{}',JSON.stringify(row.notes),s.now(),s.now(),safety?s.now():null);
}
test('conversation receives both fixed maps and symptom details independently',async()=>{
 const run=await replayTurn('Je voudrais comprendre ce que mes symptômes peuvent suggérer.',{extend:s=>seed(s)});
 assert.equal(run.stage,'done');const input=JSON.parse(run.get('ambre_writing').input);
 assert.ok(input.symptom_map?.length<=3500);assert.ok(input.childhood_cartography?.length<=3000);assert.match(input.symptom_map,/Mes symptômes/);
 assert.ok(run.get('ambre_writing').tokens<=5953*1.4,'two full maps stay within the approved writing +40% envelope');
});
test('reviews receive their own symptom allocation',async()=>{
 const run=await replayReview('initial',{extend:s=>seed(s)});assert.equal(run.r.status,202);assert.ok(JSON.stringify(run.data.symptom_map).length<=6000);assert.ok(JSON.stringify(run.data).length<=40000);assert.ok(run.tokens<=12423*1.2,'review with symptoms stays within approved +20%');
});
test('published safety question commits its timestamp; subsequent context does not repeat it',async()=>{
 const run=await replayTurn('Je voudrais comprendre mes réponses.',{extend:s=>seed(s,{safety:true})});assert.equal(run.stage,'done');
 const row=run.s.raw.prepare("SELECT * FROM symptom_maps WHERE user_id='a'").get();
 assert.equal(row.safety_asked_at,run.s.now());assert.equal(mapSafetyState(row,run.s.now()).ask_now,false);
 const turn=run.s.raw.prepare("SELECT response FROM chat_turns WHERE user_id='a' ORDER BY rowid DESC LIMIT 1").get();assert.match(turn.response,/3114/);assert.match(turn.response,/en sécurité/);
});

test('a persistent or renewed positive safety answer keeps its flag beyond fourteen days',()=>{
 const factor=data.factors.find(f=>f.safety),row={answers:{[factor.id]:1},safety_at:now-30*86400,safety_asked_at:now-15*86400};
 assert.equal(mapSafetyState(row,now).ask_now,true);
 assert.equal(mapSafetyState({...row,answers:{[factor.id]:0}},now),null);
 assert.equal(mapSafetyState({...row,safety_asked_at:now-86400},now).ask_now,false);
});

test('free note without an answer stays available and is not interpreted as No',()=>{
 const id=data.factors[0].id,view=symptomMapView({answers:{},notes:{factors:{[id]:'Une précision sans réponse.'}}});
 const a=view.themes.flatMap(t=>t.answers).find(a=>a.id===id);assert.equal(a.answer,'Non renseigné');assert.equal(a.note,'Une précision sans réponse.');
});
test('an existing safety question still receives verified resources',()=>{
 assert.match(ensureMapSafetyQuestion('En ce moment, êtes-vous en sécurité ?',{ask_now:true}),/3114/);
});

test('two full maps and maximal people/events finish a real rewrite within the reserved envelope',async()=>{
 const run=await replayTurn('Je rumine après les critiques et voudrais comprendre mes symptômes.',{rewrite:true,extend:s=>{
  seed(s);const text='Je rumine après les critiques et cherche mes limites. '.repeat(120).slice(0,6000);
  for(let i=0;i<44;i++)s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,relationship_text) VALUES(?,'a',?,'hash',?,'friend',?)").run('sxperson'+i,'sxperson'+i,'Proche '+i,text);
  for(let i=0;i<480;i++)s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,description,meaning_now,kind) VALUES(?,'a',?,'hash',?,?,?,'other')").run('sxevent'+i,'sxevent'+i,'Moment '+i,text,text.slice(0,2000));
 }});
 assert.equal(run.stage,'done');assert.equal(run.calls.filter(c=>c.purpose==='ambre_writing').length,2);assert.equal(run.calls.filter(c=>c.purpose==='ambre_review').length,2);
 assert.ok(run.s.job().reserved_tokens<=42000);for(const c of run.calls.filter(c=>c.purpose==='ambre_writing'))assert.ok(c.tokens<=5953*1.4);
});

test('review notes stay within their own allowance and are eligible exact user quotations',()=>{
 const row=fullRow(),id=data.factors[0].id;row.notes.factors[id]='Je me sens tendu avant chaque réunion importante.';
 const block=symptomReviewBlock(symptomMapView(row));assert.ok(JSON.stringify(block).length<=6000);
 assert.match(reviewCorpus({symptom_map:block}),/Je me sens tendu avant chaque réunion importante/);
 assert.doesNotMatch(reviewCorpus({symptom_map:block}),/Hypothèses, jamais diagnostics/);
});
test('past answers and notes reach Ambre as history, never current positive signs or questionnaire safety',()=>{
 const f=data.factors.find(f=>f.safety),row={answers:{[f.id]:'past'},notes:{factors:{[f.id]:'Cela se produisait il y a plusieurs années.'}},safety_at:now-10,version:1,updated_at:now};
 const view=symptomMapView(row),a=view.themes.flatMap(t=>t.answers).find(a=>a.id===f.id);
 assert.equal(a.answer,'Dans le passé');assert.equal(a.historical,true);assert.equal(a.period,'avant ces 12 derniers mois');assert.equal(view.subjects.length,0);assert.equal(mapSafetyState(row,now),null);
 const block=symptomMapBlock(view);assert.match(block,/Dans le passé \(avant ces 12 derniers mois\)/);const entries=symptomMapIndexEntries(view);assert.match(entries.map(e=>JSON.stringify(e)).join('\n'),/Cela se produisait il y a plusieurs années/);
 assert.match(MAP_HYPOTHESIS_RULES,/ne les compte pas comme Oui récent/);
});
