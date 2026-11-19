// Lot A5-P0 (plan v5.1, audit v5 defects A, B, C, D, E, G): what reaches Ambre on the real
// engine, on the filled profile of the audit, with a simulated provider.
import test from 'node:test';
import assert from 'node:assert/strict';
import {replayTurn,replayReview,MESSAGES,setup,analysis,response} from './v5-A-helper.mjs';
import {knowledgeIndex,selectDetails,headTail} from '../src/ambre-engine.mjs';
import {companionCartography,readCartography} from '../src/cartographie.mjs';
import {cautiousReply} from '../src/chat-safety.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';

const writingInput=run=>JSON.parse(run.get('ambre_writing').input);

// A5-P1: the review thread replaced last_review; these seeded reviews have no digest, so the
// thread carries the dated head + tail excerpt of the latest one, as P0 did.
for(const strategy of ['none','ideal'])test(`A: the dated last review reaches the writer for the four message types (${strategy})`,async()=>{
 for(const [label,message] of MESSAGES){
  const run=await replayTurn(message,{strategy});assert.equal(run.stage,'done',label);const input=writingInput(run),last=input.review_thread.last;
  assert.equal(input.last_review,undefined);assert.equal(last.id,'rev-p2',label);assert.equal(last.kind,'point d’étape');assert.ok(last.date);
  assert.ok(last.text.startsWith('BILAN_P2_DEBUT')&&last.text.endsWith('BILAN_P2_FIN'),label+': beginning and end of the review');
  assert.ok(last.text.length<=900);assert.ok(JSON.stringify(input.review_thread).length<=1200);assert.ok(!input.details.some(x=>x.key==='review:rev-p2'),label);
 }
});

test('B: details follow the priority order of the analysis; a later choice is shortened or dropped first',async()=>{
 const run=await replayTurn(MESSAGES[0][1],{strategy:'ideal'}),analysisInput=JSON.parse(run.get('ambre_analysis').input),rows=analysisInput.index.split('\n'),id=pattern=>rows.find(r=>pattern.test(r)).split(' ')[0];
 // Chosen in this order: synthesis, Léa, the move to Lyon, the presentation, then four more.
 const chosen=[id(/ portrait Synthèse/),id(/ person Léa/),id(/ event Déménagement/),id(/ profile Ma présentation/),id(/ person Marc/),id(/ event Burn-out/),id(/ profile Mon contexte/),id(/ person Paul/)];
 const again=await replayTurn(MESSAGES[0][1],{choose:()=>chosen}),details=writingInput(again).details;
 assert.deepEqual(details.slice(0,3).map(x=>x.id),chosen.slice(0,3),'the three first choices survive, in order');
 assert.deepEqual(details.map(x=>x.id),chosen.slice(0,details.length),'no lexical match displaces a choice');
 // Unit level: with a tight budget the first choice stays complete up to its cap, the last ones leave.
 const index=knowledgeIndex({people:[],events:[],sources:[],portrait:{summary:'S'.repeat(3000),axes:[{label:'A',why:'W'.repeat(3000),confidence:.5}],current_state:{summary:'C'.repeat(3000)},evolution:{summary:'E'.repeat(3000)}}});
 const ids=[...index].reverse().map(x=>x.id),tight=selectDetails(index,ids,'',1800);
 assert.equal(tight[0].id,ids[0]);assert.equal(tight[0].text.length,700);assert.ok(tight.length<ids.length);assert.ok(JSON.stringify(tight).length<=1800);
});

test('C: every rubric of the self description is its own index entry, reachable with its beginning and its end',async()=>{
 const run=await replayTurn(MESSAGES[3][1],{choose:index=>index.split('\n').filter(r=>/ profile /.test(r)).map(r=>r.split(' ')[0])});
 const input=writingInput(run),text=JSON.stringify(input.details);
 for(const marker of ['PRES_DEBUT','PRES_FIN','PERC_DEBUT','PERC_FIN','SCHEM_DEBUT','SCHEM_FIN','EMO_DEBUT','EMO_FIN','CTX_DEBUT','CTX_FIN'])assert.ok(text.includes(marker),marker);
 assert.deepEqual(input.details.map(x=>x.label).slice(0,5),['Ma présentation','Comment je me perçois','Les schémas que je remarque','Les émotions présentes','Mon contexte actuel']);
});

test('D: axes, current state and evolution of the synthesis are indexed and transmitted when chosen',async()=>{
 const run=await replayTurn(MESSAGES[3][1],{choose:index=>index.split('\n').filter(r=>/ portrait /.test(r)).map(r=>r.split(' ')[0])});
 const text=JSON.stringify(writingInput(run).details);
 for(const marker of ['SYNTH_DEBUT','SYNTH_FIN','AXE_Vigilance','AXE_Besoin','ETAT_ACTUEL','EVOLUTION_','piste à confirmer'])assert.ok(text.includes(marker),marker);
 const map=[{pattern_id:'rumination',label:'Ruminations',status:'exploring',observation:'O'.repeat(3000),next_question:'NEXT_Q ?',counter_evidence:['COUNTER_1']}];
 const detail=selectDetails(knowledgeIndex({people:[],events:[],sources:[],portrait:{psychological_map:map}}),['i0'],'')[0];
 assert.ok(detail.text.includes('NEXT_Q')&&detail.text.includes('COUNTER_1'),'next question and counter-evidence survive a short excerpt');
});

test('G: a moment carries its age, importance and linked people by name, and keeps the end of its account and its meaning',async()=>{
 const run=await replayTurn(MESSAGES[2][1],{choose:index=>[index.split('\n').find(r=>/Déménagement/.test(r)).split(' ')[0]]}),detail=writingInput(run).details[0];
 assert.equal(detail.key,'event:ev5');
 for(const part of ['Âge alors : 32 ans','Place aujourd’hui : 4/5','Proches liés : Julien','Repère : 2023-09','DESC_DEBUT_ev5','DESC_FIN_ev5','SENS_ev5'])assert.ok(detail.text.includes(part),part);
 const index=JSON.parse(run.get('ambre_analysis').input).index;assert.match(index,/person Léa — Frère ou sœur/);
 // Year precision gives an approximate age; an age-only moment keeps its declared words.
 const s=setup();s.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({first_name:'C',birth_date:'1991-03-02',self_description:'x'}));
 s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description,meaning_now,date_precision,date_start) VALUES('y','a','y','h','Année','other','d','', 'year','2006')").run();
 s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description,meaning_now,date_precision,date_start) VALUES('d','a','d','h','Jour','other','d','', 'day','2001-03-01')").run();
 const events=(await s.personal.context(s.env,'a')).events;
 assert.deepEqual(events.find(e=>e.id==='y').age_at_event,15);assert.equal(events.find(e=>e.id==='y').age_at_event_approx,true);
 assert.equal(events.find(e=>e.id==='d').age_at_event,9);assert.equal(events.find(e=>e.id==='d').age_at_event_approx,undefined);
});

test('index: the map follows the page sections and carries no contextual subject',async()=>{
 const run=await replayTurn(MESSAGES[0][1],{strategy:'none'}),s=run.s,context=await s.personal.context(s.env,'a'),index=knowledgeIndex(context);
 const outcomes=index.filter(x=>x.kind==='outcome');assert.ok(outcomes.length);
 assert.ok(outcomes.every(x=>!x.metadata.contextual_only));assert.equal(outcomes[0].metadata.section,'top');
 const counts=outcomes.filter(x=>x.metadata.feedback!=='non').map(x=>x.metadata.associated_count);
 assert.deepEqual(counts,[...counts].sort((a,b)=>b-a),'associated experience count order');

});

test('legacy before 0031 E: the initial analysis reads every rubric, the whole synthesis and the accounts of people and moments',async()=>{
 const {user,data}=await replayReview('initial',{legacy:true});
 for(const marker of ['PRES_FIN','PERC_FIN','SCHEM_FIN','EMO_FIN','CTX_FIN','AXE_','ETAT_ACTUEL','EVOLUTION_','DESC_FIN_ev5','SENS_ev5'])assert.ok(user.includes(marker),marker);
 assert.equal(data.sources.filter(x=>x.type==='person').length,6);assert.equal(data.sources.filter(x=>x.type==='event').length,12);
 assert.ok(JSON.stringify(data).length<=34000);assert.ok(JSON.stringify(data.childhood_cartography).length<=10000,'map halved (20 261 characters before)');
 assert.equal(data.events.find(e=>e.id==='ev5').age_at_event,32);assert.deepEqual(data.events.find(e=>e.id==='ev1').people,['Marc','Sophie']);
});

test('legacy before 0031 E: a point keeps at least 6 people and 8 moments with text, the present rubrics and six exchanges, map reduced first',async()=>{
 const {data,user}=await replayReview('point',{legacy:true});
 assert.ok(data.sources.filter(x=>x.type==='person').length>=6);assert.ok(data.sources.filter(x=>x.type==='event').length>=8);
 assert.ok(user.includes('CTX_DEBUT')&&user.includes('EMO_DEBUT'),'current context and emotions stay');
 assert.equal(data.conversation_since.length,6);assert.ok(data.previous_review.text.includes('BILAN_P2_FIN'));
 const kept=data.sources.filter(x=>x.type==='event').map(x=>x.id),importance=Object.fromEntries(data.events.map(e=>[e.id,e.current_importance]));
 const keptSet=new Set(kept.map(id=>id.slice(6))),omitted=data.events.filter(e=>!keptSet.has(e.id)).map(e=>e.current_importance||0),minKept=Math.min(...kept.map(id=>importance[id.slice(6)]||0));
 assert.ok(omitted.every(v=>v<=minKept),'omitted moments are never more important today than the moments kept');
 assert.ok(JSON.stringify(data).length<=34000);assert.equal(data.coverage.included,data.sources.length);
});

test('welcome message follows the count-based public order',async()=>{
 const s=setup();await s.ready();const answers=Object.fromEntries(CARTOGRAPHIE.factors.map((x,i)=>[x.id,i%3===0?1:0]));
 s.raw.prepare("INSERT OR REPLACE INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a',?,'{}','{}','{}','[]',1,1,2,1)").run(JSON.stringify(answers));
 const summary=companionCartography(await readCartography(s.env,'a')),welcome=(await s.call('/api/chat')).welcome.content;
 const expected=summary.sections.top.filter(x=>x.note!=='suicide').slice(0,3);assert.ok(expected.length);
 for(const item of expected)assert.ok(welcome.includes('« '+item.label+' »'),item.label);
});

test('plan v5.1: the cautious reply after a double outage is not counted in the weekly quota; a single outage still is',async t=>{
 const original=globalThis.fetch;t.after(()=>{globalThis.fetch=original});
 const outage=(purpose,body)=>purpose==='ambre_analysis'?new Response('{}',{status:500}):purpose==='ambre_risk_check'?new Response('{}',{status:502}):null;
 const play=async(handler,key)=>{const s=setup();await s.ready();globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name;return handler(purpose,body)||response({ambre_writing:{content:'Je vous écoute.',questions:[],proposals:[],cartography_notes:[],people_observations:[],change_proposals:[],followup:null},ambre_review:{criteria:Array(15).fill('yes'),reasons:[],change_criteria:Array.from({length:8},(_,i)=>({id:'C-R'+(i+1),verdict:'not_applicable'}))},ambre_memory:{items:[]},ambre_risk_check:{risk:'none',category:'none',evidence_quote:''}}[purpose])};
  await s.send('Ces temps-ci je pense souvent à la mort.',key);await s.settle(4);return s};
 const twice=await play(outage,'double_outage');
 assert.equal(twice.job().stage,'done');assert.equal(JSON.parse(twice.turn().response).content,cautiousReply('self_harm'));
 assert.equal(twice.turn().status,'completed');assert.equal(twice.turn().error_code,'uncounted_cautious_reply');
 const state=await twice.call('/api/chat');assert.equal(state.quota.used,0);assert.equal(state.quota.remaining,100);
 // Analysis available (possible risk) but verification refused: one outage, the reply is counted.
 const once=await play(purpose=>purpose==='ambre_analysis'?response(analysis({risk:'possible',posture:'safety'})):purpose==='ambre_risk_check'?new Response('{}',{status:502}):null,'single_outage');
 assert.equal(once.job().stage,'done');assert.equal(JSON.parse(once.turn().response).content,cautiousReply('self_harm'));assert.equal(once.turn().error_code,null);
 assert.equal((await once.call('/api/chat')).quota.used,1);
});

test('headTail keeps both ends within the limit',()=>{
 const text='DEBUT '+'x'.repeat(2000)+' FIN';const cut=headTail(text,300);assert.ok(cut.length<=300);assert.ok(cut.startsWith('DEBUT')&&cut.endsWith('FIN'));assert.equal(headTail('court',300),'court');
});

for(const kind of ['initial','point'])test(`full snapshot ${kind} sends every personal section to analysis without the legacy excerpts`,async()=>{
 const run=await replayReview(kind),snapshot=run.snapshot;assert.equal(run.r.status,202);assert.ok(snapshot);assert.equal(snapshot.payload.complete,true);
 const sections=snapshot.payload.sections,text=run.chunkCalls.map(c=>c.messages[1].content).join('\n');
 for(const marker of ['PRES_FIN','PERC_FIN','SCHEM_FIN','EMO_FIN','CTX_FIN','AXE_','ETAT_ACTUEL','EVOLUTION_','DESC_FIN_ev5','SENS_ev5'])assert.ok(text.includes(marker),marker);
 assert.equal(sections.people.length,6);assert.equal(sections.events.length,12);
 if(kind==='point'){assert.ok(sections.conversation.length>=8);assert.ok(JSON.stringify(sections.previous_reviews).includes('BILAN_P2_FIN'));}
 assert.equal(run.data.coverage.complete,true);assert.equal(run.data.coverage.total_blocks,run.chunkCalls.length);assert.equal(run.data.coverage.analyzed_blocks,run.chunkCalls.length);
 const paths=new Set(run.data.blocks.flatMap(b=>b.source_refs.map(r=>r.path)));
 for(const [section,items] of Object.entries(sections))for(let i=0;i<items.length;i++){
  const path=section+'/'+i;assert.ok(paths.has(path),path+' analyzed');
  const fragments=run.chunkCalls.flatMap(c=>JSON.parse(c.messages[1].content).items).filter(x=>x.path===path).sort((a,b)=>a.offset-b.offset);
  let through=0;for(const fragment of fragments){assert.equal(fragment.offset,through,path+' has no gap or duplicate');through+=fragment.text.length;assert.ok(through<=fragment.total_chars);}
  assert.equal(through,fragments[0].total_chars,path+' includes every character');
 }
});
