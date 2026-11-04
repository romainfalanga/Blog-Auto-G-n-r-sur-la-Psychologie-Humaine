import test from 'node:test';
import assert from 'node:assert/strict';
import {declaredLifeContext,assertPortraitLanguage} from '../src/portrait-language.mjs';
import {analyzePortraitSnapshot,nativePortraitClaims,portraitSupportContext} from '../src/portrait-refresh.mjs';
import {validatePsychologicalMap} from '../src/psychology-map.mjs';

const source={id:'profile:u',type:'profile',revision:1,explicit:true,fields:{self_description:'Je me décris comme créatif. Je repense souvent au travail et cela me fatigue. Avec mes amis, je sais dire non.'}};source.text='self_description: '+source.fields.self_description;
const citation={source_id:source.id,revision:1,quote:source.fields.self_description};
const hypothesis=id=>({pattern_id:id,status:'exploring',observation:'{{prenom}} décrit une difficulté au travail.',possible_function:'Une protection possible, à vérifier.',alternatives:['Une contrainte de travail.'],next_question:'Que se passe-t-il avec vos amis ?',evidence:[source.id],counter_evidence:[]});
const portrait=()=>({schema_version:2,summary:'{{prenom}} se décrit comme créatif.',summary_evidence:[source.id],axes:[{label:'Pensées répétées',why:'{{prenom}} parle du travail.',confidence:.3,claim_type:'hypothesis',evidence:[source.id]}],strengths:[],uncertainties:[],citations:[citation],psychological_map:[hypothesis('rumination')]});

test('native validation diagnostics contain fixed categories and never rejected private prose',async()=>{
 for(const [mutate,code] of [[p=>p.citations[0].revision=999,'citation'],[p=>p.summary='Elle contient une formulation privée refusée.','language']]){
  const p=structuredClone(portrait());mutate(p);let calls=0;
  await assert.rejects(()=>analyzePortraitSnapshot({sources:[source]},async()=>{calls++;return JSON.stringify(p)}),e=>e.message==='validation_error'&&JSON.stringify(e.verification_failures)===JSON.stringify(['contract.'+code]));assert.equal(calls,1);
 }
});

test('a contradicted mechanism cannot keep a weighted influence and returns a precise safe diagnostic',async()=>{
 const p=portrait(),h=p.psychological_map[0];h.status='contradicted';h.counter_evidence=[source.id];h.influence={level:'low',contexts:['au travail'],rationale:'Une influence supposée.',evidence:[source.id]};
 await assert.rejects(()=>analyzePortraitSnapshot({sources:[source]},async()=>JSON.stringify(p)),e=>e.message==='validation_error'&&JSON.stringify(e.verification_failures)===JSON.stringify(['contract.map_influence_unsupported']));
});

test('declared context has one closed identity contract and never civil identity',()=>{
 const input={first_name:'SECRET',last_name:'SECRET',birth_date:'1990-01-01',age:27,gender:'male',gender_description:'ignored',occupation:'Designer'};
 assert.deepEqual(declaredLifeContext(input),{age:27,gender:'male',gender_description:'ignored',occupation:'Designer'});
 assert.deepEqual(declaredLifeContext({age:-1,gender:'invented'}),{age:null,gender:'unspecified',gender_description:'',occupation:''});
});

test('new prose rejects ambiguous reprises and direct wrong gender without changing quotes or third parties',()=>{
 assert.throws(()=>assertPortraitLanguage({summary:'{{prenom}} parle du travail. Elle se sent fatiguée.'}));
 assert.throws(()=>assertPortraitLanguage({summary:'{{prenom}} se décrit comme créative.',identity_context:{gender:'male'}}));
 assert.doesNotThrow(()=>assertPortraitLanguage({summary:'{{prenom}} décrit sa mère comme créative. Les mots « Elle se sent fatiguée » sont cités.',identity_context:{gender:'male'}}));
 assert.doesNotThrow(()=>assertPortraitLanguage({summary:'{{prenom}} se décrit comme créatif.',psychological_map:[hypothesis('rumination')],identity_context:{gender:'male'}}));
});

test('native analyst and verifier receive the current structured context and discard an old cached prose',async()=>{
 const job={sources:[source],source_version:3,memory_version:1,self_context:{gender:'male',age:27,occupation:'Designer',first_name:'SECRET'},native_candidate:{...portrait(),summary:'Elle se décrit comme créative.'}};let calls=0;
 const result=await analyzePortraitSnapshot(job,async request=>{calls++;const data=JSON.parse(request.user_prompt);if(request.stage==='support'){assert.equal(data.current_declared_context.gender,'male');return JSON.stringify({checks:nativePortraitClaims(portrait()).map(c=>({path:c.path,supported:true}))});}assert.equal(data.life_context.gender,'male');assert.equal(data.life_context.first_name,undefined);return JSON.stringify(portrait());});
 assert.equal(calls,2);assert.equal(result.portrait.analysis_version,3);assert.equal(result.portrait.identity_context.gender,'male');
 await assert.rejects(()=>analyzePortraitSnapshot({...job,native_candidate:null},async()=>JSON.stringify({...portrait(),summary:'{{prenom}} est créative.'})));
});

test('reported influence stays independent of proof strength and interactions need both active mechanisms',()=>{
 const a=hypothesis('rumination'),b=hypothesis('boundaries');a.influence={level:'high',contexts:['au travail'],rationale:'Une fatigue rapportée, à préciser.',evidence:[source.id]};a.interactions=[{pattern_id:b.pattern_id,kind:'amplifies',explanation:'Une interaction possible au travail.',evidence:[source.id]}];
 const result=validatePsychologicalMap([a,b],[source],[citation]);assert.equal(result[0].status,'exploring');assert.equal(result[0].influence.level,'high');assert.equal(result[0].interactions[0].pattern_id,b.pattern_id);
 assert.throws(()=>validatePsychologicalMap([a],[source],[citation]));
 assert.throws(()=>validatePsychologicalMap([{...a,influence:{...a.influence,evidence:[]}},b],[source],[citation]));
 const unknown={...a,influence:{level:'unknown',contexts:[],rationale:'unverified',evidence:[]},interactions:[]};assert.equal(validatePsychologicalMap([unknown],[source],[citation])[0].influence.rationale,'Influence à préciser.');
 assert.throws(()=>validatePsychologicalMap([{...unknown,influence:{...unknown.influence,contexts:['inventé']}}],[source],[citation]));
 assert.throws(()=>validatePsychologicalMap([a,{...b,status:'contradicted',counter_evidence:[source.id]}],[source],[citation]));
});

test('semantic review keeps only verified sections and removes links to rejected mechanisms',async()=>{
 const p=portrait(),a=p.psychological_map[0],b=hypothesis('boundaries'),c=hypothesis('overcontrol');
 p.axes.push({...p.axes[0],label:'An unsupported axis'});p.uncertainties=['Que souhaitez-vous explorer ?', 'Comment le traumatisme de votre enfance explique vos relations ?'];
 a.influence={level:'high',contexts:['au travail'],rationale:'Une fatigue possible.',evidence:[source.id]};
 a.interactions=[{pattern_id:b.pattern_id,kind:'amplifies',explanation:'Un lien possible.',evidence:[source.id]}];
 c.interactions=[{pattern_id:a.pattern_id,kind:'buffers',explanation:'Un effet possible.',evidence:[source.id]}];
 p.psychological_map.push(b,c);let calls=0;
 const rejected=new Set(['axes.1','map.1.function','map.0.influence','map.2.interaction.0','uncertainties.1']);
 const result=await analyzePortraitSnapshot({sources:[source],self_context:{gender:'male'}},async request=>{calls++;return JSON.stringify(request.stage==='support'?{checks:nativePortraitClaims(p).map(claim=>({path:claim.path,supported:!rejected.has(claim.path)}))}:p);});
 assert.equal(calls,2);assert.equal(result.portrait.axes.length,1);assert.equal(result.portrait.summary,p.summary);
 assert.deepEqual(result.portrait.psychological_map.map(h=>h.pattern_id),['rumination','overcontrol']);
 assert.equal(result.portrait.psychological_map[0].influence,undefined);
 assert.ok(result.portrait.psychological_map.every(h=>!h.interactions?.length));
 assert.equal(result.portrait.citations[0].quote,citation.quote);assert.deepEqual(result.portrait.uncertainties,['Que souhaitez-vous explorer ?']);
});

test('a rejected summary still prevents publication and exposes only fixed diagnostic paths',async()=>{
 const p=portrait();await assert.rejects(()=>analyzePortraitSnapshot({sources:[source]},async request=>JSON.stringify(request.stage==='support'?{checks:nativePortraitClaims(p).map(c=>({path:c.path,supported:c.path!=='summary'}))}:p)),e=>e.message==='validation_error'&&e.verification_failures.includes('summary'));
});

test('language guard covers axis and preference labels while preserving quoted words',()=>{
 for(const p of [{axes:[{label:'Elle est anxieuse',why:'Une réaction située.'}]},{reading_preferences:{preferred:[{label:'Il souhaite comprendre'}],avoid:[]}},{reading_preferences:{preferred:[],avoid:[{label:'Elle souhaite éviter'}]}}])assert.throws(()=>assertPortraitLanguage(p));
 assert.doesNotThrow(()=>assertPortraitLanguage({axes:[{label:'Les mots « Elle est anxieuse »',why:'Une citation.'}],reading_preferences:{preferred:[{label:'Les mots « Il souhaite comprendre »'}],avoid:[]}}));
});

test('uncited current memory reaches semantic review as counter-context and cannot become silent support',async()=>{
 const memory={id:'memory:u',type:'memory',revision:2,explicit:true,fields:{text:'Correction actuelle : je ne rumine plus et je me sens reposé.'},temporal:{start:'2026-11-04'}};memory.text='text: '+memory.fields.text;
 const p=portrait(),original=structuredClone(p);let calls=0;
 await assert.rejects(()=>analyzePortraitSnapshot({sources:[source,memory],self_context:{gender:'male'}},async request=>{
  calls++;if(request.stage==='portrait')return p;
  const data=JSON.parse(request.user_prompt);
  assert.deepEqual(data.original_sources.map(x=>x.id),[source.id]);assert.deepEqual(data.correction_context,[memory]);
  assert.ok(data.claims.every(c=>!c.source_ids.includes(memory.id)));assert.deepEqual(data.citations,original.citations);
  assert.match(request.system_prompt,/jamais comme nouvelles preuves d’appui/);
  return {checks:data.claims.map(c=>({path:c.path,supported:c.path!=='summary'}))};
 }),e=>e.message==='validation_error');
 assert.equal(calls,2);
 const corrected=portrait();corrected.summary='{{prenom}} rapporte une amélioration actuelle.';corrected.summary_evidence=[memory.id];
 corrected.axes[0]={...corrected.axes[0],why:'{{prenom}} décrit une amélioration.',evidence:[memory.id]};
 corrected.citations.push({source_id:memory.id,revision:memory.revision,quote:memory.fields.text});
 corrected.psychological_map[0]={...corrected.psychological_map[0],status:'contradicted',counter_evidence:[memory.id]};
 const result=await analyzePortraitSnapshot({sources:[source,memory],self_context:{gender:'male'}},async request=>request.stage==='portrait'?corrected:{checks:JSON.parse(request.user_prompt).claims.map(c=>({path:c.path,supported:true}))});
 assert.deepEqual(result.portrait.summary_evidence,[memory.id]);assert.deepEqual(result.portrait.psychological_map[0].counter_evidence,[memory.id]);
 assert.equal(result.portrait.citations.at(-1).quote,memory.fields.text);
});

test('counter-context budget prioritizes original corrections and reports omitted and pending IDs separately',()=>{
 const p=portrait(),optional={id:'person:other',type:'person',revision:1,explicit:true,fields:{relationship_text:'Une rencontre agréable.'},text:'relationship_text: Une rencontre agréable.'};
 const current={id:'event:now',type:'event',revision:1,explicit:true,fields:{description:'Un événement.',meaning_now:'Mon regard a changé.'},text:'description: Un événement.\nmeaning_now: Mon regard a changé.',temporal:{start:'2026-11-04'}};
 p.coverage={processed:[source.id],pending:[current.id,optional.id]};const saved=structuredClone(p);
 const limit=JSON.stringify(source).length+JSON.stringify(current).length;
 const data=portraitSupportContext(p,[source,optional,current],{sourceCharLimit:limit});
 assert.deepEqual(data.correction_context,[current]);assert.deepEqual(data.correction_context_pending,[current.id]);
 assert.deepEqual(data.support_context_coverage.omitted_ids,[optional.id]);assert.deepEqual(p,saved);
 assert.throws(()=>portraitSupportContext(p,[source,current],{sourceCharLimit:limit-1}),e=>e.code==='support_context_budget');
});

test('new native analysis refuses null and absent maps before semantic review',async()=>{
 for(const absent of [false,true]){
  const p=portrait();if(absent)delete p.psychological_map;else p.psychological_map=null;let calls=0;
  await assert.rejects(()=>analyzePortraitSnapshot({sources:[source]},async()=>{calls++;return p;}),e=>e.message==='validation_error'&&JSON.stringify(e.verification_failures)===JSON.stringify(['contract.map_shape']));assert.equal(calls,1);
 }
});

test('a corrective source outside native selection cannot disappear from support or gain processed coverage',async()=>{
 const memory={id:'memory:large',type:'memory',revision:2,explicit:true,fields:{text:'Correction actuelle : '+('texte '.repeat(12000))}};memory.text='text: '+memory.fields.text;
 let calls=0,saved;
 await assert.rejects(()=>analyzePortraitSnapshot({sources:[source,memory],self_context:{gender:'male'}},async request=>{
  calls++;assert.equal(request.stage,'portrait');return portrait();
 },{saveCandidate:async candidate=>{saved=structuredClone(candidate);}}),error=>error.code==='support_context_budget');
 assert.equal(calls,1);assert.deepEqual(saved.coverage,{processed:[source.id],pending:[memory.id]});
 assert.ok(saved.citations.every(c=>c.source_id!==memory.id));assert.deepEqual(saved.summary_evidence,[source.id]);
});
