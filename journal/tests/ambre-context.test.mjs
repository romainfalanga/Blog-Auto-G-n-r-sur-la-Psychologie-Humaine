import test from 'node:test';
import assert from 'node:assert/strict';
import {declaredLifeContext,assertPortraitLanguage} from '../src/portrait-language.mjs';
import {analyzePortraitSnapshot,nativePortraitClaims} from '../src/portrait-refresh.mjs';
import {validatePsychologicalMap} from '../src/psychology-map.mjs';

const source={id:'profile:u',type:'profile',revision:1,explicit:true,fields:{self_description:'Je me décris comme créatif. Je repense souvent au travail et cela me fatigue. Avec mes amis, je sais dire non.'}};source.text='self_description: '+source.fields.self_description;
const citation={source_id:source.id,revision:1,quote:source.fields.self_description};
const hypothesis=id=>({pattern_id:id,status:'exploring',observation:'{{prenom}} décrit une difficulté au travail.',possible_function:'Une protection possible, à vérifier.',alternatives:['Une contrainte de travail.'],next_question:'Que se passe-t-il avec vos amis ?',evidence:[source.id],counter_evidence:[]});
const portrait=()=>({schema_version:2,summary:'{{prenom}} se décrit comme créatif.',summary_evidence:[source.id],axes:[{label:'Pensées répétées',why:'{{prenom}} parle du travail.',confidence:.3,claim_type:'hypothesis',evidence:[source.id]}],strengths:[],uncertainties:[],citations:[citation],psychological_map:[hypothesis('rumination')]});

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
