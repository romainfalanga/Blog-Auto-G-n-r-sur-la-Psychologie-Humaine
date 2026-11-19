// « Mes symptômes », lot C: what Ambre receives (conversation and reviews), the safety antecedent
// and the budget. Plan v1.1 (docs/plan-mes-symptomes-2026-11-19.md), contract § 7 and § 10.
import test from 'node:test';
import assert from 'node:assert/strict';
import {SYMPTOMES} from '../src/symptomes-data.generated.mjs';
import {computeModuleResult} from '../src/symptomes-engine.mjs';
import {renderSymptomsBlock,symptomsSafetyState,ensureSymptomsSafetyQuestion,descriptiveLabel,DISORDER_NAME,SYMPTOMS_BLOCK_MAX,SYMPTOMS_RULES} from '../src/symptomes-ambre.mjs';
import {symptomsForAmbre,createSymptomsBackend} from '../src/symptomes-backend.mjs';
import {reviewSafetySignal} from '../src/review-digest.mjs';
import {replayTurn,replayReview,MESSAGES} from './v5-A-helper.mjs';

const DAY=86400;
const PHQ9={phq9_1:2,phq9_2:2,phq9_3:1,phq9_4:2,phq9_5:1,phq9_6:1,phq9_7:1,phq9_8:1,phq9_9:0};
// A synthetic « above the reference » result for every published module, each positive rule
// carrying every subject the module can signal: the fullest possible block and crossing.
const subjectsOf=id=>Object.entries(SYMPTOMES.subject_modules||{}).filter(([,mods])=>mods.includes(id)).map(([s])=>s);
function fullResult(m){
 const instrument=m.rules.find(r=>r.kind==='instrument');
 const rules=m.rules.map(r=>({id:r.id,kind:r.kind,result:'positif',subjects:subjectsOf(m.id),display:true,...(r.instrument?{instrument:r.instrument,level:{id:'severe',label:'sévère',max:27},score:22}:{})}));
 return {module:m.id,state:instrument?'au_dessus':'repere_present',rules,scores:{phq9:22},positive_count:rules.length};
}
function seedSymptoms(s,{user='a',consent=true,full=true,completed=null,alertAt=null,alertKind='suicide',birth=null}={}){
 const now=s.now();
 if(birth){const p=s.raw.prepare('SELECT self_data FROM profiles WHERE user_id=?').get(user);s.raw.prepare('UPDATE profiles SET self_data=? WHERE user_id=?').run(JSON.stringify({...JSON.parse(p.self_data||'{}'),birth_date:birth}),user)}
 if(consent!==null)s.raw.prepare('INSERT OR REPLACE INTO symptom_consents(user_id,consent,content_version,granted_at,withdrawn_at,updated_at) VALUES(?,?,?,?,NULL,?)').run(user,consent?1:0,SYMPTOMES.version,now-10*DAY,now-10*DAY);
 const ins=s.raw.prepare("INSERT OR REPLACE INTO symptom_checks(id,user_id,passation_key,module,content_version,instrument_versions,answers,scores,result,state,declined,started_at,completed_at) VALUES(?,?,?,?,?,'{}',?,?,?,?,0,NULL,?)");
 const modules=full?SYMPTOMES.modules:SYMPTOMES.modules.filter(m=>m.id==='humeur');
 for(const [i,m] of modules.entries()){
  const result=m.id==='humeur'?computeModuleResult('humeur',{securite_actuelle:'non',...PHQ9,humeur_gene:'nettement'},SYMPTOMES):fullResult(m);
  ins.run('chk-'+user+'-'+m.id,user,'key-'+user+'-'+m.id+'-0001',m.id,SYMPTOMES.version,JSON.stringify(m.id==='humeur'?PHQ9:{}),JSON.stringify(result.scores||{}),JSON.stringify(result),result.state,completed??now-(i+1)*DAY);
 }
 if(alertAt!==null)s.raw.prepare("INSERT INTO symptom_safety_alerts(id,user_id,passation_key,module,trigger_id,kind,screen,check_id,created_at) VALUES(?,?,?,?,?,?,'bloquant',NULL,?)").run('al-'+user+'-'+alertAt,user,'key-alert-'+alertAt,'humeur','securite_oui',alertKind,now+alertAt);
}

test('block: descriptive labels, dates, band in words, crossing; never an item, a score or a disorder name',()=>{
 const now=1794830400,humeur=computeModuleResult('humeur',{...PHQ9},SYMPTOMES);
 assert.equal(humeur.state,'au_dessus');
 const checks=[{module:'humeur',state:'au_dessus',result:humeur,completed_at:now-2*DAY},{module:'anxiete',state:'en_dessous',result:{rules:[]},completed_at:now-DAY},{module:'sommeil',state:'repere_present',result:fullResult(SYMPTOMES.modules.find(m=>m.id==='sommeil')),completed_at:now-120*DAY}];
 const crossing={lists:{deux:[{id:'dep_generale',label:'Humeur basse ou perte d’intérêt'},{id:'ten_sommeil',label:'Difficultés de sommeil (insomnie, cauchemars)'},{id:'x',label:'Trouble panique'}]}};
 const block=renderSymptomsBlock({checks,crossing,now,data:SYMPTOMES});
 assert.match(block,/Humeur basse et perte d’intérêt : au-dessus du repère publié, palier « modéré » ; résultat du 14 novembre 2026\./);
 assert.match(block,/Sommeil et cauchemars : plusieurs repères présents \(repère non validé\) ; résultat du .* \(ancien : plus de 90 jours/);
 assert.doesNotMatch(block,/Inquiétude/,'a theme under the reference is not sent');
 assert.match(block,/Présents dans les deux questionnaires .*jamais une cause\) : Humeur basse ou perte d’intérêt ; Difficultés de sommeil\./);
 assert.doesNotMatch(block,DISORDER_NAME);assert.doesNotMatch(block,/phq|gad|itq|score|risque|points/i);
 assert.equal(descriptiveLabel('Trouble panique'),null);assert.equal(descriptiveLabel('Difficultés de sommeil (insomnie, cauchemars)'),'Difficultés de sommeil');
 // No published module label names a disorder: every theme can be sent.
 for(const m of SYMPTOMES.modules)assert.ok(descriptiveLabel(m.label),m.label);
 assert.equal(renderSymptomsBlock({checks:[checks[1]],crossing:null,now,data:SYMPTOMES}),null,'nothing above the reference: no block');
});

test('block: at most 1 200 characters with every module above the reference and a full crossing',()=>{
 const now=1794830400,checks=SYMPTOMES.modules.map((m,i)=>({module:m.id,...fullResult(m),result:fullResult(m),completed_at:now-(i*9)*DAY}));
 const deux=Object.keys(SYMPTOMES.subject_modules).map(id=>({id,label:'Sujet assez long pour remplir la liste '+id}));
 const block=renderSymptomsBlock({checks,crossing:{lists:{deux}},safety:{reported_at:now-DAY,kinds:['automutilation','suicide','violences'],ask_now:true},now,data:SYMPTOMES});
 assert.ok(block.length<=SYMPTOMS_BLOCK_MAX,String(block.length));
 assert.match(block,/^« Mes symptômes » :/);assert.match(block,/Sécurité : .*moins de 14 jours/);assert.match(block,/autres thèmes au-dessus du repère, non détaillés/);
 assert.doesNotMatch(block,DISORDER_NAME);
});

test('safety: asked at most once per 14 days, before any exploration; an alert over 14 days is not sent',()=>{
 const now=100*DAY,alert=a=>({created_at:now-a*DAY,kind:'suicide'});
 assert.equal(symptomsSafetyState([alert(15)],[],now),null,'over 14 days');
 assert.deepEqual(symptomsSafetyState([alert(2)],[],now),{reported_at:now-2*DAY,kinds:['suicide'],ask_now:true});
 assert.equal(symptomsSafetyState([alert(2)],[now-DAY],now).ask_now,false,'a reply since the alert already asked');
 assert.equal(symptomsSafetyState([alert(2)],[now-3*DAY],now).ask_now,true,'a reply before the alert did not');
 assert.equal(symptomsSafetyState([alert(20),alert(2)],[now-10*DAY],now).ask_now,false,'asked 10 days ago for an older alert: not again');
 assert.equal(symptomsSafetyState([alert(30),alert(2)],[now-16*DAY],now).ask_now,true,'asked 16 days ago: due again');
 const q=ensureSymptomsSafetyQuestion('Je vous lis.',{ask_now:true,kinds:['suicide']});
 assert.match(q,/^Vous avez indiqué récemment, dans « Mes symptômes », des idées de mort .* êtes-vous en sécurité \? .*3114/);assert.ok(q.endsWith('Je vous lis.'));
 assert.match(ensureSymptomsSafetyQuestion('x',{ask_now:true,kinds:['violences']}),/17 ou le 112/);
 assert.equal(ensureSymptomsSafetyQuestion('En ce moment, êtes-vous en sécurité ? Je suis là.',{ask_now:true,kinds:['suicide']}),'En ce moment, êtes-vous en sécurité ? Je suis là.');
 assert.equal(ensureSymptomsSafetyQuestion('x',{ask_now:false,kinds:['suicide']}),'x');
 assert.equal(ensureSymptomsSafetyQuestion('x',{ask_now:true,kinds:['suicide']},{risk:'explicit'}),'x','a current danger follows the existing safety reply');
 assert.equal(reviewSafetySignal({symptoms_safety:{kinds:['suicide']}}),'self_harm');assert.equal(reviewSafetySignal({symptoms_safety:{kinds:['violences']}}),'violence');
});

test('loader: nothing without consent, for a minor or after withdrawal; the page shows the very same text',async()=>{
 const {setup}=await import('./v4-S-helper.mjs');
 const s=setup();await s.ready();
 const json=(v,status=200)=>new Response(JSON.stringify(v),{status}),backend=createSymptomsBackend({error(status,message,code){throw Object.assign(Error(message),{status,code})},json,body:async r=>r.json(),now:()=>s.now(),random:()=>Math.random().toString(36).slice(2),limit:async()=>{}});
 seedSymptoms(s,{consent:null,full:false});
 assert.equal(await symptomsForAmbre(s.env,'a',s.now()),null,'no consent row');
 seedSymptoms(s,{consent:false,full:false});
 assert.equal(await symptomsForAmbre(s.env,'a',s.now()),null,'consent withdrawn');
 s.raw.prepare('UPDATE symptom_consents SET consent=1').run();
 const out=await symptomsForAmbre(s.env,'a',s.now());assert.match(out.block,/Humeur basse et perte d’intérêt : au-dessus du repère publié/);assert.equal(out.safety,null);
 assert.doesNotMatch(out.block,/phq9|"?2"?,/);
 const page=await backend.readState(s.env,'a');assert.equal(page.ambre.block,out.block,'Paramètres shows exactly what Ambre receives');assert.equal(page.safety.ambre_available,true);
 const p=s.raw.prepare("SELECT self_data FROM profiles WHERE user_id='a'").get();s.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({...JSON.parse(p.self_data),birth_date:'2012-01-01'}));
 assert.equal(await symptomsForAmbre(s.env,'a',s.now()),null,'a minor');
 const other=await backend.readState(s.env,'b');assert.equal(other.ambre.block,null);assert.equal(other.safety.ambre_available,false,'no prerequisites: the Ambre button would not lead to the protocol');
});

test('conversation: the block and its rules reach the writer, the safety question is published once per 14 days',async()=>{
 let s;const run=await replayTurn(MESSAGES[2][1],{strategy:'ideal',digests:true,extend:x=>{s=x;seedSymptoms(x,{full:false,alertAt:-3600})}});
 assert.equal(run.stage,'done');
 const writing=run.get('ambre_writing'),input=JSON.parse(writing.input),analysis=JSON.parse(run.get('ambre_analysis').input);
 assert.match(input.symptoms,/Humeur basse et perte d’intérêt/);assert.deepEqual(input.symptoms_safety,{kinds:['suicide'],ask_now:true});
 assert.ok(writing.system.includes(SYMPTOMS_RULES));assert.match(writing.system,/symptoms_safety\.ask_now=true/);
 assert.equal(analysis.memory.safety_seen,true,'the alert is a recent antecedent for the analysis');
 assert.doesNotMatch(writing.input,/phq9_\d|"answers"/,'no item-by-item answer');
 assert.ok(!run.calls.filter(c=>c.purpose==='ambre_risk_check').some(c=>/symptômes/i.test(c.input)),'the risk check receives no questionnaire result');
 const reply=JSON.parse(s.turn().response).content;assert.match(reply,/^Vous avez indiqué récemment, dans « Mes symptômes ».*êtes-vous en sécurité \?.*3114/s);
 // The published turn counts as asked: the next turn does not ask again.
 s.raw.prepare("UPDATE chat_turns SET created_at=datetime(?,'unixepoch') WHERE id=?").run(s.now(),s.turn().id);
 const next=await symptomsForAmbre(s.env,'a',s.now()+60);assert.equal(next.safety.ask_now,false);assert.match(next.block,/Sécurité :/);
 // Over 14 days: the alert leaves the context.
 const later=await symptomsForAmbre(s.env,'a',s.now()+15*DAY);assert.equal(later.safety,null);assert.doesNotMatch(later.block,/Sécurité/);
});

test('conversation: no block, no rule and no extra cost without consent',async()=>{
 const run=await replayTurn(MESSAGES[0][1],{strategy:'ideal',digests:true,extend:x=>seedSymptoms(x,{consent:false,alertAt:-3600})});
 const writing=run.get('ambre_writing'),input=JSON.parse(writing.input);
 assert.equal(input.symptoms,undefined);assert.equal(input.symptoms_safety,undefined);assert.ok(!writing.system.includes(SYMPTOMS_RULES));
 assert.notEqual(JSON.parse(run.get('ambre_analysis').input).memory.safety_seen,true);
});

// Budget (plan v1.1): the block is taken from the cartography share; every stage stays within
// +10 % of the frozen reference of v5-A5-budget, with the fullest block, an alert and the thread.
const REFERENCE={ideal:{proche:{ambre_analysis:2406,ambre_writing:5953,ambre_review:2460},suite_bilan:{ambre_analysis:2395,ambre_writing:5944,ambre_review:2451},moment:{ambre_analysis:2438,ambre_writing:5937,ambre_review:2444},portrait:{ambre_analysis:2478,ambre_writing:5945,ambre_review:2452}},
 none:{proche:{ambre_analysis:2406,ambre_writing:5939,ambre_review:2451},suite_bilan:{ambre_analysis:2395,ambre_writing:6321,ambre_review:2463},moment:{ambre_analysis:2438,ambre_writing:6066,ambre_review:2494},portrait:{ambre_analysis:2478,ambre_writing:6236,ambre_review:2496}},review:{initial:12423,point:12525}};
const sum=o=>Object.values(o).reduce((a,b)=>a+b,0);
for(const strategy of ['ideal','none'])for(const [label,message] of MESSAGES)test(`budget with a full « Mes symptômes » block ${strategy}/${label}: within +10 %`,async t=>{
 const run=await replayTurn(message,{strategy,digests:true,extend:x=>seedSymptoms(x,{alertAt:-3600})});assert.equal(run.stage,'done');
 const sizes=Object.fromEntries(run.calls.filter(c=>c.purpose!=='ambre_risk_check').map(c=>[c.purpose,c.tokens])),ref=REFERENCE[strategy][label],input=JSON.parse(run.get('ambre_writing').input);
 t.diagnostic(JSON.stringify({sizes,ref,ratio:+(sum(sizes)/sum(ref)).toFixed(3),block:input.symptoms.length}));
 assert.ok(input.symptoms.length>1000&&input.symptoms.length<=SYMPTOMS_BLOCK_MAX,'a full block');
 for(const [purpose,tokens] of Object.entries(ref))assert.ok(sizes[purpose]<=tokens*1.1,`${purpose} ${sizes[purpose]} > ${tokens}*1.1`);
 assert.ok(sum(sizes)<=sum(ref)*1.1);
});
for(const kind of ['initial','point'])test(`review budget with a full « Mes symptômes » block ${kind}: within +10 %, safety antecedent`,async t=>{
 const run=await replayReview(kind,{digests:true,extend:x=>seedSymptoms(x,{alertAt:-3600})});
 t.diagnostic(JSON.stringify({tokens:run.tokens,ref:REFERENCE.review[kind],ratio:+(run.tokens/REFERENCE.review[kind]).toFixed(3),block:run.data.symptoms?.length}));
 assert.ok(run.data.symptoms.length>1000);assert.deepEqual(run.data.symptoms_safety,{kinds:['suicide']});assert.ok(run.system.includes(SYMPTOMS_RULES));
 assert.doesNotMatch(run.user,/phq9_\d/);assert.equal(reviewSafetySignal(run.data),'self_harm');
 assert.ok(run.tokens<=REFERENCE.review[kind]*1.1,`${run.tokens} > ${REFERENCE.review[kind]}*1.1`);
});
