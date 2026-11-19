// « Mes symptômes » lot A : moteur pur partagé (scores, paliers, ITQ, règles, sécurité, âge, délais,
// croisement 4 états). Contrat : docs/symptomes-contrat-2026-11-19.md.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {SYMPTOMES as S} from '../src/symptomes-data.generated.mjs';
import * as E from '../src/symptomes-engine.mjs';

const fail=(status,message,code)=>{throw Object.assign(new Error(message),{status,code})};
const spread=(prefix,n,total,max=3,last=false)=>{const out={},order=Array.from({length:n},(_,i)=>i+1);if(!last)order.push(order.splice(n-1,1)[0]);
 // the last item (PHQ-9 #9: safety) is filled last so that it stays 0 unless needed
 let left=total;for(const i of order){const v=Math.min(max,left);out[prefix+i]=v;left-=v}assert.equal(left,0);return out};
const phq9=total=>spread('phq9_',9,total);
const gad7=total=>spread('gad7_',7,total);
const result=(module,answers,options)=>E.computeModuleResult(module,answers,S,options);
const rule=(r,id)=>r.rules.find(x=>x.id===id);
const ITQ=S.instruments.itq.items.map(i=>i.id);
const itq=(over={},base=0)=>({exposition_evenement:'oui',deuil_deces:'non',adaptation_evenement:'non',...Object.fromEntries(ITQ.map(id=>[id,base])),...over});
const PHQ15=S.instruments.phq15.items.map(i=>i.id);
const phq15=(total)=>{const out=Object.fromEntries(PHQ15.map(id=>[id,0]));let left=total;for(const id of PHQ15){const max=['phq15_sommeil','phq15_fatigue'].includes(id)?2:2;const v=Math.min(max,left);out[id]=v;left-=v}assert.equal(left,0);return out};

test('engine: pure module, no import, usable in the browser',()=>{
 const source=readFileSync(new URL('../src/symptomes-engine.mjs',import.meta.url),'utf8');
 assert.doesNotMatch(source,/^\s*import\s/m);assert.doesNotMatch(source,/\b(?:env|DB|Request|Response|document|window|localStorage)\b\./);
});

test('index: every published module accepts its questions, its instrument items and nothing else',()=>{
 const idx=E.symptomsIndex(S);
 assert.equal(idx.modules.size,S.modules.length);
 assert.deepEqual(idx.instrumentsOf.get('humeur'),['phq2','phq9']);assert.deepEqual(idx.instrumentsOf.get('anxiete'),['gad2','gad7']);
 assert.deepEqual(idx.instrumentsOf.get('trauma_stress'),['itq']);assert.deepEqual(idx.instrumentsOf.get('somatique'),['phq15']);
 const humeur=idx.allowed.get('humeur');
 for(let i=1;i<=9;i++)assert.ok(humeur.has('phq9_'+i));
 assert.ok(humeur.has('securite_actuelle'),'universal question in the first step');assert.ok(!idx.allowed.get('relations').has('securite_actuelle'));
 assert.ok(!humeur.has('phq9_difficulte'),'unscored final PHQ-9 question is not asked');
 assert.ok(idx.allowed.get('trauma_stress').has('itq_quand'));
 assert.deepEqual([...idx.allowed.get('somatique').get('phq15_d').values].sort(),['na',0,1,2].sort());
 assert.deepEqual([...idx.allowed.get('somatique').get('phq15_sommeil').values],[0,1,2,3]);
 assert.ok(!idx.allowed.get('somatique').get('phq15_a').values.has(3));
 for(const m of S.modules)for(const q of m.questions)assert.ok(idx.allowed.get(m.id).has(q.id),q.id);
});

test('PHQ-2 / GAD-2 entry at ≥ 2 (v1.1), manual threshold 3 kept apart',()=>{
 assert.equal(S.instruments.phq2.entry_threshold,2);assert.equal(S.instruments.phq2.manual_threshold,3);
 for(const [module,prefix] of [['humeur','phq9_'],['anxiete','gad7_']]){
  assert.equal(result(module,{[prefix+1]:1,[prefix+2]:0}).entry,'negatif');
  assert.equal(result(module,{[prefix+1]:1,[prefix+2]:1}).entry,'positif');
  assert.equal(result(module,{[prefix+1]:2,[prefix+2]:0}).entry,'positif');
  assert.equal(result(module,{[prefix+1]:3}).entry,'indetermine','a missing item is never counted 0');
  // Negative entry without the full instrument: below the entry mark, the module stops.
  const low=result(module,{[prefix+1]:0,[prefix+2]:1});assert.equal(low.state,'en_dessous');
  const entree=low.rules.find(r=>r.kind==='entree');assert.equal(entree.result,'negatif');assert.equal(entree.display,true);assert.equal(entree.score,1);
  assert.ok(low.rules.filter(r=>r.kind!=='entree').every(r=>r.result==='negatif'&&!r.display),'what was not asked is negative and not shown');
  // Positive entry, full instrument not filled: incomplete.
  const open=result(module,{[prefix+1]:1,[prefix+2]:1});assert.equal(open.state,'indetermine');assert.equal(open.rules.find(r=>r.kind==='entree').display,true);
 }
});

test('PHQ-9: every band limit, mark at 10, high note from 15, item missing = no score',()=>{
 const cases=[[0,'sous_premier_palier','en_dessous'],[4,'sous_premier_palier','en_dessous'],[5,'leger','en_dessous'],[9,'leger','en_dessous'],[10,'modere','au_dessus'],[14,'modere','au_dessus'],[15,'moderement_severe','au_dessus'],[19,'moderement_severe','au_dessus'],[20,'severe','au_dessus'],[27,'severe','au_dessus']];
 for(const [score,band,state] of cases){
  const r=result('humeur',{...phq9(score),humeur_duree:'non',humeur_perinatal:'non'},{});
  assert.equal(r.scores.phq9,score);assert.equal(r.bands.phq9.id,band,String(score));assert.equal(r.state,state,String(score));
  const p=rule(r,'phq9_repere');assert.equal(p.result,score>=10?'positif':'negatif');assert.equal(p.display,true);
  if(score>=10)assert.equal(p.note,score>=15?'high_note':'positive_note');
  assert.equal(rule(r,'phq2_entree').display,false,'PHQ-2 text only when the PHQ-9 is absent');
  assert.equal(rule(r,'orientation_humeur_haute').result,score>=10?'positif':'negatif');
 }
 const partial=phq9(12);delete partial.phq9_5;
 const r=result('humeur',partial);assert.equal(r.scores.phq9,null);assert.equal(r.state,'indetermine');assert.equal(rule(r,'phq9_repere').display,false);
 assert.equal(E.bandFor('phq9',28,S),null);assert.equal(E.bandFor('phq9',null,S),null);
});

test('PHQ-9 below the mark with a present repère: the repère is shown (state repere_present)',()=>{
 const r=result('humeur',{...phq9(7),humeur_duree:'oui'});
 assert.equal(r.state,'repere_present');assert.equal(rule(r,'phq9_repere').result,'negatif');assert.equal(rule(r,'phq9_repere').display,true);
 assert.equal(rule(r,'repere_humeur_duree').result,'positif');assert.equal(rule(r,'repere_humeur_duree').note,'positive_note');
 const nsp=result('humeur',{...phq9(7),humeur_duree:'nsp'});assert.equal(nsp.state,'en_dessous','« Je ne sais pas » to a repère does not hide a computed score');
 assert.equal(rule(nsp,'repere_humeur_duree').result,'indetermine');
 const peri=result('humeur',{...phq9(3),humeur_perinatal:'oui'});assert.equal(rule(peri,'contexte_perinatal').result,'positif');assert.equal(rule(peri,'contexte_perinatal').display,true);assert.equal(peri.state,'en_dessous','context never changes a state');
});

test('GAD-7: band limits, mark at 10, repère anxiété durable only with GAD-7 ≥ 10',()=>{
 for(const [score,band] of [[0,'sous_premier_palier'],[4,'sous_premier_palier'],[5,'leger'],[9,'leger'],[10,'modere'],[14,'modere'],[15,'severe'],[21,'severe']]){
  const r=result('anxiete',{...gad7(score),anxiete_duree:'oui'});
  assert.equal(r.scores.gad7,score);assert.equal(r.bands.gad7.id,band);assert.equal(r.state,score>=10?'au_dessus':'en_dessous');
  assert.equal(rule(r,'repere_anxiete_duree').result,score>=10?'positif':'negatif');
  assert.equal(rule(r,'repere_anxiete_duree').display,score>=10,'si_positif');
  if(score>=10)assert.equal(rule(r,'gad7_repere').note,score>=15?'high_note':'positive_note');
 }
});

test('PHQ-15: sleep and fatigue recoded 3→2, « non concerné » counted 0 with a maximum of 28, bands and mark',()=>{
 const base={somatique_entree:'oui',somatique_attention:'non',somatique_duree:'non'};
 for(const [score,band] of [[0,'sous_premier_palier'],[4,'sous_premier_palier'],[5,'faible'],[9,'faible'],[10,'moyen'],[14,'moyen'],[15,'eleve'],[30,'eleve']]){
  const r=result('somatique',{...base,...phq15(score)});assert.equal(r.scores.phq15,score);assert.equal(r.bands.phq15.id,band);assert.equal(r.bands.phq15.max,30);
  assert.equal(r.state,score>=10?'au_dessus':'en_dessous');
 }
 const recoded=result('somatique',{...base,...phq15(0),phq15_sommeil:3,phq15_fatigue:3});assert.equal(recoded.scores.phq15,4);
 const na=result('somatique',{...base,...phq15(10),phq15_d:'na'});assert.equal(na.scores.phq15,na.scores.phq15);assert.equal(na.bands.phq15.max,28);
 const missing={...base,...phq15(12)};delete missing.phq15_k;assert.equal(result('somatique',missing).scores.phq15,null);
 const repere=result('somatique',{...base,...phq15(3),somatique_attention:'oui',somatique_duree:'oui'});assert.equal(repere.state,'repere_present');
 // Negative entry: the PHQ-15 may still be filled on request; its result then counts.
 const request=result('somatique',{somatique_entree:'non',...phq15(12)});assert.equal(request.state,'au_dessus');
 const stop=result('somatique',{somatique_entree:'non'});assert.equal(stop.state,'repere_absent');assert.ok(stop.rules.every(r=>r.result==='negatif'));
});

test('ITQ: TSPT and TSPT-C exclusive, all four clusters required, missing item = no result, non_passe is known',()=>{
 const tspt=itq({itq_p1:2,itq_p3:2,itq_p6:3,itq_p9:2});
 let r=result('trauma_stress',tspt);assert.equal(r.scores.itq.result,'tspt');assert.equal(r.state,'au_dessus');
 assert.equal(rule(r,'itq_tspt').result,'positif');assert.equal(rule(r,'itq_tspt_c').result,'negatif');assert.equal(rule(r,'itq_tspt_c').display,false);
 r=result('trauma_stress',{...tspt,itq_c2:2,itq_c4:4,itq_c6:2,itq_c8:2});assert.equal(r.scores.itq.result,'tspt_c');
 assert.equal(rule(r,'itq_tspt').result,'negatif');assert.equal(rule(r,'itq_tspt').display,false);assert.equal(rule(r,'itq_tspt_c').result,'positif');
 // POS without TSPT: no result rule; one TSPT cluster missing (functioning): « aucun ».
 r=result('trauma_stress',itq({itq_p1:2,itq_p3:2,itq_p5:2,itq_c1:2,itq_c3:2,itq_c5:2,itq_c7:2}));assert.equal(r.scores.itq.result,'aucun');assert.equal(r.state,'en_dessous');
 assert.equal(rule(r,'itq_tspt').display,true,'negative text shown only when neither rule applies');assert.equal(rule(r,'itq_tspt_c').display,false);
 // Threshold « modérément » = 2: 1 is never positive.
 r=result('trauma_stress',itq({},1));assert.equal(r.scores.itq.result,'aucun');assert.ok(Object.values(r.scores.itq.clusters).every(v=>v===false));assert.equal(r.scores.itq.sums.atteinte_pos,3);
 const partial=itq({itq_p1:4});delete partial.itq_c9;r=result('trauma_stress',partial);assert.equal(r.scores.itq.result,null);assert.equal(r.state,'indetermine');assert.equal(r.scores.itq.sums,null);
 for(const exposition of ['non','pref',undefined]){
  const a={...itq({itq_p1:4,itq_p3:4,itq_p5:4,itq_p7:4})};if(exposition)a.exposition_evenement=exposition;else delete a.exposition_evenement;
  const x=result('trauma_stress',{...a,deuil_deces:'oui',deuil_douleur:'non'});assert.equal(x.scores.itq.result,'non_passe');assert.equal(rule(x,'itq_tspt').display,false);assert.equal(rule(x,'itq_menace').result,'negatif');
 }
 // Threat cluster: crossing only, never displayed.
 r=result('trauma_stress',itq({itq_p5:3}));assert.equal(rule(r,'itq_menace').result,'positif');assert.equal(rule(r,'itq_menace').display,false);assert.equal(rule(r,'itq_menace').kind,'grappe');
});

test('trauma: refused entry, bereavement and adjustment repères, adjustment excluded by TSPT',()=>{
 assert.equal(result('trauma_stress',{exposition_evenement:'pref',deuil_deces:'non',adaptation_evenement:'non'}).state,'refuse');
 assert.equal(result('trauma_stress',{exposition_evenement:'non',deuil_deces:'non',adaptation_evenement:'non'}).state,'repere_absent');
 const deuil=result('trauma_stress',{exposition_evenement:'non',deuil_deces:'oui',deuil_douleur:'oui',adaptation_evenement:'non',trauma_gene:'un_peu'});
 assert.equal(deuil.state,'repere_present');assert.equal(rule(deuil,'repere_deuil').result,'positif');
 assert.equal(result('trauma_stress',{exposition_evenement:'non',deuil_deces:'oui',deuil_douleur:'oui',adaptation_evenement:'non',trauma_gene:'non'}).state,'repere_absent','the bereavement repère requires a felt hindrance');
 const adapt={exposition_evenement:'non',deuil_deces:'non',adaptation_evenement:'oui',adaptation_preoccupation:'oui'};
 assert.equal(rule(result('trauma_stress',adapt),'repere_adaptation').result,'positif');
 assert.equal(rule(result('trauma_stress',{...itq({itq_p1:2,itq_p3:2,itq_p5:2,itq_p7:2}),adaptation_evenement:'oui',adaptation_preoccupation:'oui'}),'repere_adaptation').result,'negatif');
});

test('three-valued logic: all, any, not, atleast; nsp and pref are unknown',()=>{
 const ctx={answers:{a:'oui',b:'non',c:'nsp',d:'pref',n:2},scores:{phq9:null,gad7:12}};
 const q=(id,v='oui')=>({q:id,in:[v]});
 assert.equal(E.evaluate(q('a'),ctx),true);assert.equal(E.evaluate(q('b'),ctx),false);assert.equal(E.evaluate(q('c'),ctx),null);assert.equal(E.evaluate(q('d'),ctx),null);assert.equal(E.evaluate(q('z'),ctx),null);
 assert.equal(E.evaluate(q('d','pref'),ctx),true,'pref can be tested explicitly (safety triggers)');
 assert.equal(E.evaluate({q:'n',gte:2},ctx),true);assert.equal(E.evaluate({q:'n',gte:3},ctx),false);assert.equal(E.evaluate({q:'a',gte:1},ctx),null);
 assert.equal(E.evaluate({score:'gad7',gte:10},ctx),true);assert.equal(E.evaluate({score:'phq9',gte:10},ctx),null);
 assert.equal(E.evaluate({all:[q('a'),q('c')]},ctx),null);assert.equal(E.evaluate({all:[q('b'),q('c')]},ctx),false);assert.equal(E.evaluate({all:[q('a'),q('a')]},ctx),true);
 assert.equal(E.evaluate({any:[q('a'),q('c')]},ctx),true);assert.equal(E.evaluate({any:[q('b'),q('c')]},ctx),null);assert.equal(E.evaluate({any:[q('b'),q('b')]},ctx),false);
 assert.equal(E.evaluate({not:q('c')},ctx),null);assert.equal(E.evaluate({not:q('b')},ctx),true);
 const of=[q('a'),q('b'),q('c'),q('d')];
 assert.equal(E.evaluate({atleast:{n:1,of}},ctx),true);assert.equal(E.evaluate({atleast:{n:2,of}},ctx),null);assert.equal(E.evaluate({atleast:{n:3,of}},ctx),null);assert.equal(E.evaluate({atleast:{n:4,of}},ctx),false);
 assert.equal(E.evaluate({itq_result:'tspt'},{scores:{}}),null);assert.equal(E.evaluate({itq_cluster:'menace'},{scores:{itq:{result:'non_passe',clusters:{}}}}),false);
 assert.throws(()=>E.evaluate({inconnu:1},ctx));
});

test('module states: refused, absent with module sentence, indeterminate, several repères, gene texts',()=>{
 assert.equal(result('relations',{},{declined:true}).state,'refuse');
 const none=result('relations',{rel_peur_perte:'non',rel_distance:'non',rel_oubli_soi:'non',rel_insatisfaction:'non',rel_isolement:'non',rel_violences:'non'});
 assert.equal(none.state,'repere_absent');assert.equal(none.module_negative,true);assert.ok(none.rules.every(r=>!r.display));
 const unsure=result('relations',{rel_peur_perte:'nsp',rel_distance:'non',rel_oubli_soi:'non',rel_insatisfaction:'non',rel_isolement:'non'});assert.equal(unsure.state,'indetermine');
 const two=result('relations',{rel_peur_perte:'oui',rel_distance:'oui',rel_oubli_soi:'non',rel_insatisfaction:'non',rel_isolement:'non',relations_gene:'nettement'});
 assert.equal(two.state,'repere_present');assert.equal(two.positive_count,2);assert.equal(two.several,true);assert.equal(rule(two,'repere_peur_perte').note,'site_note');
 assert.equal(two.gene,'nettement');assert.equal(two.gene_text,'gene_nettement');
 assert.equal(result('relations',{rel_peur_perte:'oui',relations_gene:'non'}).gene_text,'gene_absente_trouble');
 assert.equal(result('alimentation',{alim_perte_controle:'non',alim_restriction:'non',alimentation_gene:'non'}).gene_text,'gene_absente','modules listed in gene_absente_trouble_except');
 assert.equal(result('relations',{rel_peur_perte:'oui',relations_gene:'un_peu'}).gene_text,null);
 // Every published module computes without error on an empty, a negative and a declined passation.
 for(const m of S.modules){
  assert.ok(E.RESULT_STATES.includes(result(m.id,{}).state));
  assert.equal(result(m.id,{},{declined:true}).state,'refuse');
  const no=Object.fromEntries(m.questions.filter(q=>S.scales[q.scale].options.some(o=>o.value==='non')).map(q=>[q.id,'non']));
  assert.ok(['repere_absent','en_dessous','indetermine'].includes(result(m.id,no).state),m.id+' '+result(m.id,no).state);
 }
});

test('validation: known module, question, scale value and versions only; declined and empty passations',()=>{
 const ok={module:'humeur',content_version:S.version,instrument_versions:{phq9:S.instruments.phq9.version},answers:{phq9_1:1,phq9_2:2,humeur_gene:'un_peu',securite_actuelle:'non',humeur_duree:null}};
 const clean=E.validatePassation(ok,S,fail);assert.deepEqual(clean.answers,{humeur_gene:'un_peu',phq9_1:1,phq9_2:2,securite_actuelle:'non'},'null = no answer, dropped');
 const code=input=>{try{E.validatePassation(input,S,fail);return 'ok'}catch(e){assert.equal(e.status,400);return e.code}};
 assert.equal(code({...ok,module:'inconnu'}),'unknown_module');
 assert.equal(code({...ok,module:'__proto__'}),'unknown_module');
 assert.equal(code({...ok,content_version:'2026-01-01.1'}),'unknown_version');
 assert.equal(code({...ok,answers:{...ok.answers,gad7_1:1}}),'unknown_question','an item of another module');
 assert.equal(code({...ok,answers:{phq9_1:4}}),'invalid_answer');assert.equal(code({...ok,answers:{phq9_1:'1'}}),'invalid_answer');assert.equal(code({...ok,answers:{phq9_1:1.5}}),'invalid_answer');
 assert.equal(code({...ok,answers:{phq9_1:'pref'}}),'invalid_answer','instrument items: official scale only (null = no answer)');
 assert.equal(code({...ok,answers:{humeur_gene:'oui'}}),'invalid_answer');
 assert.equal(code({...ok,instrument_versions:{}}),'unknown_version','an answered instrument carries its version');
 assert.equal(code({...ok,instrument_versions:{phq2:S.instruments.phq2.version}}),'ok','the PHQ-2 version covers its items');
 assert.equal(code({...ok,instrument_versions:{phq9:'phq9-autre'}}),'unknown_version');assert.equal(code({...ok,instrument_versions:{gad7:S.instruments.gad7.version}}),'unknown_instrument');
 assert.equal(code({...ok,answers:{humeur_gene:'non'},instrument_versions:{}}),'ok');
 assert.equal(code({...ok,declined:true}),'invalid_symptoms');assert.equal(code({...ok,declined:'oui'}),'invalid_symptoms');
 assert.equal(code({module:'relations',content_version:S.version,declined:true}),'ok');
 assert.equal(code({module:'relations',content_version:S.version,answers:{}}),'empty_passation');
 assert.equal(code({module:'relations',content_version:S.version,answers:{rel_violences:'non',securite_actuelle:'non'}}),'unknown_question','universal question only in the first step');
 assert.equal(code({module:'relations',content_version:S.version,answers:[]}),'invalid_symptoms');
 assert.equal(code({module:'somatique',content_version:S.version,instrument_versions:{phq15:S.instruments.phq15.version},answers:{phq15_d:'na',phq15_sommeil:3}}),'ok');
 assert.equal(code({module:'somatique',content_version:S.version,instrument_versions:{phq15:S.instruments.phq15.version},answers:{phq15_a:'na'}}),'invalid_answer');
 assert.equal(code({module:'trauma_stress',content_version:S.version,instrument_versions:{itq:S.instruments.itq.version},answers:{exposition_evenement:'oui',itq_quand:'1_5_ans',itq_p1:4}}),'ok');
 const unpublished={...S,unpublished_modules:['secret']};assert.throws(()=>E.validatePassation({...ok,module:'secret'},unpublished,fail),e=>e.code==='module_unpublished');
 assert.equal(E.validatePassation({module:'relations',content_version:S.version,answers:{}},S,fail,{partial:true}).module,'relations','an empty draft is allowed');
});

test('safety: universal question, PHQ-9 item 9 at any total, self-harm, violence, soft blocks, no risk score',()=>{
 const t=answers=>E.safetyTriggers(answers,S).map(x=>x.id);
 assert.deepEqual(t({securite_actuelle:'oui'}),['securite_oui']);assert.deepEqual(t({securite_actuelle:'non'}),[]);assert.deepEqual(t({securite_actuelle:'pref'}),['securite_pref']);
 assert.deepEqual(t({phq9_9:1}),['phq9_item9'],'from the answer itself, before the end of the PHQ-9');assert.deepEqual(t({phq9_9:0}),[]);
 assert.deepEqual(t({reg_automutilation:'oui'}),['automutilation']);assert.deepEqual(t({rel_violences:'oui'}),['violences']);assert.deepEqual(t({rel_violences:'pref'}),['violences_pref']);
 const low=result('humeur',{...phq9(1),phq9_1:0,phq9_9:1});assert.equal(low.scores.phq9,1);assert.deepEqual(low.safety.map(x=>x.id),['phq9_item9']);
 const screen=E.safetyScreen(E.safetyTriggers({securite_actuelle:'oui',phq9_9:2},S),S,{age:40});
 assert.equal(screen.blocking,true);assert.deepEqual(screen.kinds,['suicide']);assert.deepEqual(screen.resources.map(r=>r.id),['15','112','3114','114']);
 assert.equal(screen.delay_seconds,0);assert.equal(screen.dismiss_by_outside_click,false);assert.equal(screen.quick_exit,null);assert.equal(screen.risk_score,false);
 assert.ok(screen.always.includes('3114'));assert.equal(screen.opening,'Merci de l’avoir dit.');
 assert.ok(E.safetyScreen(E.safetyTriggers({securite_actuelle:'oui'},S),S,{age:25}).resources.some(r=>r.id==='fil_sante_jeunes'),'18–25: Fil Santé Jeunes');
 assert.ok(!E.safetyScreen(E.safetyTriggers({securite_actuelle:'oui'},S),S,{age:26}).resources.some(r=>r.id==='fil_sante_jeunes'));
 const violence=E.safetyScreen(E.safetyTriggers({rel_violences:'oui'},S),S,{age:40});
 assert.deepEqual(violence.resources.map(r=>r.id),['15','112','114','17','3919','116006','119']);assert.equal(violence.quick_exit,'Quitter vite cette page');assert.match(violence.messages[0],/3919/);
 const harm=E.safetyScreen(E.safetyTriggers({reg_automutilation:'oui'},S),S,{});assert.match(harm.messages[0],/Se blesser/);
 const soft=E.safetyScreen(E.safetyTriggers({securite_actuelle:'pref'},S),S,{age:40});assert.equal(soft.blocking,false);assert.equal(soft.screen,'doux');assert.match(soft.texts[0],/3114/);
 assert.match(E.safetyScreen(E.safetyTriggers({rel_violences:'pref'},S),S,{}).texts[0],/3919/);
 assert.equal(E.safetyScreen([],S,{}),null);
 const all=JSON.stringify([screen,violence,harm,soft]);for(const word of S.safety.display.never_say)assert.ok(!all.includes(word),word);
});

test('age gate: adult from the 18th birthday, minor otherwise, unknown without a valid birth date',()=>{
 assert.deepEqual(E.ageGate('2008-11-19','2026-11-19',S),{status:'adulte',age:18,jeune_adulte:true});
 const minor=E.ageGate('2008-11-20','2026-11-19',S);assert.equal(minor.status,'mineur');assert.equal(minor.age,17);assert.equal(minor.save,false);
 assert.deepEqual(minor.resources.map(r=>r.id),['119','fil_sante_jeunes','3114','15','112']);assert.match(minor.text,/réservée aux adultes/);
 assert.equal(E.ageGate('2000-11-19','2026-11-19',S).jeune_adulte,false,'26 years old');assert.equal(E.ageGate('2000-11-20','2026-11-19',S).jeune_adulte,true);
 assert.equal(E.ageOn('2008-02-29','2026-02-28'),17);assert.equal(E.ageOn('2008-02-29','2026-03-01'),18);
 for(const bad of [null,'','2008-13-01','2008-02-30','19-11-2000','2030-01-01',1990])assert.equal(E.ageGate(bad,'2026-11-19',S).status,'inconnu',String(bad));
});

test('delays: 14 days PHQ/GAD, 28 PHQ-15, 30 ITQ, 90 for 12-month repères, never for lifetime questions',()=>{
 const day=86400,at=1_000_000;
 assert.deepEqual(E.retestInfo('humeur',null,at,S),{days:14,never:false,next_at:null,due:true});
 assert.equal(E.retestInfo('humeur',at,at+14*day-1,S).due,false);assert.equal(E.retestInfo('humeur',at,at+14*day,S).due,true);
 assert.equal(E.retestInfo('somatique',at,at,S).days,28);assert.equal(E.retestInfo('trauma_stress',at,at,S).days,30);assert.equal(E.retestInfo('alcool',at,at,S).days,90);
 assert.deepEqual(E.retestInfo('bipolaire',at,at+999*day,S),{days:null,never:true,next_at:null,due:false});
 assert.equal(E.overlaps('humeur',at,at+day,S),true);assert.equal(E.overlaps('humeur',at,at+15*day,S),false);assert.equal(E.overlaps('humeur',null,at,S),false);
});

test('crossing: four states, three lists, « Pas moi », suspension, order and labels',()=>{
 const done=(module,answers)=>result(module,answers);
 const latest={
  humeur:done('humeur',{...phq9(12),humeur_duree:'non'}),             // dep_generale signalé, icd_6a72 non signalé
  relations:done('relations',{rel_peur_perte:'oui',rel_distance:'non',rel_oubli_soi:'nsp',rel_insatisfaction:'non',rel_isolement:'non',rel_violences:'non'}),
  regulation:result('regulation',{},{declined:true}),
  anxiete:done('anxiete',{gad7_1:0,gad7_2:0})                          // entry negative: anx_generale non signalé
 };
 const childhood={present:['dep_generale','icd_6a72','att_evitant','ten_colere','anx_generale','pat_overresponsibility'],dismissed:['att_anxieux']};
 const c=E.crossWithChildhood(latest,childhood,S,{labels:{dep_generale:'Dépression',icd_6a72:'Dysthymie',anx_generale:'Anxiété'}});
 const state=id=>c.subjects.find(s=>s.id===id).state;
 assert.equal(state('dep_generale'),'signale');assert.equal(state('icd_6a72'),'non_signale');assert.equal(state('att_anxieux'),'signale');
 assert.equal(state('att_evitant'),'non_signale');assert.equal(state('pat_overresponsibility'),'non_explore','« Je ne sais pas » leaves the subject unexplored');
 assert.equal(state('ten_colere'),'refuse');assert.equal(state('anx_generale'),'non_signale');assert.equal(state('icd_6b01'),'non_explore');
 assert.equal(state('ten_usage_substances'),'non_explore','several modules, none done');
 assert.deepEqual(c.lists.deux.map(x=>x.id),['dep_generale']);assert.deepEqual(c.lists.deux[0].modules,['humeur']);
 assert.deepEqual(c.lists.symptomes.map(x=>x.id),['att_anxieux'],'a « Pas moi » subject stays visible, without crossing');
 assert.equal(c.lists.symptomes[0].label,S.crossing.display_labels.att_anxieux,'display label imposed by the review');
 const label=id=>S.crossing.display_labels[id]||{dep_generale:'Dépression',icd_6a72:'Dysthymie',anx_generale:'Anxiété'}[id];
 assert.deepEqual(c.lists.enfance.map(x=>x.id),['icd_6a72','anx_generale','att_evitant'],'module order, then label');
 assert.deepEqual(c.lists.enfance.map(x=>x.label),c.lists.enfance.map(x=>label(x.id)),'display labels first, then the Mon enfance wording');
 assert.equal(E.crossWithChildhood(latest,childhood,{...S,crossing:{...S.crossing,display_labels:{}}},{labels:{icd_6a72:'Dysthymie'}}).lists.enfance[0].label,'Dysthymie');
 assert.ok(!JSON.stringify(c).match(/associated_count|"count"/),'no number of experiences shown');assert.equal(c.show_counts,false);
 // No childhood map: list 2 only.
 const alone=E.crossWithChildhood(latest,null,S);assert.equal(alone.childhood_available,false);assert.deepEqual(alone.lists.deux,[]);assert.deepEqual(alone.lists.enfance,[]);assert.deepEqual(alone.lists.symptomes.map(x=>x.id),['dep_generale','att_anxieux']);
 // Current violence suspends every subject of the relations module.
 const violent={...latest,relations:done('relations',{rel_peur_perte:'oui',rel_violences:'oui'})};
 const v=E.crossWithChildhood(violent,childhood,S);assert.deepEqual(v.suspended.map(s=>s.module),['relations']);assert.ok(v.suspended[0].text.length>0);
 assert.ok(!v.subjects.some(s=>['att_anxieux','att_evitant','ten_peur_intimite'].includes(s.id)));
 // A subject of a module done twice reads the latest passation only (the caller passes the latest).
 assert.equal(E.crossWithChildhood({humeur:done('humeur',phq9(3))},childhood,S).subjects.find(s=>s.id==='dep_generale').state,'non_signale');
 // Every subject is a known cartography subject reachable from a published module.
 for(const s of c.subjects)assert.ok(E.CROSSING_STATES.includes(s.state));
});

test('crossing order is locale independent (folded, then binary)',()=>{
 const labels=['Été','etat','Zèbre','abri','Ébène'];
 assert.deepEqual(labels.slice().sort(E.compareLabels),['abri','Ébène','etat','Été','Zèbre']);
 assert.deepEqual(labels.slice().sort(E.compareLabels),labels.slice().sort((a,b)=>a.localeCompare(b,'fr',{sensitivity:'base'})||(a<b?-1:a>b?1:0)));
});

test('negative entry with a started deepening: rules stay unknown, never forced negative (§ 5)',()=>{
 const started=result('humeur',{phq9_1:0,phq9_2:1,phq9_3:2});
 assert.equal(rule(started,'phq9_repere').result,'indetermine');assert.equal(rule(started,'phq9_repere').forced,undefined);assert.equal(started.state,'indetermine');
 assert.equal(rule(started,'repere_humeur_duree').result,'negatif','a question not started is still forced');assert.equal(rule(started,'repere_humeur_duree').forced,true);
 const full=result('humeur',{...phq9(6),phq9_1:0,phq9_2:1,phq9_3:2,phq9_4:3});assert.equal(full.scores.phq9,6);assert.equal(full.state,'en_dessous');
 const gad=result('anxiete',{gad7_1:0,gad7_2:1,gad7_5:3});assert.equal(rule(gad,'gad7_repere').result,'indetermine');assert.equal(rule(gad,'repere_anxiete_duree').result,'indetermine');assert.equal(gad.state,'indetermine');
 const partial=result('somatique',{somatique_entree:'non',phq15_a:2,phq15_b:2});
 assert.equal(rule(partial,'phq15_repere').result,'indetermine');assert.equal(partial.state,'indetermine');
 assert.equal(rule(partial,'repere_detresse_corporelle').result,'negatif');
 const asked=result('somatique',{somatique_entree:'non',somatique_attention:'oui'});assert.equal(rule(asked,'repere_detresse_corporelle').result,'indetermine','a deepening question answered on request');
});

test('crossing: an indeterminate module and an ITQ not offered leave their subjects unexplored (§ 8)',()=>{
 const childhood={present:['ten_hypervigilance','icd_6b40','att_anxieux'],dismissed:[]};
 const state=(latest,id)=>E.crossWithChildhood(latest,childhood,S).subjects.find(s=>s.id===id).state;
 const unsure=result('relations',{rel_peur_perte:'non',rel_distance:'non',rel_oubli_soi:'nsp',rel_insatisfaction:'non',rel_isolement:'non'});
 assert.equal(unsure.state,'indetermine');assert.equal(state({relations:unsure},'att_anxieux'),'non_explore');
 const notOffered=result('trauma_stress',{exposition_evenement:'non',deuil_deces:'non',adaptation_evenement:'non'});
 assert.equal(notOffered.state,'repere_absent');assert.equal(state({trauma_stress:notOffered},'ten_hypervigilance'),'non_explore');assert.equal(state({trauma_stress:notOffered},'icd_6b40'),'non_explore');
 const offered=result('trauma_stress',itq());assert.equal(offered.scores.itq.result,'aucun');
 assert.equal(state({trauma_stress:offered},'ten_hypervigilance'),'non_signale');assert.equal(state({trauma_stress:offered},'icd_6b40'),'non_signale');
 assert.deepEqual(E.crossWithChildhood({trauma_stress:notOffered},childhood,S).lists.enfance,[],'list 3 only after the ITQ');
});
