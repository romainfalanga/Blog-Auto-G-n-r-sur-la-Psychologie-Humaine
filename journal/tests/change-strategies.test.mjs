import test from 'node:test';
import assert from 'node:assert/strict';
import {CHANGE_STRATEGIES} from '../src/change-strategies.generated.mjs';
import {selectChangeCandidates,loadStrategyDetails,validateStrategySelection,changeReviewCriteria,
  CHANGE_REVIEW_CRITERIA,CHANGE_SYSTEM_RULES,normalizeChangeVerdict,normalizeChangeReview,
  deterministicChangeReview} from '../src/change-strategies.mjs';

const plan = () => ({available:true,proposals_enabled:true,reminders_paused:false,goals:[],step:null,trial:null,due_trial:null});
const analysis = () => ({risk:'none',posture:'explore',need:'act',states:[],constraints:[],closed_topics:[]});
const ids = result => result.map(item => item.id);
const choose = (message, extra = {}) => selectChangeCandidates({message,plan:plan(),...extra});
const trialPlan = () => ({...plan(),goals:[{id:'goal',title:'Dire non à ma collègue',status:'active'}],
  step:{id:'step',text:'Refuser une demande de ma collègue',strategy_id:'C08',status:'active'},
  trial:{id:'trial',step_id:'step',status:'planned'},
  due_trial:{id:'trial',step_id:'step',review_mode:'on_date',review_on:'2020-01-01',reminder_count:0}});

test('current requests select relevant methods without filling a six-item quota', () => {
  const result=choose('Je voudrais apprendre à dire non à une demande de ma collègue.');
  assert.deepEqual(ids(result),['C08']);
  assert.ok(result.length<=6);assert.ok(JSON.stringify(result).length<=900);
  assert.ok(ids(choose('Je veux retrouver un peu de plaisir avec une activité agréable.')).includes('C04'));
  assert.ok(ids(choose('Je veux être moins dure envers moi.')).includes('K01'));
  assert.deepEqual(choose('Je voudrais agir.'),[]);
});

test('diagnoses, isolated words, a care record and reported requests do not prescribe a method', () => {
  for(const message of ['Borderline.','Je suis dépressif.','TCC, exposition, confiance.','Mon diagnostic est anxiété : un conseil ?',
    'Mon amie a écrit « je veux apprendre à dire non ».','> Je veux apprendre à dire non']) {
    assert.deepEqual(choose(message,{care:{has_psychologist:true,diagnosis:'dépression'}}),[],message);
  }
});

test('listening, refusal and closed subjects override an otherwise actionable request', () => {
  for(const message of ['Je veux juste parler, sans conseil. Je veux apprendre à dire non.',
    'Je ne veux plus en parler. Je voudrais apprendre à dire non.','Je veux apprendre à dire non, mais aucun conseil.']) {
    assert.deepEqual(choose(message),[],message);
  }
  for(const patch of [{need:'listen'},{posture:'listen'},{posture:'stabilize'},{posture:'repair'},
    {constraints:['no_advice']},{closed_topics:['Je refuse ce sujet']}]) {
    assert.deepEqual(choose('Je voudrais apprendre à dire non.',{analysis:{...analysis(),...patch}}),[]);
  }
  assert.deepEqual(choose('Je veux comprendre mon enfance.',{plan:{...plan(),closed_topics:['enfance']}}),[]);
});

test('disabled or unavailable plans never produce strategy selections', () => {
  for(const patch of [{available:false},{proposals_enabled:false},{proposals_enabled:0}]) {
    assert.deepEqual(choose('Je veux retrouver du plaisir.',{plan:{...plan(),...patch}}),[]);
  }
});

test('danger has a conservative lexical backstop and a model-risk backstop', () => {
  assert.deepEqual(choose('Je veux mourir. Aide-moi à trouver une activité agréable.'),[]);
  for(const risk of ['possible','explicit','unknown']) assert.deepEqual(choose('Je veux retrouver du plaisir.',{analysis:{...analysis(),risk}}),[]);
  assert.deepEqual(validateStrategySelection(['C04'],['C04'],{analysis:{...analysis(),posture:'safety'},plan:plan()}),[]);
});

test('current violence or coercion blocks assertiveness even when a phrase is explicitly requested', () => {
  for(const danger of ['Mon conjoint me frappe.','Il contrôle mon téléphone.','Je subis des violences.','Mon partenaire violent me fait peur.']) {
    const message=danger+' Aide-moi à dire non et à formuler une demande.';
    assert.ok(choose(message).every(item=>!['C07','C08','D06'].includes(item.id)),message);
    assert.deepEqual(validateStrategySelection(['C07','C08','D06'],['C07','C08','D06'],{message,plan:plan(),analysis:analysis()}),[]);
  }
  assert.ok(ids(choose('Il y a dix ans mon ex était violent. Je voudrais apprendre à dire non à une collègue.')).includes('C08'));
});

test('C09 is a session-preparation option and never autonomous exposure', () => {
  assert.ok(!ids(choose('Je veux essayer une exposition graduée.')).includes('C09'));
  const message='Je veux préparer ma séance avec mon psychologue pour parler des situations sociales que j’évite.';
  const candidates=choose(message);
  assert.ok(ids(candidates).includes('C09'));
  assert.equal(candidates.find(item=>item.id==='C09').restriction,'prepare_session_only');
  const good={strategy_ids:['C09'],content:'Vous pouvez noter une question pour votre prochaine séance.'};
  assert.deepEqual(deterministicChangeReview({message,plan:plan(),analysis:analysis(),draft:good,candidates}).failed_ids,[]);
  const bad={strategy_ids:['C09'],content:'Allez dans une réunion et affrontez votre peur.'};
  assert.ok(deterministicChangeReview({message,plan:plan(),analysis:analysis(),draft:bad,candidates}).failed_ids.includes('C-R1'));
});

test('the model cannot invent, duplicate or select more than two IDs', () => {
  const candidates=['C04','C05','D04'];
  assert.deepEqual(validateStrategySelection(['FAKE','C04','C04','C05','D04'],candidates,{analysis:analysis(),message:'Je veux agir.',plan:plan()}),['C04','C05']);
  assert.deepEqual(validateStrategySelection(['C08'],candidates,{analysis:analysis(),message:'Je veux dire non.',plan:plan()}),[]);
  assert.deepEqual(validateStrategySelection(['C04','C05','D04'],candidates,{analysis:analysis(),plan:plan()},36),['C04','C05']);
});

test('no_questions removes guided elicitation but permits a requested question-free suggestion', () => {
  const options={analysis:{...analysis(),constraints:['no_questions']},message:'Je veux clarifier mes valeurs sans question.',plan:plan()};
  assert.deepEqual(validateStrategySelection(['A03','C04'],['A03','C04'],options),['C04']);
  assert.equal(loadStrategyDetails(['A03'])[0].requires_question,true);
  assert.equal(loadStrategyDetails(['C04'])[0].requires_question,false);
});

test('a due date alone does not hijack an unrelated conversation', () => {
  assert.deepEqual(choose('Je regarde la pluie ce soir.',{plan:trialPlan()}),[]);
  assert.deepEqual(choose('J’ai essayé une nouvelle recette.',{plan:trialPlan()}),[]);
  assert.deepEqual(choose('Je voudrais choisir une autre couleur de rideaux.',{plan:trialPlan()}),[]);
});

test('an identified trial report is reviewed before a new exercise, including shame and worsening', () => {
  const current=trialPlan();current.trial.outcome='worse';
  assert.deepEqual(ids(choose('Je n’ai pas fait mon exercice et j’ai honte.',{plan:current,analysis:{...analysis(),states:['shame']}})),['G04']);
  assert.deepEqual(ids(choose('J’ai essayé cet exercice et cela a été pire.',{plan:current})),['G04']);
  assert.deepEqual(current.trial,{id:'trial',step_id:'step',status:'planned',outcome:'worse'});
});

test('shortlists are bounded by serialized size, deterministic and free of private content', () => {
  const message='Je veux comprendre ma réaction, clarifier mes valeurs, formuler une demande, dire non, faire une pause, retrouver du plaisir, commencer une tâche et résoudre un problème.';
  const result=choose(message,{plan:{...plan(),private_secret:'NEVER_COPY'}});
  assert.ok(result.length>1&&result.length<=6);assert.ok(JSON.stringify(result).length<=900);
  assert.deepEqual(choose(message),result);assert.doesNotMatch(JSON.stringify(result),/NEVER_COPY/);
});

test('compression preserves complete permissions and exclusions or omits the entire record', () => {
  for(const item of CHANGE_STRATEGIES.strategies) {
    const details=loadStrategyDetails([item.id],{budget:10000});assert.equal(details.length,1);
    assert.deepEqual(details[0].exclusions,item.exclusions);assert.deepEqual(details[0].blocked_contexts,item.blocked_contexts);
    assert.equal(details[0].permission,item.permission);assert.equal(details[0].restriction,item.restriction);
    assert.equal(details[0].requires_safe_interpersonal_context,item.requires_safe_interpersonal_context);
    const length=JSON.stringify(details).length;
    assert.deepEqual(loadStrategyDetails([item.id],{budget:length-1}),[]);
    assert.deepEqual(loadStrategyDetails([item.id],{budget:length}),details);
  }
  assert.deepEqual(loadStrategyDetails(['C08'],{budget:0}),[]);
  assert.ok(loadStrategyDetails(['C08','C04','A03'],{budget:10000}).length<=2);
  assert.ok(JSON.stringify(loadStrategyDetails(['C08','C04'])).length<=1600);
});

test('returned details cannot mutate the shared catalogue', () => {
  const result=loadStrategyDetails(['C08']);result[0].exclusions.length=0;result[0].blocked_contexts.length=0;
  assert.ok(loadStrategyDetails(['C08'])[0].exclusions.length);assert.ok(loadStrategyDetails(['C08'])[0].blocked_contexts.length);
});

test('listening-only replies have eight genuinely inapplicable criteria', () => {
  const criteria=changeReviewCriteria({analysis:{need:'listen'},draft:{content:'Cette journée vous a beaucoup fatigué.'},changeContext:{plan:plan()}});
  assert.equal(criteria.length,8);assert.deepEqual(ids(criteria),CHANGE_REVIEW_CRITERIA.map(item=>item.id));
  assert.ok(criteria.every(item=>!item.applicable));
  const result=normalizeChangeReview(undefined,criteria);
  assert.equal(result.passed,null);assert.equal(result.checked,false);assert.equal(result.applicable_count,0);
  assert.ok(result.criteria.every(item=>item.verdict==='not_applicable'));
});

test('implicit exercises trigger criteria even when the writer omits structured IDs', () => {
  const criteria=changeReviewCriteria({draft:{content:'Vous pourriez noter une phrase moins dure envers vous.'}});
  for(const id of ['C-R1','C-R2','C-R3','C-R4','C-R6'])assert.equal(criteria.find(item=>item.id===id).applicable,true,id);
  assert.equal(criteria.find(item=>item.id==='C-R8').applicable,false);
});

test('trial reports and relevant professional instructions receive their own checks', () => {
  const criteria=changeReviewCriteria({analysis:{change_signal:{kind:'trial_reported'}},draft:{content:'Cet essai a été pénible.'},
    changeContext:{plan:trialPlan(),private_care_instructions:[{text:'Une consigne privée'}]}});
  assert.equal(criteria.find(item=>item.id==='C-R5').applicable,true);
  assert.equal(criteria.find(item=>item.id==='C-R7').applicable,true);
  assert.equal(criteria.find(item=>item.id==='C-R4').applicable,false);
  assert.doesNotMatch(JSON.stringify(criteria),/consigne privée/);
});

test('missing, unknown, duplicate or na verdicts cannot pass applicable checks', () => {
  const criteria=changeReviewCriteria({draft:{content:'Essayez de noter une pensée.'}});
  assert.equal(normalizeChangeVerdict('na',{applicable:true}),'not_checked');
  assert.equal(normalizeChangeVerdict('pass',{applicable:false}),'not_applicable');
  const good=criteria.map(item=>({id:item.id,verdict:item.applicable?'pass':'not_applicable'}));
  assert.equal(normalizeChangeReview(good,criteria).passed,true);
  assert.equal(normalizeChangeReview({change_criteria:good},criteria).passed,true);
  const missing=normalizeChangeReview([],criteria);assert.equal(missing.passed,false);assert.ok(missing.unchecked_ids.length);
  const duplicate=normalizeChangeReview([...good,{id:'C-R1',verdict:'pass'}],criteria);
  assert.ok(duplicate.unchecked_ids.includes('C-R1'));
  assert.equal(normalizeChangeReview(good.map(item=>({...item,verdict:'na'})),criteria).passed,false);
});

test('deterministic review blocks exercises under refusal and invented strategy IDs', () => {
  const draft={content:'Essayez de dire non.',strategy_ids:['C08']};
  const result=deterministicChangeReview({analysis:{...analysis(),constraints:['no_advice']},message:'Aucun conseil.',plan:plan(),draft,candidates:['C08']});
  assert.ok(result.failed_ids.includes('C-R2'));assert.ok(result.failed_ids.includes('C-R1'));
  assert.ok(deterministicChangeReview({draft:{strategy_ids:['FAKE']},plan:plan()}).failed_ids.includes('C-R1'));
});

test('followup checks the final exact invitation, target and stop/budget controls', () => {
  const current=trialPlan(),quote='Comment s’est passé cet essai ?',draft={content:quote,followup:{trial_id:'trial',quote}};
  const args={analysis:analysis(),message:'Je voudrais revenir sur mon essai.',plan:current,draft};
  assert.deepEqual(deterministicChangeReview(args).failed_ids,[]);
  assert.equal(changeReviewCriteria({draft,changeContext:{plan:current}}).find(item=>item.id==='C-R8').applicable,true);
  for(const bad of [{...draft,content:'Nous pouvons parler de la journée.'},{...draft,followup:{trial_id:'other',quote}},
    {...draft,followup:{trial_id:'trial',quote:''}}]) assert.ok(deterministicChangeReview({...args,draft:bad}).failed_ids.includes('C-R8'));
  assert.ok(deterministicChangeReview({...args,plan:{...current,reminders_paused:true}}).failed_ids.includes('C-R8'));
  assert.ok(deterministicChangeReview({...args,plan:{...current,due_trial:{...current.due_trial,reminder_count:2}}}).failed_ids.includes('C-R8'));
  assert.ok(deterministicChangeReview({...args,draft:{content:quote}}).failed_ids.includes('C-R8'));
  assert.match(CHANGE_SYSTEM_RULES,/jamais une exposition autonome/);
});
