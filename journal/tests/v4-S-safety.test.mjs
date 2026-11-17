import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
// Lot S: F17 (resources on every refused send), F18 (cautious path when the
// risk check fails or paraphrases), F19 (dedicated check when the analysis is
// down, independent reviewer criterion), F20 (quotes: negation, minimal length,
// targets, values carried by the quote).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,analysis,draft,review,response,lastReply,CARTOGRAPHIE,parisWeek} from './v4-S-helper.mjs';
import {anchorQuote,locateQuote,approximateQuote,negatedQuote,quoteSupport,cartographyNoteCandidates,similarQuotes,substantialQuote} from '../src/quote-guard.mjs';
import {prepareChangeDraft} from '../src/change-engine.mjs';
import {checkedRisk} from '../src/ambre-engine.mjs';
import {cautiousReply} from '../src/chat-safety.mjs';
const crisis='Je veux mourir ce soir.';
const hasResources=r=>/3114/.test(r.error||r.message||'');

test('F17: a danger signal receives the resources on every refusal of send',async t=>{
 provider(t);
 {const s=setup();await s.ready();s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,idempotency_key,lease_token,lease_until,created_at,attempts) VALUES('rv','a','point','running','point_key_0001','x',?,0,1)").run(s.now()+600);
  const r=await s.send(crisis,'crisis_review_01');assert.equal(r.status,409);assert.equal(r.code,'review_in_progress');assert.ok(hasResources(r));assert.equal(r.safety,true);assert.match(r.safety_reply,/3114/)}
 {const s=setup();seedCompanionPrerequisites(s.raw);await s.call('/api/chat/consent','PUT',{accepted:true});const r=await s.send(crisis,'crisis_journey_01');assert.equal(r.status,403);assert.equal(r.code,'journey_not_started');assert.ok(hasResources(r));assert.equal(r.safety,true)}
 {const s=setup();await s.ready();s.chat.engine.work=async()=>{};await s.send('Une question ordinaire sur ma semaine.','busy_turn_0001','a',{waitUntil(){}});
  const r=await s.send(crisis,'crisis_busy_0001');assert.equal(r.status,409);assert.equal(r.code,'request_in_progress');assert.ok(hasResources(r));assert.equal(r.safety,true)}
 {const s=setup();await s.ready();const week=parisWeek(new Date(s.now()*1000)).week_start,ins=s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,quota_period,source_version,consent_revision,content,response,created_at) VALUES(?,'a',?,'h','completed','consumed',?,'2026-11',1,1,'x','{}','2020-01-01 00:00:00')");for(let i=0;i<100;i++)ins.run('q'+i,'q'+i,week);
  const r=await s.send(crisis,'crisis_quota_01');assert.equal(r.status,429);assert.ok(hasResources(r));assert.equal(r.safety,true)}
 // Violence keeps its own numbers; an ordinary refused message carries nothing extra.
 {const s=setup();seedCompanionPrerequisites(s.raw);await s.call('/api/chat/consent','PUT',{accepted:true});const r=await s.send('Il menace de me tuer ce soir.','violence_journey_1');assert.match(r.error,/17 ou le 112/);assert.doesNotMatch(r.error,/3114/)}
 {const s=setup();seedCompanionPrerequisites(s.raw);await s.call('/api/chat/consent','PUT',{accepted:true});const r=await s.send('Une question ordinaire.','plain_journey_01');assert.equal(r.status,403);assert.equal(r.safety,undefined)}
});

test('F17: the interface keeps a refused safety message readable line by line',async()=>{
 const {CHAT_INTERFACE_SCRIPT}=await import('../src/chat-interface.mjs');
 assert.match(CHAT_INTERFACE_SCRIPT,/box\.safety=/);assert.match(CHAT_INTERFACE_SCRIPT,/chat-safety-error/);assert.match(CHAT_INTERFACE_SCRIPT,/pre-line/);
});

test('F18: a paraphrased or retyped anchor confirms the risk instead of failing',()=>{
 const message='J’ai pris mes dispositions pour que chacun soit tranquille. Ce soir, tout sera terminé.';
 const typed=checkedRisk(JSON.stringify({risk:'possible',category:'indirect_warning',evidence_quote:"ce soir, tout sera termine"}),{message});assert.equal(typed.evidence_quote,'Ce soir, tout sera terminé');
 const paraphrased=checkedRisk(JSON.stringify({risk:'possible',category:'indirect_warning',evidence_quote:'pris des dispositions pour que chacun soit tranquille'}),{message});assert.ok(message.includes(paraphrased.evidence_quote));
 assert.throws(()=>checkedRisk(JSON.stringify({risk:'possible',category:'self_harm',evidence_quote:'Une phrase de l’historique ancien'}),{message}),{code:'risk_check_unanchored'});
});

test('F18: a refused or exhausted risk check publishes the cautious reply at once, without a bare failure',async t=>{
 for(const failure of ['429','turn_budget']){
  const s=setup();await s.ready();let checks=0;
  const calls=provider(t,p=>{if(p==='ambre_analysis'){if(failure==='turn_budget')s.raw.prepare('UPDATE chat_engine_jobs SET reserved_tokens=29000').run();return analysis({risk:'possible',posture:'safety'})}if(p==='ambre_risk_check'){checks++;return new Response('{}',{status:429,headers:{'Retry-After':'120'}})}});
  const sent=await s.send('Une question de diagnostic.','cautious_'+failure);await s.settle(2);
  assert.equal(s.job().stage,'done',failure);const reply=lastReply(s);assert.equal(reply,cautiousReply('self_harm'));assert.match(reply,/3114/);assert.doesNotMatch(reply,/pensez au suicide ou risquez/);
  assert.equal(calls.some(x=>x.purpose==='ambre_writing'||x.purpose==='ambre_review'),false);assert.ok(checks<=1);
  assert.equal(JSON.parse(s.job().analysis).risk_check.status,'unavailable');assert.equal(sent.status<500,true);
 }
});

test('F19: an analysis outage runs the dedicated risk check before any writing',async t=>{
 // Hint + check confirmed: safety path with resources.
 {const s=setup();await s.ready();const calls=provider(t,p=>p==='ambre_analysis'?new Response('{}',{status:500}):p==='ambre_risk_check'?{risk:'possible',category:'indirect_warning',evidence_quote:'je pense souvent à la mort'}:undefined);
  await s.send('Ces temps-ci je pense souvent à la mort.','outage_confirmed');await s.settle(2);
  assert.deepEqual(calls.map(x=>x.purpose).slice(0,2),['ambre_analysis','ambre_risk_check']);assert.equal(JSON.parse(s.job().analysis).risk,'possible');assert.match(lastReply(s),/3114/)}
 // Hint + check failing as well: cautious reply with resources.
 {const s=setup();await s.ready();provider(t,p=>p==='ambre_analysis'?new Response('{}',{status:500}):p==='ambre_risk_check'?new Response('{}',{status:502}):undefined);
  await s.send('Ces temps-ci je pense souvent à la mort.','outage_cautious');await s.settle(4);
  assert.equal(s.job().stage,'done');assert.equal(lastReply(s),cautiousReply('self_harm'))}
 // No lexical cue + check failing: the ordinary path, never an invented crisis.
 {const s=setup();await s.ready();const calls=provider(t,p=>p==='ambre_analysis'?new Response('{}',{status:500}):p==='ambre_risk_check'?new Response('{}',{status:502}):undefined);
  await s.send('Je voudrais organiser ma semaine autrement.','outage_plain');await s.settle(4);
  assert.equal(s.job().stage,'done');assert.doesNotMatch(lastReply(s),/3114/);assert.ok(calls.some(x=>x.purpose==='ambre_writing'));
  const a=JSON.parse(s.job().analysis);assert.equal(a.risk,'none');assert.equal(a.risk_check.status,'unavailable')}
 // No cue + check answering none: ordinary path.
 {const s=setup();await s.ready();const calls=provider(t,p=>p==='ambre_analysis'?new Response('{}',{status:500}):undefined);
  await s.send('Je voudrais organiser ma semaine autrement.','outage_checked');await s.settle(2);
  assert.equal(calls.filter(x=>x.purpose==='ambre_risk_check').length,1);assert.equal(JSON.parse(s.job().analysis).risk_check.status,'rejected')}
});

test('F19: the reviewer may flag an unhandled danger; the fallback then carries resources',async t=>{
 const s=setup();await s.ready();const calls=provider(t,p=>{if(p==='ambre_review'){const r=review();r.criteria[0]='no';return r}});
 await s.send('Je voudrais parler de ma semaine.','reviewer_flag_01');await s.settle(2);
 assert.equal(calls.filter(x=>x.purpose==='ambre_writing').length,2);assert.match(lastReply(s),/3114/);
 assert.match(calls.find(x=>x.purpose==='ambre_review').system,/danger actuel non traité/);
});

test('F20: quotes are located up to typography, never negated and long enough',()=>{
 assert.equal(locateQuote('Je n’arrive plus à dormir depuis l’accident.',"je n'arrive plus à dormir depuis l'accident").text,'Je n’arrive plus à dormir depuis l’accident');
 const m1='Je ne dirais pas que mon père était violent, il était surtout absent.';
 assert.equal(anchorQuote(m1,'mon père était violent').reason,'negated');assert.equal(anchorQuote(m1,'il était surtout absent').ok,true);
 assert.equal(anchorQuote('Je ne veux surtout pas arrêter de boire, ce n’est pas le sujet.','arrêter de boire').reason,'negated');
 assert.equal(anchorQuote('Bonjour, je voulais juste reprendre là où nous en étions.','e').ok,false);
 assert.equal(anchorQuote('Bonjour, je voulais juste reprendre.','juste').reason,'too_short');
 assert.equal(anchorQuote('Je bois. Je veux arrêter.','Je bois').ok,true,'a whole short sentence is a complete proposition');
 assert.equal(quoteSupport('Je voudrais arrêter de boire pour mes enfants.','Je voudrais arrêter de boire pour mes enfants','Arrêter de boire'),'supported');
 assert.equal(quoteSupport('Je ne veux surtout pas arrêter de boire.','Je ne veux surtout pas arrêter de boire','Arrêter de boire'),'negated');
 assert.equal(quoteSupport('Je suis architecte et mon genre est non binaire.','Je suis architecte et mon genre est non binaire','non binaire'),'supported');
 assert.equal(quoteSupport('Bonjour, je voulais juste reprendre là où nous en étions.','Bonjour, je voulais juste reprendre là où nous en étions','Traverse un épisode dépressif sévère'),'absent');
 assert.ok(similarQuotes('Ma voisine Anne est un appui précieux, semaine 3','Ma voisine Anne est un appui précieux, semaine 4'));
 assert.ok(approximateQuote('J’ai rangé mes affaires et écrit une lettre à chacun.','écrit des lettres à chacun'));
});

test('F20: cartography notes need a supplied target, a non-negated quote and never a "no" answer',()=>{
 const negative=CARTOGRAPHIE.factors[0].id,positive=CARTOGRAPHIE.factors[1].id,map={explicit_negative:[{id:negative}],dismissed:[{id:'o_dismissed'}]};
 const message='Je ne dirais pas que mon père était violent, il était surtout absent et je me sentais seul.';
 assert.deepEqual(cartographyNoteCandidates([{target:'factor',target_id:negative,quote:'il était surtout absent'}],message,map,[{target:'factor',target_id:negative}]),[]);
 assert.deepEqual(cartographyNoteCandidates([{target:'factor',target_id:positive,quote:'mon père était violent'}],message,map,[{target:'factor',target_id:positive}]),[]);
 assert.deepEqual(cartographyNoteCandidates([{target:'factor',target_id:positive,quote:'il était surtout absent'}],message,map,[]),[]);
 assert.deepEqual(cartographyNoteCandidates([{target:'outcome',target_id:'o_dismissed',quote:'je me sentais seul'}],message,map,[{target:'outcome',target_id:'o_dismissed'}]),[]);
 const kept=cartographyNoteCandidates([{target:'factor',target_id:positive,quote:'il était surtout absent'}],message,map,[{target:'factor',target_id:positive}]);
 assert.equal(kept.length,1);assert.equal(kept[0].text,'Vous dites : « il était surtout absent »');
});

test('F20: the audit demonstration 5 writes no note on a lived experience answered "no"',async t=>{
 const s=setup();await s.ready();const answers=Object.fromEntries(CARTOGRAPHIE.factors.map((x,i)=>[x.id,i===0?0:1]));
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a',?,'{}','{}','{}','[]',1,1,1,1) ON CONFLICT(user_id) DO UPDATE SET answers=excluded.answers").run(JSON.stringify(answers));
 provider(t,p=>p==='ambre_writing'?draft({cartography_notes:[{target:'factor',target_id:CARTOGRAPHIE.factors[0].id,quote:'mon père était violent'}]}):undefined);
 await s.send('Je ne dirais pas que mon père était violent, il était surtout absent.','carto_negated_1');await s.settle(1);
 assert.equal(s.raw.prepare("SELECT ambre_notes FROM cartographies WHERE user_id='a'").get().ambre_notes,'[]');assert.equal(JSON.parse(s.turn().response).cartography_notes,undefined);
});

test('F20: the audit demonstrations 6 and 7 propose nothing',async t=>{
 const context={plan:{available:true,proposals_enabled:true,goals:[]},strategies:[],active:true},ordinary={risk:'none',posture:'explore',need:'understand',constraints:[]};
 const negated='Je ne veux surtout pas arrêter de boire, ce n’est pas le sujet aujourd’hui.';
 const goal=quote=>({content:'x',change_proposals:[{kind:'goal',target_id:null,quote,payload:{title:'Arrêter de boire',why:'Pour ma santé',step_text:'',text:'',measure:'',strategy_id:null,result_text:''}}]});
 assert.deepEqual(prepareChangeDraft(goal('arrêter de boire'),{message:negated,analysis:ordinary,context}).change_proposals,[]);
 assert.deepEqual(prepareChangeDraft(goal(negated),{message:negated,analysis:ordinary,context}).change_proposals,[]);
 const wanted='Je voudrais arrêter de boire le soir, pour mieux dormir.',kept=prepareChangeDraft(goal('Je voudrais arrêter de boire le soir'),{message:wanted,analysis:ordinary,context}).change_proposals;
 assert.equal(kept.length,1);assert.equal(kept[0].payload.why,undefined,'an invented motive is dropped');
 const s=setup();await s.ready();
 provider(t,p=>p==='ambre_writing'?draft({proposals:[{kind:'self_profile_patch',target_id:null,summary:'Mettre à jour votre contexte',evidence_quote:'e',payload:{changes:[{field:'current_context',mode:'set',value_json:JSON.stringify('Traverse un épisode dépressif sévère et des idées noires.')}]}}]}):undefined);
 await s.send('Bonjour, je voulais juste reprendre là où nous en étions.','profile_invented');await s.settle(1);
 assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_proposals').get().n,0);
});

test('F20: an unquoted profile value is labelled as Ambre’s rewording, a negated one is refused',async t=>{
 const s=setup();await s.ready();const message='Je travaille comme infirmière de nuit depuis deux ans et je dors mal.';
 provider(t,p=>p==='ambre_writing'?draft({proposals:[{kind:'self_profile_patch',target_id:null,summary:'Votre activité',evidence_quote:message,payload:{changes:[{field:'occupation',mode:'set',value_json:JSON.stringify('Soignante en horaires décalés')}]}}]}):undefined);
 await s.send(message,'profile_reworded');await s.settle(1);
 const row=s.raw.prepare('SELECT summary,binding FROM chat_proposals').get();assert.match(row.summary,/^Reformulation d’Ambre/);assert.equal(JSON.parse(row.binding).reformulated,true);
});
