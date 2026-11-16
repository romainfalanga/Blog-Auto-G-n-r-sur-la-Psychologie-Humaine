// Lot S: clinical signals and their effects on writing and review (R1–R8,
// R10, R13, R14, R16, R17, R20, R21, R24, R27, R33, #12, #34).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,analysis,draft,review,lastReply} from './v4-S-helper.mjs';
import {ipvSignal,realitySignal,eatingSignal,minorSignal,intrusiveSignal,substanceSignal,metaAiSignal,perpetratorSignal,refusesProposal,REFUSAL_EXPRESSIONS,declinedStreak,signalReview,missingReferral,signalGuidance,VERIFIED_NUMBERS,REFERRAL_RULES} from '../src/clinical-signals.mjs';
import {DEPTH_RULES,FRAME_RULES,deterministicReview,deterministicAnalysis} from '../src/ambre-engine.mjs';
import {safetyReply,IPV_RESOURCES} from '../src/chat-safety.mjs';
const base={constraints:[],risk:'none'};

test('R1, R24, R27, R33: the depth and frame rules are in the active writer prompt',async t=>{
 assert.ok(DEPTH_RULES.length<=700);for(const words of [/vu ou entendu, ressenti, et ce qu’elle imagine/,/explications possibles/,/exceptions et ressources/,/Emprise : jamais de confrontation/,/persécution ou de grandeur/])assert.match(DEPTH_RULES,words);
 for(const words of [/cause biographique pour un épisode isolé/,/Orientation et identité de genre ne sont jamais à corriger/,/sans supposer culture ni religion/,/la foi peut aider/])assert.match(FRAME_RULES,words);
 const s=setup();await s.ready();const calls=provider(t);await s.send('Je me demande pourquoi ma sœur réagit ainsi.','depth_rules_01');await s.settle(1);
 const system=calls.find(x=>x.purpose==='ambre_writing').system;assert.ok(system.includes(DEPTH_RULES));assert.ok(system.includes(FRAME_RULES));assert.doesNotMatch(system,/Signaux de ce tour/,'nominal turns pay no signal guidance');
});

test('R2: intimate partner violence levels, never from a denial',()=>{
 assert.equal(ipvSignal('Mon mari contrôle mon téléphone et mon argent.'),'control');
 assert.equal(ipvSignal('Ma compagne m’a encore frappé hier soir.'),'physical');
 assert.equal(ipvSignal('Mon copain m’a serrée par le cou la semaine dernière.'),'severe');
 assert.equal(ipvSignal('Mon conjoint a menacé de me tuer s’il me voyait partir.'),'severe');
 assert.equal(ipvSignal('Mon mari ne m’a jamais frappée, on se dispute sur l’argent.'),'none');
 assert.equal(ipvSignal('Je me suis disputée avec ma collègue.'),'none');
});

test('R2–R4: IPV guidance with verified numbers, no confrontation on any reply, referral required',async t=>{
 const guidance=signalGuidance({ipv:'control'});for(const n of ['3919','17','112','114'])assert.ok(guidance.includes(n));assert.match(guidance,/jamais de conseil de confronter/);assert.match(guidance,/effacer/);
 assert.deepEqual(signalReview('Dites-lui calmement ce que vous ressentez.',{ipv:'control'}),[2]);
 assert.deepEqual(signalReview('Dites-lui calmement ce que vous ressentez.',{}),[]);
 assert.deepEqual(missingReferral('Vous avez le droit d’être en sécurité.',{ipv:'control'}),['ipv']);
 assert.deepEqual(missingReferral('Le 3919 peut vous écouter, 24 h/24.',{ipv:'control'}),[]);
 assert.deepEqual(missingReferral('Vous avez le droit d’être en sécurité.',{ipv:'control'},[{assistant:'Le 3919 écoute 24 h/24.'}]),[],'no repetition within three replies');
 assert.deepEqual(missingReferral('Vous avez le droit d’être en sécurité.',{ipv:'severe'},[{assistant:'Le 3919 écoute 24 h/24.'}]),['ipv'],'a severe factor always repeats it');
 // End to end: a confronting draft is rewritten, the guidance reaches the writer.
 const s=setup();await s.ready();let writes=0;const calls=provider(t,p=>p==='ambre_writing'?(++writes===1?draft({content:'Dites-lui calmement ce que vous ressentez, il comprendra.'}):draft({content:'Ce contrôle de votre téléphone et de votre argent est pesant. Le 3919 écoute gratuitement, 24 h/24 ; en danger, le 17 ou le 112.'})):undefined);
 await s.send('Mon mari contrôle mon téléphone et mon argent, je ne sais plus quoi faire.','ipv_control_01');await s.settle(2);
 assert.equal(writes,2);assert.match(lastReply(s),/3919/);assert.match(calls.find(x=>x.purpose==='ambre_writing').system,/Signaux de ce tour/);
 assert.deepEqual(JSON.parse(s.job().analysis).signals.ipv,'control');
});

test('R3: severe partner violence receives the fixed 3919 / 17 / 112 / 114 reply, also on a refused send',async t=>{
 assert.match(safetyReply('ipv_severe'),/3919/);assert.match(IPV_RESOURCES,/114/);assert.match(IPV_RESOURCES,/pas un numéro d’urgence/);
 const s=setup();await s.ready();provider(t,p=>p==='ambre_review'?'invalid JSON':undefined);
 await s.send('Mon copain m’a serrée par le cou hier soir, j’ai peur.','ipv_severe_0001');await s.settle(3);
 const reply=lastReply(s);assert.match(reply,/3919/);assert.match(reply,/17 ou le 112/);assert.doesNotMatch(reply,/3114/);
 const t2=setup();await t2.call('/api/chat/consent','PUT',{accepted:true});const refused=await t2.send('Mon copain m’a serrée par le cou hier soir.','ipv_refused_001');assert.match(refused.error,/3919/);
});

test('R5–R6: reality signals never from faith alone; no exploration, no method, medical advice',async t=>{
 assert.equal(realitySignal('Mes voisins m’espionnent depuis des semaines.'),'unusual_belief');
 assert.equal(realitySignal('Je n’ai pas besoin de dormir, je déborde d’idées.'),'mania_signs');
 assert.equal(realitySignal('Je crois que Dieu me guide et la prière m’apaise.'),'none');
 assert.equal(realitySignal('Dans ma famille on croit aux esprits des ancêtres.'),'none');
 const s=setup();await s.ready();const calls=provider(t);await s.send('Mes voisins m’espionnent depuis des semaines, je le sais.','reality_turn_01');await s.settle(1);
 const a=JSON.parse(s.job().analysis);assert.ok(a.constraints.includes('no_exploration'));assert.deepEqual(a.strategy_ids||[],[]);
 assert.match(calls.find(x=>x.purpose==='ambre_writing').system,/avis médical/);
 assert.match(signalGuidance({reality:'unusual_belief',reality_count:3}),/IA qui ne peut pas vérifier/);
});

test('R7: eating concerns forbid figures; R8: only a current declared age is a minor signal',()=>{
 assert.equal(eatingSignal('Je me fais vomir après les repas.'),true);assert.equal(eatingSignal('J’ai bien mangé ce midi.'),false);
 assert.deepEqual(signalReview('Visez 1200 kcal par jour.',{eating:true}),[2]);assert.deepEqual(signalReview('Votre corps mérite de l’attention.',{eating:true}),[]);
 assert.equal(minorSignal('J’ai 15 ans et mes parents me crient dessus.'),true);
 assert.equal(minorSignal('Quand j’avais 15 ans, mes parents criaient.'),false);
 assert.equal(minorSignal('J’ai 35 ans.'),false);assert.equal(minorSignal('Bonjour',{age:16}),true);assert.equal(minorSignal('Bonjour',{age:42}),false);
 assert.match(signalGuidance({minor:true}),/119/);assert.match(signalGuidance({minor:true}),/adulte de confiance/);
 assert.deepEqual(missingReferral('Je vous écoute.',{minor:true}),['minor']);
});

test('R10: an unwanted intrusive thought is not a crisis; a deterministic signal is never lowered',async t=>{
 assert.equal(intrusiveSignal('J’ai peur de faire du mal à mon bébé, ces images me terrifient.'),true);
 assert.equal(intrusiveSignal('Je vais faire du mal à mon voisin.'),false);
 {const s=setup();await s.ready();const calls=provider(t,p=>p==='ambre_analysis'?analysis({risk:'possible',posture:'safety'}):p==='ambre_risk_check'?{risk:'possible',category:'intrusive_unwanted',evidence_quote:'des images où je fais tomber mon bébé'}:undefined);
  await s.send('J’ai des images où je fais tomber mon bébé et ça me terrifie, je ne le ferais jamais.','intrusive_0001');await s.settle(1);
  const a=JSON.parse(s.job().analysis);assert.equal(a.risk,'none');assert.equal(a.signals.intrusive,true);assert.doesNotMatch(lastReply(s),/3114/);
  assert.match(calls.find(x=>x.purpose==='ambre_writing').system,/pas de bloc de crise automatique/)}
 {const s=setup();await s.ready();const calls=provider(t,p=>p==='ambre_risk_check'?{risk:'possible',category:'intrusive_unwanted',evidence_quote:'tuer mon voisin'}:undefined);
  await s.send('Je pense à tuer mon voisin.','deterministic_violence');await s.settle(1);
  assert.equal(calls.some(x=>x.purpose==='ambre_risk_check'),false);assert.equal(JSON.parse(s.job().analysis).risk,'explicit')}
});

test('R13: one referral grid with verified numbers only',()=>{
 assert.deepEqual([...VERIFIED_NUMBERS].sort(),['112','114','119','15','17','3114','3919'].sort());
 for(const rule of Object.values(REFERRAL_RULES))for(const n of rule.resources)assert.ok(VERIFIED_NUMBERS.includes(n));
 assert.ok(deterministicReview({content:'Appelez le 0800 pour parler.'},base).includes(2));
 assert.equal(deterministicReview({content:'En danger, composez le 114 par SMS.'},base).includes(2),false);
});

test('R14: two refused proposals bring the reply back to listening unless the person asks again',async t=>{
 for(const phrase of ['Non merci.','Je ne veux pas d’exercice.','Ça ne m’aide pas.','Arrêtez de me proposer des choses.','J’ai déjà essayé ça.'])assert.equal(refusesProposal(phrase),true,phrase);
 assert.ok(REFUSAL_EXPRESSIONS.length>=10);assert.equal(refusesProposal('Merci, c’est une bonne idée.'),false);
 assert.equal(declinedStreak({plan:{document:{proposals:[{status:'dismissed',updated_at:3},{status:'dismissed',updated_at:2},{status:'accepted',updated_at:1}]}}}),2);
 assert.equal(declinedStreak({history:[{user:'Non merci, pas d’exercice.'}],message:'Ça ne m’aide pas.'}),2);
 const s=setup();await s.ready();provider(t);
 s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,created_at) VALUES('p1','a','p1','h','completed','consumed','2025-01-06',1,1,'Non merci, je ne veux pas d’exercice.','{\"companion_reply\":1,\"content\":\"D’accord.\"}','2026-01-01 10:00:00')").run();
 await s.send('Ça ne m’aide pas vraiment.','reactance_0001');await s.settle(1);
 const a=JSON.parse(s.job().analysis);assert.ok(a.constraints.includes('no_advice'));assert.deepEqual(a.strategy_ids||[],[]);
 const s2=setup();await s2.ready();provider(t);s2.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,created_at) VALUES('p1','a','p1','h','completed','consumed','2025-01-06',1,1,'Non merci, je ne veux pas d’exercice.','{\"companion_reply\":1,\"content\":\"D’accord.\"}','2026-01-01 10:00:00')").run();
 await s2.send('Ça ne m’aide pas. Que dois-je faire alors, un conseil ?','reactance_asks');await s2.settle(1);assert.equal(JSON.parse(s2.job().analysis).constraints.includes('no_advice'),false);
});

test('R17: after a repair, no method in the next turn and a light check',async t=>{
 const s=setup();await s.ready();const calls=provider(t);
 await s.send('Tu ne comprends rien, c’est une réponse de robot.','repair_turn_01');await s.settle(1);assert.equal(JSON.parse(s.job().analysis).posture,'repair');
 await s.send('Bon, je reprends : ma semaine a été difficile.','repair_next_01');await s.settle(1);
 const a=JSON.parse(s.job().analysis);assert.ok(a.constraints.includes('no_advice'));
 assert.match(calls.filter(x=>x.purpose==='ambre_writing').at(-1).system,/réparait une incompréhension/);
});

test('R20, R21, R27, #34: honest AI, no attachment, no approval of harmful acts, no identity correction, no abrupt stop',()=>{
 assert.equal(metaAiSignal('Est-ce que tu m’aimes ?'),true);assert.equal(metaAiSignal('Tu es consciente ?'),true);assert.equal(metaAiSignal('Ma mère m’aime-t-elle ?'),false);
 assert.match(signalGuidance({meta_ai:true}),/IA sans sentiments/);
 assert.deepEqual(signalReview('Je tiens à vous, je serai toujours là.',{}),[15]);
 assert.deepEqual(signalReview('Vous avez entièrement raison, il a totalement tort.',{}),[11]);
 assert.deepEqual(signalReview('Vous avez eu raison de le frapper.',{}),[11]);
 assert.deepEqual(signalReview('C’est compréhensible de ressentir de la colère après cela.',{}),[],'validating an emotion stays allowed');
 assert.deepEqual(signalReview('Nous pourrions travailler à changer votre orientation.',{}),[10]);
 assert.equal(substanceSignal('Je bois tous les soirs pour dormir.'),true);
 assert.deepEqual(signalReview('Arrêtez l’alcool du jour au lendemain.',{substance:true}),[2]);
 assert.match(signalGuidance({substance:true}),/jamais de conseil d’arrêt brutal/);
 assert.equal(perpetratorSignal('Je l’ai giflée hier et je deviens violent.'),true);
});

test('#12: adverse events are counted without any content',async t=>{
 const s=setup();await s.ready();provider(t,p=>p==='ambre_analysis'?analysis({risk:'possible',posture:'safety'}):p==='ambre_risk_check'?new Response('{}',{status:429,headers:{'Retry-After':'60'}}):p==='ambre_review'?'invalid JSON':undefined);
 await s.send('Une question de diagnostic.','adverse_events_1');await s.settle(2);
 const rows=s.raw.prepare("SELECT name,value FROM service_metrics WHERE name LIKE 'adverse_%'").all();
 assert.ok(rows.some(r=>r.name==='adverse_cautious_reply'&&r.value>=1),JSON.stringify(rows));
 for(const row of rows)assert.match(row.name,/^adverse_[a-z_]+$/);
});
