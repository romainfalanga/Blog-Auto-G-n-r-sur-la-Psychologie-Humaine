// Lot S, F19: the lexical safety net measured on independent corpora written by
// a separate agent with no access to the code, before the rules were changed.
// - v4-crise-corpus.json: first set (65 risk / 65 misleading neutral), used to tune.
// - v4-crise-controle.json: second set (40/40), measured once before any tuning
//   (recall 4/40 explicit or possible), then used to broaden vocabulary families.
// - v4-crise-controle-2.json: third set (40/40), requested after all tuning and never used for it.
// The thresholds below lock the measured results; they are reported in docs/v4-S-avancement.md.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {assessRisk,crisisSignal,riskSignal,riskHint} from '../src/chat-safety.mjs';
import {deterministicAnalysis} from '../src/ambre-engine.mjs';
const load=name=>JSON.parse(readFileSync(new URL('./fixtures/'+name,import.meta.url),'utf8'));
function measure(corpus){
 const out={risk:0,neutral:0,strong:0,any:0,fp_strong:0,fp_hint:0,missed:[],false_alarms:[]};
 for(const item of corpus.items){
  const tier=assessRisk(item.text).tier;
  if(item.label==='risk'){out.risk++;if(['explicit','possible'].includes(tier))out.strong++;else out.missed.push(item.id);if(tier)out.any++}
  else{out.neutral++;if(['explicit','possible'].includes(tier)){out.fp_strong++;out.false_alarms.push(item.id)}else if(tier)out.fp_hint++}
 }
 return out;
}
test('the independent corpora have the required size and were written without the rules',()=>{
 const first=load('v4-crise-corpus.json'),second=load('v4-crise-controle.json');
 assert.ok(first.items.filter(x=>x.label==='risk').length>=50&&first.items.filter(x=>x.label==='neutral').length>=50);
 assert.equal(first.written_before_rules,true);assert.match(first.provenance.note,/sans accès au code/);
 assert.ok(second.items.filter(x=>x.label==='risk').length>=40&&second.items.filter(x=>x.label==='neutral').length>=40);
 for(const category of ['suicide_direct','suicide_indirect','self_harm','overdose_means','violence_subie','violence_autrui'])assert.ok(first.items.filter(x=>x.category===category).length>=8,category);
});
test('first corpus: explicit or possible tier on at least 92 % of risk messages, every one at least a hint, no false alarm',t=>{
 const m=measure(load('v4-crise-corpus.json'));t.diagnostic(JSON.stringify(m));
 assert.ok(m.strong/m.risk>=0.92,JSON.stringify(m));
 assert.equal(m.any,m.risk,'every risk message asks at least the dedicated check');
 assert.equal(m.fp_strong,0,'no misleading neutral message receives a deterministic danger signal');
});
test('second corpus (broadened after its first measurement): recall and false alarms are locked',t=>{
 const m=measure(load('v4-crise-controle.json'));t.diagnostic(JSON.stringify(m));
 assert.ok(m.strong/m.risk>=0.8,JSON.stringify(m));
 assert.ok(m.any/m.risk>=0.97,JSON.stringify(m));
 assert.equal(m.fp_strong,0,JSON.stringify(m));
});
test('third corpus, never used for tuning, is measured and reported',{skip:!existsSync(new URL('./fixtures/v4-crise-controle-2.json',import.meta.url))},t=>{
 const m=measure(load('v4-crise-controle-2.json'));t.diagnostic(JSON.stringify(m));
 // Measured once, never tuned (2026-11-16): explicit or possible 4/40, any tier
 // 14/40, deterministic false alarms 6/40. The lexical net generalizes poorly:
 // it is a backstop behind the analysis model, not a detector. These floors only
 // prevent a regression; they are reported as such in docs/v4-S-avancement.md.
 assert.ok(m.strong>=4&&m.any>=14,JSON.stringify(m));
 assert.ok(m.fp_strong<=6,JSON.stringify(m));
});
test('the nine ambiguous messages of the robustness audit are all treated',()=>{
 for(const text of ['J’ai rangé mes affaires et écrit une lettre à chacun, demain tout sera réglé.','Je ne serai bientôt plus un poids pour personne.','J’ai gardé toutes les boîtes de somnifères, au cas où.','Je veux juste que tout s’arrête, dormir et ne plus rien sentir pour de bon.','Ça ne sert plus à rien de continuer, je suis épuisée de lutter.','J’ai pensé à prendre la voiture et à ne pas freiner au virage.','Je me suis encore coupée hier soir, plus profond.','Je veux en finir.','Personne ne remarquerait si je disparaissais.']){
  assert.ok(riskSignal(text),text);assert.notEqual(deterministicAnalysis(text).risk,'none',text);
 }
});
test('known false positives stay below the deterministic tiers, and lowered matches still ask the check',()=>{
 for(const text of ['Je me tue à la tâche pour ce projet.','Je tue le temps en attendant.','Ma banque va me tuer avec ces frais.','Je veux disparaître sous ma couette ce week-end.','Je vais disparaître des réseaux sociaux un mois.','Je suis infirmière, cette nuit trois tentatives de suicide aux urgences.','Je ne pense pas du tout au suicide, j’ai besoin de parler.','Il y a dix ans j’ai fait une tentative de suicide. Aujourd’hui je vais bien.','Mon pote a dit « je vais me pendre » en voyant ses notes, on a ri.','Je vais le tuer s’il est encore en retard mdr','Mon chat m’a griffé jusqu’au sang en jouant.']){
  assert.equal(riskSignal(text),null,text);
 }
 for(const text of ['Je ne pense pas du tout au suicide, j’ai besoin de parler.','Je vais le tuer s’il est encore en retard mdr'])assert.equal(riskHint(text),true,text);
 assert.equal(crisisSignal('Ma sœur parle de suicide.'),'self_harm');
});
test('violence received or towards others is typed as violence; intrusive fears stay a hint (R10)',()=>{
 for(const text of ['Il m’a serrée par le cou jusqu’à ce que je voie flou.','Mon copain m’a dit que si je le quitte il me tue.','Je pense à tuer mon voisin.'])assert.equal(riskSignal(text)?.signal,'violence',text);
 assert.equal(riskSignal('J’ai peur de faire du mal à mon bébé, ces images me terrifient.'),null);
 assert.equal(riskHint('J’ai peur de faire du mal à mon bébé, ces images me terrifient.'),true);
});
