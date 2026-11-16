import test from 'node:test';
import assert from 'node:assert/strict';
import {crisisSignal,safetyReply} from '../src/chat-safety.mjs';
import {limitQuestions} from '../src/chat-text.mjs';
import {buildCompanionSystemPrompt} from '../src/psychology-framework.mjs';

test('quoted questions preserve the entire sentence with or without structured questions',()=>{
 for(const value of [
  'Quand vous vous demandez « est-ce que j’en fais assez ? », la culpabilité prend le dessus…',
  'Votre mère vous a dit : « Tu m’abandonnes ? » et vous êtes resté à ses côtés.',
  'Vous pensez “pourquoi moi ?” et cela vous pèse.',
  'Vous dites "pourquoi ?" sans trouver de réponse.',
  'Vous évoquez « sa question “tu pars ?” » avec tristesse.',
  'Vous dites « pourquoi ?\nPourquoi maintenant ? » et cherchez du sens.'
 ]){
  assert.equal(limitQuestions(value,0),value);
  assert.equal(limitQuestions(value+' Que ressentez-vous ? Une autre question ?',1),value+' Que ressentez-vous ?');
 }
 assert.equal(limitQuestions('Cela pèse (est-ce que cela vous parle ?) aujourd’hui. Vous pouvez prendre votre temps.',0),'Vous pouvez prendre votre temps.');
 assert.equal(limitQuestions('Pourquoi ?Comment ? Encore ? Prenez votre temps.',2),'Pourquoi ?Comment ? Prenez votre temps.');
});

test('direct, indirect, means-related and third-party danger have a deterministic backstop',()=>{
 for(const text of ['Je vais me suicider.','Je veux mourir.','J’ai envie d’en finir.','Je n’en peux plus de vivre.','Je ne peux plus vivre.','Je voudrais disparaître.','Je ne veux plus être là.','Je voudrais ne pas me réveiller.','Mes proches seraient mieux sans moi.','Je ne serai plus là demain.','J’aimerais ne plus être là.','Je suis un fardeau pour ma famille.','Quelle dose serait mortelle ?','J’ai avalé toute la boîte de médicaments.','Je vais sauter du pont.','Ma sœur parle de suicide.']){
  assert.equal(crisisSignal(text),'self_harm',text);
 }
 for(const text of ['Je vais le tuer.','Il menace de me tuer.','Je veux poignarder mon voisin.'])assert.equal(crisisSignal(text),'violence',text);
 for(const text of ['Je veux comprendre mes émotions.','Je n’en peux plus de mon travail.','Je vais me faire du bien.','Le train arrive à quinze heures.','Je veux faire disparaître cette habitude.','Ce travail est un fardeau.'])assert.equal(crisisSignal(text),null,text);
 assert.match(safetyReply('self_harm'),/3114/);assert.match(safetyReply('self_harm'),/15 ou le 112/);assert.match(safetyReply('self_harm'),/suicide/);
 assert.match(safetyReply('violence'),/17 ou le 112/);
});

test('verified crisis resources are in the system instructions without a childhood map',()=>{
 const prompt=buildCompanionSystemPrompt({cartography:false});
 assert.match(prompt,/3114 \(24 h\/24, 7 j\/7\)/);assert.match(prompt,/15 ou le 112/);
});
