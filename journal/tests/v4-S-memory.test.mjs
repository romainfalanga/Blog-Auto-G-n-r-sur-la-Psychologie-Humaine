// Lot S: conversation memory (F5 pinned safety history and key facts, F6 one
// bad quote never blocks the batch, F7 the stored safety quote contains the
// signal), R6 (reported beliefs kept out of the writer) and R16 (form preferences).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,seedTurns} from './v4-S-helper.mjs';
import {mergeMemory,memoryPrompt,MEMORY_SECTIONS,deterministicAnalysis} from '../src/ambre-engine.mjs';

test('F5: after 1 000 exchanges the 2019 attempt and the allergy remain; near-identical supports evict nothing',()=>{
 let summary={};
 const text=i=>i===3?'J’ai fait une tentative de suicide en 2019, j’en parle rarement.':i===5?'Je suis allergique à la pénicilline et je prends un traitement pour la thyroïde.':i===9?'Je vis avec Paul depuis trois ans.':i===400?'Paul et moi sommes séparés depuis une semaine.':'Ma voisine Anne est un appui précieux, semaine '+i+'.';
 for(let start=1;start<=1000;start+=8){
  const turns=Array.from({length:8},(_,k)=>{const i=start+k;return {id:'t'+i,content:text(i),created_at:'2026-0'+(1+(i%9))+'-1'+(i%9)+' 10:00:00'}});
  const items=turns.map(turn=>{const i=Number(turn.id.slice(1));return {section:i===3?'safety_history':[5,9,400].includes(i)?'key_facts':'safety',turn_id:turn.id,quote:turn.content}});
  summary=mergeMemory(summary,items,turns).summary;
 }
 assert.ok(summary.safety_history.some(x=>/tentative de suicide en 2019/.test(x)),JSON.stringify(summary.safety_history));
 assert.ok(summary.key_facts.some(x=>/pénicilline/.test(x)));assert.ok(summary.key_facts.some(x=>/Je vis avec Paul/.test(x)));assert.ok(summary.key_facts.some(x=>/séparés/.test(x)));
 assert.match(summary.key_facts[0],/^\d{4}-\d{2}-\d{2} · /,'pinned facts are dated');
 assert.equal(summary.safety.filter(x=>/voisine Anne/.test(x)).length,1,'near duplicates collapse into one support');
 const prompt=memoryPrompt(summary);assert.ok(prompt.safety_history.some(x=>/2019/.test(x)));
});

test('F7: the stored safety quote is the sentence carrying the signal, also for closed topics',()=>{
 const long='Je vous écris parce que la semaine a été longue, entre le travail, les trajets, la fatigue et les disputes à la maison qui reviennent sans cesse et que je ne sais plus comment arrêter. '.repeat(2)+'Ce soir je veux mourir.';
 const merged=mergeMemory({},[],[{id:'t1',content:long,created_at:'2026-03-02 21:00:00'}]).summary;
 assert.equal(merged.safety_seen,true);assert.match(merged.safety_history[0],/^2026-03-02 · Ce soir je veux mourir\./);
 const closed=deterministicAnalysis('Je voulais d’abord vous raconter ma journée, très longue, avec beaucoup de réunions et de trajets. '.repeat(3)+'Mais de ma mère, je ne veux pas en parler.').closed_topics;
 assert.match(closed[0],/je ne veux pas en parler/);assert.ok(closed[0].length<=240);
});

test('F6: an apostrophe no longer stops the memory, an invented quote is dropped alone, the watermark advances',async t=>{
 const s=setup();await s.ready();seedTurns(s,8,{content:i=>i===1?'Je n’arrive plus à dormir depuis l’accident.':'Message numéro '+i+' sur ma semaine.'});
 provider(t,p=>p==='ambre_memory'?{items:[{section:'facts',turn_id:'a-t1',quote:"Je n'arrive plus à dormir depuis l'accident."},{section:'facts',turn_id:'a-t2',quote:'Une citation inventée par le modèle'},...[3,4,5,6,7,8].map(i=>({section:'goals',turn_id:'a-t'+i,quote:'Message numéro '+i}))]}:undefined);
 s.raw.prepare("INSERT INTO chat_memory_jobs(user_id,updated_at) VALUES('a',0)").run();await s.chat.engine.updateMemory(s.env,'a');
 const memory=s.raw.prepare("SELECT * FROM chat_memory WHERE user_id='a'").get(),summary=JSON.parse(memory.summary);
 assert.equal(memory.through_turn,'a-t8');assert.deepEqual(summary.facts,['Je n’arrive plus à dormir depuis l’accident'],'the person’s own typography is kept');assert.equal(summary.goals.length,6);
});

test('F6: after three unusable responses the batch is marked as not summarized and the memory moves on',async t=>{
 const s=setup();await s.ready();seedTurns(s,4,{content:i=>i===2?'Je veux mourir.':'Message '+i+' ordinaire.'});let calls=0;
 provider(t,p=>{if(p==='ambre_memory'){calls++;return new Response('{}',{status:502})}});
 s.raw.prepare("INSERT INTO chat_memory_jobs(user_id,updated_at) VALUES('a',0)").run();
 for(let i=0;i<3;i++){await s.chat.engine.updateMemory(s.env,'a');s.tick(4000)}
 const memory=s.raw.prepare("SELECT * FROM chat_memory WHERE user_id='a'").get(),summary=JSON.parse(memory.summary);
 assert.equal(calls,3);assert.equal(memory.through_turn,'a-t4');assert.equal(summary.unsummarized_batches,1);assert.equal(summary.safety_seen,true,'deterministic safety rules still apply');
});

test('R6: beliefs reported during a reality signal stay out of the writer; R16: form preferences reach it, never a diagnosis',async t=>{
 const turns=[{id:'t1',content:'Mes voisins m’espionnent et ils me suivent partout depuis lundi.',created_at:'2026-03-01 10:00:00'},{id:'t2',content:'Je préfère des réponses courtes, avec un seul exemple concret.',created_at:'2026-03-01 10:05:00'},{id:'t3',content:'Je suis TDAH, faites court.',created_at:'2026-03-01 10:06:00'}];
 const merged=mergeMemory({},[{section:'facts',turn_id:'t1',quote:'Mes voisins m’espionnent et ils me suivent partout'},{section:'style',turn_id:'t2',quote:'Je préfère des réponses courtes, avec un seul exemple concret.'},{section:'style',turn_id:'t3',quote:'Je suis TDAH, faites court.'}],turns).summary;
 assert.deepEqual(merged.facts,[]);assert.equal(merged.reported_beliefs.length,1);assert.deepEqual(merged.style,['Je préfère des réponses courtes, avec un seul exemple concret.']);
 const prompt=memoryPrompt(merged);assert.equal(prompt.reported_beliefs,undefined);assert.equal(prompt.reality_count,1);assert.deepEqual(prompt.style,merged.style);
 const s=setup();await s.ready();s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a',?,0)").run(JSON.stringify(merged));
 const calls=provider(t);await s.send('Pouvez-vous m’aider à préparer ma semaine ?','style_turn_0001');await s.settle(1);
 const writing=JSON.parse(calls.find(x=>x.purpose==='ambre_writing').input);assert.deepEqual(writing.memory.style,merged.style);assert.equal(JSON.stringify(writing).includes('espionnent'),false);
 assert.match(calls.find(x=>x.purpose==='ambre_writing').system,/memory\.style/);
});

test('F5: pinned safety history is visible, editable and deletable by the person, in the export too',async t=>{
 const s=setup();await s.ready();provider(t);
 s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a',?,0)").run(JSON.stringify({safety_history:['2019-05-01 · J’ai fait une tentative en 2019.'],key_facts:['2026-01-02 · Allergie à la pénicilline.'],safety_seen:true}));
 const insight=await s.call('/api/chat/insight'),section=insight.memory.sections.find(x=>x.key==='safety_history');
 assert.equal(section.title,MEMORY_SECTIONS.safety_history);assert.equal(section.pinned,true);assert.equal(section.limit,24);assert.equal(section.items.length,1);
 const edited=await s.call('/api/chat/memory','PUT',{revision:insight.memory.revision,sections:{key_facts:['2026-01-02 · Allergie à la pénicilline.'],safety_history:[]}});
 assert.equal(edited.status,200);const stored=JSON.parse(s.raw.prepare("SELECT summary FROM chat_memory WHERE user_id='a'").get().summary);
 assert.deepEqual(stored.safety_history,[]);assert.equal(stored.safety_seen,undefined);
 const tooMany=await s.call('/api/chat/memory','PUT',{revision:edited.memory.revision,sections:{safety_history:Array.from({length:25},(_,i)=>'Ligne '+i)}});assert.equal(tooMany.status,400);
 const exported=await s.chat.exportData(s.env,'a');assert.ok(JSON.stringify(exported.chat_memory||exported).includes('pénicilline'));
});
