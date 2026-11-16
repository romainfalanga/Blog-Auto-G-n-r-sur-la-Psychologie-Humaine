// Lot S, requests from the coordination of the v4.1 lots: the daily cap of
// lot R seen from `send` and from the conversation memory.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,seedTurns} from './v4-S-helper.mjs';
import {parisDay} from '../src/provider-access.mjs';

test('send: a reached daily cap is a clear 503, with the resources when a danger signal is present',async t=>{
 provider(t);const s=setup({env:{DAILY_PROVIDER_TOKEN_CAP:'1000'}});await s.ready();
 s.raw.prepare('INSERT INTO provider_global_budget(day,reserved_tokens,calls) VALUES(?,1000,1)').run(parisDay(s.now()));
 const plain=await s.send('Une question ordinaire sur ma semaine.','daily_cap_plain');
 assert.equal(plain.status,503);assert.equal(plain.code,'daily_cap');assert.match(plain.message,/limite d’utilisation pour aujourd’hui/);assert.doesNotMatch(plain.message,/accord/);
 const danger=await s.send('Je veux mourir ce soir.','daily_cap_danger');
 assert.equal(danger.status,503);assert.equal(danger.code,'daily_cap');assert.match(danger.error,/3114/);assert.equal(danger.safety,true);
 assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_turns').get().n,0);
});

test('memory: a provider refusal or the daily cap never counts as a failed attempt',async t=>{
 {const s=setup();await s.ready();seedTurns(s,4);let calls=0;provider(t,p=>{if(p==='ambre_memory'){calls++;return new Response('{}',{status:503,headers:{'Retry-After':'60'}})}});
  s.raw.prepare("INSERT INTO chat_memory_jobs(user_id,updated_at) VALUES('a',0)").run();await s.chat.engine.updateMemory(s.env,'a');
  assert.equal(calls,1);assert.equal(s.raw.prepare("SELECT attempts FROM chat_memory_jobs WHERE user_id='a'").get().attempts,0);assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_memory').get().n,0)}
 {const s=setup({env:{DAILY_PROVIDER_TOKEN_CAP:'1000'}});await s.ready();seedTurns(s,4);let calls=0;provider(t,p=>{if(p==='ambre_memory')calls++});
  s.raw.prepare('INSERT INTO provider_global_budget(day,reserved_tokens,calls) VALUES(?,1000,1)').run(parisDay(s.now()));
  s.raw.prepare("INSERT INTO chat_memory_jobs(user_id,updated_at) VALUES('a',0)").run();await s.chat.engine.updateMemory(s.env,'a');
  assert.equal(calls,0);assert.equal(s.raw.prepare("SELECT attempts FROM chat_memory_jobs WHERE user_id='a'").get().attempts,0)}
});

test('lot C: the cartography guidance explains display_order without turning a rank into a risk',async()=>{
 const {CHILDHOOD_CARTOGRAPHY_GUIDANCE}=await import('../src/psychology-framework.mjs');
 assert.match(CHILDHOOD_CARTOGRAPHY_GUIDANCE,/display_order/);assert.match(CHILDHOOD_CARTOGRAPHY_GUIDANCE,/associations empiriques/);
 assert.match(CHILDHOOD_CARTOGRAPHY_GUIDANCE,/jamais un rang comme un risque/);assert.match(CHILDHOOD_CARTOGRAPHY_GUIDANCE,/« limitée » est théorique/);
});
