import test from 'node:test';
import assert from 'node:assert/strict';
import {createVoiceInput,VOICE_INPUT_SCRIPT} from '../src/voice-input.mjs';
class Element extends EventTarget {constructor(){super();this.value='';this.dataset={};this.attrs={};this.isConnected=true;}setAttribute(k,v){this.attrs[k]=v;}setSelectionRange(a,b){this.selectionStart=a;this.selectionEnd=b;}}
function setup({available,unsupported=false,stall=false,clock={}}={}){
 const instances=[],input=new Element(),button=new Element(),status=new Element(),document=new Element(),changes=[],states=[];
 class Recognition{constructor(){instances.push(this);this.processLocally=false;}start(){if(!stall)this.onstart?.();}stop(){this.stopped=true;}abort(){this.aborted=true;}emit(results){this.onresult?.({results:results.map(([text,final])=>Object.assign([{transcript:text}],{isFinal:final}))});}}
 if(available)Recognition.available=available;
 const platform={document,...clock,...(!unsupported?{SpeechRecognition:Recognition}:{})};const controller=createVoiceInput({input,button,status,onChange:value=>changes.push(value),onState:value=>states.push(value)},platform);
 return {controller,input,button,status,document,instances,changes,states,click:async()=>{button.dispatchEvent(new Event('click'));await Promise.resolve();await Promise.resolve();}};
}
test('interim replacements and final results update one editable draft without repeated words',async()=>{
 const f=setup();f.input.value='Bonjour';await f.click();const r=f.instances[0];r.emit([['je veux',false]]);assert.equal(f.input.value,'Bonjour je veux');r.emit([['je voudrais parler',false]]);assert.equal(f.input.value,'Bonjour je voudrais parler');r.emit([['je voudrais parler',true],['de mon travail',false]]);assert.equal(f.input.value,'Bonjour je voudrais parler de mon travail');f.controller.stop();assert.ok(r.aborted);assert.equal(f.button.attrs['aria-pressed'],'false');assert.equal(f.input.disabled,undefined);
});
test('a manual correction aborts dictation and queued results cannot replace that correction',async()=>{
 const f=setup();await f.click();const r=f.instances[0];r.emit([['ma relation',false]]);const queued=r.onresult;f.input.value='la relation avec ma sœur';f.input.dispatchEvent(new Event('input'));queued({results:[[{transcript:'ma relation'}]]});assert.equal(f.input.value,'la relation avec ma sœur');assert.ok(r.aborted);
});
test('stopping before send prevents delayed recognition from writing into the next draft',async()=>{
 const f=setup();await f.click();const r=f.instances[0],queued=r.onresult;r.emit([['Premier message',true]]);f.controller.stop();f.input.value='Un autre brouillon';queued({results:[[{transcript:'suite'}]]});assert.equal(f.input.value,'Un autre brouillon');
});
test('dictation replaces only the selected text and preserves following text',async()=>{
 const f=setup();f.input.value='Avant erreur, après';f.input.selectionStart=6;f.input.selectionEnd=12;await f.click();f.instances[0].emit([['correction',true]]);assert.equal(f.input.value,'Avant correction, après');
});
test('installed local French recognition is preferred and missing packs trigger no download',async()=>{
 for(const value of ['available','downloadable','unavailable']){let options;const f=setup({available:async v=>{options=v;return value}});await Promise.resolve();await f.click();assert.deepEqual(options,{langs:['fr-FR'],processLocally:true});assert.equal(f.instances[0].processLocally,value==='available');f.controller.dispose();}
});
test('permission refusal and unsupported browsers preserve the editor with a useful state',async()=>{
 const f=setup();f.input.value='À conserver';await f.click();f.instances[0].onerror({error:'not-allowed'});assert.equal(f.input.value,'À conserver');assert.match(f.status.textContent,/Autorisez/);assert.equal(f.controller.isActive(),false);
 const u=setup({unsupported:true});assert.equal(u.button.disabled,true);assert.match(u.status.textContent,/clavier/);
});
test('hiding the page and disposing stop capture without waiting for language availability',async()=>{
 const f=setup();await f.click();f.document.visibilityState='hidden';f.document.dispatchEvent(new Event('visibilitychange'));assert.ok(f.instances[0].aborted);
 let resolve;const waiting=setup({available:()=>new Promise(r=>{resolve=r})});await waiting.click();assert.equal(waiting.instances[0].processLocally,false);waiting.controller.dispose();resolve('available');await Promise.resolve();assert.equal(waiting.instances.length,1);assert.equal(waiting.instances[0].aborted,true);
});
test('tapping the stop icon waits for the last final segment and releases the composer',async()=>{
 const f=setup();await f.click();const r=f.instances[0];r.emit([['Je voudrais',false]]);await f.click();assert.ok(r.stopped);assert.equal(r.aborted,undefined);assert.equal(f.button.dataset.voicePhase,'finishing');assert.equal(f.button.disabled,true);assert.match(f.status.textContent,/dernière phrase/);
 r.emit([['Je voudrais en parler.',true]]);r.onend();assert.equal(f.input.value,'Je voudrais en parler.');assert.equal(f.controller.isActive(),false);assert.equal(f.button.disabled,false);assert.equal(f.button.dataset.voicePhase,'idle');assert.match(f.status.textContent,/relire/);
});
test('a stalled final segment leaves the visible draft editable and cancels late capture',async()=>{
 let finish;const f=setup({clock:{setTimeout(fn){finish=fn;return 1},clearTimeout(){}}});await f.click();const r=f.instances[0];r.emit([['Mes mots',false]]);await f.click();finish();assert.ok(r.aborted);assert.equal(f.input.value,'Mes mots');assert.equal(f.controller.isActive(),false);
});
test('withdrawn provisional words restore the original selection and final segments are not duplicated',async()=>{
 const f=setup();f.input.value='Salut toi.';f.input.setSelectionRange(6,9);await f.click();const r=f.instances[0];r.emit([['Marie',false]]);assert.equal(f.input.value,'Salut Marie.');r.emit([]);assert.equal(f.input.value,'Salut toi.');assert.equal(f.input.selectionStart,6);assert.equal(f.input.selectionEnd,9);
 r.emit([['tout le monde',true],[', bonjour',false]]);assert.equal(f.input.value,'Salut tout le monde, bonjour.');r.emit([['tout le monde',true]]);assert.equal(f.input.value,'Salut tout le monde.');
});
test('length limits retain existing text and never submit a message',async()=>{
 const f=setup();f.input.value='x'.repeat(5990);await f.click();f.instances[0].emit([['une phrase beaucoup trop longue',false]]);assert.equal(f.input.value.length,6000);assert.ok(f.input.value.startsWith('x'.repeat(5990)));assert.equal(f.controller.isActive(),false);assert.match(f.status.textContent,/Limite/);
});
test('the inline adapter is self-contained and does not include network or audio-upload code',()=>{
 assert.doesNotThrow(()=>new Function(VOICE_INPUT_SCRIPT));assert.ok(!VOICE_INPUT_SCRIPT.includes('fetch('));assert.ok(!VOICE_INPUT_SCRIPT.includes('MediaRecorder'));
});

test('a recognition engine that never starts releases the composer with its draft intact',async()=>{
 let startup;
 const f=setup({stall:true,clock:{setTimeout(fn,ms){assert.equal(ms,12000);startup=fn;return 1},clearTimeout(){}}});
 f.input.value='Mon brouillon';
 await f.click();
 assert.equal(f.button.dataset.voicePhase,'starting');
 startup();
 assert.equal(f.input.value,'Mon brouillon');
 assert.equal(f.controller.isActive(),false);
 assert.ok(f.instances[0].aborted);
 assert.match(f.status.textContent,/n’a pas démarré/);
});
