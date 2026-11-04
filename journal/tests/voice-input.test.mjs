import test from 'node:test';
import assert from 'node:assert/strict';
import {createVoiceInput,VOICE_INPUT_SCRIPT} from '../src/voice-input.mjs';
class Element extends EventTarget {constructor(){super();this.value='';this.dataset={};this.attrs={};this.isConnected=true;}setAttribute(k,v){this.attrs[k]=v;}setSelectionRange(a,b){this.selectionStart=a;this.selectionEnd=b;}}
function setup({available,unsupported=false}={}){
 const instances=[],input=new Element(),button=new Element(),status=new Element(),document=new Element(),changes=[],states=[];
 class Recognition{constructor(){instances.push(this);this.processLocally=false;}start(){this.onstart?.();}abort(){this.aborted=true;}emit(results){this.onresult?.({results:results.map(([text,final])=>Object.assign([{transcript:text}],{isFinal:final}))});}}
 if(available)Recognition.available=available;
 const platform={document,...(!unsupported?{SpeechRecognition:Recognition}:{})};const controller=createVoiceInput({input,button,status,onChange:value=>changes.push(value),onState:value=>states.push(value)},platform);
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
 for(const value of ['available','downloadable','unavailable']){let options;const f=setup({available:async v=>{options=v;return value}});await f.click();assert.deepEqual(options,{langs:['fr-FR'],processLocally:true});assert.equal(f.instances[0].processLocally,value==='available');f.controller.dispose();}
});
test('permission refusal and unsupported browsers preserve the editor with a useful state',async()=>{
 const f=setup();f.input.value='À conserver';await f.click();f.instances[0].onerror({error:'not-allowed'});assert.equal(f.input.value,'À conserver');assert.match(f.status.textContent,/Autorisez/);assert.equal(f.controller.isActive(),false);
 const u=setup({unsupported:true});assert.equal(u.button.disabled,true);assert.match(u.status.textContent,/clavier/);
});
test('hiding the page and disposing stop capture, including asynchronous preparation',async()=>{
 const f=setup();await f.click();f.document.visibilityState='hidden';f.document.dispatchEvent(new Event('visibilitychange'));assert.ok(f.instances[0].aborted);
 let resolve;const waiting=setup({available:()=>new Promise(r=>{resolve=r})});await waiting.click();waiting.controller.dispose();resolve('available');await Promise.resolve();await Promise.resolve();assert.equal(waiting.instances.length,0);
});
test('length limits retain existing text and never submit a message',async()=>{
 const f=setup();f.input.value='x'.repeat(5990);await f.click();f.instances[0].emit([['une phrase beaucoup trop longue',false]]);assert.equal(f.input.value.length,6000);assert.ok(f.input.value.startsWith('x'.repeat(5990)));assert.equal(f.controller.isActive(),false);assert.match(f.status.textContent,/Limite/);
});
test('the inline adapter is self-contained and does not include network or audio-upload code',()=>{
 assert.doesNotThrow(()=>new Function(VOICE_INPUT_SCRIPT));assert.ok(!VOICE_INPUT_SCRIPT.includes('fetch('));assert.ok(!VOICE_INPUT_SCRIPT.includes('MediaRecorder'));
});
