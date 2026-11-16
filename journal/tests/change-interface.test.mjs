import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {CHANGE_INTERFACE_SCRIPT,CHANGE_INTERFACE_STYLE} from '../src/change-interface.mjs';
import {CHAT_INTERFACE_SCRIPT} from '../src/chat-interface.mjs';
import {SETTINGS_INTERFACE_SCRIPT} from '../src/settings-interface.mjs';

class Element {
  constructor(tag,doc){this.tagName=tag.toUpperCase();this.doc=doc;this.children=[];this.listeners={};this.dataset={};this.attributes={};this.value='';this.checked=false;this.disabled=false;this.hidden=false;this.isConnected=true;this._text='';}
  append(...items){for(const item of items){if(item&&typeof item==='object'){item.parentElement=this;}this.children.push(item);}}
  replaceChildren(...items){this.children=[];this._text='';this.append(...items);}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(item=>typeof item==='string'?item:item.textContent).join('');}
  setAttribute(key,value){this.attributes[key]=value;}
  removeAttribute(key){delete this.attributes[key];}
  addEventListener(key,fn){(this.listeners[key]??=[]).push(fn);}
  async fire(key,extra={}){for(const fn of this.listeners[key]||[])await fn({target:this,currentTarget:this,preventDefault(){},...extra});}
  focus(){this.doc.activeElement=this;}
  scrollIntoView(){}
  showModal(){this.open=true;}
  close(){this.open=false;}
  remove(){if(this.parentElement)this.parentElement.children=this.parentElement.children.filter(item=>item!==this);this.isConnected=false;}
  matches(selector){return selector.split(',').some(part=>part[0]==='.'?String(this.className||'').split(' ').includes(part.slice(1)):part[0]==='#'?this.id===part.slice(1):part.startsWith('[name=')?this.name===part.slice(6,-1):this.tagName===part.toUpperCase());}
  querySelectorAll(selector){return this.children.flatMap(item=>typeof item==='object'?[...(item.matches(selector)?[item]:[]),...item.querySelectorAll(selector)]:[]);}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  get elements(){return Object.fromEntries(this.querySelectorAll('input,textarea,select').map(item=>[item.name,item]));}
}
const empty=()=>({available:true,revision:4,epoch:1,proposals_enabled:true,reminders_paused:false,goals:[],steps:[],trials:[],proposals:[],instructions:[],preparations:[],counts:{goals:0,steps:0,trials:0,proposals:0,instructions:0,preparations:0},limits:{goals:20,active_goals:3,steps:60,trials:200,proposals:40,instructions:10,preparations:20},next_offsets:{goals:null,steps:null,trials:null,proposals:null,instructions:null,preparations:null}});
const json=value=>JSON.parse(JSON.stringify(value));
function environment(handler){
 const doc={createElement(tag){return new Element(tag,this)},createTextNode(value){return String(value)},getElementById(id){return this.body.querySelector('#'+id)}};doc.body=new Element('body',doc);let serial=0;const calls=[];
 const ctx=vm.createContext({document:doc,location:{hash:''},state:{},Set,Map,JSON,console,confirm:()=>true,alert(){},newClientId:()=>`request_${++serial}`,api:async(path,method='GET',data)=>{calls.push({path,method,data:json(data??null)});return handler(path,method,data)},resetChatSession(){},clearTimeout(){},renderChat:async()=>{},el:(tag,cls,text)=>{const item=doc.createElement(tag);item.className=cls||'';if(text!==undefined)item.textContent=text;return item},actionButton:(label,fn,cls)=>{const button=doc.createElement('button');button.textContent=label;button.type='button';button.className=cls||'';button.addEventListener('click',fn);return button}});
 vm.runInContext(CHANGE_INTERFACE_SCRIPT,ctx);return {ctx,doc,calls,host(){const host=doc.createElement('section');doc.body.append(host);return host},dialog:()=>doc.body.querySelector('dialog')};
}
const field=(dialog,name)=>dialog.querySelectorAll('input,textarea,select').find(item=>item.name===name);
const button=(host,label)=>host.querySelectorAll('button').find(item=>item.textContent===label);
const submit=dialog=>dialog.querySelector('form').fire('submit');

test('embedded scripts compile, responsive rules cover the dialog and touch controls',()=>{
 for(const source of [CHANGE_INTERFACE_SCRIPT,CHAT_INTERFACE_SCRIPT,SETTINGS_INTERFACE_SCRIPT])assert.doesNotThrow(()=>new vm.Script(source));
 assert.match(CHANGE_INTERFACE_STYLE,/max-width:480px/);assert.match(CHANGE_INTERFACE_STYLE,/min-height:44px/);
 assert.match(SETTINGS_INTERFACE_SCRIPT,/\[\['next','Au prochain message'\],\['stored','Mes informations'\],\['turns','Réponses précédentes'\]\]/);
});

test('opening an empty plan reads data without creating a goal or professional linkage',async()=>{
 const env=environment(()=>empty()),host=env.host();await env.ctx.renderChangePlan(host,()=>true);
 for(const text of ['Mon plan','Mes objectifs','Mes étapes','Mes essais et mes retours','Mes consignes professionnelles privées','Mes points personnels'])assert.ok(host.textContent.includes(text),text);
 assert.deepEqual(env.calls.map(call=>[call.method,call.path]),[['GET','/api/plan']]);
 assert.ok(button(host,'Préparer mon point personnel'));
});

test('a revision conflict retains the member text and requires comparing before explicit resubmission',async()=>{
 let conflict=true;const old={id:'g1',revision:2,title:'Titre enregistré',why:'Ancien motif',status:'active',priority:2};
 const env=environment((path,method,data)=>{if(method==='GET')return {revision:conflict?4:5,epoch:1,item:{...old,title:conflict?old.title:'Titre ailleurs',revision:conflict?2:3}};if(conflict){conflict=false;throw Object.assign(Error('Version modifiée'),{status:409,code:'revision_conflict'});}return empty()});
 await env.ctx.changeOpenEditor('goals','g1',null,()=>true,()=>{});const dialog=env.dialog();field(dialog,'title').value='Ma saisie conservée';await submit(dialog);
 assert.equal(field(dialog,'title').value,'Ma saisie conservée');assert.ok(dialog.isConnected);assert.equal(button(dialog,'Enregistrer').disabled,true);
 await button(dialog,'Voir la version actuelle').fire('click');assert.ok(dialog.textContent.includes('Titre ailleurs'));assert.equal(env.calls.filter(call=>call.method==='PATCH').length,1);
 await button(dialog,'Garder ma saisie sur cette version').fire('click');await submit(dialog);
 const payload=env.calls.filter(call=>call.method==='PATCH').at(-1).data;assert.equal(payload.title,'Ma saisie conservée');assert.equal(payload.revision,5);assert.equal(payload.entity_revision,3);assert.equal(payload.epoch,1);
});

test('an erased epoch never recreates data from an old form',async()=>{
 let stale=false;const env=environment((path,method)=>{if(method==='GET')return {...empty(),epoch:stale?2:1};stale=true;throw Object.assign(Error('Plan effacé'),{status:409,code:'stale_epoch'})});
 await env.ctx.changeOpenEditor('goals',null,null,()=>true,()=>{});const dialog=env.dialog();field(dialog,'title').value='Texte encore copiable';await submit(dialog);await button(dialog,'Voir la version actuelle').fire('click');
 assert.equal(field(dialog,'title').value,'Texte encore copiable');assert.equal(button(dialog,'Enregistrer').disabled,true);assert.equal(button(dialog,'Garder ma saisie sur cette version'),undefined);
});

test('an uncertain POST retry reuses its request key and never adopts a proposal on read',async()=>{
 const proposal={id:'p1',revision:3,status:'proposed',kind:'goal',payload:{title:'Une piste',step_text:'Marcher un peu'}};let attempts=0;
 const env=environment((path,method)=>{if(method==='GET')return {revision:4,epoch:1,item:proposal};if(++attempts===1)throw Error('Réseau interrompu');return empty()});
 await env.ctx.changeProposalDialog('p1',()=>true,()=>{},true,false);assert.equal(env.calls.length,1);const dialog=env.dialog();field(dialog,'title').value='Mon objectif choisi';assert.equal(field(dialog,'create_trial').checked,false);assert.equal(field(dialog,'followup_opt_in').checked,false);
 await submit(dialog);await submit(dialog);const writes=env.calls.filter(call=>call.method==='POST');assert.equal(writes.length,2);assert.equal(writes[0].data.request_key,writes[1].data.request_key);assert.equal(writes[0].data.edited.title,'Mon objectif choisi');assert.equal(writes[0].data.create_trial,false);assert.equal(writes[0].data.followup_opt_in,false);assert.equal(writes[0].data.entity_revision,3);
});

test('a reported trial keeps unknown dates and a non-tried or worse result without reopening the old trial',async()=>{
 const item={id:'t1',revision:2,step_id:'s1',status:'reported',attempt:'not_tried',result_text:'Je n’ai pas essayé.',outcome:'worse',occurred_on:null,planned_on:null};
 const env=environment((path,method)=>method==='GET'&&path!=='/api/plan'?{revision:4,epoch:1,item}:empty());await env.ctx.changeOpenEditor('trials','t1',null,()=>true,()=>{});const dialog=env.dialog();assert.equal(field(dialog,'status').children.length,1);assert.equal(field(dialog,'occurred_on').value,'');await submit(dialog);
 const write=env.calls.at(-1);assert.equal(write.method,'PATCH');assert.equal(write.data.attempt,'not_tried');assert.equal(write.data.outcome,'worse');assert.equal(write.data.occurred_on,null);assert.equal(write.data.status,'reported');
 await env.ctx.changeOpenEditor('trials',null,'s1',()=>true,()=>{});assert.equal(env.calls.at(-1).path,'/api/plan');assert.equal(field(env.dialog(),'status').value,'planned');
});

test('personal preparation excludes conversation notes unless the member explicitly adds them',async()=>{
 const prepared={revision:4,epoch:1,content:'Mon point personnel',dependencies:[],memory_revision:9,memory_notes:[{text:'Une note à vérifier'}],memory_notice:'Vérifiez cette note.'};
 const env=environment((path,method)=>method==='GET'?prepared:empty());await env.ctx.changeOpenPreparation(()=>true,()=>{});let dialog=env.dialog();assert.equal(field(dialog,'content').value,'Mon point personnel');await submit(dialog);assert.equal(env.calls.at(-1).data.include_memory,false);
 await env.ctx.changeOpenPreparation(()=>true,()=>{});dialog=env.dialog();const check=field(dialog,'memory_0');check.checked=true;await check.fire('change');assert.ok(field(dialog,'content').value.includes('Une note à vérifier'));await submit(dialog);assert.equal(env.calls.at(-1).data.include_memory,true);assert.equal(env.calls.at(-1).data.memory_revision,9);
 assert.ok(env.calls.every(call=>!call.path.includes('/api/care')));
});

test('collection pagination fetches the next page and labels are plain text',async()=>{
 const first=empty();first.trials=[{id:'t1',revision:1,status:'reported',step_id:'s1',step_snapshot:{text:'<img src=x onerror=alert(1)>'},result_text:'Retour'}];first.next_offsets.trials=50;first.counts.trials=51;
 const env=environment(path=>path.includes('collection=trials')?{trials:[{id:'t2',status:'planned',step_id:'s1',step_snapshot:{text:'Deuxième essai'}}],next_offsets:{trials:null}}:first),host=env.host();await env.ctx.renderChangePlan(host,()=>true);assert.ok(host.textContent.includes('<img src=x onerror=alert(1)>'));assert.equal(host.querySelector('img'),null);await button(host,'Voir les éléments suivants').fire('click');assert.equal(env.calls.at(-1).path,'/api/plan?collection=trials&offset=50');assert.ok(host.textContent.includes('Deuxième essai'));
});

test('chat deletion has a separate unchecked plan deletion choice and restrictive followup stop needs no revision',async()=>{
 const env=environment(()=>({retained:'Votre plan adopté est conservé.'}));env.ctx.openChangeChatClear(()=>true);const dialog=env.dialog();assert.equal(field(dialog,'include_plan').checked,false);assert.ok(dialog.textContent.includes('Par défaut, votre plan adopté'));await submit(dialog);assert.deepEqual(env.calls[0],{path:'/api/chat',method:'DELETE',data:{include_plan:false}});
 await env.ctx.changePauseFollowups();assert.deepEqual(env.calls.at(-1),{path:'/api/plan/followups',method:'PUT',data:{enabled:false,target_id:'all'}});
});

test('a chat stop command displays its server notice without keeping an optimistic message pending',async()=>{
 const env=environment(()=>({followups_paused:true,messages:[],access:{allowed:true}}));
 env.ctx.state.chatData={access:{allowed:true}};env.ctx.chatIsCurrent=()=>true;env.ctx.drawChat=()=>{};env.ctx.updateChatControls=()=>{};env.ctx.status=()=>{};
 vm.runInContext(CHAT_INTERFACE_SCRIPT.slice(CHAT_INTERFACE_SCRIPT.indexOf('async function sendChat('),CHAT_INTERFACE_SCRIPT.indexOf('function retryChatMessage(')),env.ctx);
 const host=env.host();host.id='view';const form=env.doc.createElement('form'),input=env.doc.createElement('textarea');input.name='content';input.value='Ne me relancez plus';form.append(input);
 await env.ctx.sendChat(form,()=>true);assert.equal(env.calls.length,1);assert.equal(env.ctx.state.chatSending,false);assert.equal(env.ctx.state.chatOutbox,null);assert.match(env.ctx.state.chatFollowupNotice,/sans appel à Ambre/);
});

test('accepting a proposal and authorizing followups are separate explicit controls',async()=>{
 const proposal={id:'p2',revision:1,status:'proposed',kind:'step',payload:{text:'Sortir cinq minutes',measure:'Comment cela se passe'}};
 const env=environment((path,method)=>method==='GET'?{revision:4,epoch:1,item:proposal}:empty());await env.ctx.changeProposalDialog('p2',()=>true,()=>{},false,false);const dialog=env.dialog(),create=field(dialog,'create_trial');create.checked=true;await create.fire('change');assert.equal(field(dialog,'followup_opt_in').checked,false);await submit(dialog);assert.equal(env.calls.at(-1).data.create_trial,true);assert.equal(env.calls.at(-1).data.followup_opt_in,false);
});
