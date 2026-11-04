import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { HTML } from '../src/ui.mjs';

const script = HTML.match(/<script>([\s\S]*?)<\/script>/)[1];
function functionCode(name) {
  const start = script.search(new RegExp('(?:async )?function '+name+'\\('));
  assert.ok(start >= 0, name);
  const tail = script.slice(start);
  for (const match of tail.matchAll(/\n(?:async )?function |\nvar |\nwindow\.|\ndocument\.|\nrefresh\(/g)) {
    const candidate=tail.slice(0,match.index);
    try { new vm.Script(candidate); return candidate; } catch {}
  }
  return tail;
}
class Element {
  constructor(tag, doc) { this.tagName=tag.toUpperCase(); this.doc=doc; this.children=[]; this.listeners={}; this.dataset={}; this.attributes={}; this.value=''; this.checked=false; this.disabled=false; this.hidden=false; this.isConnected=true; this._text=''; }
  set type(value) { if(this.tagName==='TEXTAREA')throw new TypeError('textarea.type is readonly');this._type=value; }
  get type() { return this._type; }
  set id(value) { this._id=value; this.doc.ids.set(value,this); }
  get id() { return this._id; }
  append(...items) { for (const item of items) { if(typeof item==='object'&&item.parentElement){const old=item.parentElement.children;const index=old.indexOf(item);if(index>=0)old.splice(index,1);}this.children.push(item); if(typeof item==='object')item.parentElement=this; } }
  replaceChildren(...items) { this.children=[];this.append(...items); }
  set textContent(value) { this._text=String(value);this.children=[]; }
  get textContent() { return this._text + this.children.map(x=>typeof x==='string'?x:x.textContent).join(''); }
  setAttribute(name,value) { this.attributes[name]=value; }
  removeAttribute(name) { delete this.attributes[name]; }
  getAttribute(name) { return this.attributes[name]; }
  remove() {if(this.parentElement){const siblings=this.parentElement.children;const index=siblings.indexOf(this);if(index>=0)siblings.splice(index,1);this.parentElement=null;}this.isConnected=false;}
  scrollIntoView() {}
  addEventListener(name,fn) { (this.listeners[name]??=[]).push(fn); }
  async fire(name,extra={}) { for (const fn of this.listeners[name]||[]) await fn({target:this,currentTarget:this,preventDefault(){},...extra}); }
  focus() {this.doc.activeElement=this;}
  matches(selector) {
    if(selector==='input,select,textarea')return ['INPUT','SELECT','TEXTAREA'].includes(this.tagName);
    if(selector==='[data-reload-source]')return this.dataset.reloadSource!==undefined;
    if(selector==='[data-continue-character]')return this.dataset.continueCharacter!==undefined;
    if(selector==='[data-map-mode]')return this.dataset.mapMode!==undefined;
    if(selector==='button[type=submit]')return this.tagName==='BUTTON'&&this.type==='submit';
    if(selector==='[name=people_ids]:checked')return this.name==='people_ids'&&this.checked;
    if(selector.startsWith('[name='))return this.name===selector.slice(6,-1);
    return selector.startsWith('.')?String(this.className||'').split(' ').includes(selector.slice(1)):this.tagName===selector.toUpperCase();
  }
  querySelectorAll(selector) { return this.children.flatMap(x=>typeof x==='object'?[...(x.matches(selector)?[x]:[]),...x.querySelectorAll(selector)]:[]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0]||null; }
  get elements() { return Object.fromEntries(this.querySelectorAll('input,select,textarea').map(x=>[x.name,x])); }
}
function environment(extra={}) {
  const doc={ids:new Map(),createElement(tag){return new Element(tag,this)},createTextNode(text){return String(text)},getElementById(id){return this.ids.get(id)||null},querySelectorAll(selector){return [...this.ids.values()].filter(x=>x.matches(selector))},querySelector(selector){if(selector.startsWith('#')){const [id,...rest]=selector.slice(1).split(' ');return this.getElementById(id)?.querySelector(rest.join(' '))||null;}return this.querySelectorAll(selector)[0]||null}};
  class FormDataMock { constructor(form){this.values=form.querySelectorAll('input,select,textarea').filter(x=>x.name&&!x.disabled&&(x.type!=='checkbox'||x.checked)).map(x=>[x.name,x.value]);}entries(){return this.values.values();} }
  const ctx=vm.createContext({document:doc,FormData:FormDataMock,URL,URLSearchParams,crypto:{randomUUID:()=> 'create-uuid'},state:{people:[],events:[],sourceDrafts:new Map(),continueKeys:new Map(),tab:'journal',characterData:{},characters:[]},renderTokens:new WeakMap(),confirm:()=>true,refreshRecommendations(){},status(){},clearTimeout(){},setTimeout(){return 1},console,...extra});
  for (const name of ['renderGuard','el','actionButton','formValues','busy','sourceField','newClientId','captureSourceDraft','sourceDate','drawSourceEditor','characterSearch','requestIsActive','ambrePrompt','chatConsent','chatFailureMessage','applyChatAvailability','scheduleChatCooldown']) vm.runInContext(functionCode(name),ctx);
  vm.runInContext(script.match(/var PERSON_ROLES=.*\n/)[0]+script.match(/var EVENT_KINDS=.*\n/)[0],ctx);
  ctx.drawSourceSection=()=>{};ctx.sourceSaved=()=>{};
  return {ctx,doc,host(id){const host=doc.createElement('div');host.id=id;return host;}};
}
function editFixture(kind,values,extra={}) { const env=environment(extra);env.host(kind+'-editor');env.ctx.state.sourceDrafts.set(kind,{id:null,creation_id:'creation-stable',values});env.ctx.drawSourceEditor(kind);return {...env,form:env.doc.getElementById(kind+'-form')}; }
function field(form,name,value) {const input=form.elements[name];assert.ok(input,name);input.value=value;return input;}

test('the generated application script parses, and canonical URLs keep both identities',()=>{
  new vm.Script(script);
  const {ctx}=environment();
  assert.equal(ctx.characterSearch('char & a','old/story?1'),'?personnage=char+%26+a&story=old%2Fstory%3F1');
  assert.equal(ctx.requestIsActive({status:'waiting_portrait'}),true);
  assert.equal(ctx.requestIsActive({status:'completed'}),false);
  assert.match(functionCode('renderLikes'),/legacy.id='suggestions';legacy.tabIndex=-1/);
  assert.doesNotMatch(HTML,/Mes histoires|une histoire chaque jour|quotidiennes/);
});

test('a retry retains the person creation key and full unsaved text after a failed write',async()=>{
  const calls=[];const env=editFixture('people',{display_name:'N.',role:'friend',relationship_text:'Un texte personnel complet.'},{api:async(...args)=>{calls.push(args);throw new Error('Réseau indisponible')}});
  await env.form.fire('submit');
  assert.equal(env.ctx.state.sourceDrafts.get('people').creation_id,'creation-stable');
  assert.equal(env.form.elements.relationship_text.value,'Un texte personnel complet.');
  await env.form.fire('submit');
  assert.equal(calls.length,2);
  assert.equal(calls[0][2].creation_id,calls[1][2].creation_id);
  assert.equal(calls[0][2].relationship_text,'Un texte personnel complet.');
});

test('life-event form preserves an approximate age, associations and understanding-only choice',async()=>{
  let sent;const env=editFixture('events',{title:'Un tournant',kind:'turning_point',date_precision:'age',age_text:'Vers 15 ans',date_start:'2020',description:'Mon ressenti.',people_ids:['p1'],fiction_use:'understanding_only'},{api:async(path,method,payload)=>{sent=payload;throw new Error('offline')}});
  const form=env.form; // Simulate a selected checkbox exactly as FormData exposes it.
  const person=env.doc.createElement('input');person.type='checkbox';person.name='people_ids';person.value='p1';person.checked=true;form.append(person);
  await form.fire('submit');
  assert.equal(sent.age_text,'Vers 15 ans');assert.equal(sent.date_start,'');assert.equal(sent.date_end,'');
  assert.equal(sent.fiction_use,'understanding_only');assert.deepEqual(Array.from(sent.people_ids),['p1']);assert.equal(sent.current_importance,null);
  assert.equal(env.ctx.sourceDate(sent),'Vers 15 ans');
});

test('saving a year never invents a month or day, and changing precision updates required fields',async()=>{
  let sent;const env=editFixture('events',{title:'2020',kind:'positive',date_precision:'unknown',description:'Un souvenir',people_ids:[],fiction_use:'abstract'},{api:async(p,m,d)=>{sent=d;throw new Error('offline')}});
  const precision=field(env.form,'date_precision','year');await precision.fire('change');field(env.form,'date_start','2020');
  assert.equal(env.form.elements.date_start.required,true);assert.equal(env.form.elements.age_text.required,false);
  await env.form.fire('submit');assert.equal(sent.date_start,'2020');assert.equal(sent.date_precision,'year');assert.equal(sent.age_text,'');
});

test('revision conflicts do not overwrite an edit draft',async()=>{
  const env=editFixture('people',{display_name:'Alias',role:'parent',relationship_text:'Mon brouillon'},{api:async()=>{const error=new Error('conflit');error.code='revision_conflict';throw error}});
  env.ctx.state.sourceDrafts.set('people',{id:'p1',revision:3,values:{display_name:'Alias',role:'parent',relationship_text:'Mon brouillon'}});env.ctx.drawSourceEditor('people');const form=env.doc.getElementById('people-form');
  await form.fire('submit');assert.equal(env.ctx.state.sourceDrafts.get('people').revision,3);assert.equal(env.ctx.state.sourceDrafts.get('people').values.relationship_text,'Mon brouillon');assert.match(form.textContent,/brouillon est conservé/);
});

test('text typed while saving remains a draft with the newly saved revision',async()=>{
  let resolve;const request=new Promise(r=>resolve=r);const env=editFixture('people',{display_name:'Alias',role:'friend',relationship_text:'Avant'},{api:()=>request});
  const pending=env.form.fire('submit');field(env.form,'relationship_text','Après la demande');await env.form.fire('input');resolve({person:{id:'p1',revision:1,display_name:'Alias',relationship_text:'Avant'},source_version:8});await pending;
  const draft=env.ctx.state.sourceDrafts.get('people');assert.equal(draft.id,'p1');assert.equal(draft.revision,1);assert.equal(draft.values.relationship_text,'Après la demande');assert.equal(draft.creation_id,undefined);
});

test('continuation retries use the same idempotency key and latest parent, never the episode being read',async()=>{
  const calls=[];const env=environment({api:async(...args)=>{calls.push(args);throw new Error('network')}});env.host('view');env.host('character-preparation');env.ctx.drawCharacterCards=()=>{};env.ctx.renderQuota=()=>{};env.ctx.drawRequestState=()=>{};
  vm.runInContext(functionCode('continueCharacter'),env.ctx);env.ctx.state.selected='old-story';const button=env.doc.createElement('button'),character={id:'c1',last_story_id:'latest-story'};
  await env.ctx.continueCharacter(character,button);await env.ctx.continueCharacter(character,button);
  assert.equal(calls.length,2);assert.equal(calls[0][2].idempotency_key,calls[1][2].idempotency_key);assert.equal(calls[0][2].expected_parent_id,'latest-story');
});

test('Netlify proxies both canonical forms with private no-store and keeps legacy query paths',()=>{
  const config=readFileSync(new URL('../../netlify.toml',import.meta.url),'utf8');
  for(const path of ['/mes-personnages','/mes-personnages/*','/mes-histoires','/journal'])assert.ok(config.includes('from = "'+path+'"'));
  assert.match(config,/for = "\/mes-personnages"[\s\S]*?Cache-Control = "private, no-store"/);
  assert.match(config,/for = "\/mes-personnages\/\*"[\s\S]*?Cache-Control = "private, no-store"/);
});

test('a late source-save response cannot replace another editor opened in the same section',async()=>{
  let resolve;const request=new Promise(r=>resolve=r);const env=editFixture('people',{display_name:'Premier',role:'friend',relationship_text:'Avant'},{api:()=>request});
  const pending=env.form.fire('submit');env.ctx.state.sourceDrafts.set('people',{id:'other-person',revision:2,values:{display_name:'Autre',role:'parent',relationship_text:'Autre brouillon'}});env.ctx.drawSourceEditor('people');
  resolve({person:{id:'saved-person',revision:1,display_name:'Premier'},source_version:10});await pending;
  assert.equal(env.ctx.state.sourceDrafts.get('people').id,'other-person');assert.equal(env.doc.getElementById('people-form').elements.relationship_text.value,'Autre brouillon');
});

test('a second continuation action is ignored while the first response is pending',async()=>{
  let reject;const request=new Promise((resolve,r)=>reject=r),calls=[];const env=environment({api:async(...args)=>{calls.push(args);return request}});env.host('view');env.host('character-preparation');env.ctx.drawCharacterCards=()=>{};
  vm.runInContext(functionCode('continueCharacter'),env.ctx);
  const first=env.ctx.continueCharacter({id:'c1',last_story_id:null},env.doc.createElement('button'));
  await env.ctx.continueCharacter({id:'c2',last_story_id:null},env.doc.createElement('button'));
  assert.equal(calls.length,1);reject(new Error('network'));await first;assert.equal(env.ctx.state.characterRequestPending,false);
});

test('V3 separates relationship and life pages from the declared and interpreted portrait',()=>{
  const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('pageForPath'),env.ctx);
  assert.equal(env.ctx.pageForPath('/mes-proches'),'people');assert.equal(env.ctx.pageForPath('/ma-ligne-de-vie/'),'events');assert.equal(env.ctx.pageForPath('/mon-personnage/'),'journal');
  assert.doesNotMatch(functionCode('renderPortrait'),/renderPersonalSources/);
  assert.match(functionCode('renderPortrait'),/renderSelfProfile/);
  assert.match(HTML,/À propos de moi/);assert.match(HTML,/Ma synthèse/);
  assert.doesNotMatch(HTML,/Trois personnages|trois personnages|\[1,2,3\]\.forEach/);
});

test('self presentation retains the full draft and original revision on 409 conflict',async()=>{
  let sent;const env=environment({api:async(path,method,payload)=>{if(!method)return {profile:{self_description:'Ma présentation',occupation:'Mon métier'},revision:4};sent=payload;const error=new Error('conflit');error.status=409;throw error;},location:{hash:''}});
  env.host('self-profile');vm.runInContext(functionCode('status')+'\n'+functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();
  const form=env.doc.getElementById('self-profile-form');field(form,'self_description','Mon texte personnel complet');await form.fire('input');await form.fire('submit');
  assert.equal(sent.expected_revision,4);assert.equal(sent.self_description,'Mon texte personnel complet');assert.equal(sent.occupation,'Mon métier');
  assert.equal(env.ctx.state.selfDraft.self_description,'Mon texte personnel complet');assert.equal(env.ctx.state.selfRevision,4);
  assert.match(env.doc.getElementById('self-profile-status').textContent,/brouillon est conservé/);
  assert.equal(form.elements.self_description.maxLength,10000);assert.equal(form.elements.current_context.maxLength,4000);assert.equal(form.elements.gender.required,undefined);
  assert.equal('relationship_text' in sent,false);assert.equal('consent' in sent,false);
});

test('self presentation keeps edits typed during a save with the new revision',async()=>{
  let resolve;const pending=new Promise(r=>resolve=r);const env=environment({api:async(path,method)=>method?pending:{profile:{self_description:'Avant'},revision:2},location:{hash:''}});
  env.host('self-profile');vm.runInContext(functionCode('status')+'\n'+functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();
  const form=env.doc.getElementById('self-profile-form');const saving=form.fire('submit');field(form,'self_description','Après');await form.fire('input');resolve({profile:{self_description:'Avant'},revision:3});await saving;
  assert.equal(env.ctx.state.selfRevision,3);assert.equal(env.ctx.state.selfDraft.self_description,'Après');assert.match(env.doc.getElementById('self-profile-status').textContent,/restent à sauvegarder/);
});

test('one active character is gated by readiness and archives have no continuation action',()=>{
  const env=environment();env.host('character-cards');Object.assign(env.ctx.state,{me:{subscription:{active:true}},archivedCharacters:[{id:'old'}],characters:[{id:'active',identity:{name:'Nina'}},{id:'ignored'}],selectedCharacter:'active',characterData:{readiness:{eligible:false},quota:{remaining:7}}});
  vm.runInContext(functionCode('characterName')+'\n'+functionCode('characterEligible')+'\n'+functionCode('drawCharacterCards'),env.ctx);
  env.ctx.drawCharacterCards();let cards=env.doc.getElementById('character-cards');assert.equal(cards.children.length,1);assert.equal(cards.querySelector('[data-continue-character]').disabled,true);
  env.ctx.state.characterData.readiness.eligible=true;env.ctx.drawCharacterCards();assert.equal(cards.querySelector('[data-continue-character]').disabled,false);
  env.ctx.state.selectedCharacter='old';env.ctx.drawCharacterCards();assert.equal(cards.querySelector('[data-continue-character]'),null);assert.match(cards.textContent,/Revenir à mon personnage actif/);
});

test('V4 global navigation has exactly the shared public links in source order',()=>{
  const navigation=JSON.parse(readFileSync(new URL('../../data/navigation.json',import.meta.url),'utf8'));
  const menu=HTML.match(/<ul class="nav-links">([\s\S]*?)<\/ul>/)[1];
  const links=[...menu.matchAll(/<a href="([^"]+)">([^<]+)<\/a>/g)].map(match=>({href:match[1],label:match[2]}));
  assert.deepEqual(links,navigation);assert.equal(links.length,7);assert.doesNotMatch(menu,/Se connecter|Se déconnecter/);
  assert.match(readFileSync(new URL('../../layouts/partials/header.html',import.meta.url),'utf8'),/\.Site\.Data\.navigation/);
});

test('locked character page only renders prerequisites and never fetches a legacy story',async()=>{
  const calls=[];const env=environment({api:async(path)=>{calls.push(path);return {characters:[{id:'old',identity:{name:'Alex'}}],archived_characters:[{id:'archive'}],readiness:{eligible:false,likes_count:2,target:10}}},capturePageDrafts(){}});
  const host=env.host('view');Object.assign(env.ctx.state,{me:{subscription:{active:false}},selected:'old-story',selectedCharacter:null});
  vm.runInContext(functionCode('characterEligible')+'\n'+functionCode('characterProgress')+'\n'+functionCode('renderJournal'),env.ctx);
  await env.ctx.renderJournal();assert.deepEqual(calls,['/api/characters']);assert.match(host.textContent,/2 \/ 10 articles aimés/);assert.doesNotMatch(host.textContent,/Alex|archives|quota|première histoire/i);assert.equal(env.ctx.state.characters[0].id,'old');
});

test('portrait rendering no longer exposes the obsolete fictional character card',()=>{
  assert.doesNotMatch(functionCode('appendPortrait'),/fictional_character/);
  assert.doesNotMatch(functionCode('renderAccount'),/10 €|likes_count>=10|Aimez au moins 10/);
  assert.match(functionCode('renderAccount'),/29 €/);assert.match(functionCode('renderAccount'),/billing\/cancel/);
});

test('chat network retries keep the same message key and complete draft',async()=>{
  const calls=[];const env=environment({api:async(...args)=>{calls.push(args);throw new Error('offline')}});
  Object.assign(env.ctx.state,{chatDraft:'',chatSending:false,chatSendKey:null});vm.runInContext(functionCode('sendChat'),env.ctx);
  const form=env.doc.createElement('form'),input=env.doc.createElement('textarea'),send=env.doc.createElement('button');input.name='content';input.value='Mon texte complet et sensible';send.type='submit';form.append(input,send);
  await env.ctx.sendChat(form,()=>true);await env.ctx.sendChat(form,()=>true);
  assert.equal(calls.length,2);assert.equal(calls[0][2].idempotency_key,calls[1][2].idempotency_key);assert.equal(env.ctx.state.chatDraft,input.value);assert.equal(send.disabled,false);
});

test('chat proposals submit edited visible fields and preserve unedited structured data',async()=>{
  let sent;const env=environment({api:async(path,method,payload)=>{sent={path,payload};throw new Error('conflict')}});env.ctx.state.chatProposalDrafts=new Map();env.ctx.drawChatPeople=()=>{};vm.runInContext(functionCode('drawChatProposal'),env.ctx);
  const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'p1',kind:'life_event_create',revision:3,status:'pending',payload:{title:'Avant',description:'Détail',people_ids:['person1'],current_importance:2}},()=>true);
  const form=host.children[0],title=field(form,'title','Mon titre corrigé');await title.fire('input');await form.fire('submit');
  assert.equal(sent.path,'/api/chat/proposals/p1/accept');assert.equal(sent.payload.revision,3);assert.equal(sent.payload.payload.title,'Mon titre corrigé');assert.deepEqual(Array.from(sent.payload.payload.people_ids),['person1']);assert.equal(sent.payload.payload.current_importance,2);assert.equal(env.ctx.state.chatProposalDrafts.get('p1').title,'Mon titre corrigé');assert.match(form.textContent,/modifications sont conservées/);
});

test('quota exhaustion retains conversation, proposals and privacy actions while blocking sends',()=>{
  const env=environment();env.ctx.state.chatProposalDrafts=new Map();env.ctx.state.chatData={consent:true,access:{allowed:false,reason:'quota_exhausted'},quota:{remaining:0,limit:100},messages:[{role:'user',content:'Mon historique privé'}],proposals:[{id:'proposal',status:'pending'}]};env.ctx.drawChatProposal=(host)=>host.append(env.doc.createElement('form'));vm.runInContext(functionCode('chatConsent')+'\n'+functionCode('drawChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);
  assert.match(host.textContent,/Mon historique privé/);assert.match(host.textContent,/Effacer la conversation et désactiver Ambre/);assert.doesNotMatch(host.textContent,/Voir mon abonnement/);assert.equal(env.doc.getElementById('chat-proposals').children.length,1);assert.equal(env.doc.getElementById('chat-form').querySelector('button').disabled,true);assert.equal(env.doc.getElementById('chat-content').maxLength,6000);
});

test('confirmed terminal failure renews message key, unlike ambiguous network failure',async()=>{
  let scheduled;const env=environment({clearTimeout(){},setTimeout(fn){scheduled=fn;return 1},api:async()=>({last_request:{id:'r1',status:'failed'},last_error:{code:'provider_error'}})});const form=env.host('chat-form'),button=env.doc.createElement('button');button.type='submit';form.append(button);Object.assign(env.ctx.state,{tab:'chat',chatDraft:'Mon brouillon',chatSendKey:{key:'old-key',content:'Mon brouillon'},chatSending:true});vm.runInContext(functionCode('pollChat'),env.ctx);env.ctx.pollChat({id:'r1'},()=>true,'Mon brouillon');await scheduled();assert.equal(env.ctx.state.chatSendKey,null);assert.equal(env.ctx.state.chatDraft,'Mon brouillon');assert.equal(button.disabled,false);
});

test('proposal citations are plain text, before values visible and person associations editable',async()=>{
  const env=environment({api:async()=>({people:[{id:'p1',display_name:'Camille'},{id:'p2',display_name:'Sam'}]})});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatPeople')+'\n'+functionCode('drawChatProposal'),env.ctx);const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'x',kind:'life_event_update',revision:1,status:'pending',evidence_quote:'<script>citation privée</script>',before:{title:'Avant'},payload:{title:'Après',people_ids:['p1']}},()=>true);await new Promise(resolve=>setImmediate(resolve));assert.match(host.textContent,/<script>citation privée<\/script>/);assert.equal(host.querySelector('script'),null);assert.match(host.textContent,/Titre : Avant/);const checkboxes=host.querySelectorAll('input');assert.equal(checkboxes.length,2);checkboxes[0].checked=false;await checkboxes[0].fire('change');checkboxes[1].checked=true;await checkboxes[1].fire('change');assert.deepEqual(Array.from(env.ctx.state.chatProposalDrafts.get('x').people_ids),['p2']);
});

test('chat ignores duplicate sends during latency and keeps text typed while a request is pending',async()=>{
  let resolve;const pending=new Promise(r=>resolve=r),calls=[];const env=environment({api:async(...args)=>{calls.push(args);return pending},pollChat(){}});Object.assign(env.ctx.state,{chatDraft:'Message initial',chatSending:false,chatSendKey:null,chatData:{access:{allowed:true}}});vm.runInContext(functionCode('sendChat'),env.ctx);const form=env.doc.createElement('form'),input=env.doc.createElement('textarea'),button=env.doc.createElement('button');input.name='content';input.value='Message initial';button.type='submit';form.append(input,button);const sending=env.ctx.sendChat(form,()=>true);input.value='Mon prochain message';env.ctx.state.chatDraft=input.value;await env.ctx.sendChat(form,()=>true);assert.equal(calls.length,1);resolve({pending_request:{id:'pending',status:'running'},request_id:'pending'});await sending;assert.equal(env.ctx.state.chatDraft,'Mon prochain message');assert.equal(env.ctx.state.chatSending,true);assert.equal(button.disabled,true);
});


test('proposal importance is numeric from null and clearing restores null',async()=>{
  const env=environment();env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'importance',kind:'life_event_create',revision:1,payload:{current_importance:null}},()=>true);const input=field(host.children[0],'current_importance','4');await input.fire('input');assert.equal(env.ctx.state.chatProposalDrafts.get('importance').current_importance,4);input.value='';await input.fire('input');assert.equal(env.ctx.state.chatProposalDrafts.get('importance').current_importance,null);
});


test('V5 removes the global likes badge and every stale badge update',()=>{
  assert.doesNotMatch(functionCode('renderApp'),/plan-status|likes_count/);
  assert.doesNotMatch(functionCode('renderLikes'),/plan-status|progressBlock/);
});

test('V5 empty conversation renders no invented message and places quota in the page tools',()=>{
  const env=environment();env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:99,limit:100},messages:[],proposals:[]};
  env.host('chat-quota');env.ctx.drawChatProposal=()=>{};
  vm.runInContext(functionCode('chatConsent')+'\n'+functionCode('drawChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);
  assert.equal(host.querySelector('.chat-thread').children.length,0);
  assert.equal(env.doc.getElementById('chat-quota').textContent,'99 / 100 messages');
  assert.equal(env.doc.getElementById('chat-form').querySelector('button').disabled,false);
  assert.doesNotMatch(host.textContent,/Un espace pour prendre du recul|Vous pouvez commencer par une situation/);
});

test('V5 blocked sends retain editable draft, history and proposals for all access reasons',()=>{
  for(const reason of ['consent_required','subscription_required','quota_exhausted','provider_missing','generation_paused']){
    const env=environment();env.ctx.state.chatDraft='Mon brouillon';env.ctx.state.chatData={consent:reason!=='consent_required',access:{allowed:false,reason},quota:{remaining:reason==='quota_exhausted'?0:100,limit:100},messages:[{role:'assistant',content:'Historique conservé'}],proposals:[{status:'pending'}]};
    env.ctx.drawChatProposal=host=>host.append(env.doc.createElement('form'));
    vm.runInContext(functionCode('chatConsent')+'\n'+functionCode('drawChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);
    assert.match(host.textContent,/Historique conservé/);assert.match(host.textContent,/Effacer la conversation et désactiver Ambre/);
    assert.equal(env.doc.getElementById('chat-proposals').children.length,1);
    assert.equal(env.doc.getElementById('chat-content').value,'Mon brouillon');assert.equal(env.doc.getElementById('chat-form').querySelector('button').disabled,true);
  }
});

test('V5 portrait panes respect deep links and switching preserves form nodes',async()=>{
 for(const [hash,key] of [['#ma-cartographie','map'],['#cartographie','map'],['#self-profile','self'],['#mes-articles','readings'],['#suggestions','readings'],['#correction-form','synthesis'],['','synthesis']]){
  const location={hash,pathname:'/mon-portrait/',search:'?article=old'};const env=environment({location,history:{replaceState(a,b,url){location.hash=url.includes('#')?url.slice(url.indexOf('#')):''}}});
  env.ctx.state.pendingArticle=hash?'old':null;const view=env.host('view'),panes=['synthesis','self','readings','map'].map(key=>{const e=env.doc.createElement('section');e.dataset.portraitPane=key;view.append(e);return e}),tabs=['synthesis','self','readings','map'].map((key,i)=>{const e=env.doc.createElement('a');e.dataset.portraitTab=key;e.getAttribute=()=>['#synthese','#self-profile','#mes-articles','#ma-cartographie'][i];view.append(e);return e});
  const form=env.doc.createElement('form');panes[1].append(form);
  view.querySelectorAll=selector=>selector==='[data-portrait-pane]'?panes:tabs;
  vm.runInContext(functionCode('setupPortraitViews'),env.ctx);env.ctx.setupPortraitViews(view);
  assert.equal(panes.find(p=>!p.hidden).dataset.portraitPane,key);
  await tabs[1].fire('click');assert.equal(panes[1].hidden,false);assert.equal(panes[1].children[0],form);
  await tabs[2].fire('click');assert.equal(panes[2].hidden,false);assert.equal(panes[1].children[0],form);
 }
});


test('V5 an immediate idempotent chat response leaves the new composer enabled',async()=>{
 const env=environment({api:async()=>({consent:true,access:{allowed:true},quota:{remaining:99,limit:100},messages:[],proposals:[]})});
 Object.assign(env.ctx.state,{chatDraft:'Bonjour',chatSending:false,chatData:{access:{allowed:true}}});env.ctx.drawChatProposal=()=>{};
 vm.runInContext(functionCode('chatConsent')+'\n'+functionCode('drawChat')+'\n'+functionCode('sendChat'),env.ctx);
 env.host('view');const form=env.doc.createElement('form'),input=env.doc.createElement('textarea'),button=env.doc.createElement('button');input.name='content';input.value='Bonjour';button.type='submit';form.append(input,button);
 await env.ctx.sendChat(form,()=>true);
 assert.equal(env.doc.getElementById('chat-form').querySelector('button').disabled,false);assert.equal(env.ctx.state.chatSending,false);
});

test('V5 retrying a pending turn keeps one visible user message and its draft',async()=>{
 const env=environment({api:async()=>({pending_request:{id:'same-turn',status:'running'}}),pollChat(){}});
 Object.assign(env.ctx.state,{chatDraft:'Un seul message',chatSending:false,chatData:{access:{allowed:true}}});vm.runInContext(functionCode('sendChat'),env.ctx);
 const thread=env.host('thread');thread.className='chat-thread';const form=env.doc.createElement('form'),input=env.doc.createElement('textarea'),button=env.doc.createElement('button');input.name='content';input.value='Un seul message';button.type='submit';form.append(input,button);
 await env.ctx.sendChat(form,()=>true);env.ctx.state.chatSending=false;env.ctx.state.chatRequest=null;await env.ctx.sendChat(form,()=>true);
 assert.equal(thread.children.length,1);assert.equal(thread.children[0].dataset.chatTurnId,'same-turn');assert.equal(input.value,'Un seul message');
});


test('V6 self information and reflection groups open with all fields in their own compact grids',async()=>{
 const env=environment({api:async()=>({profile:{},revision:0}),location:{hash:''}});env.host('self-profile');vm.runInContext(functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();
 const form=env.doc.getElementById('self-profile-form'),groups=form.querySelectorAll('details');assert.equal(groups.length,2);assert.ok(groups.every(group=>group.open));
 assert.equal(env.doc.getElementById('self-identity-fields').elements.occupation.tagName,'INPUT');assert.equal(env.doc.getElementById('self-description-fields').elements.self_description.tagName,'TEXTAREA');
 assert.deepEqual(Object.keys(env.doc.getElementById('self-reflection-fields').elements),['self_perception','recurring_patterns','emotions','current_context']);assert.equal(form.querySelectorAll('textarea').length,5);
 assert.match(HTML,/#self-profile-form \.self-presentation textarea\{min-height:160px/);
});

test('V6 application contains no footer element and labels its liked-article tab directly',()=>{
 assert.doesNotMatch(HTML,/<footer\b/);assert.match(functionCode('renderPortrait'),/data-portrait-tab="readings">Articles aimés/);
 assert.doesNotMatch(functionCode('renderLikes'),/likes-search|likes-character|likes-keyword|likes-more|recommendations-body|À découvrir/);
});

function likedFixture(api){const env=environment({api,location:{origin:'https://example.test',hash:''},refreshPortraitData(){}});Object.assign(env.ctx.state,{tab:'portrait',me:{},likes:[{article_id:'a',reason:'Mon ancien retour',differences:'Une différence ancienne'},{article_id:'b',reason:''}],examples:[{id:'a',title:'Article A',url:'/a/'},{id:'b',title:'Article B',url:'/b/'},{id:'c',title:'Non aimé',url:'/c/'}],drafts:new Map(),differencesDrafts:new Map()});vm.runInContext(functionCode('safeArticleLink')+'\n'+functionCode('updateLikeState')+'\n'+functionCode('drawLikedArticles'),env.ctx);const list=env.host('likes-list');env.ctx.drawLikedArticles(list,()=>true);return {...env,list};}

test('V6 liked articles are a consecutive title disclosure list with separate reading links and legacy returns preserved',async()=>{
 const calls=[];const env=likedFixture(async(path,method,data)=>{calls.push({path,method,data});return {likes:env.ctx.state.likes}});const cards=env.list.children;assert.equal(cards.length,2);assert.doesNotMatch(env.list.textContent,/Non aimé/);
 const first=cards[0],details=first.children[0];assert.equal(details.tagName,'DETAILS');assert.equal(details.children[0].textContent,'Article A');assert.equal(first.children[1].tagName,'A');assert.equal(first.children[1].textContent,'Lire');assert.equal(details.open,false);
 assert.equal(first.querySelectorAll('textarea').length,2);assert.equal(cards[1].querySelectorAll('textarea').length,1);assert.match(first.textContent,/Ce qui me parle et pourquoi je me reconnais/);
 const form=first.querySelector('form');field(form,'reason','Nouveau retour');await form.elements.reason.fire('input');await form.fire('submit');assert.equal(calls[0].data.reason,'Nouveau retour');assert.equal(calls[0].data.differences,'Une différence ancienne');
 await cards[1].querySelector('form').fire('submit');assert.equal('differences' in calls[1].data,false);
});

test('V6 liked article network failures keep both editable drafts',async()=>{
 const env=likedFixture(async()=>{throw new Error('offline')});const form=env.list.children[0].querySelector('form');field(form,'reason','Brouillon complet');await form.elements.reason.fire('input');field(form,'differences','Différence préservée');await form.elements.differences.fire('input');await form.fire('submit');
 assert.equal(env.ctx.state.drafts.get('a'),'Brouillon complet');assert.equal(env.ctx.state.differencesDrafts.get('a'),'Différence préservée');assert.match(form.textContent,/offline/);
});

const psychologyFixture={catalog_version:1,domains:[{id:'ties',label:'Mes relations'},{id:'self',label:'Mon regard sur moi'}],patterns:[{id:'belong',domain:'ties',label:'Besoin d’appartenance',description:'Chercher sa place.',signals:['Faire attention au groupe.'],questions:['Comment trouvez-vous votre place ?'],alternatives:['Un contexte nouveau.']},{id:'autonomy',domain:'self',label:'Besoin d’autonomie',description:'Choisir pour soi.',questions:['Quand choisissez-vous librement ?']}],hypotheses:[{pattern_id:'belong',status:'exploring',observation:'Vous évoquez votre place dans le groupe.',possible_function:'Préserver les liens.',evidence:['person:p1'],counter_evidence:['profile:self'],citations:[{source_id:'person:p1',quote:'Mes mots exacts'}],related_articles:[{title:'Une lecture',url:'/article/'},{title:'Lien refusé',url:'javascript:alert(1)'}],related_characters:[{name:'Nina',url:'/personnages/nina/'}],next_question:'Que souhaitez-vous préciser ?'}],source_version:4,analyzed_version:4,outdated:false};
function mapFixture(extra={}){const env=environment({refreshPortraitData(){},location:{origin:'https://example.test'},...extra});Object.assign(env.ctx.state,{tab:'portrait',likes:[],examples:[],mapReturnDrafts:new Map()});for(const name of ['normalized','personalizePortraitText','safeArticleLink','portraitEvidence','mapStateLabel','selectMapPattern','renderMapExplorer','drawMapPattern','drawMapReturn','drawPsychologyMap'])vm.runInContext(functionCode(name),env.ctx);const host=env.host('ma-cartographie');env.ctx.drawPsychologyMap(host,psychologyFixture,()=>true);return {...env,host};}

test('V6 psychology map distinguishes supported exploration from unknown patterns and links exact source evidence',()=>{
 const env=mapFixture(),known=env.host.mapCards.get('belong'),unknown=env.host.mapCards.get('autonomy');assert.equal(known.card.parentElement.id,'map-detail');assert.equal(known.card.hidden,false);assert.equal(unknown.card.hidden,true);assert.equal(known.row.hidden,false);assert.equal(unknown.row.hidden,true);
 assert.match(known.card.textContent,/À explorer/);assert.match(unknown.card.textContent,/Non exploré/);assert.doesNotMatch(env.host.textContent,/\d+\s*%|probabilité|absence de syndrome/i);
 assert.ok(known.card.querySelectorAll('a').some(link=>link.href==='/mes-proches/#person-p1'));assert.ok(known.card.querySelectorAll('a').some(link=>link.href==='https://example.test/article/'));assert.doesNotMatch(known.card.textContent,/Lien refusé/);
 assert.match(known.card.textContent,/Mes mots exacts/);assert.match(known.card.textContent,/Ce qui invite à nuancer/);assert.match(known.card.textContent,/Fonction possible/);assert.doesNotMatch(known.card.textContent,/Ce que cela pourrait vous apporter/);
});

test('V6 map refresh preserves open disclosures, reply form node and text while evidence changes',async()=>{
 const env=mapFixture(),entry=env.host.mapCards.get('belong'),form=entry.card.querySelector('form'),input=form.elements.text;entry.card.querySelector('.map-return').open=true;input.value='Un brouillon sensible';await input.fire('input');
 const update=structuredClone(psychologyFixture);update.hypotheses[0].observation='Observation mise à jour';env.ctx.drawPsychologyMap(env.host,update,()=>true);
 assert.equal(entry.card.querySelector('form'),form);assert.equal(input.value,'Un brouillon sensible');assert.equal(entry.card.querySelector('.map-return').open,true);assert.match(entry.card.textContent,/Observation mise à jour/);assert.equal(env.ctx.state.mapReturnDrafts.get('belong').text,'Un brouillon sensible');
});

test('V6 map feedback retries append an explicit typed correction with the same creation key',async()=>{
 const calls=[];const env=mapFixture({api:async(path,method,payload)=>{calls.push({path,method,payload});throw new Error('Réseau indisponible')}});const form=env.host.mapCards.get('belong').card.querySelector('form');field(form,'text','Cette piste ne correspond pas à mon vécu.');await form.elements.text.fire('input');field(form,'type','not_concerned');await form.elements.type.fire('change');await form.fire('submit');await form.fire('submit');
 assert.equal(calls.length,2);assert.equal(calls[0].path,'/api/portrait/feedback');assert.equal(calls[0].method,'POST');assert.equal(calls[0].payload.kind,'not_concerned');assert.equal(calls[0].payload.text,'Cette piste ne correspond pas à mon vécu.');assert.equal(calls[0].payload.creation_id,calls[1].payload.creation_id);assert.equal('label' in calls[0].payload,false);assert.equal(env.ctx.state.mapReturnDrafts.get('belong').text,form.elements.text.value);
});

test('V6 completed map feedback retains edits made during the write with a new creation key',async()=>{
 let resolve,sequence=0;const pending=new Promise(r=>resolve=r);const env=mapFixture({api:async(path)=>path==='/api/portrait/feedback'?pending:{status:'ready'},crypto:{randomUUID:()=> 'key-'+(++sequence)},applyPortraitData(){},schedulePortraitPoll(){}});const form=env.host.mapCards.get('belong').card.querySelector('form');field(form,'text','Avant');await form.elements.text.fire('input');const key=env.ctx.state.mapReturnDrafts.get('belong').creation_id;const saving=form.fire('submit');field(form,'text','Après');await form.elements.text.fire('input');resolve({memory_version:2});await saving;
 assert.equal(env.ctx.state.mapReturnDrafts.get('belong').text,'Après');assert.notEqual(env.ctx.state.mapReturnDrafts.get('belong').creation_id,key);
});

test('V6 portrait polling refreshes content without route rendering and stops after navigation',async()=>{
 const timers=[],calls=[],applied=[];const env=environment({setTimeout(fn){timers.push(fn);return timers.length},clearTimeout(){},api:async(path)=>{calls.push(path);return {status:'ready',source_version:5}},applyPortraitData(data){applied.push(data)}});env.ctx.state.tab='portrait';vm.runInContext(functionCode('schedulePortraitPoll'),env.ctx);env.ctx.schedulePortraitPoll(()=>true,{status:'queued'});assert.equal(timers.length,1);await timers.shift()();assert.deepEqual(calls,['/api/portrait']);assert.equal(applied.length,1);assert.equal(timers.length,0);
 env.ctx.schedulePortraitPoll(()=>true,{status:'updating'});env.ctx.state.tab='journal';await timers.shift()();assert.equal(calls.length,1);
});

test('V6 free preview permits an eligible character and does not bypass portrait prerequisites',()=>{
 const env=environment();vm.runInContext(functionCode('characterEligible')+'\n'+functionCode('characterProgress'),env.ctx);Object.assign(env.ctx.state,{me:{access:{mode:'free_preview',allowed:true},subscription:{active:false}},characterData:{readiness:{eligible:true}}});assert.equal(env.ctx.characterEligible(),true);env.ctx.state.characterData.readiness.eligible=false;assert.equal(env.ctx.characterEligible(),false);assert.match(env.ctx.characterProgress({readiness:{}}).textContent,/Accès gratuit ouvert/);assert.doesNotMatch(env.ctx.characterProgress({readiness:{}}).textContent,/abonnement actif/);
});

test('V6 free preview account displays access and quotas without checkout or subscription offers',()=>{
 const env=environment();const host=env.host('view'),accountRow=env.host('account-row');host.querySelector=()=>accountRow;for(const id of ['account-email','account-price','subscription-copy','account-progress','billing-action','export-button','delete-button'])env.host(id);env.ctx.state.me={user:{email:'test@example.test'},access:{mode:'free_preview',allowed:true},subscription:{active:false,trial_eligible:true},configuration:{billing:true,ai:true}};env.ctx.logout=()=>{};vm.runInContext(functionCode('renderAccount'),env.ctx);env.ctx.renderAccount();assert.equal(env.doc.getElementById('account-price').hidden,true);assert.match(env.doc.getElementById('subscription-copy').textContent,/Accès gratuit/);assert.doesNotMatch(env.doc.getElementById('billing-action').textContent,/abonner|essai|factur|29/);assert.doesNotMatch(env.doc.getElementById('account-progress').textContent,/29|7 jours d’essai/);
});

test('V6 re-saving identical map feedback replays its saved key and edits renew it',async()=>{
 const calls=[];let sequence=0;const env=mapFixture({crypto:{randomUUID:()=> 'key-'+(++sequence)},api:async(path,method,payload)=>{calls.push(payload);return {memory_version:calls.length}}});const form=env.host.mapCards.get('belong').card.querySelector('form');field(form,'text','Un retour explicite');await form.elements.text.fire('input');await form.fire('submit');await form.fire('submit');assert.equal(calls[0].creation_id,calls[1].creation_id);field(form,'text','Un retour précisé');await form.elements.text.fire('input');await form.fire('submit');assert.notEqual(calls[1].creation_id,calls[2].creation_id);
});


test('V6 portrait refresh rejects older source and analyzed snapshots before replacing visible content',()=>{
 const env=environment();env.ctx.state.portraitData={source_version:9,analyzed_version:8};vm.runInContext(functionCode('applyPortraitData'),env.ctx);const latest=env.ctx.state.portraitData;env.ctx.applyPortraitData({source_version:8,analyzed_version:8},()=>true);assert.equal(env.ctx.state.portraitData,latest);env.ctx.applyPortraitData({source_version:9,analyzed_version:7},()=>true);assert.equal(env.ctx.state.portraitData,latest);
});


test('V6 gender description is conditional without clearing an existing or typed draft',async()=>{
 for(const initial of ['', 'Mes mots déjà enregistrés']){
  const env=environment({api:async()=>({profile:{gender:'unspecified',gender_description:initial},revision:1}),location:{hash:''}});env.host('self-profile');vm.runInContext(functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();const form=env.doc.getElementById('self-profile-form'),gender=form.elements.gender,description=form.elements.gender_description;
  assert.equal(description.parentElement.hidden,!initial);gender.value='self_described';await gender.fire('change');assert.equal(description.parentElement.hidden,false);description.value='Mon brouillon conservé';await description.fire('input');gender.value='female';await gender.fire('change');assert.equal(description.parentElement.hidden,false);assert.equal(description.value,'Mon brouillon conservé');assert.equal(env.ctx.formValues(form).gender_description,'Mon brouillon conservé');description.value='';await description.fire('change');assert.equal(description.parentElement.hidden,true);assert.equal(description.value,'');
 }
});


test('V7 atlas switches domains and subdomains, searches without accents and focuses the selected detail',async()=>{
 const env=mapFixture(),map=structuredClone(psychologyFixture);map.subdomains=[{id:'social',domain:'ties',label:'Vie sociale'},{id:'liberty',domain:'self',label:'Liberté'}];map.patterns[0].subdomain='social';map.patterns[1].subdomain='liberty';map.patterns[1].label='Liberté de décision';env.ctx.drawPsychologyMap(env.host,map,()=>true);
 await env.host.mapDomains.get('self').button.fire('click');assert.equal(env.host.mapFilter.domain,'self');assert.equal(env.host.mapFilter.mode,'all');assert.equal(env.host.mapCards.get('autonomy').row.hidden,false);assert.equal(env.host.mapCards.get('belong').row.hidden,true);
 const sub=env.doc.getElementById('map-subdomain');assert.equal(sub.children.length,2);sub.value='liberty';await sub.fire('change');assert.equal(env.host.mapFilter.subdomain,'liberty');
 await env.host.mapCards.get('autonomy').row.fire('click');assert.equal(env.doc.activeElement,env.host.mapCards.get('autonomy').title);assert.equal(env.host.mapCards.get('autonomy').row.getAttribute('aria-pressed'),'true');
 await env.doc.getElementById('map-reset').fire('click');const search=env.doc.getElementById('map-search');search.value='liberte';await search.fire('input');assert.equal(env.host.mapCards.get('autonomy').row.hidden,false);assert.equal(env.host.mapCards.get('belong').row.hidden,true);assert.match(env.doc.getElementById('map-results').textContent,/1 repère/);
 search.value='sans correspondance';await search.fire('input');assert.equal(env.doc.getElementById('map-active-empty').hidden,false);assert.equal(env.doc.getElementById('map-detail').hidden,true);
});

test('V7 changing filters and polling preserve a map reply node, its draft and selection',async()=>{
 const env=mapFixture(),entry=env.host.mapCards.get('belong'),reply=entry.card.querySelector('.map-return'),form=reply.querySelector('form');reply.open=true;field(form,'text','Ma nuance encore non enregistrée');await form.elements.text.fire('input');
 await env.host.mapDomains.get('self').button.fire('click');assert.equal(entry.card.hidden,true);assert.equal(form.isConnected,true);
 await env.doc.getElementById('map-reset').fire('click');await entry.row.fire('click');const selected=env.host.mapSelected;const update=structuredClone(psychologyFixture);update.hypotheses[0].observation='Une observation révisée';env.ctx.drawPsychologyMap(env.host,update,()=>true);
 assert.equal(env.host.mapSelected,selected);assert.equal(entry.card.querySelector('form'),form);assert.equal(form.elements.text.value,'Ma nuance encore non enregistrée');assert.equal(reply.open,true);assert.match(entry.content.textContent,/observation révisée/);
});

test('V7 a synthesis uses the declared first name and targeted agreements, while citations stay exact',()=>{
 const env=mapFixture();env.ctx.state.portraitData={subject:{first_name:'Romain <em>',gender:'male'}};vm.runInContext(functionCode('appendPortrait'),env.ctx);const summary=env.doc.createElement('section');env.ctx.appendPortrait(summary,{summary:'La personne semble créative et réfléchie.',axes:[],strengths:['La personne se montre attentive.']},'Synthèse');assert.match(summary.textContent,/Romain <em> semble créatif et réfléchi/);assert.match(summary.textContent,/Romain <em> se montre attentif/);assert.equal(summary.querySelector('em'),null);
 const update=structuredClone(psychologyFixture);update.hypotheses[0].observation='La personne semble créative.';update.hypotheses[0].citations=[{quote:'La personne semble créative.'}];env.ctx.drawPsychologyMap(env.host,update,()=>true);const entry=env.host.mapCards.get('belong');assert.match(entry.content.querySelector('.map-observation').textContent,/Romain <em> semble créatif/);assert.equal(entry.content.querySelector('blockquote').textContent,'La personne semble créative.');
 assert.equal(env.ctx.personalizePortraitText('La personne est créative.',{first_name:'Camille',gender:'female'}),'Camille est créative.');assert.equal(env.ctx.personalizePortraitText('La personne est créative.',{gender:'male'}),'La personne est créative.');
});

test('V7 assistant reading suggestions are safe links attached to their message, capped at three',()=>{
 const env=environment({location:{origin:'https://example.test'}});env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:99},messages:[{role:'assistant',content:'Une réponse.',reading_suggestions:[{id:'a',title:'Une lecture',url:'/lecture/',character:'Nina',reason:'Pour réfléchir à votre place.'},{title:'Refusé',url:'javascript:alert(1)'},{title:'Deuxième',url:'/suite/'},{title:'Au-delà de la limite',url:'/trop/'}]},{role:'user',content:'Merci',reading_suggestions:[{title:'Ne pas afficher',url:'/non/'}]}],proposals:[]};env.ctx.drawChatProposal=()=>{};vm.runInContext(functionCode('safeArticleLink')+'\n'+functionCode('drawChatReadings')+'\n'+functionCode('chatConsent')+'\n'+functionCode('drawChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);const messages=host.querySelector('.chat-thread').children;assert.equal(messages[0].querySelectorAll('.chat-reading').length,2);assert.equal(messages[1].querySelector('.chat-readings'),null);assert.match(messages[0].textContent,/Pour réfléchir à votre place/);assert.doesNotMatch(host.textContent,/Refusé|Au-delà de la limite|Ne pas afficher/);assert.equal(messages[0].querySelector('.chat-reading').querySelector('a').href,'https://example.test/lecture/');
});


test('V7 personalization changes direct attributes and preserves complements and quoted words',()=>{
 const env=mapFixture(),subject={first_name:'Romain',gender:'male'},personalize=value=>env.ctx.personalizePortraitText(value,subject);
 assert.equal(personalize('La personne parle de sa mère créative et de son amie anxieuse.'),'Romain parle de sa mère créative et de son amie anxieuse.');
 assert.equal(personalize('La personne décrit sa mère comme créative.'),'Romain décrit sa mère comme créative.');
 assert.equal(personalize('La personne se décrit comme très créative, intuitive et attentive à son amie anxieuse.'),'Romain se décrit comme très créatif, intuitif et attentif à son amie anxieuse.');
 assert.equal(personalize('La personne semble créative et réfléchie dans sa relation avec une amie inquiète.'),'Romain semble créatif et réfléchi dans sa relation avec une amie inquiète.');
 assert.equal(personalize('La personne est sensible et anxieuse, sa mère est attentive.'),'Romain est sensible et anxieux, sa mère est attentive.');
 assert.equal(personalize('La personne dit : « La personne est créative. » Puis évoque "la personne anxieuse".'),'Romain dit : « La personne est créative. » Puis évoque "la personne anxieuse".');
});

test('V7 the pattern index uses one tab stop and supports arrow, home and end keys',async()=>{
 const env=mapFixture();await env.doc.getElementById('map-reset').fire('click');const first=env.host.mapCards.get('belong'),last=env.host.mapCards.get('autonomy');assert.equal(first.row.tabIndex,0);assert.equal(last.row.tabIndex,-1);await first.row.fire('keydown',{key:'End'});assert.equal(env.doc.activeElement,last.row);assert.equal(last.row.tabIndex,0);assert.equal(first.row.tabIndex,-1);assert.equal(last.card.hidden,false);await last.row.fire('keydown',{key:'ArrowUp'});assert.equal(env.doc.activeElement,first.row);await first.row.fire('keydown',{key:'ArrowDown'});assert.equal(env.doc.activeElement,last.row);await last.row.fire('keydown',{key:'Home'});assert.equal(env.doc.activeElement,first.row);
});


test('V7 a provider 429 keeps its precise message, quota and draft, then automatically restores sending after cooldown',async()=>{
 const timers=[],clock=Date.now(),calls=[];let now=clock;const data={consent:true,access:{allowed:true},quota:{remaining:100,limit:100},messages:[],proposals:[]},env=environment({Date:{now:()=>now},setTimeout(fn,delay){timers.push({fn,delay});return timers.length},clearTimeout(){},api:async(path,method)=>{calls.push({path,method});return {...data,last_error:{message:'Le fournisseur limite temporairement les demandes. Aucun message n’a été décompté.',action:'retry_later'}}}});
 Object.assign(env.ctx.state,{tab:'chat',chatDraft:'Mes mots complets',chatSending:true,chatSendKey:{key:'failed-key'},chatData:data});env.ctx.drawChatProposal=()=>{};vm.runInContext(functionCode('drawChatReadings')+'\n'+functionCode('drawChat')+'\n'+functionCode('pollChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);const input=env.doc.getElementById('chat-content'),form=env.doc.getElementById('chat-form'),button=form.querySelector('button');
 env.ctx.api=async(path,method)=>{calls.push({path,method});return {...data,access:{allowed:false,reason:'provider_cooldown',retry_at:(clock+45000)/1000},last_request:{id:'failed',status:'failed'},last_error:{code:'provider_http_429_request',message:'Le fournisseur limite temporairement les demandes. Réessayez dans 45 secondes. Aucun message n’a été décompté.',action:'retry_later',retry_at:(clock+45000)/1000}}};env.ctx.pollChat({id:'failed'},()=>true,'Mes mots complets');await timers.shift().fn();assert.match(env.doc.getElementById('chat-status').textContent,/Réessayez dans 45 secondes/);assert.match(env.doc.getElementById('chat-status').textContent,/Aucun message n’a été décompté/);assert.equal(env.ctx.state.chatSendKey,null);assert.equal(button.disabled,true);assert.equal(input.value,'Mes mots complets');assert.equal(env.ctx.state.chatData.quota.remaining,100);
 input.value='Un complément pendant l’attente';await input.fire('input');assert.ok(timers[0].delay>=45000&&timers[0].delay<=60000);now=clock+46000;env.ctx.api=async(path,method)=>{calls.push({path,method});return data};await timers.shift().fn();assert.equal(button.disabled,false);assert.equal(env.doc.getElementById('chat-form'),form);assert.equal(input.value,'Un complément pendant l’attente');assert.equal(env.ctx.state.chatDraft,input.value);assert.ok(calls.every(call=>call.path==='/api/chat'&&!call.method));
});

test('V7 provider cooldown uses bounded timers and ignores a response after leaving the conversation',async()=>{
 const timers=[],clock=Date.now();let active=true,resolve;const pending=new Promise(r=>resolve=r),env=environment({Date:{now:()=>clock},setTimeout(fn,delay){timers.push({fn,delay});return timers.length},clearTimeout(){},api:()=>pending});env.ctx.state.tab='chat';const data={access:{allowed:false,reason:'provider_cooldown',retry_at:(clock+120000)/1000}};env.ctx.scheduleChatCooldown(()=>active,data);assert.equal(timers[0].delay,60000);active=false;await timers.shift().fn();assert.equal(timers.length,0);
 active=true;env.ctx.scheduleChatCooldown(()=>active,{access:{allowed:false,reason:'provider_cooldown',retry_at:(clock-1)/1000}});const checking=timers.shift().fn();active=false;env.ctx.state.chatData={marker:'new conversation'};resolve({access:{allowed:true}});await checking;assert.equal(env.ctx.state.chatData.marker,'new conversation');
});


test('V7 portrait placeholders follow the current first name without changing portrait or quoted evidence',()=>{
 const env=mapFixture();for(const name of ['appendPortrait','applyPortraitData'])vm.runInContext(functionCode(name),env.ctx);const body=env.doc.createElement('div');body.id='portrait-body';const statusBox=env.doc.createElement('div');statusBox.id='portrait-status';
 const portrait={summary:'{{prenom}} se décrit comme créatif. Ses mots : « {{prenom}} se décrit comme créative. »',axes:[]},map=structuredClone(psychologyFixture);map.hypotheses[0].observation='{{prenom}} se décrit comme créatif.';map.hypotheses[0].citations=[{quote:'{{prenom}} se décrit comme créative.'}];const data={portrait,subject:{first_name:'Romain',gender:'male'},source_version:4,analyzed_version:4,status:'ready',psychology_map:map};
 env.ctx.applyPortraitData(data,()=>true);assert.match(body.textContent,/Romain se décrit comme créatif/);assert.match(body.textContent,/« \{\{prenom\}\} se décrit comme créative\. »/);assert.match(env.host.mapCards.get('belong').content.querySelector('.map-observation').textContent,/Romain se décrit comme créatif/);assert.equal(env.host.mapCards.get('belong').content.querySelector('blockquote').textContent,'{{prenom}} se décrit comme créative.');
 env.ctx.applyPortraitData({...data,subject:{first_name:'Alex',gender:'male'}},()=>true);assert.match(body.textContent,/Alex se décrit comme créatif/);assert.doesNotMatch(body.textContent,/Romain/);assert.match(env.host.mapCards.get('belong').content.querySelector('.map-observation').textContent,/Alex se décrit comme créatif/);assert.equal(data.portrait,portrait);assert.ok(portrait.summary.startsWith('{{prenom}}'));
 assert.equal(env.ctx.personalizePortraitText('{{prenom}}',{first_name:''}),'La personne');assert.equal(env.ctx.personalizePortraitText('« {{prenom}} » et {{prenom}}',{first_name:'Romain'}),'« {{prenom}} » et Romain');assert.equal(env.ctx.personalizePortraitText('"{{prenom}}" et {{prenom}}',{}),'"{{prenom}}" et La personne');
});


test('V7 atlas search includes keywords, aliases and readable identifiers when the label differs',async()=>{
 const env=mapFixture(),map=structuredClone(psychologyFixture);map.patterns[1]={...map.patterns[1],id:'thought_loop',label:'Pensées en boucle',keywords:['rumination'],aliases:['boucle mentale']};env.ctx.drawPsychologyMap(env.host,map,()=>true);const search=env.doc.getElementById('map-search');
 for(const query of ['rumination','boucle mentale','thought_loop','thought loop']){search.value=query;await search.fire('input');assert.equal(env.host.mapCards.get('thought_loop').row.hidden,false,query);assert.equal(env.host.mapCards.get('belong').row.hidden,true,query);assert.match(env.doc.getElementById('map-results').textContent,/1 repère/);assert.equal(env.host.mapSelected,'thought_loop');}
});


test('V7 legacy direct descriptions stay grammatical without inferring a gender from the first name',()=>{
 const env=mapFixture();for(const gender of ['unspecified','non_binary',undefined]){const subject={first_name:'Romain',...(gender?{gender}:{})},personalize=text=>env.ctx.personalizePortraitText(text,subject);
  assert.equal(personalize('La personne se décrit comme créative.'),'Romain se décrit comme une personne créative.');
  assert.equal(personalize('La personne se perçoit comme très attentive à son amie anxieuse.'),'Romain se perçoit comme une personne très attentive à son amie anxieuse.');
  assert.equal(personalize('La personne est sensible et anxieuse.'),'Romain est une personne sensible et anxieuse.');
  assert.equal(personalize('La personne semble réfléchie.'),'Romain semble être une personne réfléchie.');
  assert.equal(personalize('La personne semble être créative.'),'Romain semble être une personne créative.');
  assert.equal(personalize('La personne est une personne créative.'),'Romain est une personne créative.');
  assert.equal(personalize('La personne parle de sa mère créative et de son amie anxieuse.'),'Romain parle de sa mère créative et de son amie anxieuse.');
  assert.equal(personalize('La personne se sent créative.'),'Romain se sent créative.');
  assert.equal(personalize('La personne dit : « La personne se décrit comme créative. »'),'Romain dit : « La personne se décrit comme créative. »');
  assert.equal(personalize('{{prenom}} se décrit comme créatif.'),'Romain se décrit comme créatif.');
 }
 assert.equal(env.ctx.personalizePortraitText('{{prenom}}',{first_name:''}),'La personne');
});


test('V8 Ambre keeps the canonical route and identifies the chat as AI',()=>{
 const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('drawChat'),env.ctx);assert.equal(env.ctx.PAGES.chat.title,'Ambre');assert.equal(env.ctx.PAGES.chat.path,'/mon-accompagnement/');env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:100},messages:[{role:'assistant',content:'Bonjour.'},{role:'user',content:'Mon accompagnement, dans mes propres mots.'}],proposals:[]};const host=env.host('view');env.ctx.drawChat(host,()=>true);assert.match(host.textContent,/Ambre · psychologue IA/);assert.equal(host.querySelector('.chat-assistant').querySelector('h3').textContent,'Ambre');assert.match(host.querySelector('.chat-user').textContent,/Mon accompagnement, dans mes propres mots/);assert.doesNotMatch(host.textContent,/Assistant IA/);assert.match(functionCode('landing'),/<h1>Ambre<\/h1>/);
});

test('V8 the synthesis invites clarification with Ambre and never fetches the retired correction form',async()=>{
 const calls=[],env=environment({api:async(path)=>{calls.push(path);return {status:'ready'}},renderSelfProfile(){},renderLikes(){},setupPortraitViews(){},applyPortraitData(){},schedulePortraitPoll(){}});env.ctx.state.tab='portrait';env.host('view');env.host('portrait-status');vm.runInContext(functionCode('renderPortrait'),env.ctx);await env.ctx.renderPortrait();assert.deepEqual(calls,['/api/portrait']);assert.match(env.doc.getElementById('view').innerHTML,/Pour préciser votre synthèse, parlez-en à Ambre/);assert.match(env.doc.getElementById('view').innerHTML,/href="\/mon-accompagnement\/"/);assert.doesNotMatch(HTML,/Corriger ma synthèse|id="portrait-correction"|\/api\/memory|function loadPortraitCorrection/);
});

test('V8 manual presentation, close-person and life-event forms retain links to Ambre',async()=>{
 const env=environment({api:async()=>({profile:{first_name:'Nina'},revision:1}),location:{hash:''}});env.host('self-profile');vm.runInContext(functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();const self=env.doc.getElementById('self-profile');assert.match(self.querySelector('.ambre-hint').textContent,/Ambre.*compléter votre présentation/);assert.equal(self.querySelector('.ambre-hint').querySelector('a').href,'/mon-accompagnement/');assert.equal(env.doc.getElementById('self-profile-form').elements.first_name.value,'Nina');
 const rendered=[];env.ctx.drawSourceSection=kind=>rendered.push(kind);env.ctx.api=async()=>({people:[],events:[]});vm.runInContext(functionCode('renderPersonalSources'),env.ctx);const host=env.host('view');for(const page of ['people','events']){env.ctx.state.tab=page;await env.ctx.renderPersonalSources();assert.match(host.innerHTML,/Ambre peut aussi vous aider à ajouter/);assert.match(host.innerHTML,/href="\/mon-accompagnement\/"/);assert.match(host.innerHTML,new RegExp('id="'+page+'-content"'))}assert.deepEqual(rendered,['people','events']);
});

test('V8 memory append is an explicit editable proposal that retains the text and revision on failure',async()=>{
 let sent;const env=environment({api:async(path,method,payload)=>{sent={path,payload};throw new Error('Conflit')}});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'memory-1',kind:'memory_append',revision:7,payload:{text:'Mon contexte actuel.'},evidence_quote:'Mes mots exacts.'},()=>true);const form=host.children[0],input=form.elements.text;assert.match(form.textContent,/Préciser mon portrait/);assert.match(form.textContent,/Votre précision/);assert.match(form.textContent,/informations déjà enregistrées seront conservées/);assert.equal(input.maxLength,2000);assert.equal(input.required,true);assert.equal(sent,undefined);input.value='Ma précision modifiée.';await input.fire('input');await form.fire('submit');assert.equal(sent.path,'/api/chat/proposals/memory-1/accept');assert.equal(sent.payload.revision,7);assert.equal(sent.payload.payload.text,'Ma précision modifiée.');assert.equal('expected_version' in sent.payload.payload,false);assert.equal(env.ctx.state.chatProposalDrafts.get('memory-1').text,'Ma précision modifiée.');assert.match(form.textContent,/Vos modifications sont conservées/);assert.equal(form.querySelector('blockquote').textContent,'Mes mots exacts.');
});

test('V8 contextual influence and mechanisms remain qualitative and separate from the evidence status',async()=>{
 const env=mapFixture(),map=structuredClone(psychologyFixture);Object.assign(map.hypotheses[0],{status:'exploring',influence:{level:'high',contexts:['Dans le groupe','Au travail','En famille','Ne pas afficher'],rationale:'Ce fonctionnement revient dans plusieurs contextes décrits.',evidence:['event:e1']},interactions:[{pattern_id:'autonomy',kind:'buffers',explanation:'Cette ressource pourrait modérer ce besoin dans certaines situations.',evidence:['profile:self']},{pattern_id:'unknown',kind:'amplifies',explanation:'Introuvable'}]});env.ctx.drawPsychologyMap(env.host,map,()=>true);const entry=env.host.mapCards.get('belong'),weight=entry.content.querySelector('.map-influence');assert.equal(entry.badge.textContent,'À explorer');assert.match(weight.textContent,/Influence décrite : marquée/);assert.equal(weight.querySelectorAll('li').length,3);assert.doesNotMatch(weight.textContent,/Ne pas afficher|%|probabilité/);assert.ok(weight.querySelector('a').href.includes('event-e1'));const relations=entry.content.querySelector('.map-interactions');assert.match(relations.textContent,/Mécanismes liés/);assert.match(relations.textContent,/Peut atténuer/);assert.match(relations.textContent,/pourrait modérer/);assert.doesNotMatch(relations.textContent,/Introuvable/);await relations.querySelector('button').fire('click');assert.equal(env.host.mapSelected,'autonomy');assert.equal(env.host.mapCards.get('autonomy').card.hidden,false);
 const labels={unknown:'à préciser',low:'discrète',moderate:'modérée'};for(const [level,label] of Object.entries(labels)){map.hypotheses[0].influence.level=level;env.ctx.drawPsychologyMap(env.host,map,()=>true);assert.ok(entry.content.querySelector('.map-influence').textContent.includes('Influence décrite : '+label))}
});

test('V8 legacy pronouns with ambiguous referents are left for source-based analysis repair',()=>{
 const env=mapFixture(),subject={first_name:'Romain',gender:'male'};assert.equal(env.ctx.personalizePortraitText('La personne décrit sa mère. Elle est créative.',subject),'Romain décrit sa mère. Elle est créative.');assert.equal(env.ctx.personalizePortraitText('Sa mère est attentive. Elle se sent anxieuse.',subject),'Sa mère est attentive. Elle se sent anxieuse.');assert.equal(env.ctx.personalizePortraitText('« La personne est créative. Elle est réfléchie. »',subject),'« La personne est créative. Elle est réfléchie. »');assert.equal(env.ctx.personalizePortraitText('{{prenom}} est créatif. Sa mère est créative.',subject),'Romain est créatif. Sa mère est créative.');
});
