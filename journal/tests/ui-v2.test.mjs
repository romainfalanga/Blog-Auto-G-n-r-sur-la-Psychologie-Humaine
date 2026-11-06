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
  constructor(tag, doc) { this.tagName=tag.toUpperCase(); this.doc=doc; this.children=[]; this.listeners={}; this.dataset={}; this.attributes={}; this.value=''; this.checked=false; this.disabled=false; this.hidden=false; this.isConnected=true; this._text='';this.scrollTop=0;this.scrollHeight=0;this.clientHeight=0; }
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
  scrollTo(options) {this.scrollTop=options.top;this.lastScroll=options;}
  removeEventListener(name,fn) {this.listeners[name]=(this.listeners[name]||[]).filter(item=>item!==fn);}
  addEventListener(name,fn) { (this.listeners[name]??=[]).push(fn); }
  async fire(name,extra={}) { for (const fn of this.listeners[name]||[]) await fn({target:this,currentTarget:this,preventDefault(){},...extra}); }
  focus() {this.doc.activeElement=this;}
  matches(selector) {
    if(selector==='input,select,textarea')return ['INPUT','SELECT','TEXTAREA'].includes(this.tagName);
    if(selector==='[data-reload-source]')return this.dataset.reloadSource!==undefined;
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
  const doc={ids:new Map(),addEventListener(){},removeEventListener(){},createElement(tag){return new Element(tag,this)},createTextNode(text){return String(text)},getElementById(id){return this.ids.get(id)||null},querySelectorAll(selector){return [...this.ids.values()].filter(x=>x.matches(selector))},querySelector(selector){if(selector.startsWith('#')){const [id,...rest]=selector.slice(1).split(' ');return this.getElementById(id)?.querySelector(rest.join(' '))||null;}return this.querySelectorAll(selector)[0]||null}};
  class FormDataMock { constructor(form){this.values=form.querySelectorAll('input,select,textarea').filter(x=>x.name&&!x.disabled&&(x.type!=='checkbox'||x.checked)).map(x=>[x.name,x.value]);}entries(){return this.values.values();} }
  const ctx=vm.createContext({document:doc,FormData:FormDataMock,URL,URLSearchParams,crypto:{randomUUID:()=> 'create-uuid'},state:{people:[],events:[],sourceDrafts:new Map(),tab:'chat'},renderTokens:new WeakMap(),confirm:()=>true,refreshRecommendations(){},status(){},clearTimeout(){},setTimeout(){return 1},console,...extra});
  for (const name of ['renderGuard','el','actionButton','formValues','busy','sourceField','newClientId','captureSourceDraft','sourceDate','drawSourceEditor','chatConsent','chatFailureMessage','applyChatAvailability','scheduleChatCooldown','chatIsCurrent','chatCapabilityMessage','chatHasPending','updateChatControls','disposeChatInterface','resetChatSession','reconcileChatTurn','createChatShell','drawChatGate','scrollChatTo','revealChatResponse','chatMessageEntry','drawChatReadings','retryChatMessage','mapEvidenceContent']) vm.runInContext(functionCode(name),ctx);
  vm.runInContext(script.match(/var PERSON_ROLES=.*\n/)[0]+script.match(/var EVENT_KINDS=.*\n/)[0],ctx);
  vm.runInContext(functionCode('createVoiceInput'),ctx);
  ctx.drawSourceSection=()=>{};ctx.sourceSaved=()=>{};
  return {ctx,doc,host(id){const host=doc.createElement('div');host.id=id;return host;}};
}
function editFixture(kind,values,extra={}) { const env=environment(extra);env.host(kind+'-editor');env.ctx.state.sourceDrafts.set(kind,{id:null,creation_id:'creation-stable',values});env.ctx.drawSourceEditor(kind);return {...env,form:env.doc.getElementById(kind+'-form')}; }
function field(form,name,value) {const input=form.elements[name];assert.ok(input,name);input.value=value;return input;}

function chatFixture(extra={}) {
 let clientSerial=0;const env=environment({crypto:{randomUUID:()=> 'client-key-'+(++clientSerial)},...extra});Object.assign(env.ctx.state,{tab:'chat',chatDraft:'',chatOutbox:null,chatProposalDrafts:new Map(),chatSending:false,chatData:{consent:true,access:{allowed:true},quota:{remaining:100,limit:100},messages:[],proposals:[]}});
 for(const name of ['drawChat','sendChat','pollChat','renderChat'])vm.runInContext(functionCode(name),env.ctx);
 const view=env.host('view');env.ctx.drawChat(view,()=>true);
 return {...env,view,get form(){return env.doc.getElementById('chat-form')},get input(){return env.doc.getElementById('chat-content')},get send(){return env.doc.getElementById('chat-form').querySelector('button[type=submit]')}};
}

test('the generated application script parses without personal story controls',()=>{
  new vm.Script(script);
  assert.match(functionCode('renderLikes'),/legacy.id='suggestions';legacy.tabIndex=-1/);
  assert.doesNotMatch(HTML,/Mon personnage|Mes histoires|une histoire chaque jour/);
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

test('life-event form preserves an approximate age, and associated people',async()=>{
  let sent;const env=editFixture('events',{title:'Un tournant',kind:'turning_point',date_precision:'age',age_text:'Vers 15 ans',date_start:'2020',description:'Mon ressenti.',people_ids:['p1']},{api:async(path,method,payload)=>{sent=payload;throw new Error('offline')}});
  const form=env.form; // Simulate a selected checkbox exactly as FormData exposes it.
  const person=env.doc.createElement('input');person.type='checkbox';person.name='people_ids';person.value='p1';person.checked=true;form.append(person);
  await form.fire('submit');
  assert.equal(sent.age_text,'Vers 15 ans');assert.equal(sent.date_start,'');assert.equal(sent.date_end,'');
  assert.equal('fiction_use' in sent,false);assert.deepEqual(Array.from(sent.people_ids),['p1']);assert.equal(sent.current_importance,null);
  assert.equal(env.ctx.sourceDate(sent),'Vers 15 ans');
});

test('saving a year never invents a month or day, and changing precision updates required fields',async()=>{
  let sent;const env=editFixture('events',{title:'2020',kind:'positive',date_precision:'unknown',description:'Un souvenir',people_ids:[]},{api:async(p,m,d)=>{sent=d;throw new Error('offline')}});
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

test('V3 separates relationship and life pages from the declared and interpreted portrait',()=>{
  const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('pageForPath'),env.ctx);
  assert.equal(env.ctx.pageForPath('/mes-proches'),'people');assert.equal(env.ctx.pageForPath('/ma-ligne-de-vie/'),'events');assert.equal(env.ctx.pageForPath('/mon-accompagnement/'),'chat');
  assert.doesNotMatch(functionCode('renderPortrait'),/renderPersonalSources/);
  assert.match(functionCode('renderPortrait'),/renderSelfProfile/);
  assert.match(HTML,/À propos de moi/);assert.match(functionCode('renderPortrait'),/data-portrait-tab="synthesis">Mon portrait/);
  assert.match(HTML,/@media\(max-width:600px\)[^\n]*\.portrait-tabs \[data-portrait-tab="synthesis"\]\{grid-column:1\/-1\}/);
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

test('V4 global navigation has exactly the shared public links in source order',()=>{
  const navigation=JSON.parse(readFileSync(new URL('../../data/navigation.json',import.meta.url),'utf8'));
  const menu=HTML.match(/<ul class="nav-links">([\s\S]*?)<\/ul>/)[1];
  const links=[...menu.matchAll(/<a href="([^"]+)">([^<]+)<\/a>/g)].map(match=>({href:match[1],label:match[2]}));
  assert.deepEqual(links,navigation);assert.equal(links.length,6);assert.doesNotMatch(menu,/Se connecter|Se déconnecter/);
  assert.match(readFileSync(new URL('../../layouts/partials/header.html',import.meta.url),'utf8'),/\.Site\.Data\.navigation/);
});

test('private navigation renders the destination after updating history',()=>{
  const location={pathname:'/mon-portrait/',search:'',hash:''};
  const rendered=[];
  const {ctx}=environment({
    location,
    history:{pushState(_state,_unused,path){location.pathname=path}},
    window:{scrollTo(){}},
    capturePageDrafts(){},
    syncPageLocation(){ctx.state.tab='people'},
    renderApp(){rendered.push(ctx.state.tab)},
  });
  vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('go'),ctx);
  ctx.go('people');
  assert.equal(location.pathname,'/mes-proches/');
  assert.deepEqual(rendered,['people']);
});

test('portrait rendering no longer exposes the obsolete fictional character card',()=>{
  assert.doesNotMatch(functionCode('appendPortrait'),/fictional_character/);
  assert.doesNotMatch(functionCode('renderAccount'),/10 €|likes_count>=10|Aimez au moins 10/);
  assert.match(functionCode('renderAccount'),/29 €/);assert.match(functionCode('renderAccount'),/billing\/cancel/);
});

test('chat network retries keep the same message key and preserve a separate next draft',async()=>{
 const calls=[],env=chatFixture({api:async(...args)=>{calls.push(args);throw new Error('Réseau indisponible')}}),form=env.form;
 env.input.value='Mon premier message';await env.input.fire('input');await env.ctx.sendChat(form,()=>true);
 assert.equal(env.input.value,'');assert.equal(env.ctx.state.chatOutbox.content,'Mon premier message');assert.equal(env.ctx.state.chatOutbox.status,'uncertain');assert.equal(env.send.disabled,true);assert.equal(env.input.disabled,false);
 env.input.value='Mon prochain brouillon';await env.input.fire('input');const retry=env.view.querySelector('.chat-turn-state').querySelector('button');await retry.fire('click');
 const posts=calls.filter(call=>call[1]==='POST');assert.equal(posts.length,2);assert.equal(posts[0][2].idempotency_key,posts[1][2].idempotency_key);assert.equal(env.input.value,'Mon prochain brouillon');assert.equal(env.form,form);assert.match(env.view.textContent,/Mon premier message/);
});

test('chat proposals submit edited visible fields and preserve unedited structured data',async()=>{
  let sent;const env=environment({api:async(path,method,payload)=>{sent={path,payload};throw new Error('conflict')}});env.ctx.state.chatProposalDrafts=new Map();env.ctx.drawChatPeople=()=>{};vm.runInContext(functionCode('drawChatProposal'),env.ctx);
  const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'p1',kind:'life_event_create',revision:3,status:'pending',payload:{title:'Avant',description:'Détail',people_ids:['person1'],current_importance:2}},()=>true);
  const form=host.children[0],title=field(form,'title','Mon titre corrigé');await title.fire('input');await form.fire('submit');
  assert.equal(sent.path,'/api/chat/proposals/p1/accept');assert.equal(sent.payload.revision,3);assert.equal(sent.payload.payload.title,'Mon titre corrigé');assert.deepEqual(Array.from(sent.payload.payload.people_ids),['person1']);assert.equal(sent.payload.payload.current_importance,2);assert.equal(env.ctx.state.chatProposalDrafts.get('p1').title,'Mon titre corrigé');assert.match(form.textContent,/modifications sont conservées/);
});

test('quota exhaustion retains conversation, proposals and privacy actions while blocking sends',()=>{
  const env=environment();env.ctx.state.chatProposalDrafts=new Map();env.ctx.state.chatData={consent:true,access:{allowed:false,reason:'quota_exhausted'},quota:{remaining:0,limit:100},messages:[{role:'user',content:'Mon historique privé'}],proposals:[{id:'proposal',status:'pending'}]};env.ctx.drawChatProposal=(host)=>host.append(env.doc.createElement('form'));vm.runInContext(functionCode('chatConsent')+'\n'+functionCode('drawChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);
  assert.match(host.textContent,/Mon historique privé/);assert.doesNotMatch(host.textContent,/Informations et historique|Effacer la conversation et désactiver Ambre/);assert.doesNotMatch(host.textContent,/Voir mon abonnement/);assert.equal(env.doc.getElementById('chat-proposals').children.length,1);assert.equal(env.doc.getElementById('chat-form').querySelector('button[type=submit]').disabled,true);assert.equal(env.doc.getElementById('chat-content').maxLength,6000);
});

test('confirmed terminal failure renews message key, unlike ambiguous network failure',async()=>{
  let scheduled;const env=environment({clearTimeout(){},setTimeout(fn){scheduled=fn;return 1},api:async()=>({consent:true,access:{allowed:true},quota:{remaining:100},last_request:{id:'r1',status:'failed'},last_error:{code:'provider_error'}})});const form=env.host('chat-form'),button=env.doc.createElement('button');button.type='submit';form.append(button);Object.assign(env.ctx.state,{tab:'chat',chatDraft:'Mon brouillon',chatSendKey:{key:'old-key',content:'Mon brouillon'},chatSending:true});vm.runInContext(functionCode('pollChat'),env.ctx);env.ctx.pollChat({id:'r1'},()=>true,'Mon brouillon');await scheduled();assert.equal(env.ctx.state.chatSendKey,null);assert.equal(env.ctx.state.chatDraft,'Mon brouillon');assert.equal(button.disabled,false);
});

test('proposal citations are plain text, before values visible and person associations editable',async()=>{
  const env=environment({api:async()=>({people:[{id:'p1',display_name:'Camille'},{id:'p2',display_name:'Sam'}]})});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatPeople')+'\n'+functionCode('drawChatProposal'),env.ctx);const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'x',kind:'life_event_update',revision:1,status:'pending',evidence_quote:'<script>citation privée</script>',before:{title:'Avant'},payload:{title:'Après',people_ids:['p1']}},()=>true);await new Promise(resolve=>setImmediate(resolve));assert.match(host.textContent,/<script>citation privée<\/script>/);assert.equal(host.querySelector('script'),null);assert.match(host.textContent,/Titre : Avant/);const checkboxes=host.querySelectorAll('input');assert.equal(checkboxes.length,2);checkboxes[0].checked=false;await checkboxes[0].fire('change');checkboxes[1].checked=true;await checkboxes[1].fire('change');assert.deepEqual(Array.from(env.ctx.state.chatProposalDrafts.get('x').people_ids),['p2']);
});

test('chat ignores duplicate sends during latency and keeps text typed while a request is pending',async()=>{
  let resolve;const pending=new Promise(r=>resolve=r),calls=[];const env=environment({api:async(...args)=>{calls.push(args);return pending},pollChat(){}});Object.assign(env.ctx.state,{tab:'chat',chatDraft:'Message initial',chatSending:false,chatSendKey:null,chatData:{access:{allowed:true}}});vm.runInContext(functionCode('sendChat'),env.ctx);const form=env.doc.createElement('form'),input=env.doc.createElement('textarea'),button=env.doc.createElement('button');input.name='content';input.value='Message initial';button.type='submit';form.append(input,button);const sending=env.ctx.sendChat(form,()=>true);input.value='Mon prochain message';env.ctx.state.chatDraft=input.value;await env.ctx.sendChat(form,()=>true);assert.equal(calls.length,1);resolve({pending_request:{id:'pending',status:'running'},request_id:'pending'});await sending;assert.equal(env.ctx.state.chatDraft,'Mon prochain message');assert.equal(env.ctx.state.chatSending,true);assert.equal(button.disabled,true);
});


test('proposal importance is numeric from null and clearing restores null',async()=>{
  const env=environment();env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'importance',kind:'life_event_create',revision:1,payload:{current_importance:null}},()=>true);const input=field(host.children[0],'current_importance','4');await input.fire('input');assert.equal(env.ctx.state.chatProposalDrafts.get('importance').current_importance,4);input.value='';await input.fire('input');assert.equal(env.ctx.state.chatProposalDrafts.get('importance').current_importance,null);
});


test('V5 removes the global likes badge and every stale badge update',()=>{
  assert.doesNotMatch(functionCode('renderApp'),/plan-status|likes_count/);
  assert.doesNotMatch(functionCode('renderLikes'),/plan-status|progressBlock/);
});

test('empty conversation renders no invented message and places weekly usage in the conversation toolbar',()=>{
  const env=environment();env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{used:1,remaining:99,limit:100},messages:[],proposals:[]};
  env.ctx.drawChatProposal=()=>{};
  vm.runInContext(functionCode('chatConsent')+'\n'+functionCode('drawChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);
  assert.equal(host.querySelector('.chat-thread').children.length,0);
  const toolbar=host.querySelector('.conversation-toolbar');
  assert.equal(toolbar.querySelector('h1').textContent,'Ambre · psychologue IA');
  assert.equal(toolbar.querySelector('strong').textContent,'1 / 100');
  assert.equal(env.doc.getElementById('chat-form').querySelector('button[type=submit]').disabled,false);
  assert.doesNotMatch(host.textContent,/Un espace pour prendre du recul|Vous pouvez commencer par une situation/);
});

test('V5 blocked sends retain editable draft, history and proposals for all access reasons',()=>{
  for(const reason of ['consent_required','subscription_required','quota_exhausted','provider_missing','generation_paused']){
    const env=environment();env.ctx.state.chatDraft='Mon brouillon';env.ctx.state.chatData={consent:reason!=='consent_required',access:{allowed:false,reason},quota:{remaining:reason==='quota_exhausted'?0:100,limit:100},messages:[{role:'assistant',content:'Historique conservé'}],proposals:[{status:'pending'}]};
    env.ctx.drawChatProposal=host=>host.append(env.doc.createElement('form'));
    vm.runInContext(functionCode('chatConsent')+'\n'+functionCode('drawChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);
    assert.match(host.textContent,/Historique conservé/);assert.doesNotMatch(host.textContent,/Informations et historique|Effacer la conversation et désactiver Ambre/);
    assert.equal(env.doc.getElementById('chat-proposals').children.length,1);
    assert.equal(env.doc.getElementById('chat-content').value,'Mon brouillon');assert.equal(env.doc.getElementById('chat-form').querySelector('button[type=submit]').disabled,true);
  }
});

test('portrait panes preserve deep links and drafts while redirecting the former map pane',async()=>{
 for(const [hash,key] of [['#ma-cartographie','synthesis'],['#cartographie','synthesis'],['#mon-portrait','synthesis'],['#synthese','synthesis'],['#self-profile','self'],['#mes-articles','readings'],['#suggestions','readings'],['#correction-form','synthesis'],['','synthesis']]){
  const location={hash,pathname:'/mon-portrait/',search:'?article=old'};const env=environment({location,history:{replaceState(a,b,url){location.hash=url.includes('#')?url.slice(url.indexOf('#')):''}}});
  env.ctx.state.pendingArticle=hash?'old':null;const view=env.host('view'),panes=['synthesis','self','readings'].map(key=>{const e=env.doc.createElement('section');e.dataset.portraitPane=key;view.append(e);return e}),tabs=['synthesis','self','readings'].map((key,i)=>{const e=env.doc.createElement('a');e.dataset.portraitTab=key;e.getAttribute=()=>['#mon-portrait','#self-profile','#mes-articles'][i];view.append(e);return e});
  const form=env.doc.createElement('form');panes[1].append(form);
  view.querySelectorAll=selector=>selector==='[data-portrait-pane]'?panes:tabs;
  vm.runInContext(functionCode('setupPortraitViews'),env.ctx);env.ctx.setupPortraitViews(view);
  assert.equal(panes.find(p=>!p.hidden).dataset.portraitPane,key);
  if(['#ma-cartographie','#cartographie'].includes(hash))assert.equal(location.hash,'#mon-portrait');
  await tabs[1].fire('click');assert.equal(panes[1].hidden,false);assert.equal(panes[1].children[0],form);
  await tabs[2].fire('click');assert.equal(panes[2].hidden,false);assert.equal(panes[1].children[0],form);
 }
});


test('an immediate completed chat turn reconciles its optimistic bubble and enables the existing composer',async()=>{
 const env=chatFixture({api:async(path,method,payload)=>({consent:true,access:{allowed:true},quota:{remaining:99,limit:100},request_id:'turn',last_request:{id:'turn',status:'completed'},messages:[{id:'turn-user',client_key:payload.idempotency_key,role:'user',content:payload.content,status:'completed'},{id:'turn-assistant',role:'assistant',content:'Réponse complète.'}],proposals:[]})});
 const form=env.form;env.input.value='Bonjour';await env.input.fire('input');await env.ctx.sendChat(form,()=>true);
 assert.equal(env.form,form);assert.equal(env.send.disabled,false);assert.equal(env.ctx.state.chatSending,false);assert.equal(env.ctx.state.chatOutbox,null);assert.equal(env.input.value,'');assert.equal(env.view.querySelectorAll('.chat-user').length,1);assert.equal(env.doc.getElementById('chat-pending-indicator').parentElement,null);
});

test('a pending turn keeps one user bubble while the cleared composer accepts the next draft',async()=>{
 let resolve;const post=new Promise(r=>resolve=r);let client;const env=chatFixture({api:async(path,method,payload)=>{if(method==='POST'){client=payload.idempotency_key;return post}return {consent:true,access:{allowed:true},quota:{remaining:99,limit:100},pending_request:{id:'same-turn',status:'running',client_key:client},messages:[{id:'same-turn-user',client_key:client,role:'user',content:'Un seul message',status:'running'}],proposals:[]}}});
 env.input.value='Un seul message';await env.input.fire('input');const sending=env.ctx.sendChat(env.form,()=>true);const bubble=env.view.querySelector('.chat-user');assert.ok(bubble);assert.match(env.view.textContent,/Ambre réfléchit/);assert.equal(env.input.value,'');
 env.input.value='Le message suivant';await env.input.fire('input');resolve({request_id:'same-turn',pending_request:{id:'same-turn',status:'running',client_key:client}});await sending;
 assert.equal(env.view.querySelectorAll('.chat-user').length,1);assert.equal(env.view.querySelector('.chat-user'),bubble);assert.equal(bubble.dataset.chatTurnId,'same-turn');assert.equal(env.input.value,'Le message suivant');assert.equal(env.send.disabled,true);assert.equal(env.input.disabled,false);
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
 assert.match(known.card.textContent,/À préciser/);assert.match(unknown.card.textContent,/Non exploré/);assert.doesNotMatch(env.host.textContent,/\d+\s*%|probabilité|absence de syndrome/i);
 assert.ok(known.card.querySelectorAll('a').some(link=>link.href==='/mes-proches/#person-p1'));assert.ok(known.card.querySelectorAll('a').some(link=>link.href==='https://example.test/article/'));assert.doesNotMatch(known.card.textContent,/Lien refusé/);
 assert.match(known.card.textContent,/Mes mots exacts/);assert.match(known.card.textContent,/Ce qui invite à nuancer/);assert.match(known.card.textContent,/Ce que cela pourrait expliquer/);assert.doesNotMatch(known.card.textContent,/Ce que cela pourrait vous apporter/);
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
 env.ctx.schedulePortraitPoll(()=>true,{status:'updating'});env.ctx.state.tab='chat';await timers.shift()();assert.equal(calls.length,1);
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
 Object.assign(env.ctx.state,{tab:'chat',chatDraft:'Mes mots complets',chatSending:true,chatSendKey:{key:'failed-key'},chatData:data});env.ctx.drawChatProposal=()=>{};vm.runInContext(functionCode('drawChatReadings')+'\n'+functionCode('drawChat')+'\n'+functionCode('pollChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);const input=env.doc.getElementById('chat-content'),form=env.doc.getElementById('chat-form'),button=form.querySelector('button[type=submit]');
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
 env.ctx.applyPortraitData(data,()=>true);assert.match(body.textContent,/Romain se décrit comme créatif/);assert.match(body.textContent,/« \{\{prenom\}\} se décrit comme créative\. »/);
 env.ctx.applyPortraitData({...data,subject:{first_name:'Alex',gender:'male'}},()=>true);assert.match(body.textContent,/Alex se décrit comme créatif/);assert.doesNotMatch(body.textContent,/Romain/);assert.equal(data.portrait,portrait);assert.ok(portrait.summary.startsWith('{{prenom}}'));
 assert.equal(env.ctx.personalizePortraitText('{{prenom}}',{first_name:''}),'La personne');assert.equal(env.ctx.personalizePortraitText('« {{prenom}} » et {{prenom}}',{first_name:'Romain'}),'« {{prenom}} » et Romain');assert.equal(env.ctx.personalizePortraitText('"{{prenom}}" et {{prenom}}',{}),'"{{prenom}}" et La personne');
});


test('V7 atlas search includes keywords, aliases and readable identifiers when the label differs',async()=>{
 const env=mapFixture(),map=structuredClone(psychologyFixture);map.patterns[1]={...map.patterns[1],id:'thought_loop',label:'Pensées en boucle',keywords:['rumination'],aliases:['boucle mentale']};env.ctx.drawPsychologyMap(env.host,map,()=>true);await env.host.querySelectorAll('[data-map-mode]')[1].fire('click');const search=env.doc.getElementById('map-search');
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
 const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('drawChat'),env.ctx);assert.equal(env.ctx.PAGES.chat.title,'Ambre');assert.equal(env.ctx.PAGES.chat.path,'/mon-accompagnement/');env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:100},messages:[{role:'assistant',content:'Bonjour.'},{role:'user',content:'Mon accompagnement, dans mes propres mots.'}],proposals:[]};const host=env.host('view');env.ctx.drawChat(host,()=>true);assert.match(host.textContent,/Ambre · psychologue IA/);assert.equal(host.querySelector('.chat-assistant').querySelector('h3').textContent,'Ambre');assert.match(host.querySelector('.chat-user').textContent,/Mon accompagnement, dans mes propres mots/);assert.doesNotMatch(host.textContent,/Assistant IA/);assert.equal(host.querySelectorAll('h1').length,1);assert.doesNotMatch(functionCode('landing'),/<h1>Ambre<\/h1>/);
});

test('the public portrait renders three views without the cartography or a repeated Ambre note',async()=>{
 const calls=[],env=environment({api:async(path)=>{calls.push(path);return {status:'ready'}},renderSelfProfile(){},renderLikes(){},setupPortraitViews(){},applyPortraitData(){},schedulePortraitPoll(){}});env.ctx.state.tab='portrait';env.host('view');env.host('portrait-status');vm.runInContext(functionCode('renderPortrait'),env.ctx);await env.ctx.renderPortrait();assert.deepEqual(calls,['/api/portrait']);
 const view=env.doc.getElementById('view'),labels=[...view.innerHTML.matchAll(/data-portrait-tab="[^"]+">([^<]+)/g)].map(match=>match[1]);
 assert.deepEqual(labels,['Mon portrait','À propos de moi','Articles aimés']);assert.doesNotMatch(view.innerHTML,/ambre-hint|Ma cartographie|role="note"/);assert.equal(env.doc.getElementById('ma-cartographie'),null);assert.doesNotMatch(functionCode('applyPortraitData'),/drawPsychologyMap/);assert.doesNotMatch(HTML,/Corriger ma synthèse|id="portrait-correction"|\/api\/memory|function loadPortraitCorrection/);
});

test('manual presentation, close-person and life-event forms omit the repeated Ambre notes',async()=>{
 const env=environment({api:async()=>({profile:{first_name:'Nina'},revision:1}),location:{hash:''}});env.host('self-profile');vm.runInContext(functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();const self=env.doc.getElementById('self-profile');assert.equal(self.querySelector('.ambre-hint'),null);assert.equal(env.doc.getElementById('self-profile-form').elements.first_name.value,'Nina');
 const rendered=[];env.ctx.drawSourceSection=kind=>rendered.push(kind);env.ctx.api=async()=>({people:[],events:[]});vm.runInContext(functionCode('renderPersonalSources'),env.ctx);const host=env.host('view');for(const page of ['people','events']){env.ctx.state.tab=page;await env.ctx.renderPersonalSources();assert.doesNotMatch(host.innerHTML,/ambre-hint|Ambre peut aussi vous aider à ajouter|role="note"/);assert.match(host.innerHTML,new RegExp('id="'+page+'-content"'))}assert.deepEqual(rendered,['people','events']);
});

test('V8 memory append is an explicit editable proposal that retains the text and revision on failure',async()=>{
 let sent;const env=environment({api:async(path,method,payload)=>{sent={path,payload};throw new Error('Conflit')}});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'memory-1',kind:'memory_append',revision:7,payload:{text:'Mon contexte actuel.'},evidence_quote:'Mes mots exacts.'},()=>true);const form=host.children[0],input=form.elements.text;assert.match(form.textContent,/Préciser mon portrait/);assert.match(form.textContent,/Votre précision/);assert.match(form.textContent,/informations déjà enregistrées seront conservées/);assert.equal(input.maxLength,2000);assert.equal(input.required,true);assert.equal(sent,undefined);input.value='Ma précision modifiée.';await input.fire('input');await form.fire('submit');assert.equal(sent.path,'/api/chat/proposals/memory-1/accept');assert.equal(sent.payload.revision,7);assert.equal(sent.payload.payload.text,'Ma précision modifiée.');assert.equal('expected_version' in sent.payload.payload,false);assert.equal(env.ctx.state.chatProposalDrafts.get('memory-1').text,'Ma précision modifiée.');assert.match(form.textContent,/Vos modifications sont conservées/);assert.equal(form.querySelector('blockquote').textContent,'Mes mots exacts.');
});

test('V8 contextual influence and mechanisms remain qualitative and separate from the evidence status',async()=>{
 const env=mapFixture(),map=structuredClone(psychologyFixture);Object.assign(map.hypotheses[0],{status:'exploring',influence:{level:'high',contexts:['Dans le groupe','Au travail','En famille','Ne pas afficher'],rationale:'Ce fonctionnement revient dans plusieurs contextes décrits.',evidence:['event:e1']},interactions:[{pattern_id:'autonomy',kind:'buffers',explanation:'Cette ressource pourrait modérer ce besoin dans certaines situations.',evidence:['profile:self']},{pattern_id:'unknown',kind:'amplifies',explanation:'Introuvable'}]});env.ctx.drawPsychologyMap(env.host,map,()=>true);const entry=env.host.mapCards.get('belong'),weight=entry.content.querySelector('.map-influence');assert.equal(entry.badge.textContent,'À préciser');assert.match(weight.textContent,/Influence décrite : marquée/);assert.equal(weight.querySelectorAll('li').length,3);assert.doesNotMatch(weight.textContent,/Ne pas afficher|%|probabilité/);assert.ok(weight.querySelector('a').href.includes('event-e1'));const relations=entry.content.querySelector('.map-interactions');assert.match(relations.textContent,/Mécanismes liés/);assert.match(relations.textContent,/Peut atténuer/);assert.match(relations.textContent,/pourrait modérer/);assert.doesNotMatch(relations.textContent,/Introuvable/);await relations.querySelector('button').fire('click');assert.equal(env.host.mapSelected,'autonomy');assert.equal(env.host.mapCards.get('autonomy').card.hidden,false);
 const labels={unknown:'à préciser',low:'discrète',moderate:'modérée'};for(const [level,label] of Object.entries(labels)){map.hypotheses[0].influence.level=level;env.ctx.drawPsychologyMap(env.host,map,()=>true);assert.ok(entry.content.querySelector('.map-influence').textContent.includes('Influence décrite : '+label))}
});

test('V8 legacy pronouns with ambiguous referents are left for source-based analysis repair',()=>{
 const env=mapFixture(),subject={first_name:'Romain',gender:'male'};assert.equal(env.ctx.personalizePortraitText('La personne décrit sa mère. Elle est créative.',subject),'Romain décrit sa mère. Elle est créative.');assert.equal(env.ctx.personalizePortraitText('Sa mère est attentive. Elle se sent anxieuse.',subject),'Sa mère est attentive. Elle se sent anxieuse.');assert.equal(env.ctx.personalizePortraitText('« La personne est créative. Elle est réfléchie. »',subject),'« La personne est créative. Elle est réfléchie. »');assert.equal(env.ctx.personalizePortraitText('{{prenom}} est créatif. Sa mère est créative.',subject),'Romain est créatif. Sa mère est créative.');
});

test('a late chat POST cannot cancel the new view polling or alter its draft and request state',async()=>{
 let resolve;const pending=new Promise(r=>resolve=r),timers=new Map(),cleared=[];let serial=0,active=true,drawn=0;
 const env=environment({api:async(path,method)=>method==='POST'?pending:{messages:[{role:'assistant',content:'Réponse terminée'}],access:{allowed:true}},setTimeout(fn){timers.set(++serial,fn);return serial},clearTimeout(id){if(timers.delete(id))cleared.push(id)}});
 vm.runInContext(functionCode('sendChat')+'\n'+functionCode('pollChat'),env.ctx);env.host('view');env.ctx.drawChat=()=>drawn++;
 Object.assign(env.ctx.state,{tab:'chat',chatData:{access:{allowed:true}},chatDraft:'Message envoyé'});
 const form=env.doc.createElement('form'),input=env.doc.createElement('textarea');input.name='content';input.value='Message envoyé';form.append(input);
 const sending=env.ctx.sendChat(form,()=>active);drawn=0;active=false;
 Object.assign(env.ctx.state,{chatSending:true,chatRequest:{id:'pending-current'},chatDraft:'Nouveau brouillon',chatSendKey:{key:'current-key'},chatData:{marker:'current'},chatOutbox:{client_key:'current-key',request_id:'pending-current',content:'Current',status:'pending'}});
 env.ctx.api=async()=>({messages:[{id:'pending-current-user',client_key:'current-key',role:'user',content:'Current',status:'completed'},{id:'pending-current-assistant',role:'assistant',content:'Réponse terminée'}],access:{allowed:true},last_request:{id:'pending-current',status:'completed'}});
 env.ctx.pollChat({id:'pending-current'},()=>true,null);const currentTimer=env.ctx.state.chatTimer;
 resolve({pending_request:{id:'pending-old'}});await sending;
 assert.equal(env.ctx.state.chatTimer,currentTimer);assert.equal(cleared.includes(currentTimer),false);assert.equal(env.ctx.state.chatRequest.id,'pending-current');assert.equal(env.ctx.state.chatDraft,'Nouveau brouillon');assert.equal(env.ctx.state.chatSendKey.key,'current-key');assert.equal(env.ctx.state.chatData.marker,'current');
 env.ctx.pollChat({id:'obsolete'},()=>false,null);assert.equal(env.ctx.state.chatTimer,currentTimer);assert.equal(timers.size,1);
 const callback=timers.get(currentTimer);timers.delete(currentTimer);await callback();assert.equal(drawn,1);assert.equal(env.ctx.state.chatSending,false);assert.equal(env.ctx.state.chatDraft,'Nouveau brouillon');
});

test('chat completion and fallback refresh cannot clear another view draft after navigation',async()=>{
 let resolveRefresh,started;const refresh=new Promise(r=>resolveRefresh=r),loading=new Promise(r=>started=r);let active=true;
 const env=environment({api:async(path,method)=>{if(method==='POST')return {};started();return refresh}});vm.runInContext(functionCode('sendChat'),env.ctx);
 Object.assign(env.ctx.state,{tab:'chat',chatData:{access:{allowed:true}}});const form=env.doc.createElement('form'),input=env.doc.createElement('textarea');input.name='content';input.value='Message initial';form.append(input);
 const sending=env.ctx.sendChat(form,()=>active);await loading;active=false;
 Object.assign(env.ctx.state,{chatDraft:'Brouillon actuel',chatSendKey:{key:'current'},chatRequest:{id:'current'},chatSending:true,chatData:{marker:'current'}});
 resolveRefresh({messages:[]});await sending;
 assert.equal(env.ctx.state.chatDraft,'Brouillon actuel');assert.equal(env.ctx.state.chatSendKey.key,'current');assert.equal(env.ctx.state.chatRequest.id,'current');assert.equal(env.ctx.state.chatSending,true);assert.equal(env.ctx.state.chatData.marker,'current');
});

test('append proposals show and submit only the added fragment while keeping the full original in the preview',async()=>{
 let sent,confirmations=0;const original='Une longue présentation déjà enregistrée. '.repeat(180),env=environment({confirm(){confirmations++;return true},api:async(path,method,payload)=>{sent={path,payload};throw new Error('Réseau indisponible')}});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);
 const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'append',kind:'self_profile_patch',revision:2,payload:{self_description:'Une précision nouvelle.'},before:{self_description:original},field_modes:{self_description:'append'},evidence_quote:'Mes mots exacts.'},()=>true);
 const form=host.children[0],input=form.elements.self_description;assert.equal(input.value,'Une précision nouvelle.');assert.match(form.querySelector('label').textContent,/^Ajouter à/);assert.ok(form.querySelector('.proposal-before').textContent.includes(original));assert.match(form.textContent,/texte déjà enregistré, qui sera conservé/);
 input.value='Une précision modifiée.';await input.fire('input');await form.fire('submit');assert.equal(sent.payload.payload.self_description,'Une précision modifiée.');assert.equal(sent.payload.revision,2);assert.equal(confirmations,0);assert.equal(env.ctx.state.chatProposalDrafts.get('append').self_description,'Une précision modifiée.');assert.equal(form.querySelector('blockquote').textContent,'Mes mots exacts.');
});

test('an explicit replacement names the affected field and requires confirmation before sending',async()=>{
 let sent,confirmed=false;const prompts=[],env=environment({confirm(text){prompts.push(text);return confirmed},api:async(path,method,payload)=>{sent={path,payload};throw new Error('Conflit')}});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);
 const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'replace',kind:'self_profile_patch',revision:4,payload:{self_description:'Ma nouvelle présentation.'},before:{self_description:'Ancienne présentation.',emotions:'Un champ conservé.'},field_modes:{self_description:'set'}},()=>true);
 const form=host.children[0];assert.equal(form.querySelector('label').textContent,'Remplacer : Votre présentation');assert.equal(form.querySelector('button').textContent,'Remplacer « Votre présentation »');await form.fire('submit');assert.equal(sent,undefined);assert.match(prompts[0],/^Remplacer « Votre présentation »/);assert.doesNotMatch(prompts[0],/émotions/);
 confirmed=true;await form.fire('submit');assert.equal(sent.payload.payload.self_description,'Ma nouvelle présentation.');assert.equal('emotions' in sent.payload.payload,false);assert.equal(sent.payload.revision,4);
});

test('mixed proposal modes distinguish replacements and additions while legacy proposals retain their original behavior',async()=>{
 const prompts=[],env=environment({confirm(text){prompts.push(text);return false}});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);
 const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'mixed',kind:'life_event_update',revision:1,payload:{title:'Titre inchangé',description:'Texte corrigé.',meaning_now:'Nouveau recul.'},before:{title:'Titre inchangé',description:'Texte original.',meaning_now:'Recul existant.'},field_modes:{title:'set',description:'set',meaning_now:'append'}},()=>true);
 const form=host.children[0];assert.match(form.textContent,/Remplacer : Description/);assert.match(form.textContent,/Ajouter à : Ce que ce moment représente aujourd’hui/);assert.equal(form.querySelector('button').textContent,'Confirmer les remplacements et ajouts');await form.fire('submit');assert.match(prompts[0],/^Remplacer « Description »/);assert.doesNotMatch(prompts[0],/représente|Titre/);assert.doesNotMatch(form.textContent,/Remplacer : Titre/);
 const legacy=env.host('legacy');env.ctx.drawChatProposal(legacy,{id:'old',kind:'person_update',revision:1,payload:{relationship_text:'Texte complet hérité.'},before:{relationship_text:'Avant.'}},()=>true);assert.equal(legacy.children[0].elements.relationship_text.value,'Texte complet hérité.');assert.equal(legacy.querySelector('label').textContent,'Votre relation');assert.equal(legacy.querySelector('button').textContent,'Accepter cette proposition');
});

test('V10 a POST lost after completion reconciles by client key without duplicates or a stale network error',async()=>{
 let key;const env=chatFixture({api:async(path,method,payload)=>{if(method==='POST'){key=payload.idempotency_key;throw new Error('Réponse réseau perdue')}return {consent:true,access:{allowed:true},quota:{remaining:99,limit:100},messages:[{id:'done-user',client_key:key,role:'user',content:'Un message',status:'completed'},{id:'done-assistant',role:'assistant',content:'Une réponse enregistrée.'}],last_request:{id:'done',status:'completed'},proposals:[]}}});
 env.input.value='Un message';await env.input.fire('input');await env.ctx.sendChat(env.form,()=>true);
 assert.equal(env.view.querySelectorAll('.chat-user').length,1);assert.equal(env.view.querySelectorAll('.chat-assistant').length,1);assert.equal(env.ctx.state.chatOutbox,null);assert.equal(env.send.disabled,false);assert.equal(env.doc.getElementById('chat-status').textContent,'');assert.equal(env.input.value,'');
});

test('V10 a definitive rejection restores only an empty composer and preserves a next draft and its exact error',async()=>{
 for(const nextDraft of ['', 'Mon prochain brouillon']){
  let reject;const post=new Promise((resolve,r)=>reject=r),error=Object.assign(new Error('Le quota est atteint. Aucun message n’a été décompté.'),{status:429,code:'quota_exhausted'}),env=chatFixture({api:async(path,method)=>method==='POST'?post:{consent:true,access:{allowed:false,reason:'quota_exhausted'},quota:{remaining:0,limit:100},messages:[],proposals:[]}});
  env.input.value='Message refusé';await env.input.fire('input');const sending=env.ctx.sendChat(env.form,()=>true);env.input.value=nextDraft;await env.input.fire('input');reject(error);await sending;
  assert.equal(env.input.value,nextDraft||'Message refusé');assert.equal(env.ctx.state.chatDraft,nextDraft||'Message refusé');assert.equal(env.ctx.state.chatSendKey,null);assert.equal(env.ctx.state.chatOutbox.status,'failed');assert.match(env.view.textContent,/Le quota est atteint\. Aucun message n’a été décompté/);assert.equal(env.input.disabled,false);assert.equal(env.send.disabled,true);assert.equal(env.view.querySelector('.chat-turn-state').querySelector('button').disabled,true);
 }
});

test('V10 polling failure restores no old text over a following draft and can retry the uncertain turn with the same key',async()=>{
 const timers=[],calls=[];let key;const env=chatFixture({setTimeout(fn){timers.push(fn);return timers.length},clearTimeout(){},api:async(path,method,payload)=>{calls.push({path,method,payload});if(method==='POST'){key=payload.idempotency_key;return {consent:true,access:{allowed:true},quota:{remaining:99,limit:100},pending_request:{id:'pending',client_key:key,status:'running'},messages:[{id:'pending-user',client_key:key,role:'user',content:payload.content,status:'running'}],proposals:[]}}throw new Error('Vérification interrompue')}});
 env.input.value='Message initial';await env.input.fire('input');await env.ctx.sendChat(env.form,()=>true);env.input.value='Suite en préparation';await env.input.fire('input');await timers.shift()();
 assert.equal(env.ctx.state.chatOutbox.status,'uncertain');assert.equal(env.input.value,'Suite en préparation');const retry=env.view.querySelector('.chat-turn-state').querySelector('button');assert.equal(retry.disabled,false);await retry.fire('click');assert.equal(calls.filter(call=>call.method==='POST').length,2);assert.ok(calls.filter(call=>call.method==='POST').every(call=>call.payload.idempotency_key===key));assert.equal(env.input.value,'Suite en préparation');
});

test('new welcome remains first in conversations with existing history and uses no quota',()=>{
 let calls=0;const env=chatFixture({api:async()=>{calls++;throw new Error('Unexpected')}}),welcome={id:'ambre-welcome-v3',role:'assistant',content:'Bonjour, je suis Ambre. Qu’aimeriez-vous comprendre ?'};
 env.ctx.state.chatData.welcome=welcome;env.ctx.drawChat(env.view,()=>true);const greeting=env.view.querySelector('.chat-assistant');assert.equal(greeting.querySelector('.chat-content').textContent,welcome.content);assert.equal(env.ctx.state.chatData.messages.length,0);assert.equal(env.ctx.state.chatData.quota.remaining,100);assert.equal(calls,0);
 env.ctx.drawChat(env.view,()=>true);assert.equal(env.view.querySelector('.chat-assistant'),greeting);assert.equal(env.view.querySelectorAll('.chat-assistant').length,1);
 env.ctx.state.chatData.messages=[{id:'first-user',client_key:'first-client',role:'user',content:'Bonjour',status:'completed'}];env.ctx.drawChat(env.view,()=>true);assert.equal(env.view.querySelectorAll('.chat-assistant').length,1);assert.equal(env.view.querySelector('.chat-thread').children[0],greeting);assert.equal(env.view.querySelectorAll('.chat-user').length,1);assert.equal(calls,0);
});

test('V10 a validated long response reveals once and anchors its beginning instead of its bottom',async()=>{
 const timers=[],env=chatFixture({setTimeout(fn){timers.push(fn);return timers.length},clearTimeout(){},matchMedia:()=>({matches:false})}),text='Une phrase complète avec ses mots exacts. '.repeat(80),shell=env.view.chatShell,form=env.form;
 const create=env.doc.createElement.bind(env.doc);env.doc.createElement=tag=>{const element=create(tag);if(tag==='article')element.offsetTop=420;return element};shell.scroll.scrollHeight=5000;shell.scroll.clientHeight=500;shell.follow=true;
 env.ctx.state.chatData.messages=[{id:'reply-assistant',role:'assistant',content:text}];env.ctx.drawChat(env.view,()=>true,{animate:true});const item=env.view.querySelector('.chat-assistant'),content=item.querySelector('.chat-content');assert.ok(content.textContent.length<text.length);assert.equal(content.getAttribute('aria-hidden'),'true');assert.equal(shell.scroll.lastScroll.top,420);assert.notEqual(shell.scroll.lastScroll.top,shell.scroll.scrollHeight);
 while(timers.length)await timers.shift()();assert.equal(content.textContent,text);assert.equal(content.getAttribute('aria-hidden'),undefined);assert.equal(shell.scroll.lastScroll.top,420);assert.equal(env.form,form);
 env.ctx.drawChat(env.view,()=>true,{animate:true});assert.equal(timers.length,0);assert.equal(content.textContent,text);assert.equal(env.view.querySelector('.chat-assistant'),item);
});

test('V10 a reader above the latest exchange receives a new-response control without a forced scroll',async()=>{
 const env=chatFixture({matchMedia:()=>({matches:true})}),shell=env.view.chatShell;shell.scroll.scrollHeight=5000;shell.scroll.clientHeight=500;shell.scroll.scrollTop=50;await shell.scroll.fire('scroll');assert.equal(shell.follow,false);
 const create=env.doc.createElement.bind(env.doc);env.doc.createElement=tag=>{const element=create(tag);if(tag==='article')element.offsetTop=600;return element};env.ctx.state.chatData.messages=[{id:'new-assistant',role:'assistant',content:'Une réponse nouvelle.'}];env.ctx.drawChat(env.view,()=>true,{animate:true});assert.equal(shell.scroll.scrollTop,50);assert.equal(shell.next.hidden,false);await shell.next.fire('click');assert.equal(shell.scroll.scrollTop,600);assert.equal(shell.next.hidden,true);
});

test('V10 reduced motion and reopening history render complete responses without reanimation',async()=>{
 const timers=[],text='Un texte long et validé. '.repeat(100),data={consent:true,access:{allowed:true},quota:{remaining:100,limit:100},messages:[{id:'history-assistant',role:'assistant',content:text}],proposals:[]},env=chatFixture({setTimeout(fn){timers.push(fn);return timers.length},clearTimeout(){},matchMedia:query=>({matches:query.includes('reduced-motion')}),api:async()=>data});
 env.ctx.state.chatData=data;env.ctx.drawChat(env.view,()=>true,{animate:true});assert.equal(env.view.querySelector('.chat-content').textContent,text);assert.equal(timers.length,0);env.ctx.matchMedia=()=>({matches:false});await env.ctx.renderChat();assert.equal(env.view.querySelector('.chat-content').textContent,text);assert.equal(timers.length,0);
});

test('V10 the real dictation adapter updates the next draft, keeps its input through refresh and ignores disposed callbacks',async()=>{
 const instances=[];class Recognition{constructor(){instances.push(this)}start(){this.onstart()}abort(){this.aborted=true}}
 const env=chatFixture({SpeechRecognition:Recognition}),mic=env.view.chatShell.mic,input=env.input;assert.equal(mic.disabled,false);await mic.fire('click');const recognition=instances[0],late=recognition.onresult;assert.equal(recognition.lang,'fr-FR');assert.equal(recognition.interimResults,true);
 recognition.onresult({results:[[{transcript:'Une phrase dictée'}]]});assert.equal(input.value,'Une phrase dictée');assert.equal(env.ctx.state.chatDraft,input.value);env.ctx.drawChat(env.view,()=>true);assert.equal(env.input,input);assert.equal(env.ctx.state.chatVoice.isActive(),true);
 input.value='Une phrase corrigée';await input.fire('input');assert.equal(recognition.aborted,true);late({results:[[{transcript:'Un ancien résultat'}]]});assert.equal(input.value,'Une phrase corrigée');
 await mic.fire('click');const second=instances[1],stale=second.onresult;env.ctx.disposeChatInterface();assert.equal(second.aborted,true);assert.equal(env.ctx.state.chatVoice,null);stale({results:[[{transcript:'Après navigation'}]]});assert.equal(input.value,'Une phrase corrigée');assert.equal(env.ctx.state.chatDraft,'Une phrase corrigée');
});

test('V10 sending stops dictation before clearing the editor and a late voice result cannot overwrite the next draft',async()=>{
 const instances=[];class Recognition{constructor(){instances.push(this)}start(){this.onstart()}abort(){this.aborted=true}}
 let resolve,client;const post=new Promise(r=>resolve=r),env=chatFixture({SpeechRecognition:Recognition,api:async(path,method,payload)=>{client=payload.idempotency_key;return post}});await env.view.chatShell.mic.fire('click');const recognition=instances[0],late=recognition.onresult;recognition.onresult({results:[[{transcript:'Message dicté'}]]});const sending=env.ctx.sendChat(env.form,()=>true);assert.equal(recognition.aborted,true);assert.equal(env.input.value,'');
 env.input.value='Brouillon suivant';await env.input.fire('input');late({results:[[{transcript:'Résultat arrivé trop tard'}]]});assert.equal(env.input.value,'Brouillon suivant');resolve({consent:true,access:{allowed:true},quota:{remaining:99,limit:100},pending_request:{id:'spoken',client_key:client,status:'running'},messages:[{id:'spoken-user',client_key:client,role:'user',content:'Message dicté',status:'running'}],proposals:[]});await sending;assert.equal(env.input.value,'Brouillon suivant');assert.equal(env.view.querySelectorAll('.chat-user').length,1);
});

test('the microphone icon waits for the final French transcript before re-enabling send',async()=>{
 const instances=[];class Recognition{constructor(){instances.push(this)}start(){this.onstart()}stop(){this.stopped=true}abort(){this.aborted=true}}
 const env=chatFixture({SpeechRecognition:Recognition}),mic=env.view.chatShell.mic;
 assert.match(mic.innerHTML,/class="microphone-glyph"/);assert.match(mic.innerHTML,/class="stop-glyph"/);assert.equal(mic.getAttribute('aria-describedby'),'chat-voice-status');assert.equal(mic.type,'button');
 await mic.fire('click');const r=instances[0];assert.equal(env.send.disabled,true);assert.equal(mic.dataset.voicePhase,'listening');r.onresult({results:[[{transcript:'J’aimerais'}]]});
 await mic.fire('click');assert.equal(r.stopped,true);assert.equal(r.aborted,undefined);assert.equal(env.send.disabled,true);assert.equal(mic.dataset.voicePhase,'finishing');
 r.onresult({results:[[{transcript:'J’aimerais parler à Ambre.'}]]});r.onend();assert.equal(env.input.value,'J’aimerais parler à Ambre.');assert.equal(env.ctx.state.chatDraft,env.input.value);assert.equal(env.send.disabled,false);assert.equal(mic.getAttribute('aria-label'),'Dicter un message');
});

test('V10 proposal availability updates without destroying its edited fields or the composer',async()=>{
 const env=chatFixture({api:async()=>{throw new Error('Must not accept')}}),form=env.form;
 env.ctx.state.chatData.capabilities={self_profile_patch:{allowed:false,reasons:['target_missing']}};env.ctx.state.chatData.proposals=[{id:'self-preview',kind:'self_profile_patch',status:'pending',revision:1,payload:{self_description:'Une précision.'}},{id:'old-story',kind:'story_request',status:'pending',revision:1,payload:{requested_focus:'Retiré'}}];vm.runInContext(functionCode('drawChatProposal'),env.ctx);env.ctx.drawChat(env.view,()=>true);const proposal=env.doc.getElementById('chat-proposals').querySelector('form'),accept=proposal.querySelector('button[type=submit]');assert.equal(accept.disabled,true);assert.match(proposal.textContent,/Cette opération n’est pas disponible/);assert.equal(env.doc.getElementById('chat-proposals').querySelectorAll('form').length,1);const focus=proposal.elements.self_description;focus.value='Mon thème édité.';await focus.fire('input');await proposal.fire('submit');assert.equal(accept.disabled,true);
 env.ctx.state.chatData.capabilities.self_profile_patch.allowed=true;env.ctx.drawChat(env.view,()=>true);assert.equal(env.doc.getElementById('chat-proposals').querySelector('form'),proposal);assert.equal(focus.value,'Mon thème édité.');assert.equal(accept.disabled,false);assert.equal(env.form,form);
});

test('V10 personal map search stays personal until the user explicitly explores the catalogue',async()=>{
 const env=mapFixture(),atlas=env.host.mapAtlas;assert.equal(atlas.hidden,true);assert.equal(atlas.open,false);assert.equal(env.doc.getElementById('map-overview-count').textContent,'1 piste personnelle');assert.doesNotMatch(env.doc.getElementById('map-overview-count').textContent,/repères|domaines/);
 const search=env.doc.getElementById('map-search');search.value='autonomie';await search.fire('input');assert.equal(env.host.mapFilter.mode,'active');assert.equal(env.host.mapCards.get('autonomy').row.hidden,true);const empty=env.doc.getElementById('map-active-empty');assert.equal(empty.hidden,false);assert.equal(empty.querySelector('button').textContent,'Chercher dans les autres repères');await empty.querySelector('button').fire('click');assert.equal(env.host.mapFilter.mode,'all');assert.equal(atlas.hidden,false);assert.equal(env.host.mapCards.get('autonomy').row.hidden,false);assert.equal(search.value,'autonomie');
});

test('V10 a short personal map starts with its cards while catalogue exploration exposes the filters',async()=>{
 const env=mapFixture();assert.equal(env.host.mapSearchField.hidden,true);assert.equal(env.host.mapFilters.hidden,true);assert.equal(env.doc.getElementById('map-results').hidden,true);
 const explore=env.host.querySelectorAll('[data-map-mode]').find(button=>button.dataset.mapMode==='all');await explore.fire('click');assert.equal(env.host.mapSearchField.hidden,false);assert.equal(env.host.mapFilters.hidden,false);assert.equal(env.host.mapAtlas.hidden,false);
 const personal=env.host.querySelectorAll('[data-map-mode]').find(button=>button.dataset.mapMode==='active');await personal.fire('click');assert.equal(env.host.mapSearchField.hidden,true);assert.equal(env.host.mapFilters.hidden,true);
});

test('V10 a map detail explains observation, evidence, context, hypothesis and counter-evidence in that order',()=>{
 const env=mapFixture(),map=structuredClone(psychologyFixture);Object.assign(map.hypotheses[0],{influence:{level:'moderate',contexts:['Dans le groupe'],rationale:'Dans ce contexte décrit.',evidence:['person:p1']},citations:[{source_id:'person:p1',quote:'Mes mots exacts'},{source_id:'profile:self',quote:'Parfois je décide seul.'}]});env.ctx.drawPsychologyMap(env.host,map,()=>true);const content=env.host.mapCards.get('belong').content,classes=content.children.map(node=>node.className||''),position=name=>classes.findIndex(value=>value.split(' ').includes(name));
 assert.ok(position('map-observation')<position('map-evidence'));assert.ok(position('map-evidence')<position('map-influence'));assert.ok(position('map-influence')<position('map-function'));assert.ok(position('map-function')<position('map-counter'));assert.ok(position('map-counter')<position('map-inquiry'));assert.ok(position('map-inquiry')<position('map-definition-more'));
 assert.equal(content.querySelector('.map-evidence').querySelector('blockquote').textContent,'Mes mots exacts');assert.equal(content.querySelector('.map-counter').querySelector('blockquote').textContent,'Parfois je décide seul.');assert.match(content.querySelector('.map-function').textContent,/pourrait expliquer/);assert.match(content.querySelector('.map-influence').textContent,/Influence décrite : modérée/);assert.doesNotMatch(content.textContent,/\d+\s*%/);
});

test('V10 the mobile map opens one detail and returns to its list without losing a correction draft',async()=>{
 const scrolls=[],env=mapFixture({matchMedia:query=>({matches:query.includes('max-width')}),window:{scrollY:230,scrollTo(options){scrolls.push(options)}}}),entry=env.host.mapCards.get('belong');assert.equal(env.host.dataset.mapReading,'false');await entry.row.fire('click');assert.equal(env.host.dataset.mapReading,'true');assert.equal(env.doc.activeElement,entry.title);
 const form=entry.card.querySelector('form'),input=form.elements.text;input.value='Une nuance personnelle en cours';await input.fire('input');await env.doc.getElementById('map-back').fire('click');assert.equal(env.host.dataset.mapReading,'false');assert.equal(env.doc.activeElement,entry.row);assert.equal(scrolls[0].top,230);
 env.ctx.drawPsychologyMap(env.host,structuredClone(psychologyFixture),()=>true);await entry.row.fire('click');assert.equal(entry.card.querySelector('form'),form);assert.equal(input.value,'Une nuance personnelle en cours');assert.equal(env.host.dataset.mapReading,'true');
});

test('V10 an outbox provider cooldown keeps the precise error and automatically enables retry without changing the next draft',async()=>{
 const timers=new Map();let serial=0,now=Date.now(),client,answer;const env=chatFixture({Date:{now:()=>now},setTimeout(fn){timers.set(++serial,fn);return serial},clearTimeout(id){timers.delete(id)},api:async(path,method,payload)=>{if(method==='POST'){client=payload.idempotency_key;return {consent:true,access:{allowed:true},quota:{remaining:99,limit:100},pending_request:{id:'limited',client_key:client,status:'running'},messages:[{id:'limited-user',client_key:client,role:'user',content:payload.content,status:'running'}],proposals:[]}}return answer}});
 env.input.value='Message initial';await env.input.fire('input');await env.ctx.sendChat(env.form,()=>true);
 answer={consent:true,access:{allowed:false,reason:'provider_cooldown',retry_at:(now+30000)/1000},quota:{remaining:100,limit:100},last_request:{id:'limited',status:'failed'},last_error:{code:'provider_http_429_request',message:'Le fournisseur limite temporairement les demandes. Aucun message n’a été décompté.',action:'retry_later'},messages:[{id:'limited-user',client_key:client,role:'user',content:'Message initial',status:'failed'}],proposals:[]};
 let [id,callback]=[...timers][0];timers.delete(id);await callback();const form=env.form,retry=env.view.querySelector('.chat-turn-state').querySelector('button');assert.equal(retry.disabled,true);assert.equal(env.input.value,'Message initial');assert.match(env.doc.getElementById('chat-status').textContent,/Le fournisseur limite temporairement les demandes\. Aucun message n’a été décompté/);assert.equal(env.ctx.state.chatSendKey,null);
 env.input.value='Un prochain message';await env.input.fire('input');now+=31000;answer={...answer,access:{allowed:true}};[id,callback]=[...timers][0];timers.delete(id);await callback();assert.equal(env.form,form);assert.equal(env.input.value,'Un prochain message');assert.equal(retry.disabled,false);assert.equal(env.send.disabled,false);assert.equal(env.ctx.state.chatData.quota.remaining,100);
});

test('a liked article renders from the real application state and preserves its note draft',async()=>{
 const env=environment({location:{origin:'https://example.test',pathname:'/mon-portrait/'},query:new URLSearchParams()});
 vm.runInContext(functionCode('pageForPath')+'\n'+script.match(/var PAGES=.*\n/)[0]+'\n'+script.match(/var state=.*\n/)[0],env.ctx);
 for(const name of ['safeArticleLink','drawLikedArticles'])vm.runInContext(functionCode(name),env.ctx);
 env.ctx.state.examples=[{id:'a',title:'Une lecture',url:'/posts/une-lecture/'}];
 env.ctx.state.likes=[{article_id:'a',reason:'Mon retour'}];
 const list=env.host('liked');env.ctx.drawLikedArticles(list,()=>true);
 assert.match(list.textContent,/Une lecture/);
 const input=env.doc.getElementById('reason-0');input.value='Mon brouillon';await input.fire('input');
 env.ctx.drawLikedArticles(list,()=>true);
 assert.equal(env.doc.getElementById('reason-0').value,'Mon brouillon');
});

test('leaving an untouched profile does not hide a later update accepted in Ambre',async()=>{
 let revision=1;
 const env=environment({api:async()=>({profile:{self_description:revision===1?'Avant':'Après Ambre'},revision}),location:{hash:''}});
 env.host('self-profile');
 for(const name of ['capturePageDrafts','renderSelfProfile'])vm.runInContext(functionCode(name),env.ctx);
 await env.ctx.renderSelfProfile();env.ctx.capturePageDrafts();
 revision=2;await env.ctx.renderSelfProfile();
 assert.equal(env.doc.getElementById('self-profile-form').elements.self_description.value,'Après Ambre');
 assert.equal(env.ctx.state.selfRevision,2);
 const form=env.doc.getElementById('self-profile-form');field(form,'self_description','Mon brouillon');await form.fire('input');
 env.ctx.capturePageDrafts();revision=3;await env.ctx.renderSelfProfile();
 assert.equal(env.doc.getElementById('self-profile-form').elements.self_description.value,'Mon brouillon');
 assert.equal(env.ctx.state.selfRevision,2);
});

test('saving a timeline change reloads its chronological cursor without discarding an edit draft',async()=>{
 let reloads=0;
 const env=environment({renderPersonalSources:async()=>{reloads++;},refreshPortraitData(){}});
 env.ctx.state.tab='events';
 env.ctx.state.eventsCursor='old-boundary';
 env.ctx.state.sourceDrafts.set('events',{values:{title:'New unsaved text'}});
 vm.runInContext(functionCode('sourceSaved'),env.ctx);
 env.ctx.sourceSaved({source_version:3});await Promise.resolve();
 assert.equal(reloads,1);
 assert.equal(env.ctx.state.sourceDrafts.get('events').values.title,'New unsaved text');
 env.ctx.state.tab='people';env.ctx.sourceSaved({source_version:4});
 assert.equal(reloads,1);
});
