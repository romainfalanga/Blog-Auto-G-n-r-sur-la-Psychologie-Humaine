import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { HTML } from '../src/ui.mjs';
import { SETTINGS_INTERFACE_SCRIPT } from '../src/settings-interface.mjs';

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
  prepend(item) { this.children.unshift(item);item.parentElement=this; }
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
  const ctx=vm.createContext({document:doc,FormData:FormDataMock,URL,URLSearchParams,crypto:{randomUUID:()=> 'create-uuid'},state:{people:[],events:[],sourceDrafts:new Map(),tab:'chat'},renderTokens:new WeakMap(),query:new URLSearchParams(),confirm:()=>true,refreshRecommendations(){},status(){},clearTimeout(){},setTimeout(){return 1},console,...extra});
  // The site confirmation window is answered by the same hook as the former confirm().
  if(!ctx.confirmDialog)ctx.confirmDialog=(title,text)=>Promise.resolve(ctx.confirm([title,text].filter(Boolean).join(' ')));
  for (const name of ['renderGuard','el','actionButton','finePointer','focusEntry','formValues','busy','sourceField','appendEventDateFields','appendSourceContextFields','appendSourceContextText','newClientId','captureSourceDraft','sourceDate','drawSourceEditor','chatConsent','chatFailureMessage','applyChatAvailability','scheduleChatCooldown','chatIsCurrent','chatCapabilityMessage','chatHasPending','updateChatControls','disposeChatInterface','resetChatSession','reconcileChatTurn','createChatShell','drawChatGate','scrollChatTo','revealChatResponse','chatMessageEntry','chatJourney','chatReviewRunning','chatJourneyBlocked','chatPointNote','chatPointSuggestion','drawChatNotes','chatInitialPending','journeyPanel','closeJourneyPanel','showJourneyPanel','journeySourceItem','journeyCount','drawJourneyPanel','openPointDialog','pollJourney','dateLabel','retryChatMessage','mapEvidenceContent','displayDate','navigationPending','prepareMomentForPerson','momentReturn']) vm.runInContext(functionCode(name),ctx);
  vm.runInContext(script.match(/var MOMENT_FOR_PERSON=.*\n/)[0],ctx);
  vm.runInContext(['SOURCE_CONTEXT_CHOICES','SOURCE_CONTEXT_LIMITS','SOURCE_CONTEXT_LABELS','SOURCE_CONTEXT_ORDER'].map(name=>script.match(new RegExp('var '+name+'=.*\\n'))[0]).join('')+script.match(/var PERSON_ROLES=.*\n/)[0]+script.match(/var EVENT_KINDS=.*\n/)[0],ctx);
  vm.runInContext(functionCode('createVoiceInput'),ctx);
  ctx.drawSourceSection=()=>{};ctx.sourceSaved=()=>{};
  return {ctx,doc,host(id){const host=doc.createElement('div');host.id=id;return host;}};
}
function editFixture(kind,values,extra={}) { const env=environment(extra);env.host(kind+'-editor');env.ctx.state.sourceDrafts.set(kind,{id:null,creation_id:'creation-stable',values});env.ctx.drawSourceEditor(kind);return {...env,form:env.doc.getElementById(kind+'-form')}; }
function field(form,name,value) {const input=form.elements[name];assert.ok(input,name);input.value=value;return input;}
async function selfFixture(api,extra={}) {
 const env=environment({api,location:{hash:''},...extra}),host=env.host('self-profile');
 vm.runInContext(functionCode('status')+'\n'+functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();
 return {...env,host,get form(){return env.doc.getElementById('self-profile-form')},get input(){return env.doc.getElementById('self-profile-form')?.elements.first_name},get message(){return env.doc.getElementById('self-profile-status')}};
}

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
  assert.equal('fiction_use' in sent,false);assert.deepEqual(Array.from(sent.people_ids),['p1']);assert.equal('current_importance' in sent,false);
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

test('personal pages keep the declared portrait without a synthesis subpage',()=>{
  const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('personIdForPath')+'\n'+functionCode('pageForPath'),env.ctx);
  assert.equal(env.ctx.pageForPath('/mes-proches'),'people');assert.equal(env.ctx.pageForPath('/mes-proches/'),'people');assert.equal(env.ctx.pageForPath('/ma-ligne-de-vie/'),'events');assert.equal(env.ctx.pageForPath('/mon-accompagnement/'),'chat');
  assert.doesNotMatch(functionCode('renderPortrait'),/renderPersonalSources/);
  assert.match(functionCode('renderPortrait'),/renderSelfProfile/);
  assert.doesNotMatch(HTML,/À propos de moi/);assert.doesNotMatch(functionCode('renderPortrait'),/data-portrait-tab|portrait-synthesis|portrait-body/);
  assert.doesNotMatch(HTML,/\.portrait-tabs/);
  assert.doesNotMatch(HTML,/Trois personnages|trois personnages|\[1,2,3\]\.forEach/);
});

test('Mon compte exposes only identity controls and keeps historical paragraphs invisible',async()=>{
 const env=await selfFixture(async()=>({profile:{self_description:'Ancien texte',first_name:'Nina',emotions:'Calme',occupation:'Ancienne activité'},revision:4}));
 assert.deepEqual(env.host.querySelectorAll('input,select,textarea').map(x=>x.name),['first_name','last_name','birth_date','gender','gender_description']);
 assert.equal(env.input.name,'first_name');assert.equal(env.input.maxLength,100);assert.equal(env.input.value,'Nina');
 assert.equal(env.host.querySelectorAll('textarea').length,0);assert.equal(env.form.querySelector('button[type=submit]').textContent,'Enregistrer');
 assert.doesNotMatch(env.host.textContent,/Articles aimés|Ancien texte|Calme|Ancienne activité/);
});

test('saving the identity sends only its field and revision, with hidden historical fields left to the backend',async()=>{
 const historical={self_description:'Ancienne présentation',gender:'female',occupation:'Mon métier',self_perception:'Mon regard',what_brings_me:'Comprendre',what_helps_me:'Marcher',life_rhythm:'Le matin'};let sent;
 const env=await selfFixture(async(path,method,payload)=>{if(!method)return {profile:{...historical,first_name:'Avant'},revision:7};sent={path,method,payload};return {profile:{...historical,first_name:payload.first_name},revision:8}});
 env.input.value='Ma présentation complète';await env.form.fire('input');await env.form.fire('submit');
 assert.equal(sent.path,'/api/self-profile');assert.equal(sent.method,'PUT');assert.deepEqual(Object.keys(sent.payload).sort(),['expected_revision','first_name']);
 assert.equal(sent.payload.first_name,'Ma présentation complète');assert.equal(sent.payload.expected_revision,7);
 assert.equal(env.ctx.state.selfDraft,null);assert.equal(env.ctx.state.selfRevision,8);assert.match(env.message.textContent,/Informations enregistrées/);
});

test('409 comparison requires an explicit revision choice and never replaces typed identity',async()=>{
 let stored={profile:{first_name:'Avant'},revision:4},failure=true;const calls=[];
 const env=await selfFixture(async(path,method,payload)=>{if(!method)return stored;calls.push(payload);if(failure)throw Object.assign(new Error('conflit'),{status:409});return {profile:{first_name:payload.first_name},revision:6}});
 env.input.value='Mon brouillon complet';await env.form.fire('input');stored={profile:{first_name:'Présentation acceptée avec Ambre'},revision:5};await env.form.fire('submit');
 assert.equal(env.ctx.state.selfRevision,4);assert.equal(env.input.value,'Mon brouillon complet');assert.match(env.message.textContent,/brouillon est conservé/);
 await env.form.querySelector('[data-reload-source]').fire('click');assert.match(env.form.querySelector('.notice').textContent,/Présentation acceptée avec Ambre/);
 assert.equal(env.ctx.state.selfRevision,4);assert.equal(calls.length,1);assert.equal(env.input.value,'Mon brouillon complet');
 await env.form.querySelector('.notice').querySelector('button').fire('click');assert.equal(env.ctx.state.selfRevision,5);assert.equal(calls.length,1);
 failure=false;await env.form.fire('submit');assert.equal(calls[1].expected_revision,5);assert.equal(calls[1].first_name,'Mon brouillon complet');assert.equal(env.ctx.state.selfRevision,6);assert.equal(env.ctx.state.selfDraft,null);
});

test('typing while a save is pending stays editable and becomes a draft at the returned revision',async()=>{
 let resolve,calls=0;const env=await selfFixture(async(path,method,payload)=>{if(!method)return {profile:{first_name:'Avant'},revision:2};calls++;return new Promise(r=>resolve=r)});
 env.input.value='Premier texte';await env.form.fire('input');const saving=env.form.fire('submit');assert.equal(env.form.querySelector('button[type=submit]').disabled,true);
 env.input.value='Texte saisi pendant la sauvegarde';await env.form.fire('input');await env.form.fire('submit');assert.equal(calls,1);assert.equal(env.input.disabled,false);
 resolve({profile:{first_name:'Premier texte'},revision:3});await saving;
 assert.equal(env.input.value,'Texte saisi pendant la sauvegarde');assert.equal(env.ctx.state.selfDraft.first_name,env.input.value);assert.equal(env.ctx.state.selfRevision,3);
 assert.equal(env.form.querySelector('button[type=submit]').disabled,false);assert.match(env.message.textContent,/restent à sauvegarder/);
});

test('a failed save keeps the full identity and original revision available for retry',async()=>{
 let offline=true;const sent=[];const env=await selfFixture(async(path,method,payload)=>{if(!method)return {profile:{first_name:'Avant'},revision:3};sent.push(payload);if(offline)throw new Error('Connexion interrompue');return {profile:{first_name:payload.first_name},revision:4}});
 env.input.value='Mes mots à garder\nUne seconde ligne';await env.form.fire('input');await env.form.fire('submit');assert.match(env.message.textContent,/Connexion interrompue/);
 assert.equal(env.ctx.state.selfRevision,3);assert.equal(env.input.value,env.ctx.state.selfDraft.first_name);assert.equal(env.form.querySelector('button[type=submit]').disabled,false);
 offline=false;await env.form.fire('submit');assert.deepEqual(sent[1],sent[0]);assert.equal(env.ctx.state.selfDraft,null);assert.equal(env.ctx.state.selfRevision,4);
});

test('a failed comparison can be retried without adopting an unseen revision or losing input',async()=>{
 let failRead=false;const env=await selfFixture(async(path,method)=>{if(method)throw Object.assign(new Error('conflit'),{status:409});if(failRead)throw new Error('Lecture indisponible');return {profile:{first_name:'Enregistré'},revision:3}});
 env.input.value='Brouillon';await env.form.fire('input');await env.form.fire('submit');const compare=env.form.querySelector('[data-reload-source]');failRead=true;await compare.fire('click');
 assert.equal(compare.disabled,false);assert.equal(env.ctx.state.selfRevision,3);assert.equal(env.input.value,'Brouillon');assert.match(env.message.textContent,/Lecture indisponible/);
 failRead=false;await compare.fire('click');assert.match(env.form.querySelector('.notice').textContent,/Enregistré/);assert.equal(env.input.value,'Brouillon');
});

test('an initial profile read failure offers a reload and does not create an empty editable form',async()=>{
 let offline=true;const env=await selfFixture(async()=>{if(offline)throw new Error('Lecture indisponible');return {profile:{first_name:'Version enregistrée'},revision:9}});
 assert.equal(env.host.querySelector('form'),null);assert.match(env.host.textContent,/Lecture indisponible/);assert.equal(env.host.getAttribute('aria-busy'),undefined);
 offline=false;await env.host.querySelector('button').fire('click');assert.equal(env.input.value,'Version enregistrée');assert.equal(env.ctx.state.selfRevision,9);
});

test('identity-only edits and explicit clears patch only changed fields, never the unchanged presentation',async()=>{
 let stored={profile:{first_name:'Camille',last_name:'Martin',birth_date:'1990-05-02',gender:'female',gender_description:'',occupation:'Activité',self_description:'Présentation conservée',emotions:'Calme'},revision:4};const sent=[];
 const env=await selfFixture(async(path,method,payload)=>{if(!method)return stored;sent.push(payload);const {expected_revision,...changes}=payload;stored={profile:{...stored.profile,...changes},revision:stored.revision+1};return stored});
 assert.equal(env.form.elements.first_name.value,'Camille');assert.equal(env.form.elements.birth_date.type,'date');assert.equal(env.form.elements.occupation,undefined);
 field(env.form,'first_name','Prénom modifié');field(env.form,'last_name','');await env.form.fire('input');await env.form.fire('submit');
 assert.deepEqual(Object.keys(sent[0]).sort(),['expected_revision','first_name','last_name']);assert.equal(sent[0].last_name,'');assert.equal(sent[0].expected_revision,4);
 assert.equal(stored.profile.self_description,'Présentation conservée');assert.equal(stored.profile.emotions,'Calme');assert.equal(env.ctx.state.selfDraft,null);
});

test('a conflict rebases untouched identity while preserving local edits and hidden history',async()=>{
 let stored={profile:{first_name:'Camille',last_name:'Avant',self_description:'Présentation'},revision:2},conflict=true;const sent=[];
 const env=await selfFixture(async(path,method,payload)=>{if(!method)return stored;sent.push(payload);if(conflict)throw Object.assign(new Error('conflit'),{status:409});const {expected_revision,...changes}=payload;stored={profile:{...stored.profile,...changes},revision:stored.revision+1};return stored});
 field(env.form,'first_name','Prénom local');await env.form.fire('input');stored={profile:{first_name:'Prénom distant',last_name:'Nom distant',self_description:'Historique distant'},revision:3};
 await env.form.fire('submit');await env.form.querySelector('[data-reload-source]').fire('click');assert.equal(env.ctx.state.selfRevision,2);
 await env.form.querySelector('.notice').querySelector('button').fire('click');assert.equal(env.form.elements.first_name.value,'Prénom local');assert.equal(env.form.elements.last_name.value,'Nom distant');
 conflict=false;await env.form.fire('submit');assert.deepEqual(Object.keys(sent[1]).sort(),['expected_revision','first_name']);assert.equal(sent[1].expected_revision,3);assert.equal(stored.profile.self_description,'Historique distant');
});

test('gender details appear when requested and preserve previously declared words',async()=>{
 const env=await selfFixture(async()=>({profile:{gender:'unspecified',gender_description:'',self_description:''},revision:1}));const gender=env.form.elements.gender,description=env.form.elements.gender_description;
 assert.equal(description.parentElement.hidden,true);gender.value='self_described';await gender.fire('change');await env.form.fire('change');assert.equal(description.parentElement.hidden,false);
 description.value='Mes mots';await description.fire('input');await env.form.fire('input');gender.value='female';await gender.fire('change');await env.form.fire('change');assert.equal(description.parentElement.hidden,true);assert.equal(description.value,'Mes mots');assert.equal(env.ctx.state.selfDraft.gender_description,'Mes mots');
});

test('V4 global navigation has exactly the shared public links in source order',()=>{
  const navigation=JSON.parse(readFileSync(new URL('../../data/navigation.json',import.meta.url),'utf8'));
  const menu=HTML.match(/<ul class="nav-links">([\s\S]*?)<\/ul>/)[1];
  const links=[...menu.matchAll(/<a href="([^"]+)">([^<]+)<\/a>/g)].map(match=>({href:match[1],label:match[2]}));
  assert.deepEqual(links,navigation);assert.equal(links.length,6);assert.deepEqual(links[0],{href:'/mon-accompagnement/',label:'Ambre'});assert.deepEqual(links.map(link=>link.label),['Ambre','Mon enfance','Mes proches','Ma ligne de vie','Mes bilans','Mon compte']);assert.equal(links.some(link=>link.label==='Mon portrait'),false);assert.equal(links.some(link=>link.href==='/parametres/'),false);assert.doesNotMatch(menu,/Se connecter|Se déconnecter/);
  assert.match(readFileSync(new URL('../../layouts/partials/header.html',import.meta.url),'utf8'),/\.Site\.Data\.navigation/);
});

function navigationFixture(preparePageData){
 const location={pathname:'/mon-portrait/',search:'',hash:''},rendered=[];
 const env=environment({location,preparedResponses:{},preparePageData,history:{pushState(_state,_unused,path){location.pathname=path}},window:{scrollTo(){}},capturePageDrafts(){},syncPageLocation(){env.ctx.state.tab=env.ctx.pageForPath(location.pathname)},renderApp(){rendered.push(env.ctx.state.tab)}});
 env.ctx.state.me={user:{id:'owner'}};env.ctx.state.tab='portrait';vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('personIdForPath')+'\n'+functionCode('pagePath')+'\n'+functionCode('pageForPath')+'\n'+functionCode('go'),env.ctx);
 return {...env,location,rendered};
}

test('private navigation retains the complete current page and URL until destination data arrives',async()=>{
 let resolve;const pending=new Promise(r=>resolve=r),env=navigationFixture(()=>pending),responses={'/api/people':{status:200,body:{people:[]}}};
 const moving=env.ctx.go('people');assert.equal(env.location.pathname,'/mon-portrait/');assert.equal(env.ctx.state.tab,'portrait');assert.deepEqual(env.rendered,[]);
 resolve(responses);await moving;assert.equal(env.location.pathname,'/mes-proches/');assert.deepEqual(env.rendered,['people']);assert.equal(env.ctx.preparedResponses,responses);
});

test('rapid navigation only renders the latest requested destination, even when old data arrives last',async()=>{
 const releases={};const env=navigationFixture(tab=>new Promise(resolve=>releases[tab]=resolve));
 const first=env.ctx.go('people'),last=env.ctx.go('events');releases.events({events:'latest'});await last;releases.people({people:'old'});await first;
 assert.equal(env.location.pathname,'/ma-ligne-de-vie/');assert.deepEqual(env.rendered,['events']);assert.equal(env.ctx.preparedResponses.events,'latest');
});

test('navigation data from the previous signed-in owner cannot be installed after sign-out',async()=>{
 let resolve;const env=navigationFixture(()=>new Promise(r=>resolve=r));const moving=env.ctx.go('people');env.ctx.state.me={user:null};resolve({'/api/people':{status:200,body:{people:[{display_name:'Private'}]}}});await moving;
 assert.equal(env.location.pathname,'/mon-portrait/');assert.deepEqual(env.rendered,[]);assert.deepEqual(Object.keys(env.ctx.preparedResponses),[]);
});

test('prepared API responses are consumed once; writes and subsequent reads use the network',async()=>{
 const calls=[],env=environment({preparedResponses:{'/api/self-profile':{status:200,body:{profile:{first_name:'Initial'}}}},fetch:async(path,options)=>{calls.push({path,options});return {ok:true,json:async()=>({profile:{first_name:'Fresh'}})}}});vm.runInContext(functionCode('api'),env.ctx);
 assert.equal((await env.ctx.api('/api/self-profile')).profile.first_name,'Initial');assert.equal(calls.length,0);
 assert.equal((await env.ctx.api('/api/self-profile')).profile.first_name,'Fresh');await env.ctx.api('/api/self-profile','PUT',{first_name:'Edited',expected_revision:1});
 assert.equal(calls.length,2);assert.equal(calls[0].options.cache,'no-store');assert.equal(calls[1].options.method,'PUT');
 env.ctx.preparedResponses={'/api/self-profile':{status:409,body:{error:'Conflit',code:'revision_conflict'}}};await assert.rejects(env.ctx.api('/api/self-profile'),error=>error.status===409&&error.code==='revision_conflict');assert.deepEqual(Object.keys(env.ctx.preparedResponses),[]);
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
  assert.equal(toolbar.querySelector('h1').textContent,'Ambre · assistante psychologue IA');
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

test('all removed portrait and reading subpage links lead to the single presentation',()=>{
 for(const hash of ['#ma-cartographie','#cartographie','#mon-portrait','#synthese','#correction-form','#self-profile','#mes-articles','#suggestions','']){
  const location={hash,pathname:'/mon-portrait/',search:'?article=old'};const historyCalls=[];
  const env=environment({location,history:{replaceState(a,b,url){historyCalls.push(url);location.hash=url.includes('#')?url.slice(url.indexOf('#')):''}}});
  const view=env.host('view'),form=env.doc.createElement('form');view.append(form);
  vm.runInContext(functionCode('setupPortraitViews'),env.ctx);env.ctx.setupPortraitViews(view);
  const obsolete=['#ma-cartographie','#cartographie','#mon-portrait','#synthese','#correction-form','#mes-articles','#suggestions'].includes(hash);
  assert.equal(location.hash,obsolete?'#self-profile':hash);assert.equal(historyCalls.length,obsolete?1:0);assert.equal(view.children[0],form);
  if(obsolete)assert.equal(historyCalls[0],'/mon-portrait/?article=old#self-profile');
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



test('the lists of close people and of moments repeat neither the page title nor any counter',()=>{
 const env=environment();vm.runInContext(functionCode('emptyState')+'\n'+functionCode('drawSourceSection'),env.ctx);env.ctx.drawSourceItem=()=>env.doc.createElement('article');env.ctx.drawSourceEditor=()=>{};
 const people=['Anaïs','Bernard','Chloé','Driss','Élise'].map((display_name,index)=>({id:'p'+index,display_name,role:'friend'}));
 const events=[{id:'e1',title:'Un déclic',date_precision:'year',date_start:'2020'},{id:'e2',title:'Sans date',date_precision:'unknown'}];
 for(const [kind,rows,cursor] of [['people',people,null],['people',[],null],['events',[],null],['events',events.slice(0,1),null],['events',events,'next-page']]){
  const box=env.host(kind+'-content');env.ctx.state[kind]=rows;env.ctx.state.eventsCursor=cursor;env.ctx.drawSourceSection(kind);
  const toolbar=box.querySelector('.source-toolbar');assert.equal(toolbar.textContent,'');assert.equal(toolbar.children.length,0);assert.equal(box.querySelector('h2')?.textContent??null,rows.length?null:kind==='people'?'Aucun proche pour l’instant':'Aucun moment pour l’instant');
  assert.equal(box.querySelector('.source-count'),null);assert.doesNotMatch(box.textContent,/\d+ (?:proches?|moments?)|affichés/);
  assert.equal(box.querySelectorAll('article').length,rows.length);assert.equal(box.querySelectorAll('.pagination-button').length,cursor?1:0);
 }
 assert.doesNotMatch(HTML,/source-count/);assert.doesNotMatch(functionCode('drawSourceSection'),/affichés|' proches'|' moments'/);assert.match(HTML,/\.source-tag \{/);
});

test('the application footer holds the site name (to Ambre) and the catalogue link, and liked articles stay out of the portrait menu',()=>{
 assert.equal(HTML.match(/<footer\b/g).length,1);assert.match(HTML,/<footer class="site-footer">\s*<div class="footer-inner footer-minimal">\s*<a href="\/mon-accompagnement\/" class="site-logo">Décode ton esprit<\/a>\s*<\/div>\s*<\/footer>/);assert.doesNotMatch(functionCode('renderPortrait'),/data-portrait-tab="readings"/);
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
 const timers=[],calls=[],applied=[];const env=environment({setTimeout(fn){timers.push(fn);return timers.length},clearTimeout(){},api:async(path)=>{calls.push(path);return {status:'ready',source_version:5}},applyPortraitData(data){applied.push(data)}});env.ctx.state.tab='portrait';env.host('portrait-body');for(const name of ['refreshPortraitData','schedulePortraitPoll'])vm.runInContext(functionCode(name),env.ctx);env.ctx.schedulePortraitPoll(()=>true,{status:'queued'});assert.equal(timers.length,1);await timers.shift()();assert.deepEqual(calls,['/api/portrait']);assert.equal(applied.length,1);assert.equal(timers.length,0);
 env.ctx.schedulePortraitPoll(()=>true,{status:'updating'});env.ctx.state.tab='chat';await timers.shift()();assert.equal(calls.length,1);
});

test('V6 free preview account displays access and quotas without checkout or subscription offers',()=>{
 const env=environment();const host=env.host('view'),accountRow=env.host('account-row');host.querySelector=()=>accountRow;for(const id of ['account-email','account-price','subscription-copy','account-progress','billing-action','export-button','delete-button'])env.host(id);env.ctx.state.me={user:{email:'test@example.test'},access:{mode:'free_preview',allowed:true},subscription:{active:false,trial_eligible:true},configuration:{billing:true,ai:true}};env.doc.getElementById('account-email').parentElement=accountRow;vm.runInContext(functionCode('appendAccountQuota'),env.ctx);env.ctx.setupPortraitViews=()=>{};env.ctx.renderSelfProfile=()=>{};env.ctx.logout=()=>{};vm.runInContext(functionCode('renderAccount'),env.ctx);env.ctx.renderAccount();assert.equal(env.doc.getElementById('account-price').hidden,true);assert.match(env.doc.getElementById('subscription-copy').textContent,/Accès gratuit/);assert.doesNotMatch(env.doc.getElementById('billing-action').textContent,/abonner|essai|factur|29/);assert.doesNotMatch(env.doc.getElementById('account-progress').textContent,/29|7 jours d’essai/);
});

test('V6 re-saving identical map feedback replays its saved key and edits renew it',async()=>{
 const calls=[];let sequence=0;const env=mapFixture({crypto:{randomUUID:()=> 'key-'+(++sequence)},api:async(path,method,payload)=>{calls.push(payload);return {memory_version:calls.length}}});const form=env.host.mapCards.get('belong').card.querySelector('form');field(form,'text','Un retour explicite');await form.elements.text.fire('input');await form.fire('submit');await form.fire('submit');assert.equal(calls[0].creation_id,calls[1].creation_id);field(form,'text','Un retour précisé');await form.elements.text.fire('input');await form.fire('submit');assert.notEqual(calls[1].creation_id,calls[2].creation_id);
});


test('V6 portrait refresh rejects older source and analyzed snapshots before replacing visible content',()=>{
 const env=environment();env.ctx.state.portraitData={source_version:9,analyzed_version:8};vm.runInContext(functionCode('applyPortraitData'),env.ctx);const latest=env.ctx.state.portraitData;env.ctx.applyPortraitData({source_version:8,analyzed_version:8},()=>true);assert.equal(env.ctx.state.portraitData,latest);env.ctx.applyPortraitData({source_version:9,analyzed_version:7},()=>true);assert.equal(env.ctx.state.portraitData,latest);
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

test('assistant messages render no reading suggestion; notes written on the map and reviews are marked',()=>{
 const env=chatFixture();env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:99},messages:[
  {id:'t1-assistant',role:'assistant',content:'Une réponse.',reading_suggestions:[{id:'a',title:'Une lecture',url:'/lecture/',character:'Nina',reason:'Raison.'}],cartography_notes:[{target:'factor',target_id:'f',label:'Cris à la maison',text:'Vous dites que votre père criait souvent.'},{target:'general',target_id:null,text:'Vous voulez oser dire non.'},{target:'general',text:'  '}]},
  {id:'t2-user',role:'user',content:'Merci',cartography_notes:[{target:'general',text:'Ne pas afficher'}]},
  {id:'t3-assistant',role:'assistant',content:'Premier paragraphe.\n\nSecond paragraphe.',review_id:'r1',review_kind:'initial'},
  {id:'t4-assistant',role:'assistant',content:'Un point.',review_id:'r2',review_kind:'point'}],proposals:[]};
 env.ctx.drawChat(env.view,()=>true);const messages=env.view.querySelector('.chat-thread').children;
 assert.equal(env.view.querySelector('.chat-readings'),null);assert.doesNotMatch(env.view.textContent,/Une lecture|Nina|Pour prolonger/);
 const noted=messages[0].querySelector('.chat-noted');assert.match(noted.textContent,/^Noté dans Mon enfance : Cris à la maison — « Vous dites que votre père criait souvent\. » · « Vous voulez oser dire non\. »/);
 assert.equal(noted.querySelector('a').href,'/cartographie/');assert.equal(messages[1].querySelector('.chat-noted'),null);
 assert.equal(messages[2].querySelector('h3').textContent,'Ambre · Analyse initiale');assert.equal(messages[3].querySelector('h3').textContent,'Ambre · Point d’étape');
 assert.equal(messages[2].querySelector('.chat-review-link').href,'/mes-bilans/');assert.equal(messages[0].querySelector('.chat-review-link'),null);
 env.ctx.drawChat(env.view,()=>true);assert.equal(messages[0].querySelectorAll('.chat-noted').length,1);assert.equal(messages[2].querySelectorAll('.chat-review-link').length,1);
 assert.doesNotMatch(script,/drawChatReadings|reading_suggestions|Lectures suggérées/);
});

const journeyState=(journey,extra={})=>({consent:false,access:{allowed:false,reason:'consent_required'},quota:{used:0,remaining:100,limit:100},messages:[],proposals:[],journey,...extra});
const notStarted=(overrides={})=>({started:false,can_start:true,sources:{cartography:{done:true,count:7},portrait:{done:false},people:{done:true,count:1},life_events:{done:false,count:0}},review:{running_id:null,last_at:null,next_point_at:null},...overrides});

test('the welcome panel lists the four sources, confirms with an explicit agreement, then waits for the initial analysis',async()=>{
 const calls=[];let next;
 const env=chatFixture({api:async(path,method,data)=>{calls.push([path,method||'GET',data]);return next;}});
 env.ctx.state.chatData=journeyState(notStarted({can_start:false,sources:{cartography:{done:false,count:0},portrait:{done:false},people:{done:false,count:0},life_events:{done:false,count:0}}}));
 env.ctx.drawChat(env.view,()=>true);
 const dialog=env.doc.getElementById('journey-panel');
 assert.equal(dialog.hidden,false);assert.equal(dialog.dataset.step,'intro');assert.equal(dialog.attributes['aria-labelledby'],'journey-title');
 assert.equal(dialog.querySelector('h2').textContent,'Avant de commencer');assert.equal(env.doc.activeElement,dialog.querySelector('h2'));
 assert.match(dialog.textContent,/analyse approfondie de ce que vous avez renseigné, puis la conversation se poursuit/);
 let items=dialog.querySelectorAll('.journey-source');assert.equal(items.length,4);
 assert.deepEqual(items.map(i=>i.querySelector('a').href),['/cartographie/','/mon-compte/','/mes-proches/','/ma-ligne-de-vie/']);
 assert.ok(items.every(i=>i.querySelector('.journey-source-state').textContent==='À compléter'));
 let start=env.doc.getElementById('journey-start');assert.equal(start.disabled,true);assert.match(env.doc.getElementById('journey-hint').textContent,/Complétez vos informations/);
 assert.equal(env.send.disabled,true);assert.equal(env.view.querySelector('.chat-gate'),null,'the agreement is asked in the panel, not in a second gate');
 assert.equal(dialog.tagName,'SECTION');assert.equal(dialog.parentElement.id,'conversation-scroll');assert.match(dialog.textContent,/assistante psychologue IA/);assert.match(dialog.textContent,/à vos côtés un professionnel de santé/);assert.match(dialog.textContent,/ne pose pas de diagnostic/);
 env.ctx.state.chatData=journeyState(notStarted());env.ctx.drawChat(env.view,()=>true);
 items=dialog.querySelectorAll('.journey-source');assert.deepEqual(items.map(i=>i.querySelector('.journey-source-state').textContent),['Renseigné','À compléter','Renseigné','À compléter']);
 assert.match(items[0].textContent,/7 réponses/);assert.match(items[2].textContent,/1 proche/);
 start=env.doc.getElementById('journey-start');assert.equal(start.disabled,false);await start.fire('click');
 assert.equal(dialog.dataset.step,'confirm');assert.equal(dialog.querySelector('h2').textContent,'Voulez-vous lancer la discussion ?');
 assert.match(dialog.textContent,/Cela va initier la conversation à partir des informations que vous avez renseignées\./);
 let buttons=dialog.querySelectorAll('button');assert.deepEqual(buttons.map(b=>b.textContent),['Oui, lancer','Pas encore']);assert.equal(buttons[0].disabled,true);
 await buttons[1].fire('click');assert.equal(dialog.dataset.step,'intro');assert.equal(calls.length,0);
 await env.doc.getElementById('journey-start').fire('click');buttons=dialog.querySelectorAll('button');
 const checkbox=env.doc.getElementById('journey-consent');checkbox.checked=true;await checkbox.fire('change');assert.equal(buttons[0].disabled,false);
 next=journeyState({started:true,can_start:false,sources:notStarted().sources,review:{running_id:'r1',last_at:null,next_point_at:null}},{consent:true,access:{allowed:true,reason:null},review_id:'r1'});
 await buttons[0].fire('click');
 assert.equal(calls.length,1);assert.equal(calls[0][0],'/api/chat/start');assert.equal(calls[0][1],'POST');assert.equal(calls[0][2].accepted,true);assert.match(calls[0][2].idempotency_key,/^[\w-]{8,}$/);
 assert.equal(dialog.dataset.step,'waiting');assert.equal(dialog.hidden,false);assert.equal(dialog.querySelector('h2').textContent,'Ambre prépare votre analyse…');assert.equal(dialog.querySelectorAll('button').length,0);
 assert.equal(env.send.disabled,true);
 env.ctx.state.chatData=journeyState({started:true,can_start:false,sources:notStarted().sources,review:{running_id:null,last_at:100,next_point_at:Math.floor(Date.now()/1000)+3600}},{consent:true,access:{allowed:true,reason:null},messages:[{id:'t1-assistant',role:'assistant',content:'Votre analyse.',review_id:'r1',review_kind:'initial'}]});
 env.ctx.drawChat(env.view,()=>true);
 assert.equal(dialog.hidden,true);assert.equal(env.view.querySelector('.chat-thread').children.at(-1).querySelector('h3').textContent,'Ambre · Analyse initiale');
 env.input.value='Bonjour';assert.equal(env.send.disabled,false);
});

test('a failed initial analysis returns to the welcome panel with an explanation and a fresh key',async()=>{
 const keys=[];const env=chatFixture({api:async(_path,_method,data)=>{keys.push(data.idempotency_key);return journeyState({started:true,can_start:false,sources:notStarted().sources,review:{running_id:'r'+keys.length,last_at:null,next_point_at:null}},{consent:true,access:{allowed:true}});}});
 const launch=async()=>{await env.doc.getElementById('journey-start').fire('click');const box=env.doc.getElementById('journey-consent');box.checked=true;await box.fire('change');await env.doc.getElementById('journey-panel').querySelectorAll('button')[0].fire('click');};
 env.ctx.state.chatData=journeyState(notStarted());env.ctx.drawChat(env.view,()=>true);await launch();
 const dialog=env.doc.getElementById('journey-panel');assert.equal(dialog.dataset.step,'waiting');
 env.ctx.state.chatData=journeyState(notStarted(),{consent:true,access:{allowed:false,reason:'journey_not_started'}});env.ctx.drawChat(env.view,()=>true);
 assert.equal(dialog.dataset.step,'intro');assert.match(dialog.textContent,/n’a pas pu être préparée/);
 await launch();assert.equal(keys.length,2);assert.notEqual(keys[0],keys[1]);
});

test('“Faire le point” explains why it is unavailable, confirms lightly and shows the wait in the thread',async()=>{
 const calls=[];let next;const soon=Math.floor(Date.now()/1000)+7200;
 const started=(review,extra={})=>journeyState({started:true,can_start:false,sources:notStarted().sources,review},{consent:true,access:{allowed:true,reason:null},messages:[{id:'t1-assistant',role:'assistant',content:'Analyse.',review_id:'r1',review_kind:'initial'}],...extra});
 const env=chatFixture({api:async(path,method,data)=>{calls.push([path,method||'GET',data]);return next;}});
 assert.equal(env.doc.getElementById('chat-point').hidden,true,'no journey information: the button stays hidden');
 env.ctx.state.chatData=started({running_id:null,last_at:100,next_point_at:soon});env.ctx.drawChat(env.view,()=>true);
 const point=env.doc.getElementById('chat-point'),note=env.doc.getElementById('chat-point-note');
 assert.equal(point.hidden,false);assert.equal(point.disabled,true);assert.equal(note.hidden,false);assert.match(note.textContent,/^Prochain point possible à partir du /);
 await point.fire('click');assert.notEqual(env.doc.getElementById('journey-panel')?.dataset.step,'point');
 env.ctx.state.chatData=started({running_id:null,last_at:100,next_point_at:null});env.ctx.drawChat(env.view,()=>true);
 assert.equal(point.disabled,false);assert.equal(note.hidden,true);
 await point.fire('click');const dialog=env.doc.getElementById('journey-panel');
 assert.equal(dialog.hidden,false);assert.equal(dialog.dataset.step,'point');assert.equal(dialog.querySelector('h2').textContent,'Faire le point ?');
 let buttons=dialog.querySelectorAll('button');assert.deepEqual(buttons.map(b=>b.textContent),['Oui, faire le point','Pas maintenant']);
 await buttons[1].fire('click');assert.equal(dialog.hidden,true);assert.equal(calls.length,0);assert.equal(env.doc.activeElement,point);
 await point.fire('click');next=started({running_id:'r2',last_at:100,next_point_at:null},{review_id:'r2'});
 await dialog.querySelectorAll('button')[0].fire('click');
 assert.deepEqual(calls.map(c=>[c[0],c[1]]),[['/api/chat/review','POST']]);assert.match(calls[0][2].idempotency_key,/^[\w-]{8,}$/);
 assert.equal(dialog.hidden,true);const waiting=env.doc.getElementById('chat-review-indicator');
 assert.match(waiting.textContent,/Ambre prépare votre bilan…/);assert.equal(waiting.parentElement,env.view.querySelector('.chat-thread'));
 env.input.value='Un message';assert.equal(env.send.disabled,true);assert.equal(point.disabled,true);assert.match(note.textContent,/prépare déjà un bilan/);
 env.ctx.state.chatData=started({running_id:null,last_at:200,next_point_at:soon},{messages:[{id:'t1-assistant',role:'assistant',content:'Analyse.',review_id:'r1',review_kind:'initial'},{id:'t2-assistant',role:'assistant',content:'Le point.',review_id:'r2',review_kind:'point'}]});
 env.ctx.drawChat(env.view,()=>true);
 assert.equal(waiting.isConnected,false);assert.equal(env.send.disabled,false);assert.equal(env.view.querySelector('.chat-thread').children.at(-1).querySelector('h3').textContent,'Ambre · Point d’étape');
});

test('“Mes bilans” lists dated reviews, reads one in full, deletes with confirmation and explains an empty history',async()=>{
 const calls=[];let confirmed=false,answer=true;
 const responses={'/api/reviews':{reviews:[{id:'r2',kind:'point',status:'ready',created_at:1794700800,completed_at:1794700900,excerpt:'Depuis le dernier bilan',length:4000},{id:'r1',kind:'initial',status:'ready',created_at:1794600000,completed_at:1794600100,excerpt:'Ce qui semble ressortir',length:5000},{id:'r0',kind:'point',status:'failed',created_at:1794500000,completed_at:1794500100,excerpt:'',length:0}],next_cursor:null},'/api/reviews/r1':{review:{id:'r1',kind:'initial',status:'ready',content:'Premier paragraphe.\n\nSecond paragraphe.'}}};
 const env=environment({api:async(path,method)=>{calls.push([path,method||'GET']);return method==='DELETE'?{deleted:true}:responses[path];},confirm:message=>{confirmed=message;return answer;},status(id,message){env.doc.getElementById(id).textContent=message;}});
 for(const name of ['reviewTitle','renderReviews','drawReviewPage','drawReviewItem'])vm.runInContext(functionCode(name),env.ctx);
 env.ctx.state.tab='reviews';const view=env.host('view');await env.ctx.renderReviews();
 const items=view.querySelectorAll('.review-item');assert.equal(items.length,3);
 assert.match(items[0].querySelector('h2').textContent,/^Point du \d{1,2} novembre 2026$/);assert.equal(items[1].querySelector('h2').textContent,'Analyse initiale');
 assert.match(items[1].textContent,/Ce qui semble ressortir…/);assert.match(items[2].textContent,/n’a pas abouti/);assert.equal(items[2].querySelectorAll('button').length,1,'a failed review can only be removed');
 const [read,remove]=items[1].querySelectorAll('button');await read.fire('click');
 assert.deepEqual(calls.at(-1),['/api/reviews/r1','GET']);assert.deepEqual(items[1].querySelector('.review-body').children.map(p=>p.textContent),['Premier paragraphe.','Second paragraphe.']);assert.equal(read.textContent,'Replier');
 await read.fire('click');assert.equal(items[1].querySelector('.review-body').hidden,true);assert.equal(calls.filter(c=>c[0]==='/api/reviews/r1').length,1,'the full text is fetched once');
 answer=false;await remove.fire('click');assert.match(confirmed,/Analyse initiale/);assert.match(confirmed,/relancée/);assert.equal(calls.some(c=>c[1]==='DELETE'),false);
 answer=true;await remove.fire('click');assert.deepEqual(calls.at(-1),['/api/reviews/r1','DELETE']);assert.equal(view.querySelectorAll('.review-item').length,2);
 responses['/api/reviews']={reviews:[],next_cursor:null};await env.ctx.renderReviews();
 assert.equal(view.querySelectorAll('.review-item').length,0);assert.match(view.textContent,/Aucun bilan pour le moment/);assert.match(view.textContent,/Faire le point/);
 const empty=view.querySelector('.workspace-empty');assert.equal(empty.querySelector('a'),null,'the empty history no longer opens the conversation');assert.equal(empty.querySelector('button'),null);assert.doesNotMatch(empty.textContent,/Ouvrir la conversation/);
});

test('the application shell routes “Mes bilans” like the other personal pages',()=>{
 assert.match(script,/reviews:\{path:'\/mes-bilans\/',title:'Mes bilans'\}/);assert.match(functionCode('renderApp'),/state\.tab==='reviews'\)renderReviews\(\)/);
 assert.doesNotMatch(HTML,/<dialog[^>]*journey/);
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
 Object.assign(env.ctx.state,{tab:'chat',chatDraft:'Mes mots complets',chatSending:true,chatSendKey:{key:'failed-key'},chatData:data});env.ctx.drawChatProposal=()=>{};vm.runInContext(functionCode('drawChat')+'\n'+functionCode('pollChat'),env.ctx);const host=env.host('view');env.ctx.drawChat(host,()=>true);const input=env.doc.getElementById('chat-content'),form=env.doc.getElementById('chat-form'),button=form.querySelector('button[type=submit]');
 env.ctx.api=async(path,method)=>{calls.push({path,method});return {...data,access:{allowed:false,reason:'provider_cooldown',retry_at:(clock+45000)/1000},last_request:{id:'failed',status:'failed'},last_error:{code:'provider_http_429_request',message:'Le fournisseur limite temporairement les demandes. Réessayez dans 45 secondes. Aucun message n’a été décompté.',action:'retry_later',retry_at:(clock+45000)/1000}}};env.ctx.pollChat({id:'failed'},()=>true,'Mes mots complets');await timers.shift().fn();assert.match(env.doc.getElementById('chat-status').textContent,/Réessayez dans 45 secondes/);assert.match(env.doc.getElementById('chat-status').textContent,/Aucun message n’a été décompté/);assert.equal(env.ctx.state.chatSendKey,null);assert.equal(button.disabled,true);assert.equal(input.value,'Mes mots complets');assert.equal(env.ctx.state.chatData.quota.remaining,100);
 input.value='Un complément pendant l’attente';await input.fire('input');assert.ok(timers[0].delay>=45000&&timers[0].delay<=60000);now=clock+46000;env.ctx.api=async(path,method)=>{calls.push({path,method});return data};await timers.shift().fn();assert.equal(button.disabled,false);assert.equal(env.doc.getElementById('chat-form'),form);assert.equal(input.value,'Un complément pendant l’attente');assert.equal(env.ctx.state.chatDraft,input.value);assert.ok(calls.every(call=>!call.method&&(call.path==='/api/chat'||/^\/api\/chat\/requests\/[\w-]+\?view=status$/.test(call.path))));
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
 const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('drawChat'),env.ctx);assert.equal(env.ctx.PAGES.chat.title,'Ambre');assert.equal(env.ctx.PAGES.chat.path,'/mon-accompagnement/');env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:100},messages:[{role:'assistant',content:'Bonjour.'},{role:'user',content:'Mon accompagnement, dans mes propres mots.'}],proposals:[]};const host=env.host('view');env.ctx.drawChat(host,()=>true);assert.match(host.textContent,/Ambre · assistante psychologue IA/);assert.equal(host.querySelector('.chat-assistant').querySelector('h3').textContent,'Ambre');assert.match(host.querySelector('.chat-user').textContent,/Mon accompagnement, dans mes propres mots/);assert.doesNotMatch(host.textContent,/Assistant IA/);assert.equal(host.querySelectorAll('h1').length,1);assert.doesNotMatch(functionCode('landing'),/<h1>Ambre<\/h1>/);
});

test('the single portrait renders only its presentation without requesting readings or a synthesis',async()=>{
 const calls=[],env=environment({api:async(path)=>{calls.push(path);return {status:'ready'}},renderSelfProfile(){calls.push('self')},renderLikes(){calls.push('readings')},setupPortraitViews(){},applyPortraitData(){},schedulePortraitPoll(){}});
 env.ctx.state.tab='portrait';env.host('view');for(const name of ['renderPortrait','refreshPortraitData'])vm.runInContext(functionCode(name),env.ctx);
 await env.ctx.renderPortrait();assert.deepEqual(calls,['self']);
 const view=env.doc.getElementById('view');assert.equal(env.doc.getElementById('self-profile').hidden,false);assert.equal(env.doc.getElementById('mes-articles'),null);
 assert.equal(env.doc.getElementById('portrait-body'),null);assert.doesNotMatch(functionCode('renderPortrait'),/data-portrait-tab|portrait-synthesis/);
 await env.ctx.refreshPortraitData();assert.deepEqual(calls,['self']);
});

test('portrait transport failures leave useful options and a retry, while a failed synthesis keeps its provisional reading',async()=>{
 const calls=[];let unavailable=true;
 const env=environment({api:async(path)=>{
  calls.push(path);if(unavailable)throw new Error('Backend transport details');
  return {status:'error',source_version:2,analyzed_version:0,provisional_portrait:{summary:'Piste liée à mes lectures.'}};
 }});
 env.ctx.state.tab='portrait';env.ctx.state.portraitCurrent=()=>true;
 for(const name of ['appendPortrait','personalizePortraitText','emptyState','appendPortraitEmpty','applyPortraitData','showPortraitLoadFailure','refreshPortraitData','schedulePortraitPoll'])vm.runInContext(functionCode(name),env.ctx);
 const body=env.host('portrait-body'),statusBox=env.host('portrait-status');
 await env.ctx.refreshPortraitData();
 assert.match(body.textContent,/Voir ma présentation/);assert.doesNotMatch(body.textContent,/Voir mes articles aimés/);
 assert.equal(body.querySelector('a').href,'/mon-accompagnement/');assert.doesNotMatch(statusBox.textContent,/Backend transport details/);
 const retry=statusBox.querySelector('button');assert.equal(retry.className,'text');
 unavailable=false;await retry.fire('click');
 const reading=body.querySelector('.portrait-reading');assert.match(reading.textContent,/Piste liée à mes lectures/);
 assert.match(statusBox.textContent,/La synthèse complète n’a pas pu être préparée/);assert.equal(statusBox.querySelector('button'),null);
 unavailable=true;await env.ctx.refreshPortraitData();
 assert.equal(body.querySelector('.portrait-reading'),reading);assert.match(statusBox.textContent,/n’a pas pu être actualisé/);
 assert.equal(statusBox.querySelector('button').textContent,'Réessayer');assert.deepEqual(calls,Array(3).fill('/api/portrait'));
});

test('manual presentation, close-person and life-event forms omit the repeated Ambre notes',async()=>{
 const env=environment({api:async()=>({profile:{first_name:'Ma présentation'},revision:1}),location:{hash:''}});env.host('self-profile');vm.runInContext(functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();const self=env.doc.getElementById('self-profile');assert.equal(self.querySelector('.ambre-hint'),null);assert.equal(env.doc.getElementById('self-profile-form').elements.first_name.value,'Ma présentation');
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
 vm.runInContext(functionCode('personIdForPath')+'\n'+functionCode('pageForPath')+'\n'+script.match(/var PAGES=.*\n/)[0]+'\n'+script.match(/var state=.*\n/)[0],env.ctx);
 for(const name of ['safeArticleLink','drawLikedArticles'])vm.runInContext(functionCode(name),env.ctx);
 env.ctx.state.examples=[{id:'a',title:'Une lecture',url:'/posts/une-lecture/'}];
 env.ctx.state.likes=[{article_id:'a',reason:'Mon retour'}];
 const list=env.host('liked');env.ctx.drawLikedArticles(list,()=>true);
 assert.match(list.textContent,/Une lecture/);
 const input=env.doc.getElementById('reason-0');input.value='Mon brouillon';await input.fire('input');
 env.ctx.drawLikedArticles(list,()=>true);
 assert.equal(env.doc.getElementById('reason-0').value,'Mon brouillon');
});

test('leaving an untouched presentation shows later Ambre updates; a real draft retains its original revision',async()=>{
 let revision=1;const accepted={1:'Avant',2:'Après Ambre',3:'Encore précisé avec Ambre'};
 const env=await selfFixture(async()=>({profile:{first_name:accepted[revision]},revision}));vm.runInContext(functionCode('capturePageDrafts'),env.ctx);
 assert.equal(env.input.value,'Avant');env.ctx.capturePageDrafts();revision=2;await env.ctx.renderSelfProfile();assert.equal(env.input.value,'Après Ambre');assert.equal(env.ctx.state.selfRevision,2);
 env.input.value='Mon brouillon';field(env.form,'last_name','Nom local');await env.form.fire('input');env.ctx.capturePageDrafts();revision=3;await env.ctx.renderSelfProfile();
 assert.equal(env.input.value,'Mon brouillon');assert.equal(env.ctx.state.selfRevision,2);assert.equal(env.ctx.state.selfDraft.first_name,'Mon brouillon');assert.equal(env.form.elements.last_name.value,'Nom local');assert.equal(env.ctx.state.selfBase.first_name,'Après Ambre');
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

test('A5-P2 optional relation fields use shared choices and limits, retain input after failure, and allow clearing',async()=>{
 const sent=[],env=environment({api:async(path,method,payload)=>{sent.push(payload);throw new Error('Connexion interrompue')}});env.host('people-editor');env.ctx.state.peopleFieldsReady=true;
 const values={display_name:'Camille',role:'friend',bond_quality:'ambivalent',bond_note:'Une nuance',presence_now:'regular',helps:'Nos échanges',hurts:'Les silences',evolution:'Plus de distance',feels_safe:'not_always'};
 env.ctx.state.sourceDrafts.set('people',{id:'p1',revision:4,values});env.ctx.drawSourceEditor('people');const form=env.doc.getElementById('people-form');
 for(const name of ['bond_quality','bond_note','presence_now','helps','hurts','evolution','feels_safe']){assert.equal(form.elements[name].value,values[name]);assert.ok(!form.elements[name].required)}
 assert.equal(form.elements.bond_note.maxLength,300);assert.equal(form.elements.helps.maxLength,600);assert.equal(form.elements.feels_safe.querySelectorAll('option')[0].value,'');
 await form.fire('submit');assert.equal(sent[0].revision,4);assert.equal(sent[0].feels_safe,'not_always');assert.equal(form.elements.hurts.value,'Les silences');
 form.elements.feels_safe.value='';form.elements.hurts.value='';await form.fire('submit');assert.equal(sent[1].feels_safe,'');assert.equal(sent[1].hurts,'');assert.equal(env.ctx.state.sourceDrafts.get('people').values.hurts,'');
});

test('reduced event editor omits historical optional fields from every write',async()=>{
 let sent;const hidden={meaning_now:'Mon sens',current_importance:4,felt_then:'Surprise',felt_now:'Apaisement',impact:'Un autre quotidien',support_level:'some',support_text:'Une présence',exposure:'witnessed'};
 const env=editFixture('events',{title:'Un changement',kind:'ambivalent',date_precision:'unknown',description:'Mon récit',people_ids:[],...hidden},{api:async(path,method,payload)=>{sent=payload;throw new Error('Indisponible')}});
 for(const key of Object.keys(hidden))assert.equal(env.form.elements[key],undefined);await env.form.fire('submit');for(const key of Object.keys(hidden))assert.equal(key in sent,false);
 assert.equal(sent.description,'Mon récit');assert.equal(env.form.elements.description.value,'Mon récit');
});

test('A5-P2 pre-migration forms omit new payload fields while preserving a newer unsaved draft',async()=>{
 let sent;const env=editFixture('people',{display_name:'Camille',role:'friend',helps:'Texte non enregistré',feels_safe:'no'},{api:async(path,method,payload)=>{sent=payload;throw new Error('Hors ligne')}});
 assert.equal(env.form.elements.helps,undefined);await env.form.fire('submit');assert.equal('helps' in sent,false);assert.equal('feels_safe' in sent,false);assert.equal(env.ctx.state.sourceDrafts.get('people').values.helps,'Texte non enregistré');
 env.ctx.state.peopleFieldsReady=true;env.ctx.drawSourceEditor('people');assert.equal(env.doc.getElementById('people-form').elements.helps.value,'Texte non enregistré');
});

test('A5-P2 conflict compares new fields and rebases only after keeping the draft explicitly',async()=>{
 const calls=[],env=environment({api:async(path,method,payload)=>{if(!method)return {person:{id:'p1',revision:9,display_name:'Camille',role:'friend',helps:'Version ailleurs',feels_safe:'yes'}};calls.push(payload);throw Object.assign(new Error('Conflit'),{code:'revision_conflict'})}});env.host('people-editor');env.ctx.state.peopleFieldsReady=true;env.ctx.state.sourceDrafts.set('people',{id:'p1',revision:8,values:{display_name:'Camille',role:'friend',helps:'Mon brouillon',feels_safe:'no'}});env.ctx.drawSourceEditor('people');const form=env.doc.getElementById('people-form');
 await form.fire('submit');await form.querySelector('[data-reload-source]').fire('click');const comparison=form.querySelector('.source-comparison');assert.match(comparison.textContent,/Version ailleurs/);assert.match(comparison.textContent,/sécurité.*oui/);assert.equal(form.elements.helps.value,'Mon brouillon');assert.equal(env.ctx.state.sourceDrafts.get('people').revision,8);
 await comparison.querySelector('button').fire('click');assert.equal(env.ctx.state.sourceDrafts.get('people').revision,9);await form.fire('submit');assert.equal(calls[1].helps,'Mon brouillon');assert.equal(calls[1].feels_safe,'no');assert.equal(calls[1].revision,9);
});

test('A5-P2 readable context uses labels, preserves newlines, and never interprets unset fields',()=>{
 const env=environment(),host=env.host('context');env.ctx.appendSourceContextText(host,'people',{bond_quality:'ambivalent',bond_note:'Nuance\nsuite',feels_safe:'no',helps:'<b>Mes mots</b>'});assert.match(host.textContent,/Qualité du lienambivalent/);assert.match(host.textContent,/Nuance\nsuite/);assert.match(host.textContent,/<b>Mes mots<\/b>/);assert.doesNotMatch(host.textContent,/Présence aujourd’hui/);env.ctx.appendSourceContextText(host,'events',{exposure:'learned',support_level:'none'});assert.doesNotMatch(host.textContent,/je l’ai appris|aucun/);
});



test('A5-P2 observation proposals require an explicit decision and retain edited quote after conflict',async()=>{
 const calls=[],env=environment({api:async(path,method,payload)=>{calls.push({path,payload});throw Object.assign(new Error('Cette proposition a changé.'),{status:409})}});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(script.match(/var PEOPLE_AXES = \{[\s\S]*?\};/)[0]+functionCode('drawChatProposal'),env.ctx);
 const host=env.host('observation-proposal');env.ctx.drawChatProposal(host,{id:'obs1',kind:'person_observation',target_id:'p1',revision:2,payload:{axis:'communication',quote:'Camille prend le temps de me répondre.'},evidence_quote:'Camille prend le temps de me répondre.'},()=>true);const form=host.children[0];assert.equal(calls.length,0);assert.equal(form.elements.axis.tagName,'SELECT');assert.equal(form.elements.quote.maxLength,280);assert.equal(form.elements.quote.minLength,20);assert.match(form.textContent,/votre point de vue/);assert.doesNotMatch(form.textContent,/person_observation/);
 form.elements.quote.value='Camille me répond avec attention.';await form.elements.quote.fire('input');form.elements.axis.value='appuis';await form.elements.axis.fire('change');await form.fire('submit');assert.equal(calls[0].path,'/api/chat/proposals/obs1/accept');assert.equal(calls[0].payload.revision,2);assert.equal(calls[0].payload.payload.quote,'Camille me répond avec attention.');assert.equal(calls[0].payload.payload.axis,'appuis');assert.match(form.textContent,/Vos modifications sont conservées/);assert.equal(form.elements.quote.value,'Camille me répond avec attention.');
 await form.querySelectorAll('button').find(button=>button.textContent==='Refuser').fire('click');assert.equal(calls[1].path,'/api/chat/proposals/obs1/reject');assert.equal(calls[1].payload.revision,2);assert.equal(calls[1].payload.payload,undefined);
});

test('A5-P2 Settings translates optional context and separates pending proposals from recorded observations',async()=>{
 const {SETTINGS_INTERFACE_SCRIPT}=await import('../src/settings-interface.mjs'),calls=[],env=environment({api:async path=>{calls.push(path);return path==='/api/people'?{people:[{id:'p1',display_name:'Camille'}]}:{available:true,proposals:[{status:'pending',payload:{axis:'appuis',quote:'Une présence à vérifier.'}}],observations:[{origin:'accepted',axis:'communication',quote:'Une parole déjà validée.'},{origin:'auto',quote:'Un ancien repère.'}]}}});
 vm.runInContext(SETTINGS_INTERFACE_SCRIPT.slice(0,SETTINGS_INTERFACE_SCRIPT.indexOf('async function renderSettings(')),env.ctx);const info=env.ctx.settingsValue({bond_quality:'ambivalent',feels_safe:'not_always',exposure:'witnessed',what_brings_me:'Comprendre',life_rhythm:'Le matin'});assert.match(info.textContent,/Qualité du lienambivalent/);assert.match(info.textContent,/pas toujours/);assert.match(info.textContent,/j’en ai été témoin/);assert.doesNotMatch(info.textContent,/what_brings_me|life_rhythm|not_always/);
 const box=env.ctx.settingsObservationRecords(()=>true);assert.equal(calls.length,0);box.open=true;await box.fire('toggle');const person=box.querySelectorAll('details').find(node=>node.querySelector('summary').textContent==='Camille');person.open=true;await person.fire('toggle');assert.deepEqual(calls,['/api/people','/api/people/p1/observations']);assert.match(person.textContent,/Propositions à vérifier/);assert.match(person.textContent,/Repères enregistrés/);assert.match(person.textContent,/Validé par vous/);assert.match(person.textContent,/Ancien repère automatique/);assert.match(person.textContent,/À vérifier/);
});


test('Settings hides research bibliographies in rendered values and JSON without changing private evidence or stored data',()=>{
 const env=environment();vm.runInContext(SETTINGS_INTERFACE_SCRIPT.slice(0,SETTINGS_INTERFACE_SCRIPT.indexOf('async function renderSettings(')),env.ctx);
 const original={label:'Mon enfance',evidence_source:{id:'afifi2017',citation:'Citation scientifique',url:'https://pubmed.ncbi.nlm.nih.gov/28126359/'},evidence_sources:[{id:'afifi2017',citation:'Citation scientifique'}],evidence:[{source_ids:['afifi2017'],summary:'Une association prudente.'}],sources:[{id:'afifi2017',citation:'Citation scientifique',url:'https://pubmed.ncbi.nlm.nih.gov/28126359/'},{id:'person:p1',content:'Mes propres mots',source_quote:'Un souvenir précis.'}],source_ids:['afifi2017','person:p1'],reference_ids:['S01'],content:JSON.stringify({evidence_sources:[{id:'afifi2017',citation:'Citation scientifique'}],source_id:'person:p1',quote:'Ma citation personnelle'})};
 original.trace={messages:[{role:'user',content:'Contexte privé enregistré (données à examiner avec prudence) : '+JSON.stringify({evidence_sources:[{id:'afifi2017',citation:'Citation scientifique',url:'https://pubmed.ncbi.nlm.nih.gov/28126359/'}],quote:'Mes mots dans la trace'})}]};
 const before=JSON.stringify(original),view=env.ctx.settingsValue(original),jsonView=env.ctx.settingsJSON('Données',original);
 for(const node of [view,jsonView]){assert.doesNotMatch(node.textContent,/afifi2017|pubmed|Citation scientifique|S01/);assert.match(node.textContent,/person:p1/);assert.match(node.textContent,/Mes propres mots/);assert.match(node.textContent,/Ma citation personnelle/);assert.match(node.textContent,/Une association prudente/);assert.match(node.textContent,/Contexte privé enregistré/);assert.match(node.textContent,/Mes mots dans la trace/)}
 assert.equal(JSON.stringify(original),before);
});

test('date controls keep precision drafts without manufacturing a month or a day',async()=>{
 const env=editFixture('events',{title:'Souvenir',description:'Récit',kind:'positive',date_precision:'year',date_start:'1998',people_ids:[]});
 const {date_precision:precision,date_start:start}=env.form.elements;assert.equal(start.type,'text');assert.equal(start.value,'1998');
 precision.value='day';await precision.fire('change');assert.equal(start.type,'date');assert.equal(start.value,'');start.value='1998-06-17';
 precision.value='month';await precision.fire('change');assert.equal(start.type,'month');assert.equal(start.value,'');start.value='1998-06';
 precision.value='year';await precision.fire('change');assert.equal(start.value,'1998');precision.value='day';await precision.fire('change');assert.equal(start.value,'1998-06-17');
 await env.form.fire('input');env.ctx.drawSourceEditor('events');assert.equal(env.doc.getElementById('events-form').elements.date_start.value,'1998-06-17');
});

test('meeting dates use the same partial controls and preserve legacy period text',async()=>{
 let sent;const env=editFixture('people',{display_name:'Camille',role:'friend',meeting_date_precision:'month',meeting_date_start:'2010-05',period_start:'Pendant mes études',period_end:''},{api:async(p,m,data)=>{sent=data;throw new Error('offline')}});
 assert.equal(env.form.elements.meeting_date_start.type,'month');assert.equal(env.form.elements.meeting_date_start.value,'2010-05');await env.form.fire('submit');assert.equal(sent.meeting_date_start,'2010-05');assert.equal(sent.period_start,'Pendant mes études');
});

test('an already-started conversation is blocked when the strict readiness checklist becomes incomplete',()=>{
 const readiness={complete:false,identity:{complete:true,missing_fields:[]},people:{complete:false,count:2,required:3},events:{complete:true,count:3,required:3},childhood:{complete:true,answered:60,required:60}};
 const env=chatFixture();env.ctx.state.chatData=journeyState({started:true,can_start:false,readiness,review:{}},{consent:true,access:{allowed:true}});env.ctx.drawChat(env.view,()=>true);
 assert.equal(env.send.disabled,true);assert.equal(env.form.hidden,true);assert.equal(env.doc.getElementById('journey-panel').hidden,false);assert.match(env.doc.getElementById('journey-panel').textContent,/2 proches sur 3/);
 assert.equal(env.doc.getElementById('journey-panel').querySelectorAll('.journey-source')[2].querySelector('a').href,'/mes-proches/');
 readiness.complete=true;readiness.people.complete=true;env.ctx.drawChat(env.view,()=>true);assert.equal(env.form.hidden,false);assert.equal(env.send.disabled,false);
});

function noteFixture(note,api){
 let saved=0;const env=environment({api});for(const name of ['noteTarget','noteTimestamp','applyNoteRevision','sourceNoteEditor','sourceNoteCard'])vm.runInContext(functionCode(name),env.ctx);
 const target='/api/life-events/event-1/notes',editor=env.ctx.sourceNoteEditor(target,note,async()=>saved++);env.host('notes').append(editor);return {...env,target,editor,form:editor.querySelector('form'),input:editor.querySelector('textarea'),get saved(){return saved}};
}

test('a failed note creation retries with one creation id and preserves the complete draft',async()=>{
 const calls=[],env=noteFixture(null,async(path,method,payload)=>{calls.push({path,method,payload});throw new Error('Offline')});env.input.value='Mes mots à conserver';await env.input.fire('input');await env.form.fire('submit');await env.form.fire('submit');
 assert.equal(env.input.maxLength,2000);assert.equal(calls[0].method,'POST');assert.equal(calls[0].payload.creation_id,calls[1].payload.creation_id);assert.equal(env.input.value,'Mes mots à conserver');assert.equal(env.ctx.state.noteDrafts.get(env.target+':new').text,env.input.value);
});

test('typing during note creation keeps a draft and turns the next save into a revision-checked edit',async()=>{
 let resolve;const calls=[],env=noteFixture(null,async(path,method,payload)=>{calls.push({path,method,payload});if(calls.length===1)return new Promise(r=>resolve=r);return {note:{id:'note-1',revision:2,text:payload.text},parent_revision:6}});
 env.ctx.state.events=[{id:'event-1',revision:4}];env.input.value='Premier texte';await env.input.fire('input');const pending=env.form.fire('submit');env.input.value='Premier texte et suite';await env.input.fire('input');resolve({note:{id:'note-1',revision:1,text:'Premier texte'},parent_revision:5});await pending;
 assert.equal(env.saved,0);assert.equal(env.ctx.state.events[0].revision,5);assert.equal(env.input.value,'Premier texte et suite');await env.form.fire('submit');assert.equal(calls[1].method,'PATCH');assert.equal(calls[1].path,env.target+'/note-1');assert.equal(calls[1].payload.revision,1);assert.equal(calls[1].payload.text,'Premier texte et suite');assert.equal(env.saved,1);
});

test('note conflict shows the remote text and adopts its revision only after explicit choice',async()=>{
 let conflict=true;const writes=[],env=noteFixture({id:'n',revision:2,text:'Ancien'},async(path,method,payload)=>{if(!method)return {notes:[{id:'n',revision:3,text:'Version distante'}],next_cursor:null};writes.push(payload);if(conflict)throw Object.assign(new Error('Changed'),{status:409});return {note:{id:'n',revision:4,text:payload.text}}});
 env.input.value='Mon brouillon';await env.input.fire('input');await env.form.fire('submit');await env.editor.querySelectorAll('button').find(b=>b.textContent==='Comparer la note enregistrée').fire('click');assert.match(env.editor.textContent,/Version distante/);assert.equal(env.ctx.state.noteDrafts.get(env.target+':n').revision,2);assert.equal(env.input.value,'Mon brouillon');
 await env.editor.querySelectorAll('button').find(b=>b.textContent==='Conserver ma saisie pour un nouvel enregistrement').fire('click');conflict=false;await env.form.fire('submit');assert.equal(writes[1].revision,3);assert.equal(writes[1].text,'Mon brouillon');
});

test('the Ambre bubble is gone: no launcher, iframe bridge or embedded mode remains in the shell',()=>{
 assert.doesNotMatch(HTML,/ambre-launcher|ambre-panel|ambre-embedded|ambre-widget|dte-ambre|postMessage|embedded|BroadcastChannel|dte:session/);
 assert.doesNotMatch(HTML,/<iframe/);assert.match(HTML,/<li><a href="\/mon-accompagnement\/">Ambre<\/a><\/li>/);
 for(const file of ['../../static/js/ambre-widget.js','../../static/css/ambre-widget.css'])assert.throws(()=>readFileSync(new URL(file,import.meta.url)),{code:'ENOENT'});
 assert.doesNotMatch(readFileSync(new URL('../../layouts/_default/baseof.html',import.meta.url),'utf8'),/ambre-widget/);
 assert.doesNotMatch(readFileSync(new URL('../../scripts/build_navigation.mjs',import.meta.url),'utf8'),/AMBRE_WIDGET/);
});
test('a close person has an address of their own, prepared and opened like the other pages',async()=>{
 const seen=[],env=navigationFixture((tab,search,personId)=>{seen.push([tab,personId]);return Promise.resolve({ready:true})});
 assert.equal(env.ctx.personIdForPath('/mes-proches/p-1_A/'),'p-1_A');assert.equal(env.ctx.personIdForPath('/mes-proches/p-1_A'),'p-1_A');
 for(const path of ['/mes-proches/','/mes-proches','/mes-proches/a/b/','/mes-proches/a%2Fb/','/ma-ligne-de-vie/p1/'])assert.equal(env.ctx.personIdForPath(path),null,path);
 assert.equal(env.ctx.pageForPath('/mes-proches/p1/'),'person');assert.equal(env.ctx.pagePath('person','p1'),'/mes-proches/p1/');assert.equal(env.ctx.pagePath('people'),'/mes-proches/');
 await env.ctx.go('person','','p1');assert.equal(env.location.pathname,'/mes-proches/p1/');assert.deepEqual(seen,[['person','p1']]);assert.deepEqual(env.rendered,['person']);
 await env.ctx.go('person','');assert.equal(env.location.pathname,'/mes-proches/','a person page without identifier falls back to the list');
 const paths=[],prepare=environment({api:async path=>{paths.push(path);return {}}});vm.runInContext(functionCode('preparePageData'),prepare.ctx);
 await prepare.ctx.preparePageData('person','','p1');assert.deepEqual(paths,['/api/people/p1','/api/people/p1/timeline']);
});
function personFixture(api,extra={}){
 const env=environment({api,location:{pathname:'/mes-proches/p1/'},history:{replaceState(){}},queueMicrotask(fn){fn()},...extra});
 for(const name of ['personRoleLabel','personIdForPath','pagePath','noteTarget','noteTimestamp','noteDay','sourceNoteEditor','sourceNoteCard','eventNotes','personTimeline','renderPerson','drawPersonPage','personSaved','drawSourceItem'])vm.runInContext(functionCode(name),env.ctx);
 Object.assign(env.ctx.state,{tab:'person',personId:'p1',peopleFieldsReady:false});
 const head=env.doc.createElement('div'),title=env.doc.createElement('h1'),description=env.doc.createElement('p');title.id='private-page-title';title.textContent='Mes proches';description.id='page-description';description.hidden=true;head.append(title,description);
 const view=env.host('view');let html='';Object.defineProperty(view,'innerHTML',{set(value){html=value;const status=env.doc.createElement('p'),box=env.doc.createElement('div');status.id='source-page-status';box.id='person-content';view.replaceChildren(status,box)},get(){return html}});
 return {...env,head,title,description,view,get box(){return env.doc.getElementById('person-content')}};
}
const personRow={id:'p1',display_name:'Léa',role:'friend',relationship_text:'On se parle chaque semaine.',revision:3};
const personTimelineData={person:personRow,available:true,next_cursor:null,items:[{kind:'relationship_start',date:{precision:'year',start:'2012',end:'',age_text:''}},{kind:'life_event',event:{id:'e1',title:'Un voyage ensemble',description:'Trois jours en Bretagne.',date_precision:'year',date_start:'2019',notes:[],notes_available:true}},{kind:'user_note',note:{id:'n1',text:'Une note à moi',revision:1,created_at:'2026-11-01T10:00:00Z',updated_at:'2026-11-01T10:00:00Z'}},{kind:'ambre_observation',note:{id:'o1',text:'Léa me rappelle toujours.',created_at:'2026-11-02T10:00:00Z'}}]};
test('the person page shows the name, role, information, edit action and the complete timeline at once',async()=>{
 const calls=[],env=personFixture(async path=>{calls.push(path);return path.endsWith('/timeline')?personTimelineData:{person:personRow,fields_ready:true}});
 await env.ctx.renderPerson();
 assert.deepEqual(calls,['/api/people/p1','/api/people/p1/timeline'],'no further request is needed for the first display');
 assert.equal(env.title.textContent,'Léa');assert.equal(env.description.textContent,'Ami ou amie');assert.equal(env.description.hidden,false);
 const back=env.head.children[0];assert.equal(back.tagName,'A');assert.equal(back.href,'/mes-proches/');assert.equal(back.textContent,'Mes proches');
 const text=env.box.textContent;for(const expected of ['Informations','On se parle chaque semaine.','Votre histoire avec Léa','Rencontre','2012','Un voyage ensemble','Trois jours en Bretagne.','Une note à moi','Léa me rappelle toujours.','Repère validé','Ajouter une note à ce moment','Votre note'])assert.ok(text.includes(expected),expected);
 assert.doesNotMatch(text,/Repères sur cette relation|Chargement/);assert.equal(env.box.querySelector('details.person-timeline'),null,'the timeline is no longer a disclosure');
 const buttons=env.box.querySelector('.person-info').querySelectorAll('button').map(button=>button.textContent);assert.deepEqual(buttons.slice(0,2),['Modifier','Supprimer']);
 assert.equal(env.ctx.state.peopleFieldsReady,true);
 await env.box.querySelector('.person-info').querySelector('button').fire('click');
 const form=env.doc.getElementById('people-form');assert.ok(form,'Modifier opens the editor on the person page');assert.equal(form.elements.display_name.value,'Léa');assert.equal(env.ctx.state.sourceDrafts.get('people').revision,3);
});
test('an unknown person, or one of another account, gets a clear message and a way back to the list',async()=>{
 for(const [error,pattern,retry] of [[Object.assign(new Error('Élément introuvable.'),{status:404,code:'not_found'}),/Ce proche est introuvable/,false],[Object.assign(new Error('Service momentanément indisponible.'),{status:503}),/Service momentanément indisponible/,true]]){
  const env=personFixture(async()=>{throw error});await env.ctx.renderPerson();
  assert.match(env.box.textContent,pattern);assert.equal(env.title.textContent,'Mes proches','no name is invented');
  const links=env.box.querySelectorAll('a');assert.equal(links.length,1);assert.equal(links[0].href,'/mes-proches/');assert.equal(links[0].textContent,'Revenir à Mes proches');
  assert.equal(env.box.querySelectorAll('button').some(button=>button.textContent==='Réessayer'),retry);assert.equal(env.box.querySelector('form'),null);assert.equal(env.ctx.state.person,null);
 }
});
test('a directory card is short: name, role, one line and the action that opens the person page',()=>{
 const env=personFixture(async()=>({}));
 const card=env.ctx.drawSourceItem('people',{...personRow,what_helps:'Sa patience',period_start:'2012'},env.host('people-content'));
 assert.equal(card.id,'person-p1');assert.equal(card.querySelector('h3').textContent,'Léa');assert.equal(card.querySelector('.source-tag').textContent,'Ami ou amie');assert.equal(card.querySelector('.person-line').textContent,'On se parle chaque semaine.');
 const links=card.querySelectorAll('a');assert.equal(links.length,1);assert.equal(links[0].href,'/mes-proches/p1/');assert.equal(links[0].textContent,'Léa');assert.equal(links[0].className,'person-open','the whole card is the link');
 assert.equal(card.querySelectorAll('button').length,0);assert.equal(card.querySelectorAll('details').length,0);assert.doesNotMatch(card.textContent,/Sa patience|Repères|Votre histoire avec/);
 assert.doesNotMatch(HTML,/Repères sur cette relation|personObservationDetails|person-observations/);assert.match(HTML,/var PEOPLE_AXES = \{/,'the conversation proposal form keeps its axis labels');
});
test('a note saved on the person page keeps the page and its open edit draft at the new revision',()=>{
 const env=personFixture(async()=>({}));vm.runInContext(functionCode('applyNoteRevision'),env.ctx);
 env.ctx.state.person={...personRow};env.ctx.state.sourceDrafts.set('people',{id:'p1',revision:3,values:{display_name:'Léa modifiée'}});
 env.ctx.applyNoteRevision('/api/people/p1/notes',{parent_revision:4});assert.equal(env.ctx.state.person.revision,4);assert.equal(env.ctx.state.sourceDrafts.get('people').revision,4);assert.equal(env.ctx.state.sourceDrafts.get('people').values.display_name,'Léa modifiée');
 env.ctx.applyNoteRevision('/api/people/other/notes',{parent_revision:9});assert.equal(env.ctx.state.person.revision,4);
});
test('saving from the person page refreshes that page; an unrelated draft never opens there',async()=>{
 const saved={...personRow,display_name:'Léa B.',revision:4},calls=[];
 const env=personFixture(async(path,method)=>{calls.push([path,method||'GET']);return method==='PATCH'?{person:saved}:personTimelineData});
 env.view.innerHTML='';env.ctx.state.person={...personRow};env.ctx.state.personTimeline=personTimelineData;env.ctx.state.sourceDrafts.set('people',{id:null,creation_id:'c1',values:{role:'friend',display_name:'Brouillon'}});
 env.ctx.drawPersonPage();assert.equal(env.doc.getElementById('people-form'),null,'the pending new-person draft belongs to the list');assert.equal(env.ctx.state.sourceDrafts.get('people').values.display_name,'Brouillon');
 await env.box.querySelector('.person-info').querySelector('button').fire('click');const form=env.doc.getElementById('people-form');field(form,'display_name','Léa B.');await form.fire('submit');
 assert.deepEqual(calls.find(call=>call[1]==='PATCH'),['/api/people/p1','PATCH']);assert.equal(env.title.textContent,'Léa B.');assert.equal(env.ctx.state.person.revision,4);assert.equal(env.ctx.state.sourceDrafts.has('people'),false);assert.equal(env.box.querySelectorAll('form').filter(node=>node.id==='people-form').length,0,'the editor closes after a saved change');assert.ok(calls.some(call=>call[0]==='/api/people/p1/timeline'),'the timeline is reloaded because the meeting date may have changed');
});
test('the person page offers « Ajouter un moment » beside « Ajouter une note », toward Ma ligne de vie with this person chosen',async()=>{
 const env=personFixture(async path=>path.endsWith('/timeline')?personTimelineData:{person:personRow,fields_ready:true});
 await env.ctx.renderPerson();
 const adds=env.box.querySelector('.person-timeline-adds');assert.ok(adds,'one row closes the line');
 assert.equal(adds.querySelector('.person-note-add').querySelector('summary').textContent,'Ajouter une note');
 const moment=adds.querySelectorAll('a').find(link=>/person-moment-add/.test(link.className));assert.equal(moment.textContent,'Ajouter un moment');assert.equal(moment.href,'/ma-ligne-de-vie/?ajouter-avec=p1');assert.match(moment.className,/\bbutton secondary\b/);
});
function momentFixture(search,api){
 const calls=[],gone=[];const env=environment({api:api||(async(path,method,data)=>{calls.push([path,method||'GET',data]);return {event:{id:'e9',title:data&&data.title,people_ids:data&&data.people_ids,revision:1},source_version:12}}),query:new URLSearchParams(search),refreshPortraitData(){},go:(...args)=>{gone.push(args)}});
 vm.runInContext(functionCode('pagePath'),env.ctx);
 env.ctx.state.people=[personRow,{id:'p2',display_name:'Sam',role:'friend'}];env.ctx.state.tab='events';env.host('events-editor');
 return {...env,calls,gone};
}
test('« Ajouter un moment » from a person opens the new moment with that person chosen, saves it and returns to the person page',async()=>{
 const env=momentFixture('ajouter-avec=p1');
 assert.equal(env.ctx.prepareMomentForPerson().id,'p1');env.ctx.drawSourceEditor('events',true);
 const form=env.doc.getElementById('events-form');assert.ok(form,'the form is open');
 const boxes=form.querySelectorAll('input').filter(input=>input.name==='people_ids');assert.deepEqual(boxes.filter(box=>box.checked).map(box=>box.value),['p1'],'only this person is chosen');
 field(form,'title','Un dîner');field(form,'description','Chez elle, tous les deux.');await form.fire('submit');
 const post=env.calls.find(call=>call[1]==='POST');assert.equal(post[0],'/api/life-events','the existing moments API');assert.deepEqual([...post[2].people_ids],['p1']);assert.equal(post[2].title,'Un dîner');assert.ok(post[2].creation_id);
 assert.equal(JSON.stringify(env.gone),JSON.stringify([['person','','p1']]),'back to the person page, whose timeline is fetched again');assert.equal(env.ctx.state.sourceDrafts.has('events'),false);
});
test('« Ajouter un moment »: Annuler returns to the person and keeps the typed text; an edited moment and an unknown person are left alone',async()=>{
 const env=momentFixture('ajouter-avec=p1');env.ctx.prepareMomentForPerson();env.ctx.drawSourceEditor('events');
 const form=env.doc.getElementById('events-form');field(form,'title','Pas fini');
 await form.querySelectorAll('button').find(button=>button.textContent==='Annuler').fire('click');
 assert.equal(JSON.stringify(env.gone),JSON.stringify([['person','','p1']]));const kept=env.ctx.state.sourceDrafts.get('events');assert.equal(kept.values.title,'Pas fini');assert.deepEqual([...kept.values.people_ids],['p1']);assert.equal(kept.returnTo,undefined);
 // Coming back from another person adds that person to the same unsaved moment.
 env.ctx.query=new URLSearchParams('ajouter-avec=p2');vm.runInContext('query=globalThis.query',env.ctx);env.ctx.prepareMomentForPerson();assert.deepEqual([...env.ctx.state.sourceDrafts.get('events').values.people_ids],['p1','p2']);
 const edit=momentFixture('ajouter-avec=p1');edit.ctx.state.sourceDrafts.set('events',{id:'e1',revision:2,values:{title:'En cours',people_ids:[]}});
 assert.equal(edit.ctx.prepareMomentForPerson(),null);assert.equal(JSON.stringify(edit.ctx.state.sourceDrafts.get('events').values),JSON.stringify({title:'En cours',people_ids:[]}));
 const unknown=momentFixture('ajouter-avec=nobody');assert.equal(unknown.ctx.prepareMomentForPerson(),null);assert.equal(unknown.ctx.state.sourceDrafts.has('events'),false);
 // Without the parameter, saving a moment stays on Ma ligne de vie.
 const plain=momentFixture('');plain.ctx.state.sourceDrafts.set('events',{id:null,creation_id:'c',values:{kind:'positive',date_precision:'unknown',people_ids:[]},returnTo:'p1'});plain.ctx.drawSourceEditor('events');
 const plainForm=plain.doc.getElementById('events-form');field(plainForm,'title','Seul');field(plainForm,'description','Texte');await plainForm.fire('submit');assert.equal(plain.gone.length,0);
});
test('account renders API monthly credits and the Paris renewal date without another request',()=>{
 const env=environment({api(){throw new Error('No extra quota fetch')}}),host=env.host('quota');vm.runInContext(functionCode('appendAccountQuota'),env.ctx);
 env.ctx.appendAccountQuota(host,{access:{mode:'paid'},subscription:{active:true,status:'active',paid_until:Date.parse('2026-12-17T12:00:00Z')/1000},quotas:{messages:{limit:100,used:7,reserved:1,remaining:92,renews_at:'2026-11-30T23:00:00.000Z'},reviews:{limit:1,used:1,remaining:0}}});
 assert.match(host.textContent,/100 messages et 1 bilan par mois, bilan initial inclus/);assert.match(host.textContent,/Il vous reste 92 messages et 0 bilan ce mois-ci, jusqu’au 1ᵉʳ décembre\./);assert.match(host.textContent,/facturation en cours se termine le 17 décembre 2026/);assert.doesNotMatch(host.textContent,/100 messages.*disponibles/);
});
