import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import {HTML as SHELL,APP_SCRIPT,APP_STYLE} from '../src/ui.mjs';
// Lot P: the private shell plus its two content-addressed files, as the browser assembles them.
const HTML=SHELL.replace('</head>',()=>'<style>'+APP_STYLE+'</style></head>')+'<script>'+APP_SCRIPT+'</script>';
import { REVIEW_ARCHIVE_SCRIPT } from '../src/review-archive-interface.mjs';

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
  for (const name of ['renderGuard','el','actionButton','finePointer','focusEntry','formValues','busy','sourceField','monthNames','periodLength','appendEventDateFields','appendSourceContextFields','appendSourceContextText','newClientId','captureSourceDraft','sourceDate','drawSourceEditor','chatConsent','chatFailureMessage','applyChatAvailability','scheduleChatCooldown','chatIsCurrent','chatCapabilityMessage','chatHasPending','updateChatControls','disposeChatInterface','resetChatSession','reconcileChatTurn','createChatShell','drawChatGate','scrollChatTo','revealChatResponse','chatSentences','chatIsQuestion','chatReviewParts','drawChatStructured','chatReviewLinks','chatMessageEntry','chatJourney','chatReviewRunning','chatJourneyBlocked','chatPointNote','chatPointSuggestion','drawChatNotes','chatInitialPending','journeyPanel','closeJourneyPanel','showJourneyPanel','chatGuest','ambreFailureReason','ambreReview','ambreFailure','ambreReadiness','ambreClock','ambreNeeds','ambreNeedItem','ambreLaunchReason','ambreRequestReview','ambreLaunchArea','ambreIntroStatus','ambreIntroMessage','ambreIntroSignature','drawAmbreIntro','ambreProgress','ambreEstimate','ambreProgressMessage','ambreFailureMessage','drawAmbreReviewStatus','ambreComposerClosed','ambreComposerLock','drawJourneyPanel','openPointDialog','pollJourney','dateLabel','retryChatMessage','displayDate','navigationPending','prepareMomentForPerson','momentReturn']) vm.runInContext(functionCode(name),ctx);
  vm.runInContext(script.match(/var MOMENT_FOR_PERSON=.*\n/)[0],ctx);
  vm.runInContext(['AMBRE_INTRO','AMBRE_DAILY_LAUNCHES','AMBRE_FAILURES'].map(name=>script.match(new RegExp('var '+name+'=.*\\n'))[0]).join(''),ctx);
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
  assert.doesNotMatch(script,/\/api\/(?:likes|examples|recommendations|inspirations)|renderLikes|drawLikedArticles|safeArticleLink|applyPendingLike|pendingArticle|likeVersion/);
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
  assert.equal(env.form.elements.date_start.type,'hidden','dates are chosen in lists, never typed');assert.equal(env.form.elements.age_text.type,'hidden');
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
  const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('personIdForPath')+'\n'+functionCode('topicIdForPath')+'\n'+functionCode('pageForPath'),env.ctx);
  assert.equal(env.ctx.pageForPath('/mes-proches'),'people');assert.equal(env.ctx.pageForPath('/mes-proches/'),'people');assert.equal(env.ctx.pageForPath('/ma-ligne-de-vie/'),'events');assert.equal(env.ctx.pageForPath('/ambre/'),'chat');
  assert.doesNotMatch(HTML,/À propos de moi/);
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

test('there is no menu: the header holds the back button and the eye, and the dashboard lists every page in order',()=>{
  const navigation=JSON.parse(readFileSync(new URL('../../data/navigation.json',import.meta.url),'utf8'));
  assert.doesNotMatch(HTML,/site-navigation"|nav-links|site-menu-toggle/);
  assert.match(HTML,/<a href="\/accueil\/" class="site-back">[\s\S]*?<span>Accueil<\/span>\s*<\/a>\s*<a href="\/accueil\/" class="site-logo"/);
  assert.deepEqual(JSON.parse(script.match(/var HOME_PAGES=(\[.*?\]);/)[1]),navigation);
  assert.deepEqual(navigation.map(link=>link.label),['Ambre','Ma ligne de vie','Mes proches','Mon enfance','Mes symptômes','Mes sujets','Ma cartographie','Mes bilans','Mes diagnostics','Mon compte']);
  for(const link of navigation)assert.match(script,new RegExp("'"+link.href.replace(/\//g,'\\/')+"':\\["),'the tile of '+link.href+' has its icon and sentence');
  assert.doesNotMatch(readFileSync(new URL('../../layouts/partials/header.html',import.meta.url),'utf8'),/<nav|navigation/);
});

function navigationFixture(preparePageData){
 const location={pathname:'/mon-portrait/',search:'',hash:''},rendered=[];
 const env=environment({location,preparedResponses:{},preparePageData,history:{pushState(_state,_unused,path){location.pathname=path}},window:{scrollTo(){}},capturePageDrafts(){},syncPageLocation(){env.ctx.state.tab=env.ctx.pageForPath(location.pathname)},renderApp(){rendered.push(env.ctx.state.tab)}});
 env.ctx.state.me={user:{id:'owner'}};env.ctx.state.tab='portrait';vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('personIdForPath')+'\n'+functionCode('topicIdForPath')+'\n'+functionCode('pagePath')+'\n'+functionCode('pageForPath')+'\n'+functionCode('go'),env.ctx);
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

test('all removed portrait subpage links lead to the single presentation',()=>{
 for(const hash of ['#ma-cartographie','#cartographie','#mon-portrait','#synthese','#correction-form','#self-profile','']){
  const location={hash,pathname:'/mon-portrait/',search:'?article=old'};const historyCalls=[];
  const env=environment({location,history:{replaceState(a,b,url){historyCalls.push(url);location.hash=url.includes('#')?url.slice(url.indexOf('#')):''}}});
  const view=env.host('view'),form=env.doc.createElement('form');view.append(form);
  vm.runInContext(functionCode('setupPortraitViews'),env.ctx);env.ctx.setupPortraitViews(view);
  const obsolete=['#ma-cartographie','#cartographie','#mon-portrait','#synthese','#correction-form'].includes(hash);
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

test('the application footer holds the site name (to Ambre), and no reading tab stays in the portrait menu',()=>{
 assert.equal(HTML.match(/<footer\b/g).length,1);assert.match(HTML,/<footer class="site-footer">\s*<div class="footer-inner footer-minimal">\s*<a href="\/accueil\/" class="site-logo">Décode ton esprit<\/a>\s*<\/div>\s*<\/footer>/);
 assert.doesNotMatch(script,/likes-search|likes-character|likes-keyword|likes-more|recommendations-body|À découvrir/);
});

const psychologyFixture={catalog_version:1,domains:[{id:'ties',label:'Mes relations'},{id:'self',label:'Mon regard sur moi'}],patterns:[{id:'belong',domain:'ties',label:'Besoin d’appartenance',description:'Chercher sa place.',signals:['Faire attention au groupe.'],questions:['Comment trouvez-vous votre place ?'],alternatives:['Un contexte nouveau.']},{id:'autonomy',domain:'self',label:'Besoin d’autonomie',description:'Choisir pour soi.',questions:['Quand choisissez-vous librement ?']}],hypotheses:[{pattern_id:'belong',status:'exploring',observation:'Vous évoquez votre place dans le groupe.',possible_function:'Préserver les liens.',evidence:['person:p1'],counter_evidence:['profile:self'],citations:[{source_id:'person:p1',quote:'Mes mots exacts'}],related_articles:[{title:'Une lecture',url:'/article/'},{title:'Lien refusé',url:'javascript:alert(1)'}],related_characters:[{name:'Nina',url:'/personnages/nina/'}],next_question:'Que souhaitez-vous préciser ?'}],source_version:4,analyzed_version:4,outdated:false};
function mapFixture(extra={}){const env=environment({refreshPortraitData(){},location:{origin:'https://example.test'},...extra});Object.assign(env.ctx.state,{tab:'portrait',mapReturnDrafts:new Map()});for(const name of ['normalized','personalizePortraitText','portraitEvidence','mapStateLabel','selectMapPattern','renderMapExplorer','drawMapPattern','drawMapReturn','drawPsychologyMap'])vm.runInContext(functionCode(name),env.ctx);const host=env.host('ma-cartographie');env.ctx.drawPsychologyMap(host,psychologyFixture,()=>true);return {...env,host};}

test('V6 free preview account displays access and quotas without checkout or subscription offers',()=>{
 const env=environment();const host=env.host('view'),accountRow=env.host('account-row');host.querySelector=()=>accountRow;for(const id of ['account-appointment','account-email','account-price','subscription-copy','account-progress','billing-action','export-button','delete-button'])env.host(id);env.ctx.state.me={user:{email:'test@example.test'},access:{mode:'free_preview',allowed:true},subscription:{active:false,trial_eligible:true},configuration:{billing:true,ai:true}};env.doc.getElementById('account-email').parentElement=accountRow;vm.runInContext(functionCode('appendAccountQuota'),env.ctx);env.ctx.setupPortraitViews=()=>{};env.ctx.renderSelfProfile=()=>{};env.ctx.logout=()=>{};vm.runInContext(functionCode('renderAccount'),env.ctx);env.ctx.renderAccount();assert.equal(env.doc.getElementById('account-price').hidden,true);assert.match(env.doc.getElementById('subscription-copy').textContent,/Accès gratuit/);assert.doesNotMatch(env.doc.getElementById('billing-action').textContent,/abonner|essai|factur|29/);assert.doesNotMatch(env.doc.getElementById('account-progress').textContent,/29|7 jours d’essai/);
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

// Plan 19/11 lot 3: the page is always the conversation; Ambre's first message (code-written) replaces
// the former « Avant de commencer » card, the welcome text and the consent gate.
const fullReadiness=()=>({version:2,complete:true,identity:{complete:true,missing_fields:[]},people:{complete:true,count:4,required:3},events:{complete:true,count:3,required:3},childhood:{complete:true,answered:60,required:60},symptoms:{complete:true,answered:516,required:516},missing:[]});
const partialReadiness=()=>({version:2,complete:false,identity:{complete:false,missing_fields:['gender']},people:{complete:false,count:1,required:3},events:{complete:true,count:3,required:3},childhood:{complete:false,answered:60,required:60,consent:false},symptoms:{complete:false,answered:212,required:516},missing:['identity','people','childhood','symptoms']});
const ambreState=(readiness,journey={},extra={})=>journeyState({started:false,can_start:readiness.complete,readiness,quotas:{reviews:{remaining:1,limit:1,renews_at:'2026-12-01'}},...journey,review:{running_id:null,last_at:null,next_point_at:null,last_failure:null,...journey.review}},{readiness,welcome:{id:'ambre-welcome-v9',role:'assistant',content:'Ancien message d’accueil.'},...extra});
const runningReview=(review={})=>({running_id:'r1',last_at:null,next_point_at:null,stage:'snapshot',blocks_done:5,blocks_total:12,started_at:Math.floor(Date.now()/1000)-120,estimate:{p50_seconds:540,p90_seconds:780,basis:'initial'},last_failure:null,...review});

test('Ambre’s first message introduces her, lists the five pages with counters and launches the review without an agreement box',async()=>{
 const calls=[];let next;const env=chatFixture({api:async(path,method,data)=>{calls.push([path,method||'GET',data]);return next;}});
 env.ctx.state.chatData=ambreState(partialReadiness());env.ctx.drawChat(env.view,()=>true);
 const thread=env.view.querySelector('.chat-thread'),intro=env.doc.getElementById('ambre-intro');
 assert.equal(thread.hidden,false);assert.equal(env.form.hidden,false);assert.ok(thread.children[0]===intro,'the first message opens the thread');assert.equal(env.doc.getElementById('journey-panel').hidden,true);
 assert.equal(intro.querySelector('h2').textContent,'Bonjour, je suis Ambre.');assert.deepEqual(intro.querySelector('.ambre-intro-lead').children.map(p=>p.textContent),['Je suis une assistante psychologue IA, et je suis là pour vous aider à mieux vous comprendre, à votre rythme.','Pour commencer, je lis tout ce que vous avez renseigné et je vous prépare un premier bilan : ce que je comprends de votre histoire, avec des hypothèses sur ce qui me paraît probable.','Ensuite, nous en parlons ensemble : vous me dites ce qui vous parle, ce qui vous ressemble moins, et ce que vous aimeriez approfondir.']);assert.match(intro.textContent,/Ce dont j’ai besoin/);
 assert.doesNotMatch(env.view.textContent,/Ancien message d’accueil|Avant de commencer/,'the former welcome and card are gone');assert.equal(env.view.querySelectorAll('.chat-assistant').length,1);
 const items=intro.querySelectorAll('.ambre-need');assert.deepEqual(items.map(i=>i.querySelector('a').href),['/mon-compte/','/mes-proches/','/ma-ligne-de-vie/','/cartographie/','/mes-symptomes/']);
 assert.deepEqual(items.map(i=>i.querySelector('.ambre-need-count').textContent),['3/4','1/3','3/3','60/60','212/516']);
 assert.deepEqual(items.map(i=>i.querySelector('.ambre-need-state').textContent),['À compléter','À compléter','Complet','Enregistrez la page','À compléter']);
 assert.equal(items[2].querySelector('.ambre-need-mark').textContent,'✓');assert.equal(items[0].querySelector('.ambre-need-mark').textContent,'');assert.equal(items[4].querySelector('a').getAttribute('aria-label'),'Mes symptômes : à compléter, 212 sur 516');
 let start=env.doc.getElementById('ambre-start'),reason=env.doc.getElementById('ambre-start-reason');assert.equal(start.textContent,'Lancer mon bilan');assert.equal(start.disabled,true);
 assert.equal(reason.textContent,'Complétez d’abord Mon compte, Mes proches, Mes symptômes. Enregistrez Mon enfance.');assert.equal(start.getAttribute('aria-describedby'),reason.id);
 assert.ok(env.view.querySelector('.chat-gate')===null,'no agreement gate in the conversation');assert.ok(env.doc.getElementById('ambre-consent')===null&&env.doc.getElementById('chat-consent')===null,'no agreement box');assert.equal(env.view.querySelectorAll('input').filter(i=>i.type==='checkbox').length,0);
 assert.equal(env.doc.getElementById('chat-composer-row').hidden,true,'no input bar before the first review');assert.equal(env.form.dataset.closed,'true');assert.equal(env.doc.getElementById('chat-memory-notice').hidden,true);
 assert.equal(env.doc.getElementById('chat-locked-note').textContent,'Vous pourrez écrire à Ambre après votre premier bilan.');assert.equal(env.doc.getElementById('chat-locked-note').hidden,false);
 env.input.value='Bonjour';assert.equal(env.send.disabled,true);assert.ok(env.doc.activeElement!==env.input,'no keyboard is opened');
 env.ctx.state.chatData=ambreState(fullReadiness());env.ctx.drawChat(env.view,()=>true);
 start=env.doc.getElementById('ambre-start');reason=env.doc.getElementById('ambre-start-reason');assert.equal(start.disabled,false,'the launch button alone is enough');assert.equal(reason.hidden,true);assert.equal(start.getAttribute('aria-describedby'),undefined);assert.equal(env.doc.getElementById('ambre-consent-info'),null);
 assert.ok(env.doc.getElementById('ambre-intro').querySelectorAll('.ambre-need').every(i=>i.querySelector('.ambre-need-mark').textContent==='✓'));
 assert.equal(calls.length,0);
 next=ambreState(fullReadiness(),{started:true,can_start:false,review:runningReview()},{consent:true,access:{allowed:true,reason:null},review_id:'r1'});
 await start.fire('click');
 assert.equal(calls.length,1);assert.deepEqual([calls[0][0],calls[0][1],calls[0][2].accepted],['/api/chat/start','POST',true]);assert.match(calls[0][2].idempotency_key,/^[\w-]{8,}$/);
 const progress=env.doc.getElementById('chat-review-indicator');assert.ok(progress.parentElement===thread);assert.ok(thread.children.at(-1)===progress);
 const bar=progress.querySelector('.ambre-bar');assert.equal(bar.getAttribute('role'),'progressbar');assert.equal(bar.getAttribute('aria-valuemin'),'0');assert.equal(bar.getAttribute('aria-valuemax'),'100');assert.equal(bar.getAttribute('aria-valuetext'),'Lecture de vos informations : 5 sur 12');assert.ok(Number(bar.getAttribute('aria-valuenow'))>5&&Number(bar.getAttribute('aria-valuenow'))<70);
 assert.equal(progress.getAttribute('role'),undefined,'the wait is not a second live region');assert.equal(env.view.chatShell.announcement.textContent,'Je prépare votre bilan. Lecture de vos informations : 5 sur 12.');
 assert.match(progress.textContent,/Durée estimée : entre 9 et 13 minutes, prêt au plus tard vers \d\d:\d\d\./);assert.match(progress.textContent,/Vous pouvez fermer cette page : votre bilan apparaîtra ici et dans Mes bilans\./);assert.equal(progress.querySelector('.ambre-late').hidden,true);
 assert.equal(env.doc.getElementById('chat-composer-row').hidden,true,'still no input bar during the first review');
 assert.ok(env.doc.activeElement===progress.querySelector('h3'),'focus follows the launch');assert.match(env.doc.getElementById('ambre-intro').textContent,/Votre bilan est lancé/);assert.equal(env.doc.getElementById('ambre-intro').querySelectorAll('button').length,0);
 assert.equal(env.doc.getElementById('chat-point').hidden,true,'no « Faire le point » during the first review');assert.equal(env.send.disabled,true);
 env.ctx.state.chatData=ambreState(fullReadiness(),{started:true,review:runningReview({stage:'write',blocks_done:12})},{consent:true,access:{allowed:true}});env.ctx.drawChat(env.view,()=>true);
 let current=env.doc.getElementById('chat-review-indicator');assert.equal(current.querySelector('.ambre-bar').getAttribute('aria-valuetext'),'Rédaction de votre bilan');assert.equal(env.view.chatShell.announcement.textContent,'Rédaction de votre bilan.');assert.equal(current.querySelectorAll('.ambre-step')[0].className.includes('is-done'),true);assert.equal(current.querySelectorAll('.ambre-step')[1].getAttribute('aria-current'),'step');
 env.ctx.state.chatData=ambreState(fullReadiness(),{started:true,review:runningReview({stage:'check',blocks_done:12,started_at:Math.floor(Date.now()/1000)-900})},{consent:true,access:{allowed:true}});env.ctx.drawChat(env.view,()=>true);
 current=env.doc.getElementById('chat-review-indicator');assert.equal(current.querySelector('.ambre-bar').getAttribute('aria-valuetext'),'Relecture de votre bilan');assert.equal(current.querySelector('.ambre-late').hidden,false);assert.equal(current.querySelector('.ambre-late').textContent,'C’est plus long que prévu : je continue, rien n’est perdu.');assert.equal(current.querySelector('.ambre-estimate').textContent,'Durée estimée : entre 9 et 13 minutes.');
 env.ctx.state.chatData=ambreState(fullReadiness(),{started:true,review:{last_at:100}},{consent:true,access:{allowed:true,reason:null},messages:[{id:'t1-assistant',role:'assistant',content:'Le bilan complet, très long.',chat_message:'Ce qui ressort : vous vous taisez pour protéger le lien.\n\nQu’aimeriez-vous creuser ?',review_id:'r1',review_kind:'initial'}]});
 env.ctx.drawChat(env.view,()=>true);
 assert.equal(current.isConnected,false);assert.equal(env.view.chatShell.announcement.textContent,'Votre bilan est prêt.');
 const reply=thread.children.at(-1);assert.deepEqual(reply.querySelectorAll('.chat-paragraph').map(p=>p.textContent),['Ce qui ressort : vous vous taisez pour protéger le lien.']);assert.deepEqual(reply.querySelector('.chat-questions').querySelectorAll('li').map(li=>li.textContent),['Qu’aimeriez-vous creuser ?']);assert.doesNotMatch(reply.textContent,/très long/);
 assert.deepEqual(reply.querySelectorAll('a').map(a=>[a.textContent,a.href]),[['Lire le bilan complet','/mes-bilans/'],['Voir Mes diagnostics','/mes-diagnostics/']]);
 assert.equal(env.doc.getElementById('chat-composer-row').hidden,false,'the input bar appears once the review is ready');assert.equal(env.form.dataset.closed,'');
 assert.equal(env.input.getAttribute('aria-disabled'),undefined);assert.equal(env.input.readOnly,false);assert.equal(env.input.disabled,false);assert.equal(env.doc.getElementById('chat-locked-note').hidden,true);assert.equal(env.send.disabled,false);
 assert.match(env.doc.getElementById('ambre-intro').textContent,/Votre premier bilan est fait/);assert.equal(thread.children[0].id,'ambre-intro');
});

test('a failed review is explained from last_failure, survives a reload and is relaunched only when retryable',async()=>{
 const calls=[];let next;const env=chatFixture({api:async(path,method,data)=>{calls.push([path,method||'GET',data]);if(next instanceof Error)throw next;return next;}});
 const failed=(failure,journey={},extra={})=>ambreState(fullReadiness(),{...journey,review:{last_failure:failure,...journey.review}},{consent:true,...extra});
 env.ctx.state.chatData=failed({id:'r1',error_code:'provider_timeout',retryable:true,retry_at:null});env.ctx.drawChat(env.view,()=>true);
 let message=env.doc.getElementById('ambre-review-failure');assert.ok(message.parentElement===env.view.querySelector('.chat-thread'));assert.equal(message.querySelector('h3').textContent,'Le bilan n’a pas pu aboutir');
 assert.match(message.textContent,/Le service de rédaction a mis trop de temps à répondre\./);assert.match(message.textContent,/Rien n’a été perdu/);assert.match(message.textContent,/n’est pas décompté de votre bilan du mois/);assert.match(message.textContent,/jusqu’à 8 bilans par 24 heures/);
 assert.equal(env.view.chatShell.announcement.textContent,'','a reload does not announce an old failure');assert.ok(env.doc.getElementById('ambre-start')===null,'one action: « Relancer »');assert.match(env.doc.getElementById('ambre-intro').textContent,/n’a pas abouti/);
 let retry=env.doc.getElementById('ambre-retry');assert.equal(retry.textContent,'Relancer');assert.equal(retry.disabled,false);
 const lost=Object.assign(new Error('Trop de bilans demandés aujourd’hui. Réessayez demain.'),{status:429,code:'request_limit'});next=lost;await retry.fire('click');
 assert.deepEqual([calls[0][0],calls[0][2].accepted],['/api/chat/start',true]);assert.match(message.textContent,/Trop de bilans demandés aujourd’hui/);assert.equal(retry.disabled,false);
 next=ambreState(fullReadiness(),{started:true,review:runningReview({running_id:'r2'})},{consent:true,access:{allowed:true}});await retry.fire('click');
 assert.equal(calls.length,2);assert.notEqual(calls[0][2].idempotency_key,calls[1][2].idempotency_key,'a definitive refusal renews the key');assert.equal(env.doc.getElementById('ambre-review-failure').isConnected,false);assert.equal(env.doc.getElementById('chat-review-indicator').isConnected,true);
 env.ctx.state.chatData=failed({id:'r2',error_code:'review_not_approved',retryable:true},{started:false});env.ctx.drawChat(env.view,()=>true);
 assert.match(env.view.chatShell.announcement.textContent,/^Le bilan n’a pas pu aboutir\. La relecture n’a pas validé mon texte/);
 env.ctx.state.chatData=failed({id:'r3',error_code:'request_limit',retryable:false,retry_at:Math.floor(Date.now()/1000)+7200});env.ctx.drawChat(env.view,()=>true);
 message=env.doc.getElementById('ambre-review-failure');assert.equal(message.querySelectorAll('button').length,0);assert.match(message.textContent,/Trop de bilans ont été demandés aujourd’hui\./);assert.match(message.textContent,/Un nouveau bilan sera possible à partir du/);
 env.ctx.state.chatData=failed({id:'r4',error_code:'quota_reservation_missing',retryable:false});env.ctx.drawChat(env.view,()=>true);
 assert.doesNotMatch(env.doc.getElementById('ambre-review-failure').textContent,/n’est pas décompté/);
 env.ctx.state.chatData=failed({id:'r5',error_code:'provider_http_503',retryable:true,retry_at:Math.floor(Date.now()/1000)+600,daily_remaining:3});env.ctx.drawChat(env.view,()=>true);
 message=env.doc.getElementById('ambre-review-failure');assert.match(message.textContent,/Le service de rédaction a rencontré une erreur\./);assert.equal(env.doc.getElementById('ambre-retry').disabled,true);assert.match(message.textContent,/Relance possible à partir de \d\d:\d\d\. .*\(il en reste 3\)/);
 assert.equal(env.ctx.ambreFailureReason('something_new'),'Une erreur inattendue a interrompu la préparation.');assert.equal(env.ctx.ambreFailureReason('provider_http_429'),env.ctx.AMBRE_FAILURES.provider_cooldown);
 // A failed point (conversation already started) is relaunched as a point, below the conversation.
 env.ctx.state.chatData=failed({id:'p1',error_code:'context_changed',retryable:true},{started:true,review:{last_at:100}},{access:{allowed:true},messages:[{id:'t1-assistant',role:'assistant',content:'Analyse.',review_id:'r1',review_kind:'initial'}]});env.ctx.drawChat(env.view,()=>true);
 assert.equal(env.view.querySelector('.chat-thread').children.at(-1).id,'ambre-review-failure');next=ambreState(fullReadiness(),{started:true,review:runningReview({running_id:'p2',last_at:100})},{consent:true,access:{allowed:true}});
 await env.doc.getElementById('ambre-retry').fire('click');assert.equal(calls.at(-1)[0],'/api/chat/review');assert.equal(calls.at(-1)[2].accepted,undefined);
 assert.equal(env.doc.getElementById('chat-locked-note').textContent,'Vous pourrez écrire à Ambre quand votre bilan sera prêt.');
});

test('the server list of pages (readiness.pages) gives the counters; « enregistrez » only when every answer is given',()=>{
 const readiness={...partialReadiness(),childhood:{complete:false,answered:40,required:60,saved:false},pages:[{id:'account',label:'Mon compte',href:'/mon-compte/',done:4,total:4,saved:true,complete:true},{id:'people',label:'Mes proches',href:'/mes-proches/',done:3,total:3,saved:true,complete:true},{id:'timeline',label:'Ma ligne de vie',href:'/ma-ligne-de-vie/',done:2,total:3,saved:true,complete:false},{id:'childhood',label:'Mon enfance',href:'/cartographie/',done:40,total:60,saved:false,complete:false},{id:'symptoms',label:'Mes symptômes',href:'/mes-symptomes/',done:516,total:516,saved:false,complete:false}]};
 const env=chatFixture();env.ctx.state.chatData=ambreState(readiness);env.ctx.drawChat(env.view,()=>true);
 const items=env.doc.getElementById('ambre-intro').querySelectorAll('.ambre-need');
 assert.deepEqual(items.map(i=>[i.querySelector('a').textContent,i.querySelector('.ambre-need-count').textContent,i.querySelector('.ambre-need-state').textContent]),[['Mon compte','4/4','Complet'],['Mes proches','3/3','Complet'],['Ma ligne de vie','2/3','À compléter'],['Mon enfance','40/60','À compléter'],['Mes symptômes','516/516','Enregistrez la page']]);
 assert.equal(env.doc.getElementById('ambre-start-reason').textContent,'Complétez d’abord Ma ligne de vie, Mon enfance. Enregistrez Mes symptômes.');
});

test('a visitor sees the same first message without counters and signs in from it',()=>{
 const env=chatFixture();env.ctx.state.chatData={access:{allowed:false,reason:'authentication_required'},messages:[],proposals:[]};env.ctx.drawChat(env.view,()=>true);
 const intro=env.doc.getElementById('ambre-intro');assert.equal(intro.querySelector('h2').textContent,'Bonjour, je suis Ambre.');assert.match(intro.textContent,/assistante psychologue IA/);assert.equal(intro.querySelectorAll('.ambre-need').length,5);
 assert.equal(intro.querySelectorAll('.ambre-need-count').length,0);assert.equal(intro.querySelectorAll('.ambre-need-state').length,0);assert.match(intro.textContent,/Les 516 questions/);
 assert.deepEqual(intro.querySelectorAll('button').map(b=>[b.textContent,b.dataset.auth]),[['Se connecter','login'],['Créer un compte','register']]);
 assert.ok(env.doc.getElementById('ambre-consent')===null);assert.ok(env.doc.getElementById('ambre-start')===null);assert.ok(env.view.querySelector('.chat-gate')===null);
 assert.equal(env.doc.getElementById('chat-locked-note').textContent,'Connectez-vous pour écrire à Ambre.');assert.equal(env.doc.getElementById('chat-composer-row').hidden,true,'no input bar for a visitor');assert.equal(env.send.disabled,true);assert.equal(env.doc.getElementById('chat-quota').parentElement.hidden,true);
});

test('the review state is read every 15 seconds and never while the page is hidden',async()=>{
 const timers=[],listeners={};let calls=0;const env=chatFixture({setTimeout(fn,delay){timers.push({fn,delay});return timers.length},clearTimeout(){},api:async()=>{calls++;return ambreState(fullReadiness(),{started:true,review:runningReview({blocks_done:6})},{consent:true,access:{allowed:true}})}});
 env.doc.addEventListener=(name,fn)=>{(listeners[name]??=[]).push(fn)};
 env.ctx.state.chatData=ambreState(fullReadiness(),{started:true,review:runningReview()},{consent:true,access:{allowed:true}});env.ctx.drawChat(env.view,()=>true);
 let timer=timers.at(-1);assert.equal(timer.delay,15000);
 env.doc.hidden=true;await timer.fn();assert.equal(calls,0,'no read while hidden');const pending=timers.length;
 env.doc.hidden=false;for(const fn of listeners.visibilitychange||[])fn();assert.equal(timers.length,pending+1,'reading resumes when the page is shown again');assert.equal(timers.at(-1).delay,0,'on return to the tab the state is read at once');
 await timers.at(-1).fn();assert.equal(calls,1);assert.equal(env.doc.getElementById('chat-review-indicator').querySelector('.ambre-bar').getAttribute('aria-valuetext'),'Lecture de vos informations : 6 sur 12');
});

test('an old review envelope keeps its content and link, and the former welcome text is not shown',()=>{
 const env=chatFixture();env.ctx.state.chatData=ambreState(fullReadiness(),{started:true,review:{last_at:100}},{consent:true,access:{allowed:true},messages:[{id:'ambre-welcome-v3',role:'assistant',content:'Ancien accueil v3 : bonjour.'},{id:'t1-assistant',role:'assistant',content:'Votre analyse initiale.',review_id:'r1',review_kind:'initial'}]});env.ctx.drawChat(env.view,()=>true);
 const thread=env.view.querySelector('.chat-thread');assert.equal(thread.children.length,2);assert.equal(thread.children[0].id,'ambre-intro');assert.doesNotMatch(thread.textContent,/Ancien accueil v3|Ancien message/);
 const legacy=thread.children[1];assert.equal(legacy.querySelector('h3').textContent,'Ambre · Analyse initiale');assert.equal(legacy.querySelector('.chat-content').textContent,'Votre analyse initiale.');assert.deepEqual(legacy.querySelectorAll('a').map(a=>[a.textContent,a.href]),[['Retrouver ce bilan dans Mes bilans','/mes-bilans/']]);
 env.ctx.state.chatData.messages[1].chat_message='   ';env.ctx.drawChat(env.view,()=>true);assert.equal(legacy.querySelector('.chat-content').textContent,'Votre analyse initiale.','an empty chat_message keeps content');
});

test('after a withdrawn agreement « Reprendre la conversation » gives it again, without any box',async()=>{
 const calls=[];const env=chatFixture({api:async(path,method,data)=>{calls.push([path,method,data]);return {}}});env.ctx.renderChat=()=>calls.push(['render']);
 env.ctx.state.chatData=ambreState(fullReadiness(),{started:true,review:{last_at:100}},{consent:false,access:{allowed:false,reason:'consent_required'}});env.ctx.drawChat(env.view,()=>true);
 const resume=env.doc.getElementById('ambre-start');assert.equal(resume.textContent,'Reprendre la conversation');assert.equal(resume.disabled,false);assert.ok(env.doc.getElementById('ambre-consent')===null);assert.match(env.doc.getElementById('chat-locked-note').textContent,/Touchez « Reprendre la conversation »/);assert.equal(env.doc.getElementById('chat-composer-row').hidden,false,'the conversation is open: the bar stays, disabled');
 await resume.fire('click');assert.equal(JSON.stringify(calls[0]),JSON.stringify(['/api/chat/consent','PUT',{accepted:true}]));
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
 assert.match(waiting.textContent,/Je prépare votre bilan/);assert.equal(waiting.querySelector('.ambre-bar').getAttribute('role'),'progressbar');assert.equal(waiting.parentElement,env.view.querySelector('.chat-thread'));
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
 const buttons=items[1].querySelectorAll('button'),read=buttons[0],remove=buttons.at(-1);await read.fire('click');
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


test('V8 Ambre keeps the canonical route and identifies the chat as AI',()=>{
 const env=environment();vm.runInContext(script.match(/var PAGES=.*\n/)[0]+functionCode('drawChat'),env.ctx);assert.equal(env.ctx.PAGES.chat.title,'Ambre');assert.equal(env.ctx.PAGES.chat.path,'/ambre/');env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:100},messages:[{role:'assistant',content:'Bonjour.'},{role:'user',content:'Mon accompagnement, dans mes propres mots.'}],proposals:[]};const host=env.host('view');env.ctx.drawChat(host,()=>true);assert.match(host.textContent,/Ambre · assistante psychologue IA/);assert.equal(host.querySelector('.chat-assistant').querySelector('h3').textContent,'Ambre');assert.match(host.querySelector('.chat-user').textContent,/Mon accompagnement, dans mes propres mots/);assert.doesNotMatch(host.textContent,/Assistant IA/);assert.equal(host.querySelectorAll('h1').length,1);assert.doesNotMatch(functionCode('landing'),/<h1>Ambre<\/h1>/);
});

test('manual presentation, close-person and life-event forms omit the repeated Ambre notes',async()=>{
 const env=environment({api:async()=>({profile:{first_name:'Ma présentation'},revision:1}),location:{hash:''}});env.host('self-profile');vm.runInContext(functionCode('renderSelfProfile'),env.ctx);await env.ctx.renderSelfProfile();const self=env.doc.getElementById('self-profile');assert.equal(self.querySelector('.ambre-hint'),null);assert.equal(env.doc.getElementById('self-profile-form').elements.first_name.value,'Ma présentation');
 const rendered=[];env.ctx.drawSourceSection=kind=>rendered.push(kind);env.ctx.api=async()=>({people:[],events:[]});vm.runInContext(functionCode('renderPersonalSources')+'\n'+functionCode('eventsListQuery'),env.ctx);const host=env.host('view');for(const page of ['people','events']){env.ctx.state.tab=page;await env.ctx.renderPersonalSources();assert.doesNotMatch(host.innerHTML,/ambre-hint|Ambre peut aussi vous aider à ajouter|role="note"/);assert.match(host.innerHTML,new RegExp('id="'+page+'-content"'))}assert.deepEqual(rendered,['people','events']);
});

test('V8 memory append is an explicit editable proposal that retains the text and revision on failure',async()=>{
 let sent;const env=environment({api:async(path,method,payload)=>{sent={path,payload};throw new Error('Conflit')}});env.ctx.state.chatProposalDrafts=new Map();vm.runInContext(functionCode('drawChatProposal'),env.ctx);const host=env.host('proposals');env.ctx.drawChatProposal(host,{id:'memory-1',kind:'memory_append',revision:7,payload:{text:'Mon contexte actuel.'},evidence_quote:'Mes mots exacts.'},()=>true);const form=host.children[0],input=form.elements.text;assert.match(form.textContent,/Préciser mon portrait/);assert.match(form.textContent,/Votre précision/);assert.match(form.textContent,/informations déjà enregistrées seront conservées/);assert.equal(input.maxLength,2000);assert.equal(input.required,true);assert.equal(sent,undefined);input.value='Ma précision modifiée.';await input.fire('input');await form.fire('submit');assert.equal(sent.path,'/api/chat/proposals/memory-1/accept');assert.equal(sent.payload.revision,7);assert.equal(sent.payload.payload.text,'Ma précision modifiée.');assert.equal('expected_version' in sent.payload.payload,false);assert.equal(env.ctx.state.chatProposalDrafts.get('memory-1').text,'Ma précision modifiée.');assert.match(form.textContent,/Vos modifications sont conservées/);assert.equal(form.querySelector('blockquote').textContent,'Mes mots exacts.');
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

test('V10 an outbox provider cooldown keeps the precise error and automatically enables retry without changing the next draft',async()=>{
 const timers=new Map();let serial=0,now=Date.now(),client,answer;const env=chatFixture({Date:{now:()=>now},setTimeout(fn){timers.set(++serial,fn);return serial},clearTimeout(id){timers.delete(id)},api:async(path,method,payload)=>{if(method==='POST'){client=payload.idempotency_key;return {consent:true,access:{allowed:true},quota:{remaining:99,limit:100},pending_request:{id:'limited',client_key:client,status:'running'},messages:[{id:'limited-user',client_key:client,role:'user',content:payload.content,status:'running'}],proposals:[]}}return answer}});
 env.input.value='Message initial';await env.input.fire('input');await env.ctx.sendChat(env.form,()=>true);
 answer={consent:true,access:{allowed:false,reason:'provider_cooldown',retry_at:(now+30000)/1000},quota:{remaining:100,limit:100},last_request:{id:'limited',status:'failed'},last_error:{code:'provider_http_429_request',message:'Le fournisseur limite temporairement les demandes. Aucun message n’a été décompté.',action:'retry_later'},messages:[{id:'limited-user',client_key:client,role:'user',content:'Message initial',status:'failed'}],proposals:[]};
 let [id,callback]=[...timers][0];timers.delete(id);await callback();const form=env.form,retry=env.view.querySelector('.chat-turn-state').querySelector('button');assert.equal(retry.disabled,true);assert.equal(env.input.value,'Message initial');assert.match(env.doc.getElementById('chat-status').textContent,/Le fournisseur limite temporairement les demandes\. Aucun message n’a été décompté/);assert.equal(env.ctx.state.chatSendKey,null);
 env.input.value='Un prochain message';await env.input.fire('input');now+=31000;answer={...answer,access:{allowed:true}};[id,callback]=[...timers][0];timers.delete(id);await callback();assert.equal(env.form,form);assert.equal(env.input.value,'Un prochain message');assert.equal(retry.disabled,false);assert.equal(env.send.disabled,false);assert.equal(env.ctx.state.chatData.quota.remaining,100);
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

test('date pickers never manufacture a month or a day when the precision changes',async()=>{
 const env=editFixture('events',{title:'Souvenir',description:'Récit',kind:'positive',date_precision:'year',date_start:'1998',people_ids:[]});
 const {date_precision:precision,date_start:start}=env.form.elements;assert.equal(start.type,'hidden');assert.equal(start.value,'1998');
 precision.value='day';await precision.fire('change');assert.equal(start.value,'','a day needs its month and its day');
 precision.value='month';await precision.fire('change');assert.equal(start.value,'','a month needs its month');
 precision.value='year';await precision.fire('change');assert.equal(start.value,'1998');precision.value='range';await precision.fire('change');assert.equal(start.value,'1998');assert.equal(env.form.elements.date_end.value,'');
 precision.value='year';await precision.fire('change');await env.form.fire('input');env.ctx.drawSourceEditor('events');assert.equal(env.doc.getElementById('events-form').elements.date_start.value,'1998');
});

test('meeting dates use the same partial controls and preserve legacy period text',async()=>{
 let sent;const env=editFixture('people',{display_name:'Camille',role:'friend',meeting_date_precision:'month',meeting_date_start:'2010-05',period_start:'Pendant mes études',period_end:''},{api:async(p,m,data)=>{sent=data;throw new Error('offline')}});
 assert.equal(env.form.elements.meeting_date_start.type,'hidden');assert.equal(env.form.elements.meeting_date_start.value,'2010-05');await env.form.fire('submit');assert.equal(sent.meeting_date_start,'2010-05');assert.equal(sent.period_start,'Pendant mes études');
});

test('an already-started conversation is blocked when the strict readiness checklist becomes incomplete',()=>{
 const readiness={complete:false,identity:{complete:true,missing_fields:[]},people:{complete:false,count:2,required:3},events:{complete:true,count:3,required:3},childhood:{complete:true,answered:60,required:60},symptoms:{complete:true,answered:516,required:516}};
 const env=chatFixture();env.ctx.state.chatData=journeyState({started:true,can_start:false,readiness,review:{last_at:100}},{consent:true,access:{allowed:true}});env.ctx.drawChat(env.view,()=>true);
 assert.equal(env.send.disabled,true);assert.equal(env.form.hidden,false,'the composer stays visible');assert.equal(env.input.getAttribute('aria-disabled'),'true');assert.match(env.doc.getElementById('chat-locked-note').textContent,/Complétez les pages indiquées/);
 const intro=env.doc.getElementById('ambre-intro'),people=intro.querySelectorAll('.ambre-need')[1];assert.equal(people.querySelector('.ambre-need-count').textContent,'2/3');assert.equal(people.querySelector('a').href,'/mes-proches/');assert.match(intro.textContent,/Pour reprendre la conversation, complétez les pages/);
 readiness.complete=true;readiness.people.complete=true;readiness.people.count=3;env.ctx.drawChat(env.view,()=>true);assert.equal(env.send.disabled,false);assert.equal(env.input.getAttribute('aria-disabled'),undefined);
});

function noteFixture(note,api){
 let saved=0;const env=environment({api});for(const name of ['noteTarget','parseStamp','shortDay','addedOn','historyButton','applyNoteRevision','sourceNoteEditor','sourceNoteCard'])vm.runInContext(functionCode(name),env.ctx);
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
 assert.doesNotMatch(HTML,/<iframe/);
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
 for(const name of ['personRoleLabel','personIdForPath','topicIdForPath','pagePath','noteTarget','parseStamp','shortDay','addedOn','historyButton','noteDay','sourceNoteEditor','sourceNoteCard','eventNotes','personTimeline','renderPerson','drawPersonPage','personSaved','drawSourceItem'])vm.runInContext(functionCode(name),env.ctx);
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
 const buttons=env.box.querySelector('.person-info').querySelectorAll('button').map(button=>button.textContent);assert.deepEqual(buttons.slice(0,3),['Modifier','Historique','Supprimer']);
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

function archiveFixture(extra={}) {const env=environment(extra);vm.runInContext(REVIEW_ARCHIVE_SCRIPT,env.ctx);return env;}
test('archived assessment separates associations, uncertainty, missing declarations and contextual gates',()=>{
 const env=archiveFixture();const result=env.ctx.drawArchivedAssessment({rules_version:'test',outcomes:[{label:'Sujet A',hypothesis_state:'criterion_signals',context_gates:{type:'perinatal',current_window:false,historical:true},evidence:{current_window:[{label:'Signe',answer:1,note_excerpt:'Extrait personnel',note_excerpted:true}],unknown:[{label:'Question incertaine',answer:'unknown'},{label:'Question sans réponse',answer:'unanswered'}]},dimensions:{}},{label:'Sujet B',hypothesis_state:'association_only',evidence:{},dimensions:{}}]},{captured_at:'2026-11-19T12:00:00Z',schema_version:1});
 assert.match(result.textContent,/Signes caractéristiques/);assert.match(result.textContent,/Associations non spécifiques/);assert.match(result.textContent,/Grossesse ou période après la naissance/);assert.match(result.textContent,/uniquement pour le passé/);assert.match(result.textContent,/note complète/);assert.match(result.textContent,/Aucun pourcentage de probabilité/);
 const sections=result.querySelectorAll('.diagnosis-evidence');assert.match(sections.find(s=>s.textContent.includes('Je ne sais pas')).textContent,/Question incertaine/);assert.doesNotMatch(sections.find(s=>s.textContent.includes('Je ne sais pas')).textContent,/Question sans réponse/);const missing=sections.find(s=>s.textContent.includes('non renseignés'));assert.match(missing.textContent,/Question sans réponse/);assert.doesNotMatch(missing.textContent,/réponse déclarée/);
});
test('archive reads immutable sections lazily, paginates, and keeps personal text inert',async()=>{
 const calls=[];const env=archiveFixture({api:async path=>{calls.push(path);return calls.length===1?{items:[{id:'hidden-id',title:'<img src=x>',description:'Premier souvenir'}],next_cursor:50}:{items:[{title:'Second souvenir'}],next_cursor:null}}});
 const host=env.ctx.archiveSnapshotSections('r1',{sections:[{id:'events',count:51}]},()=>true);assert.equal(calls.length,0);const details=host.querySelector('details');details.open=true;await details.fire('toggle');await new Promise(r=>setImmediate(r));assert.match(host.textContent,/Premier souvenir/);assert.match(host.textContent,/<img src=x>/);assert.equal(host.querySelector('img'),null);assert.doesNotMatch(host.textContent,/hidden-id/);await host.querySelector('button').fire('click');assert.match(calls[1],/cursor=50/);assert.match(host.textContent,/Second souvenir/);await details.fire('toggle');assert.equal(calls.length,2);
});
test('archive rejects stale section responses and retries failed reads',async()=>{
 let resolve,active=true;const env=archiveFixture({api:()=>new Promise(r=>{resolve=r})});const host=env.ctx.archiveSnapshotSections('r',{sections:['notes']},()=>active);const details=host.querySelector('details');details.open=true;await details.fire('toggle');active=false;resolve({items:[{text:'Private stale note'}]});await new Promise(r=>setImmediate(r));assert.doesNotMatch(host.textContent,/Private stale note/);
 let n=0;env.ctx.api=async()=>{if(!n++)throw Error('Hors ligne');return {items:[{text:'Note retrouvée'}]}};const retry=env.ctx.archiveSnapshotSections('r',{sections:['notes']},()=>true),d=retry.querySelector('details');d.open=true;await d.fire('toggle');await new Promise(r=>setImmediate(r));assert.match(retry.textContent,/Hors ligne/);await retry.querySelector('button').fire('click');await new Promise(r=>setImmediate(r));assert.match(retry.textContent,/Note retrouvée/);
});
test('shared archive sections use the authorized professional endpoint without owner requests',async()=>{
 const calls=[];const env=archiveFixture({api:async path=>{calls.push(path);return {items:[{person_label:'Nina',event_title:'Rencontre'}],next_cursor:null}}});const host=env.ctx.archiveSnapshotSections('review',{sections:[{id:'event_people',count:1}]},()=>true,'/api/care/dossiers/link/reviews/review/snapshot'),d=host.querySelector('details');d.open=true;await d.fire('toggle');await new Promise(r=>setImmediate(r));assert.equal(calls[0],'/api/care/dossiers/link/reviews/review/snapshot?section=event_people&limit=50');assert.match(host.textContent,/Nina/);assert.match(host.textContent,/Rencontre/);assert.doesNotMatch(host.textContent,/Autre information/);assert.match(env.ctx.archiveMetadata({scope_limited:true,captured_at:'2026-11-19T12:00:00Z'}).textContent,/rubriques autorisées/);
});

 test('Ambre checklist shows 515 of 516 symptoms and links to complete them',()=>{
 const readiness={complete:false,identity:{complete:true,missing_fields:[]},people:{complete:true,count:3,required:3},events:{complete:true,count:3,required:3},childhood:{complete:true,answered:60,required:60},symptoms:{complete:false,answered:515,required:516}};const env=chatFixture();env.ctx.state.chatData=journeyState({started:false,can_start:false,readiness,review:{}},{consent:true,access:{allowed:false}});env.ctx.drawChat(env.view,()=>true);const symptoms=env.doc.getElementById('ambre-intro').querySelectorAll('.ambre-need')[4];assert.equal(symptoms.querySelector('.ambre-need-count').textContent,'515/516');assert.match(symptoms.querySelector('a').getAttribute('aria-label'),/515 sur 516/);assert.equal(symptoms.querySelector('a').href,'/mes-symptomes/');assert.equal(env.doc.getElementById('ambre-start').disabled,true);assert.match(env.doc.getElementById('ambre-start-reason').textContent,/Complétez d’abord Mes symptômes/);assert.equal(env.send.disabled,true);
});

// Relecture du 19/11 : un message exact par code ; les limites de taille sont définitives, sans relance.
test('every review failure code has its own plain message; definitive limits say there is no relaunch',()=>{
 const env=chatFixture(),F=env.ctx.AMBRE_FAILURES;
 for(const code of ['sources_changed','context_changed','safety_changed','review_not_approved','consent_withdrawn','access_required','prerequisites_required','snapshot_too_large','snapshot_analysis_limit','generation_budget','daily_cap','provider_cooldown','provider_timeout','attempts_exhausted','nothing_to_analyze','quota_reservation_missing'])assert.ok(F[code],code);
 assert.equal(new Set(Object.values(F)).size,Object.keys(F).length,'no message shared by two codes');
 for(const code of ['snapshot_too_large','snapshot_analysis_limit'])assert.match(F[code],/limite fixe : relancer ne changerait rien/);
 assert.doesNotMatch(F.snapshot_analysis_limit,/aujourd’hui/);assert.doesNotMatch(F.generation_budget,/aujourd’hui/);assert.match(F.generation_budget,/semaine/);
});
test('the launch feedback is announced, and no state shows an agreement box',async()=>{
 const calls=[];const env=chatFixture({api:async(...args)=>{calls.push(args);return {consent:true,access:{allowed:true},messages:[]}}});
 env.ctx.state.chatData=ambreState(fullReadiness(),{},{consent:false,access:{allowed:false,reason:'consent_required'}});env.ctx.drawChat(env.view,()=>true);
 assert.equal(env.doc.getElementById('ambre-launch-status').getAttribute('role'),'status');
 env.ctx.state.chatData={consent:false,access:{allowed:false,reason:'consent_required'},messages:[]};env.ctx.drawChatGate(env.view.chatShell,env.ctx.state.chatData,()=>true);
 assert.ok(env.doc.getElementById('chat-consent')===null&&env.doc.getElementById('chat-consent-accept')===null&&env.doc.getElementById('chat-consent-gate')===null);assert.equal(env.view.chatShell.gate.children.length,0);assert.equal(calls.length,0);
 assert.doesNotMatch(script,/chat-consent-gate|chat-consent-accept|ambre-consent'|type='checkbox';checkbox\.id/,'no agreement box anywhere in the conversation');
});

// Corrections 19/11 (recette réelle): in the new envelopes content equals chat_message and the
// server sends links; the two links are shown, and a single long block becomes short paragraphs
// with the questions in a separate block.
test('a review message shows the server links and is drawn as short paragraphs with its questions apart',()=>{
 const long='Votre bilan est prêt. Deux pistes ressortent : hypothèse : une exigence forte envers vous-même, appuyée par vos réponses sur le travail ; il manque des exemples récents. Vos appuis sont nets : vos amis et la marche. Une autre piste concerne la difficulté à demander de l’aide, appuyée par le récit de votre premier poste ; il manque votre avis. Vous avez aussi décrit une grande fatigue le soir, qui mérite qu’on en parle. Qu’aimeriez-vous creuser en premier ? Qu’est-ce qui vous semble juste, ou à corriger ?';
 const links=[{label:'Lire le bilan complet',href:'/mes-bilans/'},{label:'Voir Mes diagnostics',href:'/mes-diagnostics/'},{label:'Piège',href:'javascript:alert(1)'}];
 const env=chatFixture();env.ctx.state.chatData={consent:true,access:{allowed:true},quota:{remaining:99},messages:[{id:'t1-assistant',role:'assistant',content:long,chat_message:long,links,review_id:'r1',review_kind:'initial'},{id:'t2-assistant',role:'assistant',content:'Ancien bilan.',review_id:'r0',review_kind:'initial'}],proposals:[]};
 env.ctx.drawChat(env.view,()=>true);const [reply,legacy]=env.view.querySelector('.chat-thread').children.filter(x=>x.dataset&&x.dataset.chatTurnId);
 assert.deepEqual(reply.querySelectorAll('.chat-review-link').map(a=>[a.textContent,a.href]),[['Lire le bilan complet','/mes-bilans/'],['Voir Mes diagnostics','/mes-diagnostics/']],'server links shown even when content equals chat_message; unsafe href dropped');
 const paragraphs=reply.querySelectorAll('.chat-paragraph').map(p=>p.textContent);assert.ok(paragraphs.length>=2,'the long block is cut');assert.ok(paragraphs.every(p=>p.length<=360&&!/\?/.test(p)));
 assert.deepEqual(reply.querySelector('.chat-questions').querySelectorAll('li').map(li=>li.textContent),['Qu’aimeriez-vous creuser en premier ?','Qu’est-ce qui vous semble juste, ou à corriger ?']);
 assert.deepEqual(legacy.querySelectorAll('.chat-review-link').map(a=>a.textContent),['Retrouver ce bilan dans Mes bilans']);
 env.ctx.drawChat(env.view,()=>true);assert.equal(reply.querySelectorAll('.chat-review-link').length,2,'a redraw does not duplicate the links');
});
