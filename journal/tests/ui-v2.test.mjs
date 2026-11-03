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
  append(...items) { for (const item of items) { this.children.push(item); if(typeof item==='object')item.parentElement=this; } }
  replaceChildren(...items) { this.children=[];this.append(...items); }
  set textContent(value) { this._text=String(value);this.children=[]; }
  get textContent() { return this._text + this.children.map(x=>typeof x==='string'?x:x.textContent).join(''); }
  setAttribute(name,value) { this.attributes[name]=value; }
  addEventListener(name,fn) { (this.listeners[name]??=[]).push(fn); }
  async fire(name) { for (const fn of this.listeners[name]||[]) await fn({target:this,currentTarget:this,preventDefault(){}}); }
  focus() {this.doc.activeElement=this;}
  matches(selector) {
    if(selector==='input,select,textarea')return ['INPUT','SELECT','TEXTAREA'].includes(this.tagName);
    if(selector==='[data-reload-source]')return this.dataset.reloadSource!==undefined;
    if(selector==='[data-continue-character]')return this.dataset.continueCharacter!==undefined;
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
  const ctx=vm.createContext({document:doc,FormData:FormDataMock,URLSearchParams,crypto:{randomUUID:()=> 'create-uuid'},state:{people:[],events:[],sourceDrafts:new Map(),continueKeys:new Map(),tab:'journal',characterData:{},characters:[]},renderTokens:new WeakMap(),confirm:()=>true,refreshRecommendations(){},status(){},console,...extra});
  for (const name of ['renderGuard','el','actionButton','formValues','busy','sourceField','newClientId','captureSourceDraft','sourceDate','drawSourceEditor','characterSearch','requestIsActive']) vm.runInContext(functionCode(name),ctx);
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
  assert.match(HTML,/id="suggestions" tabindex="-1"/);
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
  assert.match(HTML,/Mon portrait déclaré/);assert.match(HTML,/L’interprétation de votre portrait par l’IA/);
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
