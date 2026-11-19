// « Mes symptômes » lot B : page /mes-symptomes/ (flow of questions, payloads, safety screen
// computed in the page, states of the page, results, crossing, Paramètres line, Worker routes).
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {setup} from './v4-R-helper.mjs';
import {SYMPTOMES as S} from '../src/symptomes-data.generated.mjs';
import {createSymptomsBackend} from '../src/symptomes-backend.mjs';
import {SYMPTOMES_INTERFACE_SCRIPT,SYMPTOMES_INTERFACE_STYLE,symptomsClientScript,SYMPTOMES_ENGINE_PATH} from '../src/symptomes-interface.mjs';
import {SYMPTOMES_ENGINE_SOURCE} from '../src/symptomes-engine-source.generated.mjs';
import {SETTINGS_INTERFACE_SCRIPT} from '../src/settings-interface.mjs';
import {HTML as SHELL,APP_SCRIPT,APP_STYLE} from '../src/ui.mjs';
// Lot P: the private shell plus its two content-addressed files, as the browser assembles them.
const HTML=SHELL.replace('</head>',()=>'<style>'+APP_STYLE+'</style></head>')+'<script>'+APP_SCRIPT+'</script>';

// ---- Minimal DOM -------------------------------------------------------------------------------
class Node {
 constructor(tag,doc){this.tagName=tag.toUpperCase();this.doc=doc;this.children=[];this.listeners={};this.dataset={};this.attributes={};this.style={props:{},setProperty(k,v){this.props[k]=v}};this.className='';this.hidden=false;this.disabled=false;this.open=false;this._text='';this.parent=null}
 get isConnected(){let n=this;while(n.parent)n=n.parent;return n===this.doc.body||n===this.doc.head}
 append(...items){for(const item of items){if(item==null)continue;if(typeof item==='object'){if(item.parent)item.remove();item.parent=this}this.children.push(item)}}
 replaceChildren(...items){for(const c of this.children)if(typeof c==='object')c.parent=null;this.children=[];this._text='';this.append(...items)}
 remove(){if(this.parent){this.parent.children=this.parent.children.filter(c=>c!==this);this.parent=null}}
 set textContent(v){this.children=[];this._text=String(v)}
 get textContent(){return this._text+this.children.map(c=>typeof c==='string'?c:c.textContent).join('')}
 setAttribute(k,v){this.attributes[k]=String(v);if(k==='id')this.id=String(v)}
 getAttribute(k){return this.attributes[k]??null}
 removeAttribute(k){delete this.attributes[k]}
 addEventListener(k,fn){(this.listeners[k]??=[]).push(fn)}
 async fire(k,extra={}){let prevented=false;for(const fn of this.listeners[k]||[])await fn({target:this,currentTarget:this,preventDefault(){prevented=true},...extra});return prevented}
 focus(){this.doc.activeElement=this}
 scrollIntoView(){}
 showModal(){this.open=true;this.doc.modals.push(this)}
 close(){this.open=false;this.fire('close')}
 matches(sel){
  return sel.split(',').map(s=>s.trim()).some(s=>{const m=s.match(/^([a-z]+)?(?:#([\w-]+))?((?:\.[\w-]+)*)(?:\[([\w-]+)(?:="([^"]*)")?\])?$/i);if(!m)return false;const [,tag,id,cls,attr,val]=m;
   if(tag&&this.tagName!==tag.toUpperCase())return false;if(id&&this.id!==id)return false;
   if(cls&&cls.split('.').filter(Boolean).some(c=>!String(this.className).split(/\s+/).includes(c)))return false;
   if(attr){const key=attr.startsWith('data-')?attr.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase()):null;const v=key?this.dataset[key]:this.attributes[attr]??this[attr];if(v===undefined||v===null)return false;if(val!==undefined&&String(v)!==val)return false}
   return true});
 }
 querySelectorAll(sel){return this.children.flatMap(c=>typeof c==='object'?[...(c.matches(sel)?[c]:[]),...c.querySelectorAll(sel)]:[])}
 querySelector(sel){return this.querySelectorAll(sel)[0]||null}
}
const button=(host,label)=>host.querySelectorAll('button').find(b=>b.textContent===label||b.textContent.startsWith(label));
const choice=(host,q,value)=>host.querySelector('[data-sy-focus="'+q+':'+value+'"]')||host.querySelectorAll('input').find(b=>b.dataset.syFocus&&b.dataset.syFocus.endsWith(':'+q+':'+value));
const plain=v=>JSON.parse(JSON.stringify(v));
const tick=node=>{node.checked=true;return node.fire('change')};
const flush=()=>new Promise(r=>setTimeout(r,0));

function environment({handler}={}){
 const doc={modals:[],activeElement:null,createElement(tag){return new Node(tag,this)},getElementById(id){return this.body.querySelector('#'+id)||this.head.querySelector('#'+id)}};
 doc.body=new Node('body',doc);doc.head=new Node('head',doc);
 const calls=[],storage=new Map(),session=new Map(),timers=[],delays=[];
 const store=map=>({getItem:k=>map.get(k)??null,setItem:(k,v)=>map.set(k,String(v)),removeItem:k=>map.delete(k),key:i=>[...map.keys()][i]??null,get length(){return map.size}});
 const ctx={document:doc,console,JSON,Math,Date,Promise,Object,Array,String,Number,Boolean,Error,Set,Map,WeakMap,RegExp,Uint8Array,crypto:globalThis.crypto,location:{replaced:null,replace(u){this.replaced=u},assign(){}},
  localStorage:store(storage),sessionStorage:store(session),
  setTimeout:(fn,ms)=>{timers.push(fn);delays.push(ms);return timers.length},clearTimeout:()=>{},
  state:{me:{user:{id:'a'}}},dateLabel:v=>'date:'+v,
  el(tag,cls,text){const n=doc.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n},
  actionButton(text,fn,cls){const b=ctx.el('button',cls||'',text);b.type='button';b.addEventListener('click',fn);return b},
  confirmDialog:async()=>true,
  api:async(path,method='GET',data)=>{calls.push({path,method,data:data===undefined?undefined:JSON.parse(JSON.stringify(data))});return handler(path,method,data)}};
 ctx.window=ctx;vm.createContext(ctx);
 vm.runInContext(symptomsClientScript(),ctx);vm.runInContext(SYMPTOMES_INTERFACE_SCRIPT,ctx);
 const view=doc.createElement('div');doc.body.append(view);
 return {ctx,doc,view,calls,timers,delays,storage,session,async runTimers(){while(timers.length){await timers.shift()();await flush()}}};
}

// ---- Real backend for the page states ----------------------------------------------------------
function backend(){
 const s=setup({until:'9999',users:['a']});// The page dates its passations with Date.now(): the server clock starts from the real time (a fixed
 // clock rejected every « started_at » once the real time passed it by five minutes).
 let clock=Math.floor(Date.now()/1000),serial=0;
 const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),body:async req=>JSON.parse(await req.text()),now:()=>clock,random:()=>'sym_'+String(++serial).padStart(6,'0'),limit:async()=>{}};
 const b=createSymptomsBackend(utils);
 async function handler(path,method,data){
  const req=new Request('https://example.test'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});
  const url=new URL(req.url),res=await b.routes(req,s.env,{id:'a'},url.pathname,url);const body=await res.json();
  if(res.status>=400)throw Object.assign(new Error(body.error||'erreur'),{status:res.status,code:body.code});return body;
 }
 const birth=date=>s.raw.prepare('UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id=?').run(JSON.stringify({birth_date:date}),'a');
 return {...s,handler,birth,tick:n=>{clock+=n}};
}

test('the page engine is the Worker engine: generated copy current, served as a plain script',async()=>{
 const engine=readFileSync(new URL('../src/symptomes-engine.mjs',import.meta.url),'utf8');
 assert.equal(SYMPTOMES_ENGINE_SOURCE,engine.replace(/^export\s+(?=(?:async\s+)?(?:function|const|let)\s)/gm,''),'run node scripts/build_symptomes_page_engine.mjs');
 const script=symptomsClientScript();assert.doesNotMatch(script,/^\s*(import|export)\s/m);
 const ctx={};ctx.window=ctx;vm.createContext(ctx);vm.runInContext(script,ctx);
 assert.equal(ctx.SYMPTOMES_CONTENT.version,S.version);assert.equal(typeof ctx.SymptomesEngine.computeModuleResult,'function');
 assert.equal(ctx.SYMPTOMES_SUBJECT_LINKS.dep_generale,'/personnages/louise/');
 assert.equal(SYMPTOMES_ENGINE_PATH,'/mes-symptomes/moteur.js');
 assert.doesNotThrow(()=>new vm.Script(SYMPTOMES_INTERFACE_SCRIPT));
 // The page is part of the shared shell (menu, style), after Mon enfance.
 assert.match(HTML,/<li><a href="\/cartographie\/">Mon enfance<\/a><\/li>\n<li><a href="\/mes-symptomes\/">Mes symptômes<\/a><\/li>/);
 assert.match(HTML,/symptoms:\{path:'\/mes-symptomes\/',title:'Mes symptômes'\}/);assert.match(HTML,/renderSymptoms\(/);
 assert.match(SYMPTOMES_INTERFACE_STYLE,/min-height: 44px/);assert.match(SYMPTOMES_INTERFACE_STYLE,/:active/);assert.match(SYMPTOMES_INTERFACE_STYLE,/max-width: 599px/);
});

test('flow: PHQ-2 then PHQ-9 when the entry reaches 2, full instrument on request, reference period first',()=>{
 const {ctx}=environment({handler:()=>({})});
 let f=ctx.syFlow('humeur',{});assert.deepEqual(plain(f.ids),['phq9_1','phq9_2']);assert.equal(f.groups[0].head,S.instruments.phq9.stem);assert.equal(f.offer,null);
 f=ctx.syFlow('humeur',{phq9_1:0,phq9_2:1});assert.deepEqual(plain(f.ids),['phq9_1','phq9_2']);assert.equal(f.offer,'phq9');
 f=ctx.syFlow('humeur',{phq9_1:1,phq9_2:1});
 assert.deepEqual(plain(f.ids),['phq9_1','phq9_2','phq9_3','phq9_4','phq9_5','phq9_6','phq9_7','phq9_8','phq9_9','humeur_duree','humeur_perinatal','humeur_gene']);
 assert.equal(f.groups[0].items.length,9,'PHQ-2 and PHQ-9 share one group under the official stem');assert.match(f.groups[0].notice,/Pfizer/);
 assert.equal(f.groups[1].head,S.periods['2_ans'].label);
 f=ctx.syFlow('humeur',{phq9_1:0,phq9_2:0},{requested:true});assert.equal(f.ids.length,12);
 // Anxiety: GAD-2 → GAD-7.
 assert.deepEqual(plain(ctx.syFlow('anxiete',{gad7_1:1,gad7_2:0}).ids),['gad7_1','gad7_2']);
 assert.equal(ctx.syFlow('anxiete',{gad7_1:2,gad7_2:0}).ids.length,9);
 // Trauma: the ITQ only after « oui » to the exposure question, with its own event question first.
 const t=ctx.syFlow('trauma_stress',{exposition_evenement:'oui',deuil_deces:'non',adaptation_evenement:'non'});
 assert.equal(t.ids[3],'itq_quand');assert.ok(t.ids.includes('itq_c9'));assert.equal(t.ids.at(-1),'trauma_gene');assert.ok(t.groups.some(g=>/Cloitre/.test(g.notice)));
 assert.ok(!ctx.syFlow('trauma_stress',{exposition_evenement:'pref'}).ids.includes('itq_quand'));
 // Sensitive module: nothing beyond the entry before a « oui ».
 assert.deepEqual(plain(ctx.syFlow('alcool',{alc_usage:'non'}).ids),['alc_usage']);
});

test('payload: PHQ-15 copies sleep and fatigue from the PHQ-9 of the session; « Je préfère ne pas répondre » is sent as no answer',()=>{
 const {ctx}=environment({handler:()=>({})});
 ctx.SY.page={modules:[],lifetime_answers:{}};
 ctx.SY.sessionAnswers={phq9_3:3,phq9_4:1};
 const w={answers:{somatique_entree:'oui',phq15_a:1,phq15_b:'pref_local',ghost:'x'},requested:false};
 const f=ctx.syFlow('somatique',w.answers,{reuse:ctx.syReuse()});
 assert.ok(!f.ids.includes('phq15_sommeil')&&!f.ids.includes('phq15_fatigue'));
 const p=ctx.syPayloadAnswers('somatique',w);
 assert.deepEqual(JSON.parse(JSON.stringify(p)),{somatique_entree:'oui',phq15_a:1,phq15_sommeil:3,phq15_fatigue:1});
 ctx.SY.sessionAnswers={};const asked=ctx.syFlow('somatique',w.answers,{reuse:ctx.syReuse()});assert.ok(asked.ids.includes('phq15_sommeil'));
 assert.equal(asked.groups.find(g=>g.items.some(i=>i.id==='phq15_sommeil')).head,S.instruments.phq9.stem);
 // A follow-up answer that no longer follows the entry is not sent.
 assert.deepEqual(JSON.parse(JSON.stringify(ctx.syPayloadAnswers('panique',{answers:{panique_crises:'non',agoraphobie_peur:'non',panique_inattendues:'oui'}}))),{panique_crises:'non',agoraphobie_peur:'non'});
 assert.deepEqual(JSON.parse(JSON.stringify(ctx.syVersions('humeur'))),{phq2:S.instruments.phq2.version,phq9:S.instruments.phq9.version});
});

test('page states: unavailable, birth date missing, minor (nothing saved), consent, then the two steps',async()=>{
 const before={available:false,message:'« Mes symptômes » sera bientôt disponible.',content_version:S.version};
 let env=environment({handler:()=>before});await env.ctx.renderSymptoms(env.view,()=>true);
 assert.match(env.view.textContent,/bientôt disponible/);assert.match(env.view.textContent,/Ce n’est pas un diagnostic/);
 const b=backend();
 env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);
 assert.match(env.view.textContent,/réservée aux adultes/);assert.equal(env.view.querySelector('a').attributes.href??env.view.querySelector('a').href,'/mon-compte/#self-profile');
 b.birth('2010-03-01');env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);
 assert.match(env.view.textContent,/Fil Santé Jeunes/);assert.ok(env.view.querySelector('a.sy-resource'));assert.equal(button(env.view,'J’accepte'),undefined);
 assert.equal(env.calls.filter(c=>c.method!=='GET').length,0);
 b.birth('1990-05-04');env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);
 assert.match(env.view.textContent,/Avant de commencer/);await button(env.view,'J’accepte et je commence').fire('click');await flush();
 assert.deepEqual(env.calls.find(c=>c.method==='PUT'),{path:'/api/symptomes/consent',method:'PUT',data:{consent:true}});
 const text=env.view.textContent;for(const s of ['Pour commencer',S.safety.universal_question.text,'Humeur basse et perte d’intérêt','Inquiétude et tension','D’autres thèmes','Choisir des thèmes','Mes résultats','Avec Mon enfance','Mes données'])assert.ok(text.includes(s),s);
 // Sensitive themes: only offered in the chooser, never shown before being chosen.
 assert.ok(!env.view.querySelector('[data-module="alcool"]'));assert.match(text,/Thèmes plus intimes/);
 const box=env.view.querySelectorAll('input').find(i=>i.dataset.syFocus==='choose:alcool');box.checked=true;await box.fire('change');
 assert.ok(env.view.querySelector('[data-module="alcool"]'));assert.match(env.view.querySelector('[data-module="alcool"]').textContent,/données de santé/);
 assert.equal(env.ctx.localStorage.getItem('symptomes-themes:a'),'["alcool"]');
});

test('safety: « oui » to item 9 opens the blocking screen at once, before the autosave, then resources stay visible',async()=>{
 const b=backend();b.birth('2003-05-04');
 const env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);await button(env.view,'J’accepte et je commence').fire('click');await flush();
 const writes=()=>env.calls.filter(c=>c.method==='PUT'&&c.path.startsWith('/api/symptomes/drafts')).length;
 await tick(choice(env.view,'phq9_1','2'));await tick(choice(env.view,'phq9_2','1'));
 assert.ok(env.view.querySelector('[data-q="phq9_9"]'),'PHQ-9 opened');
 const before=writes();await tick(choice(env.view,'phq9_9','1'));
 const dialog=env.doc.modals.at(-1);assert.ok(dialog&&dialog.open,'screen shown');assert.equal(writes(),before,'no wait for the API');
 assert.match(dialog.textContent,/Merci de l’avoir dit/);assert.match(dialog.textContent,/3114/);assert.match(dialog.textContent,/0 800 235 236/,'Fil Santé Jeunes for 18–25');
 assert.doesNotMatch(dialog.textContent,/risque|score/i);
 // No AI key here: Ambre does not follow the safety protocol for this account, its button is hidden.
 assert.equal(env.ctx.SY.page.safety.ambre_available,false);
 assert.deepEqual(plain(dialog.querySelectorAll('button').map(x=>x.textContent)),S.safety.display.blocking_screen.buttons.filter(b=>!/Ambre/.test(b)));
 assert.equal(await dialog.fire('cancel'),true,'Escape does not close it');
 await env.runTimers();assert.ok(writes()>before,'autosave writes the dated alert');
 assert.equal(b.raw.prepare("SELECT count(*) n FROM symptom_safety_alerts WHERE trigger_id='phq9_item9'").get().n,1);
 await button(dialog,'Arrêter').fire('click');
 assert.match(env.view.textContent,/Vous avez arrêté/);assert.ok(env.view.querySelector('#sy-safety-box').querySelector('a.sy-resource'));assert.ok(!env.view.querySelector('[data-module="humeur"]'));
 await button(env.view,'Reprendre le questionnaire').fire('click');assert.ok(env.view.querySelector('[data-module="humeur"]'));assert.ok(env.view.querySelector('#sy-safety-box'));
 // Same answer again: no second screen; violences: quick exit.
 const count=env.doc.modals.length;await tick(choice(env.view,'phq9_8','0'));assert.equal(env.doc.modals.length,count);
 env.ctx.SY.page.safety.ambre_available=true;
 env.ctx.syChosen().relations=true;env.ctx.syRender();await tick(choice(env.view,'rel_violences','oui'));
 const violence=env.doc.modals.at(-1);
 assert.deepEqual(plain(violence.querySelector('.sy-safety-actions').querySelectorAll('button').map(x=>x.textContent)),S.safety.display.blocking_screen.buttons,'Ambre shown when available');assert.match(violence.textContent,/3919/);await button(violence,'Quitter vite cette page').fire('click');assert.equal(env.ctx.location.replaced,'https://www.google.fr/');
});

test('the universal question is asked once on screen for the first step; « pref » shows the soft block only',async()=>{
 const b=backend();b.birth('1980-01-01');const env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);await button(env.view,'J’accepte et je commence').fire('click');await flush();
 await tick(choice(env.view,'securite_actuelle','pref'));
 assert.equal(env.doc.modals.length,0);assert.match(env.view.textContent,new RegExp(S.safety.display.soft_block.text.slice(0,40)));
 await env.runTimers();
 const drafts=b.raw.prepare('SELECT module,answers FROM symptom_drafts ORDER BY module').all();
 // One question on screen, recorded by one module only (no duplicate alert).
 assert.deepEqual(drafts.map(d=>[d.module,JSON.parse(d.answers).securite_actuelle]),[['anxiete',undefined],['humeur','pref']]);
 await tick(choice(env.view,'securite_actuelle','oui'));assert.equal(env.doc.modals.length,1);await env.runTimers();
 assert.equal(b.raw.prepare("SELECT count(*) n FROM symptom_safety_alerts WHERE trigger_id='securite_oui'").get().n,1);
});

test('recording a theme: result « au-dessus / en dessous du repère publié », paliers, fixed sentences, delay, delete; late autosave is dropped',async()=>{
 const b=backend();b.birth('1980-01-01');const env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);await button(env.view,'J’accepte et je commence').fire('click');await flush();
 for(const [q,v] of [['phq9_1',2],['phq9_2',2],['phq9_3',2],['phq9_4',1],['phq9_5',1],['phq9_6',1],['phq9_7',1],['phq9_8',0],['phq9_9',0],['humeur_duree','non'],['humeur_perinatal','non'],['humeur_gene','nettement']])await tick(choice(env.view.querySelector('[data-module="humeur"]'),q,String(v)));
 assert.match(env.view.querySelector('[data-module="humeur"]').querySelector('.sy-module-sub').textContent,/^12 \/ 12$/);
 await button(env.view.querySelector('[data-module="humeur"]'),'Enregistrer ce thème').fire('click');await flush();
 const post=env.calls.find(c=>c.method==='POST');assert.equal(post.data.module,'humeur');assert.equal(post.data.answers.phq9_1,2);assert.equal(post.data.answers.securite_actuelle,undefined);
 const card=env.view.querySelector('#sy-result-humeur');assert.ok(card);
 assert.match(card.textContent,/Au-dessus du repère publié par PHQ-9 · palier « modéré »/);
 assert.match(card.textContent,/score 10 sur 27/);assert.ok(card.textContent.includes(S.fixed_texts.positive_note));assert.ok(card.textContent.includes(S.fixed_texts.what_next.replace(', en parler avec Ambre','')),'no Ambre without its safety protocol');assert.ok(card.textContent.includes(S.fixed_texts.gene_nettement));
 assert.doesNotMatch(card.textContent,/vous avez un trouble|diagnostic probable|dépression/i);
 // One card, one conclusion: no secondary negative text, no negative fixed sentence, reference in small print.
 assert.ok(!card.textContent.includes('humeur basse durable'));assert.ok(!card.textContent.includes(S.fixed_texts.negative_note));
 assert.match(card.textContent,/\(score 10 sur 27\)\./);assert.doesNotMatch(card.querySelectorAll('p').filter(p=>!p.className).map(p=>p.textContent).join(' '),/Kroenke/);
 assert.match(card.querySelector('.sy-result-ref').textContent,/Kroenke et al\. 2001/);
 const order=['sy-result-state','sy-next','sy-result-notes','sy-result-ref','sy-result-actions'].map(c=>card.children.findIndex(n=>n.className===c));
 assert.deepEqual(order,[...order].sort((x,y)=>x-y));
 assert.match(env.view.querySelector('[data-module="humeur"]').textContent,/Vous pourrez le refaire à partir du date:/);
 // Late autosave of the recorded passation: finished, the draft is not resumed.
 const late=await b.handler('/api/symptomes/drafts/humeur','PUT',{passation_key:post.data.passation_key,content_version:S.version,instrument_versions:post.data.instrument_versions,answers:{phq9_1:2}});
 assert.equal(late.finished,true);assert.equal(late.draft,null);
 env.ctx.SY.work.humeur={key:post.data.passation_key,started_at:null,answers:{phq9_1:2},prefilled:{},requested:false,dirty:1,saving:false,status:''};
 await env.ctx.sySaveDraft('humeur');assert.equal(env.ctx.SY.work.humeur,undefined);
 // Below the reference for anxiety (GAD-2 negative, GAD-7 not asked).
 await tick(choice(env.view.querySelector('[data-module="anxiete"]'),'gad7_1','0'));await tick(choice(env.view.querySelector('[data-module="anxiete"]'),'gad7_2','1'));
 await button(env.view.querySelector('[data-module="anxiete"]'),'Enregistrer ce thème').fire('click');await flush();
 const below=env.view.querySelector('#sy-result-anxiete');assert.match(below.textContent,/En dessous du seuil d’entrée \(GAD-2\)/);assert.ok(below.textContent.includes(S.fixed_texts.negative_note));assert.ok(!below.textContent.includes(S.fixed_texts.what_next));
 await button(below,'Supprimer ce résultat').fire('click');await flush();
 assert.equal(env.view.querySelector('#sy-result-anxiete'),null);assert.ok(env.calls.some(c=>c.method==='DELETE'&&c.path.startsWith('/api/symptomes/checks/')));
 // Redo early: allowed, the overlap is announced.
 await button(env.view.querySelector('[data-module="humeur"]'),'Refaire ce thème').fire('click');assert.match(env.view.querySelector('[data-module="humeur"]').textContent,/chevaucheront/);
});

test('results of a draft resume, delete all and withdraw consent',async()=>{
 const b=backend();b.birth('1980-01-01');await b.handler('/api/symptomes/consent','PUT',{consent:true});
 await b.handler('/api/symptomes/drafts/humeur','PUT',{passation_key:'humeur-draft-0001',content_version:S.version,instrument_versions:{phq9:S.instruments.phq9.version},answers:{phq9_1:1,securite_actuelle:'non'}});
 const env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);
 assert.equal(env.ctx.SY.work.humeur.key,'humeur-draft-0001');assert.match(env.view.querySelector('[data-module="humeur"]').querySelector('.sy-module-sub').textContent,/^1 \/ 2$/);
 assert.equal(choice(env.view,'phq9_1','1').checked,true);
 await button(env.view,'Supprimer tous mes résultats').fire('click');await flush();
 assert.equal(b.raw.prepare('SELECT count(*) n FROM symptom_drafts').get().n,0);
 await button(env.view,'Retirer mon accord').fire('click');await flush();assert.match(env.view.textContent,/Avant de commencer/);
});

test('crossing with Mon enfance: three lists with the display labels, the fixed sentence and links; nothing counted',()=>{
 const env=environment({handler:()=>({})}),ctx=env.ctx;
 ctx.SY.view=env.view;ctx.SY.current=()=>true;
 ctx.SY.page={available:true,consent:{given:true},age:{status:'adulte',age:40},modules:S.modules.map(m=>({id:m.id,order:m.order,state:'non_explore',last:null,retest:{},draft:null})),lifetime_answers:{},safety:{banner:null},
  crossing:{childhood_available:true,lists:{deux:[{id:'dep_generale',label:'Humeur basse ou perte d’intérêt',modules:['humeur']}],symptomes:[{id:'anx_generale',label:'Anxiété',modules:['anxiete']}],enfance:[{id:'att_anxieux',label:'Peur de perdre ses proches',modules:[]}]},subjects:[{id:'icd_6b01',state:'non_explore',childhood:'present'},{id:'att_anxieux',state:'non_signale',childhood:'present'}],suspended:[],show_counts:false}};
 ctx.syRender();const cross=env.view.querySelector('.sy-cross'),text=cross.textContent;
 assert.ok(text.indexOf(S.fixed_texts.crossing_title)<text.indexOf(S.fixed_texts.crossing_symptoms_only));
 assert.ok(text.includes('Beaucoup de personnes ayant ces vécus n’ont pas ces difficultés, et inversement ; ce rapprochement ne montre pas une cause.'));
 assert.ok(text.includes('Humeur basse ou perte d’intérêt'));assert.ok(!text.includes('Dépression'));
 const fold=cross.querySelector('details');assert.equal(fold.open,false);assert.ok(fold.textContent.includes('Peur de perdre ses proches'));
 assert.equal(cross.querySelector('a.sy-cross-link').href,'/personnages/louise/');
 assert.match(text,/pas encore explorés.*Crises de peur soudaines/);assert.doesNotMatch(text,/\d+ (vécus|sujets)/);
 ctx.SY.page.crossing={childhood_available:false,lists:{}};ctx.syRender();assert.match(env.view.querySelector('.sy-cross').textContent,/Quand vous aurez rempli Mon enfance/);
});

test('Paramètres: « Mes symptômes » line shows verbatim the text Ambre receives, never the answers',async()=>{
 const doc={createElement:tag=>new Node(tag,doc)};doc.body=new Node('body',doc);
 const ctx={document:doc,JSON,String,el(tag,cls,text){const n=doc.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n},actionButton(){},careDate:v=>'le '+v};
 vm.createContext(ctx);vm.runInContext(SETTINGS_INTERFACE_SCRIPT,ctx);
 const block='« Mes symptômes » : questionnaires de repérage.\nThèmes au-dessus du repère :\n- Humeur basse et perte d’intérêt : au-dessus du repère publié ; résultat du 5 novembre 2026.';
 const body=ctx.settingsSymptomsSummary({available:true,consent:{given:true},modules:[{id:'humeur',last:{state:'au_dessus',completed_at:5,answers:{phq9_1:3}}}],ambre:{block}});
 assert.ok(body.textContent.includes(block.split('\n').join('')),'the exact text, line by line');assert.match(body.textContent,/Jamais vos réponses/);assert.doesNotMatch(body.textContent,/phq9_1/);
 assert.match(ctx.settingsSymptomsSummary({available:true,consent:{given:true},modules:[],ambre:{block:null}}).textContent,/Ambre ne reçoit rien/);
 assert.match(ctx.settingsSymptomsSummary({available:true,consent:{given:false}}).textContent,/Ambre ne reçoit rien/);
 assert.match(SETTINGS_INTERFACE_SCRIPT,/elsewhere\.append\(settingsSymptoms\(current\)\)/);
});

test('« Je préfère ne pas répondre » to an official item: quiet link, sent as no answer, kept on this device after a reload',async()=>{
 const b=backend();b.birth('1980-01-01');await b.handler('/api/symptomes/consent','PUT',{consent:true});
 let env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);
 const pref=choice(env.view,'phq9_1','pref');assert.equal(pref.type,'radio');assert.match(pref.parent.className,/sy-pref/);assert.equal(pref.parent.parent.className,'sy-q','outside the list of answers, in the same group');assert.equal(pref.parent.parent.tagName,'FIELDSET');
 await tick(pref);await tick(choice(env.view,'phq9_2','1'));await env.runTimers();
 const draft=JSON.parse(b.raw.prepare("SELECT answers FROM symptom_drafts WHERE module='humeur'").get().answers);assert.equal(draft.phq9_1,undefined);assert.equal(draft.phq9_2,1);
 const storage=env.ctx.localStorage;env=environment({handler:b.handler});env.ctx.localStorage=storage;await env.ctx.renderSymptoms(env.view,()=>true);
 assert.equal(choice(env.view,'phq9_1','pref').checked,true);assert.equal(choice(env.view,'phq9_2','1').checked,true);
});

const adultEnv=async()=>{const b=backend();b.birth('1980-01-01');await b.handler('/api/symptomes/consent','PUT',{consent:true});const env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);return {b,env}};

test('relecture: an account change or a sign-out empties the page memory and its device keys',async()=>{
 const {env}=await adultEnv(),ctx=env.ctx;
 await tick(choice(env.view,'securite_actuelle','oui'));assert.equal(env.doc.modals.length,1);
 ctx.SY.overlap.humeur=true;ctx.SY.saved.humeur=true;ctx.SY.open.humeur=true;ctx.SY.redoEarly.humeur=true;
 env.storage.set('symptomes-themes:a','["alcool"]');env.storage.set('symptomes-pref:a:k1','["phq9_1"]');env.storage.set('symptomes-themes:b','["soi"]');env.storage.set('autre','x');
 ctx.syLeave('a');
 for(const key of ['acknowledged','overlap','saved','open','redoEarly','work','sessionAnswers'])assert.deepEqual(Object.keys(ctx.SY[key]),[],key);
 assert.equal(ctx.SY.sessionScreen,null);assert.equal(env.doc.getElementById('sy-safety-dialog'),null);
 assert.deepEqual([...env.storage.keys()].sort(),['autre','symptomes-themes:b']);
 // renderSymptoms for another account resets too.
 ctx.SY.acknowledged['x:securite_oui']=true;ctx.state.me.user.id='b';await ctx.renderSymptoms(env.view,()=>true);assert.deepEqual(Object.keys(ctx.SY.acknowledged),[]);
 assert.match(SYMPTOMES_INTERFACE_SCRIPT,/function syResetSession\(\)/);
 assert.match(HTML,/state\.events=\[\];\n syResetSession\(\);\n\}/,'clearSessionState');
 assert.equal((HTML.match(/syLeave\(leaving\)/g)||[]).length,2,'logout and account deletion');
});

test('relecture: after recording the first step, a new « oui » opens the blocking screen again',async()=>{
 const {env}=await adultEnv();
 await tick(choice(env.view,'securite_actuelle','oui'));assert.equal(env.doc.modals.length,1);
 const dialog=env.doc.modals[0];await button(dialog,'Continuer').fire('click');
 for(const id of ['humeur','anxiete']){const box=env.view.querySelector('[data-module="'+id+'"]');await tick(choice(box,id==='humeur'?'phq9_1':'gad7_1','0'));await button(env.view.querySelector('[data-module="'+id+'"]'),'Enregistrer ce thème').fire('click');await flush()}
 await button(env.view.querySelector('[data-module="humeur"]'),'Refaire ce thème').fire('click');
 await tick(choice(env.view,'securite_actuelle','oui'));assert.equal(env.doc.modals.length,2,'screen shown again');
});

test('relecture: a second Escape that closes the screen shows the box of resources at once',async()=>{
 const {env}=await adultEnv();
 await tick(choice(env.view,'securite_actuelle','oui'));const dialog=env.doc.getElementById('sy-safety-dialog');
 assert.equal(env.view.querySelector('#sy-safety-box'),null);
 dialog.close();await flush();
 assert.equal(env.doc.getElementById('sy-safety-dialog'),null);assert.ok(env.view.querySelector('#sy-safety-box'));
});

test('relecture: device keys purged at « Supprimer tous mes résultats », « Retirer mon accord » and for orphan passations',async()=>{
 const {b,env}=await adultEnv();
 env.storage.set('symptomes-pref:a:ancienne-passation','["phq9_1"]');
 await tick(choice(env.view,'phq9_1','pref'));await env.runTimers();
 const key=[...env.storage.keys()].find(k=>k.startsWith('symptomes-pref:a:humeur'));assert.ok(key);
 const reload=environment({handler:b.handler});for(const [k,v] of env.storage)reload.storage.set(k,v);await reload.ctx.renderSymptoms(reload.view,()=>true);
 assert.ok(reload.storage.has(key));assert.ok(!reload.storage.has('symptomes-pref:a:ancienne-passation'),'orphan dropped');
 reload.storage.set('symptomes-themes:a','["soi"]');
 await button(reload.view,'Supprimer tous mes résultats').fire('click');await flush();
 assert.deepEqual([...reload.storage.keys()].filter(k=>k.startsWith('symptomes-')),[]);
 reload.storage.set('symptomes-themes:a','["soi"]');await button(reload.view,'Retirer mon accord').fire('click');await flush();
 assert.deepEqual([...reload.storage.keys()].filter(k=>k.startsWith('symptomes-')),[]);
});

test('relecture: an unconfirmed blocking answer is retried with a growing delay; the box shows offline and without consent',async()=>{
 const b=backend();b.birth('1980-01-01');await b.handler('/api/symptomes/consent','PUT',{consent:true});
 let failing=3;const env=environment({handler:async(path,method,data)=>{if(method==='PUT'&&path==='/api/symptomes/drafts/humeur'&&failing>0){failing--;throw Object.assign(new Error('Réseau indisponible'),{status:0})}return b.handler(path,method,data)}});
 await env.ctx.renderSymptoms(env.view,()=>true);
 await tick(choice(env.view,'securite_actuelle','oui'));env.delays.length=0;
 for(let i=0;i<12&&env.timers.length;i++){const fn=env.timers.shift();await fn();await flush()}
 assert.deepEqual(env.delays.filter(ms=>ms>=1000),[2000,4000,8000]);
 assert.equal(b.raw.prepare("SELECT count(*) n FROM symptom_safety_alerts WHERE trigger_id='securite_oui'").get().n,1,'recorded on the fourth try');
 // Offline afterwards: the resources of this session stay on screen.
 const offline=env.ctx;offline.api=async()=>{throw new Error('Hors ligne.')};await offline.renderSymptoms(env.view,()=>true);
 assert.ok(env.view.querySelector('#sy-safety-box'));assert.match(env.view.textContent,/Hors ligne/);
 // Without consent, the 14-day banner stays above the consent card.
 await b.handler('/api/symptomes/consent','PUT',{consent:false});const again=environment({handler:b.handler});await again.ctx.renderSymptoms(again.view,()=>true);
 assert.ok(again.view.querySelector('#sy-safety-box'));assert.match(again.view.textContent,/Avant de commencer/);
});

test('relecture: « Ne pas faire ce thème » keeps one key across a retry and never posts twice at once',async()=>{
 const b=backend();b.birth('1980-01-01');await b.handler('/api/symptomes/consent','PUT',{consent:true});
 let fail=true;const env=environment({handler:async(path,method,data)=>{if(method==='POST'&&fail){fail=false;throw Object.assign(new Error('Erreur temporaire'),{status:503})}return b.handler(path,method,data)}});
 await env.ctx.renderSymptoms(env.view,()=>true);env.ctx.syChosen().soi=true;env.ctx.syRender();
 await button(env.view.querySelector('[data-module="soi"]'),'Ne pas faire ce thème').fire('click');await flush();
 const first=env.calls.filter(c=>c.method==='POST')[0].data.passation_key;
 const second=env.ctx.syComplete('soi',true),third=env.ctx.syComplete('soi',true);await second;await third;
 const posts=env.calls.filter(c=>c.method==='POST');assert.equal(posts.length,2,'the concurrent call is dropped');assert.equal(posts[1].data.passation_key,first);
 assert.equal(b.raw.prepare("SELECT count(*) n FROM symptom_checks WHERE module='soi'").get().n,1);
});

test('relecture: answers are real radio buttons in a fieldset with its legend',async()=>{
 const {env}=await adultEnv();const row=env.view.querySelector('[data-q="phq9_1"]');
 assert.equal(row.tagName,'FIELDSET');assert.equal(row.children[0].tagName,'LEGEND');
 const radios=row.querySelectorAll('input');assert.ok(radios.length>=5&&radios.every(r=>r.type==='radio'&&r.name===radios[0].name));
 assert.equal(env.doc.getElementById('sy-live')?.getAttribute('aria-live')??'polite','polite');
 assert.doesNotMatch(SYMPTOMES_INTERFACE_SCRIPT,/status\.setAttribute\('aria-live'/);
});

test('recette: « Et maintenant ? » proposes Ambre only when Ambre follows the safety protocol',()=>{
 const env=environment({handler:()=>({})}),ctx=env.ctx;ctx.SY.view=env.view;ctx.SY.current=()=>true;
 const text=S.fixed_texts.what_next;assert.match(text,/, en parler avec Ambre,/,'published sentence unchanged');
 ctx.SY.page={safety:{ambre_available:false}};
 const without=ctx.syWhatNext(text);
 assert.doesNotMatch(without,/Ambre/);assert.doesNotMatch(without,/,\s*,|\s,|\s{2}|,\./);
 assert.equal(without,text.replace(', en parler avec Ambre',''));assert.match(without,/\(CMP, gratuit\), ou refaire ce module plus tard\.$/);
 assert.equal(ctx.syWhatNext('Vous pouvez en parler à votre médecin, ou en parler avec Ambre.'),'Vous pouvez en parler à votre médecin.');
 ctx.SY.page.safety.ambre_available=true;assert.equal(ctx.syWhatNext(text),text);
 ctx.SY.page=null;assert.doesNotMatch(ctx.syWhatNext(text),/Ambre/,'unknown availability: not proposed');
});

test('recette: the result card « Et maintenant ? » follows ambre_available',async()=>{
 const {env}=await adultEnv(),ctx=env.ctx;assert.equal(ctx.SY.page.safety.ambre_available,false);
 for(const [q,v] of [['phq9_1',2],['phq9_2',2],['phq9_3',2],['phq9_4',1],['phq9_5',1],['phq9_6',1],['phq9_7',1],['phq9_8',0],['phq9_9',0],['humeur_duree','non'],['humeur_perinatal','non'],['humeur_gene','nettement']])await tick(choice(env.view.querySelector('[data-module="humeur"]'),q,String(v)));
 await button(env.view.querySelector('[data-module="humeur"]'),'Enregistrer ce thème').fire('click');await flush();
 const next=()=>env.view.querySelector('#sy-result-humeur').querySelector('.sy-next').textContent;
 assert.match(next(),/^Et maintenant \?/);assert.doesNotMatch(next(),/Ambre/);
 ctx.SY.page.safety.ambre_available=true;ctx.syRender();assert.equal(next(),S.fixed_texts.what_next);
});

test('recette: a confirmed blocking screen stays confirmed after a reload, in this tab, for its passation only',async()=>{
 const {b,env}=await adultEnv();
 await tick(choice(env.view,'securite_actuelle','oui'));assert.equal(env.doc.modals.length,1);
 assert.deepEqual([...env.session.keys()],[],'not kept before it is confirmed');
 await button(env.doc.modals[0],'Continuer').fire('click');
 const keys=[...env.session.keys()];assert.ok(keys.length>=1&&keys.every(k=>/^symptomes-ack:a:/.test(k)));assert.ok(keys.every(k=>JSON.parse(env.session.get(k)).includes('securite_oui')));
 await env.runTimers();
 // Reload: a new page, same tab (sessionStorage kept), the draft is resumed; the next answer does not reopen the screen.
 const reload=environment({handler:b.handler});for(const [k,v] of env.session)reload.session.set(k,v);await reload.ctx.renderSymptoms(reload.view,()=>true);
 await tick(choice(reload.view,'phq9_1','1'));assert.equal(reload.doc.modals.length,0,'no blocking screen at the next answer');
 // Another tab (no sessionStorage): the screen opens again, as before.
 const other=environment({handler:b.handler});await other.ctx.renderSymptoms(other.view,()=>true);await tick(choice(other.view,'phq9_1','1'));assert.equal(other.doc.modals.length,1);
 // Unavailable storage: the page still works (memory only).
 const broken=environment({handler:b.handler});broken.ctx.sessionStorage={get length(){throw new Error('blocked')},getItem(){throw new Error('blocked')},setItem(){throw new Error('blocked')},removeItem(){},key(){return null}};
 await broken.ctx.renderSymptoms(broken.view,()=>true);await tick(choice(broken.view,'phq9_1','1'));assert.equal(broken.doc.modals.length,1);await button(broken.doc.modals[0],'Continuer').fire('click');
});

test('recette: confirmations purged at recording, « Refaire », sign-out, delete all and withdrawn consent',async()=>{
 const {b,env}=await adultEnv();
 await tick(choice(env.view,'securite_actuelle','oui'));await button(env.doc.modals[0],'Continuer').fire('click');
 assert.ok(env.session.size>=1);
 for(const id of ['humeur','anxiete']){await tick(choice(env.view.querySelector('[data-module="'+id+'"]'),id==='humeur'?'phq9_1':'gad7_1','0'));await button(env.view.querySelector('[data-module="'+id+'"]'),'Enregistrer ce thème').fire('click');await flush()}
 assert.deepEqual([...env.session.keys()],[],'recorded passations leave no key');
 // « Refaire » then « oui »: the screen opens again, even with the old keys of another passation around.
 env.session.set('symptomes-ack:a:passation-ancienne','["securite_oui"]');
 await button(env.view.querySelector('[data-module="humeur"]'),'Refaire ce thème').fire('click');
 await tick(choice(env.view,'securite_actuelle','oui'));assert.equal(env.doc.modals.length,2,'screen shown again');
 await button(env.doc.modals[1],'Continuer').fire('click');assert.ok([...env.session.keys()].some(k=>k!=='symptomes-ack:a:passation-ancienne'));
 // Reload drops orphan keys.
 await env.runTimers();const reload=environment({handler:b.handler});for(const [k,v] of env.session)reload.session.set(k,v);await reload.ctx.renderSymptoms(reload.view,()=>true);
 assert.ok(!reload.session.has('symptomes-ack:a:passation-ancienne'));assert.ok(reload.session.size>=1);
 // Delete all, withdraw consent, sign-out: nothing left (other accounts untouched).
 const keep='symptomes-ack:b:x';reload.session.set(keep,'["securite_oui"]');
 await button(reload.view,'Supprimer tous mes résultats').fire('click');await flush();assert.deepEqual([...reload.session.keys()],[keep]);
 reload.session.set('symptomes-ack:a:y','["securite_oui"]');await button(reload.view,'Retirer mon accord').fire('click');await flush();assert.deepEqual([...reload.session.keys()],[keep]);
 reload.session.set('symptomes-ack:a:z','["securite_oui"]');reload.ctx.syLeave('a');assert.deepEqual([...reload.session.keys()],[keep]);
 // An answer that no longer meets the trigger re-arms it: its key goes too.
 const {env:e2}=await adultEnv();await tick(choice(e2.view,'securite_actuelle','oui'));await button(e2.doc.modals[0],'Continuer').fire('click');assert.ok(e2.session.size>=1);
 await tick(choice(e2.view,'securite_actuelle','non'));assert.equal(e2.session.size,0);
});

test('recette: Paramètres shows French labels for people, links and sources, and the gender as chosen',()=>{
 const doc={createElement:tag=>new Node(tag,doc)};doc.body=new Node('body',doc);
 const ctx={document:doc,JSON,String,Set,el(tag,cls,text){const n=doc.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n},actionButton(){},careDate:v=>'le '+v};
 vm.createContext(ctx);vm.runInContext(SETTINGS_INTERFACE_SCRIPT.slice(0,SETTINGS_INTERFACE_SCRIPT.indexOf('async function renderSettings(')),ctx);
 ctx.settingsDisclosure=(label,body)=>{const d=ctx.el('details');d.append(ctx.el('summary','',label));if(body)d.append(body);return d};
 const shown=ctx.settingsValue({people_ids:['p1'],people:['Camille'],links:[{label:'Lien'}],sources:[{label:'Source'}],gender:'unspecified'}).textContent;
 for(const label of ['Proches liés','Proches','Liens','Sources'])assert.ok(shown.includes(label),label);
 assert.doesNotMatch(shown,/people ids|\bpeople\b|\blinks\b|\bsources\b|—/);
 assert.match(shown,/GenreJe préfère ne pas le préciser/);
 // Every gender reads as the choice offered in « Mon compte ».
 const choices=Object.fromEntries(JSON.parse(HTML.match(/var genderChoices=(\[\[.*?\]\]);/)[1].replace(/'/g,'"')).filter(([v])=>v));
 for(const [value,label] of Object.entries(choices))assert.equal(ctx.settingsValue(value,0,'gender',true).textContent,label,value);
});
