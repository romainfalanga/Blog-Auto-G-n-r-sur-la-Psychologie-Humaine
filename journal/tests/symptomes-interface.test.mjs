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
import {HTML} from '../src/ui.mjs';

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
const choice=(host,q,value)=>host.querySelector('[data-sy-focus="'+q+':'+value+'"]')||host.querySelectorAll('button').find(b=>b.dataset.syFocus&&b.dataset.syFocus.endsWith(':'+q+':'+value));
const plain=v=>JSON.parse(JSON.stringify(v));
const flush=()=>new Promise(r=>setTimeout(r,0));

function environment({handler}={}){
 const doc={modals:[],activeElement:null,createElement(tag){return new Node(tag,this)},getElementById(id){return this.body.querySelector('#'+id)||this.head.querySelector('#'+id)}};
 doc.body=new Node('body',doc);doc.head=new Node('head',doc);
 const calls=[],storage=new Map(),timers=[];
 const ctx={document:doc,console,JSON,Math,Date,Promise,Object,Array,String,Number,Boolean,Error,Set,Map,WeakMap,RegExp,Uint8Array,crypto:globalThis.crypto,location:{replaced:null,replace(u){this.replaced=u},assign(){}},
  localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},
  setTimeout:(fn,ms)=>{timers.push(fn);return timers.length},clearTimeout:()=>{},
  state:{me:{user:{id:'a'}}},dateLabel:v=>'date:'+v,
  el(tag,cls,text){const n=doc.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n},
  actionButton(text,fn,cls){const b=ctx.el('button',cls||'',text);b.type='button';b.addEventListener('click',fn);return b},
  confirmDialog:async()=>true,
  api:async(path,method='GET',data)=>{calls.push({path,method,data:data===undefined?undefined:JSON.parse(JSON.stringify(data))});return handler(path,method,data)}};
 ctx.window=ctx;vm.createContext(ctx);
 vm.runInContext(symptomsClientScript(),ctx);vm.runInContext(SYMPTOMES_INTERFACE_SCRIPT,ctx);
 const view=doc.createElement('div');doc.body.append(view);
 return {ctx,doc,view,calls,timers,async runTimers(){while(timers.length){await timers.shift()();await flush()}}};
}

// ---- Real backend for the page states ----------------------------------------------------------
function backend(){
 const s=setup({until:'9999',users:['a']});let clock=Date.UTC(2026,10,19,12)/1000,serial=0;
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
 await choice(env.view,'phq9_1','2').fire('click');await choice(env.view,'phq9_2','1').fire('click');
 assert.ok(env.view.querySelector('[data-q="phq9_9"]'),'PHQ-9 opened');
 const before=writes();await choice(env.view,'phq9_9','1').fire('click');
 const dialog=env.doc.modals.at(-1);assert.ok(dialog&&dialog.open,'screen shown');assert.equal(writes(),before,'no wait for the API');
 assert.match(dialog.textContent,/Merci de l’avoir dit/);assert.match(dialog.textContent,/3114/);assert.match(dialog.textContent,/0 800 235 236/,'Fil Santé Jeunes for 18–25');
 assert.doesNotMatch(dialog.textContent,/risque|score/i);
 assert.deepEqual(plain(dialog.querySelectorAll('button').map(x=>x.textContent)),S.safety.display.blocking_screen.buttons);
 assert.equal(await dialog.fire('cancel'),true,'Escape does not close it');
 await env.runTimers();assert.ok(writes()>before,'autosave writes the dated alert');
 assert.equal(b.raw.prepare("SELECT count(*) n FROM symptom_safety_alerts WHERE trigger_id='phq9_item9'").get().n,1);
 await button(dialog,'Arrêter').fire('click');
 assert.match(env.view.textContent,/Vous avez arrêté/);assert.ok(env.view.querySelector('#sy-safety-box').querySelector('a.sy-resource'));assert.ok(!env.view.querySelector('[data-module="humeur"]'));
 await button(env.view,'Reprendre le questionnaire').fire('click');assert.ok(env.view.querySelector('[data-module="humeur"]'));assert.ok(env.view.querySelector('#sy-safety-box'));
 // Same answer again: no second screen; violences: quick exit.
 const count=env.doc.modals.length;await choice(env.view,'phq9_8','0').fire('click');assert.equal(env.doc.modals.length,count);
 env.ctx.syChosen().relations=true;env.ctx.syRender();await choice(env.view,'rel_violences','oui').fire('click');
 const violence=env.doc.modals.at(-1);assert.match(violence.textContent,/3919/);await button(violence,'Quitter vite cette page').fire('click');assert.equal(env.ctx.location.replaced,'https://www.google.fr/');
});

test('the universal question is asked once on screen for the first step; « pref » shows the soft block only',async()=>{
 const b=backend();b.birth('1980-01-01');const env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);await button(env.view,'J’accepte et je commence').fire('click');await flush();
 await choice(env.view,'securite_actuelle','pref').fire('click');
 assert.equal(env.doc.modals.length,0);assert.match(env.view.textContent,new RegExp(S.safety.display.soft_block.text.slice(0,40)));
 await env.runTimers();
 const drafts=b.raw.prepare('SELECT module,answers FROM symptom_drafts ORDER BY module').all();
 assert.deepEqual(drafts.map(d=>[d.module,JSON.parse(d.answers).securite_actuelle]),[['anxiete','pref'],['humeur','pref']]);
});

test('recording a theme: result « au-dessus / en dessous du repère publié », paliers, fixed sentences, delay, delete; late autosave is dropped',async()=>{
 const b=backend();b.birth('1980-01-01');const env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);await button(env.view,'J’accepte et je commence').fire('click');await flush();
 for(const [q,v] of [['phq9_1',2],['phq9_2',2],['phq9_3',2],['phq9_4',1],['phq9_5',1],['phq9_6',1],['phq9_7',1],['phq9_8',0],['phq9_9',0],['humeur_duree','non'],['humeur_perinatal','non'],['humeur_gene','nettement']])await choice(env.view.querySelector('[data-module="humeur"]'),q,String(v)).fire('click');
 assert.match(env.view.querySelector('[data-module="humeur"]').querySelector('.sy-module-sub').textContent,/^12 \/ 12$/);
 await button(env.view.querySelector('[data-module="humeur"]'),'Enregistrer ce thème').fire('click');await flush();
 const post=env.calls.find(c=>c.method==='POST');assert.equal(post.data.module,'humeur');assert.equal(post.data.answers.phq9_1,2);assert.equal(post.data.answers.securite_actuelle,undefined);
 const card=env.view.querySelector('#sy-result-humeur');assert.ok(card);
 assert.match(card.textContent,/Au-dessus du repère publié par PHQ-9 · palier « modéré »/);
 assert.match(card.textContent,/score 10 sur 27/);assert.ok(card.textContent.includes(S.fixed_texts.positive_note));assert.ok(card.textContent.includes(S.fixed_texts.what_next));assert.ok(card.textContent.includes(S.fixed_texts.gene_nettement));
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
 await choice(env.view.querySelector('[data-module="anxiete"]'),'gad7_1','0').fire('click');await choice(env.view.querySelector('[data-module="anxiete"]'),'gad7_2','1').fire('click');
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
 assert.equal(choice(env.view,'phq9_1','1').getAttribute('aria-pressed'),'true');
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

test('Paramètres: « Mes symptômes » line says what Ambre will receive, never the answers',async()=>{
 const doc={createElement:tag=>new Node(tag,doc)};doc.body=new Node('body',doc);
 const ctx={document:doc,JSON,el(tag,cls,text){const n=doc.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n},actionButton(){},careDate:v=>'le '+v};
 vm.createContext(ctx);vm.runInContext(SETTINGS_INTERFACE_SCRIPT,ctx);
 const body=ctx.settingsSymptomsSummary({available:true,consent:{given:true},modules:[{id:'humeur',last:{state:'au_dessus',completed_at:5,answers:{phq9_1:3}}},{id:'anxiete',last:{state:'en_dessous',completed_at:6}}]});
 assert.match(body.textContent,/Humeur basse et perte d’intérêt · au-dessus du repère publié · le 5/);assert.doesNotMatch(body.textContent,/Inquiétude/);assert.match(body.textContent,/Jamais vos réponses/);
 assert.match(ctx.settingsSymptomsSummary({available:true,consent:{given:false}}).textContent,/Ambre ne recevra rien/);
 assert.match(SETTINGS_INTERFACE_SCRIPT,/elsewhere\.append\(settingsSymptoms\(current\)\)/);
});

test('« Je préfère ne pas répondre » to an official item: quiet link, sent as no answer, kept on this device after a reload',async()=>{
 const b=backend();b.birth('1980-01-01');await b.handler('/api/symptomes/consent','PUT',{consent:true});
 let env=environment({handler:b.handler});await env.ctx.renderSymptoms(env.view,()=>true);
 const pref=choice(env.view,'phq9_1','pref');assert.match(pref.className,/sy-pref/);assert.equal(pref.parent.className,'sy-q','outside the list of answers');
 await pref.fire('click');await choice(env.view,'phq9_2','1').fire('click');await env.runTimers();
 const draft=JSON.parse(b.raw.prepare("SELECT answers FROM symptom_drafts WHERE module='humeur'").get().answers);assert.equal(draft.phq9_1,undefined);assert.equal(draft.phq9_2,1);
 const storage=env.ctx.localStorage;env=environment({handler:b.handler});env.ctx.localStorage=storage;await env.ctx.renderSymptoms(env.view,()=>true);
 assert.equal(choice(env.view,'phq9_1','pref').getAttribute('aria-pressed'),'true');assert.equal(choice(env.view,'phq9_2','1').getAttribute('aria-pressed'),'true');
});
