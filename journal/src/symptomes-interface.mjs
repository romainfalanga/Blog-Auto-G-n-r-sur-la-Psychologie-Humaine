// « Mes symptômes » (plan v1.1, lot B) : the private page /mes-symptomes/.
// The page runs the engine the Worker uses (symptomes-engine.mjs, copied by
// scripts/build_symptomes_page_engine.mjs and served at /mes-symptomes/moteur.js with the published
// content) to show the safety screen as soon as an answer calls for it, without waiting for the API,
// and to decide which questions follow. The server recomputes every score and state; the page only
// displays the results it returns. Contract: docs/symptomes-contrat-2026-11-19.md, § 10.
import {SYMPTOMES} from './symptomes-data.generated.mjs';
import {CARTOGRAPHIE} from './cartographie.generated.mjs';
import {SYMPTOMES_ENGINE_SOURCE,SYMPTOMES_ENGINE_EXPORTS} from './symptomes-engine-source.generated.mjs';

export const SYMPTOMES_ENGINE_PATH='/mes-symptomes/moteur.js';
let CLIENT=null;
// Engine + content + links to the public pages of the subjects of the crossing (Mon enfance urls).
export function symptomsClientScript(){
 if(CLIENT)return CLIENT;
 const subjects=SYMPTOMES.subject_modules||{};
 const links=Object.fromEntries(CARTOGRAPHIE.outcomes.filter(o=>Object.hasOwn(subjects,o.id)&&typeof o.url==='string'&&/^\/(?!\/)/.test(o.url)).map(o=>[o.id,o.url]));
 const json=value=>JSON.stringify(value).replace(/[\u2028\u2029]/g,c=>'\\u'+c.charCodeAt(0).toString(16));
 CLIENT='/* Mes symptômes : moteur partagé avec le Worker (symptomes-engine.mjs) et contenu publié. */\n(function(){\'use strict\';\n'+SYMPTOMES_ENGINE_SOURCE+
  '\nwindow.SymptomesEngine=Object.freeze({'+SYMPTOMES_ENGINE_EXPORTS.join(',')+'});\nwindow.SYMPTOMES_CONTENT='+json(SYMPTOMES)+';\nwindow.SYMPTOMES_SUBJECT_LINKS='+json(links)+';\n})();\n';
 return CLIENT;
}
// Module labels for the Paramètres transparency line (no answer, no content beyond the titles).
export const SYMPTOMES_MODULE_LABELS=Object.fromEntries(SYMPTOMES.modules.map(m=>[m.id,m.label]));

export const SYMPTOMES_INTERFACE_SCRIPT=String.raw`
var SY={page:null,work:{},open:{},acknowledged:{},sessionScreen:null,stopped:false,overlap:{},saved:{},sessionAnswers:{},chosen:null,loading:null,view:null,current:null,user:null,redoEarly:{},declineKeys:{},posting:{}};
// Everything this page remembers for one account; emptied at sign-out, account change or deletion.
function syResetSession(){
 Object.keys(SY.work||{}).forEach(function(id){var w=SY.work[id];if(w){clearTimeout(w.timer);clearTimeout(w.retryTimer)}});
 SY.work={};SY.acknowledged={};SY.sessionScreen=null;SY.sessionAnswers={};SY.overlap={};SY.saved={};SY.open={};SY.redoEarly={};SY.declineKeys={};SY.posting={};
 SY.stopped=false;SY.chosen=null;SY.chooserOpen=undefined;SY.page=null;SY.user=null;
 var dialog=typeof document!=='undefined'&&document.getElementById&&document.getElementById('sy-safety-dialog');if(dialog){dialog.handled=true;if(dialog.open)dialog.close();dialog.remove()}
}
// Sign-out or account deletion: nothing of this page stays in memory or on the device.
function syLeave(userId){syResetSession();if(userId)syPurgeStorage(userId)}
function syUserId(){return state.me&&state.me.user&&state.me.user.id||''}
// Device storage of this page: keys « symptomes-<kind>:<user>[:<passation>] ». keep(key) keeps a key.
// Both stores: localStorage (choices, « ne pas répondre ») and sessionStorage (confirmed screens).
function syPurgeStorage(userId,keep){
 [function(){return localStorage},function(){return sessionStorage}].forEach(function(get){
  try{var store=get(),drop=[];for(var i=0;i<store.length;i++){var key=store.key(i);if(typeof key!=='string'||key.indexOf('symptomes-')!==0)continue;var owner=key.split(':')[1];if(userId&&owner!==userId)continue;if(keep&&keep(key))continue;drop.push(key)}drop.forEach(function(key){store.removeItem(key)})}catch(e){}
 });
}
// One live region for the whole page, never re-created by a render.
function syAnnounce(text){var node=document.getElementById('sy-live');if(!node){node=el('p','visually-hidden');node.id='sy-live';node.setAttribute('aria-live','polite');node.setAttribute('role','status');document.body.append(node)}node.textContent='';node.textContent=text||''}
var SY_DISCLAIMER='Ce n’est pas un diagnostic : seul un professionnel peut en poser un, en entretien.';
var SY_PREF_LABEL='Je préfère ne pas répondre';
function syEngine(){return typeof window!=='undefined'&&window.SymptomesEngine||null}
function syData(){return typeof window!=='undefined'&&window.SYMPTOMES_CONTENT||null}
function syLinks(){return typeof window!=='undefined'&&window.SYMPTOMES_SUBJECT_LINKS||{}}
function syLoadEngine(){
 if(syEngine()&&syData())return Promise.resolve();if(SY.loading)return SY.loading;
 SY.loading=new Promise(function(resolve,reject){var script=document.createElement('script');script.src='/mes-symptomes/moteur.js';script.async=true;script.onload=function(){if(syEngine()&&syData())resolve();else{SY.loading=null;reject(new Error('Le questionnaire n’a pas pu être chargé. Rechargez la page.'))}};script.onerror=function(){SY.loading=null;script.remove();reject(new Error('Le questionnaire n’a pas pu être chargé. Vérifiez votre connexion puis réessayez.'))};document.head.append(script)});
 return SY.loading;
}
function syModule(id){return (syData().modules||[]).find(function(m){return m.id===id})||null}
function syQuestion(m,id){return (m.questions||[]).find(function(q){return q.id===id})||null}
function syOptions(scale){var s=syData().scales[scale];return s?s.options.slice():[]}
function syPeriod(id){var p=syData().periods[id];return p?p.label:''}
function syInstrumentName(id){var inst=syData().instruments[id];if(!inst)return id;var short=/\(([^)]+)\)\s*$/.exec(inst.name);return short?short[1]:inst.name}
function syRow(id){return SY.page&&(SY.page.modules||[]).find(function(row){return row.id===id})||null}
function syInstrumentItems(instrumentId){
 var D=syData(),inst=D.instruments[instrumentId];if(!inst)return [];var parentId=inst.derived_from||instrumentId,parent=D.instruments[parentId]||inst;
 return (inst.items||[]).map(function(ref){var id=typeof ref==='string'?ref:ref.id,item=(parent.items||[]).find(function(p){return (typeof p==='string'?p:p.id)===id})||ref;if(typeof item==='string')item={id:item,text:item};
  return {id:id,text:item.text,scale:item.scale||parent.scale,head:item.stem||parent.stem||syPeriod(item.period||parent.period),reuse:item.reuse||null,na:parent.not_applicable&&parent.not_applicable.item===id?parent.not_applicable.label:null,instrument:parentId,official:true}});
}
// The questions a module asks for the current answers: entry, then (if the entry opens the module,
// or the full instrument is asked for) the follow-up and the question about the impact on life.
// Each group starts with its reference period (the instrument stem or the period of the questions).
function syFlow(moduleId,answers,options){
 options=options||{};answers=answers||{};var E=syEngine(),D=syData(),m=syModule(moduleId),groups=[],ids=[],reused={};
 function push(head,item,extra){if(ids.indexOf(item.id)>=0)return;var last=groups[groups.length-1];extra=extra||{};if(!last||last.head!==head||extra.fresh){last={head:head,items:[],intro:extra.intro||'',notice:extra.notice||''};groups.push(last)}else if(extra.notice&&!last.notice)last.notice=extra.notice;last.items.push(item);ids.push(item.id)}
 function site(list){(list||[]).forEach(function(id){var q=syQuestion(m,id);if(q)push(syPeriod(q.period),{id:q.id,text:q.text,hint:q.hint||'',scale:q.scale,period:q.period,role:q.role,official:false})})}
 function instrument(id,skip){
  var inst=D.instruments[id];if(!inst)return;
  if(inst.scoring&&inst.scoring.type==='algorithme'){
   var items={};syInstrumentItems(id).forEach(function(item){items[item.id]=item});
   if(inst.event&&inst.event.when)push(inst.event.instruction,{id:inst.event.when.id,text:inst.event.when.text,scale:inst.event.when.scale,instrument:id,official:true},{fresh:true,intro:inst.site_intro||'',notice:inst.header_official||''});
   (inst.parts||[]).forEach(function(part){var head=[part.intro,part.question].filter(Boolean).join(' ');(part.items||[]).forEach(function(itemId,i){if(items[itemId])push(head,items[itemId],i===0?{fresh:true}:null)})});
   return;
  }
  syInstrumentItems(id).forEach(function(item){
   if(skip&&skip.indexOf(item.id)>=0)return;
   // PHQ-15 sleep and fatigue: the PHQ-9 answers of the same session are copied, never asked twice.
   if(item.reuse&&options.reuse&&typeof options.reuse[item.reuse]==='number'){reused[item.id]=options.reuse[item.reuse];return}
   var source=D.instruments[item.instrument]||{};push(item.head,item,{notice:source.notice||'',intro:source.medical_note||''});
  });
 }
 var entry=m.entry||{};
 if(entry.instrument)instrument(entry.instrument);else site(entry.questions);
 var ctx={answers:answers,scores:E.computeScores(answers,D)},value=entry.positive_if?E.evaluate(entry.positive_if,ctx):null;
 var open=value===true||Boolean(options.requested&&entry.deepen_on_request&&value===false);
 var offer=!open&&value===false&&entry.deepen_on_request&&m.deepen&&m.deepen.instrument?m.deepen.instrument:null;
 if(open&&m.deepen){
  var d=m.deepen;
  if(d.instrument)instrument(d.instrument,d.reuse_entry_items&&entry.instrument?syInstrumentItems(entry.instrument).map(function(item){return item.id}):null);
  site(d.questions);
  (d.branches||[]).forEach(function(branch){if(E.evaluate(branch.if,ctx)!==true)return;if(branch.instrument)instrument(branch.instrument);site(branch.questions)});
 }
 if(open){var gene=(m.questions||[]).find(function(q){return q.role==='gene'});if(gene)site([gene.id])}
 return {module:m,groups:groups,ids:ids,entry:value,open:open,offer:offer,reused:reused};
}
function syReuse(){var out={};Object.keys(SY.sessionAnswers).forEach(function(k){out[k]=SY.sessionAnswers[k]});var humeur=SY.work.humeur;if(humeur)Object.keys(humeur.answers).forEach(function(k){out[k]=humeur.answers[k]});return out}
function syCurrentFlow(id){var w=SY.work[id];return syFlow(id,w?w.answers:{},{requested:Boolean(w&&w.requested),reuse:syReuse()})}
// Payload: only the questions currently asked (an answer to a question that no longer follows is not
// sent), the universal safety question for the first step, the copied PHQ-15 items. A local
// « Je préfère ne pas répondre » to an instrument item is sent as « no answer » (null, never 0).
function syPayloadAnswers(id,work,flow){
 var out={},m=syModule(id);work=work||SY.work[id]||{answers:{}};flow=flow||syFlow(id,work.answers,{requested:work.requested,reuse:syReuse()});
 var allowed=flow.ids.slice();if(m.temps==='temps1'&&syData().safety.universal_question&&work.safetyOwner!==false)allowed.push(syData().safety.universal_question.id);
 allowed.forEach(function(q){var v=work.answers[q];if(v!==undefined&&v!==null&&v!=='pref_local')out[q]=v});
 Object.keys(flow.reused).forEach(function(q){out[q]=flow.reused[q]});
 return out;
}
function syVersions(id){var E=syEngine(),D=syData(),out={};(E.symptomsIndex(D).instrumentsOf.get(id)||[]).forEach(function(inst){out[inst]=D.instruments[inst].version});return out}
function syKey(id){var hex='';try{var bytes=new Uint8Array(6);crypto.getRandomValues(bytes);bytes.forEach(function(b){hex+=('0'+b.toString(16)).slice(-2)})}catch(e){hex=Math.random().toString(16).slice(2,14)}return id.replace(/_/g,'-')+'-'+new Date().toISOString().slice(0,10).replace(/-/g,'')+'-'+hex}
function syTemps1(){return (syData().modules||[]).filter(function(m){return m.temps==='temps1'}).map(function(m){return m.id})}
function syDone(id){var row=syRow(id);return Boolean(row&&row.last)&&!SY.work[id]}
// The universal safety question opens each passation of the first step: one question on screen,
// copied into every first-step module in progress.
function sySafetyTargets(){return syTemps1().filter(function(id){return !syDone(id)})}
function sySharedSafety(){var found;sySafetyTargets().some(function(id){var w=SY.work[id];if(w&&w.answers.securite_actuelle!==undefined){found=w.answers.securite_actuelle;return true}return false});return found}
function syWork(id,create){
 var w=SY.work[id];if(w||!create)return w||null;var m=syModule(id),answers={},prefilled={},life=SY.page&&SY.page.lifetime_answers||{};
 (m.questions||[]).forEach(function(q){if(q.period==='vie'&&life[q.id]!=null){answers[q.id]=life[q.id];prefilled[q.id]=true}});
 if(m.temps==='temps1'){var shared=sySharedSafety();if(shared!==undefined)answers.securite_actuelle=shared}
 w=SY.work[id]={key:syKey(id),started_at:Math.floor(Date.now()/1000),answers:answers,prefilled:prefilled,requested:false,dirty:0,saving:false,status:'',error:false};
 // The first-step passations in progress share the scope of the universal question: one screen per
 // answer on screen. A first-step passation started when no other is in progress gets a new scope,
 // so a new « oui » opens the blocking screen again.
 if(m.temps==='temps1'){var peer=syTemps1().map(function(t){return t!==id&&SY.work[t]}).filter(Boolean)[0];w.safetyScope=peer&&peer.safetyScope||w.key;w.safetyOwner=!(peer&&peer.answers.securite_actuelle!==undefined)}
 return w;
}
function syAdoptDrafts(page){
 SY.work={};
 (page.modules||[]).forEach(function(row){var d=row.draft;if(!d||!d.current_version||!d.answers)return;var answers={};Object.keys(d.answers).forEach(function(k){answers[k]=d.answers[k]});
  var w={key:d.passation_key,started_at:d.started_at,answers:answers,prefilled:{},requested:false,dirty:0,saving:false,status:'Brouillon repris.',error:false};syRestorePrefs(w);
  var m=syModule(row.id);if(m&&m.entry&&m.entry.deepen_on_request){var plain=syFlow(row.id,answers,{reuse:answers}).ids,asked=syFlow(row.id,answers,{requested:true,reuse:answers}).ids;w.requested=Object.keys(answers).some(function(k){return plain.indexOf(k)<0&&asked.indexOf(k)>=0})}
  SY.work[row.id]=w});
 // First-step drafts: one shared scope; the universal answer is recorded by one module only.
 var firsts=syTemps1().map(function(id){return SY.work[id]}).filter(Boolean),owner=firsts.filter(function(w){return w.answers.securite_actuelle!==undefined})[0];
 firsts.forEach(function(w){w.safetyScope=firsts[0].key;w.safetyOwner=w===owner});
 if(owner)firsts.forEach(function(w){if(w!==owner&&w.answers.securite_actuelle===undefined)w.answers.securite_actuelle=owner.answers.securite_actuelle});
 // Orphan « ne pas répondre » keys (passations recorded or deleted elsewhere) are dropped.
 var live=Object.keys(SY.work).map(function(id){return syPrefKey(SY.work[id])}),user=syUserId();
 syPurgeStorage(user,function(key){return key.indexOf('symptomes-pref:')!==0||live.indexOf(key)>=0});
 syRestoreAcks();
}
// « Je préfère ne pas répondre » to an official item is sent as no answer; the page keeps it on this
// device for the passation in progress (best effort: storage may be unavailable).
function syPrefKey(work){return 'symptomes-pref:'+(state.me&&state.me.user&&state.me.user.id||'')+':'+work.key}
function syStorePrefs(work){try{var ids=Object.keys(work.answers).filter(function(k){return work.answers[k]==='pref_local'});if(ids.length)localStorage.setItem(syPrefKey(work),JSON.stringify(ids));else localStorage.removeItem(syPrefKey(work))}catch(e){}}
function syRestorePrefs(work){try{var ids=JSON.parse(localStorage.getItem(syPrefKey(work))||'[]');if(Array.isArray(ids))ids.forEach(function(k){if(typeof k==='string'&&work.answers[k]===undefined)work.answers[k]='pref_local'})}catch(e){}}
function syDropPrefs(work){try{localStorage.removeItem(syPrefKey(work))}catch(e){}}
function syChosenKey(){return 'symptomes-themes:'+(state.me&&state.me.user&&state.me.user.id||'')}
function syChosen(){if(!SY.chosen){SY.chosen={};try{var saved=JSON.parse(localStorage.getItem(syChosenKey())||'[]');if(Array.isArray(saved))saved.forEach(function(id){if(syModule(id))SY.chosen[id]=true})}catch(e){}}return SY.chosen}
function sySaveChosen(){try{localStorage.setItem(syChosenKey(),JSON.stringify(Object.keys(SY.chosen).filter(function(id){return SY.chosen[id]})))}catch(e){}}

// ---- Safety -----------------------------------------------------------------------------------
function syTriggerDefs(){return syData().safety.triggers||[]}
function syAckScope(trigger,work){return trigger.id.indexOf('securite')===0?(work.safetyScope||work.key):work.key}
function syCheckSafety(id){
 var E=syEngine(),D=syData(),w=SY.work[id];if(!w)return false;
 var triggers=E.safetyTriggers(syPayloadAnswers(id,w),D),blocking=triggers.filter(function(t){return t.screen==='bloquant'});
 var fresh=blocking.filter(function(t){return !SY.acknowledged[syAckScope(t,w)+':'+t.id]});if(!fresh.length)return false;
 fresh.forEach(function(t){SY.acknowledged[syAckScope(t,w)+':'+t.id]=true});
 var screen=E.safetyScreen(blocking,D,{age:SY.page&&SY.page.age?SY.page.age.age:null});
 SY.sessionScreen=syMergeScreens(SY.sessionScreen,screen);sySafetyDialog(screen);return true;
}
// A blocking screen confirmed (« Continuer », « Arrêter »…) stays confirmed for its passation after a
// reload, in this tab only: « symptomes-ack:<user>:<passation> » lists its trigger ids. Rewritten from
// memory, so a re-armed trigger, a recorded passation or a new one (« Refaire ») leaves no key.
function syAckKey(work){return 'symptomes-ack:'+syUserId()+':'+work.key}
function syStoreAcks(){
 try{var user=syUserId(),prefix='symptomes-ack:'+user+':',live={};if(!syData())return;
  Object.keys(SY.work).forEach(function(id){var w=SY.work[id],ids=syTriggerDefs().filter(function(t){return SY.acknowledged[syAckScope(t,w)+':'+t.id]}).map(function(t){return t.id});if(ids.length)live[syAckKey(w)]=JSON.stringify(ids)});
  var drop=[];for(var i=0;i<sessionStorage.length;i++){var key=sessionStorage.key(i);if(typeof key==='string'&&key.indexOf(prefix)===0&&!live[key])drop.push(key)}
  drop.forEach(function(key){sessionStorage.removeItem(key)});Object.keys(live).forEach(function(key){sessionStorage.setItem(key,live[key])});
 }catch(e){}
}
function syRestoreAcks(){
 Object.keys(SY.work).forEach(function(id){var w=SY.work[id];try{var ids=JSON.parse(sessionStorage.getItem(syAckKey(w))||'[]');if(Array.isArray(ids))syTriggerDefs().forEach(function(t){if(ids.indexOf(t.id)>=0)SY.acknowledged[syAckScope(t,w)+':'+t.id]=true})}catch(e){}});
 syStoreAcks();
}
function syMergeScreens(a,b){if(!a)return b;if(!b)return a;var seen={},resources=[];a.resources.concat(b.resources).forEach(function(r){if(!seen[r.id]){seen[r.id]=true;resources.push(r)}});var order=syData()?syData().safety.display.blocking_screen.order:[];resources.sort(function(x,y){return order.indexOf(x.id)-order.indexOf(y.id)});return {always:a.always||b.always,resources:resources,kinds:a.kinds.concat(b.kinds.filter(function(k){return a.kinds.indexOf(k)<0})),quick_exit:a.quick_exit||b.quick_exit}}
function syResources(list){var ul=el('ul','sy-resources');(list||[]).forEach(function(r){var li=el('li'),a=el('a','sy-resource');a.href='tel:'+String(r.number).replace(/\s+/g,'');a.append(el('strong','',r.number),el('span','',r.label));li.append(a);ul.append(li)});return ul}
function syQuickExit(){try{if(SY.view)SY.view.replaceChildren()}catch(e){}location.replace('https://www.google.fr/')}
// Blocking screen: immediate, no delay, no score, never closed by a click outside or Escape.
function sySafetyDialog(screen){
 var old=document.getElementById('sy-safety-dialog');if(old){if(old.open)old.close();old.remove()}
 var dialog=el('dialog','sy-safety');dialog.id='sy-safety-dialog';dialog.setAttribute('aria-labelledby','sy-safety-title');
 var title=el('h2','',screen.opening);title.id='sy-safety-title';title.setAttribute('tabindex','-1');dialog.append(title);
 // Violences: the quick exit comes first, before any reading.
 if(screen.quick_exit)dialog.append(actionButton(screen.quick_exit,syQuickExit,'danger sy-quick-exit'));
 (screen.messages||[]).forEach(function(text){dialog.append(el('p','sy-safety-message',String(text).replace(/^Merci de l’avoir dit\.\s*/,'')))});
 dialog.append(el('p','sy-safety-always',screen.always));dialog.append(syResources(screen.resources));
 var actions=el('div','sy-safety-actions');
 // « Écrire à Ambre » only when Ambre follows the deterministic safety protocol for this account (lot C).
 var ambre=syAmbreAvailable();
 (screen.buttons||[]).filter(function(label){return ambre||!/Ambre/.test(label)}).forEach(function(label){var kind=/Ambre/.test(label)?'ambre':/^Arr/.test(label)?'stop':'continue';
  actions.append(actionButton(label,function(){dialog.handled=true;dialog.close();dialog.remove();syStoreAcks();
   if(kind==='ambre'){if(typeof go==='function')go('chat');else location.assign('/mon-accompagnement/');return}
   if(kind==='stop'){SY.stopped=true;syRender();var top=document.getElementById('sy-safety-box');if(top&&top.scrollIntoView)top.scrollIntoView({block:'start'});return}
   syRender();
  },kind==='continue'?'':'secondary'))});
 dialog.append(actions);dialog.addEventListener('cancel',function(event){event.preventDefault()});
 // Some browsers close a modal on a second Escape despite « cancel »: the box of resources then
 // appears at once on the page.
 dialog.addEventListener('close',function(){if(dialog.handled)return;dialog.handled=true;dialog.remove();syStoreAcks();syRender()});
 document.body.append(dialog);dialog.showModal();if(title.focus)title.focus({preventScroll:true});return dialog;
}
function sySoftTexts(id){
 var E=syEngine(),D=syData(),w=SY.work[id];if(!w)return {};var out={},defs={};syTriggerDefs().forEach(function(t){defs[t.id]=t});
 E.safetyTriggers(syPayloadAnswers(id,w),D).filter(function(t){return t.screen==='doux'}).forEach(function(t){var q=defs[t.id]&&defs[t.id].when&&defs[t.id].when.q,text=t.kind==='suicide'?D.safety.display.soft_block.text:(D.safety.display.soft_block_by_kind||{})[t.kind];if(q&&text)out[q]=text});
 return out;
}

// ---- Answers, autosave, recording ------------------------------------------------------------
function syAnswer(id,question,value){
 var targets=id==='temps1'?sySafetyTargets():[id];
 targets.forEach(function(target,i){var w=syWork(target,true);if(id==='temps1')w.safetyOwner=i===0;
  if(value==='pref'&&question.official)w.answers[question.id]='pref_local';else w.answers[question.id]=value;
  delete w.prefilled[question.id];w.status='';if(question.official)syStorePrefs(w);
  // An answer that no longer meets a trigger re-arms its screen for a later « oui ».
  syTriggerDefs().forEach(function(t){if(t.when&&t.when.q===question.id&&!syEngine().evaluate(t.when,{answers:w.answers}))delete SY.acknowledged[syAckScope(t,w)+':'+t.id]});
 });
 syStoreAcks();
 if(question.official||question.id==='phq9_3'||question.id==='phq9_4')SY.sessionAnswers[question.id]=value==='pref'?null:value;
 var shown=false;targets.forEach(function(target){if(syCheckSafety(target))shown=true});
 targets.forEach(function(target){syScheduleSave(target,shown)});
 if(!shown)syRender();
}
function syScheduleSave(id,now){var w=SY.work[id];if(!w)return;w.dirty++;clearTimeout(w.timer);w.timer=setTimeout(function(){sySaveDraft(id)},now?0:700)}
function syErrorText(err){if(err&&err.code==='unknown_version')return 'Le questionnaire a changé. Rechargez la page pour continuer.';return err&&err.message||'Enregistrement impossible pour le moment.'}
async function sySaveDraft(id){
 var w=SY.work[id];if(!w)return;if(w.saving){w.again=true;return}
 var revision=w.dirty;w.saving=true;
 try{
  var result=await api('/api/symptomes/drafts/'+encodeURIComponent(id),'PUT',{passation_key:w.key,content_version:syData().version,instrument_versions:syVersions(id),answers:syPayloadAnswers(id,w),started_at:w.started_at});
  if(SY.work[id]!==w)return;
  // A late autosave of a passation already recorded: no draft is recreated, nothing to resume.
  if(result.finished){delete SY.work[id];syRender();return}
  w.error=false;w.status='Brouillon enregistré.';w.retries=0;clearTimeout(w.retryTimer);if(w.dirty===revision)w.dirty=0;
  var screen=result.safety&&result.safety.screen;
  if(screen&&screen.blocking&&!document.getElementById('sy-safety-dialog')){var fresh=(result.safety.triggers||[]).filter(function(t){return t.screen==='bloquant'&&!SY.acknowledged[syAckScope(t,w)+':'+t.id]});if(fresh.length){fresh.forEach(function(t){SY.acknowledged[syAckScope(t,w)+':'+t.id]=true});SY.sessionScreen=syMergeScreens(SY.sessionScreen,screen);sySafetyDialog(screen)}}
 }catch(err){if(SY.work[id]!==w)return;w.error=true;w.status=syErrorText(err)+' Vos réponses restent affichées ici.';syAnnounce(w.status);if(['consent_required','minor','birth_date_required'].indexOf(err.code)>=0){w.saving=false;return syReload()}
  // A safety answer is never lost: while a blocking trigger is not confirmed by the server, retry
  // with a growing delay (2 s, 4 s, 8 s… up to one minute).
  if(!err.status||err.status>=500||err.status===429){var pending=syEngine().safetyTriggers(syPayloadAnswers(id,w),syData()).some(function(t){return t.screen==='bloquant'});
   if(pending){w.retries=(w.retries||0)+1;clearTimeout(w.retryTimer);w.retryTimer=setTimeout(function(){if(SY.work[id]===w){w.dirty++;sySaveDraft(id)}},Math.min(60000,1000*Math.pow(2,w.retries)))}}}
 finally{if(SY.work[id]===w){w.saving=false;syStatusUpdate(id);if(w.again||w.dirty&&w.dirty!==revision){w.again=false;syScheduleSave(id)}}}
}
function syStatusUpdate(id){var node=document.getElementById('sy-status-'+id),w=SY.work[id];if(node&&w){node.textContent=w.status||'';node.className='sy-save'+(w.error?' is-error':'')}}
async function syComplete(id,declined){
 var w=SY.work[id]||null;if(w){clearTimeout(w.timer);clearTimeout(w.retryTimer)}
 if(SY.posting[id])return;
 // A refusal without passation keeps one key until it is recorded: a retry never makes a duplicate.
 var key=w?w.key:SY.declineKeys[id]||(SY.declineKeys[id]=syKey(id));
 var body={module:id,content_version:syData().version,instrument_versions:declined?{}:syVersions(id),answers:declined?{}:syPayloadAnswers(id,w),declined:Boolean(declined),passation_key:key,started_at:w?w.started_at:null};
 var result;SY.posting[id]=true;
 try{
  try{result=await api('/api/symptomes/checks','POST',body)}
  catch(err){if(err.code!=='passation_conflict')throw err;body.passation_key=syKey(id);if(!w)SY.declineKeys[id]=body.passation_key;result=await api('/api/symptomes/checks','POST',body)}
 }finally{delete SY.posting[id]}
 delete SY.declineKeys[id];
 // The universal question of a recorded first-step passation is settled: once no first-step
 // passation of its scope is in progress, a new « oui » opens the blocking screen again.
 if(w&&w.safetyScope&&!syTemps1().some(function(t){return t!==id&&SY.work[t]&&SY.work[t].safetyScope===w.safetyScope}))Object.keys(SY.acknowledged).forEach(function(k){if(k.indexOf(w.safetyScope+':')===0)delete SY.acknowledged[k]});
 if(w)syDropPrefs(w);delete SY.work[id];syStoreAcks();syAnnounce(declined?'Choix enregistré.':'Thème enregistré.');SY.overlap[id]=Boolean(result.overlap);SY.saved[id]=true;SY.open[id]=true;
 syApply(result.page,false);syRender();
}
async function syReload(){try{var page=await api('/api/symptomes');syApply(page,false)}catch(err){}syRender()}
function syApply(page,adopt){SY.page=page||SY.page;if(adopt)syAdoptDrafts(SY.page)}

// ---- Results ---------------------------------------------------------------------------------
function syFill(text,rule){
 var level=rule.level||null,score=rule.score,inst=syData().instruments[rule.instrument]||{},max=level&&level.max!=null?level.max:inst.scoring&&inst.scoring.max;
 return String(text||'').replace(/\{score\}/g,score==null?'':String(score)).replace(/\{level\}/g,level?level.label:'').replace(/\{max\}/g,max==null?'':String(max)).replace(/\{level_clause\}/g,level&&score>=5&&score<=9?', palier « '+level.label+' »':'');
}
// What a result card says, from the result the server computed (with the version of the passation):
// « au-dessus / en dessous du repère publié par <instrument> », paliers, the fixed sentences;
// never the name of a disorder as a conclusion.
function syResultModel(id,check){
 var D=syData(),F=D.fixed_texts,m=syModule(id),r=check&&check.result||{},stateName=check&&(check.state||r.state)||'non_explore',defs={};
 (m.rules||[]).forEach(function(rule){defs[rule.id]=rule});
 var model={module:id,title:m.label,date:check?check.completed_at:null,state:stateName,phrase:'',flag:false,paragraphs:[],notes:[],refs:[],what_next:false,gene:null,positives:0,id:check?check.id:null};
 if(stateName==='refuse'){model.phrase='Vous avez choisi de ne pas répondre à ce thème.';return model}
 var rules=r.rules||[],instrumentOf=function(rule){return rule?syInstrumentName(rule.instrument):''},levelOf=function(rule){return rule&&rule.instrument!=='itq'&&rule.level&&rule.level.label?' · palier « '+rule.level.label+' »':''};
 if(stateName==='au_dessus'){var above=rules.find(function(x){return x.kind==='instrument'&&x.result==='positif'});model.phrase=above?'Au-dessus du repère publié par '+instrumentOf(above)+levelOf(above):'Au-dessus du repère publié';model.flag=true}
 else if(stateName==='en_dessous'){var below=rules.find(function(x){return x.kind==='instrument'&&x.result==='negatif'&&x.display});var entry=rules.find(function(x){return x.kind==='entree'&&x.result==='negatif'&&x.display});model.phrase=below?'En dessous du repère publié par '+instrumentOf(below)+levelOf(below):entry?'En dessous du seuil d’entrée ('+instrumentOf(entry)+')':'En dessous du repère'}
 else if(stateName==='repere_present'){model.phrase=r.several?'Plusieurs repères présents':'Repère présent';model.flag=true}
 else if(stateName==='repere_absent')model.phrase='Aucun repère présent';
 else if(stateName==='indetermine')model.phrase='Résultat non lisible';
 var negative=false,note=function(text){if(text&&model.notes.indexOf(text)<0)model.notes.push(text)};
 rules.forEach(function(rule){var def=defs[rule.id];if(!rule.display||!def)return;
  if(rule.result==='positif'){model.paragraphs.push(syShort(syFill(def.result_positive,rule),model.refs));if(rule.kind!=='contexte')model.positives++;
   var key=rule.note||def.positive_note;note(key==='positive_note'?F.positive_note:key==='site_note'?F.site_note:key==='high_note'?F.high_note:key==='texte'?def.positive_note_text:'');
   if(rule.kind==='repere')note(F.unvalidated_label);if(rule.kind==='repere_site')note(F.unvalidated_label_site)}
  // One card, one conclusion: a card above the reference does not list the secondary negative texts.
  else if(rule.result==='negatif'&&def.result_negative&&!model.flag){model.paragraphs.push(syShort(syFill(def.result_negative,rule),model.refs));negative=true}
 });
 if(r.module_negative&&m.module_negative&&!model.flag){model.paragraphs.push(m.module_negative);negative=true}
 if(stateName==='indetermine')model.paragraphs.push(F.indetermine);
 if(negative&&!model.flag)note(F.negative_note);
 model.what_next=model.flag;
 if(r.gene_text&&F[r.gene_text])model.gene=F[r.gene_text];
 if(check&&check.current_version===false)model.notes.push('Résultat calculé avec une version précédente du questionnaire.');
 return model;
}
// « (score 10 sur 27, palier « modéré », Kroenke et al. 2001) » → « (score 10 sur 27) »: the level is
// in the conclusion, the reference goes in small print under the card.
function syShort(text,refs){
 return String(text).replace(/\s*\(([^()]*)\)/g,function(all,inner){
  var kept=[];inner.split(/\s*[,;]\s*/).forEach(function(part){if(/et al\.\s*\d{4}/.test(part)){if(refs.indexOf(part)<0)refs.push(part)}else if(!/^palier\s*«/.test(part)&&part)kept.push(part)});
  if(kept.length===inner.split(/\s*[,;]\s*/).filter(Boolean).length)return all;
  return kept.length?' ('+kept.join(', ')+')':'';
 });
}
function syRetestText(row){var r=row&&row.retest||{};if(r.never)return 'Ces questions portent sur toute votre vie : elles ne sont pas reposées d’office.';if(!r.next_at)return '';return r.due?'Vous pouvez refaire ce thème.':'Vous pourrez le refaire à partir du '+dateLabel(r.next_at)+'.'}
function syRedo(id){var row=syRow(id);syWork(id,true);SY.open[id]=true;if(syModule(id).temps!=='temps1'){syChosen()[id]=true;sySaveChosen()}SY.redoEarly=SY.redoEarly||{};SY.redoEarly[id]=Boolean(row&&row.retest&&!row.retest.due&&!row.retest.never);SY.stopped=false;syRender();var node=SY.view&&SY.view.querySelector('[data-module="'+id+'"]');if(node&&node.scrollIntoView)node.scrollIntoView({block:'start'})}

// ---- Rendering -------------------------------------------------------------------------------
function syQuestionRow(scope,question,value,extra){
 extra=extra||{};var row=el('fieldset','sy-q');row.dataset.q=question.id;var legend=el('legend','sy-q-text',question.text);row.append(legend);
 if(question.hint)row.append(el('p','sy-q-hint',question.hint));
 var options=syOptions(question.scale);if(question.na)options.push({value:'na',label:question.na});
 if(question.official&&!options.some(function(o){return o.value==='pref'}))options.push({value:'pref',label:SY_PREF_LABEL});
 // Real radio buttons (one group, arrow keys), drawn as a list of fine rows on a phone and as a fine
 // segmented row on a computer when the answers fit. « Je préfère ne pas répondre » is a quiet link
 // under the answers, in the same group.
 var pref=options.filter(function(o){return o.value==='pref'})[0],main=options.filter(function(o){return o.value!=='pref'});
 var list=main.length>5||main.some(function(o){return String(o.label).length>30});
 var group=el('div','sy-choices'+(list?' is-list':''));group.style.setProperty('--n',String(main.length));
 var current=value==='pref_local'?'pref':value,name='sy-'+scope+'-'+question.id;
 function radio(o,cls){var on=current!==undefined&&current!==null&&String(current)===String(o.value);var label=el('label',cls+(on?' is-on':'')),input=el('input');input.type='radio';input.name=name;input.value=String(o.value);input.checked=on;input.dataset.value=String(o.value);input.dataset.syFocus=scope+':'+question.id+':'+o.value;
  input.addEventListener('change',function(){if(input.checked)syAnswer(scope,question,o.value)});label.append(input,el('span','',o.label));return label}
 main.forEach(function(o){group.append(radio(o,'sy-choice'))});
 row.append(group);
 if(pref)row.append(radio(pref,'sy-pref'));
 if(extra.prefilled)row.append(el('p','sy-q-hint','Réponse reprise de votre dernière passation ; vous pouvez la changer.'));
 if(extra.soft)row.append(el('p','sy-soft',extra.soft));
 return row;
}
function syProgress(flow,work){var n=0;flow.ids.forEach(function(q){var v=work&&work.answers[q];if(v!==undefined&&v!==null)n++});return n+' / '+flow.ids.length}
function syModuleSub(id){var row=syRow(id),w=SY.work[id];if(!w&&row&&row.last)return row.last.declined?'Non fait, par choix':'Fait le '+dateLabel(row.last.completed_at);return syProgress(syCurrentFlow(id),w)}
function syModuleSection(id){
 var m=syModule(id),details=el('details','sy-module');details.dataset.module=id;details.open=Boolean(SY.open[id]);
 details.addEventListener('toggle',function(){SY.open[id]=details.open});
 var summary=el('summary','sy-module-head'),text=el('span','sy-module-text');text.append(el('strong','',m.label),el('span','sy-module-sub',syModuleSub(id)));summary.append(text,el('span','sy-chevron','›'));
 var body=el('div','sy-module-body');details.append(summary,body);
 if(syDone(id))syDoneBody(body,id);else syFormBody(body,id);
 return details;
}
function syDoneBody(body,id){
 var row=syRow(id),model=syResultModel(id,row.last);
 if(SY.saved[id])body.append(el('p','sy-saved','Thème enregistré.'));
 var line=el('p','sy-done-state',model.phrase);body.append(line);
 if(!row.last.declined){var link=el('a','text-link','Voir le résultat');link.href='#sy-result-'+id;body.append(link)}
 var retest=syRetestText(row);if(retest&&!row.last.declined)body.append(el('p','sy-retest',retest));
 var actions=el('div','sy-module-actions');actions.append(actionButton(row.last.declined?'Faire ce thème maintenant':'Refaire ce thème',function(){syRedo(id)},'secondary'));body.append(actions);
}
function syFormBody(body,id){
 var m=syModule(id),D=syData(),F=D.fixed_texts,w=SY.work[id],flow=syCurrentFlow(id),soft=sySoftTexts(id);
 if(m.sensible&&m.intro_privacy)body.append(el('p','sy-intro',F[m.intro_privacy]||F.privacy_sensitive));
 if(m.intro)body.append(el('p','sy-intro',m.intro));
 if(m.medical_note)body.append(el('p','sy-intro',m.medical_note));
 if(SY.redoEarly&&SY.redoEarly[id])body.append(el('p','sy-intro','Vous refaites ce thème avant la fin de sa période : les deux périodes se chevaucheront en partie.'));
 flow.groups.forEach(function(group){
  var box=el('div','sy-group');if(group.intro)box.append(el('p','sy-intro',group.intro));box.append(el('p','sy-period',group.head));
  group.items.forEach(function(q){box.append(syQuestionRow(id,q,w?w.answers[q.id]:undefined,{prefilled:w&&w.prefilled[q.id],soft:soft[q.id]}))});
  if(group.notice)box.append(el('p','sy-notice',group.notice));body.append(box);
 });
 if(Object.keys(flow.reused).length)body.append(el('p','sy-notice','Sommeil et fatigue : vos réponses au questionnaire sur l’humeur de cette séance sont reprises.'));
 if(flow.offer)body.append(actionButton('Répondre aussi au '+syInstrumentName(flow.offer)+' complet',function(){var work=syWork(id,true);work.requested=true;syScheduleSave(id);syRender()},'secondary sy-offer'));
 var payload=syPayloadAnswers(id,w),answered=Object.keys(payload).filter(function(k){return k!=='securite_actuelle'}).length,missing=flow.ids.some(function(q){return !w||w.answers[q]===undefined||w.answers[q]===null});
 if(answered&&missing)body.append(el('p','sy-notice','Certaines questions sont sans réponse : vous pouvez enregistrer quand même, le résultat pourra rester non lisible.'));
 var actions=el('div','sy-module-actions'),feedback=el('p','status');
 var save=actionButton('Enregistrer ce thème',async function(){save.disabled=true;decline.disabled=true;feedback.textContent='Enregistrement…';feedback.className='status';try{await syComplete(id,false)}catch(err){feedback.textContent=syErrorText(err);feedback.className='status error';syAnnounce(feedback.textContent);save.disabled=false;decline.disabled=false}});save.disabled=!answered||Boolean(SY.posting[id]);
 var decline=actionButton('Ne pas faire ce thème',async function(){if(!await confirmDialog('Ne pas faire ce thème ?','Ce choix est enregistré ; vous pourrez faire ce thème plus tard.','Confirmer'))return;save.disabled=true;decline.disabled=true;try{await syComplete(id,true)}catch(err){feedback.textContent=syErrorText(err);feedback.className='status error';syAnnounce(feedback.textContent);decline.disabled=false;save.disabled=!answered}},'text');decline.disabled=Boolean(SY.posting[id]);
 actions.append(save,decline);body.append(actions,feedback);
 var status=el('p','sy-save'+(w&&w.error?' is-error':''),w&&w.status||'');status.id='sy-status-'+id;body.append(status);
}
function sySafetyQuestionBlock(){
 var D=syData(),q=D.safety.universal_question;if(!q||!sySafetyTargets().length)return null;var value=sySharedSafety();
 var box=el('div','sy-group sy-safety-question');box.append(el('p','sy-period',syPeriod(q.period)));
 box.append(syQuestionRow('temps1',{id:q.id,text:q.text,scale:q.scale,official:false},value,{soft:value==='pref'?D.safety.display.soft_block.text:''}));
 return box;
}
function sySafetyBox(){
 var banner=SY.page&&SY.page.safety&&SY.page.safety.banner,screen=syMergeScreens(banner,SY.sessionScreen);if(!screen)return null;
 var box=el('section','sy-safety-box');box.id='sy-safety-box';box.setAttribute('aria-label','Numéros utiles');
 if(SY.stopped)box.append(el('p','sy-safety-stopped','Vous avez arrêté. Vos réponses sont gardées en brouillon ; vous pourrez reprendre quand vous le voudrez.'));
 if(screen.quick_exit)box.append(actionButton(screen.quick_exit,syQuickExit,'danger sy-quick-exit'));
 box.append(el('p','',screen.always));box.append(syResources(screen.resources));
 if(SY.stopped)box.append(actionButton('Reprendre le questionnaire',function(){SY.stopped=false;syRender()},'secondary'));
 return box;
}
function syResultCard(row){
 var model=syResultModel(row.id,row.last),card=el('article','sy-result'+(model.flag?' is-flag':''));card.id='sy-result-'+row.id;
 card.append(el('h3','',model.title),el('p','sy-result-meta',dateLabel(model.date)));
 // Conclusion, short explanation, « Et maintenant ? », fixed sentence, reference in small print, actions.
 card.append(el('p','sy-result-state',model.phrase));
 model.paragraphs.forEach(function(text){card.append(el('p','',text))});
 if(model.gene)card.append(el('p','',model.gene));
 if(model.what_next)card.append(el('p','sy-next',syWhatNext(syData().fixed_texts.what_next)));
 if(SY.overlap[row.id])card.append(el('p','sy-result-note',syData().fixed_texts.overlap_note));
 if(model.notes.length){var notes=el('div','sy-result-notes');model.notes.forEach(function(text){notes.append(el('p','',text))});card.append(notes)}
 if(model.refs.length)card.append(el('p','sy-result-ref','Référence : '+model.refs.join(' ; ')+'.'));
 var actions=el('div','sy-result-actions'),feedback=el('p','status');
 if(!SY.work[row.id])actions.append(actionButton(row.last.declined?'Faire ce thème':'Refaire',function(){syRedo(row.id)},'secondary'));
 actions.append(actionButton('Supprimer ce résultat',async function(){if(!await confirmDialog('Supprimer ce résultat ?','Ce résultat sera effacé de votre compte. Cette action est définitive.','Supprimer',true))return;try{var result=await api('/api/symptomes/checks/'+encodeURIComponent(row.last.id),'DELETE');delete SY.overlap[row.id];delete SY.saved[row.id];syApply(result.page,false);syRender()}catch(err){feedback.textContent=err.message;feedback.className='status error';syAnnounce(err.message)}},'text danger'));
 card.append(actions,feedback);
 return card;
}
// « en parler avec Ambre » only when Ambre follows the safety protocol for this account (as the
// button of the blocking screen); the published sentence stays as it is, the clause is removed here.
function syAmbreAvailable(){return Boolean(SY.page&&SY.page.safety&&SY.page.safety.ambre_available)}
function syWhatNext(text){text=String(text||'');if(syAmbreAvailable())return text;
 return text.replace(/,\s*(?:ou\s+)?en parler avec Ambre(?=\s*[,.;])/g,'')}
function syResultsSection(){
 var rows=(SY.page.modules||[]).filter(function(row){return row.last}),section=el('section','sy-section');section.id='sy-resultats';
 section.append(el('h2','','Mes résultats'));
 if(!rows.length){section.append(el('p','sy-empty','Vos résultats apparaîtront ici, thème par thème.'));return section}
 var positives=rows.reduce(function(sum,row){return sum+syResultModel(row.id,row.last).positives},0);
 if(positives>=3)section.append(el('p','sy-section-intro',syData().fixed_texts.several_positive));
 var list=el('div','sy-results');rows.forEach(function(row){list.append(syResultCard(row))});section.append(list);
 section.append(el('p','sy-notice',syData().fixed_texts.methode_absences));
 return section;
}
function syCrossItems(items){
 var ul=el('ul','sy-cross-list'),links=syLinks();
 items.forEach(function(item){var li=el('li');li.append(el('span','sy-cross-label',item.label));if(links[item.id]){var a=el('a','text-link sy-cross-link','Lire');a.href=links[item.id];a.setAttribute('aria-label','Lire : '+item.label);li.append(a)}ul.append(li)});
 return ul;
}
function syCrossingSection(){
 var crossing=SY.page.crossing||{},F=syData().fixed_texts,section=el('section','sy-section sy-cross');section.append(el('h2','','Avec Mon enfance'));
 if(!crossing.childhood_available){var p=el('p','sy-empty','Quand vous aurez rempli Mon enfance, cette partie rapprochera vos deux questionnaires. ');var a=el('a','','Ouvrir Mon enfance');a.href='/cartographie/';p.append(a);section.append(p);return section}
 var lists=crossing.lists||{};
 section.append(el('h3','',F.crossing_title));
 if((lists.deux||[]).length){section.append(syCrossItems(lists.deux),el('p','sy-cross-note',F.crossing_note))}else section.append(el('p','sy-empty','Rien pour le moment.'));
 section.append(el('h3','',F.crossing_symptoms_only));
 section.append((lists.symptomes||[]).length?syCrossItems(lists.symptomes):el('p','sy-empty','Rien pour le moment.'));
 var fold=el('details','sy-fold');fold.open=!F.crossing_childhood_only_collapsed;fold.append(el('summary','',F.crossing_childhood_only));var inner=el('div','sy-fold-body');inner.append(el('p','sy-cross-note',F.crossing_childhood_only_note));inner.append((lists.enfance||[]).length?syCrossItems(lists.enfance):el('p','sy-empty','Rien pour le moment.'));fold.append(inner);section.append(fold);
 (crossing.suspended||[]).forEach(function(s){if(s.text)section.append(el('p','sy-cross-note',s.text))});
 // The two other states: themes still to explore, themes set aside — named, never counted.
 var later={},refused={},subjectModules=syData().subject_modules||{};
 (crossing.subjects||[]).forEach(function(s){if(s.childhood!=='present')return;var target=s.state==='non_explore'?later:s.state==='refuse'?refused:null;if(!target)return;(subjectModules[s.id]||[]).forEach(function(id){var row=syRow(id);if(s.state==='refuse'||!row||!row.last||row.last.state==='indetermine')target[id]=true})});
 var label=function(map){return (syData().modules||[]).filter(function(m){return map[m.id]}).map(function(m){return m.label}).join(' ; ')};
 if(label(later))section.append(el('p','sy-cross-more','Thèmes pas encore explorés qui compléteraient ce rapprochement : '+label(later)+'.'));
 if(label(refused))section.append(el('p','sy-cross-more','Thèmes que vous avez choisi de ne pas faire : '+label(refused)+'.'));
 return section;
}
function syDataSection(){
 var section=el('section','sy-section sy-data');section.append(el('h2','','Mes données'));
 section.append(el('p','sy-section-intro','Vous pouvez effacer vos résultats et retirer votre accord à tout moment. L’export de vos données se fait depuis Mon compte.'));
 var actions=el('div','sy-data-actions'),feedback=el('p','status');
 if((SY.page.modules||[]).some(function(row){return row.last||row.draft})||Object.keys(SY.work).length)actions.append(actionButton('Supprimer tous mes résultats',async function(){if(!await confirmDialog('Supprimer tous vos résultats ?','Vos résultats, vos brouillons et les alertes de sécurité de cette page seront effacés. Votre accord reste tel quel.','Tout supprimer',true))return;try{var result=await api('/api/symptomes/checks','DELETE');Object.keys(SY.work).forEach(function(k){clearTimeout(SY.work[k].timer);clearTimeout(SY.work[k].retryTimer)});SY.work={};SY.overlap={};SY.saved={};SY.sessionScreen=null;SY.acknowledged={};SY.chosen=null;syPurgeStorage(syUserId());syApply(result.page,false);syRender();syAnnounce('Vos résultats ont été supprimés.')}catch(err){feedback.textContent=err.message;feedback.className='status error';syAnnounce(err.message)}},'text danger'));
 if(SY.page.consent&&SY.page.consent.given)actions.append(actionButton('Retirer mon accord',async function(){if(!await confirmDialog('Retirer votre accord ?','Rien de nouveau ne sera enregistré et Ambre ne recevra plus de résumé. Vos résultats restent jusqu’à ce que vous les supprimiez.','Retirer',true))return;try{await api('/api/symptomes/consent','PUT',{consent:false});Object.keys(SY.work).forEach(function(k){clearTimeout(SY.work[k].timer);clearTimeout(SY.work[k].retryTimer)});SY.work={};SY.chosen=null;syPurgeStorage(syUserId());await syReload();syAnnounce('Votre accord est retiré.')}catch(err){feedback.textContent=err.message;feedback.className='status error';syAnnounce(err.message)}},'text danger'));
 var account=el('a','text-link','Exporter depuis Mon compte');account.href='/mon-compte/';actions.append(account);
 section.append(actions,feedback);return section;
}
function syConsentSection(){
 var F=syData().fixed_texts,section=el('section','sy-section sy-consent');section.append(el('h2','','Avant de commencer'));
 var list=el('ul','sy-consent-list');
 [F.privacy_sensitive,'Ces questionnaires repèrent des difficultés ; ils ne posent pas de diagnostic.','Avec votre accord, Ambre reçoit seulement un résumé des thèmes au-dessus du repère, avec leur date ; jamais vos réponses ni aucun score. Paramètres vous montre ce résumé.','Vous pouvez retirer votre accord à tout moment.'].forEach(function(text){list.append(el('li','',text))});
 section.append(list);
 var feedback=el('p','status');
 var accept=actionButton('J’accepte et je commence',async function(){accept.disabled=true;try{await api('/api/symptomes/consent','PUT',{consent:true});await syReload()}catch(err){feedback.textContent=err.message;feedback.className='status error';syAnnounce(err.message);accept.disabled=false;if(['minor','birth_date_required'].indexOf(err.code)>=0)syReload()}});
 var actions=el('div','actions');actions.append(accept);section.append(actions,feedback);return section;
}
function syThemeChooser(){
 var chosen=syChosen(),fold=el('details','sy-fold sy-chooser'),others=(syData().modules||[]).filter(function(m){return m.temps!=='temps1'});
 fold.open=SY.chooserOpen!==undefined?SY.chooserOpen:!others.some(function(m){return chosen[m.id]||SY.work[m.id]});
 fold.addEventListener('toggle',function(){SY.chooserOpen=fold.open});
 fold.append(el('summary','','Choisir des thèmes'));var inner=el('div','sy-fold-body');
 [[false,''],[true,'Thèmes plus intimes, proposés seulement si vous les cochez']].forEach(function(part){
  var list=others.filter(function(m){return Boolean(m.sensible)===part[0]});if(!list.length)return;
  var set=el('fieldset','sy-checks');if(part[1])set.append(el('legend','',part[1]));else set.append(el('legend','visually-hidden','Thèmes'));
  list.forEach(function(m){var label=el('label','check sy-check'),input=el('input');input.type='checkbox';input.checked=Boolean(chosen[m.id]||SY.work[m.id]);input.disabled=Boolean(SY.work[m.id]);input.dataset.syFocus='choose:'+m.id;
   input.addEventListener('change',function(){chosen[m.id]=input.checked;if(input.checked)SY.open[m.id]=true;sySaveChosen();syRender()});label.append(input,el('span','',m.label));set.append(label)});
  inner.append(set);
 });
 fold.append(inner);return fold;
}
function syRender(){
 var view=SY.view;if(!view||view.isConnected===false||SY.current&&!SY.current())return;
 var active=document.activeElement,focusKey=active&&active.dataset&&active.dataset.syFocus,root=el('div','sy'),keyboard=false;
 try{keyboard=Boolean(active&&active.matches&&active.matches(':focus-visible'))}catch(e){keyboard=false}
 var page=SY.page||{};root.append(el('p','sy-disclaimer',SY_DISCLAIMER));
 if(!page.available){root.append(el('div','notice',page.message||'« Mes symptômes » sera bientôt disponible.'));view.replaceChildren(root);return}
 var age=page.age||{};
 if(age.status==='mineur'){var minor=el('section','sy-section sy-minor');minor.append(el('p','',age.text));minor.append(syResources(age.resources));root.append(minor);view.replaceChildren(root);return}
 if(age.status!=='adulte'){var unknown=el('section','sy-section');var p=el('p','','Cette page est réservée aux adultes : indiquez votre date de naissance dans Mon compte pour l’ouvrir. ');var a=el('a','','Compléter Mon compte');a.href='/mon-compte/#self-profile';p.append(a);unknown.append(p);root.append(unknown);view.replaceChildren(root);return}
 var box=sySafetyBox();
 if(!syEngine()||!syData()){if(box)root.append(box);root.append(el('p','status error','Le questionnaire n’a pas pu être chargé.'),actionButton('Réessayer',function(){renderSymptoms(view,SY.current)},'secondary'));view.replaceChildren(root);return}
 if(!(page.consent&&page.consent.given)){
  if(box)root.append(box);root.append(syConsentSection());
  if((page.modules||[]).some(function(row){return row.last})){root.append(syResultsSection(),syDataSection())}
  view.replaceChildren(root);return;
 }
 if(box)root.append(box);
 if(!SY.stopped){
  var first=el('section','sy-section');first.append(el('h2','','Pour commencer'));
  var question=sySafetyQuestionBlock();if(question)first.append(question);
  var mods=el('div','sy-modules');syTemps1().forEach(function(id){mods.append(syModuleSection(id))});first.append(mods);root.append(first);
  var second=el('section','sy-section');second.append(el('h2','','D’autres thèmes, si vous le souhaitez'));second.append(syThemeChooser());
  var chosen=syChosen(),shown=(syData().modules||[]).filter(function(m){return m.temps!=='temps1'&&(chosen[m.id]||SY.work[m.id])});
  if(shown.length){var others=el('div','sy-modules');shown.forEach(function(m){others.append(syModuleSection(m.id))});second.append(others)}
  root.append(second);
 }
 root.append(syResultsSection(),syCrossingSection(),syDataSection());
 view.replaceChildren(root);
 // Focus comes back to the same answer for keyboard users only (no ring after a tap or a click).
 if(focusKey&&keyboard){var again=view.querySelector('[data-sy-focus="'+focusKey+'"]');if(again&&again.focus)again.focus({preventScroll:true})}
}
async function renderSymptoms(view,current){
 if(SY.user!==syUserId()){syResetSession();SY.user=syUserId()}
 SY.view=view;SY.current=current;SY.chosen=null;SY.stopped=false;
 view.replaceChildren(el('p','muted page-loading','Chargement…'));
 var engine=syLoadEngine().catch(function(err){return err}),page;
 try{page=await api('/api/symptomes')}catch(err){if(current&&!current())return;
  // Offline: the resources of this session stay on screen.
  var box=sySafetyBox(),parts=[];if(box)parts.push(box);parts.push(el('p','status error',err.message),actionButton('Réessayer',function(){renderSymptoms(view,current)},'secondary'));view.replaceChildren.apply(view,parts);return}
 if(page&&page.available)await engine;
 if(current&&!current())return;
 if(page&&page.available&&syEngine()&&syData()){syApply(page,true)}else SY.page=page;
 syRender();
}
`;

export const SYMPTOMES_INTERFACE_STYLE=String.raw`
/* Mes symptômes : a sober notebook. Thin rows, folded themes, fine segmented answers. */
.sy { max-width: var(--page); }
.sy-disclaimer { margin: calc(-1 * var(--s4)) 0 var(--s5); color: var(--muted); font-size: var(--t-xs); line-height: 1.45; }
.sy-section { margin: 0 0 var(--s7); }
.sy-section > h2 { margin: 0 0 var(--s3); }
.sy-section > h3 { margin: var(--s5) 0 var(--s2); font-size: var(--t-md); }
.sy-section-intro, .sy-empty { margin: 0 0 var(--s3); color: var(--muted); font-size: var(--t-sm); line-height: 1.5; }
.sy-modules { border-bottom: 1px solid var(--line); }
.sy-module { border-top: 1px solid var(--line); }
.sy-module > summary { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; column-gap: var(--s3); min-height: 56px; padding: var(--s3) 0; border-radius: var(--r-control); transition: background-color .12s; }
.sy-module > summary:active, .sy-fold > summary:active { background: var(--accent-wash); }
.sy-module-text strong { display: block; font: 600 var(--t-md)/1.35 var(--font-ui); }
.sy-module[open] > summary .sy-module-text strong { color: var(--accent); }
.sy-module-sub { display: block; margin-top: 2px; color: var(--muted); font: 400 var(--t-xs)/1.4 var(--font-ui); font-variant-numeric: tabular-nums; }
.sy-chevron { width: 24px; color: var(--muted); font: 400 20px/1 var(--font-ui); text-align: center; transition: transform .12s; }
.sy-module[open] > summary .sy-chevron { transform: rotate(90deg); color: var(--accent); }
.sy-module-body { padding: 0 0 var(--s5); }
.sy-intro { max-width: 62ch; margin: var(--s2) 0 0; color: var(--muted); font-size: var(--t-sm); line-height: 1.5; }
.sy-group { padding-top: var(--s2); }
.sy-period { max-width: 62ch; margin: var(--s4) 0 0; padding-left: var(--s3); border-left: 2px solid var(--accent); color: var(--ink); font: 400 var(--t-sm)/1.45 var(--font-ui); font-style: italic; }
.sy-safety-question { margin-bottom: var(--s3); }
/* A fieldset per question: the legend floats into the content so that the separator and the spacing
   behave like any block (a legend otherwise sits in the top border). */
.sy-q { display: block; padding: var(--s4) 0 var(--s3); }
.sy-q > * + * { margin-top: var(--s2); }
.sy-q + .sy-q { box-shadow: inset 0 1px 0 color-mix(in srgb, var(--line) 60%, transparent); }
.sy-q-text { max-width: 62ch; margin: 0; font: 600 var(--t-md)/1.4 var(--font-ui); }
.sy-q-hint { max-width: 62ch; margin: 0; color: var(--muted); font-size: var(--t-sm); line-height: 1.5; }
.sy-choices { display: grid; grid-template-columns: repeat(var(--n, 3), minmax(0, 1fr)); gap: 1px; width: min(100%, calc(var(--n, 3) * 11rem)); margin-top: var(--s1); padding: 0; border: 1px solid var(--line); border-radius: var(--r-control); background: var(--line); overflow: hidden; }
.sy-choice, .sy-pref { position: relative; display: flex; align-items: center; justify-content: center; margin: 0; cursor: pointer; }
.sy-choice > input, .sy-pref > input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: pointer; }
.sy-q { min-width: 0; margin: 0; border: 0; }
.sy-q > legend.sy-q-text { float: left; width: 100%; margin: 0 0 var(--s2); padding: 0; }
.sy-q > legend + * { clear: both; margin-top: 0; }
.sy-q > .sy-pref { display: flex; width: fit-content; }
.sy-choice:has(input:focus-visible), .sy-pref:has(input:focus-visible) { outline: 2px solid var(--accent); outline-offset: -3px; }
.sy-choice { min-width: 0; min-height: 44px; padding: 6px var(--s2); border: 0; border-radius: 0; background: var(--surface); color: var(--ink); font: 400 var(--t-sm)/1.25 var(--font-ui); white-space: normal; text-align: center; }
@media (hover: hover) { .sy-choice:hover { background: var(--accent-wash); } }
.sy-choice:not(:disabled):active { transform: none; background: var(--accent-wash); }
.sy-choice:focus-visible { outline-offset: -3px; }
.sy-choice.is-on { background: var(--accent-wash); color: var(--accent); font-weight: 600; box-shadow: inset 0 -2px 0 var(--accent); }
.sy-choice.is-on::before { content: '✓' / ''; margin-right: 3px; font-weight: 700; }
/* The list: one answer per fine row, a radio dot, thin separators, no frame. */
.sy-choices.is-list { display: flex; flex-direction: column; gap: 0; width: 100%; max-width: 32rem; border: 0; border-radius: 0; background: none; }
.sy-choices.is-list > .sy-choice { justify-content: flex-start; gap: var(--s3); padding: 6px var(--s2); border-bottom: 1px solid color-mix(in srgb, var(--line) 70%, transparent); border-radius: 0; background: none; text-align: left; box-shadow: none; }
.sy-choices.is-list > .sy-choice:first-child { border-top: 1px solid color-mix(in srgb, var(--line) 70%, transparent); }
.sy-choices.is-list > .sy-choice::before { content: '' / ''; flex: none; box-sizing: border-box; width: 18px; height: 18px; margin: 0; border: 1.5px solid var(--line-strong); border-radius: 50%; background: var(--surface); color: #fff; font: 700 11px/15px var(--font-ui); text-align: center; }
.sy-choices.is-list > .sy-choice.is-on { background: var(--accent-wash); color: var(--ink); font-weight: 600; }
.sy-choices.is-list > .sy-choice.is-on::before { content: '✓' / ''; border-color: var(--accent); background: var(--accent); }
.sy-choices.is-list > .sy-choice:not(:disabled):active { background: var(--accent-wash); }
.sy-pref { align-self: flex-start; min-height: 44px; color: var(--muted); font-size: var(--t-xs); font-weight: 400; }
.sy-pref.is-on { color: var(--accent); font-weight: 600; }
.sy-pref.is-on::before { content: '✓' / ''; margin-right: 4px; }
.sy-soft { max-width: 62ch; margin: 0; padding-left: var(--s3); border-left: 2px solid var(--danger); font-size: var(--t-sm); line-height: 1.5; }
.sy-notice { max-width: 62ch; margin: var(--s3) 0 0; color: var(--muted); font-size: var(--t-xs); line-height: 1.45; }
.sy-offer { margin-top: var(--s4); }
.sy-module-actions, .sy-result-actions, .sy-data-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s5); }
.sy-module-actions { margin-top: var(--s5); }
.sy-save { margin: var(--s2) 0 0; color: var(--muted); font-size: var(--t-xs); }
.sy-save:empty { display: none; }
.sy-save.is-error { color: var(--danger); }
.sy-saved { margin: var(--s2) 0 0; color: var(--accent); font-size: var(--t-sm); font-weight: 600; }
.sy-done-state { margin: var(--s2) 0 0; font-weight: 600; }
.sy-retest { margin: 0; color: var(--muted); font-size: var(--t-sm); }
.sy-module-body > .text-link { font-size: var(--t-sm); }
.sy-fold { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.sy-fold > summary { position: relative; min-height: 52px; padding: var(--s2) var(--s6) var(--s2) 0; border-radius: var(--r-control); }
.sy-fold > summary::after { content: '›' / ''; position: absolute; right: var(--s1); top: 50%; color: var(--muted); font: 400 22px/1 var(--font-ui); transform: translateY(-50%); transition: transform .12s; }
.sy-fold[open] > summary::after { transform: translateY(-50%) rotate(90deg); }
.sy-fold-body { padding: 0 0 var(--s4); }
.sy-chooser + .sy-modules { margin-top: var(--s4); }
.sy-checks + .sy-checks { margin-top: var(--s3); }
.sy-checks legend { margin: var(--s2) 0 0; color: var(--muted); font-weight: 400; }
.sy-check { padding: 10px 0; font-size: var(--t-sm); }
.sy-check:active { background: var(--accent-wash); }
.sy-results { display: grid; gap: var(--s3); }
.sy-result { min-width: 0; padding: var(--s4) var(--s5); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); overflow-wrap: anywhere; scroll-margin-top: calc(var(--header-height) + var(--s3)); }
.sy-result.is-flag { border-left: 3px solid var(--accent); }
.sy-result h3 { margin: 0; font-size: var(--t-lg); }
.sy-result-meta { margin: 2px 0 var(--s3); color: var(--muted); font-size: var(--t-xs); }
.sy-result-state { margin: 0 0 var(--s3); font-weight: 600; }
.sy-result > p { font-size: var(--t-sm); line-height: 1.55; }
.sy-next { padding: var(--s3) var(--s4); border-radius: var(--r-control); background: var(--accent-wash); }
.sy-result-notes p, .sy-result-note { margin: 0 0 var(--s2); color: var(--muted); font-size: var(--t-xs); line-height: 1.45; }
.sy-result-ref { margin: var(--s2) 0 0; color: var(--muted); font-size: 12px; line-height: 1.4; }
.sy-result-actions { margin-top: var(--s3); padding-top: var(--s2); border-top: 1px solid var(--line); }
.sy-result-actions .text, .sy-data-actions .text, .sy-data-actions .text-link { font-size: var(--t-sm); }
.sy-resources { display: grid; gap: var(--s2); margin: var(--s3) 0 0; padding: 0; list-style: none; }
.sy-resource { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0 var(--s3); min-height: 44px; padding: var(--s2) var(--s3); border: 1px solid var(--line); border-radius: var(--r-control); background: var(--surface); color: var(--ink); text-decoration: none; transition: background-color .12s; }
.sy-resource strong { flex: none; min-width: 4.5rem; color: var(--accent); font: 700 var(--t-lg)/1.25 var(--font-ui); white-space: nowrap; }
.sy-resource span { flex: 1 1 14rem; min-width: 0; font-size: var(--t-sm); line-height: 1.4; }
.sy-resource:active { background: var(--accent-wash); }
.sy-safety-box { margin: 0 0 var(--s6); padding: var(--s4); border-left: 2px solid var(--danger); background: var(--surface); font-size: var(--t-sm); }
.sy-safety-box > p { margin: 0 0 var(--s2); }
.sy-safety-box > button { margin-top: var(--s3); margin-right: var(--s3); }
.sy-safety-stopped { font-weight: 600; }
.sy-quick-exit { margin: 0 0 var(--s4); }
@media (min-width: 600px) { dialog.sy-safety { width: min(560px, calc(100% - 2 * var(--gutter))); } }
dialog.sy-safety h2:focus { outline: none; }
.sy-safety-message, .sy-safety-always { font-size: var(--t-sm); line-height: 1.55; }
.sy-safety-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--s3); margin-top: var(--s5); }
.sy-cross-list { margin: 0; padding: 0; list-style: none; border-bottom: 1px solid var(--line); }
.sy-cross-list li { display: flex; align-items: center; justify-content: space-between; gap: var(--s3); min-height: 48px; padding: var(--s1) 0; border-top: 1px solid var(--line); font-size: var(--t-sm); }
.sy-cross-label { font-weight: 600; }
.sy-cross-link { flex: none; font-size: var(--t-sm); }
.sy-cross-link::after { content: '›' / ''; margin-left: 4px; }
.sy-cross-note, .sy-cross-more { max-width: 62ch; margin: var(--s3) 0 0; color: var(--muted); font-size: var(--t-sm); line-height: 1.5; }
.sy-cross .sy-fold { margin-top: var(--s5); }
.sy-consent-list { margin: 0 0 var(--s2); padding-left: 1.2em; font-size: var(--t-sm); line-height: 1.55; }
.sy-consent-list li + li { margin-top: var(--s2); }
.sy-minor p { max-width: 62ch; }
@media (max-width: 599px) {
  /* Phone: always the list of fine rows. */
  .sy-choices, .sy-choices.is-list { display: flex; flex-direction: column; gap: 0; width: 100%; max-width: none; border: 0; border-radius: 0; background: none; }
  .sy-choices > .sy-choice { justify-content: flex-start; gap: var(--s3); padding: 6px var(--s2); border-bottom: 1px solid color-mix(in srgb, var(--line) 70%, transparent); background: none; text-align: left; box-shadow: none; }
  .sy-choices > .sy-choice:first-child { border-top: 1px solid color-mix(in srgb, var(--line) 70%, transparent); }
  .sy-choices > .sy-choice::before { content: '' / ''; flex: none; box-sizing: border-box; width: 18px; height: 18px; margin: 0; border: 1.5px solid var(--line-strong); border-radius: 50%; background: var(--surface); color: #fff; font: 700 11px/15px var(--font-ui); text-align: center; }
  .sy-choices > .sy-choice.is-on { background: var(--accent-wash); color: var(--ink); box-shadow: none; }
  .sy-choices > .sy-choice.is-on::before { content: '✓' / ''; margin: 0; border-color: var(--accent); background: var(--accent); }
  .sy-result { padding: var(--s4); }
  .sy-safety-actions { flex-direction: column; }
  .sy-safety-actions > button, .sy-safety-box > button { width: 100%; margin-right: 0; }
}
`;
