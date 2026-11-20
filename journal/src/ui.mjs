import {REQUIRED_SYMPTOMS} from './companion-readiness.mjs';
import {CARE_INTERFACE_SCRIPT} from './care-interface.mjs';
import { CHANGE_INTERFACE_SCRIPT, CHANGE_INTERFACE_STYLE } from './change-interface.mjs';
import { HEADER_BRAND_HTML, NAVIGATION, SITE_FOOTER_HTML, SITE_SYSTEM_CSS, SITE_NAVIGATION_SCRIPT } from './navigation.generated.mjs';
import { HOME_INTERFACE_SCRIPT, HOME_INTERFACE_STYLE } from './home-interface.mjs';
import { VOICE_INPUT_SCRIPT } from './voice-input.mjs';
import { WORKSPACE_STYLE } from './workspace-style.mjs';
import { CHAT_INTERFACE_SCRIPT } from './chat-interface.mjs';
import { TOPICS_INTERFACE_SCRIPT, TOPICS_INTERFACE_STYLE } from './topics-interface.mjs';
import { GRAPH_INTERFACE_SCRIPT, GRAPH_INTERFACE_STYLE } from './graph-interface.mjs';
import { REVIEW_ARCHIVE_SCRIPT, REVIEW_ARCHIVE_STYLE } from './review-archive-interface.mjs';
import { PERSON_CHOICES, PERSON_TEXTS, EVENT_CHOICES, EVENT_TEXTS } from './context-fields.mjs';
import { PEOPLE_AXES_UI } from './people-observations-ui.mjs';
const DOCUMENT_HTML = String.raw`<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, interactive-widget=resizes-content"><meta name="theme-color" content="#F2F5F7"><title>Décode ton esprit</title><style>
${SITE_SYSTEM_CSS}
${WORKSPACE_STYLE}
${CHANGE_INTERFACE_STYLE}
${TOPICS_INTERFACE_STYLE}
${GRAPH_INTERFACE_STYLE}
${HOME_INTERFACE_STYLE}
${REVIEW_ARCHIVE_STYLE}
</style></head><body><header class="site-header">
  <div class="header-inner">
    ${HEADER_BRAND_HTML}
  </div>
</header>
<div class="shell">
<main id="main"></main>
</div>
${SITE_FOOTER_HTML}
<dialog id="auth-dialog"><div class="dialog-top"><h2 id="auth-title">Bienvenue.</h2><button type="button" class="close" data-close="auth-dialog" aria-label="Fermer">×</button></div><p class="muted" id="auth-description"></p><form id="auth-form"><div class="field"><label for="auth-email">Adresse e-mail</label><input id="auth-email" name="email" type="email" required autocomplete="email" maxlength="254"></div><div class="field"><label for="auth-password">Mot de passe</label><input id="auth-password" name="password" type="password" required autocomplete="current-password" minlength="12" maxlength="128"><div class="helper" id="auth-password-help">Au moins 12 caractères.</div></div><div id="auth-register-fields" hidden><label class="field" for="auth-role" hidden>Type de compte<select id="auth-role" name="role"><option value="member">Personne accompagnée</option><option value="psychologist">Psychologue</option></select></label><button type="button" class="text" id="auth-role-toggle">Je suis psychologue</button><div id="auth-professional" hidden><p class="helper">Informations professionnelles déclarées, non vérifiées. Les dossiers ne deviennent accessibles qu’après activation d’un code transmis par la personne.</p><label class="field">Nom professionnel<input name="professional_name" maxlength="160"></label><label class="field">Titre professionnel<input name="professional_title" maxlength="160"></label><label class="field">Numéro professionnel<input name="registration_number" maxlength="80"></label></div></div><button id="auth-submit" type="submit">Se connecter</button><p id="auth-status" class="status" role="status"></p></form><p class="auth-switch" id="auth-switch"></p></dialog>
<dialog id="delete-dialog"><div class="dialog-top"><h2>Supprimer mon compte</h2><button type="button" class="close" data-close="delete-dialog" aria-label="Fermer">×</button></div><p>Votre compte et les informations que vous avez enregistrées seront supprimés. L’abonnement sera résilié. Cette action est définitive.</p><p class="muted">Vous pouvez d’abord exporter vos données depuis votre compte.</p><form id="delete-form"><label for="delete-password">Confirmez avec votre mot de passe</label><input type="password" id="delete-password" name="password" autocomplete="current-password" required><div class="actions"><button class="secondary" type="button" data-close="delete-dialog">Annuler</button><button type="submit" class="danger">Supprimer définitivement</button></div><p class="status" id="delete-status" role="status"></p></form></dialog>
<script>
${VOICE_INPUT_SCRIPT}
'use strict';
${SITE_NAVIGATION_SCRIPT}

var sessionRefreshRequest=0;
function clearSessionState(){
 closeReviewArchives();
 resetChatSession();clearTimeout(state.chatTimer);clearTimeout(state.chatCooldownTimer);clearTimeout(state.journeyTimer);state.chatDraft='';state.chatData=null;state.chatRequest=null;state.chatSending=false;state.chatProposalDrafts.clear();state.journeyKey=null;state.journeyWaiting=false;state.journeyStep=null;preparedResponses={};state.navigationRequest=(state.navigationRequest||0)+1;
 state.sourceDrafts.clear();state.selfDraft=null;state.selfBase=null;state.selfRevision=null;if(state.noteDrafts)state.noteDrafts.clear();state.people=[];state.events=[];
}
var PAGES={home:{path:'/accueil/',title:'Accueil'},chat:{path:'/ambre/',title:'Ambre'},people:{path:'/mes-proches/',title:'Mes proches'},person:{path:'/mes-proches/',title:'Mes proches'},graph:{path:'/ma-cartographie/',title:'Ma cartographie'},topics:{path:'/mes-sujets/',title:'Mes sujets'},topic:{path:'/mes-sujets/',title:'Mes sujets'},events:{path:'/ma-ligne-de-vie/',title:'Ma ligne de vie'},reviews:{path:'/mes-bilans/',title:'Mes bilans'},diagnoses:{path:'/mes-diagnostics/',title:'Mes diagnostics'},account:{path:'/mon-compte/',title:'Mon compte'},care:{path:'/mon-suivi/',title:'Mon suivi'},psychologist:{path:'/espace-psychologue/',title:'Espace psychologue'}};
var PAGE_DESCRIPTIONS={
 reviews:'',
 care:'',
 psychologist:'Consultez les dossiers que les personnes ont choisi de partager avec vous.',
 account:''
};
// « Ajouter un moment » on a person page opens Ma ligne de vie with ?ajouter-avec=<id>: the new
// moment's form is open with this person already chosen, and saving returns to the person page.
var MOMENT_FOR_PERSON='ajouter-avec';
function prepareMomentForPerson(){
 var personId=query.get(MOMENT_FOR_PERSON),person=personId&&state.people.find(function(row){return row.id===personId});if(!person)return null;
 var draft=state.sourceDrafts.get('events');
 // A moment being edited keeps its draft: nothing typed is replaced.
 if(draft&&draft.id)return null;
 if(!draft){draft={id:null,creation_id:newClientId(),values:{kind:'positive',date_precision:'unknown',people_ids:[]}};state.sourceDrafts.set('events',draft)}
 var ids=draft.values.people_ids||[];if(ids.indexOf(person.id)<0)draft.values.people_ids=ids.concat(person.id);
 draft.returnTo=person.id;return person
}
function momentReturn(kind,draft){return kind==='events'&&draft&&draft.returnTo&&query.get(MOMENT_FOR_PERSON)===draft.returnTo?draft.returnTo:null}
// One close person has an address of their own: /mes-proches/<id>/.
function personIdForPath(path){var match=String(path).match(/^\/mes-proches\/([\w-]+)\/?$/);return match?match[1]:null}
// One topic of « Mes sujets » has an address of its own: /mes-sujets/<id>/.
function topicIdForPath(path){var match=String(path).match(/^\/mes-sujets\/([\w-]+)\/?$/);return match?match[1]:null}
function pagePath(tab,personId){return tab==='person'?'/mes-proches/'+encodeURIComponent(personId)+'/':tab==='topic'?'/mes-sujets/'+encodeURIComponent(personId)+'/':PAGES[tab].path}
function pageForPath(path){if(personIdForPath(path))return 'person';if(topicIdForPath(path))return 'topic';var normalizedPath=path.replace(/\/+$/,'')+'/';if(normalizedPath==='/mon-portrait/')return 'account';return Object.keys(PAGES).find(function(key){return PAGES[key].path===normalizedPath})||'chat'}
function capturePageDrafts(){var chatInput=document.getElementById('chat-content');if(chatInput)state.chatDraft=chatInput.value;document.querySelectorAll('[data-source-form]').forEach(function(form){captureSourceDraft(form)})}
function syncPageLocation(){query=new URLSearchParams(location.search);state.tab=pageForPath(location.pathname);state.personId=personIdForPath(location.pathname);state.topicId=topicIdForPath(location.pathname);syncPageChrome()}
function syncPageChrome(){document.title='Décode ton esprit';var root=document.documentElement;if(root&&root.dataset){if(state.tab==='home')root.dataset.home='';else delete root.dataset.home}}

function navigationPending(on){var root=document.documentElement;if(root&&root.classList)root.classList.toggle('is-navigating',on);if(!on&&document.querySelectorAll)document.querySelectorAll('[data-pending]').forEach(function(node){node.removeAttribute('data-pending')})}
async function go(tab,search,personId){
 if(tab==='likes'||tab==='portrait'){tab='account';search=(search||'').split('#')[0]+'#self-profile'}
 if(tab==='person'&&!personId)tab='people';
 if(tab==='topic'&&!personId)tab='topics';
 if(!PAGES[tab])return;
 capturePageDrafts();
 var request=state.navigationRequest=(state.navigationRequest||0)+1,owner=state.me&&state.me.user&&state.me.user.id;
 // The current page stays until the destination is ready; past a short wait, a thin bar says it is coming.
 clearTimeout(state.navigationSlow);state.navigationSlow=setTimeout(function(){if(request===state.navigationRequest)navigationPending(true)},180);
 var responses;try{responses=owner?await preparePageData(tab,search,personId):{}}finally{if(request===state.navigationRequest){clearTimeout(state.navigationSlow);navigationPending(false)}}
 if(request!==state.navigationRequest||owner!==(state.me&&state.me.user&&state.me.user.id))return;
 preparedResponses=responses;
 var target=pagePath(tab,personId)+(search||'');
 if(location.pathname+location.search+location.hash!==target)history.pushState({page:tab},'',target);
 syncPageLocation();
 renderApp();
 window.scrollTo({top:0,behavior:'instant'})
}
window.addEventListener('popstate',function(){go(pageForPath(location.pathname),location.search+location.hash,personIdForPath(location.pathname)||topicIdForPath(location.pathname))});
document.addEventListener('click',function(event){var link=event.target.closest('a[href]');if(!link||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||link.target==='_blank'||link.hasAttribute('download'))return;var url=new URL(link.href,location.href);if(url.origin!==location.origin)return;var personId=personIdForPath(url.pathname),topicId=topicIdForPath(url.pathname),page=personId?'person':topicId?'topic':Object.keys(PAGES).find(function(key){return PAGES[key].path===url.pathname.replace(/\/+$/,'')+'/'});if(!page)return;event.preventDefault();navigationPending(false);link.setAttribute('data-pending','');go(page,url.search+url.hash,personId||topicId)});

var initialPageData=readInitialPageData();
function readInitialPageData(){var node=document.getElementById('initial-page-data');if(!node)return null;try{return JSON.parse(node.textContent)}catch(error){return null}}
var preparedResponses=initialPageData&&initialPageData.page===location.pathname.replace(/\/+$/,'')+'/'?initialPageData.responses||{}:{};
async function preparePageData(tab,search,personId){
 var paths={account:['/api/self-profile'],people:['/api/people'],topics:['/api/topics'],graph:['/api/graph'],events:['/api/people','/api/life-events']}[tab]||[];
 if(tab==='person'){var base='/api/people/'+encodeURIComponent(personId);paths=[base,base+'/timeline']}
 if(tab==='topic')paths=['/api/topics/'+encodeURIComponent(personId)];
 if(tab==='events'){var filter=new URLSearchParams((search||'').split('#')[0]).get('person');if(filter)paths[1]+='?person_id='+encodeURIComponent(filter)}
 var entries=await Promise.all(paths.map(async function(path){try{return [path,{status:200,body:await api(path)}]}catch(error){return [path,{status:error.status||503,body:{error:error.message,code:error.code}}]}}));return Object.fromEntries(entries)
}
var query=new URLSearchParams(location.search);
var state={chatOutbox:null,chatSeenResponses:new Set(),chatDraft:'',chatProposalDrafts:new Map(),chatSendKey:null,chatSending:false,me:null,tab:pageForPath(location.pathname),personId:personIdForPath(location.pathname),topicId:topicIdForPath(location.pathname),authMode:'login',people:[],events:[],sourceDrafts:new Map(),selfDraft:null,selfRevision:null,mapReturnDrafts:new Map()};
var main=document.getElementById('main');
function el(tag,cls,text){var n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n}
function actionButton(text,fn,cls){var b=el('button',cls||'',text);b.type='button';b.addEventListener('click',fn);return b}
// Opening a form or a window: a field gets the focus only with a mouse or trackpad. A touch screen
// gets the title instead (tabindex -1, no scrolling jump), so the keyboard never opens by itself.
function finePointer(){try{return typeof matchMedia==='function'&&matchMedia('(hover: hover) and (pointer: fine)').matches}catch(error){return false}}
function focusEntry(field,heading){if(field&&finePointer()){field.focus();return}if(!heading)return;if(heading.getAttribute('tabindex')==null&&!(heading.tabIndex>=0))heading.setAttribute('tabindex','-1');heading.focus({preventScroll:true})}
// One site window replaces the browser's confirm(): it resolves true only on the confirm button.
function confirmDialog(title,text,confirmLabel,danger){return new Promise(function(resolve){var dialog=el('dialog','confirm-dialog'),heading=el('h2','',title),actions=el('div','actions'),answer=false,opener=document.activeElement;heading.id='confirm-dialog-title';dialog.setAttribute('aria-labelledby',heading.id);dialog.append(heading);if(text){var body=el('p','',text);body.id='confirm-dialog-text';dialog.setAttribute('aria-describedby',body.id);dialog.append(body)}var cancel=actionButton('Annuler',function(){dialog.close()},'secondary'),confirm=actionButton(confirmLabel||'Confirmer',function(){answer=true;dialog.close()},danger?'danger':'');confirm.dataset.confirm='true';actions.append(cancel,confirm);dialog.append(actions);dialog.addEventListener('close',function(){dialog.remove();if(opener&&opener.isConnected&&opener.focus)opener.focus();resolve(answer)});document.body.append(dialog);dialog.showModal();cancel.focus()})}
function status(id,message,error){var n=document.getElementById(id);if(n){n.textContent=message;n.className='status '+(error?'error':'success')}}
function busy(form,yes){form.querySelectorAll('button[type=submit]').forEach(function(b){b.disabled=yes})}
function formValues(form){return Object.fromEntries(Array.from(new FormData(form).entries()).filter(function(entry){return entry[0]}))}
async function api(path,method,data){
 if((!method||method==='GET')&&Object.prototype.hasOwnProperty.call(preparedResponses,path)){
  var ready=preparedResponses[path];delete preparedResponses[path];
  if(ready.status>=400){var failure=new Error(ready.body&&ready.body.error||'La demande n’a pas abouti.');failure.status=ready.status;failure.code=ready.body&&ready.body.code;throw failure}return ready.body
 }
 var options={method:method||'GET',credentials:'same-origin',headers:{},cache:'no-store'};if(data!==undefined){options.headers['Content-Type']='application/json';options.body=JSON.stringify(data)}var response=await fetch(path,options);var result;try{result=await response.json()}catch(e){result={}}if(!response.ok){var error=new Error(typeof result.error==='string'?result.error:'La demande n’a pas abouti. Réessayez dans un instant.');error.code=result.code;error.status=response.status;throw error;}return result}
function normalized(value){return String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase()}
function dateLabel(value){var d=new Date(typeof value==='number'&&value<100000000000?value*1000:String(value).length===10?value+'T12:00:00':value);return isNaN(d)?'':d.toLocaleDateString('fr-FR',{day:'numeric',month:'long',year:'numeric'})}
function landing(){
 syncPageChrome();
 // Plan 19/11 lot 3: a visitor sees Ambre's first message (without counters) and signs in from it.
 if(state.tab==='chat'&&typeof drawChat==='function'){main.className='conversation-page';main.innerHTML='<div id="app-messages"></div><div id="view"></div>';var guestView=document.getElementById('view');guestView.chatShell=null;state.chatData={access:{allowed:false,reason:'authentication_required'},messages:[],proposals:[]};drawChat(guestView,renderGuard(guestView));return}
 if(state.tab==='home'){main.className='';main.innerHTML='<div class="app-head"><h1>Accueil</h1></div><div id="app-messages"></div><div id="view"></div>';renderHome();return}
 main.className='';
 main.innerHTML='<section class="journal-welcome"><h1 id="public-page-title"></h1><p>Connectez-vous pour ouvrir votre espace.</p><div class="actions"><button type="button" data-auth="login">Me connecter</button><button type="button" class="secondary" data-auth="register">Créer mon compte</button></div></section>';
 document.getElementById('public-page-title').textContent=PAGES[state.tab].title;
}
function renderPageLoading(){
 syncPageChrome();main.className=state.tab==='chat'?'conversation-page':'';
 var content=el('section','panel narrow personal-section'),loading=el('p','muted page-loading','Chargement…');loading.setAttribute('role','status');content.setAttribute('aria-busy','true');content.append(loading);
 main.replaceChildren();if(state.tab!=='chat'){var head=el('div','app-head');head.append(el('h1','',PAGES[state.tab].title));main.append(head)}main.append(content)
}
async function refresh(){var request=++sessionRefreshRequest,me=await api('/api/me');if(request!==sessionRefreshRequest)return;if(state.me&&state.me.user&&(!me.user||state.me.user.id!==me.user.id))clearSessionState();state.me=me;syncRoleNavigation();if(state.me.user)await renderApp();else landing();}
function syncRoleNavigation(){}
function updateAuthRole(){var register=state.authMode==='register',role=document.getElementById('auth-role'),fields=document.getElementById('auth-register-fields'),professional=document.getElementById('auth-professional');fields.hidden=!register;professional.hidden=!register||role.value!=='psychologist';var toggle=document.getElementById('auth-role-toggle');toggle.textContent=role.value==='psychologist'?'Je ne suis pas psychologue':'Je suis psychologue';document.getElementById('auth-password-help').hidden=!register;professional.querySelectorAll('input').forEach(function(input){input.required=!professional.hidden;input.disabled=professional.hidden})}
function openAuth(mode){state.authMode=mode;updateAuthRole();var register=mode==='register';document.getElementById('auth-title').textContent=register?'Créer mon compte':'Me connecter';document.getElementById('auth-description').textContent='';document.getElementById('auth-submit').textContent=register?'Créer mon compte':'Se connecter';document.getElementById('auth-password').autocomplete=register?'new-password':'current-password';document.getElementById('auth-password').minLength=12;document.getElementById('auth-switch').replaceChildren(document.createTextNode(register?'Déjà un compte ? ':'Pas encore de compte ? '),actionButton(register?'Se connecter':'Créer un compte',function(){openAuth(register?'login':'register')},'text'));status('auth-status','');var dialog=document.getElementById('auth-dialog');if(!dialog.open){dialog.showModal();focusEntry(document.getElementById('auth-email'),document.getElementById('auth-title'))}}
function renderApp(){
 closeReviewArchives();
 disposeChatInterface();
 clearTimeout(state.portraitTimer);
 clearTimeout(state.chatTimer);
 clearTimeout(state.chatCooldownTimer);
 clearTimeout(state.journeyTimer);
 if(typeof closeJourneyPanel==='function'){state.journeyStep=null;closeJourneyPanel()}
 state.chatSending=false;
 state.chatRequest=null;
 if(!state.me){renderPageLoading();return}
 if(!state.me.user){landing();return}
 syncRoleNavigation();
 if(state.me.user.role==='psychologist'&&state.tab==='chat'){state.tab='psychologist';history.replaceState({},'','/espace-psychologue/')}
 syncPageChrome();
 var chat=state.tab==='chat';
 main.className=chat?'conversation-page':'';
 main.innerHTML=(chat?'':'<div class="app-head"><div><h1 id="private-page-title"></h1><p class="page-description" id="page-description"></p></div></div>')+'<div id="app-messages"></div><div id="view"></div>';
 if(!chat){
  document.getElementById('private-page-title').textContent=PAGES[state.tab].title;
  var description=document.getElementById('page-description');
  description.textContent=PAGE_DESCRIPTIONS[state.tab]||'';
  description.hidden=!description.textContent
 }
 var messages=document.getElementById('app-messages');
 if(chat&&state.me.configuration&&state.me.configuration.ai===false){
  messages.append(el('div','notice warn','Ambre est momentanément indisponible. Vos informations enregistrées restent accessibles.'))
 }
 if(state.tab==='care')renderCareMember(document.getElementById('view'),renderGuard(document.getElementById('view')));
 else if(state.tab==='psychologist')renderCarePsychologist(document.getElementById('view'),renderGuard(document.getElementById('view')));
 else if(chat)return renderChat();
 else if(state.tab==='account')renderAccount();
 else if(state.tab==='reviews')renderReviews();
 else if(state.tab==='diagnoses')renderDiagnoses();
 else if(state.tab==='person')renderPerson();
 else if(state.tab==='home')renderHome();
 else if(state.tab==='graph')renderGraph();
 else if(state.tab==='topics')renderTopics();
 else if(state.tab==='topic')renderTopic();
 else renderPersonalSources()
}

var renderTokens=new WeakMap();
function renderGuard(node){var token={};renderTokens.set(node,token);return function(){return node.isConnected&&renderTokens.get(node)===token}}

async function renderSelfProfile(){
 var host=document.getElementById('self-profile'),current=renderGuard(host);
 var loading=el('p','muted page-loading','Chargement de vos informations…');loading.setAttribute('role','status');host.replaceChildren(loading);host.setAttribute('aria-busy','true');
 try{
  var result=await api('/api/self-profile');if(!current())return;
  var fields=[['first_name','Prénom ou pseudonyme'],['last_name','Nom'],['birth_date','Date de naissance'],['gender','Genre'],['gender_description','Votre genre, avec vos mots']];
  function valuesOf(profile){var values={};fields.forEach(function(field){values[field[0]]=profile&&profile[field[0]]||''});return values}
  var savedValues=valuesOf(result.profile),sending=false;
  if(!state.selfDraft){state.selfRevision=result.revision;state.selfBase=savedValues}
  if(!state.selfBase)state.selfBase=savedValues;
  var values=Object.assign({},savedValues,state.selfDraft||{}),form=el('form');form.id='self-profile-form';
  var identity=el('div','form-grid self-identity-fields');identity.id='account-identity';
  sourceField(identity,'first_name','Prénom ou pseudonyme',values.first_name,{max:100});
  sourceField(identity,'last_name','Nom',values.last_name,{max:100});
  var birth=sourceField(identity,'birth_date','Date de naissance',values.birth_date,{});birth.type='date';
  var genderChoices=[['','—'],['female','Femme'],['male','Homme'],['non_binary','Non binaire'],['self_described','Je le décris avec mes mots'],['unspecified','Je préfère ne pas le préciser']];
  var gender=sourceField(identity,'gender','Genre',values.gender,{choices:genderChoices});
  var genderDescription=sourceField(identity,'gender_description','Votre genre, avec vos mots',values.gender_description,{max:200,full:true});
  function showGenderDescription(){genderDescription.parentElement.hidden=gender.value!=='self_described'}
  gender.addEventListener('change',showGenderDescription);genderDescription.addEventListener('input',showGenderDescription);showGenderDescription();
  var actions=el('div','actions'),save=el('button','','Enregistrer'),message=el('p','status'),conflict=el('div');
  save.type='submit';message.id='self-profile-status';message.setAttribute('role','status');conflict.className='self-profile-conflict';actions.append(save);form.append(identity,actions,message,conflict);host.replaceChildren(form);host.removeAttribute('aria-busy');
  function remember(){var values=formValues(form);state.selfDraft=JSON.stringify(values)===JSON.stringify(state.selfBase)?null:values}
  function changes(values){var changed={};fields.forEach(function(field){var key=field[0];if(values[key]!==state.selfBase[key])changed[key]=values[key]});return changed}
  form.addEventListener('input',remember);form.addEventListener('change',remember);
  form.addEventListener('submit',async function(event){
   event.preventDefault();if(sending)return;remember();var submitted=formValues(form),payload=changes(submitted);payload.expected_revision=state.selfRevision;
   sending=true;save.disabled=true;status(message.id,'Enregistrement…');
   try{
    // Only edited visible fields are patched. Historical paragraphs stay untouched.
    var saved=await api('/api/self-profile','PUT',payload);if(!current())return;
    state.selfRevision=saved.revision;state.selfBase=valuesOf(saved.profile);
    fields.forEach(function(field){var key=field[0];if(form.elements[key].value===submitted[key])form.elements[key].value=state.selfBase[key]});showGenderDescription();
    remember();conflict.replaceChildren();sourceSaved(saved);
    status(message.id,state.selfDraft?'Enregistré. Les modifications saisies pendant l’enregistrement restent à sauvegarder.':'Informations enregistrées.');
   }catch(err){
    if(!current())return;
    status(message.id,err.status===409?'Vos informations ont été modifiées ailleurs. Votre brouillon est conservé. Comparez la version enregistrée avant de réessayer.':err.message,true);
    if(err.status===409){
     var compare=actionButton('Comparer la version enregistrée',async function(){
      compare.disabled=true;
      try{
       var latest=await api('/api/self-profile');if(!current())return;
       var comparison=el('div','notice'),stored=valuesOf(latest.profile);
       comparison.append(el('h3','','Informations actuellement enregistrées'));
       fields.forEach(function(field){var text=stored[field[0]];if(field[0]==='gender')text=(genderChoices.find(function(choice){return choice[0]===text})||[text,text])[1];comparison.append(el('p','self-profile-saved',field[1]+' : '+(text||'—')))});
       comparison.append(actionButton('Conserver mon brouillon pour un nouvel enregistrement',function(){
        var edited=changes(formValues(form));fields.forEach(function(field){var key=field[0];if(!Object.prototype.hasOwnProperty.call(edited,key))form.elements[key].value=stored[key]});
        state.selfRevision=latest.revision;state.selfBase=stored;showGenderDescription();remember();conflict.replaceChildren();
        status(message.id,'Votre brouillon est conservé. Vérifiez vos informations, puis enregistrez pour remplacer les champs modifiés.');
       },'secondary'));conflict.replaceChildren(comparison);
      }catch(error){if(current()){compare.disabled=false;status(message.id,error.message,true)}}
     },'secondary');compare.dataset.reloadSource='true';conflict.replaceChildren(compare);
    }
   }finally{sending=false;save.disabled=false}
  });
 }catch(err){if(current()){host.removeAttribute('aria-busy');host.replaceChildren(el('p','status error',err.message),actionButton('Recharger mes informations',renderSelfProfile,'secondary'))}}
}

var PERSON_ROLES=[['father','Père'],['mother','Mère'],['parent','Parent'],['partner','Partenaire'],['child','Enfant'],['sibling','Frère ou sœur'],['friend','Ami ou amie'],['best_friend','Meilleur ami ou meilleure amie'],['other','Autre']];
var SOURCE_CONTEXT_CHOICES=${JSON.stringify({people:PERSON_CHOICES,events:EVENT_CHOICES})};
var SOURCE_CONTEXT_LIMITS=${JSON.stringify({people:PERSON_TEXTS,events:EVENT_TEXTS})};
var SOURCE_CONTEXT_LABELS={bond_quality:'Qualité du lien',bond_note:'Une nuance sur ce lien',presence_now:'Présence aujourd’hui',helps:'Qu’est-ce qui vous aide dans cette relation ?',hurts:'Qu’est-ce qui vous pèse ?',evolution:'Comment le lien a-t-il évolué ?',feels_safe:'Vous sentez-vous en sécurité avec cette personne ?',felt_then:'Ce que j’ai ressenti alors',felt_now:'Ce que je ressens aujourd’hui en y repensant',impact:'Ce que cela a changé pour moi',support_level:'Soutien reçu',support_text:'Ce soutien, avec mes mots',exposure:'Comment ce moment me concerne'};
var SOURCE_CONTEXT_ORDER={people:['bond_quality','bond_note','presence_now','helps','hurts','evolution','feels_safe'],events:[]};
var EVENT_KINDS=[['positive','Moment positif'],['difficult','Moment difficile'],['turning_point','Déclic'],['ambivalent','Moment ambivalent'],['other','Autre']];
var SIGNIFICANCE_CHOICES=[['anecdotal','Anecdotique'],['major','Marquant']];
function newClientId(){return crypto.randomUUID()}
function captureSourceDraft(form){var kind=form.dataset.sourceForm,previous=state.sourceDrafts.get(kind);if(!previous)return;var values=formValues(form);if(!state[kind+'FieldsReady'])SOURCE_CONTEXT_ORDER[kind].forEach(function(name){if(Object.prototype.hasOwnProperty.call(previous.values,name))values[name]=previous.values[name]});if(kind==='events'){values.people_ids=Array.from(form.querySelectorAll('[name=people_ids]:checked')).map(function(input){return input.value});if(form.querySelector('[name=significance]')){var significance=chosenSignificance(form);if(significance)values.significance=significance;else delete values.significance}}state.sourceDrafts.set(kind,Object.assign({},previous,{values:values}))}
function sourceField(form,name,label,value,options){options=options||{};var field=el('div','field'+(options.full?' full':'')),caption=el('label','',label),input=el(options.choices?'select':options.multiline?'textarea':'input');input.id=form.id+'-'+name;input.name=name;caption.htmlFor=input.id;if(options.choices){options.choices.forEach(function(choice){var option=el('option','',choice[1]);option.value=choice[0];input.append(option)})}else{if(!options.multiline)input.type=options.type||'text';if(options.max)input.maxLength=options.max;if(options.placeholder)input.placeholder=options.placeholder;}if(options.required)input.required=true;input.value=value==null?'':String(value);field.append(caption,input);if(options.help){var help=el('small','field-hint',options.help);help.id=input.id+'-help';input.setAttribute('aria-describedby',help.id);field.append(help)}form.append(field);return input}
function monthNames(){return ['janvier','février','mars','avril','mai','juin','juillet','août','septembre','octobre','novembre','décembre']}
// A date is chosen in lists, never typed: a day, a month, a year, or a period whose two ends each
// have their own precision. The hidden fields keep the stored format (YYYY, YYYY-MM, YYYY-MM-DD).
function appendEventDateFields(grid,values,draft,prefix){
 prefix=prefix||'';var key=function(name){return prefix+name},stored=values[key('date_precision')]||'unknown';
 var modes=[['unknown','Sans date'],['day','Un jour précis'],['month','Un mois'],['year','Une année'],['range','Une période']];
 // Earlier entries could hold an age written in words: it is kept as it is until another choice is made.
 if(stored==='age')modes.push(['age','Un âge (ancienne saisie)']);
 var precision=sourceField(grid,key('date_precision'),prefix?'Date de rencontre':'Date',stored,{choices:modes});
 var start=sourceField(grid,key('date_start'),'Date',values[key('date_start')],{type:'hidden'}),end=sourceField(grid,key('date_end'),'Fin',values[key('date_end')],{type:'hidden'}),age=sourceField(grid,key('age_text'),'Âge',values[key('age_text')],{type:'hidden'});
 [start,end,age].forEach(function(input){input.parentElement.hidden=true});
 var ageNote=el('p','helper full date-age-note',age.value?'Âge indiqué : '+age.value:'');ageNote.hidden=true;grid.append(ageNote);
 function picker(caption,target,own){
  var box=el('fieldset','field full date-picker'),legend=el('legend','',caption),row=el('div','date-picker-row'),unit=el('select'),day=el('select'),month=el('select'),year=el('select'),value=target.value||'',mode='year';
  var dayOptions=[];function option(select,v,text){var o=el('option','',text);o.value=v;select.append(o);return o}
  [['day','Au jour près'],['month','Au mois près'],['year','À l’année près']].forEach(function(entry){option(unit,entry[0],entry[1])});
  option(day,'','Jour');for(var d=1;d<=31;d++)dayOptions.push(option(day,String(d).padStart(2,'0'),String(d)));
  option(month,'','Mois');monthNames().forEach(function(name,index){option(month,String(index+1).padStart(2,'0'),name)});
  var thisYear=new Date().getFullYear(),first=Math.min(thisYear-110,Number(value.slice(0,4))||thisYear);option(year,'','Année');for(var y=thisYear;y>=first;y--)option(year,String(y),String(y));
  unit.setAttribute('aria-label',caption+' : précision');day.setAttribute('aria-label',caption+' : jour');month.setAttribute('aria-label',caption+' : mois');year.setAttribute('aria-label',caption+' : année');
  unit.value=value.length===10?'day':value.length===7?'month':'year';year.value=value.slice(0,4);month.value=value.slice(5,7);day.value=value.slice(8,10);
  function refresh(){
   var unitNow=own?unit.value:mode,shown=!box.hidden;unit.hidden=!own;month.hidden=unitNow==='year';day.hidden=unitNow!=='day';
   year.required=shown;month.required=shown&&unitNow!=='year';day.required=shown&&unitNow==='day';
   var days=year.value&&month.value?new Date(Number(year.value),Number(month.value),0).getDate():31;
   dayOptions.forEach(function(o){o.hidden=Number(o.value)>days;o.disabled=o.hidden});if(Number(day.value)>days)day.value='';
   if(!shown)return;
   target.value=!year.value?'':unitNow==='year'?year.value:!month.value?'':unitNow==='month'?year.value+'-'+month.value:!day.value?'':year.value+'-'+month.value+'-'+day.value;
  }
  [unit,day,month,year].forEach(function(select){select.addEventListener('change',refresh)});
  row.append(unit,day,month,year);box.append(legend,row);grid.append(box);
  return {box:box,legend:legend,show:function(visible,unitMode){box.hidden=!visible;if(unitMode)mode=unitMode;refresh()}};
 }
 var from=picker('Date',start,false),range=picker('Début de la période',start,true),to=picker('Fin de la période',end,true);
 var ongoingField=el('label','check full date-ongoing'),ongoing=el('input');ongoing.type='checkbox';ongoing.checked=stored==='range'&&!!start.value&&!end.value;ongoingField.append(ongoing,el('span','','Cette période est toujours en cours'));grid.append(ongoingField);
 function show(){
  var mode=precision.value,isRange=mode==='range',single=['day','month','year'].includes(mode);
  from.show(single,single?mode:null);range.show(isRange);ongoingField.hidden=!isRange;to.show(isRange&&!ongoing.checked);
  if(!isRange||ongoing.checked)end.value='';
  if(!single&&!isRange)start.value=mode==='age'?start.value:'';
  ageNote.hidden=mode!=='age'||!age.value;
 }
 precision.addEventListener('change',show);ongoing.addEventListener('change',show);show();
}
// « 2 ans et 3 mois », « 5 mois », « 12 jours »: how long a period lasted, from its two partial dates.
function periodLength(startValue,endValue){
 var from=new Date((startValue+'-01-01').slice(0,10)+'T12:00:00Z'),until=endValue?new Date((endValue.length===4?endValue+'-12-31':endValue.length===7?endValue+'-'+String(new Date(Date.UTC(Number(endValue.slice(0,4)),Number(endValue.slice(5,7)),0)).getUTCDate()).padStart(2,'0'):endValue)+'T12:00:00Z'):new Date();
 var days=Math.round((until-from)/86400000)+1;if(!Number.isFinite(days)||days<1)return '';
 if(days<60)return days+(days>1?' jours':' jour');
 var next=new Date(until.getTime()+86400000),months=(next.getUTCFullYear()-from.getUTCFullYear())*12+next.getUTCMonth()-from.getUTCMonth()-(next.getUTCDate()<from.getUTCDate()?1:0);if(months<24)return months+' mois';
 var years=Math.floor(months/12),rest=months%12;return years+' ans'+(rest?' et '+rest+' mois':'');
}
function appendSourceContextFields(grid,kind,values){
 SOURCE_CONTEXT_ORDER[kind].forEach(function(name){var choices=SOURCE_CONTEXT_CHOICES[kind][name],options=choices?{choices:[['','—']].concat(Object.entries(choices))}:{multiline:true,max:SOURCE_CONTEXT_LIMITS[kind][name],full:true};sourceField(grid,name,SOURCE_CONTEXT_LABELS[name],values[name],options)})
}
function appendSourceContextText(host,kind,item){
 var rows=SOURCE_CONTEXT_ORDER[kind].filter(function(name){return item[name]});if(!rows.length)return;
 var section=el('div','source-context');rows.forEach(function(name){var choices=SOURCE_CONTEXT_CHOICES[kind][name],text=choices?choices[item[name]]:item[name];if(!text)return;var row=el('div','source-context-row');row.append(el('h4','',SOURCE_CONTEXT_LABELS[name]),el('p','',text));section.append(row)});host.append(section)
}
function displayDate(value){
 if(!value||!/^\d{4}(-\d{2}){0,2}$/.test(value))return value||'';
 if(value.length===4)return value;
 var date=new Date(value+(value.length===7?'-01':'')+'T12:00:00Z');
 if(!Number.isFinite(date.getTime()))return value;
 var options={month:'long',year:'numeric',timeZone:'UTC'};
 if(value.length===10)options.day='numeric';
 return date.toLocaleDateString('fr-FR',options);
}
// Under a year marker of the life line, the year is not repeated.
function eventDateInYear(event){
 if(event.date_precision==='unknown'||!event.date_start&&event.date_precision!=='age')return '';
 if(event.date_precision==='year'&&!event.date_end)return '';
 if(['month','day'].includes(event.date_precision)&&!event.date_end&&/^\d{4}-\d{2}(-\d{2})?$/.test(event.date_start)){
  var date=new Date(event.date_start+(event.date_start.length===7?'-01':'')+'T12:00:00Z');
  if(Number.isFinite(date.getTime())){var options={month:'long',timeZone:'UTC'};if(event.date_start.length===10)options.day='numeric';return date.toLocaleDateString('fr-FR',options)}
 }
 if(event.date_precision==='range'){var length=periodLength(event.date_start,event.date_end);return sourceDate(event)+(length?' · '+length:'')}
 return sourceDate(event);
}
function sourceDate(event){
 if(event.date_precision==='age')return event.age_text||'Âge non précisé';
 if(event.date_precision==='unknown'||!event.date_start)return 'Sans date';
 if(event.date_precision==='range'&&!event.date_end)return 'Depuis '+displayDate(event.date_start);
 return displayDate(event.date_start)+(event.date_end?' – '+displayDate(event.date_end):'');
}
function emptyState(kind,title,description){
 var icons={portrait:'<circle cx="12" cy="8" r="3"/><path d="M5 20v-2a7 5 0 0 1 14 0v2M3 3h3m12 0h3v3M3 18v3h3m12 0h3v-3"/>',people:'<circle cx="9" cy="8" r="3"/><path d="M3 20v-2a6 4 0 0 1 12 0v2m1-15a3 3 0 0 1 0 6m2 3a4 4 0 0 1 3 4v2"/>'};
 var empty=el('div','workspace-empty');
 if(icons[kind]){
  var glyph=el('span','empty-glyph');
  glyph.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+icons[kind]+'</svg>';
  empty.append(glyph)
 }
 empty.append(el('h2','',title),el('p','',description));return empty
}
// After a deletion the list is redrawn: focus returns to the add button instead of being lost.
function sourceSaved(result,focusAdd){status('source-page-status','Enregistré.');if(state.tab==='events')renderPersonalSources().then(function(){if(state.tab==='events'){status('source-page-status','Enregistré.');var add=focusAdd&&document.querySelector('#events-editor summary');if(add)add.focus()}});}
async function renderPersonalSources(){var holder=document.getElementById('view'),current=renderGuard(holder),page=state.tab;var kind=page==='people'?'people':'events';holder.innerHTML='<section class="panel narrow personal-section source-panel"><p class="status" id="source-page-status" role="status"></p><div id="'+kind+'-content"><p class="muted page-loading" role="status">Chargement…</p></div></section>';try{var requests=kind==='people'?[api('/api/people')]:[api('/api/people'),api('/api/life-events'+eventsListQuery(''))];var results=await Promise.all(requests);if(!current()||state.tab!==page)return;state.people=results[0].people||[];state.peopleFieldsReady=results[0].fields_ready===true;if(kind==='events'){state.events=results[1].events||[];state.eventsFieldsReady=results[1].fields_ready===true;state.eventsSignificanceReady=results[1].significance_ready===true;state.eventsCursor=results[1].next_cursor}var momentPerson=kind==='events'&&prepareMomentForPerson();drawSourceSection(kind);if(momentPerson){var momentForm=document.querySelector('[data-source-form="events"]');if(momentForm){focusEntry(momentForm.querySelector('input,select,textarea'),momentForm.parentElement.querySelector('summary'))}}
 var prefix=kind==='people'?'#person-':'#event-';
 if(location.hash.startsWith(prefix)||query.get(kind==='people'?'person':'event')){
  var id=query.get(kind==='people'?'person':'event')||location.hash.slice(prefix.length);
  try{id=decodeURIComponent(id)}catch{}
  var key=prefix.slice(1)+id;
  if(kind==='events'&&!document.getElementById(key)){
   var linked=await api('/api/life-events/'+encodeURIComponent(id)).catch(function(){return null});
   if(!current())return;
   if(linked&&linked.event){state.events.push(linked.event);drawSourceSection(kind)}
  }
  var entry=document.getElementById(key);
  if(entry){entry.scrollIntoView({block:'nearest'});entry.focus({preventScroll:true})}
 }
}catch(err){if(current())document.getElementById(kind+'-content').replaceChildren(el('p','status error',err.message),actionButton('Réessayer',renderPersonalSources,'secondary'))}}

function drawSourceSection(kind){
 var box=document.getElementById(kind+'-content');if(!box)return;
 box.replaceChildren();
 var isPeople=kind==='people',rows=state[kind].slice(),toolbar=el('div','source-toolbar'),editor=el('div','source-editor-host');
 editor.id=kind+'-editor';
 var count=rows.length;
 if(!isPeople&&query.get('person')){var selected=state.people.find(function(person){return person.id===query.get('person')});if(selected)toolbar.append(el('span','source-meta','Avec '+selected.display_name));var all=el('a','text-link','Tous les événements');all.href='/ma-ligne-de-vie/';toolbar.append(all)}
 var back=!isPeople&&momentReturn(kind,state.sourceDrafts.get(kind)),backPerson=back&&state.people.find(function(person){return person.id===back});if(backPerson){var backLink=el('a','text-link moment-back','Retour à '+backPerson.display_name);backLink.href=pagePath('person',back);toolbar.append(backLink)}
 box.append(toolbar,editor);
 drawSourceEditor(kind);
 if(!count){
  box.append(isPeople?emptyState('people','Aucun proche pour l’instant','Ajoutez une personne qui compte pour vous. Un prénom ou un pseudonyme suffit.'):emptyState('events','Aucun moment pour l’instant','Ajoutez un souvenir, un changement ou un déclic. La date peut rester approximative.'));
  return
 }
 if(isPeople){
  var directory=el('div','people-directory');
  rows.sort(function(a,b){return a.display_name.localeCompare(b.display_name,'fr')});
  rows.forEach(function(item){directory.append(drawSourceItem(kind,item,box))});
  box.append(directory)
 }else{
  var pending=[],hidden=0,mode=state.momentFilter==='major'?'major':'all';
  if(state.eventsSignificanceReady){
   toolbar.append(momentFilter(mode,async function(next){var open=document.querySelector('[data-source-form="events"]');if(open)captureSourceDraft(open);var previous=state.momentFilter;state.momentFilter=next;toolbar.querySelectorAll('.moment-filter button').forEach(function(button){button.disabled=true});
    // The API applies the filter, so that the pagination follows it.
    try{var data=await api('/api/life-events'+eventsListQuery(''));if(!box.isConnected||state.tab!=='events')return;state.events=data.events||[];state.eventsCursor=data.next_cursor}catch(err){state.momentFilter=previous;toolbar.querySelectorAll('.moment-filter button').forEach(function(button){button.disabled=false});status('source-page-status',err.message);return}
    drawSourceSection('events');var pressed=document.querySelector('.moment-filter button[aria-pressed="true"]');if(pressed)pressed.focus()},'Filtrer les moments'));
   if(mode==='major'){pending=rows.filter(function(item){return item.significance!=='major'&&item.significance!=='anecdotal'});hidden=rows.filter(function(item){return item.significance==='anecdotal'}).length;rows=rows.filter(function(item){return item.significance==='major'})}
  }
  var dated=rows.filter(function(item){return item.date_precision!=='age'&&item.date_precision!=='unknown'&&item.date_start});
  var undated=rows.filter(function(item){return !dated.includes(item)});
  dated.sort(function(a,b){return b.date_start.localeCompare(a.date_start)});
  if(dated.length){
   // Each year lists first what is only dated by the year, then its months, most recent first.
   // A period is drawn where it starts and recalled in every later year it runs through.
   var timeline=el('div','event-timeline'),today=new Date().toISOString().slice(0,10),years=[];
   dated.forEach(function(item){var y=item.date_start.slice(0,4);if(years.indexOf(y)<0)years.push(y)});
   years.forEach(function(year){
    timeline.append(el('h3','timeline-year',year));
    var running=dated.filter(function(item){return item.date_precision==='range'&&item.date_start.slice(0,4)<year&&(item.date_end||today).slice(0,4)>=year});
    if(running.length){var recall=el('p','timeline-running','Se poursuit cette année : ');running.forEach(function(item,index){var link=el('a','',item.title);link.href='#event-'+item.id;recall.append(link);if(index<running.length-1)recall.append(', ')});timeline.append(recall)}
    var inYear=dated.filter(function(item){return item.date_start.slice(0,4)===year}),whole=inYear.filter(function(item){return item.date_start.length===4}),rest=inYear.filter(function(item){return item.date_start.length>4});
    if(whole.length&&rest.length)timeline.append(el('p','timeline-month','Au cours de l’année'));
    whole.forEach(function(item){timeline.append(drawSourceItem(kind,item,box))});
    var month='';rest.forEach(function(item){var m=item.date_start.slice(5,7);if(m!==month){month=m;var name=monthNames()[Number(m)-1]||'';timeline.append(el('p','timeline-month',name.charAt(0).toUpperCase()+name.slice(1)))}timeline.append(drawSourceItem(kind,item,box))});
   });
   box.append(timeline)
  }
  if(undated.length){
   var section=el('section','undated-events');section.append(el('h3','timeline-year','Moments sans date précise'));
   undated.forEach(function(item){section.append(drawSourceItem(kind,item,box))});
   box.append(section)
  }
  if(state.eventsSignificanceReady&&mode==='major'){
   if(!rows.length)box.append(el('p','muted moment-filter-empty','Aucun moment noté « Marquant » pour l’instant.'));
   if(pending.length){var later=el('section','undated-events pending-moments');later.append(el('h3','timeline-year','À préciser'),el('p','helper','Ces moments ne sont pas encore classés. Indiquez s’ils sont marquants dans votre histoire.'));pending.forEach(function(item){later.append(drawSourceItem(kind,item,box))});box.append(later)}
   if(hidden)box.append(el('p','muted moment-filter-hidden','Les moments anecdotiques sont masqués. Choisissez « Tous » pour les voir.'))
  }
 }
 if(state[kind+'Cursor'])box.append(actionButton('Afficher les événements suivants',async function(e){
  var button=e.currentTarget;button.disabled=true;
  try{
   var data=await api('/api/life-events'+eventsListQuery('cursor='+encodeURIComponent(state[kind+'Cursor'])));
   if(!box.isConnected)return;
   var known=new Set(state.events.map(function(item){return item.id}));
   state.events=state.events.concat((data.events||[]).filter(function(item){return !known.has(item.id)}));state.eventsCursor=data.next_cursor;
   var form=document.querySelector('[data-source-form="events"]');if(form)captureSourceDraft(form);
   drawSourceSection('events')
  }catch(err){if(box.isConnected){button.disabled=false;box.append(el('p','status error',err.message))}}
 },'secondary pagination-button'));
}
function noteTarget(kind,id){return '/api/'+(kind==='person'?'people':'life-events')+'/'+encodeURIComponent(id)+'/notes'}
function applyNoteRevision(target,result){var match=target.match(/^\/api\/(people|life-events)\/([^/]+)\/notes$/);if(!match)return;var row=state[match[1]==='people'?'people':'events'].find(function(item){return item.id===decodeURIComponent(match[2])});if(row&&result.parent_revision)row.revision=result.parent_revision;var open=match[1]==='people'&&state.person&&state.person.id===decodeURIComponent(match[2])?state.person:null;if(open&&result.parent_revision){var draft=state.sourceDrafts.get('people');if(draft&&draft.id===open.id&&draft.revision===open.revision)draft.revision=result.parent_revision;open.revision=result.parent_revision}}
function sourceNoteEditor(target,note,onSaved,addLabel){
 var drafts=state.noteDrafts||(state.noteDrafts=new Map()),key=target+':'+(note?note.id:'new'),draft=drafts.get(key);
 if(!draft){draft={text:note?note.text:'',revision:note&&note.revision,creation_id:newClientId()};drafts.set(key,draft)}
 var details=el('details','source-note-editor'),summary=el('summary','',note?'Modifier la note':addLabel||'Ajouter une note');details.open=Boolean(draft.text&&(!note||draft.text!==note.text));
 var form=el('form'),label=el('label','','Votre note'),input=el('textarea');input.id='note-'+newClientId();label.htmlFor=input.id;input.name='text';input.rows=4;input.required=true;input.maxLength=2000;input.value=draft.text;
 var save=el('button','','Enregistrer la note');save.type='submit';var message=el('p','status'),comparison=el('div');message.setAttribute('role','status');
 input.addEventListener('input',function(){draft.text=input.value});
 form.append(label,input,save,message,comparison);details.append(summary,form);
 form.addEventListener('submit',async function(event){event.preventDefault();if(save.disabled)return;draft.text=input.value;var sent=input.value;if(!sent.trim())return;save.disabled=true;message.textContent='Enregistrement…';
  try{if(note){var signature=String(draft.revision)+'\n'+sent;if(!draft.mutation||draft.mutation.signature!==signature)draft.mutation={signature:signature,id:newClientId()}}var result=await api(target+(note?'/'+encodeURIComponent(note.id):''),note?'PATCH':'POST',note?{text:sent,revision:draft.revision,mutation_id:draft.mutation.id}:{text:sent,creation_id:draft.creation_id});
   if(!details.isConnected)return;applyNoteRevision(target,result);
   if(input.value!==sent){draft.text=input.value;draft.revision=result.note.revision;drafts.delete(key);note=result.note;key=target+':'+note.id;drafts.set(key,draft);message.textContent='Note enregistrée. Votre nouvelle saisie reste à sauvegarder.'}
   else{drafts.delete(key);message.textContent='Note enregistrée.';if(details.isConnected)await onSaved()}
  }catch(error){if(!details.isConnected)return;message.textContent=error.status===409?'Cette note a changé ailleurs. Votre saisie est conservée.':error.message;
   if(error.status===409&&note){comparison.replaceChildren(actionButton('Comparer la note enregistrée',async function(){
    try{var cursor='',latest=null;do{var result=await api(target+'?limit=50'+(cursor?'&cursor='+encodeURIComponent(cursor):''));latest=(result.notes||[]).find(function(item){return item.id===note.id});cursor=result.next_cursor}while(!latest&&cursor);
     if(!details.isConnected)return;if(!latest){message.textContent='La note a été supprimée. Votre saisie reste disponible.';return}
     comparison.replaceChildren(el('p','source-note',latest.text),actionButton('Conserver ma saisie pour un nouvel enregistrement',function(){draft.revision=latest.revision;comparison.replaceChildren();message.textContent='Vérifiez votre note, puis enregistrez.'},'secondary'))
    }catch(err){message.textContent=err.message}
   },'secondary'))}
  }finally{save.disabled=false}
 });return details
}
function sourceNoteCard(target,note,onSaved,label){
 var deletionId=newClientId(),row=el('article','timeline-note'),dates=el('p','source-meta note-dates');dates.append(el('span','note-added',label||'Note ajoutée le '+shortDay(note.created_at)));row.dataset.noteId=note.id;
 if((note.revision>1||note.updated_at!==note.created_at)&&shortDay(note.updated_at))dates.append(el('span','note-edited',' · modifiée le '+shortDay(note.updated_at)));
 row.append(dates,el('p','source-note',note.text));
 row.append(sourceNoteEditor(target,note,onSaved),historyButton('note',note.id,'Note du '+shortDay(note.created_at)));var feedback=el('p','status');feedback.setAttribute('role','status');
 row.append(actionButton('Supprimer la note',async function(event){var button=event.currentTarget;if(!await confirmDialog('Supprimer cette note ?','','Supprimer',true))return;button.disabled=true;
  try{var result=await api(target+'/'+encodeURIComponent(note.id),'DELETE',{revision:note.revision,mutation_id:deletionId});if(!row.isConnected)return;applyNoteRevision(target,result);if(state.noteDrafts)state.noteDrafts.delete(target+':'+note.id);if(row.isConnected)await onSaved()}catch(error){feedback.textContent=error.status===409?'Cette note a changé. Rechargez les notes avant de la supprimer.':error.message;button.disabled=false}
 },'text danger'),feedback);return row
}
function eventNotes(item,options){
 options=options||{};
 var section=el('section','event-notes');section.setAttribute('aria-label','Notes sur cet événement');var target=noteTarget('event',item.id),notes=item.notes||[],cursor=item.notes_next_cursor,available=item.notes_available===true;
 async function reload(){try{var data=await api(target);if(!section.isConnected)return;notes=data.notes||[];cursor=data.next_cursor;available=data.available===true;draw()}catch(err){if(section.isConnected)section.append(el('p','status error',err.message),actionButton('Recharger les notes',reload,'text'))}}
 function draw(){section.replaceChildren();notes.forEach(function(note){section.append(sourceNoteCard(target,note,reload,options.noteLabel&&options.noteLabel(note)))});
  if(cursor)section.append(actionButton('Afficher les notes suivantes',async function(event){var button=event.currentTarget;button.disabled=true;try{var data=await api(target+'?cursor='+encodeURIComponent(cursor));if(!section.isConnected)return;var known=new Set(notes.map(function(n){return n.id}));notes=notes.concat((data.notes||[]).filter(function(n){return !known.has(n.id)}));cursor=data.next_cursor;draw()}catch(err){button.disabled=false;section.append(el('p','status error',err.message))}},'text'));
  if(available)section.append(sourceNoteEditor(target,null,reload,options.addLabel));else if(item.notes_available===false)section.append(el('p','helper','Les notes ne sont pas encore disponibles.'));
 }
 draw();return section
}
function noteDay(value){var date=new Date(value);return Number.isNaN(date.getTime())?'':date.toLocaleDateString('fr-FR',{day:'numeric',month:'short',year:'numeric'})}
// Migration 0034: the date an element was added and the history of its modifications.
// People and moments keep SQLite UTC text (« 2026-11-19 10:42:00 »), notes and history an ISO date.
function parseStamp(value){if(value==null||value==='')return null;var text=String(value),date=new Date(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$/.test(text)?text.replace(' ','T')+'Z':text);return Number.isNaN(date.getTime())?null:date}
function shortDay(value){var date=parseStamp(value);return date?date.toLocaleDateString('fr-FR',{day:'numeric',month:'short',year:'numeric'}):''}
function addedOn(item,feminine){var day=shortDay(item&&item.created_at);return day?(feminine?'Ajoutée':'Ajouté')+' le '+day:''}
// A discreet link: the history opens in a dialog, read on demand, never sent to Ambre.
function historyButton(type,id,subject){var button=actionButton('Historique',function(){openHistory(type,id,subject)},'text history-link');button.setAttribute('aria-haspopup','dialog');return button}
function historyValue(type,field,value,values){
 if(field==='meeting')return values.meeting_date_precision&&values.meeting_date_precision!=='unknown'?sourceDate({date_precision:values.meeting_date_precision,date_start:values.meeting_date_start,date_end:values.meeting_date_end,age_text:values.meeting_age_text}):'';
 if(field==='date')return values.date_precision&&values.date_precision!=='unknown'?sourceDate(values):'';
 var pick=function(list){var found=list.find(function(entry){return entry[0]===value});return found?found[1]:value||''};
 if(field==='role')return pick(PERSON_ROLES);if(field==='kind')return pick(EVENT_KINDS);
 if(field==='significance')return value==='major'?'Marquant':value==='anecdotal'?'Anecdotique':'À préciser';
 if(field==='current_importance')return value==null||value===''?'':value+' sur 5';
 if(field==='people_ids')return (value||[]).map(function(id){var known=(state.people||[]).concat(state.person?[state.person]:[]).find(function(person){return person.id===id});return known?known.display_name:'un proche'}).join(', ');
 var choices=(SOURCE_CONTEXT_CHOICES[type==='person'?'people':'events']||{})[field];if(choices)return choices[value]||'';
 return value==null?'':String(value)
}
function historyRows(type,values,changed){
 var labels={display_name:'Prénom ou pseudonyme',role:'Lien',custom_role:'Lien précisé',relationship_text:'Comment vous vivez cette relation',period_start:'Ancien repère de début',period_end:'Ancien repère de fin',meeting:'Date de rencontre',title:'Titre',kind:'Type',description:'Récit',meaning_now:'Ce que cela représente aujourd’hui',date:'Date',current_importance:'Poids aujourd’hui',significance:'Place dans votre histoire',people_ids:'Proches associés',text:'Texte'};
 var group=function(field){return /^meeting_/.test(field)?'meeting':['date_precision','date_start','date_end','age_text'].includes(field)?'date':field},seen=[];
 changed.forEach(function(field){var key=group(field);if(!seen.includes(key))seen.push(key)});
 return seen.map(function(key){return {key:key,label:labels[key]||SOURCE_CONTEXT_LABELS[key]||key}})
}
function historyEntry(type,version){
 var feminine=type==='note',item=el('li','history-entry'),when=el('p','history-when',(version.action==='create'?(feminine?'Ajoutée':'Ajouté'):(feminine?'Modifiée':'Modifié'))+' le '+shortDay(version.at));
 var time=parseStamp(version.at);if(time)when.append(el('span','history-time',' à '+time.toLocaleTimeString('fr-FR',{hour:'2-digit',minute:'2-digit'})));item.append(when);
 var list=el('dl','history-changes');
 if(version.action==='create'){var values=version.values||{};historyRows(type,values,Object.keys(values)).forEach(function(row){var text=historyValue(type,row.key,values[row.key],values);if(!text)return;list.append(el('dt','',row.label),el('dd','history-value',text))});if(!list.children.length)list.append(el('dd','muted','Sans contenu.'))}
 else{var before={},after={};(version.changes||[]).forEach(function(change){before[change.field]=change.before;after[change.field]=change.after});
  historyRows(type,after,Object.keys(after)).forEach(function(row){var old=historyValue(type,row.key,before[row.key],before),next=historyValue(type,row.key,after[row.key],after);if(old===next)return;
   if(type!=='note')list.append(el('dt','',row.label));var dd=el('dd','history-diff'),was=el('p','history-before'),now=el('p','history-after');
   was.append(el('span','history-tag',type==='note'?'Texte précédent':'Avant'),document.createTextNode(old||'(vide)'));now.append(el('span','history-tag',type==='note'?'Nouveau texte':'Après'),document.createTextNode(next||'(vide)'));
   dd.append(was,now);list.append(dd)})}
 if(version.action!=='create'&&!list.children.length)return null;
 if(list.children.length)item.append(list);return item
}
async function openHistory(type,id,subject){
 var opener=document.activeElement,dialog=el('dialog','history-dialog'),top=el('div','dialog-top'),heading=el('h2','','Historique'),close=actionButton('×',function(){dialog.close()},'close'),body=el('div','history-body');
 heading.id='history-dialog-title';dialog.setAttribute('aria-labelledby',heading.id);close.setAttribute('aria-label','Fermer l’historique');top.append(heading,close);dialog.append(top);
 if(subject)dialog.append(el('p','history-subject',subject));body.append(el('p','muted','Chargement…'));body.setAttribute('aria-busy','true');dialog.append(body);
 dialog.addEventListener('close',function(){dialog.remove();if(opener&&opener.isConnected&&opener.focus)opener.focus()});
 document.body.append(dialog);dialog.showModal();close.focus();
 try{
  var result=await api('/api/history?type='+encodeURIComponent(type)+'&id='+encodeURIComponent(id));if(!dialog.isConnected)return;body.removeAttribute('aria-busy');body.replaceChildren();
  if(result.available!==true){body.append(el('p','helper','L’historique des modifications n’est pas encore disponible.'));return}
  var versions=result.versions||[],entries=versions.map(function(version){return historyEntry(type,version)}).filter(Boolean);
  if(entries.length){var list=el('ol','history-list');entries.forEach(function(entry){list.append(entry)});body.append(list)}
  else body.append(el('p','muted',type==='note'?'Cette note n’a pas été modifiée.':'Aucune modification enregistrée.'));
  if(!result.complete){var since=shortDay(result.created_at);body.append(el('p','helper history-gap',(since?(type==='note'?'Ajoutée':'Ajouté')+' le '+since+'. ':'')+(versions.length>=(result.limit||50)?'Seules les '+(result.limit||50)+' dernières versions sont conservées.':'Les modifications faites avant la mise en place de l’historique ne sont pas connues.')))}
 }catch(err){if(!dialog.isConnected)return;body.removeAttribute('aria-busy');body.replaceChildren(el('p','status error',err.status===404?'Cet élément n’existe plus.':err.message))}
}
// « Votre histoire avec … » : one vertical line, one dot per step, the date above in small type.
// The meeting comes first, then the linked moments with their notes, the personal notes and the
// validated observations. The single general note form closes the line.
function personTimeline(person,initial){
 var details=el('section','person-timeline'),body=el('div','person-timeline-body'),items=[],cursor=null,loading=false,target=noteTarget('person',person.id),significance=false;
 details.append(el('h2','','Votre histoire avec '+person.display_name),body);
 async function load(more){if(loading)return;loading=true;try{var result=await api('/api/people/'+encodeURIComponent(person.id)+'/timeline'+(more&&cursor?'?cursor='+encodeURIComponent(cursor):''));if(!details.isConnected)return;apply(result,more)}catch(err){if(details.isConnected)body.append(el('p','status error',err.message),actionButton('Recharger la timeline',function(){return load(false)},'text'))}finally{loading=false}}
 function apply(result,more){items=more?items.concat(result.items||[]):result.items||[];cursor=result.next_cursor;significance=result.significance_ready===true;draw(result.available===true)}
 function reload(){return load(false)}
 function step(kind,when){var line=el('li','frise-step frise-'+kind);if(when)line.append(el('p','frise-when',when));return line}
 function noteLabel(note){var day=noteDay(note.created_at);return 'Votre note'+(day?' · '+day:'')}
 function eventStep(event){var line=step('event',sourceDate(event));line.dataset.eventId=event.id;if(significance)line.className+=' '+significanceClass(event);var link=el('a','',event.title);link.href='/ma-ligne-de-vie/?person='+encodeURIComponent(person.id)+'&event='+encodeURIComponent(event.id);var heading=el('h3','frise-title');heading.append(link);line.append(heading);
  if(significance)line.append(significanceView(event,function(saved){items=items.map(function(entry){return entry.kind==='life_event'&&entry.event.id===saved.id?Object.assign({},entry,{event:saved}):entry});draw(lastAvailable);var again=body.querySelector('[data-event-id="'+saved.id+'"] .frise-title a')||body.querySelector('.moment-filter button[aria-pressed="true"]');if(again)again.focus()}));
  if(event.description)line.append(el('p','source-note',event.description));line.append(eventNotes(event,{addLabel:'Ajouter une note à ce moment',noteLabel:noteLabel}));return line}
 var lastAvailable=false;
 function draw(available){lastAvailable=available;body.replaceChildren();var list=el('ol','person-frise'),mode=state.momentFilter==='major'?'major':'all',pending=[],hidden=0,events=items.filter(function(item){return item.kind==='life_event'});
  if(significance&&events.length){body.append(momentFilter(mode,async function(next){state.momentFilter=next;
   // The story of a person is paginated: « Marquants » first loads every remaining page, so that nothing is missed.
   if(next==='major'){var guard=0;while(cursor&&guard++<40){var before=cursor;await load(true);if(!details.isConnected||cursor===before)break}}
   draw(lastAvailable);var pressed=body.querySelector('.moment-filter button[aria-pressed="true"]');if(pressed)pressed.focus()},'Filtrer les moments de cette histoire'))}
  items.forEach(function(item){
   if(significance&&mode==='major'&&item.kind==='life_event'&&item.event.significance!=='major'){if(item.event.significance==='anecdotal')hidden++;else pending.push(item.event);return}
   if(item.kind==='relationship_start'){var date=item.date||{},known=date.precision&&date.precision!=='unknown'&&(date.start||date.age_text),line=step('meeting',known?sourceDate({date_precision:date.precision,date_start:date.start,date_end:date.end,age_text:date.age_text}):'Date non précisée');line.append(el('h3','frise-title','Rencontre'));if(item.legacy_period_start||item.legacy_period_end)line.append(el('p','source-meta',[item.legacy_period_start,item.legacy_period_end].filter(Boolean).join(' – ')));list.append(line)}
   else if(item.kind==='life_event')list.append(eventStep(item.event))
   else if(item.kind==='user_note'){var line=step('note');line.append(sourceNoteCard(target,item.note,reload,noteLabel(item.note)));list.append(line)}
   else if(item.kind==='ambre_observation'){var note=item.note,day=noteDay(note.created_at),line=step('observation','Repère validé'+(day?' · '+day:'')),feedback=el('p','status');line.append(el('p','source-note',note.text),actionButton('Retirer ce repère',async function(event){var button=event.currentTarget;button.disabled=true;try{await api('/api/people/'+encodeURIComponent(person.id)+'/observations/'+encodeURIComponent(note.id),'DELETE');await load(false)}catch(err){feedback.textContent=err.message;button.disabled=false}},'text'),feedback);list.append(line)}
  });
  if(items.length)body.append(list);else body.append(el('p','muted','Aucun moment enregistré pour cette relation.'));
  if(pending.length){var later=el('section','pending-moments');later.append(el('h3','frise-pending-title','À préciser'),el('p','helper','Ces moments ne sont pas encore classés. Indiquez s’ils sont marquants dans votre histoire.'));var laterList=el('ol','person-frise');pending.forEach(function(event){laterList.append(eventStep(event))});later.append(laterList);body.append(later)}
  if(hidden)body.append(el('p','muted moment-filter-hidden','Les moments anecdotiques sont masqués. Choisissez « Tous » pour les voir.'));
  if(cursor)body.append(actionButton('Afficher la suite',function(){return load(true)},'secondary frise-more'));
  // Two ways to add to this story, side by side: a note here, or a moment of Ma ligne de vie with this person already chosen.
  var adds=el('div','person-timeline-adds');if(available){var add=sourceNoteEditor(target,null,reload,'Ajouter une note');add.className+=' person-note-add';adds.append(add)}
  var moment=el('a','button secondary person-moment-add','Ajouter un moment');moment.href='/ma-ligne-de-vie/?'+MOMENT_FOR_PERSON+'='+encodeURIComponent(person.id);adds.append(moment);body.append(adds)
 }
 // The Worker prepares the first page of the timeline; later reloads use the API.
 if(initial)apply(initial,false);else{body.append(el('p','muted page-loading','Chargement…'));queueMicrotask(function(){if(details.isConnected)load(false)})}
 return details
}
function personRoleLabel(item){var role=PERSON_ROLES.find(function(entry){return entry[0]===item.role});return (item.role==='other'?item.custom_role:role&&role[1])||'Proche'}
async function renderPerson(){
 var holder=document.getElementById('view'),current=renderGuard(holder),id=state.personId,base='/api/people/'+encodeURIComponent(id);
 var title=document.getElementById('private-page-title'),back=el('a','text-link page-back','Mes proches');back.href='/mes-proches/';title.parentElement.prepend(back);
 holder.innerHTML='<section class="panel narrow personal-section source-panel person-page"><p class="status" id="source-page-status" role="status"></p><div id="person-content"><p class="muted page-loading" role="status">Chargement…</p></div></section>';
 try{
  var results=await Promise.all([api(base),api(base+'/timeline')]);
  if(!current()||state.tab!=='person'||state.personId!==id)return;
  state.person=results[0].person;state.peopleFieldsReady=results[0].fields_ready===true;state.personTimeline=results[1];drawPersonPage()
 }catch(err){
  if(!current()||state.tab!=='person'||state.personId!==id)return;
  state.person=null;
  var box=document.getElementById('person-content'),list=el('a','button secondary','Revenir à Mes proches');list.href='/mes-proches/';
  // The same message covers an unknown identifier and a person of another account.
  if(err.status===404)box.replaceChildren(el('h2','','Ce proche est introuvable'),el('p','','Cette page n’existe pas, ou ce proche n’est plus enregistré dans votre espace. Vous pouvez retrouver vos proches dans la liste.'),list);
  else box.replaceChildren(el('p','status error',err.message),actionButton('Réessayer',renderPerson,'secondary'),list)
 }
}
function drawPersonPage(){
 var box=document.getElementById('person-content'),person=state.person;if(!box||!person)return;
 var title=document.getElementById('private-page-title'),description=document.getElementById('page-description');
 title.textContent=person.display_name;description.textContent=personRoleLabel(person);description.hidden=false;
 var info=el('section','person-info'),actions=el('div','actions source-actions'),editor=el('div','source-editor-host'),feedback=el('p','status');
 editor.id='people-editor';feedback.setAttribute('role','status');info.append(el('h2','','Informations'));
 var added=addedOn(person);if(added)info.append(el('p','source-meta added-on',added));
 if(person.relationship_text)info.append(el('p','source-note',person.relationship_text));
 if(person.period_start||person.period_end)info.append(el('p','source-meta',[person.period_start,person.period_end].filter(Boolean).join(' – ')));
 appendSourceContextText(info,'people',person);
 if(info.children.length===(added?2:1))info.append(el('p','muted','Aucune information ajoutée pour le moment.'));
 var edit=actionButton('Modifier',async function(){
  var open=document.querySelector('[data-source-form="people"]');if(open)captureSourceDraft(open);
  var old=state.sourceDrafts.get('people');
  if(old&&old.id!==person.id&&!await confirmDialog('Remplacer le brouillon ouvert ?','Le brouillon en cours sera remplacé par ce proche enregistré.','Remplacer'))return;
  if(!old||old.id!==person.id)state.sourceDrafts.set('people',{id:person.id,revision:person.revision,values:Object.assign({},person)});
  drawSourceEditor('people',true)
 },'secondary');edit.dataset.personEdit='true';
 actions.append(edit,historyButton('person',person.id,person.display_name),actionButton('Supprimer',async function(e){
  var button=e.currentTarget;
  if(!await confirmDialog('Supprimer '+person.display_name+' ?','Ses liens avec les événements seront retirés. Les événements et vos textes resteront conservés.','Supprimer',true))return;
  button.disabled=true;feedback.textContent='';
  try{
   var result=await api('/api/people/'+encodeURIComponent(person.id),'DELETE',{revision:person.revision});
   state.people=state.people.filter(function(row){return row.id!==person.id});
   if(state.sourceDrafts.get('people')&&state.sourceDrafts.get('people').id===person.id)state.sourceDrafts.delete('people');
   state.events.forEach(function(event){event.people_ids=(event.people_ids||[]).filter(function(id){return id!==person.id})});
   var draft=state.sourceDrafts.get('events');if(draft)draft.values.people_ids=(draft.values.people_ids||[]).filter(function(id){return id!==person.id});
   if(!box.isConnected)return;
   // Replace the address of the deleted page so that Back does not return to it.
   state.person=null;history.replaceState({page:'people'},'','/mes-proches/');await go('people');sourceSaved(result);status('source-page-status','Proche supprimé.')
  }catch(err){
   if(!box.isConnected)return;
   feedback.textContent=err.code==='revision_conflict'?'Ce proche a changé dans un autre onglet. Rechargez la page pour vérifier sa dernière version avant de le supprimer.':err.message;feedback.className='status error';button.disabled=false
  }
 },'text danger'));
 info.append(actions,feedback,editor);
 box.replaceChildren(info,personTimeline(person,state.personTimeline));state.personTimeline=null;
 drawSourceEditor('people')
}
function personSaved(saved){
 state.person=saved;state.personTimeline=null;drawPersonPage();
 var edit=document.querySelector('#person-content [data-person-edit]');if(edit)edit.focus()
}

// Migration 0027: « marquant dans mon histoire », chosen by the person only (never deduced).
// It differs from current_importance (« pèse aujourd’hui »). Without a value the moment is « à préciser ».
// Query of Ma ligne de vie: the person filter, the « Marquants » filter (applied by the API), then the cursor.
function eventsListQuery(extra){var parts=[];if(query.get('person'))parts.push('person_id='+encodeURIComponent(query.get('person')));if(state.momentFilter==='major')parts.push('significance=major');if(extra)parts.push(extra);return parts.length?'?'+parts.join('&'):''}
function significanceClass(item){return item&&item.significance==='major'?'is-major':item&&item.significance==='anecdotal'?'is-anecdotal':'is-unset'}
function significanceField(host,value,options){
 options=options||{};var field=el('fieldset','field full significance-field'),legend=el('legend','','Place dans votre histoire'),group=el('div','segmented'),prefix=options.id||'significance',inputs=[];
 legend.id=prefix+'-legend';field.append(legend);group.setAttribute('role','radiogroup');group.setAttribute('aria-labelledby',legend.id);
 // A modification may bring the moment back to « à préciser » (sent as an explicit null).
 var unset=options.allowUnset&&(value==null||value===''||value==='unset');
 SIGNIFICANCE_CHOICES.concat(options.allowUnset?[['unset','À préciser']]:[]).forEach(function(choice){var label=el('label','segment'),input=el('input');input.type='radio';input.name='significance';input.value=choice[0];input.id=prefix+'-'+choice[0];input.checked=choice[0]==='unset'?unset:value===choice[0];if(options.required)input.required=true;label.htmlFor=input.id;label.append(input,el('span','',choice[1]));group.append(label);inputs.push(input)});
 var hint=el('small','field-hint',options.hint||'Marquant : ce moment compte dans votre histoire. Cela ne dit pas s’il pèse aujourd’hui.');hint.id=prefix+'-help';group.setAttribute('aria-describedby',hint.id);
 function validity(){var chosen=inputs.some(function(input){return input.checked});inputs.forEach(function(input){if(input.setCustomValidity)input.setCustomValidity(options.required&&!chosen?'Choisissez « Anecdotique » ou « Marquant ».':'')})}
 inputs.forEach(function(input){input.addEventListener('change',function(){validity();if(options.onChange)options.onChange(input.value)})});validity();
 field.append(group,hint);host.append(field);return inputs
}
function chosenSignificance(form){var picked=Array.from(form.querySelectorAll('[name=significance]')).find(function(input){return input.checked});return picked?picked.value:''}
// The place of a moment on a card: a word for a chosen value, two buttons for a moment « à préciser ».
function significanceView(item,onSaved){
 var wrap=el('div','significance-view '+significanceClass(item));
 if(item.significance==='major'||item.significance==='anecdotal'){wrap.append(el('span','significance-tag',item.significance==='major'?'Marquant':'Anecdotique'));return wrap}
 var feedback=el('p','status');feedback.setAttribute('role','status');
 var group=el('div','significance-choice');group.setAttribute('role','group');group.setAttribute('aria-label','À préciser : ce moment est-il marquant dans votre histoire ?');
 group.append(el('span','significance-tag','À préciser'));
 SIGNIFICANCE_CHOICES.forEach(function(choice){group.append(actionButton(choice[1],async function(event){var buttons=group.querySelectorAll('button');buttons.forEach(function(button){button.disabled=true});feedback.textContent='Enregistrement…';feedback.className='status';
  try{var result=await api('/api/life-events/'+encodeURIComponent(item.id),'PATCH',{revision:item.revision,significance:choice[0]});if(!wrap.isConnected)return;feedback.textContent='';await onSaved(result.event,result)}
  catch(err){if(!wrap.isConnected)return;feedback.textContent=err.code==='revision_conflict'?'Ce moment a changé ailleurs. Rechargez la page pour le préciser.':err.message;feedback.className='status error';buttons.forEach(function(button){button.disabled=false})}
 },'secondary segment-button'))});
 wrap.append(group,feedback);return wrap
}
// « Tous / Marquants »: the second view keeps the marking moments and lists the moments « à préciser » apart.
function momentFilter(current,onChange,label){
 var group=el('div','segmented moment-filter');group.setAttribute('role','group');group.setAttribute('aria-label',label||'Filtrer les moments');
 [['all','Tous'],['major','Marquants']].forEach(function(choice){var button=actionButton(choice[1],function(){if(current!==choice[0])onChange(choice[0])},'segment-button');button.setAttribute('aria-pressed',String(current===choice[0]));group.append(button)});
 return group
}
function drawSourceItem(kind,item,box){
 var isPeople=kind==='people',article=el('article','source-item '+(isPeople?'person-card':'event-card'));
 article.id=(isPeople?'person-':'event-')+item.id;article.tabIndex=-1;
 var content=el('div','source-body'),title=el(isPeople?'h3':'h4','',isPeople?undefined:item.title),meta=el('div','source-meta');
 if(isPeople){
  // The directory stays short: the details and the timeline live on the person page.
  var open=el('a','person-open',item.display_name);open.href=pagePath('person',item.id);title.append(open);var head=el('div','person-head');head.append(title,el('span','source-tag',personRoleLabel(item)));content.append(head);
  if(item.relationship_text)content.append(el('p','source-note person-line',item.relationship_text));
  var addedPerson=addedOn(item);if(addedPerson)content.append(el('p','source-meta added-on',addedPerson));
  article.append(content);return article
 }
 var eventKind=EVENT_KINDS.find(function(entry){return entry[0]===item.kind});
 var when=eventDateInYear(item);if(when)meta.append(el('span','event-date',when),document.createTextNode(' · '));
 meta.append(el('span','source-tag',eventKind?eventKind[1]:'Événement'));
 var linked=(item.people_ids||[]).map(function(id){return state.people.find(function(person){return person.id===id})}).filter(Boolean);
 if(linked.length){
  var connections=el('span','event-people');connections.append(document.createTextNode(' · avec '));
  linked.forEach(function(person,index){var link=el('a','',person.display_name);link.href=pagePath('person',person.id);if(index)connections.append(document.createTextNode(', '));connections.append(link)});
  meta.append(connections)
 }
 content.append(title,meta);
 var addedEvent=addedOn(item);if(addedEvent)content.append(el('p','source-meta added-on',addedEvent));
 if(state.eventsSignificanceReady){article.className+=' '+significanceClass(item);content.append(significanceView(item,function(saved){var open=document.querySelector('[data-source-form="events"]');if(open)captureSourceDraft(open);var editing=state.sourceDrafts.get('events');if(editing&&editing.id===saved.id){editing.revision=saved.revision;editing.values.significance=saved.significance}state.events=state.events.map(function(row){return row.id===saved.id?saved:row});drawSourceSection('events');var entry=document.getElementById('event-'+saved.id)||document.querySelector('.moment-filter button[aria-pressed="true"]');if(entry)entry.focus({preventScroll:true});status('source-page-status','Enregistré.')}))}
 content.append(el('p','source-note',item.description));
 appendSourceContextText(content,kind,item);
 var actions=el('div','actions source-actions');
 actions.append(actionButton('Modifier',async function(){
  var open=document.querySelector('[data-source-form="'+kind+'"]');if(open)captureSourceDraft(open);
  var old=state.sourceDrafts.get(kind);
  if(old&&old.id!==item.id&&!await confirmDialog('Remplacer le brouillon ouvert ?','Le brouillon en cours sera remplacé par cet élément enregistré.','Remplacer'))return;
  state.sourceDrafts.set(kind,{id:item.id,revision:item.revision,values:Object.assign({},item)});drawSourceEditor(kind,true)
 },'text'),historyButton('event',item.id,item.title),actionButton('Supprimer',async function(e){
  var button=e.currentTarget;
  if(!await confirmDialog('Supprimer ce moment ?','Il sera retiré de votre ligne de vie et des prochaines analyses.','Supprimer',true))return;
  button.disabled=true;
  try{
   var result=await api('/api/life-events/'+encodeURIComponent(item.id),'DELETE',{revision:item.revision});
   if(!box.isConnected)return;
   state[kind]=state[kind].filter(function(row){return row.id!==item.id});
   if(state.sourceDrafts.get(kind)&&state.sourceDrafts.get(kind).id===item.id)state.sourceDrafts.delete(kind);
   drawSourceSection(kind);document.querySelector('#'+kind+'-editor summary')?.focus();sourceSaved(result,true)
  }catch(err){
   if(!box.isConnected)return;
   content.append(el('p','status error',err.code==='revision_conflict'?'Cet élément a changé dans un autre onglet. Rechargez la page pour vérifier sa dernière version avant de le supprimer.':err.message));button.disabled=false
  }
 },'text danger'));
 var foot=el('div','event-foot');foot.append(eventNotes(item),actions);content.append(foot);article.append(content);return article
}
function drawSourceEditor(kind,focus){var host=document.getElementById(kind+'-editor');if(!host)return;var current=renderGuard(host);host.replaceChildren();var draft=state.sourceDrafts.get(kind);if(kind==='people'&&state.tab==='person'&&!(draft&&draft.id===state.personId))return;var details=el('details','source-editor');details.open=!!draft;var summary=el('summary','',draft&&draft.id?(kind==='people'?'Modifier ce proche':'Modifier ce moment'):kind==='people'?'Ajouter un proche':'Ajouter un moment');details.append(summary);host.append(details);if(!draft){details.addEventListener('toggle',function(){if(details.open&&!state.sourceDrafts.has(kind)){state.sourceDrafts.set(kind,{id:null,creation_id:newClientId(),values:kind==='people'?{role:'friend'}:{kind:'positive',date_precision:'unknown',people_ids:[]}});drawSourceEditor(kind,true)}});return}var form=el('form');form.id=kind+'-form';form.dataset.sourceForm=kind;var values=draft.values,grid=el('div','form-grid');grid.id=form.id+'-fields';form.append(grid);
if(kind==='people'){sourceField(grid,'display_name','Prénom ou pseudonyme',values.display_name,{required:true,max:80});var role=sourceField(grid,'role','Lien',values.role,{choices:PERSON_ROLES});var custom=sourceField(grid,'custom_role','Précisez le lien',values.custom_role,{max:80});function showRole(){custom.parentElement.hidden=role.value!=='other'}role.addEventListener('change',showRole);showRole();sourceField(grid,'relationship_text','Comment vivez-vous cette relation ?',values.relationship_text,{multiline:true,max:6000,full:true,placeholder:'Ce que vous ressentez dans cette relation…'});var more=el('details','form-more disclosure'),moreGrid=el('div','form-grid');more.append(el('summary','','Ajouter des détails'),moreGrid);moreGrid.id=form.id+'-details';grid.append(more);appendEventDateFields(moreGrid,values,draft,'meeting_');if(values.period_start||values.period_end){sourceField(moreGrid,'period_start','Ancien repère de début',values.period_start,{max:80,placeholder:'Depuis l’enfance, 2020…'});sourceField(moreGrid,'period_end','Ancien repère de fin',values.period_end,{max:80,placeholder:'Aujourd’hui, vers 2022…'})}if(state.peopleFieldsReady)appendSourceContextFields(moreGrid,kind,values);more.open=draft.detailsOpen===true||(draft.detailsOpen!==false&&['meeting_date_start','meeting_age_text','period_start','period_end'].concat(SOURCE_CONTEXT_ORDER.people).some(function(name){return values[name]}));more.addEventListener('toggle',function(){var open=state.sourceDrafts.get(kind);if(open)open.detailsOpen=more.open})}else{sourceField(grid,'title','Titre',values.title,{required:true,max:160,full:true});sourceField(grid,'kind','Type',values.kind,{choices:EVENT_KINDS});if(state.eventsSignificanceReady)significanceField(grid,values.significance,{required:!draft.id,allowUnset:Boolean(draft.id),id:form.id+'-significance'});appendEventDateFields(grid,values,draft);sourceField(grid,'description','Ce que vous souhaitez raconter',values.description,{multiline:true,max:6000,required:true,full:true});var peopleField=el('fieldset','field full');peopleField.append(el('legend','','Avec'));var options=el('div','people-options');state.people.forEach(function(person){var label=el('label','check'),checkbox=el('input');checkbox.type='checkbox';checkbox.name='people_ids';checkbox.value=person.id;checkbox.checked=(values.people_ids||[]).includes(person.id);label.append(checkbox,document.createTextNode(person.display_name));options.append(label)});if(!state.people.length){var none=el('p','helper','Aucun proche enregistré. '),peopleLink=el('a','','Ajouter un proche');peopleLink.href='/mes-proches/';none.append(peopleLink);options.append(none)};peopleField.append(options);grid.append(peopleField)}
if(kind!=='people'&&state[kind+'FieldsReady'])appendSourceContextFields(grid,kind,values);
var actions=el('div','actions'),save=el('button','','Enregistrer');save.type='submit';actions.append(save,actionButton('Annuler',function(){captureSourceDraft(form);var back=momentReturn(kind,state.sourceDrafts.get(kind));if(back){delete state.sourceDrafts.get(kind).returnTo;go('person','',back);return}details.open=false;summary.focus()},'secondary'));var message=el('p','status source-feedback');message.setAttribute('role','status');form.append(actions,message);details.append(form);form.addEventListener('input',function(){captureSourceDraft(form)});form.addEventListener('change',function(){captureSourceDraft(form)});form.addEventListener('submit',async function(event){event.preventDefault();captureSourceDraft(form);var submitted=state.sourceDrafts.get(kind),payload=Object.assign({},submitted.values);if(!state[kind+'FieldsReady'])SOURCE_CONTEXT_ORDER[kind].forEach(function(name){delete payload[name]});if(kind==='events'){['meaning_now','current_importance','exposure','felt_then','felt_now','impact','support_level','support_text'].forEach(function(name){delete payload[name]});if(payload.significance==='unset')payload.significance=submitted.id?null:'';if(!state.eventsSignificanceReady||payload.significance===''||payload.significance===undefined)delete payload.significance;if(state.eventsSignificanceReady&&!submitted.id){if(!payload.significance){message.textContent='Choisissez « Anecdotique » ou « Marquant » pour ce moment.';message.className='status error';var firstChoice=form.querySelector('[name=significance]');if(firstChoice)firstChoice.focus();return}payload.entry='form'}if(payload.people_ids.length>10){message.textContent='Associez au maximum 10 proches à cet événement.';message.className='status error';return}if(payload.date_precision==='unknown'||payload.date_precision==='age'){payload.date_start='';payload.date_end=''}if(payload.date_precision!=='range')payload.date_end='';if(payload.date_precision!=='age')payload.age_text=''}if(kind==='people'&&payload.role!=='other')payload.custom_role='';if(submitted.id)payload.revision=submitted.revision;else payload.creation_id=submitted.creation_id;var snapshot=JSON.stringify(submitted.values);busy(form,true);message.textContent='Enregistrement…';message.className='status';try{var result=await api((kind==='people'?'/api/people':'/api/life-events')+(submitted.id?'/'+encodeURIComponent(submitted.id):''),submitted.id?'PATCH':'POST',payload);if(!current())return;var saved=result[kind==='people'?'person':'event'];state[kind]=state[kind].filter(function(item){return item.id!==saved.id});state[kind].unshift(saved);captureSourceDraft(form);var latest=state.sourceDrafts.get(kind);if(latest&&JSON.stringify(latest.values)!==snapshot){latest.id=saved.id;latest.revision=saved.revision;delete latest.creation_id;if(kind==='people'&&state.tab==='person')state.person=saved;drawSourceEditor(kind);var note=document.querySelector('#'+kind+'-editor .source-feedback');note.textContent='Enregistré. Vos changements saisis pendant l’enregistrement restent à sauvegarder.'}else{state.sourceDrafts.delete(kind);var back=momentReturn(kind,submitted);if(back){go('person','',back);return}if(kind==='people'&&state.tab==='person')personSaved(saved);else{drawSourceSection(kind);document.querySelector('#'+kind+'-editor summary').focus()}}if(kind==='people'){var eventForm=document.querySelector('[data-source-form="events"]');if(eventForm)captureSourceDraft(eventForm);drawSourceSection('events')}sourceSaved(result)}catch(err){if(!current())return;message.textContent=err.code==='revision_conflict'?'Cet élément a été modifié dans un autre onglet. Votre brouillon est conservé. Rechargez la version enregistrée pour comparer avant de réessayer.':err.message;message.className='status error';if(err.code==='revision_conflict'&&!form.querySelector('[data-reload-source]')){var reload=actionButton('Afficher la version enregistrée',async function(){try{var response=await api((kind==='people'?'/api/people/':'/api/life-events/')+encodeURIComponent(submitted.id));var latest=response[kind==='people'?'person':'event'];if(!current())return;if(!latest){message.textContent='Cet élément a été supprimé ou ne figure pas sur cette page. Votre brouillon reste disponible.';return}var comparison=el('div','notice source-comparison');comparison.append(el('h3','','Version enregistrée'));form.querySelectorAll('input,select,textarea').forEach(function(input){if(!input.name||input.name==='people_ids'||input.type==='radio')return;var value=latest[input.name];if(value==null||value==='')return;if(input.tagName==='SELECT'){var option=Array.from(input.querySelectorAll('option')).find(function(option){return option.value===String(value)});if(option)value=option.textContent}var caption=input.parentElement.querySelector('label');comparison.append(el('p','',(caption?caption.textContent:input.name)+' : '+value))});if(kind==='events'&&latest.significance){var place=SIGNIFICANCE_CHOICES.find(function(choice){return choice[0]===latest.significance});if(place)comparison.append(el('p','','Place dans votre histoire : '+place[1]))}if(kind==='events'&&latest.people_ids&&latest.people_ids.length)comparison.append(el('p','','Proches associés : '+latest.people_ids.map(function(id){var person=state.people.find(function(person){return person.id===id});return person?person.display_name:'Proche'}).join(', ')));comparison.append(actionButton('Conserver mon brouillon pour un nouvel enregistrement',function(){captureSourceDraft(form);var kept=state.sourceDrafts.get(kind);kept.revision=latest.revision;comparison.remove();reload.remove();message.textContent='Votre brouillon est conservé. Vérifiez vos champs puis enregistrez pour remplacer les informations comparées.';message.className='status'},'secondary'),actionButton('Reprendre cette version',function(){if(!confirm('Remplacer votre brouillon par la version enregistrée ? Votre saisie non enregistrée sera perdue.'))return;state.sourceDrafts.set(kind,{id:latest.id,revision:latest.revision,values:latest});drawSourceEditor(kind,true)},'secondary'));form.append(comparison)}catch(error){message.textContent=error.message}},'secondary');reload.dataset.reloadSource='true';form.append(reload)}}finally{busy(form,false)}});if(focus)focusEntry(form.querySelector('input,select,textarea'),summary)}

function setupPortraitViews(view){
 if(['#ma-cartographie','#cartographie','#mon-portrait','#synthese','#correction-form'].includes(location.hash))history.replaceState({},'',location.pathname+location.search+'#self-profile');
}

${CHAT_INTERFACE_SCRIPT}
${TOPICS_INTERFACE_SCRIPT}
${GRAPH_INTERFACE_SCRIPT}
var HOME_PAGES=${JSON.stringify(NAVIGATION)};
${HOME_INTERFACE_SCRIPT}
${REVIEW_ARCHIVE_SCRIPT}
${CHANGE_INTERFACE_SCRIPT}
${CARE_INTERFACE_SCRIPT}
${PEOPLE_AXES_UI}
async function drawChatPeople(host,payload,current){host.append(el('p','helper','Chargement de vos proches…'));try{var result=await api('/api/people');if(!current()||!host.isConnected)return;host.replaceChildren(el('legend','','Proches associés'));var people=result.people||[];if(!people.length)host.append(el('p','helper','Aucun proche enregistré.'));people.forEach(function(person){var label=el('label','check'),input=el('input');input.type='checkbox';input.name='people_ids';input.value=person.id;input.checked=payload.people_ids.includes(person.id);input.addEventListener('change',function(){payload.people_ids=input.checked?Array.from(new Set(payload.people_ids.concat(person.id))):payload.people_ids.filter(function(id){return id!==person.id})});label.append(input,document.createTextNode(person.display_name||'Proche'));host.append(label)})}catch(err){if(current()){host.replaceChildren(el('legend','','Proches associés'),el('p','status error','Impossible de charger vos proches. Les associations proposées restent conservées.'),actionButton('Réessayer',function(){drawChatPeople(host,payload,current)},'secondary'))}}}
function drawChatProposal(host,proposal,current){
 var titles={person_observation:'Un repère sur votre relation',self_profile_patch:'Votre présentation',person_create:'Ajouter un proche',person_update:'Modifier un proche',life_event_create:'Ajouter un moment de vie',life_event_update:'Modifier un moment de vie',memory_append:'Préciser mon portrait'},labels={axis:'Rubrique',quote:'Vos mots à conserver',what_brings_me:'Ce qui m’amène',what_helps_me:'Ce qui m’aide',life_rhythm:'Mon rythme de vie',first_name:'Prénom ou pseudonyme',last_name:'Nom',birth_date:'Date de naissance',gender:'Genre',gender_description:'Votre genre, avec vos mots',custom_role:'Autre relation',period_start:'Début de la relation',period_end:'Fin de la relation',self_description:'Votre présentation',self_perception:'Votre perception de vous',recurring_patterns:'Les schémas que vous remarquez',emotions:'Vos émotions',occupation:'Activité',current_context:'Contexte actuel',meaning_now:'Ce que ce moment représente aujourd’hui',display_name:'Prénom ou pseudonyme',role:'Relation',relationship_text:'Votre relation',title:'Titre',description:'Description',kind:'Type',date_precision:'Précision de la date',date_start:'Début',date_end:'Fin',age_text:'Âge approximatif',current_importance:'Importance actuelle',character_id:'Personnage',expected_parent_id:'Dernier épisode',people_ids:'Proches associés',requested_focus:'Thème souhaité',text:'Votre précision'};
 var box=el('form','panel personal-section'),payload=state.chatProposalDrafts.get(proposal.id)||JSON.parse(JSON.stringify(proposal.payload||{})),modes=proposal.field_modes||{};
 var replaced=Object.keys(payload).filter(function(key){return modes[key]==='set'&&proposal.before&&Object.prototype.hasOwnProperty.call(proposal.before,key)&&JSON.stringify(payload[key])!==JSON.stringify(proposal.before[key])&&String(proposal.before[key]==null?'':proposal.before[key]).trim()!==''}),appended=Object.keys(payload).filter(function(key){return modes[key]==='append'});
 state.chatProposalDrafts.set(proposal.id,payload);box.append(el('h3','',titles[proposal.kind]||'Proposition à vérifier'));
 if(proposal.evidence_quote)box.append(el('p','helper','À partir de votre message :'),el('blockquote','chat-content',proposal.evidence_quote));
 if(proposal.before&&Object.keys(proposal.before).length){
  var before=el('details','proposal-before');before.append(el('summary','','Informations actuelles — avant modification'));
  Object.keys(payload).forEach(function(key){if(Object.prototype.hasOwnProperty.call(proposal.before,key)){var value=proposal.before[key];before.append(el('p','chat-content',(labels[key]||key)+' : '+(Array.isArray(value)?value.join(', '):value==null?'Non renseigné':typeof value==='object'?'Informations associées':String(value))))}});box.append(before);
 }
 var effect=proposal.kind==='person_observation'?'Après acceptation : ce repère sera conservé dans la fiche de ce proche. Il décrit votre point de vue, pas un diagnostic sur cette personne.':proposal.kind==='memory_append'?'Après acceptation : cette précision sera ajoutée à votre portrait. Vos informations déjà enregistrées seront conservées.':appended.length?(replaced.length?'Les champs marqués « Ajouter » conserveront leur texte actuel. Les champs marqués « Remplacer » seront remplacés.':'Les fragments ci-dessous seront ajoutés au texte déjà enregistré, qui sera conservé.'):replaced.length?'Les champs marqués « Remplacer » seront remplacés par le texte ci-dessous.':'Après acceptation : les champs ci-dessous seront enregistrés.';
 box.append(el('p','helper',effect));
 if(proposal.rationale||proposal.summary)box.append(el('p','muted',proposal.rationale||proposal.summary));
 box.append(el('p','helper','Vérifiez et modifiez cette proposition avant de l’accepter. Aucun changement n’est enregistré sans votre accord.'));
 Object.keys(payload).forEach(function(key){
  var value=payload[key];
  if(proposal.kind==='person_observation'&&key==='axis'){var axis=sourceField(box,key,labels.axis,value,{choices:Object.entries(PEOPLE_AXES)});axis.id='proposal-'+proposal.id+'-axis';axis.parentElement.querySelector('label').htmlFor=axis.id;axis.addEventListener('change',function(){payload.axis=axis.value});return}
  if(key==='people_ids'&&Array.isArray(value)){var peopleBox=el('fieldset','field');peopleBox.append(el('legend','',labels.people_ids));box.append(peopleBox);drawChatPeople(peopleBox,payload,current);return}
  if(Array.isArray(value)||value&&typeof value==='object'){box.append(el('p','helper',(labels[key]||key)+' : '+(Array.isArray(value)?value.join(', '):'Informations associées conservées')));return}
  var field=el('div','field'),labelText=(modes[key]==='append'?'Ajouter à : ':replaced.includes(key)?'Remplacer : ':'')+(labels[key]||key),label=el('label','',labelText),input=el('textarea');
  input.value=value==null?'':String(value);input.name=key;
  if(proposal.kind==='person_observation'&&key==='quote'){input.minLength=20;input.maxLength=280;input.required=true}
  if(proposal.kind==='memory_append'&&key==='text'){input.maxLength=2000;input.required=true}
  input.id='proposal-'+proposal.id+'-'+key;label.htmlFor=input.id;input.rows=2;
  input.addEventListener('input',function(){payload[key]=key==='current_importance'?(input.value.trim()===''?null:Number(input.value)):typeof value==='number'&&input.value!==''?Number(input.value):typeof value==='boolean'?input.value==='true':input.value});field.append(label,input);box.append(field);
 });
 // Migration 0027: Ambre never chooses « marquant / anecdotique ». The person chooses it here, before accepting a new moment.
 var askSignificance=proposal.kind==='life_event_create'&&Boolean(state.chatData&&state.chatData.significance_ready),choices=state.chatProposalSignificance||(state.chatProposalSignificance=new Map());
 if(askSignificance)significanceField(box,choices.get(proposal.id)||'',{required:true,id:'proposal-'+proposal.id+'-significance',hint:'À choisir par vous : Ambre ne le décide pas. Marquant : ce moment compte dans votre histoire, qu’il pèse ou non aujourd’hui.',onChange:function(value){choices.set(proposal.id,value)}});
 var capabilityNote=el('p','helper chat-capability'),unavailable=chatCapabilityMessage(proposal.kind);capabilityNote.textContent=unavailable;capabilityNote.hidden=!unavailable;box.append(capabilityNote);
 var feedback=el('p','status');feedback.setAttribute('role','status');
 var replacementScope=replaced.map(function(key){return '« '+(labels[key]||key)+' »'}).join(', '),acceptLabel=replaced.length===1&&!appended.length?'Remplacer '+replacementScope:replaced.length?'Confirmer les remplacements'+(appended.length?' et ajouts':''):'Accepter cette proposition';
 var actions=el('div','actions'),accept=el('button','',acceptLabel);accept.type='submit';accept.disabled=!!unavailable;var reject=actionButton('Refuser',function(){return decide('reject')},'secondary');actions.append(accept,reject);box.append(actions,feedback);
 box.addEventListener('submit',function(event){event.preventDefault();return decide('accept')});
 async function decide(action){
  if(!current())return;
  if(action==='accept'&&chatCapabilityMessage(proposal.kind)){capabilityNote.textContent=chatCapabilityMessage(proposal.kind);capabilityNote.hidden=false;return}
  if(action==='accept'&&askSignificance&&!choices.get(proposal.id)){feedback.textContent='Choisissez « Anecdotique » ou « Marquant » pour ce moment avant de l’ajouter.';feedback.className='status error';var firstChoice=box.querySelector('[name=significance]');if(firstChoice)firstChoice.focus();return}
  if(action==='accept'&&replaced.length&&!await confirmDialog('Remplacer '+replacementScope+' ?','Le texte proposé remplacera le texte actuel. Les autres informations seront conservées.','Remplacer'))return;
  accept.disabled=true;reject.disabled=true;box.dataset.submitting='true';feedback.textContent='Enregistrement…';
  try{var result=await api('/api/chat/proposals/'+encodeURIComponent(proposal.id)+'/'+action,'POST',action==='accept'?Object.assign({payload:payload,revision:proposal.revision},askSignificance?{significance:choices.get(proposal.id),entry:'form'}:{}):{revision:proposal.revision});if(!current()||state.tab!=='chat')return;state.chatProposalDrafts.delete(proposal.id);choices.delete(proposal.id);var latest=result.messages?result:await api('/api/chat');if(!current()||state.tab!=='chat')return;state.chatData=latest;reconcileChatTurn(latest);drawChat(document.getElementById('view'),current)}
  catch(err){if(current()){box.dataset.submitting='false';feedback.textContent=err.message+' Vos modifications sont conservées.';feedback.className='status error';accept.disabled=false;reject.disabled=false}}
 }
 host.append(box);
}
function appendAccountQuota(host,me){
 var quotas=me.quotas,messages=quotas&&quotas.messages,reviews=quotas&&quotas.reviews;
 var messageLimit=messages&&Number.isFinite(messages.limit)?messages.limit:100,reviewLimit=reviews&&Number.isFinite(reviews.limit)?reviews.limit:1;
 var until='',renews=messages&&messages.renews_at;if(renews){var date=new Date(typeof renews==='number'?renews*1000:renews);if(Number.isFinite(date.getTime())){var parts=date.toLocaleDateString('fr-FR',{day:'numeric',month:'long',timeZone:'Europe/Paris'}).split(' ');until=', jusqu’au '+(parts[0]==='1'?'1ᵉʳ':parts[0])+' '+parts.slice(1).join(' ')}}
 if(messages&&reviews&&Number.isFinite(messages.remaining)&&Number.isFinite(reviews.remaining))host.append(el('p','account-quota','Il vous reste '+messages.remaining+' message'+(messages.remaining>1?'s':'')+' et '+reviews.remaining+' bilan'+(reviews.remaining>1?'s':'')+' ce mois-ci'+until+'.'));
 host.append(el('p','helper',messageLimit+' messages et '+reviewLimit+' bilan'+(reviewLimit>1?'s':'')+' par mois, bilan initial inclus.'));
 var subscription=me.subscription;if(me.access&&me.access.mode!=='free_preview'&&subscription&&subscription.active&&!subscription.cancel_at_period_end&&subscription.status!=='trialing'&&subscription.paid_until)host.append(el('p','helper','Votre période de facturation en cours se termine le '+dateLabel(subscription.paid_until)+'.'));
}
function renderAccount(){var m=state.me,freePreview=!!(m.access&&m.access.mode==='free_preview'),active=m.subscription&&m.subscription.active,manageable=!!(m.subscription&&m.subscription.manageable),subStatus=m.subscription&&m.subscription.status,billing=m.configuration&&m.configuration.billing,canSubscribe=!active&&(!subStatus||['none','canceled','incomplete_expired'].includes(subStatus));var canStart=!!(m.configuration&&m.configuration.ai);var view=document.getElementById('view');view.innerHTML='<section class="account-panel"><div class="account-row"><h2>Identité</h2><div id="self-profile"></div></div><div class="account-row"><h2>Connexion</h2><p id="account-email" class="muted"></p></div><div class="account-row"><h2>Accès à Ambre</h2><div class="row"><p id="subscription-copy"></p><div class="price" id="account-price">29 € <small>/ mois</small></div></div><div id="account-progress"></div><div id="billing-action"></div><p id="billing-status" class="status" role="status"></p></div><div class="account-row" id="account-appointment"></div><div class="account-row"><h2>Mes données</h2><div class="actions"><button type="button" class="secondary" id="export-button">Exporter</button><button type="button" class="secondary" id="chat-clear-button">Effacer la conversation</button></div><p id="export-status" class="status" role="status"></p><details class="privacy-details"><summary>Comment mes choix sont-ils utilisés ?</summary><p class="legal-space">Vos réponses, vos proches et les événements que vous enregistrez permettent à Ambre de proposer des analyses. Celles-ci restent des interprétations, pas un diagnostic ni une mesure définitive de votre personnalité.</p><p>Les informations nécessaires à l’analyse approfondie, vos proches, vos événements, vos retours et vos corrections sont transmis à OpenRouter pour personnaliser ce service.</p><p>Vous pouvez modifier ou supprimer ce que vous avez enregistré, ou corriger une interprétation. Les analyses suivantes tiennent compte de ces changements.</p></details></div><div class="account-row"><button type="button" class="text danger account-quiet" id="delete-button">Supprimer mon compte</button></div></section>';setupPortraitViews(view);renderSelfProfile();var appointmentHost=document.getElementById('account-appointment');if(m.user.role!=='psychologist'&&typeof renderAccountAppointment==='function')renderAccountAppointment(appointmentHost);else appointmentHost.hidden=true;document.getElementById('account-email').textContent=m.user.email;document.getElementById('account-price').hidden=freePreview;document.getElementById('subscription-copy').textContent=freePreview?'Accès gratuit.':active?'Abonnement actif.':['past_due','unpaid','incomplete'].includes(subStatus)?'Votre paiement nécessite une vérification. Retrouvez vos factures et votre moyen de paiement dans l’espace de facturation.':'Un abonnement pour échanger avec Ambre et faire le point.';var offer=el('p','helper',freePreview||active?'':m.subscription&&m.subscription.trial_eligible===false?'Un nouvel abonnement est facturé immédiatement : 29 € par mois.':'7 jours d’essai, puis 29 € par mois. Annulez avant la fin de l’essai pour éviter le premier paiement.');var progress=document.getElementById('account-progress');appendAccountQuota(progress,m);if(offer.textContent)progress.append(offer);var area=document.getElementById('billing-action');function billingButton(label,path,cls){return actionButton(label,async function(e){var b=e.currentTarget;b.disabled=true;status('billing-status','Ouverture de votre espace de paiement…');try{var data=await api(path,'POST',{});var u=new URL(data.url);if(u.protocol!=='https:')throw new Error('Le lien de paiement est indisponible.');location.assign(u.href)}catch(err){status('billing-status',err.message,true);b.disabled=false}},cls)}if(!freePreview&&!billing)area.append(el('div','notice warn','Les nouveaux abonnements ne sont pas encore ouverts.'));var actions=el('div','actions');if(!freePreview&&canSubscribe&&billing){var subscribe=billingButton(m.subscription&&m.subscription.trial_eligible===false?'M’abonner pour 29 € par mois':'Commencer mes 7 jours d’essai','/api/billing/checkout');subscribe.disabled=!canStart;actions.append(subscribe);if(!canStart)area.append(el('p','helper','La personnalisation attend l’activation de l’IA avant l’ouverture de votre abonnement.'))}if(!freePreview&&manageable)actions.append(billingButton(canSubscribe?'Mes factures et moyens de paiement':'Gérer mon abonnement','/api/billing/portal',canSubscribe?'secondary':''));if(!freePreview&&active&&!(m.subscription&&m.subscription.cancel_at_period_end)){actions.append(actionButton(subStatus==='trialing'?'Annuler avant la fin de l’essai':'Résilier à la fin de la période',async function(e){var button=e.currentTarget;button.disabled=true;status('billing-status','Enregistrement de votre résiliation…');try{await api('/api/billing/cancel','POST',{});state.me=await api('/api/me');renderAccount()}catch(err){status('billing-status',err.message,true);button.disabled=false}},'secondary'))}if(!freePreview&&m.subscription&&m.subscription.cancel_at_period_end)area.append(el('p','notice','Résiliation enregistrée. Votre accès reste ouvert jusqu’au '+dateLabel(m.subscription.access_until||m.subscription.trial_end)+'.'));else if(!freePreview&&subStatus==='trialing'&&m.subscription.trial_end)area.append(el('p','notice','Votre essai se termine le '+dateLabel(m.subscription.trial_end)+'. Vous pouvez annuler avant cette date.'));area.append(actions);if(m.user.role==='psychologist'){document.getElementById('account-progress').closest('.account-row').hidden=true;var privacy=view.querySelector('.privacy-details');privacy.replaceChildren(el('summary','','Comment les dossiers sont-ils partagés ?'),el('p','','Vous consultez uniquement les rubriques autorisées après activation du code transmis par la personne. Les consultations sont journalisées, et la personne peut révoquer votre accès à tout moment. Vos informations professionnelles sont déclarées, non vérifiées.'))}var sessionActions=el('div','actions');sessionActions.append(actionButton('Se déconnecter',logout,'text'),actionButton('Se déconnecter partout',function(){logout('all')},'text'));document.getElementById('account-email').parentElement.append(sessionActions);var chatClear=document.getElementById('chat-clear-button');if(chatClear){if(typeof openChangeChatClear==='function')chatClear.addEventListener('click',function(){openChangeChatClear(function(){return state.tab==='account'})});else chatClear.remove()}document.getElementById('export-button').addEventListener('click',async function(e){var b=e.currentTarget;b.disabled=true;status('export-status','Préparation…');try{var data=await api('/api/export');var url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));var a=el('a');a.href=url;a.download='mon-journal-decode-ton-esprit.json';document.body.append(a);a.click();a.remove();setTimeout(function(){URL.revokeObjectURL(url)},1000);status('export-status','Votre téléchargement est lancé.')}catch(err){status('export-status',err.message,true)}finally{b.disabled=false}});document.getElementById('delete-button').addEventListener('click',function(){document.getElementById('delete-form').reset();status('delete-status','');var deleteDialog=document.getElementById('delete-dialog');deleteDialog.showModal();focusEntry(document.getElementById('delete-password'),deleteDialog.querySelector('h2'))})}
async function logout(scope){if(scope==='all'&&!await confirmDialog('Se déconnecter partout ?','Tous vos appareils seront déconnectés, y compris celui-ci. Vous devrez vous reconnecter.','Se déconnecter'))return;try{disposeChatInterface();var leaving=state.me&&state.me.user&&state.me.user.id;await api(scope==='all'?'/api/auth/logout-all':'/api/auth/logout','POST',{});state.me={user:null};preparedResponses={};state.navigationRequest=(state.navigationRequest||0)+1;resetChatSession();state.chatDraft='';state.chatSendKey=null;state.chatProposalDrafts.clear();clearTimeout(state.chatTimer);clearTimeout(state.chatCooldownTimer);state.chatData=null;state.sourceDrafts.clear();state.people=[];state.events=[];state.selfDraft=null;state.selfRevision=null;state.selfBase=null;state.mapReturnDrafts.clear();clearTimeout(state.portraitTimer);if(state.noteDrafts)state.noteDrafts.clear();landing()}catch(err){var failure=el('div','notice warn',err.message);failure.setAttribute('role','alert');main.prepend(failure)}}
document.addEventListener('click',function(e){var auth=e.target.closest('[data-auth]');if(auth)openAuth(auth.dataset.auth);var close=e.target.closest('[data-close]');if(close)document.getElementById(close.dataset.close).close()});
document.getElementById('auth-role').addEventListener('change',updateAuthRole);
document.getElementById('auth-role-toggle').addEventListener('click',function(){var role=document.getElementById('auth-role');role.value=role.value==='psychologist'?'member':'psychologist';updateAuthRole()});
document.getElementById('auth-form').addEventListener('submit',async function(e){e.preventDefault();var form=e.currentTarget;busy(form,true);status('auth-status','Connexion…');try{var data=formValues(form);if(state.authMode==='register'&&data.role==='psychologist')data.professional={name:data.professional_name,title:data.professional_title,registration_number:data.registration_number};await api('/api/auth/'+state.authMode,'POST',data);await refresh();document.getElementById('auth-dialog').close();form.reset()}catch(err){status('auth-status',err.message,true)}finally{busy(form,false)}});
document.getElementById('delete-form').addEventListener('submit',async function(e){e.preventDefault();var form=e.currentTarget;busy(form,true);status('delete-status','Suppression…');try{var leaving=state.me&&state.me.user&&state.me.user.id;await api('/api/account','DELETE',formValues(form));document.getElementById('delete-dialog').close();state.me={user:null};preparedResponses={};state.navigationRequest=(state.navigationRequest||0)+1;resetChatSession();state.chatDraft='';state.chatSendKey=null;state.chatProposalDrafts.clear();clearTimeout(state.chatTimer);clearTimeout(state.chatCooldownTimer);state.chatData=null;state.sourceDrafts.clear();state.people=[];state.events=[];state.selfDraft=null;state.selfRevision=null;state.selfBase=null;state.mapReturnDrafts.clear();clearTimeout(state.portraitTimer);if(state.noteDrafts)state.noteDrafts.clear();landing()}catch(err){status('delete-status',err.message,true)}finally{busy(form,false)}});
renderPageLoading();
refresh().catch(function(err){main.replaceChildren(el('div','notice warn','Votre espace n’a pas pu être ouvert. '+err.message),actionButton('Réessayer',function(){location.reload()},'secondary'))});
</script></body></html>
`;

// Lot P: the application script and style are public, identical for every account and
// served as content-addressed files (/app/<hash>.js|.css, cached immutably). The private
// page keeps only its shell and, in a non-executable JSON block, the prepared responses.
function contentHash(text){
  // Two independent 32-bit FNV-1a lanes: a stable 64-bit name, enough to bust caches.
  let a=0x811c9dc5,b=0x01000193^0x9e3779b9;
  for(let i=0;i<text.length;i++){const c=text.charCodeAt(i);a=Math.imul(a^c,0x01000193);b=Math.imul(b^c,0x01000193)^(b>>>13);}
  return (a>>>0).toString(16).padStart(8,'0')+(b>>>0).toString(16).padStart(8,'0')+text.length.toString(16);
}
const PARTS=(()=>{
  let script='',style='';
  const shell=DOCUMENT_HTML.replace(/<script>([\s\S]*?)<\/script>/,(_match,source)=>{script=source;return ''}).replace(/<style>([\s\S]*?)<\/style>/,(_match,source)=>{style=source;return ''});
  return {shell,script:'document.addEventListener("DOMContentLoaded",function(){\n'+script+'\n});\n',style};
})();
export const APP_SCRIPT=PARTS.script;
export const APP_STYLE=PARTS.style;
export const APP_ASSETS=Object.freeze({
  script:Object.freeze({path:'/app/'+contentHash(APP_SCRIPT)+'.js',type:'application/javascript; charset=utf-8',body:APP_SCRIPT}),
  style:Object.freeze({path:'/app/'+contentHash(APP_STYLE)+'.css',type:'text/css; charset=utf-8',body:APP_STYLE}),
});
export const INITIAL_PAGE_DATA_MARKER='/*INITIAL_PAGE_DATA*/null';
// The deferred script parses with the page and runs before DOMContentLoaded, as the inline script did.
export const HTML=PARTS.shell
  .replace('</head>',()=>'<link rel="stylesheet" href="'+APP_ASSETS.style.path+'"><script defer src="'+APP_ASSETS.script.path+'"></script></head>')
  .replace('</body>',()=>'<script type="application/json" id="initial-page-data">'+INITIAL_PAGE_DATA_MARKER+'</script></body>');
