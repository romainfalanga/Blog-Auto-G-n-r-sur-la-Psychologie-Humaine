import {CARE_INTERFACE_SCRIPT} from './care-interface.mjs';
import {SETTINGS_INTERFACE_SCRIPT} from './settings-interface.mjs';
import { CHANGE_INTERFACE_SCRIPT, CHANGE_INTERFACE_STYLE } from './change-interface.mjs';
import { HEADER_BRAND_HTML, NAVIGATION_HTML, SITE_FOOTER_HTML, SITE_SYSTEM_CSS, SITE_NAVIGATION_SCRIPT } from './navigation.generated.mjs';
import { VOICE_INPUT_SCRIPT } from './voice-input.mjs';
import { WORKSPACE_STYLE } from './workspace-style.mjs';
import { CHAT_INTERFACE_SCRIPT } from './chat-interface.mjs';
import { PERSON_CHOICES, PERSON_TEXTS, EVENT_CHOICES, EVENT_TEXTS } from './context-fields.mjs';
import { PEOPLE_AXES_UI } from './people-observations-ui.mjs';
const DOCUMENT_HTML = String.raw`<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, interactive-widget=resizes-content"><meta name="theme-color" content="#F2F5F7"><title>Décode ton esprit</title><style>
${SITE_SYSTEM_CSS}
${WORKSPACE_STYLE}
${CHANGE_INTERFACE_STYLE}
</style></head><body><header class="site-header">
  <div class="header-inner">
    ${HEADER_BRAND_HTML}
    <nav class="site-navigation" id="site-navigation" aria-label="Navigation principale">
      <ul class="nav-links">
        ${NAVIGATION_HTML}
      </ul>
    </nav>
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
 resetChatSession();clearTimeout(state.chatTimer);clearTimeout(state.chatCooldownTimer);clearTimeout(state.journeyTimer);state.chatDraft='';state.chatData=null;state.chatRequest=null;state.chatSending=false;state.chatProposalDrafts.clear();state.journeyKey=null;state.journeyWaiting=false;state.journeyStep=null;preparedResponses={};state.navigationRequest=(state.navigationRequest||0)+1;
 state.sourceDrafts.clear();state.selfDraft=null;state.selfBase=null;state.selfRevision=null;if(state.noteDrafts)state.noteDrafts.clear();state.people=[];state.events=[];
}
var PAGES={chat:{path:'/mon-accompagnement/',title:'Ambre'},people:{path:'/mes-proches/',title:'Mes proches'},person:{path:'/mes-proches/',title:'Mes proches'},events:{path:'/ma-ligne-de-vie/',title:'Ma ligne de vie'},reviews:{path:'/mes-bilans/',title:'Mes bilans'},account:{path:'/mon-compte/',title:'Mon compte'},settings:{path:'/parametres/',title:'Paramètres'},care:{path:'/mon-suivi/',title:'Mon suivi'},psychologist:{path:'/espace-psychologue/',title:'Espace psychologue'}};
var PAGE_DESCRIPTIONS={
 reviews:'',
 settings:'',
 care:'',
 psychologist:'Consultez les dossiers que les personnes ont choisi de partager avec vous.',
 account:''
};
// One close person has an address of their own: /mes-proches/<id>/.
function personIdForPath(path){var match=String(path).match(/^\/mes-proches\/([\w-]+)\/?$/);return match?match[1]:null}
function pagePath(tab,personId){return tab==='person'?'/mes-proches/'+encodeURIComponent(personId)+'/':PAGES[tab].path}
function pageForPath(path){if(personIdForPath(path))return 'person';var normalizedPath=path.replace(/\/+$/,'')+'/';if(normalizedPath==='/mon-portrait/')return 'account';return Object.keys(PAGES).find(function(key){return PAGES[key].path===normalizedPath})||'chat'}
function capturePageDrafts(){var chatInput=document.getElementById('chat-content');if(chatInput)state.chatDraft=chatInput.value;document.querySelectorAll('[data-source-form]').forEach(function(form){captureSourceDraft(form)})}
function syncPageLocation(){query=new URLSearchParams(location.search);state.tab=pageForPath(location.pathname);state.personId=personIdForPath(location.pathname);state.pendingArticle=query.get('article');state.pendingLike=query.get('like')==='1';state.pendingError=null;syncPageChrome()}
function syncPageChrome(){document.title='Décode ton esprit';document.querySelectorAll('#site-navigation a').forEach(function(link){if(link.getAttribute('href')===PAGES[state.tab].path)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current')})}

async function go(tab,search,personId){
 if(tab==='likes'||tab==='portrait'){tab='account';search=(search||'').split('#')[0]+'#self-profile'}
 if(tab==='person'&&!personId)tab='people';
 if(!PAGES[tab])return;
 capturePageDrafts();
 var request=state.navigationRequest=(state.navigationRequest||0)+1,owner=state.me&&state.me.user&&state.me.user.id;
 var responses=owner?await preparePageData(tab,search,personId):{};
 if(request!==state.navigationRequest||owner!==(state.me&&state.me.user&&state.me.user.id))return;
 preparedResponses=responses;
 var target=pagePath(tab,personId)+(search||'');
 if(location.pathname+location.search+location.hash!==target)history.pushState({page:tab},'',target);
 syncPageLocation();
 if(state.me&&state.me.user&&state.pendingLike)applyPendingLike().then(renderApp);
 else renderApp();
 window.scrollTo({top:0,behavior:'instant'})
}
window.addEventListener('popstate',function(){go(pageForPath(location.pathname),location.search+location.hash,personIdForPath(location.pathname))});
document.addEventListener('click',function(event){var link=event.target.closest('a[href]');if(!link||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||link.target==='_blank'||link.hasAttribute('download'))return;var url=new URL(link.href,location.href);if(url.origin!==location.origin)return;var personId=personIdForPath(url.pathname),page=personId?'person':Object.keys(PAGES).find(function(key){return PAGES[key].path===url.pathname.replace(/\/+$/,'')+'/'});if(!page)return;event.preventDefault();go(page,url.search+url.hash,personId)});

var initialPageData=/*INITIAL_PAGE_DATA*/null;
var preparedResponses=initialPageData&&initialPageData.page===location.pathname.replace(/\/+$/,'')+'/'?initialPageData.responses||{}:{};
async function preparePageData(tab,search,personId){
 var paths={account:['/api/self-profile'],people:['/api/people'],events:['/api/people','/api/life-events']}[tab]||[];
 if(tab==='person'){var base='/api/people/'+encodeURIComponent(personId);paths=[base,base+'/timeline']}
 if(tab==='events'){var filter=new URLSearchParams((search||'').split('#')[0]).get('person');if(filter)paths[1]+='?person_id='+encodeURIComponent(filter)}
 var entries=await Promise.all(paths.map(async function(path){try{return [path,{status:200,body:await api(path)}]}catch(error){return [path,{status:error.status||503,body:{error:error.message,code:error.code}}]}}));return Object.fromEntries(entries)
}
var query=new URLSearchParams(location.search);
var state={chatOutbox:null,chatSeenResponses:new Set(),chatDraft:'',chatProposalDrafts:new Map(),chatSendKey:null,chatSending:false,me:null,tab:pageForPath(location.pathname),personId:personIdForPath(location.pathname),authMode:'login',examples:null,likes:[],drafts:new Map(),differencesDrafts:new Map(),people:[],events:[],sourceDrafts:new Map(),selfDraft:null,selfRevision:null,mapReturnDrafts:new Map(),pendingArticle:query.get('article'),pendingLike:query.get('like')==='1',pendingError:null};
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
function safeArticleLink(example,label){var a=el('a','',label||'Lire');try{if(typeof example.url!=='string'||!example.url.trim())throw new Error();var u=new URL(example.url,location.origin);if(u.protocol!=='https:'&&u.protocol!=='http:')throw new Error();a.href=u.href;a.target='_blank';a.rel='noopener'}catch(e){a.removeAttribute('href')}return a}
function landing(){
 syncPageChrome();
 main.className='';
 main.innerHTML='<section class="journal-welcome"><h1 id="public-page-title"></h1><p>Connectez-vous pour ouvrir votre espace.</p><div id="pending-note" class="notice" hidden>Connectez-vous pour conserver cet article.</div><div class="actions"><button type="button" data-auth="login">Me connecter</button><button type="button" class="secondary" data-auth="register">Créer mon compte</button></div></section>';
 document.getElementById('public-page-title').textContent=PAGES[state.tab].title;
 document.getElementById('pending-note').hidden=!(state.pendingArticle&&state.pendingLike)
}
async function applyPendingLike(){if(!state.pendingArticle||!state.pendingLike||!state.me.user)return;try{await api('/api/likes/'+encodeURIComponent(state.pendingArticle),'PUT',{liked:true});state.pendingError=null;state.pendingLike=false;var url=new URL(location.href);url.searchParams.delete('like');history.replaceState({},'',url.pathname+url.search+url.hash);state.me=await api('/api/me')}catch(err){state.pendingError=err.message}}
function renderPageLoading(){
 syncPageChrome();main.className=state.tab==='chat'?'conversation-page':'';
 var content=el('section','panel narrow personal-section'),loading=el('p','muted page-loading','Chargement…');loading.setAttribute('role','status');content.setAttribute('aria-busy','true');content.append(loading);
 main.replaceChildren();if(state.tab!=='chat'){var head=el('div','app-head');head.append(el('h1','',PAGES[state.tab].title));main.append(head)}main.append(content)
}
async function refresh(){var request=++sessionRefreshRequest,me=await api('/api/me');if(request!==sessionRefreshRequest)return;if(state.me&&state.me.user&&(!me.user||state.me.user.id!==me.user.id))clearSessionState();state.me=me;syncRoleNavigation();if(state.me.user){await applyPendingLike();await renderApp()}else landing();}
function syncRoleNavigation(){
 var professional=state.me&&state.me.user&&state.me.user.role==='psychologist',list=document.querySelector('#site-navigation .nav-links');if(!list)return;
 list.querySelectorAll('li').forEach(function(item){var link=item.querySelector('a');if(link)item.hidden=professional&&link.getAttribute('href')!=='/parametres/'&&link.getAttribute('href')!=='/espace-psychologue/'});
 var existing=document.getElementById('professional-nav');if(professional&&!existing){var item=el('li'),link=el('a','','Espace psychologue');item.id='professional-nav';link.href='/espace-psychologue/';item.append(link);list.append(item)}else if(existing)existing.hidden=!professional;
}
function updateAuthRole(){var register=state.authMode==='register',role=document.getElementById('auth-role'),fields=document.getElementById('auth-register-fields'),professional=document.getElementById('auth-professional');fields.hidden=!register;professional.hidden=!register||role.value!=='psychologist';var toggle=document.getElementById('auth-role-toggle');toggle.textContent=role.value==='psychologist'?'Je ne suis pas psychologue':'Je suis psychologue';document.getElementById('auth-password-help').hidden=!register;professional.querySelectorAll('input').forEach(function(input){input.required=!professional.hidden;input.disabled=professional.hidden})}
function openAuth(mode){state.authMode=mode;updateAuthRole();var register=mode==='register';document.getElementById('auth-title').textContent=register?'Créer mon compte':'Me connecter';document.getElementById('auth-description').textContent='';document.getElementById('auth-submit').textContent=register?'Créer mon compte':'Se connecter';document.getElementById('auth-password').autocomplete=register?'new-password':'current-password';document.getElementById('auth-password').minLength=12;document.getElementById('auth-switch').replaceChildren(document.createTextNode(register?'Déjà un compte ? ':'Pas encore de compte ? '),actionButton(register?'Se connecter':'Créer un compte',function(){openAuth(register?'login':'register')},'text'));status('auth-status','');var dialog=document.getElementById('auth-dialog');if(!dialog.open){dialog.showModal();focusEntry(document.getElementById('auth-email'),document.getElementById('auth-title'))}}
function renderApp(){
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
 if(state.pendingError){
  var warning=el('div','notice warn',state.pendingError);
  warning.append(actionButton('Réessayer d’enregistrer ce J’aime',async function(){await applyPendingLike();renderApp()},'text'));
  messages.append(warning)
 }
 if(chat&&state.me.configuration&&state.me.configuration.ai===false){
  messages.append(el('div','notice warn','Ambre est momentanément indisponible. Vos informations enregistrées restent accessibles.'))
 }
 if(state.tab==='settings')renderSettings(document.getElementById('view'),renderGuard(document.getElementById('view')));
 else if(state.tab==='care')renderCareMember(document.getElementById('view'),renderGuard(document.getElementById('view')));
 else if(state.tab==='psychologist')renderCarePsychologist(document.getElementById('view'),renderGuard(document.getElementById('view')));
 else if(chat)return renderChat();
 else if(state.tab==='account')renderAccount();
 else if(state.tab==='reviews')renderReviews();
 else if(state.tab==='person')renderPerson();
 else renderPersonalSources()
}
function updateLikeState(result){if(typeof result.source_version==='number'&&typeof state.likeVersion==='number'&&result.source_version<state.likeVersion)return;if(typeof result.source_version==='number')state.likeVersion=result.source_version;state.likes=result.likes||state.likes}
var recommendationRequest=0;
async function refreshRecommendations(){var container=document.getElementById('recommendations-body');if(state.tab!=='portrait'||!container)return;var request=++recommendationRequest;status('recommendations-status','Actualisation des suggestions…');try{var result=await api('/api/recommendations');if(request!==recommendationRequest||!container.isConnected||state.tab!=='portrait')return;if(Number(result.source_version)<Number(state.likeVersion||0)){status('recommendations-status','Vos choix viennent de changer. Les suggestions seront actualisées avec vos derniers enregistrements.');return}var version=[Number(result.source_version)||0,Number(result.portrait_version)||0];if(state.recommendationVersion&&(version[0]<state.recommendationVersion[0]||version[0]===state.recommendationVersion[0]&&version[1]<state.recommendationVersion[1]))return;state.recommendationVersion=version;container.replaceChildren();var liked=new Set(state.likes.map(function(item){return item.article_id}));var items=(result.items||[]).filter(function(item){return !liked.has(item.id)}).slice(0,12);if(!items.length){container.append(el('p','muted',result.likes_count>0?'Aucune nouvelle suggestion n’est disponible pour ces choix pour le moment. Retrouvez les autres lectures sur l’accueil.':'Aimez un article ou décrivez un élément personnel pour donner un point de départ à vos suggestions. Vous pouvez aussi explorer librement les lectures sur l’accueil.'))}items.forEach(function(item){var card=el('article','recommendation-card');card.append(el('h3','',item.title));if(item.description)card.append(el('p','muted',item.description));var tags=Array.isArray(item.keywords)?item.keywords.slice(0,3):[];if(tags.length)card.append(el('p','article-keywords',tags.join(' · ')));if(Array.isArray(item.reasons)&&item.reasons.length)card.append(el('p','recommendation-reason',item.reasons.slice(0,2).join(' ')));var footer=el('div','recommendation-footer');if(item.character)footer.append(el('span','example-character',item.character));footer.append(safeArticleLink(item,'Lire'));card.append(footer);container.append(card)});status('recommendations-status','')}catch(err){if(request!==recommendationRequest||!container.isConnected||state.tab!=='portrait')return;status('recommendations-status',err.message,true);var retry=actionButton('Réessayer',refreshRecommendations,'secondary');container.replaceChildren(retry)}}
var collectionRefreshBusy=false,collectionRefreshedAt=0;
async function refreshCollectionOnReturn(){if(document.hidden||!state.me||!state.me.user||state.tab!=='portrait'||!state.refreshLikesView||collectionRefreshBusy||Date.now()-collectionRefreshedAt<1000)return;collectionRefreshBusy=true;collectionRefreshedAt=Date.now();var redraw=state.refreshLikesView;try{var result=await api('/api/likes');if(state.tab!=='portrait'||state.refreshLikesView!==redraw)return;var previous=Number(state.likeVersion)||0;updateLikeState(result);if(Number(state.likeVersion)>previous){redraw();refreshPortraitData()}}catch(err){if(state.tab==='portrait')status('likes-status','Actualisation impossible : '+err.message,true)}finally{collectionRefreshBusy=false}}
window.addEventListener('focus',refreshCollectionOnReturn);
document.addEventListener('visibilitychange',function(){if(!document.hidden)refreshCollectionOnReturn()});

var renderTokens=new WeakMap();
function renderGuard(node){var token={};renderTokens.set(node,token);return function(){return node.isConnected&&renderTokens.get(node)===token}}
async function renderLikes(){var view=document.getElementById('mes-articles'),current=renderGuard(view);view.replaceChildren();var section=el('section','panel narrow');section.append(el('h2','','Articles aimés'),el('p','panel-intro','Retrouvez les lectures qui vous ont parlé et les notes que vous avez gardées.'));var legacy=el('span');legacy.id='suggestions';legacy.tabIndex=-1;section.append(legacy);var message=el('p','status');message.id='likes-status';message.setAttribute('role','status');var list=el('div','liked-articles');list.id='likes-list';list.append(el('p','muted','Chargement de vos articles…'));section.append(message,list);view.append(section);try{var results=await Promise.all([state.examples?Promise.resolve({examples:state.examples}):api('/api/examples'),api('/api/likes')]);if(!current()||state.tab!=='portrait')return;state.examples=results[0].examples||[];updateLikeState(results[1]);state.refreshLikesView=function(){if(current()&&state.tab==='portrait')drawLikedArticles(list,current)};state.refreshLikesView();if(['#mes-articles','#suggestions'].includes(location.hash)||state.pendingArticle){var focusCard=document.getElementById('selected-liked-article')||view;focusCard.scrollIntoView({block:'start'})}collectionRefreshedAt=Date.now()}catch(err){if(current()&&state.tab==='portrait')list.replaceChildren(el('p','status error',err.message),actionButton('Réessayer',renderLikes,'secondary'))}}
function drawLikedArticles(list,current){list.replaceChildren();var examples=new Map((state.examples||[]).map(function(item){return[item.id,item]}));state.likes.forEach(function(like,index){var example=examples.get(like.article_id)||Object.assign({id:like.article_id},like),id=example.id;var card=el('article','liked-article');var details=el('details');details.open=id===state.pendingArticle||state.drafts.has(id)||state.differencesDrafts.has(id);details.append(el('summary','liked-article-title',example.title||'Article aimé'));var read=safeArticleLink(example,'Lire');read.className='article-read';read.setAttribute('aria-label','Lire : '+(example.title||'article aimé'));var form=el('form'),label=el('label','','Ce qui me parle et pourquoi je me reconnais'),input=el('textarea');input.id='reason-'+index;input.name='reason';label.htmlFor=input.id;input.maxLength=4000;input.value=state.drafts.has(id)?state.drafts.get(id):like.reason||'';input.addEventListener('input',function(){state.drafts.set(id,input.value)});form.append(label,input);var differencesInput=null;if(like.differences||state.differencesDrafts.has(id)){var legacy=el('details','legacy-return');legacy.append(el('summary','','Mon ancien retour sur les différences'));var field=el('div'),differencesLabel=el('label','','Ce qui est différent pour moi (facultatif)');differencesInput=el('textarea');differencesInput.id='differences-'+index;differencesInput.name='differences';differencesLabel.htmlFor=differencesInput.id;differencesInput.maxLength=4000;differencesInput.value=state.differencesDrafts.has(id)?state.differencesDrafts.get(id):like.differences||'';differencesInput.addEventListener('input',function(){state.differencesDrafts.set(id,differencesInput.value)});field.append(differencesLabel,differencesInput);legacy.append(field);form.append(legacy)}var actions=el('div','actions'),save=el('button','secondary','Enregistrer mon retour');save.type='submit';var remove=actionButton('Retirer des articles aimés',async function(event){var button=event.currentTarget;button.disabled=true;try{var result=await api('/api/likes/'+encodeURIComponent(id),'PUT',{liked:false});if(!current())return;updateLikeState(result);state.drafts.delete(id);state.differencesDrafts.delete(id);drawLikedArticles(list,current);refreshPortraitData();status('likes-status','Article retiré.')}catch(err){if(current()){status('likes-status',err.message,true);button.disabled=false}}},'text');actions.append(save,remove);var resultText=el('p','status');resultText.setAttribute('role','status');form.append(actions,resultText);form.addEventListener('submit',async function(event){event.preventDefault();busy(form,true);resultText.textContent='Enregistrement…';var submittedReason=input.value,submittedDifferences=differencesInput?differencesInput.value:undefined;state.drafts.set(id,submittedReason);if(differencesInput)state.differencesDrafts.set(id,submittedDifferences);try{var payload={liked:true,reason:submittedReason};if(differencesInput)payload.differences=submittedDifferences;var result=await api('/api/likes/'+encodeURIComponent(id),'PUT',payload);if(!current())return;updateLikeState(result);if(state.drafts.get(id)===submittedReason)state.drafts.delete(id);if(differencesInput&&state.differencesDrafts.get(id)===submittedDifferences)state.differencesDrafts.delete(id);refreshPortraitData();resultText.textContent='Retour enregistré.';resultText.className='status success'}catch(err){if(!current())return;resultText.textContent=err.message;resultText.className='status error'}finally{busy(form,false)}});details.append(form);card.append(details,read);if(state.pendingArticle===id)card.id='selected-liked-article';list.append(card)});if(!state.likes.length){var empty=el('p','muted','Aucun article aimé pour le moment. '),browse=el('a','','Explorer les lectures');browse.href='/';empty.append(browse);list.append(empty)}}

function portraitEvidence(sources){var evidence=el('p','helper','À l’origine de cette piste : ');sources.forEach(function(source,index){var id=String(typeof source==='object'?source.source_id||source.id:source),kind=id.split(':')[0],key=id.includes(':')?id.slice(id.indexOf(':')+1):id;var article=(state.examples||[]).find(function(item){return item.id===key})||state.likes.find(function(item){return item.article_id===key});var names={profile:'Portrait déclaré',self_profile:'Portrait déclaré',person:'Relation décrite',event:'Événement raconté',memory:'Votre correction',feedback:'Votre retour de lecture',story:'Votre retour de lecture'};var link=el('a','',names[kind]||article&&article.title||'Lecture aimée');if(kind==='feedback'||kind==='story')link=el('span','','Un retour de lecture antérieur');else if(kind==='person'||kind==='event')link.href=(kind==='person'?'/mes-proches/#person-':'/ma-ligne-de-vie/#event-')+encodeURIComponent(key);else if(kind==='self_profile'||kind==='profile')link.href='/mon-compte/#self-profile';else if(kind==='memory')link.href='/mon-accompagnement/';else link.href='/mon-portrait/?article='+encodeURIComponent(key)+'#mes-articles';evidence.append(link,document.createTextNode(index<sources.length-1?' ; ':''))});return evidence}
function personalizePortraitText(value,subject){
  var text=String(value||''),identity=subject||state.portraitData&&state.portraitData.subject||{},name=String(identity.first_name||'').trim();
  var masculine={créative:'créatif',réfléchie:'réfléchi',attentive:'attentif',curieuse:'curieux',intuitive:'intuitif',réservée:'réservé',motivée:'motivé',engagée:'engagé',ouverte:'ouvert',convaincue:'convaincu',prudente:'prudent',inquiète:'inquiet',soutenue:'soutenu',comprise:'compris',accompagnée:'accompagné',épuisée:'épuisé',reconnue:'reconnu',sollicitée:'sollicité',heureuse:'heureux',seule:'seul',exigeante:'exigeant',spontanée:'spontané',indécise:'indécis',confiante:'confiant',bienveillante:'bienveillant',protectrice:'protecteur',anxieuse:'anxieux',stressée:'stressé',fatiguée:'fatigué',blessée:'blessé',valorisée:'valorisé',jugée:'jugé',rassurée:'rassuré',attachée:'attaché',dépendante:'dépendant',présente:'présent',loyale:'loyal'};
  var unchanged=['sensible','calme','autonome','lucide','stable','fragile','vulnérable','capable','disponible','responsable','solidaire','sociable','proche','perfectionniste','enthousiaste','optimiste','pessimiste','timide','authentique'];
  function agrees(tail){
    var predicate=tail.match(/^(\s+(?:(?:est|semble|paraît|apparaît|reste)(?:\s+être)?|se\s+(?:montre|sent|dit)|se\s+(?:décrit|perçoit)(?:\s+(?:souvent|parfois|plutôt))?\s+comme|dit\s+être|peut\s+(?:se\s+sentir|être)|a\s+l[’']air)\s+)/i);if(!predicate)return tail;
    var prefix=predicate[0],rest=tail.slice(prefix.length),offset=0,output='';
    while(offset<rest.length){
      var attribute=rest.slice(offset).match(/^((?:(?:très|plutôt|particulièrement|assez|souvent|parfois|aussi|encore|plus|moins|peu|vraiment)\s+)*)([A-Za-zÀ-ÿ-]+)/i);if(!attribute)break;
      var word=attribute[2],lower=word.toLowerCase(),replacement=masculine[lower];if(!replacement&&!unchanged.includes(lower))break;
      output+=attribute[1]+(replacement?(word[0]===word[0].toUpperCase()?replacement[0].toUpperCase()+replacement.slice(1):replacement):word);offset+=attribute[0].length;
      var connector=rest.slice(offset).match(/^(\s*,\s*(?:(?:et|mais|ou)\s+)?|\s+(?:et|mais|ou)\s+)/i);if(!connector)break;
      var next=rest.slice(offset+connector[0].length).match(/^(?:(?:très|plutôt|particulièrement|assez|souvent|parfois|aussi|encore|plus|moins|peu|vraiment)\s+)*([A-Za-zÀ-ÿ-]+)/i);if(!next||!masculine[next[1].toLowerCase()]&&!unchanged.includes(next[1].toLowerCase()))break;
      output+=connector[0];offset+=connector[0].length;
    }
    return prefix+output+rest.slice(offset);
  }
  function neutral(tail){
    var predicate=tail.match(/^(\s+(?:se\s+(?:décrit|perçoit)\s+comme|est|semble(?:\s+être)?)\s+)/i);if(!predicate)return tail;
    var rest=tail.slice(predicate[0].length),attribute=rest.match(/^(?:(?:très|plutôt|particulièrement|assez|souvent|parfois|aussi|encore|plus|moins|peu|vraiment)\s+)*([A-Za-zÀ-ÿ-]+)/i);if(!attribute||!masculine[attribute[1].toLowerCase()]&&!unchanged.includes(attribute[1].toLowerCase()))return tail;
    return predicate[0]+(/^\s+semble\s+$/i.test(predicate[0])?'être une personne ':'une personne ')+rest;
  }
  function unquoted(part){var personalized=name?part.replace(/(?:[Ll]a personne|[Cc]ette personne)([^.!?\n]*)/g,function(match,tail){return name+(identity.gender==='male'?agrees(tail):identity.gender==='female'?tail:neutral(tail))}):part;return personalized.replace(/\{\{prenom\}\}/g,function(){return name||'La personne'})}
  var result='',cursor=0,quotes=/«[^»]*»|“[^”]*”|"[^"]*"|‘[^’]*’/g,quote;while((quote=quotes.exec(text))){result+=unquoted(text.slice(cursor,quote.index))+quote[0];cursor=quote.index+quote[0].length}return result+unquoted(text.slice(cursor));
}
function appendPortrait(container,portrait,title){var block=el('section','portrait-reading');if(title)block.append(el('h3','',title));if(portrait.summary)block.append(el('p','',personalizePortraitText(portrait.summary)));if(Array.isArray(portrait.summary_evidence)&&portrait.summary_evidence.length)block.append(portraitEvidence(portrait.summary_evidence));if(portrait.coverage&&Array.isArray(portrait.coverage.pending)&&portrait.coverage.pending.length)block.append(el('p','helper',portrait.coverage.pending.length+' source(s) restent à analyser. Ce portrait ne couvre pas encore tous vos ajouts.'));[['Ce que vos choix suggèrent aujourd’hui',portrait.current_state],['Évolutions observées dans vos retours',portrait.evolution]].forEach(function(entry){if(entry[1]&&entry[1].summary){var part=el('section','portrait-axis');part.append(el('h3','',entry[0]),el('p','',personalizePortraitText(entry[1].summary)));if(Array.isArray(entry[1].evidence)&&entry[1].evidence.length)part.append(portraitEvidence(entry[1].evidence));block.append(part)}});(portrait.axes||[]).forEach(function(axis){var section=el('section','portrait-axis');section.append(el('h3','',axis.label),el('p','',personalizePortraitText(axis.why)));if(typeof axis.confidence==='number')section.append(el('p','helper',axis.confidence>=.75?'Piste plus présente dans les lectures analysées':axis.confidence>=.45?'Piste à confirmer avec vos retours':'Piste fragile, à vérifier'));if(Array.isArray(axis.evidence)&&axis.evidence.length)section.append(portraitEvidence(axis.evidence));block.append(section)});[['Ressources possibles',portrait.strengths],['Ce qui reste incertain',portrait.uncertainties]].forEach(function(entry){if(Array.isArray(entry[1])&&entry[1].length){block.append(el('h3','',entry[0]));var list=el('ul');entry[1].forEach(function(item){var row=el('li','',personalizePortraitText(typeof item==='string'?item:item.text||item.label||''));if(item&&Array.isArray(item.evidence)&&item.evidence.length)row.append(portraitEvidence(item.evidence));list.append(row)});block.append(list)}});container.append(block)}
async function renderPortrait(){
 clearTimeout(state.portraitTimer);state.portraitData=null;
 var view=document.getElementById('view'),current=renderGuard(view);state.portraitCurrent=current;
 view.replaceChildren();
 var self=el('section','panel narrow personal-section');self.id='self-profile';view.append(self);
 setupPortraitViews(view);await renderSelfProfile()
}

function appendPortraitEmpty(body,mode){
 var collecting=mode==='collecting',unavailable=mode==='unavailable',canChat=state.me?.configuration?.ai!==false;
 var title=collecting?'Votre portrait prend forme ici':unavailable?'Vos informations et vos lectures':'En attendant votre synthèse';
 var description=collecting?(canChat?'Vous pouvez commencer par vous présenter ou raconter à Ambre une situation que vous aimeriez mieux comprendre.':'Vous pouvez commencer par vous présenter ou aimer un article qui vous parle.'):unavailable&&canChat?'Vous pouvez compléter votre présentation ou parler avec Ambre.':'Retrouvez votre présentation dans l’onglet Mon portrait.';
 var empty=emptyState('portrait',title,description),actions=el('div','actions');
 function tabAction(label,key,cls){return actionButton(label,function(){var tab=document.querySelector('[data-portrait-tab="'+key+'"]');if(tab){tab.click();tab.focus()}},cls)}
 actions.append(tabAction(collecting?'Me présenter':'Voir ma présentation','self',collecting?'':'secondary'));
 if(canChat&&(collecting||unavailable)){var companion=el('a','button secondary','Parler avec Ambre');companion.href='/mon-accompagnement/';actions.append(companion)}
 empty.append(actions);body.append(empty)
}

function applyPortraitData(data,current){
 if(!current())return;
 var previous=state.portraitData;
 if(previous&&(Number(data.source_version)<Number(previous.source_version)||Number(data.source_version)===Number(previous.source_version)&&Number(data.analyzed_version)<Number(previous.analyzed_version)))return;
 state.portraitData=data;
 function isPending(item){return item.outdated||['queued','updating'].includes(item.status)||Number(item.source_version)>Number(item.analyzed_version)}
 function contentKey(item){return JSON.stringify([item.subject,item.portrait,item.provisional_portrait,item.provisional,item.status==='collecting',Boolean(isPending(item))])}
 var saved=data.portrait,provisional=data.provisional_portrait,collecting=data.status==='collecting';
 var pending=isPending(data),unavailable=state.me?.configuration?.ai===false;
 var paused=data.status==='paused'||pending&&unavailable;
 var body=document.getElementById('portrait-body');
 if(!previous||contentKey(previous)!==contentKey(data)){
  body.replaceChildren();
  if(provisional&&(!saved||pending||data.provisional))appendPortrait(body,provisional,'Premières pistes, à confirmer');
  if(saved){
   if(pending&&provisional){var earlier=el('details','profile-group');earlier.append(el('summary','','Synthèse précédente'));appendPortrait(earlier,saved);body.append(earlier)}
   else appendPortrait(body,saved,'Ce qui se dessine')
  }
  if(!saved&&!provisional)appendPortraitEmpty(body,collecting?'collecting':'pending')
 }
 var statusBox=document.getElementById('portrait-status'),message='';
 if(!collecting||saved||provisional){
  if(['error','failed'].includes(data.status))message=saved?'La mise à jour de la synthèse n’a pas abouti.':'La synthèse complète n’a pas pu être préparée.';
  else if(paused)message=saved?'La mise à jour de la synthèse est en pause.':'La préparation de la synthèse est en pause.';
  else if(pending)message=saved?'La synthèse est en cours de mise à jour.':'La première synthèse est en préparation.';
  else if(saved){var date=data.updated_at&&dateLabel(data.updated_at);message=date?'Synthèse mise à jour le '+date+'.':'Synthèse à jour.'}
  else if(provisional)message='Ces pistes sont provisoires.';
  var retryTime=Number(data.retry_after)*1000;
  if(message&&!unavailable&&['paused','queued','updating'].includes(data.status)&&retryTime>Date.now())message+=' Une nouvelle tentative est possible à partir du '+new Date(retryTime).toLocaleString('fr-FR',{dateStyle:'long',timeStyle:'short'})+'.';
 }
 if(statusBox.textContent!==message)statusBox.textContent=message
}

function showPortraitLoadFailure(body,current){
 if(!state.portraitData&&!body.querySelector('.workspace-empty')){body.replaceChildren();appendPortraitEmpty(body,'unavailable')}
 var statusBox=document.getElementById('portrait-status');
 if(statusBox.querySelector('button'))return;
 var message=state.portraitData?'Le portrait affiché n’a pas pu être actualisé.':'Le portrait n’a pas pu être chargé.';
 var retry=actionButton('Réessayer',async function(){
  var focused=document.activeElement===retry;retry.disabled=true;
  var loaded=await refreshPortraitData(current);
  if(retry.isConnected)retry.disabled=false;
  if(focused&&document.activeElement===document.body){if(retry.isConnected)retry.focus();else if(loaded)body.focus()}
 },'text');
 statusBox.replaceChildren(el('span','',message),retry)
}

async function refreshPortraitData(current=state.portraitCurrent){
 if(!current||!current()||state.tab!=='portrait')return;
 var body=document.getElementById('portrait-body');if(!body)return;var request={};body.portraitRequest=request;
 clearTimeout(state.portraitTimer);
 function latest(){return current()&&state.tab==='portrait'&&body.portraitRequest===request}
 try{
  var data=await api('/api/portrait');if(!latest())return;
  if(!data||!['collecting','ready','queued','updating','error','failed','paused'].includes(data.status))throw new Error('Réponse du portrait incomplète.');
  applyPortraitData(data,current);
  schedulePortraitPoll(current,state.portraitData||data);
  return true
 }catch(err){
  if(!latest())return;
  showPortraitLoadFailure(body,current);
  if(state.portraitData)schedulePortraitPoll(current,state.portraitData)
 }
}

function schedulePortraitPoll(current,data){
 clearTimeout(state.portraitTimer);
 if(!current()||state.tab!=='portrait'||state.me?.configuration?.ai===false||!['queued','updating'].includes(data.status))return;
 state.portraitTimer=setTimeout(function(){
  if(!current()||state.tab!=='portrait')return;
  if(document.hidden){schedulePortraitPoll(current,data);return}
  return refreshPortraitData(current)
 },5000)
}

function mapStateLabel(hypothesis){return hypothesis?({supported:'Éléments concordants',mixed:'À nuancer',exploring:'À préciser',contradicted:'Éléments contraires'}[hypothesis.status]||'À explorer'):'Non exploré'}
function drawPsychologyMap(host,map,current){
  if(!host)return;
  var patterns=Array.isArray(map.patterns)?map.patterns:[],domains=map.domains||[],subdomains=map.subdomains||[],hypotheses=new Map((map.hypotheses||[]).map(function(item){return[item.pattern_id,item]}));
  host.mapData=map;host.mapCurrent=current;
  if(!host.mapCards){
    host.className='panel psychology-map';host.replaceChildren();host.mapCards=new Map();host.mapDomains=new Map();host.mapSubgroups=new Map();host.mapFilter={mode:'active',domain:'',subdomain:'',query:''};
    var heading=el('div','map-heading'),meta=el('div','map-heading-meta'),count=el('p','map-overview-count');count.id='map-overview-count';meta.append(count,el('span','map-heading-note','Chaque piste relie une hypothèse à vos éléments.'));heading.append(el('h2','','Ma cartographie'),meta);host.append(heading);
    var freshness=el('p','map-freshness');freshness.id='map-freshness';host.append(freshness);
    var atlas=el('details','map-atlas');atlas.open=false;atlas.hidden=true;host.mapAtlas=atlas;atlas.append(el('summary','','Parcourir les domaines'));var domainGrid=el('nav','map-domain-grid');domainGrid.id='map-domains';domainGrid.setAttribute('aria-label','Domaines de la cartographie');atlas.append(domainGrid);host.append(atlas);
    var workspace=el('section','map-workspace'),toolbar=el('div','map-toolbar'),modes=el('div','map-modes');modes.setAttribute('aria-label','Repères affichés');
    ['active','all'].forEach(function(mode){var button=actionButton(mode==='active'?'Mes pistes':'Explorer les repères',function(){host.mapMobileReading=false;host.mapFilter.mode=mode;host.mapFilter.domain='';host.mapFilter.subdomain='';renderMapExplorer(host)},'map-mode');button.dataset.mapMode=mode;modes.append(button)});
    var searchField=el('div','map-search-field'),searchLabel=el('label','visually-hidden','Rechercher un repère'),search=el('input');search.type='search';search.id='map-search';searchLabel.htmlFor=search.id;search.placeholder='Rechercher un besoin, une réaction…';search.autocomplete='off';search.addEventListener('input',function(){host.mapFilter.query=search.value;host.mapMobileReading=false;renderMapExplorer(host)});searchField.append(searchLabel,search);host.mapSearchField=searchField;toolbar.append(modes,searchField);workspace.append(toolbar,atlas);
    var filters=el('div','map-filters'),context=el('p','map-context');context.id='map-context';var subLabel=el('label','visually-hidden','Sous-domaine'),subselect=el('select');subselect.id='map-subdomain';subLabel.htmlFor=subselect.id;subselect.addEventListener('change',function(){host.mapFilter.subdomain=subselect.value;renderMapExplorer(host)});var reset=actionButton('Réinitialiser',function(){host.mapFilter={mode:'all',domain:'',subdomain:'',query:''};search.value='';renderMapExplorer(host)},'text map-reset');reset.id='map-reset';filters.append(context,subLabel,subselect,reset);host.mapFilters=filters;workspace.append(filters);
    var layout=el('div','map-layout'),index=el('section','map-index');index.setAttribute('aria-label','Liste des repères');var results=el('p','map-results');results.id='map-results';results.setAttribute('role','status');index.append(results);var list=el('div','map-index-list');list.id='map-index-list';index.append(list);var empty=el('div','map-empty');empty.id='map-active-empty';index.append(empty);var detail=el('section','map-detail');detail.id='map-detail';detail.setAttribute('aria-label','Comprendre la piste sélectionnée');var back=actionButton('Mes pistes',function(){host.mapMobileReading=false;host.dataset.mapReading='false';var selected=host.mapCards.get(host.mapSelected);if(selected)selected.row.focus();if(typeof window!=='undefined'&&typeof window.scrollTo==='function')window.scrollTo({top:host.mapReturnScroll||0,behavior:'instant'})},'text map-back');back.id='map-back';detail.append(back);layout.append(index,detail);workspace.append(layout);host.append(workspace);
    host.append(el('p','map-boundary','Ces repères sont des pistes de compréhension, pas des diagnostics. « Non exploré » ne signifie ni présent ni absent.'));
  }
  document.getElementById('map-overview-count').textContent=hypotheses.size+' piste'+(hypotheses.size>1?'s':'')+' personnelle'+(hypotheses.size>1?'s':'');
  var freshness=document.getElementById('map-freshness');freshness.hidden=!map.outdated;freshness.textContent=map.outdated?'Mise à jour en cours : vos derniers ajouts restent à analyser.':'';
  var domainList=document.getElementById('map-domains');
  domains.forEach(function(domain){
    var entry=host.mapDomains.get(domain.id);if(!entry){var button=actionButton('',function(){host.mapFilter.domain=host.mapFilter.domain===domain.id?'':domain.id;host.mapFilter.subdomain='';host.mapFilter.mode='all';renderMapExplorer(host)},'map-domain-tile');var title=el('span','map-domain-label'),stats=el('span','map-domain-count');button.append(title,stats);entry={button:button,title:title,stats:stats};host.mapDomains.set(domain.id,entry);domainList.append(button)}
    var domainPatterns=patterns.filter(function(pattern){return pattern.domain===domain.id}),active=domainPatterns.filter(function(pattern){return hypotheses.has(pattern.id)}).length;entry.title.textContent=domain.label;entry.stats.textContent=domainPatterns.length+' repères'+(active?' · '+active+' relié'+(active>1?'s':'')+' à vous':'');entry.button.title=domain.description||domain.label;
  });
  var validIds=new Set(patterns.map(function(pattern){return pattern.id}));host.mapCards.forEach(function(entry,id){if(!validIds.has(id)){entry.card.remove();entry.row.remove();host.mapCards.delete(id)}});
  patterns.forEach(function(pattern){
    var hypothesis=hypotheses.get(pattern.id),entry=host.mapCards.get(pattern.id),domain=domains.find(function(item){return item.id===pattern.domain}),subdomain=subdomains.find(function(item){return item.id===pattern.subdomain});
    if(!entry){
      var card=el('article','map-pattern');card.id='map-pattern-'+pattern.id;card.dataset.patternId=pattern.id;card.hidden=true;var cardHead=el('header','map-pattern-head'),category=el('p','map-category'),title=el('h3','map-pattern-heading'),badge=el('span','map-state');title.id='map-title-'+pattern.id;card.setAttribute('aria-labelledby',title.id);cardHead.append(category,title,badge);var content=el('div','map-pattern-content'),reply=drawMapReturn(pattern,current);card.append(cardHead,content,reply);document.getElementById('map-detail').append(card);
      var row=actionButton('',function(){selectMapPattern(host,pattern.id,true)},'map-index-item'),rowTitle=el('span','map-index-title'),rowState=el('span','map-index-state'),rowDescription=el('span','map-index-description');row.id='map-choice-'+pattern.id;row.setAttribute('aria-controls',card.id);row.setAttribute('aria-keyshortcuts','ArrowUp ArrowDown Home End');row.addEventListener('keydown',function(event){var ids=host.mapVisible||[],index=ids.indexOf(pattern.id),next=event.key==='ArrowDown'?Math.min(ids.length-1,index+1):event.key==='ArrowUp'?Math.max(0,index-1):event.key==='Home'?0:event.key==='End'?ids.length-1:null;if(next===null||!ids[next])return;event.preventDefault();selectMapPattern(host,ids[next],false);host.mapCards.get(ids[next]).row.focus()});row.append(rowTitle,rowState,rowDescription);
      entry={card:card,badge:badge,category:category,title:title,content:content,row:row,rowTitle:rowTitle,rowState:rowState,rowDescription:rowDescription};host.mapCards.set(pattern.id,entry);
    }
    entry.pattern=pattern;entry.hypothesis=hypothesis;entry.card.dataset.mapState=hypothesis?hypothesis.status:'unknown';entry.row.dataset.mapState=entry.card.dataset.mapState;entry.badge.textContent=mapStateLabel(hypothesis);entry.title.textContent=pattern.label;entry.category.textContent=[domain&&domain.label,subdomain&&subdomain.label].filter(Boolean).join(' / ');entry.rowTitle.textContent=pattern.label;entry.rowState.textContent=mapStateLabel(hypothesis);entry.rowDescription.textContent=hypothesis&&hypothesis.observation?personalizePortraitText(hypothesis.observation,state.portraitData&&state.portraitData.subject):pattern.description||'';
    var signature=JSON.stringify([pattern,hypothesis,state.portraitData&&state.portraitData.subject]);if(entry.signature!==signature){drawMapPattern(entry.content,pattern,hypothesis,host);entry.signature=signature}
  });
  renderMapExplorer(host);
}
function renderMapExplorer(host){
  var map=host.mapData,filter=host.mapFilter,patterns=map.patterns||[],domains=map.domains||[],subdomains=map.subdomains||[],query=normalized(filter.query.trim()),rank={supported:0,mixed:1,exploring:2,contradicted:3};
  var compact=filter.mode==='active'&&(map.hypotheses||[]).length<=6&&!filter.query&&!filter.domain&&!filter.subdomain;
  if(host.mapSearchField)host.mapSearchField.hidden=compact;if(host.mapFilters)host.mapFilters.hidden=compact;
  document.getElementById('map-results').hidden=compact;
  if(host.mapAtlas){host.mapAtlas.hidden=filter.mode!=='all';host.mapAtlas.open=filter.mode==='all'}
  host.dataset.mapReading=String(!!host.mapMobileReading);var back=document.getElementById('map-back');if(back)back.textContent=filter.mode==='active'?'Mes pistes':'Les repères';
  host.querySelectorAll('[data-map-mode]').forEach(function(button){button.setAttribute('aria-pressed',String(button.dataset.mapMode===filter.mode))});
  host.mapDomains.forEach(function(entry,id){entry.button.setAttribute('aria-pressed',String(filter.domain===id))});
  var subselect=document.getElementById('map-subdomain'),availableSubs=subdomains.filter(function(item){return !filter.domain||item.domain===filter.domain}),subSignature=JSON.stringify(availableSubs);
  if(host.mapSubSignature!==subSignature){subselect.replaceChildren();var all=el('option','','Tous les sous-domaines');all.value='';subselect.append(all);availableSubs.forEach(function(subdomain){var option=el('option','',subdomain.label);option.value=subdomain.id;subselect.append(option)});host.mapSubSignature=subSignature}subselect.value=filter.subdomain;subselect.hidden=!availableSubs.length;
  var domain=domains.find(function(item){return item.id===filter.domain});document.getElementById('map-context').textContent=domain?domain.label:filter.mode==='active'?'Les repères reliés à votre portrait':'Tous les domaines';document.getElementById('map-reset').hidden=!filter.domain&&!filter.subdomain&&!query;
  var visible=patterns.filter(function(pattern){var entry=host.mapCards.get(pattern.id),subdomain=subdomains.find(function(item){return item.id===pattern.subdomain});if(filter.mode==='active'&&!entry.hypothesis)return false;if(filter.domain&&pattern.domain!==filter.domain||filter.subdomain&&pattern.subdomain!==filter.subdomain)return false;var domain=domains.find(function(item){return item.id===pattern.domain});return !query||normalized([pattern.id,String(pattern.id||'').replace(/[_-]/g,' '),pattern.label,pattern.description,domain&&domain.label,subdomain&&subdomain.label].concat(pattern.keywords||[],pattern.aliases||[],pattern.signals||[]).join(' ')).includes(query)}).sort(function(a,b){var aEntry=host.mapCards.get(a.id),bEntry=host.mapCards.get(b.id);return (aEntry.hypothesis?(rank[aEntry.hypothesis.status]??2):4)-(bEntry.hypothesis?(rank[bEntry.hypothesis.status]??2):4)||patterns.indexOf(a)-patterns.indexOf(b)});
  host.mapVisible=visible.map(function(pattern){return pattern.id});var visibleIds=new Set(host.mapVisible),list=document.getElementById('map-index-list');
  host.mapSubgroups.forEach(function(group){group.hidden=true});host.mapCards.forEach(function(entry,id){entry.row.hidden=!visibleIds.has(id)});
  visible.forEach(function(pattern){var key=filter.mode==='active'?'active':pattern.domain+':'+(pattern.subdomain||''),group=host.mapSubgroups.get(key);if(!group){group=el('section','map-subgroup');var heading=el('h4');heading.id='map-group-'+key.replace(':','-');var rows=el('div');group.append(heading,rows);group.mapHeading=heading;group.mapRows=rows;host.mapSubgroups.set(key,group);list.append(group)}group.hidden=false;var subdomain=subdomains.find(function(item){return item.id===pattern.subdomain}),domain=domains.find(function(item){return item.id===pattern.domain});group.mapHeading.textContent=filter.mode==='active'?'À partir de vos informations':subdomain?subdomain.label:domain?domain.label:'Autres repères';var entry=host.mapCards.get(pattern.id);if(entry.row.parentElement!==group.mapRows)group.mapRows.append(entry.row)});
  document.getElementById('map-results').textContent=visible.length+' repère'+(visible.length>1?'s':'')+(query?' pour « '+filter.query.trim()+' »':'');
  var empty=document.getElementById('map-active-empty');empty.hidden=visible.length>0;if(!visible.length){empty.replaceChildren(el('h4','',query?'Aucun repère trouvé':'Vos pistes apparaîtront ici'),el('p','',query?'Essayez un mot plus simple ou choisissez un autre domaine.':'Elles relieront vos éléments à des hypothèses de compréhension. Vous pouvez aussi explorer les repères.'));empty.append(actionButton(query&&filter.mode==='active'?'Chercher dans les autres repères':'Explorer les repères',function(){filter.mode='all';renderMapExplorer(host)},'secondary'))}
  if(!visibleIds.has(host.mapSelected))host.mapSelected=visible.length?visible[0].id:null;
  selectMapPattern(host,host.mapSelected,false);
}
function selectMapPattern(host,id,moveFocus){
  if(id&&moveFocus){host.mapReturnScroll=typeof window!=='undefined'?window.scrollY||0:0;host.mapMobileReading=true}if(!id)host.mapMobileReading=false;host.dataset.mapReading=String(!!host.mapMobileReading);
  host.mapSelected=id;host.mapCards.forEach(function(entry,key){var selected=key===id;entry.card.hidden=!selected;entry.row.setAttribute('aria-pressed',String(selected));entry.row.tabIndex=selected?0:-1});var detail=document.getElementById('map-detail');detail.hidden=!id;
  if(id&&moveFocus){var title=host.mapCards.get(id).title;title.tabIndex=-1;title.focus();if(typeof matchMedia==='function'&&matchMedia('(max-width: 740px)').matches)detail.scrollIntoView({behavior:typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'start'})}
}
function mapEvidenceContent(title,ids,hypothesis,cls){
 var section=el('section','map-insight '+(cls||'map-evidence'));section.append(el('h4','',title),portraitEvidence(ids));
 var quotes=(hypothesis.citations||[]).filter(function(citation){return citation.quote&&ids.includes(citation.source_id)});
 quotes.slice(0,2).forEach(function(citation){section.append(el('blockquote','map-quote',citation.quote))});
 if(quotes.length>2){var more=el('details','map-more');more.append(el('summary','','Tous les mots d’origine'));quotes.slice(2).forEach(function(citation){more.append(el('blockquote','map-quote',citation.quote))});section.append(more)}
 return section;
}
function drawMapPattern(content,pattern,hypothesis,host){
 content.replaceChildren();var subject=state.portraitData&&state.portraitData.subject,personal=function(value){return personalizePortraitText(value,subject)};
 if(hypothesis){
  if(hypothesis.observation){var observation=el('section','map-insight map-observation');observation.append(el('h4','','Ce qui se dessine'),el('p','',personal(hypothesis.observation)));content.append(observation)}
  if((hypothesis.evidence||[]).length)content.append(mapEvidenceContent('Les éléments qui l’appuient',hypothesis.evidence,hypothesis,'map-evidence'));
  var influence=hypothesis.influence,levels={unknown:'à préciser',low:'discrète',moderate:'modérée',high:'marquée'};
  if(influence&&levels[influence.level]){var weight=el('section','map-influence');weight.dataset.influenceLevel=influence.level;weight.append(el('h4','','Dans quels contextes'),el('p','helper','Influence décrite : '+levels[influence.level]));if(influence.rationale)weight.append(el('p','',personal(influence.rationale)));var contexts=Array.isArray(influence.contexts)?influence.contexts.slice(0,3):[];if(contexts.length){var list=el('ul','map-contexts');contexts.forEach(function(context){list.append(el('li','',personal(context)))});weight.append(list)}if(Array.isArray(influence.evidence)&&influence.evidence.length)weight.append(portraitEvidence(influence.evidence));content.append(weight)}
  if(hypothesis.possible_function){var possible=el('section','map-insight map-function');possible.append(el('h4','','Ce que cela pourrait expliquer'),el('p','',personal(hypothesis.possible_function)));content.append(possible)}
  if((hypothesis.counter_evidence||[]).length)content.append(mapEvidenceContent('Ce qui invite à nuancer',hypothesis.counter_evidence,hypothesis,'map-counter'));
  else content.append(el('p','helper map-counter-empty','Pas de contre-indice renseigné pour le moment. Cela ne confirme pas la piste.'));
 }else{if(pattern.description)content.append(el('p','map-definition',pattern.description));content.append(el('p','map-unexplored','Ce repère n’est pas relié à vos informations pour le moment. L’explorer ne vous attribue aucune étiquette.'))}
 var alternatives=hypothesis&&hypothesis.alternatives&&hypothesis.alternatives.length?hypothesis.alternatives:pattern.alternatives||[];
 if(alternatives.length){var other=el('details','map-more');other.append(el('summary','','Autres explications possibles'));var list=el('ul');alternatives.forEach(function(text){list.append(el('li','',personal(text)))});other.append(list);content.append(other)}
 var question=hypothesis&&hypothesis.next_question||(pattern.questions||[])[0];
 if(question){var inquiry=el('section','map-inquiry');inquiry.append(el('h4','','Une question pour avancer'),el('p','',personal(question)));content.append(inquiry)}
 var resources=hypothesis&&(hypothesis.resources||hypothesis.strengths)||pattern.resources||[];
 if(resources.length){var resource=el('section','map-insight map-resources');resource.append(el('h4','','Vos ressources'));var list=el('ul');resources.forEach(function(text){list.append(el('li','',personal(typeof text==='string'?text:text.text||text.label||'')))});resource.append(list);content.append(resource)}
 if(pattern.experiment){var experiment=el('details','map-more');experiment.append(el('summary','','Un petit essai'),el('p','',pattern.experiment));content.append(experiment)}
 if(hypothesis&&pattern.description){var definition=el('details','map-more map-definition-more');definition.append(el('summary','','Comprendre ce repère'),el('p','',pattern.description));content.append(definition)}
 if(!hypothesis&&(pattern.signals||[]).length){var signals=el('details','map-more');signals.append(el('summary','','Quand ce repère peut aider'));var list=el('ul');pattern.signals.forEach(function(text){list.append(el('li','',text))});signals.append(list);content.append(signals)}
 if(hypothesis&&(hypothesis.citations||[]).some(function(citation){return citation.quote})){var citations=el('details','map-more map-citations');citations.append(el('summary','','Tous les mots à l’origine'));hypothesis.citations.forEach(function(citation){if(!citation.quote)return;var quote=el('blockquote','map-quote',citation.quote);if(citation.source_id)quote.append(portraitEvidence([citation.source_id]));citations.append(quote)});content.append(citations)}
 var articles=hypothesis&&hypothesis.related_articles||[],characters=hypothesis&&hypothesis.related_characters||[];
 if(articles.length||characters.length){var related=el('details','map-related map-more');related.append(el('summary','','Lectures pour prolonger'));articles.forEach(function(article){var link=safeArticleLink(article,article.title||'Lire cet article');if(link.href)related.append(link)});characters.forEach(function(character){var link=safeArticleLink(character,'Les histoires de '+character.name);if(link.href)related.append(link)});content.append(related)}
 if(host&&hypothesis&&Array.isArray(hypothesis.interactions)){var mechanisms=el('details','map-interactions map-more'),kinds={amplifies:'Peut renforcer',buffers:'Peut atténuer',alternative:'Autre explication possible'};mechanisms.append(el('summary','','Mécanismes liés'));hypothesis.interactions.slice(0,3).forEach(function(interaction){var linked=(host.mapData.patterns||[]).find(function(item){return item.id===interaction.pattern_id});if(!linked||!kinds[interaction.kind])return;var relation=el('article','map-interaction');relation.append(el('p','helper',kinds[interaction.kind]),actionButton(linked.label,function(){host.mapFilter={mode:'all',domain:'',subdomain:'',query:''};document.getElementById('map-search').value='';host.mapSelected=linked.id;renderMapExplorer(host);selectMapPattern(host,linked.id,true)},'map-neighbor'));if(interaction.explanation)relation.append(el('p','',personal(interaction.explanation)));if(Array.isArray(interaction.evidence)&&interaction.evidence.length)relation.append(portraitEvidence(interaction.evidence));mechanisms.append(relation)});if(mechanisms.children.length>1)content.append(mechanisms)}
 var relatedIds=pattern.related_patterns||pattern.related||[];
 if(host&&relatedIds.length){var links=el('details','map-related-patterns map-more');links.append(el('summary','','Repères voisins'));relatedIds.forEach(function(id){var linked=(host.mapData.patterns||[]).find(function(item){return item.id===(typeof id==='string'?id:id.id||id.pattern_id)});if(!linked)return;links.append(actionButton(linked.label,function(){host.mapFilter={mode:'all',domain:'',subdomain:'',query:''};document.getElementById('map-search').value='';host.mapSelected=linked.id;renderMapExplorer(host);selectMapPattern(host,linked.id,true)},'map-neighbor'))});if(links.children.length>1)content.append(links)}
}
function drawMapReturn(pattern,current){var details=el('details','map-return');details.append(el('summary','','Nuancer ou préciser cette piste'));var form=el('form'),kindLabel=el('label','','Mon retour'),kind=el('select');kind.id='map-return-type-'+pattern.id;kind.name='type';kindLabel.htmlFor=kind.id;[['nuance','Je souhaite nuancer cette piste'],['not_concerned','Cela ne me concerne pas']].forEach(function(choice){var option=el('option','',choice[1]);option.value=choice[0];kind.append(option)});var label=el('label','','Avec mes mots'),input=el('textarea');input.id='map-return-'+pattern.id;input.name='text';input.maxLength=4000;input.required=true;label.htmlFor=input.id;var lastSaved=null,draft=state.mapReturnDrafts.get(pattern.id);kind.value=draft?draft.type:'nuance';input.value=draft?draft.text:'';details.open=!!draft;function remember(){var previous=state.mapReturnDrafts.get(pattern.id),same=previous&&previous.type===kind.value&&previous.text===input.value,savedSame=lastSaved&&lastSaved.type===kind.value&&lastSaved.text===input.value;state.mapReturnDrafts.set(pattern.id,{type:kind.value,text:input.value,creation_id:same?previous.creation_id:savedSame?lastSaved.creation_id:newClientId()})}kind.addEventListener('change',remember);input.addEventListener('input',remember);var save=el('button','secondary','Enregistrer mon retour');save.type='submit';var message=el('p','status');message.setAttribute('role','status');form.append(kindLabel,kind,label,input,save,message);form.addEventListener('submit',async function(event){event.preventDefault();if(!input.value.trim())return;remember();var submitted=state.mapReturnDrafts.get(pattern.id);var snapshot=JSON.stringify({type:submitted.type,text:submitted.text});busy(form,true);message.textContent='Enregistrement…';try{var result=await api('/api/portrait/feedback','POST',{pattern_id:pattern.id,kind:submitted.type,text:submitted.text,creation_id:submitted.creation_id});if(!current())return;lastSaved=submitted;if(JSON.stringify({type:kind.value,text:input.value})===snapshot){state.mapReturnDrafts.delete(pattern.id)}else{var remaining=state.mapReturnDrafts.get(pattern.id);if(remaining)remaining.creation_id=newClientId()}message.textContent='Retour enregistré. La cartographie en tiendra compte.';if(typeof result.source_version==='number')state.likeVersion=result.source_version;refreshPortraitData()}catch(err){if(current()){message.textContent=err.message;message.className='status error'}}finally{busy(form,false)}});details.append(form);return details}

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
function newClientId(){return crypto.randomUUID()}
function captureSourceDraft(form){var kind=form.dataset.sourceForm,previous=state.sourceDrafts.get(kind);if(!previous)return;var values=formValues(form);if(!state[kind+'FieldsReady'])SOURCE_CONTEXT_ORDER[kind].forEach(function(name){if(Object.prototype.hasOwnProperty.call(previous.values,name))values[name]=previous.values[name]});if(kind==='events'){values.people_ids=Array.from(form.querySelectorAll('[name=people_ids]:checked')).map(function(input){return input.value});}state.sourceDrafts.set(kind,Object.assign({},previous,{values:values}))}
function sourceField(form,name,label,value,options){options=options||{};var field=el('div','field'+(options.full?' full':'')),caption=el('label','',label),input=el(options.choices?'select':options.multiline?'textarea':'input');input.id=form.id+'-'+name;input.name=name;caption.htmlFor=input.id;if(options.choices){options.choices.forEach(function(choice){var option=el('option','',choice[1]);option.value=choice[0];input.append(option)})}else{if(!options.multiline)input.type=options.type||'text';if(options.max)input.maxLength=options.max;if(options.placeholder)input.placeholder=options.placeholder;}if(options.required)input.required=true;input.value=value==null?'':String(value);field.append(caption,input);if(options.help){var help=el('small','field-hint',options.help);help.id=input.id+'-help';input.setAttribute('aria-describedby',help.id);field.append(help)}form.append(field);return input}
function appendEventDateFields(grid,values,draft,prefix){
 prefix=prefix||'';var key=function(name){return prefix+name},cache=draft.dateDrafts||(draft.dateDrafts={});
 function granularity(value){return value&&value.length===10?'day':value&&value.length===7?'month':'year'}
 var modes=[['unknown','Sans date'],['year','Une année'],['month','Un mois'],['day','Un jour'],['range','Une période'],['age','Un âge approximatif']];
 var precision=sourceField(grid,key('date_precision'),prefix?'Date de rencontre':'Date',values[key('date_precision')]||'unknown',{choices:modes});
 var startKind=sourceField(grid,'','Précision du début',granularity(values[key('date_start')]),{choices:modes.slice(1,4)});startKind.id=grid.id+'-'+prefix+'start-granularity';startKind.parentElement.querySelector('label').htmlFor=startKind.id;
 var start=sourceField(grid,key('date_start'),'Date ou début de période',values[key('date_start')],{max:10});
 var endKind=sourceField(grid,'','Précision de la fin',granularity(values[key('date_end')]),{choices:modes.slice(1,4)});endKind.id=grid.id+'-'+prefix+'end-granularity';endKind.parentElement.querySelector('label').htmlFor=endKind.id;
 var end=sourceField(grid,key('date_end'),'Fin de période',values[key('date_end')],{max:10});
 var age=sourceField(grid,key('age_text'),'Âge ou période de vie',values[key('age_text')],{max:80,placeholder:'Vers 15 ans, pendant mon adolescence…'});
 function control(input,mode,visible){
  var previous=input.dataset.dateMode;if(previous)cache[input.name+':'+previous]=input.value;
  var remembered=cache[input.name+':'+mode],original=values[input.name]||'',length={year:4,month:7,day:10}[mode];
  input.type=mode==='day'?'date':mode==='month'?'month':'text';input.inputMode=mode==='year'?'numeric':'';
  if(previous!==mode)input.value=remembered!==undefined?remembered:original.length===length?original:'';
  input.dataset.dateMode=mode;input.parentElement.hidden=!visible;input.required=visible;
  if(mode==='year'){input.pattern='[0-9]{4}';input.maxLength=4;input.placeholder='2020'}else{input.removeAttribute('pattern');input.removeAttribute('maxlength');input.placeholder=''}
 }
 function show(){var mode=precision.value,range=mode==='range';startKind.parentElement.hidden=!range;endKind.parentElement.hidden=!range;
  control(start,range?startKind.value:['year','month','day'].includes(mode)?mode:start.dataset.dateMode||'year',!['unknown','age'].includes(mode));
  control(end,endKind.value,range);age.parentElement.hidden=mode!=='age';age.required=mode==='age';
 }
 precision.addEventListener('change',show);startKind.addEventListener('change',show);endKind.addEventListener('change',show);show();
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
 return sourceDate(event);
}
function sourceDate(event){
 if(event.date_precision==='age')return event.age_text||'Âge non précisé';
 if(event.date_precision==='unknown'||!event.date_start)return 'Sans date';
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
function sourceSaved(result,focusAdd){status('source-page-status','Enregistré.');if(state.tab==='events')renderPersonalSources().then(function(){if(state.tab==='events'){status('source-page-status','Enregistré.');var add=focusAdd&&document.querySelector('#events-editor summary');if(add)add.focus()}});if(Number.isFinite(Number(result.source_version)))state.likeVersion=Number(result.source_version);refreshRecommendations();refreshPortraitData()}
async function renderPersonalSources(){var holder=document.getElementById('view'),current=renderGuard(holder),page=state.tab;var kind=page==='people'?'people':'events';holder.innerHTML='<section class="panel narrow personal-section source-panel"><p class="status" id="source-page-status" role="status"></p><div id="'+kind+'-content"><p class="muted page-loading" role="status">Chargement…</p></div></section>';try{var requests=kind==='people'?[api('/api/people')]:[api('/api/people'),api('/api/life-events'+(query.get('person')?'?person_id='+encodeURIComponent(query.get('person')):''))];var results=await Promise.all(requests);if(!current()||state.tab!==page)return;state.people=results[0].people||[];state.peopleFieldsReady=results[0].fields_ready===true;if(kind==='events'){state.events=results[1].events||[];state.eventsFieldsReady=results[1].fields_ready===true;state.eventsCursor=results[1].next_cursor}drawSourceSection(kind);
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
  var dated=rows.filter(function(item){return item.date_precision!=='age'&&item.date_precision!=='unknown'&&item.date_start});
  var undated=rows.filter(function(item){return !dated.includes(item)});
  dated.sort(function(a,b){return b.date_start.localeCompare(a.date_start)});
  if(dated.length){
   var timeline=el('div','event-timeline'),year='';
   dated.forEach(function(item){
    var nextYear=item.date_start.slice(0,4);
    if(nextYear!==year){year=nextYear;timeline.append(el('h3','timeline-year',year))}
    timeline.append(drawSourceItem(kind,item,box))
   });
   box.append(timeline)
  }
  if(undated.length){
   var section=el('section','undated-events');section.append(el('h3','timeline-year','Moments sans date précise'));
   undated.forEach(function(item){section.append(drawSourceItem(kind,item,box))});
   box.append(section)
  }
 }
 if(state[kind+'Cursor'])box.append(actionButton('Afficher les événements suivants',async function(e){
  var button=e.currentTarget;button.disabled=true;
  try{
   var data=await api('/api/life-events?cursor='+encodeURIComponent(state[kind+'Cursor'])+(query.get('person')?'&person_id='+encodeURIComponent(query.get('person')):''));
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
function noteTimestamp(value){var date=new Date(value);return Number.isNaN(date.getTime())?'':date.toLocaleString('fr-FR',{dateStyle:'medium',timeStyle:'short'})}
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
 var deletionId=newClientId(),row=el('article','timeline-note');row.dataset.noteId=note.id;row.append(el('p','source-meta',label||'Note ajoutée le '+noteTimestamp(note.created_at)),el('p','source-note',note.text));
 if(note.updated_at&&note.updated_at!==note.created_at)row.append(el('p','source-meta','Modifiée le '+noteTimestamp(note.updated_at)));
 row.append(sourceNoteEditor(target,note,onSaved));var feedback=el('p','status');feedback.setAttribute('role','status');
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
// « Votre histoire avec … » : one vertical line, one dot per step, the date above in small type.
// The meeting comes first, then the linked moments with their notes, the personal notes and the
// validated observations. The single general note form closes the line.
function personTimeline(person,initial){
 var details=el('section','person-timeline'),body=el('div','person-timeline-body'),items=[],cursor=null,loading=false,target=noteTarget('person',person.id);
 details.append(el('h2','','Votre histoire avec '+person.display_name),body);
 async function load(more){if(loading)return;loading=true;try{var result=await api('/api/people/'+encodeURIComponent(person.id)+'/timeline'+(more&&cursor?'?cursor='+encodeURIComponent(cursor):''));if(!details.isConnected)return;apply(result,more)}catch(err){if(details.isConnected)body.append(el('p','status error',err.message),actionButton('Recharger la timeline',function(){return load(false)},'text'))}finally{loading=false}}
 function apply(result,more){items=more?items.concat(result.items||[]):result.items||[];cursor=result.next_cursor;draw(result.available===true)}
 function reload(){return load(false)}
 function step(kind,when){var line=el('li','frise-step frise-'+kind);if(when)line.append(el('p','frise-when',when));return line}
 function noteLabel(note){var day=noteDay(note.created_at);return 'Votre note'+(day?' · '+day:'')}
 function draw(available){body.replaceChildren();var list=el('ol','person-frise');
  items.forEach(function(item){
   if(item.kind==='relationship_start'){var date=item.date||{},known=date.precision&&date.precision!=='unknown'&&(date.start||date.age_text),line=step('meeting',known?sourceDate({date_precision:date.precision,date_start:date.start,date_end:date.end,age_text:date.age_text}):'Date non précisée');line.append(el('h3','frise-title','Rencontre'));if(item.legacy_period_start||item.legacy_period_end)line.append(el('p','source-meta',[item.legacy_period_start,item.legacy_period_end].filter(Boolean).join(' – ')));list.append(line)}
   else if(item.kind==='life_event'){var event=item.event,line=step('event',sourceDate(event));line.dataset.eventId=event.id;var link=el('a','',event.title);link.href='/ma-ligne-de-vie/?person='+encodeURIComponent(person.id)+'&event='+encodeURIComponent(event.id);var heading=el('h3','frise-title');heading.append(link);line.append(heading);if(event.description)line.append(el('p','source-note',event.description));line.append(eventNotes(event,{addLabel:'Ajouter une note à ce moment',noteLabel:noteLabel}));list.append(line)}
   else if(item.kind==='user_note'){var line=step('note');line.append(sourceNoteCard(target,item.note,reload,noteLabel(item.note)));list.append(line)}
   else if(item.kind==='ambre_observation'){var note=item.note,day=noteDay(note.created_at),line=step('observation','Repère validé'+(day?' · '+day:'')),feedback=el('p','status');line.append(el('p','source-note',note.text),actionButton('Retirer ce repère',async function(event){var button=event.currentTarget;button.disabled=true;try{await api('/api/people/'+encodeURIComponent(person.id)+'/observations/'+encodeURIComponent(note.id),'DELETE');await load(false)}catch(err){feedback.textContent=err.message;button.disabled=false}},'text'),feedback);list.append(line)}
  });
  if(items.length)body.append(list);else body.append(el('p','muted','Aucun moment enregistré pour cette relation.'));
  if(cursor)body.append(actionButton('Afficher la suite',function(){return load(true)},'secondary frise-more'));
  if(available){var add=sourceNoteEditor(target,null,reload,'Ajouter une note');add.className+=' person-note-add';body.append(add)}
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
 if(person.relationship_text)info.append(el('p','source-note',person.relationship_text));
 if(person.period_start||person.period_end)info.append(el('p','source-meta',[person.period_start,person.period_end].filter(Boolean).join(' – ')));
 appendSourceContextText(info,'people',person);
 if(info.children.length===1)info.append(el('p','muted','Aucune information ajoutée pour le moment.'));
 var edit=actionButton('Modifier',async function(){
  var open=document.querySelector('[data-source-form="people"]');if(open)captureSourceDraft(open);
  var old=state.sourceDrafts.get('people');
  if(old&&old.id!==person.id&&!await confirmDialog('Remplacer le brouillon ouvert ?','Le brouillon en cours sera remplacé par ce proche enregistré.','Remplacer'))return;
  if(!old||old.id!==person.id)state.sourceDrafts.set('people',{id:person.id,revision:person.revision,values:Object.assign({},person)});
  drawSourceEditor('people',true)
 },'secondary');edit.dataset.personEdit='true';
 actions.append(edit,actionButton('Supprimer',async function(e){
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

function drawSourceItem(kind,item,box){
 var isPeople=kind==='people',article=el('article','source-item '+(isPeople?'person-card':'event-card'));
 article.id=(isPeople?'person-':'event-')+item.id;article.tabIndex=-1;
 var content=el('div','source-body'),title=el(isPeople?'h3':'h4','',isPeople?undefined:item.title),meta=el('div','source-meta');
 if(isPeople){
  // The directory stays short: the details and the timeline live on the person page.
  var open=el('a','person-open',item.display_name);open.href=pagePath('person',item.id);title.append(open);var head=el('div','person-head');head.append(title,el('span','source-tag',personRoleLabel(item)));content.append(head);
  if(item.relationship_text)content.append(el('p','source-note person-line',item.relationship_text));
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
 content.append(title,meta,el('p','source-note',item.description));
 appendSourceContextText(content,kind,item);
 var actions=el('div','actions source-actions');
 actions.append(actionButton('Modifier',async function(){
  var open=document.querySelector('[data-source-form="'+kind+'"]');if(open)captureSourceDraft(open);
  var old=state.sourceDrafts.get(kind);
  if(old&&old.id!==item.id&&!await confirmDialog('Remplacer le brouillon ouvert ?','Le brouillon en cours sera remplacé par cet élément enregistré.','Remplacer'))return;
  state.sourceDrafts.set(kind,{id:item.id,revision:item.revision,values:Object.assign({},item)});drawSourceEditor(kind,true)
 },'text'),actionButton('Supprimer',async function(e){
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
if(kind==='people'){sourceField(grid,'display_name','Prénom ou pseudonyme',values.display_name,{required:true,max:80});var role=sourceField(grid,'role','Lien',values.role,{choices:PERSON_ROLES});var custom=sourceField(grid,'custom_role','Précisez le lien',values.custom_role,{max:80});function showRole(){custom.parentElement.hidden=role.value!=='other'}role.addEventListener('change',showRole);showRole();sourceField(grid,'relationship_text','Comment vivez-vous cette relation ?',values.relationship_text,{multiline:true,max:6000,full:true,placeholder:'Ce que vous ressentez dans cette relation…'});var more=el('details','form-more disclosure'),moreGrid=el('div','form-grid');more.append(el('summary','','Ajouter des détails'),moreGrid);moreGrid.id=form.id+'-details';grid.append(more);appendEventDateFields(moreGrid,values,draft,'meeting_');if(values.period_start||values.period_end){sourceField(moreGrid,'period_start','Ancien repère de début',values.period_start,{max:80,placeholder:'Depuis l’enfance, 2020…'});sourceField(moreGrid,'period_end','Ancien repère de fin',values.period_end,{max:80,placeholder:'Aujourd’hui, vers 2022…'})}if(state.peopleFieldsReady)appendSourceContextFields(moreGrid,kind,values);more.open=draft.detailsOpen===true||(draft.detailsOpen!==false&&['meeting_date_start','meeting_age_text','period_start','period_end'].concat(SOURCE_CONTEXT_ORDER.people).some(function(name){return values[name]}));more.addEventListener('toggle',function(){var open=state.sourceDrafts.get(kind);if(open)open.detailsOpen=more.open})}else{sourceField(grid,'title','Titre',values.title,{required:true,max:160,full:true});sourceField(grid,'kind','Type',values.kind,{choices:EVENT_KINDS});appendEventDateFields(grid,values,draft);sourceField(grid,'description','Ce que vous souhaitez raconter',values.description,{multiline:true,max:6000,required:true,full:true});var peopleField=el('fieldset','field full');peopleField.append(el('legend','','Avec'));var options=el('div','people-options');state.people.forEach(function(person){var label=el('label','check'),checkbox=el('input');checkbox.type='checkbox';checkbox.name='people_ids';checkbox.value=person.id;checkbox.checked=(values.people_ids||[]).includes(person.id);label.append(checkbox,document.createTextNode(person.display_name));options.append(label)});if(!state.people.length){var none=el('p','helper','Aucun proche enregistré. '),peopleLink=el('a','','Ajouter un proche');peopleLink.href='/mes-proches/';none.append(peopleLink);options.append(none)};peopleField.append(options);grid.append(peopleField)}
if(kind!=='people'&&state[kind+'FieldsReady'])appendSourceContextFields(grid,kind,values);
var actions=el('div','actions'),save=el('button','','Enregistrer');save.type='submit';actions.append(save,actionButton('Annuler',function(){captureSourceDraft(form);details.open=false;summary.focus()},'secondary'));var message=el('p','status source-feedback');message.setAttribute('role','status');form.append(actions,message);details.append(form);form.addEventListener('input',function(){captureSourceDraft(form)});form.addEventListener('change',function(){captureSourceDraft(form)});form.addEventListener('submit',async function(event){event.preventDefault();captureSourceDraft(form);var submitted=state.sourceDrafts.get(kind),payload=Object.assign({},submitted.values);if(!state[kind+'FieldsReady'])SOURCE_CONTEXT_ORDER[kind].forEach(function(name){delete payload[name]});if(kind==='events'){['meaning_now','current_importance','exposure','felt_then','felt_now','impact','support_level','support_text'].forEach(function(name){delete payload[name]});if(payload.people_ids.length>10){message.textContent='Associez au maximum 10 proches à cet événement.';message.className='status error';return}if(payload.date_precision==='unknown'||payload.date_precision==='age'){payload.date_start='';payload.date_end=''}if(payload.date_precision!=='range')payload.date_end='';if(payload.date_precision!=='age')payload.age_text=''}if(kind==='people'&&payload.role!=='other')payload.custom_role='';if(submitted.id)payload.revision=submitted.revision;else payload.creation_id=submitted.creation_id;var snapshot=JSON.stringify(submitted.values);busy(form,true);message.textContent='Enregistrement…';message.className='status';try{var result=await api((kind==='people'?'/api/people':'/api/life-events')+(submitted.id?'/'+encodeURIComponent(submitted.id):''),submitted.id?'PATCH':'POST',payload);if(!current())return;var saved=result[kind==='people'?'person':'event'];state[kind]=state[kind].filter(function(item){return item.id!==saved.id});state[kind].unshift(saved);captureSourceDraft(form);var latest=state.sourceDrafts.get(kind);if(latest&&JSON.stringify(latest.values)!==snapshot){latest.id=saved.id;latest.revision=saved.revision;delete latest.creation_id;if(kind==='people'&&state.tab==='person')state.person=saved;drawSourceEditor(kind);var note=document.querySelector('#'+kind+'-editor .source-feedback');note.textContent='Enregistré. Vos changements saisis pendant l’enregistrement restent à sauvegarder.'}else{state.sourceDrafts.delete(kind);if(kind==='people'&&state.tab==='person')personSaved(saved);else{drawSourceSection(kind);document.querySelector('#'+kind+'-editor summary').focus()}}if(kind==='people'){var eventForm=document.querySelector('[data-source-form="events"]');if(eventForm)captureSourceDraft(eventForm);drawSourceSection('events')}sourceSaved(result)}catch(err){if(!current())return;message.textContent=err.code==='revision_conflict'?'Cet élément a été modifié dans un autre onglet. Votre brouillon est conservé. Rechargez la version enregistrée pour comparer avant de réessayer.':err.message;message.className='status error';if(err.code==='revision_conflict'&&!form.querySelector('[data-reload-source]')){var reload=actionButton('Afficher la version enregistrée',async function(){try{var response=await api((kind==='people'?'/api/people/':'/api/life-events/')+encodeURIComponent(submitted.id));var latest=response[kind==='people'?'person':'event'];if(!current())return;if(!latest){message.textContent='Cet élément a été supprimé ou ne figure pas sur cette page. Votre brouillon reste disponible.';return}var comparison=el('div','notice source-comparison');comparison.append(el('h3','','Version enregistrée'));form.querySelectorAll('input,select,textarea').forEach(function(input){if(!input.name||input.name==='people_ids')return;var value=latest[input.name];if(value==null||value==='')return;if(input.tagName==='SELECT'){var option=Array.from(input.querySelectorAll('option')).find(function(option){return option.value===String(value)});if(option)value=option.textContent}var caption=input.parentElement.querySelector('label');comparison.append(el('p','',(caption?caption.textContent:input.name)+' : '+value))});if(kind==='events'&&latest.people_ids&&latest.people_ids.length)comparison.append(el('p','','Proches associés : '+latest.people_ids.map(function(id){var person=state.people.find(function(person){return person.id===id});return person?person.display_name:'Proche'}).join(', ')));comparison.append(actionButton('Conserver mon brouillon pour un nouvel enregistrement',function(){captureSourceDraft(form);var kept=state.sourceDrafts.get(kind);kept.revision=latest.revision;comparison.remove();reload.remove();message.textContent='Votre brouillon est conservé. Vérifiez vos champs puis enregistrez pour remplacer les informations comparées.';message.className='status'},'secondary'),actionButton('Reprendre cette version',function(){if(!confirm('Remplacer votre brouillon par la version enregistrée ? Votre saisie non enregistrée sera perdue.'))return;state.sourceDrafts.set(kind,{id:latest.id,revision:latest.revision,values:latest});drawSourceEditor(kind,true)},'secondary'));form.append(comparison)}catch(error){message.textContent=error.message}},'secondary');reload.dataset.reloadSource='true';form.append(reload)}}finally{busy(form,false)}});if(focus)focusEntry(form.querySelector('input,select,textarea'),summary)}

function setupPortraitViews(view){
 if(['#ma-cartographie','#cartographie','#mon-portrait','#synthese','#correction-form','#mes-articles','#suggestions'].includes(location.hash))history.replaceState({},'',location.pathname+location.search+'#self-profile');
}

${CHAT_INTERFACE_SCRIPT}
${CHANGE_INTERFACE_SCRIPT}
${CARE_INTERFACE_SCRIPT}
${SETTINGS_INTERFACE_SCRIPT}
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
 var capabilityNote=el('p','helper chat-capability'),unavailable=chatCapabilityMessage(proposal.kind);capabilityNote.textContent=unavailable;capabilityNote.hidden=!unavailable;box.append(capabilityNote);
 var feedback=el('p','status');feedback.setAttribute('role','status');
 var replacementScope=replaced.map(function(key){return '« '+(labels[key]||key)+' »'}).join(', '),acceptLabel=replaced.length===1&&!appended.length?'Remplacer '+replacementScope:replaced.length?'Confirmer les remplacements'+(appended.length?' et ajouts':''):'Accepter cette proposition';
 var actions=el('div','actions'),accept=el('button','',acceptLabel);accept.type='submit';accept.disabled=!!unavailable;var reject=actionButton('Refuser',function(){return decide('reject')},'secondary');actions.append(accept,reject);box.append(actions,feedback);
 box.addEventListener('submit',function(event){event.preventDefault();return decide('accept')});
 async function decide(action){
  if(!current())return;
  if(action==='accept'&&chatCapabilityMessage(proposal.kind)){capabilityNote.textContent=chatCapabilityMessage(proposal.kind);capabilityNote.hidden=false;return}
  if(action==='accept'&&replaced.length&&!await confirmDialog('Remplacer '+replacementScope+' ?','Le texte proposé remplacera le texte actuel. Les autres informations seront conservées.','Remplacer'))return;
  accept.disabled=true;reject.disabled=true;box.dataset.submitting='true';feedback.textContent='Enregistrement…';
  try{var result=await api('/api/chat/proposals/'+encodeURIComponent(proposal.id)+'/'+action,'POST',action==='accept'?{payload:payload,revision:proposal.revision}:{revision:proposal.revision});if(!current()||state.tab!=='chat')return;state.chatProposalDrafts.delete(proposal.id);var latest=result.messages?result:await api('/api/chat');if(!current()||state.tab!=='chat')return;state.chatData=latest;reconcileChatTurn(latest);drawChat(document.getElementById('view'),current)}
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
function renderAccount(){var m=state.me,freePreview=!!(m.access&&m.access.mode==='free_preview'),active=m.subscription&&m.subscription.active,manageable=!!(m.subscription&&m.subscription.manageable),subStatus=m.subscription&&m.subscription.status,billing=m.configuration&&m.configuration.billing,canSubscribe=!active&&(!subStatus||['none','canceled','incomplete_expired'].includes(subStatus));var canStart=!!(m.configuration&&m.configuration.ai);var view=document.getElementById('view');view.innerHTML='<section class="account-panel"><div class="account-row"><h2>Identité</h2><div id="self-profile"></div></div><div class="account-row"><h2>Connexion</h2><p id="account-email" class="muted"></p></div><div class="account-row"><h2>Accès à Ambre</h2><div class="row"><p id="subscription-copy"></p><div class="price" id="account-price">29 € <small>/ mois</small></div></div><div id="account-progress"></div><div id="billing-action"></div><p id="billing-status" class="status" role="status"></p></div><div class="account-row"><h2>Mes données</h2><div class="actions"><button type="button" class="secondary" id="export-button">Exporter</button><button type="button" class="secondary" id="chat-clear-button">Effacer la conversation</button></div><p id="export-status" class="status" role="status"></p><details class="privacy-details"><summary>Comment mes choix sont-ils utilisés ?</summary><p class="legal-space">Vos réponses, vos proches et les événements que vous enregistrez permettent à Ambre de proposer des analyses. Celles-ci restent des interprétations, pas un diagnostic ni une mesure définitive de votre personnalité.</p><p>Les informations nécessaires à l’analyse approfondie, vos proches, vos événements, vos retours et vos corrections sont transmis à OpenRouter pour personnaliser ce service.</p><p>Vous pouvez modifier ou supprimer ce que vous avez enregistré, ou corriger une interprétation. Les analyses suivantes tiennent compte de ces changements.</p></details></div><div class="account-row"><button type="button" class="text danger account-quiet" id="delete-button">Supprimer mon compte</button></div></section>';setupPortraitViews(view);renderSelfProfile();document.getElementById('account-email').textContent=m.user.email;document.getElementById('account-price').hidden=freePreview;document.getElementById('subscription-copy').textContent=freePreview?'Accès gratuit.':active?'Abonnement actif.':['past_due','unpaid','incomplete'].includes(subStatus)?'Votre paiement nécessite une vérification. Retrouvez vos factures et votre moyen de paiement dans l’espace de facturation.':'Un abonnement pour échanger avec Ambre et faire le point.';var offer=el('p','helper',freePreview||active?'':m.subscription&&m.subscription.trial_eligible===false?'Un nouvel abonnement est facturé immédiatement : 29 € par mois.':'7 jours d’essai, puis 29 € par mois. Annulez avant la fin de l’essai pour éviter le premier paiement.');var progress=document.getElementById('account-progress');appendAccountQuota(progress,m);if(offer.textContent)progress.append(offer);var area=document.getElementById('billing-action');function billingButton(label,path,cls){return actionButton(label,async function(e){var b=e.currentTarget;b.disabled=true;status('billing-status','Ouverture de votre espace de paiement…');try{var data=await api(path,'POST',{});var u=new URL(data.url);if(u.protocol!=='https:')throw new Error('Le lien de paiement est indisponible.');location.assign(u.href)}catch(err){status('billing-status',err.message,true);b.disabled=false}},cls)}if(!freePreview&&!billing)area.append(el('div','notice warn','Les nouveaux abonnements ne sont pas encore ouverts.'));var actions=el('div','actions');if(!freePreview&&canSubscribe&&billing){var subscribe=billingButton(m.subscription&&m.subscription.trial_eligible===false?'M’abonner pour 29 € par mois':'Commencer mes 7 jours d’essai','/api/billing/checkout');subscribe.disabled=!canStart;actions.append(subscribe);if(!canStart)area.append(el('p','helper','La personnalisation attend l’activation de l’IA avant l’ouverture de votre abonnement.'))}if(!freePreview&&manageable)actions.append(billingButton(canSubscribe?'Mes factures et moyens de paiement':'Gérer mon abonnement','/api/billing/portal',canSubscribe?'secondary':''));if(!freePreview&&active&&!(m.subscription&&m.subscription.cancel_at_period_end)){actions.append(actionButton(subStatus==='trialing'?'Annuler avant la fin de l’essai':'Résilier à la fin de la période',async function(e){var button=e.currentTarget;button.disabled=true;status('billing-status','Enregistrement de votre résiliation…');try{await api('/api/billing/cancel','POST',{});state.me=await api('/api/me');renderAccount()}catch(err){status('billing-status',err.message,true);button.disabled=false}},'secondary'))}if(!freePreview&&m.subscription&&m.subscription.cancel_at_period_end)area.append(el('p','notice','Résiliation enregistrée. Votre accès reste ouvert jusqu’au '+dateLabel(m.subscription.access_until||m.subscription.trial_end)+'.'));else if(!freePreview&&subStatus==='trialing'&&m.subscription.trial_end)area.append(el('p','notice','Votre essai se termine le '+dateLabel(m.subscription.trial_end)+'. Vous pouvez annuler avant cette date.'));area.append(actions);if(m.user.role==='psychologist'){document.getElementById('account-progress').closest('.account-row').hidden=true;var privacy=view.querySelector('.privacy-details');privacy.replaceChildren(el('summary','','Comment les dossiers sont-ils partagés ?'),el('p','','Vous consultez uniquement les rubriques autorisées après activation du code transmis par la personne. Les consultations sont journalisées, et la personne peut révoquer votre accès à tout moment. Vos informations professionnelles sont déclarées, non vérifiées.'))}var sessionActions=el('div','actions');sessionActions.append(actionButton('Se déconnecter',logout,'text'),actionButton('Se déconnecter partout',function(){logout('all')},'text'));document.getElementById('account-email').parentElement.append(sessionActions);var chatClear=document.getElementById('chat-clear-button');if(chatClear){if(typeof openChangeChatClear==='function')chatClear.addEventListener('click',function(){openChangeChatClear(function(){return state.tab==='account'})});else chatClear.remove()}document.getElementById('export-button').addEventListener('click',async function(e){var b=e.currentTarget;b.disabled=true;status('export-status','Préparation…');try{var data=await api('/api/export');var url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));var a=el('a');a.href=url;a.download='mon-journal-decode-ton-esprit.json';document.body.append(a);a.click();a.remove();setTimeout(function(){URL.revokeObjectURL(url)},1000);status('export-status','Votre téléchargement est lancé.')}catch(err){status('export-status',err.message,true)}finally{b.disabled=false}});document.getElementById('delete-button').addEventListener('click',function(){document.getElementById('delete-form').reset();status('delete-status','');var deleteDialog=document.getElementById('delete-dialog');deleteDialog.showModal();focusEntry(document.getElementById('delete-password'),deleteDialog.querySelector('h2'))})}
async function logout(scope){if(scope==='all'&&!await confirmDialog('Se déconnecter partout ?','Tous vos appareils seront déconnectés, y compris celui-ci. Vous devrez vous reconnecter.','Se déconnecter'))return;try{disposeChatInterface();await api(scope==='all'?'/api/auth/logout-all':'/api/auth/logout','POST',{});state.me={user:null};preparedResponses={};state.navigationRequest=(state.navigationRequest||0)+1;resetChatSession();state.chatDraft='';state.chatSendKey=null;state.chatProposalDrafts.clear();clearTimeout(state.chatTimer);clearTimeout(state.chatCooldownTimer);state.chatData=null;state.likes=[];state.drafts.clear();state.differencesDrafts.clear();state.likeVersion=null;state.recommendationVersion=null;state.refreshLikesView=null;state.sourceDrafts.clear();state.people=[];state.events=[];state.selfDraft=null;state.selfRevision=null;state.selfBase=null;state.mapReturnDrafts.clear();clearTimeout(state.portraitTimer);if(state.noteDrafts)state.noteDrafts.clear();landing()}catch(err){var failure=el('div','notice warn',err.message);failure.setAttribute('role','alert');main.prepend(failure)}}
document.addEventListener('click',function(e){var auth=e.target.closest('[data-auth]');if(auth)openAuth(auth.dataset.auth);var close=e.target.closest('[data-close]');if(close)document.getElementById(close.dataset.close).close()});
document.getElementById('auth-role').addEventListener('change',updateAuthRole);
document.getElementById('auth-role-toggle').addEventListener('click',function(){var role=document.getElementById('auth-role');role.value=role.value==='psychologist'?'member':'psychologist';updateAuthRole()});
document.getElementById('auth-form').addEventListener('submit',async function(e){e.preventDefault();var form=e.currentTarget;busy(form,true);status('auth-status','Connexion…');try{var data=formValues(form);if(state.authMode==='register'&&data.role==='psychologist')data.professional={name:data.professional_name,title:data.professional_title,registration_number:data.registration_number};await api('/api/auth/'+state.authMode,'POST',data);await refresh();document.getElementById('auth-dialog').close();form.reset()}catch(err){status('auth-status',err.message,true)}finally{busy(form,false)}});
document.getElementById('delete-form').addEventListener('submit',async function(e){e.preventDefault();var form=e.currentTarget;busy(form,true);status('delete-status','Suppression…');try{await api('/api/account','DELETE',formValues(form));document.getElementById('delete-dialog').close();state.me={user:null};preparedResponses={};state.navigationRequest=(state.navigationRequest||0)+1;resetChatSession();state.chatDraft='';state.chatSendKey=null;state.chatProposalDrafts.clear();clearTimeout(state.chatTimer);clearTimeout(state.chatCooldownTimer);state.chatData=null;state.likes=[];state.drafts.clear();state.differencesDrafts.clear();state.likeVersion=null;state.recommendationVersion=null;state.refreshLikesView=null;state.sourceDrafts.clear();state.people=[];state.events=[];state.selfDraft=null;state.selfRevision=null;state.selfBase=null;state.mapReturnDrafts.clear();clearTimeout(state.portraitTimer);if(state.noteDrafts)state.noteDrafts.clear();landing()}catch(err){status('delete-status',err.message,true)}finally{busy(form,false)}});
renderPageLoading();
refresh().catch(function(err){main.replaceChildren(el('div','notice warn','Votre espace n’a pas pu être ouvert. '+err.message),actionButton('Réessayer',function(){location.reload()},'secondary'))});
</script></body></html>
`;

// Parse the application before the body can paint; hydrated pages need no intermediate scaffold.
export const HTML = (() => {
  let script='';
  const document=DOCUMENT_HTML.replace(/<script>([\s\S]*?)<\/script>/,(_match,source)=>{script=source;return ''});
  return document.replace('</head>',()=>'<script>document.addEventListener("DOMContentLoaded",function(){\n'+script+'\n});</script></head>');
})();
