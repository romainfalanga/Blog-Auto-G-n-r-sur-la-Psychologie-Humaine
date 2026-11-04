// Embedded in the app shell: no framework, network client or persisted history.
export const CHAT_INTERFACE_SCRIPT=String.raw`
function chatConsent(data){return data.consent===true||!!(data.consent&&data.consent.accepted)}
function chatFailureMessage(data){var error=data&&data.last_error;return (error&&error.message||'La réponse n’a pas pu être préparée.')+' Votre texte reste disponible.'}
function chatIsCurrent(current){return current()&&state.tab==='chat'}
function chatHasPending(){return !!(state.chatSending||state.chatOutbox&&['sending','pending','uncertain'].includes(state.chatOutbox.status))}
function chatCapabilityMessage(kind){
 var capability=state.chatData&&state.chatData.capabilities&&state.chatData.capabilities[kind];if(!capability||capability.allowed!==false)return '';
 var reasons={likes_required:'Aimez encore quelques lectures pour préparer votre personnage.',self_description_required:'Complétez votre présentation pour préparer votre personnage.',portrait_required:'Votre portrait doit être analysé et à jour.',character_required:'Votre personnage doit être prêt.',character_not_ready:'Votre personnage doit être prêt.',quota_exhausted:'Les sept histoires de cette semaine sont utilisées.',request_in_progress:'Une histoire est déjà en préparation.',active_request:'Une histoire est déjà en préparation.',provider_missing:'La préparation est momentanément indisponible.',provider_cooldown:'Le fournisseur est temporairement indisponible.',generation_budget:'Le budget de génération est momentanément atteint.',request_limit:'La limite de demandes du jour est atteinte.',subscription_required:'Un accès actif est nécessaire.'};
 return (capability.reasons||[]).map(function(reason){return reasons[reason]}).filter(Boolean).join(' ')||'Cette opération n’est pas disponible pour le moment.';
}
function updateChatControls(){
 var data=state.chatData||{},form=document.getElementById('chat-form');if(!form)return;
 var send=form.querySelector('button[type=submit]');if(send)send.disabled=chatHasPending()||!chatConsent(data)||!data.access||!data.access.allowed||!!(data.quota&&Number(data.quota.remaining||0)<=0);
 var mounted=document.getElementById('view'),shell=mounted&&mounted.chatShell;if(shell)shell.thread.querySelectorAll('button').forEach(function(button){if(button.dataset.chatRetry)button.disabled=!!(state.chatSending||!data.access||!data.access.allowed||data.quota&&Number(data.quota.remaining||0)<=0)});
 var input=form.elements.content;if(input)input.disabled=!!(data.access&&data.access.reason==='authentication_required');
 var quota=document.getElementById('chat-quota');if(quota&&data.quota)quota.textContent=Number(data.quota.remaining||0)+' / '+Number(data.quota.limit||100)+' messages';
}
function disposeChatInterface(){
 if(state.chatVoice){state.chatVoice.dispose();state.chatVoice=null}
 if(state.chatReveals){state.chatReveals.forEach(function(cancel){cancel()});state.chatReveals.clear()}
}
function resetChatSession(){disposeChatInterface();state.chatOutbox=null;state.chatSeenResponses=new Set()}
function applyChatAvailability(data,current){
 if(!chatIsCurrent(current))return;
 var wasCooling=state.chatData&&state.chatData.access&&state.chatData.access.reason==='provider_cooldown';state.chatData=Object.assign({},state.chatData||{},data);var latest=state.chatData;
 updateChatControls();var gate=document.getElementById('chat-access-gate');if(gate&&latest.access&&latest.access.allowed)gate.remove();
 var waiting=document.getElementById('chat-pending-indicator');if(waiting&&!state.chatSending)waiting.remove();
 var feedback=document.getElementById('chat-status');if(feedback&&latest.last_error&&!state.chatSending&&!(state.chatOutbox&&state.chatOutbox.error&&['failed','uncertain'].includes(state.chatOutbox.status))){feedback.textContent=chatFailureMessage(latest);feedback.className='status error';if(latest.last_error.action==='refresh')feedback.append(actionButton('Recharger la conversation',function(){if(chatIsCurrent(current))renderChat()},'text'));if(wasCooling&&latest.access&&latest.access.allowed)feedback.append(el('span','',' Vous pouvez réessayer maintenant.'))}
 scheduleChatCooldown(current,latest);
}
function scheduleChatCooldown(current,data){
 clearTimeout(state.chatCooldownTimer);var access=data&&data.access||{},retryAt=Number(access.retry_at||0);if(!chatIsCurrent(current)||access.reason!=='provider_cooldown'||!retryAt)return;
 var failures=0;async function check(){if(!chatIsCurrent(current)||state.chatSending)return;var remaining=retryAt*1000-Date.now();if(remaining>0){state.chatCooldownTimer=setTimeout(check,Math.min(60000,Math.max(1000,remaining+100)));return}try{var latest=await api('/api/chat');if(!chatIsCurrent(current)||state.chatSending)return;applyChatAvailability(latest,current)}catch(err){if(!chatIsCurrent(current))return;if(++failures<=3)state.chatCooldownTimer=setTimeout(check,5000);else status('chat-status','La disponibilité n’a pas pu être vérifiée. Votre texte est conservé. Rechargez la conversation pour réessayer.',true)}}
 state.chatCooldownTimer=setTimeout(check,Math.min(60000,Math.max(1000,retryAt*1000-Date.now()+100)));
}
function reconcileChatTurn(data){
 var pending=data.pending_request||null,box=state.chatOutbox,user=box&&(data.messages||[]).find(function(message){return message.role==='user'&&(message.client_key===box.client_key||box.request_id&&String(message.id)===box.request_id+'-user')});
 state.chatRequest=pending;state.chatSending=!!pending;
 if(!box)return;
 if(user){box.request_id=String(user.id).replace(/-user$/,'');if(['completed','failed','cancelled','expired'].includes(user.status))box.status=user.status==='completed'?'completed':'failed';else box.status='pending'}
 if(pending&&pending.client_key===box.client_key){box.request_id=pending.id;box.status='pending'}
 var last=data.last_request;if(last&&box.request_id===last.id){if(['failed','cancelled','expired'].includes(last.status))box.status='failed';else if(last.status==='completed')box.status='completed'}
 if(box.status==='failed'){
  box.error=data.last_error&&box.request_id&&data.last_request&&data.last_request.id===box.request_id?chatFailureMessage(data):box.error||chatFailureMessage(data);state.chatSendKey=null;
  if(!box.restored){box.restored=true;if(!state.chatDraft){state.chatDraft=box.content;var input=document.getElementById('chat-content');if(input&&!input.value)input.value=box.content}}
 }else if(box.status==='completed'){state.chatSendKey=null;if(user)state.chatOutbox=null;}
 else if(!pending&&['sending','pending'].includes(box.status)){
  if(Date.now()-box.started_at<30000)state.chatSending=true;
  else{box.status='uncertain';box.error='L’envoi n’a pas été confirmé. Vérifiez ou réessayez ce message.'}
 }
}
async function renderChat(){
 disposeChatInterface();var view=document.getElementById('view'),current=renderGuard(view);view.chatShell=null;view.replaceChildren(el('p','muted','Ouverture de la conversation avec Ambre…'));
 try{var data=await api('/api/chat');if(!chatIsCurrent(current))return;state.chatData=data;reconcileChatTurn(data);drawChat(view,current);if(state.chatSending)pollChat(state.chatRequest||{client_key:state.chatOutbox&&state.chatOutbox.client_key},current,null)}
 catch(err){if(current())view.replaceChildren(el('p','status error',err.message),actionButton('Réessayer',renderChat,'secondary'))}
}
function drawChatReadings(host,suggestions){
 var valid=[];(suggestions||[]).slice(0,3).forEach(function(article){var link=safeArticleLink(article,article.title||'Découvrir cette lecture');if(!link.href)return;var card=el('article','chat-reading');if(article.character)card.append(el('span','chat-reading-character',article.character));var title=el('h4');title.append(link);card.append(title);if(article.reason)card.append(el('p','',article.reason));valid.push(card)});
 if(!valid.length)return;var readings=el('aside','chat-readings');readings.setAttribute('aria-label','Lectures suggérées');readings.append(el('p','chat-readings-title','Pour prolonger cet échange'));valid.forEach(function(card){readings.append(card)});host.append(readings);
}
function createChatShell(view,current){
 var data=state.chatData||{},guest=data.access&&data.access.reason==='authentication_required',surface=el('section','conversation');surface.setAttribute('aria-label','Ambre');
 var toolbar=el('div','conversation-toolbar');toolbar.append(el('span','','Ambre · psychologue IA'));
 var tools=el('details','conversation-tools');tools.append(el('summary','','Informations et historique'));var privacy=el('div','conversation-info');
 privacy.append(el('p','','Ambre, psychologue IA, vous aide à réfléchir à votre vécu. Elle ne remplace pas un professionnel de santé.'),el('p','helper','Vos messages et le contexte utile sont transmis à OpenRouter. Les modifications de votre portrait, de vos proches et de votre ligne de vie nécessitent votre validation.'),el('p','helper','La dictée utilise votre navigateur et peut transmettre la voix à son service vocal. Le texte reste modifiable avant l’envoi. Aucun audio n’est conservé ici.'));
 if(!guest){privacy.append(el('p','helper','Effacer la conversation désactive les échanges avec Ambre. Les informations déjà acceptées restent enregistrées. Le quota reste inchangé.'),actionButton('Effacer la conversation et désactiver Ambre',async function(event){
  if(!confirm('Effacer cet historique et désactiver Ambre ? Les informations déjà acceptées dans votre profil restent conservées.'))return;
  disposeChatInterface();event.currentTarget.disabled=true;
  try{await api('/api/chat','DELETE');if(!chatIsCurrent(current))return;resetChatSession();state.chatDraft='';state.chatSendKey=null;state.chatProposalDrafts.clear();renderChat()}
  catch(err){if(current()){privacy.append(el('p','status error',err.message));event.currentTarget.disabled=false}}
 },'text'))}
 tools.append(privacy);toolbar.append(tools);surface.append(toolbar);
 var scroll=el('div','conversation-scroll');scroll.id='conversation-scroll';var thread=el('section','chat-thread');thread.setAttribute('aria-label','Conversation');thread.setAttribute('role','log');thread.setAttribute('aria-live','off');scroll.append(thread);
 var proposals=el('section');proposals.id='chat-proposals';scroll.append(proposals);surface.append(scroll);
 var gate=el('div');gate.id='chat-gates';surface.append(gate);
 var form=el('form','chat-composer');form.id='chat-form';var row=el('div','composer-row'),label=el('label','visually-hidden','Votre message');label.htmlFor='chat-content';var input=el('textarea');input.id='chat-content';input.name='content';input.maxLength=6000;input.required=true;input.placeholder=guest?'Connectez-vous pour écrire…':'Votre message…';input.value=state.chatDraft||'';input.disabled=guest;
 input.addEventListener('input',function(){state.chatDraft=input.value});var mic=el('button','chat-microphone','Dicter');mic.type='button';mic.setAttribute('aria-label','Dicter un message');mic.title='Dicter un message';
 var send=el('button','','Envoyer');send.type='submit';input.addEventListener('keydown',function(event){if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing&&!send.disabled){event.preventDefault();form.requestSubmit()}});
 var feedback=el('p','status');feedback.id='chat-status';feedback.setAttribute('role','status');var voiceStatus=el('p','chat-voice-status');voiceStatus.id='chat-voice-status';voiceStatus.setAttribute('role','status');
 row.append(label,input,mic,send);form.append(row,voiceStatus,feedback);form.addEventListener('submit',async function(event){event.preventDefault();await sendChat(form,current)});surface.append(form);
 var announcement=el('p','visually-hidden');announcement.setAttribute('role','status');surface.append(announcement);
 var next=actionButton('Nouvelle réponse',function(){if(shell.latestResponse){scrollChatTo(shell,shell.latestResponse,false);next.hidden=true}},'chat-new-response');next.hidden=true;surface.append(next);
 var shell={surface:surface,scroll:scroll,thread:thread,proposals:proposals,gate:gate,form:form,input:input,mic:mic,voiceStatus:voiceStatus,feedback:feedback,announcement:announcement,next:next,messages:new Map(),proposalNodes:new Map(),follow:true};
 scroll.addEventListener('scroll',function(){shell.follow=scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<100});view.replaceChildren(surface);view.chatShell=shell;
 if(!guest&&typeof createVoiceInput==='function')state.chatVoice=createVoiceInput({input:input,button:mic,status:voiceStatus,maxLength:6000,isCurrent:function(){return chatIsCurrent(current)},onChange:function(value){if(chatIsCurrent(current))state.chatDraft=value},onState:function(voice){if(!chatIsCurrent(current))return;mic.textContent=voice.active||voice.busy?'Arrêter':'Dicter';mic.className='chat-microphone'+(voice.active?' is-listening':'')}});else mic.disabled=true;
 return shell;
}
function drawChatGate(shell,data,current){
 var signature=JSON.stringify([data.consent,data.access]);if(shell.gateSignature===signature)return;shell.gateSignature=signature;shell.gate.replaceChildren();
 var guest=data.access&&data.access.reason==='authentication_required';
 if(guest){var guestBox=el('div','chat-gate'),actions=el('div','actions');['login','register'].forEach(function(mode){var button=el('button',mode==='register'?'secondary':'',mode==='register'?'Créer mon compte':'Me connecter');button.type='button';button.dataset.auth=mode;actions.append(button)});guestBox.append(actions);shell.gate.append(guestBox);return}
 if(!chatConsent(data)){var box=el('div','chat-gate'),consent=el('label','check'),checkbox=el('input');checkbox.type='checkbox';consent.append(checkbox,document.createTextNode('J’accepte le traitement de mes messages et du contexte personnel utile par OpenRouter.'));var activate=actionButton('Activer la conversation',async function(){activate.disabled=true;try{await api('/api/chat/consent','PUT',{accepted:true});if(current())renderChat()}catch(err){if(current()){box.append(el('p','status error',err.message));activate.disabled=!checkbox.checked}}});activate.disabled=true;checkbox.addEventListener('change',function(){activate.disabled=!checkbox.checked});box.append(consent,activate);shell.gate.append(box)}
 var reason=data.access&&data.access.reason,reasons={quota_exhausted:'Vos 100 messages sont utilisés. Reprise au prochain renouvellement.',provider_missing:'Ambre est momentanément indisponible.',generation_paused:'Les réponses sont momentanément en pause.',provider_cooldown:'Le fournisseur est temporairement indisponible. La disponibilité sera vérifiée automatiquement.',subscription_required:'Un abonnement ou un essai actif est nécessaire.'};
 if(reason&&reason!=='consent_required'){var blocked=el('div','chat-gate');blocked.id='chat-access-gate';blocked.append(el('p','',reasons[reason]||'L’envoi est momentanément indisponible.'));if(reason==='subscription_required'){var link=el('a','text-link','Voir mon abonnement');link.href='/mon-compte/';blocked.append(link)}shell.gate.append(blocked)}
}
function scrollChatTo(shell,node,smooth){
 if(!node)return;var top;
 if(typeof node.getBoundingClientRect==='function'&&typeof shell.scroll.getBoundingClientRect==='function')top=shell.scroll.scrollTop+node.getBoundingClientRect().top-shell.scroll.getBoundingClientRect().top-16;
 else top=Number(node.offsetTop)||0;
 if(typeof shell.scroll.scrollTo==='function')shell.scroll.scrollTo({top:Math.max(0,top),behavior:smooth?'smooth':'auto'});else shell.scroll.scrollTop=Math.max(0,top);
}
function revealChatResponse(entry,text,current){
 var reduced=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
 if(reduced||document.hidden||text.length<80){entry.content.textContent=text;return}
 var words=[...text.matchAll(/\S+(?:\s+|$)/g)],steps=Math.min(18,words.length),position=0,timer,done=false;
 var accessible=el('span','visually-hidden',text);entry.item.append(accessible);entry.content.setAttribute('aria-hidden','true');
 function finish(){if(done)return;done=true;clearTimeout(timer);entry.content.textContent=text;entry.content.removeAttribute('aria-hidden');accessible.remove();if(state.chatReveals)state.chatReveals.delete(entry.item)}
 function advance(){if(!chatIsCurrent(current)){finish();return}position++;var last=words[Math.min(words.length-1,Math.ceil(words.length*position/steps)-1)];entry.content.textContent=text.slice(0,last.index+last[0].length);if(position>=steps)finish();else timer=setTimeout(advance,Math.min(80,1400/steps))}
 if(!state.chatReveals)state.chatReveals=new Map();state.chatReveals.set(entry.item,finish);advance();
}
function chatMessageEntry(shell,message,current,animate){
 var id=String(message.id),entry=shell.messages.get(id),client=message.client_key;
 if(!entry&&client){entry=shell.messages.get('client:'+client);if(entry)shell.messages.delete('client:'+client)}
 if(!entry){var item=el('article','chat-message '+(message.role==='user'?'chat-user':'chat-assistant')),heading=el('h3','',message.role==='user'?'Vous':'Ambre'),content=el('p','chat-content'),meta=el('div','chat-turn-state');item.append(heading,content,meta);entry={item:item,content:content,meta:meta};}
 shell.messages.set(id,entry);entry.item.dataset.chatTurnId=id.replace(/-(user|assistant)$/,'');if(client)entry.item.dataset.clientKey=client;
 if(entry.text!==message.content){entry.text=message.content;entry.content.textContent=message.content;
  if(message.role==='assistant'){
   if(!state.chatSeenResponses)state.chatSeenResponses=new Set();var unseen=!state.chatSeenResponses.has(id);state.chatSeenResponses.add(id);
   if(animate&&unseen){revealChatResponse(entry,message.content,current);shell.latestResponse=entry.item;shell.responseToAnchor=entry.item;shell.announcement.textContent='Nouvelle réponse d’Ambre.'}
   if(Array.isArray(message.reading_suggestions))drawChatReadings(entry.item,message.reading_suggestions);
  }
 }
 return entry;
}
function drawChat(view,current,options){
 if(!current())return;var data=state.chatData||{},shell=view.chatShell||createChatShell(view,current),animate=!!(options&&options.animate),messages=(data.messages||[]).map(function(message,index){return Object.assign({},message,{id:message.id||'history-'+index+'-'+message.role})}),box=state.chatOutbox,keys=new Set();
 if(!messages.length&&!box&&data.welcome){messages.push(data.welcome)}
 if(box&&!messages.some(function(message){return message.role==='user'&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')}))messages.push({id:'client:'+box.client_key,client_key:box.client_key,role:'user',content:box.content,status:box.status});
 messages.forEach(function(message){var entry=chatMessageEntry(shell,message,current,animate);keys.add(String(message.id));entry.meta.replaceChildren();shell.thread.append(entry.item);
  if(message.role==='user'&&['failed','expired','cancelled'].includes(message.status)&&!(box&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')))entry.meta.append(el('p','helper','Réponse non aboutie.'));
  if(message.role==='user'&&box&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')&&['uncertain','failed'].includes(box.status)){
   var retryButton=actionButton('Réessayer ce message',function(){retryChatMessage(box,current)},'text');retryButton.dataset.chatRetry=box.client_key;entry.meta.append(el('p','status error',box.error||'Ce message n’a pas abouti.'),retryButton);
  }
 });
 shell.messages.forEach(function(entry,id){if(!keys.has(id)){entry.item.remove();shell.messages.delete(id)}});
 if(state.chatSending){if(!shell.waiting){var waiting=el('article','chat-message chat-assistant chat-thinking');waiting.id='chat-pending-indicator';waiting.setAttribute('role','status');waiting.append(el('h3','','Ambre'),el('span','','Ambre réfléchit…'));var dots=el('span','thinking-dots');dots.setAttribute('aria-hidden','true');for(var i=0;i<3;i++)dots.append(el('i'));waiting.append(dots);shell.waiting=waiting}shell.thread.append(shell.waiting)}else if(shell.waiting){shell.waiting.remove();shell.waiting=null}
 var proposals=(data.proposals||[]).filter(function(proposal){return proposal.status==='pending'}),proposalIds=new Set();proposals.forEach(function(proposal){proposalIds.add(proposal.id);var signature=JSON.stringify(proposal),stored=shell.proposalNodes.get(proposal.id);if(!stored){var host=el('div');shell.proposals.append(host);drawChatProposal(host,proposal,current);shell.proposalNodes.set(proposal.id,{host:host,signature:signature})}var mounted=shell.proposalNodes.get(proposal.id),note=mounted.host.querySelector('.chat-capability'),form=mounted.host.querySelector('form'),submit=form&&form.querySelector('button[type=submit]'),unavailable=chatCapabilityMessage(proposal.kind);if(note){note.textContent=unavailable;note.hidden=!unavailable}if(submit&&form.dataset.submitting!=='true')submit.disabled=!!unavailable;});shell.proposalNodes.forEach(function(stored,id){if(!proposalIds.has(id)){stored.host.remove();shell.proposalNodes.delete(id)}});
 if(shell.responseToAnchor){if(shell.follow){scrollChatTo(shell,shell.responseToAnchor,false);shell.follow=false}else shell.next.hidden=false;shell.responseToAnchor=null}
 drawChatGate(shell,data,current);updateChatControls();if(!state.chatSending){if(box&&box.error&&['failed','uncertain'].includes(box.status)){shell.feedback.textContent=box.error;shell.feedback.className='status error'}else if(data.last_error){shell.feedback.textContent=chatFailureMessage(data);shell.feedback.className='status error'}else{shell.feedback.textContent='';shell.feedback.className='status'}}scheduleChatCooldown(current,data);
 if(options&&options.sent){shell.follow=true;scrollChatTo(shell,shell.waiting||shell.thread.children[shell.thread.children.length-1],false)}
 if(!shell.initialPositioned){shell.initialPositioned=true;var last=messages.slice().reverse().find(function(message){return message.role==='assistant'});if(last){var entry=shell.messages.get(String(last.id));scrollChatTo(shell,entry&&entry.item,false)}}
}
async function sendChat(form,current,retry){
 if(!chatIsCurrent(current)||state.chatSending||!retry&&state.chatOutbox&&['sending','pending','uncertain'].includes(state.chatOutbox.status)||state.chatData&&(!state.chatData.access||!state.chatData.access.allowed))return;
 if(state.chatVoice)state.chatVoice.stop('');var input=form.elements.content,content=retry?retry.content:input.value.trim();if(!content)return;
 var sameRetry=retry&&retry.status==='uncertain',key=sameRetry?retry.client_key:newClientId(),box={client_key:key,content:content,status:'sending',started_at:Date.now()};
 state.chatOutbox=box;state.chatSendKey={content:content,key:key};state.chatSending=true;state.chatRequest=null;
 if(!retry){input.value='';state.chatDraft=''}
 var view=document.getElementById('view');if(view)drawChat(view,current,{sent:true});updateChatControls();var submit=form.querySelector('button[type=submit]');if(submit)submit.disabled=true;status('chat-status','');
 try{
  var result=await api('/api/chat/messages','POST',{content:content,idempotency_key:key});if(!chatIsCurrent(current)||state.chatOutbox!==box)return;
  box.request_id=result.request_id||result.pending_request&&result.pending_request.id||result.request&&result.request.id||null;
  if(result.pending_request||result.request&&['queued','running','pending'].includes(result.request.status||'queued'))box.status='pending';
  var latest=result.messages?result:await api('/api/chat');if(!chatIsCurrent(current)||state.chatOutbox!==box)return;
  state.chatData=latest;reconcileChatTurn(latest);
  if(!latest.pending_request&&!latest.messages&&!latest.last_error){box.status='completed';state.chatSending=false;state.chatSendKey=null}
  if(view)drawChat(view,current,{animate:true});updateChatControls();
  if(state.chatSending)pollChat(state.chatRequest||{id:box.request_id,client_key:key},current,null);
 }catch(err){
  if(!chatIsCurrent(current)||state.chatOutbox!==box)return;
  var definitive=Number(err.status)>=400&&Number(err.status)<500||['provider_cooldown','provider_missing','generation_paused','subscription_required','consent_required'].includes(err.code);
  box.status=definitive?'failed':'uncertain';box.error=err.message+(definitive?' Votre texte reste disponible.':' L’envoi n’a pas été confirmé. Vous pouvez réessayer ce message.');state.chatSending=false;state.chatRequest=null;
  if(definitive){state.chatSendKey=null;if(!state.chatDraft&&!input.value){input.value=content;state.chatDraft=content;box.restored=true}}
  if(view)drawChat(view,current);updateChatControls();status('chat-status',box.error,true);
  try{var latest=await api('/api/chat');if(!chatIsCurrent(current)||state.chatOutbox!==box)return;state.chatData=latest;reconcileChatTurn(latest);if(view)drawChat(view,current,{animate:true});applyChatAvailability(latest,current);if(state.chatSending)pollChat(state.chatRequest||{id:box.request_id,client_key:key},current,null)}catch(ignore){}
 }
}
function retryChatMessage(box,current){
 if(!chatIsCurrent(current)||state.chatOutbox!==box||state.chatSending)return;var form=document.getElementById('chat-form');if(form)return sendChat(form,current,box);
}
function pollChat(request,current,sentDraft){
 if(!chatIsCurrent(current))return;clearTimeout(state.chatTimer);state.chatTimer=setTimeout(async function(){
  if(!chatIsCurrent(current))return;
  try{
   var result=await api('/api/chat');if(!chatIsCurrent(current))return;state.chatData=Object.assign({},state.chatData||{},result);reconcileChatTurn(result);
   var failed=result.last_error||result.last_request&&['failed','cancelled','expired'].includes(result.last_request.status)||result.request&&['failed','cancelled','expired'].includes(result.request.status);
   if(failed&&!state.chatSending){state.chatSendKey=null;status('chat-status',chatFailureMessage(result),true)}
   if(!state.chatOutbox&&!result.pending_request)state.chatSending=false;
   var view=document.getElementById('view');if(view)drawChat(view,current,{animate:true});updateChatControls();applyChatAvailability(result,current);
   if(state.chatSending)pollChat(state.chatRequest||request,current,null);
  }catch(err){
   if(!chatIsCurrent(current))return;state.chatRequest=null;state.chatSending=false;if(state.chatOutbox&&['sending','pending'].includes(state.chatOutbox.status)){state.chatOutbox.status='uncertain';state.chatOutbox.error='La réponse n’a pas pu être vérifiée. Vous pouvez réessayer ce message.'}
   status('chat-status',err.message+' Votre texte est conservé.',true);var view=document.getElementById('view');if(view)drawChat(view,current);updateChatControls();
  }
 },1500);
}
`;
