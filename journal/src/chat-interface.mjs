// Embedded in the app shell: no framework, network client or persisted history.
export const CHAT_INTERFACE_SCRIPT=String.raw`
function chatConsent(data){return data.consent===true||!!(data.consent&&data.consent.accepted)}
function chatFailureMessage(data){var error=data&&data.last_error;return (error&&error.message||'La réponse n’a pas pu être préparée.')+' Votre texte reste disponible.'}
function chatIsCurrent(current){return current()&&state.tab==='chat'}
function chatJourney(data){return data&&data.journey||null}
function chatReviewRunning(data){var journey=chatJourney(data);return !!(journey&&journey.review&&journey.review.running_id)}
function chatJourneyBlocked(data){var journey=chatJourney(data);return !!(journey&&(!journey.started||chatReviewRunning(data)))}
function chatPointNote(data){
 var journey=chatJourney(data);if(!journey||!journey.started)return '';
 if(chatReviewRunning(data))return 'Ambre prépare déjà un bilan.';
 var next=Number(journey.review&&journey.review.next_point_at||0);
 if(next&&next*1000>Date.now())return 'Prochain point possible à partir du '+new Date(next*1000).toLocaleString('fr-FR',{day:'numeric',month:'long',hour:'2-digit',minute:'2-digit'})+'.';
 return '';
}
function chatHasPending(){return !!(state.chatSending||state.chatOutbox&&['sending','pending','uncertain'].includes(state.chatOutbox.status))}
function chatCapabilityMessage(kind){
 var capability=state.chatData&&state.chatData.capabilities&&state.chatData.capabilities[kind];if(!capability||capability.allowed!==false)return '';
 var messages={memory_append:'La mémoire de précisions est pleine.',person_create:'La limite de cinquante proches est atteinte.',life_event_create:'La limite de cinq cents moments de vie est atteinte.',person_update:'Aucun proche enregistré ne correspond à cette modification.',life_event_update:'Aucun moment enregistré ne correspond à cette modification.'};
 return messages[kind]||'Cette opération n’est pas disponible pour le moment.';
}
function updateChatControls(){
 var data=state.chatData||{},form=document.getElementById('chat-form');if(!form)return;
 var send=form.querySelector('button[type=submit]');if(send)send.disabled=chatHasPending()||!!state.chatVoice?.isActive()||!chatConsent(data)||!data.access||!data.access.allowed||chatJourneyBlocked(data)||!!(data.quota&&Number(data.quota.remaining||0)<=0);
 var point=document.getElementById('chat-point'),pointNote=document.getElementById('chat-point-note'),journey=chatJourney(data);if(point){var reason=chatPointNote(data);point.hidden=!(journey&&journey.started);point.disabled=!!reason||chatHasPending()||!chatConsent(data);if(pointNote){pointNote.textContent=reason;pointNote.hidden=!reason||point.hidden}}
 var mounted=document.getElementById('view'),shell=mounted&&mounted.chatShell;if(shell)shell.thread.querySelectorAll('button').forEach(function(button){if(button.dataset.chatRetry)button.disabled=!!(state.chatSending||!data.access||!data.access.allowed||data.quota&&Number(data.quota.remaining||0)<=0)});
 var input=form.elements.content;if(input)input.disabled=!!(data.access&&data.access.reason==='authentication_required');
 var notice=document.getElementById('chat-memory-notice');if(notice){notice.hidden=!(chatConsent(data)&&data.people_observations_enabled);if(notice.hidden)input.removeAttribute('aria-describedby');else input.setAttribute('aria-describedby',notice.id)}
 var quota=document.getElementById('chat-quota');
 if(quota&&data.quota)quota.textContent=data.quota.used+' / '+data.quota.limit;
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
function drawChatNotes(host,notes){
 var valid=(notes||[]).filter(function(note){return note&&typeof note.text==='string'&&note.text.trim()}).slice(0,3);if(!valid.length)return;
 var line=el('p','chat-noted');line.append(el('span','chat-noted-label','Noté dans votre cartographie : '));
 valid.forEach(function(note,index){if(index)line.append(document.createTextNode(' · '));line.append(document.createTextNode((note.label?note.label+' — ':'')+'« '+note.text+' »'))});
 var link=el('a','text-link','Voir ma cartographie');link.href='/cartographie/';line.append(document.createTextNode(' '),link);host.append(line);
}
function createChatShell(view,current){
 var data=state.chatData||{},guest=data.access&&data.access.reason==='authentication_required',surface=el('section','conversation');
 var toolbar=el('div','conversation-toolbar'),title=el('h1');
 title.append(el('span','','Ambre'),el('span','conversation-title-divider',' · '),el('span','conversation-role','psychologue IA'));
 title.id='conversation-title';surface.setAttribute('aria-labelledby',title.id);
 var quota=el('div','message-quota'),count=el('strong','','—');count.id='chat-quota';
 quota.setAttribute('role','status');quota.append(count,el('span','',' messages'));
 var point=actionButton('Faire le point',function(){openPointDialog(current)},'secondary chat-point');point.id='chat-point';point.hidden=true;
 var pointNote=el('p','chat-point-note');pointNote.id='chat-point-note';pointNote.hidden=true;point.setAttribute('aria-describedby',pointNote.id);
 toolbar.append(title,point,quota);
 surface.append(toolbar,pointNote);
 var scroll=el('div','conversation-scroll');scroll.id='conversation-scroll';var thread=el('section','chat-thread');thread.setAttribute('aria-label','Conversation');thread.setAttribute('role','log');thread.setAttribute('aria-live','off');scroll.append(thread);
 var proposals=el('section');proposals.id='chat-proposals';scroll.append(proposals);surface.append(scroll);
 var gate=el('div');gate.id='chat-gates';surface.append(gate);
 var form=el('form','chat-composer');form.id='chat-form';var row=el('div','composer-row'),label=el('label','visually-hidden','Votre message');label.htmlFor='chat-content';var input=el('textarea');input.id='chat-content';input.name='content';input.maxLength=6000;input.required=true;input.placeholder=guest?'Connectez-vous pour écrire…':'Votre message…';input.value=state.chatDraft||'';input.disabled=guest;
 input.addEventListener('input',function(){state.chatDraft=input.value});var mic=el('button','chat-microphone');mic.type='button';mic.setAttribute('aria-label','Dicter un message');mic.setAttribute('aria-describedby','chat-voice-status');mic.title='Dicter un message';
 mic.innerHTML='<svg class="microphone-glyph" aria-hidden="true" focusable="false" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3m-3 0h6"/></svg><svg class="stop-glyph" aria-hidden="true" focusable="false" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>';
 var send=el('button','','Envoyer');send.type='submit';input.addEventListener('keydown',function(event){if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing&&!send.disabled){event.preventDefault();form.requestSubmit()}});
 var feedback=el('p','status');feedback.id='chat-status';feedback.setAttribute('role','status');var voiceStatus=el('p','chat-voice-status');voiceStatus.id='chat-voice-status';voiceStatus.setAttribute('role','status');
 var notice=el('p','chat-memory-notice','Ambre peut retenir des repères sur les proches déjà ajoutés.');notice.id='chat-memory-notice';notice.hidden=true;
 row.append(label,input,mic,send);form.append(notice,row,voiceStatus,feedback);form.addEventListener('submit',async function(event){event.preventDefault();await sendChat(form,current)});surface.append(form);
 var announcement=el('p','visually-hidden');announcement.setAttribute('role','status');surface.append(announcement);
 var next=actionButton('Nouvelle réponse',function(){if(shell.latestResponse){scrollChatTo(shell,shell.latestResponse,false);next.hidden=true}},'chat-new-response');next.hidden=true;surface.append(next);
 var shell={surface:surface,scroll:scroll,thread:thread,proposals:proposals,gate:gate,form:form,input:input,mic:mic,voiceStatus:voiceStatus,feedback:feedback,announcement:announcement,next:next,messages:new Map(),proposalNodes:new Map(),follow:true};
 scroll.addEventListener('scroll',function(){shell.follow=scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<100});view.replaceChildren(surface);view.chatShell=shell;
 if(!guest&&typeof createVoiceInput==='function')state.chatVoice=createVoiceInput({input:input,button:mic,status:voiceStatus,maxLength:6000,isCurrent:function(){return chatIsCurrent(current)},onChange:function(value){if(chatIsCurrent(current))state.chatDraft=value},onState:function(){if(chatIsCurrent(current))updateChatControls()}});else mic.disabled=true;
 return shell;
}
function drawChatGate(shell,data,current){
 var signature=JSON.stringify([data.consent,data.access,chatJourney(data)&&chatJourney(data).started]);if(shell.gateSignature===signature)return;shell.gateSignature=signature;shell.gate.replaceChildren();
 var guest=data.access&&data.access.reason==='authentication_required';
 if(guest){var guestBox=el('div','chat-gate'),actions=el('div','actions');['login','register'].forEach(function(mode){var button=el('button',mode==='register'?'secondary':'',mode==='register'?'Créer mon compte':'Me connecter');button.type='button';button.dataset.auth=mode;actions.append(button)});guestBox.append(actions);shell.gate.append(guestBox);return}
 if(!chatConsent(data)&&!(chatJourney(data)&&!chatJourney(data).started)){var box=el('div','chat-gate'),consent=el('label','check'),checkbox=el('input');checkbox.type='checkbox';consent.append(checkbox,document.createTextNode('J’accepte le traitement de mes messages et du contexte personnel utile par OpenRouter. Ambre peut retenir des repères concrets tirés de mes échanges sur les proches que j’ai déjà ajoutés.'));var activate=actionButton('Activer la conversation',async function(){activate.disabled=true;try{await api('/api/chat/consent','PUT',{accepted:true});if(current())renderChat()}catch(err){if(current()){box.append(el('p','status error',err.message));activate.disabled=!checkbox.checked}}});activate.disabled=true;checkbox.addEventListener('change',function(){activate.disabled=!checkbox.checked});box.append(consent,activate);shell.gate.append(box)}
 var reason=data.access&&data.access.reason,reasons={quota_exhausted:'Vos 100 messages sont utilisés. Reprise au prochain renouvellement.',provider_missing:'Ambre est momentanément indisponible.',generation_paused:'Les réponses sont momentanément en pause.',provider_cooldown:'Le fournisseur est temporairement indisponible. La disponibilité sera vérifiée automatiquement.',subscription_required:'Un abonnement ou un essai actif est nécessaire.'};
 if(reason&&reason!=='consent_required'&&reason!=='journey_not_started'){var blocked=el('div','chat-gate');blocked.id='chat-access-gate';blocked.append(el('p','',reasons[reason]||'L’envoi est momentanément indisponible.'));if(reason==='subscription_required'){var link=el('a','text-link','Voir mon abonnement');link.href='/mon-compte/';blocked.append(link)}shell.gate.append(blocked)}
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
 if(!entry){var item=el('article','chat-message '+(message.role==='user'?'chat-user':'chat-assistant')),heading=el('h3','',message.role==='user'?'Vous':message.review_id?(message.review_kind==='point'?'Ambre · Point d’étape':'Ambre · Analyse initiale'):'Ambre'),content=el('p','chat-content'),meta=el('div','chat-turn-state');item.append(heading,content,meta);entry={item:item,content:content,meta:meta};}
 shell.messages.set(id,entry);entry.item.dataset.chatTurnId=id.replace(/-(user|assistant)$/,'');if(client)entry.item.dataset.clientKey=client;
 if(entry.text!==message.content){entry.text=message.content;entry.content.textContent=message.content;
  if(message.role==='assistant'){
   if(!state.chatSeenResponses)state.chatSeenResponses=new Set();var unseen=!state.chatSeenResponses.has(id);state.chatSeenResponses.add(id);
   if(animate&&unseen){revealChatResponse(entry,message.content,current);shell.latestResponse=entry.item;shell.responseToAnchor=entry.item;shell.announcement.textContent='Nouvelle réponse d’Ambre.'}
   if(Array.isArray(message.cartography_notes))drawChatNotes(entry.item,message.cartography_notes);
   if(message.review_id&&!entry.reviewLink){var bilans=el('a','text-link chat-review-link','Retrouver ce bilan dans Mes bilans');bilans.href='/mes-bilans/';entry.reviewLink=bilans;entry.item.append(bilans)}
  }
 }
 return entry;
}
function drawChat(view,current,options){
 if(!current())return;var data=state.chatData||{},shell=view.chatShell||createChatShell(view,current),animate=!!(options&&options.animate),messages=(data.messages||[]).map(function(message,index){return Object.assign({},message,{id:message.id||'history-'+index+'-'+message.role})}),box=state.chatOutbox,keys=new Set();
 if(data.welcome){messages=messages.filter(function(message){return !/^ambre-welcome-/.test(message.id)});messages.unshift(data.welcome)}
 if(box&&!messages.some(function(message){return message.role==='user'&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')}))messages.push({id:'client:'+box.client_key,client_key:box.client_key,role:'user',content:box.content,status:box.status});
 messages.forEach(function(message){var entry=chatMessageEntry(shell,message,current,animate&&message.id!==data.welcome?.id);keys.add(String(message.id));entry.meta.replaceChildren();shell.thread.append(entry.item);
  if(message.role==='user'&&['failed','expired','cancelled'].includes(message.status)&&!(box&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')))entry.meta.append(el('p','helper','Réponse non aboutie.'));
  if(message.role==='user'&&box&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')&&['uncertain','failed'].includes(box.status)){
   var retryButton=actionButton('Réessayer ce message',function(){retryChatMessage(box,current)},'text');retryButton.dataset.chatRetry=box.client_key;entry.meta.append(el('p','status error',box.error||'Ce message n’a pas abouti.'),retryButton);
  }
 });
 shell.messages.forEach(function(entry,id){if(!keys.has(id)){entry.item.remove();shell.messages.delete(id)}});
 if(state.chatSending){if(!shell.waiting){var waiting=el('article','chat-message chat-assistant chat-thinking');waiting.id='chat-pending-indicator';waiting.setAttribute('role','status');waiting.append(el('h3','','Ambre'),el('span','','Ambre réfléchit…'));var dots=el('span','thinking-dots');dots.setAttribute('aria-hidden','true');for(var i=0;i<3;i++)dots.append(el('i'));waiting.append(dots);shell.waiting=waiting}shell.thread.append(shell.waiting)}else if(shell.waiting){shell.waiting.remove();shell.waiting=null}
 if(chatReviewRunning(data)&&!chatInitialPending(data)){if(!shell.reviewWaiting){var preparing=el('article','chat-message chat-assistant chat-thinking');preparing.id='chat-review-indicator';preparing.setAttribute('role','status');preparing.append(el('h3','','Ambre'),el('span','','Ambre prépare votre bilan…'));var reviewDots=el('span','thinking-dots');reviewDots.setAttribute('aria-hidden','true');for(var d=0;d<3;d++)reviewDots.append(el('i'));preparing.append(reviewDots);shell.reviewWaiting=preparing}shell.thread.append(shell.reviewWaiting)}else if(shell.reviewWaiting){shell.reviewWaiting.remove();shell.reviewWaiting=null}
 var proposals=(data.proposals||[]).filter(function(proposal){return proposal.status==='pending'&&proposal.kind!=='story_request'}),proposalIds=new Set();proposals.forEach(function(proposal){proposalIds.add(proposal.id);var signature=JSON.stringify(proposal),stored=shell.proposalNodes.get(proposal.id);if(!stored){var host=el('div');shell.proposals.append(host);drawChatProposal(host,proposal,current);shell.proposalNodes.set(proposal.id,{host:host,signature:signature})}var mounted=shell.proposalNodes.get(proposal.id),note=mounted.host.querySelector('.chat-capability'),form=mounted.host.querySelector('form'),submit=form&&form.querySelector('button[type=submit]'),unavailable=chatCapabilityMessage(proposal.kind);if(note){note.textContent=unavailable;note.hidden=!unavailable}if(submit&&form.dataset.submitting!=='true')submit.disabled=!!unavailable;});shell.proposalNodes.forEach(function(stored,id){if(!proposalIds.has(id)){stored.host.remove();shell.proposalNodes.delete(id)}});
 if(shell.responseToAnchor){if(shell.follow){scrollChatTo(shell,shell.responseToAnchor,false);shell.follow=false}else shell.next.hidden=false;shell.responseToAnchor=null}
 drawChatGate(shell,data,current);updateChatControls();if(!state.chatSending){if(box&&box.error&&['failed','uncertain'].includes(box.status)){shell.feedback.textContent=box.error;shell.feedback.className='status error'}else if(data.last_error){shell.feedback.textContent=chatFailureMessage(data);shell.feedback.className='status error'}else{shell.feedback.textContent='';shell.feedback.className='status'}}scheduleChatCooldown(current,data);
 if(options&&options.sent){shell.follow=true;scrollChatTo(shell,shell.waiting||shell.thread.children[shell.thread.children.length-1],false)}
 drawJourneyDialog(current);if(chatReviewRunning(data))pollJourney(current);
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
function chatInitialPending(data){var journey=chatJourney(data);return !!(journey&&journey.review&&journey.review.running_id&&journey.review.last_at==null)}
function journeyDialog(){
 var dialog=document.getElementById('journey-dialog');
 if(!dialog){dialog=el('dialog','journey-dialog');dialog.id='journey-dialog';if(document.body)document.body.append(dialog)}
 if(!dialog.dataset.bound){dialog.dataset.bound='1';dialog.setAttribute('aria-labelledby','journey-title');
  // The welcome window cannot be dismissed by accident; a simple confirmation can.
  dialog.addEventListener('cancel',function(event){if(dialog.dataset.step!=='point')event.preventDefault();else state.journeyStep=null});
 }
 return dialog;
}
function closeJourneyDialog(){var dialog=document.getElementById('journey-dialog');if(!dialog)return;if(dialog.open&&typeof dialog.close==='function')dialog.close();else dialog.open=false;dialog.dataset.step=''}
function showJourneyDialog(dialog,step,nodes){
 var changed=dialog.dataset.step!==step;dialog.dataset.step=step;dialog.replaceChildren.apply(dialog,nodes);
 if(!dialog.open){if(typeof dialog.showModal==='function')dialog.showModal();else dialog.open=true}
 if(changed){var target=dialog.querySelector('h2');if(target){target.tabIndex=-1;target.focus()}}
}
function journeySourceItem(label,href,source,detail){
 var item=el('li','journey-source'+(source&&source.done?' is-done':'')),name=el('span','journey-source-name',label),badge=el('span','journey-source-state',source&&source.done?'Renseigné':'À compléter'),link=el('a','text-link',source&&source.done?'Modifier':'Compléter');
 link.href=href;link.setAttribute('aria-label',(source&&source.done?'Modifier : ':'Compléter : ')+label);
 var text=el('span','journey-source-text');text.append(name);if(source&&source.done&&detail)text.append(el('span','journey-source-detail',detail));
 item.append(text,badge,link);return item;
}
function journeyCount(count,one,many){count=Number(count)||0;return count?count+' '+(count>1?many:one):''}
function drawJourneyDialog(current){
 var data=state.chatData||{},journey=chatJourney(data),guest=data.access&&data.access.reason==='authentication_required';
 if(!journey||guest||!chatIsCurrent(current)){if(state.journeyStep!=='point')closeJourneyDialog();return}
 if(state.journeyStep==='point'){if(chatReviewRunning(data)||!journey.started){state.journeyStep=null;closeJourneyDialog()}else return}
 var pending=chatInitialPending(data);
 if(journey.started&&!pending){state.journeyStep=null;state.journeyWaiting=false;state.journeyKey=null;closeJourneyDialog();return}
 var dialog=journeyDialog(),title=el('h2');title.id='journey-title';
 if(pending){
  state.journeyWaiting=true;title.textContent='Ambre prépare votre analyse…';
  var wait=el('p','journey-wait');wait.setAttribute('role','status');wait.append(document.createTextNode('Elle relit ce que vous avez renseigné et rédige votre analyse initiale. Cela prend en général une à deux minutes. Vous pouvez laisser cette page ouverte : l’analyse apparaîtra dans la conversation.'));
  var dots=el('span','thinking-dots');dots.setAttribute('aria-hidden','true');for(var i=0;i<3;i++)dots.append(el('i'));
  showJourneyDialog(dialog,'waiting',[title,wait,dots]);return;
 }
 var failed=state.journeyWaiting;if(failed){state.journeyWaiting=false;state.journeyKey=null;state.journeyStep=null;state.journeyError='L’analyse n’a pas pu être préparée. Rien n’a été perdu : vous pouvez la relancer.'}
 if(state.journeyStep==='confirm'&&journey.can_start){
  title.textContent='Êtes-vous certain ?';
  var consent=el('label','check'),checkbox=el('input');checkbox.type='checkbox';checkbox.id='journey-consent';
  consent.append(checkbox,document.createTextNode('J’accepte que mes messages et les informations que j’ai renseignées soient traités par OpenRouter pour l’analyse et la conversation. Ambre peut retenir des repères concrets sur les proches que j’ai déjà ajoutés et écrire de courtes notes dans ma cartographie.'));
  var feedback=el('p','status');feedback.setAttribute('role','status');
  var yes=actionButton('Oui, lancer',async function(){
   yes.disabled=true;no.disabled=true;feedback.textContent='Lancement…';feedback.className='status';
   try{
    if(!state.journeyKey)state.journeyKey=newClientId();
    var result=await api('/api/chat/start','POST',{idempotency_key:state.journeyKey,accepted:true});if(!chatIsCurrent(current))return;
    state.journeyError=null;state.chatData=Object.assign({},state.chatData||{},result);state.journeyWaiting=true;
    var view=document.getElementById('view');if(view)drawChat(view,current,{animate:true});else drawJourneyDialog(current);
   }catch(err){
    if(!chatIsCurrent(current))return;if(Number(err.status)>=400&&Number(err.status)<500)state.journeyKey=null;
    feedback.textContent=err.message;feedback.className='status error';no.disabled=false;yes.disabled=!checkbox.checked;
   }
  });
  var no=actionButton('Pas encore',function(){state.journeyStep=null;drawJourneyDialog(current)},'secondary');
  yes.disabled=true;checkbox.addEventListener('change',function(){yes.disabled=!checkbox.checked});
  var actions=el('div','actions');actions.append(yes,no);
  showJourneyDialog(dialog,'confirm',[title,el('p','','Cela va initier la conversation à partir des informations que vous avez renseignées.'),consent,actions,feedback]);return;
 }
 state.journeyStep=null;title.textContent='Avant de commencer avec Ambre';
 var sources=journey.sources||{},list=el('ul','journey-sources');
 list.append(journeySourceItem('Ma cartographie','/cartographie/',sources.cartography,journeyCount(sources.cartography&&sources.cartography.count,'élément coché','éléments cochés')),journeySourceItem('Mon portrait','/mon-portrait/',sources.portrait,''),journeySourceItem('Mes proches','/mes-proches/',sources.people,journeyCount(sources.people&&sources.people.count,'proche','proches')),journeySourceItem('Ma ligne de vie','/ma-ligne-de-vie/',sources.life_events,journeyCount(sources.life_events&&sources.life_events.count,'moment','moments')));
 var start=actionButton('Lancer la discussion avec Ambre',function(){state.journeyError=null;state.journeyStep='confirm';drawJourneyDialog(current)});start.id='journey-start';start.disabled=!journey.can_start;
 var hint=el('p','helper',journey.can_start?'Plus vous renseignez de sources, plus l’analyse est précise. Vous pourrez les compléter ensuite et demander un nouveau point.':'Renseignez au moins une de ces sources pour lancer la discussion.');hint.id='journey-hint';start.setAttribute('aria-describedby',hint.id);
 var nodes=[title,el('p','','Ambre commence par une analyse approfondie de ce que vous avez renseigné : votre cartographie, votre portrait, vos proches et votre ligne de vie. La conversation se poursuit ensuite autour de ce que vous voulez comprendre et de ce que vous voulez changer.'),el('p','muted','Ambre est une IA de réflexion psychologique. Elle ne pose pas de diagnostic et ne remplace pas un professionnel de santé.'),list,hint];
 if(state.journeyError){var error=el('p','status error',state.journeyError);error.setAttribute('role','alert');nodes.push(error)}
 nodes.push(start);showJourneyDialog(dialog,'intro',nodes);
}
function openPointDialog(current){
 var data=state.chatData||{};if(!chatIsCurrent(current)||chatPointNote(data)||!chatJourney(data)||!chatJourney(data).started)return;
 var dialog=journeyDialog(),title=el('h2','','Faire le point ?');title.id='journey-title';state.journeyStep='point';
 var feedback=el('p','status');feedback.setAttribute('role','status');
 function close(){state.journeyStep=null;closeJourneyDialog();var point=document.getElementById('chat-point');if(point)point.focus()}
 var yes=actionButton('Oui, faire le point',async function(){
  yes.disabled=true;no.disabled=true;feedback.textContent='Demande envoyée…';feedback.className='status';
  try{
   if(!state.pointKey)state.pointKey=newClientId();
   var result=await api('/api/chat/review','POST',{idempotency_key:state.pointKey});if(!chatIsCurrent(current))return;
   state.pointKey=null;state.chatData=Object.assign({},state.chatData||{},result);close();
   var view=document.getElementById('view');if(view)drawChat(view,current,{sent:true});updateChatControls();
  }catch(err){
   if(!chatIsCurrent(current))return;if(Number(err.status)>=400&&Number(err.status)<500)state.pointKey=null;
   feedback.textContent=err.message;feedback.className='status error';no.disabled=false;yes.disabled=false;
  }
 });
 var no=actionButton('Pas maintenant',close,'secondary'),actions=el('div','actions');actions.append(yes,no);
 showJourneyDialog(dialog,'point',[title,el('p','','Ambre rédige un nouveau bilan à partir de vos informations et de vos échanges depuis le dernier bilan. Cela prend en général une à deux minutes ; il sera aussi conservé dans Mes bilans.'),actions,feedback]);
}
function pollJourney(current){
 if(!chatIsCurrent(current))return;clearTimeout(state.journeyTimer);state.journeyTimer=setTimeout(async function(){
  if(!chatIsCurrent(current))return;
  try{
   var result=await api('/api/chat');if(!chatIsCurrent(current))return;state.journeyPollFailures=0;
   state.chatData=Object.assign({},state.chatData||{},result);reconcileChatTurn(result);
   var view=document.getElementById('view');if(view)drawChat(view,current,{animate:true});updateChatControls();
  }catch(err){
   if(!chatIsCurrent(current))return;state.journeyPollFailures=(state.journeyPollFailures||0)+1;
   if(state.journeyPollFailures<=5)pollJourney(current);else status('chat-status','L’avancement du bilan n’a pas pu être vérifié. Rechargez la page pour le retrouver.',true);
  }
 },3000);
}
function reviewTitle(review){return review.kind==='point'?'Point du '+dateLabel(Number(review.created_at)):'Analyse initiale'}
async function renderReviews(){
 var view=document.getElementById('view'),current=renderGuard(view);view.replaceChildren();
 var section=el('section','panel narrow reviews'),message=el('p','status');message.id='reviews-status';message.setAttribute('role','status');
 var list=el('div','review-list');list.id='reviews-list';list.append(el('p','muted','Chargement de vos bilans…'));section.append(message,list);view.append(section);
 try{var result=await api('/api/reviews');if(!current()||state.tab!=='reviews')return;list.replaceChildren();drawReviewPage(list,result,current)}
 catch(err){if(current()&&state.tab==='reviews')list.replaceChildren(el('p','status error',err.message),actionButton('Réessayer',renderReviews,'secondary'))}
}
function drawReviewPage(list,result,current){
 var reviews=result.reviews||[];
 if(!reviews.length&&!list.children.length){
  var empty=el('div','workspace-empty');empty.append(el('h2','','Aucun bilan pour le moment'),el('p','','Les bilans sont rédigés par Ambre. Le premier, l’analyse initiale, est écrit au lancement de la discussion à partir de votre cartographie, de votre portrait, de vos proches et de votre ligne de vie. Ensuite, un nouveau point s’ajoute chaque fois que vous le demandez avec « Faire le point » dans la conversation.'));
  var link=el('a','text-link','Ouvrir la conversation avec Ambre');link.href='/mon-accompagnement/';empty.append(link);list.append(empty);return;
 }
 reviews.forEach(function(review){list.append(drawReviewItem(review,current))});
 if(result.next_cursor){var more=actionButton('Afficher les bilans plus anciens',async function(){more.disabled=true;try{var next=await api('/api/reviews?cursor='+encodeURIComponent(result.next_cursor));if(!current())return;more.remove();drawReviewPage(list,next,current)}catch(err){if(current()){status('reviews-status',err.message,true);more.disabled=false}}},'secondary');list.append(more)}
}
function drawReviewItem(review,current){
 var item=el('article','review-item'),title=el('h2','review-title',reviewTitle(review)),meta=el('p','review-meta'),body=el('div','review-body'),actions=el('div','actions');
 item.dataset.reviewId=review.id;
 meta.textContent=review.status==='running'?'En préparation…':review.status==='failed'?'Ce bilan n’a pas abouti. Aucun message n’a été décompté.':'Rédigé le '+dateLabel(Number(review.completed_at||review.created_at));
 if(review.status==='ready'&&review.excerpt)body.append(el('p','review-excerpt',review.excerpt+(Number(review.length)>review.excerpt.length?'…':'')));
 var loaded=false;
 if(review.status==='ready'){
  var read=actionButton('Lire le bilan',async function(){
   if(loaded){body.hidden=!body.hidden;read.textContent=body.hidden?'Lire le bilan':'Replier';read.setAttribute('aria-expanded',String(!body.hidden));return}
   read.disabled=true;
   try{var result=await api('/api/reviews/'+encodeURIComponent(review.id));if(!current())return;loaded=true;body.replaceChildren();String(result.review.content||'').split(/\n{2,}/).forEach(function(paragraph){if(paragraph.trim())body.append(el('p','',paragraph.trim()))});body.hidden=false;read.textContent='Replier';read.setAttribute('aria-expanded','true')}
   catch(err){if(current())status('reviews-status',err.message,true)}
   finally{read.disabled=false}
  },'secondary');read.setAttribute('aria-expanded','false');actions.append(read);
 }
 var remove=actionButton('Supprimer',async function(){
  if(!confirm('Supprimer définitivement « '+reviewTitle(review)+' » ? Il disparaîtra aussi de la conversation.'+(review.kind==='initial'?' La discussion devra être relancée par une nouvelle analyse.':'')))return;
  remove.disabled=true;
  try{await api('/api/reviews/'+encodeURIComponent(review.id),'DELETE');if(!current())return;var list=item.parentElement;item.remove();status('reviews-status','Bilan supprimé.');if(list&&!list.children.length)drawReviewPage(list,{reviews:[]},current)}
  catch(err){if(current()){status('reviews-status',err.message,true);remove.disabled=false}}
 },'text');actions.append(remove);
 item.append(title,meta,body,actions);return item;
}
`;
