// Embedded in the app shell: no framework, network client or persisted history.
import {REQUIRED_SYMPTOMS} from './companion-readiness.mjs';
export const CHAT_INTERFACE_SCRIPT=String.raw`
function chatConsent(data){return data.consent===true||!!(data.consent&&data.consent.accepted)}
function chatFailureMessage(data){var error=data&&data.last_error;return (error&&error.message||'La réponse n’a pas pu être préparée.')+' Votre texte reste disponible.'}
function chatIsCurrent(current){return current()&&state.tab==='chat'}
function chatJourney(data){return data&&data.journey||null}
function chatReviewRunning(data){var journey=chatJourney(data);return !!(journey&&journey.review&&journey.review.running_id)}
function chatJourneyBlocked(data){var journey=chatJourney(data),readiness=data&&(data.readiness||journey&&journey.readiness);return !!(journey&&(!journey.started||chatReviewRunning(data))||readiness&&!readiness.complete)}
function chatPointNote(data){
 var journey=chatJourney(data);if(!journey||!journey.started)return '';
 if(chatReviewRunning(data))return 'Ambre prépare déjà un bilan.';
 var readiness=data.readiness||journey.readiness;if(readiness&&!readiness.complete)return 'Complétez les informations nécessaires pour faire le point.';
 var next=Number(journey.review&&journey.review.next_point_at||0);
 if(next&&next*1000>Date.now())return 'Prochain point possible à partir du '+new Date(next*1000).toLocaleString('fr-FR',{day:'numeric',month:'long',hour:'2-digit',minute:'2-digit'})+'.';
 return '';
}
// Lot A5-P1: a point may be suggested (a session soon, or many exchanges); it is never started for the person.
function chatPointSuggestion(data){
 var journey=chatJourney(data),suggestion=journey&&journey.review&&journey.review.suggestion;if(!suggestion)return '';
 if(suggestion.reason==='session_soon'&&suggestion.session_at)return 'Une séance est prévue le '+new Date(suggestion.session_at*1000).toLocaleString('fr-FR',{weekday:'long',day:'numeric',month:'long',hour:'2-digit',minute:'2-digit'})+' : vous pouvez faire le point avant, si vous le souhaitez.';
 if(suggestion.reason==='many_exchanges')return 'Vous avez beaucoup échangé depuis le dernier bilan : vous pouvez faire le point quand vous le souhaitez.';
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
 var send=form.querySelector('button[type=submit]');if(send)send.disabled=chatHasPending()||!!state.chatVoice?.isActive()||!chatConsent(data)||!data.access||!data.access.allowed||chatJourneyBlocked(data)||!!ambreComposerLock(data)||!!(data.quota&&Number(data.quota.remaining||0)<=0);
 var point=document.getElementById('chat-point'),pointNote=document.getElementById('chat-point-note'),journey=chatJourney(data);if(point){var reason=chatPointNote(data);point.hidden=!(journey&&journey.started)||chatInitialPending(data);point.disabled=!!reason||chatHasPending()||!chatConsent(data);if(pointNote){var hint=reason||(point.disabled?'':chatPointSuggestion(data));pointNote.textContent=hint;pointNote.hidden=!hint||point.hidden}}
 var mounted=document.getElementById('view'),shell=mounted&&mounted.chatShell;if(shell)shell.thread.querySelectorAll('button').forEach(function(button){if(button.dataset.chatRetry)button.disabled=!!(state.chatSending||!data.access||!data.access.allowed||data.quota&&Number(data.quota.remaining||0)<=0)});
 // Before the first review the composer stays visible but unusable, with its reason attached.
 var input=form.elements.content,lock=ambreComposerLock(data),lockNote=document.getElementById('chat-locked-note'),notice=document.getElementById('chat-memory-notice'),described=[];
 if(lockNote){lockNote.textContent=lock;lockNote.hidden=!lock}
 if(notice)notice.hidden=!!lock||!(chatConsent(data)&&data.people_observations_enabled);
 if(input){input.disabled=!!lock;input.readOnly=!!lock;if(lock){input.setAttribute('aria-disabled','true');described.push('chat-locked-note')}else input.removeAttribute('aria-disabled');if(notice&&!notice.hidden)described.push(notice.id);if(described.length)input.setAttribute('aria-describedby',described.join(' '));else input.removeAttribute('aria-describedby')}
 var mic=form.querySelector('.chat-microphone');if(mic){if(lock){mic.disabled=true;mic.dataset.ambreLocked='true'}else if(mic.dataset.ambreLocked==='true'){mic.dataset.ambreLocked='';mic.disabled=!state.chatVoice||!(typeof SpeechRecognition!=='undefined'||typeof webkitSpeechRecognition!=='undefined')}}
 var quota=document.getElementById('chat-quota');
 if(quota&&data.quota)quota.textContent=data.quota.used+' / '+data.quota.limit;
 if(quota&&quota.parentElement)quota.parentElement.hidden=chatGuest(data);
}
function disposeChatInterface(){
 if(state.chatVoice){state.chatVoice.dispose();state.chatVoice=null}
 if(state.chatReveals){state.chatReveals.forEach(function(cancel){cancel()});state.chatReveals.clear()}
}
function resetChatSession(){disposeChatInterface();state.chatOutbox=null;state.chatSeenResponses=new Set();state.chatFollowupNotice=''}
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
 var line=el('p','chat-noted');line.append(el('span','chat-noted-label','Noté dans Mon enfance : '));
 valid.forEach(function(note,index){if(index)line.append(document.createTextNode(' · '));line.append(document.createTextNode((note.label?note.label+' — ':'')+'« '+note.text+' »'))});
 var link=el('a','text-link','Voir Mon enfance');link.href='/cartographie/';line.append(document.createTextNode(' '),link);host.append(line);
}
function createChatShell(view,current){
 var data=state.chatData||{},guest=data.access&&data.access.reason==='authentication_required',surface=el('section','conversation');
 var toolbar=el('div','conversation-toolbar'),title=el('h1');
 title.append(el('span','','Ambre'),el('span','conversation-title-divider',' · '),el('span','conversation-role','assistante psychologue IA'));
 title.id='conversation-title';surface.setAttribute('aria-labelledby',title.id);
 var quota=el('div','message-quota'),count=el('strong','','—');count.id='chat-quota';
 quota.setAttribute('role','status');quota.append(count,el('span','',' messages'));
 var point=actionButton('Faire le point',function(){openPointDialog(current)},'secondary chat-point');point.id='chat-point';point.hidden=true;
 var pointNote=el('p','chat-point-note');pointNote.id='chat-point-note';pointNote.hidden=true;point.setAttribute('aria-describedby',pointNote.id);
 toolbar.append(title,point,quota);
 surface.append(toolbar,pointNote);
 var scroll=el('div','conversation-scroll');scroll.id='conversation-scroll';var panel=el('section','journey-panel');panel.id='journey-panel';panel.hidden=true;panel.setAttribute('aria-labelledby','journey-title');scroll.append(panel);var thread=el('section','chat-thread');thread.setAttribute('aria-label','Conversation');thread.setAttribute('role','log');thread.setAttribute('aria-live','off');scroll.append(thread);
 var proposals=el('section');proposals.id='chat-proposals';scroll.append(proposals);surface.append(scroll);
 var gate=el('div');gate.id='chat-gates';surface.append(gate);
 var form=el('form','chat-composer');form.id='chat-form';var row=el('div','composer-row'),label=el('label','visually-hidden','Votre message');label.htmlFor='chat-content';var input=el('textarea');input.id='chat-content';input.name='content';input.maxLength=6000;input.required=true;input.placeholder='Votre message…';input.value=state.chatDraft||'';
 input.addEventListener('input',function(){state.chatDraft=input.value});var mic=el('button','chat-microphone');mic.type='button';mic.setAttribute('aria-label','Dicter un message');mic.setAttribute('aria-describedby','chat-voice-status');mic.title='Dicter un message';
 mic.innerHTML='<svg class="microphone-glyph" aria-hidden="true" focusable="false" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3m-3 0h6"/></svg><svg class="stop-glyph" aria-hidden="true" focusable="false" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>';
 var send=el('button','','Envoyer');send.type='submit';input.addEventListener('keydown',function(event){if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing&&!send.disabled){event.preventDefault();form.requestSubmit()}});
 var feedback=el('p','status');feedback.id='chat-status';feedback.setAttribute('role','status');var voiceStatus=el('p','chat-voice-status');voiceStatus.id='chat-voice-status';voiceStatus.setAttribute('role','status');
 var notice=el('p','chat-memory-notice','Ambre peut retenir des repères sur les proches déjà ajoutés.');notice.id='chat-memory-notice';notice.hidden=true;
 var lockNote=el('p','chat-locked-note');lockNote.id='chat-locked-note';lockNote.hidden=true;
 row.append(label,input,mic,send);form.append(notice,lockNote,row,voiceStatus,feedback);form.addEventListener('submit',async function(event){event.preventDefault();await sendChat(form,current)});surface.append(form);
 var announcement=el('p','visually-hidden');announcement.setAttribute('role','status');surface.append(announcement);
 var next=actionButton('Nouvelle réponse',function(){if(shell.latestResponse){scrollChatTo(shell,shell.latestResponse,false);next.hidden=true}},'chat-new-response');next.hidden=true;surface.append(next);
 var shell={surface:surface,scroll:scroll,thread:thread,proposals:proposals,gate:gate,form:form,input:input,mic:mic,voiceStatus:voiceStatus,feedback:feedback,announcement:announcement,next:next,messages:new Map(),proposalNodes:new Map(),follow:true};
 scroll.addEventListener('scroll',function(){shell.follow=scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<100});view.replaceChildren(surface);view.chatShell=shell;
 if(!guest&&typeof createVoiceInput==='function')state.chatVoice=createVoiceInput({input:input,button:mic,status:voiceStatus,maxLength:6000,isCurrent:function(){return chatIsCurrent(current)},onChange:function(value){if(chatIsCurrent(current))state.chatDraft=value},onState:function(){if(chatIsCurrent(current))updateChatControls()}});else mic.disabled=true;
 return shell;
}
function drawChatGate(shell,data,current){
 var journey=chatJourney(data),signature=JSON.stringify([data.access,journey&&journey.started]);if(shell.gateSignature===signature)return;shell.gateSignature=signature;shell.gate.replaceChildren();
 // Before the first review, the first message of Ambre explains every blocking reason itself.
 if(chatGuest(data)||journey&&!journey.started)return;
 // A state without journey (older server envelope) has no first message: the agreement box stays.
 if(!journey&&!chatConsent(data)){var consentGate=el('div','chat-gate');consentGate.id='chat-consent-gate';var consentLabel=el('label','check'),consentBox=el('input');consentBox.type='checkbox';consentBox.id='chat-consent';consentLabel.append(consentBox,document.createTextNode(AMBRE_CONSENT_TEXT));var consentStatus=el('p','status');consentStatus.setAttribute('role','status');var consentButton=actionButton('Donner mon accord',async function(){if(!consentBox.checked)return;consentButton.disabled=true;try{var result=await api('/api/chat/consent','PUT',{accepted:true});if(!chatIsCurrent(current))return;state.chatData=Object.assign({},state.chatData||{},result);var view=document.getElementById('view');if(view)drawChat(view,current)}catch(err){if(chatIsCurrent(current)){consentStatus.textContent=err.message;consentStatus.className='status error';consentButton.disabled=!consentBox.checked}}});consentButton.id='chat-consent-accept';consentButton.disabled=true;consentBox.addEventListener('change',function(){consentButton.disabled=!consentBox.checked});consentGate.append(consentLabel,consentButton,consentStatus);shell.gate.append(consentGate);return}
 var reason=data.access&&data.access.reason,reasons={quota_exhausted:'Vos 100 messages sont utilisés. Reprise au prochain renouvellement.',provider_missing:'Ambre est momentanément indisponible.',generation_paused:'Les réponses sont momentanément en pause.',provider_cooldown:'Le fournisseur est temporairement indisponible. La disponibilité sera vérifiée automatiquement.',subscription_required:'Un abonnement ou un essai actif est nécessaire.'};
 if(reason&&reason!=='consent_required'&&reason!=='journey_not_started'&&reason!=='prerequisites_required'){var blocked=el('div','chat-gate');blocked.id='chat-access-gate';blocked.append(el('p','',reasons[reason]||'L’envoi est momentanément indisponible.'));if(reason==='subscription_required'){var link=el('a','text-link','Voir mon abonnement');link.href='/mon-compte/';blocked.append(link)}shell.gate.append(blocked)}
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
// Corrections 19/11: a review message is shown as short paragraphs, then its questions in a
// separate block (one per line), even when it was written as one long block (older messages).
// Same rules as the server (review-digest.mjs): a quote is never cut, and a question ends with « ? »
// outside any quote.
function chatSentences(text){var t=String(text),out=[],depth=0,start=0;for(var i=0;i<t.length;i++){var c=t[i];if(c==='«'||c==='“'){depth++;continue}if((c==='»'||c==='”')&&depth>0){depth--;if(!depth){var k=i-1;while(k>start&&/\s/.test(t[k]))k--;var m=i+1;while(m<t.length&&/[\s)]/.test(t[m]))m++;if(/[.!?…]/.test(t[k])&&m<t.length&&/[A-ZÀ-ÖØ-Þ]/.test(t[m])&&m>i+1){out.push(t.slice(start,m));start=m;i=m-1}}continue}if(depth===0&&/[.!?…]/.test(c)){var j=i+1;while(j<t.length&&/[.!?…)]/.test(t[j]))j++;while(j<t.length&&/\s/.test(t[j]))j++;out.push(t.slice(start,j));start=j;i=j-1}}if(start<t.length)out.push(t.slice(start));return out}
function chatIsQuestion(text){return /\?\)?$/.test(String(text).trim())}
function chatReviewParts(text){
 var blocks=String(text||'').replace(/\r\n?/g,'\n').split(/\n{2,}/).map(function(x){return x.trim()}).filter(Boolean),questions=[],paragraphs=[];
 while(blocks.length){
  var lines=blocks[blocks.length-1].split('\n').map(function(x){return x.trim()}).filter(Boolean);
  while(lines.length&&chatIsQuestion(lines[lines.length-1])){var sentences=chatSentences(lines[lines.length-1]),tail=[];while(sentences.length&&chatIsQuestion(sentences[sentences.length-1]))tail.unshift(sentences.pop().trim());if(!tail.length)break;questions=tail.concat(questions);var rest=sentences.join('').trim();lines.pop();if(rest){lines.push(rest);break}}
  if(lines.length){blocks[blocks.length-1]=lines.join('\n');break}
  blocks.pop();
 }
 blocks.forEach(function(block){
  if(block.length<=360||block.indexOf('\n')>=0){paragraphs.push(block);return}
  var current='';chatSentences(block).map(function(x){return x.trim()}).filter(Boolean).forEach(function(part){if(current&&current.length+part.length+1>360){paragraphs.push(current);current=part}else current=current?current+' '+part:part});if(current)paragraphs.push(current);
 });
 return {paragraphs:paragraphs,questions:questions};
}
function drawChatStructured(node,text){
 var parts=chatReviewParts(text);node.replaceChildren();
 parts.paragraphs.forEach(function(paragraph){node.append(el('p','chat-paragraph',paragraph))});
 if(parts.questions.length){var list=el('ul','chat-questions');list.setAttribute('aria-label','Questions d’Ambre');parts.questions.forEach(function(question){list.append(el('li','',question))});node.append(list)}
}
// The links come from the server (message.links); a message with chat_message but without links
// keeps the two default links; an older envelope keeps the single « Mes bilans » link.
function chatReviewLinks(message){
 if(Array.isArray(message.links)){var safe=message.links.filter(function(link){return link&&typeof link.label==='string'&&link.label.trim()&&typeof link.href==='string'&&/^\/[\w\-\/]*$/.test(link.href)});if(safe.length)return safe}
 return typeof message.chat_message==='string'&&message.chat_message.trim()?[{label:'Lire le bilan complet',href:'/mes-bilans/'},{label:'Voir Mes diagnostics',href:'/mes-diagnostics/'}]:null;
}
function chatMessageEntry(shell,message,current,animate){
 var id=String(message.id),entry=shell.messages.get(id),client=message.client_key;
 if(!entry&&client){entry=shell.messages.get('client:'+client);if(entry)shell.messages.delete('client:'+client)}
 if(!entry){var item=el('article','chat-message '+(message.role==='user'?'chat-user':'chat-assistant')),heading=el('h3','',message.role==='user'?'Vous':message.review_id?(message.review_kind==='point'?'Ambre · Point d’étape':'Ambre · Analyse initiale'):'Ambre'),content=message.role==='assistant'&&message.review_id?el('div','chat-content chat-structured'):el('p','chat-content'),meta=el('div','chat-turn-state');item.append(heading,content,meta);entry={item:item,content:content,meta:meta};}
 shell.messages.set(id,entry);entry.item.dataset.chatTurnId=id.replace(/-(user|assistant)$/,'');if(client)entry.item.dataset.clientKey=client;
 // Lot 2/3: a review publishes its short conversation message (chat_message); older envelopes keep content.
 var shown=message.role==='assistant'&&message.review_id&&typeof message.chat_message==='string'&&message.chat_message.trim()?message.chat_message:message.content;
 var structured=message.role==='assistant'&&!!message.review_id;
 if(entry.text!==shown){entry.text=shown;if(structured)drawChatStructured(entry.content,shown);else entry.content.textContent=shown;
  if(message.role==='assistant'){
   if(!state.chatSeenResponses)state.chatSeenResponses=new Set();var unseen=!state.chatSeenResponses.has(id);state.chatSeenResponses.add(id);
   if(animate&&unseen){if(!structured)revealChatResponse(entry,shown,current);shell.latestResponse=entry.item;shell.responseToAnchor=entry.item;shell.announcement.textContent='Nouvelle réponse d’Ambre.'}
   if(Array.isArray(message.cartography_notes))drawChatNotes(entry.item,message.cartography_notes);
   // Corrections 19/11: the links no longer depend on content differing from chat_message (in the
   // new envelopes both are equal, so the two links were replaced by the older single one).
   if(message.review_id){var reviewLinks=chatReviewLinks(message),linkKind=reviewLinks?JSON.stringify(reviewLinks):'legacy';if(entry.reviewLinkKind!==linkKind){if(entry.reviewLink)entry.reviewLink.remove();entry.reviewLinkKind=linkKind;
    if(reviewLinks){var links=el('p','chat-review-links');reviewLinks.forEach(function(link){var anchor=el('a','text-link chat-review-link',link.label);anchor.href=link.href;links.append(anchor)});entry.reviewLink=links}
    else{var bilans=el('a','text-link chat-review-link','Retrouver ce bilan dans Mes bilans');bilans.href='/mes-bilans/';entry.reviewLink=bilans}
    entry.item.append(entry.reviewLink)}}
  }
 }
 if(message.role==='assistant'&&typeof renderChatChangeProposal==='function'){
  var changeIds=Array.isArray(message.change_proposal_ids)?Array.from(new Set(message.change_proposal_ids.filter(function(value){return typeof value==='string'}))):[],changeSignature=JSON.stringify(changeIds);
  if(entry.changeSignature!==changeSignature){entry.changeSignature=changeSignature;if(entry.changeCards)entry.changeCards.remove();if(changeIds.length){entry.changeCards=el('section','chat-change-cards');entry.changeCards.setAttribute('aria-label','Propositions à choisir pour votre plan');entry.item.append(entry.changeCards);changeIds.forEach(function(proposalId){var card=el('div','change-inline');entry.changeCards.append(card);renderChatChangeProposal(card,proposalId,current)})}}
 }
 return entry;
}
function drawChat(view,current,options){
 if(!current())return;var data=state.chatData||{},shell=view.chatShell||createChatShell(view,current),animate=!!(options&&options.animate),messages=(data.messages||[]).map(function(message,index){return Object.assign({},message,{id:message.id||'history-'+index+'-'+message.role})}),box=state.chatOutbox,keys=new Set();
 // The first message of Ambre (code-written) replaces the former welcome text.
 messages=messages.filter(function(message){return !/^ambre-welcome-/.test(message.id)});drawAmbreIntro(shell,data,current);
 if(box&&!messages.some(function(message){return message.role==='user'&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')}))messages.push({id:'client:'+box.client_key,client_key:box.client_key,role:'user',content:box.content,status:box.status});
 messages.forEach(function(message){var entry=chatMessageEntry(shell,message,current,animate);keys.add(String(message.id));entry.meta.replaceChildren();shell.thread.append(entry.item);
  if(message.role==='user'&&['failed','expired','cancelled'].includes(message.status)&&!(box&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')))entry.meta.append(el('p','helper','Réponse non aboutie.'));
  if(message.role==='user'&&box&&(message.client_key===box.client_key||box.request_id&&message.id===box.request_id+'-user')&&['uncertain','failed'].includes(box.status)){
   var retryButton=actionButton('Réessayer ce message',function(){retryChatMessage(box,current)},'text');retryButton.dataset.chatRetry=box.client_key;var errorLine=el('p','status error'+(box.safety?' chat-safety-error':''),box.error||'Ce message n’a pas abouti.');if(box.safety){if(errorLine.style)errorLine.style.whiteSpace='pre-line';if(errorLine.setAttribute)errorLine.setAttribute('role','alert')}entry.meta.append(errorLine,retryButton);
  }
 });
 shell.messages.forEach(function(entry,id){if(!keys.has(id)){entry.item.remove();shell.messages.delete(id)}});
 if(state.chatSending){if(!shell.waiting){var waiting=el('article','chat-message chat-assistant chat-thinking');waiting.id='chat-pending-indicator';waiting.setAttribute('role','status');waiting.append(el('h3','','Ambre'),el('span','','Ambre réfléchit…'));var dots=el('span','thinking-dots');dots.setAttribute('aria-hidden','true');for(var i=0;i<3;i++)dots.append(el('i'));waiting.append(dots);shell.waiting=waiting}shell.thread.append(shell.waiting)}else if(shell.waiting){shell.waiting.remove();shell.waiting=null}
 drawAmbreReviewStatus(shell,data,current);
 var proposals=(data.proposals||[]).filter(function(proposal){return proposal.status==='pending'&&proposal.kind!=='story_request'}),proposalIds=new Set();proposals.forEach(function(proposal){proposalIds.add(proposal.id);var signature=JSON.stringify(proposal),stored=shell.proposalNodes.get(proposal.id);if(!stored){var host=el('div');shell.proposals.append(host);drawChatProposal(host,proposal,current);shell.proposalNodes.set(proposal.id,{host:host,signature:signature})}var mounted=shell.proposalNodes.get(proposal.id),note=mounted.host.querySelector('.chat-capability'),form=mounted.host.querySelector('form'),submit=form&&form.querySelector('button[type=submit]'),unavailable=chatCapabilityMessage(proposal.kind);if(note){note.textContent=unavailable;note.hidden=!unavailable}if(submit&&form.dataset.submitting!=='true')submit.disabled=!!unavailable;});shell.proposalNodes.forEach(function(stored,id){if(!proposalIds.has(id)){stored.host.remove();shell.proposalNodes.delete(id)}});
 if(shell.responseToAnchor){if(shell.follow){scrollChatTo(shell,shell.responseToAnchor,false);shell.follow=false}else shell.next.hidden=false;shell.responseToAnchor=null}
 drawChatGate(shell,data,current);updateChatControls();if(!state.chatSending){if(box&&box.error&&['failed','uncertain'].includes(box.status)){shell.feedback.textContent=box.error;shell.feedback.className='status error'}else if(data.last_error){shell.feedback.textContent=chatFailureMessage(data);shell.feedback.className='status error'}else{shell.feedback.textContent=state.chatFollowupNotice||'';shell.feedback.className='status'}}scheduleChatCooldown(current,data);
 if(options&&options.sent){shell.follow=true;scrollChatTo(shell,shell.waiting||shell.thread.children[shell.thread.children.length-1],false)}
 drawJourneyPanel(current);if(chatReviewRunning(data))pollJourney(current);
 if(!shell.initialPositioned){shell.initialPositioned=true;var last=messages.slice().reverse().find(function(message){return message.role==='assistant'});if(last){var entry=shell.messages.get(String(last.id));scrollChatTo(shell,entry&&entry.item,false)}}
}
async function sendChat(form,current,retry){
 if(!chatIsCurrent(current)||state.chatSending||!retry&&state.chatOutbox&&['sending','pending','uncertain'].includes(state.chatOutbox.status)||state.chatData&&(!state.chatData.access||!state.chatData.access.allowed))return;
 if(state.chatVoice)state.chatVoice.stop('');var input=form.elements.content,content=retry?retry.content:input.value.trim();if(!content)return;
 var sameRetry=retry&&retry.status==='uncertain',key=sameRetry?retry.client_key:newClientId(),box={client_key:key,content:content,status:'sending',started_at:Date.now()};
 state.chatFollowupNotice='';
 state.chatOutbox=box;state.chatSendKey={content:content,key:key};state.chatSending=true;state.chatRequest=null;
 if(!retry){input.value='';state.chatDraft=''}
 var view=document.getElementById('view');if(view)drawChat(view,current,{sent:true});updateChatControls();var submit=form.querySelector('button[type=submit]');if(submit)submit.disabled=true;status('chat-status','');
 try{
  var result=await api('/api/chat/messages','POST',{content:content,idempotency_key:key});if(!chatIsCurrent(current)||state.chatOutbox!==box)return;
  if(result.followups_paused===true){state.chatOutbox=null;state.chatSending=false;state.chatRequest=null;state.chatSendKey=null;state.chatData=Object.assign({},state.chatData||{},result);state.chatFollowupNotice='Les invitations de retour sont suspendues. Ce choix a été enregistré sans appel à Ambre.';if(view)drawChat(view,current);updateChatControls();return}
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
  // F17: a refused send that carries safety resources keeps them visible, line by line.
  box.safety=/\b(?:3114|3919)\b|17 ou le 112|15 ou le 112/.test(err.message||'');var safetyFeedback=document.getElementById('chat-status');if(safetyFeedback&&safetyFeedback.style)safetyFeedback.style.whiteSpace=box.safety?'pre-line':'';
  if(definitive){state.chatSendKey=null;if(!state.chatDraft&&!input.value){input.value=content;state.chatDraft=content;box.restored=true}}
  if(view)drawChat(view,current);updateChatControls();status('chat-status',box.error,true);
  try{var latest=await api('/api/chat');if(!chatIsCurrent(current)||state.chatOutbox!==box)return;state.chatData=latest;reconcileChatTurn(latest);if(view)drawChat(view,current,{animate:true});applyChatAvailability(latest,current);if(state.chatSending)pollChat(state.chatRequest||{id:box.request_id,client_key:key},current,null)}catch(ignore){}
 }
}
function retryChatMessage(box,current){
 if(!chatIsCurrent(current)||state.chatOutbox!==box||state.chatSending)return;var form=document.getElementById('chat-form');if(form)return sendChat(form,current,box);
}
function pollChat(request,current,sentDraft,delay){
 if(!chatIsCurrent(current))return;clearTimeout(state.chatTimer);state.chatTimer=setTimeout(async function(){
  if(!chatIsCurrent(current))return;
  try{
   // While the reply is prepared, a light status poll (a few hundred bytes) replaces
   // the full conversation; the full state is read once the status changes.
   var pending=state.chatRequest||request;
   if(pending&&pending.id){var light=await api('/api/chat/requests/'+encodeURIComponent(pending.id)+'?view=status');if(!chatIsCurrent(current))return;if(light&&light.request&&light.request.id===pending.id&&light.request.status==='running'){var wait=light.request.waiting;if(wait&&wait.retry_after_seconds>0)status('chat-status','Le service est très sollicité : votre message est conservé et sera traité dans environ '+Math.ceil(wait.retry_after_seconds)+' secondes.');if(state.chatSending)pollChat(pending,current,null,wait&&wait.retry_after_seconds>0?Math.min(10000,Math.max(1500,wait.retry_after_seconds*1000)):1500);return}}
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
 },delay||1500);
}
function chatInitialPending(data){var journey=chatJourney(data);return !!(journey&&journey.review&&journey.review.running_id&&journey.review.last_at==null)}
// Plan 19/11, lot 3: Ambre's first message is written by the code (no model, no storage). It replaces
// the former welcome text, the « Avant de commencer » card and the consent gate: one message, one
// agreement, one launch button. Missing fields keep the former behaviour (no journey: no message).
var AMBRE_CONSENT_TEXT='J’accepte que mes messages et les informations que j’ai renseignées soient traités par OpenRouter pour l’analyse et la conversation. Ambre peut retenir des repères concrets sur les proches que j’ai déjà ajoutés et écrire de courtes notes dans Mon enfance.';
var AMBRE_DAILY_LAUNCHES=8;
var AMBRE_FAILURES={'sources_changed':'Une information que je lisais a été supprimée pendant la préparation : par respect de cette suppression, j’ai effacé ce que j’avais préparé. Vous pouvez relancer un bilan, qui partira de vos informations actuelles.','context_changed':'Vos informations ont été modifiées au même moment à plusieurs reprises et je n’ai pas pu en prendre une copie cohérente. Vous pouvez relancer le bilan.','safety_changed':'Une information sur votre sécurité a été enregistrée au moment où je publiais le bilan : je dois en tenir compte avant de le publier. Vous pouvez relancer le bilan.','review_not_approved':'La relecture n’a pas validé mon texte, même après une correction, et je ne publie pas un bilan qui ne serait pas fiable. Vous pouvez relancer le bilan.','invalid_response':'Une réponse a bien été reçue du service de rédaction, mais elle n’était pas conforme aux vérifications du bilan. Vous pouvez relancer le bilan.','invalid_check':'La relecture n’a pas pu être menée jusqu’au bout. Vous pouvez relancer le bilan.','provider_timeout':'Le service de rédaction a mis trop de temps à répondre. Vous pouvez relancer le bilan.','provider_cooldown':'Le service de rédaction est momentanément saturé. Vous pourrez relancer le bilan dans quelques minutes.','provider_error':'Le service de rédaction a rencontré une erreur. Vous pouvez relancer le bilan un peu plus tard.','provider_refused':'Le service de rédaction a refusé la demande. Vous pouvez relancer le bilan un peu plus tard ; si cela se répète, contactez le support.','provider_missing':'La rédaction des bilans est momentanément indisponible. Une relance n’est pas possible pour l’instant.','generation_paused':'La rédaction des bilans est momentanément en pause. Une relance n’est pas possible pour l’instant.','attempts_exhausted':'La préparation a été interrompue trois fois de suite. Vous pouvez relancer le bilan.','lease_lost':'La préparation a été interrompue. Vous pouvez relancer le bilan.','request_expired':'La préparation a dépassé le délai prévu. Vous pouvez relancer le bilan.','review_deferred':'La préparation a été mise en pause entre deux étapes et n’a pas repris à temps. Vous pouvez relancer le bilan.','snapshot_too_large':'Vos informations dépassent la taille qu’un bilan complet peut lire. C’est une limite fixe : relancer ne changerait rien, et je ne produis pas de bilan partiel.','snapshot_analysis_limit':'La synthèse de toutes vos informations dépasse la taille qu’un bilan peut contenir. C’est une limite fixe : relancer ne changerait rien, et je ne produis pas de bilan partiel.','generation_budget':'Le budget de rédaction de la semaine est atteint. Une relance sera possible après son renouvellement.','daily_cap':'Le service a atteint sa limite d’utilisation pour aujourd’hui. Vous pourrez relancer le bilan demain.','request_limit':'Trop de bilans ont été demandés aujourd’hui. Vous pourrez relancer le bilan plus tard.','consent_withdrawn':'Votre accord a été retiré pendant la préparation, et ce que j’avais préparé a été effacé. Pour un nouveau bilan, il faudra de nouveau donner votre accord.','consent_required':'Votre accord est nécessaire pour que je lise vos informations.','access_required':'Votre accès a pris fin pendant la préparation. Une relance sera possible avec un abonnement ou un essai actif.','subscription_required':'Un abonnement ou un essai actif est nécessaire. Une relance sera possible avec un accès actif.','prerequisites_required':'Une des pages nécessaires n’est plus complète ou enregistrée. Complétez-la : vous pourrez ensuite relancer le bilan.','quota_reservation_missing':'Votre bilan du mois n’était plus réservé pour cette préparation. Une relance n’est possible que s’il vous reste un bilan ce mois-ci.','review_quota_exhausted':'Votre bilan du mois est déjà utilisé. Un nouveau bilan sera possible au renouvellement.','nothing_to_analyze':'Je n’ai trouvé aucune information à lire. Complétez vos pages avant de relancer.'};
function ambreFailureReason(code){code=String(code||'');if(AMBRE_FAILURES[code])return AMBRE_FAILURES[code];if(code==='provider_http_429')return AMBRE_FAILURES.provider_cooldown;if(/^provider_http_/.test(code))return AMBRE_FAILURES.provider_error;return 'Une erreur inattendue a interrompu la préparation.'}
function chatGuest(data){return !!(data&&data.access&&data.access.reason==='authentication_required')}
function ambreReview(data){var journey=chatJourney(data);return journey&&journey.review||{}}
function ambreFailure(data){var review=ambreReview(data);return !chatReviewRunning(data)&&review.last_failure&&typeof review.last_failure==='object'?review.last_failure:null}
function ambreReadiness(data){var journey=chatJourney(data);return data&&(data.readiness||journey&&journey.readiness)||null}
function ambreClock(seconds){return new Date(seconds*1000).toLocaleTimeString('fr-FR',{hour:'2-digit',minute:'2-digit'})}
// The five pages of « Ce dont j’ai besoin », in the order a person usually fills them.
function ambreNeeds(readiness){
 var r=readiness||{},n=function(value){return Math.max(0,Number(value)||0)};
 function need(key,label,href,page,done,total,detail){
  var known=!!(page&&typeof page==='object'),complete=known&&page.complete===true,counted=known&&done!=null&&total>0;
  // « Enregistrez » only when every answer is given but the page is not recorded yet.
  var save=known&&!complete&&(counted?done>=total&&page.saved!==true:page.saved===false||page.needs_save===true);
  return {key:key,label:label,href:href,known:known,complete:complete,save:save,done:counted?Math.min(done,total):null,total:counted?total:null,detail:detail};
 }
 // The server may send its own list (readiness.pages); its counters are then used as given.
 var pages=Array.isArray(r.pages)?r.pages:null,DETAILS={account:'Prénom, nom, date de naissance et genre',people:'Au moins trois personnes qui comptent pour vous',timeline:'Au moins trois moments de votre vie',childhood:'Toutes les questions, « Je ne sais pas » compris',symptoms:'Les ${REQUIRED_SYMPTOMS} questions, « Je ne sais pas » compris'},KEYS={account:'identity',timeline:'events'};
 if(pages&&pages.length&&pages.every(function(page){return page&&typeof page.label==='string'&&typeof page.href==='string'&&/^\/[\w-]+\/$/.test(page.href)}))return pages.map(function(page){return need(KEYS[page.id]||page.id,page.label,page.href,page,page.done==null?null:n(page.done),n(page.total),DETAILS[page.id]||'')});
 var identity=r.identity,fields=identity&&Array.isArray(identity.missing_fields)?4-Math.min(4,identity.missing_fields.length):identity&&identity.complete===true?4:null;
 return [
  need('identity','Mon compte','/mon-compte/',identity,fields,4,'Prénom, nom, date de naissance et genre'),
  need('people','Mes proches','/mes-proches/',r.people,r.people&&n(r.people.count),r.people&&n(r.people.required)||3,'Au moins trois personnes qui comptent pour vous'),
  need('events','Ma ligne de vie','/ma-ligne-de-vie/',r.events,r.events&&n(r.events.count),r.events&&n(r.events.required)||3,'Au moins trois moments de votre vie'),
  need('childhood','Mon enfance','/cartographie/',r.childhood,r.childhood&&n(r.childhood.answered),r.childhood&&n(r.childhood.required),'Toutes les questions, « Je ne sais pas » compris'),
  need('symptoms','Mes symptômes','/mes-symptomes/',r.symptoms,r.symptoms&&n(r.symptoms.answered),r.symptoms&&n(r.symptoms.required)||${REQUIRED_SYMPTOMS},'Les ${REQUIRED_SYMPTOMS} questions, « Je ne sais pas » compris')
 ];
}
function ambreNeedItem(need,guest){
 var stateText=guest||!need.known?'':need.complete?'Complet':need.save?'Enregistrez la page':'À compléter';
 var item=el('li','ambre-need'+(need.complete?' is-done':need.save?' is-save':'')),mark=el('span','ambre-need-mark',need.complete?'✓':'');mark.setAttribute('aria-hidden','true');
 var text=el('span','ambre-need-text'),link=el('a','ambre-need-link',need.label);link.href=need.href;text.append(link);
 if(guest||!need.known||need.done==null)text.append(el('span','ambre-need-detail',need.detail));
 else if(need.save)text.append(el('span','ambre-need-detail','Toutes les réponses sont données : il reste à les enregistrer.'));
 item.append(mark,text);
 if(!guest&&need.done!=null){var count=el('span','ambre-need-count',need.done+'/'+need.total);count.setAttribute('aria-hidden','true');item.append(count)}
 if(stateText)item.append(el('span','ambre-need-state',stateText));
 link.setAttribute('aria-label',need.label+(stateText?' : '+stateText.toLowerCase():'')+(!guest&&need.done!=null?', '+need.done+' sur '+need.total:''));
 return item;
}
function ambreLaunchReason(data,needs,checked){
 var journey=chatJourney(data)||{},access=data.access||{},quotas=data.quotas||journey.quotas||{},reviews=quotas.reviews;
 var missing=needs.filter(function(need){return need.known&&!need.complete&&!need.save}).map(function(need){return need.label}),save=needs.filter(function(need){return need.save}).map(function(need){return need.label});
 if(missing.length||save.length)return [missing.length?'Complétez d’abord '+missing.join(', ')+'.':'',save.length?'Enregistrez '+save.join(', ')+'.':''].filter(Boolean).join(' ');
 if(reviews&&Number(reviews.remaining)<=0)return 'Votre bilan du mois est utilisé.'+(reviews.renews_at?' Prochain bilan possible le '+dateLabel(reviews.renews_at)+'.':'');
 var reasons={subscription_required:'Un abonnement ou un essai actif est nécessaire.',provider_missing:'Les bilans sont momentanément indisponibles.',generation_paused:'Les bilans sont momentanément en pause.',provider_cooldown:'Le service de rédaction est momentanément saturé. Réessayez dans quelques minutes.',daily_cap:'Le service a atteint sa limite pour aujourd’hui. Réessayez demain.'};
 if(reasons[access.reason])return reasons[access.reason];
 if(journey.can_start===false)return 'Le bilan ne peut pas être lancé pour le moment.';
 if(!checked)return 'Cochez la case d’accord pour lancer votre bilan.';
 return '';
}
async function ambreRequestReview(kind,current,button,feedback){
 if(!chatIsCurrent(current)||button.dataset.busy==='true')return;button.dataset.busy='true';button.disabled=true;feedback.textContent='Lancement…';feedback.className='status ambre-launch-status';
 var path=kind==='point'?'/api/chat/review':'/api/chat/start';
 try{
  if(!state.journeyKey)state.journeyKey=newClientId();
  var result=await api(path,'POST',kind==='point'?{idempotency_key:state.journeyKey}:{idempotency_key:state.journeyKey,accepted:true});if(!chatIsCurrent(current))return;
  state.journeyKey=null;state.ambreFocusProgress=true;state.chatData=Object.assign({},state.chatData||{},result);
  var view=document.getElementById('view');if(view)drawChat(view,current,{animate:true});
 }catch(err){
  if(!chatIsCurrent(current))return;if(Number(err.status)>=400&&Number(err.status)<500)state.journeyKey=null;
  button.dataset.busy='';button.disabled=false;feedback.textContent=err.message||'Le bilan n’a pas pu être lancé.';feedback.className='status error ambre-launch-status';
  var gateBox=document.getElementById('ambre-consent');if(gateBox&&kind!=='point')button.disabled=!gateBox.checked;
 }
}
function ambreLaunchArea(data,current,needs){
 var journey=chatJourney(data)||{},area=el('div','ambre-launch');
 var box=el('label','check ambre-consent'),checkbox=el('input');checkbox.type='checkbox';checkbox.id='ambre-consent';checkbox.checked=!!state.ambreConsentChecked;box.append(checkbox,document.createTextNode(AMBRE_CONSENT_TEXT));
 var resume=journey.started&&!chatConsent(data),button=actionButton(resume?'Reprendre la conversation':'Lancer mon bilan',async function(){
  if(button.disabled)return;
  if(!resume)return ambreRequestReview('initial',current,button,feedback);
  button.disabled=true;try{await api('/api/chat/consent','PUT',{accepted:true});if(current())renderChat()}catch(err){if(current()){feedback.textContent=err.message;feedback.className='status error ambre-launch-status';button.disabled=!checkbox.checked}}
 });button.id='ambre-start';
 var reason=el('p','helper ambre-start-reason');reason.id='ambre-start-reason';var feedback=el('p','status ambre-launch-status');feedback.id='ambre-launch-status';feedback.setAttribute('role','status');
 function refresh(){var why=resume?(checkbox.checked?'':'Cochez la case d’accord pour reprendre la conversation.'):ambreLaunchReason(data,needs,checkbox.checked);button.disabled=!!why||button.dataset.busy==='true';reason.textContent=why;reason.hidden=!why;if(why)button.setAttribute('aria-describedby',reason.id);else button.removeAttribute('aria-describedby')}
 checkbox.addEventListener('change',function(){state.ambreConsentChecked=checkbox.checked;refresh()});refresh();
 area.append(box,button,reason,feedback);return area;
}
function ambreIntroStatus(data){
 var journey=chatJourney(data)||{},readiness=ambreReadiness(data),failure=ambreFailure(data);
 if(chatReviewRunning(data))return journey.started&&!chatInitialPending(data)?'':'Votre bilan est lancé : je vous montre l’avancement plus bas.';
 if(!journey.started&&failure&&chatConsent(data))return 'Votre bilan n’a pas abouti : je vous explique pourquoi plus bas.';
 if(journey.started&&readiness&&readiness.complete===false)return 'Pour reprendre la conversation, complétez les pages indiquées ci-dessus.';
 if(journey.started&&chatConsent(data))return 'Votre premier bilan est fait : vous le retrouvez dans Mes bilans.';
 return null;
}
function ambreIntroMessage(data,current){
 var guest=chatGuest(data),needs=ambreNeeds(guest?null:ambreReadiness(data));
 var item=el('article','chat-message chat-assistant ambre-intro');item.id='ambre-intro';item.setAttribute('aria-labelledby','ambre-intro-title');
 var from=el('p','ambre-from','Ambre · premier message'),title=el('h2','ambre-intro-title','Je commence par un bilan');title.id='ambre-intro-title';
 var lead=el('div','ambre-intro-lead');
 lead.append(el('p','','Je commence par un bilan complet de ce que vous avez renseigné, pour vous aider à mieux vous comprendre.'),el('p','','Ensuite, nous en discutons ensemble : vous me dites ce qui vous parle et ce que vous voulez creuser.'));
 var needsTitle=el('h3','ambre-needs-title','Ce dont j’ai besoin');needsTitle.id='ambre-needs-title';
 var list=el('ul','ambre-needs');list.setAttribute('aria-labelledby',needsTitle.id);needs.forEach(function(need){list.append(ambreNeedItem(need,guest))});
 item.append(from,title,lead,needsTitle,list);
 if(guest){
  var actions=el('div','actions ambre-auth');[['login','Se connecter',''],['register','Créer un compte','secondary']].forEach(function(entry){var button=el('button',entry[2],entry[1]);button.type='button';button.dataset.auth=entry[0];actions.append(button)});
  item.append(el('p','ambre-guest-note','Connectez-vous pour remplir ces pages : elles restent privées.'),actions);
 }else{
  var statusLine=ambreIntroStatus(data);
  if(statusLine===null)item.append(ambreLaunchArea(data,current,needs));else if(statusLine)item.append(el('p','ambre-intro-status',statusLine));
 }
 item.append(el('p','ambre-intro-note','Je suis une assistante psychologue IA : je propose des hypothèses sur ce qui est probable, à partir de tout ce que vous avez renseigné.'));
 return item;
}
function ambreIntroSignature(data){var journey=chatJourney(data)||{};return JSON.stringify([chatGuest(data),ambreReadiness(data),chatConsent(data),journey.started,journey.can_start,journey.quotas||data.quotas,data.access&&data.access.reason,chatReviewRunning(data),chatInitialPending(data),!!ambreFailure(data)])}
function drawAmbreIntro(shell,data,current){
 if(!chatGuest(data)&&!chatJourney(data)){if(shell.intro){shell.intro.remove();shell.intro=null;shell.introSignature=null}return}
 var signature=ambreIntroSignature(data);
 if(!shell.intro||shell.introSignature!==signature){var fresh=ambreIntroMessage(data,current);if(shell.intro)shell.intro.remove();shell.intro=fresh;shell.introSignature=signature}
 shell.thread.append(shell.intro);
}
function ambreProgress(review){
 var stage=String(review.stage||''),total=Math.max(0,Number(review.blocks_total)||0),done=Math.min(total,Math.max(0,Number(review.blocks_done)||0));
 var phase=stage==='write'||stage==='rewrite'?'write':stage==='check'||stage==='publish'||stage==='done'?'check':'read';
 var value=phase==='read'?Math.round(5+60*(total?done/total:0)):phase==='write'?(stage==='rewrite'?82:72):(stage==='check'?90:97);
 var reading='Lecture de vos informations'+(total?' : '+done+' sur '+total:'');
 var text=phase==='read'?reading:phase==='write'?(stage==='rewrite'?'Rédaction : je reprends quelques passages':'Rédaction de votre bilan'):'Relecture de votre bilan';
 return {phase:phase,value:value,text:text,reading:reading};
}
function ambreEstimate(review,nowMs){
 var estimate=review.estimate||{},p50=Number(estimate.p50_seconds)||0,p90=Number(estimate.p90_seconds)||0;if(!p50&&!p90)return null;if(!p50)p50=p90;if(!p90||p90<p50)p90=p50;
 var low=Math.max(1,Math.round(p50/60)),high=Math.max(low,Math.ceil(p90/60)),started=Number(review.started_at)||0;
 var range=low===high?'environ '+low+' minute'+(low>1?'s':''):'entre '+low+' et '+high+' minutes';
 return {range:range,late:!!started&&nowMs/1000-started>p90,until:started?started+p90:0};
}
// Built once per review and updated in place, so a poll never moves the focus or the reading position.
function ambreProgressMessage(){
 var item=el('article','chat-message chat-assistant ambre-progress');item.id='chat-review-indicator';item.setAttribute('aria-labelledby','ambre-progress-title');
 var title=el('h3','ambre-progress-title','Je prépare votre bilan');title.id='ambre-progress-title';title.tabIndex=-1;
 var bar=el('div','ambre-bar');bar.setAttribute('role','progressbar');bar.setAttribute('aria-labelledby',title.id);bar.setAttribute('aria-valuemin','0');bar.setAttribute('aria-valuemax','100');
 var fill=el('span','ambre-bar-fill');bar.append(fill);
 var steps=el('ol','ambre-steps'),stepItems=[0,1,2].map(function(){var li=el('li','ambre-step');steps.append(li);return li});
 var estimate=el('p','ambre-estimate'),late=el('p','ambre-late','C’est plus long que prévu : je continue, rien n’est perdu.');estimate.hidden=true;late.hidden=true;
 item.append(el('p','ambre-from','Ambre'),title,bar,steps,estimate,late,el('p','ambre-leave','Vous pouvez fermer cette page : votre bilan apparaîtra ici et dans Mes bilans.'));
 return {item:item,title:title,update:function(data){
  var review=ambreReview(data),progress=ambreProgress(review),timing=ambreEstimate(review,Date.now()),at=['read','write','check'].indexOf(progress.phase);
  bar.setAttribute('aria-valuenow',String(progress.value));bar.setAttribute('aria-valuetext',progress.text);if(fill.style)fill.style.width=progress.value+'%';
  [progress.reading,'Rédaction','Relecture'].forEach(function(text,index){var li=stepItems[index];li.textContent=text;li.className='ambre-step'+(index<at?' is-done':index===at?' is-current':'');if(index===at)li.setAttribute('aria-current','step');else li.removeAttribute('aria-current')});
  estimate.hidden=!timing;late.hidden=!(timing&&timing.late);if(timing)estimate.textContent='Durée estimée : '+timing.range+(timing.until&&!timing.late?', prêt au plus tard vers '+ambreClock(timing.until):'')+'.';
  return progress;
 }};
}
function ambreFailureMessage(data,current,failure){
 var journey=chatJourney(data)||{},kind=failure.kind==='point'||failure.kind==='initial'?failure.kind:journey.started?'point':'initial',retryAt=Number(failure.retry_at||0),waiting=retryAt*1000>Date.now();
 var item=el('article','chat-message chat-assistant ambre-failure');item.id='ambre-review-failure';item.setAttribute('aria-labelledby','ambre-failure-title');
 var title=el('h3','ambre-failure-title','Le bilan n’a pas pu aboutir');title.id='ambre-failure-title';title.tabIndex=-1;
 item.append(el('p','ambre-from','Ambre'),title,el('p','',ambreFailureReason(failure.error_code)),el('p','ambre-safe','Rien n’a été perdu : vos informations sont intactes.'+(/quota/.test(String(failure.error_code||''))?'':' Un bilan qui n’aboutit pas n’est pas décompté de votre bilan du mois.')));
 if(failure.retryable===true){
  var feedback=el('p','status ambre-launch-status');feedback.setAttribute('role','status');var retry=actionButton('Relancer',function(){return ambreRequestReview(kind,current,retry,feedback)});retry.id='ambre-retry';retry.disabled=waiting;
  var limit=Number(failure.daily_limit)||AMBRE_DAILY_LAUNCHES,left=failure.daily_remaining!=null?Number(failure.daily_remaining):null;
  var note=el('p','helper ambre-retry-note',(waiting?'Relance possible à partir de '+ambreClock(retryAt)+'. ':'')+'Vous pouvez lancer jusqu’à '+limit+' bilans par 24 heures'+(left!=null&&Number.isFinite(left)?' (il en reste '+Math.max(0,left)+')':'')+'.');note.id='ambre-retry-note';retry.setAttribute('aria-describedby',note.id);
  if(waiting)setTimeout(function(){if(chatIsCurrent(current)&&retry.isConnected&&retry.dataset.busy!=='true'){retry.disabled=false;note.textContent=note.textContent.replace(/^Relance possible à partir de [^.]*\. /,'')}},Math.min(2147483000,retryAt*1000-Date.now()+500));
  item.append(retry,note,feedback);
 }else item.append(el('p','helper',retryAt?'Un nouveau bilan sera possible à partir du '+dateLabel(retryAt)+' à '+ambreClock(retryAt)+'.':'Un nouveau bilan n’est pas possible pour le moment.'));
 return {item:item,title:title};
}
// The wait, the failure and the announcements of the review; one polite live region (the shell's).
function drawAmbreReviewStatus(shell,data,current){
 var running=chatReviewRunning(data),failure=ambreFailure(data),review=ambreReview(data),wasRunning=!!shell.reviewKey,announce='';
 if(running){
  if(shell.reviewFailure){shell.reviewFailure.item.remove();shell.reviewFailure=null}
  if(!shell.reviewWaiting||shell.reviewWaiting.id!==String(review.running_id)){if(shell.reviewWaiting)shell.reviewWaiting.view.item.remove();shell.reviewWaiting={id:String(review.running_id),view:ambreProgressMessage()}}
  var view=shell.reviewWaiting.view,progress=view.update(data),key=review.running_id+':'+progress.phase;shell.thread.append(view.item);
  if(shell.reviewKey!==key){announce=wasRunning&&shell.reviewKey.split(':')[0]===String(review.running_id)?progress.text+'.':'Je prépare votre bilan. '+progress.text+'.';shell.reviewKey=key}
  if(state.ambreFocusProgress){state.ambreFocusProgress=false;if(view.title.focus)view.title.focus()}
 }else{
  if(shell.reviewWaiting){shell.reviewWaiting.view.item.remove();shell.reviewWaiting=null}
  shell.reviewKey=null;
  var failureKey=failure&&!chatGuest(data)?JSON.stringify([failure,chatJourney(data)&&chatJourney(data).started]):null;
  if(shell.reviewFailure&&shell.reviewFailure.key!==failureKey){shell.reviewFailure.item.remove();shell.reviewFailure=null}
  if(failureKey){if(!shell.reviewFailure)shell.reviewFailure={key:failureKey,item:ambreFailureMessage(data,current,failure).item};shell.thread.append(shell.reviewFailure.item);if(wasRunning)announce='Le bilan n’a pas pu aboutir. '+ambreFailureReason(failure.error_code)}
  else if(wasRunning)announce='Votre bilan est prêt.';
 }
 if(announce)shell.announcement.textContent=announce;
}
// The composer stays visible; while it cannot be used it is disabled and says why (disabled, aria-disabled + description).
function ambreComposerLock(data){
 if(chatGuest(data))return 'Connectez-vous pour écrire à Ambre.';
 var journey=chatJourney(data);if(!journey)return '';
 if(chatReviewRunning(data))return journey.started&&!chatInitialPending(data)?'Vous pourrez écrire à Ambre quand votre bilan sera prêt.':'Vous pourrez écrire à Ambre après votre premier bilan.';
 if(!journey.started)return 'Vous pourrez écrire à Ambre après votre premier bilan.';
 var readiness=ambreReadiness(data);if(readiness&&readiness.complete===false)return 'Complétez les pages indiquées dans le premier message d’Ambre pour reprendre la conversation.';
 if(!chatConsent(data))return 'Cochez l’accord dans le premier message d’Ambre pour reprendre la conversation.';
 return '';
}
function journeyPanel(){
 var panel=document.getElementById('journey-panel'),scroll=document.getElementById('conversation-scroll');
 if(!panel&&scroll){panel=el('section','journey-panel');panel.id='journey-panel';panel.hidden=true;panel.setAttribute('aria-labelledby','journey-title');scroll.append(panel)}
 return panel;
}
function closeJourneyPanel(){var panel=document.getElementById('journey-panel');if(!panel)return;panel.hidden=true;panel.dataset.step=''}
function showJourneyPanel(panel,step,nodes){
 var changed=panel.dataset.step!==step;panel.dataset.step=step;panel.replaceChildren.apply(panel,nodes);panel.hidden=false;
 if(changed){var target=panel.querySelector('h2');if(target){target.tabIndex=-1;target.focus()}var scroll=document.getElementById('conversation-scroll');if(scroll)scroll.scrollTop=0}
}
// The panel now only hosts the « Faire le point » confirmation; the first message replaces the rest.
function drawJourneyPanel(current){
 var data=state.chatData||{},journey=chatJourney(data);
 if(state.journeyStep==='point'&&(!journey||!journey.started||chatReviewRunning(data)||!chatIsCurrent(current)))state.journeyStep=null;
 if(state.journeyStep!=='point')closeJourneyPanel();
}
function openPointDialog(current){
 var data=state.chatData||{};if(!chatIsCurrent(current)||chatPointNote(data)||!chatJourney(data)||!chatJourney(data).started)return;
 var dialog=journeyPanel(),title=el('h2','','Faire le point ?');title.id='journey-title';state.journeyStep='point';
 var feedback=el('p','status');feedback.setAttribute('role','status');
 function close(){state.journeyStep=null;closeJourneyPanel();var point=document.getElementById('chat-point');if(point)point.focus()}
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
 showJourneyPanel(dialog,'point',[title,el('p','','Ambre rédige un nouveau bilan à partir de vos informations et de vos échanges depuis le dernier bilan. Cela prend en général une à deux minutes ; il sera aussi conservé dans Mes bilans.'),actions,feedback]);
}
// While a review is prepared the state is read every 15 s, never while the page is hidden.
function pollJourney(current,delay){
 if(!chatIsCurrent(current))return;clearTimeout(state.journeyTimer);state.journeyPollCurrent=current;
 if(!state.journeyVisibility&&document.addEventListener){state.journeyVisibility=true;document.addEventListener('visibilitychange',function(){var pending=state.journeyPollCurrent;if(!document.hidden&&pending&&chatIsCurrent(pending)&&chatReviewRunning(state.chatData)){state.journeyPollPaused=false;pollJourney(pending,0)}})}
 state.journeyTimer=setTimeout(async function(){
  if(!chatIsCurrent(current))return;
  if(document.hidden){state.journeyPollPaused=true;return}
  try{
   var result=await api('/api/chat');if(!chatIsCurrent(current))return;state.journeyPollFailures=0;
   state.chatData=Object.assign({},state.chatData||{},result);reconcileChatTurn(result);
   var view=document.getElementById('view');if(view)drawChat(view,current,{animate:true});updateChatControls();
  }catch(err){
   if(!chatIsCurrent(current))return;state.journeyPollFailures=(state.journeyPollFailures||0)+1;
   if(state.journeyPollFailures<=5)pollJourney(current);else status('chat-status','L’avancement du bilan n’a pas pu être vérifié. Rechargez la page pour le retrouver.',true);
  }
 },delay==null?15000:delay);
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
  var empty=el('div','workspace-empty');empty.append(el('h2','','Aucun bilan pour le moment'),el('p','','Le premier bilan, l’analyse initiale, est écrit par Ambre au lancement de la discussion. Un nouveau bilan s’ajoute chaque fois que vous le demandez avec « Faire le point ».'));
  list.append(empty);return;
 }
 reviews.forEach(function(review){list.append(drawReviewItem(review,current))});
 if(result.next_cursor){var more=actionButton('Afficher les bilans plus anciens',async function(){more.disabled=true;try{var next=await api('/api/reviews?cursor='+encodeURIComponent(result.next_cursor));if(!current())return;more.remove();drawReviewPage(list,next,current)}catch(err){if(current()){status('reviews-status',err.message,true);more.disabled=false}}},'secondary');list.append(more)}
}
// Lot A5-P1: what Ambre keeps from a review (its structured part, as prose) and the person's answer.
function drawReviewDigest(text){
 var box=el('section','review-digest'),heading=el('h3','','Ce qu’Ambre retient de ce bilan');box.append(heading);
 String(text).split(/\n{2,}/).forEach(function(paragraph){if(paragraph.trim())box.append(el('p','',paragraph.trim()))});
 box.append(el('p','helper','Chaque élément s’appuie sur vos propres mots. Ce résumé accompagne les prochains échanges avec Ambre ; votre retour ci-dessous aussi.'));return box;
}
var REVIEW_FEEDBACK=[['resembles','Ce bilan me ressemble'],['not_really','Pas vraiment'],['to_correct','À corriger']];
function drawReviewFeedback(review,current){
 var box=el('section','review-feedback'),legend=el('h3','','Ce bilan vous ressemble-t-il ?'),choices=el('div','review-feedback-choices'),label=el('label','field','Ce que vous voudriez préciser ou corriger'),text=el('textarea'),status=el('p','status'),chosen=review.feedback&&review.feedback.verdict||null;
 choices.setAttribute('role','group');choices.setAttribute('aria-label','Votre réponse');status.setAttribute('role','status');
 text.maxLength=600;text.rows=3;text.value=review.feedback&&review.feedback.text||'';label.append(text);
 var buttons=REVIEW_FEEDBACK.map(function(item){var button=actionButton(item[1],function(){chosen=item[0];buttons.forEach(function(b,i){b.setAttribute('aria-pressed',String(REVIEW_FEEDBACK[i][0]===chosen))});save.disabled=false},'secondary');button.setAttribute('aria-pressed',String(item[0]===chosen));choices.append(button);return button});
 var save=actionButton('Enregistrer mon retour',async function(){
  if(!chosen)return;save.disabled=true;
  try{var result=await api('/api/reviews/'+encodeURIComponent(review.id)+'/feedback','PUT',{verdict:chosen,text:text.value.trim()});if(!current())return;review.feedback=result.feedback;status.className='status';status.textContent='Merci, votre retour est enregistré. Ambre en tiendra compte au prochain point.'}
  catch(err){if(current()){status.className='status error';status.textContent=err.message}}
  finally{save.disabled=false}
 });save.disabled=!chosen;
 var actions=el('div','actions');actions.append(save);
 box.append(legend,choices,label,actions,status);return box;
}
function drawReviewItem(review,current){
 var item=el('article','review-item'),title=el('h2','review-title',reviewTitle(review)),meta=el('p','review-meta'),body=el('div','review-body'),actions=el('div','actions');
 item.dataset.reviewId=review.id;
 meta.textContent=review.status==='running'?'En préparation…':review.status==='failed'?(['snapshot_too_large','snapshot_analysis_limit'].includes(review.error_code)?'Ce dossier dépasse la capacité actuelle d’un bilan complet. Aucun bilan partiel n’a été publié. Vos données restent enregistrées.':'Ce bilan n’a pas abouti. Aucun message n’a été décompté.'):'Rédigé le '+dateLabel(Number(review.completed_at||review.created_at))+(review.feedback&&review.feedback.verdict?' · Votre retour : '+(REVIEW_FEEDBACK.find(function(item){return item[0]===review.feedback.verdict})||['',''])[1]:'');
 if(review.status==='ready'&&review.excerpt)body.append(el('p','review-excerpt',review.excerpt+(Number(review.length)>review.excerpt.length?'…':'')));
 var loaded=false;
 if(review.status==='ready'){
  var read=actionButton('Lire le bilan',async function(){
   if(loaded){body.hidden=!body.hidden;read.textContent=body.hidden?'Lire le bilan':'Replier';read.setAttribute('aria-expanded',String(!body.hidden));return}
   read.disabled=true;
   try{var result=await api('/api/reviews/'+encodeURIComponent(review.id));if(!current())return;loaded=true;body.replaceChildren();String(result.review.content||'').split(/\n{2,}/).forEach(function(paragraph){if(paragraph.trim())body.append(el('p','',paragraph.trim()))});if(result.review.digest_text)body.append(drawReviewDigest(result.review.digest_text));if(result.review.feedback!==undefined)body.append(drawReviewFeedback(result.review,current));body.hidden=false;read.textContent='Replier';read.setAttribute('aria-expanded','true')}
   catch(err){if(current())status('reviews-status',err.message,true)}
   finally{read.disabled=false}
  },'secondary');read.setAttribute('aria-expanded','false');actions.append(read);
  actions.append(actionButton('Informations de cette date',function(){openReviewArchive(review,current)},'secondary'));var hypotheses=el('a','text-link','Voir les hypothèses de ce bilan');hypotheses.href='/mes-diagnostics/?bilan='+encodeURIComponent(review.id);actions.append(hypotheses);
 }
 var remove=actionButton('Supprimer',async function(){
  if(!await confirmDialog('Supprimer « '+reviewTitle(review)+' » ?','Ce bilan sera supprimé définitivement et disparaîtra aussi de la conversation.'+(review.kind==='initial'?' La discussion devra être relancée par une nouvelle analyse.':''),'Supprimer',true))return;
  remove.disabled=true;
  try{await api('/api/reviews/'+encodeURIComponent(review.id),'DELETE');if(!current())return;var list=item.parentElement;item.remove();status('reviews-status','Bilan supprimé.');if(list&&!list.children.length)drawReviewPage(list,{reviews:[]},current)}
  catch(err){if(current()){status('reviews-status',err.message,true);remove.disabled=false}}
 },'text danger');actions.append(remove);
 item.append(title,meta,body,actions);return item;
}
`;
