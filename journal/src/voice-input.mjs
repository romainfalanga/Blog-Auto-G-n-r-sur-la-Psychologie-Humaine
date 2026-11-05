// The browser's recognizer owns capture. The app only receives editable text.
// This function is self-contained because the app embeds it in its inline script.
export function createVoiceInput(options, platform=globalThis){
 const {input,button,status,onChange=()=>{},onState=()=>{},isCurrent=()=>true,maxLength=6000}=options;
 const Recognition=platform.SpeechRecognition||platform.webkitSpeechRecognition;
 const delay=platform.setTimeout?.bind(platform)||setTimeout;
 const clearDelay=platform.clearTimeout?.bind(platform)||clearTimeout;
 const review='Vous pouvez relire le texte avant de l’envoyer.';
 let recognition=null,phase='idle',disposed=false,phaseTimer=null,localReady=false;
 let original='',lastValue=input.value||'',prefix='',suffix='';
 const current=()=>!disposed&&isCurrent()&&input.isConnected!==false;
 const inProgress=()=>phase!=='idle';
 function announce(message){if(current()&&status)status.textContent=message;}
 function state(){
  if(!current())return;
  const listening=phase==='listening';
  const label=!Recognition?'Dictée indisponible':phase==='finishing'?'Finalisation de la dictée':inProgress()?'Arrêter la dictée':'Dicter un message';
  button.disabled=!Recognition||phase==='finishing';
  button.setAttribute('aria-pressed',String(listening));
  button.setAttribute('aria-label',label);
  button.title=label;
  if(button.dataset)button.dataset.voicePhase=phase;
  onState({active:listening,busy:phase==='starting'||phase==='finishing',available:Boolean(Recognition)});
 }
 function reset(message,abort){
  const previous=recognition;
  recognition=null;
  phase='idle';
  clearDelay(phaseTimer);
  phaseTimer=null;
  if(previous){
   previous.onresult=previous.onerror=previous.onend=previous.onstart=previous.onaudiostart=null;
   if(abort)try{previous.abort?previous.abort():previous.stop();}catch{}
  }
  state();announce(message);
 }
 function stop(message=review){reset(message,true);}
 function finish(){
  if(phase==='starting'){stop('Dictée annulée. Votre texte est conservé.');return;}
  const instance=recognition;
  phase='finishing';state();announce('Finalisation de la dernière phrase…');
  // Keep accepting the final result after stop(). If the service stalls,
  // keep the visible transcript and release the composer.
  phaseTimer=delay(()=>{if(recognition===instance)stop(review);},3000);
  try{instance.stop();}catch{stop(review);}
 }
 function dispose(){
  if(disposed)return;
  stop('');disposed=true;button.removeEventListener('click',toggle);input.removeEventListener('input',manualEdit);
  platform.document?.removeEventListener('visibilitychange',visibility);platform.removeEventListener?.('pagehide',pagehide);
 }
 function manualEdit(){
  if(inProgress()&&input.value!==lastValue){stop('Dictée arrêtée. Votre modification est conservée.');onChange(input.value);}
 }
 function visibility(){if(platform.document?.visibilityState==='hidden'&&inProgress())stop('Dictée arrêtée lorsque la page est masquée.');}
 function pagehide(){if(inProgress())stop('');}
 function start(){
  if(!Recognition||!current()||input.disabled||inProgress())return;
  let instance;
  try{instance=new Recognition();}catch{announce('La dictée est indisponible dans ce navigateur.');return;}
  recognition=instance;phase='starting';
  instance.lang='fr-FR';instance.continuous=true;instance.interimResults=true;instance.maxAlternatives=1;
  const local=localReady&&'processLocally' in instance;
  if(local)instance.processLocally=true;
  original=String(input.value||'');lastValue=original;
  const startAt=Number.isInteger(input.selectionStart)?input.selectionStart:original.length;
  const endAt=Number.isInteger(input.selectionEnd)?input.selectionEnd:startAt;
  prefix=original.slice(0,startAt);suffix=original.slice(endAt);
  const valid=()=>current()&&recognition===instance;
  function listening(){
   if(!valid()||phase!=='starting')return;
   clearDelay(phaseTimer);phaseTimer=null;phase='listening';state();
   announce(local?'Micro actif. Dictée en français sur cet appareil.':'Micro actif. Parlez, puis appuyez sur Arrêter la dictée.');
  }
  instance.onstart=listening;instance.onaudiostart=listening;
  instance.onresult=event=>{
   if(!valid())return;
   if(input.value!==lastValue){stop('Dictée arrêtée pour conserver votre modification.');return;}
   listening();
   // The result list contains all final segments and the current provisional
   // text. Provisional segments can be revised or removed by the recognizer.
   let spoken='';
   for(let i=0;i<event.results.length;i++){
    const transcript=event.results[i]?.[0]?.transcript;
    const words=typeof transcript==='string'?transcript.trim():'';
    if(words)spoken+=(spoken&&!/^[,.…)]/.test(words)?' ':'')+words;
   }
   const left=prefix&&!/\s$/.test(prefix)&&!/^[,.…)]/.test(spoken)?' ':'',right=suffix&&!/^\s|^[,.!?;:]/.test(suffix)?' ':'';
   const available=Math.max(0,maxLength-prefix.length-suffix.length-left.length-right.length),tooLong=spoken.length>available;
   if(tooLong)spoken=spoken.slice(0,available).replace(/[\uD800-\uDBFF]$/u,'');
   const value=spoken?prefix+left+spoken+right+suffix:original;
   if(value!==lastValue){
    lastValue=value;input.value=value;
    try{input.setSelectionRange(spoken?startAt+left.length+spoken.length:startAt,spoken?startAt+left.length+spoken.length:endAt);}catch{}
    onChange(value);
   }
   if(tooLong)stop('Limite du message atteinte. Relisez le texte avant de l’envoyer.');
  };
  instance.onerror=event=>{
   if(!valid())return;
   if(local&&event.error==='language-not-supported')localReady=false;
   const errors={'not-allowed':'Microphone bloqué. Autorisez-le dans les réglages du site ou utilisez le micro de votre clavier.','service-not-allowed':'La dictée est bloquée par votre navigateur. Essayez le micro de votre clavier.','audio-capture':'Aucun microphone accessible. Vérifiez le périphérique utilisé.','no-speech':'Aucune parole détectée. Rapprochez-vous du micro, puis réessayez.','network':'Le service de dictée du navigateur est indisponible. Votre texte est conservé.','language-not-supported':local?'La dictée locale est indisponible. Appuyez à nouveau pour utiliser le service du navigateur.':'La dictée française est indisponible dans ce navigateur.','aborted':'Dictée arrêtée. Votre texte est conservé.'};
   reset(errors[event.error]||'La dictée s’est interrompue. Votre texte est conservé.',false);
  };
  instance.onend=()=>{if(valid())reset(lastValue!==original?review:'Aucun texte reconnu. Appuyez sur le micro pour réessayer.',false);};
  state();announce(local?'Autorisez le microphone pour commencer.':'Autorisez le microphone. Votre navigateur peut utiliser son service vocal.');
  phaseTimer=delay(()=>{if(valid()&&phase==='starting')stop('Le navigateur n’a pas démarré la dictée. Vérifiez l’accès au micro ou utilisez le micro du clavier.');},12000);
  // Start inside the click handler so the browser keeps the user activation.
  try{instance.start();}catch{if(valid())stop('Impossible de démarrer la dictée. Votre texte est conservé.');}
 }
 function toggle(){if(phase==='finishing')return;if(inProgress())finish();else start();}
 if(!Recognition){state();announce('La dictée est indisponible ici. Vous pouvez utiliser le micro de votre clavier.');return {stop,dispose,isActive:()=>false};}
 if(typeof Recognition.available==='function'){
  try{Promise.resolve(Recognition.available({langs:['fr-FR'],processLocally:true})).then(value=>{localReady=value==='available';},()=>{});}catch{}
 }
 button.addEventListener('click',toggle);input.addEventListener('input',manualEdit);
 platform.document?.addEventListener('visibilitychange',visibility);platform.addEventListener?.('pagehide',pagehide);state();
 return {stop,dispose,isActive:inProgress};
}
export const VOICE_INPUT_SCRIPT='var createVoiceInput = '+createVoiceInput.toString()+';';
