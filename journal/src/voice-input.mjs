// Browser dictation only. No audio blob, upload, backend key or automatic send.
// This function is self-contained because the app embeds it in its inline script.
export function createVoiceInput(options, platform=globalThis){
 const {input,button,status,onChange=()=>{},onState=()=>{},isCurrent=()=>true,maxLength=6000}=options;
 const Recognition=platform.SpeechRecognition||platform.webkitSpeechRecognition;
 let recognition=null,active=false,busy=false,disposed=false,epoch=0,lastValue=input.value||'',prefix='',suffix='';
 const current=()=>!disposed&&isCurrent()&&input.isConnected!==false;
 function announce(message){if(current()&&status)status.textContent=message;}
 function state(){
  if(!current())return;
  button.setAttribute('aria-pressed',String(active));
  button.setAttribute('aria-label',active||busy?'Arrêter la dictée':'Dicter un message');
  button.title=active||busy?'Arrêter la dictée':'Dicter un message';
  if(button.dataset){button.dataset.voiceActive=String(active);button.dataset.voiceBusy=String(busy);}
  onState({active,busy,available:Boolean(Recognition)});
 }
 function stop(message='Vous pouvez corriger le texte avant de l’envoyer.'){
  epoch++;active=false;busy=false;const previous=recognition;recognition=null;
  if(previous){previous.onresult=previous.onerror=previous.onend=previous.onstart=null;try{previous.abort?previous.abort():previous.stop();}catch{}}
  state();if(message)announce(message);
 }
 function dispose(){
  if(disposed)return;
  stop('');disposed=true;button.removeEventListener('click',toggle);input.removeEventListener('input',manualEdit);
  platform.document?.removeEventListener('visibilitychange',visibility);platform.removeEventListener?.('pagehide',pagehide);
 }
 function manualEdit(){
  if((active||busy)&&input.value!==lastValue){stop('Dictée arrêtée. Vous pouvez modifier votre message.');onChange(input.value);}
 }
 function visibility(){if(platform.document?.visibilityState==='hidden'&&(active||busy))stop('Dictée arrêtée lorsque la page est masquée.');}
 function pagehide(){stop('');}
 async function start(){
  if(!Recognition||!current()||input.disabled||active||busy)return;
  const token=++epoch;busy=true;state();announce('Préparation de la dictée…');
  let local=false;
  // Use an already-installed French pack when available. Do not trigger a
  // large model download or silently promise offline support on every browser.
  if(typeof Recognition.available==='function'){
   try{local=await Recognition.available({langs:['fr-FR'],processLocally:true})==='available';}catch{}
  }
  if(!current()||token!==epoch)return;
  let instance;
  try{instance=new Recognition();}catch{stop('La dictée est indisponible dans ce navigateur.');return;}
  recognition=instance;instance.lang='fr-FR';instance.continuous=true;instance.interimResults=true;instance.maxAlternatives=1;
  if(local&&'processLocally' in instance)instance.processLocally=true;
  const value=String(input.value||''),startAt=Number.isInteger(input.selectionStart)?input.selectionStart:value.length,endAt=Number.isInteger(input.selectionEnd)?input.selectionEnd:startAt;
  prefix=value.slice(0,startAt);suffix=value.slice(endAt);lastValue=value;
  const valid=()=>current()&&token===epoch&&recognition===instance;
  instance.onstart=()=>{if(!valid()){try{instance.abort();}catch{}return;}busy=false;active=true;state();announce(local?'Dictée sur cet appareil. Parlez, puis corrigez le texte.':'Dictée du navigateur. Parlez, puis corrigez le texte.');};
  instance.onresult=event=>{
   if(!valid())return;
   // A keystroke or a send can change the editor before a queued result event.
   // Never overwrite the next draft or a user's manual correction.
   if(input.value!==lastValue){stop('Dictée arrêtée pour conserver votre modification.');return;}
   const parts=[];for(let i=0;i<event.results.length;i++){const words=event.results[i]?.[0]?.transcript;if(typeof words==='string'&&words.trim())parts.push(words.trim());}
   let spoken=parts.join(' ');if(!spoken)return;
   const left=prefix&&!/\s$/.test(prefix)?' ':'',right=suffix&&!/^\s|^[,.!?;:]/.test(suffix)?' ':'';
   const available=Math.max(0,maxLength-prefix.length-suffix.length-left.length-right.length),tooLong=spoken.length>available;
   if(tooLong)spoken=spoken.slice(0,available);
   lastValue=prefix+left+spoken+right+suffix;input.value=lastValue;
   try{input.setSelectionRange(prefix.length+left.length+spoken.length,prefix.length+left.length+spoken.length);}catch{}
   onChange(lastValue);
   if(tooLong)stop('Limite du message atteinte. Relisez le texte avant de l’envoyer.');
  };
  instance.onerror=event=>{
   if(!valid())return;
   const errors={'not-allowed':'Le microphone est bloqué. Autorisez-le dans votre navigateur.','service-not-allowed':'La dictée est bloquée par votre navigateur.','audio-capture':'Aucun microphone accessible. Vérifiez le périphérique utilisé.','no-speech':'Aucune parole détectée. Vous pouvez relancer la dictée.','network':'Le service de dictée du navigateur est indisponible. Votre texte est conservé.','language-not-supported':'La dictée française est indisponible dans ce navigateur.','aborted':'Dictée arrêtée. Votre texte est conservé.'};
   stop(errors[event.error]||'La dictée s’est interrompue. Votre texte est conservé.');
  };
  instance.onend=()=>{if(valid())stop();};
  announce(local?'Autorisez le microphone pour commencer.':'Votre navigateur peut transmettre la voix à son service de dictée. Aucun audio n’est enregistré sur Décode ton esprit.');
  try{instance.start();}catch{if(valid())stop('Impossible de démarrer la dictée. Votre texte est conservé.');}
 }
 function toggle(){if(active||busy)stop();else void start();}
 button.setAttribute('aria-pressed','false');button.setAttribute('aria-label','Dicter un message');
 if(!Recognition){button.disabled=true;button.title='Dictée indisponible dans ce navigateur';announce('Pour dicter ici, utilisez un navigateur compatible ou le micro de votre clavier.');state();return {stop,dispose,isActive:()=>false};}
 button.addEventListener('click',toggle);input.addEventListener('input',manualEdit);
 platform.document?.addEventListener('visibilitychange',visibility);platform.addEventListener?.('pagehide',pagehide);state();
 return {stop,dispose,isActive:()=>active||busy};
}
export const VOICE_INPUT_SCRIPT='var createVoiceInput = '+createVoiceInput.toString()+';';
