export const PEOPLE_OBSERVATIONS_UI = String.raw`
var PEOPLE_AXES = {
 communication: 'Communication',
 emotions: 'Émotions exprimées',
 besoins: 'Besoins exprimés',
 limites: 'Limites et préférences',
 appuis: 'Ce qui aide',
 tensions: 'Moments de tension'
};

function personObservationDetails(person,options){
 options=options||{};
 function changed(){window.dispatchEvent(new CustomEvent('dte:person-observations-changed',{detail:{personId:person.id}}))}
 var details=el('details','person-observations'),summary=el('summary','','Repères sur cette relation'),body=el('div','person-observation-content');
 details.append(summary,body);
 var path='/api/people/'+encodeURIComponent(person.id)+'/observations',loading=false,notes=null,proposals=[],available=true;
 var feedback=el('p','status');feedback.setAttribute('role','status');
 function draw(){
  body.replaceChildren();
  body.append(el('p','helper','Ambre peut proposer un repère à partir de vos mots. Rien n’est enregistré comme repère sans votre validation individuelle. Vous pouvez corriger ou refuser chaque proposition.'));
  if(!available)body.append(el('p','helper','Les nouvelles propositions de repères ne sont pas encore disponibles.'));
  proposals.filter(function(p){return p.status==='pending'}).forEach(function(p){
   var card=el('section','chat-proposal'),form=el('form'),axisLabel=el('label','','Rubrique'),axis=el('select'),quoteLabel=el('label','','Repère à conserver'),quote=el('textarea');
   axis.name='axis';Object.entries(PEOPLE_AXES).forEach(function(entry){var option=el('option','',entry[1]);option.value=entry[0];axis.append(option)});axis.value=p.payload.axis;
   quote.name='quote';quote.value=p.payload.quote;quote.minLength=20;quote.maxLength=280;quote.required=true;quote.rows=4;
   axisLabel.append(axis);quoteLabel.append(quote);
   card.append(el('h4','','Proposition à vérifier'),el('p','helper','Vos mots dans la conversation :'),el('blockquote','',p.evidence_quote));
   var notice=el('p','status');notice.setAttribute('role','status');
   async function decide(action){
    accept.disabled=true;reject.disabled=true;notice.textContent='';
    try{
     var payload={revision:p.revision};if(action==='accept')payload.payload={axis:axis.value,quote:quote.value};
     await api('/api/chat/proposals/'+encodeURIComponent(p.id)+'/'+action,'POST',payload);
     if(!details.isConnected)return;
     await loadNotes();changed();summary.focus();feedback.textContent=action==='accept'?'Repère validé et enregistré.':'Proposition refusée. Aucun repère enregistré.'
    }catch(err){notice.textContent=err.message;accept.disabled=false;reject.disabled=false}
   }
   var accept=actionButton('Valider ce repère',function(){return decide('accept')},'primary'),reject=actionButton('Refuser',function(){return decide('reject')},'text');
   accept.disabled=!available;form.addEventListener('submit',function(event){event.preventDefault()});
   form.append(axisLabel,quoteLabel,accept,reject,notice);card.append(form);body.append(card)
  });
  var refused=proposals.filter(function(p){return p.status==='rejected'});
  if(refused.length){var history=el('details'),list=el('ul');history.append(el('summary','','Propositions refusées'));refused.forEach(function(p){list.append(el('li','',p.payload.quote))});history.append(list);body.append(history)}
  if(!notes.length){body.append(el('p','helper','Aucun repère enregistré pour cette personne.'),feedback);return}
  body.append(el('p','helper','Ces repères décrivent votre point de vue, sans définir la personnalité de votre proche.'));
  var groups=Object.entries(PEOPLE_AXES).map(function(entry){
   return {label:entry[1],notes:notes.filter(function(note){return note.axis===entry[0]})}
  }).filter(function(group){return group.notes.length});
  var map=el('ul','person-observation-map');
  groups.forEach(function(group){
   var marker=el('li','',group.label);marker.append(el('strong','',String(group.notes.length)));map.append(marker)
  });
  body.append(el('h4','','Repères enregistrés'),map);
  groups.forEach(function(group){
   var list=el('ul','person-observation-notes');
   group.notes.forEach(function(note){
    var row=el('li'),quote=el('blockquote','',note.quote);
    var remove=actionButton('Retirer ce repère',async function(){
     remove.disabled=true;feedback.textContent='';
     try{
      await api(path+'/'+encodeURIComponent(note.id),'DELETE');
      if(!details.isConnected)return;
      notes=notes.filter(function(entry){return entry.id!==note.id});
      draw();changed();summary.focus();feedback.textContent='Repère retiré.'
     }catch(err){feedback.textContent=err.message;remove.disabled=false}
    },'text');
    row.append(quote,el('p','helper',note.origin==='accepted'?'Validé par vous':'Ancien repère automatique, non validé individuellement'),remove);list.append(row)
   });
   body.append(el('h5','',group.label),list)
  });
  body.append(feedback)
 }
 async function loadNotes(){
  if(loading)return;
  loading=true;body.replaceChildren(el('p','helper','Chargement des repères…'));
  try{
   var result=await api(path);
   if(!details.isConnected)return;
   available=result.available!==false;proposals=result.proposals||[];
   notes=(result.observations||[]).filter(function(note){return Object.hasOwn(PEOPLE_AXES,note.axis)&&(!options.proposalsOnly||note.origin!=='accepted')});
   draw()
  }catch(err){
   if(details.isConnected)body.replaceChildren(el('p','status error',err.message),actionButton('Réessayer',loadNotes,'text'))
  }finally{loading=false}
 }
 details.addEventListener('toggle',function(){if(details.open&&notes===null)return loadNotes()});
 return details
}
`;
