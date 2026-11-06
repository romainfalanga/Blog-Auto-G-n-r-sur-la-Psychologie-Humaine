export const PEOPLE_OBSERVATIONS_UI = String.raw`
var PEOPLE_AXES = {
 communication: 'Communication',
 emotions: 'Émotions exprimées',
 besoins: 'Besoins exprimés',
 limites: 'Limites et préférences',
 appuis: 'Ce qui aide',
 tensions: 'Moments de tension'
};

function peopleObservationPreference(){
 var box=el('div','people-observation-preference'),label=el('label','check'),input=el('input');
 input.type='checkbox';
 input.disabled=true;
 label.append(input,document.createTextNode('Mémoriser les repères sur mes proches'));
 var help=el('p','helper','Ambre peut retenir des exemples concrets que vous racontez au sujet d’un proche déjà ajouté. Vous pourrez les relire et les supprimer dans sa fiche.');
 var feedback=el('p','status');feedback.setAttribute('role','status');
 box.append(label,help,feedback);
 var endpoint='/api/people-observations/preferences',enabled=false;
 api(endpoint).then(function(result){
  if(!box.isConnected)return;
  enabled=result.enabled===true;input.checked=enabled;input.disabled=false
 }).catch(function(){
  if(box.isConnected)feedback.textContent='Le réglage est momentanément indisponible.'
 });
 input.addEventListener('change',async function(){
  input.disabled=true;feedback.textContent='';
  try{
   var result=await api(endpoint,'PUT',{enabled:input.checked});
   enabled=result.enabled===true;input.checked=enabled;
   feedback.textContent=enabled?'La mémoire des proches est activée.':'Les nouveaux échanges ne créeront plus de repères.'
  }catch(err){input.checked=enabled;feedback.textContent=err.message}
  finally{input.disabled=false}
 });
 return box
}

function personObservationDetails(person){
 var details=el('details','person-observations'),summary=el('summary','','Portrait et cartographie'),body=el('div','person-observation-content');
 details.append(summary,body);
 var path='/api/people/'+encodeURIComponent(person.id)+'/observations',loading=false,notes=null,enabled=false;
 var feedback=el('p','status');feedback.setAttribute('role','status');
 function draw(){
  body.replaceChildren(el('p','helper','Ces repères viennent de vos échanges avec Ambre et décrivent votre point de vue.'));
  if(!notes.length){
   body.append(el('p','helper',enabled?'Les exemples retenus au fil de la conversation apparaîtront ici.':'Activez la mémoire des proches pour retrouver ici des exemples tirés de vos échanges.'),feedback);
   return
  }
  var groups=Object.entries(PEOPLE_AXES).map(function(entry){
   return {label:entry[1],notes:notes.filter(function(note){return note.axis===entry[0]})}
  }).filter(function(group){return group.notes.length});
  var map=el('ul','person-observation-map');
  groups.forEach(function(group){
   var marker=el('li','',group.label);marker.append(el('strong','',String(group.notes.length)));map.append(marker)
  });
  body.append(el('h4','','Cartographie'),map,el('h4','','Portrait relationnel'));
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
      draw();summary.focus();feedback.textContent='Repère retiré.'
     }catch(err){feedback.textContent=err.message;remove.disabled=false}
    },'text');
    row.append(quote,remove);list.append(row)
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
   enabled=result.enabled===true;
   notes=(result.observations||[]).filter(function(note){return Object.hasOwn(PEOPLE_AXES,note.axis)});
   draw()
  }catch(err){
   if(details.isConnected)body.replaceChildren(el('p','status error',err.message),actionButton('Réessayer',loadNotes,'text'))
  }finally{loading=false}
 }
 details.addEventListener('toggle',function(){if(details.open&&notes===null)loadNotes()});
 return details
}
`;
