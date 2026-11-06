export const PEOPLE_OBSERVATIONS_UI = String.raw`
var PEOPLE_AXES = {
 communication: 'Communication',
 emotions: 'Émotions exprimées',
 besoins: 'Besoins exprimés',
 limites: 'Limites et préférences',
 appuis: 'Ce qui aide',
 tensions: 'Moments de tension'
};

function personObservationDetails(person){
 var details=el('details','person-observations'),summary=el('summary','','Portrait et cartographie'),body=el('div','person-observation-content');
 details.append(summary,body);
 var path='/api/people/'+encodeURIComponent(person.id)+'/observations',loading=false,notes=null;
 var feedback=el('p','status');feedback.setAttribute('role','status');
 function draw(){
  body.replaceChildren();
  if(!notes.length){
   body.append(el('p','helper','Les repères qu’Ambre retient au fil de vos échanges sur cette personne apparaîtront ici.'),feedback);
   return
  }
  body.append(el('p','helper','Ces repères viennent de vos échanges avec Ambre et décrivent votre point de vue.'));
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
