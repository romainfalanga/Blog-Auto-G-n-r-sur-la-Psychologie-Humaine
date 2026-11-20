// « Mes sujets »: the list of topics and, for one topic, the map of its positions with what supports
// and what contradicts each of them. Uses the shared helpers of ui.mjs (el, api, actionButton, go…).
export const TOPICS_INTERFACE_STYLE=String.raw`
.topic-intro { max-width: 62ch; margin: 0 0 var(--s5); color: var(--muted); line-height: 1.6; }
.topic-new { display: flex; flex-wrap: wrap; align-items: flex-end; gap: var(--s3); margin: 0 0 var(--s5); }
.topic-new .field { flex: 1 1 260px; margin: 0; }
.topic-new button { min-height: 48px; }
.topic-list { display: grid; gap: var(--s3); margin: 0; padding: 0; list-style: none; }
.topic-card { display: flex; flex-direction: column; gap: 2px; min-height: 64px; padding: var(--s4); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); color: var(--ink); text-decoration: none; }
.topic-card strong { font: 600 var(--t-md)/1.35 var(--font-ui); }
.topic-card span { color: var(--muted); font-size: var(--t-sm); }
@media (hover: hover) { .topic-card:hover { border-color: var(--accent); } }
.topic-page { --topic-for: var(--accent); --topic-against: #B4530A; }
.topic-all { display: inline-flex; align-items: center; min-height: 44px; }
.topic-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2) var(--s4); margin: 0 0 var(--s4); }
.topic-head h2 { margin: 0; font-size: var(--t-xl); overflow-wrap: anywhere; }
.topic-head .actions { margin: 0; }
.topic-overview { margin: 0 0 var(--s5); padding: var(--s4); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); }
.topic-overview h3 { margin: 0 0 var(--s2); font: 600 var(--t-md)/1.35 var(--font-ui); }
.topic-overview p { margin: 0 0 var(--s2); color: var(--muted); font-size: var(--t-sm); }
.topic-gaps { margin: 0; padding: 0; list-style: none; }
.topic-gaps li { border-top: 1px solid var(--line); }
.topic-gaps button { display: block; width: 100%; min-height: 44px; padding: var(--s2) 0; border: 0; background: none; color: var(--ink); font: inherit; font-size: var(--t-sm); line-height: 1.45; text-align: left; cursor: pointer; }
.topic-gaps button strong { display: block; font-weight: 600; }
.topic-gaps button span { color: var(--muted); }
.topic-position { margin: 0 0 var(--s5); padding: var(--s4); border: 1px solid var(--line-strong); border-radius: var(--r-card); background: var(--surface); scroll-margin-top: calc(var(--header-height, 64px) + var(--s4)); }
.topic-label { margin: 0 0 var(--s1); color: var(--muted); font: 600 var(--t-xs)/1.4 var(--font-ui); letter-spacing: .02em; text-transform: uppercase; }
.topic-text { margin: 0; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
.topic-position > .topic-node-body > .topic-text { font-size: var(--t-md); font-weight: 600; }
.topic-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0 var(--s3); margin-top: var(--s1); }
.topic-chip { padding: 2px var(--s2); border-radius: var(--r-control); background: var(--accent-wash); color: var(--ink); font-size: var(--t-xs); line-height: 1.5; }
.topic-meta button, .topic-add button { min-height: 44px; padding: 0; font-size: var(--t-sm); }
.topic-sides { display: grid; gap: var(--s4); margin-top: var(--s4); }
@media (min-width: 760px) { .topic-sides { grid-template-columns: 1fr 1fr; } }
.topic-side > h4 { margin: 0 0 var(--s2); padding-bottom: var(--s1); border-bottom: 2px solid var(--topic-for); font: 600 var(--t-sm)/1.4 var(--font-ui); }
.topic-side.is-against > h4 { border-bottom-color: var(--topic-against); }
.topic-side-empty { margin: 0 0 var(--s2); color: var(--muted); font-size: var(--t-sm); }
.topic-node { margin: 0 0 var(--s3); padding: var(--s2) 0 var(--s2) var(--s3); border-left: 3px solid var(--topic-for); scroll-margin-top: calc(var(--header-height, 64px) + var(--s4)); }
.topic-node.is-against { border-left-color: var(--topic-against); }
.topic-node .topic-label { text-transform: none; letter-spacing: 0; color: var(--topic-for); }
.topic-node.is-against > .topic-node-body > .topic-label { color: var(--topic-against); }
.topic-children { margin: var(--s2) 0 0 var(--s2); }
.topic-children .topic-node:last-child { margin-bottom: 0; }
.topic-add { display: flex; flex-wrap: wrap; gap: 0 var(--s4); }
.topic-form { margin: var(--s2) 0; padding: var(--s3); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--bg); }
.topic-form .field { margin: 0 0 var(--s3); }
.topic-form textarea { width: 100%; min-height: 96px; }
.topic-form .actions { margin: 0; }
.topic-flash { animation: topic-flash 1.6s ease; }
@keyframes topic-flash { from { background: var(--accent-wash); } to { background: transparent; } }
@media (prefers-reduced-motion: reduce) { .topic-flash { animation: none; } }
`;
export const TOPICS_INTERFACE_SCRIPT=String.raw`
var TOPIC_BASES=[['','Je ne précise pas'],['fact','Un fait vérifié'],['experience','Mon expérience'],['value','Une valeur qui compte pour moi'],['intuition','Une intuition'],['hearsay','Quelque chose que j’ai entendu ou lu']];
var TOPIC_CONFIDENCE=[['','Je ne précise pas'],['1','Très peu de certitude'],['2','Peu de certitude'],['3','Certitude moyenne'],['4','Bonne certitude'],['5','Certitude très forte']];
function topicChoice(list,value){var found=list.find(function(row){return row[0]===String(value==null?'':value)});return found&&found[0]?found[1]:''}
function topicCount(n,one,many){return n+' '+(n>1?many:one)}
function topicError(host,error){host.replaceChildren(el('div','notice warn',error&&error.message||'Cette page n’a pas pu être chargée.'),actionButton('Réessayer',function(){renderApp()},'secondary'))}
async function renderTopics(){
 var view=document.getElementById('view'),current=renderGuard(view),data;
 try{data=await api('/api/topics')}catch(error){if(current())topicError(view,error);return}
 if(!current())return;view.replaceChildren();
 view.append(el('p','topic-intro','Écrivez ce que vous pensez d’un sujet, puis ce qui vous amène à le penser et ce qui pourrait vous contredire. Voir vos arguments posés aide à prendre du recul et à repérer ce qui mérite d’être approfondi.'));
 var form=el('form','topic-new'),field=el('label','field','Nouveau sujet'),input=el('input'),add=el('button','','Ajouter'),message=el('p','status');
 input.name='title';input.maxLength=120;input.required=true;input.placeholder='Par exemple : l’écologie, le travail, l’éducation…';input.autocomplete='off';field.append(input);add.type='submit';message.setAttribute('role','status');form.append(field,add);
 form.addEventListener('submit',async function(event){event.preventDefault();busy(form,true);message.className='status';message.textContent='';
  try{var result=await api('/api/topics','POST',{title:input.value});go('topic','',result.topic.id)}catch(error){message.textContent=error.message;message.className='status error'}finally{busy(form,false)}});
 view.append(form,message);
 if(!data.topics.length){view.append(el('p','helper','Vous n’avez encore ajouté aucun sujet.'));return}
 var list=el('ul','topic-list');
 data.topics.forEach(function(topic){var item=el('li'),link=el('a','topic-card');link.href='/mes-sujets/'+encodeURIComponent(topic.id)+'/';link.append(el('strong','',topic.title),el('span','',topic.positions?topicCount(topic.positions,'position','positions')+' · '+topicCount(topic.arguments,'argument','arguments'):'Rien d’écrit pour le moment'));item.append(link);list.append(item)});
 view.append(list);
}
// One form for every element: a position (with a degree of certainty) or an argument (with its basis).
function topicForm(options){
 var form=el('form','topic-form'),field=el('label','field',options.label),area=el('textarea'),actions=el('div','actions'),save=el('button','','Enregistrer'),message=el('p','status'),select=null;
 area.name='content';area.maxLength=1500;area.required=true;area.value=options.value||'';if(options.placeholder)area.placeholder=options.placeholder;field.append(area);form.append(field);
 if(options.choices){var choiceField=el('label','field',options.choiceLabel);select=el('select');options.choices.forEach(function(row){var option=el('option','',row[1]);option.value=row[0];select.append(option)});select.value=String(options.choice==null?'':options.choice);choiceField.append(select);form.append(choiceField)}
 save.type='submit';message.setAttribute('role','status');actions.append(save,actionButton('Annuler',function(){options.cancel()},'secondary'));form.append(actions,message);
 form.addEventListener('submit',async function(event){event.preventDefault();busy(form,true);message.className='status';message.textContent='Enregistrement…';
  try{await options.save(area.value,select?select.value:'')}catch(error){message.textContent=error.message;message.className='status error';busy(form,false)}});
 setTimeout(function(){if(form.isConnected)area.focus()},0);
 return form;
}
// What still deserves attention, read from the tree alone: no model, nothing leaves the page.
function topicGaps(nodes,children){
 var gaps=[];
 nodes.forEach(function(node){
  var below=children.get(node.id)||[],supports=below.filter(function(row){return row.kind==='support'}),objections=below.filter(function(row){return row.kind==='objection'});
  if(node.kind==='position'){
   if(!supports.length)gaps.push({node:node,title:'Une position sans raison écrite',help:'Qu’est-ce qui vous amène à le penser ?'});
   else if(!objections.length)gaps.push({node:node,title:'Une position sans objection notée',help:'Que répondrait quelqu’un qui n’est pas d’accord ?'});
  }else if(node.kind==='objection'){
   var parent=nodes.find(function(row){return row.id===node.parent_id});
   if(parent&&parent.kind!=='objection'&&!objections.length)gaps.push({node:node,title:'Une objection sans réponse',help:'Que lui répondez-vous, ou change-t-elle votre avis ?'});
  }else if(node.basis==='hearsay'&&!supports.length)gaps.push({node:node,title:'Une raison entendue ou lue, non vérifiée',help:'Sur quoi pourriez-vous l’appuyer ?'});
 });
 return gaps;
}
async function renderTopic(){
 var view=document.getElementById('view'),current=renderGuard(view),id=state.topicId,data;
 try{data=await api('/api/topics/'+encodeURIComponent(id))}catch(error){if(current()){topicError(view,error);var back=el('a','text-link topic-all','Tous mes sujets');back.href='/mes-sujets/';view.prepend(back)}return}
 if(!current())return;
 var base='/api/topics/'+encodeURIComponent(id),openForm=null;
 function reload(){return api(base).then(function(next){if(!current())return;data=next;openForm=null;draw()})}
 function draw(focusId){
  var nodes=data.nodes,children=new Map();nodes.forEach(function(node){var key=node.parent_id||'';if(!children.has(key))children.set(key,[]);children.get(key).push(node)});
  var page=el('div','topic-page'),all=el('a','text-link topic-all','Tous mes sujets');all.href='/mes-sujets/';
  var head=el('div','topic-head'),title=el('h2','',data.topic.title),headActions=el('div','actions');
  headActions.append(actionButton('Renommer',function(){openForm={rename:true};draw()},'text'),actionButton('Supprimer ce sujet',async function(){if(!await confirmDialog('Supprimer ce sujet ?','Ses positions et tous leurs arguments seront supprimés. Cette action est définitive.','Supprimer',true))return;try{await api(base,'DELETE');go('topics','')}catch(error){status('topic-status',error.message,true)}},'text'));
  head.append(title,headActions);page.append(all,head);
  var feedback=el('p','status');feedback.id='topic-status';feedback.setAttribute('role','status');page.append(feedback);
  if(openForm&&openForm.rename){var rename=el('form','topic-form'),renameField=el('label','field','Nom du sujet'),renameInput=el('input'),renameActions=el('div','actions'),renameSave=el('button','','Enregistrer'),renameMessage=el('p','status');renameInput.maxLength=120;renameInput.required=true;renameInput.value=data.topic.title;renameField.append(renameInput);renameSave.type='submit';renameActions.append(renameSave,actionButton('Annuler',function(){openForm=null;draw()},'secondary'));rename.append(renameField,renameActions,renameMessage);
   rename.addEventListener('submit',async function(event){event.preventDefault();busy(rename,true);try{await api(base,'PUT',{title:renameInput.value});await reload()}catch(error){renameMessage.textContent=error.message;renameMessage.className='status error';busy(rename,false)}});page.append(rename);setTimeout(function(){if(renameInput.isConnected)renameInput.focus()},0)}
  var positions=children.get('')||[];
  if(positions.length){
   var overview=el('section','topic-overview'),gaps=topicGaps(nodes,children),args=nodes.length-positions.length;
   overview.append(el('h3','','Vue d’ensemble'),el('p','',topicCount(positions.length,'position','positions')+' · '+topicCount(args,'argument','arguments')));
   if(gaps.length){overview.append(el('p','','À approfondir :'));var gapList=el('ul','topic-gaps');gaps.forEach(function(gap){var item=el('li'),button=el('button');button.type='button';var text=String(gap.node.content).replace(/\s+/g,' ');button.append(el('strong','',gap.title),el('span','','« '+(text.length>90?text.slice(0,90)+'…':text)+' » — '+gap.help));button.addEventListener('click',function(){var target=document.getElementById('topic-node-'+gap.node.id);if(!target)return;target.scrollIntoView({behavior:'smooth',block:'center'});target.classList.remove('topic-flash');void target.offsetWidth;target.classList.add('topic-flash')});item.append(button);gapList.append(item)});overview.append(gapList)}
   else if(args)overview.append(el('p','','Chaque position a ses raisons, et chaque objection a une réponse.'));
   page.append(overview);
  }
  function editForm(node){return topicForm({label:node.kind==='position'?'Ce que je pense':'Votre texte',value:node.content,choices:node.kind==='position'?TOPIC_CONFIDENCE:TOPIC_BASES,choiceLabel:node.kind==='position'?'Mon degré de certitude':'Sur quoi cela repose',choice:node.kind==='position'?node.confidence:node.basis,cancel:function(){openForm=null;draw()},save:async function(text,choice){var payload={content:text};if(node.kind==='position')payload.confidence=choice?Number(choice):null;else payload.basis=choice||null;await api(base+'/nodes/'+encodeURIComponent(node.id),'PUT',payload);await reload()}})}
  function addForm(parent,kind,label,placeholder){return topicForm({label:label,placeholder:placeholder,choices:TOPIC_BASES,choiceLabel:'Sur quoi cela repose',cancel:function(){openForm=null;draw()},save:async function(text,choice){await api(base+'/nodes','POST',{parent_id:parent.id,kind:kind,content:text,basis:choice||null});await reload()}})}
  function removeNode(node){return async function(){var below=(children.get(node.id)||[]).length;if(!await confirmDialog(node.kind==='position'?'Supprimer cette position ?':'Supprimer cet élément ?',below?'Ce qui lui est rattaché sera supprimé aussi. Cette action est définitive.':'Cette action est définitive.','Supprimer',true))return;try{await api(base+'/nodes/'+encodeURIComponent(node.id),'DELETE');await reload()}catch(error){status('topic-status',error.message,true)}}}
  function nodeLabel(node,parent){if(parent.kind==='position')return '';if(parent.kind==='objection')return node.kind==='objection'?'Ma réponse':'Ce qui lui donne du poids';return node.kind==='support'?'Ce qui l’appuie':'Ce qui la fragilise'}
  function drawNode(node,parent,depth){
   var item=el('div','topic-node'+(node.kind==='objection'?' is-against':'')),body=el('div','topic-node-body'),label=nodeLabel(node,parent);item.id='topic-node-'+node.id;
   if(openForm&&openForm.edit===node.id){item.append(editForm(node));return item}
   if(label)body.append(el('p','topic-label',label));
   body.append(el('p','topic-text',node.content));
   var meta=el('div','topic-meta'),basis=topicChoice(TOPIC_BASES,node.basis);if(basis)meta.append(el('span','topic-chip',basis));
   meta.append(actionButton('Modifier',function(){openForm={edit:node.id};draw()},'text'),actionButton('Supprimer',removeNode(node),'text'));body.append(meta);item.append(body);
   var below=children.get(node.id)||[];if(below.length){var nest=el('div','topic-children');below.forEach(function(child){nest.append(drawNode(child,node,depth+1))});item.append(nest)}
   if(openForm&&openForm.parent===node.id)item.append(node.kind==='objection'?(openForm.kind==='objection'?addForm(node,'objection','Ma réponse à cette objection','Ce que vous lui répondez.'):addForm(node,'support','Ce qui donne du poids à cette objection','Ce qui la rend sérieuse à vos yeux.')):(openForm.kind==='support'?addForm(node,'support','Ce qui appuie cette raison','Un fait, un exemple, une expérience.'):addForm(node,'objection','Ce qui fragilise cette raison','Une limite, un contre-exemple.')));
   else if(depth<4){var add=el('div','topic-add');
    if(node.kind==='objection')add.append(actionButton('Y répondre',function(){openForm={parent:node.id,kind:'objection'};draw()},'text'),actionButton('Ce qui lui donne du poids',function(){openForm={parent:node.id,kind:'support'};draw()},'text'));
    else add.append(actionButton('Ce qui l’appuie',function(){openForm={parent:node.id,kind:'support'};draw()},'text'),actionButton('Ce qui la fragilise',function(){openForm={parent:node.id,kind:'objection'};draw()},'text'));
    item.append(add)}
   return item;
  }
  positions.forEach(function(position){
   var card=el('article','topic-position');card.id='topic-node-'+position.id;
   if(openForm&&openForm.edit===position.id)card.append(editForm(position));
   else{var body=el('div','topic-node-body');body.append(el('p','topic-label','Ce que je pense'),el('p','topic-text',position.content));var meta=el('div','topic-meta'),certainty=topicChoice(TOPIC_CONFIDENCE,position.confidence);if(certainty)meta.append(el('span','topic-chip',certainty));meta.append(actionButton('Modifier',function(){openForm={edit:position.id};draw()},'text'),actionButton('Supprimer',removeNode(position),'text'));body.append(meta);card.append(body)}
   var sides=el('div','topic-sides'),below=children.get(position.id)||[];
   [['support','Ce qui m’amène à le penser','Ajouter une raison','Aucune raison écrite pour le moment.','Une raison','Pourquoi vous le pensez : un fait, une expérience, une valeur.'],['objection','Ce qui pourrait me contredire','Ajouter une objection','Aucune objection notée pour le moment.','Une objection','Ce que dirait quelqu’un qui n’est pas d’accord, ou ce qui vous fait douter.']].forEach(function(side){
    var column=el('section','topic-side'+(side[0]==='objection'?' is-against':'')),rows=below.filter(function(row){return row.kind===side[0]});column.append(el('h4','',side[1]));
    if(!rows.length)column.append(el('p','topic-side-empty',side[3]));rows.forEach(function(row){column.append(drawNode(row,position,1))});
    if(openForm&&openForm.parent===position.id&&openForm.kind===side[0])column.append(addForm(position,side[0],side[4],side[5]));
    else{var add=el('div','topic-add');add.append(actionButton(side[2],function(){openForm={parent:position.id,kind:side[0]};draw()},'text'));column.append(add)}
    sides.append(column)});
   card.append(sides);page.append(card);
  });
  if(openForm&&openForm.position||!positions.length)page.append(topicForm({label:positions.length?'Une autre position sur ce sujet':'Ce que je pense de ce sujet',placeholder:'Une idée à la fois, en une ou deux phrases. Par exemple : « Je pense que… »',choices:TOPIC_CONFIDENCE,choiceLabel:'Mon degré de certitude',cancel:function(){if(positions.length){openForm=null;draw()}else go('topics','')},save:async function(text,choice){await api(base+'/nodes','POST',{content:text,confidence:choice?Number(choice):null});await reload()}}));
  else page.append(actionButton('Ajouter une position',function(){openForm={position:true};draw()},'secondary'));
  view.replaceChildren(page);
 }
 draw();
}
`;
