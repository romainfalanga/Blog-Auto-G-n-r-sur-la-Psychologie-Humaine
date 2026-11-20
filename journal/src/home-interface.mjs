// The dashboard: the only way into every page now that the menu is gone. It reads like the contents
// page of a personal notebook: Ambre set apart, then a typographic index in two parts (what the person
// fills in, what comes out of it), each line with its figure. No icon. On a phone the whole page fits
// one screen without scrolling. HOME_PAGES comes from data/navigation.json.
export const HOME_INTERFACE_STYLE=String.raw`
.shell:has(> .home-main) { max-width: none; padding: 0; }
body:has(.home-main) .site-footer { display: none; }
.home-main { height: calc(100dvh - var(--header-height) - 1px); }
/* On a phone the dashboard is one screen: the page itself never scrolls. */
@media (max-width: 899px) { html:has(.home-main), body:has(.home-main) { height: 100dvh; min-height: 0; overflow: hidden; overscroll-behavior: none; } .home-head.is-guest h2 { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); } .home-head.is-guest .home-auth { flex: 1; } .home-head.is-guest .home-auth button { flex: 1; } }
.home-main .app-head { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
.home-main #app-messages:empty { display: none; }
.home-main #view { max-width: none; height: 100%; }
#view > .home { max-width: none; }
.home { display: flex; flex-direction: column; gap: var(--s3); height: 100%; padding: var(--s3) var(--gutter) max(var(--s4), env(safe-area-inset-bottom)); }
.home-head { flex: none; display: flex; align-items: center; justify-content: space-between; gap: var(--s3); }
.home-head h2 { margin: 0; font: 400 var(--t-xl)/1.15 var(--font-title); overflow-wrap: anywhere; }
.home-auth { flex: none; display: flex; gap: var(--s2); }
.home-auth button { min-height: 44px; padding: 0 var(--s3); font-size: var(--t-sm); }
.home-tile { color: inherit; text-decoration: none; }
.home-ambre { flex: none; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 2px var(--s3); min-height: 64px; padding: var(--s3) var(--s4); border-radius: var(--r-card); background: var(--accent); color: #fff; }
.home-ambre strong { font: 400 var(--t-xl)/1.15 var(--font-title); }
.home-ambre .home-about { display: none; grid-column: 1 / -1; color: rgb(255 255 255 / .82); line-height: 1.55; }
.home-ambre .home-state { grid-column: 1; color: rgb(255 255 255 / .82); font-size: var(--t-sm); line-height: 1.35; }
.home-ambre .home-open { grid-column: 2; grid-row: 1 / span 2; padding: 9px var(--s3); border: 1px solid rgb(255 255 255 / .45); border-radius: 20px; font: 600 var(--t-sm)/1 var(--font-ui); white-space: nowrap; }
.home-ambre:active { background: color-mix(in srgb, var(--accent) 86%, #000); }
@media (hover: hover) { .home-ambre:hover { background: color-mix(in srgb, var(--accent) 90%, #000); } .home-ambre:hover .home-open { background: #fff; color: var(--accent); } }
.home-index { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.home-group { display: flex; flex-direction: column; min-height: 0; }
.home-group h3 { flex: none; margin: 0; padding: var(--s2) 0 var(--s1); color: var(--muted); font: 600 var(--t-xs)/1.3 var(--font-ui); letter-spacing: .06em; text-transform: uppercase; }
.home-group ul { flex: 1; min-height: 0; display: flex; flex-direction: column; margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--line-strong); }
.home-group li { flex: 1; min-height: 0; display: flex; margin: 0; border-bottom: 1px solid var(--line); }
.home-row { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 0 var(--s3); min-height: 44px; padding: 0 var(--s1); }
.home-row strong { font: 400 var(--t-lg)/1.2 var(--font-title); overflow-wrap: anywhere; }
.home-row .home-about { display: none; grid-column: 1; color: var(--muted); font-size: var(--t-sm); line-height: 1.45; }
.home-row .home-state { grid-column: 2; grid-row: 1; color: var(--muted); font-size: var(--t-sm); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.home-row .home-state:empty { display: none; }
.home-meter { grid-column: 2; grid-row: 2; justify-self: end; display: none; width: 96px; height: 3px; border-radius: 2px; background: var(--line); overflow: hidden; }
.home-meter i { display: block; height: 100%; background: var(--accent); }
.home-row:active { background: var(--accent-wash); }
@media (hover: hover) { .home-row:hover { background: var(--surface); } .home-row:hover strong { color: var(--accent); } }
.home-index .home-group:nth-child(1) { flex: 5; } .home-index .home-group:nth-child(2) { flex: 3; } .home-index .home-group:nth-child(3) { flex: 1; }
/* Short phones: the two headings give their room to the lines. */
@media (max-width: 899px) and (max-height: 640px) { .home { gap: var(--s2); padding-top: var(--s2); } .home-group h3 { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); } .home-group + .home-group ul { border-top: 0; } .home-ambre { min-height: 56px; padding-block: var(--s2); } .home-head h2, .home-ambre strong { font-size: var(--t-lg); } .home-row strong { font-size: var(--t-md); } }
@media (min-width: 900px) {
  .home-main { height: auto; min-height: calc(100dvh - var(--header-height) - 1px); }
  #view > .home, .home { max-width: calc(1040px + 2 * var(--gutter)); }
  .home { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); grid-template-rows: auto 1fr; align-content: start; gap: var(--s5) var(--s8); height: auto; margin: 0 auto; padding: var(--s6) var(--gutter) var(--s6); }
  .home-head { grid-column: 1; flex-direction: column; align-items: flex-start; }
  .home-head h2 { font-size: clamp(2rem, 3.2vw, 2.75rem); }
  .home-auth { margin-top: var(--s2); }
  .home-ambre { grid-column: 1; align-self: start; grid-template-columns: minmax(0, 1fr); gap: var(--s2); padding: var(--s5); }
  .home-ambre strong { font-size: var(--t-2xl); }
  .home-ambre .home-about { display: block; }
  .home-ambre .home-state { color: #fff; font-weight: 600; }
  .home-ambre .home-open { grid-column: 1; grid-row: auto; justify-self: start; margin-top: var(--s3); padding: 12px var(--s4); }
  .home-index { grid-column: 2; grid-row: 1 / span 2; gap: var(--s4); }
  .home-index .home-group { flex: none; }
  .home-group h3 { padding: 0 0 var(--s2); }
  .home-group li { flex: none; }
  .home-row { grid-template-rows: auto auto; padding: 10px var(--s2); }
  .home-row strong { font-size: var(--t-xl); }
  .home-row .home-about { display: block; }
  .home-row .home-state { color: var(--ink); }
  .home-meter { display: block; }
}
`;
export const HOME_INTERFACE_SCRIPT=String.raw`
var HOME_DETAILS={
 '/ambre/':['Votre assistante psychologue IA. Elle lit ce que vous renseignez, prépare vos bilans et en parle avec vous.','ambre'],
 '/ma-ligne-de-vie/':['Les moments qui ont compté, datés et reliés à vos proches.','fill'],
 '/mes-proches/':['Les personnes importantes de votre vie.','fill'],
 '/cartographie/':['Ce que vous avez vécu en grandissant.','fill'],
 '/mes-symptomes/':['Les signes que vous reconnaissez chez vous.','fill'],
 '/mes-sujets/':['Vos points de vue, et la façon dont vos arguments se construisent.','fill'],
 '/ma-cartographie/':['Toutes vos informations reliées en un seul réseau.','read'],
 '/mes-bilans/':['Les bilans préparés par Ambre.','read'],
 '/mes-diagnostics/':['Les hypothèses qui ressortent de vos bilans.','read'],
 '/mon-compte/':['Votre identité, vos données et votre abonnement.','account'],
 '/espace-psychologue/':['Les dossiers que des personnes ont choisi de partager avec vous.','read']
};
var HOME_GROUPS=[['fill','Ce que je renseigne'],['read','Ce que j’en retire'],['account','Mon compte']];
function homeCount(n,one,many){return n+' '+(n>1?many:one)}
function homeStates(extra){
 var me=state.me||{},ready=me.companion_readiness||{},out={};if(!me.user)return out;
 if(ready.people)out['/mes-proches/']=[homeCount(ready.people.count||0,'proche','proches')];
 if(ready.events)out['/ma-ligne-de-vie/']=[homeCount(ready.events.count||0,'moment','moments')];
 if(ready.childhood&&ready.childhood.required)out['/cartographie/']=[(ready.childhood.answered||0)+' / '+ready.childhood.required+' réponses',(ready.childhood.answered||0)/ready.childhood.required];
 if(ready.symptoms&&ready.symptoms.required)out['/mes-symptomes/']=[(ready.symptoms.answered||0)+' / '+ready.symptoms.required+' réponses',(ready.symptoms.answered||0)/ready.symptoms.required];
 if(ready.identity)out['/mon-compte/']=[ready.identity.complete?'Identité complète':'Identité à compléter'];
 if(extra){out['/mes-sujets/']=[homeCount(extra.topics||0,'sujet','sujets')];out['/mes-bilans/']=[homeCount(extra.reviews||0,'bilan','bilans')];out['/ma-cartographie/']=[homeCount(extra.elements||0,'élément relié','éléments reliés')];out['/ambre/']=[extra.review_running?'Bilan en préparation':extra.reviews?'Conversation ouverte':ready.complete?'Prête pour votre premier bilan':'Premier bilan à préparer']}
 return out;
}
function renderHome(){
 var view=document.getElementById('view'),current=renderGuard(view),me=state.me||{},user=me.user;
 if(main.classList)main.classList.add('home-main');
 var home=el('div','home'),head=el('div','home-head'),titles=el('div'),index=el('div','home-index');
 titles.append(el('h2','','Décode ton esprit'));head.append(titles);if(!user)head.classList.add('is-guest');
 if(!user){var auth=el('div','home-auth');[['login','Se connecter',''],['register','Créer un compte','secondary']].forEach(function(entry){var button=el('button',entry[2],entry[1]);button.type='button';button.dataset.auth=entry[0];auth.append(button)});head.append(auth)}
 var pages=HOME_PAGES.slice();if(user&&user.role==='psychologist')pages=[{label:'Espace psychologue',href:'/espace-psychologue/'},{label:'Mon compte',href:'/mon-compte/'}];
 var nodes={},lists={};home.append(head);
 pages.forEach(function(page){var detail=HOME_DETAILS[page.href]||['','read'],tile=el('a','home-tile'),status=el('span','home-state');tile.href=page.href;nodes[page.href]={status:status};
  if(detail[1]==='ambre'){tile.className='home-tile home-ambre';tile.append(el('strong','',page.label),el('span','home-about',detail[0]),status,el('span','home-open','Ouvrir'));home.append(tile);return}
  var meter=el('span','home-meter'),fill=el('i');meter.hidden=true;meter.append(fill);nodes[page.href].meter=meter;nodes[page.href].fill=fill;
  tile.className='home-tile home-row';tile.append(el('strong','',page.label),status,el('span','home-about',detail[0]),meter);
  if(!lists[detail[1]]){var group=el('section','home-group'),label=HOME_GROUPS.find(function(row){return row[0]===detail[1]}),heading=el('h3','',label?label[1]:''),list=el('ul');heading.id='home-group-'+detail[1];list.setAttribute('aria-labelledby',heading.id);group.append(heading,list);index.append(group);lists[detail[1]]=list}
  var item=el('li');item.append(tile);lists[detail[1]].append(item)});
 function fill(extra){var states=homeStates(extra);Object.keys(nodes).forEach(function(href){var value=states[href],node=nodes[href];if(!value)return;node.status.textContent=value[0];if(node.meter&&value.length>1){node.meter.hidden=false;node.fill.style.width=Math.round(Math.max(0,Math.min(1,value[1]))*100)+'%'}})}
 home.append(index);view.replaceChildren(home);fill(null);
 if(user&&user.role!=='psychologist')api('/api/dashboard').then(function(extra){if(current())fill(extra)}).catch(function(){});
}
`;
