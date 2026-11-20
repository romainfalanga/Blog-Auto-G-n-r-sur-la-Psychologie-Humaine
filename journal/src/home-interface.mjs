// The dashboard: the only way into every page now that the menu is gone. It reads like the contents
// page of a personal notebook: Ambre set apart, then a typographic index in two parts (what the person
// fills in, what comes out of it), each line with its figure. No icon. On a phone the whole page fits
// one screen without scrolling. HOME_PAGES comes from data/navigation.json.
export const HOME_INTERFACE_STYLE=String.raw`
.shell:has(> .home-main) { max-width: none; padding: 0; }
body:has(.home-main) .site-footer { display: none; }
.home-main { height: calc(100dvh - var(--header-height) - 1px); }
/* On a phone the dashboard is one screen: the page itself never scrolls. */
@media (max-width: 899px) { html:has(.home-main), body:has(.home-main) { height: 100dvh; min-height: 0; overflow: hidden; overscroll-behavior: none; } }
.home-main .app-head { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
.home-main #app-messages:empty { display: none; }
.home-main #view { max-width: none; height: 100%; }
#view > .home { max-width: none; }
.home { display: flex; flex-direction: column; gap: var(--s3); height: 100%; padding: var(--s3) var(--gutter) max(var(--s4), env(safe-area-inset-bottom)); }
.home-head { flex: none; display: flex; align-items: flex-end; justify-content: space-between; gap: var(--s3); }
.home-head h2 { margin: 0; font: 400 34px/1 var(--font-title); letter-spacing: -.01em; }
.home-head h2 em { font-style: italic; color: var(--accent); }
.home-auth { flex: none; display: flex; gap: var(--s2); }
.home-auth button { min-height: 44px; padding: 0 var(--s3); font-size: var(--t-sm); }
.home-tile { color: inherit; text-decoration: none; }
.home-ambre { position: relative; flex: none; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: 2px var(--s3); min-height: 92px; padding: var(--s3) var(--s4); border-radius: 18px; background: radial-gradient(circle at 100% 0, rgb(255 255 255 / .09) 0 34%, transparent 34.4%), radial-gradient(circle at 100% 0, rgb(255 255 255 / .05) 0 58%, transparent 58.4%), var(--accent); color: #fff; }
.home-ambre strong { grid-column: 1; font: 400 40px/1 var(--font-title); letter-spacing: -.01em; }
.home-ambre .home-about { display: none; grid-column: 1 / -1; max-width: 36ch; color: rgb(255 255 255 / .8); line-height: 1.55; }
.home-ambre .home-state { grid-column: 1; grid-row: 1; align-self: start; color: rgb(255 255 255 / .78); font: 500 11px/1.3 var(--font-ui); letter-spacing: .09em; text-transform: uppercase; }
.home-ambre .home-open { position: relative; z-index: 1; grid-column: 2; grid-row: 1 / span 2; align-self: end; padding: 11px 18px; border-radius: 22px; background: #fff; color: var(--accent); font: 600 var(--t-sm)/1 var(--font-ui); white-space: nowrap; }
.home-ambre:active { filter: brightness(.92); }
@media (hover: hover) { .home-ambre { transition: transform .18s ease, box-shadow .18s ease; } .home-ambre:hover { transform: translateY(-2px); box-shadow: 0 18px 40px -18px color-mix(in srgb, var(--accent) 70%, #000); } }
.home-index { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.home-group { display: flex; flex-direction: column; min-height: 0; }
.home-group h3 { flex: none; margin: 0; padding: var(--s2) 0 var(--s1); color: var(--muted); font: 500 11px/1.3 var(--font-ui); letter-spacing: .12em; text-transform: uppercase; }
.home-group ul { flex: 1; min-height: 0; display: flex; flex-direction: column; margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--ink); }
.home-group li { flex: 1; min-height: 0; display: flex; margin: 0; border-bottom: 1px solid var(--line); }
.home-row { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 0 var(--s3); min-height: 44px; }
.home-row strong { font: 400 25px/1.05 var(--font-title); overflow-wrap: anywhere; }
.home-row .home-about { display: none; grid-column: 1; color: var(--muted); font-size: var(--t-sm); line-height: 1.45; }
.home-row .home-state { grid-column: 2; grid-row: 1; display: inline-flex; align-items: baseline; gap: 6px; white-space: nowrap; }
.home-row .home-state:empty { display: none; }
.home-num { font: 400 25px/1 var(--font-title); font-variant-numeric: tabular-nums; }
.home-unit { color: var(--muted); font: 500 11px/1 var(--font-ui); letter-spacing: .08em; text-transform: uppercase; }
.home-meter { grid-column: 2; grid-row: 2; justify-self: end; display: none; width: 100%; min-width: 96px; height: 2px; background: var(--line); overflow: hidden; }
.home-meter i { display: block; height: 100%; background: var(--accent); }
.home-row:active strong { color: var(--accent); }
@media (hover: hover) { .home-row strong { transition: transform .18s ease, color .18s ease; } .home-row:hover strong { color: var(--accent); transform: translateX(6px); } }
.home-index .home-group:nth-child(1) { flex: 5; } .home-index .home-group:nth-child(2) { flex: 3; } .home-index .home-group:nth-child(3) { flex: 1; }
@media (max-width: 899px) { .home-head.is-guest h2 { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); } .home-head.is-guest .home-auth { flex: 1; } .home-head.is-guest .home-auth button { flex: 1; } }
/* Short phones: the headings give their room to the lines. */
@media (max-width: 899px) and (max-height: 700px) { .home { gap: var(--s2); padding-top: var(--s2); } .home-group h3 { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); } .home-group + .home-group ul { border-top: 0; } .home-ambre { min-height: 64px; align-items: center; padding-block: var(--s2); } .home-ambre .home-state { display: none; } .home-head h2 { font-size: 26px; } .home-ambre strong { font-size: 30px; } .home-row strong, .home-num { font-size: 21px; } }
@media (min-width: 900px) {
  .home-main { height: auto; min-height: calc(100dvh - var(--header-height) - 1px); }
  #view > .home, .home { max-width: calc(1120px + 2 * var(--gutter)); }
  .home { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); grid-template-rows: auto 1fr; align-content: start; gap: var(--s6) 88px; height: auto; margin: 0 auto; padding: var(--s6) var(--gutter) var(--s5); }
  .home-head { grid-column: 1; flex-direction: column; align-items: flex-start; }
  .home-head h2 { font-size: clamp(56px, 6.2vw, 84px); line-height: .95; letter-spacing: -.02em; }
  .home-auth { margin-top: var(--s4); }
  .home-ambre { grid-column: 1; align-self: start; grid-template-columns: minmax(0, 1fr); align-items: start; gap: var(--s3); min-height: 300px; padding: var(--s6); border-radius: 24px; }
  .home-ambre strong { font-size: 76px; line-height: .95; }
  .home-ambre .home-about { display: block; font-size: var(--t-md); }
  .home-ambre .home-state { grid-row: auto; }
  .home-ambre .home-open { grid-column: 1; grid-row: auto; justify-self: start; align-self: end; margin-top: var(--s3); padding: 14px 22px; font-size: var(--t-md); }
  .home-index { grid-column: 2; grid-row: 1 / span 2; gap: var(--s4); padding-top: var(--s2); }
  .home-index .home-group { flex: none; }
  .home-group h3 { padding: 0 0 var(--s2); }
  .home-group li { flex: none; }
  .home-row { grid-template-rows: auto auto; row-gap: 0; padding: 8px 0; }
  .home-row strong { font-size: 29px; }
  .home-num { font-size: 29px; }
  .home-row .home-about { display: block; }
  .home-meter { display: block; margin-top: 4px; }
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
function homeStates(extra){
 var me=state.me||{},ready=me.companion_readiness||{},out={};if(!me.user)return out;
 if(ready.events)out['/ma-ligne-de-vie/']={num:ready.events.count||0,unit:(ready.events.count||0)>1?'moments':'moment'};
 if(ready.people)out['/mes-proches/']={num:ready.people.count||0,unit:(ready.people.count||0)>1?'proches':'proche'};
 if(ready.childhood&&ready.childhood.required)out['/cartographie/']={num:ready.childhood.answered||0,unit:'sur '+ready.childhood.required,ratio:(ready.childhood.answered||0)/ready.childhood.required};
 if(ready.symptoms&&ready.symptoms.required)out['/mes-symptomes/']={num:ready.symptoms.answered||0,unit:'sur '+ready.symptoms.required,ratio:(ready.symptoms.answered||0)/ready.symptoms.required};
 if(ready.identity)out['/mon-compte/']={unit:ready.identity.complete?'Identité complète':'Identité à compléter'};
 if(extra){out['/mes-sujets/']={num:extra.topics||0,unit:(extra.topics||0)>1?'sujets':'sujet'};out['/mes-bilans/']={num:extra.reviews||0,unit:(extra.reviews||0)>1?'bilans':'bilan'};out['/ma-cartographie/']={num:extra.elements||0,unit:(extra.elements||0)>1?'éléments reliés':'élément relié'};out['/ambre/']={unit:extra.review_running?'Bilan en préparation':extra.reviews?'Conversation ouverte':ready.complete?'Prête pour votre premier bilan':'Premier bilan à préparer'}}
 return out;
}
function renderHome(){
 var view=document.getElementById('view'),current=renderGuard(view),me=state.me||{},user=me.user;
 if(main.classList)main.classList.add('home-main');
 var home=el('div','home'),head=el('div','home-head'),titles=el('div'),index=el('div','home-index');
 var title=el('h2');title.append('Décode ton ',el('em','','esprit'));titles.append(title);head.append(titles);if(!user)head.classList.add('is-guest');
 if(!user){var auth=el('div','home-auth');[['login','Se connecter',''],['register','Créer un compte','secondary']].forEach(function(entry){var button=el('button',entry[2],entry[1]);button.type='button';button.dataset.auth=entry[0];auth.append(button)});head.append(auth)}
 var pages=HOME_PAGES.slice();if(user&&user.role==='psychologist')pages=[{label:'Espace psychologue',href:'/espace-psychologue/'},{label:'Mon compte',href:'/mon-compte/'}];
 var nodes={},lists={};home.append(head);
 pages.forEach(function(page){var detail=HOME_DETAILS[page.href]||['','read'],tile=el('a','home-tile'),status=el('span','home-state');tile.href=page.href;nodes[page.href]={status:status};
  if(detail[1]==='ambre'){tile.className='home-tile home-ambre';tile.append(el('strong','',page.label),el('span','home-about',detail[0]),status,el('span','home-open','Ouvrir'));home.append(tile);return}
  var meter=el('span','home-meter'),fill=el('i');meter.hidden=true;meter.append(fill);nodes[page.href].meter=meter;nodes[page.href].fill=fill;
  tile.className='home-tile home-row';tile.append(el('strong','',page.label),status,el('span','home-about',detail[0]),meter);
  if(!lists[detail[1]]){var group=el('section','home-group'),label=HOME_GROUPS.find(function(row){return row[0]===detail[1]}),heading=el('h3','',label?label[1]:''),list=el('ul');heading.id='home-group-'+detail[1];list.setAttribute('aria-labelledby',heading.id);group.append(heading,list);index.append(group);lists[detail[1]]=list}
  var item=el('li');item.append(tile);lists[detail[1]].append(item)});
 function fill(extra){var states=homeStates(extra);Object.keys(nodes).forEach(function(href){var value=states[href],node=nodes[href];if(!value)return;node.status.replaceChildren();if(value.num!==undefined&&node.meter)node.status.append(el('span','home-num',String(value.num)));if(node.meter)node.status.append(el('span','home-unit',value.unit));else node.status.textContent=value.unit;if(node.meter&&value.ratio!==undefined){node.meter.hidden=false;node.fill.style.width=Math.round(Math.max(0,Math.min(1,value.ratio))*100)+'%'}})}
 home.append(index);view.replaceChildren(home);fill(null);
 if(user&&user.role!=='psychologist')api('/api/dashboard').then(function(extra){if(current())fill(extra)}).catch(function(){});
}
`;
