// The dashboard: the only way into every page now that the menu is gone. A grid of blocks of different
// sizes: Ambre is the large one, each other page shows its name and its figure in large numerals. No
// icon, and the same sans-serif as the rest of the interface. On a phone the whole page fits one screen
// without scrolling. HOME_PAGES comes from data/navigation.json.
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
.home { display: flex; flex-direction: column; gap: var(--s3); height: 100%; padding: var(--s3) var(--gutter) max(var(--s3), env(safe-area-inset-bottom)); font-family: var(--font-ui); }
.home-head { flex: none; display: flex; align-items: center; justify-content: space-between; gap: var(--s3); min-height: 32px; }
.home-head h2 { margin: 0; font: 700 22px/1.1 var(--font-ui); letter-spacing: -.02em; }
.home-auth { flex: none; display: flex; gap: var(--s2); }
.home-auth button { min-height: 44px; padding: 0 var(--s3); font-size: var(--t-sm); }
.home-grid { flex: 1; min-height: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: minmax(0, 1.25fr) repeat(4, minmax(0, 1fr)) auto; gap: 10px; margin: 0; padding: 0; list-style: none; }
.home-grid li { min-width: 0; min-height: 0; margin: 0; display: flex; }
.home-tile { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: space-between; gap: 4px; padding: 14px 16px; border: 1px solid var(--line); border-radius: 18px; background: var(--surface); color: var(--ink); text-decoration: none; }
.home-tile strong { font: 600 15px/1.25 var(--font-ui); letter-spacing: -.01em; }
.home-about { display: none; color: var(--muted); font-size: var(--t-sm); line-height: 1.5; }
.home-state { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0 6px; min-height: 1em; }
.home-num { font: 650 30px/1 var(--font-ui); letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
.home-unit { color: var(--muted); font: 500 var(--t-xs)/1.2 var(--font-ui); }
.home-meter { display: block; height: 4px; margin-top: 6px; border-radius: 2px; background: var(--line); overflow: hidden; }
.home-meter[hidden] { display: none; }
.home-meter i { display: block; height: 100%; border-radius: 2px; background: var(--accent); }
.home-tile:active { background: var(--accent-wash); }
@media (hover: hover) { .home-tile { transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease; } .home-tile:hover { transform: translateY(-2px); border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); box-shadow: 0 14px 30px -18px rgb(15 23 42 / .35); } }
.home-grid .is-ambre { grid-column: 1 / -1; }
.home-ambre { flex-direction: row; align-items: flex-end; justify-content: space-between; border-color: var(--accent); background: var(--accent); color: #fff; }
.home-ambre .home-text { display: flex; flex-direction: column; justify-content: space-between; align-self: stretch; gap: 4px; min-width: 0; }
.home-ambre strong { font-size: 26px; font-weight: 700; letter-spacing: -.02em; }
.home-ambre .home-unit, .home-ambre .home-about { color: rgb(255 255 255 / .82); }
.home-ambre .home-open { flex: none; padding: 10px 18px; border-radius: 22px; background: #fff; color: var(--accent); font: 600 var(--t-sm)/1 var(--font-ui); }
.home-ambre:active { background: var(--accent); filter: brightness(.92); }
.home-grid .is-account { grid-column: 1 / -1; }
.home-account { flex-direction: row; align-items: center; min-height: 44px; padding-block: 10px; }
/* Short phones: tighter blocks, smaller numerals. */
@media (max-width: 899px) and (max-height: 700px) { .home { gap: var(--s2); padding-top: var(--s2); } .home-grid { gap: 8px; } .home-tile { padding: 8px 12px; border-radius: 14px; } .home-num { font-size: 22px; } .home-tile strong { font-size: 14px; } .home-ambre strong { font-size: 20px; } .home-head h2 { font-size: 19px; } .home-meter { margin-top: 3px; height: 3px; } .home-ambre .home-open { padding: 8px 14px; } }
@media (max-width: 359px) { .home-tile { padding: 6px 10px; } .home-num { font-size: 20px; } .home-tile strong { font-size: 13px; } }
@media (max-width: 899px) { .home-head.is-guest h2 { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); } .home-head.is-guest .home-auth { flex: 1; } .home-head.is-guest .home-auth button { flex: 1; } }
@media (min-width: 900px) {
  .home-main { height: auto; min-height: calc(100dvh - var(--header-height) - 1px); }
  #view > .home, .home { max-width: calc(1120px + 2 * var(--gutter)); }
  .home { height: auto; margin: 0 auto; gap: var(--s5); padding: var(--s6) var(--gutter) var(--s6); }
  .home-head h2 { font-size: 34px; letter-spacing: -.03em; }
  .home-grid { flex: none; grid-template-columns: repeat(4, minmax(0, 1fr)); grid-template-rows: repeat(3, 196px) auto; gap: 16px; }
  .home-tile { padding: 22px; border-radius: 22px; }
  .home-tile strong { font-size: 18px; }
  .home-about { display: block; margin-top: 6px; }
  .home-num { font-size: 46px; }
  .home-unit { font-size: var(--t-sm); }
  .home-grid .is-ambre { grid-column: 1 / span 2; grid-row: 1 / span 2; }
  .home-ambre { flex-direction: column; align-items: flex-start; padding: 32px; }
  .home-ambre .home-text { align-self: auto; justify-content: flex-start; gap: 10px; }
  .home-ambre strong { font-size: 54px; line-height: 1; letter-spacing: -.04em; }
  .home-ambre .home-about { max-width: 38ch; margin: 0; font-size: 17px; line-height: 1.55; }
  .home-ambre .home-unit { order: -1; font-size: var(--t-sm); font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
  .home-ambre .home-open { padding: 14px 24px; font-size: var(--t-md); }
  .home-account { min-height: 56px; padding: 0 22px; }
}
`;
export const HOME_INTERFACE_SCRIPT=String.raw`
var HOME_DETAILS={
 '/ambre/':['Votre assistante psychologue IA. Elle lit ce que vous renseignez, prépare vos bilans et en parle avec vous.','ambre'],
 '/ma-ligne-de-vie/':['Les moments qui ont compté.',''],
 '/mes-proches/':['Les personnes importantes de votre vie.',''],
 '/cartographie/':['Ce que vous avez vécu en grandissant.',''],
 '/mes-symptomes/':['Les signes que vous reconnaissez chez vous.',''],
 '/mes-sujets/':['Vos points de vue et vos arguments.',''],
 '/ma-cartographie/':['Tout ce que vous avez renseigné, relié.',''],
 '/mes-bilans/':['Les bilans préparés par Ambre.',''],
 '/mes-diagnostics/':['Les hypothèses qui ressortent de vos bilans.',''],
 '/mon-compte/':['','account'],
 '/espace-psychologue/':['Les dossiers partagés avec vous.','']
};
function homeStates(extra){
 var me=state.me||{},ready=me.companion_readiness||{},out={};if(!me.user)return out;
 if(ready.events)out['/ma-ligne-de-vie/']={num:ready.events.count||0,unit:(ready.events.count||0)>1?'moments':'moment'};
 if(ready.people)out['/mes-proches/']={num:ready.people.count||0,unit:(ready.people.count||0)>1?'proches':'proche'};
 if(ready.childhood&&ready.childhood.required)out['/cartographie/']={num:ready.childhood.answered||0,unit:'sur '+ready.childhood.required,ratio:(ready.childhood.answered||0)/ready.childhood.required};
 if(ready.symptoms&&ready.symptoms.required)out['/mes-symptomes/']={num:ready.symptoms.answered||0,unit:'sur '+ready.symptoms.required,ratio:(ready.symptoms.answered||0)/ready.symptoms.required};
 if(ready.identity)out['/mon-compte/']={unit:ready.identity.complete?'Identité complète':'Identité à compléter'};
 if(extra){out['/mes-sujets/']={num:extra.topics||0,unit:(extra.topics||0)>1?'sujets':'sujet'};out['/mes-bilans/']={num:extra.reviews||0,unit:(extra.reviews||0)>1?'bilans':'bilan'};out['/ma-cartographie/']={num:extra.elements||0,unit:(extra.elements||0)>1?'éléments':'élément'};out['/ambre/']={unit:extra.review_running?'Bilan en préparation':extra.reviews?'Conversation ouverte':ready.complete?'Prête pour votre premier bilan':'Premier bilan à préparer'}}
 return out;
}
function renderHome(){
 var view=document.getElementById('view'),current=renderGuard(view),me=state.me||{},user=me.user;
 if(main.classList)main.classList.add('home-main');
 var home=el('div','home'),head=el('div','home-head'),grid=el('ul','home-grid');
 head.append(el('h2','','Décode ton esprit'));if(!user){head.classList.add('is-guest');var auth=el('div','home-auth');[['login','Se connecter',''],['register','Créer un compte','secondary']].forEach(function(entry){var button=el('button',entry[2],entry[1]);button.type='button';button.dataset.auth=entry[0];auth.append(button)});head.append(auth)}
 var pages=HOME_PAGES.slice();if(user&&user.role==='psychologist')pages=[{label:'Espace psychologue',href:'/espace-psychologue/'},{label:'Mon compte',href:'/mon-compte/'}];
 var nodes={};
 pages.forEach(function(page){var detail=HOME_DETAILS[page.href]||['',''],item=el('li'),tile=el('a','home-tile'),status=el('span','home-state'),name=el('strong','',page.label);tile.href=page.href;nodes[page.href]={status:status};
  if(detail[1]==='ambre'){item.className='is-ambre';tile.className='home-tile home-ambre';var text=el('span','home-text');text.append(name,el('span','home-about',detail[0]),status);tile.append(text,el('span','home-open','Ouvrir'))}
  else if(detail[1]==='account'){item.className='is-account';tile.className='home-tile home-account';tile.append(name,status)}
  else{var top=el('span','home-text'),bottom=el('span','home-figure'),meter=el('span','home-meter'),bar=el('i');meter.hidden=true;meter.append(bar);nodes[page.href].meter=meter;nodes[page.href].bar=bar;top.append(name,el('span','home-about',detail[0]));bottom.append(status,meter);tile.append(top,bottom)}
  item.append(tile);grid.append(item)});
 function fill(extra){var states=homeStates(extra);Object.keys(nodes).forEach(function(href){var value=states[href],node=nodes[href];if(!value)return;node.status.replaceChildren();if(value.num!==undefined)node.status.append(el('span','home-num',String(value.num)));node.status.append(el('span','home-unit',value.unit));if(node.meter&&value.ratio!==undefined){node.meter.hidden=false;node.bar.style.width=Math.round(Math.max(0,Math.min(1,value.ratio))*100)+'%'}})}
 home.append(head,grid);view.replaceChildren(home);fill(null);
 if(user&&user.role!=='psychologist')api('/api/dashboard').then(function(extra){if(current())fill(extra)}).catch(function(){});
}
`;
