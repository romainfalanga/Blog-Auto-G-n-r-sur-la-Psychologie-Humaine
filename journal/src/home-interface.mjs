// The dashboard: the only way into every page now that the menu is gone. On a phone the ten tiles
// fill exactly one screen, without scrolling; on a wide screen each card adds a sentence and what
// the person has already filled in. HOME_PAGES comes from data/navigation.json.
export const HOME_INTERFACE_STYLE=String.raw`
.shell:has(> .home-main) { max-width: none; padding: 0; }
body:has(.home-main) .site-footer { display: none; }
.home-main { height: calc(100dvh - var(--header-height) - 1px); min-height: 420px; }
.home-main .app-head { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
.home-main #app-messages:empty { display: none; }
.home-main #view { max-width: none; height: 100%; }
#view > .home { max-width: none; }
.home { display: flex; flex-direction: column; gap: var(--s3); height: 100%; padding: var(--s3) var(--s4) max(var(--s3), env(safe-area-inset-bottom)); }
.home-head { flex: none; display: flex; align-items: center; justify-content: space-between; gap: var(--s3); min-height: 44px; }
.home-head h2 { margin: 0; font: 400 var(--t-xl)/1.2 var(--font-title); overflow-wrap: anywhere; }
.home-head p { display: none; margin: var(--s1) 0 0; color: var(--muted); }
.home-auth { display: flex; gap: var(--s2); }
.home-auth button { min-height: 44px; padding: 0 var(--s3); font-size: var(--t-sm); }
.home-grid { flex: 1; min-height: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-rows: minmax(0, 1fr); gap: var(--s2); margin: 0; padding: 0; list-style: none; }
.home-grid li { min-width: 0; min-height: 0; margin: 0; }
.home-tile { display: flex; flex-direction: column; justify-content: center; gap: 6px; width: 100%; height: 100%; min-height: 56px; padding: var(--s2) var(--s3); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); color: var(--ink); text-decoration: none; }
.home-tile:active { background: var(--accent-wash); }
@media (hover: hover) { .home-tile:hover { border-color: var(--accent); box-shadow: 0 6px 20px rgb(15 23 42 / .08); } }
.home-icon { flex: none; display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 10px; background: var(--accent-wash); color: var(--accent); }
.home-icon svg { width: 20px; height: 20px; }
.home-tile strong { font: 600 var(--t-md)/1.25 var(--font-ui); }
.home-tile .home-about { display: none; color: var(--muted); font-size: var(--t-sm); line-height: 1.45; }
.home-tile .home-state { color: var(--muted); font-size: var(--t-xs); line-height: 1.3; font-variant-numeric: tabular-nums; }
.home-tile .home-state:empty { display: none; }
.home-tile.is-main { border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); }
.home-tile.is-main .home-icon { background: var(--accent); color: #fff; }
/* Short phones: the icon sits beside the name so that five rows always fit. */
@media (max-width: 899px) and (max-height: 700px) { .home-tile { flex-direction: row; align-items: center; justify-content: flex-start; gap: var(--s2); } .home-tile .home-state { display: none; } .home-icon { width: 32px; height: 32px; } }
@media (max-width: 359px) { .home { padding-inline: var(--s3); } .home-tile { padding: var(--s1) var(--s2); } .home-tile strong { font-size: var(--t-sm); } .home-icon { width: 28px; height: 28px; border-radius: 8px; } .home-icon svg { width: 16px; height: 16px; } }
@media (min-width: 900px) {
  .home-main { height: auto; min-height: calc(100dvh - var(--header-height) - 1px); }
  .home { max-width: calc(1120px + 2 * var(--gutter)); height: auto; margin: 0 auto; padding: var(--s6) var(--gutter) var(--s7); gap: var(--s5); }
  #view > .home { max-width: calc(1120px + 2 * var(--gutter)); }
  .home-head { align-items: flex-end; }
  .home-head h2 { font-size: var(--t-2xl); }
  .home-head p { display: block; }
  .home-grid { flex: none; grid-template-columns: repeat(5, minmax(0, 1fr)); grid-auto-rows: auto; gap: var(--s4); }
  .home-tile { justify-content: flex-start; gap: var(--s2); min-height: 204px; padding: var(--s4); }
  .home-icon { width: 44px; height: 44px; margin-bottom: var(--s1); border-radius: 12px; }
  .home-icon svg { width: 24px; height: 24px; }
  .home-tile .home-about { display: block; }
  .home-tile .home-state { margin-top: auto; padding-top: var(--s2); color: var(--ink); font-size: var(--t-sm); font-weight: 600; }
}
@media (min-width: 900px) and (max-width: 1099px) { .home-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
`;
export const HOME_INTERFACE_SCRIPT=String.raw`
var HOME_DETAILS={
 '/ambre/':['M4 5h16v11H9l-5 4z','Votre assistante psychologue IA : bilans et conversation.'],
 '/ma-ligne-de-vie/':['M4 12h16M7 12V7m5 5v-7m5 7V9M7 17h.01M12 17h.01M17 17h.01','Les moments qui ont compté, datés et reliés à vos proches.'],
 '/mes-proches/':['M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm-6 8v-1a6 6 0 0 1 12 0v1m1-13a3 3 0 0 1 0 6m2 3a4 4 0 0 1 3 4v1','Les personnes importantes de votre vie.'],
 '/cartographie/':['M12 3 4 9v11h5v-6h6v6h5V9z','Ce que vous avez vécu en grandissant.'],
 '/mes-symptomes/':['M4 13h3l2-6 4 12 2-6h5','Les signes que vous reconnaissez chez vous.'],
 '/mes-sujets/':['M12 4a6 6 0 0 0-4 10.5V17h8v-2.5A6 6 0 0 0 12 4zM9.5 20h5','Vos points de vue et la façon dont vos arguments se construisent.'],
 '/ma-cartographie/':['M6 7a2 2 0 1 0 0-.01zM18 6a2 2 0 1 0 0-.01zM12 18a2 2 0 1 0 0-.01zM7.5 8.5l3.5 7.5M16.5 7.5 13 16M8 6.5h8','Toutes vos informations reliées en un seul réseau.'],
 '/mes-bilans/':['M7 3h8l4 4v14H7zM14 3v5h5M10 13h6M10 17h6','Les bilans préparés par Ambre.'],
 '/mes-diagnostics/':['M11 4a7 7 0 1 0 4.9 12l4.1 4M11 8v6m-3-3h6','Les hypothèses qui ressortent de vos bilans.'],
 '/mon-compte/':['M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm-7 8a7 7 0 0 1 14 0','Votre identité, vos données et votre abonnement.']
};
function homeCount(n,one,many){return n+' '+(n>1?many:one)}
function homeStates(extra){
 var me=state.me||{},ready=me.companion_readiness||{},out={};if(!me.user)return out;
 if(ready.people)out['/mes-proches/']=homeCount(ready.people.count||0,'proche','proches');
 if(ready.events)out['/ma-ligne-de-vie/']=homeCount(ready.events.count||0,'moment','moments');
 if(ready.childhood&&ready.childhood.required)out['/cartographie/']=(ready.childhood.answered||0)+' / '+ready.childhood.required+' réponses';
 if(ready.symptoms&&ready.symptoms.required)out['/mes-symptomes/']=(ready.symptoms.answered||0)+' / '+ready.symptoms.required+' réponses';
 if(ready.identity)out['/mon-compte/']=ready.identity.complete?'Identité complète':'Identité à compléter';
 if(extra){out['/mes-sujets/']=homeCount(extra.topics||0,'sujet','sujets');out['/mes-bilans/']=homeCount(extra.reviews||0,'bilan','bilans');out['/ma-cartographie/']=homeCount(extra.elements||0,'élément relié','éléments reliés');out['/ambre/']=extra.review_running?'Bilan en préparation':extra.reviews?'Conversation ouverte':ready.complete?'Prête pour votre premier bilan':'Premier bilan à préparer'}
 return out;
}
function renderHome(){
 var view=document.getElementById('view'),current=renderGuard(view),me=state.me||{},user=me.user;
 if(main.classList)main.classList.add('home-main');
 var home=el('div','home'),head=el('div','home-head'),titles=el('div'),grid=el('ul','home-grid'),first=me.profile&&String(me.profile.first_name||'').trim();
 titles.append(el('h2','',user?(first?'Bonjour '+first:'Bonjour'):'Décode ton esprit'),el('p','',user?'Où souhaitez-vous aller ?':'Un espace personnel pour mieux vous comprendre.'));head.append(titles);
 if(!user){var auth=el('div','home-auth');[['login','Se connecter',''],['register','Créer un compte','secondary']].forEach(function(entry){var button=el('button',entry[2],entry[1]);button.type='button';button.dataset.auth=entry[0];auth.append(button)});head.append(auth)}
 var pages=HOME_PAGES.slice();if(user&&user.role==='psychologist')pages=[{label:'Espace psychologue',href:'/espace-psychologue/'},{label:'Mon compte',href:'/mon-compte/'}];
 var stateNodes={};
 pages.forEach(function(page,index){var detail=HOME_DETAILS[page.href]||['M5 12h14',''],item=el('li'),tile=el('a','home-tile'+(index===0?' is-main':'')),icon=el('span','home-icon'),status=el('span','home-state');
  tile.href=page.href;icon.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="'+detail[0]+'"/></svg>';
  stateNodes[page.href]=status;tile.append(icon,el('strong','',page.label),el('span','home-about',detail[1]),status);item.append(tile);grid.append(item)});
 function fill(extra){var states=homeStates(extra);Object.keys(stateNodes).forEach(function(href){if(states[href])stateNodes[href].textContent=states[href]})}
 home.append(head,grid);view.replaceChildren(home);fill(null);
 if(user&&user.role!=='psychologist')api('/api/dashboard').then(function(extra){if(current())fill(extra)}).catch(function(){});
}
`;
