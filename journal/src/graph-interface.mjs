// « Ma cartographie »: the personal graph, full screen under the menu like a note graph. Floating
// search, filters and zoom; a card opens on the selected element (side panel on wide screens, bottom
// sheet on phones). Labels never overlap and appear by importance as one zooms in.
export const GRAPH_INTERFACE_STYLE=String.raw`
.shell:has(> .graph-main) { max-width: none; padding: 0; }
body:has(.graph-main) .site-footer { display: none; }
.graph-main { height: calc(100dvh - var(--header-height) - 1px); min-height: 320px; }
.graph-main .app-head { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
.graph-main #app-messages:empty { display: none; }
.graph-main #view { max-width: none; height: 100%; }
#view > .graph-app { max-width: none; }
.graph-app { position: relative; width: 100%; height: 100%; overflow: hidden; background: var(--bg); touch-action: none; }
.graph-app canvas { display: block; width: 100%; height: 100%; cursor: grab; }
.graph-app canvas.is-dragging { cursor: grabbing; }
.graph-top { position: absolute; top: var(--s3); left: var(--s3); right: var(--s3); display: flex; gap: var(--s2); max-width: 460px; pointer-events: none; }
.graph-top > * { pointer-events: auto; }
.graph-top input[type=search] { flex: 1; min-width: 0; height: 48px; padding: 0 var(--s4); border: 1px solid var(--line); border-radius: 24px; background: var(--surface); box-shadow: 0 2px 10px rgb(15 23 42 / .08); }
.graph-pill { display: inline-flex; align-items: center; justify-content: center; min-width: 48px; height: 48px; padding: 0 var(--s4); border: 1px solid var(--line); border-radius: 24px; background: var(--surface); color: var(--ink); font: 600 var(--t-sm)/1 var(--font-ui); box-shadow: 0 2px 10px rgb(15 23 42 / .08); cursor: pointer; }
.graph-pill[aria-expanded="true"] { border-color: var(--accent); color: var(--accent); }
@media (hover: hover) { .graph-pill:hover { background: var(--accent-wash); } }
.graph-drop { position: absolute; top: calc(var(--s3) + 56px); left: var(--s3); right: var(--s3); max-width: 460px; max-height: min(60%, 420px); margin: 0; padding: var(--s2); overflow-y: auto; border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); box-shadow: 0 8px 28px rgb(15 23 42 / .14); list-style: none; }
.graph-drop[hidden] { display: none; }
.graph-drop button, .graph-card-links button { display: flex; align-items: center; justify-content: flex-start; gap: var(--s3); width: 100%; min-height: 44px; padding: var(--s1) var(--s2); border: 0; border-radius: var(--r-control); background: none; color: var(--ink); font: 400 var(--t-sm)/1.35 var(--font-ui); text-align: left; cursor: pointer; }
@media (hover: hover) { .graph-drop button:hover, .graph-card-links button:hover { background: var(--accent-wash); } }
.graph-drop button[aria-pressed="false"] { color: var(--muted); }
.graph-drop button[aria-pressed="false"] i { opacity: .25; }
.graph-drop .graph-count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.graph-dot { flex: none; width: 12px; height: 12px; border-radius: 50%; }
.graph-drop-empty { margin: 0; padding: var(--s3); color: var(--muted); font-size: var(--t-sm); }
.graph-zoom { position: absolute; left: var(--s3); bottom: var(--s3); display: flex; gap: var(--s2); }
.graph-zoom .graph-pill { width: 48px; padding: 0; font-size: 20px; font-weight: 400; }
.graph-zoom .graph-fit { width: auto; padding: 0 var(--s4); font: 600 var(--t-sm)/1 var(--font-ui); }
.graph-fit svg { display: none; }
.graph-hint { position: absolute; left: 50%; bottom: var(--s3); transform: translateX(-50%); margin: 0; padding: var(--s2) var(--s4); border-radius: 20px; background: color-mix(in srgb, var(--surface) 92%, transparent); color: var(--muted); font-size: var(--t-sm); white-space: nowrap; pointer-events: none; }
.graph-card { position: absolute; top: var(--s3); right: var(--s3); width: 340px; max-height: calc(100% - 2 * var(--s3)); display: flex; flex-direction: column; padding: var(--s4); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); box-shadow: 0 8px 28px rgb(15 23 42 / .14); }
.graph-card[hidden] { display: none; }
.graph-card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--s3); }
.graph-card-type { display: inline-flex; align-items: center; gap: var(--s2); margin: 0 0 var(--s1); color: var(--muted); font-size: var(--t-xs); }
.graph-card h2 { margin: 0; font: 600 var(--t-md)/1.35 var(--font-ui); overflow-wrap: anywhere; }
.graph-card-close { flex: none; width: 44px; height: 44px; margin: -10px -10px 0 0; padding: 0; border: 0; border-radius: 50%; background: none; color: var(--muted); font-size: 24px; line-height: 1; cursor: pointer; }
.graph-card-body { min-height: 0; margin-top: var(--s2); overflow-y: auto; }
.graph-card-body p { margin: 0 0 var(--s3); line-height: 1.55; overflow-wrap: anywhere; }
.graph-card-body .button { min-height: 44px; }
.graph-card-body h3 { margin: var(--s4) 0 var(--s1); color: var(--muted); font: 600 var(--t-xs)/1.4 var(--font-ui); }
.graph-card-links { margin: 0 calc(-1 * var(--s2)); padding: 0; list-style: none; }
.graph-empty { position: absolute; left: 50%; top: 50%; width: min(420px, calc(100% - 2 * var(--s4))); transform: translate(-50%, -50%); padding: var(--s5); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); box-shadow: 0 8px 28px rgb(15 23 42 / .1); text-align: center; }
.graph-empty h2 { margin: 0 0 var(--s2); font: 600 var(--t-md)/1.35 var(--font-ui); }
.graph-empty p { margin: 0 0 var(--s3); color: var(--muted); line-height: 1.55; }
@media (max-width: 760px) {
  .graph-top, .graph-drop { max-width: none; }
  .graph-zoom { left: auto; right: var(--s3); top: calc(var(--s3) + 60px); bottom: auto; flex-direction: column; align-items: flex-end; }
  .graph-zoom .graph-fit { width: 48px; padding: 0; }
  .graph-fit svg { display: block; }
  .graph-fit span { display: none; }
  .graph-hint { bottom: max(var(--s3), env(safe-area-inset-bottom)); }
  .graph-card { top: auto; left: var(--s2); right: var(--s2); bottom: max(var(--s2), env(safe-area-inset-bottom)); width: auto; max-height: 46%; }
  .graph-app:has(.graph-card:not([hidden])) .graph-hint { display: none; }
}
`;
export const GRAPH_INTERFACE_SCRIPT=String.raw`
var GRAPH_KINDS=[['me','Moi','#1F5C4D',0],['page','Pages','#475569',0],['person','Proches','#3B82F6',1],['event','Moments','#D97706',2],['note','Notes','#94A3B8',2],['childhood','Enfance','#0891B2',2],['theme','Thèmes de symptômes','#DB2777',1],['symptom','Symptômes','#F472B6',2],['past','Symptômes passés','#D4B5C3',2],['topic','Sujets','#7C3AED',1],['position','Positions','#A78BFA',2],['support','Raisons','#16A34A',2],['objection','Objections','#DC2626',2],['review','Bilans','#CA8A04',1]];
function graphKind(type){return GRAPH_KINDS.find(function(row){return row[0]===type})||GRAPH_KINDS[1]}
async function renderGraph(){
 var view=document.getElementById('view'),current=renderGuard(view),data;
 if(main.classList)main.classList.add('graph-main');
 try{data=await api('/api/graph')}catch(error){if(current()){if(main.classList)main.classList.remove('graph-main');view.replaceChildren(el('div','notice warn',error.message||'La cartographie n’a pas pu être chargée.'),actionButton('Réessayer',function(){renderApp()},'secondary'))}return}
 if(!current())return;
 var coarse=typeof matchMedia==='function'&&matchMedia('(pointer: coarse)').matches;
 var nodes=data.nodes.map(function(n){return Object.assign({},n,{x:0,y:0,vx:0,vy:0,links:[]})}),byId=new Map(nodes.map(function(n){return [n.id,n]}));
 var edges=data.edges.map(function(e){return {a:byId.get(e.source),b:byId.get(e.target)}}).filter(function(e){return e.a&&e.b&&e.a!==e.b});
 edges.forEach(function(e){e.a.links.push(e.b);e.b.links.push(e.a)});
 nodes.forEach(function(n){var kind=graphKind(n.type);n.color=kind[2];n.tier=kind[3];n.q=n.type==='me'||n.type==='page'?3.4:n.tier===1?1.7:1;n.r=(n.type==='me'?16:n.type==='page'?11:Math.min(12,5+Math.sqrt(n.links.length)*1.7))+(coarse?1:0)});
 // Every family starts from its page: a breadth-first placement gives neat clusters before the forces run.
 (function seed(){var root=byId.get('me')||nodes[0];if(!root)return;var seen=new Set([root]),queue=[root];root.angle=0;root.depth=0;while(queue.length){var parent=queue.shift(),kids=parent.links.filter(function(n){return !seen.has(n)});kids.forEach(function(child,index){seen.add(child);child.depth=parent.depth+1;var spread=parent===root?6.2832:Math.min(2.4,.5+kids.length*.16),angle=parent===root?index/kids.length*6.2832:parent.angle+(kids.length>1?(index/(kids.length-1)-.5)*spread:0),reach=parent===root?170:child.links.length>1?80:52;child.angle=angle;child.x=parent.x+Math.cos(angle)*reach;child.y=parent.y+Math.sin(angle)*reach;queue.push(child)})}nodes.forEach(function(n,i){if(!seen.has(n)){n.x=Math.cos(i)*320;n.y=Math.sin(i)*320}})})();
 var hidden=new Set(),selected=null,hovered=null,query='',cam={x:0,y:0,k:1},goal={x:0,y:0,k:1},alpha=1,frame=0,size={w:0,h:0},touched=false,settled=false;
 var app=el('div','graph-app'),canvas=el('canvas'),top=el('div','graph-top'),search=el('input'),filterButton=el('button','graph-pill','Filtres'),results=el('ul','graph-drop'),filters=el('ul','graph-drop'),zoom=el('div','graph-zoom'),hint=el('p','graph-hint',coarse?'Touchez un point pour voir ses liens':'Cliquez sur un point pour voir ses liens'),card=el('aside','graph-card');
 search.type='search';search.placeholder='Rechercher';search.setAttribute('aria-label','Rechercher dans ma cartographie');search.autocomplete='off';
 filterButton.type='button';filterButton.setAttribute('aria-expanded','false');results.hidden=true;filters.hidden=true;card.hidden=true;card.setAttribute('aria-live','polite');card.setAttribute('aria-label','Élément choisi');
 canvas.setAttribute('role','img');canvas.setAttribute('aria-label','Cartographie de vos informations : '+nodes.length+' éléments et '+edges.length+' liens. La recherche permet de la parcourir au clavier.');
 function pill(text,label,action){var button=el('button','graph-pill',text);button.type='button';button.setAttribute('aria-label',label);button.addEventListener('click',action);return button}
 zoom.append(pill('+','Zoomer',function(){zoomBy(1.4)}),pill('−','Dézoomer',function(){zoomBy(1/1.4)}),(function(){var b=pill('','Recentrer la carte',function(){fit();run()});b.className='graph-pill graph-fit';b.innerHTML='<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/></svg><span>Recentrer</span>';return b})());
 top.append(search,filterButton);app.append(canvas,top,results,filters,zoom,hint,card);view.replaceChildren(app);
 var styles=getComputedStyle(document.documentElement),ink=styles.getPropertyValue('--ink').trim()||'#1B2733',muted=styles.getPropertyValue('--muted').trim()||'#5B6B7A',line=styles.getPropertyValue('--line-strong').trim()||'#B8C2CC',paper=styles.getPropertyValue('--bg').trim()||'#F2F5F7',accent=styles.getPropertyValue('--accent').trim()||'#1F5C4D';
 function visible(n){return !hidden.has(n.type)}
 function matches(n){return !query||normalized(n.label+' '+(n.detail||'')).indexOf(query)>=0}
 function dot(color){var i=el('i','graph-dot');i.style.background=color;return i}
 function nodeButton(n){var item=el('li'),button=el('button');button.type='button';button.append(dot(n.color),el('span','',n.label));button.addEventListener('click',function(){results.hidden=true;select(n,true)});item.append(button);return item}
 var present=new Map();nodes.forEach(function(n){present.set(n.type,(present.get(n.type)||0)+1)});
 GRAPH_KINDS.filter(function(row){return present.has(row[0])&&row[0]!=='me'&&row[0]!=='page'}).forEach(function(row){var item=el('li'),button=el('button');button.type='button';button.setAttribute('aria-pressed','true');button.append(dot(row[2]),el('span','',row[1]),el('span','graph-count',String(present.get(row[0]))));
  button.addEventListener('click',function(){var on=hidden.has(row[0]);if(on)hidden.delete(row[0]);else hidden.add(row[0]);button.setAttribute('aria-pressed',String(on));if(selected&&!visible(selected))select(null);else drawCard();alpha=Math.max(alpha,.4);run()});item.append(button);filters.append(item)});
 if(!filters.children.length)filterButton.hidden=true;
 filterButton.addEventListener('click',function(){var open=filters.hidden;filters.hidden=!open;results.hidden=true;filterButton.setAttribute('aria-expanded',String(open))});
 function drawCard(){
  card.replaceChildren();card.hidden=!selected;if(!selected)return;
  var head=el('div','graph-card-head'),titles=el('div'),type=el('p','graph-card-type'),close=el('button','graph-card-close','×'),body=el('div','graph-card-body'),kind=graphKind(selected.type);
  type.append(dot(selected.color),el('span','',kind[1]));titles.append(type,el('h2','',selected.label));close.type='button';close.setAttribute('aria-label','Fermer');close.addEventListener('click',function(){select(null)});head.append(titles,close);
  if(selected.detail)body.append(el('p','',selected.detail));
  if(selected.href){var open=el('a','button secondary','Ouvrir');open.href=selected.href;body.append(open)}
  var linked=selected.links.filter(visible);if(linked.length){body.append(el('h3','','Relié à '+linked.length+(linked.length>1?' éléments':' élément')));var list=el('ul','graph-card-links');linked.forEach(function(n){list.append(nodeButton(n))});body.append(list)}
  card.append(head,body);
 }
 // The part of the stage that the floating controls and the open card leave free.
 function free(){var narrow=size.w<=760,cardOpen=!card.hidden;return {left:24,top:84,right:size.w-(cardOpen&&!narrow?380:24),bottom:size.h-(cardOpen&&narrow?size.h*.46+24:narrow?64:84)}}
 function select(n,centre){touched=true;selected=n;hint.hidden=true;drawCard();if(n&&centre)frame_(n.links.filter(visible).concat(n),.45,1.5);run()}
 function drawResults(){results.replaceChildren();results.hidden=!query;if(!query)return;filters.hidden=true;filterButton.setAttribute('aria-expanded','false');var found=nodes.filter(function(n){return visible(n)&&matches(n)}).sort(function(a,b){return a.tier-b.tier||b.links.length-a.links.length}).slice(0,14);if(!found.length){var none=el('li');none.append(el('p','graph-drop-empty','Aucun élément ne correspond.'));results.append(none);return}found.forEach(function(n){results.append(nodeButton(n))})}
 // Frame a set of elements inside the free part of the stage, labels included.
 function frame_(shown,least,most){if(!shown.length||!size.w)return;var minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9;shown.forEach(function(n){minX=Math.min(minX,n.x-n.r);maxX=Math.max(maxX,n.x+n.r);minY=Math.min(minY,n.y-n.r);maxY=Math.max(maxY,n.y+n.r)});var f=free(),w=Math.max(120,f.right-f.left-(size.w<=760?16:90)),h=Math.max(120,f.bottom-f.top-50);goal.k=Math.max(least,Math.min(most,Math.min(w/(maxX-minX+1),h/(maxY-minY+1))));goal.x=(minX+maxX)/2-((f.left+f.right)/2-size.w/2)/goal.k;goal.y=(minY+maxY)/2-((f.top+f.bottom)/2-size.h/2)/goal.k}
 function fit(){frame_(nodes.filter(visible),.06,1.6)}
 function zoomBy(factor){touched=true;goal.k=Math.max(.06,Math.min(4,goal.k*factor));run()}
 function step(){
  var shown=nodes.filter(visible),i,j,a,b,dx,dy,d2,f;
  for(i=0;i<shown.length;i++){a=shown[i];a.vx-=a.x*.014*alpha;a.vy-=a.y*.014*alpha;for(j=i+1;j<shown.length;j++){b=shown[j];dx=a.x-b.x;dy=a.y-b.y;d2=dx*dx+dy*dy||.01;if(d2>52900)continue;f=Math.min(260*a.q*b.q/d2,3/Math.sqrt(d2))*alpha;dx*=f;dy*=f;a.vx+=dx;a.vy+=dy;b.vx-=dx;b.vy-=dy}}
  edges.forEach(function(e){if(!visible(e.a)||!visible(e.b))return;var x=e.b.x-e.a.x,y=e.b.y-e.a.y,d=Math.sqrt(x*x+y*y)||1,want=e.a.type==='me'||e.b.type==='me'?170:e.a.type==='page'||e.b.type==='page'?80:52,k=(d-want)/d*.2*alpha;x*=k;y*=k;e.a.vx+=x;e.a.vy+=y;e.b.vx-=x;e.b.vy-=y});
  shown.forEach(function(n){if(n.fixed){n.vx=0;n.vy=0;return}n.vx*=.8;n.vy*=.8;n.x+=Math.max(-20,Math.min(20,n.vx));n.y+=Math.max(-20,Math.min(20,n.vy))});
  alpha*=.985;
 }
 function draw(){
  if(!size.w)return;var ratio=window.devicePixelRatio||1,k=cam.k;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,size.w,size.h);
  function sx(x){return (x-cam.x)*k+size.w/2}function sy(y){return (y-cam.y)*k+size.h/2}
  var focus=hovered||selected,near=focus?new Set(focus.links.concat(focus)):null;
  function strong(n){return near?near.has(n):matches(n)}
  ctx.lineCap='round';
  edges.forEach(function(e){if(!visible(e.a)||!visible(e.b))return;var on=focus?(e.a===focus||e.b===focus):matches(e.a)&&matches(e.b),trunk=e.a.type==='me'||e.b.type==='me';ctx.globalAlpha=on?(focus?.95:trunk?.75:.5):.07;ctx.strokeStyle=on&&focus?accent:line;ctx.lineWidth=on&&focus?1.8:trunk?1.6:1;ctx.beginPath();ctx.moveTo(sx(e.a.x),sy(e.a.y));ctx.lineTo(sx(e.b.x),sy(e.b.y));ctx.stroke()});
  var scale=Math.max(.55,Math.min(1.5,Math.sqrt(k)));
  nodes.forEach(function(n){if(!visible(n))return;var x=sx(n.x),y=sy(n.y),r=n.r*scale;n.sr=r;ctx.globalAlpha=strong(n)?1:.14;ctx.beginPath();ctx.arc(x,y,r,0,6.2832);ctx.fillStyle=n.color;ctx.fill();ctx.lineWidth=1.5;ctx.strokeStyle=paper;ctx.stroke();if(n===selected||n===hovered){ctx.globalAlpha=1;ctx.beginPath();ctx.arc(x,y,r+4,0,6.2832);ctx.lineWidth=2;ctx.strokeStyle=n===selected?accent:muted;ctx.stroke()}});
  // Labels by importance; one that would cover another is left out until the zoom makes room.
  var placed=[],wanted=nodes.filter(function(n){if(!visible(n)||!strong(n))return false;n.rank=n===focus?0:near&&near.has(n)?1:n.type==='me'?2:n.type==='page'?3:query&&matches(n)?4:5+n.tier;return n.rank<5||n.tier<=1&&k>=.75||k>=1.35}).sort(function(a,b){return a.rank-b.rank||b.links.length-a.links.length});
  ctx.textAlign='center';ctx.textBaseline='top';ctx.lineJoin='round';ctx.globalAlpha=1;
  wanted.forEach(function(n){var big=n.type==='me'||n.type==='page'||n===focus,px=big?13:12;ctx.font=(big?'600 ':'400 ')+px+'px Inter, system-ui, sans-serif';var text=n.label.length>30?n.label.slice(0,29)+'…':n.label,w=ctx.measureText(text).width,x=sx(n.x),y=sy(n.y)+n.sr+5,box=[x-w/2-4,y-2,x+w/2+4,y+px+4];if(box[2]<0||box[0]>size.w||box[3]<0||box[1]>size.h)return;if(n.rank>1&&placed.some(function(p){return box[0]<p[2]&&box[2]>p[0]&&box[1]<p[3]&&box[3]>p[1]}))return;placed.push(box);ctx.lineWidth=4;ctx.strokeStyle=paper;ctx.strokeText(text,x,y);ctx.fillStyle=n.rank<4?ink:muted;ctx.fillText(text,x,y)});
 }
 function run(){if(frame)return;frame=requestAnimationFrame(function tick(){frame=0;if(!canvas.isConnected)return;if(alpha>.02)step();cam.x+=(goal.x-cam.x)*.2;cam.y+=(goal.y-cam.y)*.2;cam.k+=(goal.k-cam.k)*.2;draw();var moving=Math.abs(goal.x-cam.x)*cam.k>.4||Math.abs(goal.y-cam.y)*cam.k>.4||Math.abs(goal.k-cam.k)>.002;if(alpha<=.02&&!settled){settled=true;if(!touched){fit();moving=true}}if(alpha>.02||moving)frame=requestAnimationFrame(tick);else{cam.x=goal.x;cam.y=goal.y;cam.k=goal.k;draw()}})}
 var ctx=canvas.getContext('2d');
 function resize(){var box=app.getBoundingClientRect(),ratio=window.devicePixelRatio||1;if(!box.width||!box.height)return;var first=!size.w;size={w:box.width,h:box.height};canvas.width=Math.round(box.width*ratio);canvas.height=Math.round(box.height*ratio);if(first){for(var i=0;i<420;i++){if(i<300)alpha=1;step()}alpha=.06;fit();cam.x=goal.x;cam.y=goal.y;cam.k=goal.k}draw();run()}
 function world(event){var box=canvas.getBoundingClientRect();return {x:(event.clientX-box.left-size.w/2)/cam.k+cam.x,y:(event.clientY-box.top-size.h/2)/cam.k+cam.y}}
 function pick(event){var box=canvas.getBoundingClientRect(),px=event.clientX-box.left,py=event.clientY-box.top,best=null,bestD=1e9,reach=event.pointerType==='touch'?22:8;nodes.forEach(function(n){if(!visible(n))return;var d=Math.hypot((n.x-cam.x)*cam.k+size.w/2-px,(n.y-cam.y)*cam.k+size.h/2-py)-(n.sr||n.r);if(d<reach&&d<bestD){best=n;bestD=d}});return best}
 var pointers=new Map(),grab=null,moved=false,pinch=0;
 canvas.addEventListener('pointerdown',function(event){touched=true;canvas.setPointerCapture(event.pointerId);pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});moved=false;results.hidden=true;if(pointers.size===1){grab=pick(event);if(grab)grab.fixed=true;canvas.classList.add('is-dragging')}else{if(grab)grab.fixed=false;grab=null;var pts=Array.from(pointers.values());pinch=Math.hypot(pts[0].x-pts[1].x,pts[0].y-pts[1].y)}});
 canvas.addEventListener('pointermove',function(event){var last=pointers.get(event.pointerId);if(!last){var over=pick(event);if(over!==hovered){hovered=over;canvas.style.cursor=over?'pointer':'';draw()}return}
  var dx=event.clientX-last.x,dy=event.clientY-last.y;if(Math.abs(dx)+Math.abs(dy)>4)moved=true;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
  if(pointers.size===2){var pts=Array.from(pointers.values()),dist=Math.hypot(pts[0].x-pts[1].x,pts[0].y-pts[1].y);if(pinch){goal.k=cam.k=Math.max(.06,Math.min(4,cam.k*dist/pinch))}pinch=dist;draw();return}
  if(grab&&moved){var p=world(event);grab.x=p.x;grab.y=p.y;alpha=Math.max(alpha,.25);run()}else if(!grab){cam.x-=dx/cam.k;cam.y-=dy/cam.k;goal.x=cam.x;goal.y=cam.y;draw()}});
 function release(event){if(!pointers.has(event.pointerId))return;pointers.delete(event.pointerId);canvas.classList.remove('is-dragging');if(grab){grab.fixed=false;if(!moved)select(grab,true)}else if(!moved&&pointers.size===0){var hit=pick(event);if(hit)select(hit,true);else if(selected)select(null)}grab=null;pinch=0}
 canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);canvas.addEventListener('pointerleave',function(){if(hovered){hovered=null;draw()}});
 canvas.addEventListener('wheel',function(event){touched=true;event.preventDefault();var before=world(event);cam.k=goal.k=Math.max(.06,Math.min(4,cam.k*Math.exp(-event.deltaY*.0016)));var after=world(event);cam.x+=before.x-after.x;cam.y+=before.y-after.y;goal.x=cam.x;goal.y=cam.y;draw()},{passive:false});
 search.addEventListener('input',function(){query=normalized(search.value.trim());drawResults();draw()});
 search.addEventListener('focus',function(){if(query)results.hidden=false});
 app.addEventListener('keydown',function(event){if(event.key==='Escape'){if(!results.hidden||!filters.hidden){results.hidden=true;filters.hidden=true;filterButton.setAttribute('aria-expanded','false')}else if(selected)select(null)}});
 if(nodes.length<=7){var empty=el('div','graph-empty'),start=el('a','button','Ajouter un proche');start.href='/mes-proches/';empty.append(el('h2','','Votre cartographie est encore vide'),el('p','','Elle se dessine toute seule à mesure que vous remplissez vos pages : vos proches, vos moments, votre enfance, vos symptômes et vos sujets.'),start);app.append(empty);hint.hidden=true}
 if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe(app);else window.addEventListener('resize',resize);
 resize();
}
`;
