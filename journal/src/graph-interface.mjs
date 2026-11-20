// « Ma cartographie »: the personal graph drawn like a note graph — nodes sized by their links,
// coloured by kind, pulled together by their links; pan, zoom, drag, search, filters by kind, and a
// panel listing what a selected element is linked to. Uses the shared helpers of ui.mjs.
export const GRAPH_INTERFACE_STYLE=String.raw`
#view > .graph-page { max-width: none; }
.graph-intro { max-width: 70ch; margin: var(--s4) 0 0; color: var(--muted); font-size: var(--t-sm); line-height: 1.6; }
.graph-filters { margin: var(--s4) 0 0; }
.graph-filters > summary { display: inline-flex; align-items: center; min-height: 44px; color: var(--accent); font: 600 var(--t-sm)/1.4 var(--font-ui); cursor: pointer; }
.graph-tools { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s3); margin: 0 0 var(--s3); }
.graph-tools input[type=search] { flex: 1 1 220px; min-width: 0; min-height: 48px; }
.graph-tools button { min-height: 48px; }
.graph-kinds { display: flex; flex-wrap: wrap; gap: var(--s2); margin: var(--s2) 0 0; padding: 0; list-style: none; }
.graph-kind { display: inline-flex; align-items: center; gap: 6px; min-height: 44px; padding: 0 var(--s3); border: 1px solid var(--line); border-radius: 22px; background: var(--surface); color: var(--ink); font: 400 var(--t-sm)/1.2 var(--font-ui); cursor: pointer; }
.graph-kind[aria-pressed="false"] { background: transparent; color: var(--muted); text-decoration: line-through; }
.graph-kind i { width: 10px; height: 10px; border-radius: 50%; }
.graph-layout { display: grid; gap: var(--s4); }
@media (min-width: 900px) { .graph-layout { grid-template-columns: minmax(0, 1fr) 320px; align-items: start; } }
.graph-stage { position: relative; height: min(68dvh, 720px); min-height: 340px; border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); overflow: hidden; touch-action: none; }
.graph-stage canvas { display: block; width: 100%; height: 100%; cursor: grab; }
.graph-stage canvas.is-dragging { cursor: grabbing; }
.graph-panel { padding: var(--s4); border: 1px solid var(--line); border-radius: var(--r-card); background: var(--surface); }
.graph-panel h2 { margin: 0 0 var(--s1); font: 600 var(--t-md)/1.35 var(--font-ui); overflow-wrap: anywhere; }
.graph-panel .graph-type { display: inline-flex; align-items: center; gap: 6px; margin: 0 0 var(--s2); color: var(--muted); font-size: var(--t-xs); }
.graph-panel .graph-type i { width: 10px; height: 10px; border-radius: 50%; }
.graph-panel p { margin: 0 0 var(--s3); line-height: 1.55; overflow-wrap: anywhere; }
.graph-panel h3 { margin: var(--s3) 0 var(--s1); color: var(--muted); font: 600 var(--t-xs)/1.4 var(--font-ui); }
.graph-links { margin: 0; padding: 0; list-style: none; max-height: 320px; overflow-y: auto; }
.graph-links button { display: flex; align-items: center; justify-content: flex-start; gap: var(--s2); width: 100%; min-height: 44px; padding: var(--s1) 0; border: 0; border-top: 1px solid var(--line); background: none; color: var(--ink); font: inherit; font-size: var(--t-sm); text-align: left; cursor: pointer; }
.graph-links button i { flex: none; width: 10px; height: 10px; border-radius: 50%; }
.graph-results { margin: 0 0 var(--s3); }
`;
export const GRAPH_INTERFACE_SCRIPT=String.raw`
var GRAPH_KINDS=[['me','Moi','#1F5C4D'],['page','Pages','#334155'],['person','Proches','#2563EB'],['event','Moments','#B4530A'],['note','Notes','#94A3B8'],['childhood','Enfance','#0E7490'],['theme','Thèmes de symptômes','#BE185D'],['symptom','Symptômes','#E0679A'],['past','Symptômes passés','#C4A1B1'],['topic','Sujets','#7C3AED'],['position','Positions','#A78BFA'],['support','Raisons','#15803D'],['objection','Objections','#B91C1C'],['review','Bilans','#CA8A04']];
function graphKind(type){return GRAPH_KINDS.find(function(row){return row[0]===type})||GRAPH_KINDS[1]}
async function renderGraph(){
 var view=document.getElementById('view'),current=renderGuard(view),data;
 try{data=await api('/api/graph')}catch(error){if(current())view.replaceChildren(el('div','notice warn',error.message||'La cartographie n’a pas pu être chargée.'),actionButton('Réessayer',function(){renderApp()},'secondary'));return}
 if(!current())return;
 var nodes=data.nodes.map(function(n,i){var a=i*2.399963,r=30+12*Math.sqrt(i);return Object.assign({},n,{x:Math.cos(a)*r,y:Math.sin(a)*r,vx:0,vy:0,links:[]})}),byId=new Map(nodes.map(function(n){return [n.id,n]}));
 var edges=data.edges.map(function(e){return {a:byId.get(e.source),b:byId.get(e.target)}}).filter(function(e){return e.a&&e.b&&e.a!==e.b});
 edges.forEach(function(e){e.a.links.push(e.b);e.b.links.push(e.a)});
 nodes.forEach(function(n){n.q=n.type==='me'||n.type==='page'?3.2:n.type==='theme'||n.type==='topic'?1.6:1;n.r=n.type==='me'?13:n.type==='page'?9:Math.min(11,4+Math.sqrt(n.links.length)*1.6)});
 var hidden=new Set(),selected=null,hovered=null,query='',cam={x:0,y:0,k:1},alpha=1,frame=0,size={w:0,h:0};
 var page=el('div','graph-page'),tools=el('div','graph-tools'),search=el('input'),recenter=actionButton('Recentrer',function(){fit();draw()},'secondary'),kinds=el('ul','graph-kinds'),layout=el('div','graph-layout'),stage=el('div','graph-stage'),canvas=el('canvas'),panel=el('aside','graph-panel'),results=el('ul','graph-links graph-results');
 search.type='search';search.placeholder='Rechercher dans ma cartographie';search.setAttribute('aria-label','Rechercher dans ma cartographie');
 canvas.setAttribute('role','img');canvas.setAttribute('aria-label','Cartographie de vos informations : '+nodes.length+' éléments et '+edges.length+' liens. La recherche et la liste des liens permettent de la parcourir au clavier.');
 tools.append(search,recenter);stage.append(canvas);panel.setAttribute('aria-live','polite');layout.append(stage,panel);
 var present=new Set(nodes.map(function(n){return n.type}));
 GRAPH_KINDS.filter(function(row){return present.has(row[0])&&row[0]!=='me'}).forEach(function(row){var item=el('li'),button=el('button','graph-kind'),dot=el('i');button.type='button';dot.style.background=row[2];button.append(dot,el('span','',row[1]));button.setAttribute('aria-pressed','true');
  button.addEventListener('click',function(){var on=hidden.has(row[0]);if(on)hidden.delete(row[0]);else hidden.add(row[0]);button.setAttribute('aria-pressed',String(on));if(selected&&!visible(selected))select(null);else drawPanel();drawResults();alpha=Math.max(alpha,.5);run()});item.append(button);kinds.append(item)});
 var filters=el('details','graph-filters');filters.append(el('summary','','Choisir ce qui est affiché'),kinds);
 page.append(tools,results,layout,filters,el('p','graph-intro','Tout ce que vous avez renseigné, relié en un seul réseau. Déplacez la carte, zoomez, et touchez un point pour voir ce qu’il contient et à quoi il est lié.'));
 view.replaceChildren(page);
 var ctx=canvas.getContext('2d'),ink=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim()||'#1B2733',line=getComputedStyle(document.documentElement).getPropertyValue('--line-strong').trim()||'#B8C2CC';
 function visible(n){return !hidden.has(n.type)}
 function matches(n){return !query||normalized(n.label+' '+(n.detail||'')).indexOf(query)>=0}
 function dot(row){var i=el('i');i.style.background=row[2];return i}
 function nodeButton(n){var item=el('li'),button=el('button');button.type='button';button.append(dot(graphKind(n.type)),el('span','',n.label));button.addEventListener('click',function(){select(n,true)});item.append(button);return item}
 function drawPanel(){
  panel.replaceChildren();
  if(!selected){panel.append(el('h2','','Votre cartographie'),el('p','',nodes.length+' éléments et '+edges.length+' liens.'+(data.truncated?' Seuls les premiers éléments sont affichés.':'')),el('p','helper','Touchez un point pour voir ce qu’il contient et à quoi il est lié.'));return}
  var kind=graphKind(selected.type),type=el('p','graph-type');type.append(dot(kind),el('span','',kind[1]));panel.append(type,el('h2','',selected.label));
  if(selected.detail)panel.append(el('p','',selected.detail));
  if(selected.href){var open=el('a','button secondary','Ouvrir');open.href=selected.href;panel.append(open)}
  var linked=selected.links.filter(visible);panel.append(el('h3','',linked.length?'Relié à ('+linked.length+')':'Aucun lien affiché'));
  if(linked.length){var list=el('ul','graph-links');linked.forEach(function(n){list.append(nodeButton(n))});panel.append(list)}
 }
 function select(n,focusIt){selected=n;drawPanel();if(n&&focusIt){cam.x=n.x;cam.y=n.y;cam.k=Math.max(cam.k,1.4);panel.scrollIntoView({block:'nearest'})}draw()}
 function drawResults(){results.replaceChildren();if(!query)return;var found=nodes.filter(function(n){return visible(n)&&matches(n)}).slice(0,12);if(!found.length){var none=el('li');none.append(el('span','helper','Aucun élément ne correspond.'));results.append(none);return}found.forEach(function(n){results.append(nodeButton(n))})}
 function fit(){var shown=nodes.filter(visible);if(!shown.length||!size.w)return;var minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9;shown.forEach(function(n){minX=Math.min(minX,n.x);maxX=Math.max(maxX,n.x);minY=Math.min(minY,n.y);maxY=Math.max(maxY,n.y)});cam.x=(minX+maxX)/2;cam.y=(minY+maxY)/2;cam.k=Math.max(.15,Math.min(2.5,Math.min(size.w/(maxX-minX+80),size.h/(maxY-minY+80))))}
 function step(){
  var shown=nodes.filter(visible),i,j,a,b,dx,dy,d2,f;
  for(i=0;i<shown.length;i++){a=shown[i];a.vx-=a.x*.012*alpha;a.vy-=a.y*.012*alpha;for(j=i+1;j<shown.length;j++){b=shown[j];dx=a.x-b.x;dy=a.y-b.y;d2=dx*dx+dy*dy||.01;if(d2>160000)continue;f=900*a.q*b.q/d2*alpha;dx*=f;dy*=f;a.vx+=dx;a.vy+=dy;b.vx-=dx;b.vy-=dy}}
  edges.forEach(function(e){if(!visible(e.a)||!visible(e.b))return;var x=e.b.x-e.a.x,y=e.b.y-e.a.y,d=Math.sqrt(x*x+y*y)||1,want=e.a.type==='me'||e.b.type==='me'?190:e.a.type==='page'||e.b.type==='page'?75:48,k=(d-want)/d*.09*alpha;x*=k;y*=k;e.a.vx+=x;e.a.vy+=y;e.b.vx-=x;e.b.vy-=y});
  shown.forEach(function(n){if(n.fixed){n.vx=0;n.vy=0;return}n.vx*=.82;n.vy*=.82;n.x+=Math.max(-25,Math.min(25,n.vx));n.y+=Math.max(-25,Math.min(25,n.vy))});
  alpha*=.985;
 }
 function draw(){
  if(!size.w)return;var ratio=window.devicePixelRatio||1;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,size.w,size.h);ctx.translate(size.w/2,size.h/2);ctx.scale(cam.k,cam.k);ctx.translate(-cam.x,-cam.y);
  var focus=hovered||selected,near=focus?new Set(focus.links.concat(focus)):null;
  function strong(n){return near?near.has(n):matches(n)}
  ctx.lineWidth=1/cam.k;edges.forEach(function(e){if(!visible(e.a)||!visible(e.b))return;var on=focus?(e.a===focus||e.b===focus):matches(e.a)&&matches(e.b);ctx.globalAlpha=on?(focus?.9:.45):.08;ctx.strokeStyle=on&&focus?ink:line;ctx.beginPath();ctx.moveTo(e.a.x,e.a.y);ctx.lineTo(e.b.x,e.b.y);ctx.stroke()});
  nodes.forEach(function(n){if(!visible(n))return;ctx.globalAlpha=strong(n)?1:.15;ctx.fillStyle=graphKind(n.type)[2];ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,6.2832);ctx.fill();if(n===selected){ctx.lineWidth=2.5/cam.k;ctx.strokeStyle=ink;ctx.stroke();ctx.lineWidth=1/cam.k}});
  ctx.fillStyle=ink;ctx.textAlign='center';ctx.textBaseline='top';
  nodes.forEach(function(n){if(!visible(n)||!strong(n))return;var always=n.type==='me'||n.type==='page'||n===focus||near&&near.has(n)||query&&matches(n);if(!always&&cam.k<1.25)return;if(!always&&n.r*cam.k<5)return;ctx.globalAlpha=1;ctx.font=(n.type==='me'||n.type==='page'?'600 ':'400 ')+(12/cam.k)+'px Inter, system-ui, sans-serif';var text=n.label.length>34?n.label.slice(0,33)+'…':n.label;ctx.fillText(text,n.x,n.y+n.r+3/cam.k)});
  ctx.globalAlpha=1;
 }
 function run(){if(frame)return;frame=requestAnimationFrame(function tick(){frame=0;if(!canvas.isConnected)return;step();draw();if(alpha>.02)frame=requestAnimationFrame(tick)})}
 function resize(){var box=stage.getBoundingClientRect(),ratio=window.devicePixelRatio||1;if(!box.width)return;var first=!size.w;size={w:box.width,h:box.height};canvas.width=Math.round(box.width*ratio);canvas.height=Math.round(box.height*ratio);if(first){for(var i=0;i<220;i++)step();alpha=.25;fit()}draw()}
 function world(event){var box=canvas.getBoundingClientRect();return {x:(event.clientX-box.left-size.w/2)/cam.k+cam.x,y:(event.clientY-box.top-size.h/2)/cam.k+cam.y}}
 function pick(event){var p=world(event),best=null,bestD=1e9,reach=(event.pointerType==='touch'?22:10)/cam.k;nodes.forEach(function(n){if(!visible(n))return;var d=Math.hypot(n.x-p.x,n.y-p.y)-n.r;if(d<reach&&d<bestD){best=n;bestD=d}});return best}
 var pointers=new Map(),grab=null,moved=false,pinch=0;
 canvas.addEventListener('pointerdown',function(event){canvas.setPointerCapture(event.pointerId);pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});moved=false;if(pointers.size===1){grab=pick(event);if(grab)grab.fixed=true;canvas.classList.add('is-dragging')}else{if(grab)grab.fixed=false;grab=null;var pts=Array.from(pointers.values());pinch=Math.hypot(pts[0].x-pts[1].x,pts[0].y-pts[1].y)}});
 canvas.addEventListener('pointermove',function(event){var last=pointers.get(event.pointerId);if(!last){var over=pick(event);if(over!==hovered){hovered=over;canvas.style.cursor=over?'pointer':'';draw()}return}
  var dx=event.clientX-last.x,dy=event.clientY-last.y;if(Math.abs(dx)+Math.abs(dy)>3)moved=true;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
  if(pointers.size===2){var pts=Array.from(pointers.values()),dist=Math.hypot(pts[0].x-pts[1].x,pts[0].y-pts[1].y);if(pinch)cam.k=Math.max(.15,Math.min(4,cam.k*dist/pinch));pinch=dist;draw();return}
  if(grab){var p=world(event);grab.x=p.x;grab.y=p.y;alpha=Math.max(alpha,.3);run()}else{cam.x-=dx/cam.k;cam.y-=dy/cam.k;draw()}});
 function release(event){if(!pointers.has(event.pointerId))return;pointers.delete(event.pointerId);canvas.classList.remove('is-dragging');if(grab){grab.fixed=false;if(!moved)select(grab)}else if(!moved&&pointers.size===0)select(pick(event));grab=null;pinch=0}
 canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);canvas.addEventListener('pointerleave',function(){if(hovered){hovered=null;draw()}});
 canvas.addEventListener('wheel',function(event){event.preventDefault();var before=world(event);cam.k=Math.max(.15,Math.min(4,cam.k*Math.exp(-event.deltaY*.0015)));var after=world(event);cam.x+=before.x-after.x;cam.y+=before.y-after.y;draw()},{passive:false});
 search.addEventListener('input',function(){query=normalized(search.value.trim());drawResults();draw()});
 if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe(stage);else window.addEventListener('resize',resize);
 drawPanel();resize();run();
}
`;
