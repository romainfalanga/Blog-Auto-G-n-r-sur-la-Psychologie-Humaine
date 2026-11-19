const test=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const source=readFileSync(path.join(root,'static/js/character-search.js'),'utf8');
function fixture({lazy=null}={}){
  const nodes=new Map();let document;const created=[];
  function node(id){const n={id,children:[],listeners:{},hidden:false,value:'',attributes:{},dataset:{},classList:{add(){}},addEventListener(type,fn){this.listeners[type]=fn},setAttribute(key,value){this.attributes[key]=value},append(child){this.children=this.children.filter(x=>x!==child);this.children.push(child)},replaceChildren(){this.children=[]},focus(){document.activeElement=this;this.listeners.focus?.({})},querySelector(selector){return selector==='button'?this.children[0]:this.match},querySelectorAll(selector){return this.children.filter(child=>child.attributes[selector.slice(1,-1)]!==undefined)},fire(type,event={}){this.listeners[type]?.({...event,preventDefault(){event.prevented=true}})}};nodes.set(id,n);return n}
  const input=node('character-search'),index=lazy?null:node('character-search-index'),grid=node('character-results');
  const entries=[{id:'camille',name:'Camille',keywords:['Rumination'],profile:'pensées en boucle',stories:''},{id:'thomas',name:'Thomas',keywords:['Peur de décevoir'],profile:'Peur de décevoir',stories:''}];
  for(const entry of entries){
    const card=node(entry.id);card.dataset.characterId=entry.id;card.setAttribute('data-character-id',entry.id);card.link={href:'/personnages/'+entry.id+'/'};grid.append(card);
  }
  if(index)index.textContent=JSON.stringify(entries);else node('character-search-config').textContent=JSON.stringify({index:'/js/character-search-index.abc.json'});
  node('character-search-clear');node('character-count');node('character-search-empty');
  document={activeElement:null,getElementById:id=>nodes.get(id),createElement:tag=>{const element=node('new-'+created.length);created.push(element);return element}};
  const window={document};if(lazy)window.fetch=(url,options)=>lazy(url,options,entries);
  vm.runInNewContext(source,{window,document,module:{exports:{}},Set,Map});
  return {nodes,input,grid,document,created};
}
test('focusing and typing only filter cards, without creating a suggestions popup',()=>{
  const f=fixture();const cards=f.grid.children.slice();f.input.focus();
  assert.equal(f.document.activeElement,f.input);assert.deepEqual(f.grid.children,cards);assert.equal(f.created.length,0);
  f.input.value='pensées en boucle';f.input.fire('input');
  assert.equal(f.nodes.get('camille').hidden,false);assert.equal(f.nodes.get('thomas').hidden,true);assert.equal(f.created.length,0);
  assert.equal(f.nodes.get('character-count').textContent,'1 personnage pour cette recherche');
  assert.equal(f.input.attributes['aria-expanded'],undefined);
});
test('a matching card retains its native character link and receives no click interception',()=>{
  const f=fixture();const card=f.nodes.get('thomas'),link=card.link;f.input.focus();f.input.value='peur décevoir';f.input.fire('input');
  assert.equal(card.hidden,false);assert.equal(f.nodes.get('camille').hidden,true);assert.equal(card.link,link);assert.equal(link.href,'/personnages/thomas/');
  assert.equal(card.listeners.mousedown,undefined);assert.equal(card.listeners.click,undefined);
});
test('clearing a query restores the original cards and order without recreating them',()=>{
  const f=fixture();const cards=f.grid.children.slice();f.input.value='Peur de décevoir';f.input.fire('input');
  assert.equal(f.nodes.get('character-search-clear').hidden,false);f.nodes.get('character-search-clear').fire('click');
  assert.equal(f.input.value,'');assert.deepEqual(f.grid.children,cards);assert.ok(cards.every(card=>!card.hidden));assert.equal(f.nodes.get('character-search-clear').hidden,true);assert.equal(f.created.length,0);
});
test('the accessible search controls only the results and contains no suggestions markup',()=>{
  const template=readFileSync(path.join(root,'layouts/partials/character-directory.html'),'utf8');
  const css=readFileSync(path.join(root,'static/css/style.css'),'utf8');
  assert.match(template,/placeholder="Peur de décevoir, pensées en boucle…"/);assert.match(template,/aria-label="Rechercher un personnage, un trouble, une émotion ou un thème"/);
  assert.match(template,/aria-controls="character-results"/);
  assert.doesNotMatch(template,/character-search-suggestions|aria-expanded|aria-haspopup|role="combobox"/);
  assert.doesNotMatch(template,/character-group-links|character-group-heading/);
  assert.doesNotMatch(css,/\.character-search-suggestions/);
});

test('Escape prevents the native search-field clearing and retains the active query',()=>{
  const f=fixture();f.input.focus();f.input.value='rumination';f.input.fire('input');const event={key:'Escape'};f.input.fire('keydown',event);
  assert.equal(event.prevented,true);assert.equal(f.input.value,'rumination');assert.equal(f.nodes.get('camille').hidden,false);assert.equal(f.nodes.get('thomas').hidden,true);assert.equal(f.created.length,0);
});

test('an unmatched query reports no results and clearing it restores discovery',()=>{
  const f=fixture();f.input.value='zzzzzzz';f.input.fire('input');assert.equal(f.nodes.get('character-search-empty').hidden,false);assert.ok(f.grid.children.every(card=>card.hidden));
  f.nodes.get('character-search-clear').fire('click');assert.equal(f.nodes.get('character-search-empty').hidden,true);assert.equal(f.nodes.get('character-count').textContent,'2 personnages');
});

test('lot P: the index is fetched once, on focus or first keystroke, and a query typed meanwhile is applied on arrival',async()=>{
  const requests=[];let deliver;
  const f=fixture({lazy:(url,options,entries)=>{requests.push([url,options]);return new Promise(resolve=>{deliver=()=>resolve({ok:true,json:async()=>entries})})}});
  const cards=f.grid.children.slice();
  assert.equal(requests.length,0,'nothing is downloaded with the page');
  assert.ok(cards.every(card=>!card.hidden));
  f.input.focus();assert.equal(requests.length,1);assert.equal(requests[0][0],'/js/character-search-index.abc.json');
  f.input.value='pensées en boucle';f.input.fire('input');assert.equal(requests.length,1,'a pending download is reused');
  assert.ok(cards.every(card=>!card.hidden),'cards stay visible until the index arrives');assert.equal(f.nodes.get('character-search-clear').hidden,false);
  deliver();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.nodes.get('camille').hidden,false);assert.equal(f.nodes.get('thomas').hidden,true);
  assert.equal(f.nodes.get('character-count').textContent,'1 personnage pour cette recherche');
  f.input.value='';f.input.fire('input');f.input.focus();assert.equal(requests.length,1);assert.ok(cards.every(card=>!card.hidden));
});

test('lot P: a failed index download keeps every card and is retried on the next keystroke',async()=>{
  let calls=0;
  const f=fixture({lazy:async(url,options,entries)=>{calls++;if(calls===1)return {ok:false,status:503};return {ok:true,json:async()=>entries}}});
  f.input.value='Peur de décevoir';f.input.fire('input');await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls,1);assert.ok(f.grid.children.every(card=>!card.hidden));
  f.input.fire('input');await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls,2);assert.equal(f.nodes.get('thomas').hidden,false);assert.equal(f.nodes.get('camille').hidden,true);
});

test('lot P: the directory page links a content-addressed index instead of embedding it',()=>{
  const template=readFileSync(path.join(root,'layouts/partials/character-directory.html'),'utf8');
  assert.doesNotMatch(template,/id="character-search-index"/);
  assert.match(template,/resources\.FromString \(printf "js\/character-search-index\.%s\.json" \(md5 \$indexJSON\)\)/);
  assert.match(template,/id="character-search-config"/);
});
