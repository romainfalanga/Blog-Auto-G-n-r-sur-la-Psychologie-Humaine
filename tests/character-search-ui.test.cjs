const test=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const source=readFileSync(path.join(root,'static/js/character-search.js'),'utf8');
function fixture(){
  const nodes=new Map();let document;
  function node(id){const n={id,children:[],listeners:{},hidden:false,value:'',attributes:{},dataset:{},classList:{add(){}},addEventListener(type,fn){this.listeners[type]=fn},setAttribute(key,value){this.attributes[key]=value},append(child){this.children=this.children.filter(x=>x!==child);this.children.push(child)},replaceChildren(){this.children=[]},focus(){document.activeElement=this;this.listeners.focus?.({})},querySelector(selector){return selector==='button'?this.children[0]:this.match},querySelectorAll(){return this.children},fire(type,event={}){this.listeners[type]?.({...event,preventDefault(){event.prevented=true}})}};nodes.set(id,n);return n}
  const input=node('character-search'),index=node('character-search-index'),grid=node('character-results'),panel=node('character-search-suggestions'),box=node('box');
  const entries=[{id:'camille',name:'Camille',keywords:['Rumination'],profile:'pensées en boucle',stories:''},{id:'thomas',name:'Thomas',keywords:['Peur de décevoir'],profile:'Peur de décevoir',stories:''}];
  for(const entry of entries){const card=node(entry.id);card.dataset.characterId=entry.id;card.match=node(entry.id+'-match');grid.append(card)}
  index.textContent=JSON.stringify(entries);input.closest=()=>box;box.contains=target=>target===input||target===nodes.get('character-search-clear')||panel.children.includes(target);
  node('character-search-clear');node('character-count');node('character-search-empty');
  document={activeElement:null,getElementById:id=>nodes.get(id),createElement:tag=>node('new-'+Math.random())};
  vm.runInNewContext(source,{window:{document},document,module:{exports:{}},Set,Map});
  return {nodes,input,panel,box,grid,document};
}
test('suggestions can be reached with the keyboard and closed without changing search results',()=>{
  const f=fixture();f.input.focus();assert.equal(f.panel.hidden,false);assert.equal(f.input.attributes['aria-expanded'],'true');
  f.input.fire('keydown',{key:'ArrowDown'});assert.equal(f.document.activeElement,f.panel.children[0]);
  f.panel.fire('keydown',{key:'Escape'});assert.equal(f.document.activeElement,f.input);assert.equal(f.panel.hidden,true);assert.equal(f.input.attributes['aria-expanded'],'false');assert.equal(f.grid.children.length,2);
});
test('choosing a suggestion closes its overlay and keeps the selected result available',()=>{
  const f=fixture();f.input.focus();const suggestion=f.panel.children.find(x=>x.textContent==='Peur de décevoir');f.document.activeElement=suggestion;suggestion.fire('click');
  assert.equal(f.input.value,'Peur de décevoir');assert.equal(f.panel.hidden,true);assert.equal(f.nodes.get('thomas').hidden,false);assert.equal(f.nodes.get('camille').hidden,true);
  assert.equal(f.nodes.get('thomas').listeners.mousedown,undefined);assert.equal(f.nodes.get('thomas').listeners.click,undefined);
});
test('leaving the search closes the panel without rebuilding or moving the card DOM',()=>{
  const f=fixture();f.input.focus();const cards=f.grid.children.slice();f.box.fire('focusout',{relatedTarget:f.nodes.get('camille')});assert.equal(f.panel.hidden,true);assert.deepEqual(f.grid.children,cards);
});
test('the search label stays accessible and the suggestions are removed from document flow',()=>{
  const template=readFileSync(path.join(root,'layouts/partials/character-directory.html'),'utf8');
  const css=readFileSync(path.join(root,'static/css/style.css'),'utf8');
  assert.match(template,/placeholder="Peur de décevoir, pensées en boucle…"/);assert.match(template,/aria-label="Rechercher un personnage, une émotion ou un thème"/);
  assert.doesNotMatch(template,/character-directory-title|character-search-help|<label/);
  assert.match(css,/\.character-search-box\s*\{[^}]*position:relative/);assert.match(css,/\.character-search-suggestions\s*\{[^}]*position:absolute/);assert.match(css,/\.character-search-suggestions\s*\{[^}]*overflow-y:auto/);
});

test('Escape prevents the native search-field clearing and retains the active query',()=>{
  const f=fixture();f.input.focus();f.input.value='rumination';f.input.fire('input');const event={key:'Escape'};f.input.fire('keydown',event);
  assert.equal(event.prevented,true);assert.equal(f.input.value,'rumination');assert.equal(f.panel.hidden,true);assert.equal(f.nodes.get('camille').hidden,false);assert.equal(f.nodes.get('thomas').hidden,true);
});
