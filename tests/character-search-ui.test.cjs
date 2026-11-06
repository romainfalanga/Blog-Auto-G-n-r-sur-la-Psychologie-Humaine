const test=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const source=readFileSync(path.join(root,'static/js/character-search.js'),'utf8');
function fixture(){
  const nodes=new Map();let document;const created=[];
  function node(id){const n={id,children:[],listeners:{},hidden:false,value:'',attributes:{},dataset:{},classList:{add(){}},addEventListener(type,fn){this.listeners[type]=fn},setAttribute(key,value){this.attributes[key]=value},append(child){this.children=this.children.filter(x=>x!==child);this.children.push(child)},replaceChildren(){this.children=[]},focus(){document.activeElement=this;this.listeners.focus?.({})},querySelector(selector){return selector==='button'?this.children[0]:this.match},querySelectorAll(selector){return this.children.filter(child=>child.attributes[selector.slice(1,-1)]!==undefined)},fire(type,event={}){this.listeners[type]?.({...event,preventDefault(){event.prevented=true}})}};nodes.set(id,n);return n}
  const input=node('character-search'),index=node('character-search-index'),grid=node('character-results');
  const entries=[{id:'camille',name:'Camille',keywords:['Rumination'],profile:'pensées en boucle',stories:''},{id:'thomas',name:'Thomas',keywords:['Peur de décevoir'],profile:'Peur de décevoir',stories:''}];
  for(const entry of entries){
    const heading=node('group-'+entry.id);heading.setAttribute('data-character-group-heading','');grid.append(heading);
    const card=node(entry.id);card.dataset.characterId=entry.id;card.setAttribute('data-character-id',entry.id);card.link={href:'/personnages/'+entry.id+'/'};grid.append(card);
  }
  index.textContent=JSON.stringify(entries);
  node('character-search-clear');node('character-count');node('character-search-empty');node('character-group-links');
  document={activeElement:null,getElementById:id=>nodes.get(id),createElement:tag=>{const element=node('new-'+created.length);created.push(element);return element}};
  vm.runInNewContext(source,{window:{document},document,module:{exports:{}},Set,Map});
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
  assert.equal(f.nodes.get('character-group-links').hidden,true);
  assert.equal(f.nodes.get('group-thomas').hidden,true);
  assert.equal(f.nodes.get('character-search-clear').hidden,false);f.nodes.get('character-search-clear').fire('click');
  assert.equal(f.input.value,'');assert.deepEqual(f.grid.children,cards);assert.ok(cards.every(card=>!card.hidden));assert.equal(f.nodes.get('character-search-clear').hidden,true);assert.equal(f.created.length,0);
  assert.equal(f.nodes.get('character-group-links').hidden,false);
});
test('the accessible search controls only the results and contains no suggestions markup',()=>{
  const template=readFileSync(path.join(root,'layouts/partials/character-directory.html'),'utf8');
  const css=readFileSync(path.join(root,'static/css/style.css'),'utf8');
  assert.match(template,/placeholder="Peur de décevoir, pensées en boucle…"/);assert.match(template,/aria-label="Rechercher un personnage, un trouble, une émotion ou un thème"/);
  assert.match(template,/aria-controls="character-results"/);
  assert.doesNotMatch(template,/character-search-suggestions|aria-expanded|aria-haspopup|role="combobox"/);
  assert.doesNotMatch(css,/\.character-search-suggestions/);
});

test('Escape prevents the native search-field clearing and retains the active query',()=>{
  const f=fixture();f.input.focus();f.input.value='rumination';f.input.fire('input');const event={key:'Escape'};f.input.fire('keydown',event);
  assert.equal(event.prevented,true);assert.equal(f.input.value,'rumination');assert.equal(f.nodes.get('camille').hidden,false);assert.equal(f.nodes.get('thomas').hidden,true);assert.equal(f.created.length,0);
});

test('an unmatched query reports no results and clearing it restores discovery',()=>{
  const f=fixture();f.input.value='zzzzzzz';f.input.fire('input');assert.equal(f.nodes.get('character-search-empty').hidden,false);assert.ok(f.grid.children.every(card=>card.hidden));
  f.nodes.get('character-search-clear').fire('click');assert.equal(f.nodes.get('character-search-empty').hidden,true);assert.equal(f.nodes.get('character-count').textContent,'2 personnages à explorer');
});
