// Plan v5.1, lot C5a: contextual subjects hidden, variant L ranking with strength
// guard, sections shared by the page and Ambre, frozen contract for the Ambre lots.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {computeCartography,summarizeCartography,cartographyPromptBlock,cartographyIndexEntries,cartographyEvidence,chooseAmbreNotes,SECTION_ORDER,SECTION_TOP} from '../src/cartographie.mjs';
const require=createRequire(import.meta.url);
const publicPage=require('../../static/js/cartographie.js');
const fullData=JSON.parse(readFileSync(new URL('../../data/cartographie.json',import.meta.url),'utf8'));
const ids=CARTOGRAPHIE.factors.map(f=>f.id);
const outcomeIds=CARTOGRAPHIE.outcomes.map(o=>o.id);
const contextual=new Set(CARTOGRAPHIE.outcomes.filter(o=>o.contextual_only).map(o=>o.id));
const GENERIC=new Set(['dep_generale','anx_generale']);

function rng(seed){let state=seed;return ()=>(state=(state*1103515245+12345)%2147483648)/2147483648;}
function profile(rand,size){
 const answers={},pool=ids.slice();
 while(Object.keys(answers).length<size&&pool.length){
  const id=pool.splice(Math.floor(rand()*pool.length),1)[0];
  if((CARTOGRAPHIE.exclusive||[]).some(g=>g.includes(id)&&g.some(x=>answers[x])))continue;
  answers[id]=rand()<0.3?2:1;
 }
 return answers;
}
const mixed=(count,seed)=>{const rand=rng(seed);return Array.from({length:count},()=>profile(rand,1+Math.floor(rand()*20)));};
const shape=r=>({id:r.outcome.id,section:r.section,relevance:r.relevance,max_points:r.max_points,frequent:r.frequent,has_empirical:r.has_empirical,tier:r.tier,score:r.score,associated_count:r.associated_count});

test('page and Worker give the same sections, order and relevance on 2 000 profiles with feedback',()=>{
 const rand=rng(51);let checked=0;
 for(const answers of mixed(2000,20261119)){
  const feedback={};
  for(let k=0;k<3;k++)if(rand()<0.4)feedback[outcomeIds[Math.floor(rand()*outcomeIds.length)]]=rand()<0.5?'oui':'non';
  const page=publicPage.compute(fullData,answers,feedback),worker=computeCartography(answers,CARTOGRAPHIE,feedback);
  assert.deepEqual(worker.results.map(shape),page.results.map(shape));
  for(const name of SECTION_ORDER)assert.deepEqual(worker.sections[name].map(r=>r.outcome.id),page.sections[name].map(r=>r.outcome.id),name);
  assert.deepEqual(worker.results.map(r=>r.outcome.id),SECTION_ORDER.flatMap(name=>worker.sections[name].map(r=>r.outcome.id)),'results = sections concatenated');
  checked++;
 }
 assert.equal(checked,2000);
});

test('sections partition the visible subjects; no contextual subject is ever displayed',()=>{
 for(const answers of mixed(500,7)){
  const feedback={[outcomeIds[0]]:'oui',[outcomeIds[1]]:'non',icd_6b44:'oui'};
  const state=computeCartography(answers,CARTOGRAPHIE,feedback);
  assert.ok(!state.results.some(r=>contextual.has(r.outcome.id)||r.outcome.retired));
  assert.ok(state.sections.top.length<=SECTION_TOP);
  assert.ok(state.sections.recognized.every(r=>feedback[r.outcome.id]==='oui'));
  assert.ok(state.sections.dismissed.every(r=>feedback[r.outcome.id]==='non'));
  // Clinical review (lot C5a): frequent subjects with a moderate or strong link stay visible as cards.
  assert.ok(state.sections.frequent.every(r=>r.frequent&&r.max_points>=8),'frequent section: frequent subjects with a moderate or strong link');
  assert.ok(!state.sections.other.some(r=>r.frequent&&r.max_points>=8));
  assert.ok(!state.sections.top.some(r=>r.serious&&r.max_points<12),'a serious disorder needs a strong link to be among the five first');
  const trauma=[...state.sections.other,...state.sections.frequent].filter(r=>r.trauma_first);assert.equal(trauma.length,0,'trauma subjects linked to violence answered « Souvent » are among the first');
  if(state.sections.other.some(r=>!r.serious)||state.sections.frequent.length)assert.equal(state.sections.top.length,SECTION_TOP);
  const unique=new Set(state.results.map(r=>r.outcome.id));assert.equal(unique.size,state.results.length);
 }
});

test('strength guard: a subject held only by limited links or models never precedes a subject with a moderate or strong link',()=>{
 let moved=0;
 for(const answers of mixed(2000,99)){
  const shown=computeCartography(answers).results.filter(r=>r.section!=='recognized'&&r.section!=='dismissed');
  // Serious disorders without a strong link are kept out of the five first on purpose.
  const lastStrong=shown.map(r=>r.max_points>=8&&!r.familial&&!(r.serious&&r.max_points<12)).lastIndexOf(true);
  const firstWeak=shown.findIndex(r=>r.max_points<=4);
  if(lastStrong>=0&&firstWeak>=0)assert.ok(firstWeak>lastStrong,JSON.stringify(answers));
  // Without the guard a weak subject would sometimes lead: the guard is not a no-op.
  const raw=shown.slice().sort((a,b)=>Number(b.has_empirical)-Number(a.has_empirical)||b.relevance-a.relevance);
  if(raw[0]&&raw[0].max_points<=4&&shown[0].max_points>4)moved++;
 }
 assert.ok(moved>0);
});

test('simulation on 2 000 mixed profiles stays within the agreed bounds',()=>{
 let first=0,top3=0,weak=0,n=0;
 for(const answers of mixed(2000,20261119)){
  const order=computeCartography(answers).results.map(r=>r.outcome.id);if(!order.length)continue;n++;
  if(GENERIC.has(order[0]))first++;
  if(order.slice(0,3).some(id=>GENERIC.has(id)))top3++;
  const head=computeCartography(answers).results[0],measured=head.links.filter(l=>!l.definitional&&l.relation_type==='empirical_association');
  if(!measured.length||(measured.length===1&&measured[0].strength==='limitee'))weak++;
 }
 assert.ok(first/n<0.40,'depression or anxiety first: '+first/n);
 assert.ok(top3/n<0.70,'depression or anxiety in top 3: '+top3/n);
 assert.ok(weak/n<0.12,'head held by a single limited link: '+weak/n);
 // Clinical counter-review: a first card resting only on familial transmission, or a
 // serious disorder without a strong link among the five first, should not happen.
 let familial=0,serious=0;
 for(const answers of mixed(2000,20261119)){const state=computeCartography(answers);if(state.results[0]?.familial)familial++;if(state.sections.top.some(r=>r.serious&&r.max_points<12))serious++;}
 assert.ok(familial/n<0.01,'first card only familial: '+familial/n);assert.equal(serious,0);
});

test('« Souvent » may change the order but never the tier, the evidence or the count',()=>{
 for(const answers of mixed(300,5)){
  const often=Object.fromEntries(Object.keys(answers).map(id=>[id,2]));
  const a=new Map(computeCartography(answers).results.map(r=>[r.outcome.id,r])),b=computeCartography(often).results;
  for(const r of b){const x=a.get(r.outcome.id);assert.equal(r.tier,x.tier);assert.equal(r.score,x.score);assert.equal(r.max_points,x.max_points);assert.equal(r.associated_count,x.associated_count);}
 }
});

test('two experiences of the same family do not double the relevance',()=>{
 const data={factors:[{id:'a',label:'A',overlap_group:'g'},{id:'b',label:'B',overlap_group:'g'},{id:'c',label:'C'}],outcomes:[{id:'x',label:'X',kind:'trouble'},{id:'y',label:'Y',kind:'trouble'}],
  links:[{factor:'a',outcome:'x',strength:'moderee',relation_type:'empirical_association'},{factor:'b',outcome:'x',strength:'moderee',relation_type:'empirical_association'},{factor:'c',outcome:'y',strength:'moderee',relation_type:'empirical_association'}],ranking:{specificity:{x:1000,y:1000},frequent:[],section_top:5}};
 const both=computeCartography({a:1,b:1},data).results[0],one=computeCartography({a:1},data).results[0];
 assert.equal(both.relevance,one.relevance);
 assert.deepEqual(publicPage.compute(data,{a:1,b:1}).results.map(shape),computeCartography({a:1,b:1},data).results.map(shape));
});

test('contract: summary sections, prompt block without score, index entries in section order',()=>{
 const answers={abus_physique:2,abus_emotionnel:2,temoin_violences:1,ressenti_peur:1,pas_protege:1},feedback={att_anxieux:'oui',ten_sommeil:'non',icd_6b44:'oui'};
 const summary=summarizeCartography(answers,feedback),state=computeCartography(answers,CARTOGRAPHIE,feedback);
 assert.deepEqual(summary.contextual,[]);
 assert.deepEqual(Object.keys(summary.sections),SECTION_ORDER);
 for(const name of ['recognized','top','frequent','dismissed'])assert.deepEqual(summary.sections[name].map(x=>x.id),state.sections[name].map(r=>r.outcome.id));
 assert.deepEqual(summary.display_order.map(x=>x.id),state.results.slice(0,10).map(r=>r.outcome.id));
 assert.ok(summary.display_order.every(x=>SECTION_ORDER.includes(x.section)));
 assert.equal(summary.selection_policy.ranking,'relevance_specificity_v5_strength_guard');
 assert.ok(!summary.recognized.some(x=>x.id==='icd_6b44'),'an older recognition of a hidden subject is not sent to Ambre');
 assert.equal(summary.suicide_recognized,false);
 assert.deepEqual(chooseAmbreNotes([{target:'outcome',target_id:'icd_6b44',text:'Note.'},{target:'outcome',target_id:'ten_conduites_suicidaires',text:'Note.'}]),[],'Ambre cannot write on a hidden subject');
 const block=cartographyPromptBlock({version:1,locked_at:5,...summary,...cartographyEvidence(answers,feedback)});
 const text=JSON.stringify(block);
 assert.doesNotMatch(text,/"relevance"|"specificity"|"score"/,'no number from the ranking is sent');
 assert.doesNotMatch(text,/https?:\/\//,'no public URL');
 assert.deepEqual(block.contextual,[]);
 assert.deepEqual(block.sections.frequent.map(x=>x.id),summary.sections.frequent.map(x=>x.id));
 assert.equal(cartographyPromptBlock(null),null);
 const small=cartographyPromptBlock({version:1,locked_at:5,...summary,...cartographyEvidence(answers,feedback)},4000);
 assert.ok(JSON.stringify(small).length<text.length);
 assert.deepEqual(small.sections,block.sections,'top and frequent subjects are never removed');
 assert.deepEqual(small.recognized,block.recognized);
 const entries=cartographyIndexEntries({locked_at:5,...summary});
 const subjects=entries.filter(e=>e.kind==='outcome').map(e=>e.key.split(':').at(-1));
 const expected=[...summary.sections.recognized,...summary.sections.top,...summary.sections.frequent].map(x=>x.id);
 assert.deepEqual(subjects.slice(0,expected.length),expected);
 assert.ok(!subjects.some(id=>contextual.has(id)&&feedback[id]!=='oui'));
 assert.ok(entries.every(e=>e.key&&e.kind&&typeof e.label==='string'&&e.metadata&&e.updated_at===5));
 assert.equal(entries.find(e=>e.key==='cartography:outcome:ten_sommeil').metadata.feedback,'non');
 assert.deepEqual(cartographyIndexEntries(null),[]);
});

test('a tight budget also reduces display_order duplicates, explicit negatives and checked causes, never recognized, top or frequent',()=>{
 const answers=Object.fromEntries(ids.filter(id=>id!=='divorce').map((id,i)=>[id,i%7===0?0:1])),feedback={att_anxieux:'oui'};
 const summary={version:1,locked_at:5,...summarizeCartography(answers,feedback),...cartographyEvidence(answers,feedback)};
 const full=cartographyPromptBlock(summary),tight=cartographyPromptBlock(summary,6000);
 assert.ok(JSON.stringify(full).length>9000);
 assert.ok(JSON.stringify(tight).length<JSON.stringify(full).length);
 assert.ok(tight.display_order.length<full.display_order.length&&tight.display_order_truncated);
 assert.ok(tight.display_order.every(x=>x.feedback==='oui'||tight.sections.top.includes(x.id)));
 assert.equal(tight.explicit_negative.length,0);assert.equal(tight.explicit_negative_omitted,full.explicit_negative.length);
 assert.ok(tight.causes.length>=12&&tight.causes.length<full.causes.length);assert.equal(tight.causes_omitted,full.causes.length-tight.causes.length);
 assert.deepEqual(tight.sections,full.sections);assert.deepEqual(tight.recognized,full.recognized);
 assert.equal(summary.causes.length,full.causes.length,'the summary itself is never mutated');
 assert.equal(summary.display_order.length,full.display_order.length);
});
