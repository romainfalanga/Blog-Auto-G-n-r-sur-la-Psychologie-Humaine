import test from 'node:test';
import assert from 'node:assert/strict';
import {createRecommender,classifyText,RELEVANCE_THRESHOLD} from '../src/recommendations.mjs';
import {EXAMPLES} from '../src/examples.mjs';
import {scenarios} from './fixtures/recommendation-scenarios.mjs';
const rows=[['anger','Colère','Conflit'],['grief','Deuil','Famille'],['perfection','Perfectionnisme','Travail'],['seed','Limites','Travail'],['gratitude','Gratitude','Finances']].map(([id,...keywords],index)=>({id,keywords,title:keywords.join(' '),character:'Personnage'+index,url:'/'+id,sujet:keywords[0]}));
const engine=createRecommender(rows),at=Date.parse('2026-11-03');
const r=ctx=>engine.recommend(ctx,{now:at});
test('zero likes with declared event, relationship or memory produces sourced suggestions',()=>{
 for(const ctx of [{events:[{id:'e',description:'Je souhaite explorer la colère.'}]},{people:[{id:'p',relationship_text:'Je souhaite explorer la colère.'}]},{memory:{text:'Je souhaite explorer la colère.'}}]){
  const a=r(ctx);assert.equal(a[0].id,'anger');assert.equal(a[0].reason_sources[0].kind,'request');assert(engine.provisionalPortrait(ctx));
 }
});
test('role/name alone is insufficient and is never a psychological signal',()=>{assert.deepEqual(r({people:[{id:'p',display_name:'Colère',role:'father'}]}),[]);});
test('plain nominal differences cannot positively boost their theme',()=>{
 const base={likes:[{article_id:'seed'}]};const withDifference=r({likes:[{article_id:'seed',differences:'Le deuil familial'}]});
 assert(!withDifference.some(x=>x.id==='grief'));assert(!r(base).some(x=>x.id==='grief'));assert.equal(classifyText('Le deuil familial','difference')[0].kind,'difference');
});
test('double negation remains an explicit reading request without a hard filter',()=>{assert.equal(classifyText('Je ne veux pas éviter la colère.')[0].kind,'request');assert.equal(r({memory:{text:'Je ne veux pas éviter la colère.'}})[0].id,'anger');});
test('new positive declaration in differences is handled separately',()=>{const parts=classifyText('Le deuil familial, mais je souhaite explorer la colère.','difference');assert.equal(parts[0].kind,'difference');assert.equal(parts[1].kind,'request');assert.equal(r({likes:[{article_id:'seed',differences:'Le deuil familial, mais je souhaite explorer la colère.'} ]})[0].id,'anger');});
test('explicit avoidance filters only requested topic; nonresemblance never means blanket avoidance',()=>{
 const context={memory:{text:'Je souhaite lire sur la colère et le deuil.'},feedback:[{story_id:'s',resonance:4,avoid:'Deuil'}]};assert(!r(context).some(x=>x.id==='grief'));assert(r(context).some(x=>x.id==='anger'));
});
test('correction beyond 120 characters is not truncated',()=>{assert.equal(r({memory:{text:'Voici un contexte détaillé et long. '.repeat(10)+' Je souhaite explorer la gratitude et les finances.'}})[0].id,'gratitude');});
test('derived portrait cannot duplicate original proof or introduce fiction',()=>{const c={memory:{text:'Je souhaite explorer la colère.'}};assert.deepEqual(r(c),r({...c,portrait:{schema_version:2,summary:'Deuil',axes:[{label:'Deuil',confidence:1,evidence:['memory']}],fictional_character:{name:'Grief'}}}));});
test('irrelevant results are not added to fill twelve slots',()=>{const a=r({memory:{text:'Je souhaite explorer la gratitude.'}});assert.equal(a.length,1);assert(a.every(x=>x.relevance>=RELEVANCE_THRESHOLD));});
test('source deletion clears cached assumptions and user requests remain isolated',()=>{const a={memory:{text:'Gratitude finances'}};r(a);assert.deepEqual(r({}),[]);assert.deepEqual(r({memory:{text:'Colère conflit'}}),r({memory:{text:'Colère conflit'}}));});
test('thirty French annotated scenarios reference existing unliked catalogue articles',()=>{const set=scenarios(EXAMPLES);assert.equal(set.length,30);for(const s of set){assert(Object.keys(s.judgments).length>=1);assert(s.annotation);for(const id of Object.keys(s.judgments)){assert(EXAMPLES.some(x=>x.id===id));assert(!(s.context.likes||[]).some(l=>l.article_id===id));}}});
test('all 231 catalogue entries carry faithful extractive metadata and content hashes',()=>{assert.equal(EXAMPLES.length,231);for(const e of EXAMPLES){assert.equal(e.catalogue_version,2);assert.match(e.content_hash,/^[a-f0-9]{64}$/);assert(e.summary_evidence.length);assert(e.summary_evidence.every(q=>e.summary.includes(q.quote)));assert.equal(e.summary_method,'extractive_opening_and_final_scene');}});

test('ambiguous double negation inside avoidance field never creates a hard filter',()=>{const a=r({memory:{text:'Je souhaite lire sur la colère.'},feedback:[{story_id:'s',avoid:'Je ne veux pas éviter la colère.'}]});assert(a.some(x=>x.id==='anger'));assert.equal(classifyText('Je veux éviter le deuil.')[0].kind,'difference');});
