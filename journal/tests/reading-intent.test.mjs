import test from 'node:test';
import assert from 'node:assert/strict';
import {expandReadingQuery} from '../src/reading-intent.mjs';
import {createRecommender} from '../src/recommendations.mjs';
import {EXAMPLES} from '../src/examples.mjs';

test('ordinary refusal and repetitive thinking retrieve useful actual catalogue readings',()=>{
 const query='Je repense parfois aux réunions, surtout quand je n’ai pas osé refuser une demande. Peux-tu me proposer un article existant et une question pour comprendre ce qui se passe ?';
 const engine=createRecommender(EXAMPLES),result=engine.recommend({sources:[{id:'reading-request',type:'profile',fields:{current_context:expandReadingQuery(query)}}]},{limit:6,includeLiked:true});
 assert.ok(result.length>0&&result.length<=6);
 assert.ok(result.some(r=>/rumination|overthinking|surpensée/i.test(r.title)));
 for(const item of result)assert.ok(EXAMPLES.some(e=>e.id===item.id&&e.url===item.url));
});

test('unmatched wording stays untouched and declared negatives remain available to the recommender',()=>{
 assert.equal(expandReadingQuery('Mon contexte actuel.'),'Mon contexte actuel.');
 const query='Je ne repense plus à cela. Je veux éviter les articles sur la rumination.';
 assert.ok(expandReadingQuery(query).startsWith(query));
 assert.ok(expandReadingQuery('').length===0);
});
