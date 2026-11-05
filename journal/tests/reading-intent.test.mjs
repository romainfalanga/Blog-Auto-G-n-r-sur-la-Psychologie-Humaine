import test from 'node:test';
import assert from 'node:assert/strict';
import {expandReadingQuery,lookupReadingCatalogue} from '../src/reading-intent.mjs';
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

test('negative or ambiguous clauses never manufacture affirmative editorial interests',()=>{
 for(const query of ['Je ne repense plus à cela. Je veux éviter les articles sur la rumination.','Je ne veux pas trop penser à cette question.','Je ne veux pas éviter de décevoir.'])assert.equal(expandReadingQuery(query),query);
 const mixed='Je repense parfois aux réunions, surtout quand je n’ai pas osé refuser une demande.';
 assert.match(expandReadingQuery(mixed),/Thèmes de recherche : rumination overthinking/);
});

test('a current explicit refusal of rumination does not retrieve it through added positive themes',()=>{
 const query='Je ne repense plus à cela. Je veux éviter les articles sur la rumination.';
 const engine=createRecommender(EXAMPLES),expanded=expandReadingQuery(query),result=engine.recommend({sources:[]},{query:expanded,limit:6,includeLiked:true});
 assert.equal(expanded,query);assert.equal(result.some(r=>/rumination|overthinking|surpensée/i.test(r.title)),false);
});

test('legacy saviour requests still find related readings before dedicated V10 stories are added',()=>{
 const original=EXAMPLES.filter(e=>e.pattern_source!=='editorial_pack_v10_reviewed_story');
 assert.equal(original.length,231);
 const query='As-tu un personnage sur le syndrome du sauveur ?',lookup=lookupReadingCatalogue(original,query),engine=createRecommender(original);
 assert.equal(lookup.scope,'complete_public_catalogue');assert.equal(lookup.catalogue_size,original.length);assert.equal(lookup.searched_all,true);assert.ok(lookup.total_matches>=4);assert.ok(lookup.matches.some(x=>x.character==='Thomas'&&/Karpman/.test(x.title)));assert.ok(lookup.matches.some(x=>/Abnégation/.test(x.title)));assert.ok(lookup.matches.every(x=>['related','metadata_match'].includes(x.match_kind)&&original.some(e=>e.id===x.id&&e.url===x.url)));assert.equal(lookup.absence_proven,false);
 const results=engine.recommend({sources:[]},{query:expandReadingQuery(query),limit:6,includeLiked:true});assert.ok(results.slice(0,4).some(x=>x.character==='Thomas'&&/Karpman/.test(x.title)));assert.ok(results.slice(0,4).some(x=>/Abnégation/.test(x.title)));
 const unknown=lookupReadingCatalogue(EXAMPLES,'un thème absent zzzzzzz');assert.equal(unknown.total_matches,0);assert.equal(unknown.absence_proven,false);
});

test('catalogue existence stays global while suggestions preserve explicit reading avoidances',()=>{
 const original=EXAMPLES.filter(e=>e.pattern_source!=='editorial_pack_v10_reviewed_story');
 const query='Je souhaite éviter les articles sur le syndrome du sauveur.';assert.equal(expandReadingQuery(query),query);const engine=createRecommender(original),lookup=lookupReadingCatalogue(original,query);assert.ok(lookup.total_matches>0);
 const context={sources:[],avoid_topics:[{id:'no-self-sacrifice',label:'abnégation sacrifice de soi triangle de Karpman',confirmed:true,updated_at:'2026-10-01T00:00:00Z'}]};
 const result=engine.recommend(context,{query:expandReadingQuery('syndrome du sauveur'),limit:6,includeLiked:true});assert.equal(result.some(x=>lookup.matches.some(m=>m.id===x.id)),false);
});

test('the complete V10 catalogue retrieves the actual dedicated Aline story for equivalent saviour requests',()=>{
 assert.equal(EXAMPLES.length,387);
 for(const query of ['Un article sur le syndrome du sauveur ?', 'Un récit sur le complexe du sauveur ?', 'Une lecture sur le syndrome de la sauveuse ?']){
  const result=lookupReadingCatalogue(EXAMPLES,query);
  assert.equal(result.catalogue_size,387);assert.equal(result.searched_all,true);assert.equal(result.absence_proven,false);
  assert.ok(result.matches.some(m=>m.character==='Aline'&&/Syndrome du sauveur/.test(m.title)&&m.match_kind==='metadata_match'));
  const suggestions=createRecommender(EXAMPLES).recommend({sources:[]},{query:expandReadingQuery(query),limit:6,includeLiked:true});
  assert.ok(suggestions.some(m=>m.character==='Aline'&&/Syndrome du sauveur/.test(m.title)));
 }
});

test('natural catalogue questions keep only their actual topic across the real editorial corpus',()=>{
 for(const [query,expected] of [
  ['Est-ce que tu connais un article qui parle de l’intellectualisation ?',/Intellectualisation/i],
  ['Une histoire sur la dissociation ?',/Dissociation/i],
  ['Connais-tu un récit qui parle de la rumination ?',/Rumination|Overthinking/i],
  ['Comment trouver un article qui parle de la peur de décevoir ?',/approbation|abnégation|Karpman|décevoir|déception|limite/i],
  ['Connais-tu un article sur le HPI ?',/HPI|Intellectualisation/i]
 ]){
  const result=lookupReadingCatalogue(EXAMPLES,query);
  assert.ok(result.total_matches>0,query);
  const articles=result.matches.map(item=>{
   const article=EXAMPLES.find(e=>e.id===item.id);
   assert.ok(article&&article.title===item.title&&article.url===item.url);
   return article;
  });
  if(query.includes('HPI'))assert.ok(articles.every(article=>article.keywords.includes('HPI')));
  else assert.ok(articles.some(article=>[article.title,article.sujet,...article.keywords].some(value=>expected.test(value))),query);
  assert.equal(result.absence_proven,false);
  assert.equal(result.searched_all,true);
 }
 const syndrome=lookupReadingCatalogue(EXAMPLES,'As-tu un personnage autour du syndrome du sauveur ?'),complex=lookupReadingCatalogue(EXAMPLES,'Connais-tu un récit sur le complexe du sauveur ?');assert.deepEqual(complex.matches,syndrome.matches);assert.match(expandReadingQuery('Je cherche une lecture sur le complexe du sauveur.'),/abnégation sacrifice de soi/);
 for(const query of ['Est-ce que tu connais un article qui parle de la zzzthemeinexistant ?', 'Une histoire sur la dissociation et zzzthemeinexistant ?', 'Un article sur la peur de décevoir et zzzthemeinexistant ?', 'Un récit sur le complexe du sauveur et zzzthemeinexistant ?', 'Connais-tu un article ?']){const result=lookupReadingCatalogue(EXAMPLES,query);assert.equal(result.total_matches,0,query);assert.deepEqual(result.matches,[]);assert.equal(result.absence_proven,false);}
});

test('a related editorial concept precedes a passing mention in an article description',()=>{
 const passing={id:'a',title:'Reconnaître sa fatigue',description:'Décrire ses limites au repos.',url:'/passing/'},focused={id:'b',title:'Exprimer ses limites',keywords:['Limites'],url:'/focused/'};
 const result=lookupReadingCatalogue([passing,focused],'Un article sur la peur de décevoir ?');
 assert.equal(result.matches[0].id,'b');assert.equal(result.matches[0].match_kind,'related');assert.equal(result.matches[1].id,'a');
});


test('a new dedicated saviour story is found ahead of related themes, including equivalent labels',()=>{
 const related={id:'karpman',title:'Le triangle de Karpman en couple',sujet:'Triangle de Karpman',character:'Thomas',keywords:['Karpman'],url:'/posts/karpman/'},dedicated={id:'aline',title:'Syndrome du sauveur : aider sans s’effacer',sujet:'Rôle de sauveur',character:'Aline',keywords:['Syndrome du sauveur','Surresponsabilité','Limites'],url:'/posts/aline/'};
 for(const query of ['Un article sur le syndrome du sauveur ?', 'Un récit sur le complexe du sauveur ?', 'Une lecture sur le syndrome de la sauveuse ?', 'Besoin de sauver les autres']){
  const lookup=lookupReadingCatalogue([related,dedicated],query);assert.equal(lookup.matches[0].id,'aline',query);assert.equal(lookup.matches[0].match_kind,'metadata_match');assert.equal(lookup.matches[1].match_kind,'related');assert.equal(lookup.absence_proven,false);
 }
 assert.equal(lookupReadingCatalogue([related,dedicated],'syndrome du sauveur Aline').matches[0].id,'aline');
 assert.equal(lookupReadingCatalogue([related,dedicated],'syndrome du sauveur et zzzinconnu').total_matches,0);
});
