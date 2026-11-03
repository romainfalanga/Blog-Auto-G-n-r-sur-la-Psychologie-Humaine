import test from 'node:test';
import assert from 'node:assert/strict';
import {createRecommender} from '../src/recommendations.mjs';
const article=(id,character,keywords,description='')=>({id,title:keywords.join(' et '),description,character,keywords,url:'/posts/'+id,character_keywords:[]});
const examples=[
 article('seed','Sophie',['Poser ses limites','Culpabilité'],'Dire non au travail et poser une limite sans culpabiliser.'),
 article('same-unrelated','Sophie',['Deuil','Famille'],'Apprivoiser le deuil de son parent.'),
 article('limits','Nadia',['Poser ses limites','Travail'],'Dire non à une demande professionnelle.'),
 article('guilt','Thomas',['Culpabilité','Famille'],'Se sentir coupable après avoir posé ses limites.'),
 article('perfection','Sophie',['Perfectionnisme','Travail'],'Accepter une erreur au travail.'),
 article('anger','Adrien',['Colère','Émotions'],'Exprimer la colère.'),
 article('grief','Monique',['Deuil','Famille'],'Traverser un deuil.'),
 article('approval','Lucas',['Approbation','Comparaison'],'Chercher la reconnaissance et se comparer aux autres.'),
 article('limits2','Patrick',['Poser ses limites','Culpabilité'],'Refuser sans culpabiliser.'),
 article('limits3','Patrick',['Poser ses limites','Travail'],'Trouver la bonne limite.'),
 article('limits4','Patrick',['Poser ses limites','Travail'],'Dire non avec calme.'),
 article('limits5','Patrick',['Poser ses limites','Travail'],'Exprimer un refus.'),
 article('other','Camille',['Anxiété','Charge mentale'],'Une inquiétude dans son travail.')
];
const rank=(rows,id)=>{const i=rows.findIndex(x=>x.id===id);return i<0?Infinity:i};
const engine=createRecommender(examples),at=Date.parse('2026-11-01T12:00:00Z');
const like=(id,reason='',updated_at='2026-11-01T10:00:00Z')=>({article_id:id,reason,created_at:updated_at,updated_at});
test('first like has immediate non-liked recommendations and a sourced provisional portrait',()=>{const ctx={likes:[like('seed')]};const r=engine.recommend(ctx,{limit:6,now:at});assert(r.length>=3&&r.length<=6);assert(!r.some(x=>x.id==='seed'));assert(r.findIndex(x=>x.id==='limits')<r.findIndex(x=>x.id==='same-unrelated')||!r.some(x=>x.id==='same-unrelated'));const p=engine.provisionalPortrait(ctx);assert(p.provisional);assert(p.axes.every(x=>x.evidence.includes('seed')));assert(!p.fictional_character);});
test('explicit subtheme changes ranking across characters',()=>{const a=engine.recommend({likes:[like('seed','Je me reconnais dans la culpabilité familiale envers ma famille et mes parents.')]},{limit:3,now:at});const b=engine.recommend({likes:[like('seed','Ce qui me parle, c’est dire non au travail et poser une limite professionnelle.')]},{limit:3,now:at});assert.equal(a[0].id,'guilt');assert.equal(b[0].id,'limits');});
test('negative explanations are not promoted as positive interests',()=>{const positive=engine.recommend({likes:[like('seed','Je me reconnais dans le perfectionnisme au travail.')]},{limit:12,now:at});const negative=engine.recommend({likes:[like('seed','Je ne me reconnais pas dans le perfectionnisme au travail.')]},{limit:12,now:at});assert(rank(positive,'perfection')<rank(negative,'perfection'));});
test('recent corrections can shift ranking while older likes remain evidence',()=>{const likes=[like('seed','Je me reconnais dans la culpabilité familiale.','2025-01-01T00:00:00Z')];const before=engine.recommend({likes},{limit:3,now:at});const after=engine.recommend({likes,memory:{text:'Actuellement, je veux surtout explorer la colère et exprimer mes émotions.'}},{limit:3,now:at});assert.equal(after[0].id,'anger');assert.notEqual(before[0].id,after[0].id);assert(engine.provisionalPortrait({likes}).axes.length>0);});
test('diversification caps character repetition in first six and explains recommendations',()=>{const r=engine.recommend({likes:[like('seed')]},{limit:12,now:at});const counts={};for(const x of r.slice(0,6))counts[x.character]=(counts[x.character]||0)+1;assert(Object.values(counts).every(x=>x<=2));assert(r.every(x=>x.reasons.length>0&&x.keywords.length<=3));assert(r.slice(0,6).some(x=>x.character!=='Sophie'));});
test('request-local data cannot leak through subsequent ranking',()=>{const a={likes:[like('seed','Private SENTINEL')]},b={likes:[like('grief')]};const expected=engine.recommend(b,{now:at});engine.recommend(a,{now:at});assert.deepEqual(engine.recommend(b,{now:at}),expected);assert(!JSON.stringify(expected).includes('SENTINEL'));});
test('removing the last or unknown like clears personal recommendations and provisional portrait',()=>{assert.deepEqual(engine.recommend({likes:[]}),[]);assert.equal(engine.provisionalPortrait({likes:[]}),null);assert.deepEqual(engine.recommend({likes:[like('removed-source')]}),[]);});

test('low-resonance feedback is a rejection signal rather than a normalized positive affinity',()=>{const base={likes:[like('seed')]};const yes=engine.recommend({...base,feedback:[{resonance:5,why:'Colère et émotions',updated_at:'2026-11-01T10:00:00Z'}]},{limit:12,now:at});const no=engine.recommend({...base,feedback:[{resonance:1,why:'Colère et émotions',updated_at:'2026-11-01T10:00:00Z'}]},{limit:12,now:at});assert(rank(yes,'anger')<rank(no,'anger'));assert.notDeepEqual(yes,no);});
test('colloquial French negation without ne is never a positive recommendation signal',()=>{const positive=engine.recommend({likes:[like('seed','Intéressé par le deuil et la famille.')]},{limit:12,now:at});const negative=engine.recommend({likes:[like('seed','Pas intéressé par le deuil et la famille.')]},{limit:12,now:at});assert(rank(positive,'grief')<rank(negative,'grief'));});
