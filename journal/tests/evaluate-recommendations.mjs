import {EXAMPLES} from '../src/examples.mjs';
import {createRecommender} from '../src/recommendations.mjs';
import {createRecommender as baseline} from '../src/recommendations-baseline.mjs';
import {scenarios} from './fixtures/recommendation-scenarios.mjs';
const at=Date.parse('2026-11-03T12:00:00Z');
const cases=scenarios(EXAMPLES), engines={baseline:baseline(EXAMPLES),v2:createRecommender(EXAMPLES)};
const dcg=grades=>grades.reduce((sum,g,i)=>sum+(2**g-1)/Math.log2(i+2),0);
const metrics={};
const keys=['count','precision@6','precision@12','ndcg@6','ndcg@12','characters@6'];
const mean=rows=>Object.fromEntries(keys.map(k=>[k,rows.reduce((n,r)=>n+r[k],0)/rows.length]));
for(const [name,engine] of Object.entries(engines)){
 const rows=cases.map(s=>{const result=engine.recommend(s.context,{limit:12,now:at});const row={scenario:s.id,count:result.length,top:result.slice(0,3).map(x=>x.id)};
  for(const k of [6,12]){const grades=result.slice(0,k).map(x=>s.judgments[x.id]||0);row['precision@'+k]=grades.filter(x=>x>0).length/k;row['ndcg@'+k]=dcg(grades)/dcg(Object.values(s.judgments).sort((a,b)=>b-a).slice(0,k));}
  row['characters@6']=new Set(result.slice(0,6).map(x=>x.character)).size;return row;});
 metrics[name]={means:mean(rows),cohorts:{with_likes:{count:15,means:mean(rows.slice(0,15))},zero_likes:{count:15,means:mean(rows.slice(15))}},scenarios:rows};
}
console.log(JSON.stringify({catalogue_count:EXAMPLES.length,scenario_count:cases.length,as_of:'2026-11-03',annotation_limit:'Synthetic engineering judgments, sparse graded labels; unjudged scored zero; no independent editorial or clinical validation. No real provider calls. Weights frozen before these 30 scenarios were written; no coefficient tuning on this set. Zero-like baseline returns no results by design; compare cohorts separately. No independent holdout yet.',metrics},null,2));
