import test from 'node:test';
import assert from 'node:assert/strict';
import {replayTurn} from './v5-A-helper.mjs';
import {SYMPTOMES_CARTE} from '../src/symptomes-carte.generated.mjs';
for(const variant of ['edit','delete-recreate','absent-insert'])test('review: symptom '+variant+' immediately before publication cannot publish stale context',async()=>{
 let raced=false;
 const run=await replayTurn('Je voudrais comprendre mes réactions.',{extend:s=>{
  const id=SYMPTOMES_CARTE.factors.find(f=>!f.safety).id;
  const insert=value=>s.raw.prepare("INSERT INTO symptom_maps(user_id,answers,locked_at,updated_at) VALUES('a',?,?,?)").run(JSON.stringify({[id]:value}),s.now(),s.now());
  if(variant!=='absent-insert')insert(1);
  const prepare=s.env.DB.prepare.bind(s.env.DB),batch=s.env.DB.batch.bind(s.env.DB);
  s.env.DB.prepare=sql=>{const statement=prepare(sql);statement.reviewSQL=sql;return statement};
  s.env.DB.batch=statements=>{
   if(!raced&&statements.some(x=>x.reviewSQL?.startsWith("UPDATE chat_turns SET status='completed',credit_status='consumed',error_code="))){
    raced=true;
    if(variant==='edit')s.raw.prepare("UPDATE symptom_maps SET answers=?,version=version+1 WHERE user_id='a'").run(JSON.stringify({[id]:0}));
    else{if(variant==='delete-recreate')s.raw.prepare("DELETE FROM symptom_maps WHERE user_id='a'").run();insert(0);}
   }
   return batch(statements);
  };
 }});
 try{assert.equal(raced,true,'mutation reaches the actual publication batch');assert.equal(run.stage,'failed');const turn=run.s.raw.prepare('SELECT status,response FROM chat_turns WHERE id=?').get(run.s.job().turn_id);assert.notEqual(turn.status,'completed');assert.equal(turn.response,'');}finally{run.s.raw.close()}
});
