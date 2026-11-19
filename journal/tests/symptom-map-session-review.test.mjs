import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import worker from '../src/worker.mjs';
import {setup} from './v4-S-helper.mjs';
import {SYMPTOMES_CARTE} from '../src/symptomes-carte.generated.mjs';
test('review: an old A window cannot read, replace or erase B map after session replacement',async t=>{
 const s=setup();t.after(()=>s.raw.close());const token='review-B',origin='https://example.test',factor=SYMPTOMES_CARTE.factors[0].id;
 s.raw.prepare('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)').run(createHash('sha256').update(token).digest('hex'),'b',Math.floor(Date.now()/1000)+3600);
 s.raw.prepare("INSERT INTO symptom_maps(user_id,answers,notes,locked_at,updated_at) VALUES('b',?,?,1,1)").run(JSON.stringify({[factor]:0}),JSON.stringify({factors:{[factor]:'PRIVATE B'}}));
 for(const method of ['GET','PUT','DELETE']){
  const response=await worker.fetch(new Request(origin+'/api/symptomes',{method,headers:{Origin:origin,Cookie:'__Host-dte_session='+token,'X-Expected-User-Id':'a',...(method==='PUT'?{'Content-Type':'application/json'}:{})},...(method==='PUT'?{body:JSON.stringify({answers:{[factor]:1},version:1})}:{})}),s.env,{waitUntil(){}});
  assert.equal(response.status,409,method);const text=await response.text();assert.match(text,/session_changed/);assert.doesNotMatch(text,/PRIVATE B/);
  const row=s.raw.prepare("SELECT answers,notes,version FROM symptom_maps WHERE user_id='b'").get();assert.equal(JSON.parse(row.answers)[factor],0);assert.equal(row.version,1);assert.match(row.notes,/PRIVATE B/);
 }
});
