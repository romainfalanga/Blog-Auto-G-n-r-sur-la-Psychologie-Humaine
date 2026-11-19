import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import worker from '../src/worker.mjs';
import {setup} from './v4-S-helper.mjs';
import {SYMPTOMES_CARTE} from '../src/symptomes-carte.generated.mjs';
test('real Worker requires authentication, bounds requests and isolates map/export/erase',async t=>{
 const s=setup();t.after(()=>s.raw.close());const origin='https://example.test';
 for(const id of ['a','b'])s.raw.prepare('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)').run(createHash('sha256').update('v2-'+id).digest('hex'),id,Math.floor(Date.now()/1000)+3600);
 const call=(path,user='a',method='GET',body)=>worker.fetch(new Request(origin+path,{method,headers:{Origin:origin,...(user?{Cookie:'__Host-dte_session=v2-'+user}:{}),...(body?{'Content-Type':'application/json'}:{})},...(body?{body:JSON.stringify(body)}:{})}),s.env,{waitUntil(){}});
 assert.equal((await call('/api/symptomes',null)).status,401);
 const id=SYMPTOMES_CARTE.factors[0].id;
 assert.equal((await call('/api/symptomes','a','PUT',{answers:{[id]:1},notes:{factors:{[id]:'x'.repeat(121000)}},version:0})).status,413);
 assert.equal((await call('/api/symptomes','a','PUT',{answers:{[id]:1},version:0})).status,200);
 assert.equal((await (await call('/api/symptomes','b')).json()).cartographie,null);
 const exp=await call('/api/export');assert.equal(exp.status,200);assert.equal((await exp.json()).symptom_map.answers[id],1);
 assert.equal((await call('/api/symptomes','a','DELETE')).status,200);assert.equal((await (await call('/api/export')).json()).symptom_map,null);
});
