import test from 'node:test';
import assert from 'node:assert/strict';
import {careSetup} from './care-helper.mjs';
import {createCartographyBackend} from '../src/cartographie.mjs';
import {SYMPTOMES_CARTE} from '../src/symptomes-carte.generated.mjs';
const seed=s=>{for(const user of ['a','b'])s.raw.prepare('INSERT INTO symptom_maps(user_id,answers,notes,locked_at,updated_at) VALUES(?,?,?,?,?)').run(user,JSON.stringify({[SYMPTOMES_CARTE.factors[0].id]:1}),JSON.stringify({factors:{[SYMPTOMES_CARTE.factors[0].id]:'PRIVATE '+user}}),s.now(),s.now());};
test('review: symptoms sharing is opt-in, isolated and rechecked after a concurrent scope removal',async t=>{
 const s=careSetup();t.after(()=>s.close());seed(s);
 const absent=await s.share(['conversation']);assert.equal((await s.call(`/api/care/dossiers/${absent.id}/symptoms`,'GET',undefined,'pro')).httpStatus,404);
 const link=await s.share(['symptoms']);const path=`/api/care/dossiers/${link.id}/symptoms`;
 const result=await s.call(path,'GET',undefined,'pro');assert.equal(result.httpStatus,200);assert.match(JSON.stringify(result),/PRIVATE a/);assert.doesNotMatch(JSON.stringify(result),/PRIVATE b/);
 for(const user of ['b','other'])assert.equal((await s.call(path,'GET',undefined,user)).httpStatus,404);
 s.setHook(({sql,phase})=>{if(sql.startsWith('SELECT answers,feedback,notes,consent')&&phase==='after'){s.setHook(null);s.raw.prepare("UPDATE care_links SET scopes='[]',revision=revision+1 WHERE id=?").run(link.id)}});
 assert.equal((await s.call(path,'GET',undefined,'pro')).httpStatus,404);
});
test('review: backend table selection is a fixed allowlist',()=>{
 for(const table of ['users','symptom_maps; DROP TABLE users','symptom_maps WHERE user_id=1'])assert.throws(()=>createCartographyBackend({table}),/Unsupported map table/);
});
