import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {setup,provider} from './v4-S-helper.mjs';
import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {readCompanionReadiness,companionReadinessSql} from '../src/companion-readiness.mjs';
import {readMonthlyQuotas,parisMonth} from '../src/monthly-quotas.mjs';
import worker from '../src/worker.mjs';
const active=CARTOGRAPHIE.factors.filter(f=>!f.historical).map(f=>f.id);
const stamp=value=>Math.floor(Date.parse(value)/1000);
function fixture(t,options){const s=setup(options);t.after(()=>s.raw.close());return s}
function seedMessages(s,n,{period=parisMonth(s.now()).id,user='a',prefix='history'}={}){
 const insert=s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,quota_period,source_version,consent_revision,content,response,created_at) VALUES(?,?,?,'hash','completed','consumed','2025-01-06',?,1,1,'Message','Réponse','2025-01-01 10:00:00')");
 for(let i=0;i<n;i++)insert.run(prefix+'-'+user+'-'+i,user,prefix+'-'+i,period);
}

test('readiness uses persisted identity, all counts, 60 active answers including Non/unknown/legacy2, and consent',async t=>{
 const s=fixture(t);assert.equal((await readCompanionReadiness(s.env,'a')).complete,false);
 for(const answer of [0,'unknown',2]){
  seedCompanionPrerequisites(s.raw,'a',{answer});s.raw.prepare('UPDATE cartographies SET answers=? WHERE user_id=?').run(JSON.stringify(Object.fromEntries(active.map(id=>[id,answer]))),'a');
  let r=await readCompanionReadiness(s.env,'a');assert.equal(r.complete,true);assert.equal(r.childhood.required,60);assert.equal(r.identity.complete,true);assert.equal(s.raw.prepare('SELECT '+companionReadinessSql('?')+' ok').get('a').ok,1);
 }
 const answers=Object.fromEntries(active.filter(id=>id!=='parents_stricts').map(id=>[id,0]));answers.controle=2;s.raw.prepare('UPDATE cartographies SET answers=? WHERE user_id=?').run(JSON.stringify(answers),'a');
 assert.deepEqual((await readCompanionReadiness(s.env,'a')).childhood.missing_ids,['parents_stricts']);assert.equal(s.raw.prepare('SELECT '+companionReadinessSql('?')+' ok').get('a').ok,0);
 seedCompanionPrerequisites(s.raw);s.raw.prepare("UPDATE cartographies SET consent=0 WHERE user_id='a'").run();assert.equal((await readCompanionReadiness(s.env,'a')).complete,false);
 seedCompanionPrerequisites(s.raw);s.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.birth_date','2025-02-30') WHERE user_id='a'").run();assert.deepEqual((await readCompanionReadiness(s.env,'a')).identity.missing_fields,['birth_date']);assert.equal(s.raw.prepare('SELECT '+companionReadinessSql('?')+' ok').get('a').ok,0);
 assert.equal((await readCompanionReadiness(s.env,'b')).people.count,0);
});

test('real Worker rejects forged starts/messages/reviews for existing journeys without readiness and makes no provider call',async t=>{
 const s=fixture(t);let calls=0;t.mock.method(globalThis,'fetch',async()=>{calls++;throw Error('not allowed')});
 const token='test-token';s.raw.prepare('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)').run(createHash('sha256').update(token).digest('hex'),'a',Math.floor(Date.now()/1000)+3600);
 s.env.PUBLIC_ORIGIN='https://example.test';s.env.FREE_ACCESS_ENABLED='true';
 s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES('old','a','initial','ready','Old','old-initial',0,0)").run();
 for(const path of ['/api/chat/start','/api/chat/messages','/api/chat/review']){
  const response=await worker.fetch(new Request('https://example.test'+path,{method:'POST',headers:{Origin:'https://example.test',Cookie:'__Host-dte_session='+token,'Content-Type':'application/json'},body:JSON.stringify({accepted:true,content:'Bonjour',idempotency_key:'forged-'+path.split('/').at(-1)})}),s.env,{});
  // Starting an already-started conversation is still harmless; test fresh start below.
  assert.equal(response.status,path.endsWith('/start')?409:403);if(!path.endsWith('/start'))assert.equal((await response.json()).code,'prerequisites_required');
 }
 s.raw.prepare("DELETE FROM ambre_reviews WHERE id='old'").run();const start=await s.call('/api/chat/start','POST',{accepted:true,idempotency_key:'fresh-forged-start'});assert.equal(start.status,403);assert.equal(start.code,'prerequisites_required');
 assert.equal(calls,0);assert.equal(s.raw.prepare('SELECT count(*) n FROM chat_turns').get().n,0);assert.equal(s.raw.prepare('SELECT count(*) n FROM provider_budgets').get().n,0);
});

test('monthly messages reserve atomically at99, initial does not consume messages, replay is free, deletion retains usage',async t=>{
 const s=fixture(t);await s.ready();seedMessages(s,99);const calls=provider(t);
 const results=await Promise.all([s.send('Je voudrais parler de ma journée.','last-message-a'),s.send('Je voudrais parler de mon travail.','last-message-b')]);
 assert.equal(results.filter(r=>r.status===200).length,1);assert.ok(results.some(r=>[409,429].includes(r.status)));
 let q=await readMonthlyQuotas(s.env,'a',s.now());assert.equal(q.messages.used,100);assert.equal(q.messages.reserved,0);assert.equal(q.reviews.used,0);
 const turn=s.raw.prepare("SELECT id,idempotency_key,content FROM chat_turns WHERE idempotency_key LIKE 'last-message-%'").get(),count=calls.length;
 assert.equal((await s.send(turn.content,turn.idempotency_key)).status,200);assert.equal(calls.length,count);
 s.raw.prepare('DELETE FROM chat_turns WHERE id=?').run(turn.id);q=await readMonthlyQuotas(s.env,'a',s.now());assert.equal(q.messages.used,100);
 assert.equal((await s.send('Une nouvelle question.','quota-full-message')).code,'quota_exhausted');assert.equal(calls.length,count);
 s.tick(32*86400);s.env.FREE_ACCESS_ENABLED='true';assert.equal((await readMonthlyQuotas(s.env,'a',s.now())).messages.used,0);
});

test('initial and point share one atomic monthly reservation; failures release, publication and content deletion do not reset it',async t=>{
 const s=fixture(t,{env:{REVIEW_INLINE_TIMEOUT_MS:'0'}});seedCompanionPrerequisites(s.raw);
 t.mock.method(globalThis,'fetch',()=>assert.fail('Queued reviews must not call the provider'));
 const results=await Promise.all(['review-first-a','review-first-b'].map(idempotency_key=>s.call('/api/chat/start','POST',{accepted:true,idempotency_key})));
 assert.equal(results.filter(r=>r.status===202).length,1);assert.ok(results.some(r=>[409,429].includes(r.status)));let q=await readMonthlyQuotas(s.env,'a',s.now());assert.equal(q.reviews.reserved,1);assert.equal(q.messages.used,0);
 let row=s.raw.prepare("SELECT * FROM ambre_reviews WHERE status='running'").get();s.raw.prepare("UPDATE ambre_reviews SET status='failed',error_code='provider_error' WHERE id=?").run(row.id);
 assert.equal((await readMonthlyQuotas(s.env,'a',s.now())).reviews.remaining,1);
 const retry=await s.call('/api/chat/start','POST',{accepted:true,idempotency_key:'review-retry-new'});assert.equal(retry.status,202);
 row=s.raw.prepare("SELECT * FROM ambre_reviews WHERE status='running'").get();s.raw.prepare("UPDATE ambre_reviews SET status='ready',completed_at=?,content='Approved initial' WHERE id=?").run(s.now(),row.id);
 assert.equal((await s.call('/api/chat/review','POST',{idempotency_key:'point-same-month'})).code,'review_quota_exhausted');
 s.raw.prepare('DELETE FROM ambre_reviews WHERE id=?').run(row.id);q=await readMonthlyQuotas(s.env,'a',s.now());assert.equal(q.reviews.used,1);assert.equal(q.messages.remaining,100);
 assert.equal((await s.call('/api/chat/start','POST',{accepted:true,idempotency_key:'delete-cannot-reset'})).code,'review_quota_exhausted');
});

test('a readiness change during generation prevents publication and refunds the message reservation',async t=>{
 const s=fixture(t);await s.ready();let removed=false;const calls=provider(t,purpose=>{if(purpose==='ambre_writing'&&!removed){removed=true;s.raw.prepare("DELETE FROM personal_people WHERE id='ready-person-a-1'").run()}});
 const result=await s.send('Je voudrais parler de ma journée.','remove-during-writing');assert.equal(result.last_request.status,'failed');assert.equal(result.quota.used,0);assert.equal(result.quota.reserved,0);assert.equal(result.access.reason,'prerequisites_required');assert.equal(calls.filter(x=>x.purpose==='ambre_review').length,0);assert.equal(result.messages.some(x=>x.role==='assistant'),false);
});

test('calendar periods use Paris midnight at DST boundaries and never reset when free access changes',async t=>{
 const s=fixture(t);
 for(const [at,id,start,end] of [
  ['2026-03-31T21:59:59Z','2026-03','2026-02-28T23:00:00.000Z','2026-03-31T22:00:00.000Z'],
  ['2026-03-31T22:00:00Z','2026-04','2026-03-31T22:00:00.000Z','2026-04-30T22:00:00.000Z'],
  ['2026-10-31T23:00:00Z','2026-11','2026-10-31T23:00:00.000Z','2026-11-30T23:00:00.000Z'],
  ['2028-02-29T23:00:00Z','2028-03','2028-02-29T23:00:00.000Z','2028-03-31T22:00:00.000Z']
 ])assert.deepEqual(parisMonth(stamp(at)),{id,basis:'calendar_month',start,end,time_zone:'Europe/Paris'});
 seedMessages(s,3);const before=await readMonthlyQuotas(s.env,'a',s.now());s.env.FREE_ACCESS_ENABLED='true';s.raw.prepare("UPDATE subscriptions SET status='none',paid_until=0 WHERE user_id='a'").run();assert.deepEqual(await readMonthlyQuotas(s.env,'a',s.now()),before);
});

test('migration backfills historical purged messages and deleted review receipts once, in their Paris month',()=>{
 const raw=new DatabaseSync(':memory:');try{
  const directory=new URL('../migrations/',import.meta.url);for(const name of readdirSync(directory).filter(n=>n.endsWith('.sql')&&n<'0026').sort())raw.exec(readFileSync(new URL(name,directory),'utf8'));
  raw.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('a','a@test','x','x');INSERT INTO profiles(user_id) VALUES('a');");
  const turn=raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,purged,created_at) VALUES(?,'a',?,'x','completed','consumed','2026-03-30',1,1,'','',1,?)");
  turn.run('m1','message-1','2026-03-31 21:59:59');turn.run('m2','message-2','2026-03-31 22:00:00');turn.run('r1','review-deleted-id','2026-03-31 22:00:00');turn.run('r2','review-existing-id','2026-03-31 22:00:00');
  raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,idempotency_key,created_at) VALUES('existing-id','a','initial','ready','existing-key',?)").run(stamp('2026-03-31T22:00:00Z'));
  raw.exec(readFileSync(new URL('0026_monthly_usage.sql',directory),'utf8'));
  const rows=raw.prepare('SELECT kind,operation_id,period_id,state FROM monthly_usage ORDER BY kind,operation_id').all().map(r=>({...r}));
  assert.deepEqual(rows,[{kind:'message',operation_id:'m1',period_id:'2026-03',state:'consumed'},{kind:'message',operation_id:'m2',period_id:'2026-04',state:'consumed'},{kind:'review',operation_id:'deleted-id',period_id:'2026-04',state:'consumed'},{kind:'review',operation_id:'existing-id',period_id:'2026-04',state:'consumed'}]);
  raw.exec("UPDATE ambre_reviews SET status='failed' WHERE id='existing-id';DELETE FROM ambre_reviews;DELETE FROM chat_turns;");assert.equal(raw.prepare("SELECT count(*) n FROM monthly_usage WHERE state='consumed'").get().n,4);
 }finally{raw.close()}
});

test('review and message budget admission rechecks readiness atomically and sends no request after withdrawal',async t=>{
 for(const kind of ['review','message'])await t.test(kind,async t=>{
  const s=fixture(t);if(kind==='message')await s.ready();else seedCompanionPrerequisites(s.raw);
  const prepare=s.env.DB.prepare.bind(s.env.DB);let changed=false,calls=0;
  s.env.DB.prepare=sql=>{const stmt=prepare(sql);if(sql.startsWith('INSERT INTO provider_budgets')){const run=stmt.run.bind(stmt);stmt.run=async()=>{if(!changed){changed=true;s.raw.prepare("DELETE FROM personal_people WHERE id='ready-person-a-1'").run()}return run()}}return stmt};
  t.mock.method(globalThis,'fetch',async()=>{calls++;throw Error('Provider must not be reached')});
  if(kind==='review')await s.call('/api/chat/start','POST',{accepted:true,idempotency_key:'review-admission-race'});else await s.send('Parlons de ma journée.','message-admission-race');
  assert.equal(changed,true);assert.equal(calls,0);assert.equal((await readCompanionReadiness(s.env,'a')).complete,false);
  const quotas=await readMonthlyQuotas(s.env,'a',s.now());assert.equal(quotas[kind==='review'?'reviews':'messages'].reserved,0);assert.equal(quotas[kind==='review'?'reviews':'messages'].used,0);
  assert.equal(s.raw.prepare('SELECT coalesce(sum(provider_calls),0) n FROM provider_budgets').get().n,0);
 });
});

test('incomplete accounts retain static crisis resources without any generation or quota consumption',async t=>{
 const s=fixture(t);let calls=0;t.mock.method(globalThis,'fetch',async()=>{calls++;throw Error('Provider forbidden')});
 for(const [content,pattern] of [['Je veux mourir ce soir.',/3114/],['Mon conjoint menace de me tuer ce soir.',/17|112/]]){
  const result=await s.send(content,'incomplete-crisis-'+content.length);assert.equal(result.status,403);assert.equal(result.code,'prerequisites_required');assert.equal(result.readiness.complete,false);assert.equal(result.safety,true);assert.match(result.safety_reply,pattern);assert.match(result.error,pattern);
 }
 const ordinary=await s.send('Bonjour.','incomplete-ordinary');assert.equal(ordinary.status,403);assert.equal(ordinary.safety,undefined);
 assert.equal(calls,0);assert.equal(s.raw.prepare('SELECT count(*) n FROM monthly_usage').get().n,0);assert.equal(s.raw.prepare('SELECT count(*) n FROM provider_budgets').get().n,0);
});

test('all 516 symptoms are required with JS/SQL parity and past precision',async t=>{
 const {SYMPTOMES_CARTE}=await import('../src/symptomes-carte.generated.mjs');const ids=SYMPTOMES_CARTE.factors.filter(f=>!f.historical).map(f=>f.id),s=fixture(t);seedCompanionPrerequisites(s.raw);assert.equal(ids.length,516);
 const check=async(expected)=>{const r=await readCompanionReadiness(s.env,'a');assert.equal(r.complete,expected);assert.equal(r.symptoms.required,516);assert.equal(s.raw.prepare('SELECT '+companionReadinessSql('?')+' ok').get('a').ok,Number(expected));return r;};
 for(const value of [0,1,'unknown','past']){const answers=Object.fromEntries(ids.map(id=>[id,value])),notes={factors:Object.fromEntries(ids.map(id=>[id,'Avant les douze derniers mois.']))};s.raw.prepare("UPDATE symptom_maps SET answers=?,notes=? WHERE user_id='a'").run(JSON.stringify(answers),JSON.stringify(notes));await check(true);}
 s.raw.prepare("UPDATE symptom_maps SET notes='{}' WHERE user_id='a'").run();await check(false);
 const answers=Object.fromEntries(ids.map(id=>[id,0]));delete answers[ids[0]];answers.unrelated=1;s.raw.prepare("UPDATE symptom_maps SET answers=? WHERE user_id='a'").run(JSON.stringify(answers));assert.equal((await check(false)).symptoms.answered,515);
 answers[ids[0]]=2;s.raw.prepare("UPDATE symptom_maps SET answers=? WHERE user_id='a'").run(JSON.stringify(answers));await check(false);
 answers[ids[0]]=0;s.raw.prepare("UPDATE symptom_maps SET answers=? WHERE user_id='a'").run(JSON.stringify(answers));await check(true);
 s.raw.prepare("DELETE FROM symptom_maps WHERE user_id='a'").run();assert.equal((await check(false)).symptoms.answered,0);
});

test('existing Ambre journeys cannot send or request a review with only 515 symptoms',async t=>{
 const {SYMPTOMES_CARTE}=await import('../src/symptomes-carte.generated.mjs');const s=fixture(t);await s.ready();const answers=Object.fromEntries(SYMPTOMES_CARTE.factors.slice(1).map(f=>[f.id,0]));s.raw.prepare("UPDATE symptom_maps SET answers=? WHERE user_id='a'").run(JSON.stringify(answers));let calls=0;provider(t,()=>{calls++;throw Error('Must not call provider')});
 for(const [path,data] of [['/api/chat/messages',{content:'Bonjour Ambre',idempotency_key:'symptoms-missing-send'}],['/api/chat/review',{idempotency_key:'symptoms-missing-review'}]]){const r=await s.call(path,'POST',data);assert.equal(r.status,403,JSON.stringify(r));assert.equal(r.code,'prerequisites_required');}assert.equal(calls,0);
});
