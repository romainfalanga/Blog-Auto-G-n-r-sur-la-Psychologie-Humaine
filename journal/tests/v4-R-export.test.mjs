// Lot R, F4: complete, paginated chat export with declared coverage.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider,filledProfile,seedTurns} from './v4-R-helper.mjs';

test('F4: 3 000 traces are all exported with an explicit coverage and a bounded size',async t=>{
 const base=setup();filledProfile(base);await base.ready();provider(t);
 await base.call('/api/chat/messages','POST',{content:'Je rumine quand ma sœur me critique.',idempotency_key:'real_turn_0001'});
 const trace=base.raw.prepare('SELECT trace FROM chat_turn_traces').get().trace;
 const s=setup();await s.ready();seedTurns(s,'a',3000,{trace,traceAt:i=>s.now()-5000+i,jobAt:s.now()});
 for(let i=0;i<450;i++)s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,'point','ready','Bilan',?,?,?)").run('r'+i,'a','rk'+i,i,i);
 const data=await s.chat.exportData(s.env,'a'),bytes=JSON.stringify(data).length,c=data.chat_export_coverage;
 t.diagnostic(JSON.stringify({bytes,traces:c.traces,included:c.included}));
 assert.equal(data.chat_traces.length,3000);assert.equal(c.included.chat_traces,3000);assert.equal(c.traces.total,3000);
 assert.equal(c.traces.with_exact_inputs+c.traces.exact_inputs_omitted_for_size+c.traces.summarized_by_retention,3000);
 assert.ok(c.traces.exact_inputs_omitted_for_size>0);
 assert.equal(data.chat_messages.length,6000);assert.equal(data.ambre_reviews.length,451);
 assert.ok(bytes<32e6,`${bytes} bytes`);
 // Newest traces keep their exact inputs; omitted ones say so.
 assert.ok(JSON.parse(data.chat_traces[0].trace).details.values.length>0);
 assert.equal(JSON.parse(data.chat_traces.at(-1).trace).details.export_omitted,1);
 assert.equal(new Set(data.chat_traces.map(x=>x.turn_id)).size,3000);
});
