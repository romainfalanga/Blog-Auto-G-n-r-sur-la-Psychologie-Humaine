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

// Lot A5-P1: the export carries each review's digest (with its prose) and the person's feedback,
// and the review thread as Ambre receives it at a point (bounded, whatever the number of reviews).
test('A5-P1: digests, feedback and the bounded review thread are exported',async()=>{
 const s=setup();await s.ready();
 const digest=JSON.stringify({v:1,formulation:{holds:[{text:'Le silence protège du conflit',quote:'je me tais pour éviter les disputes'}],revised:[],new:[]},goals:[{key:'parler',words:'oser parler en réunion',status:'ongoing',quote:'je me tais pour éviter les disputes',next_step:'une phrase mardi'}],tried:[],alliance:null,open_questions:['Qu’est-ce qui aide ?'],next_focus:'La parole en réunion',safety:'',dropped:0});
 for(let i=0;i<300;i++)s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at,digest,feedback) VALUES(?,?,'point','ready','Bilan',?,?,?,?,?)").run('r'+i,'a','rk'+i,i+1,i+1,digest,i===299?JSON.stringify({verdict:'to_correct',text:'Ce n’était pas en réunion.',at:5}):null);
 const data=await s.chat.exportData(s.env,'a'),latest=data.ambre_reviews.find(r=>r.id==='r299');
 assert.equal(data.ambre_reviews.length,301);
 assert.equal(latest.digest.goals[0].status,'ongoing');assert.match(latest.digest_text,/« oser parler en réunion » : en cours/);assert.deepEqual(latest.feedback,{verdict:'to_correct',text:'Ce n’était pas en réunion.',at:5});
 assert.equal(data.ambre_reviews.find(r=>r.id==='seed-a').feedback,null);
 const thread=data.ambre_review_thread;assert.equal(thread.thread.last.id,'r299');assert.equal(thread.thread.reviews,12,'the thread reads twelve reviews at most');
 assert.ok(JSON.stringify(thread.thread).length<=3000);assert.match(thread.text,/Objectifs suivis : « oser parler en réunion »/);assert.match(thread.text,/À corriger/);
 assert.equal((await s.chat.exportData(s.env,'b')).ambre_review_thread,null);
});
