// Plan « Ambre premier message » v1.1, lots 1 and 2 (server) and the lot 3 API contract.
// Simulated provider only: no paid call. Point numbers refer to the v1.1 section of
// docs/plan-ambre-premier-message-2026-11-19.md.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider} from './v4-R-helper.mjs';
import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
import {SYMPTOMES_CARTE} from '../src/symptomes-carte.generated.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {readReviewSnapshot,snapshotChunks,snapshotPersonCorpus,INTERFACE_LABELS} from '../src/review-snapshot.mjs';
import {reviewChecks,classifyQuotes,chatMessageChecks,fallbackChatMessage,reviewVerdict,REVIEW_CHECK_CRITERIA,CHAT_MESSAGE_MAX} from '../src/review-digest.mjs';
import {HYPOTHESIS_RULE,buildReviewSystemPrompt} from '../src/psychology-framework.mjs';
import {MAP_HYPOTHESIS_RULES} from '../src/symptom-map-context.mjs';
import {initialReviewEstimate,observedReviewEstimate} from '../src/chat-backend.mjs';

const PASS={criteria:{no_diagnosis:'yes',quotes_faithful:'yes',safety_resources:'na',hypotheses_open:'yes',no_saved_claim:'yes',sources_honest:'yes',no_internal:'yes'},reasons:[]};
const FAIL_DIAGNOSIS={...PASS,criteria:{...PASS.criteria,no_diagnosis:'no'}};
const SELF='Je me tais souvent pour éviter les conflits avec ma sœur.';
const PAST=SYMPTOMES_CARTE.factors.find(f=>f.id==='tristesse_persistante');
const DIGEST={formulation:{holds:[],revised:[],new:[]},goals:[],tried:[],alliance:{text:'',quote:''},open_questions:[],next_focus:'',safety:''};
// The two contradictory verdicts of the 19/11 acceptance run: an interface label (« Dans le passé »)
// taken for an invented quote, and a hypothesis taken for a diagnosis.
const CONTENT='Ce qui semble ressortir. Vous écrivez : « '+SELF+' ». À la question « '+PAST.label+' », vous avez répondu « Dans le passé » : cela concerne une période antérieure aux douze derniers mois, pas un signe actuel. Hypothèse : un évitement des conflits, appuyée par ce que vous écrivez et par vos réponses ; il manque des exemples récents. '+'Nous pourrons vérifier ensemble ce qui vous ressemble et ce qui ne vous ressemble pas, sans conclusion hâtive. '.repeat(12)+'Qu’avez-vous envie de comprendre en priorité ?';
const CHAT='Votre bilan est prêt. Ce qui ressort : hypothèse : un évitement des conflits, appuyée par ce que vous écrivez ; il manque des exemples récents. À la question « '+PAST.label+' », vous avez répondu « Dans le passé ». Qu’aimeriez-vous creuser en premier ? Qu’en pensez-vous ?';
// Corrections 19/11: the published message keeps every word, its questions at the end, one per line.
const CHAT_SHOWN=CHAT.replace(' Qu’aimeriez-vous creuser en premier ? Qu’en pensez-vous ?','\n\nQu’aimeriez-vous creuser en premier ?\nQu’en pensez-vous ?');

async function fresh(t,{env}={}){
 const s=setup({env});t.after(()=>s.raw.close());
 s.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({first_name:'Alix',self_description:SELF}));
 seedCompanionPrerequisites(s.raw,'a');
 // Complete synthetic profile: 516 answers with Oui, Je ne sais pas and Dans le passé; childhood with Oui.
 const answers=JSON.parse(s.raw.prepare("SELECT answers FROM symptom_maps WHERE user_id='a'").get().answers),plain=SYMPTOMES_CARTE.factors.filter(f=>!f.historical&&!f.safety);
 answers[PAST.id]='past';plain.slice(1,6).forEach(f=>{answers[f.id]=1});plain.slice(6,9).forEach(f=>{answers[f.id]='unknown'});
 s.raw.prepare("UPDATE symptom_maps SET answers=?,notes=? WHERE user_id='a'").run(JSON.stringify(answers),JSON.stringify({factors:{[PAST.id]:'Surtout pendant mes études, plus maintenant.'}}));
 const child=JSON.parse(s.raw.prepare("SELECT answers FROM cartographies WHERE user_id='a'").get().answers);CARTOGRAPHIE.factors.filter(f=>!f.historical).slice(0,4).forEach(f=>{child[f.id]=1});
 s.raw.prepare("UPDATE cartographies SET answers=? WHERE user_id='a'").run(JSON.stringify(child));
 return s;
}
function mock(t,{onChunk,write,check}={}){
 return provider(t,async(purpose,body,calls)=>{
  if(purpose==='ambre_snapshot_chunk'){await onChunk?.(calls.filter(c=>c.purpose===purpose).length);return {summary:'Bloc analysé : faits déclarés, réponses et nuances.',quotes:[]};}
  if(purpose==='ambre_review')return write?write(calls):{content:CONTENT,chat_message:CHAT,digest:DIGEST};
  if(purpose==='ambre_review_check')return check?check(calls):PASS;
 });
}
const start=(s,key='start-key-0001')=>s.call('/api/chat/start','POST',{idempotency_key:key,accepted:true});
const review=(s,id)=>s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(id);
const consent=s=>s.raw.prepare("SELECT consent FROM chat_settings WHERE user_id='a'").get()?.consent??null;

test('rehearsal: complete synthetic initial review with a portrait changing during processing and « Dans le passé » ends ready',async t=>{
 const s=await fresh(t),progress=[];
 const calls=mock(t,{onChunk:async n=>{
  if(n===1)s.raw.prepare("INSERT INTO user_portraits(user_id,portrait,analyzed_version) VALUES('a',?,99) ON CONFLICT(user_id) DO UPDATE SET portrait=excluded.portrait,analyzed_version=99").run(JSON.stringify({summary:'Portrait recalculé par la tâche planifiée pendant le bilan.'}));
  progress.push((await s.call('/api/chat')).journey.review);
 }});
 assert.equal(consent(s),null,'no agreement before the launch');
 const r=await start(s);assert.equal(r.status,202,JSON.stringify(r));
 const row=review(s,r.review_id);assert.equal(row.status,'ready',JSON.stringify({code:row.error_code}));
 const snap=await readReviewSnapshot(s.env,'a',r.review_id),sections=snap.payload.sections;
 assert.equal(Object.keys(sections.symptoms[0].answers).length,516);assert.equal(sections.symptoms[0].answers[PAST.id],'past');
 assert.equal(sections.people.length,3);assert.equal(sections.events.length,3);assert.equal(sections.childhood.length,1);assert.deepEqual(sections.portrait,[],'the portrait written after the capture is not mixed in');
 const chunks=snapshotChunks(snap.payload),chunkCalls=calls.filter(c=>c.purpose==='ambre_snapshot_chunk');assert.equal(chunkCalls.length,chunks.length);
 // Progress seen by the page during the processing (lot 3 contract).
 assert.equal(progress.length,chunks.length);
 progress.forEach((p,i)=>{assert.equal(p.running_id,r.review_id);assert.equal(p.stage,'reading');assert.equal(p.blocks_total,chunks.length);assert.equal(p.blocks_done,i);assert.equal(p.started_at,row.created_at);assert.equal(p.last_failure,null);assert.equal(p.estimate.basis,'initial');assert.ok(p.estimate.p50_seconds<=p.estimate.p90_seconds);});
 assert.match(chunkCalls[0].body.messages[0].content,/past = Dans le passé/);
 // One writing call (document + conversation message), checked with the same criteria.
 const write=calls.filter(c=>c.purpose==='ambre_review');assert.equal(write.length,1);assert.equal(write[0].body.max_tokens,5600);assert.ok(write[0].body.response_format.json_schema.schema.properties.chat_message);
 assert.ok(write[0].body.messages[0].content.includes(HYPOTHESIS_RULE));
 const checkCall=calls.find(c=>c.purpose==='ambre_review_check'),check=JSON.parse(checkCall.body.messages[1].content);
 assert.deepEqual(check.deterministic_failures,[]);assert.deepEqual(check.unfound_quotes,[]);assert.ok(check.interface_labels_quoted.includes('Dans le passé'));assert.equal(check.chat_message,CHAT_SHOWN);
 assert.ok(checkCall.body.messages[0].content.includes(HYPOTHESIS_RULE),'the check uses the writer’s hypothesis rule');
 // Mes bilans keeps the document; the conversation shows the message and its links.
 assert.equal(row.content,CONTENT);
 const state=await s.call('/api/chat'),message=state.messages.at(-1);
 assert.equal(message.review_id,r.review_id);assert.equal(message.chat_message,CHAT_SHOWN);assert.equal(message.content,CHAT_SHOWN);assert.deepEqual(message.links,[{label:'Lire le bilan complet',href:'/mes-bilans/'},{label:'Voir Mes diagnostics',href:'/mes-diagnostics/'}]);
 assert.equal(state.consent,true);assert.equal(consent(s),1);assert.equal(state.journey.started,true);assert.equal(state.journey.review.running_id,null);assert.equal(state.journey.review.last_failure,null);
 assert.equal((await s.call('/api/reviews/'+r.review_id)).review.content,CONTENT);
 assert.equal(JSON.parse(row.sources).blocks_total,chunks.length);
});

test('point 2: deleting a proche present in the snapshot ends the review with sources_changed and erases the snapshot; relaunch works',async t=>{
 const s=await fresh(t);s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role) VALUES('extra','a','extra','h','Quatrième proche','friend')").run();
 let calls=mock(t,{onChunk:n=>{if(n===1)s.raw.prepare("DELETE FROM personal_people WHERE id='extra'").run();}});
 const r=await start(s),row=review(s,r.review_id);
 assert.equal(row.status,'failed');assert.equal(row.error_code,'sources_changed');assert.equal(await readReviewSnapshot(s.env,'a',r.review_id),null);
 assert.equal(calls.filter(c=>c.purpose==='ambre_review').length,0,'no synthesis bought after the deletion');
 assert.equal(s.raw.prepare("SELECT count(*) n FROM chat_turns WHERE user_id='a' AND idempotency_key LIKE 'review-%'").get().n,0);
 assert.equal(s.raw.prepare('SELECT state FROM monthly_usage WHERE operation_id=?').get(r.review_id).state,'released','the monthly credit is given back');
 const failed=(await s.call('/api/chat')).journey;
 assert.deepEqual(failed.review.last_failure,{id:r.review_id,kind:'initial',error_code:'sources_changed',retryable:true,retry_at:null,daily_limit:8,daily_remaining:7});
 assert.equal((await s.call('/api/chat')).journey.review.last_failure.id,r.review_id,'the failure survives a reload');
 calls=mock(t);const again=await start(s,'start-key-0002');assert.equal(again.status,202);assert.equal(review(s,again.review_id).status,'ready');
 assert.equal((await s.call('/api/chat')).journey.review.last_failure,null);
});

test('point 2: withdrawing the agreement or deleting the account during the review publishes nothing and keeps no archive',async t=>{
 const s=await fresh(t);mock(t,{onChunk:async n=>{if(n===1)await s.call('/api/chat/consent','PUT',{accepted:false});}});
 const r=await start(s),row=review(s,r.review_id);assert.equal(row.status,'failed');assert.equal(row.error_code,'consent_withdrawn');assert.equal(await readReviewSnapshot(s.env,'a',r.review_id),null);
 const failure=(await s.call('/api/chat')).journey.review.last_failure;assert.equal(failure.error_code,'consent_withdrawn');assert.equal(failure.retryable,false);
 const d=await fresh(t);mock(t,{onChunk:n=>{if(n===1)d.raw.prepare("DELETE FROM users WHERE id='a'").run();}});
 await start(d);
 for(const table of ['ambre_reviews','review_snapshots','chat_turns','chat_settings'])assert.equal(d.raw.prepare(`SELECT count(*) n FROM ${table} WHERE user_id='a'`).get().n,0,table);
});

test('point 3: a safety signal declared after the capture imposes 3114 and 15/112 in the message and the document',async t=>{
 const s=await fresh(t);
 mock(t,{onChunk:n=>{if(n===1){const answers=JSON.parse(s.raw.prepare("SELECT answers FROM symptom_maps WHERE user_id='a'").get().answers);answers.idees_de_mort=1;s.raw.prepare("UPDATE symptom_maps SET answers=?,safety_at=? WHERE user_id='a'").run(JSON.stringify(answers),s.now());}}});
 const r=await start(s),row=review(s,r.review_id);assert.equal(row.status,'ready',row.error_code);
 const message=(await s.call('/api/chat')).messages.at(-1);
 assert.match(message.chat_message,/^Depuis la préparation de ce bilan/);assert.match(message.chat_message,/\b3114\b/);assert.match(message.chat_message,/\b(?:15|112)\b/);
 assert.match(row.content,/^En France, vous pouvez appeler le 3114/,'resources in first position of the document');assert.equal(JSON.parse(row.sources).check.safety_added,'self_harm');
 assert.equal(JSON.parse(row.sources).check.passed,true);
});

test('point 4: the capture re-reads a concurrent write up to twice before context_changed',async t=>{
 for(const [races,status] of [[1,'ready'],[3,'failed']]){
  const s=await fresh(t);mock(t);const prepare=s.env.DB.prepare.bind(s.env.DB);let raced=0;
  s.env.DB.prepare=sql=>{const st=prepare(sql);if(sql.startsWith('WITH owner AS')){const first=st.first.bind(st);st.first=async()=>{const row=await first();if(raced<races){raced++;s.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.occupation',?) WHERE user_id='a'").run('Métier '+raced);}return row;};}return st;};
  const r=await start(s),row=review(s,r.review_id);assert.equal(row.status,status,races+' race(s)');
  if(status==='failed'){assert.equal(row.error_code,'context_changed');assert.equal(raced,3);}
 }
});

test('point 5: a pending portrait refresh is awaited up to three minutes, then the capture is taken without it and says so',async t=>{
 const s=await fresh(t),calls=mock(t);
 s.raw.prepare("INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,status) SELECT 'job1','a','portrait',version,1,'pending' FROM profiles WHERE user_id='a'").run();
 const r=await start(s);let row=review(s,r.review_id);
 assert.equal(row.status,'running');assert.equal(row.error_code,'portrait_pending');assert.equal(row.attempts,0);assert.equal(calls.length,0);
 assert.equal((await s.call('/api/chat')).journey.review.stage,'capture');
 s.tick(60);await s.chat.resumeReviews(s.env);assert.equal(review(s,r.review_id).status,'running','still waiting under three minutes');assert.equal(calls.length,0);
 s.tick(130);await s.chat.resumeReviews(s.env);row=review(s,r.review_id);assert.equal(row.status,'ready',row.error_code);
 const snap=await readReviewSnapshot(s.env,'a',r.review_id);assert.deepEqual(snap.payload.omitted.map(x=>x.reason),['refresh_pending']);
 assert.match(calls.find(c=>c.purpose==='ambre_review').body.messages[1].content,/refresh_pending/);assert.deepEqual(JSON.parse(row.sources).omitted,['portrait']);
});

test('point 6: quotes in three classes — the person’s words, interface labels as labels, the rest is invented',()=>{
 const corpus=SELF,labels=INTERFACE_LABELS;
 const ok=classifyQuotes(CONTENT,{corpus,labels});assert.deepEqual(ok.invented,[]);assert.deepEqual(ok.misattributed,[]);assert.ok(ok.label.includes('Dans le passé'));assert.ok(ok.person.includes(SELF));
 assert.deepEqual(reviewChecks(CONTENT,{corpus,labels}).failed,[],'label and hypothesis pass the deterministic checks');
 const asWords=reviewChecks('Vous écrivez « Dans le passé » pour décrire cette période.',{corpus,labels});assert.ok(asWords.failed.includes('invented_quote'));assert.deepEqual(asWords.misattributed,['Dans le passé']);
 // A text produced by an AI (here a portrait sentence) is no longer a valid source of quotes.
 const portrait='Vous semblez porter une vigilance ancienne envers les critiques.';
 const ai=reviewChecks('Le portrait note « '+portrait+' ».',{supplied:corpus+'\n'+portrait,corpus,labels});assert.ok(ai.failed.includes('invented_quote'));
 assert.ok(reviewChecks('Vous dites « je ne supporte plus rien du tout ».',{corpus,labels}).failed.includes('invented_quote'));
 assert.ok(reviewChecks('Vous avez un trouble anxieux.',{corpus,labels}).failed.includes('diagnosis'),'a verdict stays blocking');
});

test('point 7: one hypothesis rule shared by the writer, the check, the symptom rules; contradictory verdicts are removed',()=>{
 for(const prompt of [buildReviewSystemPrompt('initial'),buildReviewSystemPrompt('point'),REVIEW_CHECK_CRITERIA.no_diagnosis,MAP_HYPOTHESIS_RULES])assert.ok(prompt.includes(HYPOTHESIS_RULE));
 assert.match(HYPOTHESIS_RULE,/hypothèse : X, appuyée par A et B ; il manque C/);assert.match(HYPOTHESIS_RULE,/jamais « vous avez \[un trouble\] »/);
 assert.match(REVIEW_CHECK_CRITERIA.hypotheses_open,/hypothèse : X, appuyée par A et B ; il manque C/);
 assert.match(REVIEW_CHECK_CRITERIA.quotes_faithful,/Dans le passé/);
 // No criterion is disabled: a "no" from the model still fails the review.
 assert.equal(reviewVerdict(FAIL_DIAGNOSIS,{failed:[]}).passed,false);assert.equal(reviewVerdict({...PASS,criteria:{...PASS.criteria,quotes_faithful:'no'}},{failed:[]}).passed,false);
});

test('point 8: an oversized chat_message is replaced by the deterministic fallback, checked like the document; old envelopes keep content',async t=>{
 const s=await fresh(t);const calls=mock(t,{write:()=>({content:CONTENT,chat_message:'Ce qui ressort de votre bilan. '.repeat(120),digest:DIGEST})});
 const r=await start(s),row=review(s,r.review_id);assert.equal(row.status,'ready',row.error_code);assert.equal(JSON.parse(row.sources).check.chat_fallback,true);
 const message=(await s.call('/api/chat')).messages.at(-1);assert.ok(message.chat_message.length<=CHAT_MESSAGE_MAX);assert.match(message.chat_message,/^Votre bilan est prêt/);
 const questions=(message.chat_message.match(/\?/g)||[]).length;assert.ok(questions>=2&&questions<=3);
 assert.equal(JSON.parse(calls.find(c=>c.purpose==='ambre_review_check').body.messages[1].content).chat_message,message.chat_message);
 // Shape checks of the message: questions are counted, the deterministic checks apply.
 assert.ok(chatMessageChecks('Une seule question ?',{corpus:SELF,labels:INTERFACE_LABELS}).failed.includes('chat_questions'));
 assert.match(fallbackChatMessage(DIGEST,CONTENT,{signal:'self_harm'}),/^En France, vous pouvez appeler le 3114/);
 // An envelope written before this change has no chat_message: the client shows content.
 s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response) VALUES('old','a','review-old','review','completed','consumed','2026-01-05',1,1,'',?)").run(JSON.stringify({companion_reply:1,content:'Ancien bilan complet.',review_id:'old',review_kind:'initial'}));
 const old=(await s.call('/api/chat')).messages.find(m=>m.review_id==='old');assert.equal(old.content,'Ancien bilan complet.');assert.equal(old.chat_message,undefined);assert.equal(old.links,undefined);
});

test('point 9: the agreement is recorded only with a created review; a refused launch records nothing',async t=>{
 const s=await fresh(t);mock(t);
 s.raw.prepare("UPDATE symptom_maps SET answers=json_remove(answers,'$.'||?) WHERE user_id='a'").run(PAST.id);
 const refused=await start(s);assert.equal(refused.status,403);assert.equal(refused.code,'prerequisites_required');assert.equal(consent(s),null,'refused: no agreement recorded');
 s.raw.prepare("UPDATE symptom_maps SET answers=json_set(answers,'$.'||?,'past') WHERE user_id='a'").run(PAST.id);
 const r=await start(s,'start-key-0003');assert.equal(r.status,202);assert.equal(consent(s),1);
});

test('lot 3 contract: readiness.pages requires the answers and the recording; reload during the wait, failure, relaunch and quotas',async t=>{
 const s=await fresh(t,{env:{REVIEW_INLINE_TIMEOUT_MS:'0'}});
 s.raw.prepare("UPDATE symptom_maps SET locked_at=NULL WHERE user_id='a'").run();
 let state=await s.call('/api/chat');const page=state.readiness.pages.find(p=>p.id==='symptoms');
 assert.deepEqual(page,{id:'symptoms',label:'Mes symptômes',href:'/mes-symptomes/',done:516,total:516,saved:false,complete:false,answered:516,required:516});
 assert.deepEqual(state.readiness.pages.map(p=>[p.id,p.complete]),[['account',true],['people',true],['timeline',true],['childhood',true],['symptoms',false]]);assert.equal(state.journey.can_start,false);
 s.raw.prepare("UPDATE symptom_maps SET locked_at=1 WHERE user_id='a'").run();
 state=await s.call('/api/chat');assert.equal(state.journey.can_start,true);assert.equal(state.journey.review.estimate.basis,'initial_default');
 // Launch without an inline attempt: the reload shows the waiting review before the capture.
 const r=await start(s);assert.equal(r.status,202);
 for(let i=0;i<2;i++){const review=(await s.call('/api/chat')).journey.review;assert.equal(review.running_id,r.review_id);assert.equal(review.stage,'capture');assert.equal(review.blocks_total,null);assert.ok(review.estimate.p90_seconds>=review.estimate.p50_seconds);}
 // The check refuses twice (initial write and the single rewrite): review_not_approved, retryable.
 const calls=mock(t,{check:()=>FAIL_DIAGNOSIS});s.tick(61);await s.chat.resumeReviews(s.env,{timeoutMs:180000});
 assert.equal(review(s,r.review_id).error_code,'review_not_approved');assert.equal(calls.filter(c=>c.purpose==='ambre_review').length,2);
 const failure=(await s.call('/api/chat')).journey.review.last_failure;assert.equal(failure.retryable,true);assert.equal(failure.kind,'initial');
 assert.equal((await s.call('/api/chat')).quotas.reviews.remaining,1,'the monthly credit is restored after a failure');
 // Daily creation limit: shown and enforced.
 s.raw.prepare("UPDATE rate_limits SET count=8 WHERE key=?").run('review-created:a');
 const limited=(await s.call('/api/chat')).journey.review;assert.equal(limited.last_failure.daily_remaining,0);assert.equal(limited.last_failure.retry_at,limited.daily.resets_at);
 assert.equal((await start(s,'start-key-0004')).code,'request_limit');
 s.raw.prepare("UPDATE rate_limits SET count=1 WHERE key=?").run('review-created:a');
 mock(t);const again=await start(s,'start-key-0005');assert.equal(again.status,202);assert.equal((await s.call('/api/chat')).journey.review.last_failure,null,'a new review clears the failure');
 s.tick(61);await s.chat.resumeReviews(s.env);assert.equal(review(s,again.review_id).status,'ready');
 assert.equal((await s.call('/api/chat')).quotas.reviews.remaining,0);
 // A final cause is not retryable.
 const big=await fresh(t);big.raw.prepare("UPDATE profiles SET self_data=json_set(self_data,'$.self_description',?) WHERE user_id='a'").run('x'.repeat(530000));mock(t);
 const tooLarge=await start(big);assert.equal(review(big,tooLarge.review_id).error_code,'snapshot_too_large');
 assert.equal((await big.call('/api/chat')).journey.review.last_failure.retryable,false);
});

test('estimate: initial values from the execution model, observed p50/p90 from three ready reviews',async t=>{
 // Calibrated on the paid acceptance run of 19/11 (6 blocks ready in 122 s): 6 blocks → 2 to 4 minutes.
 assert.deepEqual(initialReviewEstimate(6),{p50_seconds:145,p90_seconds:215,basis:'initial'});assert.deepEqual(initialReviewEstimate(8),{p50_seconds:170,p90_seconds:240,basis:'initial'});assert.ok(initialReviewEstimate(6).p50_seconds>=122,'the measured run lies under p50');
 assert.equal(initialReviewEstimate(null).basis,'initial_default');assert.ok(initialReviewEstimate(20).p50_seconds>initialReviewEstimate(2).p50_seconds);
 assert.equal(observedReviewEstimate([100,200]),null);assert.deepEqual(observedReviewEstimate([300,100,200]),{p50_seconds:200,p90_seconds:300,basis:'observed',sample:3});
 const s=await fresh(t);
 for(const [i,d] of [[1,240],[2,300],[3,420]])s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at,sources) VALUES(?,'b','initial','ready','x',?,?,?,?)").run('obs'+i,'obs-'+i,1000*i,1000*i+d,JSON.stringify({snapshot_id:'obs'+i}));
 const estimate=(await s.call('/api/chat')).journey.review.estimate;assert.deepEqual(estimate,{p50_seconds:300,p90_seconds:420,basis:'observed',sample:3});
});

// ---- Corrections after the independent review (19/11) ----
test('review fix 2: Mes précisions (memories.text) are the person’s words for quotes',()=>{
 const words='Je note ici que les dimanches soirs sont les plus durs.';
 const corpus=snapshotPersonCorpus({personal_memory:[{text:words,proposals:'Proposition non validée de l’IA.'}]});
 assert.ok(corpus.includes(words));assert.ok(!corpus.includes('Proposition non validée'));
 assert.deepEqual(reviewChecks('Vous écrivez « '+words+' ».',{corpus,labels:INTERFACE_LABELS}).failed,[]);
});

test('review fix 3: an approved review that meets a safety change at publication resumes at the publication, without a new check nor a counted attempt',async t=>{
 const s=await fresh(t),calls=mock(t),prepare=s.env.DB.prepare.bind(s.env.DB);let raced=false;
 s.env.DB.prepare=sql=>{const st=prepare(sql);if(sql.startsWith('SELECT (COALESCE((SELECT safety_at')&&!raced){const first=st.first.bind(st);st.first=async()=>{const row=await first();raced=true;const answers=JSON.parse(s.raw.prepare("SELECT answers FROM symptom_maps WHERE user_id='a'").get().answers);answers.idees_de_mort=1;s.raw.prepare("UPDATE symptom_maps SET answers=?,safety_at=? WHERE user_id='a'").run(JSON.stringify(answers),s.now());return row;};}return st;};
 const r=await start(s);let row=review(s,r.review_id);
 assert.equal(raced,true);assert.equal(row.status,'running');assert.equal(row.error_code,'safety_changed');assert.equal(row.attempts,0,'no attempt consumed');assert.equal(row.stage,'check');assert.equal(JSON.parse(row.draft).approved,true);
 assert.equal((await s.call('/api/chat')).journey.review.stage,'publish');
 const checks=calls.filter(c=>c.purpose==='ambre_review_check').length;
 s.tick(61);await s.chat.resumeReviews(s.env);row=review(s,r.review_id);
 assert.equal(row.status,'ready',row.error_code);assert.equal(calls.filter(c=>c.purpose==='ambre_review_check').length,checks,'no second check');
 assert.equal(JSON.parse(row.sources).check.safety_added,'self_harm');assert.match((await s.call('/api/chat')).messages.at(-1).chat_message,/\b3114\b/);
});

test('review fix 5: a guard failing at publication reports the real cause (access, prerequisites), not context_changed',async t=>{
 for(const [cause,mutate] of [['access_required',s=>s.raw.prepare("UPDATE subscriptions SET status='canceled',paid_until=0 WHERE user_id='a'").run()],['prerequisites_required',s=>s.raw.prepare("UPDATE symptom_maps SET answers=json_remove(answers,'$.'||?) WHERE user_id='a'").run(PAST.id)]]){
  const s=await fresh(t);mock(t,{check:()=>{mutate(s);return PASS}});
  const r=await start(s),row=review(s,r.review_id);assert.equal(row.status,'failed');assert.equal(row.error_code,cause);assert.equal(await readReviewSnapshot(s.env,'a',r.review_id),null);
 }
});

test('review fix 6: only a portrait job the worker can run is awaited (batch, call cap and budget are not)',async t=>{
 for(const [columns,waits] of [["'pending',0,NULL",true],["'pending',0,'batch_required'",false],["'pending',20,NULL",false],["'running',0,'generation_budget'",false]]){
  const s=await fresh(t),calls=mock(t);
  s.raw.prepare(`INSERT INTO journal_jobs(id,user_id,kind,source_version,memory_version,status,provider_calls,error_code) SELECT 'job1','a','portrait',version,1,${columns} FROM profiles WHERE user_id='a'`).run();
  const r=await start(s),row=review(s,r.review_id);
  assert.equal(row.status,waits?'running':'ready',columns);assert.equal(calls.length>0,!waits);
 }
});

test('review minors: the fallback message asks only its own 2–3 questions; retry_at follows the oldest published review',async t=>{
 const content='Premier constat. Est-ce que cela vous parle ? Deuxième constat, plus nuancé. Que voulez-vous en faire ?\n\nTroisième paragraphe sans question.';
 const message=fallbackChatMessage(DIGEST,content);assert.equal((message.match(/\?/g)||[]).length,2);assert.doesNotMatch(message,/cela vous parle|voulez-vous en faire/);
 const withQuestions=fallbackChatMessage({...DIGEST,open_questions:['Par quoi voulez-vous commencer ?','Est-ce juste ? Ou pas ?']},content);assert.match(withQuestions,/Par quoi voulez-vous commencer \?/);assert.doesNotMatch(withQuestions,/Ou pas/);
 const s=await fresh(t);mock(t);
 for(const [i,hours] of [[1,20],[2,10],[3,5],[4,1]])s.raw.prepare(`INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,created_at) VALUES(?,'a',?,'review','completed','consumed','2026-11-16',1,1,'','{}',datetime('now','-${hours} hours'))`).run('pub'+i,'review-pub'+i);
 s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,idempotency_key,created_at,completed_at,error_code) VALUES('f1','a','initial','failed','f1',?,?,'review_not_approved')").run(s.now(),s.now());
 const failure=(await s.call('/api/chat')).journey.review.last_failure,expected=Math.floor(Date.now()/1000)-20*3600+86400;
 assert.equal(failure.retryable,true);assert.ok(Math.abs(failure.retry_at-expected)<=5,JSON.stringify({retry_at:failure.retry_at,expected}));
});

// Corrections 19/11 after the paid acceptance run: the document began with a technical sentence
// and the conversation message was one block of about 780 characters.
const INTERNAL='Ce bilan s’appuie sur des résumés de blocs d’un instantané, pas sur vos mots directs : les citations y sont rares.';
const ONE_BLOCK='Votre bilan est prêt. Hypothèse : une forte exigence envers vous-même, appuyée par vos réponses sur le travail et par votre présentation ; il manque des exemples récents. Une autre piste concerne la difficulté à demander de l’aide, appuyée par ce que vous racontez de vos proches ; il manque votre avis. Vos appuis sont nets : vos amis proches, la marche et la musique. Votre histoire peut éclairer ces pistes, à vérifier ensemble, sans conclusion hâtive. '+INTERNAL+' Qu’aimeriez-vous creuser en premier ? Qu’est-ce qui vous semble juste, ou à corriger ? Qu’en pensez-vous ?';
test('the conversation message is restructured into short paragraphs and questions one per line, without changing a word',async()=>{
 const {structureChatMessage}=await import('../src/review-digest.mjs');const out=structureChatMessage(ONE_BLOCK);
 assert.ok(out.paragraphs.length>=2&&out.paragraphs.length<=4,JSON.stringify(out.paragraphs));assert.ok(out.paragraphs.every(p=>p.length<=360&&!p.includes('?')));
 assert.deepEqual(out.questions,['Qu’aimeriez-vous creuser en premier ?','Qu’est-ce qui vous semble juste, ou à corriger ?','Qu’en pensez-vous ?']);
 assert.equal(out.text.replace(/\s+/g,' '),ONE_BLOCK.replace(/\s+/g,' '),'same words, same order');assert.match(out.text,/\n\nQu’aimeriez-vous creuser en premier \?\nQu’est-ce qui vous semble juste, ou à corriger \?\nQu’en pensez-vous \?$/);
 assert.equal(structureChatMessage(out.text).text,out.text,'idempotent');
 const fallback=fallbackChatMessage({...DIGEST,formulation:{holds:[{text:'Hypothèse : une forte exigence envers vous-même.',quote:''}],revised:[],new:[{text:'Hypothèse : une difficulté à demander de l’aide.',quote:''}]}},CONTENT);
 assert.ok(fallback.split('\n\n').length>=3,'the fallback also has paragraphs');assert.match(fallback,/\?\n.*\?$/);
});
test('the internal process is never mentioned: deterministic check, review criterion, one rewrite, then removal',async()=>{
 const {mentionsInternalProcess,withoutInternalProcess}=await import('../src/review-digest.mjs');
 assert.equal(mentionsInternalProcess(INTERNAL),true);for(const text of ['Le modèle de langage a relu vos citations vérifiables.','Selon l’instantané, vous…','Je me sens bloquée quand on me critique.','Vous avez surtout répondu aux questions ; quelques mots de votre part aideraient.'])assert.equal(mentionsInternalProcess(text),/modèle|instantané/.test(text),text);
 assert.equal(withoutInternalProcess(INTERNAL+' Vos appuis sont nets.\n\nDeuxième paragraphe.').text,'Vos appuis sont nets.\n\nDeuxième paragraphe.');
 assert.ok(reviewChecks(INTERNAL+' '+CONTENT,{corpus:SELF,labels:INTERFACE_LABELS}).failed.includes('internal_process'));
 assert.ok(REVIEW_CHECK_CRITERIA.no_internal);for(const kind of ['initial','point'])assert.match(buildReviewSystemPrompt(kind),/Fonctionnement interne\. Ni content ni chat_message ne décrivent comment le bilan est préparé/);
 assert.match(buildReviewSystemPrompt('initial'),/deux à quatre paragraphes courts séparés par une ligne vide/);
 assert.equal(reviewVerdict(PASS,{failed:['internal_process']}).passed,false,'a first mention asks for the rewrite');
 assert.equal(reviewVerdict(PASS,{failed:['internal_process']},{rewritten:true}).passed,true,'after the rewrite, the sentence is removed instead of losing the review');
 assert.equal(reviewVerdict({...PASS,criteria:{...PASS.criteria,no_internal:'no'}},{failed:[]}).passed,false);
});
test('a review that keeps mentioning its internal process after the rewrite is published without those sentences',async t=>{
 const s=await fresh(t);let writes=0;
 mock(t,{write:()=>(writes++,{content:INTERNAL+'\n\n'+CONTENT,chat_message:ONE_BLOCK,digest:DIGEST})});
 const r=await start(s),row=review(s,r.review_id);assert.equal(row.status,'ready',row.error_code);assert.equal(writes,2,'one rewrite');
 assert.doesNotMatch(row.content,/blocs|instantané|citations y sont rares/);assert.ok(row.content.startsWith('Ce qui semble ressortir.'));assert.ok(JSON.parse(row.sources).check.internal_removed>=2);
 const message=(await s.call('/api/chat')).messages.at(-1);assert.doesNotMatch(message.chat_message,/blocs|instantané/);
 assert.match(message.chat_message,/\n\nQu’aimeriez-vous creuser en premier \?\nQu’est-ce qui vous semble juste, ou à corriger \?\nQu’en pensez-vous \?$/);assert.ok(message.chat_message.split('\n\n').length>=3);
 assert.deepEqual(message.links.map(l=>l.href),['/mes-bilans/','/mes-diagnostics/']);
});
