import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {CARE_SCOPES} from '../src/care-backend.mjs';
import {CARE_INTERFACE_SCRIPT} from '../src/care-interface.mjs';
import {emptyPlan} from '../src/change-plan.mjs';
import {careSetup} from './care-helper.mjs';

function setup(t){const s=careSetup();t.after(()=>s.close());return s}
function seedPlan(s,{owner='a',trials=1,malformed=false}={}){
 const at=s.now()-30,document=emptyPlan();
 document.goals=[{id:'goal',revision:1,title:'Marcher avec une amie',why:'Retrouver un moment agréable',priority:1,status:'active',accepted_at:at,updated_at:at,source_quote:'PRIVATE_GOAL_QUOTE',source_turn_id:'PRIVATE_TURN_ID'}];
 document.steps=[{id:'step',revision:1,goal_id:'goal',text:'Choisir dix minutes de promenade',measure:'Mon ressenti après',strategy_id:'C04',status:'active',accepted_at:at,updated_at:at,professional_context:'PRIVATE_PROFESSIONAL_CONTEXT',source_quote:'PRIVATE_STEP_QUOTE'}];
 document.trials=Array.from({length:trials},(_,i)=>({id:'trial'+i,revision:1,step_id:'step',step_snapshot:{text:'Choisir dix minutes de promenade',measure:'Mon ressenti après',strategy_id:'C04',source_quote:'PRIVATE_NESTED_QUOTE',instructions:{text:'PRIVATE_NESTED_INSTRUCTION'}},status:'reported',attempt:'done',result_text:'J’ai apprécié ce moment',outcome:'helpful',occurred_on:null,timing_note:'Date non précisée',accepted_at:at,updated_at:at,source_quote:'PRIVATE_RESULT_QUOTE'}));
 document.proposals=[{id:'proposal',revision:1,status:'proposed',payload:{title:'PRIVATE_PROPOSAL'},source_quote:'PRIVATE_PROPOSAL_QUOTE'}];
 document.instructions=[{id:'instruction',text:'PRIVATE_CARE_INSTRUCTION',status:'current'}];
 document.preparations=[{id:'preparation',content:'PRIVATE_PERSONAL_PREPARATION',private:true}];
 if(malformed){document.goals.push(null,{id:'unadopted',title:'PRIVATE_UNADOPTED_GOAL'},{id:'typed',accepted_at:at,title:{source_quote:'PRIVATE_MALFORMED_FIELD'}});document.steps.push({id:'orphan',goal_id:'unadopted',accepted_at:at,text:'PRIVATE_ORPHAN_STEP'});document.trials.push({id:'unadopted-trial',step_id:'step',result_text:'PRIVATE_UNADOPTED_TRIAL'},{id:'typed-trial',step_id:'step',accepted_at:at,step_snapshot:{text:{source_quote:'PRIVATE_NESTED_MALFORMED'},measure:{instructions:'PRIVATE_BAD_MEASURE'}}})}
 s.raw.prepare('INSERT INTO change_plans(user_id,revision,document_json,updated_at) VALUES(?,1,?,?)').run(owner,JSON.stringify(document),s.now());return document;
}
function replacePlan(s,document,owner='a'){s.raw.prepare('UPDATE change_plans SET document_json=?,revision=revision+1,updated_at=? WHERE user_id=?').run(JSON.stringify(document),s.now(),owner)}
async function prepare(s,link,id='session'){s.readyGeneration(link.user_id);s.session(id);return s.call('/api/care/briefs','POST',{link_id:link.id,session_id:id})}

test('plan remains an opt-in scope alongside symptoms and never appears on preexisting links',async t=>{
 const s=setup(t),legacy=await s.share(['conversation','cartography','portrait','people','life_events','reviews']);seedPlan(s);
 assert.equal(CARE_SCOPES.length,10);assert.equal(CARE_SCOPES.at(-1),'plan');
 const state=await s.call('/api/care'),overview=await s.call(`/api/care/dossiers/${legacy.id}`,'GET',undefined,'pro');
 assert.equal(state.links[0].scopes.includes('plan'),false);assert.equal(overview.scopes.includes('plan'),false);
 assert.match(overview.sharing_notice,/mémoire.*explicitement partagée/i);assert.doesNotMatch(JSON.stringify(overview),/PRIVATE_|Marcher avec une amie|instructions|preparations/);
 assert.equal((await s.call(`/api/care/dossiers/${legacy.id}/plan`,'GET',undefined,'pro')).httpStatus,404);
 assert.equal((await s.call('/api/care/links','POST',{email:'pro@example.test',scopes:CARE_SCOPES,accepted:true})).httpStatus,201);
 assert.equal((await s.call('/api/care/links','POST',{email:'pro@example.test',scopes:[...CARE_SCOPES,'provider_traces'],accepted:true})).code,'invalid_scopes');
});

test('shared plan strips private and malformed nested fields, unadopted objects and foreign tenants',async t=>{
 const s=setup(t);seedPlan(s,{malformed:true});seedPlan(s,{owner:'b'});const link=await s.share(['plan']);
 const result=await s.call(`/api/care/dossiers/${link.id}/plan`,'GET',undefined,'pro');assert.equal(result.httpStatus,200);assert.match(JSON.stringify(result.data),/Marcher avec une amie|J’ai apprécié ce moment/);
 assert.doesNotMatch(JSON.stringify(result.data),/PRIVATE_|source_quote|source_turn_id|instructions|professional_context|proposals|preparations/);
 assert.deepEqual(result.data.trials[0].step_snapshot,{text:'Choisir dix minutes de promenade',measure:'Mon ressenti après',strategy_id:'C04'});
 for(const user of ['b','other'])assert.equal((await s.call(`/api/care/dossiers/${link.id}/plan`,'GET',undefined,user)).httpStatus,404);
 assert.equal((await s.call(`/api/care/dossiers/${link.id}/plan`,'PATCH',{title:'Mutation'},'pro')).httpStatus,404);
 assert.equal((await s.call(`/api/care/dossiers/${link.id}/conversation`,'GET',undefined,'pro')).httpStatus,404);
 assert.ok((await s.call('/api/care/access-log')).entries.some(entry=>entry.action==='read'&&entry.scope==='plan'));
});

test('shared plan pagination covers every adopted object in pages of at most fifty',async t=>{
 const s=setup(t);seedPlan(s,{trials:67});const link=await s.share(['plan']);let offset=0,ids=[];
 do{const result=await s.call(`/api/care/dossiers/${link.id}/plan?offset=${offset}`,'GET',undefined,'pro');assert.equal(result.httpStatus,200);const page=['goals','steps','trials'].flatMap(kind=>result.data[kind]);assert.ok(page.length<=50);ids.push(...page.map(item=>item.id));offset=result.next_offset}while(offset!==null);
 assert.equal(ids.length,69);assert.equal(new Set(ids).size,69);
});

test('plan-only point sends adopted choices and reported results, with no raw private sources',async t=>{
 const s=setup(t);seedPlan(s);s.turn('private-conversation','PRIVATE_CONVERSATION');s.raw.prepare("INSERT INTO chat_memory(user_id,summary,updated_at) VALUES('a','{\"goals\":[\"PRIVATE_MEMORY_GOAL\"]}',?)").run(s.now());const link=await s.share(['plan']),queued=await prepare(s,link);
 assert.equal(queued.httpStatus,202);assert.equal(s.calls.length,1);const input=JSON.parse(s.calls[0].user);assert.equal(input.sources.length,3);assert.ok(input.sources.every(row=>row.id.startsWith('plan:')));assert.doesNotMatch(s.calls[0].user,/PRIVATE_|source_quote|instructions|preparations|proposals/);
 assert.match(input.sources.find(row=>row.id==='plan:trials:trial0').provenance,/date non précisée/);assert.match(s.calls[0].system,/Un objectif choisi ne prouve pas une action réalisée/);
 const result=await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro');assert.equal(result.brief.status,'ready');assert.equal(result.brief.coverage.processed,3);
 const refs=s.raw.prepare("SELECT entity_kind,entity_id FROM change_artifact_refs WHERE artifact_kind='brief' AND artifact_id=?").all(queued.id);assert.equal(refs.length,3);assert.deepEqual(new Set(refs.map(row=>row.entity_id)),new Set(['goal','step','trial0']));
});

test('an unshared plan never enters a point and its changes do not invalidate shared conversation',async t=>{
 const s=setup(t),document=seedPlan(s),link=await s.share(['conversation']);s.turn('turn');s.setGenerator(async(_env,options)=>{assert.doesNotMatch(options.user,/Marcher avec|J’ai apprécié|PRIVATE_|plan:/);document.goals[0].title='Choix modifié hors partage';replacePlan(s,document);return {text:'{"items":[]}'}});
 const queued=await prepare(s,link);assert.equal((await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro')).brief.status,'ready');assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_artifact_refs').get().n,0);
});

test('revoking or removing plan consent during generation clears content and prevents publication',async t=>{
 for(const mode of ['revoke','reduce']){const s=careSetup();try{seedPlan(s);const link=await s.share(['plan','people']);s.setGenerator(async()=>{await s.call(`/api/care/links/${link.id}`,mode==='revoke'?'DELETE':'PUT',{revision:link.revision,accepted:true,scopes:['people']});return {text:JSON.stringify({items:[{source_id:'plan:goals:goal',quote:'Marcher avec une amie',theme:'Faits rapportés'}]})}});const queued=await prepare(s,link),row=s.raw.prepare('SELECT * FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,'revoked');assert.equal(row.content,'');assert.equal(row.notes,'[]');assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_artifact_refs').get().n,3);assert.equal((await s.call(`/api/care/dossiers/${link.id}/plan`,'GET',undefined,'pro')).httpStatus,404);assert.equal((await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro')).httpStatus,404)}finally{s.close()}}
});

test('plan source changes and physical deletion during generation invalidate all excerpts',async t=>{
 for(const mode of ['edit','delete']){const s=careSetup();try{const document=seedPlan(s),link=await s.share(['plan']);s.setGenerator(async()=>{if(mode==='edit'){document.goals[0].title='Objectif corrigé';replacePlan(s,document)}else s.raw.prepare("DELETE FROM change_plans WHERE user_id='a'").run();return {text:'{"items":[]}'}});const queued=await prepare(s,link),row=s.raw.prepare('SELECT * FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,'failed',mode);assert.equal(row.error_code,'sources_changed');assert.equal(row.content,'');assert.equal(row.notes,'[]');assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_artifact_refs').get().n,3)}finally{s.close()}}
});

test('a shared plan read rechecks source and consent after retrieving the document',async t=>{
 for(const mode of ['edit','delete','revoke','reduce']){const s=careSetup();try{const document=seedPlan(s),link=await s.share(['plan','people']);s.setHook(async({sql,phase})=>{if(phase==='after'&&sql==='SELECT * FROM change_plans WHERE user_id=?'){s.setHook(null);if(mode==='edit'){document.goals[0].title='Corrigé pendant lecture';replacePlan(s,document)}else if(mode==='delete')s.raw.prepare("DELETE FROM change_plans WHERE user_id='a'").run();else await s.call(`/api/care/links/${link.id}`,mode==='revoke'?'DELETE':'PUT',{revision:link.revision,accepted:true,scopes:['people']})}});const result=await s.call(`/api/care/dossiers/${link.id}/plan`,'GET',undefined,'pro');assert.equal(result.httpStatus,404,mode);assert.equal(result.data,undefined)}finally{s.close()}}
});

test('plan revision guards final publication while retaining metadata of already transmitted inputs',async t=>{
 const s=setup(t),document=seedPlan(s),link=await s.share(['plan']);s.setHook(({sql,phase})=>{if(phase==='before'&&sql.includes('SET status=?,content=?,notes=?,coverage=?')){s.setHook(null);document.goals[0].title='Corrigé juste avant publication';replacePlan(s,document)}});
 const queued=await prepare(s,link),row=s.raw.prepare('SELECT * FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,'failed');assert.equal(row.error_code,'sources_changed');assert.equal(row.content,'');assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_artifact_refs').get().n,3);
});

test('losing source, consent, session or lease guards before transmission sends nothing and records no inputs',async()=>{
 for(const mode of ['source','consent','session','previous_session','lease']){const s=careSetup();try{
  const document=seedPlan(s),link=await s.share(['plan']);s.setHook(({sql,phase})=>{if(phase==='before'&&sql.startsWith("UPDATE care_briefs SET updated_at=? WHERE id=? AND status='running'")){s.setHook(null);if(mode==='source'){document.goals[0].title='Source corrigée avant envoi';replacePlan(s,document)}else if(mode==='consent')s.raw.prepare("UPDATE care_links SET status='revoked',revision=revision+1 WHERE id=?").run(link.id);else if(mode==='session')s.raw.prepare("UPDATE care_sessions SET revision=revision+1 WHERE id='session'").run();else if(mode==='previous_session')s.session('inserted-prior',s.now()+1800);else s.raw.prepare("UPDATE care_briefs SET lease_token='new-worker',lease_until=?").run(s.now()+300)}});
  const queued=await prepare(s,link),row=s.raw.prepare('SELECT content,notes FROM care_briefs WHERE id=?').get(queued.id);assert.equal(s.calls.length,0,mode);assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_artifact_refs').get().n,0,mode);assert.equal(row.content,'');assert.equal(row.notes,'[]');
 }finally{s.close()}}
});

test('a failed provider call keeps metadata of the plan inputs already transmitted without storing their text',async t=>{
 const s=setup(t);seedPlan(s);const link=await s.share(['plan']);s.setGenerator(async()=>{const refs=s.raw.prepare("SELECT * FROM change_artifact_refs WHERE artifact_kind='brief'").all();assert.equal(refs.length,3);assert.doesNotMatch(JSON.stringify(refs),/Marcher|Choisir dix|apprécié|PRIVATE_/);throw Error('Provider unavailable')});const queued=await prepare(s,link),row=s.raw.prepare('SELECT status,content,notes FROM care_briefs WHERE id=?').get(queued.id);assert.equal(row.status,'pending');assert.equal(row.content,'');assert.equal(row.notes,'[]');assert.equal(s.raw.prepare('SELECT count(*) AS n FROM change_artifact_refs').get().n,3);
});

test('a plan mutation racing the final point read returns no stale body',async t=>{
 const s=setup(t),document=seedPlan(s),link=await s.share(['plan']),queued=await prepare(s,link);
 s.setHook(({sql,phase})=>{if(phase==='before'&&sql.startsWith('SELECT b.* FROM care_briefs')){s.setHook(null);document.trials[0].result_text='Retour corrigé';replacePlan(s,document)}});
 const result=await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro');assert.equal(result.httpStatus,404);assert.equal(result.brief,undefined);
});

test('ready points no longer expose removed plan data after edit or physical removal',async t=>{
 for(const mode of ['edit','delete']){const s=careSetup();try{const document=seedPlan(s),link=await s.share(['plan']),queued=await prepare(s,link);if(mode==='delete')s.raw.prepare("DELETE FROM change_plans WHERE user_id='a'").run();else{document.trials[0].result_text='Retour corrigé';replacePlan(s,document)}const result=await s.call(`/api/care/briefs/${queued.id}`,'GET',undefined,'pro');assert.equal(result.brief.status,'failed');assert.equal(result.brief.content,'');assert.equal(result.brief.error_code,'sources_changed')}finally{s.close()}}
});

test('care exports omit stale plan-derived point text before any ordinary point read',async t=>{
 const s=setup(t),document=seedPlan(s),link=await s.share(['plan']);await prepare(s,link);const before=await s.care.exportData(s.env,'a');assert.equal(before.care_briefs[0].status,'ready');assert.ok(before.care_briefs[0].content);
 document.trials=[];replacePlan(s,document);const exported=await s.care.exportData(s.env,'a');assert.equal(exported.care_briefs[0].status,'failed');assert.equal(exported.care_briefs[0].content,'');assert.doesNotMatch(JSON.stringify(exported),/J’ai apprécié/);
});

test('the final export query detects a concurrent plan source mutation',async t=>{
 const s=setup(t),document=seedPlan(s),link=await s.share(['plan']);await prepare(s,link);s.setHook(({sql,phase})=>{if(phase==='before'&&sql.startsWith('SELECT b.content,b.id FROM care_briefs')){s.setHook(null);document.goals=[];document.steps=[];document.trials=[];replacePlan(s,document)}});
 const exported=await s.care.exportData(s.env,'a');assert.equal(exported.care_briefs[0].status,'failed');assert.equal(exported.care_briefs[0].content,'');
});

test('an export rejects a source mutation that invalidates content read on an earlier page',async t=>{
 const s=setup(t),document=seedPlan(s),link=await s.share(['plan']);await prepare(s,link);
 const insert=s.raw.prepare("INSERT INTO care_briefs(id,user_id,link_id,session_id,link_revision,session_revision,status,created_at,updated_at) VALUES(?,'a',?,'session',?,1,'revoked',?,?)");for(let i=0;i<200;i++)insert.run('old-'+i,link.id,1000+i,s.now(),s.now());
 s.setHook(({sql,args,phase})=>{if(phase==='before'&&sql.startsWith('SELECT rowid AS export_rowid,')&&sql.includes('FROM care_briefs WHERE')&&args[1]>0){s.setHook(null);document.goals=[];document.steps=[];document.trials=[];replacePlan(s,document)}});
 await assert.rejects(s.care.exportData(s.env,'a'),error=>error.status===409&&error.code==='export_sources_changed');
});

test('export pages remain bounded and cursor-based when an earlier access record is removed',async t=>{
 const s=setup(t),link=await s.share(['plan']),insert=s.raw.prepare("INSERT INTO care_access_log(id,user_id,actor_id,link_id,action,scope,created_at) VALUES(?,'a','pro',?,'read','plan',?)");for(let i=0;i<205;i++)insert.run('access-'+i,link.id,s.now());
 let pages=0,deleted=false;s.setHook(({sql,args,phase})=>{if(phase==='before'&&sql.includes('FROM care_access_log WHERE')&&sql.startsWith('SELECT rowid AS export_rowid,')){pages++;assert.match(sql,/ORDER BY rowid LIMIT 200$/);if(args[1]>0&&!deleted){deleted=true;s.raw.prepare("DELETE FROM care_access_log WHERE id='access-0'").run()}}});
 const exported=await s.care.exportData(s.env,'a');assert.equal(pages,2);assert.equal(exported.care_access_log.filter(row=>row.id.startsWith('access-')).length,205,'a deletion behind the cursor must not skip the next retained record');assert.equal(new Set(exported.care_access_log.map(row=>row.id)).size,exported.care_access_log.length);assert.equal(exported.care_export_coverage.complete,true);assert.equal(exported.care_export_coverage.page_size,200);assert.equal(exported.care_export_coverage.included.care_access_log,exported.care_access_log.length);
});

test('the five hundred session limit is enforced at creation and the entire retained list is exported',async t=>{
 const s=setup(t);for(let i=0;i<499;i++)s.session('session-'+i,s.now()+i);assert.equal((await s.call('/api/care/sessions','POST',{session_at:s.now()+1000})).httpStatus,201);assert.equal((await s.call('/api/care/sessions','POST',{session_at:s.now()+1001})).code,'session_limit');const exported=await s.care.exportData(s.env,'a');assert.equal(exported.care_sessions.length,500);assert.equal(exported.care_export_coverage.included.care_sessions,500);
});

test('AI hypotheses remain discussion items even when a provider labels them as facts',async t=>{
 const s=setup(t),link=await s.share(['reviews']);s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES('review','a','point','ready','Hypothèse IA à vérifier','review',?,?)").run(s.now(),s.now());
 s.setGenerator(async()=>({text:JSON.stringify({items:[{source_id:'review:review',quote:'Hypothèse IA à vérifier',theme:'Faits rapportés'}]})}));const queued=await prepare(s,link),row=s.raw.prepare('SELECT notes,content FROM care_briefs WHERE id=?').get(queued.id),notes=JSON.parse(row.notes);assert.equal(notes[0].theme,'À discuter en séance');assert.match(notes[0].provenance,/aucune hypothèse confirmée/);assert.ok(row.content.indexOf('Hypothèse IA à vérifier')>row.content.indexOf('À discuter en séance'));
});

test('missing plan migration fails closed without pretending to have an empty shared plan',async t=>{
 const s=setup(t),link=await s.share(['plan']);s.raw.exec('DROP TABLE change_plans');assert.equal((await s.call(`/api/care/dossiers/${link.id}/plan`,'GET',undefined,'pro')).code,'plan_unavailable');const queued=await prepare(s,link);assert.equal(s.raw.prepare('SELECT status FROM care_briefs WHERE id=?').get(queued.id).status,'failed');assert.equal(s.calls.length,0);
});

test('member UI renders the personal plan with no linked professional and never selects sharing by default',async()=>{
 class Node{constructor(tag,cls,text){this.tag=tag;this.className=cls;this.textContent=text||'';this.children=[];this.isConnected=true}append(...items){this.children.push(...items)}replaceChildren(...items){this.children=items}setAttribute(){}addEventListener(){}querySelectorAll(){return []}}
 const apiCalls=[],plans=[],context=vm.createContext({console,Promise,Date,Number,Object,Array,String,document:{createTextNode:text=>new Node('text','',text)},el:(...args)=>new Node(...args),actionButton:(label,fn)=>Object.assign(new Node('button','',label),{action:fn}),api:async path=>{apiCalls.push(path);if(path==='/api/care')return {care:{has_psychologist:null,frequency_days:null,last_session_at:null,next_session_at:null},links:[],sessions:[],briefs:[]};if(path==='/api/care/access-log?offset=0')return {entries:[],next_offset:null};throw Error('Unexpected API '+path)},renderChangePlan:async(host,current)=>{plans.push({host,current:current()})},setTimeout(){}});
 vm.runInContext(CARE_INTERFACE_SCRIPT,context);const host=new Node('main');await context.renderCareMember(host,()=>true);assert.equal(plans.length,1);assert.equal(plans[0].current,true);assert.ok(host.children.includes(plans[0].host));const scopesHost=new Node('div'),read=context.careScopes(scopesHost,[],()=>true);assert.deepEqual(Array.from(read()),[]);assert.equal(scopesHost.children[0].children.filter(node=>node.tag==='label').length,10);assert.equal(apiCalls.includes('/api/plan/sharing-preview'),false);
 const text=CARE_INTERFACE_SCRIPT;assert.match(text,/\/api\/plan\/sharing-preview/);assert.match(text,/déjà téléchargé, copié ou capturé/);
});
