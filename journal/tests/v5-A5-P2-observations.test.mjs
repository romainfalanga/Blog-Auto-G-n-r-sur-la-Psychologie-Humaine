import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {setup,provider,draft} from './v4-S-helper.mjs';
import {PEOPLE_OBSERVATIONS_UI} from '../src/people-observations-ui.mjs';

const quote='Mila et moi cuisinons ensemble le dimanche, et je me sens soutenue dans ces moments.';
const path='/api/people/mila/observations';
function addPerson(s,id='mila',user='a'){
 s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role) VALUES(?,?,?,'hash','Mila','friend')").run(id,user,id);
}
async function pending(s,key='observation_initial',text=quote){
 await s.send(text,key);await s.settle();
 const state=await s.call('/api/chat');assert.equal(state.last_request.status,'completed',JSON.stringify(state.last_error));
 const proposal=state.proposals.find(p=>p.kind==='person_observation'&&p.status==='pending');assert.ok(proposal);return proposal;
}
const accept=(s,p,payload,user='a')=>s.call('/api/chat/proposals/'+p.id+'/accept','POST',{revision:p.revision,...(payload?{payload}:{})},user);
function observationsProvider(t,extra){return provider(t,p=>p==='ambre_writing'?draft({people_observations:[{person_id:'mila',axis:'appuis',quote}],...extra}):undefined)}

test('P2 engine proposes an expanded exact experience, edits need individual consent and retain source in export',async t=>{
 const s=setup();addPerson(s);await s.ready();observationsProvider(t);
 const p=await pending(s);assert.equal(p.evidence_quote,quote);assert.equal(p.target_id,'mila');
 assert.equal(s.raw.prepare('SELECT count(*) n FROM people_observations').get().n,0);
 const listed=await s.call(path);assert.equal(listed.available,true);assert.equal(listed.proposals[0].kind,'person_observation');assert.deepEqual(listed.observations,[]);
 assert.equal((await accept(s,p,undefined,'b')).status,404);
 assert.equal((await s.call('/api/chat/proposals/'+p.id+'/reject','POST',{revision:p.revision},'b')).status,404);
 assert.equal((await accept(s,p,{axis:'appuis',quote,person_id:'foreign'})).code,'invalid_input');
 assert.equal((await accept(s,p,{axis:'diagnostic',quote})).status,400);
 assert.equal((await accept(s,p,{axis:'appuis',quote:'Court'})).status,400);
 const edited='Avec Mila, cuisiner le dimanche me permet de me sentir soutenue.';
 const saved=await accept(s,p,{axis:'emotions',quote:edited});assert.equal(saved.status,200,JSON.stringify(saved));
 const notes=(await s.call(path)).observations;assert.equal(notes.length,1);assert.equal(notes[0].quote,edited);assert.equal(notes[0].axis,'emotions');assert.equal(notes[0].origin,'accepted');
 assert.equal((await accept(s,p)).status,200,'replayed accept is idempotent');assert.equal((await s.call(path)).observations.length,1);
 const exported=await s.chat.exportData(s.env,'a'),proposal=exported.chat_proposals.find(v=>v.id===p.id);assert.equal(proposal.evidence_quote,quote);assert.equal(proposal.payload.quote,edited);assert.equal(proposal.kind,'person_observation');assert.equal(proposal.status,'accepted');
 assert.equal((await s.personal.exportData(s.env,'a')).people_observations[0].origin,'accepted');assert.deepEqual((await s.personal.exportData(s.env,'b')).people_observations,[]);
 assert.equal((await s.call(path,'GET',undefined,'b')).status,404);assert.equal((await s.call(path+'/'+notes[0].id,'DELETE',undefined,'b')).status,404);
 assert.equal((await s.call(path+'/'+notes[0].id,'DELETE')).status,200);assert.equal((await accept(s,p)).status,200);assert.deepEqual((await s.call(path)).observations,[]);
});

test('P2 refusal persists and cannot become an observation or be reproposed on the next matching turn',async t=>{
 const s=setup();addPerson(s);await s.ready();observationsProvider(t);const p=await pending(s);
 assert.equal((await s.call('/api/chat/proposals/'+p.id+'/reject','POST',{revision:p.revision})).status,200);
 assert.equal((await accept(s,{...p,revision:p.revision+1})).code,'proposal_closed');
 await s.send(quote,'observation_again');await s.settle();
 assert.deepEqual((await s.call(path)).observations,[]);assert.equal((await s.call(path)).proposals.length,1);assert.equal((await s.call(path)).proposals[0].status,'rejected');
});

test('P2 accept enforces proposal revision, target revision, consent and deletion with no partial insert',async t=>{
 for(const mode of ['proposal','person','consent','delete'])await t.test(mode,async sub=>{
  const s=setup();addPerson(s);await s.ready();observationsProvider(sub);const p=await pending(s);
  if(mode==='proposal')p.revision++;
  if(mode==='person')s.raw.prepare("UPDATE personal_people SET revision=revision+1 WHERE id='mila'").run();
  if(mode==='consent')await s.call('/api/chat/consent','PUT',{accepted:false});
  if(mode==='delete')assert.equal((await s.call('/api/people/mila','DELETE',{revision:1})).status,200);
  const result=await accept(s,p);assert.equal(result.status,409,JSON.stringify(result));assert.equal(result.code,'revision_conflict');
  assert.equal(s.raw.prepare('SELECT count(*) n FROM people_observations').get().n,0);assert.equal(s.raw.prepare('SELECT status FROM chat_proposals WHERE id=?').get(p.id).status,'pending');
  if(mode==='person'){const renewed=await pending(s,'renewed_after_person_edit');assert.notEqual(renewed.id,p.id);assert.equal((await accept(s,renewed)).status,200);}
 });
});

test('P2 two proposals from one turn are accepted separately and concurrent duplicate accepts cannot double-save',async t=>{
 const s=setup();addPerson(s);await s.ready();const second='Mila me prépare du thé quand nous terminons la cuisine.';
 provider(t,p=>p==='ambre_writing'?draft({people_observations:[{person_id:'mila',axis:'appuis',quote},{person_id:'mila',axis:'communication',quote:second}]}):undefined);
 await pending(s,'two_observations',quote+' '+second);const proposals=(await s.call(path)).proposals;assert.equal(proposals.length,2);
 const results=await Promise.all([accept(s,proposals[0]),accept(s,proposals[0])]);assert.ok(results.some(r=>r.status===200));assert.ok(results.every(r=>[200,409].includes(r.status)));assert.equal((await s.call(path)).observations.length,1);
 assert.equal((await accept(s,proposals[1])).status,200);assert.equal((await s.call(path)).observations.length,2);
 assert.equal((await s.call('/api/chat','DELETE',{})).status,200);assert.deepEqual((await s.call(path)).observations,[]);assert.deepEqual((await s.call(path)).proposals,[]);
});

test('P2 migration absence disables new observations while old notes stay readable, deletable and exportable',async t=>{
 const s=setup();s.raw.exec('ALTER TABLE people_observations DROP COLUMN origin');addPerson(s);await s.ready();observationsProvider(t);
 await s.send(quote,'before_migration');await s.settle();assert.equal(s.turn().status,'completed');
 const listed=await s.call(path);assert.equal(listed.available,false);assert.equal(listed.enabled,false);assert.deepEqual(listed.proposals,[]);assert.deepEqual(listed.observations,[]);
 s.raw.prepare("INSERT INTO people_observations(id,user_id,person_id,turn_id,axis,quote,fingerprint,created_at) VALUES('old','a','mila',?,'appuis',?,'old',1)").run(s.turn().id,quote);
 assert.equal((await s.call(path)).observations[0].origin,'auto');assert.equal((await s.personal.exportData(s.env,'a')).people_observations[0].origin,'auto');
 assert.equal((await s.call(path+'/old','DELETE')).status,200);
 // A running Worker sees the additive migration without a restart.
 s.raw.exec("ALTER TABLE people_observations ADD COLUMN origin TEXT NOT NULL DEFAULT 'auto' CHECK(origin IN ('auto','accepted'))");
 const p=await pending(s,'after_migration');assert.equal((await accept(s,p)).status,200);
});

test('P2 old automatic observations are not recalled as individually accepted evidence',async t=>{
 const s=setup();addPerson(s);await s.ready();const calls=observationsProvider(t);const p=await pending(s);
 s.raw.prepare("INSERT INTO people_observations(id,user_id,person_id,turn_id,axis,quote,fingerprint,created_at) VALUES('old','a','mila',?,'appuis',?,'old',1)").run(s.turn().id,quote);
 await s.send('Je repense à Mila.','recall_auto');await s.settle();
 const input=JSON.parse(calls.filter(c=>c.purpose==='ambre_writing').at(-1).input);assert.equal(input.reported_relationship_observations,null);
 assert.equal((await s.call(path)).observations[0].origin,'auto');assert.equal(p.status,'pending');
});

// Minimal DOM only for the observation component's explicit consent controls.
class Element{
 constructor(tag){this.tagName=tag;this.children=[];this.listeners={};this.isConnected=true;this.value='';this.text='';this.disabled=false}
 append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=nodes}setAttribute(){}focus(){}
 set textContent(value){this.text=String(value);this.children=[]}get textContent(){return this.text+this.children.map(n=>n.textContent||'').join('')}
 addEventListener(event,fn){this.listeners[event]=fn}async fire(event){return this.listeners[event]?.({preventDefault(){}})}
}
const descendants=node=>[node,...node.children.flatMap(descendants)];
function observationUi(api){const ctx=vm.createContext({api,el:(tag,cls,text)=>{const node=new Element(tag);node.className=cls||'';node.text=text||'';return node},actionButton:(text,fn)=>{const node=new Element('button');node.text=text;node.addEventListener('click',fn);return node}});vm.runInContext(PEOPLE_OBSERVATIONS_UI,ctx);const root=ctx.personObservationDetails({id:'mila'});root.open=true;return {root,button:text=>descendants(root).find(n=>n.tagName==='button'&&n.text===text)}}

test('P2 close-person UI requires a per-proposal click, sends edits and preserves them after a conflict',async()=>{
 const calls=[],p={id:'proposal',kind:'person_observation',status:'pending',revision:3,payload:{axis:'appuis',quote},evidence_quote:quote};let conflict=true,accepted=false;
 const ui=observationUi(async(url,method,data)=>{calls.push({url,method,data});if(method==='POST'){if(conflict)throw new Error('Les données ont changé.');accepted=true;return {ok:true}}return {available:true,proposals:accepted?[]:[p],observations:accepted?[{id:'saved',axis:'appuis',quote:data?.payload?.quote||quote,origin:'accepted'}]:[]}});
 await ui.root.fire('toggle');assert.equal(calls.length,1);assert.ok(ui.root.textContent.includes('Rien n’est enregistré'));
 const textarea=descendants(ui.root).find(n=>n.tagName==='textarea');textarea.value='Mila et moi avons cuisiné ensemble un dimanche précis.';
 await ui.button('Valider ce repère').fire('click');assert.equal(calls[1].data.revision,3);assert.equal(calls[1].data.payload.quote,textarea.value);assert.ok(ui.root.textContent.includes('Les données ont changé.'));assert.equal(textarea.value,'Mila et moi avons cuisiné ensemble un dimanche précis.');assert.equal(ui.button('Valider ce repère').disabled,false);
 conflict=false;await ui.button('Valider ce repère').fire('click');assert.ok(ui.root.textContent.includes('Validé par vous'));assert.equal(ui.button('Valider ce repère'),undefined);
});

test('P2 a declared unsafe relationship blocks confrontation in a review despite a permissive model check',async t=>{
 const s=setup();addPerson(s);seedCompanionPrerequisites(s.raw);s.raw.prepare("UPDATE personal_people SET feels_safe='not_always' WHERE id='mila'").run();
 const calls=provider(t,p=>{
  if(p==='ambre_review')return {content:'Confrontez Mila ce soir. '+('Ce bilan décrit des informations limitées sur votre relation. Vous pouvez les préciser selon ce qui vous convient. '.repeat(12)),digest:{}};
  if(p==='ambre_review_check')return {criteria:{no_diagnosis:'yes',quotes_faithful:'yes',safety_resources:'na',hypotheses_open:'yes',no_saved_claim:'yes',sources_honest:'yes',no_internal:'yes'},reasons:[]};
 });
 const result=await s.call('/api/chat/start','POST',{idempotency_key:'review_unsafe_relation',accepted:true});assert.equal(result.status,202,JSON.stringify(result));
 const row=s.raw.prepare('SELECT status,error_code FROM ambre_reviews WHERE id=?').get(result.review_id);assert.equal(row.status,'failed');assert.equal(row.error_code,'review_not_approved');
 assert.match(calls.find(c=>c.purpose==='ambre_review').body.messages[0].content,/jamais de conseil de confronter/);
 const checks=calls.filter(c=>c.purpose==='ambre_review_check');assert.equal(checks.length,2);assert.ok(checks.every(c=>JSON.parse(c.input).deterministic_failures.includes('unsafe_confrontation')));
});
