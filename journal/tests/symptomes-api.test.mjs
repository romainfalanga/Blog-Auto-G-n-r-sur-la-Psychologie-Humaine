// « Mes symptômes » lot A : migration 0028 et API (consentement, âge, passations, brouillons,
// alertes de sécurité, délais, croisement, export, suppression), avant et après la migration.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {setup,database} from './v4-R-helper.mjs';
import {SYMPTOMES as S} from '../src/symptomes-data.generated.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {createSymptomsBackend,symptomsReady,latestSafetyReport,childhoodSide} from '../src/symptomes-backend.mjs';
import {crossWithChildhood} from '../src/symptomes-engine.mjs';

const SQL=readFileSync(new URL('../migrations/0028_symptom_checks.sql',import.meta.url),'utf8');
const DAY=86400,TODAY=Date.UTC(2026,10,19,12)/1000;
const V={phq9:S.instruments.phq9.version,gad7:S.instruments.gad7.version,itq:S.instruments.itq.version,phq15:S.instruments.phq15.version};
const phq9=(total,extra={})=>{const out={};let left=total;for(const i of [1,2,3,4,5,6,7,8,9]){const v=Math.min(3,left);out['phq9_'+i]=v;left-=v}return {...out,...extra}};
let keys=0;const key=()=>'passation-'+String(++keys).padStart(6,'0');
const humeur=(answers,extra={})=>({module:'humeur',content_version:S.version,instrument_versions:{phq9:V.phq9},answers,passation_key:key(),...extra});

function harness({until='9999',users=['a','b']}={}){
 const s=setup({until,users});let clock=TODAY,serial=0;const limits=[];
 const utils={error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),
  body:async(req,max)=>{const raw=await req.text();if(raw.length>max)throw Object.assign(new Error('trop gros'),{status:413});return JSON.parse(raw)},now:()=>clock,random:()=>'sym_'+String(++serial).padStart(6,'0'),limit:async(env,k,max,seconds)=>{limits.push([k,max,seconds])}};
 const backend=createSymptomsBackend(utils);
 async function call(path,method='GET',data,user='a'){
  const req=new Request('https://example.test'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});
  try{const url=new URL(req.url),res=await backend.routes(req,s.env,{id:user},url.pathname,url);if(!res)return {status:'unrouted'};return {status:res.status,...await res.json()}}
  catch(e){if(!e.status)throw e;return {status:e.status,code:e.code,message:e.message}}
 }
 const birth=(user,date)=>s.raw.prepare('UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id=?').run(JSON.stringify({birth_date:date}),user);
 const count=(table,user='a')=>s.raw.prepare(`SELECT count(*) n FROM ${table} WHERE user_id=?`).get(user).n;
 async function adult(user='a'){birth(user,'1990-05-04');const r=await call('/api/symptomes/consent','PUT',{consent:true},user);assert.equal(r.status,200,JSON.stringify(r));return r}
 return {...s,backend,call,birth,count,adult,limits,tick:n=>{clock+=n},now:()=>clock};
}

test('0028: additive tables, CHECK constraints, unique passation and alert, cascade with the account',()=>{
 const body=SQL.replace(/--[^\n]*/g,'').replace(/ON DELETE (CASCADE|SET NULL)/g,'');
 assert.doesNotMatch(body,/\b(DROP|DELETE|UPDATE|RENAME|ALTER|INSERT)\b/i);
 for(const statement of body.split(';').map(x=>x.trim()).filter(Boolean))assert.match(statement,/^CREATE (TABLE|INDEX) IF NOT EXISTS (symptom_\w+)/,statement);
 const db=database('0028');db.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','u@example.test','h','s'),('v','v@example.test','h','s')");
 const before=db.prepare("SELECT count(*) n FROM sqlite_master WHERE name LIKE 'symptom%'").get().n;assert.equal(before,0);
 db.exec(SQL);db.exec(SQL);// idempotent (IF NOT EXISTS)
 const ins=(over={})=>{const r={id:'c'+Math.random(),user_id:'u',passation_key:'key-0000001',module:'humeur',content_version:'2026-11-19.1',instrument_versions:'{}',answers:'{"phq9_1":1}',scores:'{}',result:'{}',state:'en_dessous',declined:0,completed_at:1,...over};return db.prepare(`INSERT INTO symptom_checks(${Object.keys(r)}) VALUES(${Object.keys(r).map(()=>'?')})`).run(...Object.values(r))};
 ins();
 assert.throws(()=>ins(),/UNIQUE/,'one record per passation');
 ins({passation_key:'key-0000002',state:'refuse',declined:1,answers:'{}'});
 for(const [over,why] of [[{state:'au-dessus'},'state'],[{state:'non_explore'},'non_explore is never stored'],[{declined:2},'declined'],[{declined:1,state:'refuse'},'declined with answers'],[{declined:1,answers:'{}'},'declined state'],[{answers:'[]'},'answers object'],[{answers:'nope'},'json'],[{passation_key:'short'},'key length'],[{module:''},'module'],[{result:'[1]'},'result object']])assert.throws(()=>ins({passation_key:'key-'+Math.random(),...over}),/CHECK|constraint/,why);
 assert.throws(()=>db.prepare("INSERT INTO symptom_checks(id,user_id,passation_key,module,content_version,result,state,completed_at) VALUES('x','ghost','key-0000009','humeur','v','{}','indetermine',1)").run(),/FOREIGN KEY/);
 db.prepare("INSERT INTO symptom_consents(user_id,consent,content_version,granted_at,updated_at) VALUES('u',1,'v',1,1)").run();
 assert.throws(()=>db.prepare("INSERT INTO symptom_consents(user_id,consent,content_version,updated_at) VALUES('v',1,'v',1)").run(),/CHECK/,'a given consent is dated');
 assert.throws(()=>db.prepare("INSERT INTO symptom_consents(user_id,consent,content_version,granted_at,updated_at) VALUES('v',2,'v',1,1)").run(),/CHECK/);
 const alert=(over={})=>{const r={id:'a'+Math.random(),user_id:'u',passation_key:'key-0000001',module:'humeur',trigger_id:'phq9_item9',kind:'suicide',screen:'bloquant',created_at:1,...over};return db.prepare(`INSERT INTO symptom_safety_alerts(${Object.keys(r)}) VALUES(${Object.keys(r).map(()=>'?')})`).run(...Object.values(r))};
 alert();assert.throws(()=>alert(),/UNIQUE/);
 assert.throws(()=>alert({trigger_id:'x',kind:'risque'}),/CHECK/);assert.throws(()=>alert({trigger_id:'y',screen:'score'}),/CHECK/);
 db.prepare("INSERT INTO symptom_drafts(user_id,module,passation_key,content_version,answers,updated_at) VALUES('u','humeur','key-0000003','v','{}',1)").run();
 assert.throws(()=>db.prepare("INSERT INTO symptom_drafts(user_id,module,passation_key,content_version,answers,updated_at) VALUES('u','humeur','key-0000004','v','{}',1)").run(),/UNIQUE|PRIMARY/,'one draft per module');
 // A deleted passation keeps its dated alert (check_id set to NULL).
 const id=db.prepare("SELECT id FROM symptom_checks WHERE passation_key='key-0000001'").get().id;db.prepare("UPDATE symptom_safety_alerts SET check_id=?").run(id);
 db.prepare('DELETE FROM symptom_checks WHERE id=?').run(id);assert.equal(db.prepare('SELECT check_id FROM symptom_safety_alerts').get().check_id,null);
 const plan=db.prepare("EXPLAIN QUERY PLAN SELECT * FROM symptom_checks WHERE user_id=? AND module=? ORDER BY completed_at DESC").all('u','humeur').map(r=>r.detail).join(' ');assert.match(plan,/symptom_checks_user_module/);
 db.prepare("DELETE FROM users WHERE id='u'").run();
 for(const table of ['symptom_consents','symptom_checks','symptom_drafts','symptom_safety_alerts'])assert.equal(db.prepare(`SELECT count(*) n FROM ${table}`).get().n,0,table);
 db.close();
});

test('before 0028: « bientôt disponible », no error 500, nothing written; after: available at once on a new binding',async()=>{
 const h=harness({until:'0028'});h.birth('a','1990-01-01');
 assert.equal(await symptomsReady(h.env),false);
 const page=await h.call('/api/symptomes');assert.equal(page.status,200);assert.equal(page.available,false);assert.match(page.message,/bientôt disponible/);
 for(const [path,method,data] of [['/api/symptomes/consent','PUT',{consent:true}],['/api/symptomes/consent','PUT',{consent:false}],['/api/symptomes/checks','POST',humeur(phq9(3))],['/api/symptomes/checks','GET'],['/api/symptomes/checks','DELETE'],['/api/symptomes/drafts/humeur','PUT',{passation_key:key(),content_version:S.version,answers:{}}]]){
  const r=await h.call(path,method,data);assert.equal(r.status,503,path+' '+method);assert.equal(r.code,'feature_unavailable');
 }
 assert.deepEqual(await h.backend.exportData(h.env,'a'),{symptomes:null});assert.equal(await latestSafetyReport(h.env,'a'),null);
 h.raw.exec(SQL);
 const env={...h.env,DB:{...h.env.DB}};assert.equal(await symptomsReady(env),true);
});

test('age: birth date of Mon compte required; a minor is refused everywhere and nothing is recorded',async()=>{
 const h=harness();
 let r=await h.call('/api/symptomes');assert.equal(r.available,true);assert.equal(r.age.status,'inconnu');
 r=await h.call('/api/symptomes/consent','PUT',{consent:true});assert.equal(r.status,403);assert.equal(r.code,'birth_date_required');
 h.birth('a','2008-11-20');// 18 tomorrow
 r=await h.call('/api/symptomes');assert.equal(r.age.status,'mineur');assert.equal(r.age.save,false);assert.ok(r.age.resources.some(x=>x.id==='119'));assert.ok(r.age.resources.some(x=>x.id==='fil_sante_jeunes'));
 r=await h.call('/api/symptomes/consent','PUT',{consent:true});assert.equal(r.status,403);assert.equal(r.code,'minor');
 // Even with an older consent (birth date corrected since), a minor's answers are never stored.
 h.raw.prepare("INSERT INTO symptom_consents(user_id,consent,content_version,granted_at,updated_at) VALUES('a',1,?,1,1)").run(S.version);
 r=await h.call('/api/symptomes/checks','POST',humeur(phq9(3,{phq9_9:2})));assert.equal(r.status,403);assert.equal(r.code,'minor');
 r=await h.call('/api/symptomes/drafts/humeur','PUT',{passation_key:key(),content_version:S.version,answers:{phq9_9:2}});assert.equal(r.code,'minor');
 assert.equal(h.count('symptom_checks'),0);assert.equal(h.count('symptom_drafts'),0);assert.equal(h.count('symptom_safety_alerts'),0);
 r=await h.call('/api/symptomes/consent','PUT',{consent:false});assert.equal(r.status,200,'withdrawal is always possible');assert.equal(r.consent.given,false);
 h.tick(DAY);r=await h.call('/api/symptomes');assert.equal(r.age.status,'adulte');assert.equal(r.age.age,18);assert.equal(r.age.jeune_adulte,true);
 h.birth('a','2030-01-01');assert.equal((await h.call('/api/symptomes')).age.status,'inconnu','a future date is not a birth date');
});

test('consent: distinct, required before any answer, withdrawable; invalid input refused',async()=>{
 const h=harness();h.birth('a','1990-05-04');
 let r=await h.call('/api/symptomes/checks','POST',humeur(phq9(3)));assert.equal(r.status,403);assert.equal(r.code,'consent_required');
 for(const bad of [{},{consent:'true'},{consent:1}])assert.equal((await h.call('/api/symptomes/consent','PUT',bad)).code,'invalid_consent');
 r=await h.call('/api/symptomes/consent','PUT',{consent:true});assert.deepEqual(r.consent,{given:true,content_version:S.version,granted_at:TODAY,withdrawn_at:null});
 h.tick(10);r=await h.call('/api/symptomes/consent','PUT',{consent:true});assert.equal(r.consent.granted_at,TODAY,'granting again keeps the first date');
 assert.equal((await h.call('/api/symptomes/checks','POST',humeur(phq9(3)))).status,201);
 h.tick(10);r=await h.call('/api/symptomes/consent','PUT',{consent:false});assert.equal(r.consent.given,false);assert.equal(r.consent.withdrawn_at,TODAY+20);
 assert.equal((await h.call('/api/symptomes/checks','POST',humeur(phq9(3)))).code,'consent_required');
 assert.equal(h.count('symptom_checks'),1,'withdrawal keeps the results until the person deletes them');
 assert.equal((await h.call('/api/symptomes')).consent.given,false);
 assert.equal((await h.call('/api/symptomes/consent','GET')).status,405);
 assert.equal(h.raw.prepare("SELECT consent_ai FROM profiles WHERE user_id='a'").get().consent_ai,0,'not the AI consent');
});

test('passation: scores recomputed by the server, client scores ignored, idempotent retry, conflict, history',async()=>{
 const h=harness();await h.adult();
 const data=humeur(phq9(12,{humeur_duree:'non',humeur_gene:'nettement'}),{scores:{phq9:0},state:'repere_absent',result:{state:'repere_absent'}});
 let r=await h.call('/api/symptomes/checks','POST',data);assert.equal(r.status,201,JSON.stringify(r));
 assert.equal(r.check.state,'au_dessus');assert.equal(r.check.scores.phq9,12);assert.equal(r.check.scores.phq2,6);assert.equal(r.check.result.bands.phq9.id,'modere');
 assert.equal(r.check.completed_at,TODAY,'server clock');assert.equal(r.check.current_version,true);assert.equal(r.overlap,false);
 assert.equal(r.page.modules.find(m=>m.id==='humeur').state,'au_dessus');
 const stored=h.raw.prepare('SELECT * FROM symptom_checks').get();assert.equal(JSON.parse(stored.scores).phq9,12);assert.ok(!stored.answers.includes('scores'));
 assert.deepEqual(h.limits.at(-1),['symptoms-check:a',60,3600]);
 const again=await h.call('/api/symptomes/checks','POST',data);assert.equal(again.status,200);assert.equal(again.replayed,true);assert.equal(again.check.id,r.check.id);
 const conflict=await h.call('/api/symptomes/checks','POST',{...data,answers:phq9(3)});assert.equal(conflict.status,409);assert.equal(conflict.code,'passation_conflict');
 assert.equal(h.count('symptom_checks'),1);
 // A passation is never overwritten: a new one is added.
 h.tick(DAY);const second=await h.call('/api/symptomes/checks','POST',humeur(phq9(4)));assert.equal(second.check.state,'en_dessous');assert.equal(second.overlap,true,'before 14 days: overlap note');
 const history=await h.call('/api/symptomes/checks?module=humeur');assert.deepEqual(history.checks.map(c=>c.state),['en_dessous','au_dessus']);
 assert.equal((await h.call('/api/symptomes/checks?module=inconnu')).code,'unknown_module');
 const declined=await h.call('/api/symptomes/checks','POST',{module:'regulation',content_version:S.version,declined:true,passation_key:key()});assert.equal(declined.check.state,'refuse');assert.deepEqual(declined.check.answers,{});
 assert.equal((await h.call('/api/symptomes/checks','PATCH',{})).status,405);
 assert.equal((await h.call('/api/symptomes/inconnu')).status,404);assert.equal((await h.call('/api/symptomesx')).status,'unrouted','other routes are left to the Worker');
});

test('passation: strict validation through the API (module, question, value, version, key, dates)',async()=>{
 const h=harness();await h.adult();
 const code=async data=>(await h.call('/api/symptomes/checks','POST',data)).code;
 assert.equal(await code(humeur(phq9(3),{module:'inconnu'})),'unknown_module');
 assert.equal(await code(humeur({...phq9(3),phq9_10:1})),'unknown_question');
 assert.equal(await code(humeur({...phq9(3),gad7_1:1})),'unknown_question');
 assert.equal(await code(humeur({phq9_1:7})),'invalid_answer');
 assert.equal(await code(humeur({humeur_gene:'beaucoup'})),'invalid_answer');
 assert.equal(await code(humeur(phq9(3),{content_version:'2025-01-01.1'})),'unknown_version');
 assert.equal(await code(humeur(phq9(3),{instrument_versions:{phq9:'ancienne'}})),'unknown_version');
 assert.equal(await code(humeur(phq9(3),{instrument_versions:{}})),'unknown_version');
 assert.equal(await code(humeur(phq9(3),{passation_key:'court'})),'passation_key_required');
 assert.equal(await code(humeur(phq9(3),{passation_key:undefined})),'passation_key_required');
 assert.equal(await code(humeur(phq9(3),{started_at:TODAY+3600})),'invalid_symptoms');
 assert.equal(await code(humeur(phq9(3),{started_at:'hier'})),'invalid_symptoms');
 assert.equal(await code(humeur({})),'empty_passation');
 assert.equal(await code(humeur(phq9(3),{declined:true})),'invalid_symptoms');
 assert.equal(await code({...humeur({}),answers:'phq9_1=1'}),'invalid_symptoms');
 assert.equal(h.count('symptom_checks'),0);
 assert.equal((await h.call('/api/symptomes/checks','POST',humeur(phq9(3),{started_at:TODAY-600}))).check.started_at,TODAY-600);
});

test('other user: no read, no deletion, no history of another account',async()=>{
 const h=harness();await h.adult('a');await h.adult('b');
 const mine=await h.call('/api/symptomes/checks','POST',humeur(phq9(15)),'a');
 const other=await h.call('/api/symptomes',undefined,undefined,'b');assert.equal(other.modules.find(m=>m.id==='humeur').state,'non_explore');assert.equal(other.modules.find(m=>m.id==='humeur').last,null);
 assert.deepEqual((await h.call('/api/symptomes/checks',undefined,undefined,'b')).checks,[]);
 const r=await h.call('/api/symptomes/checks/'+mine.check.id,'DELETE',undefined,'b');assert.equal(r.status,404);
 assert.equal((await h.call('/api/symptomes/checks','DELETE',undefined,'b')).deleted.checks,0);
 assert.equal(h.count('symptom_checks','a'),1);
 // The same passation key in another account is another passation.
 const b=await h.call('/api/symptomes/checks','POST',humeur(phq9(2),{passation_key:mine.check.passation_key}),'b');assert.equal(b.status,201);
 assert.deepEqual((await h.backend.exportData(h.env,'b')).symptomes.checks.map(c=>c.id),[b.check.id]);
});

test('safety: alerts dated as soon as the draft is saved, kept if abandoned, no duplicate, 14-day banner, no score',async()=>{
 const h=harness();await h.adult();
 const passation=key();
 let r=await h.call('/api/symptomes/drafts/humeur','PUT',{passation_key:passation,content_version:S.version,instrument_versions:{phq9:V.phq9},answers:{securite_actuelle:'non',phq9_1:0,phq9_9:1},started_at:TODAY-60});
 assert.equal(r.status,200,JSON.stringify(r));assert.deepEqual(r.safety.triggers.map(t=>t.id),['phq9_item9']);assert.equal(r.safety.screen.blocking,true);assert.ok(!JSON.stringify(r).match(/risque|risk_level/));
 assert.deepEqual(r.draft.answers,{phq9_1:0,phq9_9:1,securite_actuelle:'non'});
 assert.deepEqual(h.raw.prepare('SELECT trigger_id,kind,screen,check_id,created_at FROM symptom_safety_alerts').all().map(x=>({...x})),[{trigger_id:'phq9_item9',kind:'suicide',screen:'bloquant',check_id:null,created_at:TODAY}]);
 r=await h.call('/api/symptomes/drafts/humeur','PUT',{passation_key:passation,content_version:S.version,instrument_versions:{phq9:V.phq9},answers:{securite_actuelle:'non',phq9_1:0,phq9_9:2}});assert.equal(h.count('symptom_safety_alerts'),1,'one alert per trigger and passation');
 let page=await h.call('/api/symptomes');assert.equal(page.safety.last_reported_at,TODAY);assert.deepEqual(page.safety.banner.kinds,['suicide']);assert.equal(page.safety.banner.until,TODAY+14*DAY);assert.ok(page.safety.banner.resources.some(x=>x.id==='3114'));
 assert.equal(page.modules.find(m=>m.id==='humeur').state,'non_explore','an abandoned passation is not a result');assert.equal(page.modules.find(m=>m.id==='humeur').draft.passation_key,passation);
 assert.deepEqual(await latestSafetyReport(h.env,'a'),{reported_at:TODAY,kind:'suicide'});
 // Finishing the same passation links the alert and removes the draft.
 h.tick(120);r=await h.call('/api/symptomes/checks','POST',humeur(phq9(4,{phq9_9:2,securite_actuelle:'non'}),{passation_key:passation}));
 assert.equal(r.status,201);assert.deepEqual(r.safety.triggers.map(t=>t.id),['phq9_item9']);assert.equal(h.count('symptom_safety_alerts'),1);
 assert.equal(h.raw.prepare('SELECT check_id FROM symptom_safety_alerts').get().check_id,r.check.id);assert.equal(h.count('symptom_drafts'),0);
 // Soft block: dated too, without banner.
 await h.call('/api/symptomes/drafts/anxiete','PUT',{passation_key:key(),content_version:S.version,answers:{securite_actuelle:'pref'}});
 assert.equal(h.raw.prepare("SELECT screen FROM symptom_safety_alerts WHERE trigger_id='securite_pref'").get().screen,'doux');
 // Violence in a passation abandoned before the end; then the banner ends after 14 days.
 h.tick(DAY);await h.call('/api/symptomes/drafts/relations','PUT',{passation_key:key(),content_version:S.version,answers:{rel_violences:'oui'}});
 page=await h.call('/api/symptomes');assert.deepEqual(page.safety.banner.kinds,['suicide','violences']);assert.equal(page.safety.banner.quick_exit,'Quitter vite cette page');
 h.tick(15*DAY);page=await h.call('/api/symptomes');assert.equal(page.safety.banner,null);assert.equal(page.safety.last_reported_at,TODAY+120+DAY);
 assert.deepEqual(h.limits.find(l=>l[0]==='symptoms-draft:a'),['symptoms-draft:a',600,3600]);
 // Drafts follow the same validation; a module mismatch is refused; a draft can be dropped.
 assert.equal((await h.call('/api/symptomes/drafts/humeur','PUT',{passation_key:key(),content_version:S.version,answers:{rel_violences:'oui'}})).code,'unknown_question');
 assert.equal((await h.call('/api/symptomes/drafts/humeur','PUT',{module:'anxiete',passation_key:key(),content_version:S.version,answers:{}})).code,'invalid_symptoms');
 assert.equal((await h.call('/api/symptomes/drafts/inconnu','PUT',{passation_key:key(),content_version:S.version,answers:{}})).code,'unknown_module');
 assert.equal((await h.call('/api/symptomes/drafts/relations','DELETE')).ok,true);assert.equal(h.count('symptom_drafts'),1);
 assert.equal((await h.call('/api/symptomes/drafts/relations','POST',{})).status,405);
});

test('delays, lifetime answers and crossing in the page state',async()=>{
 const h=harness();await h.adult();
 await h.call('/api/symptomes/checks','POST',humeur(phq9(12,{humeur_duree:'non'})));
 const itq=Object.fromEntries(S.instruments.itq.items.map(i=>[i.id,0]));
 await h.call('/api/symptomes/checks','POST',{module:'trauma_stress',content_version:S.version,instrument_versions:{itq:V.itq},answers:{...itq,exposition_evenement:'oui',deuil_deces:'non',adaptation_evenement:'non',itq_quand:'1_5_ans'},passation_key:key()});
 await h.call('/api/symptomes/checks','POST',{module:'bipolaire',content_version:S.version,answers:{bipolaire_periode:'non',bipolaire_signes:'non'},passation_key:key()});
 await h.call('/api/symptomes/checks','POST',{module:'relations',content_version:S.version,declined:true,passation_key:key()});
 let page=await h.call('/api/symptomes');const mod=id=>page.modules.find(m=>m.id===id);
 assert.deepEqual(mod('humeur').retest,{days:14,never:false,next_at:TODAY+14*DAY,due:false});
 assert.deepEqual(mod('trauma_stress').retest,{days:30,never:false,next_at:TODAY+30*DAY,due:false});
 assert.equal(mod('bipolaire').retest.never,true);assert.equal(mod('relations').state,'refuse');assert.equal(mod('relations').retest.due,true,'a refusal starts no delay');
 assert.deepEqual(page.lifetime_answers,{exposition_evenement:'oui',deuil_deces:'non',bipolaire_periode:'non',bipolaire_signes:'non'});
 assert.equal(page.crossing.childhood_available,false);assert.ok(page.crossing.lists.symptomes.some(s=>s.id==='dep_generale'));
 assert.equal(page.crossing.subjects.find(s=>s.id==='att_anxieux').state,'refuse');
 // With a saved childhood map: the server uses computeCartography and the « Pas moi » answers.
 const answers=Object.fromEntries(CARTOGRAPHIE.factors.filter(f=>!f.historical).map(f=>[f.id,1]));
 const side=childhoodSide({answers:JSON.stringify(answers),feedback:'{}',consent:1});
 assert.ok(side.present.length>0);
 const target=side.present.find(id=>id in S.subject_modules)||side.present[0];
 const feedback={[target]:'non'};
 h.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES('a',?,?,'{}','{}','[]',1,1,1,1)").run(JSON.stringify(answers),JSON.stringify(feedback));
 page=await h.call('/api/symptomes');assert.equal(page.crossing.childhood_available,true);
 const latest=Object.fromEntries(page.modules.filter(m=>m.last).map(m=>[m.id,m.last.result]));
 assert.deepEqual(page.crossing,crossWithChildhood(latest,childhoodSide({answers:JSON.stringify(answers),feedback:JSON.stringify(feedback),consent:1}),S,{labels:Object.fromEntries(CARTOGRAPHIE.outcomes.map(o=>[o.id,o.label]))}),'same result as the shared engine');
 for(const list of ['deux','enfance'])assert.ok(!page.crossing.lists[list].some(x=>x.id===target),'« Pas moi » never in lists 1 and 3');
 assert.ok(!JSON.stringify(page.crossing).includes('associated_count'));
});

test('deletion of results (all or one) keeps the consent; export carries every table',async()=>{
 const h=harness();await h.adult();
 const one=await h.call('/api/symptomes/checks','POST',humeur(phq9(3,{phq9_9:1})));
 await h.call('/api/symptomes/checks','POST',humeur(phq9(13)));
 await h.call('/api/symptomes/drafts/anxiete','PUT',{passation_key:key(),content_version:S.version,answers:{securite_actuelle:'oui'}});
 const exported=(await h.backend.exportData(h.env,'a')).symptomes;
 assert.equal(exported.content_version,S.version);assert.equal(exported.consent.given,true);assert.equal(exported.checks.length,2);
 assert.deepEqual(exported.checks[0].answers,phq9(3,{phq9_9:1}));assert.equal(exported.checks[0].result.state,'en_dessous');
 assert.equal(exported.drafts.length,1);assert.deepEqual(exported.safety_alerts.map(a=>a.trigger_id).sort(),['phq9_item9','securite_oui']);
 let r=await h.call('/api/symptomes/checks/'+one.check.id,'DELETE');assert.equal(r.status,200);assert.equal(h.count('symptom_checks'),1);
 assert.equal((await h.call('/api/symptomes/checks/'+one.check.id,'DELETE')).status,404);
 assert.equal(h.raw.prepare("SELECT check_id FROM symptom_safety_alerts WHERE trigger_id='phq9_item9'").get().check_id,null,'the dated alert outlives a deleted passation until all results are deleted');
 r=await h.call('/api/symptomes/checks','DELETE');assert.deepEqual(r.deleted,{checks:1,drafts:1,alerts:2});
 for(const table of ['symptom_checks','symptom_drafts','symptom_safety_alerts'])assert.equal(h.count(table),0,table);
 assert.equal(r.page.consent.given,true);assert.equal(r.page.safety.banner,null);
 assert.deepEqual((await h.backend.exportData(h.env,'b')).symptomes.checks,[]);
});

// Real Worker routes: authentication, same origin, no-store, export and account deletion.
const source=readFileSync(new URL('../src/worker.mjs',import.meta.url),'utf8').replace("import { HTML } from './ui.mjs';","const HTML='<title>T</title>';").replace("import { EXAMPLES } from './examples.mjs';",'const EXAMPLES=[];')
 .replace("import { createRecommender } from './recommendations.mjs';",'const createRecommender=()=>({recommend:()=>[],provisionalPortrait:()=>null});');
const rewritten=source.replace(/from '(\.\/[^']+)'/g,(_,specifier)=>'from '+JSON.stringify(new URL('../src/'+specifier.slice(2),import.meta.url).href));
const {default:worker}=await import('data:text/javascript;base64,'+Buffer.from(rewritten).toString('base64'));
const origin='https://journal.example';
function d1(raw){
 const prepare=sql=>{let args=[];return {bind(...a){args=a;return this},async first(){return raw.prepare(sql).get(...args)||null},async all(){return {results:raw.prepare(sql).all(...args)}},async run(){return {meta:{changes:raw.prepare(sql).run(...args).changes}}}}};
 return {raw,prepare,async batch(statements){raw.exec('BEGIN');try{const r=[];for(const s of statements)r.push(await s.run());raw.exec('COMMIT');return r}catch(e){raw.exec('ROLLBACK');throw e}}};
}
const request=(env,path,{method='GET',data,cookie,originHeader=origin}={})=>worker.fetch(new Request(origin+path,{method,headers:{...(data?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{}),...(originHeader?{Origin:originHeader}:{}),'CF-Connecting-IP':'192.0.2.9'},body:data?JSON.stringify(data):undefined}),env);
async function register(env,email){const r=await request(env,'/api/auth/register',{method:'POST',data:{email,password:'long secure password 123'}});assert.equal(r.status,200);return {cookie:r.headers.get('set-cookie').split(';')[0],user:(await r.json()).user};}

test('Worker: authentication, same origin, no-store, export and account deletion include « Mes symptômes »',async()=>{
 const env={DB:d1(database()),PUBLIC_ORIGIN:origin},a=await register(env,'sym-a@example.test'),b=await register(env,'sym-b@example.test');
 assert.equal((await request(env,'/api/symptomes')).status,401);
 env.DB.raw.prepare('UPDATE profiles SET self_data=json_patch(self_data,?) WHERE user_id=?').run(JSON.stringify({birth_date:'1985-03-02'}),a.user.id);
 assert.equal((await request(env,'/api/symptomes/consent',{method:'PUT',cookie:a.cookie,data:{consent:true},originHeader:'https://evil.example'})).status,403);
 assert.equal((await request(env,'/api/symptomes/consent',{method:'PUT',cookie:a.cookie,data:{consent:true},originHeader:null})).status,403);
 assert.equal((await request(env,'/api/symptomes/consent',{method:'PUT',cookie:a.cookie,data:{consent:true}})).status,200);
 const r=await request(env,'/api/symptomes/checks',{method:'POST',cookie:a.cookie,data:humeur(phq9(11,{securite_actuelle:'oui'}))});
 assert.equal(r.status,201);assert.equal(r.headers.get('cache-control'),'private, no-store');
 const page=await request(env,'/api/symptomes',{cookie:a.cookie});assert.equal(page.headers.get('cache-control'),'private, no-store');assert.equal((await page.json()).modules[0].state,'au_dessus');
 const big=await request(env,'/api/symptomes/checks',{method:'POST',cookie:a.cookie,data:{...humeur(phq9(1)),padding:'x'.repeat(25000)}});assert.equal(big.status,413);
 const exported=await (await request(env,'/api/export',{cookie:a.cookie})).json();
 assert.equal(exported.symptomes.checks.length,1);assert.equal(exported.symptomes.safety_alerts[0].trigger_id,'securite_oui');assert.equal(exported.symptomes.consent.given,true);
 assert.deepEqual((await (await request(env,'/api/export',{cookie:b.cookie})).json()).symptomes.checks,[]);
 assert.equal((await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}})).status,200);
 for(const table of ['symptom_consents','symptom_checks','symptom_drafts','symptom_safety_alerts'])assert.equal(env.DB.raw.prepare(`SELECT count(*) n FROM ${table} WHERE user_id=?`).get(a.user.id).n,0,table);
 env.DB.raw.close();
});

test('Worker before 0028: the page state says « bientôt disponible » and the export still works',async()=>{
 const env={DB:d1(database('0028')),PUBLIC_ORIGIN:origin},a=await register(env,'sym-old@example.test');
 const page=await request(env,'/api/symptomes',{cookie:a.cookie});assert.equal(page.status,200);assert.equal((await page.json()).available,false);
 assert.equal((await request(env,'/api/symptomes/checks',{method:'POST',cookie:a.cookie,data:humeur(phq9(1))})).status,503);
 const exported=await request(env,'/api/export',{cookie:a.cookie});assert.equal(exported.status,200);assert.equal((await exported.json()).symptomes,null);
 assert.equal((await request(env,'/api/account',{method:'DELETE',cookie:a.cookie,data:{password:'long secure password 123'}})).status,200);
 env.DB.raw.close();
});
