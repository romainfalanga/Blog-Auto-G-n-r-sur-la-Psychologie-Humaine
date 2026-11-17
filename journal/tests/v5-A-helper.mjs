// Lot A5-P0 harness: the filled profile of the v5 audit (docs/audit-sources-bilans-ambre-v5-2026-11-19.md),
// with start and end markers in every field, replayed on the real engine with a simulated provider.
import {setup,analysis,draft,review,response,CARTOGRAPHIE} from './v4-S-helper.mjs';
export {setup,analysis,draft,review,response};
const filler=(seed,n)=>{const s=`${seed} : je remarque que cela revient souvent dans ma vie, avec des nuances selon les moments et les personnes. `;return s.repeat(Math.ceil(n/s.length)).slice(0,n)};
export const MARK={self_description:['PRES_DEBUT','PRES_FIN'],self_perception:['PERC_DEBUT','PERC_FIN'],recurring_patterns:['SCHEM_DEBUT','SCHEM_FIN'],emotions:['EMO_DEBUT','EMO_FIN'],current_context:['CTX_DEBUT','CTX_FIN']};
export const PEOPLE=[['lea','Léa','sibling','','Ma sœur aînée. Elle me coupe souvent la parole au dîner.'],['marc','Marc','father','','Mon père, sévère et critique pendant mon enfance.'],['sophie','Sophie','mother','','Ma mère, aimante mais débordée.'],['julien','Julien','partner','','Mon compagnon depuis 2019, rassurant.'],['nadia','Nadia','best_friend','','Ma meilleure amie depuis le lycée.'],['paul','Paul','other','Collègue','Mon responsable, exigeant.']];
export const EVENTS=[
 ['ev1','Divorce de mes parents','difficult','age','','','Vers 9 ans',['marc','sophie'],5],
 ['ev2','Entrée au lycée','positive','year','2006','','',['nadia'],2],
 ['ev3','Départ de la maison','turning_point','year','2009','','',[],3],
 ['ev4','Rencontre avec Julien','positive','year','2019','','',['julien'],4],
 ['ev5','Déménagement à Lyon','ambivalent','month','2023-09','','',['julien'],4],
 ['ev6','Burn-out au travail','difficult','range','2024-01','2024-06','',['paul'],5],
 ['ev7','Mariage de Léa','ambivalent','day','2022-06-18','','',['lea'],2],
 ['ev8','Décès de ma grand-mère','difficult','year','2015','','',[],3],
 ['ev9','Premier emploi','positive','year','2012','','',[],1],
 ['ev10','Dispute avec mon père','difficult','year','2021','','',['marc'],4],
 ['ev11','Voyage seule','positive','year','2017','','',[],2],
 ['ev12','Reprise du travail','turning_point','month','2024-09','','',['paul'],4]
];
export function seedProfile(s,user='a'){
 const self={first_name:'Camille',gender:'female',birth_date:'1991-03-02',occupation:'Cheffe de projet'};
 const sizes={self_description:2400,self_perception:800,recurring_patterns:800,emotions:800,current_context:800};
 for(const [k,[a,b]] of Object.entries(MARK))self[k]=a+' '+filler(k,sizes[k]-a.length-b.length-2)+' '+b;
 s.raw.prepare('UPDATE profiles SET self_data=? WHERE user_id=?').run(JSON.stringify(self),user);
 for(const [id,name,role,custom,intro] of PEOPLE)s.raw.prepare("INSERT INTO personal_people(id,user_id,creation_id,payload_hash,display_name,role,custom_role,relationship_text,period_start,period_end) VALUES(?,?,?,'hash',?,?,?,?,?,?)").run(id,user,'c-'+id,name,role,custom,intro+' '+filler('relation '+name,1100)+' REL_FIN_'+id,'Depuis l’enfance','Aujourd’hui');
 for(const [id,title,kind,prec,start,end,age,people,imp] of EVENTS){
  s.raw.prepare("INSERT INTO life_events(id,user_id,creation_id,payload_hash,title,kind,description,meaning_now,date_precision,date_start,date_end,age_text,current_importance) VALUES(?,?,?,'hash',?,?,?,?,?,?,?,?,?)").run(id,user,'c-'+id,title,kind,'DESC_DEBUT_'+id+' '+filler('moment '+title,1000)+' DESC_FIN_'+id,'SENS_'+id+' '+filler('aujourd’hui',300),prec,start,end,age,imp);
  for(const p of people)s.raw.prepare('INSERT INTO life_event_people(event_id,person_id,user_id) VALUES(?,?,?)').run(id,p,user);
 }
 const answers=Object.fromEntries(CARTOGRAPHIE.factors.map((x,i)=>[x.id,i%4===0?0:i%5===0?2:1]));
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,feedback,notes,resources,ambre_notes,consent,locked_at,version,updated_at) VALUES(?,?,'{}','{}','{}','[]',1,1,1,1)").run(user,JSON.stringify(answers));
 const v=s.raw.prepare('SELECT version FROM profiles WHERE user_id=?').get(user).version;
 const portrait={schema_version:2,summary:'SYNTH_DEBUT {{prenom}} décrit une vigilance forte face à la critique. '+filler('synthèse',1800)+' SYNTH_FIN',summary_evidence:['profile:'+user],axes:[{label:'AXE_Vigilance à la critique',why:'AXEWHY '+filler('axe',600),confidence:0.6,claim_type:'hypothesis',evidence:['profile:'+user]},{label:'AXE_Besoin de limites',why:'AXEWHY2 '+filler('axe2',400),confidence:0.5,claim_type:'hypothesis',evidence:['person:lea']}],strengths:[{text:'FORCE_ lucidité et humour',evidence:['profile:'+user]}],uncertainties:['INCERT_ rôle du travail'],current_state:{summary:'ETAT_ACTUEL '+filler('état',500),evidence:['profile:'+user]},evolution:{summary:'EVOLUTION_ '+filler('évolution',500),evidence:['profile:'+user]},citations:[{source_id:'profile:'+user,revision:1,quote:'PRES_DEBUT'}],psychological_map:[],coverage:{processed:[],pending:[]}};
 s.raw.prepare('INSERT OR REPLACE INTO user_portraits(user_id,portrait,analyzed_version,schema_version,updated_at) VALUES(?,?,?,2,CURRENT_TIMESTAMP)').run(user,JSON.stringify(portrait),v);
}
export function seedReviews(s,user='a'){
 s.raw.prepare('DELETE FROM ambre_reviews WHERE user_id=?').run(user);
 const ins=s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES(?,?,?,'ready',?,?,?,?)");
 ins.run('rev-init',user,'initial','BILAN_INIT_DEBUT objectif initial : moins ruminer après les critiques. '+filler('analyse initiale',4200)+' BILAN_INIT_FIN','k-init',100,100);
 ins.run('rev-p1',user,'point','BILAN_P1_DEBUT objectif : dire non à Paul une fois par semaine. '+filler('point 1',4000)+' BILAN_P1_FIN','k-p1',200,200);
 ins.run('rev-p2',user,'point','BILAN_P2_DEBUT objectif confirmé : poser une limite avec Léa ; essai réalisé : quitter la table calmement. '+filler('point 2',4000)+' BILAN_P2_FIN','k-p2',300,300);
}
export function seedMemoryAndHistory(s,user='a'){
 const ins=s.raw.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response,created_at) VALUES(?,?,?,'hash','completed','consumed','2026-01-01',1,1,?,?,?)");
 for(let i=1;i<=8;i++)ins.run('h'+i,user,'h'+i,'HIST_'+i+' '+filler('échange '+i,500),JSON.stringify({companion_reply:1,content:'REP_'+i+' '+filler('réponse',700)}),'2026-11-1'+(i%9)+' 10:00:00');
 const memory={goals:['MEM_GOAL poser une limite avec Léa'],facts:['MEM_FACT je rumine après les critiques'],tried:['MEM_TRIED quitter la table calmement'],alliance:['MEM_ALLIANCE j’aime les réponses courtes'],hypotheses:[],session:[]};
 s.raw.prepare("INSERT OR REPLACE INTO chat_memory(user_id,summary,through_turn,turns_covered,updated_at) VALUES(?,?,'h8',8,0)").run(user,JSON.stringify(memory));
}
export const MESSAGES=[
 ['proche','Hier soir Léa m’a encore coupé la parole au dîner et je me suis sentie minuscule, comme avec mon père quand j’étais petite.'],
 ['suite_bilan','Je n’arrive pas à avancer sur ce qu’on s’était dit la dernière fois, je me sens bloquée.'],
 ['moment','Depuis le déménagement à Lyon je dors mal et je pense beaucoup au burn-out.'],
 ['portrait','Je me demande pourquoi je réagis si fort à la critique, comme vous l’aviez noté dans mon portrait.']
];
// Simulated analysis choices: "none" selects nothing; "ideal" selects, in priority order,
// the latest review, the portrait synthesis, the self description, then the person or moment cited.
export function pick(strategy,index){
 if(strategy!=='ideal')return [];
 const rows=index.split('\n'),pri=r=>/ review /.test(r)?0:/ portrait /.test(r)?1:/ profile /.test(r)?2:/Léa|Marc|Lyon|Burn/.test(r)?3:9;
 return rows.filter(r=>pri(r)<9).sort((x,y)=>pri(x)-pri(y)).slice(0,8).map(r=>r.split(' ')[0]);
}
const stageTokens=c=>Math.ceil((c.system.length+c.input.length+c.schema.length)/3);
// Runs one turn on the filled profile; returns every provider call (purpose, system, input, schema, tokens).
export async function replayTurn(message,{strategy='ideal',choose=null}={}){
 const s=setup();await s.ready();seedProfile(s);seedReviews(s);seedMemoryAndHistory(s);
 const calls=[],original=globalThis.fetch;
 globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body),purpose=body.response_format?.json_schema.name,input=body.messages[1].content,system=Array.isArray(body.messages[0].content)?body.messages[0].content[0].text:body.messages[0].content,call={purpose,input,system,schema:JSON.stringify(body.response_format?.json_schema.schema||{})};call.tokens=stageTokens(call);calls.push(call);
  if(purpose==='ambre_analysis'){const data=JSON.parse(input);return response(analysis({selected_ids:choose?choose(data.index):pick(strategy,data.index)}))}
  return response({ambre_risk_check:{risk:'none',category:'none',evidence_quote:''},ambre_writing:draft(),ambre_review:review(),ambre_memory:{items:[]}}[purpose])};
 try{await s.send(message,'k-'+Math.random().toString(36).slice(2,10));for(let n=0;n<8&&!['done','failed'].includes(s.job()?.stage);n++)await s.chat.engine.work(s.env,s.job().turn_id)}finally{globalThis.fetch=original}
 return {s,calls,stage:s.job()?.stage,get:p=>calls.find(c=>c.purpose===p)};
}
// Runs an initial analysis or a point review on the filled profile; returns the provider input.
export async function replayReview(kind){
 const s=setup();seedProfile(s);
 if(kind==='point'){await s.ready();seedReviews(s);seedMemoryAndHistory(s)}
 const calls=[],original=globalThis.fetch;
 globalThis.fetch=async(url,init)=>{const body=JSON.parse(init.body);calls.push(body);return response({content:'BILAN_GENERE '+'Ce qui a évolué depuis le dernier bilan. '.repeat(60)})};
 let r;try{r=kind==='point'?await s.call('/api/chat/review','POST',{idempotency_key:'point-0001'}):await s.call('/api/chat/start','POST',{idempotency_key:'start-0001',accepted:true})}finally{globalThis.fetch=original}
 const body=calls[0],system=typeof body.messages[0].content==='string'?body.messages[0].content:body.messages[0].content[0].text,user=body.messages[1].content;
 return {s,r,system,user,data:JSON.parse(user.slice(user.indexOf('{'))),tokens:Math.ceil((system.length+user.length+JSON.stringify(body.response_format?.json_schema?.schema||{}).length)/3)};
}
