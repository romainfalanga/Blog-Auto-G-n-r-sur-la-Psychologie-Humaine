// Recette du 19/11 : l'analyse du bloc 0 d'un instantané complet était rejetée trois fois
// (invalid_response). Le fragment est un extrait du texte JSON de l'élément : les mots de la
// personne y figurent échappés (\" pour un guillemet, \n pour un retour à la ligne), et le
// résumé du modèle dépassait 800 caractères (maxLength non garanti en mode strict). Ces tests
// reproduisent le refus avec une sortie réaliste, vérifient la correction et la répétition
// complète d'un bilan de 6 blocs. Fournisseur simulé uniquement, données synthétiques.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup,provider} from './v4-R-helper.mjs';
import {seedCompanionPrerequisites} from './companion-ready-helper.mjs';
import {SYMPTOMES_CARTE} from '../src/symptomes-carte.generated.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {readReviewSnapshot,snapshotChunks,validateSnapshotChunk,normalizeQuoteText,SNAPSHOT_CHUNK_SCHEMA} from '../src/review-snapshot.mjs';

const SELF='Je me tais souvent pour éviter les conflits avec ma sœur. Quand on me demande « ça va ? », je réponds "oui" par réflexe.\nJ’ai du mal à dire ce que je ressens, même à mon compagnon.';
const PASS={criteria:{no_diagnosis:'yes',quotes_faithful:'yes',safety_resources:'na',hypotheses_open:'yes',no_saved_claim:'yes',sources_honest:'yes'},reasons:[]};
const DIGEST={formulation:{holds:[],revised:[],new:[]},goals:[],tried:[],alliance:{text:'',quote:''},open_questions:[],next_focus:'',safety:''};
const CONTENT='Ce qui semble ressortir de vos réponses et de ce que vous écrivez. Hypothèse : une difficulté à identifier et nommer vos émotions, à explorer avec vous ; il manque des exemples récents. '+'Nous pourrons vérifier ensemble ce qui vous ressemble et ce qui ne vous ressemble pas, sans conclusion hâtive. '.repeat(12)+'Qu’avez-vous envie de comprendre en priorité ?';
const CHAT='Votre bilan est prêt. Ce qui ressort : hypothèse : une difficulté à nommer vos émotions, à vérifier avec vous ; il manque des exemples récents. Qu’aimeriez-vous creuser en premier ? Qu’en pensez-vous ?';
// The validation of the code before the fix, kept to prove that the realistic output was refused.
const legacyValid=(value,chunk)=>Boolean(value&&typeof value.summary==='string'&&value.summary.trim()&&value.summary.length<=800&&Array.isArray(value.quotes)&&value.quotes.length<=3&&value.quotes.every(q=>chunk.items.some(x=>x.path===q.path&&x.text.includes(q.quote))&&q.quote.length<=160));

// Complete synthetic account (recette du 19/11): 516 answers of which 21 « Dans le passé » and 4
// notes, Mon enfance 60/60, 3 proches and 3 moments with multi-line texts.
function fullProfile(t){
 const s=setup();t.after(()=>s.raw.close());
 s.raw.prepare("UPDATE profiles SET self_data=? WHERE user_id='a'").run(JSON.stringify({first_name:'Alix',self_description:SELF,emotions:'Je sens « une boule » dans le ventre avant les repas de famille.\nMa mère dit : "tu exagères".',current_context:'Je travaille beaucoup ; le soir je n’arrive plus à savoir si je suis triste ou fatiguée.'}));
 seedCompanionPrerequisites(s.raw,'a');
 const answers=JSON.parse(s.raw.prepare("SELECT answers FROM symptom_maps WHERE user_id='a'").get().answers),plain=SYMPTOMES_CARTE.factors.filter(f=>!f.safety&&!f.historical);
 plain.slice(0,21).forEach(f=>{answers[f.id]='past'});plain.slice(30,110).forEach(f=>{answers[f.id]=1});plain.slice(120,160).forEach(f=>{answers[f.id]='unknown'});
 const notes=Object.fromEntries(plain.slice(30,34).map((f,i)=>[f.id,['Surtout le dimanche soir, avant le travail.','Ma sœur me l’a fait remarquer : "tu ne dis jamais rien".','Plus fort depuis le déménagement.\nMoins depuis l’été.','Je ne sais pas si c’est de la colère ou de la fatigue.'][i]]));
 s.raw.prepare("UPDATE symptom_maps SET answers=?,notes=? WHERE user_id='a'").run(JSON.stringify(answers),JSON.stringify({factors:notes}));
 const child=Object.fromEntries(CARTOGRAPHIE.factors.map((f,i)=>[f.id,i%3?1:0]));
 s.raw.prepare("UPDATE cartographies SET answers=? WHERE user_id='a'").run(JSON.stringify(child));
 const relation='Nous nous voyons chaque semaine. Quand elle me demande ce que je ressens, je réponds "je ne sais pas" et elle s’agace.\nJ’aimerais lui expliquer que ce n’est pas de la mauvaise volonté. ';
 s.raw.prepare("SELECT id FROM personal_people WHERE user_id='a' ORDER BY id").all().forEach((p,i)=>s.raw.prepare('UPDATE personal_people SET relationship_text=? WHERE id=?').run(relation.repeat(4+i),p.id));
 const moment='Le jour du mariage de mon frère, tout le monde pleurait et moi je ne ressentais rien de net ; j’ai fait « comme si ».\nAprès coup je me suis sentie coupable. ';
 s.raw.prepare("SELECT id FROM life_events WHERE user_id='a' ORDER BY id").all().forEach((e,i)=>s.raw.prepare('UPDATE life_events SET description=? WHERE id=?').run(moment.repeat(4+i),e.id));
 s.raw.prepare("UPDATE memories SET text=? WHERE user_id='a'").run('Précision : je préfère qu’on me pose des questions concrètes.\nJe ne veux pas parler de mon père pour l’instant.');
 return s;
}
// A realistic block analysis: a summary over 800 characters, danger signs first, and quotes copied
// as a model writes them (unescaped, straight apostrophes, a line break turned into a space).
function realistic(chunk,{invent=false}={}){
 const quotes=[];
 for(const item of chunk.items){
  if(quotes.length>=2)break;
  const plainText=item.text.replace(/\\n/g,'\n').replace(/\\"/g,'"');
  const said=plainText.match(/je réponds "[^"]+" par réflexe\.\nJ’ai du mal à dire ce que je ressens/)||plainText.match(/je réponds "je ne sais pas" et elle s’agace\.\nJ’aimerais/)||plainText.match(/j’ai fait « comme si »\.\nAprès coup/)||plainText.match(/"tu ne dis jamais rien"/);
  if(said)quotes.push({path:item.path,quote:said[0].replace(/\n/g,' ').replace(/’/g,"'")});
 }
 if(invent)quotes.push({path:chunk.items[0].path,quote:'Je déteste ma famille et je veux tout quitter'});
 const summary='Signes de danger : aucune idée suicidaire actuelle déclarée dans ce bloc. Bloc '+chunk.index+' : la personne décrit une difficulté à identifier et nommer ses émotions, des réponses « Oui » nombreuses sur la régulation émotionnelle, plusieurs réponses « Dans le passé » qui concernent une période antérieure aux douze derniers mois et non un signe actuel, et des réponses « Je ne sais pas » à explorer. '+'Les déclarations restent à vérifier avec elle, sans diagnostic ; contradictions possibles entre fatigue et tristesse, à préciser. '.repeat(6);
 return {summary,quotes};
}

test('the realistic analysis of block 0 is refused by the former check and accepted now, with its exact quotes',async t=>{
 const s=fullProfile(t);provider(t,(purpose)=>purpose==='ambre_snapshot_chunk'?{summary:'x',quotes:[]}:purpose==='ambre_review'?{content:CONTENT,chat_message:CHAT,digest:DIGEST}:purpose==='ambre_review_check'?PASS:undefined);
 const r=await s.call('/api/chat/start','POST',{idempotency_key:'start-key-0001',accepted:true});assert.equal(r.status,202,JSON.stringify(r));
 const chunks=snapshotChunks((await readReviewSnapshot(s.env,'a',r.review_id)).payload),block=chunks[0];
 assert.equal(chunks.length,6,'same size as the 19/11 snapshot');
 assert.deepEqual(block.items.slice(0,3).map(x=>x.path),['profile/0','childhood/0','symptoms/0']);
 const output=realistic(block);
 assert.ok(output.summary.length>800&&output.quotes.length>=1);
 assert.equal(JSON.parse(JSON.stringify(output)).quotes[0].path,'profile/0');
 assert.equal(legacyValid(output,block),false,'the former check refused this output (reproduction)');
 assert.equal(legacyValid({...output,summary:'Court.'},block),false,'even with a short summary: the escaped quote was never found');
 const kept=validateSnapshotChunk(output,block);
 assert.ok(kept.summary.length<=800&&kept.summary.endsWith('.'),'cut at the last full sentence');assert.match(kept.summary,/^Signes de danger/);
 assert.equal(kept.adjusted.summary_chars,output.summary.length);
 assert.deepEqual(kept.quotes,output.quotes);
 // The schema still states the limits sent to the provider.
 assert.equal(SNAPSHOT_CHUNK_SCHEMA.properties.summary.maxLength,800);assert.equal(SNAPSHOT_CHUNK_SCHEMA.properties.quotes.maxItems,3);
});

test('a quote absent from the block is still refused, with a precise reason and no content',()=>{
 const chunk={index:0,items:[{path:'profile/0',offset:0,total_chars:80,text:'{"self_description":"je réponds \\"oui\\" par réflexe.\\nJ’ai du mal"}'},{path:'people/0',offset:0,total_chars:40,text:'{"relationship_text":"Elle s’agace."}'}]};
 assert.equal(validateSnapshotChunk({summary:'Résumé.',quotes:[{path:'profile/0',quote:'je réponds "oui" par réflexe. J\'ai du mal'}]},chunk).quotes.length,1);
 // Wrong path but a real quote: attributed to the fragment where it is.
 assert.deepEqual(validateSnapshotChunk({summary:'Résumé.',quotes:[{path:'profile/0',quote:'Elle s\'agace.'}]},chunk).quotes,[{path:'people/0',quote:'Elle s\'agace.'}]);
 for(const [quote,reason] of [['je réponds "non" par réflexe','quote_not_found'],['je réponds oui par réflexe','quote_not_found'],['réflexe. J’ai du mal à dormir','quote_not_found']]){
  assert.throws(()=>validateSnapshotChunk({summary:'Résumé.',quotes:[{path:'profile/0',quote}]},chunk),e=>e.code==='invalid_response'&&e.rejection.reason===reason&&!JSON.stringify(e.rejection).includes('réflexe'));
 }
 assert.throws(()=>validateSnapshotChunk({summary:'Résumé.',quotes:[{path:'inconnu/9',quote:'invention'}]},chunk),e=>e.rejection.reason==='quote_not_found_unknown_path');
 assert.throws(()=>validateSnapshotChunk({summary:' ',quotes:[]},chunk),e=>e.rejection.reason==='summary_missing');
 assert.throws(()=>validateSnapshotChunk({summary:'Résumé.'},chunk),e=>e.rejection.reason==='quotes_missing');
 // A quote straddling two contiguous fragments of the same item is found; too many quotes are kept to three.
 const split={index:1,items:[{path:'events/0',offset:0,total_chars:30,text:'{"description":"Je fais '},{path:'events/0',offset:24,total_chars:30,text:'comme si."}'}]};
 assert.equal(validateSnapshotChunk({summary:'R.',quotes:[{path:'events/0',quote:'Je fais comme si.'}]},split).quotes.length,1);
 const four=validateSnapshotChunk({summary:'R.',quotes:Array(4).fill({path:'people/0',quote:'Elle s’agace.'})},chunk);assert.equal(four.quotes.length,3);assert.equal(four.adjusted.dropped,1);
 assert.equal(normalizeQuoteText('« ça va ? »  l’été'),normalizeQuoteText('"ça va ?" l\'été'));
});

test('rehearsal: complete review of 6 blocks with realistic analyses ends ready',async t=>{
 const s=fullProfile(t);
 const calls=provider(t,(purpose,body)=>{if(purpose==='ambre_snapshot_chunk')return realistic(JSON.parse(body.messages[1].content));if(purpose==='ambre_review')return {content:CONTENT,chat_message:CHAT,digest:DIGEST};if(purpose==='ambre_review_check')return PASS;});
 const r=await s.call('/api/chat/start','POST',{idempotency_key:'start-key-0001',accepted:true});assert.equal(r.status,202,JSON.stringify(r));
 const row=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);assert.equal(row.status,'ready',JSON.stringify({code:row.error_code,sources:row.sources}));
 const blocks=calls.filter(c=>c.purpose==='ambre_snapshot_chunk');assert.equal(blocks.length,6);
 assert.match(blocks[0].body.messages[0].content,/800 caractères espaces compris/);assert.match(blocks[0].body.messages[0].content,/sans les barres obliques d’échappement/);
 const write=JSON.parse(calls.find(c=>c.purpose==='ambre_review').body.messages[1].content.replace(/^[^{]*/,''));
 assert.equal(write.coverage.analyzed_blocks,6);assert.ok(write.blocks.every(b=>b.summary.length<=800));assert.ok(write.blocks.some(b=>b.quotes.length));
 assert.equal(JSON.parse(row.sources).rejections,undefined);
});

test('a block refused three times leaves its exact reasons in the review sources and metrics',async t=>{
 const s=fullProfile(t);
 const calls=provider(t,(purpose,body)=>purpose==='ambre_snapshot_chunk'?realistic(JSON.parse(body.messages[1].content),{invent:true}):undefined);
 const r=await s.call('/api/chat/start','POST',{idempotency_key:'start-key-0001',accepted:true});assert.equal(r.status,202);
 for(let i=0;i<3;i++){const row=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);if(row.status!=='running')break;s.tick(61);await s.chat.resumeReviews(s.env);}
 const row=s.raw.prepare('SELECT * FROM ambre_reviews WHERE id=?').get(r.review_id);
 assert.equal(row.status,'failed');assert.equal(row.error_code,'invalid_response');
 assert.equal(calls.filter(c=>c.purpose==='ambre_snapshot_chunk').length,3);
 const rejections=JSON.parse(row.sources).rejections;
 assert.equal(rejections.length,3);assert.deepEqual(rejections.map(x=>[x.stage,x.block,x.reason,x.attempt]),[['snapshot_chunk',0,'quote_not_found',1],['snapshot_chunk',0,'quote_not_found',2],['snapshot_chunk',0,'quote_not_found',3]]);
 assert.doesNotMatch(row.sources,/déteste|Alix|réflexe/,'no personal content in the diagnostics');
 assert.equal(s.raw.prepare("SELECT value FROM service_metrics WHERE name='review_rejected_quote_not_found'").get().value,3);
 const page=await s.call('/api/chat');assert.equal(page.journey.review.last_failure.error_code,'invalid_response');
});
