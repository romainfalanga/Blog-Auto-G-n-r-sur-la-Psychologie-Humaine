// Descriptive assessment of member declarations, never an automated diagnosis.
import { SYMPTOMES_CARTE } from './symptomes-carte.generated.mjs';
export const CLINICAL_ASSESSMENT_VERSION = '2026-11-19.2';
const MISSING = ['duration','contemporaneity','impairment','onset','exclusions','differential'];
const SOURCES = ['https://www.who.int/publications/i/item/9789240077263'];
const CONTEXT_TYPES = {ten_symptomes_depressifs_perinataux:'perinatal',ten_violences_amoureuses_jeunes:'relationship',ten_violences_couple:'relationship',ten_automutilations_non_suicidaires:'intent',ten_procrastination:'behaviour'};
const excerpt = value => { let out='',bytes=0;for(const char of value){const size=new TextEncoder().encode(char).length;if(bytes+size>160)break;out+=char;bytes+=size;}return out; };
const object = value => value && typeof value === 'object' && !Array.isArray(value);
function coverage(outcome) {
 const code=outcome.code||'';
 if (/^(7A|HA|8A)/.test(code)) return 'outside_cddr_pending';
 if (code.startsWith('6D11')||outcome.category==='pattern_qualifier') return 'specifier_not_standalone';
 if (outcome.category==='diagnosis') return 'clinical_reference_partial';
 if (['clinical_group','symptom_group','categorical_personality_diagnosis'].includes(outcome.category)) return 'aggregate_not_diagnosis';
 return 'non_diagnostic_construct';
}
export function buildClinicalAssessment(snapshot) {
 if (!object(snapshot)||snapshot.schema_version!==1||snapshot.complete!==true||!object(snapshot.sections)||typeof snapshot.review_id!=='string'||!snapshot.review_id||!Number.isFinite(Date.parse(snapshot.captured_at))) throw new TypeError('Instantané complet et daté requis.');
 const rows=snapshot.sections.symptoms||[];
 if (!Array.isArray(rows)||rows.length>1) throw new TypeError('Une seule carte de symptômes est attendue.');
 const map=rows[0]||{},answers=map.answers||{},notes=map.notes?.factors||{};
 if (!object(answers)||!object(notes)) throw new TypeError('Réponses de symptômes invalides.');
 const factors=new Map(SYMPTOMES_CARTE.factors.map(f=>[f.id,f]));
 for (const [id,value] of Object.entries(answers)) {
  if (!factors.has(id)||![0,1,2,'unknown','past'].includes(value)) throw new TypeError('Réponse de symptômes inconnue.');
  if (value==='past'&&(typeof notes[id]!=='string'||!notes[id].trim())) throw new TypeError('Une précision est requise pour le passé.');
 }
 const links=new Map();
 for(const link of SYMPTOMES_CARTE.links){if(!links.has(link.outcome))links.set(link.outcome,[]);links.get(link.outcome).push(link);}
 const outcomes=SYMPTOMES_CARTE.outcomes.map(outcome=>{
  const evidence={current_window:[],historical:[],denied:[],unknown:[]};
  for(const link of links.get(outcome.id)||[]){
   const value=answers[link.factor],factor=factors.get(link.factor);
   const bucket=value===1||value===2?'current_window':value==='past'?'historical':value===0?'denied':'unknown';
   evidence[bucket].push({factor_id:factor.id,label:factor.label,answer:value===undefined?'unanswered':value,note_ref:typeof notes[factor.id]==='string'?`sections.symptoms[0].notes.factors.${factor.id}`:null,note_excerpt:typeof notes[factor.id]==='string'?excerpt(notes[factor.id]):'',note_excerpted:typeof notes[factor.id]==='string'&&excerpt(notes[factor.id])!==notes[factor.id],source_ref:`sections.symptoms[0].answers.${factor.id}`,provenance:'user_answer',strength:link.strength,overlap_group:factor.overlap_group||factor.id});
  }
  const required=SYMPTOMES_CARTE.requires?.[outcome.id]||[];
  const gates={current_window:required.every(id=>answers[id]===1||answers[id]===2),historical:required.every(id=>answers[id]==='past')};
  const contextType=required.length?(CONTEXT_TYPES[outcome.id]||'product'):null;
  const current=evidence.current_window.some(e=>e.strength==='critere')&&gates.current_window;
  const past=evidence.historical.some(e=>e.strength==='critere')&&gates.historical;
  const association=evidence.current_window.some(e=>!required.includes(e.factor_id))&&gates.current_window;
  const pastAssociation=evidence.historical.some(e=>!required.includes(e.factor_id))&&gates.historical;
  const status=coverage(outcome);
  return {outcome_id:outcome.id,label:outcome.label,kind:outcome.category,coverage_status:status,
   rule_refs:status==='clinical_reference_partial'||status==='specifier_not_standalone'?SOURCES.slice():[],evidence,
   dimensions:{manifestations:current?'supported':past?'historical':'unknown',...Object.fromEntries(MISSING.map(k=>[k,'unknown']))},
   hypothesis_state:current?'criterion_signals':past?'historical_signals':association?'association_only':pastAssociation?'historical_association_only':'insufficient_information',
   missing_fact_ids:[...MISSING,...(contextType?[contextType==='product'?'symptom_product_attribution':'symptom_context_attribution']:[]),...(!gates.current_window?[contextType==='product'?'current_product_use':`current_${contextType}_context`]:[])],
   context_gates:{type:contextType,required_factor_ids:required,current_window:gates.current_window,historical:gates.historical},
   clinician_report_refs:[],explanation_refs:[]};
 });
 return {schema_version:1,rules_version:CLINICAL_ASSESSMENT_VERSION,snapshot_id:snapshot.review_id,review_id:snapshot.review_id,captured_at:snapshot.captured_at,
  dataset_version:'symptomes-v2',dataset_sha256:DATASET_SHA256,period:'12_mois',method:'declared_manifestations_only',
  diagnostic_status:'not_diagnostic',outcomes,counts:{outcomes:outcomes.length,associations:outcomes.filter(o=>o.hypothesis_state==='association_only'||o.hypothesis_state==='historical_association_only').length,current:outcomes.filter(o=>o.hypothesis_state==='criterion_signals').length,historical:outcomes.filter(o=>o.hypothesis_state==='historical_signals').length}};
}

const DATASET_SHA256 = '622de9067654b84a84c9cda9c7564c3f7700fda2981df5479142eebb0d05c128';
