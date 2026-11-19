// Mes symptômes v2: declared answers and hypotheses, never questionnaire scores.
import {SYMPTOMES_CARTE} from './symptomes-carte.generated.mjs';
import {computeCartography,readSymptomMap} from './cartographie.mjs';
const parse=(x,f={})=>{if(x&&typeof x==='object')return x;try{return JSON.parse(x)||f}catch{return f}};
export const MAP_HYPOTHESIS_RULES='Mon enfance décrit des vécus ; symptom_map décrit des manifestations déclarées pendant ces 12 derniers mois. Croise ces informations avec la conversation sans déduire de causalité individuelle. Tu peux nommer une hypothèse probable en précisant ce qui l’appuie et ce qui manque : « hypothèse : X, appuyée par A et B ; il manque C ». Jamais un verdict, « vous avez [un trouble] », « vous souffrez de [un trouble] », un score ou une probabilité chiffrée. Respecte les Non, Je ne sais pas et Pas moi ; distingue une hypothèse d’un fait. Une orientation professionnelle est pertinente en cas de danger, de forte gêne au quotidien ou sur demande, pas systématiquement.';
export const MAP_SAFETY_RULE='map_safety.ask_now=true : un signe de sécurité a été déclaré dans Mes symptômes. Ce n’est pas la preuve d’un danger actuel. Pose calmement une question directe de sécurité, avec le 3114 et le 15/112 si danger immédiat, avant de poursuivre. Ne répète pas cette question à partir du seul questionnaire pendant 14 jours ; un danger actuel dans le message reste prioritaire.';
export const SAFETY_WINDOW=14*86400;
export function mapSafetyState(row,now,data=SYMPTOMES_CARTE){
 if(!row||!Number.isFinite(row.safety_at)||row.safety_at>now)return null;
 const answers=parse(row.answers),kinds=[...new Set(data.factors.filter(f=>f.safety&&Number(answers[f.id])>0).map(f=>f.safety))];
 if(!kinds.length&&now-row.safety_at>=SAFETY_WINDOW)return null;
 return {reported_at:row.safety_at,kinds:kinds.length?kinds:['suicide'],ask_now:!Number.isFinite(row.safety_asked_at)||now-row.safety_asked_at>=SAFETY_WINDOW};
}
export function ensureMapSafetyQuestion(content,safety,{risk='none'}={}){
 if(!safety?.ask_now||risk!=='none')return String(content||'');
 if(/(?:êtes|etes)-vous[^?]{0,80}en s[ée]curit[ée][^?]{0,60}\?/iu.test(content))return /\b3114\b/.test(content)?content:content+' En présence d’idées suicidaires, le 3114 est disponible gratuitement, 24 h/24. En cas de danger immédiat, appelez le 15 ou le 112.';
 return 'Vous avez signalé des idées de mort ou des gestes pour vous faire du mal dans « Mes symptômes ». En ce moment, êtes-vous en sécurité ? Si des idées suicidaires sont présentes, le 3114 est disponible gratuitement, 24 h/24. En cas de danger immédiat, appelez le 15 ou le 112.\n\n'+String(content||'');
}
const answerWord=v=>v===undefined?'Non renseigné':v==='unknown'?'Je ne sais pas':Number(v)>0?'Oui':'Non';
const clip=(s,n)=>String(s||'').length>n?String(s).slice(0,n-1)+'…':String(s||'');
export function symptomMapView(row,data=SYMPTOMES_CARTE){
 if(!row)return null;
 const answers=parse(row.answers),feedback=parse(row.feedback),notes=parse(row.notes),state=computeCartography(answers,data,feedback);
 const ranked=[...(state.sections?.top||[]),...(state.sections?.other||[]),...(state.sections?.dismissed||[])];
 const themes=data.groups.map(g=>({id:g.id,label:g.label||g.title,answers:data.factors.filter(f=>(f.group||f.theme)===g.id&&(Object.hasOwn(answers,f.id)||notes.factors?.[f.id])).map(f=>({id:f.id,label:f.label,answer:answerWord(answers[f.id]),...(notes.factors?.[f.id]?{note:notes.factors?.[f.id]}:{})}))}));
 const factorById=new Map(data.factors.map(f=>[f.id,f]));
 const subjects=ranked.map(r=>({id:r.outcome.id,label:r.outcome.label,c:r.essential_count,C:r.essential_total,m:r.moderate_count,feedback:feedback[r.outcome.id]||null,links:data.links.filter(l=>l.outcome===r.outcome.id).map(l=>({factor:l.factor,label:factorById.get(l.factor)?.label,answer:answerWord(answers[l.factor]),strength:l.strength,summary:l.summary,...(notes.factors?.[l.factor]?{note:notes.factors[l.factor]}:{})})),...(notes.outcomes?.[r.outcome.id]?{note:notes.outcomes?.[r.outcome.id]}:{})}));
 for(const o of data.outcomes)if(!subjects.some(s=>s.id===o.id)&&(feedback[o.id]||notes.outcomes?.[o.id]))subjects.push({id:o.id,label:o.label,feedback:feedback[o.id]||null,note:notes.outcomes?.[o.id]||null,ranked:false});
 return {version:row.version,updated_at:row.updated_at,period:'ces 12 derniers mois',answered:Object.keys(answers).length,total:data.factors.length,themes,subjects,feedback,notes,safety_at:row.safety_at};
}
export function symptomMapBlock(view,max=3500){
 if(!view)return null;
 const lines=['Mes symptômes — '+view.period+' : '+view.answered+'/'+view.total+' réponses. Hypothèses, jamais diagnostics.'];
 for(const theme of view.themes){const yes=theme.answers.filter(a=>a.answer==='Oui'),uncertain=theme.answers.filter(a=>a.answer==='Je ne sais pas');if(yes.length||uncertain.length)lines.push(theme.label+' : Oui '+yes.map(a=>a.label).join(' ; ')+' | Je ne sais pas '+uncertain.map(a=>a.label).join(' ; '));}
 const ranked=view.subjects.filter(s=>s.ranked!==false&&s.feedback!=='non').slice(0,10).map(s=>s.label+' (c/C='+s.c+'/'+s.C+', m='+s.m+')'+(s.feedback?' — '+s.feedback:''));
 const feedback=Object.entries(view.feedback).filter(([,v])=>v==='oui'||v==='non').map(([id,v])=>id+': '+(v==='oui'?'Je me reconnais':'Pas moi'));
 // Reserve ranking/feedback before shortening theme text; full details remain in the index.
 const tail='\nSujets à explorer : '+ranked.join('; ')+'\nRetours : '+feedback.join('; ');
 const boundedTail=clip(tail,Math.floor(max*.55));
 return clip(lines.join('\n'),max-boundedTail.length)+boundedTail;
}
export function symptomMapIndexEntries(view){
 if(!view)return [];
 return [...view.themes.filter(t=>t.answers.length).map(t=>({key:'symptom_map:theme:'+t.id,kind:'symptom_map',label:'Mes symptômes · '+t.label,content:t,parts:t.answers.map(a=>({name:a.label,text:a.answer+(a.note?' — '+a.note:'')})),header:t.label+' — ces 12 derniers mois',updated_at:view.updated_at})),...view.subjects.map(s=>({key:'symptom_map:subject:'+s.id,kind:'symptom_map',label:'Hypothèse à explorer · '+s.label,content:s,header:s.label+' — hypothèse, c/C='+s.c+'/'+s.C+', m='+s.m,parts:[...(s.links||[]).map(l=>({name:l.label||l.factor,text:l.answer+' ('+l.strength+') — '+l.summary+(l.note?' — '+l.note:'')})),...(s.note?[{name:'Vos mots',text:s.note}]:[])],updated_at:view.updated_at}))];
}
export async function readSymptomContext(env,userId,now){
 const row=await readSymptomMap(env,userId);if(!row)return null;
 const view=symptomMapView(row);return {view,block:symptomMapBlock(view),safety:mapSafetyState(row,now),revision:[row.version,row.updated_at,row.safety_at,row.source_revision||0]};
}
// Fixed childhood overview; its full evidence and notes remain in the existing knowledge index.
export function childhoodMapBlock(summary,max=3000){
 if(!summary)return null;
 const lines=['Mon enfance : vécus déclarés, pas une cause individuelle démontrée.', 'Oui : '+(summary.causes||[]).map(x=>x.label).join('; '),'Je ne sais pas : '+(summary.explicit_uncertain||[]).map(x=>x.label).join('; '),'Je me reconnais : '+(summary.recognized||[]).map(x=>x.label).join('; '),'Pas moi : '+(summary.dismissed||[]).map(x=>x.label).join('; ')];
 return clip(lines.join('\n'),max);
}

export async function symptomMapStorageAvailable(env){return Boolean(await env.DB.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='symptom_maps'").first());}
export function symptomReviewBlock(view,max=6000){
 if(!view)return null;
 const all=[...Object.entries(view.notes?.factors||{}).map(([id,text])=>({target:'factor',id,text})),...Object.entries(view.notes?.outcomes||{}).map(([id,text])=>({target:'outcome',id,text}))];
 const notes=[];let used=0;for(const note of all){if(used+JSON.stringify(note).length>Math.floor(max*.45))continue;notes.push(note);used+=JSON.stringify(note).length;}
 const result={overview:'',person_notes:notes,notes_omitted:all.length-notes.length};
 result.overview=symptomMapBlock(view,Math.max(300,max-JSON.stringify(result).length-32));
 return result;
}
