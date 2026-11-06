export const PORTRAIT_ANALYSIS_VERSION=3;
// Same identity contract as the Python analyst. Civil identity never crosses
// into story-writing context; declared context is grammar/context, not proof.
export function declaredLifeContext(input={}){
 const age=Number.isInteger(input.age)&&input.age>=0&&input.age<=130?input.age:null;
 const gender=['male','female','non_binary','self_described','unspecified'].includes(input.gender)?input.gender:'unspecified';
 return {age,gender,gender_description:String(input.gender_description||'').trim().slice(0,200),occupation:String(input.occupation||'').trim().slice(0,400)};
}
export const withoutQuotedText=value=>String(value||'').replace(/«[^»]*»|“[^”]*”|"[^"\n]*"|‘[^’]*’/g,'');
// Only allow forms where « il » has no referent. « Il y a » can also refer
// to a person (e.g. « il y a vécu »). Keep hyphenated inversions.
const impersonal=String.raw`(?:s[’']agit|se\s+peut)\b|(?:semble|paraît)\s+qu(?:e\b|[’'])|est\s+possible\s+(?:de\b|d[’'])|reste\s+(?:un\s+point|une\s+question)\s+à\s+(?:préciser|clarifier|explorer|comprendre)\b`;
const ambiguousPronoun=new RegExp(String.raw`(?:^|[^\p{L}\p{M}\p{N}_‐‑-])(?:elles?|ils|il(?!\s+(?:${impersonal})))(?![\p{L}\p{M}\p{N}_])`,'iu');
export function assertPortraitLanguage(p,identity=p.identity_context||{}){
 // Explicit names/roles for third parties avoid ambiguous pronoun resolution.
 // We regenerate unsafe prose, rather than rewriting a mother's or friend's words.
 const prose=[p.summary,...(p.axes||[]).flatMap(x=>[x.label,x.why]),...(p.strengths||[]).map(x=>x.text),...(p.uncertainties||[]),p.current_state?.summary,p.evolution?.summary,...['preferred','avoid'].flatMap(field=>(p.reading_preferences?.[field]||[]).map(x=>x.label))];
 for(const h of p.psychological_map||[])prose.push(h.observation,h.possible_function,...h.alternatives,h.next_question,h.influence?.rationale,...(h.interactions||[]).map(x=>x.explanation));
 if(prose.some(text=>ambiguousPronoun.test(withoutQuotedText(text))))throw Error('portrait_language');
 const attributes={créative:'créatif',attentive:'attentif',réfléchie:'réfléchi',curieuse:'curieux',intuitive:'intuitif',anxieuse:'anxieux',fatiguée:'fatigué',épuisée:'épuisé',heureuse:'heureux',réservée:'réservé',confiante:'confiant',exigeante:'exigeant'};
 const wrong=String(identity.gender_description||'').trim()?new Set():identity.gender==='male'?new Set(Object.keys(attributes)):identity.gender==='female'?new Set(Object.values(attributes)):new Set();
 for(const text of prose){const clean=withoutQuotedText(text),direct=/\{\{prenom\}\}\s+(?:se\s+(?:décrit|perçoit)\s+comme|se\s+dit|est|semble(?:\s+être)?|se\s+sent)\s+(?:(?:très|plutôt|souvent|parfois|assez)\s+)*([A-Za-zÀ-ÿ]+)/gi;for(const match of clean.matchAll(direct))if(wrong.has(match[1].toLowerCase()))throw Error('portrait_gender');}
}
