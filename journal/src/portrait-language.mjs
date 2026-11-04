export const PORTRAIT_ANALYSIS_VERSION=3;
// Same identity contract as the Python analyst. Civil identity never crosses
// into story-writing context; declared context is grammar/context, not proof.
export function declaredLifeContext(input={}){
 const age=Number.isInteger(input.age)&&input.age>=0&&input.age<=130?input.age:null;
 const gender=['male','female','non_binary','self_described','unspecified'].includes(input.gender)?input.gender:'unspecified';
 return {age,gender,gender_description:String(input.gender_description||'').trim().slice(0,200),occupation:String(input.occupation||'').trim().slice(0,400)};
}
export const withoutQuotedText=value=>String(value||'').replace(/«[^»]*»|“[^”]*”|"[^"\n]*"|‘[^’]*’/g,'');
export function assertPortraitLanguage(p,identity=p.identity_context||{}){
 // Explicit names/roles for third parties avoid ambiguous pronoun resolution.
 // We regenerate unsafe prose, rather than rewriting a mother's or friend's words.
 const prose=[p.summary,...(p.axes||[]).map(x=>x.why),...(p.strengths||[]).map(x=>x.text),...(p.uncertainties||[]),p.current_state?.summary,p.evolution?.summary];
 for(const h of p.psychological_map||[])prose.push(h.observation,h.possible_function,...h.alternatives,h.next_question,h.influence?.rationale,...(h.interactions||[]).map(x=>x.explanation));
 if(prose.some(text=>/(?:^|[\s(’'])\b(?:elle|elles|il|ils)\b/i.test(withoutQuotedText(text))))throw Error('portrait_language');
 const attributes={créative:'créatif',attentive:'attentif',réfléchie:'réfléchi',curieuse:'curieux',intuitive:'intuitif',anxieuse:'anxieux',fatiguée:'fatigué',épuisée:'épuisé',heureuse:'heureux',réservée:'réservé',confiante:'confiant',exigeante:'exigeant'};
 const wrong=String(identity.gender_description||'').trim()?new Set():identity.gender==='male'?new Set(Object.keys(attributes)):identity.gender==='female'?new Set(Object.values(attributes)):new Set();
 for(const text of prose){const clean=withoutQuotedText(text),direct=/\{\{prenom\}\}\s+(?:se\s+(?:décrit|perçoit)\s+comme|se\s+dit|est|semble(?:\s+être)?|se\s+sent)\s+(?:(?:très|plutôt|souvent|parfois|assez)\s+)*([A-Za-zÀ-ÿ]+)/gi;for(const match of clean.matchAll(direct))if(wrong.has(match[1].toLowerCase()))throw Error('portrait_gender');}
}
