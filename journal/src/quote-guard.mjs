// Quote anchoring shared by memory, cartography notes, plan and profile
// proposals and the current-risk check. A quote is the person's own words:
// typography may differ (apostrophes, spaces, case), meaning may not.
const APOSTROPHES=/[’‘ʼ`´′]/,QUOTES=/[«»“”„‟″]/,DASHES=/[‐‑‒–—―−]/,SPACES=/[\s   ⁠]/;
export const MIN_QUOTE_CHARS=12;
// Folds a text for comparison and keeps, for each folded character, the index
// of the original character it came from, so a match maps back to the exact
// original words.
export function foldWithMap(value){
 const source=String(value??'').normalize('NFC'),out=[],map=[];let space=true;
 for(let i=0;i<source.length;i++){
  let c=source[i];
  if(SPACES.test(c)){if(!space){out.push(' ');map.push(i)}space=true;continue}
  space=false;
  if(APOSTROPHES.test(c))c="'";else if(QUOTES.test(c))c='"';else if(DASHES.test(c))c='-';else if(c==='…'){for(const d of '...'){out.push(d);map.push(i)}continue}
  for(const d of c.normalize('NFD').replace(/[̀-ͯ]/g,'').toLowerCase()){out.push(d);map.push(i)}
 }
 while(out.at(-1)===' '){out.pop();map.pop()}
 return {source,text:out.join(''),map};
}
export const foldText=value=>foldWithMap(value).text;
const trimQuote=value=>foldText(value).replace(/^[\s"'.,;:!?-]+|[\s"'.,;:!?-]+$/g,'');
function span(folded,start,length){
 const begin=folded.map[start],last=folded.map[start+length-1];
 let end=last+1;while(end<folded.source.length&&/[̀-ͯ]/.test(folded.source[end]))end++;
 return {start:begin,end,text:folded.source.slice(begin,end)};
}
// Exact match up to typography. Returns the original words, never the model's.
export function locateQuote(message,quote){
 const raw=String(message??''),given=String(quote??'').trim(),exact=given.length>=2?raw.indexOf(given):-1;
 if(exact>=0)return {start:exact,end:exact+given.length,text:given};
 const needle=trimQuote(quote);if(needle.length<2)return null;
 const folded=foldWithMap(message),at=folded.text.indexOf(needle);
 return at<0?null:span(folded,at,needle.length);
}
// Semi-global edit distance (Sellers): the best window of the message within
// a small number of edits. Reserved for the current-risk check, where a
// paraphrased anchor must not silence a danger both models saw.
export function approximateQuote(message,quote,{maxRatio=0.2,minChars=8}={}){
 const needle=trimQuote(quote),folded=foldWithMap(message),text=folded.text.slice(0,12000);
 if(needle.length<minChars||needle.length>600||!text)return null;
 const exact=text.indexOf(needle);if(exact>=0)return {...span(folded,exact,needle.length),distance:0};
 const m=needle.length,limit=Math.floor(m*maxRatio);let prev=new Array(m+1),prevStart=new Array(m+1),best={distance:Infinity,end:-1,start:0};
 for(let j=0;j<=m;j++){prev[j]=j;prevStart[j]=0}
 for(let i=1;i<=text.length;i++){
  const cur=new Array(m+1),curStart=new Array(m+1);cur[0]=0;curStart[0]=i;
  for(let j=1;j<=m;j++){
   const sub=prev[j-1]+(text[i-1]===needle[j-1]?0:1),del=prev[j]+1,ins=cur[j-1]+1;
   if(sub<=del&&sub<=ins){cur[j]=sub;curStart[j]=prevStart[j-1]}else if(del<=ins){cur[j]=del;curStart[j]=prevStart[j]}else{cur[j]=ins;curStart[j]=curStart[j-1]}
  }
  if(cur[m]<best.distance){best={distance:cur[m],end:i,start:curStart[m]}}
  prev=cur;prevStart=curStart;
 }
 if(best.distance>limit||best.end<=best.start)return null;
 return {...span(folded,best.start,best.end-best.start),distance:best.distance};
}
// A clause boundary ends the scope of a preceding negation.
const CLAUSE=/[.!?;:\n]|,|\b(?:mais|car|parce que|puisque|alors que|tandis que|pourtant|cependant|sauf que)\b|\bet (?=je\b|j'|il\b|elle\b|on\b|nous\b|vous\b)/g;
const NEGATION=/(?:^|[\s"'(-])(?:ne|n'|pas|jamais|sans|ni|non pas|aucune?|nullement|plutot que|loin de|faux que)(?=$|[\s"',.;:!?)-])/;
// True when the quoted words are placed under a negation in the same clause
// of the person's message ("Je ne dirais pas que…", "pas que…", "sans…").
export function negatedQuote(message,located){
 if(!located)return false;
 const folded=foldWithMap(message),start=folded.map.findIndex(i=>i>=located.start);
 const before=folded.text.slice(0,start<0?folded.text.length:start);
 let cut=0;for(const match of before.matchAll(CLAUSE))cut=match.index+match[0].length;
 // Scope: the same clause and at most the six words before the quote.
 const words=before.slice(cut).trim().split(/\s+/).slice(-6).join(' ');
 return NEGATION.test(words);
}
// The quote is long enough to carry meaning: twelve characters, or a whole
// sentence of the message with at least two words.
export function substantialQuote(message,located){
 if(!located)return false;
 const words=foldText(located.text).replace(/[^a-z0-9' -]/g,' ').trim().split(/\s+/).filter(Boolean);
 if(foldText(located.text).replace(/[^a-z0-9]/g,'').length>=MIN_QUOTE_CHARS-2&&located.text.trim().length>=MIN_QUOTE_CHARS)return true;
 if(words.length<2)return false;
 const folded=foldText(message),own=trimQuote(located.text);
 return folded.split(/[.!?\n]+/).map(s=>s.trim().replace(/^[\s"'-]+|[\s"'-]+$/g,'')).includes(own);
}
// One verdict for a person's quote: located, long enough and not negated.
export function anchorQuote(message,quote,{allowNegated=false,minimal=true}={}){
 if(typeof quote!=='string')return {ok:false,reason:'missing'};
 const located=locateQuote(message,quote);
 if(!located)return {ok:false,reason:'not_found'};
 if(minimal&&!substantialQuote(message,located))return {ok:false,reason:'too_short',located};
 if(!allowNegated&&negatedQuote(message,located))return {ok:false,reason:'negated',located};
 return {ok:true,located,text:located.text};
}
const STOP=new Set('avec dans pour mais cette comme vous nous votre notre avoir etre suis fait faire plus tout quand alors sans aussi veux voudrais peux parce quelque chose elle elles ils leur leurs mes mon ma ses son sa des les une un est sont etait ete tres bien peu moi toi lui qui que quoi dont cela ceci tous toutes'.split(' '));
const stem=word=>word.length>5?word.slice(0,5):word;
export const contentWords=value=>[...new Set((foldText(value).match(/[a-z]{4,}/g)||[]).filter(word=>!STOP.has(word)).map(stem))];
// The words that carry a proposed value must come from the quote and must not
// be negated there ("Je ne veux surtout pas arrêter de boire" never supports
// a goal "Arrêter de boire"). Returns supported, negated or absent.
export function quoteSupport(message,quote,value,{share=0.5}={}){
 const words=contentWords(value);if(!words.length)return 'supported';
 const located=locateQuote(message,quote);if(!located)return 'absent';
 const quoted=foldText(located.text),present=words.filter(word=>new RegExp('\\b'+word).test(quoted));
 if(!present.length||present.length/words.length<share)return 'absent';
 if(NEGATION.test(' '+foldText(value)+' '))return 'supported';
 const folded=foldWithMap(located.text),hit=present.map(word=>folded.text.search(new RegExp('\\b'+word))).filter(i=>i>=0).sort((a,b)=>a-b)[0];
 const offset=(folded.map[hit]??0),inner={start:located.start+offset,end:located.end,text:located.text.slice(offset)};
 return negatedQuote(message,inner)?'negated':'supported';
}
export const quoteSupports=(message,quote,value,options)=>quoteSupport(message,quote,value,options)==='supported';
// Near-duplicate test for memory sections (token overlap, typography folded).
export function similarQuotes(a,b,threshold=0.8){
 const x=new Set(foldText(a).match(/[a-z0-9]+/g)||[]),y=new Set(foldText(b).match(/[a-z0-9]+/g)||[]);
 if(!x.size||!y.size)return foldText(a)===foldText(b);
 let common=0;for(const word of x)if(y.has(word))common++;
 return common/Math.max(x.size,y.size)>=threshold;
}
// The sentence of the message around a matched position, at most 240
// characters, so a safety quote always contains the words that triggered it.
export function sentenceAround(message,index,length=0,max=240){
 const text=String(message??'');if(!(index>=0))return text.slice(0,max);
 let start=index,end=index+length;
 while(start>0&&!/[.!?\n]/.test(text[start-1])&&index-start<120)start--;
 while(end<text.length&&!/[.!?\n]/.test(text[end])&&end-index-length<120)end++;
 if(end<text.length&&/[.!?]/.test(text[end]))end++;
 let out=text.slice(start,end).trim();
 if(out.length>max){const from=Math.max(0,index-start-Math.floor((max-length)/2));out=text.slice(start+from,start+from+max).trim()}
 return out;
}
// Cartography notes are written without a confirmation step: each one needs a
// substantial, non-negated quote, a target that was actually supplied to the
// writer, and never a lived experience the person answered "no" to (or an
// outcome they dismissed).
export function cartographyNoteCandidates(notes,message,map,targets){
 if(!Array.isArray(notes))return [];
 const negative=new Set((map?.explicit_negative||[]).map(x=>x?.id)),dismissed=new Set((map?.dismissed||[]).map(x=>x?.id));
 const allowed=Array.isArray(targets)?new Set(targets.map(t=>t.target+'|'+t.target_id)):null,out=[];
 for(const note of notes){
  if(!note||typeof note.quote!=='string'||note.quote.length>300)continue;
  if(note.target==='factor'&&negative.has(note.target_id)||note.target==='outcome'&&dismissed.has(note.target_id))continue;
  if(note.target!=='general'&&allowed&&!allowed.has(note.target+'|'+note.target_id))continue;
  const anchored=anchorQuote(message,note.quote);if(!anchored.ok)continue;
  out.push({...note,quote:anchored.text,text:'Vous dites : « '+anchored.text+' »'});
 }
 return out;
}
