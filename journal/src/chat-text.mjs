// Keep quoted speech and parenthetical asides inside their complete sentence.
// Quoted question marks are evidence, not questions addressed to the member.
export function responseSentences(content) {
 const parts=[],stack=[];let start=0,questions=0;
 const pairs={'«':'»','“':'”','‘':'’','"':'"','(':')','[':']'};
 for(let i=0;i<content.length;i++){
  const char=content[i],top=stack.at(-1);
  if(top===char){stack.pop();continue;}
  if(Object.hasOwn(pairs,char)){stack.push(pairs[char]);continue;}
  const quoted=stack.some(close=>['»','”','’','"'].includes(close));
  if(char==='?'&&!quoted&&content[i-1]!=='?')questions++;
  if(!stack.length&&((/[.!?…]/u.test(char)&&(/\s/u.test(content[i+1]||'')||i===content.length-1))||char==='\n'||char==='?'&&/\p{L}/u.test(content[i+1]||''))){
   parts.push({text:content.slice(start,i+1),questions});start=i+1;questions=0;
  }
 }
 if(start<content.length)parts.push({text:content.slice(start),questions});
 return parts;
}
export function limitQuestions(content,maximum=2){
 let used=0;
 return responseSentences(content).filter(sentence=>{
  if(used+sentence.questions>maximum)return false;
  used+=sentence.questions;return true;
 }).map(sentence=>sentence.text).join('').trim();
}
