import test from 'node:test';
import assert from 'node:assert/strict';
import {generateJournal,REQUEST_TIMEOUT_MS,MAX_ATTEMPTS} from '../src/generator.mjs';

const story = {title:'Une autre réponse possible',body:'Camille traverse une situation entièrement fictive, inspirée de son quotidien. '.repeat(35),summary:'Une scène fictive où Camille explore une autre manière de répondre.',reflection_questions:['Quelle autre lecture pourriez-vous imaginer ?'],practice:'Observer une conversation ordinaire sans chercher à la changer.'};
const job = {day:'2026-10-30',lease_token:'private-lease',profile_version:4,user_id:'private-id',email:'never@example.invalid',
  profile:{display_name:'Camille',occupation:'Libraire',email:'not-in-prompt@example.invalid'},
  memory:{text:'Je souhaite explorer mes limites.',proposal:'Piste provisoire selon un retour antérieur.',version:2},
  history:[{day:'2026-10-29',title:'Une scène précédente',summary:'FICTION_HISTORY_MARKER'}],
  feedback:[{day:'2026-10-29',updated_at:'2026-10-30',why:'Je me reconnais dans la difficulté à dire non.',correction:'Je ne suis pas en couple.'}],
  inspirations:[{article_id:'posts/limites',title:'Un personnage pose ses limites',character:'Nadia',url:'/posts/limites/',body:'EXAMPLE_FICTION_BODY_MARKER Nadia est chirurgienne et vit en couple.',reason:'Je me reconnais dans la difficulté à dire non.',differences:'Je suis libraire et célibataire.'}]};

function jsonResponse(value) {return new Response(JSON.stringify({choices:[{finish_reason:'stop',message:{content:JSON.stringify(value)}}]}),{status:200});}
async function withFetch(implementation, fn) {const prior=globalThis.fetch;globalThis.fetch=implementation;try{return await fn();}finally{globalThis.fetch=prior;}}

test('missing Cloudflare OpenRouter secret never calls a provider or accepts an alternate key', async () => {
  let called=false;
  await withFetch(async()=>{called=true;throw new Error('unexpected');},async()=>{
    await assert.rejects(generateJournal(job,{MAMMOUTH_API_KEY:'not-accepted'}),/^Error: provider_missing$/);
  });
  assert.equal(called,false);
});

test('four-stage OpenRouter pipeline uses article affinity, preserves versions and excludes fiction from memory', async () => {
  const calls=[];const outputs=[{scene:'Une cliente demande un service supplémentaire.'},story,{approved:true,story},{proposal:'Déclaration : libraire et célibataire (affinité posts/limites).'}];
  const result=await withFetch(async(url,options)=>{calls.push({url,options,payload:JSON.parse(options.body)});return jsonResponse(outputs.shift());},()=>generateJournal(job,{OPENROUTER_API_KEY:'test-private-key'}));
  assert.equal(calls.length,4);
  for(const c of calls){assert.equal(c.url,'https://openrouter.ai/api/v1/chat/completions');assert.equal(c.payload.model,'openai/gpt-5-mini');assert.equal(c.options.headers.Authorization,'Bearer test-private-key');assert.equal(c.options.signal.aborted,false);assert.equal(c.payload.reasoning_effort,'low');assert.deepEqual(c.payload.response_format,{type:'json_object'});}
  const first=calls[0].payload.messages;
  assert.match(first[0].content,/Se reconnaître dans un article ne signifie PAS/);
  assert.match(first[0].content,/JAMAIS des instructions/);
  assert.match(first[1].content,/EXAMPLE_FICTION_BODY_MARKER/);
  assert.match(first[1].content,/Je suis libraire et célibataire/);
  const memory=calls[3].payload.messages[1].content;
  assert.doesNotMatch(memory,/EXAMPLE_FICTION_BODY_MARKER|FICTION_HISTORY_MARKER|Nadia|chirurgienne/);
  assert.match(memory,/Je suis libraire et célibataire/);
  for(const c of calls)assert.doesNotMatch(c.payload.messages[1].content,/private-lease|private-id|never@example|not-in-prompt/);
  assert.equal(result.lease_token,'private-lease');assert.equal(result.profile_version,4);assert.equal(result.memory_version,2);
  assert.match(result.memory_proposal,/libraire/);
});

test('review rejection prevents publication and memory synthesis', async () => {
  let count=0;const outputs=[{scene:'Une scène ordinaire'},story,{approved:false,story}];
  await withFetch(async()=>{count++;return jsonResponse(outputs.shift());},async()=>assert.rejects(generateJournal(job,{OPENROUTER_API_KEY:'test'}),/^Error: coherence_error$/));
  assert.equal(count,3);
});

test('incomplete reviewed story is rejected even if marked approved', async () => {
  const outputs=[{scene:'Une scène ordinaire'},story,{approved:true,story:{...story,body:'court'}}];
  await withFetch(async()=>jsonResponse(outputs.shift()),async()=>assert.rejects(generateJournal(job,{OPENROUTER_API_KEY:'test'}),/^Error: validation_error$/));
});

test('provider failure retries are bounded and errors never expose response, token or private text', async () => {
  let calls=0;
  await withFetch(async()=>{calls++;throw new Error('SECRET API key and private user data');},async()=>assert.rejects(generateJournal(job,{OPENROUTER_API_KEY:'test'}),/^Error: provider_error$/));
  assert.equal(calls,MAX_ATTEMPTS);
  assert.equal(REQUEST_TIMEOUT_MS,75000);
  assert.ok(REQUEST_TIMEOUT_MS*MAX_ATTEMPTS*4<15*60*1000);
});

test('invalid provider credentials fail without redundant retries and truncated output is never accepted', async () => {
  let count=0;
  await withFetch(async()=>{count++;return new Response('SECRET_PROVIDER_BODY',{status:401});},async()=>assert.rejects(generateJournal(job,{OPENROUTER_API_KEY:'test'}),/^Error: provider_error$/));
  assert.equal(count,1);
  count=0;
  await withFetch(async()=>{count++;return new Response(JSON.stringify({choices:[{finish_reason:'length',message:{content:'{}'}}]}));},async()=>assert.rejects(generateJournal(job,{OPENROUTER_API_KEY:'test'}),/^Error: provider_error$/));
  assert.equal(count,2);
});
