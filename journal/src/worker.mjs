import { HTML } from './ui.mjs';
import { EXAMPLES } from './examples.mjs';

const encoder = new TextEncoder();
const COOKIE = '__Host-dte_session';
const REPOSITORY = 'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine';
const ISSUER = 'https://token.actions.githubusercontent.com';
const FIELDS = ['display_name','pronouns','age_range','occupation','daily_life','relationships','values','strengths','challenges','goals','boundaries'];
const now = () => Math.floor(Date.now()/1000);
const random = () => crypto.randomUUID();
const hex = b => [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,'0')).join('');
const digest = async s => hex(await crypto.subtle.digest('SHA-256',encoder.encode(s)));
class HTTPError extends Error { constructor(status,message) { super(message); this.status=status; } }
const error = (status,message) => { throw new HTTPError(status,message); };
const json = (value,status=200,headers={}) => new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json; charset=utf-8',...headers}});
const first = (env,sql,...params) => env.DB.prepare(sql).bind(...params).first();
const all = async (env,sql,...params) => (await env.DB.prepare(sql).bind(...params).all()).results;
const run = (env,sql,...params) => env.DB.prepare(sql).bind(...params).run();
const parse = (value,fallback={}) => { try {return JSON.parse(value);} catch {return fallback;} };
const text = (value,max=3000) => { if(typeof value!=='string'||value.length>max) error(400,'Texte invalide ou trop long.'); return value.trim(); };
async function body(request,max=50000) {
  if(!request.headers.get('content-type')?.startsWith('application/json')) error(415,'Un contenu JSON est requis.');
  if(Number(request.headers.get('content-length')||0)>max) error(413,'Contenu trop volumineux.');
  const raw=await request.text(); if(raw.length>max) error(413,'Contenu trop volumineux.');
  try {const v=JSON.parse(raw); if(!v||typeof v!=='object'||Array.isArray(v)) throw 0; return v;} catch {error(400,'Contenu JSON invalide.');}
}
export function constantEqual(a,b) {
  if(typeof a!=='string'||typeof b!=='string') return false;
  let mismatch=a.length^b.length; for(let i=0;i<Math.max(a.length,b.length);i++) mismatch|=(a.charCodeAt(i)||0)^(b.charCodeAt(i)||0); return mismatch===0;
}
export async function passwordHash(password,salt) {
  const key=await crypto.subtle.importKey('raw',encoder.encode(password),'PBKDF2',false,['deriveBits']);
  return hex(await crypto.subtle.deriveBits({name:'PBKDF2',hash:'SHA-256',salt:encoder.encode(salt),iterations:100000},key,256));
}
export function activeSubscription(subscription,at=now()) {return subscription?.status==='active' && Number(subscription.paid_until)>at;}
async function limit(env,key,max,seconds) {
  const k=await digest(key),t=now();
  const row=await first(env,`INSERT INTO rate_limits(key,count,expires_at) VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET count=CASE WHEN expires_at<=? THEN 1 ELSE count+1 END, expires_at=CASE WHEN expires_at<=? THEN ? ELSE expires_at END RETURNING count`,k,t+seconds,t,t,t+seconds);
  if(row.count>max) error(429,'Trop de tentatives. Réessayez plus tard.');
}
function originCheck(request,env) {
  const expected=env.PUBLIC_ORIGIN||new URL(request.url).origin;
  if(request.headers.get('origin')!==expected) error(403,'Origine de la requête refusée.');
}
async function userFor(request,env,required=true) {
  const token=request.headers.get('cookie')?.split(';').map(s=>s.trim()).find(s=>s.startsWith(COOKIE+'='))?.slice(COOKIE.length+1);
  const user=token?await first(env,'SELECT u.id,u.email,u.created_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?',await digest(token),now()):null;
  if(!user&&required) error(401,'Connectez-vous pour accéder à votre journal.'); return user;
}
async function session(env,user) {
  const token=random()+random(); await run(env,'INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)',await digest(token),user.id,now()+2592000);
  return json({user:{id:user.id,email:user.email}},200,{'Set-Cookie':`${COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000`});
}
const billingReady=env=>Boolean(env.STRIPE_SECRET_KEY&&env.STRIPE_PRICE_ID&&env.STRIPE_WEBHOOK_SECRET);
async function stripe(env,path,method='GET',data,idem) {
  if(!env.STRIPE_SECRET_KEY) error(503,'Le paiement sera disponible après sa configuration.');
  const headers={'Authorization':`Bearer ${env.STRIPE_SECRET_KEY}`,'Stripe-Version':'2024-06-20'};
  if(data) headers['Content-Type']='application/x-www-form-urlencoded'; if(idem) headers['Idempotency-Key']=idem;
  const response=await fetch('https://api.stripe.com/v1/'+path,{method,headers,body:data?new URLSearchParams(data):undefined});
  if(!response.ok) error(502,'Le service de paiement est momentanément indisponible.'); return response.json();
}
async function validPrice(env) {
  const price=await stripe(env,'prices/'+encodeURIComponent(env.STRIPE_PRICE_ID));
  if(!price.active||price.currency!=='eur'||price.unit_amount!==1000||price.recurring?.interval!=='month'||price.recurring?.interval_count!==1) error(503,'Le tarif mensuel doit être configuré à 10 euros.');
  return price;
}
async function ensureCustomer(env,user) {
  let sub=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
  if(sub?.customer_id) return sub.customer_id;
  const customer=await stripe(env,'customers','POST',{'email':user.email,'metadata[user_id]':user.id},'dte-customer-'+user.id);
  await run(env,'UPDATE subscriptions SET customer_id=? WHERE user_id=?',customer.id,user.id); return customer.id;
}
async function synchronizeSubscription(env,subscriptionId,eventCreated=0) {
  const sub=await stripe(env,'subscriptions/'+encodeURIComponent(subscriptionId));
  const customer=typeof sub.customer==='string'?sub.customer:sub.customer?.id;
  const local=await first(env,'SELECT * FROM subscriptions WHERE customer_id=?',customer||'');
  if(!local) return;
  const valid=sub.items?.data?.length===1&&sub.items.data[0].quantity===1&&sub.items.data[0].price?.id===env.STRIPE_PRICE_ID;
  // Re-fetching canonical state makes delayed events harmless. Never use client metadata as entitlement.
  await run(env,`UPDATE subscriptions SET subscription_id=?,status=?,paid_until=?,updated_at=? WHERE user_id=? AND updated_at<=? AND NOT (subscription_id=? AND status='canceled' AND ?!='canceled')`,sub.id,valid?sub.status:'invalid_price',valid?Number(sub.current_period_end||0):0,eventCreated,local.user_id,eventCreated,sub.id,sub.status);
}
export async function verifyStripeSignature(raw,header,secret,at=now()) {
  if(!secret||!header) return false;
  const parts=header.split(',').map(x=>x.split('=')); const timestamp=parts.find(x=>x[0]==='t')?.[1];
  if(!timestamp||!/^\d+$/.test(timestamp)||Math.abs(at-Number(timestamp))>300) return false;
  const key=await crypto.subtle.importKey('raw',encoder.encode(secret),{name:'HMAC',hash:'SHA-256'},false,['sign']);
  const signature=hex(await crypto.subtle.sign('HMAC',key,encoder.encode(timestamp+'.'+raw)));
  return parts.some(([k,v])=>k==='v1'&&constantEqual(signature,v));
}
async function webhook(request,env) {
  const raw=await request.text(); if(raw.length>1000000) error(413,'Requête trop volumineuse.');
  if(!await verifyStripeSignature(raw,request.headers.get('stripe-signature'),env.STRIPE_WEBHOOK_SECRET)) error(400,'Signature invalide.');
  const event=parse(raw,null); if(!event?.id||!event?.type) error(400,'Événement invalide.');
  if(await first(env,'SELECT id FROM webhook_events WHERE id=?',event.id)) return json({received:true});
  let sid;
  if(event.type.startsWith('customer.subscription.')) sid=event.data?.object?.id;
  if(['checkout.session.completed','checkout.session.async_payment_succeeded','invoice.paid','invoice.payment_failed'].includes(event.type)) sid=event.data?.object?.subscription;
  if(sid) await synchronizeSubscription(env,sid,Number(event.created)||0);
  await run(env,'INSERT OR IGNORE INTO webhook_events(id,processed_at) VALUES(?,?)',event.id,now());
  return json({received:true});
}
let jwksCache;
const decode64=s=>Uint8Array.from(atob(s.replace(/-/g,'+').replace(/_/g,'/')),c=>c.charCodeAt(0));
export function validateClaims(claims,repository=REPOSITORY,at=now()) {
  return claims.iss===ISSUER && claims.aud==='decode-ton-esprit-journal' && claims.repository===repository && claims.ref==='refs/heads/main' && claims.workflow_ref===`${repository}/.github/workflows/generate-articles.yml@refs/heads/main` && ['schedule','workflow_dispatch','push'].includes(claims.event_name) && Number(claims.exp)>at && Number(claims.iat)<=at+30 && Number(claims.nbf||0)<=at+30 && Number(claims.exp)-Number(claims.iat)<=900;
}
async function githubIdentity(request,env) {
  const token=request.headers.get('authorization')?.match(/^Bearer ([A-Za-z0-9._-]+)$/)?.[1];
  if(!token||token.length>15000) error(401,'Identité automatique requise.');
  const parts=token.split('.'); if(parts.length!==3) error(401,'Identité invalide.');
  let header,claims; try {header=JSON.parse(new TextDecoder().decode(decode64(parts[0]))); claims=JSON.parse(new TextDecoder().decode(decode64(parts[1])));} catch {error(401,'Identité invalide.');}
  if(header.alg!=='RS256'||!validateClaims(claims,env.GITHUB_REPOSITORY||REPOSITORY)) error(403,'Workflow non autorisé.');
  if(!jwksCache||jwksCache.until<now()||!jwksCache.keys.some(k=>k.kid===header.kid)) {
    const response=await fetch(ISSUER+'/.well-known/jwks',{signal:AbortSignal.timeout(10000)}); if(!response.ok) error(503,'Vérification indisponible.');
    jwksCache={keys:(await response.json()).keys,until:now()+3600};
  }
  const jwk=jwksCache.keys.find(k=>k.kid===header.kid); if(!jwk) error(401,'Clé de signature inconnue.');
  const key=await crypto.subtle.importKey('jwk',jwk,{name:'RSASSA-PKCS1-v1_5',hash:'SHA-256'},false,['verify']);
  if(!await crypto.subtle.verify('RSASSA-PKCS1-v1_5',key,decode64(parts[2]),encoder.encode(parts[0]+'.'+parts[1]))) error(401,'Signature invalide.');
  return claims;
}
const frenchDay=()=>new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Paris',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
const TARGET_LIKES=10;
const catalog=new Map(EXAMPLES.map(example=>[example.id,example]));
function decorateLikes(rows) {
  return rows.flatMap(row=>{
    const article=catalog.get(row.article_id);
    return article?[{...row,title:article.title,url:article.url,character:article.character,keywords:article.keywords||[],character_keywords:article.character_keywords||[],sujet:article.sujet||'',category:article.category||'',summary:article.summary||article.description||''}]:[];
  });
}
async function likesFor(env,userId) {return decorateLikes(await all(env,'SELECT article_id,reason,differences,created_at,updated_at FROM article_likes WHERE user_id=? ORDER BY created_at,article_id',userId));}
function readinessFor(likes) {return {likes_count:likes.length,explained_count:likes.filter(l=>l.reason.trim()).length,target:TARGET_LIKES,ready:likes.length>=TARGET_LIKES};}
async function likesResponse(env,userId) {
  // One SQLite read snapshot binds the collection and its version. A concurrent edit
  // can return a newer snapshot, but can never tag this older collection with it.
  const snapshot=await first(env,`SELECT p.version,(SELECT json_group_array(json_object('article_id',article_id,'reason',reason,'differences',differences,'created_at',created_at,'updated_at',updated_at)) FROM (SELECT article_id,reason,differences,created_at,updated_at FROM article_likes WHERE user_id=p.user_id ORDER BY created_at,article_id)) AS likes_json FROM profiles p WHERE p.user_id=?`,userId);
  if(!snapshot) error(401,'Compte introuvable.');
  const likes=decorateLikes(parse(snapshot.likes_json,[])),r=readinessFor(likes);
  return {likes,count:r.likes_count,explained_count:r.explained_count,target:TARGET_LIKES,source_version:snapshot.version};
}
async function portraitFor(env,userId) {
  const p=await first(env,'SELECT version,consent_ai FROM profiles WHERE user_id=?',userId),saved=await first(env,'SELECT * FROM user_portraits WHERE user_id=?',userId),sub=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',userId),likes=await likesFor(env,userId);
  const job=await first(env,"SELECT status,lease_until,attempts,provider_calls,created_at FROM journal_jobs WHERE user_id=? AND kind='portrait' AND source_version=? AND day=''",userId,p.version);
  let status='collecting';
  if(activeSubscription(sub)&&p.consent_ai&&likes.length>0) {
    status=saved?.analyzed_version===p.version?'ready':job?.status==='running'&&job.lease_until>now()?'updating':job&&(job.attempts>=3||job.provider_calls>=20)?'error':'queued';
  }
  const retry_after=status==='error'&&job?.created_at?Math.floor(Date.parse(job.created_at.replace(' ','T')+'Z')/1000)+86400:null;
  return {status,retry_after,source_version:p.version,analyzed_version:saved?.analyzed_version||0,updated_at:saved?.updated_at||null,portrait:saved?parse(saved.portrait,null):null};
}
async function storyInspirations(env,likes) {
  const selected=[...likes].sort((a,b)=>Number(Boolean(b.reason))-Number(Boolean(a.reason))||b.updated_at.localeCompare(a.updated_at)).slice(0,3);
  const result=[];
  for(const selectedLike of selected) {
    const source=await first(env,'SELECT body FROM public_examples WHERE id=?',selectedLike.article_id);
    if(!source?.body?.trim()) error(503,'catalog_missing');
    result.push({...selectedLike,body:source.body});
  }
  return result;
}
export async function claimJobs(env) {
  if(!env.OPENROUTER_API_KEY) return json({jobs:[],status:'provider_missing'});
  const day=frenchDay(),t=now();
  // A failed source version gets a fresh bounded budget once per 24-hour window.
  // This never reopens a completed portrait or an unexpired lease.
  await run(env,"UPDATE journal_jobs SET status='pending',attempts=0,provider_calls=0,lease_token=NULL,lease_until=0,error_code=NULL,created_at=CURRENT_TIMESTAMP WHERE kind='portrait' AND status!='complete' AND (attempts>=3 OR provider_calls>=20) AND lease_until<=? AND created_at<=datetime('now','-24 hours')",t);
  // A portrait refresh always precedes a story for the same source version.
  for(const kind of ['portrait','story']) {
    const dirty=kind==='portrait'?"(up.analyzed_version IS NULL OR up.analyzed_version!=p.version)":"up.analyzed_version=p.version AND NOT EXISTS(SELECT 1 FROM stories st WHERE st.user_id=u.id AND st.day=?)";
    const jobDay=kind==='portrait'?'':day;
    const params=[t,...(kind==='story'?[day]:[]),kind,jobDay,t];
    const candidates=await all(env,`SELECT u.id,p.version,m.version AS memory_version FROM users u JOIN profiles p ON p.user_id=u.id JOIN memories m ON m.user_id=u.id JOIN subscriptions s ON s.user_id=u.id LEFT JOIN user_portraits up ON up.user_id=u.id WHERE s.status='active' AND s.paid_until>? AND p.consent_ai=1 AND (SELECT COUNT(*) FROM article_likes l WHERE l.user_id=u.id)>=1 AND ${dirty} AND NOT EXISTS(SELECT 1 FROM journal_jobs j WHERE j.user_id=u.id AND j.kind=? AND j.source_version=p.version AND j.day=? AND (j.attempts>=3 OR j.provider_calls>=20 OR j.status='complete' OR j.lease_until>?)) ORDER BY u.created_at,u.id LIMIT 10`,...params);
    for(const candidate of candidates) {
      const likes=await likesFor(env,candidate.id);if(!likes.length) continue;
      let inspirations=[];
      if(kind==='story') {
        try {inspirations=await storyInspirations(env,likes);} catch {return json({jobs:[],status:'catalog_missing'});}
      }
      await run(env,'INSERT OR IGNORE INTO journal_jobs(id,user_id,kind,source_version,memory_version,day) VALUES(?,?,?,?,?,?)',random(),candidate.id,kind,candidate.version,candidate.memory_version,jobDay);
      const lease=random()+random();
      const job=await first(env,`UPDATE journal_jobs SET status='running',attempts=attempts+1,lease_token=?,lease_until=?,error_code=NULL WHERE user_id=? AND kind=? AND source_version=? AND day=? AND status!='complete' AND lease_until<=? AND attempts<3 AND provider_calls<20 AND EXISTS(SELECT 1 FROM profiles WHERE user_id=? AND version=? AND consent_ai=1) RETURNING id`,lease,t+1800,candidate.id,kind,candidate.version,jobDay,t,candidate.id,candidate.version);
      if(!job) continue;
      const memory=await first(env,'SELECT text,version FROM memories WHERE user_id=?',candidate.id),previous=await first(env,'SELECT portrait,analyzed_version FROM user_portraits WHERE user_id=?',candidate.id);
      const feedback=await all(env,'SELECT f.story_id,s.day,s.title AS fictional_source_title,s.summary AS fictional_source_summary,f.resonance,f.feelings,f.why,f.correction,f.updated_at FROM feedback f JOIN stories s ON s.id=f.story_id WHERE f.user_id=? ORDER BY f.updated_at DESC,s.day DESC LIMIT 30',candidate.id);
      const result={id:job.id,kind,user_id:candidate.id,day:jobDay||day,lease_token:lease,source_version:candidate.version,profile_version:candidate.version,memory_version:candidate.memory_version,memory,likes,feedback:feedback.reverse(),portrait_previous:previous?parse(previous.portrait,null):null,portrait_previous_version:previous?.analyzed_version||0};
      if(kind==='story') {result.portrait=result.portrait_previous;result.inspirations=inspirations;result.history=(await all(env,'SELECT day,title,summary FROM stories WHERE user_id=? ORDER BY day DESC LIMIT 20',candidate.id)).reverse();}
      return json({jobs:[result]});
    }
  }
  return json({jobs:[]});
}
// Kept as a harmless operational endpoint: GitHub owns orchestration and retries.
export async function runGenerationBatch() {return {status:'workflow-managed',claimed:0,completed:0,failed:0};}
async function validJob(env,id,lease,sourceVersion,memoryVersion) {
  const job=await first(env,'SELECT * FROM journal_jobs WHERE id=?',id);
  if(!job||!constantEqual(job.lease_token,lease)||job.status!=='running'||job.lease_until<=now()) error(409,'Bail expiré ou inconnu.');
  const p=await first(env,'SELECT version,consent_ai FROM profiles WHERE user_id=?',job.user_id),m=await first(env,'SELECT version FROM memories WHERE user_id=?',job.user_id),s=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',job.user_id);
  if(!p?.consent_ai||!activeSubscription(s)||!(await likesFor(env,job.user_id)).length) error(409,'Génération annulée.');
  if(p.version!==job.source_version||m.version!==job.memory_version||(sourceVersion!==undefined&&sourceVersion!==p.version)||(memoryVersion!==undefined&&memoryVersion!==m.version)) {
    await run(env,"UPDATE journal_jobs SET status='obsolete',lease_until=0 WHERE id=? AND lease_token=?",id,lease);error(409,'Le contexte a changé.');
  }
  if(job.kind==='story') {const portrait=await first(env,'SELECT analyzed_version FROM user_portraits WHERE user_id=?',job.user_id);if(portrait?.analyzed_version!==p.version) error(409,'Le portrait doit être actualisé.');}
  return job;
}
const jobGuard=`EXISTS(SELECT 1 FROM journal_jobs j JOIN profiles p ON p.user_id=j.user_id JOIN memories m ON m.user_id=j.user_id JOIN subscriptions s ON s.user_id=j.user_id WHERE j.id=? AND j.status='running' AND j.lease_token=? AND j.lease_until>? AND p.version=j.source_version AND m.version=j.memory_version AND p.consent_ai=1 AND s.status='active' AND s.paid_until>? AND (SELECT COUNT(*) FROM article_likes l WHERE l.user_id=j.user_id)>=1)`;
function jobArgs(id,lease) {const t=now();return [id,lease,t,t];}
export function validatePortrait(value,likedIds) {
  if(!value||typeof value!=='object') error(400,'Portrait invalide.');
  const summary=text(value.summary,6000);if(!summary) error(400,'Résumé du portrait manquant.');
  if(!Array.isArray(value.axes)||value.axes.length<1||value.axes.length>12) error(400,'Axes du portrait invalides.');
  const axes=value.axes.map(axis=>{
    const label=text(axis.label,160),why=text(axis.why,2400);
    if(!label||!why||typeof axis.confidence!=='number'||!Number.isFinite(axis.confidence)||axis.confidence<0||axis.confidence>1||!Array.isArray(axis.evidence)||!axis.evidence.length||axis.evidence.length>30||axis.evidence.some(id=>!likedIds.has(id))) error(400,'Sources du portrait invalides.');
    return {label,why,confidence:axis.confidence,evidence:[...new Set(axis.evidence)]};
  });
  const list=(values)=>{if(!Array.isArray(values)||values.length>20) error(400,'Liste du portrait invalide.');return values.map(value=>text(value,1200));};
  const fictional_character={name:text(value.fictional_character?.name,100),description:text(value.fictional_character?.description,5000)};
  if(!fictional_character.name||!fictional_character.description) error(400,'Personnage fictif manquant.');
  return {summary,axes,strengths:list(value.strengths),uncertainties:list(value.uncertainties),fictional_character};
}
export async function completeJob(env,id,data) {
  const sourceVersion=data.source_version??data.profile_version;
  if(!Number.isInteger(sourceVersion)||!Number.isInteger(data.memory_version)) error(400,'Versions du contexte requises.');
  const job=await validJob(env,id,data.lease_token,sourceVersion,data.memory_version),args=jobArgs(id,data.lease_token);
  if(job.kind==='portrait') {
    const likedIds=new Set((await likesFor(env,job.user_id)).map(like=>like.article_id));
    for(const feedback of await all(env,'SELECT story_id FROM feedback WHERE user_id=?',job.user_id)) likedIds.add('story:'+feedback.story_id);
    const saved=await first(env,'SELECT portrait FROM user_portraits WHERE user_id=?',job.user_id);
    const previousName=saved?parse(saved.portrait,null)?.fictional_character?.name:null;
    const incoming=previousName?{...data.portrait,fictional_character:{...data.portrait?.fictional_character,name:previousName}}:data.portrait;
    const portrait=validatePortrait(incoming,likedIds);
    const results=await env.DB.batch([
      env.DB.prepare(`INSERT INTO user_portraits(user_id,portrait,analyzed_version) SELECT ?,?,? WHERE ${jobGuard} ON CONFLICT(user_id) DO UPDATE SET portrait=excluded.portrait,analyzed_version=excluded.analyzed_version,updated_at=CURRENT_TIMESTAMP WHERE user_portraits.analyzed_version<excluded.analyzed_version`).bind(job.user_id,JSON.stringify(portrait),job.source_version,...args),
      env.DB.prepare("UPDATE journal_jobs SET status='complete',lease_until=0 WHERE id=? AND lease_token=? AND EXISTS(SELECT 1 FROM user_portraits p WHERE p.user_id=journal_jobs.user_id AND p.analyzed_version=journal_jobs.source_version)").bind(id,data.lease_token)
    ]);
    if(!results[0].meta.changes) error(409,'Le contexte a changé ou ce portrait existe déjà.');
    return json({ok:true,kind:'portrait',source_version:job.source_version,analyzed_version:job.source_version});
  }
  const title=text(data.title,160),content=text(data.body,60000),summary=text(data.summary,3000),practice=text(data.practice||'',3000);
  if(!title||content.length<500||!summary) error(400,'Histoire incomplète.');
  if(!Array.isArray(data.reflection_questions)||data.reflection_questions.length<1||data.reflection_questions.length>5) error(400,'Questions invalides.');
  const questions=data.reflection_questions.map(q=>text(q,500));
  const results=await env.DB.batch([
    env.DB.prepare(`INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice) SELECT ?,?,?,?,?,?,?,? WHERE ${jobGuard} AND EXISTS(SELECT 1 FROM user_portraits p WHERE p.user_id=? AND p.analyzed_version=?) ON CONFLICT(user_id,day) DO NOTHING`).bind(id,job.user_id,job.day,title,content,summary,JSON.stringify(questions),practice,...args,job.user_id,job.source_version),
    env.DB.prepare("UPDATE journal_jobs SET status='complete',lease_until=0 WHERE id=? AND lease_token=? AND EXISTS(SELECT 1 FROM stories WHERE id=?)").bind(id,data.lease_token,id)
  ]);
  if(!results[0].meta.changes) error(409,'Publication déjà effectuée ou contexte modifié.');
  // A fictional episode never updates a person's portrait, declared memory or source version.
  return json({ok:true,kind:'story',story_id:id});
}
export async function proxyJobLLM(env,id,data) {
  if(!env.OPENROUTER_API_KEY) error(503,'provider_missing');
  const job=await validJob(env,id,data.lease_token,data.source_version??data.profile_version,data.memory_version);
  const allowed=[env.OPENROUTER_MODEL||'openai/gpt-5-mini',env.OPENROUTER_WRITER_MODEL,env.OPENROUTER_ANALYST_MODEL].filter(Boolean);
  const model=data.model||(data.purpose==='writer'?env.OPENROUTER_WRITER_MODEL:data.purpose==='analyst'?env.OPENROUTER_ANALYST_MODEL:null)||allowed[0];if(!allowed.includes(model)) error(400,'Modèle non autorisé.');
  const system=text(data.system_prompt,50000),prompt=text(data.user_prompt,900000),purpose=text(data.purpose||'generation',80);
  if(!system||!prompt||!purpose) error(400,'Prompts requis.');
  const maxTokens=data.max_tokens??6000;if(!Number.isInteger(maxTokens)||maxTokens<128||maxTokens>8000) error(400,'Limite de tokens invalide.');
  const counter=await first(env,`UPDATE journal_jobs SET provider_calls=provider_calls+1 WHERE id=? AND provider_calls<20 AND ${jobGuard} RETURNING provider_calls`,id,...jobArgs(id,data.lease_token));
  if(!counter) error(429,'Limite de génération atteinte ou contexte expiré.');
  let response;
  try {response=await fetch('https://openrouter.ai/api/v1/chat/completions',{method:'POST',headers:{'Authorization':`Bearer ${env.OPENROUTER_API_KEY}`,'Content-Type':'application/json','HTTP-Referer':'https://decodetonesprit.fr','X-Title':'Decode ton esprit'},body:JSON.stringify({model,messages:[{role:'system',content:system},{role:'user',content:prompt}],temperature:0.75,reasoning:{effort:'low'},max_tokens:maxTokens}),signal:AbortSignal.timeout(75000)});} catch {error(502,'provider_error');}
  if(!response.ok) error(502,'provider_error');
  let result;try {result=await response.json();} catch {error(502,'provider_error');}
  const content=result?.choices?.[0]?.message?.content;
  if(typeof content!=='string'||!content.trim()||content.length>180000||result.choices[0].finish_reason==='length') error(502,'provider_error');
  return json({content});
}
async function internal(request,env,path) {
  await githubIdentity(request,env);
  if(request.method!=='POST') error(405,'Méthode non autorisée.');
  if(path==='/api/internal/jobs/run') return json(await runGenerationBatch());
  if(path==='/api/internal/jobs/claim') return claimJobs(env);
  const match=path.match(/^\/api\/internal\/jobs\/([\w-]+)\/(complete|fail|llm)$/);if(!match) error(404,'Route inconnue.');
  const data=await body(request,1000000);
  if(match[2]==='llm') return proxyJobLLM(env,match[1],data);
  if(match[2]==='complete') return completeJob(env,match[1],data);
  const job=await first(env,'SELECT attempts FROM journal_jobs WHERE id=? AND lease_token=? AND status=\'running\'',match[1],text(data.lease_token,100));
  if(!job) error(409,'Bail expiré ou inconnu.');
  const allowed=['provider_error','validation_error','coherence_error','unexpected_error','context_changed'];
  await run(env,"UPDATE journal_jobs SET status='pending',lease_until=?,error_code=? WHERE id=? AND status='running' AND lease_token=?",now()+Math.min(3600,300*2**(job.attempts-1)),allowed.includes(data.reason)?data.reason:'unexpected_error',match[1],data.lease_token);
  return json({ok:true});
}
async function routes(request,env) {
  const url=new URL(request.url),path=url.pathname,method=request.method;
  if(method==='GET'&&path==='/') return Response.redirect('https://decodetonesprit.fr/',302);
  if(method==='GET'&&['/journal','/journal/'].includes(path)) return new Response(HTML,{headers:{'Content-Type':'text/html; charset=utf-8'}});
  if(method==='GET'&&path==='/api/examples') return json({examples:EXAMPLES.map(({id,title,url,character,description,keywords,character_keywords,sujet,category})=>({id,title,url,character,description,keywords:keywords||[],character_keywords:character_keywords||[],sujet:sujet||'',category:category||''}))});
  if(method==='GET'&&path==='/api/health') return json({ok:true,billing:billingReady(env),ai:Boolean(env.OPENROUTER_API_KEY)});
  if(method==='GET'&&path==='/robots.txt') return new Response('User-agent: *\nDisallow: /\n');
  if(path==='/api/billing/webhook'&&method==='POST') return webhook(request,env);
  if(path.startsWith('/api/internal/')) return internal(request,env,path);
  if(!['GET','HEAD','OPTIONS'].includes(method)) originCheck(request,env);
  if(path==='/api/me'&&method==='GET') {
    const user=await userFor(request,env,false); if(!user) return json({user:null,configuration:{billing:billingReady(env),ai:Boolean(env.OPENROUTER_API_KEY)}});
    const p=await first(env,'SELECT * FROM profiles WHERE user_id=?',user.id),s=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id),m=await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id);
    const readiness=readinessFor(await likesFor(env,user.id)),inspirations_count=readiness.likes_count;
    return json({user,inspirations_count,readiness,consent_ai:Boolean(p.consent_ai),source_version:p.version,profile:{...parse(p.data),consent_ai:Boolean(p.consent_ai)},subscription:{status:s.status,active:activeSubscription(s),paid_until:s.paid_until,manageable:Boolean(s.customer_id)},memory:m,configuration:{billing:billingReady(env),ai:Boolean(env.OPENROUTER_API_KEY)}});
  }
  if(['/api/auth/register','/api/auth/login'].includes(path)&&method==='POST') {
    const data=await body(request,3000),email=text(data.email,254).toLowerCase(),password=data.password;
    if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)||typeof password!=='string'||password.length<12||password.length>256) error(400,'Saisissez une adresse valide et un mot de passe de 12 caractères minimum.');
    await limit(env,'auth-ip:'+request.headers.get('CF-Connecting-IP'),30,900); await limit(env,'auth-email:'+email,12,900);
    if(path.endsWith('register')) {
      if(await first(env,'SELECT id FROM users WHERE email=?',email)) error(400,'Création impossible avec ces identifiants. Essayez de vous connecter.');
      const id=random(),salt=random(),hash=await passwordHash(password,salt);
      await env.DB.batch([env.DB.prepare('INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,?,?)').bind(id,email,hash,salt),env.DB.prepare('INSERT INTO profiles(user_id) VALUES(?)').bind(id),env.DB.prepare('INSERT INTO memories(user_id) VALUES(?)').bind(id),env.DB.prepare('INSERT INTO subscriptions(user_id) VALUES(?)').bind(id)]);
      return session(env,{id,email});
    }
    const user=await first(env,'SELECT * FROM users WHERE email=?',email);
    const hash=await passwordHash(password,user?.password_salt||'no-account-constant-salt');
    if(!user||!constantEqual(hash,user.password_hash)) error(401,'Adresse ou mot de passe incorrect.');
    return session(env,user);
  }
  const user=await userFor(request,env);
  if(path==='/api/auth/logout'&&method==='POST') {
    const token=request.headers.get('cookie')?.split(';').map(s=>s.trim()).find(s=>s.startsWith(COOKIE+'='))?.slice(COOKIE.length+1);
    if(token) await run(env,'DELETE FROM sessions WHERE token_hash=?',await digest(token));
    return json({ok:true},200,{'Set-Cookie':`${COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`});
  }
  if(path==='/api/profile'&&method==='PUT') {
    const data=await body(request,30000),profile={}; for(const field of FIELDS) profile[field]=text(data[field]??'',field==='display_name'?80:2000);
    if(typeof data.consent_ai!=='boolean') error(400,'Indiquez votre choix concernant la personnalisation IA.');
    await run(env,'UPDATE profiles SET data=?,consent_ai=?,version=version+1 WHERE user_id=?',JSON.stringify(profile),Number(data.consent_ai),user.id);
    return json({ok:true,profile:{...profile,consent_ai:data.consent_ai}});
  }
  if(path==='/api/consent'&&method==='PUT') {
    const data=await body(request,1000);if(typeof data.consent_ai!=='boolean') error(400,'Indiquez votre choix concernant la personnalisation IA.');
    await run(env,'UPDATE profiles SET consent_ai=?,version=version+1 WHERE user_id=? AND consent_ai!=?',Number(data.consent_ai),user.id,Number(data.consent_ai));
    const p=await first(env,'SELECT version,consent_ai FROM profiles WHERE user_id=?',user.id);
    return json({ok:true,consent_ai:Boolean(p.consent_ai),source_version:p.version});
  }
  if(path==='/api/portrait'&&method==='GET') return json(await portraitFor(env,user.id));
  if(path==='/api/likes'&&method==='GET') return json(await likesResponse(env,user.id));
  const likeMatch=path.match(/^\/api\/likes\/(.+)$/);
  if(likeMatch&&method==='PUT') {
    let articleId;try {articleId=decodeURIComponent(likeMatch[1]);} catch {error(400,'Identifiant d’article invalide.');}
    if(!catalog.has(articleId)) error(404,'Article introuvable.');
    const data=await body(request,6000);if(typeof data.liked!=='boolean') error(400,'Indiquez votre choix pour cet article.');
    if(data.reason!==undefined) text(data.reason,4000);
    if(data.liked) {
      if(data.reason===undefined) await run(env,'INSERT OR IGNORE INTO article_likes(user_id,article_id) VALUES(?,?)',user.id,articleId);
      else await run(env,`INSERT INTO article_likes(user_id,article_id,reason) VALUES(?,?,?) ON CONFLICT(user_id,article_id) DO UPDATE SET reason=excluded.reason,updated_at=CURRENT_TIMESTAMP WHERE article_likes.reason!=excluded.reason`,user.id,articleId,text(data.reason,4000));
    } else await run(env,'DELETE FROM article_likes WHERE user_id=? AND article_id=?',user.id,articleId);
    return json({ok:true,...await likesResponse(env,user.id)});
  }
  if(path==='/api/inspirations'&&method==='GET') return json({selections:(await likesFor(env,user.id)).map(({article_id,reason,differences})=>({article_id,reason,differences}))});
  if(path==='/api/inspirations'&&method==='PUT') {
    const data=await body(request,1000000);if(!Array.isArray(data.selections)) error(400,'Liste d’articles invalide.');
    const ids=new Set();
    const selections=data.selections.map(selection=>{
      if(!selection||!catalog.has(selection.article_id)||ids.has(selection.article_id)) error(400,'Article inconnu ou sélection en double.');
      ids.add(selection.article_id);return {article_id:selection.article_id,reason:text(selection.reason||'',4000),differences:text(selection.differences||'',4000)};
    });
    const serialized=JSON.stringify(selections);
    await env.DB.batch([
      env.DB.prepare("DELETE FROM article_likes WHERE user_id=? AND article_id NOT IN(SELECT json_extract(value,'$.article_id') FROM json_each(?))").bind(user.id,serialized),
      env.DB.prepare("INSERT INTO article_likes(user_id,article_id,reason,differences) SELECT ?,json_extract(value,'$.article_id'),json_extract(value,'$.reason'),json_extract(value,'$.differences') FROM json_each(?) WHERE true ON CONFLICT(user_id,article_id) DO UPDATE SET reason=excluded.reason,differences=excluded.differences,updated_at=CURRENT_TIMESTAMP WHERE article_likes.reason!=excluded.reason OR article_likes.differences!=excluded.differences").bind(user.id,serialized)
    ]);
    return json({ok:true,selections,...await likesResponse(env,user.id)});
  }
  if(path==='/api/memory'&&method==='GET') return json(await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id));
  if(path==='/api/memory'&&method==='PUT') {
    const data=await body(request,15000); const value=text(data.text,10000);
    await env.DB.batch([
      env.DB.prepare('UPDATE profiles SET version=version+1 WHERE user_id=? AND EXISTS(SELECT 1 FROM memories WHERE user_id=? AND text!=?)').bind(user.id,user.id,value),
      env.DB.prepare("UPDATE memories SET text=?,proposal='',version=version+1 WHERE user_id=? AND text!=?").bind(value,user.id,value)
    ]);
    return json({ok:true,source_version:(await first(env,'SELECT version FROM profiles WHERE user_id=?',user.id)).version});
  }
  if(path==='/api/stories'&&method==='GET') return json({stories:await all(env,'SELECT id,day,title,summary,created_at FROM stories WHERE user_id=? ORDER BY day DESC LIMIT 366',user.id)});
  const storyMatch=path.match(/^\/api\/stories\/([\w-]+)(\/feedback)?$/);
  if(storyMatch) {
    const story=await first(env,'SELECT * FROM stories WHERE id=? AND user_id=?',storyMatch[1],user.id); if(!story) error(404,'Histoire introuvable.');
    if(!storyMatch[2]&&method==='GET') return json({story:{...story,reflection_questions:parse(story.reflection_questions,[])},feedback:await first(env,'SELECT resonance,feelings,why,correction FROM feedback WHERE story_id=? AND user_id=?',story.id,user.id)});
    if(storyMatch[2]&&method==='PUT') {
      const data=await body(request,15000); if(!Number.isInteger(data.resonance)||data.resonance<1||data.resonance>5) error(400,'Choisissez une note de 1 à 5.');
      const feelings=text(data.feelings||'',3000),why=text(data.why||'',4000),correction=text(data.correction||'',4000);
      await env.DB.batch([env.DB.prepare('INSERT INTO feedback(story_id,user_id,resonance,feelings,why,correction) VALUES(?,?,?,?,?,?) ON CONFLICT(story_id) DO UPDATE SET resonance=excluded.resonance,feelings=excluded.feelings,why=excluded.why,correction=excluded.correction,updated_at=CURRENT_TIMESTAMP').bind(story.id,user.id,data.resonance,feelings,why,correction),env.DB.prepare('UPDATE profiles SET version=version+1 WHERE user_id=?').bind(user.id)]);
      return json({ok:true,source_version:(await first(env,'SELECT version FROM profiles WHERE user_id=?',user.id)).version});
    }
  }
  if(path==='/api/billing/checkout'&&method==='POST') {
    if(!billingReady(env)||!env.OPENROUTER_API_KEY) error(503,'Les abonnements ne sont pas encore ouverts.');
    const profile=await first(env,'SELECT consent_ai FROM profiles WHERE user_id=?',user.id);
    if(!profile?.consent_ai) error(400,'Autorisez la personnalisation avant de vous abonner.');
    if((await likesFor(env,user.id)).length<TARGET_LIKES) error(400,'Aimez au moins dix articles avant de vous abonner.');
    await limit(env,'checkout:'+user.id,5,3600); await validPrice(env);
    const current=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
    if(current.subscription_id&&['active','trialing','past_due','unpaid','incomplete'].includes(current.status)) error(409,'Un abonnement existe déjà. Gérez-le depuis votre espace abonnement.');
    const customer=await ensureCustomer(env,user),origin=env.PUBLIC_ORIGIN||url.origin;
    let state=await first(env,'SELECT checkout_attempt,checkout_session_id FROM subscriptions WHERE user_id=?',user.id);
    if(state.checkout_session_id) {
      const previous=await stripe(env,'checkout/sessions/'+encodeURIComponent(state.checkout_session_id));
      if(previous.status==='open'&&previous.url) return json({url:previous.url});
      if(previous.status==='complete') {
        if(previous.subscription) await synchronizeSubscription(env,typeof previous.subscription==='string'?previous.subscription:previous.subscription.id,now());
        const refreshed=await first(env,'SELECT status FROM subscriptions WHERE user_id=?',user.id);
        if(!['canceled','incomplete_expired'].includes(refreshed.status)) error(409,'Un paiement existe déjà. Gérez votre abonnement depuis votre espace abonnement.');
      }
      // Compare-and-swap prevents two requests from creating different attempts after expiry.
      await run(env,'UPDATE subscriptions SET checkout_attempt=?,checkout_session_id=NULL WHERE user_id=? AND checkout_session_id=?',random(),user.id,state.checkout_session_id);
    }
    await run(env,'UPDATE subscriptions SET checkout_attempt=COALESCE(checkout_attempt,?) WHERE user_id=?',random(),user.id);
    state=await first(env,'SELECT checkout_attempt,checkout_session_id FROM subscriptions WHERE user_id=?',user.id);
    // Reconcile remote subscriptions even if a webhook or the previous local write was lost.
    const existing=await stripe(env,'subscriptions?customer='+encodeURIComponent(customer)+'&status=all&limit=100');
    const live=existing.data?.find(s=>!['canceled','incomplete_expired'].includes(s.status));
    if(live) {await synchronizeSubscription(env,live.id,now()); error(409,'Un abonnement existe déjà. Gérez-le depuis votre espace abonnement.');}
    const sessions=await stripe(env,'checkout/sessions?customer='+encodeURIComponent(customer)+'&status=open&limit=100');
    const open=sessions.data?.find(s=>s.mode==='subscription'&&s.url);
    if(open) {await run(env,'UPDATE subscriptions SET checkout_session_id=? WHERE user_id=? AND checkout_attempt=?',open.id,user.id,state.checkout_attempt);return json({url:open.url});}
    const result=await stripe(env,'checkout/sessions','POST',{'mode':'subscription','customer':customer,'line_items[0][price]':env.STRIPE_PRICE_ID,'line_items[0][quantity]':'1','success_url':origin+'/journal/?payment=success','cancel_url':origin+'/journal/?payment=cancelled','client_reference_id':user.id,'subscription_data[metadata][user_id]':user.id},'dte-checkout-'+user.id+'-'+state.checkout_attempt);
    await run(env,'UPDATE subscriptions SET checkout_session_id=? WHERE user_id=? AND checkout_attempt=?',result.id,user.id,state.checkout_attempt);
    return json({url:result.url});
  }
  if(path==='/api/billing/portal'&&method==='POST') {
    const sub=await first(env,'SELECT customer_id FROM subscriptions WHERE user_id=?',user.id); if(!sub?.customer_id) error(400,'Aucun abonnement à gérer.');
    const result=await stripe(env,'billing_portal/sessions','POST',{'customer':sub.customer_id,'return_url':(env.PUBLIC_ORIGIN||url.origin)+'/journal/'}); return json({url:result.url});
  }
  if(path==='/api/export'&&method==='GET') {
    const profile=await first(env,'SELECT data,consent_ai FROM profiles WHERE user_id=?',user.id),memory=await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id);
    return json({user,profile:{...parse(profile.data),consent_ai:Boolean(profile.consent_ai)},memory,likes:await likesFor(env,user.id),portrait:await portraitFor(env,user.id),selections:await all(env,'SELECT article_id,reason,differences FROM article_likes WHERE user_id=? ORDER BY article_id',user.id),stories:await all(env,'SELECT day,title,body,summary,reflection_questions,practice FROM stories WHERE user_id=? ORDER BY day',user.id),feedback:await all(env,'SELECT story_id,resonance,feelings,why,correction,updated_at FROM feedback WHERE user_id=?',user.id)},200,{'Content-Disposition':'attachment; filename="mon-journal-decode-ton-esprit.json"'});
  }
  if(path==='/api/account'&&method==='DELETE') {
    const data=await body(request,1000),record=await first(env,'SELECT * FROM users WHERE id=?',user.id);
    await limit(env,'delete:'+user.id,5,3600);
    if(typeof data.password!=='string'||data.password.length>256||!constantEqual(await passwordHash(data.password,record.password_salt),record.password_hash)) error(401,'Mot de passe incorrect.');
    const sub=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
    if(sub?.customer_id) {
      const customer=await stripe(env,'customers/'+encodeURIComponent(sub.customer_id));
      if(!customer.deleted) await stripe(env,'customers/'+encodeURIComponent(sub.customer_id),'DELETE');
    } else if(sub?.subscription_id&&!['canceled','incomplete_expired'].includes(sub.status)) await stripe(env,'subscriptions/'+encodeURIComponent(sub.subscription_id),'DELETE');
    await run(env,'DELETE FROM users WHERE id=?',user.id);
    return json({ok:true},200,{'Set-Cookie':`${COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`});
  }
  error(404,'Route inconnue.');
}
export default {
  async scheduled(controller,env,ctx) {
    const work=runGenerationBatch(env);
    if(ctx?.waitUntil) ctx.waitUntil(work);
    return work;
  },
  async fetch(request,env) {
    let response;
    try {response=await routes(request,env);} catch(e) {response=json({error:e instanceof HTTPError?e.message:'Une erreur temporaire est survenue. Réessayez plus tard.'},e instanceof HTTPError?e.status:500);}
    const headers=new Headers(response.headers);
    headers.set('Cache-Control','no-store'); headers.set('X-Content-Type-Options','nosniff'); headers.set('Referrer-Policy','no-referrer'); headers.set('X-Frame-Options','DENY'); headers.set('Permissions-Policy','camera=(), microphone=(), geolocation=()');
    headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    return new Response(response.body,{status:response.status,headers});
  }
};
