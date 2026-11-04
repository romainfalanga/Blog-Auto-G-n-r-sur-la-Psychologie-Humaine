import { isEntitled, accessUntil, platformAccess } from './entitlements.mjs';
import { createChatBackend } from './chat-backend.mjs';
import { HTML } from './ui.mjs';
import { EXAMPLES } from './examples.mjs';
import { createRecommender } from './recommendations.mjs';
import { createPersonalBackend, hasPersonalizationSources } from './personal-backend.mjs';

const encoder = new TextEncoder();
const COOKIE = '__Host-dte_session';
const REPOSITORY = 'romainfalanga/Blog-Auto-G-n-r-sur-la-Psychologie-Humaine';
const ISSUER = 'https://token.actions.githubusercontent.com';
const FIELDS = ['display_name','pronouns','age_range','occupation','daily_life','relationships','values','strengths','challenges','goals','boundaries'];
const now = () => Math.floor(Date.now()/1000);
const random = () => crypto.randomUUID();
const hex = b => [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,'0')).join('');
const digest = async s => hex(await crypto.subtle.digest('SHA-256',encoder.encode(s)));
class HTTPError extends Error { constructor(status,message,code) { super(message); this.status=status; this.code=code; } }
const error = (status,message,code) => { throw new HTTPError(status,message,code); };
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
export function activeSubscription(subscription,at=now()) {return isEntitled(subscription,at);}
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
  if(!price.active||price.currency!=='eur'||price.unit_amount!==2900||price.recurring?.interval!=='month'||price.recurring?.interval_count!==1) error(503,'Le tarif mensuel doit être configuré à 29 euros.');
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
  eventCreated=now(); // Canonical state was fetched now, regardless of event delivery order.
  const customer=typeof sub.customer==='string'?sub.customer:sub.customer?.id;
  const local=await first(env,'SELECT * FROM subscriptions WHERE customer_id=?',customer||'');
  if(!local) return sub;
  const valid=sub.items?.data?.length===1&&sub.items.data[0].quantity===1&&sub.items.data[0].price?.id===env.STRIPE_PRICE_ID;
  // Re-fetching canonical state makes delayed events harmless. Never use client metadata as entitlement.
  await run(env,`UPDATE subscriptions SET subscription_id=?,status=?,paid_until=?,trial_end=?,trial_used_at=CASE WHEN trial_used_at=0 AND ?>0 THEN ? ELSE trial_used_at END,cancel_at_period_end=?,updated_at=? WHERE user_id=? AND updated_at<=? AND NOT (subscription_id=? AND status='canceled' AND ?!='canceled')`,sub.id,valid?sub.status:'invalid_price',valid?Number(sub.current_period_end||0):0,valid?Number(sub.trial_end||0):0,Number(sub.trial_start||sub.trial_end||0),Number(sub.trial_start||sub.trial_end||0),sub.cancel_at_period_end?1:0,eventCreated,local.user_id,eventCreated,sub.id,sub.status);
  return sub;
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
const recommender=createRecommender(EXAMPLES);
const FREE_PORTRAIT_STARTS=3;
const FREE_PORTRAIT_INTERVAL=600;
function decorateLikes(rows) {
  return rows.flatMap(row=>{
    const article=catalog.get(row.article_id);
    return article?[{...row,title:article.title,url:article.url,character:article.character,keywords:article.keywords||[],character_keywords:article.character_keywords||[],sujet:article.sujet||'',category:article.category||'',summary:article.summary||article.description||''}]:[];
  });
}
async function likesFor(env,userId) {return decorateLikes(await all(env,'SELECT article_id,reason,differences,created_at,updated_at FROM article_likes WHERE user_id=? ORDER BY created_at,article_id',userId));}
function readinessFor(likes) {return {likes_count:likes.length,explained_count:likes.filter(l=>l.reason.trim()||l.differences.trim()).length,target:TARGET_LIKES,ready:likes.length>=TARGET_LIKES};}
async function likesResponse(env,userId) {
  // One SQLite read snapshot binds the collection and its version. A concurrent edit
  // can return a newer snapshot, but can never tag this older collection with it.
  const snapshot=await first(env,`SELECT p.version,(SELECT json_group_array(json_object('article_id',article_id,'reason',reason,'differences',differences,'created_at',created_at,'updated_at',updated_at)) FROM (SELECT article_id,reason,differences,created_at,updated_at FROM article_likes WHERE user_id=p.user_id ORDER BY created_at,article_id)) AS likes_json FROM profiles p WHERE p.user_id=?`,userId);
  if(!snapshot) error(401,'Compte introuvable.');
  const likes=decorateLikes(parse(snapshot.likes_json,[])),r=readinessFor(likes);
  return {likes,count:r.likes_count,explained_count:r.explained_count,target:TARGET_LIKES,source_version:snapshot.version};
}
const personal=createPersonalBackend({error,json,text,body,now,random,decorateLikes,recommender,examples:EXAMPLES});
const chat=createChatBackend({error,json,text,body,now,random,personal});
const readingContext=(env,userId)=>personal.context(env,userId);
const portraitFor=(env,userId)=>personal.portrait(env,userId);
export const claimJobs=(env,...filters)=>personal.claimJobs(env,...filters);
export const completeJob=(env,id,data)=>personal.completeJob(env,id,data);
export const proxyJobLLM=(env,id,data)=>personal.proxyJobLLM(env,id,data);
export const failJob=(env,id,data)=>personal.failJob(env,id,data);
export const refreshUserPortrait=(env,userId,options)=>personal.refreshUserPortrait(env,userId,options);
export async function runGenerationBatch(env) {return env?.OPENROUTER_API_KEY?personal.refreshPendingPortraits(env):{status:'workflow-managed',claimed:0,completed:0,failed:0};}
async function internal(request,env,path) {
  await githubIdentity(request,env);
  if(request.method!=='POST') error(405,'Méthode non autorisée.');
  if(path==='/api/internal/jobs/run') return json(await runGenerationBatch(env));
  if(path==='/api/internal/jobs/claim') return claimJobs(env);
  const match=path.match(/^\/api\/internal\/jobs\/([\w-]+)\/(complete|fail|llm)$/);if(!match) error(404,'Route inconnue.');
  const data=await body(request,1000000);
  if(match[2]==='llm') return proxyJobLLM(env,match[1],data);
  if(match[2]==='complete') return completeJob(env,match[1],data);
  return personal.failJob(env,match[1],data);
}
async function routes(request,env,ctx) {
  const url=new URL(request.url),path=url.pathname,method=request.method;
  if(method==='GET'&&path==='/') return Response.redirect('https://decodetonesprit.fr/',302);
  if(['GET','HEAD'].includes(method)&&['/mes-articles','/mes-articles/'].includes(path)) return Response.redirect(new URL('/mon-portrait/'+url.search+'#mes-articles',env.PUBLIC_ORIGIN||'https://decodetonesprit.fr').href,302);
  if(['GET','HEAD'].includes(method)&&['/mes-histoires','/mes-histoires/','/mes-personnages','/mes-personnages/'].includes(path)) return Response.redirect(new URL('/mon-personnage/'+url.search,env.PUBLIC_ORIGIN||'https://decodetonesprit.fr').href,302);
  const pageTitles={'/mon-accompagnement':'Ambre','/mon-personnage':'Mon personnage','/mes-proches':'Mes proches','/ma-ligne-de-vie':'Ma ligne de vie','/mon-portrait':'Mon portrait','/mon-compte':'Mon compte'};
  const pagePath=path.replace(/\/$/,'');
  if(['GET','HEAD'].includes(method)&&pageTitles[pagePath]) {
    const page=HTML.replace(/<title>[^<]*<\/title>/,'<title>Décode ton esprit</title>');
    return new Response(method==='HEAD'?null:page,{headers:{'Content-Type':'text/html; charset=utf-8'}});
  }
  if(['GET','HEAD'].includes(method)&&['/journal','/journal/'].includes(path)) {
    const tab=url.searchParams.get('tab');
    const destination=url.searchParams.has('payment')?'/mon-compte/':url.searchParams.has('story')?'/mon-personnage/':url.searchParams.has('article')||url.searchParams.has('character')||tab==='likes'?'/mon-portrait/':tab==='portrait'?'/mon-portrait/':tab==='account'?'/mon-compte/':'/mon-personnage/';
    return Response.redirect(new URL(destination+url.search+(destination==='/mon-portrait/'&&(url.searchParams.has('article')||url.searchParams.has('character')||tab==='likes')?'#mes-articles':''),env.PUBLIC_ORIGIN||'https://decodetonesprit.fr').href,302);
  }
  if(method==='GET'&&path==='/api/examples') return json({examples:EXAMPLES.map(({id,title,url,character,description,keywords,character_keywords,sujet,category})=>({id,title,url,character,description,keywords:keywords||[],character_keywords:character_keywords||[],sujet:sujet||'',category:category||''}))});
  if(method==='GET'&&path==='/api/health') return json({ok:true,billing:billingReady(env),ai:Boolean(env.OPENROUTER_API_KEY)});
  if(method==='GET'&&path==='/robots.txt') return new Response('User-agent: *\nDisallow: /\n');
  if(path==='/api/billing/webhook'&&method==='POST') return webhook(request,env);
  if(path.startsWith('/api/internal/')) return internal(request,env,path);
  if(!['GET','HEAD','OPTIONS'].includes(method)) originCheck(request,env);
  if(path==='/api/me'&&method==='GET') {
    const user=await userFor(request,env,false); if(!user) return json({user:null,configuration:{billing:billingReady(env),ai:Boolean(env.OPENROUTER_API_KEY),free_access:platformAccess(env,null).mode==='free_preview'}});
    const p=await first(env,'SELECT * FROM profiles WHERE user_id=?',user.id),s=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id),m=await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id);
    const readiness=readinessFor(await likesFor(env,user.id)),inspirations_count=readiness.likes_count;
    return json({user,access:platformAccess(env,s),inspirations_count,readiness,consent_ai:Boolean(p.consent_ai),source_version:p.version,profile:{...parse(p.data),consent_ai:Boolean(p.consent_ai)},subscription:{status:s.status,active:activeSubscription(s),paid_until:s.paid_until,trial_end:s.trial_end,trial_used_at:s.trial_used_at,trial_eligible:!s.trial_used_at,cancel_at_period_end:Boolean(s.cancel_at_period_end),access_until:accessUntil(s),manageable:Boolean(s.customer_id)},memory:m,configuration:{billing:billingReady(env),ai:Boolean(env.OPENROUTER_API_KEY),free_access:platformAccess(env,null).mode==='free_preview'}});
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
  const chatResponse=await chat.routes(request,env,user,path,url,ctx);if(chatResponse)return chatResponse;
  const personalResponse=await personal.routes(request,env,user,path,url,ctx);if(personalResponse)return personalResponse;
  if(path==='/api/auth/logout'&&method==='POST') {
    const token=request.headers.get('cookie')?.split(';').map(s=>s.trim()).find(s=>s.startsWith(COOKIE+'='))?.slice(COOKIE.length+1);
    if(token) await run(env,'DELETE FROM sessions WHERE token_hash=?',await digest(token));
    return json({ok:true},200,{'Set-Cookie':`${COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`});
  }
  if(path==='/api/profile'&&method==='PUT') {
    const data=await body(request,30000),profile={}; for(const field of FIELDS) profile[field]=text(data[field]??'',field==='display_name'?80:2000);
    if(typeof data.consent_ai!=='boolean') error(400,'Indiquez votre choix concernant la personnalisation IA.');
    await run(env,'UPDATE profiles SET data=?,consent_ai=?,version=version+1 WHERE user_id=? AND (data!=? OR consent_ai!=?)',JSON.stringify(profile),Number(data.consent_ai),user.id,JSON.stringify(profile),Number(data.consent_ai));
    await personal.sourcesChanged(env,user.id,ctx);
    return json({ok:true,profile:{...profile,consent_ai:data.consent_ai}});
  }
  if(path==='/api/consent'&&method==='PUT') {
    const data=await body(request,1000);if(typeof data.consent_ai!=='boolean') error(400,'Indiquez votre choix concernant la personnalisation IA.');
    await run(env,'UPDATE profiles SET consent_ai=?,version=version+1 WHERE user_id=? AND consent_ai!=?',Number(data.consent_ai),user.id,Number(data.consent_ai));
    const p=await first(env,'SELECT version,consent_ai FROM profiles WHERE user_id=?',user.id);
    await personal.sourcesChanged(env,user.id,ctx);
    return json({ok:true,consent_ai:Boolean(p.consent_ai),source_version:p.version});
  }
  if(path==='/api/portrait'&&method==='GET') {await personal.sourcesChanged(env,user.id,ctx);return json(await portraitFor(env,user.id));}
  if(path==='/api/recommendations'&&method==='GET') {
    const context=await readingContext(env,user.id);
    const rankingContext={...context,portrait:context.portrait_version===context.source_version?context.portrait:null};
    const items=hasPersonalizationSources(context)?recommender.recommend(rankingContext,{limit:12}):[];
    await env.DB.batch([env.DB.prepare("INSERT INTO recommendation_exposures(user_id,article_id) SELECT ?,value FROM json_each(?) WHERE true ON CONFLICT(user_id,article_id) DO UPDATE SET shown_at=CURRENT_TIMESTAMP").bind(user.id,JSON.stringify(items.map(x=>x.id))),env.DB.prepare("DELETE FROM recommendation_exposures WHERE user_id=? AND (shown_at<=datetime('now','-7 days') OR article_id NOT IN(SELECT article_id FROM recommendation_exposures WHERE user_id=? ORDER BY shown_at DESC,article_id LIMIT 100))").bind(user.id,user.id)]);
    return json({items,source_version:context.source_version,portrait_version:context.portrait_version,likes_count:context.likes.length});
  }
  if(path==='/api/likes'&&method==='GET') return json(await likesResponse(env,user.id));
  const likeMatch=path.match(/^\/api\/likes\/(.+)$/);
  if(likeMatch&&method==='PUT') {
    let articleId;try {articleId=decodeURIComponent(likeMatch[1]);} catch {error(400,'Identifiant d’article invalide.');}
    if(!catalog.has(articleId)) error(404,'Article introuvable.');
    const data=await body(request,32000);if(typeof data.liked!=='boolean') error(400,'Indiquez votre choix pour cet article.');
    if(data.reason!==undefined) text(data.reason,4000);
    if(data.differences!==undefined) text(data.differences,4000);
    if(data.liked) {
      await run(env,`INSERT INTO article_likes(user_id,article_id,reason,differences) VALUES(?,?,?,?) ON CONFLICT(user_id,article_id) DO UPDATE SET reason=CASE WHEN ? THEN excluded.reason ELSE article_likes.reason END,differences=CASE WHEN ? THEN excluded.differences ELSE article_likes.differences END,updated_at=CURRENT_TIMESTAMP WHERE (? AND article_likes.reason!=excluded.reason) OR (? AND article_likes.differences!=excluded.differences)`,user.id,articleId,data.reason===undefined?'':text(data.reason,4000),data.differences===undefined?'':text(data.differences,4000),Number(data.reason!==undefined),Number(data.differences!==undefined),Number(data.reason!==undefined),Number(data.differences!==undefined));
    } else await run(env,'DELETE FROM article_likes WHERE user_id=? AND article_id=?',user.id,articleId);
    await personal.sourcesChanged(env,user.id,ctx);
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
    await personal.sourcesChanged(env,user.id,ctx);
    return json({ok:true,selections,...await likesResponse(env,user.id)});
  }
  if(path==='/api/memory'&&method==='GET') return json(await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id));
  if(path==='/api/memory'&&method==='PUT') {
    const data=await body(request,50000),value=text(data.text,10000),expected=data.expected_version??null;
    if(expected!==null&&(!Number.isInteger(expected)||expected<1))error(400,'Version de mémoire invalide.','invalid_version');
    // Optional for older clients; versioned editors cannot overwrite an appended
    // portrait correction or another tab's draft with an older memory snapshot.
    const saved=await env.DB.batch([
      env.DB.prepare('UPDATE profiles SET version=version+1 WHERE user_id=? AND EXISTS(SELECT 1 FROM memories WHERE user_id=? AND text!=? AND (? IS NULL OR version=?))').bind(user.id,user.id,value,expected,expected),
      env.DB.prepare("UPDATE memories SET text=?,proposal='',version=version+CASE WHEN text!=? THEN 1 ELSE 0 END WHERE user_id=? AND (? IS NULL OR version=?)").bind(value,value,user.id,expected,expected)
    ]);
    if(expected!==null&&!saved[1].meta.changes)error(409,'Votre mémoire a changé. Rechargez-la avant de réessayer.','revision_conflict');
    await personal.sourcesChanged(env,user.id,ctx);
    return json({ok:true,source_version:(await first(env,'SELECT version FROM profiles WHERE user_id=?',user.id)).version,memory_version:(await first(env,'SELECT version FROM memories WHERE user_id=?',user.id)).version});
  }
  if(path==='/api/stories'&&method==='GET') return json({stories:await all(env,'SELECT id,day,title,summary,created_at FROM stories WHERE user_id=? ORDER BY day DESC LIMIT 366',user.id)});
  const storyMatch=path.match(/^\/api\/stories\/([\w-]+)(\/feedback)?$/);
  if(storyMatch) {
    const story=await first(env,'SELECT * FROM stories WHERE id=? AND user_id=?',storyMatch[1],user.id); if(!story) error(404,'Histoire introuvable.');
    if(!storyMatch[2]&&method==='GET') return json({story:{...story,reflection_questions:parse(story.reflection_questions,[]),narrative_delta:parse(story.narrative_delta)},feedback:await first(env,'SELECT resonance,feelings,why,correction,differences,scope,preference,avoid,revision FROM feedback WHERE story_id=? AND user_id=?',story.id,user.id)});
    if(storyMatch[2]&&method==='PUT') {
      const data=await body(request,60000); if(!Number.isInteger(data.resonance)||data.resonance<1||data.resonance>5) error(400,'Choisissez une note de 1 à 5.');
      if(!Number.isInteger(data.revision)||data.revision<0)error(400,'La révision du retour est requise, ou 0 pour un premier retour.','revision_required');
      const activeStory="EXISTS(SELECT 1 FROM stories s JOIN personal_characters c ON c.id=s.character_id AND c.user_id=s.user_id WHERE s.id=? AND s.user_id=? AND c.is_primary=1)";
      if(!await first(env,'SELECT 1 AS active WHERE '+activeStory,story.id,user.id))error(409,'Ce personnage est archivé. Son histoire reste en lecture seule.','character_archived');
      const fields={resonance:data.resonance,feelings:text(data.feelings||'',3000),why:text(data.why||'',4000),correction:text(data.correction||'',4000),differences:text(data.differences||'',4000),scope:text(data.scope||'overall',30),preference:text(data.preference||'',4000),avoid:text(data.avoid||'',4000)};
      if(!['overall','situation','reaction','character','style'].includes(fields.scope))error(400,'Portée invalide.','invalid_scope');
      const keys=Object.keys(fields),values=Object.values(fields),expected=data.revision;
      const result=await env.DB.batch([
        env.DB.prepare(`INSERT INTO feedback(story_id,user_id,${keys.join(',')}) SELECT ?,?,${keys.map(()=>'?').join(',')} WHERE ${activeStory} AND ((?=0 AND NOT EXISTS(SELECT 1 FROM feedback WHERE story_id=? AND user_id=?)) OR EXISTS(SELECT 1 FROM feedback WHERE story_id=? AND user_id=? AND revision=?)) ON CONFLICT(story_id) DO UPDATE SET ${keys.map(k=>k+'=excluded.'+k).join(',')},revision=feedback.revision+1,updated_at=CURRENT_TIMESTAMP WHERE feedback.revision=? AND ${activeStory} AND (${keys.map(k=>'feedback.'+k+'!=excluded.'+k).join(' OR ')})`).bind(story.id,user.id,...values,story.id,user.id,expected,story.id,user.id,story.id,user.id,expected,expected,story.id,user.id),
        env.DB.prepare('UPDATE profiles SET version=version+1 WHERE user_id=? AND changes()>0').bind(user.id)
      ]);
      const saved=await first(env,'SELECT resonance,feelings,why,correction,differences,scope,preference,avoid,revision FROM feedback WHERE story_id=? AND user_id=?',story.id,user.id);
      if(!result[0].meta.changes){
        if(!await first(env,'SELECT 1 AS active WHERE '+activeStory,story.id,user.id))error(409,'Ce personnage est archivé. Son histoire reste en lecture seule.','character_archived');
        if(!saved||saved.revision!==expected||keys.some(k=>saved[k]!==fields[k]))error(409,'Ce retour a été modifié. Votre brouillon est conservé.','revision_conflict');
      }
      await personal.sourcesChanged(env,user.id,ctx);
      return json({ok:true,feedback:saved,source_version:(await first(env,'SELECT version FROM profiles WHERE user_id=?',user.id)).version});
    }
  }
  if(path==='/api/billing/checkout'&&method==='POST') {
    if(!billingReady(env)||!env.OPENROUTER_API_KEY) error(503,'Les abonnements ne sont pas encore ouverts.');
    await limit(env,'checkout:'+user.id,5,3600); await validPrice(env);
    const current=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
    if(current.subscription_id&&['active','trialing','past_due','unpaid','incomplete'].includes(current.status)) error(409,'Un abonnement existe déjà. Gérez-le depuis votre espace abonnement.');
    const customer=await ensureCustomer(env,user),origin=env.PUBLIC_ORIGIN||url.origin;
    let state=await first(env,'SELECT checkout_attempt,checkout_session_id FROM subscriptions WHERE user_id=?',user.id);
    if(state.checkout_session_id) {
      const previous=await stripe(env,'checkout/sessions/'+encodeURIComponent(state.checkout_session_id));
      if(previous.status==='open'&&previous.url&&previous.metadata?.offer==='v4-29-trial7') return json({url:previous.url});
      if(previous.status==='open') await stripe(env,'checkout/sessions/'+encodeURIComponent(previous.id)+'/expire','POST',{});
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
    const existing={data:[]};let after='';
    do {const page=await stripe(env,'subscriptions?customer='+encodeURIComponent(customer)+'&status=all&limit=100'+(after?'&starting_after='+encodeURIComponent(after):''));existing.data.push(...(page.data||[]));after=page.has_more?page.data?.at(-1)?.id:'';if(page.has_more&&!after)error(502,'Historique de paiement indisponible.');} while(after);
    const priorTrial=existing.data.find(s=>s.trial_start||s.trial_end);
    if(priorTrial) await run(env,'UPDATE subscriptions SET trial_used_at=CASE WHEN trial_used_at=0 THEN ? ELSE trial_used_at END WHERE user_id=?',Number(priorTrial.trial_start||priorTrial.trial_end),user.id);
    const trialEligible=!(await first(env,'SELECT trial_used_at FROM subscriptions WHERE user_id=?',user.id)).trial_used_at;
    const live=existing.data?.find(s=>!['canceled','incomplete_expired'].includes(s.status));
    if(live) {await synchronizeSubscription(env,live.id,now()); error(409,'Un abonnement existe déjà. Gérez-le depuis votre espace abonnement.');}
    const sessions=await stripe(env,'checkout/sessions?customer='+encodeURIComponent(customer)+'&status=open&limit=100');
    const open=sessions.data?.find(s=>s.mode==='subscription'&&s.url&&s.metadata?.offer==='v4-29-trial7');
    if(open) {await run(env,'UPDATE subscriptions SET checkout_session_id=? WHERE user_id=? AND checkout_attempt=?',open.id,user.id,state.checkout_attempt);return json({url:open.url});}
    const result=await stripe(env,'checkout/sessions','POST',{'mode':'subscription','metadata[offer]':'v4-29-trial7','customer':customer,'line_items[0][price]':env.STRIPE_PRICE_ID,'line_items[0][quantity]':'1','success_url':origin+'/mon-compte/?payment=success','cancel_url':origin+'/mon-compte/?payment=cancelled','client_reference_id':user.id,'subscription_data[metadata][user_id]':user.id,'payment_method_collection':'always',...(trialEligible?{'subscription_data[trial_period_days]':'7','subscription_data[trial_settings][end_behavior][missing_payment_method]':'cancel'}:{})},'dte-checkout-'+user.id+'-'+state.checkout_attempt);
    await run(env,'UPDATE subscriptions SET checkout_session_id=? WHERE user_id=? AND checkout_attempt=?',result.id,user.id,state.checkout_attempt);
    return json({url:result.url});
  }
  if(path==='/api/billing/cancel'&&method==='POST') {
    const observed=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
    if(!observed?.subscription_id) error(400,'Aucun abonnement à résilier.');
    const subscriptionId=observed.subscription_id;
    const canonical=await synchronizeSubscription(env,subscriptionId,now());
    if(canonical.status==='canceled') return json({ok:true,cancel_at_period_end:false,access_until:0});
    if(canonical.cancel_at_period_end){
      await run(env,'UPDATE subscriptions SET cancellation_confirmed=1 WHERE user_id=? AND cancellation_subscription_id=?',user.id,subscriptionId);
      const updated=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
      return json({ok:true,cancel_at_period_end:true,access_until:accessUntil(updated)});
    }
    // A transport retry keeps its durable key. Once Stripe has accepted an
    // attempt, a later canonical false means a resumed subscription/new intent.
    // CAS prevents simultaneous clicks from opening two attempts.
    if(!observed.cancellation_attempt||observed.cancellation_subscription_id!==subscriptionId||observed.cancellation_confirmed){
      await run(env,'UPDATE subscriptions SET cancellation_attempt=?,cancellation_subscription_id=?,cancellation_confirmed=0 WHERE user_id=? AND subscription_id=? AND cancellation_attempt IS ?',random(),subscriptionId,user.id,subscriptionId,observed.cancellation_attempt);
    }
    const attempt=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
    if(attempt.subscription_id!==subscriptionId||attempt.cancellation_subscription_id!==subscriptionId||!attempt.cancellation_attempt)error(409,'Votre abonnement a changé. Actualisez votre compte avant de réessayer.','subscription_changed');
    await stripe(env,'subscriptions/'+encodeURIComponent(subscriptionId),'POST',{'cancel_at_period_end':'true'},'dte-cancel-'+subscriptionId+'-'+attempt.cancellation_attempt);
    await run(env,'UPDATE subscriptions SET cancellation_confirmed=1 WHERE user_id=? AND cancellation_attempt=? AND cancellation_subscription_id=?',user.id,attempt.cancellation_attempt,subscriptionId);
    const verified=await synchronizeSubscription(env,subscriptionId,now());
    if(verified.status!=='canceled'&&!verified.cancel_at_period_end)error(502,'La résiliation n’a pas été confirmée par le service de paiement. Réessayez ou vérifiez votre abonnement dans l’espace de facturation.','cancellation_not_confirmed');
    const updated=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id);
    return json({ok:true,cancel_at_period_end:Boolean(verified.cancel_at_period_end),access_until:accessUntil(updated)});
  }
  if(path==='/api/billing/portal'&&method==='POST') {
    const sub=await first(env,'SELECT customer_id FROM subscriptions WHERE user_id=?',user.id); if(!sub?.customer_id) error(400,'Aucun abonnement à gérer.');
    const result=await stripe(env,'billing_portal/sessions','POST',{'customer':sub.customer_id,'return_url':(env.PUBLIC_ORIGIN||url.origin)+'/mon-compte/'}); return json({url:result.url});
  }
  if(path==='/api/export'&&method==='GET') {
    const profile=await first(env,'SELECT data,consent_ai FROM profiles WHERE user_id=?',user.id),memory=await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id);
    return json({user,profile:{...parse(profile.data),consent_ai:Boolean(profile.consent_ai)},memory,likes:await likesFor(env,user.id),portrait:await portraitFor(env,user.id),selections:await all(env,'SELECT article_id,reason,differences FROM article_likes WHERE user_id=? ORDER BY article_id',user.id),...await personal.exportData(env,user.id),...await chat.exportData(env,user.id)},200,{'Content-Disposition':'attachment; filename="mon-journal-decode-ton-esprit.json"'});
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
  async fetch(request,env,ctx) {
    let response;
    try {response=await routes(request,env,ctx);} catch(e) {response=json({error:e instanceof HTTPError?e.message:'Une erreur temporaire est survenue. Réessayez plus tard.',...(e instanceof HTTPError&&e.code?{code:e.code}:{})},e instanceof HTTPError?e.status:500);}
    const headers=new Headers(response.headers);
    headers.set('Cache-Control','private, no-store'); headers.set('X-Content-Type-Options','nosniff'); headers.set('Referrer-Policy','no-referrer'); headers.set('X-Frame-Options','DENY'); headers.set('Permissions-Policy','camera=(), microphone=(), geolocation=()');
    headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    return new Response(response.body,{status:response.status,headers});
  }
};
