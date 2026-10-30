import { HTML } from './ui.mjs';

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
  return claims.iss===ISSUER && claims.aud==='decode-ton-esprit-journal' && claims.repository===repository && claims.ref==='refs/heads/main' && claims.workflow_ref===`${repository}/.github/workflows/journal-daily.yml@refs/heads/main` && ['schedule','workflow_dispatch'].includes(claims.event_name) && Number(claims.exp)>at && Number(claims.iat)<=at+30 && Number(claims.nbf||0)<=at+30 && Number(claims.exp)-Number(claims.iat)<=900;
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
export async function claimJobs(env) {
  const day=frenchDay(),t=now();
  const candidates=await all(env,`SELECT u.id FROM users u JOIN subscriptions s ON s.user_id=u.id JOIN profiles p ON p.user_id=u.id WHERE s.status='active' AND s.paid_until>? AND p.consent_ai=1 AND length(json_extract(p.data,'$.display_name'))>0 AND NOT EXISTS(SELECT 1 FROM stories WHERE user_id=u.id AND day=?) AND NOT EXISTS(SELECT 1 FROM generation_jobs j WHERE j.user_id=u.id AND j.day=? AND (j.attempts>=3 OR j.status='complete' OR j.lease_until>?)) ORDER BY u.created_at LIMIT 1`,t,day,day,t);
  const jobs=[];
  for(const u of candidates) {
    await run(env,'INSERT OR IGNORE INTO generation_jobs(id,user_id,day) VALUES(?,?,?)',random(),u.id,day);
    const p=await first(env,'SELECT * FROM profiles WHERE user_id=?',u.id),m=await first(env,'SELECT * FROM memories WHERE user_id=?',u.id);
    const lease=random()+random();
    const j=await first(env,`UPDATE generation_jobs SET status='running',attempts=attempts+1,lease_token=?,lease_until=?,profile_version=?,memory_version=?,error_code=NULL WHERE user_id=? AND day=? AND status!='complete' AND lease_until<=? AND attempts<3 RETURNING id`,lease,t+1800,p.version,m.version,u.id,day,t);
    if(!j) continue;
    const history=await all(env,'SELECT day,title,summary FROM stories WHERE user_id=? ORDER BY day DESC LIMIT 20',u.id);
    const feedback=await all(env,'SELECT s.day,s.title,f.resonance,f.feelings,f.why,f.correction,f.updated_at FROM feedback f JOIN stories s ON s.id=f.story_id WHERE f.user_id=? ORDER BY f.updated_at DESC,s.day DESC LIMIT 20',u.id);
    jobs.push({id:j.id,user_id:u.id,day,lease_token:lease,profile:parse(p.data),profile_version:p.version,memory:{text:m.text,proposal:m.proposal,version:m.version},history:history.reverse(),feedback:feedback.reverse()});
  }
  return json({jobs});
}
export async function completeJob(env,id,data) {
  const j=await first(env,'SELECT * FROM generation_jobs WHERE id=?',id);
  if(!j||!constantEqual(j.lease_token,data.lease_token)||j.status!=='running'||j.lease_until<now()) error(409,'Bail expiré ou inconnu.');
  const p=await first(env,'SELECT * FROM profiles WHERE user_id=?',j.user_id),m=await first(env,'SELECT * FROM memories WHERE user_id=?',j.user_id),sub=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',j.user_id);
  if(!p?.consent_ai||!activeSubscription(sub)) error(409,'Génération annulée.');
  if(p.version!==data.profile_version||m.version!==data.memory_version||j.profile_version!==p.version||j.memory_version!==m.version) {
    await run(env,"UPDATE generation_jobs SET status='pending',lease_until=0,lease_token=NULL,error_code='context_changed' WHERE id=?",id); error(409,'Le contexte a changé.');
  }
  const title=text(data.title,160),content=text(data.body,40000),summary=text(data.summary,3000),practice=text(data.practice,2000),proposal=text(data.memory_proposal||'',5000);
  if(!title||content.length<500||!summary) error(400,'Histoire incomplète.');
  if(!Array.isArray(data.reflection_questions)||data.reflection_questions.length<1||data.reflection_questions.length>5) error(400,'Questions invalides.');
  const questions=data.reflection_questions.map(q=>text(q,500));
  // Every write repeats the version/consent/lease predicates inside one atomic D1 batch.
  const guard=`EXISTS(SELECT 1 FROM generation_jobs j JOIN profiles p ON p.user_id=j.user_id JOIN memories m ON m.user_id=j.user_id JOIN subscriptions s ON s.user_id=j.user_id WHERE j.id=? AND j.status='running' AND j.lease_token=? AND j.lease_until>=? AND p.version=? AND m.version=? AND p.consent_ai=1 AND s.status='active' AND s.paid_until>?)`;
  const args=[id,data.lease_token,now(),p.version,m.version,now()];
  const result=await env.DB.batch([
    env.DB.prepare(`INSERT INTO stories(id,user_id,day,title,body,summary,reflection_questions,practice) SELECT ?,?,?,?,?,?,?,? WHERE ${guard} ON CONFLICT(user_id,day) DO NOTHING`).bind(id,j.user_id,j.day,title,content,summary,JSON.stringify(questions),practice,...args),
    env.DB.prepare(`UPDATE memories SET proposal=? WHERE user_id=? AND ${guard}`).bind(proposal,j.user_id,...args),
    env.DB.prepare("UPDATE generation_jobs SET status='complete',lease_until=0 WHERE id=? AND lease_token=? AND EXISTS(SELECT 1 FROM stories WHERE id=?)").bind(id,data.lease_token,id)
  ]);
  if(!result[0].meta.changes) error(409,'Publication déjà effectuée ou contexte modifié.');
  return json({ok:true,story_id:id});
}
async function internal(request,env,path) {
  await githubIdentity(request,env);
  if(request.method!=='POST') error(405,'Méthode non autorisée.');
  if(path==='/api/internal/jobs/claim') return claimJobs(env);
  const match=path.match(/^\/api\/internal\/jobs\/([\w-]+)\/(complete|fail)$/); if(!match) error(404,'Route inconnue.');
  const data=await body(request,60000);
  if(match[2]==='complete') return completeJob(env,match[1],data);
  const allowed=['provider_error','validation_error','coherence_error','unexpected_error'];
  await run(env,"UPDATE generation_jobs SET status='pending',lease_until=0,error_code=? WHERE id=? AND status='running' AND lease_token=?",allowed.includes(data.reason)?data.reason:'unexpected_error',match[1],text(data.lease_token,100));
  return json({ok:true});
}
async function routes(request,env) {
  const url=new URL(request.url),path=url.pathname,method=request.method;
  if(method==='GET'&&path==='/') return new Response(HTML,{headers:{'Content-Type':'text/html; charset=utf-8'}});
  if(method==='GET'&&path==='/api/health') return json({ok:true,billing:billingReady(env)});
  if(method==='GET'&&path==='/robots.txt') return new Response('User-agent: *\nDisallow: /\n');
  if(path==='/api/billing/webhook'&&method==='POST') return webhook(request,env);
  if(path.startsWith('/api/internal/')) return internal(request,env,path);
  if(!['GET','HEAD','OPTIONS'].includes(method)) originCheck(request,env);
  if(path==='/api/me'&&method==='GET') {
    const user=await userFor(request,env,false); if(!user) return json({user:null,configuration:{billing:billingReady(env)}});
    const p=await first(env,'SELECT * FROM profiles WHERE user_id=?',user.id),s=await first(env,'SELECT * FROM subscriptions WHERE user_id=?',user.id),m=await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id);
    return json({user,profile:{...parse(p.data),consent_ai:Boolean(p.consent_ai)},subscription:{status:s.status,active:activeSubscription(s),paid_until:s.paid_until,manageable:Boolean(s.customer_id)},memory:m,configuration:{billing:billingReady(env)}});
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
    if(!profile.display_name) error(400,'Choisissez un prénom ou un pseudonyme pour votre personnage.');
    if(typeof data.consent_ai!=='boolean') error(400,'Indiquez votre choix concernant la personnalisation IA.');
    await run(env,'UPDATE profiles SET data=?,consent_ai=?,version=version+1 WHERE user_id=?',JSON.stringify(profile),Number(data.consent_ai),user.id);
    return json({ok:true,profile:{...profile,consent_ai:data.consent_ai}});
  }
  if(path==='/api/memory'&&method==='GET') return json(await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id));
  if(path==='/api/memory'&&method==='PUT') {
    const data=await body(request,15000); const value=text(data.text,10000);
    await run(env,"UPDATE memories SET text=?,proposal='',version=version+1 WHERE user_id=?",value,user.id); return json({ok:true});
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
      return json({ok:true});
    }
  }
  if(path==='/api/billing/checkout'&&method==='POST') {
    if(!billingReady(env)) error(503,'Les abonnements ne sont pas encore ouverts.');
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
    const result=await stripe(env,'checkout/sessions','POST',{'mode':'subscription','customer':customer,'line_items[0][price]':env.STRIPE_PRICE_ID,'line_items[0][quantity]':'1','success_url':origin+'/?payment=success','cancel_url':origin+'/?payment=cancelled','client_reference_id':user.id,'subscription_data[metadata][user_id]':user.id},'dte-checkout-'+user.id+'-'+state.checkout_attempt);
    await run(env,'UPDATE subscriptions SET checkout_session_id=? WHERE user_id=? AND checkout_attempt=?',result.id,user.id,state.checkout_attempt);
    return json({url:result.url});
  }
  if(path==='/api/billing/portal'&&method==='POST') {
    const sub=await first(env,'SELECT customer_id FROM subscriptions WHERE user_id=?',user.id); if(!sub?.customer_id) error(400,'Aucun abonnement à gérer.');
    const result=await stripe(env,'billing_portal/sessions','POST',{'customer':sub.customer_id,'return_url':env.PUBLIC_ORIGIN||url.origin}); return json({url:result.url});
  }
  if(path==='/api/export'&&method==='GET') {
    const profile=await first(env,'SELECT data,consent_ai FROM profiles WHERE user_id=?',user.id),memory=await first(env,'SELECT text,proposal,version FROM memories WHERE user_id=?',user.id);
    return json({user,profile:{...parse(profile.data),consent_ai:Boolean(profile.consent_ai)},memory,stories:await all(env,'SELECT day,title,body,summary,reflection_questions,practice FROM stories WHERE user_id=? ORDER BY day',user.id),feedback:await all(env,'SELECT story_id,resonance,feelings,why,correction,updated_at FROM feedback WHERE user_id=?',user.id)},200,{'Content-Disposition':'attachment; filename="mon-journal-decode-ton-esprit.json"'});
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
  async fetch(request,env) {
    let response;
    try {response=await routes(request,env);} catch(e) {response=json({error:e instanceof HTTPError?e.message:'Une erreur temporaire est survenue. Réessayez plus tard.'},e instanceof HTTPError?e.status:500);}
    const headers=new Headers(response.headers);
    headers.set('Cache-Control','no-store'); headers.set('X-Content-Type-Options','nosniff'); headers.set('Referrer-Policy','no-referrer'); headers.set('X-Frame-Options','DENY'); headers.set('Permissions-Policy','camera=(), microphone=(), geolocation=()');
    headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    return new Response(response.body,{status:response.status,headers});
  }
};
