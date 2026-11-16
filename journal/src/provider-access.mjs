// Shared access policy for every OpenRouter call (conversation, memory, reviews,
// care briefs, portrait). It never stores prompts or provider messages: only
// typed codes, timestamps and aggregated counters.
export const PROVIDER_URL='https://openrouter.ai/api/v1/chat/completions';
// Statuses that consumed nothing upstream: the reservation is refunded and the
// call does not count as an attempt.
export const REFUSAL_STATUSES=[401,402,403,429,503];
// Statuses worth waiting for. Others (401/403) end the work with a clear code.
const WAIT_STATUSES=[402,429,503];
const DEFAULT_COOLDOWN={401:300,402:900,403:300,429:30,503:20};
const missingTable=e=>/no such table:\s*(?:provider_state|provider_global_budget|service_metrics)/i.test(String(e?.message||''));
const tolerant=async(query,fallback)=>{try{return await query()}catch(e){if(missingTable(e))return fallback;throw e}};

export const retryAfterSeconds=(header,clock)=>{if(!header)return null;const seconds=/^\d+(?:\.\d+)?$/.test(String(header).trim())?Math.ceil(Number(header)):Math.ceil((Date.parse(header)/1000)-clock);return Number.isFinite(seconds)?Math.max(1,Math.min(86400,seconds)):null};
// Only typed upstream metadata can refine the diagnosis. Never persist a raw
// provider message, account identifier, spend amount or an arbitrary error type.
export const provider429Suffix=metadata=>{
 const sources={openrouter_key_limit:'key_budget',openrouter_workspace_budget:'workspace_budget',openrouter_guardrail_budget:'guardrail_budget',openrouter_in_flight_budget:'in_flight_budget'};
 const source=Object.hasOwn(sources,metadata?.limit_source)?sources[metadata.limit_source]:null;
 return source?'_'+source:metadata?.error_type==='rate_limit_exceeded'?'_rate_limit':'';
};
// A key-wide limit (budget, credits, authentication) blocks every model; an
// upstream rate limit or outage is local to the model that answered.
export function refusalScope(status,metadata,model){
 if([401,402,403].includes(status))return 'key';
 if(status===429&&/_(?:key|workspace|guardrail|in_flight)_budget$/.test(provider429Suffix(metadata)))return 'key';
 return 'model:'+String(model||'unknown').slice(0,160);
}
export const modelScopes=models=>['key',...[...new Set(models.filter(Boolean))].map(model=>'model:'+String(model).slice(0,160))];
// Latest retry-not-before among the given models and the key. Indexed by primary key.
export async function providerBlock(env,models,now){
 const scopes=modelScopes(models);
 const row=await tolerant(()=>env.DB.prepare(`SELECT scope,retry_at,reason FROM provider_state WHERE scope IN (${scopes.map(()=>'?').join(',')}) AND retry_at>? ORDER BY retry_at DESC LIMIT 1`).bind(...scopes,now).first(),null);
 return row?{scope:row.scope,retry_at:Number(row.retry_at),reason:row.reason||'provider_cooldown'}:null;
}
// Records the cooldown carried by a refused response and returns its typed error.
export async function recordRefusal(env,{status,headers,metadata,model,now}){
 const scope=refusalScope(status,metadata,model),seconds=Math.min(3600,retryAfterSeconds(headers?.get?.('Retry-After'),now)??DEFAULT_COOLDOWN[status]??30),retryAt=now+seconds;
 const code='provider_http_'+status+(status===429?provider429Suffix(metadata):'');
 await tolerant(()=>env.DB.prepare('INSERT INTO provider_state(scope,retry_at,reason,updated_at) VALUES(?,?,?,?) ON CONFLICT(scope) DO UPDATE SET retry_at=max(provider_state.retry_at,excluded.retry_at),reason=excluded.reason,updated_at=excluded.updated_at').bind(scope,retryAt,code,now).run(),null);
 await countMetric(env,now,'provider_refusal_'+status);
 return refusalError(code,retryAt,{wait:WAIT_STATUSES.includes(status)});
}
// A refusal is a deferral: callers that already rethrow `stage_deferred` keep the
// durable stage untouched; the engine decides whether the turn can still wait.
export const refusalError=(providerCode,retryAt,{wait=true}={})=>Object.assign(Error('Provider refused or cooling down'),{code:'stage_deferred',provider_code:providerCode,deferred_until:retryAt,refusal:true,wait});
export const isRefusal=error=>Boolean(error?.refusal);
export const parisDay=now=>new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Paris',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(now*1000));
// Global daily token cap (DAILY_PROVIDER_TOKEN_CAP, 0 or unset = disabled).
// Background work (memory, reviews, care, portrait) may use 80 % of the cap so
// that the conversation always keeps priority.
export const dailyCap=env=>Math.max(0,Math.floor(Number(env?.DAILY_PROVIDER_TOKEN_CAP)||0));
export async function reserveDaily(env,{tokens,now,conversation=false}){
 const cap=dailyCap(env);if(!cap)return true;
 const ceiling=conversation?cap:Math.floor(cap*0.8),day=parisDay(now);
 const result=await tolerant(()=>env.DB.prepare('INSERT INTO provider_global_budget(day,reserved_tokens,calls) SELECT ?,?,1 WHERE ?<=? ON CONFLICT(day) DO UPDATE SET reserved_tokens=reserved_tokens+excluded.reserved_tokens,calls=calls+1 WHERE reserved_tokens+excluded.reserved_tokens<=?').bind(day,tokens,tokens,ceiling,ceiling).run(),{meta:{changes:1}});
 if(!result.meta.changes)await countMetric(env,now,'daily_cap_refused');
 return result.meta.changes>0;
}
export async function releaseDaily(env,{tokens,now}){
 if(!dailyCap(env))return;
 await tolerant(()=>env.DB.prepare('UPDATE provider_global_budget SET reserved_tokens=max(0,reserved_tokens-?),calls=max(0,calls-1) WHERE day=?').bind(tokens,parisDay(now)).run(),null);
}
export async function dailyCapReached(env,now,{conversation=true}={}){
 const cap=dailyCap(env);if(!cap)return false;
 const row=await tolerant(()=>env.DB.prepare('SELECT reserved_tokens FROM provider_global_budget WHERE day=?').bind(parisDay(now)).first(),null);
 return Number(row?.reserved_tokens||0)>=(conversation?cap:Math.floor(cap*0.8));
}
export const DAILY_CAP_CODE='daily_cap';
export const dailyCapError=()=>Object.assign(Error('Daily provider cap reached'),{code:DAILY_CAP_CODE});
// Aggregated counters only (names are fixed codes, never user content).
export async function countMetric(env,now,name,delta=1){
 if(!/^[a-z0-9_]{1,64}$/.test(name)||!Number.isFinite(delta))return;
 await tolerant(()=>env.DB.prepare('INSERT INTO service_metrics(day,name,value) VALUES(?,?,?) ON CONFLICT(day,name) DO UPDATE SET value=value+excluded.value').bind(parisDay(now),name,Math.round(delta)).run(),null).catch(()=>null);
}
// Several counters in one statement (one D1 write per provider call).
export async function countMetrics(env,now,values){
 const rows=Object.entries(values).filter(([name,delta])=>/^[a-z0-9_]{1,64}$/.test(name)&&Number.isFinite(delta)).map(([name,delta])=>[name,Math.round(delta)]);
 if(!rows.length)return;
 await tolerant(()=>env.DB.prepare("INSERT INTO service_metrics(day,name,value) SELECT ?,json_extract(value,'$[0]'),json_extract(value,'$[1]') FROM json_each(?) WHERE true ON CONFLICT(day,name) DO UPDATE SET value=value+excluded.value").bind(parisDay(now),JSON.stringify(rows)).run(),null).catch(()=>null);
}
