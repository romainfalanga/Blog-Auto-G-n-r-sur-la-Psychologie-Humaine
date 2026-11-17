// Commercial credits use a Paris calendar month in every access mode. Provider
// weekly/daily token budgets remain separate operational safeguards.
const LIMITS={message:100,review:1};
function parisMidnight(year,month) {
 const utc=Date.UTC(year,month,1),local=new Intl.DateTimeFormat('en-GB',{timeZone:'Europe/Paris',hour:'2-digit',hourCycle:'h23'}).format(new Date(utc));
 return new Date(utc-Number(local)*3600000).toISOString();
}
export function parisMonth(at=Math.floor(Date.now()/1000)) {
 const date=at instanceof Date?at:new Date(at*1000),parts=new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Paris',year:'numeric',month:'2-digit'}).formatToParts(date),year=Number(parts.find(x=>x.type==='year').value),month=Number(parts.find(x=>x.type==='month').value);
 return {id:year+'-'+String(month).padStart(2,'0'),basis:'calendar_month',start:parisMidnight(year,month-1),end:parisMidnight(year,month),time_zone:'Europe/Paris'};
}
export function monthlyAvailableSql(userExpression,kind,periodId) {
 if(!/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$/.test(userExpression)||!Object.hasOwn(LIMITS,kind)||!/^\d{4}-\d{2}$/.test(periodId))throw Error('Invalid monthly quota guard');
 return `(SELECT count(*) FROM monthly_usage mu WHERE mu.user_id=${userExpression} AND mu.kind='${kind}' AND mu.period_id='${periodId}' AND mu.state IN ('reserved','consumed'))<${LIMITS[kind]}`;
}
export function reservedUsageSql(userExpression,kind,operationExpression) {
 for(const value of [userExpression,operationExpression])if(!/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$/.test(value))throw Error('Invalid usage owner');
 if(!Object.hasOwn(LIMITS,kind))throw Error('Invalid usage kind');
 return `EXISTS(SELECT 1 FROM monthly_usage mu WHERE mu.user_id=${userExpression} AND mu.kind='${kind}' AND mu.operation_id=${operationExpression} AND mu.state='reserved')`;
}
export async function readMonthlyQuotas(env,userId,at=Math.floor(Date.now()/1000)) {
 const period=parisMonth(at),rows=(await env.DB.prepare("SELECT kind,sum(state='consumed') used,sum(state='reserved') reserved FROM monthly_usage WHERE user_id=? AND period_id=? GROUP BY kind").bind(userId,period.id).all()).results;
 const make=kind=>{const row=rows.find(row=>row.kind===kind),used=Number(row?.used||0),reserved=Number(row?.reserved||0),limit=LIMITS[kind];return {limit,used,reserved,remaining:Math.max(0,limit-used-reserved),period,renews_at:period.end,time_zone:'Europe/Paris'}};
 return {messages:make('message'),reviews:make('review')};
}
