// Trial and paid access share one contract across HTTP and atomic SQL guards.
export function accessUntil(s) {return s?.status==='trialing'?Number(s.trial_end||0):s?.status==='active'?Number(s.paid_until||0):0;}
export function isEntitled(s,at=Math.floor(Date.now()/1000)) {return accessUntil(s)>at;}
export function entitlementSql(alias='s',nowExpression='?') {
 if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(alias)) throw new Error('Invalid subscription alias');
 return `(${alias}.status IN ('active','trialing') AND CASE WHEN ${alias}.status='trialing' THEN ${alias}.trial_end ELSE ${alias}.paid_until END>${nowExpression})`;
}

// Preview grants platform access without inventing a Stripe entitlement.
export const freeAccessEnabled=env=>['true','1','yes'].includes(String(env?.FREE_ACCESS_ENABLED||'').trim().toLowerCase());
export function platformAccess(env,subscription,at=Math.floor(Date.now()/1000)) {
 const entitled=isEntitled(subscription,at),preview=freeAccessEnabled(env);
 return {allowed:preview||entitled,mode:preview?'free_preview':entitled?(subscription.status==='trialing'?'trial':'paid'):'subscription_required'};
}
export function accessSql(env,alias='s',nowExpression='?') {
 // Keep the same placeholder contract in both modes for atomic SQL guards.
 return `(${freeAccessEnabled(env)?'1':'0'}=1 OR ${entitlementSql(alias,nowExpression)})`;
}
