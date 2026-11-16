// F23: the private personal context can weigh several megabytes (50 people, 500
// moments). The conversation reads it at the request, then at every engine stage.
// This wrapper keeps the last few contexts and revalidates each one with a single
// small row of every revision it depends on; any change reloads the full context.
// Callers must treat the returned context as read-only.
const STAMP_SQL='SELECT p.version,p.self_revision,m.version AS memory_version,up.updated_at,up.analyzed_version,up.schema_version,s.status,s.paid_until,s.trial_end,c.version AS map_version,c.consent,c.locked_at,c.ambre_notes FROM profiles p JOIN memories m ON m.user_id=p.user_id JOIN subscriptions s ON s.user_id=p.user_id LEFT JOIN user_portraits up ON up.user_id=p.user_id LEFT JOIN cartographies c ON c.user_id=p.user_id WHERE p.user_id=?';
export function cachedContext(load,{now,size=2}={}){
 const cache=new Map();
 return async function context(env,userId){
  let stamp;
  // The minute bucket bounds time-dependent fields (access, declared age) to one minute.
  try{stamp=JSON.stringify([Math.floor(now()/60),await env.DB.prepare(STAMP_SQL).bind(userId).first()])}catch{return load(env,userId)}
  const hit=cache.get(userId);
  if(hit&&hit.env===env&&hit.stamp===stamp)return hit.context;
  const value=await load(env,userId);
  cache.delete(userId);cache.set(userId,{env,stamp,context:value});
  while(cache.size>size)cache.delete(cache.keys().next().value);
  return value;
 };
}
