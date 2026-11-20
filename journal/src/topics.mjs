// « Mes sujets » (migration 0035): topics, positions and the arguments that support or contradict
// them. Plain storage for the owner: no model is called and nothing here is sent to Ambre.
export const TOPIC_LIMITS=Object.freeze({topics:60,nodes:300,title:120,content:1500,depth:5});
export const TOPIC_BASES=Object.freeze(['fact','experience','value','intuition','hearsay']);
const ID=/^[\w-]{1,120}$/;
const PROBED=new WeakMap();
// Like historyReady: a missing table is retried after a few seconds, any other error is raised.
export async function topicsReady(env){
 const db=env?.DB;if(!db)return false;const cached=PROBED.get(db);if(cached===true)return true;if(typeof cached==='number'&&cached>Date.now())return false;
 try{await db.prepare('SELECT id FROM personal_topic_nodes LIMIT 0').bind().all();PROBED.set(db,true);return true}
 catch(error){if(/no such (?:column|table)/i.test(String(error?.message||''))){PROBED.set(db,Date.now()+5000);return false}throw error}
}
const clean=value=>typeof value==='string'?value.replace(/\r\n?/g,'\n').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g,'').trim():'';
export function createTopicsBackend({error,json,body,now,random}){
 const first=(env,sql,...args)=>env.DB.prepare(sql).bind(...args).first();
 const all=async(env,sql,...args)=>(await env.DB.prepare(sql).bind(...args).all()).results;
 const run=(env,sql,...args)=>env.DB.prepare(sql).bind(...args).run();
 const fail=(status,message,code)=>error(status,message,code);
 const topicOut=row=>({id:row.id,title:row.title,created_at:row.created_at,updated_at:row.updated_at});
 const nodeOut=row=>({id:row.id,parent_id:row.parent_id||null,kind:row.kind,content:row.content,basis:row.basis||null,confidence:row.confidence??null,created_at:row.created_at,updated_at:row.updated_at});
 const title=data=>{const value=clean(data?.title).replace(/\s+/g,' ');if(!value)fail(400,'Donnez un nom à ce sujet.','invalid_input');if(value.length>TOPIC_LIMITS.title)fail(400,'Ce nom est trop long.','invalid_input');return value};
 const content=data=>{const value=clean(data?.content);if(!value)fail(400,'Écrivez quelques mots avant d’enregistrer.','invalid_input');if(value.length>TOPIC_LIMITS.content)fail(400,'Ce texte est trop long ('+TOPIC_LIMITS.content+' caractères au plus).','invalid_input');return value};
 const basis=data=>{const value=data?.basis;if(value==null||value==='')return null;if(!TOPIC_BASES.includes(value))fail(400,'Choix invalide.','invalid_input');return value};
 const confidence=data=>{const value=data?.confidence;if(value==null||value==='')return null;const n=Number(value);if(!Number.isInteger(n)||n<1||n>5)fail(400,'Choix invalide.','invalid_input');return n};
 // The same answer covers an unknown identifier and a topic of another account.
 const owned=async(env,userId,id)=>{const row=ID.test(id)?await first(env,'SELECT * FROM personal_topics WHERE id=? AND user_id=?',id,userId):null;if(!row)fail(404,'Ce sujet est introuvable.','not_found');return row};
 const nodes=(env,userId,topicId)=>all(env,'SELECT * FROM personal_topic_nodes WHERE topic_id=? AND user_id=? ORDER BY created_at,id',topicId,userId);
 const touch=(env,topicId)=>run(env,'UPDATE personal_topics SET updated_at=? WHERE id=?',now(),topicId);
 async function list(env,userId){
  const topics=await all(env,'SELECT * FROM personal_topics WHERE user_id=? ORDER BY updated_at DESC,id',userId);
  const counts=await all(env,"SELECT topic_id,sum(kind='position') positions,sum(kind<>'position') arguments FROM personal_topic_nodes WHERE user_id=? GROUP BY topic_id",userId);
  const byTopic=new Map(counts.map(row=>[row.topic_id,row]));
  return json({topics:topics.map(row=>({...topicOut(row),positions:Number(byTopic.get(row.id)?.positions||0),arguments:Number(byTopic.get(row.id)?.arguments||0)}))});
 }
 const detail=async(env,userId,topic)=>json({topic:topicOut(topic),nodes:(await nodes(env,userId,topic.id)).map(nodeOut)});
 async function createTopic(request,env,userId){
  const value=title(await body(request,2000));
  if(Number((await first(env,'SELECT count(*) n FROM personal_topics WHERE user_id=?',userId)).n)>=TOPIC_LIMITS.topics)fail(409,'Vous avez atteint le nombre maximal de sujets.','limit_reached');
  const id=random(),at=now();await run(env,'INSERT INTO personal_topics(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)',id,userId,value,at,at);
  return json({topic:{id,title:value,created_at:at,updated_at:at,positions:0,arguments:0}},201);
 }
 async function createNode(request,env,userId,topic){
  const data=await body(request,8000),text=content(data),parentId=data.parent_id==null||data.parent_id===''?null:String(data.parent_id);
  const rows=await nodes(env,userId,topic.id);
  if(rows.length>=TOPIC_LIMITS.nodes)fail(409,'Ce sujet contient déjà beaucoup d’éléments. Créez un autre sujet pour continuer.','limit_reached');
  let kind='position';
  if(parentId){
   const byId=new Map(rows.map(row=>[row.id,row]));if(!byId.has(parentId))fail(404,'L’élément auquel vous répondez est introuvable.','not_found');
   kind=data.kind;if(!['support','objection'].includes(kind))fail(400,'Choix invalide.','invalid_input');
   let depth=1;for(let row=byId.get(parentId);row;row=byId.get(row.parent_id))depth++;
   if(depth>TOPIC_LIMITS.depth)fail(409,'Cet enchaînement est déjà long : reformulez-le plutôt comme une nouvelle position.','limit_reached');
  }
  const id=random(),at=now(),row={id,parent_id:parentId,kind,content:text,basis:kind==='position'?null:basis(data),confidence:kind==='position'?confidence(data):null,created_at:at,updated_at:at};
  await run(env,'INSERT INTO personal_topic_nodes(id,user_id,topic_id,parent_id,kind,content,basis,confidence,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',id,userId,topic.id,parentId,kind,text,row.basis,row.confidence,at,at);
  await touch(env,topic.id);return json({node:nodeOut(row)},201);
 }
 async function nodeRoute(request,env,userId,topic,nodeId){
  const row=ID.test(nodeId)?await first(env,'SELECT * FROM personal_topic_nodes WHERE id=? AND topic_id=? AND user_id=?',nodeId,topic.id,userId):null;
  if(!row)fail(404,'Cet élément est introuvable.','not_found');
  if(request.method==='DELETE'){await run(env,'DELETE FROM personal_topic_nodes WHERE id=? AND user_id=?',row.id,userId);await touch(env,topic.id);return json({ok:true})}
  if(request.method!=='PUT')fail(405,'Méthode non autorisée.','method_not_allowed');
  const data=await body(request,8000),next={...row,content:content(data),basis:row.kind==='position'?null:basis(data),confidence:row.kind==='position'?confidence(data):null,updated_at:now()};
  await run(env,'UPDATE personal_topic_nodes SET content=?,basis=?,confidence=?,updated_at=? WHERE id=? AND user_id=?',next.content,next.basis,next.confidence,next.updated_at,row.id,userId);
  await touch(env,topic.id);return json({node:nodeOut(next)});
 }
 async function routes(request,env,user,path){
  if(path!=='/api/topics'&&!path.startsWith('/api/topics/'))return null;
  if(!await topicsReady(env))fail(503,'Cette page n’est pas encore disponible. Réessayez dans un instant.','topics_unavailable');
  const method=request.method,parts=path.split('/').slice(3);
  if(parts.length===0){if(method==='GET')return list(env,user.id);if(method==='POST')return createTopic(request,env,user.id);fail(405,'Méthode non autorisée.','method_not_allowed')}
  const topic=await owned(env,user.id,parts[0]);
  if(parts.length===1){
   if(method==='GET')return detail(env,user.id,topic);
   if(method==='PUT'){const value=title(await body(request,2000)),at=now();await run(env,'UPDATE personal_topics SET title=?,updated_at=? WHERE id=? AND user_id=?',value,at,topic.id,user.id);return json({topic:{...topicOut(topic),title:value,updated_at:at}})}
   if(method==='DELETE'){await run(env,'DELETE FROM personal_topics WHERE id=? AND user_id=?',topic.id,user.id);return json({ok:true})}
   fail(405,'Méthode non autorisée.','method_not_allowed');
  }
  if(parts[1]==='nodes'&&parts.length===2&&method==='POST')return createNode(request,env,user.id,topic);
  if(parts[1]==='nodes'&&parts.length===3)return nodeRoute(request,env,user.id,topic,parts[2]);
  fail(404,'Adresse introuvable.','not_found');
 }
 async function exportData(env,userId){
  if(!await topicsReady(env))return {topics:[]};
  const topics=await all(env,'SELECT * FROM personal_topics WHERE user_id=? ORDER BY created_at,id',userId),rows=await all(env,'SELECT * FROM personal_topic_nodes WHERE user_id=? ORDER BY created_at,id',userId);
  return {topics:topics.map(topic=>({...topicOut(topic),nodes:rows.filter(row=>row.topic_id===topic.id).map(nodeOut)}))};
 }
 return {routes,exportData};
}
