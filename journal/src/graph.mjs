// « Ma cartographie »: every piece of information of the personal space as one graph, in the manner
// of a note graph (nodes, links, neighbours). Read-only assembly for the owner: no model is called,
// nothing is stored and nothing here is sent to Ambre.
import {CARTOGRAPHIE} from './cartographie.generated.mjs';
import {SYMPTOMES_CARTE} from './symptomes-carte.generated.mjs';
export const GRAPH_LIMIT=900;
const ROLES={father:'Père',mother:'Mère',parent:'Parent',partner:'Partenaire',child:'Enfant',sibling:'Frère ou sœur',friend:'Ami ou amie',best_friend:'Meilleur ami ou meilleure amie'};
const short=(text,max=140)=>{const value=String(text||'').replace(/\s+/g,' ').trim();return value.length>max?value.slice(0,max-1).trimEnd()+'…':value};
const parse=text=>{try{const value=JSON.parse(text||'{}');return value&&typeof value==='object'?value:{}}catch{return {}}};
const MONTHS=['janvier','février','mars','avril','mai','juin','juillet','août','septembre','octobre','novembre','décembre'];
const partial=value=>!/^\d{4}(-\d{2}){0,2}$/.test(value||'')?'':value.length===4?value:(value.length===10?Number(value.slice(8))+' ':'')+MONTHS[Number(value.slice(5,7))-1]+' '+value.slice(0,4);
const when=event=>event.date_precision==='age'?event.age_text||'':event.date_precision==='unknown'||!event.date_start?'':event.date_precision==='range'?(event.date_end?partial(event.date_start)+' – '+partial(event.date_end):'Depuis '+partial(event.date_start)):partial(event.date_start);
export function createGraphBackend({error,json}){
 // A table that does not exist yet (older database) simply contributes nothing.
 const all=async(env,sql,...args)=>{try{return (await env.DB.prepare(sql).bind(...args).all()).results}catch(e){if(/no such (?:column|table)/i.test(String(e?.message||'')))return [];throw e}};
 async function build(env,userId){
  const [profile,people,events,links,notes,topics,topicNodes,childhood,symptoms,reviews]=await Promise.all([
   all(env,'SELECT data FROM profiles WHERE user_id=?',userId),
   all(env,'SELECT id,display_name,role,custom_role,relationship_text FROM personal_people WHERE user_id=? ORDER BY created_at,id',userId),
   all(env,'SELECT id,title,description,date_precision,date_start,date_end,age_text FROM life_events WHERE user_id=? ORDER BY created_at,id',userId),
   all(env,'SELECT event_id,person_id FROM life_event_people WHERE user_id=?',userId),
   all(env,'SELECT id,event_id,person_id,text FROM personal_notes WHERE user_id=? ORDER BY created_at,id',userId),
   all(env,'SELECT id,title FROM personal_topics WHERE user_id=? ORDER BY created_at,id',userId),
   all(env,'SELECT id,topic_id,parent_id,kind,content FROM personal_topic_nodes WHERE user_id=? ORDER BY created_at,id',userId),
   all(env,'SELECT answers FROM cartographies WHERE user_id=?',userId),
   all(env,'SELECT answers FROM symptom_maps WHERE user_id=?',userId),
   all(env,"SELECT id,kind,created_at,completed_at FROM ambre_reviews WHERE user_id=? AND status='ready' ORDER BY created_at,id",userId)
  ]);
  const nodes=[],edges=[],ids=new Set();
  const node=(id,type,label,extra={})=>{if(ids.has(id)||nodes.length>=GRAPH_LIMIT)return false;ids.add(id);nodes.push({id,type,label:short(label,70)||'Sans titre',...extra});return true};
  const edge=(source,target)=>{if(ids.has(source)&&ids.has(target))edges.push({source,target})};
  const firstName=short(parse(profile[0]?.data).first_name,40);
  node('me','me',firstName||'Moi',{href:'/mon-compte/',detail:'Le centre de votre cartographie : tout ce que vous avez renseigné s’y rattache.'});
  const hub=(key,label,href,detail)=>{node('page:'+key,'page',label,{href,detail});edge('me','page:'+key)};
  hub('people','Mes proches','/mes-proches/',people.length+' proche'+(people.length>1?'s':''));
  hub('events','Ma ligne de vie','/ma-ligne-de-vie/',events.length+' moment'+(events.length>1?'s':''));
  hub('childhood','Mon enfance','/cartographie/','Ce que vous avez indiqué avoir vécu.');
  hub('symptoms','Mes symptômes','/mes-symptomes/','Les signes auxquels vous avez répondu « Oui » ou « Dans le passé ».');
  hub('topics','Mes sujets','/mes-sujets/',topics.length+' sujet'+(topics.length>1?'s':''));
  hub('reviews','Mes bilans','/mes-bilans/',reviews.length+' bilan'+(reviews.length>1?'s':''));
  for(const p of people){node('person:'+p.id,'person',p.display_name,{href:'/mes-proches/'+encodeURIComponent(p.id)+'/',detail:[p.role==='other'?p.custom_role||'Proche':ROLES[p.role]||'Proche',short(p.relationship_text)].filter(Boolean).join(' — ')});edge('page:people','person:'+p.id)}
  for(const e of events){node('event:'+e.id,'event',e.title,{href:'/ma-ligne-de-vie/#event-'+e.id,detail:[when(e),short(e.description)].filter(Boolean).join(' — ')});edge('page:events','event:'+e.id)}
  for(const l of links)edge('event:'+l.event_id,'person:'+l.person_id);
  for(const n of notes){const parent=n.event_id?'event:'+n.event_id:n.person_id?'person:'+n.person_id:null;if(!parent||!ids.has(parent))continue;if(node('note:'+n.id,'note',short(n.text,48),{detail:short(n.text,400),href:n.event_id?'/ma-ligne-de-vie/#event-'+n.event_id:'/mes-proches/'+encodeURIComponent(n.person_id)+'/'}))edge(parent,'note:'+n.id)}
  for(const t of topics){node('topic:'+t.id,'topic',t.title,{href:'/mes-sujets/'+encodeURIComponent(t.id)+'/',detail:'Un sujet de « Mes sujets ».'});edge('page:topics','topic:'+t.id)}
  for(const n of topicNodes){const type=n.kind==='position'?'position':n.kind==='support'?'support':'objection';if(node('idea:'+n.id,type,short(n.content,48),{href:'/mes-sujets/'+encodeURIComponent(n.topic_id)+'/',detail:short(n.content,400)}))edge(n.parent_id?'idea:'+n.parent_id:'topic:'+n.topic_id,'idea:'+n.id)}
  const lived=parse(childhood[0]?.answers);
  for(const f of CARTOGRAPHIE.factors)if(lived[f.id]===1||lived[f.id]===2){node('childhood:'+f.id,'childhood',f.label,{href:'/cartographie/',detail:lived[f.id]===2?'Vécu souvent, d’après vos réponses à Mon enfance.':'Vécu, d’après vos réponses à Mon enfance.'});edge('page:childhood','childhood:'+f.id)}
  const signs=parse(symptoms[0]?.answers),groups=new Map(SYMPTOMES_CARTE.groups.map(g=>[g.id,g]));
  for(const f of SYMPTOMES_CARTE.factors){const answer=signs[f.id],now=answer===1||answer===2,past=answer==='past';if(!now&&!past)continue;const group=groups.get(f.group),themeId='theme:'+f.group;
   if(group&&!ids.has(themeId)){node(themeId,'theme',group.label,{href:'/mes-symptomes/',detail:short(group.description,300)});edge('page:symptoms',themeId)}
   if(node('symptom:'+f.id,past?'past':'symptom',f.label,{href:'/mes-symptomes/',detail:past?'Dans le passé, d’après vos réponses à Mes symptômes.':'Oui, d’après vos réponses à Mes symptômes.'}))edge(ids.has(themeId)?themeId:'page:symptoms','symptom:'+f.id)}
  for(const r of reviews){const at=new Date(Number(r.completed_at||r.created_at)*1000),label=(r.kind==='point'?'Point d’étape':'Bilan initial')+(Number.isNaN(at.getTime())?'':' du '+at.getUTCDate()+' '+MONTHS[at.getUTCMonth()]+' '+at.getUTCFullYear());node('review:'+r.id,'review',label,{href:'/mes-bilans/',detail:'Un bilan préparé par Ambre.'});edge('page:reviews','review:'+r.id)}
  return {nodes,edges,truncated:nodes.length>=GRAPH_LIMIT};
 }
 // The dashboard only needs a few counts; the readiness in /api/me gives the others.
 async function summary(env,userId){
  const one=async sql=>Number((await all(env,sql,userId))[0]?.n||0);
  const [topics,reviews,running,graph]=await Promise.all([one('SELECT count(*) n FROM personal_topics WHERE user_id=?'),one("SELECT count(*) n FROM ambre_reviews WHERE user_id=? AND status='ready'"),one("SELECT count(*) n FROM ambre_reviews WHERE user_id=? AND status='running'"),build(env,userId)]);
  return {topics,reviews,review_running:running>0,elements:graph.nodes.filter(node=>node.type!=='me'&&node.type!=='page').length};
 }
 async function routes(request,env,user,path){
  if(path==='/api/dashboard'){if(request.method!=='GET')error(405,'Méthode non autorisée.','method_not_allowed');return json(await summary(env,user.id))}
  if(path!=='/api/graph')return null;
  if(request.method!=='GET')error(405,'Méthode non autorisée.','method_not_allowed');
  return json(await build(env,user.id));
 }
 return {routes,build,summary};
}
