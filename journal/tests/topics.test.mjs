// Migration 0035, « Mes sujets »: topics, positions and arguments, owner-only, never sent to Ambre.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {setup,database} from './v4-R-helper.mjs';
import {createTopicsBackend,TOPIC_LIMITS} from '../src/topics.mjs';

const SQL=readFileSync(new URL('../migrations/0035_topics.sql',import.meta.url),'utf8');
function harness(options){
 const s=setup(options);let serial=0;
 const backend=createTopicsBackend({error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),body:async req=>req.json(),now:s.now,random:()=>'t'+(++serial)});
 const call=async(path,method='GET',data,user='a')=>{const req=new Request('https://example.test'+path,{method,...(data!==undefined?{body:JSON.stringify(data),headers:{'Content-Type':'application/json'}}:{})});
  try{const res=await backend.routes(req,s.env,{id:user},new URL(req.url).pathname);return res?{status:res.status,...await res.json()}:null}catch(e){return {status:e.status,error:e.message,code:e.code}}};
 return {...s,backend,call};
}

test('0035: additive only',()=>{
 const body=SQL.replace(/--[^\n]*/g,'');assert.doesNotMatch(body,/\b(DROP|ALTER|RENAME|UPDATE|INSERT|TRIGGER)\b/i);
 const db=database('0035');db.exec(SQL);assert.throws(()=>db.exec(SQL),/already exists/);db.close();
});

test('a topic holds positions, reasons, objections and replies; counts and detail follow',async()=>{
 const h=harness();
 assert.deepEqual((await h.call('/api/topics')).topics,[]);
 const topic=(await h.call('/api/topics','POST',{title:'  Écologie  '})).topic;assert.equal(topic.title,'Écologie');
 const position=(await h.call('/api/topics/'+topic.id+'/nodes','POST',{content:'Il faut réduire la voiture en ville.',confidence:4,basis:'fact'})).node;
 assert.deepEqual([position.kind,position.confidence,position.basis,position.parent_id],['position',4,null,null]);
 const reason=(await h.call('/api/topics/'+topic.id+'/nodes','POST',{parent_id:position.id,kind:'support',content:'La pollution baisse.',basis:'hearsay',confidence:3})).node;
 assert.deepEqual([reason.kind,reason.basis,reason.confidence],['support','hearsay',null]);
 const objection=(await h.call('/api/topics/'+topic.id+'/nodes','POST',{parent_id:position.id,kind:'objection',content:'Certains n’ont pas d’autre moyen.'})).node;
 const reply=(await h.call('/api/topics/'+topic.id+'/nodes','POST',{parent_id:objection.id,kind:'objection',content:'D’où le besoin de transports.'})).node;
 const list=(await h.call('/api/topics')).topics;assert.deepEqual([list[0].positions,list[0].arguments],[1,3]);
 const detail=await h.call('/api/topics/'+topic.id);assert.deepEqual(detail.nodes.map(n=>n.id).sort(),[position.id,reason.id,objection.id,reply.id].sort());
 const edited=(await h.call('/api/topics/'+topic.id+'/nodes/'+reason.id,'PUT',{content:'La pollution baisse nettement.',basis:'fact'})).node;assert.deepEqual([edited.content,edited.basis],['La pollution baisse nettement.','fact']);
 assert.equal((await h.call('/api/topics/'+topic.id,'PUT',{title:'Écologie en ville'})).topic.title,'Écologie en ville');
 // Deleting an objection removes its reply; deleting the topic removes everything.
 await h.call('/api/topics/'+topic.id+'/nodes/'+objection.id,'DELETE');
 assert.deepEqual((await h.call('/api/topics/'+topic.id)).nodes.map(n=>n.id).sort(),[position.id,reason.id].sort());
 assert.equal((await h.backend.exportData(h.env,'a')).topics[0].nodes.length,2);
 await h.call('/api/topics/'+topic.id,'DELETE');
 assert.equal(h.raw.prepare('SELECT count(*) n FROM personal_topic_nodes').get().n,0);
});

test('validation, limits and isolation between accounts',async()=>{
 const h=harness();
 assert.equal((await h.call('/api/topics','POST',{title:'   '})).status,400);
 assert.equal((await h.call('/api/topics','POST',{title:'x'.repeat(TOPIC_LIMITS.title+1)})).status,400);
 const topic=(await h.call('/api/topics','POST',{title:'Travail'})).topic,base='/api/topics/'+topic.id;
 assert.equal((await h.call(base+'/nodes','POST',{content:''})).status,400);
 assert.equal((await h.call(base+'/nodes','POST',{content:'x'.repeat(TOPIC_LIMITS.content+1)})).status,400);
 let node=(await h.call(base+'/nodes','POST',{content:'Position',confidence:9})).status;assert.equal(node,400);
 let parent=(await h.call(base+'/nodes','POST',{content:'Position'})).node;
 assert.equal((await h.call(base+'/nodes','POST',{parent_id:parent.id,kind:'position',content:'x'})).status,400);
 assert.equal((await h.call(base+'/nodes','POST',{parent_id:parent.id,kind:'support',content:'x',basis:'rumeur'})).status,400);
 assert.equal((await h.call(base+'/nodes','POST',{parent_id:'inconnu',kind:'support',content:'x'})).status,404);
 for(let depth=1;depth<TOPIC_LIMITS.depth;depth++)parent=(await h.call(base+'/nodes','POST',{parent_id:parent.id,kind:'objection',content:'niveau '+depth})).node;
 assert.deepEqual([(await h.call(base+'/nodes','POST',{parent_id:parent.id,kind:'objection',content:'trop profond'})).status],[409]);
 // Another account sees nothing and cannot read, change or delete.
 assert.deepEqual((await h.call('/api/topics','GET',undefined,'b')).topics,[]);
 for(const [path,method,data] of [[base,'GET'],[base,'PUT',{title:'Vol'}],[base,'DELETE'],[base+'/nodes','POST',{content:'x'}],[base+'/nodes/'+parent.id,'PUT',{content:'x'}],[base+'/nodes/'+parent.id,'DELETE']])assert.equal((await h.call(path,method,data,'b')).status,404,method+' '+path);
 assert.equal((await h.call(base)).nodes.length,TOPIC_LIMITS.depth);
 // Deleting the account erases its topics.
 h.raw.prepare("DELETE FROM users WHERE id='a'").run();assert.equal(h.raw.prepare('SELECT count(*) n FROM personal_topics').get().n,0);
 assert.equal(await h.call('/api/other'),null);
});

test('before the migration the page answers « pas encore disponible » and the export stays empty',async()=>{
 const h=harness({until:'0035'});
 assert.deepEqual([(await h.call('/api/topics')).status,(await h.call('/api/topics')).code],[503,'topics_unavailable']);
 assert.deepEqual(await h.backend.exportData(h.env,'a'),{topics:[]});
});
