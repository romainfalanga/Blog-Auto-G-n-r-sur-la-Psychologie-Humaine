// « Ma cartographie »: the personal graph is assembled from the owner's own rows only.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup} from './v4-R-helper.mjs';
import {createGraphBackend} from '../src/graph.mjs';
import {createTopicsBackend} from '../src/topics.mjs';
import {CARTOGRAPHIE} from '../src/cartographie.generated.mjs';
import {SYMPTOMES_CARTE} from '../src/symptomes-carte.generated.mjs';

const utils=s=>{let n=0;return {error(status,message,code){throw Object.assign(new Error(message),{status,code})},json:(v,status=200)=>new Response(JSON.stringify(v),{status}),body:async r=>r.json(),now:s.now,random:()=>'g'+(++n)}};

test('people, moments, notes, topics, childhood, symptoms and reviews become one linked graph, per account',async()=>{
 const s=setup(),u=utils(s),graph=createGraphBackend(u),topics=createTopicsBackend(u);
 const person=(await s.call('/api/people','POST',{creation_id:'graph-person-1',display_name:'Léa',role:'friend',relationship_text:'Une amie fictive.'})).person;
 const event=(await s.call('/api/life-events','POST',{creation_id:'graph-event-1',title:'Déménagement',description:'Un récit fictif.',kind:'turning_point',date_precision:'range',date_start:'2019-03',date_end:'',people_ids:[person.id]})).event;
 await s.call('/api/people/'+person.id+'/notes','POST',{creation_id:'graph-note-0001',text:'Une note fictive sur Léa.'});
 const post=(path,data)=>topics.routes(new Request('https://example.test'+path,{method:'POST',body:JSON.stringify(data)}),s.env,{id:'a'},path).then(r=>r.json());
 const topic=(await post('/api/topics',{title:'Écologie'})).topic,position=(await post('/api/topics/'+topic.id+'/nodes',{content:'Une position fictive.'})).node;
 await post('/api/topics/'+topic.id+'/nodes',{parent_id:position.id,kind:'objection',content:'Une objection fictive.'});
 const factor=CARTOGRAPHIE.factors[0],sign=SYMPTOMES_CARTE.factors[0],past=SYMPTOMES_CARTE.factors[1];
 s.raw.prepare("INSERT INTO cartographies(user_id,answers,updated_at) VALUES('a',?,1)").run(JSON.stringify({[factor.id]:1}));
 s.raw.prepare("INSERT INTO symptom_maps(user_id,answers,updated_at) VALUES('a',?,1)").run(JSON.stringify({[sign.id]:1,[past.id]:'past',[SYMPTOMES_CARTE.factors[2].id]:0}));
 s.raw.prepare("INSERT INTO ambre_reviews(id,user_id,kind,status,content,idempotency_key,created_at,completed_at) VALUES('r1','a','initial','ready','Bilan.','k1',1794830400,1794830400)").run();
 const g=await graph.build(s.env,'a'),ids=new Set(g.nodes.map(n=>n.id)),has=(a,b)=>g.edges.some(e=>e.source===a&&e.target===b||e.source===b&&e.target===a);
 for(const id of ['me','page:people','page:events','page:childhood','page:symptoms','page:topics','page:reviews','person:'+person.id,'event:'+event.id,'topic:'+topic.id,'idea:'+position.id,'childhood:'+factor.id,'theme:'+sign.group,'symptom:'+sign.id,'symptom:'+past.id,'review:r1'])assert.ok(ids.has(id),id);
 assert.ok(has('event:'+event.id,'person:'+person.id),'a moment is linked to the people it involves');assert.ok(has('page:people','person:'+person.id));assert.ok(has('topic:'+topic.id,'idea:'+position.id));assert.ok(has('theme:'+sign.group,'symptom:'+sign.id));
 assert.equal(g.nodes.find(n=>n.id==='symptom:'+past.id).type,'past');assert.equal(ids.has('symptom:'+SYMPTOMES_CARTE.factors[2].id),false,'« Non » is not drawn');
 assert.match(g.nodes.find(n=>n.id==='event:'+event.id).detail,/^Depuis mars 2019/);assert.equal(g.nodes.filter(n=>n.type==='note').length,1);
 assert.ok(g.edges.every(e=>ids.has(e.source)&&ids.has(e.target)));
 // Another account only sees its own empty frame.
 const other=await graph.build(s.env,'b');assert.deepEqual(other.nodes.map(n=>n.type).sort(),['me','page','page','page','page','page','page'].sort());
 const response=await graph.routes(new Request('https://example.test/api/graph'),s.env,{id:'a'},'/api/graph');assert.equal((await response.json()).nodes.length,g.nodes.length);
 assert.equal(await graph.routes(new Request('https://example.test/api/other'),s.env,{id:'a'},'/api/other'),null);
});
