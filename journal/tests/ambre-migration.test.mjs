import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync,readdirSync} from 'node:fs';

test('Ambre migration preserves existing conversations/proposals and invalidates only old generated portraits',()=>{
 const db=new DatabaseSync(':memory:'),dir=new URL('../migrations/',import.meta.url);
 for(const name of readdirSync(dir).filter(n=>n.endsWith('.sql')&&n<'0012').sort())db.exec(readFileSync(new URL(name,dir),'utf8'));
 for(const id of ['a','b']){
  db.prepare("INSERT INTO users(id,email,password_hash,password_salt) VALUES(?,?,'h','s')").run(id,id+'@example.invalid');
  db.prepare("INSERT INTO profiles(user_id,self_data) VALUES(?,?)").run(id,JSON.stringify({gender:'male',self_description:'Mes déclarations restent exactes.'}));
  db.prepare('INSERT INTO memories(user_id,text) VALUES(?,?)').run(id,'Mémoire préservée.');
  db.prepare('INSERT INTO user_portraits(user_id,portrait,analyzed_version,schema_version) VALUES(?,?,1,2)').run(id,JSON.stringify({analysis_version:id==='a'?2:3}));
  db.prepare("INSERT INTO chat_turns(id,user_id,idempotency_key,payload_hash,status,credit_status,week_start,source_version,consent_revision,content,response) VALUES(?,?,?,'h','completed','consumed','2026-11-02',1,1,'Question conservée','Réponse conservée')").run('turn-'+id,id,'key-'+id);
  db.prepare("INSERT INTO chat_proposals(id,user_id,turn_id,kind,payload,status) VALUES(?,?,?,'self_profile_patch','{}',?)").run('proposal-'+id,id,'turn-'+id,id==='a'?'pending':'accepted');
 }
 const before=db.prepare('SELECT * FROM chat_proposals ORDER BY id').all(),turns=db.prepare('SELECT * FROM chat_turns ORDER BY id').all();
 db.exec('BEGIN');db.exec(readFileSync(new URL('0012_ambre_portrait_corrections.sql',dir),'utf8'));db.exec('COMMIT');
 assert.deepEqual(db.prepare('SELECT * FROM chat_proposals ORDER BY id').all(),before);
 assert.deepEqual(db.prepare('SELECT * FROM chat_turns ORDER BY id').all(),turns);
 assert.equal(db.prepare("SELECT version FROM profiles WHERE user_id='a'").get().version,2);
 assert.equal(db.prepare("SELECT version FROM profiles WHERE user_id='b'").get().version,1);
 assert.equal(db.prepare("SELECT text FROM memories WHERE user_id='a'").get().text,'Mémoire préservée.');
 db.prepare("INSERT INTO chat_proposals(id,user_id,turn_id,kind,payload) VALUES('memory-proposal','a','turn-a','memory_append','{}')").run();
 assert.deepEqual(db.prepare('PRAGMA foreign_key_check').all(),[]);
 db.prepare("DELETE FROM users WHERE id='a'").run();assert.equal(db.prepare("SELECT count(*) n FROM chat_proposals WHERE user_id='a'").get().n,0);
});
