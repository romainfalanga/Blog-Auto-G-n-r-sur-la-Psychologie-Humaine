// Lot R, F22: migration history baseline (ops/d1-baseline.sql) and additive 0022.
import test from 'node:test';
import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {migrationNames} from './v4-R-helper.mjs';

const read=name=>readFileSync(new URL('../migrations/'+name,import.meta.url),'utf8');
const baseline=readFileSync(new URL('../ops/d1-baseline.sql',import.meta.url),'utf8');
function migrated(until){const db=new DatabaseSync(':memory:');db.exec('PRAGMA foreign_keys=ON');for(const name of migrationNames().filter(n=>n<until))db.exec(read(name));return db}
// What `wrangler d1 migrations list` reports: files not recorded in d1_migrations.
const pending=db=>{const done=new Set(db.prepare("SELECT name FROM sqlite_master WHERE name='d1_migrations'").get()?db.prepare('SELECT name FROM d1_migrations').all().map(r=>r.name):[]);return migrationNames().filter(n=>!done.has(n))};

test('F22: before the baseline every migration looks pending and a replay fails on 0005',()=>{
 const db=migrated('0022');
 assert.equal(pending(db).length,migrationNames().length);
 assert.throws(()=>db.exec(read('0005_personal_context_characters.sql')),/duplicate column/);
});

test('F22: after the baseline only 0022 is pending; applying it is additive and the baseline is idempotent',()=>{
 const db=migrated('0022');
 db.exec("INSERT INTO users(id,email,password_hash,password_salt) VALUES('u','u@example.test','h','s');INSERT INTO chat_memory_jobs(user_id,updated_at) VALUES('u',1234);INSERT INTO rate_limits(key,count,expires_at) VALUES('k',1,5)");
 db.exec(baseline);
 assert.deepEqual(pending(db),['0022_robustness.sql']);
 db.exec(read('0022_robustness.sql'));db.prepare('INSERT INTO d1_migrations(name) VALUES(?)').run('0022_robustness.sql');
 assert.deepEqual(pending(db),[]);
 db.exec(baseline);
 assert.equal(db.prepare('SELECT count(*) n FROM d1_migrations').get().n,migrationNames().length);
 assert.equal(db.prepare("SELECT queued_at FROM chat_memory_jobs WHERE user_id='u'").get().queued_at,1234);
 assert.equal(db.prepare('SELECT count(*) n FROM rate_limits').get().n,1);
 for(const name of ['provider_state','provider_global_budget','service_metrics','chat_engine_live','chat_turns_owner_live','chat_traces_detailed'])assert.ok(db.prepare('SELECT 1 FROM sqlite_master WHERE name=?').get(name),name);
});

test('F22: a missing migration is never hidden by the baseline',()=>{
 const db=migrated('0021');db.exec(baseline);
 assert.deepEqual(pending(db),['0021_change_plan.sql','0022_robustness.sql']);
 const partial=migrated('0018');partial.exec(baseline);
 assert.deepEqual(pending(partial).slice(0,2),['0018_ambre_reviews.sql','0019_ambre_engine.sql']);
});

test('F22: the 0022 file only adds objects (no DROP, no DELETE, no table rewrite)',()=>{
 const sql=read('0022_robustness.sql').replace(/--[^\n]*/g,'');
 assert.doesNotMatch(sql,/\b(DROP|DELETE|RENAME)\b/i);
 for(const statement of sql.split(';').map(s=>s.trim()).filter(Boolean))assert.match(statement,/^(CREATE (TABLE|INDEX) IF NOT EXISTS|ALTER TABLE \w+ ADD COLUMN|UPDATE chat_memory_jobs SET queued_at=updated_at WHERE queued_at=0)/,statement);
});
