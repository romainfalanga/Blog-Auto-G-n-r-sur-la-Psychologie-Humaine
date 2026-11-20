// Ma ligne de vie: a moment is dated by a day, a month, a year or a period; a period may still be
// running (no end) and its two ends may have different precisions.
import test from 'node:test';
import assert from 'node:assert/strict';
import {setup} from './v4-R-helper.mjs';

let serial=0;
const create=(s,dates)=>s.call('/api/life-events','POST',{creation_id:'moment-dates-'+(++serial),title:'Un moment',description:'Un récit fictif.',kind:'other',...dates});

test('periods: running, mixed precisions, and an end before the start refused',async()=>{
 const s=setup();
 const running=await create(s,{date_precision:'range',date_start:'2024-04',date_end:''});assert.equal(running.status,201,JSON.stringify(running));assert.deepEqual([running.event.date_start,running.event.date_end],['2024-04','']);
 const mixed=await create(s,{date_precision:'range',date_start:'2019-06',date_end:'2019'});assert.equal(mixed.status,201,'June 2019 to some time in 2019');
 const days=await create(s,{date_precision:'range',date_start:'2022-07-10',date_end:'2022-07-24'});assert.equal(days.status,201);
 assert.equal((await create(s,{date_precision:'range',date_start:'2022',date_end:'2021-12'})).status,400);
 assert.equal((await create(s,{date_precision:'range',date_start:'',date_end:'2021'})).status,400);
 assert.equal((await create(s,{date_precision:'range',date_start:'2021-02-30',date_end:''})).status,400);
});

test('a day, a month and a year keep exactly their precision',async()=>{
 const s=setup();
 for(const [precision,value] of [['day','2020-02-29'],['month','2020-02'],['year','2020']]){const r=await create(s,{date_precision:precision,date_start:value,date_end:'2030'});assert.equal(r.status,201);assert.deepEqual([r.event.date_start,r.event.date_end],[value,'']);}
 assert.equal((await create(s,{date_precision:'day',date_start:'2021-02-29'})).status,400);
 assert.equal((await create(s,{date_precision:'month',date_start:'2020'})).status,400);
});
