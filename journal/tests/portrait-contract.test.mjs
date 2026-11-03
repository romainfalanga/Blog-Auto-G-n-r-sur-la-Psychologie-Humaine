import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validatePortraitV2} from '../src/private-context.mjs';
const fixtures=JSON.parse(fs.readFileSync(new URL('./fixtures/portrait-provenance-v2.json',import.meta.url),'utf8'));
for(const f of fixtures)test('shared Python/Worker provenance: '+f.name,()=>{
 const validate=()=>validatePortraitV2(structuredClone(f.portrait),f.sources,(_status,message)=>{throw Error(message)});
 if(f.valid)assert.doesNotThrow(validate);else assert.throws(validate);
});
