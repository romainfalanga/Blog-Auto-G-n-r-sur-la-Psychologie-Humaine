import {readFileSync, writeFileSync} from 'node:fs';
const navigation=JSON.parse(readFileSync(new URL('../data/navigation.json',import.meta.url),'utf8'));
const escape=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
if(!Array.isArray(navigation)||navigation.some(item=>!item.label||!/^\/(?!\/)/.test(item.href)))throw new Error('Invalid navigation');
const markup=navigation.map(item=>'<li><a href="'+escape(item.href)+'">'+escape(item.label)+'</a></li>').join('\n');
const output='// Generated from data/navigation.json by scripts/build_navigation.mjs.\nexport const NAVIGATION_HTML = '+JSON.stringify(markup)+';\n';
const path=new URL('../journal/src/navigation.generated.mjs',import.meta.url);
if(process.argv.includes('--check')){if(readFileSync(path,'utf8')!==output)throw new Error('Navigation is stale; run node scripts/build_navigation.mjs');}else writeFileSync(path,output);
