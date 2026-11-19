// Reproducible local payload and cold-import comparison. No network or private data.
// Usage: node scripts/perf/compare-worker.mjs [before-ref] [after-ref]
import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const refs=process.argv.slice(2);if(!refs.length)refs.push('329891e','3ad3232');
const probe=String.raw`
import {performance} from 'node:perf_hooks';
import {gzipSync} from 'node:zlib';
const started=performance.now();
const ui=await import('./journal/src/ui.mjs');
const importMs=performance.now()-started;
const bytes=x=>Buffer.byteLength(x||'');
const gz=x=>gzipSync(x||'').length;
const assets=Object.values(ui.APP_ASSETS||{});
console.log(JSON.stringify({ui_import_ms:importMs,html_bytes:bytes(ui.HTML),html_gzip_bytes:gz(ui.HTML),assets:assets.map(x=>({path:x.path,bytes:bytes(x.body),gzip_bytes:gz(x.body)})),first_visit_gzip_bytes:gz(ui.HTML)+assets.reduce((sum,x)=>sum+gz(x.body),0),repeat_visit_gzip_bytes:gz(ui.HTML)}));
`;
const results=[];
for(const ref of refs){
 const dir=mkdtempSync(join(tmpdir(),'dte-perf-'));
 try{
  const sha=execFileSync('git',['rev-parse','--verify',ref+'^{commit}'],{encoding:'utf8'}).trim();
  const archive=execFileSync('git',['archive',sha,'journal/src'],{maxBuffer:32*1024*1024});
  execFileSync('tar',['-x','-C',dir],{input:archive});
  const runs=Array.from({length:7},()=>JSON.parse(execFileSync(process.execPath,['--input-type=module','-e',probe],{cwd:dir,encoding:'utf8'})));
  const times=runs.map(x=>x.ui_import_ms).sort((a,b)=>a-b);
  results.push({ref,sha,...runs[0],ui_import_ms:undefined,ui_import_median_ms:times[3],ui_import_range_ms:[times[0],times[6]],runs:7});
 }finally{rmSync(dir,{recursive:true,force:true})}
}
console.log(JSON.stringify({kind:'local payload comparison; gzip level default; no network, LCP or D1 latency measurement',results},null,2));
