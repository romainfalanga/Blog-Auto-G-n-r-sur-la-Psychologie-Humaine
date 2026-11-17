// Hugo has no SSR adapter. Only Mon enfance needs this response transform.
const INITIAL_URL='https://decode-journal.romainfalanga83.workers.dev/api/initial-state?page=cartographie';
const safeJSON=value=>JSON.stringify(value).replace(/[<>&\u2028\u2029]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));

export default async function personalPageData(request,context) {
  if(request.method!=='GET'||!['/cartographie','/cartographie/'].includes(new URL(request.url).pathname))return;
  const headers=new Headers(),cookie=request.headers.get('cookie');
  if(cookie)headers.set('Cookie',cookie);
  const initial=Promise.resolve().then(()=>fetch(INITIAL_URL,{method:'GET',headers,cache:'no-store',redirect:'error'}))
    .then(async response=>{
      if(!response.ok)return null;
      const value=await response.json();
      return value?.page==='/cartographie/'&&value.responses&&value.responses['/api/me']?value:null;
    }).catch(()=>null);
  const response=await context.next();
  const data=await initial,outHeaders=new Headers(response.headers);
  for(const name of ['Cache-Control','CDN-Cache-Control','Netlify-CDN-Cache-Control'])outHeaders.set(name,'private, no-store');
  const vary=outHeaders.get('Vary');
  if(!vary?.split(',').some(value=>['cookie','*'].includes(value.trim().toLowerCase())))outHeaders.set('Vary',vary?vary+', Cookie':'Cookie');
  for(const name of ['ETag','Last-Modified','Content-Length'])outHeaders.delete(name);
  let body=response.body;
  if(data&&response.ok&&(outHeaders.get('Content-Type')||'').toLowerCase().startsWith('text/html')){
    const html=await response.text();
    body=html.replace(/<\/head\s*>/i,()=>'<script id="cartographie-initial" type="application/json">'+safeJSON(data)+'</script></head>');
    outHeaders.delete('Content-Encoding');
  }
  // Failed preload leaves the ordinary page intact; the browser can retry its APIs.
  return new Response(body,{status:response.status,statusText:response.statusText,headers:outHeaders});
}
