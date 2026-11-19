// Private bootstrap for the two independent maps. Never CDN-cache user data.
const INITIAL_URL='https://decode-journal.romainfalanga83.workers.dev/api/initial-state?page=';
const safeJSON=value=>JSON.stringify(value).replace(/[<>&\u2028\u2029]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));

export default async function personalPageData(request,context) {
  const path=new URL(request.url).pathname;
  if(request.method!=='GET'||!['/cartographie','/cartographie/','/mes-symptomes','/mes-symptomes/'].includes(path))return;
  const symptoms=path.startsWith('/mes-symptomes'),page=symptoms?'/mes-symptomes/':'/cartographie/';
  const controller=new AbortController();
  let timer;
  const deadline=new Promise(resolve=>{timer=setTimeout(()=>{controller.abort();resolve(null)},300)});
  const headers=new Headers(),cookie=request.headers.get('cookie');
  if(cookie)headers.set('Cookie',cookie);
  const initial=Promise.resolve().then(()=>fetch(INITIAL_URL+(symptoms?'symptomes':'cartographie'),{method:'GET',headers,cache:'no-store',redirect:'error',signal:controller.signal}))
    .then(async response=>{
      if(!response.ok)return null;
      const value=await response.json();
      return value?.page===page&&value.responses&&value.responses['/api/me']?value:null;
    }).catch(()=>null);
  const response=await context.next();
  const data=await Promise.race([initial,deadline]);clearTimeout(timer);
  const outHeaders=new Headers(response.headers);
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
