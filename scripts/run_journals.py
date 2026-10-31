"""Run the historic editorial pipeline for private portrait and story jobs.

OpenRouter key stays in Cloudflare. GitHub OIDC authorizes bounded per-job calls.
Private context exists only in process memory, never in Git, artifacts or logs.
"""
import json
import os
import sys
import time
import re
import urllib.parse
import urllib.request
import urllib.error

from generate_personal_articles import generate


class TransportFailure(RuntimeError):
    pass


def internal(path, payload):
    token_url = os.environ['ACTIONS_ID_TOKEN_REQUEST_URL']
    token_url += ('&' if '?' in token_url else '?') + urllib.parse.urlencode({'audience':'decode-ton-esprit-journal'})
    req = urllib.request.Request(token_url, headers={'Authorization':'Bearer '+os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            token = json.load(response)['value']
    except urllib.error.HTTPError as exc:
        raise TransportFailure('oidc_http_'+str(exc.code)) from None
    base = os.environ.get('JOURNAL_URL','https://decode-journal.romainfalanga83.workers.dev').rstrip('/')
    req = urllib.request.Request(base+'/api/internal/jobs'+path,
        data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+token,
                 'User-Agent':'DecodeTonEsprit-GitHubActions/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=210) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Only fixed operational codes may enter public Actions logs.
        known={'Workflow non autorisé.':'workflow_claims','Identité automatique requise.':'identity_missing',
               'Signature invalide.':'signature_invalid','Identité invalide.':'identity_invalid'}
        try:
            raw=exc.read(8192).decode('utf-8','replace')
            code=known.get(json.loads(raw).get('error'),'request_failed')
        except Exception:
            # Only a short HTML title from a failed empty claim can identify
            # an infrastructure refusal; never log bodies, prompts or headers.
            code='response_unavailable'
            if path=='/claim':
                title=re.search(r'<title[^>]*>([^<]{0,100})</title>',raw if 'raw' in locals() else '',re.I)
                if title:
                    code+='_'+re.sub(r'[^a-zA-Z0-9_-]','_',title.group(1))[:100]
        raise TransportFailure('worker_http_'+str(exc.code)+'_'+code) from None


def proxy(job, payload):
    return internal('/'+job['id']+'/llm',payload)


def main():
    completed=failed=0
    # Last job has at most 20 x 75s provider calls, keeping its 30min lease
    # and the complete execution below GitHub's 50min timeout.
    deadline=time.monotonic()+20*60
    for _ in range(20):
        if time.monotonic()>deadline:
            break
        jobs=internal('/claim',{}).get('jobs',[])
        if not jobs:
            break
        job=jobs[0]
        try:
            result=generate(job,proxy)
            internal('/'+job['id']+'/complete',result)
            completed+=1
        except Exception:
            failed+=1
            try:
                internal('/'+job['id']+'/fail',{'lease_token':job['lease_token'],'reason':'generation_failed'})
            except Exception:
                pass
    print(f'Private jobs completed={completed}, failed={failed}. Personal data omitted.')
    return 1 if failed else 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except TransportFailure as exc:
        print('Private workflow transport: '+str(exc)+'. Personal data omitted.')
        sys.exit(1)
    except Exception:
        print('Private workflow unavailable. Check configuration and job state; personal data omitted.')
        sys.exit(1)
