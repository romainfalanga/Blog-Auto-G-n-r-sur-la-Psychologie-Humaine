"""Run the historic editorial pipeline for private portrait and story jobs.

OpenRouter key stays in Cloudflare. GitHub OIDC authorizes bounded per-job calls.
Private context exists only in process memory, never in Git, artifacts or logs.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from generate_personal_articles import generate


def internal(path, payload):
    token_url = os.environ['ACTIONS_ID_TOKEN_REQUEST_URL']
    token_url += ('&' if '?' in token_url else '?') + urllib.parse.urlencode({'audience':'decode-ton-esprit-journal'})
    req = urllib.request.Request(token_url, headers={'Authorization':'Bearer '+os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']})
    with urllib.request.urlopen(req, timeout=30) as response:
        token = json.load(response)['value']
    base = os.environ.get('JOURNAL_URL','https://decodetonesprit.fr').rstrip('/')
    req = urllib.request.Request(base+'/api/internal/jobs'+path,
        data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
    with urllib.request.urlopen(req,timeout=210) as response:
        return json.load(response)


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
    except Exception:
        print('Private workflow unavailable. Check configuration and job state; personal data omitted.')
        sys.exit(1)
