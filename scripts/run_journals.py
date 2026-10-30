"""Manual operations trigger only. Generation and the API key live on Cloudflare.

Automatic scheduling is a Cloudflare cron. No profile or story is returned here.
"""
import json
import os
import sys
import urllib.parse
import urllib.request


def main():
    url = os.environ['ACTIONS_ID_TOKEN_REQUEST_URL']
    url += ('&' if '?' in url else '?') + urllib.parse.urlencode({'audience': 'decode-ton-esprit-journal'})
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']})
    with urllib.request.urlopen(req, timeout=30) as response:
        token = json.load(response)['value']
    req = urllib.request.Request(os.environ['JOURNAL_URL'].rstrip('/') + '/api/internal/jobs/run',
                                 data=b'{}', headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(req, timeout=780) as response:
        result = json.load(response)
    print('Cloudflare journal run completed. No personal data returned in logs.')
    return 1 if result.get('failed', 0) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        print('Cloudflare generation unavailable. Check configuration and run status; private data omitted.')
        sys.exit(1)
