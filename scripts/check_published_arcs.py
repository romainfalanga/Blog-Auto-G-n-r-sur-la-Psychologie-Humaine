#!/usr/bin/env python3
"""Compare published clinical story text with the local Hugo render.

Run Hugo after integration, then pass character IDs. This checks publication
fidelity, not clinical accuracy or independent editorial review.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
        'link', 'meta', 'param', 'source', 'track', 'wbr'}


class StoryText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts = []
        self.found = 0

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get('class', '').split()
        if not self.depth and tag == 'div' and 'article-content' in classes:
            self.depth = 1
            self.found += 1
        elif self.depth and tag not in VOID:
            self.depth += 1
        if self.depth and tag in {'p', 'div', 'h2', 'h3', 'li', 'br', 'hr'}:
            self.parts.append(' ')

    def handle_startendtag(self, tag, attrs):
        if self.depth:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if self.depth and tag not in VOID:
            self.depth -= 1
            self.parts.append(' ')

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def story_text(html):
    parser = StoryText()
    parser.feed(html)
    if parser.found != 1:
        raise ValueError(f'Expected one article-content block; found {parser.found}')
    text = re.sub(r'\s+', ' ', ''.join(parser.parts)).strip()
    if 'Du côté des proches' not in text:
        raise ValueError('Missing relatives section')
    return text


def check(item):
    url, local = item
    try:
        expected = story_text(local.read_text())
        request = urllib.request.Request(url, headers={'User-Agent': 'DecodePublicationCheck/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            actual = story_text(response.read().decode('utf-8'))
            status = response.status
            final_url = response.url
        return {'url': url, 'status': status, 'final_url': final_url,
                'ok': status == 200 and actual == expected,
                'expected_text_sha256': hashlib.sha256(expected.encode()).hexdigest(),
                'served_text_sha256': hashlib.sha256(actual.encode()).hexdigest()}
    except Exception as error:
        return {'url': url, 'ok': False, 'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('characters', nargs='+')
    parser.add_argument('--origin', default='https://decodetonesprit.fr')
    parser.add_argument('--public-dir', type=Path, default=ROOT / 'public')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    items = []
    for character in args.characters:
        if not re.fullmatch(r'[a-z0-9-]+', character):
            parser.error('Invalid character ID')
        manuscript = json.loads((ROOT / 'editorial/clinical' / f'{character}.json').read_text())
        if len(manuscript['episodes']) != 12:
            parser.error(f'{character}: expected 12 episodes')
        for episode in manuscript['episodes']:
            matches = list((ROOT / 'content/posts').rglob(episode['slug'] + '.md'))
            if len(matches) != 1:
                parser.error(f'{character}: missing or ambiguous source {episode["slug"]}')
            route = matches[0].relative_to(ROOT / 'content').with_suffix('')
            items.append((f'{args.origin.rstrip("/")}/{route.as_posix()}/',
                          args.public_dir / route / 'index.html'))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(check, items))
    report = {'checked': len(results), 'passed': sum(r['ok'] for r in results), 'results': results}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}))
    for result in results:
        if not result['ok']:
            print(json.dumps(result, ensure_ascii=False))
    return 0 if all(r['ok'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
