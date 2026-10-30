"""Export the existing public articles as immutable onboarding references.

Does not modify articles, character pages or their URLs. The Worker validates every
chosen article against this generated catalogue; client-supplied excerpts are ignored.
"""
import json
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build():
    examples = []
    matrix = json.loads((ROOT / 'scripts/matrice_combinaisons.json').read_text())
    summaries = {a.get('slug'): a.get('resume_narratif', '') for a in matrix.get('articles', [])}
    for path in sorted((ROOT / 'content/posts').rglob('*.md')):
        if path.name == '_index.md':
            continue
        _, front, body = path.read_text().split('---', 2)
        meta = yaml.safe_load(front)
        if meta.get('draft'):
            continue
        relative = path.relative_to(ROOT / 'content').with_suffix('')
        slug = meta.get('slug', path.stem)
        summary = summaries.get(slug, '')
        if not isinstance(summary, str):
            summary = json.dumps(summary, ensure_ascii=False)
        # Opening scene + ending retain narrative approach at a bounded context cost.
        excerpt = body.strip()
        if len(excerpt) > 10000:
            excerpt = excerpt[:7500] + '\n\n[Extrait intermédiaire omis]\n\n' + excerpt[-2000:]
        examples.append({
            'id': relative.as_posix(), 'title': meta['title'],
            'url': '/' + relative.parent.as_posix() + '/' + slug + '/',
            'character': meta.get('personnage', ''),
            'description': meta.get('description', ''),
            'body': ('Résumé du personnage FICTIF : ' + summary + '\n\n' if summary else '') + excerpt,
        })
    if not examples or len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Empty or duplicate catalogue')
    target = ROOT / 'journal/src/examples.mjs'
    target.write_text('// Generated from preserved public articles. Run python scripts/build_examples.py.\n'
                      + 'export const EXAMPLES = ' + json.dumps([{k:v for k,v in e.items() if k != 'body'} for e in examples], ensure_ascii=False, separators=(',', ':')) + ';\n')
    return examples


if __name__ == '__main__':
    examples = build()
    print(f'{len(examples)} public articles; {len({e["character"] for e in examples})} characters.')
    import sys
    if '--seed' in sys.argv:
        dest = pathlib.Path(sys.argv[sys.argv.index('--seed') + 1])
        dest.write_text(json.dumps(examples, ensure_ascii=False))
