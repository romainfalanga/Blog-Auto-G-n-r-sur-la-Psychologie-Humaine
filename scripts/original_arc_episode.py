#!/usr/bin/env python3
"""Add or withdraw an episode of one of the twenty historical "psychology" arcs.

The article Markdown is written by its writer (never by this script). The
writer also saves a sidecar `editorial/originals/<id>/<slug>.json`:

    {"schema_version": 1, "character_id": "adrien", "author": "redacteur_adrien",
     "article": "content/posts/<catégorie>/<slug>.md",
     "concept": "Notion distincte des onze épisodes précédents",
     "resume_narratif": "Deux ou trois phrases fidèles à la scène d’ouverture.",
     "evolution": "Décision prise à la fin et conséquence réelle.",
     "topics": ["Deux", "ou trois repères"],
     "mirror": ["Un ou deux paragraphes « Du côté des proches » (40 à 160 mots)."]}

`add` inserts the sidecar data in the shared files: matrix row
(scripts/matrice_combinaisons.json), entry and `nb_articles` of
content/personnages/<id>.md, data/article_mirrors/<id>.json and
data/parcours_topics.json, then regenerates the registry (integrate_editorial_pack),
intros, keywords, examples and coverage state. `remove` withdraws an article
(surplus episode) everywhere, archives its proof and adds a 301 redirect.
`check` verifies that every sidecar, page and registry history agree.

The proof of review is recorded afterwards with scripts/refresh_original_proof.py.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from build_article_intros import HEADING, WORD, normalized, plain_text, sentences  # noqa: E402
from check_editorial_pack import CATEGORIES  # noqa: E402
import check_original_article_structure as checker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIDECARS = 'editorial/originals'
PAGE_LABELS = {'cat1_pensees': 'Biais cognitif', 'cat2_emotions': 'Émotion', 'cat3_schemas': 'Schéma répétitif'}
FOLDERS = {folder: (key, name) for key, (name, folder) in CATEGORIES.items()}
ARC_LENGTH = 12


class EpisodeError(ValueError):
    pass


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def dump_topics(topics: dict) -> str:
    rows = [f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(values, ensure_ascii=False)}'
            for key, values in sorted(topics.items())]
    return '{\n' + ',\n'.join(rows) + '\n}\n'


def article_id(relative: str) -> str:
    return Path(relative).relative_to('content').with_suffix('').as_posix()


def url(relative: str) -> str:
    return '/' + article_id(relative) + '/'


def pack_names(root: Path) -> set[str]:
    names = set()
    for path in (root / 'data').glob('editorial_pack_v*.json'):
        names |= {c['prenom'] for c in json.loads(path.read_text()).get('characters', [])}
    return names


def historical_characters(root: Path) -> dict[str, dict]:
    registry = json.loads((root / 'scripts/personnages.json').read_text())['personnages']
    packs = pack_names(root)
    return {c['prenom']: c for c in registry if c['prenom'] not in packs}


def read_front(path: Path) -> tuple[dict, str]:
    split = path.read_text().split('---', 2)
    if len(split) != 3 or split[0].strip():
        raise EpisodeError(f'Frontmatter absent : {path}')
    meta = yaml.safe_load(split[1])
    if not isinstance(meta, dict):
        raise EpisodeError(f'Frontmatter invalide : {path}')
    return meta, split[2].strip()


def page_entry(relative: str, meta: dict, category_key: str, resume: str, evolution: str) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    link = esc(url(relative))
    return '\n'.join([
        '  <div class="parcours-entry">',
        f'    <div class="parcours-cat">{PAGE_LABELS[category_key]}</div>',
        f'    <h3 class="parcours-title"><a href="{link}">{esc(meta["title"])}</a></h3>',
        f'    <div class="parcours-date">{str(meta["date"])[:10]}</div>',
        f'    <p class="parcours-resume">{esc(resume)}</p>',
        '    <div class="parcours-evolution">',
        '      <span class="evolution-label">Évolution</span>',
        f'      <p>{esc(evolution)}</p>',
        '    </div>',
        f'    <a href="{link}" class="parcours-link">Lire l\'article complet &rarr;</a>',
        '  </div>',
    ])


ENTRY = re.compile(r'  <div class="parcours-entry">\n.*?\n  </div>(?=\n)', re.S)


def page_entries(text: str) -> list[re.Match]:
    return list(ENTRY.finditer(text))


def entry_date(block: str) -> str:
    match = re.search(r'<div class="parcours-date">([^<]+)</div>', block)
    return match.group(1) if match else ''


def set_count(text: str, count: int) -> str:
    updated, n = re.subn(r'(?m)^nb_articles: \d+$', f'nb_articles: {count}', text, count=1)
    if n != 1:
        raise EpisodeError('nb_articles absent de la page personnage')
    return updated


def validate_sidecar(root: Path, sidecar: dict, character: dict, relative: str) -> list[str]:
    errors = []
    if sidecar.get('schema_version') != 1 or sidecar.get('character_id') != character['id'] or sidecar.get('article') != relative:
        errors.append('Fiche annexe : schema_version 1, character_id et article exacts requis')
    for key in ('author', 'concept', 'resume_narratif', 'evolution'):
        if not isinstance(sidecar.get(key), str) or not sidecar[key].strip() or '\n' in sidecar[key]:
            errors.append(f'Fiche annexe : {key} requis (une ligne)')
    topics = sidecar.get('topics')
    if not isinstance(topics, list) or not 2 <= len(topics) <= 3 or any(
            not isinstance(t, str) or not t.strip() or t != t.strip() for t in topics) or len({normalized(t) for t in topics}) != len(topics or []):
        errors.append('Fiche annexe : deux ou trois repères de lecture distincts')
    mirror = sidecar.get('mirror')
    if not isinstance(mirror, list) or not 1 <= len(mirror) <= 2 or any(
            not isinstance(p, str) or not p.strip() or '\n' in p or re.search(r'<[^>]+>', p) for p in mirror) \
            or not 40 <= len(WORD.findall(' '.join(mirror))) <= 160 or sum(map(len, mirror)) > 1200:
        errors.append('Fiche annexe : « Du côté des proches » en un ou deux paragraphes de 40 à 160 mots')
    return errors


def validate_article(root: Path, relative: str, characters: dict) -> tuple[dict, str, dict, str]:
    path = root / relative
    if not path.is_file() or path.parent.parent != root / 'content/posts' or path.parent.name not in FOLDERS:
        raise EpisodeError(f'{relative} : article introuvable dans un répertoire de catégorie')
    meta, body = read_front(path)
    errors = []
    category_key, category_name = FOLDERS[path.parent.name]
    for key in ('title', 'description', 'slug', 'personnage', 'sujet', 'contexte'):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            errors.append(f'{key} requis')
    if meta.get('slug') != path.stem:
        errors.append('slug égal au nom du fichier requis')
    if meta.get('categories') != [category_name]:
        errors.append(f'categories doit valoir ["{category_name}"]')
    if meta.get('draft') is not False:
        errors.append('draft: false requis')
    if not isinstance(meta.get('tags'), list) or len(meta['tags']) < 3:
        errors.append('au moins trois tags')
    if 'date' not in meta:
        errors.append('date requise')
    if meta.get('editorial_pack') or meta.get('clinical_code'):
        errors.append('épisode de pack ou clinique : utiliser psychology_editorial.py ou clinical_editorial.py')
    if len(meta.get('description', '')) > 160:
        errors.append('description de 160 caractères maximum')
    character = characters.get(meta.get('personnage'))
    if not character:
        errors.append('personnage absent des vingt parcours historiques')
    result = checker.analyze(body)
    errors += result['mechanical_problems'] + result['prose_signals_requiring_review']
    first = HEADING.search(body)
    if not first or len(WORD.findall(plain_text(body[:first.start()]))) < 180:
        errors.append('scène d’ouverture de 180 mots minimum, sans intertitre')
    if errors:
        raise EpisodeError(f'{relative} : ' + '; '.join(errors))
    return meta, body, character, category_key


def rebuild(root: Path) -> None:
    for command in (['scripts/integrate_editorial_pack.py'], ['scripts/build_examples.py'],
                    ['scripts/build_coverage_status.py']):
        completed = subprocess.run([sys.executable, *command], cwd=root, capture_output=True, text=True)
        print(f'{command[0]} → {completed.returncode} {(completed.stdout + completed.stderr).strip().splitlines()[-1:]}')
        if completed.returncode:
            raise EpisodeError(f'Régénération interrompue par {command[0]} :\n{completed.stdout}{completed.stderr}')


def add(root: Path, relative: str, *, allow_over: bool = False, run_rebuild: bool = True, dry_run: bool = False) -> dict:
    characters = historical_characters(root)
    meta, body, character, category_key = validate_article(root, relative, characters)
    sidecar_path = root / SIDECARS / character['id'] / (meta['slug'] + '.json')
    if not sidecar_path.is_file():
        raise EpisodeError(f'Fiche annexe absente : {sidecar_path.relative_to(root)}')
    sidecar = json.loads(sidecar_path.read_text())
    errors = validate_sidecar(root, sidecar, character, relative)
    if errors:
        raise EpisodeError('\n'.join(errors))
    ident = article_id(relative)
    slug = meta['slug']
    matrix_path = root / 'scripts/matrice_combinaisons.json'
    matrix = json.loads(matrix_path.read_text())
    if any(row['slug'] == slug for row in matrix['articles']):
        raise EpisodeError(f'Slug déjà présent dans la matrice : {slug}')
    history = character.get('historique_articles', [])
    if len(history) + 1 > ARC_LENGTH and not allow_over:
        raise EpisodeError(f'{character["id"]} compterait {len(history) + 1} épisodes (douze au maximum)')
    mirrors_path = root / 'data/article_mirrors' / (character['id'] + '.json')
    mirrors = json.loads(mirrors_path.read_text())
    topics_path = root / 'data/parcours_topics.json'
    topics = json.loads(topics_path.read_text())
    if ident in mirrors or ident in topics:
        raise EpisodeError(f'Miroir ou repères déjà présents pour {ident}')
    page_path = root / 'content/personnages' / (character['id'] + '.md')
    page = page_path.read_text()
    if url(relative) in page:
        raise EpisodeError('Épisode déjà présent sur la page personnage')
    # Matrix: after the last historical row, as integrate_editorial_pack.py orders it.
    row = {'date': str(meta['date'])[:10], 'category_key': category_key, 'category_name': CATEGORIES[category_key][0],
           'sujet': meta['sujet'], 'contexte': meta['contexte'], 'angle': None, 'profil': None,
           'prenom': character['prenom'], 'age': character.get('tranche_age'), 'title': meta['title'], 'slug': slug,
           'tags': meta['tags'], 'resume_narratif': sidecar['resume_narratif'], 'evolution': sidecar['evolution']}
    position = next((i for i, r in enumerate(matrix['articles']) if r.get('editorial_pack')), len(matrix['articles']))
    matrix['articles'].insert(position, row)
    mirrors[ident] = sidecar['mirror']
    mirrors = dict(sorted(mirrors.items()))
    topics[ident] = sidecar['topics']
    entries = page_entries(page)
    block = page_entry(relative, meta, category_key, sidecar['resume_narratif'], sidecar['evolution'])
    later = next((m for m in entries if entry_date(m.group(0)) > row['date']), None)
    if later:
        page = page[:later.start()] + block + '\n' + page[later.start():]
    elif entries:
        page = page[:entries[-1].end()] + '\n' + block + page[entries[-1].end():]
    else:
        raise EpisodeError('Liste du parcours introuvable sur la page personnage')
    page = set_count(page, len(page_entries(page)))
    if not dry_run:
        matrix_path.write_text(dump(matrix))
        mirrors_path.write_text(dump(mirrors))
        topics_path.write_text(dump_topics(topics))
        page_path.write_text(page)
        if run_rebuild:
            rebuild(root)
    return {'added': relative, 'character': character['id'], 'episodes': len(page_entries(page)),
            'next': f'python3 scripts/refresh_original_proof.py {relative} --reviewer VERIFICATEUR --corrector '
                    f'{sidecar["author"]} --change "…" --review-source RAPPORT --regenerate --run-checks'}


def remove(root: Path, relative: str, *, reason: str, reviewer: str, corrector: str, redirect_to: str | None,
           run_rebuild: bool = True) -> dict:
    from refresh_original_proof import RefreshError, refresh, evidence_records
    path = root / relative
    meta, _ = read_front(path)
    characters = historical_characters(root)
    character = characters.get(meta.get('personnage'))
    if not character or meta.get('editorial_pack') or meta.get('clinical_code'):
        raise EpisodeError('Seul un épisode d’un parcours historique peut être retiré ici')
    ident = article_id(relative)
    coverage_path = root / 'data/psychology_coverage_review.json'
    if coverage_path.exists():
        cited = [record.get('pattern_id', '?') for record, _ in evidence_records(json.loads(coverage_path.read_text()))
                 if record['article_id'] == ident]
        if cited:
            raise EpisodeError(f'{relative} est cité dans {coverage_path.name} ({", ".join(cited)}) ; '
                               'relire et déplacer ces preuves vers un autre épisode avant le retrait')
    if not reason.strip():
        raise EpisodeError('Raison du retrait requise')
    page_path = root / 'content/personnages' / (character['id'] + '.md')
    page = page_path.read_text()
    match = next((m for m in page_entries(page) if f'href="{url(relative)}"' in m.group(0)), None)
    if not match:
        raise EpisodeError('Entrée absente de la page personnage')
    original = path.read_text()
    page = page[:match.start()] + page[match.end() + 1:]
    page = set_count(page, len(page_entries(page)))
    matrix_path = root / 'scripts/matrice_combinaisons.json'
    matrix = json.loads(matrix_path.read_text())
    before = len(matrix['articles'])
    matrix['articles'] = [r for r in matrix['articles'] if not (r['slug'] == meta['slug'] and r.get('prenom') == character['prenom'])]
    if len(matrix['articles']) != before - 1:
        raise EpisodeError('Ligne de matrice introuvable')
    mirrors_path = root / 'data/article_mirrors' / (character['id'] + '.json')
    mirrors = json.loads(mirrors_path.read_text())
    mirrors.pop(ident, None)
    topics_path = root / 'data/parcours_topics.json'
    topics = json.loads(topics_path.read_text())
    topics.pop(ident, None)
    path.unlink()
    try:
        refresh(root, [], reviewer=reviewer, corrector=corrector, changes={}, mirror_changes={},
                withdraw={relative: reason})
    except RefreshError:
        path.write_text(original)
        raise
    page_path.write_text(page)
    matrix_path.write_text(dump(matrix))
    mirrors_path.write_text(dump(mirrors))
    topics_path.write_text(dump_topics(topics))
    sidecar = root / SIDECARS / character['id'] / (meta['slug'] + '.json')
    if sidecar.exists():
        sidecar.unlink()
    if redirect_to:
        netlify = root / 'netlify.toml'
        block = (f'\n# Épisode retiré le {__import__("datetime").date.today().isoformat()} : {reason[:80]}\n'
                 f'[[redirects]]\n  from = "{url(relative)}"\n  to = "{redirect_to}"\n  status = 301\n  force = true\n')
        if url(relative) not in netlify.read_text():
            netlify.write_text(netlify.read_text().rstrip('\n') + '\n' + block)
    if run_rebuild:
        rebuild(root)
    return {'removed': relative, 'character': character['id'], 'episodes': len(page_entries(page))}


def check(root: Path) -> list[str]:
    errors = []
    characters = historical_characters(root)
    by_id = {c['id']: c for c in characters.values()}
    topics = json.loads((root / 'data/parcours_topics.json').read_text())
    matrix = {r['slug']: r for r in json.loads((root / 'scripts/matrice_combinaisons.json').read_text())['articles']}
    for character in characters.values():
        page = (root / 'content/personnages' / (character['id'] + '.md')).read_text()
        entries = page_entries(page)
        count = re.search(r'(?m)^nb_articles: (\d+)$', page)
        history = character.get('historique_articles', [])
        if not count or int(count.group(1)) != len(entries) or len(entries) != len(history):
            errors.append(f'{character["id"]} : nb_articles, entrées de la page et historique du registre diffèrent')
        for row in history:
            folder = CATEGORIES[row['category']][1]
            if f'href="/posts/{folder}/{row["slug"]}/"' not in page:
                errors.append(f'{character["id"]} : épisode {row["slug"]} absent de la page')
    for path in sorted((root / SIDECARS).glob('*/*.json')):
        sidecar = json.loads(path.read_text())
        character = by_id.get(path.parent.name)
        relative = sidecar.get('article', '')
        label = path.relative_to(root).as_posix()
        if not character or not (root / relative).is_file():
            errors.append(f'{label} : personnage ou article absent')
            continue
        errors += [f'{label} : {e}' for e in validate_sidecar(root, sidecar, character, relative)]
        ident = article_id(relative)
        mirrors = json.loads((root / 'data/article_mirrors' / (character['id'] + '.json')).read_text())
        row = matrix.get(Path(relative).stem)
        if mirrors.get(ident) != sidecar.get('mirror') or topics.get(ident) != sidecar.get('topics') or not row \
                or row.get('resume_narratif') != sidecar.get('resume_narratif') or row.get('evolution') != sidecar.get('evolution'):
            errors.append(f'{label} : fiche annexe non intégrée ou modifiée depuis (lancer original_arc_episode.py add ou corriger)')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    p_add = sub.add_parser('add', help='Intégrer un nouvel épisode écrit et sa fiche annexe')
    p_add.add_argument('article')
    p_add.add_argument('--allow-over-12', action='store_true')
    p_add.add_argument('--no-rebuild', action='store_true')
    p_add.add_argument('--dry-run', action='store_true')
    p_remove = sub.add_parser('remove', help='Retirer un épisode surnuméraire')
    p_remove.add_argument('article')
    p_remove.add_argument('--reason', required=True)
    p_remove.add_argument('--reviewer', required=True)
    p_remove.add_argument('--corrector', required=True)
    p_remove.add_argument('--redirect-to', help='Cible de la redirection 301 ajoutée à netlify.toml, ex. /personnages/camille/')
    p_remove.add_argument('--no-rebuild', action='store_true')
    sub.add_parser('check', help='Vérifier pages, historiques et fiches annexes')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == 'add':
            result = add(root, args.article, allow_over=args.allow_over_12, run_rebuild=not args.no_rebuild, dry_run=args.dry_run)
        elif args.command == 'remove':
            result = remove(root, args.article, reason=args.reason, reviewer=args.reviewer, corrector=args.corrector,
                            redirect_to=args.redirect_to, run_rebuild=not args.no_rebuild)
        else:
            errors = check(root)
            for error in errors:
                print(error)
            print(f'{len(historical_characters(root))} parcours historiques vérifiés ; {len(errors)} écart(s).')
            return int(bool(errors))
    except (EpisodeError, ValueError) as error:
        print('Refus :\n' + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
