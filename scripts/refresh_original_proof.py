#!/usr/bin/env python3
"""Record a named review of corrected or new "psychology" stories in originaux-v24.json.

The script never writes an article. For each listed article of a registry
character (the 33 originals and the psychology packs), it:

1. refuses the whole run if the structure does not match the original trame
   (opening scene, four H2, three manifestations with two examples each, three
   numbered techniques with an objective and three actions, conclusion after
   `---`, a valid "Du côté des proches" mirror and reading concepts);
2. computes the body and mirror digests exactly as
   scripts/check_original_article_structure.py does;
3. updates (or creates) the proof entry: `status: revised`, techniques extracted
   from the text, the changes given on the command line, reviewer, corrector and
   date, `manual_review` recomputed, previous proof archived in `revision_history`;
4. refreshes the hashes and offsets of data/psychology_coverage_review.json for
   the same articles, or stops when a cited passage no longer exists (unless a
   replacement quote is supplied with --coverage-evidence);
5. recomputes the summary and validation blocks of the proof file.

`--withdraw` archives the proof of an article that has been removed from the
corpus (see scripts/original_arc_episode.py remove). `--regenerate` rebuilds the
derived projections and `--run-checks` executes every --check of the project.

A proof records a human or agent review that actually happened. The reviewer
must differ from the corrector; the structural checks never replace that reading.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_article_intros import DEFINITION_MARKERS, HEADING, WORD, normalized, plain_text, sections, sentences  # noqa: E402
import check_original_article_structure as checker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROOF_FILE = 'editorial/reviews/originaux-v24.json'
COVERAGE_FILE = 'data/psychology_coverage_review.json'
GOAL = re.compile(r"\b(?:objectif|le but|il s.agit|l.idée est)\b", re.I)
OBJECTIVE_LABEL = re.compile(r"^(?:objectif|but)\s*[:：]\s*(.+)$", re.I)

# Every command of netlify.toml, of the CI workflow and of
# docs/verification-tests-lot3-2026-11-09.json, except the Hugo rendering.
CHECKS = [
    ['node', 'scripts/build_navigation.mjs', '--check'],
    ['node', 'scripts/build_psychology_catalog.mjs', '--check'],
    [sys.executable, 'scripts/build_article_intros.py', '--check'],
    [sys.executable, 'scripts/build_examples.py', '--check'],
    [sys.executable, 'scripts/clinical_editorial.py', 'check'],
    [sys.executable, 'scripts/check_editorial_pack.py'],
    [sys.executable, 'scripts/integrate_editorial_pack.py', '--check'],
    [sys.executable, 'scripts/psychology_editorial.py', 'check'],
    [sys.executable, 'scripts/check_original_article_structure.py', '--audit', PROOF_FILE, '--require-reviewed'],
    [sys.executable, 'scripts/build_coverage_status.py', '--check'],
]
REGENERATE = [
    [sys.executable, 'scripts/build_examples.py'],          # also rebuilds data/article_intros.json
    [sys.executable, 'scripts/build_coverage_status.py'],
]


class RefreshError(ValueError):
    pass


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def repository_head(root: Path) -> str | None:
    try:
        return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def head_body(root: Path, relative: str) -> str | None:
    try:
        raw = subprocess.run(['git', '-C', str(root), 'show', 'HEAD:' + relative], check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    split = raw.split('---', 2)
    return split[2].strip() if len(split) == 3 else None


# --------------------------------------------------------------------------
# Structure


def item_text(line: str) -> str:
    match = checker.ITEM.match(line)
    return plain_text(match.group(2)) if match else ''


def extract_technique(unit: dict) -> dict:
    """Objective and peer actions, taken verbatim (Markdown removed) from the text."""
    lines = unit['text'].splitlines()
    objective = None
    for line in lines:
        match = OBJECTIVE_LABEL.match(checker.clean_line(line))
        if match:
            objective = plain_text(match.group(1))
            break
    if objective is None:
        for paragraph in re.split(r'\n\s*\n', unit['text']):
            if checker.ITEM.match(paragraph.strip().splitlines()[0] if paragraph.strip() else ''):
                continue
            found = next((s for s in sentences(plain_text(paragraph)) if GOAL.search(s)), None)
            if found:
                objective = found
                break
    cleaned = [checker.clean_line(line) for line in lines]
    start = next((i + 1 for i, line in enumerate(cleaned) if checker.EXERCISE.match(line)), 0)
    actions, base = [], None
    for line in lines[start:]:
        match = checker.ITEM.match(line)
        if match:
            indent = len(match.group(1))
            base = indent if base is None else base
            if indent == base:
                actions.append(plain_text(match.group(2)))
            continue
        if line.strip() and actions:
            if len(actions) == 2 and checker.THIRD.match(checker.clean_line(line)):
                actions.append(plain_text(line))
            if not line.startswith((' ', '\t')):
                break
    return {'title': unit['title'], 'objective': objective or '', 'actions': actions}


def strict_problems(root: Path, relative: str, body: str, character_id: str) -> tuple[list[str], dict, list[dict]]:
    """Refusal reasons for the original trame; empty when the article conforms."""
    mirror_digest, mirror_error = checker.read_mirror(root, character_id, relative)
    result = checker.analyze(body, None, mirror_digest)
    problems = list(result['mechanical_problems']) + list(result['prose_signals_requiring_review'])
    if mirror_error:
        problems.append(mirror_error)
    else:
        key = Path(relative).relative_to('content').with_suffix('').as_posix()
        paragraphs = json.loads((root / 'data/article_mirrors' / (character_id + '.json')).read_text())[key]
        count = len(WORD.findall(' '.join(paragraphs)))
        if not 1 <= len(paragraphs) <= 2 or not 40 <= count <= 160 or sum(map(len, paragraphs)) > 1200 \
                or any('\n' in p or re.search(r'<[^>]+>', p) for p in paragraphs):
            problems.append(f'« Du côté des proches » : un ou deux paragraphes de 40 à 160 mots requis ({count} mots)')
    h2 = checker.parts(body, checker.H2)
    if len(h2) == 4:
        manifestations = checker.parts(h2[1]['text'], checker.H3)
        if len(manifestations) != 3:
            problems.append(f'Trois manifestations (H3) requises, {len(manifestations)} trouvées')
        techniques_all = checker.parts(h2[2]['text'], checker.H3)
        numbered = [t for t in techniques_all if checker.NUMBERED.match(t['title'])]
        if [int(checker.NUMBERED.match(t['title']).group(1)) for t in numbered] != [1, 2, 3]:
            problems.append('Trois techniques numérotées 1., 2., 3. requises dans la troisième partie')
        if not any(marker in normalized(plain_text(h2[0]['title'])) for marker in DEFINITION_MARKERS):
            problems.append('La première partie doit expliquer le concept (« Qu’est-ce que … ? »)')
    first_heading = HEADING.search(body)
    opening = len(WORD.findall(plain_text(body[:first_heading.start()]))) if first_heading else 0
    if opening < 180:
        problems.append(f'Scène d’ouverture trop courte ({opening} mots, minimum 180)')
    techniques = []
    if len(h2) == 4:
        units = [t for t in checker.parts(h2[2]['text'], checker.H3) if checker.NUMBERED.match(t['title'])]
        techniques = [extract_technique(unit) for unit in units]
        for technique, metrics in zip(techniques, result['techniques']):
            if not technique['objective'].strip():
                problems.append('Objectif introuvable : ' + technique['title'])
            if len(technique['actions']) < 3:
                problems.append('Trois actions introuvables : ' + technique['title'])
            if not metrics['goal_label_or_phrase']:
                problems.append('Objectif non formulé : ' + technique['title'])
    key = Path(relative).relative_to('content').with_suffix('').as_posix()
    topics = json.loads((root / 'data/parcours_topics.json').read_text()).get(key)
    if not isinstance(topics, list) or not 2 <= len(topics) <= 3:
        problems.append('Deux ou trois repères de lecture requis dans data/parcours_topics.json')
    return problems, result, techniques


def structure_block(result: dict) -> dict:
    return {
        'h2_titles': result['h2_titles'],
        'manifestation_headings': [m['heading'] for m in result['manifestations']],
        'example_markers': [m['example_markers'] for m in result['manifestations']],
        'technique_headings': [t['heading'] for t in result['techniques']],
        'action_markers': [t['peer_action_markers'] for t in result['techniques']],
        'additional_notes': result['additional_technique_notes'],
        'mechanical_problems': result['mechanical_problems'],
        'prose_signals_requiring_review': result['prose_signals_requiring_review'],
    }


# --------------------------------------------------------------------------
# Coverage evidence (data/psychology_coverage_review.json)


def evidence_records(node, in_links=False):
    """Yield (record, blocking) for every record citing an article body."""
    if isinstance(node, dict):
        if 'article_id' in node and isinstance(node.get('evidence'), list) and 'source_body_sha256' in node:
            yield node, in_links
        for key, value in node.items():
            yield from evidence_records(value, in_links or key == 'links')
    elif isinstance(node, list):
        for value in node:
            yield from evidence_records(value, in_links)


def refresh_coverage(doc: dict, article_id: str, raw_body: str, replacements: dict, today: str):
    """Recompute offsets and digests; return (touched, errors, warnings)."""
    touched, errors, warnings = [], [], []
    for record, blocking in evidence_records(doc):
        if record['article_id'] != article_id:
            continue
        pattern = record.get('pattern_id', '?')
        if blocking and pattern in replacements:
            record['evidence'] = [{'quote': quote} for quote in replacements[pattern]]
        missing = []
        for item in record['evidence']:
            quote = item.get('quote', '')
            start = raw_body.find(quote) if quote else -1
            if start < 0:
                missing.append(quote)
                continue
            item['start'], item['end'] = start, start + len(quote)
            item['quote'] = quote
            # Keep the field order of the existing records.
            ordered = {'quote': item['quote'], 'start': item['start'], 'end': item['end']}
            item.clear()
            item.update(ordered)
        label = f'{article_id} [{pattern}]'
        if missing and blocking:
            errors.append(f'Citation de couverture absente du nouveau corps, {label} : « {missing[0][:90]}… » '
                          f'(fournir --coverage-evidence {pattern}::"citation exacte" ou retirer la liaison après relecture)')
            continue
        if missing:
            record['evidence_status'] = f'citation absente depuis la révision du {today} ; à relire'
            warnings.append(f'Citation de contexte absente, marquée à relire : {label}')
            continue
        record.pop('evidence_status', None)
        record['source_body_sha256'] = sha256_text(raw_body)
        touched.append(pattern)
    return touched, errors, warnings


# --------------------------------------------------------------------------
# Proof document


def recompute(root: Path, doc: dict) -> dict:
    rows = doc['articles']
    report = checker.collect(root, {row['path']: row for row in rows})
    summary = report['summary']
    doc['scope'] = f"{summary['articles']} articles des {summary['characters']} personnages originaux et des packs psychologie"
    doc['summary'] = {
        'articles': summary['articles'], 'characters': summary['characters'],
        'owner_status': dict(sorted(Counter(row.get('status') for row in rows).items(), key=lambda kv: -kv[1])),
        'current_manual_reviews': summary['current_manual_reviews'],
        'stale_manual_reviews': summary['stale_manual_reviews'],
        'prose_signals_without_current_review': summary['prose_signals_without_current_review'],
        'independent_final_qa_articles': sum('independent_final_qa' in row for row in rows),
        'articles_with_flags': sum(bool(row.get('flags')) for row in rows),
        'bodies_changed_vs_head': sum(bool(row.get('body_changed_vs_head')) for row in rows),
        'current_body_reviews': summary['current_body_reviews'],
        'current_mirror_reviews': summary['current_mirror_reviews'],
        'stale_mirror_reviews': summary['stale_mirror_reviews'],
    }
    passed = (not report['dataset_errors'] and summary['mechanical_failures'] == 0
              and summary['current_body_reviews'] == summary['articles'] == summary['current_mirror_reviews'])
    validation = doc.setdefault('validation', {})
    validation['final_structure_check'] = {
        'command': f'python scripts/check_original_article_structure.py --audit {PROOF_FILE} --require-reviewed',
        'summary': summary,
        'checker_sha256': hashlib.sha256((ROOT / 'scripts/check_original_article_structure.py').read_bytes()).hexdigest(),
        'status': 'passed' if passed else 'failed',
    }
    if root.resolve() == ROOT:
        from check_editorial_pack import inspect
        pack = inspect()
        validation['editorial_pack'] = {key: pack[key] for key in ('public_articles_checked', 'pack', 'characters', 'articles', 'words', 'errors')}
    return report


def load_review_source(root: Path, value: str | None, reviewer: str) -> dict | None:
    if not value:
        return None
    path = (root / value).resolve()
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise RefreshError('Rapport de relecture introuvable : ' + value)
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if isinstance(data, dict):
        named = data.get('reviewer') or data.get('auditor')
        if data.get('reviewer') and data['reviewer'] != reviewer:
            raise RefreshError(f'Le rapport {value} est signé par {named}, pas par {reviewer}')
        if data.get('status') and data['status'] not in ('passed', 'accepted', 'no_blocking_finding'):
            raise RefreshError(f'Le rapport {value} n’est pas accepté (status {data["status"]})')
    return {'path': path.relative_to(root.resolve()).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()}


def parse_coverage_evidence(values: list[str], paths: list[str]) -> dict:
    result: dict[str, dict[str, list[str]]] = {}
    for value in values:
        parts = value.split('::', 2)
        if len(parts) == 3 and parts[0].startswith('content/'):
            path, pattern, quote = parts
        elif len(parts) >= 2 and len(paths) == 1:
            path, (pattern, quote) = paths[0], value.split('::', 1)
        else:
            raise RefreshError('Format --coverage-evidence : [CHEMIN::]PATTERN_ID::citation exacte')
        if path not in paths or not quote.strip():
            raise RefreshError('Citation de couverture pour un article non relu : ' + value[:80])
        key = Path(path).relative_to('content').with_suffix('').as_posix()
        result.setdefault(key, {}).setdefault(pattern, []).append(quote)
    return result


def refresh(root: Path, paths: list[str], *, reviewer: str, corrector: str, changes: dict[str, list[str]],
            mirror_changes: dict[str, list[str]], review_source: str | None = None, batch: str | None = None,
            today: str | None = None, coverage_evidence: dict | None = None, withdraw: dict[str, str] | None = None,
            dry_run: bool = False) -> dict:
    root = root.resolve()
    today = today or date.today().isoformat()
    withdraw = withdraw or {}
    if not reviewer.strip() or not corrector.strip():
        raise RefreshError('--reviewer et --corrector sont requis')
    if normalized(reviewer.strip()) == normalized(corrector.strip()):
        raise RefreshError('Le vérificateur doit être distinct du correcteur ou du rédacteur')
    if not paths and not withdraw:
        raise RefreshError('Aucun article à relire ou à retirer')
    proof_path = root / PROOF_FILE
    doc = json.loads(proof_path.read_text())
    coverage_path = root / COVERAGE_FILE
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else None
    registry = json.loads((root / 'scripts/personnages.json').read_text())['personnages']
    names = {c['prenom']: c['id'] for c in registry}
    source = load_review_source(root, review_source, reviewer)
    rows = {row['path']: row for row in doc['articles']}
    errors, warnings, refreshed = [], [], []
    coverage_evidence = coverage_evidence or {}

    for relative in withdraw:
        if (root / relative).exists():
            errors.append(f'{relative} : l’article existe encore ; le retirer d’abord (original_arc_episode.py remove)')
        if relative not in rows:
            errors.append(f'{relative} : aucune preuve à retirer')
        key = Path(relative).relative_to('content').with_suffix('').as_posix()
        if coverage and any(record['article_id'] == key for record, _ in evidence_records(coverage)):
            errors.append(f'{relative} : cité dans {COVERAGE_FILE} ; relire et déplacer ces preuves avant le retrait')

    for relative in paths:
        if relative in withdraw:
            errors.append(f'{relative} : à la fois relu et retiré')
            continue
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root / 'content/posts'):
            errors.append(f'{relative} : article introuvable dans content/posts')
            continue
        frontmatter, body = checker.read_article(path)
        if re.search(r'^clinical_code:', frontmatter, re.M):
            errors.append(f'{relative} : parcours clinique, à relire par clinical_editorial.py')
            continue
        name = checker.scalar(frontmatter, 'personnage')
        if name not in names:
            errors.append(f'{relative} : personnage absent de scripts/personnages.json ({name}) ; intégrer d’abord le parcours')
            continue
        if not changes.get(relative):
            errors.append(f'{relative} : au moins une modification relue (--change) est requise')
        character_id = names[name]
        problems, result, techniques = strict_problems(root, relative, body, character_id)
        if problems:
            errors.extend(f'{relative} : {problem}' for problem in problems)
            continue
        mirror_digest, _ = checker.read_mirror(root, character_id, relative)
        previous = rows.get(relative)
        if previous and previous.get('current_mirror_sha256') != mirror_digest and not mirror_changes.get(relative) \
                and previous.get('body_sha256') == result['body_sha256']:
            errors.append(f'{relative} : le miroir a changé ; décrire la modification avec --mirror-change')
        raw_body = path.read_text().split('---', 2)[2]
        article_id = Path(relative).relative_to('content').with_suffix('').as_posix()
        if coverage is not None:
            touched, coverage_errors, coverage_warnings = refresh_coverage(
                coverage, article_id, raw_body, coverage_evidence.get(article_id, {}), today)
            errors.extend(coverage_errors)
            warnings.extend(coverage_warnings)
        else:
            touched = []
        refreshed.append((relative, character_id, body, result, techniques, mirror_digest, previous, touched))

    if errors:
        raise RefreshError('\n'.join(errors))

    head = repository_head(root)
    for relative, character_id, body, result, techniques, mirror_digest, previous, touched in refreshed:
        entry = dict(previous or {})
        history = list(entry.get('revision_history', []))
        if previous:
            archived = {'archived_on': today,
                        'previous_body_sha256': previous.get('body_sha256'),
                        'previous_mirror_sha256': previous.get('mirror_sha256'),
                        'previous_status': previous.get('status'),
                        'previous_reviewer': previous.get('reviewer', previous.get('historical_batch_review')),
                        'previous_changes': previous.get('changes', [])}
            qa = entry.get('independent_final_qa')
            if qa and (qa.get('body_sha256') != result['body_sha256'] or qa.get('mirror_sha256') != mirror_digest):
                archived['superseded_independent_final_qa'] = entry.pop('independent_final_qa')
            history.append(archived)
        before = head_body(root, relative)
        entry.update({
            'batch': entry.get('batch', batch or f'revue-{today}'),
            'character_id': character_id,
            'path': relative,
            'body_changed_vs_head': before is None or before != body,
            'flags': entry.get('flags', []),
            'body_sha256': result['body_sha256'],
            'current_body_sha256': result['body_sha256'],
            'mirror_sha256': mirror_digest,
            'current_mirror_sha256': mirror_digest,
            'mirror_reviewed': True,
            'status': 'revised',
            'techniques': techniques,
            'review_file': PROOF_FILE,
            'historical_batch_review': entry.get('historical_batch_review'),
            'changes': list(changes[relative]),
            'reviewer': reviewer,
            'corrector': corrector,
            'reviewed_on': today,
        })
        if mirror_changes.get(relative):
            entry['mirror_changes'] = list(mirror_changes[relative])
        if source:
            entry['review_source'] = source
        if touched:
            entry['coverage_evidence_refreshed'] = sorted(set(touched))
        if history:
            entry['revision_history'] = history
        entry['manual_review'] = checker.review_proof(entry, result['body_sha256'], mirror_digest)
        entry['structure'] = structure_block(result)
        rows[relative] = entry

    withdrawn = doc.setdefault('withdrawn_articles', []) if withdraw else doc.get('withdrawn_articles')
    for relative, reason in withdraw.items():
        row = rows.pop(relative)
        withdrawn.append({'path': relative, 'withdrawn_on': today, 'reason': reason, 'reviewer': reviewer,
                          'corrector': corrector, 'last_proof': row})
    order = [row['path'] for row in doc['articles'] if row['path'] in rows]
    order += [path for path in rows if path not in order]
    doc['articles'] = [rows[path] for path in order]
    doc.setdefault('refresh_log', []).append({
        'at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'reviewed_on': today,
        'reviewer': reviewer, 'corrector': corrector, 'repository_head': head,
        'paths': [item[0] for item in refreshed], 'withdrawn': sorted(withdraw),
        **({'review_source': source} if source else {}),
    })
    report = recompute(root, doc)
    for row in report['articles']:
        if row['path'] in {item[0] for item in refreshed} and not row['manual_review']['current']:
            raise RefreshError(row['path'] + ' : la preuve recalculée n’est pas courante')
    if coverage is not None and any(item[7] for item in refreshed):
        coverage.setdefault('refresh_log', []).append({
            'reviewed_on': today, 'reviewer': reviewer,
            'articles': {Path(item[0]).relative_to('content').with_suffix('').as_posix(): sorted(set(item[7]))
                         for item in refreshed if item[7]},
            'hash_method': 'SHA256 du corps brut Markdown : texte.split("---", 2)[2], sans strip ; offsets en caractères Python dans ce même corps.',
        })
    if not dry_run:
        proof_path.write_text(dump(doc))
        if coverage is not None:
            coverage_path.write_text(dump(coverage))
    return {'refreshed': [item[0] for item in refreshed], 'withdrawn': sorted(withdraw),
            'warnings': warnings, 'summary': doc['summary'],
            'structure_status': doc['validation']['final_structure_check']['status']}


def run(commands: list[list[str]], root: Path) -> list[dict]:
    results = []
    for command in commands:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
        results.append({'command': ' '.join(Path(c).name if c == sys.executable else c for c in command),
                        'exit_code': completed.returncode,
                        'output': (completed.stdout + completed.stderr).strip().splitlines()[-3:]})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='*', help='content/posts/<catégorie>/<slug>.md relus')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reviewer', required=True, help='Identité du vérificateur indépendant')
    parser.add_argument('--corrector', required=True, help='Identité du correcteur ou du rédacteur')
    parser.add_argument('--change', action='append', default=[], help='Modification relue, pour tous les chemins')
    parser.add_argument('--changes-json', type=Path, help='{"chemin": ["modification", …]} pour plusieurs articles')
    parser.add_argument('--mirror-change', action='append', default=[], help='Modification relue de « Du côté des proches »')
    parser.add_argument('--review-source', help='Rapport du vérificateur (chemin relatif), lié par SHA256')
    parser.add_argument('--coverage-evidence', action='append', default=[],
                        help='[CHEMIN::]PATTERN_ID::citation exacte remplaçant une preuve de couverture disparue')
    parser.add_argument('--withdraw', action='append', default=[], metavar='CHEMIN::RAISON',
                        help='Archiver la preuve d’un article retiré du corpus')
    parser.add_argument('--batch', help='Étiquette de lot pour une nouvelle entrée')
    parser.add_argument('--date', help='Date de relecture ISO (défaut : aujourd’hui)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--regenerate', action='store_true', help='Régénérer intros, mots-clés, exemples et couverture')
    parser.add_argument('--run-checks', action='store_true', help='Exécuter tous les contrôles --check du projet')
    args = parser.parse_args()
    paths = [Path(p).resolve().relative_to(args.root.resolve()).as_posix() if Path(p).is_absolute() else Path(p).as_posix()
             for p in args.paths]
    changes = {path: list(args.change) for path in paths}
    if args.changes_json:
        for path, values in json.loads(args.changes_json.read_text()).items():
            changes.setdefault(path, []).extend(values)
    mirror_changes = {path: list(args.mirror_change) for path in paths if args.mirror_change}
    withdraw = {}
    for value in args.withdraw:
        path, _, reason = value.partition('::')
        if not reason.strip():
            parser.error('--withdraw CHEMIN::RAISON')
        withdraw[path] = reason.strip()
    try:
        result = refresh(args.root, paths, reviewer=args.reviewer, corrector=args.corrector, changes=changes,
                         mirror_changes=mirror_changes, review_source=args.review_source, batch=args.batch,
                         today=args.date, coverage_evidence=parse_coverage_evidence(args.coverage_evidence, paths),
                         withdraw=withdraw, dry_run=args.dry_run)
    except RefreshError as error:
        print('Refus, aucune preuve modifiée :\n' + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    status = 0
    if args.regenerate and not args.dry_run:
        for row in run(REGENERATE, args.root):
            print(json.dumps(row, ensure_ascii=False))
            status |= bool(row['exit_code'])
    if args.run_checks:
        for row in run(CHECKS, args.root):
            print(json.dumps(row, ensure_ascii=False))
            status |= bool(row['exit_code'])
    return int(status)


if __name__ == '__main__':
    raise SystemExit(main())
