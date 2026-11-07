"""Report published coverage separately from planned clinical mappings."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def build():
    clinical = read('data/clinical_catalog.json')
    releases = read('editorial/clinical/releases.json')['arcs']
    characters = {c['id']: c for c in [*clinical['characters'],
                  *read('editorial/clinical/additional_characters.json')]}
    index = read('data/article_keywords.json')
    patterns = read('data/psychology_catalog.json')['patterns']
    original_reviews_path = ROOT / 'editorial/reviews/originaux-v24.json'
    original_reviews = (json.loads(original_reviews_path.read_text()).get('articles', [])
                        if original_reviews_path.exists() else [])
    reviewed_originals = 0
    for row in original_reviews:
        body = (ROOT / row['path']).read_text().split('---', 2)[2].strip()
        assert hashlib.sha256(body.encode()).hexdigest() == row['body_sha256'], row['path'] + ': relecture périmée'
        reviewed_originals += row['status'] in ('revised', 'already_complete')
    groups = {g['id']: g for g in clinical['groups']}
    codes = {c['code']: c for c in clinical['codes']}
    published = {}
    for ident, release in releases.items():
        character = characters[ident]
        raw = (ROOT / 'editorial/clinical' / (ident + '.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == release['sha256'], ident + ': relecture périmée'
        articles = sorted(key for key, row in index.items() if row.get('character_id') == ident)
        assert len(articles) == 12, ident + ': les douze épisodes doivent être indexés'
        published[character['code']] = {'character_id': ident, 'name': character['name'],
                                       'reviewed_on': release['date'], 'articles': articles}

    def top_group(group_id):
        if group_id is None:
            return None
        while groups[group_id]['parent_id']:
            group_id = groups[group_id]['parent_id']
        return group_id

    code_rows = []
    for code, row in codes.items():
        status = ('published_primary_arc' if code in published else
                  'pending_editorial_case' if row['status'] == 'parcours' else
                  'editorial_mapping_only' if row['status'] == 'rattache' else
                  'no_standalone_arc_planned')
        code_rows.append({'code': code, 'label': row['label'], 'group_id': top_group(row['group_id']),
                          'editorial_status': row['status'], 'coverage_status': status,
                          'reason': row['reason'], 'candidate_ids': row['character_ids'],
                          'published_primary': published.get(code), 'reference_url': row['url']})
    pending = [{'character_id': c['id'], 'name': c['name'], 'code': c['code'],
                'label': codes[c['code']]['label'],
                'decision': 'À documenter puis choisir entre récit, comparaison et dossier.'}
               for c in clinical['characters'] if c['id'] not in releases]
    pattern_rows = [{'id': p['id'], 'label': p['label'],
                     'indexed_articles': sorted(key for key, row in index.items()
                                                if p['id'] in row.get('pattern_ids', []))}
                    for p in patterns]
    group_rows = []
    for group_id, group in groups.items():
        if group['parent_id']:
            continue
        rows = [r for r in code_rows if r['group_id'] == group_id]
        group_rows.append({'id': group_id, 'label': group['label'], 'codes': len(rows),
                           'statuses': dict(sorted(Counter(r['coverage_status'] for r in rows).items()))})
    ungrouped = [r for r in code_rows if r['group_id'] is None]
    assert sum(g['codes'] for g in group_rows) + len(ungrouped) == len(code_rows)
    associated = [{**r, 'published_primary': published.get(r.get('code'))}
                  for r in clinical['related_subjects']]
    extra = [{'code': code, **row} for code, row in published.items() if code not in codes]
    return {'schema_version': 1, 'as_of': max(r['date'] for r in releases.values()),
            'scope': 'État éditorial vérifiable, sans prétention à épuiser tous les concepts de la psychologie.',
            'icd11_reference': clinical['release'],
            'rules': {
                'published_primary_arc': 'Douze épisodes indexés et manuscrit relu dont l’empreinte est vérifiée.',
                'pending_editorial_case': 'Cas du registre sans parcours publié ; format à décider après documentation.',
                'editorial_mapping_only': 'Renvoi de rédaction ; aucune preuve de traitement effectif de ce code.',
                'no_standalone_arc_planned': 'Catégorie résiduelle, ouverte ou facteur associé ; besoin documentaire à examiner.',
                'indexed_pattern': 'Lien éditorial explicite, pouvant être secondaire ; ne prouve pas un traitement exhaustif.',
                'dsm': 'Les correspondances thématiques ne sont pas un inventaire diagnostic DSM-5-TR exhaustif.',
                'other_frameworks': 'APA pédagogique, RDoC recherche et CIF fonctionnement restent des grilles distinctes.'},
            'counts': {'articles': len(index), 'characters': len({r['character_id'] for r in index.values()}),
                       'clinical_arcs': len(releases), 'clinical_episodes': len(releases) * 12,
                       'original_articles_reviewed': reviewed_originals,
                       'icd11_chapter06_codes': len(code_rows),
                       'icd11_ungrouped_codes': len(ungrouped),
                       'icd11_primary_arcs': sum(code in codes for code in published),
                       'clinical_cases_pending': len(pending),
                       'other_published_arcs': len(extra),
                       'internal_patterns': len(pattern_rows),
                       'internal_patterns_indexed': sum(bool(r['indexed_articles']) for r in pattern_rows)},
            'groups': group_rows, 'codes': code_rows, 'pending_cases': pending,
            'additional_published_arcs': extra, 'associated_subjects': associated,
            'dsm_followup_topics': [
                {'label': 'Trouble de la personnalité obsessionnelle-compulsive',
                 'status': 'comparaison_documentaire_a_developper',
                 'note': 'Distinguer le tableau DSM du TOC et des traits anankastiques CIM-11.'},
                {'label': 'Trouble de la personnalité dépendante',
                 'status': 'comparaison_documentaire_a_developper',
                 'note': 'Distinguer le tableau DSM du besoin de soutien et des schémas de dépendance ordinaires.'},
            ],
            'psychology_patterns': pattern_rows,
            'broader_framework_review': 'data/psychology_coverage_review.json'}


def markdown(data):
    n = data['counts']
    lines = ['# Couverture éditoriale vérifiée', '', f"État au {data['as_of']}.", '',
             f"Le catalogue contient **{n['articles']} articles**, **{n['characters']} personnages** et "
             f"**{n['clinical_arcs']} parcours cliniques** de douze épisodes. "
             f"**{n['icd11_primary_arcs']}** parcours ont un code principal dans le chapitre 06 de la CIM-11. "
             f"**{n['other_published_arcs']}** parcours se situent dans d’autres chapitres ou dans le DSM-5-TR.", '',
             f"**{n['original_articles_reviewed']} articles d’origine** disposent d’une preuve de lecture "
             "sémantique liée à l’empreinte de leur corps. Cette preuve est distincte des signaux de structure "
             "et de la relecture indépendante des passages sensibles.", '',
             f"Il reste **{n['clinical_cases_pending']} cas du registre** à documenter et à traiter sous la forme "
             "appropriée. Les renvois à une variante ne comptent pas comme des textes écrits. "
             "Les catégories résiduelles, syndromes secondaires et différences entre référentiels restent à examiner.", '',
             f"Les **{n['internal_patterns_indexed']} repères internes indexés sur {n['internal_patterns']}** "
             "ne constituent pas un inventaire de toute la psychologie. Une liaison secondaire ne vaut pas un épisode "
             "dédié. Les lacunes précises des grilles APA, RDoC et CIF sont consignées dans "
             "[la revue des concepts](../data/psychology_coverage_review.json). Le DSM-5-TR reste à comparer "
             "diagnostic par diagnostic au-delà des seules familles thématiques.", '',
             '## Chapitre 06 de la CIM-11', '',
             '| Famille | Codes | Parcours principaux publiés | Cas en attente | Renvois seuls | Sans parcours propre |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for g in data['groups']:
        s = g['statuses']
        values = [s.get(k, 0) for k in ('published_primary_arc', 'pending_editorial_case',
                                       'editorial_mapping_only', 'no_standalone_arc_planned')]
        lines.append('| ' + g['label'].replace('|', ' / ') + f" | {g['codes']} | " + ' | '.join(map(str, values)) + ' |')
    lines += ['', f"Les {n['icd11_ungrouped_codes']} codes sans famille de premier niveau "
              "(facteurs associés et rubriques résiduelles) figurent séparément dans les données. "
              'Les 858 lignes et leur statut sont disponibles dans '
              '[le registre détaillé](../data/coverage_status.json). Les intitulés et codes proviennent de la '
              '[CIM-11 MMS 2026-01 de l’OMS](https://icd.who.int/browse/2026-01/mms/fr#334423054). '
              'International Classification of Diseases, Eleventh Revision (ICD-11), World Health '
              'Organization (WHO) 2019/2021 ; codes et intitulés sous '
              '[CC BY-ND 3.0 IGO](https://creativecommons.org/licenses/by-nd/3.0/igo/). '
              'Les choix de format et les liaisons aux personnages sont éditoriaux.', '',
              '## Autres parcours publiés', '', '| Référence principale | Personnage | Épisodes |',
              '| --- | --- | ---: |']
    for row in data['additional_published_arcs']:
        lines.append(f"| {row['code']} | {row['name']} | {len(row['articles'])} |")
    lines += ['', 'Les personnalités obsessionnelle-compulsive et dépendante restent des pistes nommées '
              'pour la comparaison DSM-5-TR, avec [la présentation officielle de l’APA]'
              '(https://www.psychiatry.org/patients-families/personality-disorders/what-are-personality-disorders). '
              'Ce rappel ne remplace pas l’inventaire de tous les diagnostics du DSM.', '',
              '## Cas du chapitre 06 encore à examiner', '',
              'Ce tableau suit les scénarios candidats déjà présents dans la cartographie. Il n’impose pas '
              'un personnage par variante. Une comparaison ou un dossier peut être plus pertinent qu’un nouveau récit.', '',
              '| Code | Sujet | Scénario candidat |', '| --- | --- | --- |']
    for row in data['pending_cases']:
        lines.append(f"| {row['code']} | {row['label'].replace('|', ' / ')} | {row['name']} |")
    lines += ['', '## Reproduire et contrôler cet état', '', '```sh',
              'python3 scripts/build_coverage_status.py', 'python3 scripts/build_coverage_status.py --check',
              '```', '', 'La génération vérifie les empreintes de relecture et les douze articles indexés de chaque '
              'parcours. Le contrôle de la forme, des sources, des nuances cliniques et de la continuité demeure '
              'une étape distincte. Un contrôle automatique ne certifie pas la qualité clinique.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = build()
    outputs = {'data/coverage_status.json': json.dumps(data, ensure_ascii=False, indent=2) + '\n',
               'docs/etat-couverture.md': markdown(data)}
    for name, content in outputs.items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit('État de couverture à régénérer : ' + name)
        else:
            path.write_text(content)
    print(json.dumps(data['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
