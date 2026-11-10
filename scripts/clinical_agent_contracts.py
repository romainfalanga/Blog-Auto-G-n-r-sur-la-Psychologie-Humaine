"""Validate evidence at the handoff between real ChatGPT editorial agents.

These checks establish structure and provenance. An independent agent must still
read every episode and its sources; hashes cannot establish clinical accuracy.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re

import clinical_editorial as clinical

ROOT = clinical.ROOT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(relative):
    path = ROOT / relative
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Chemin hors du dépôt')
    require(path.is_file(), 'Fichier requis : ' + relative)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('JSON invalide : ' + relative) from error
    require(isinstance(value, dict), 'Objet JSON requis : ' + relative)
    return value


def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def paths(ident):
    characters, _ = clinical.sources_and_characters()
    require(isinstance(ident, str) and ident in characters,
            'Personnage canonique inconnu : ' + str(ident))
    return {
        'design': f'editorial/clinical/designs/{ident}.json',
        'dossier': f'editorial/clinical/dossiers/{ident}.json',
        'story': f'editorial/clinical/{ident}.json',
        'review': f'editorial/clinical/reviews/{ident}.json',
    }


def validate_design(ident):
    locations = paths(ident)
    characters, shared = clinical.sources_and_characters()
    character = characters[ident]
    design = read(locations['design'])
    # Individual research is compulsory for a newly commissioned character.
    read(locations['dossier'])
    evidence, individual = clinical.character_evidence(character, shared)
    require(individual is not None, 'Dossier individuel requis')
    dossier = evidence['dossiers'][character['code']]
    require(design.get('schema_version') == 1
            and design.get('character_id') == ident
            and design.get('code') == character['code']
            and design.get('article_format') == 2, 'Identité ou format du plan invalide')
    bio = design.get('bio')
    require(clinical.paragraph(bio) and 60 <= clinical.words(bio) <= 105,
            'Biographie : 60 à 105 mots en un paragraphe')
    require(clinical.normalized(dossier['name']) in clinical.normalized(bio),
            'Trouble documenté absent de la biographie')
    age = character['canon']['age']
    small_numbers = ('zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept',
                     'huit', 'neuf', 'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize')
    age_forms = [str(age), *([small_numbers[age]] if 0 <= age < len(small_numbers) else [])]
    require(re.search(r'(?<!\w)' + re.escape(character['name']) + r'(?!\w)', bio, re.I)
            and re.search(r'(?<!\w)(?:' + '|'.join(age_forms) + r')\s+ans\b', bio, re.I),
            'Prénom et âge canoniques requis dans la biographie')
    topics = design.get('profile_topics')
    require(isinstance(topics, list) and len(topics) == 5
            and all(clinical.paragraph(topic) for topic in topics)
            and all(len(topic) <= 72 for topic in topics[1:])
            and topics[0] == dossier['name']
            and len({clinical.normalized(topic) for topic in topics}) == 5,
            'Cinq repères distincts, le premier étant le trouble exact')
    require(all(clinical.prose(design.get(key)) for key in ('voice', 'series_goal')),
            'Voix et enjeu de série requis')
    rules = design.get('continuity_rules')
    require(isinstance(rules, list) and len(rules) >= 3
            and all(clinical.prose(rule) for rule in rules), 'Règles de continuité insuffisantes')
    episodes = design.get('episodes')
    require(isinstance(episodes, list) and len(episodes) == 12, 'Douze épisodes prévus requis')
    focuses = set()
    diagnosis = character['diagnosis']
    for number, episode in enumerate(episodes, 1):
        require(isinstance(episode, dict), f'Plan de l’épisode {number} : objet requis')
        require(episode.get('number') == number, f'Ordre du plan : épisode {number}')
        focus = episode.get('focus')
        require(clinical.paragraph(focus) and 2 <= clinical.words(focus) <= 9 and len(focus) <= 95,
                f'Aspect clinique précis : épisode {number}')
        key = clinical.normalized(focus)
        require(key not in focuses, f'Aspect répété : épisode {number}')
        focuses.add(key)
        require(all(clinical.prose(episode.get(key))
                    for key in ('scene', 'development', 'ending', 'next')),
                f'Scène, développement, fin et raccord requis : épisode {number}')
        refs = episode.get('source_refs')
        primary = character.get('primary_reference', 'oms-cddr')
        require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs)
                and len(set(refs)) >= 2 and primary in refs
                and set(refs) <= set(dossier['references']),
                f'Sources insuffisantes ou non autorisées : épisode {number}')
        if diagnosis['timing'] == 'established':
            expected = 'known'
        else:
            announced = diagnosis.get('episode')
            require(isinstance(announced, int) and 1 <= announced <= 12,
                    'Chronologie canonique de diagnostic non prise en charge')
            expected = ('under_assessment' if number < announced else
                        'diagnosis_explained' if number == announced else 'known')
        require(episode.get('diagnostic_stage') == expected,
                f'Chronologie du diagnostic : épisode {number}, attendu {expected}')
    files = {locations[key]: digest(locations[key]) for key in ('design', 'dossier')}
    return {'character_id': ident, 'files': files, 'episodes': 12}


def validate_story(ident):
    result = validate_design(ident)
    locations = paths(ident)
    design, story = read(locations['design']), read(locations['story'])
    clinical.validate_manuscript(ident, date.today().isoformat())
    require(story.get('bio') == design['bio']
            and story.get('profile_topics') == design['profile_topics'],
            'Le manuscrit diverge de la présentation validée')
    for episode, plan in zip(story['episodes'], design['episodes']):
        require(clinical.normalized(episode['clinical_focus']) == clinical.normalized(plan['focus']),
                f'Aspect différent du plan : épisode {plan["number"]}')
    result['files'][locations['story']] = digest(locations['story'])
    return result


def validate_review(ident):
    result = validate_story(ident)
    locations = paths(ident)
    report = read(locations['review'])
    story = read(locations['story'])
    require(report.get('schema_version') == 1 and report.get('character_id') == ident,
            'Identité du rapport de relecture invalide')
    require(report.get('status') in ('passed', 'revision_requested')
            and clinical.paragraph(report.get('reviewer'))
            and clinical.prose(report.get('summary')), 'Statut ou relecteur absent')
    try:
        date.fromisoformat(report['reviewed_on'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Date de relecture ISO requise') from error
    require(report.get('files') == result['files'],
            'Relecture périmée : empreintes exactes des trois entrées requises')
    findings = report.get('findings')
    require(isinstance(findings, list), 'Liste des remarques requise')
    for finding in findings:
        require(isinstance(finding, dict) and type(finding.get('episode')) is int
                and 1 <= finding['episode'] <= 12
                and clinical.prose(finding.get('issue'))
                and clinical.prose(finding.get('requested_change')), 'Remarque incomplète')
    require((report['status'] == 'passed') == (len(findings) == 0),
            'Une relecture acceptée ne comporte aucune correction en attente')
    checks = report.get('episode_checks')
    require(isinstance(checks, list) and len(checks) == 12, 'Lecture des douze épisodes requise')
    for number, (check, episode) in enumerate(zip(checks, story['episodes']), 1):
        require(isinstance(check, dict), f'Preuve de lecture {number} : objet requis')
        require(check.get('number') == number
                and clinical.normalized(check.get('focus', '')) == clinical.normalized(episode['clinical_focus']),
                f'Aspect de la relecture incorrect : épisode {number}')
        for quote_key, source_key in (('opening_quote', 'opening'), ('ending_quote', 'ending')):
            quote = check.get(quote_key)
            require(isinstance(quote, str) and len(quote.strip()) >= 20
                    and quote in episode[source_key], f'Citation de lecture absente : {number}, {quote_key}')
        require(all(clinical.prose(check.get(key))
                    for key in ('continuity_note', 'clinical_note', 'relatives_note')),
                f'Relecture clinique, narrative et des proches incomplète : {number}')
    result.update(status=report['status'], findings=findings, reviewer=report['reviewer'])
    result['files'][locations['review']] = digest(locations['review'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('character', 'story', 'review'))
    parser.add_argument('character')
    args = parser.parse_args()
    validator = {'character': validate_design, 'story': validate_story, 'review': validate_review}[args.role]
    try:
        print(json.dumps(validator(args.character), ensure_ascii=False, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
