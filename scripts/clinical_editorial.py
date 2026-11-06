"""Prepare individual writing assignments and integrate reviewed clinical arcs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml

from build_article_intros import HEADING, WORD, normalized, plain_text
from check_editorial_pack import CATEGORIES, ROOT
from integrate_editorial_pack import public_page

EDITORIAL = ROOT / 'editorial/clinical'
CATEGORY_KEY = 'cat2_emotions'
CATEGORY, FOLDER = CATEGORIES[CATEGORY_KEY]
PACK = 'clinical-v17'
ARTICLE_KEYS = ('number', 'slug', 'title', 'description', 'context', 'topics', 'clinical_focus', 'clinical_intro',
                'opening', 'sections', 'ending', 'mirror', 'sources', 'continuity')


def read(path):
    return json.loads(path.read_text())


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def require(value, message):
    if not value:
        raise ValueError(message)


def prose(value):
    return isinstance(value, str) and bool(value.strip()) and not re.search(r'<[a-zA-Z!/?]|javascript\s*:', value, re.I)


def words(value):
    return len(WORD.findall(plain_text(value)))


def validate_episode(episode, character, number, dossier):
    label = f'{character["id"]}, épisode {number}'
    require(isinstance(episode, dict) and all(key in episode for key in ARTICLE_KEYS), f'Champs incomplets : {label}')
    require(episode['number'] == number, f'Numéro inattendu : {label}')
    require(re.fullmatch(re.escape(character['id']) + r'-[a-z0-9]+(?:-[a-z0-9]+)*', str(episode['slug'])), f'Adresse invalide : {label}')
    for key in ('title', 'description', 'context', 'opening', 'ending'):
        require(prose(episode[key]), f'Texte absent ou balise interdite dans {key} : {label}')
    focus, intro = episode['clinical_focus'], episode['clinical_intro']
    require(prose(focus) and '\n' not in focus and len(focus) <= 95 and 2 <= words(focus) <= 9,
            f'Notion clinique ou psychologique précise requise : {label}')
    require(prose(intro) and '\n' not in intro and 40 <= words(intro) <= 90
            and re.search(r'[.!?]["»”’)]*$', intro),
            f'Introduction pédagogique complète requise : {label}')
    require(prose(episode.get('ending_title')) and '\n' not in episode['ending_title'], f'Intertitre de fin manquant : {label}')
    require(len(episode['description']) <= 155, f'Description trop longue : {label}')
    require(not HEADING.search(episode['opening']) and not episode['opening'].lstrip().startswith('#') and words(episode['opening']) >= 180, f'Scène trop courte ou avec intertitre : {label}')
    topics = episode['topics']
    require(isinstance(topics, list) and len(topics) == 3 and all(prose(v) and '\n' not in v and v == v.strip() for v in topics) and len({normalized(v) for v in topics}) == 3, f'Trois repères distincts requis : {label}')
    sections = episode['sections']
    require(isinstance(sections, list) and len(sections) >= 2 and all(isinstance(s, dict) and prose(s.get('heading')) and '\n' not in s['heading'] and prose(s.get('text')) for s in sections), f'Deux sections éditoriales requises : {label}')
    mirror = episode['mirror']
    require(isinstance(mirror, list) and 1 <= len(mirror) <= 2 and all(prose(p) and '\n' not in p for p in mirror) and 40 <= words(' '.join(mirror)) <= 160, f'Rubrique des proches à revoir : {label}')
    require(words(' '.join([episode['opening'], episode['ending'], *[s['text'] for s in sections], *mirror])) >= 550, f'Épisode trop court : {label}')
    refs = episode['sources']
    require(isinstance(refs, list) and len(set(refs)) >= 2 and 'oms-cddr' in refs and set(refs) <= set(dossier['references']), f'Sources cliniques non autorisées ou insuffisantes : {label}')
    continuity = episode['continuity']
    require(isinstance(continuity, dict) and prose(continuity.get('choice')) and prose(continuity.get('next')), f'Continuité à documenter : {label}')


def sources_and_characters():
    catalogue = read(ROOT / 'data/clinical_catalog.json')
    evidence = read(ROOT / 'data/clinical_sources.json')
    return {character['id']: character for character in catalogue['characters']}, evidence


def dossier_path(ident):
    return EDITORIAL / 'dossiers' / (ident+'.json')


def character_evidence(character, shared):
    """Keep research and editorial direction scoped to the writing agent."""
    path = dossier_path(character['id'])
    code = character['code']
    if not path.exists():
        return shared, None
    raw = path.read_bytes()
    profile = json.loads(raw)
    sources, dossier, direction = profile['references'], profile['dossier'], profile['direction']
    require(prose(direction), f'Direction individuelle manquante : {path.name}')
    require(isinstance(sources, dict) and 'oms-cddr' not in sources, f'Références invalides : {path.name}')
    for key, source in sources.items():
        require(re.fullmatch(r'[a-z0-9-]+', key) and isinstance(source, dict)
                and prose(source.get('title')) and prose(source.get('url')),
                f'Référence incomplète : {path.name}, {key}')
        url = urlsplit(source['url'])
        require(url.scheme == 'https' and url.hostname and not url.username and not url.password,
                f'Adresse de source invalide : {path.name}, {key}')
    references = {**shared['references'], **sources}
    refs, points = dossier['references'], dossier['points']
    require(prose(dossier['name']) and isinstance(refs, list) and 'oms-cddr' in refs
            and len(set(refs)) >= 2 and set(refs) <= references.keys()
            and isinstance(points, list) and len(points) >= 3 and all(prose(point) for point in points)
            and isinstance(dossier.get('aliases', []), list) and all(prose(v) for v in dossier.get('aliases', [])),
            f'Dossier clinique incomplet : {path.name}')
    return {'references':references, 'dossiers':{code:dossier}}, {
        'direction':direction, 'sha256':hashlib.sha256(raw).hexdigest(),
    }


def arc_outputs(character, manifest, publication, order, evidence):
    ident, code = character['id'], character['code']
    dossier = evidence['dossiers'][code]
    bio = manifest.get('bio')
    require(manifest.get('character_id') == ident and prose(bio) and 60 <= words(bio) <= 105,
            f'Présentation du personnage invalide : {ident}')
    require(normalized(dossier['name']) in normalized(bio), f'Trouble absent de la présentation : {ident}')
    cues = manifest.get('profile_topics')
    require(isinstance(cues, list) and len(cues) == 5 and cues[0] == dossier['name']
            and all(prose(cue) and cue == cue.strip() and '\n' not in cue and len(cue) <= 72 for cue in cues[1:])
            and len({normalized(cue) for cue in cues}) == 5,
            f'Intitulé du trouble et quatre repères distincts requis dans le portrait : {ident}')
    episodes = manifest.get('episodes')
    require(isinstance(episodes, list) and len(episodes) == 12, f'Douze épisodes requis : {ident}')
    outputs, topics, mirrors, rows = {}, {}, {}, []
    titles, passages = set(), set()
    for number, episode in enumerate(episodes, 1):
        validate_episode(episode, character, number, dossier)
        title = normalized(episode['title'])
        require(title not in titles, f'Titre répété : {ident}, épisode {number}')
        titles.add(title)
        for passage in (episode['clinical_intro'], episode['opening'], episode['ending'], ' '.join(episode['mirror'])):
            key = normalized(passage)
            require(key not in passages, f'Passage répété : {ident}, épisode {number}')
            passages.add(key)
        slug = episode['slug']
        identifier = f'posts/{FOLDER}/{slug}'
        require(identifier not in topics, f'Adresse répétée : {identifier}')
        topics[identifier], mirrors[identifier] = episode['topics'], episode['mirror']
        front = {
            'title':episode['title'], 'description':episode['description'], 'slug':slug,
            'date':publication+'T00:00:00+00:00', 'draft':False,
            'personnage':character['name'], 'sujet':episode['clinical_focus'], 'contexte':episode['context'],
            'tags':list(dict.fromkeys([dossier['name'], *dossier.get('aliases', []), *episode['topics']])),
            'categories':[CATEGORY], 'editorial_pack':PACK,
            'clinical_code':code, 'clinical_diagnosis':dossier['name'], 'episode':number,
        }
        body = [episode['opening'].strip()]
        body.extend('## '+s['heading']+'\n\n'+s['text'].strip() for s in episode['sections'])
        body.append('## '+episode['ending_title']+'\n\n'+episode['ending'].strip())
        refs = [evidence['references'][ref] for ref in dict.fromkeys(episode['sources'])]
        body.append('## Sources\n\n'+'\n'.join(f'- [{ref["title"]}]({ref["url"]})' for ref in refs))
        outputs[ROOT / 'content' / (identifier+'.md')] = '---\n'+yaml.safe_dump(front,allow_unicode=True,sort_keys=False)+'---\n\n'+'\n\n'.join(body)+'\n'
        rows.append({'episode':number,'category_key':CATEGORY_KEY,'slug':slug,'title':episode['title'],
                     'resume_narratif':episode['clinical_intro']})
    canon = character['canon']
    profile = {
        'prenom':character['name'],'id':ident,'description_dynamique':bio,
        'date_naissance':f'{date.fromisoformat(publication).year-canon["age"]}-01-01',
        'profession':canon.get('occupation',''), 'situation_familiale':canon['setting'],
    }
    outputs[ROOT / 'content/personnages' / (ident+'.md')] = public_page(
        profile, rows, 34+order, published_on=publication+'T00:00:00+00:00',
        profile_fields={'clinical_code':code,'clinical_diagnosis':dossier['name'],'reading_topics':cues},
    )
    outputs[ROOT / 'data/article_mirrors' / (ident+'.json')] = dump(mirrors)
    return outputs, topics


def integrate(check):
    characters, shared = sources_and_characters()
    reviewed = read(EDITORIAL / 'releases.json')['arcs']
    outputs, new_topics, published_codes = {}, {}, set()
    for order, (ident, release) in enumerate(reviewed.items()):
        require(ident in characters, f'Personnage inconnu : {ident}')
        character = characters[ident]
        require(character['code'] not in published_codes, f'Plusieurs parcours pour {character["code"]}')
        published_codes.add(character['code'])
        require(prose(release.get('review')) and release.get('status') == 'editorially_reviewed', f'Relecture documentée nécessaire : {ident}')
        manuscript = (EDITORIAL / (ident+'.json')).read_bytes()
        require(hashlib.sha256(manuscript).hexdigest() == release['sha256'], f'Manuscrit modifié depuis la relecture : {ident}')
        evidence, dossier_review = character_evidence(character, shared)
        digest = dossier_review['sha256'] if dossier_review else None
        require(digest == release.get('dossier_sha256'), f'Dossier modifié depuis la relecture : {ident}')
        publication = date.fromisoformat(release['date']).isoformat()
        arc, topics = arc_outputs(character, json.loads(manuscript), publication, order, evidence)
        require(not (outputs.keys() & arc.keys()), f'Collision de chemins : {ident}')
        outputs.update(arc)
        new_topics.update(topics)
    topics_path = ROOT / 'data/parcours_topics.json'
    topics = read(topics_path)
    rows = [f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(values, ensure_ascii=False)}'
            for key, values in sorted({**topics, **new_topics}.items())]
    outputs[topics_path] = '{\n' + ',\n'.join(rows) + '\n}\n'
    clinical_posts = {path for path in (ROOT / 'content/posts').rglob('*.md')
                      if re.search(r'^editorial_pack:\s*clinical-v17\s*$', path.read_text(), re.MULTILINE)}
    require(clinical_posts <= outputs.keys(), 'Un parcours public doit figurer dans le registre de relecture.')
    stale = [path for path, content in outputs.items() if not path.exists() or path.read_text() != content]
    if check:
        require(not stale, 'Parcours à régénérer : '+', '.join(str(path.relative_to(ROOT)) for path in stale))
    else:
        for path in stale:
            if path.exists() and path.is_relative_to(ROOT / 'content'):
                require('clinical_code:' in path.read_text().split('---',2)[1], f'Collision avec un contenu existant : {path}')
        for path in stale:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(outputs[path])
    print(f'{len(reviewed)} parcours relus, {len(new_topics)} épisodes contrôlés.')


def brief(ident):
    characters, shared = sources_and_characters()
    require(ident in characters, f'Personnage inconnu : {ident}')
    character = characters[ident]
    evidence, individual = character_evidence(character, shared)
    dossier = evidence['dossiers'].get(character['code'])
    path = EDITORIAL / (ident+'.json')
    episodes = read(path)['episodes'] if path.exists() else []
    return {
        'agent_id':'redaction-'+ident, 'character':character,
        'direction':individual['direction'] if individual else None, 'dossier':dossier,
        'references':{ref:evidence['references'][ref] for ref in dossier['references']} if dossier else {},
        'charter':(EDITORIAL / 'charter.md').read_text(),
        'files':{'manuscript':str(path.relative_to(ROOT)), 'dossier':str(dossier_path(ident).relative_to(ROOT))},
        'manifest_fields':['character_id', 'bio', 'profile_topics', 'episodes'],
        'profile_topics_count':5,
        'episode_fields':[*ARTICLE_KEYS, 'ending_title'],
        'total_episodes':12,
        'continuity':[{'number':e['number'], 'title':e['title'], **e['continuity']} for e in episodes],
        'ready_to_draft':bool(dossier),
        'instruction':'Rédiger les épisodes dans l’ordre et consigner les choix à reprendre.' if dossier else 'Vérifier les sources cliniques et renseigner un dossier avant de lancer la rédaction.',
    }


def validate_manuscript(ident, publication):
    characters, shared = sources_and_characters()
    require(ident in characters, f'Personnage inconnu : {ident}')
    character = characters[ident]
    path = EDITORIAL / (ident+'.json')
    require(path.exists(), f'Manuscrit absent : {ident}')
    evidence, dossier_review = character_evidence(character, shared)
    require(character['code'] in evidence['dossiers'], f'Dossier clinique absent : {ident}')
    manuscript = path.read_bytes()
    arc_outputs(character, json.loads(manuscript), publication, 0, evidence)
    hashes = {'sha256':hashlib.sha256(manuscript).hexdigest()}
    if dossier_review:
        hashes['dossier_sha256'] = dossier_review['sha256']
    return hashes


def record_review(ident, publication, note):
    require(prose(note), 'Observations de relecture requises.')
    path = EDITORIAL / 'releases.json'
    registry = read(path)
    hashes = validate_manuscript(ident, publication)
    release = {'date':publication, 'status':'editorially_reviewed', **hashes, 'review':note}
    registry['arcs'][ident] = release
    path.write_text(dump(registry))


def progress():
    characters, shared = sources_and_characters()
    reviewed = read(EDITORIAL / 'releases.json')['arcs']
    priorities = read(EDITORIAL / 'priorities.json')
    by_code = {character['code']: ident for ident, character in characters.items()}
    require(len(priorities) == len(set(priorities)) and set(priorities) <= by_code.keys(),
            'Les priorités doivent correspondre à des codes distincts du registre.')
    counts = Counter()
    drafts = {}
    for ident, character in characters.items():
        path = EDITORIAL / (ident+'.json')
        count = len(read(path)['episodes']) if path.exists() else 0
        has_dossier = character['code'] in shared['dossiers'] or dossier_path(ident).exists()
        counts['episodes_written'] += count
        counts['dossiers'] += has_dossier
        counts['complete'] += count == 12
        if ident not in reviewed:
            drafts[ident] = {'id':ident, 'code':character['code'], 'episodes':count, 'has_dossier':has_dossier}
    pending = [drafts[by_code[code]] for code in priorities if by_code[code] in drafts]
    return {'reference_cases':len(characters), 'priority_candidates':len(pending),
            'other_unpublished_cases':len(drafts)-len(pending),
            'reviewed':len(reviewed), 'reviewed_episodes':len(reviewed)*12,
            **counts, 'pending':pending}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    sub.add_parser('check')
    sub.add_parser('integrate')
    make_brief = sub.add_parser('brief', aliases=['agent'])
    make_brief.add_argument('character')
    sub.add_parser('progress')
    validate = sub.add_parser('validate')
    validate.add_argument('character')
    review = sub.add_parser('record-review')
    review.add_argument('character')
    review.add_argument('--date',required=True)
    review.add_argument('--note',required=True)
    args = parser.parse_args()
    if args.action in {'check','integrate'}:
        integrate(check=args.action=='check')
    elif args.action in {'brief','agent'}:
        print(dump(brief(args.character)),end='')
    elif args.action=='progress':
        print(dump(progress()),end='')
    elif args.action=='validate':
        print(dump(validate_manuscript(args.character,date.today().isoformat())),end='')
    else:
        record_review(args.character,date.fromisoformat(args.date).isoformat(),args.note)


if __name__ == '__main__':
    main()
