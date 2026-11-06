"""Prepare clinical briefs, generate review drafts, and integrate reviewed arcs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from build_article_intros import HEADING, WORD, normalized, plain_text
from check_editorial_pack import CATEGORIES, ROOT
from integrate_editorial_pack import public_page

EDITORIAL = ROOT / 'editorial/clinical'
CATEGORY_KEY = 'cat2_emotions'
CATEGORY, FOLDER = CATEGORIES[CATEGORY_KEY]
PACK = 'clinical-v17'
ARTICLE_KEYS = ('number', 'slug', 'title', 'description', 'context', 'topics', 'opening', 'sections', 'ending', 'mirror', 'sources', 'continuity')


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


def arc_outputs(character, manifest, publication, order, evidence):
    ident, code = character['id'], character['code']
    dossier = evidence['dossiers'][code]
    require(manifest.get('character_id') == ident and prose(manifest.get('bio')), f'Présentation du personnage invalide : {ident}')
    episodes = manifest.get('episodes')
    require(isinstance(episodes, list) and len(episodes) == 12, f'Douze épisodes requis : {ident}')
    outputs, topics, mirrors, rows = {}, {}, {}, []
    for number, episode in enumerate(episodes, 1):
        validate_episode(episode, character, number, dossier)
        slug = episode['slug']
        identifier = f'posts/{FOLDER}/{slug}'
        require(identifier not in topics, f'Adresse répétée : {identifier}')
        topics[identifier], mirrors[identifier] = episode['topics'], episode['mirror']
        front = {
            'title':episode['title'], 'description':episode['description'], 'slug':slug,
            'date':publication+'T00:00:00+00:00', 'draft':False,
            'personnage':character['name'], 'sujet':dossier['name'], 'contexte':episode['context'],
            'tags':list(dict.fromkeys(dossier.get('aliases', []) + episode['topics'])),
            'categories':[CATEGORY], 'editorial_pack':PACK,
            'clinical_code':code, 'episode':number,
        }
        body = [episode['opening'].strip()]
        body.extend('## '+s['heading']+'\n\n'+s['text'].strip() for s in episode['sections'])
        body.append('## '+episode['ending_title']+'\n\n'+episode['ending'].strip())
        refs = [evidence['references'][ref] for ref in dict.fromkeys(episode['sources'])]
        body.append('## Sources\n\n'+'\n'.join(f'- [{ref["title"]}]({ref["url"]})' for ref in refs))
        outputs[ROOT / 'content' / (identifier+'.md')] = '---\n'+yaml.safe_dump(front,allow_unicode=True,sort_keys=False)+'---\n\n'+'\n\n'.join(body)+'\n'
        rows.append({'episode':number,'category_key':CATEGORY_KEY,'slug':slug,'title':episode['title'],'resume_narratif':episode['description']})
    canon = character['canon']
    profile = {
        'prenom':character['name'],'id':ident,'description_dynamique':manifest['bio'],
        'date_naissance':f'{date.fromisoformat(publication).year-canon["age"]}-01-01',
        'profession':canon.get('occupation',''), 'situation_familiale':canon['setting'],
    }
    counts = Counter(topic for values in topics.values() for topic in values)
    cues = [dossier['name']]
    cues.extend(topic for topic,count in counts.most_common() if topic not in cues and count > 1)
    outputs[ROOT / 'content/personnages' / (ident+'.md')] = public_page(
        profile, rows, 34+order, published_on=publication+'T00:00:00+00:00',
        profile_fields={'clinical_code':code,'reading_topics':cues[:5]},
    )
    outputs[ROOT / 'data/article_mirrors' / (ident+'.json')] = dump(mirrors)
    return outputs, topics


def integrate(check):
    characters, evidence = sources_and_characters()
    reviewed = read(EDITORIAL / 'releases.json')['arcs']
    outputs, new_topics, published_codes = {}, {}, set()
    for order, (ident, release) in enumerate(reviewed.items()):
        require(ident in characters, f'Personnage inconnu : {ident}')
        character = characters[ident]
        require(character['code'] not in published_codes, f'Plusieurs parcours pour {character["code"]}')
        published_codes.add(character['code'])
        require(prose(release.get('review')) and release.get('status') == 'editorially_reviewed', f'Relecture documentée nécessaire : {ident}')
        manifest_path = EDITORIAL / (ident+'.json')
        require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == release['sha256'], f'Manuscrit modifié depuis la relecture : {ident}')
        publication = date.fromisoformat(release['date']).isoformat()
        arc, topics = arc_outputs(character, read(manifest_path), publication, order, evidence)
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
    characters, evidence = sources_and_characters()
    require(ident in characters, f'Personnage inconnu : {ident}')
    character = characters[ident]
    dossier = evidence['dossiers'].get(character['code'])
    return {
        'character':character, 'dossier':dossier,
        'references':{ref:evidence['references'][ref] for ref in dossier['references']} if dossier else {},
        'ready_to_draft':bool(dossier),
        'instruction':'Rédiger les épisodes dans l’ordre et consigner les choix à reprendre.' if dossier else 'Vérifier les sources cliniques et renseigner un dossier avant de lancer la rédaction.',
    }


def provider(system, payload):
    key = os.environ.get('OPENROUTER_API_KEY')
    model = os.environ.get('CLINICAL_EDITOR_MODEL') or 'openai/gpt-5-mini'
    require(key, 'Configurer OPENROUTER_API_KEY dans GitHub Actions.')
    request = Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps({'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
                         'response_format':{'type':'json_object'},'max_tokens':4200}).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},
        method='POST',
    )
    try:
        with urlopen(request, timeout=100) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise ValueError(f'Le fournisseur a refusé la requête (HTTP {exc.code}).') from None
    choice = result['choices'][0]
    require(choice['finish_reason'] == 'stop', 'Le fournisseur a interrompu la rédaction.')
    return json.loads(choice['message']['content'])


def draft(ident, count, output):
    pack = brief(ident)
    require(pack['ready_to_draft'], 'Le dossier clinique de ce personnage doit être vérifié avant rédaction.')
    require(not output.resolve().is_relative_to(ROOT), 'Écrire les brouillons en dehors du dépôt.')
    require(1 <= count <= 12, 'Entre un et douze épisodes par exécution.')
    manuscript = read(output) if output.exists() else {'character_id':ident,'bio':'','episodes':[]}
    require(manuscript['character_id'] == ident and len(manuscript['episodes']) <= count, 'Ce fichier contient un autre brouillon ou plus d’épisodes.')
    system = (EDITORIAL / 'charter.md').read_text() + '\nRends un objet JSON {bio, episode}. Champs de episode : '+', '.join(ARTICLE_KEYS)+', ending_title. sections est un tableau {heading,text}. mirror est une liste de paragraphes. sources reprend les identifiants fournis. continuity est un objet {choice,next}. Aucune autre clé.'
    for number, episode in enumerate(manuscript['episodes'], 1):
        validate_episode(episode,pack['character'],number,pack['dossier'])
    for number in range(len(manuscript['episodes'])+1, count+1):
        payload = {**pack,'number':number,'total_episodes':12,'previous':[{'number':e['number'],'title':e['title'],'continuity':e['continuity']} for e in manuscript['episodes']]}
        result = provider(system,payload)
        episode = result['episode']
        validate_episode(episode,pack['character'],number,pack['dossier'])
        require(all(e['slug'] != episode['slug'] for e in manuscript['episodes']), 'Adresse répétée dans le brouillon.')
        if not manuscript['bio']:
            require(prose(result.get('bio')), 'Présentation manquante.')
            manuscript['bio'] = result['bio']
        manuscript['episodes'].append(episode)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dump(manuscript))
        print(f'{ident} : épisode {number} prêt à relire.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    sub.add_parser('check')
    sub.add_parser('integrate')
    make_brief = sub.add_parser('brief')
    make_brief.add_argument('character')
    make_draft = sub.add_parser('draft')
    make_draft.add_argument('character')
    make_draft.add_argument('--episodes',type=int,default=12)
    make_draft.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.action in {'check','integrate'}:
        integrate(check=args.action=='check')
    elif args.action=='brief':
        print(dump(brief(args.character)),end='')
    else:
        draft(args.character,args.episodes,args.output)


if __name__ == '__main__':
    main()
