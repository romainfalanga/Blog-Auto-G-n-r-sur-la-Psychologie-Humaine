#!/usr/bin/env python3
"""Validate and integrate the new "psychology" characters (pack psychology-v11).

No model, provider or generation API is called. Three distinct agents write
three files (see editorial/agents/psychology.md):

* creator  → editorial/psychology/designs/<id>.json   (canon, proches, 12 concepts)
* writer   → editorial/psychology/<id>.json           (12 episodes, written in order)
* reviewer → editorial/psychology/reviews/<id>.json   (independent report, SHA256-bound)

The pilot records an accepted review with `record-release`, then `integrate`
renders the twelve Markdown articles with the exact trame of the original
stories, the "Du côté des proches" mirrors, the reading concepts and the pack
manifest data/editorial_pack_v11.json consumed by integrate_editorial_pack.py.

Commands:
  design <id> | story <id> [--complete] | review <id>   validate one file
  brief <id> --role creator|writer|reviewer              JSON brief for an agent
  record-release <id> [--date YYYY-MM-DD]                pilot, after a passed review
  integrate [--check]                                    render released arcs
  proof-command <id>                                     proof command for the 12 articles
  check                                                  every file + integrate --check
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from build_article_intros import HEADING, WORD, normalized, plain_text, sentences  # noqa: E402
from check_editorial_pack import CATEGORIES, SOURCE_BLOCK, check_article, title_starts_with_name, word_count  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACK = 'psychology-v11'
EDITORIAL = 'editorial/psychology'
PROPOSALS = 'docs/analyse-couverture-concepts-2026-11-12.json'
MANIFEST = 'data/editorial_pack_v11.json'
SLUG = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+){0,5}')
FORBIDDEN_FORMULAS = (
    'après avoir pratiqué ces techniques', 'grâce à ces techniques', 'en appliquant ces techniques',
    'prend tout son sens', 'il est tout à fait naturel', 'comme nous l’avons vu', "comme nous l'avons vu",
    'dans cet article', 'cet épisode', 'voici trois techniques', 'voici 3 techniques', 'en conclusion',
    'il est important de noter', 'n’hésitez pas', "n'hésitez pas", 'observe un changement dans son quotidien',
    'force est de constater', 'il convient de',
)
PROFESSIONAL = re.compile(r'professionnel|psychologue|médecin|soignant', re.I)
REVIEW_CHECKS = ('proches_presentes_par_incise', 'scenes_concretes', 'sans_formules_procedurales',
                 'raccords_exacts', 'concepts_distincts', 'conseils_proches_surs')


class EditorialError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EditorialError(message)


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(value, minimum=1) -> bool:
    return isinstance(value, str) and value == value.strip() and '\n' not in value and len(WORD.findall(value)) >= minimum


def paragraphs(value, minimum=1) -> bool:
    return isinstance(value, list) and len(value) >= minimum and all(line(p) for p in value)


def text_words(values) -> int:
    return len(WORD.findall(plain_text(' '.join(values))))


def paths(root: Path, ident: str) -> dict[str, Path]:
    base = root / EDITORIAL
    return {'design': base / 'designs' / f'{ident}.json', 'manuscript': base / f'{ident}.json',
            'review': base / 'reviews' / f'{ident}.json', 'releases': base / 'releases.json'}


def read(path: Path):
    return json.loads(path.read_text())


def allowed_concepts(root: Path) -> dict[str, str]:
    concepts = {p['id']: p['label'] for p in read(root / 'data/psychology_catalog.json')['patterns']}
    proposals = root / PROPOSALS
    if proposals.exists():
        concepts.update({c['id']: c['label'] for c in read(proposals).get('missing_concepts', [])})
    return concepts


def proposal_for(root: Path, prenom: str) -> dict | None:
    proposals = root / PROPOSALS
    if not proposals.exists():
        return None
    return next((c for c in read(proposals).get('proposed_characters', []) if c.get('prenom') == prenom), None)


def existing_names(root: Path, own_id: str) -> dict[str, str]:
    """Lower-cased first names already used by another character page."""
    names = {}
    for page in (root / 'content/personnages').glob('*.md'):
        if page.name == '_index.md' or page.stem == own_id:
            continue
        meta = yaml.safe_load(page.read_text().split('---', 2)[1]) or {}
        names[str(meta.get('prenom', '')).casefold()] = page.stem
    return names


# --------------------------------------------------------------------------
# Design (créateur)


def validate_design(root: Path, ident: str) -> dict:
    path = paths(root, ident)['design']
    require(path.is_file(), f'Plan absent : {path.relative_to(root)}')
    design = read(path)
    require(design.get('schema_version') == 1 and design.get('character_id') == ident and re.fullmatch(r'[a-z]+', ident),
            f'schema_version 1 et character_id « {ident} » (lettres ASCII minuscules) requis')
    require(line(design.get('creator')), 'creator : identité du créateur requise')
    canon = design.get('canon')
    require(isinstance(canon, dict), 'canon requis')
    for key in ('prenom', 'genre', 'profession', 'situation_familiale', 'lieu', 'projet'):
        require(line(canon.get(key)), f'canon.{key} requis (une ligne)')
    require(isinstance(canon.get('age'), int) and 18 <= canon['age'] <= 90, 'canon.age entier entre 18 et 90')
    require(canon['genre'] in ('F', 'M', 'X'), 'canon.genre F, M ou X')
    prenom = canon['prenom']
    require(prenom.casefold() not in existing_names(root, ident), f'Prénom déjà utilisé par un autre personnage : {prenom}')
    proches = canon.get('proches')
    require(isinstance(proches, list) and len(proches) >= 2, 'canon.proches : au moins deux proches')
    seen = {prenom.casefold()}
    for proche in proches:
        require(isinstance(proche, dict) and line(proche.get('prenom')) and line(proche.get('lien'))
                and line(proche.get('incise')), 'Chaque proche : prenom, lien et incise')
        require(proche['prenom'].casefold() not in seen, 'Prénoms des proches distincts et différents du personnage')
        seen.add(proche['prenom'].casefold())
        require(proche['incise'].startswith(proche['prenom'] + ', ') and len(proche['incise']) <= 90,
                f'Incise « {proche["prenom"]}, … » de 90 caractères maximum requise')
        require(role_terms(proche), f'Le lien de {proche["prenom"]} doit contenir un mot de quatre lettres ou plus')
    presentation = design.get('presentation')
    require(line(presentation) and 25 <= len(WORD.findall(presentation)) <= 90 and prenom in presentation,
            'presentation : 25 à 90 mots, une ligne, avec le prénom')
    tendencies = design.get('tendances_psychologiques')
    require(isinstance(tendencies, list) and len(tendencies) == 5 and all(line(t) and len(t) <= 60 for t in tendencies)
            and len({normalized(t) for t in tendencies}) == 5, 'tendances_psychologiques : cinq repères distincts')
    for key in ('voice', 'series_goal'):
        require(line(design.get(key), 5), f'{key} requis')
    require(paragraphs(design.get('continuity_rules'), 3), 'continuity_rules : au moins trois règles')
    episodes = design.get('episodes')
    require(isinstance(episodes, list) and len(episodes) == 12, 'Douze épisodes planifiés requis')
    concepts = allowed_concepts(root)
    ids, labels = set(), set()
    for number, episode in enumerate(episodes, 1):
        label = f'plan, épisode {number}'
        require(isinstance(episode, dict) and episode.get('number') == number, f'Numéro attendu : {label}')
        require(episode.get('concept_id') in concepts, f'concept_id inconnu (catalogue ou analyse de couverture) : {label}')
        require(episode['concept_id'] not in ids, f'Concept répété : {label}')
        ids.add(episode['concept_id'])
        require(line(episode.get('concept')) and len(episode['concept']) <= 80, f'concept : intitulé court : {label}')
        require(normalized(episode['concept']) not in labels, f'Intitulé de concept répété : {label}')
        labels.add(normalized(episode['concept']))
        require(episode.get('category_key') in CATEGORIES, f'category_key parmi {sorted(CATEGORIES)} : {label}')
        for key in ('scene', 'development', 'ending', 'next'):
            require(line(episode.get(key), 6), f'{key} : une phrase concrète requise : {label}')
        sources = episode.get('sources')
        require(isinstance(sources, list) and len(set(sources)) >= 2 and all(isinstance(u, str) and u.startswith('https://') for u in sources),
                f'Au moins deux sources https distinctes : {label}')
    proposal = proposal_for(root, prenom)
    if proposal:
        planned = {e['concept_id'] for e in proposal['episodes']}
        deviations = design.get('deviations', [])
        require(isinstance(deviations, list) and all(isinstance(d, dict) and line(d.get('reason'), 5) for d in deviations),
                'deviations : chaque écart à la proposition doit être justifié')
        removed = {d.get('removed') for d in deviations}
        added = {d.get('added') for d in deviations}
        require(ids == (planned - removed) | added and not (planned - removed) & added,
                f'Concepts différents de la proposition pour {prenom} sans écart justifié')
    return design


def role_terms(proche: dict) -> set[str]:
    return {w for w in re.findall(r'[^\W\d_]+', normalized(proche['lien'])) if len(w) >= 4}


# --------------------------------------------------------------------------
# Manuscript (rédacteur)

EPISODE_KEYS = ('number', 'slug', 'title', 'description', 'date', 'category_key', 'concept_id', 'sujet', 'contexte',
                'tags', 'topics', 'opening', 'definition', 'manifestations', 'techniques', 'ending', 'conclusion',
                'mirror', 'resume_narratif', 'evolution', 'sources', 'continuity')


def render_body(episode: dict) -> str:
    out = ['\n\n'.join(episode['opening']), '']
    out += ['## ' + episode['definition']['heading'], '\n\n'.join(episode['definition']['paragraphs']), '']
    out += ['## ' + episode['manifestations']['heading']]
    for point in episode['manifestations']['points']:
        out += ['### ' + point['heading'], '\n\n'.join(point['paragraphs']), '', 'Exemples concrets :',
                *('* ' + example for example in point['examples']), '']
    out += ['## ' + episode['techniques']['heading']]
    for number, point in enumerate(episode['techniques']['points'], 1):
        out += [f'### {number}. {point["heading"]}', 'Objectif : ' + point['objective'], '']
        if point.get('paragraphs'):
            out += ['\n\n'.join(point['paragraphs']), '']
        out += ['Exercice pratique :', *('* ' + action for action in point['actions']), '',
                'Utilité possible : ' + point['benefit'], '']
    out += ['## ' + episode['ending']['heading'], '\n\n'.join(episode['ending']['paragraphs']), '', '---', '',
            '\n\n'.join(episode['conclusion'])]
    return '\n'.join(out).strip() + '\n'


def frontmatter(episode: dict, design: dict) -> str:
    canon = design['canon']
    front = {'title': episode['title'], 'description': episode['description'], 'slug': episode['slug'],
             'date': episode['date'] + 'T09:00:00+01:00', 'draft': False,
             'categories': [CATEGORIES[episode['category_key']][0]], 'tags': episode['tags'],
             'personnage': canon['prenom'], 'age': canon['age'], 'sujet': episode['sujet'],
             'contexte': episode['contexte'], 'editorial_pack': PACK, 'episode': episode['number']}
    return '---\n' + yaml.safe_dump(front, allow_unicode=True, sort_keys=False) + '---\n\n'


def article_path(episode: dict) -> str:
    return f'content/posts/{CATEGORIES[episode["category_key"]][1]}/{episode["slug"]}.md'


def episode_passages(episode: dict) -> list[str]:
    passages = [*episode['opening'], episode['definition']['heading'], *episode['definition']['paragraphs'],
                episode['manifestations']['heading'], episode['techniques']['heading'],
                episode['ending']['heading'], *episode['ending']['paragraphs'], *episode['conclusion'], *episode['mirror']]
    for point in episode['manifestations']['points']:
        passages += [point['heading'], *point['paragraphs'], *point['examples']]
    for point in episode['techniques']['points']:
        passages += [point['heading'], point['objective'], *point.get('paragraphs', []), *point['actions'], point['benefit']]
    return passages


def reading_order(episode: dict) -> list[str]:
    """The article text in display order, followed by the relatives' section."""
    passages = [*episode['opening'], *episode['definition']['paragraphs']]
    for point in episode['manifestations']['points']:
        passages += [*point['paragraphs'], *point['examples']]
    for point in episode['techniques']['points']:
        passages += [point['objective'], *point.get('paragraphs', []), *point['actions'], point['benefit']]
    return passages + [*episode['ending']['paragraphs'], *episode['conclusion'], *episode['mirror']]


def validate_episode(root: Path, episode: dict, number: int, design: dict) -> list[str]:
    errors = []
    label = f'épisode {number}'
    planned = design['episodes'][number - 1]
    canon = design['canon']
    if not isinstance(episode, dict) or any(key not in episode for key in EPISODE_KEYS):
        return [f'Champs incomplets ({", ".join(k for k in EPISODE_KEYS if k not in (episode or {}))}) : {label}']
    check = lambda condition, message: condition or errors.append(f'{message} : {label}')
    check(episode['number'] == number, 'Numéro inattendu')
    check(episode['concept_id'] == planned['concept_id'] and episode['category_key'] == planned['category_key'],
          'concept_id et category_key du plan requis')
    check(episode['sujet'] == planned['concept'], 'sujet égal à l’intitulé du concept du plan')
    check(isinstance(episode['slug'], str) and SLUG.fullmatch(episode['slug']) and episode['slug'].startswith(design['character_id'] + '-'),
          f'slug ASCII de six mots maximum commençant par « {design["character_id"]}- »')
    check(line(episode['title']) and len(episode['title']) <= 65 and not title_starts_with_name(episode['title'], canon['prenom']),
          'Titre centré sur le sujet, 65 caractères maximum')
    check(line(episode['description']) and len(episode['description']) <= 155, 'Description de 155 caractères maximum')
    try:
        date.fromisoformat(episode['date'])
    except (TypeError, ValueError):
        errors.append(f'date ISO requise : {label}')
    check(line(episode['contexte'], 3), 'contexte requis')
    check(isinstance(episode['tags'], list) and len(episode['tags']) >= 5 and all(line(t) for t in episode['tags'])
          and len(set(episode['tags'])) == len(episode['tags']), 'Au moins cinq tags distincts')
    topics = episode['topics']
    check(isinstance(topics, list) and 2 <= len(topics) <= 3 and all(line(t) and len(t) <= 60 for t in topics)
          and len({normalized(t) for t in topics}) == len(topics), 'Deux ou trois repères de lecture distincts')
    check(paragraphs(episode['opening']) and text_words(episode['opening']) >= 180, 'Scène d’ouverture de 180 mots minimum')
    definition = episode['definition']
    check(isinstance(definition, dict) and line(definition.get('heading')) and re.match(r'^Qu[’\']est-ce', definition.get('heading', ''))
          and paragraphs(definition.get('paragraphs'), 2), 'Définition : « Qu’est-ce que … ? » et au moins deux paragraphes')
    if not errors:
        first = sentences(plain_text(definition['paragraphs'][0]))
        check(first and canon['prenom'] not in first[0], 'La définition commence par une phrase générale, sans le personnage')
    manifestations = episode['manifestations']
    check(isinstance(manifestations, dict) and str(manifestations.get('heading', '')).startswith('Comment ')
          and str(manifestations.get('heading', '')).endswith('?'), 'Manifestations : intertitre « Comment … ? »')
    points = manifestations.get('points') if isinstance(manifestations, dict) else None
    check(isinstance(points, list) and len(points) == 3 and all(
        isinstance(p, dict) and line(p.get('heading')) and not re.match(r'^\d', p['heading']) and paragraphs(p.get('paragraphs'))
        and isinstance(p.get('examples'), list) and len(p['examples']) == 2 and all(line(e, 6) for e in p['examples'])
        and normalized(p['examples'][0]) != normalized(p['examples'][1]) for p in points),
        'Trois manifestations avec intertitre, explication et deux exemples concrets')
    techniques = episode['techniques']
    check(isinstance(techniques, dict) and str(techniques.get('heading', '')).startswith('3 techniques'),
          'Techniques : intertitre « 3 techniques … »')
    points = techniques.get('points') if isinstance(techniques, dict) else None
    check(isinstance(points, list) and len(points) == 3 and all(
        isinstance(p, dict) and line(p.get('heading')) and not re.match(r'^\d', p['heading']) and line(p.get('objective'), 4)
        and (p.get('paragraphs') is None or paragraphs(p['paragraphs'])) and isinstance(p.get('actions'), list)
        and len(p['actions']) == 3 and all(line(a, 4) for a in p['actions'])
        and len({normalized(a) for a in p['actions']}) == 3 and line(p.get('benefit'), 4) for p in points),
        'Trois techniques : objectif, trois actions distinctes et utilité possible')
    ending = episode['ending']
    check(isinstance(ending, dict) and line(ending.get('heading')) and paragraphs(ending.get('paragraphs')),
          'Retour à la scène : intertitre et paragraphes')
    check(paragraphs(episode['conclusion']) and text_words(episode['conclusion']) >= 25, 'Conclusion de 25 mots minimum')
    mirror = episode['mirror']
    check(paragraphs(mirror) and len(mirror) <= 2 and 40 <= text_words(mirror) <= 160 and sum(map(len, mirror)) <= 1200,
          '« Du côté des proches » : un ou deux paragraphes, 40 à 160 mots')
    check(any(p['prenom'] in ' '.join(mirror) for p in canon['proches']) if paragraphs(mirror) else False,
          '« Du côté des proches » ancré dans la scène avec au moins un proche nommé')
    check(line(episode['resume_narratif'], 10) and line(episode['evolution'], 8), 'resume_narratif et evolution requis')
    sources = episode['sources']
    check(isinstance(sources, list) and len(set(sources)) >= 2 and set(sources) <= set(planned['sources']),
          'Au moins deux sources, toutes autorisées par le plan')
    continuity = episode['continuity']
    check(isinstance(continuity, dict) and line(continuity.get('choice'), 4) and line(continuity.get('next'), 4),
          'continuity.choice et continuity.next requis')
    if errors:
        return errors
    passages = episode_passages(episode)
    text = '\n'.join(passages)
    if number > 1:
        callback = continuity.get('callback')
        check(isinstance(callback, str) and len(callback) >= 20 and callback in '\n'.join(reading_order(episode))
              and isinstance(continuity.get('callback_episode'), int) and 1 <= continuity['callback_episode'] < number,
              'continuity.callback : extrait exact (20 caractères ou plus) rappelant un épisode antérieur indiqué')
    check('**' not in text and not re.search('[—–]', text), 'Ni gras ni tirets cadratin ou demi-cadratin')
    check('/' not in re.sub(r'\[[^\]]*\]\([^)]*\)|https?://\S+', '', text), 'Pas de barre oblique hors liens')
    check(not re.search(r'<[a-zA-Z!/?]|javascript\s*:', text, re.I), 'Pas de balise HTML')
    check(not any(HEADING.search(p) or p.lstrip().startswith(('#', '* ', '- ')) or SOURCE_BLOCK.search(p) for p in passages),
          'Pas d’intertitre, de liste ou de bibliographie dans les champs de texte')
    lowered = text.casefold()
    found = [formula for formula in FORBIDDEN_FORMULAS if formula in lowered]
    check(not found, 'Formules procédurales à reformuler en scène : ' + ', '.join(found))
    check(PROFESSIONAL.search(text), 'Rappel d’un relais professionnel requis')
    # Each episode may be read alone: a relative is introduced by an incise at first mention.
    ordered = reading_order(episode)
    for proche in canon['proches']:
        name = proche['prenom']
        first = next((s for p in ordered for s in sentences(plain_text(p)) if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', s)), None)
        if first and not role_terms(proche) & set(re.findall(r'[^\W\d_]+', normalized(first))):
            errors.append(f'Présenter {name} par une incise à sa première mention (« {proche["incise"]} ») : {label}')
    words = word_count(render_body(episode))
    check(1800 <= words <= 2500, f'{words} mots, 1800 à 2500 requis comme pour les packs psychologie')
    return errors


def validate_manuscript(root: Path, ident: str, complete: bool = False) -> tuple[dict, dict]:
    design = validate_design(root, ident)
    files = paths(root, ident)
    require(files['manuscript'].is_file(), f'Manuscrit absent : {files["manuscript"].relative_to(root)}')
    manuscript = read(files['manuscript'])
    require(manuscript.get('schema_version') == 1 and manuscript.get('character_id') == ident, 'schema_version 1 et character_id requis')
    require(line(manuscript.get('author')), 'author : identité du rédacteur requise')
    require(normalized(manuscript['author']) != normalized(design['creator']), 'Le rédacteur doit être distinct du créateur')
    require(manuscript.get('design_sha256') == sha(files['design']), 'Plan modifié depuis le début de la rédaction (design_sha256)')
    episodes = manuscript.get('episodes')
    require(isinstance(episodes, list) and 1 <= len(episodes) <= 12, 'episodes : un à douze épisodes, dans l’ordre')
    require(not complete or len(episodes) == 12, f'Douze épisodes requis, {len(episodes)} sauvegardés')
    errors, slugs, titles, first_topics, previous = [], set(), set(), set(), None
    for number, episode in enumerate(episodes, 1):
        errors += validate_episode(root, episode, number, design)
        if not isinstance(episode, dict) or errors:
            continue
        for key, seen in (('slug', slugs), ('title', titles)):
            value = normalized(episode[key])
            if value in seen:
                errors.append(f'{key} répété : épisode {number}')
            seen.add(value)
        if normalized(episode['topics'][0]) in first_topics:
            errors.append(f'Premier repère de lecture répété : épisode {number}')
        first_topics.add(normalized(episode['topics'][0]))
        if previous and episode['date'] <= previous:
            errors.append(f'Dates croissantes requises : épisode {number}')
        previous = episode['date']
        target = root / article_path(episode)
        if target.exists():
            meta = yaml.safe_load(target.read_text().split('---', 2)[1]) or {}
            if meta.get('editorial_pack') != PACK or meta.get('personnage') != design['canon']['prenom']:
                errors.append(f'Adresse déjà utilisée par un autre article : {article_path(episode)}')
    require(not errors, '\n'.join(errors))
    return design, manuscript


# --------------------------------------------------------------------------
# Review (relecteur indépendant)


def validate_review(root: Path, ident: str) -> dict:
    design, manuscript = validate_manuscript(root, ident, complete=True)
    files = paths(root, ident)
    require(files['review'].is_file(), f'Rapport absent : {files["review"].relative_to(root)}')
    review = read(files['review'])
    require(review.get('schema_version') == 1 and review.get('character_id') == ident, 'schema_version 1 et character_id requis')
    reviewer = review.get('reviewer')
    require(line(reviewer) and normalized(reviewer) not in {normalized(design['creator']), normalized(manuscript['author'])},
            'reviewer distinct du créateur et du rédacteur')
    try:
        date.fromisoformat(review.get('reviewed_on', ''))
    except ValueError:
        raise EditorialError('reviewed_on ISO requis')
    require(line(review.get('summary'), 15), 'summary : synthèse spécifique de quinze mots minimum')
    expected = {files['design'].relative_to(root).as_posix(): sha(files['design']),
                files['manuscript'].relative_to(root).as_posix(): sha(files['manuscript'])}
    require(review.get('files') == expected, 'files : chemins et SHA256 actuels du plan et du manuscrit')
    status, findings = review.get('status'), review.get('findings')
    require(status in ('passed', 'revision_requested'), 'status passed ou revision_requested')
    if status == 'passed':
        require(findings == [], 'Une correction en attente interdit passed')
        require(review.get('checks') == {key: True for key in REVIEW_CHECKS}, 'checks : ' + ', '.join(REVIEW_CHECKS) + ' tous vérifiés')
    else:
        require(isinstance(findings, list) and findings and all(
            isinstance(f, dict) and isinstance(f.get('episode'), int) and line(f.get('issue'), 4) and line(f.get('requested_change'), 4)
            for f in findings), 'findings : episode, issue, requested_change')
    checks = review.get('episode_checks')
    require(isinstance(checks, list) and len(checks) == 12, 'Douze episode_checks dans l’ordre')
    notes = set()
    for number, (item, episode) in enumerate(zip(checks, manuscript['episodes']), 1):
        label = f'relecture, épisode {number}'
        require(isinstance(item, dict) and item.get('number') == number and item.get('concept') == design['episodes'][number - 1]['concept'],
                f'number et concept du plan : {label}')
        for key, source in (('opening_quote', episode['opening']), ('ending_quote', episode['ending']['paragraphs']),
                            ('mirror_quote', episode['mirror'])):
            quote = item.get(key)
            require(isinstance(quote, str) and len(quote) >= 20 and any(quote in p for p in source),
                    f'{key} : extrait exact de 20 caractères ou plus : {label}')
        for key in ('concept_note', 'continuity_note', 'readability_note', 'relatives_note'):
            note = item.get(key)
            require(line(note, 8), f'{key} : observation spécifique de huit mots minimum : {label}')
            require(normalized(note) not in notes, f'{key} répète une autre note : {label}')
            notes.add(normalized(note))
    return review


# --------------------------------------------------------------------------
# Release and integration


def releases(root: Path) -> dict:
    path = paths(root, 'x')['releases']
    if not path.exists():
        return {'schema_version': 1, 'pack': PACK, 'arcs': {}}
    data = read(path)
    require(data.get('schema_version') == 1 and data.get('pack') == PACK and isinstance(data.get('arcs'), dict),
            'releases.json invalide')
    return data


def record_release(root: Path, ident: str, on: str | None = None) -> dict:
    review = validate_review(root, ident)
    require(review['status'] == 'passed', 'Relecture indépendante non acceptée')
    design, manuscript = read(paths(root, ident)['design']), read(paths(root, ident)['manuscript'])
    data = releases(root)
    files = paths(root, ident)
    data['arcs'][ident] = {'date': on or date.today().isoformat(), 'design_sha256': sha(files['design']),
                           'manuscript_sha256': sha(files['manuscript']), 'review_sha256': sha(files['review']),
                           'creator': design['creator'], 'author': manuscript['author'], 'reviewer': review['reviewer'],
                           'status': 'independently_reviewed'}
    files['releases'].parent.mkdir(parents=True, exist_ok=True)
    files['releases'].write_text(dump(data))
    return data['arcs'][ident]


def pack_character(design: dict, published: str) -> dict:
    canon = design['canon']
    affinities: dict[str, list[str]] = {}
    for episode in design['episodes']:
        affinities.setdefault(episode['category_key'], []).append(episode['concept'])
    return {
        'id': design['character_id'], 'prenom': canon['prenom'], 'age': canon['age'], 'genre': canon['genre'],
        'profession': canon['profession'], 'situation_familiale': canon['situation_familiale'],
        'date_naissance': f'{date.fromisoformat(published).year - canon["age"]}-01-01',
        'description_dynamique': design['presentation'], 'presentation': design['presentation'],
        'traits_personnalite': [], 'tendances_psychologiques': design['tendances_psychologiques'],
        'histoire_de_fond': design['presentation'],
        'details_physiques': 'Aucun détail physique stable établi dans ce cycle.',
        'habitudes': 'Se référer aux événements et aux appuis concrets de l’historique publié ; ne pas inventer de rituel permanent.',
        'relations': {p['prenom']: p['lien'] for p in canon['proches']},
        'affinites_thematiques': {key: affinities[key] for key in CATEGORIES if key in affinities},
        'canon_editorial': {'prenom': canon['prenom'], 'age': canon['age'], 'genre': canon['genre'],
                            'profession': canon['profession'], 'situation_familiale': canon['situation_familiale'],
                            'lieu': canon['lieu'], 'projet': canon['projet'],
                            'secondaires': [{'prenom': p['prenom'], 'lien': p['lien'], 'incise': p['incise']} for p in canon['proches']]},
        'published_on': published + 'T08:00:00+00:00',
    }


def integrate(root: Path, check: bool = False) -> list[str]:
    data = releases(root)
    outputs: dict[Path, str] = {}
    characters, rows, topics_add = [], [], {}
    for ident, release in data['arcs'].items():
        files = paths(root, ident)
        for key, name in (('design_sha256', 'design'), ('manuscript_sha256', 'manuscript'), ('review_sha256', 'review')):
            require(files[name].is_file() and sha(files[name]) == release.get(key), f'{ident} : {name} modifié depuis la publication enregistrée')
        review = validate_review(root, ident)
        require(review['status'] == 'passed' and review['reviewer'] == release['reviewer'], f'{ident} : relecture non acceptée')
        design, manuscript = read(files['design']), read(files['manuscript'])
        mirrors = {}
        for episode in manuscript['episodes']:
            relative = article_path(episode)
            body = render_body(episode)
            outputs[root / relative] = frontmatter(episode, design) + body
            identifier = Path(relative).relative_to('content').with_suffix('').as_posix()
            mirrors[identifier] = episode['mirror']
            topics_add[identifier] = episode['topics']
            rows.append({'personnage': design['canon']['prenom'], 'episode': episode['number'], 'slug': episode['slug'],
                         'path': relative, 'title': episode['title'], 'sujet': episode['sujet'], 'contexte': episode['contexte'],
                         'category_key': episode['category_key'], 'resume_narratif': episode['resume_narratif'],
                         'evolution': episode['evolution'], 'concept_id': episode['concept_id'], 'pattern_ids': [],
                         'source_urls': episode['sources'], 'words': word_count(body)})
        outputs[root / 'data/article_mirrors' / f'{ident}.json'] = dump(dict(sorted(mirrors.items())))
        characters.append(pack_character(design, release['date']))
    manifest_path = root / MANIFEST
    if characters:
        topics_path = root / 'data/parcours_topics.json'
        topics = {**read(topics_path), **topics_add}
        outputs[topics_path] = '{\n' + ',\n'.join(f'  {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}'
                                                  for k, v in sorted(topics.items())) + '\n}\n'
        outputs[manifest_path] = dump({
            'pack': PACK, 'version': 1, 'writing_method': 'direct_authored_no_generation_api',
            'scope': f'{len(characters)} personnages « psychologie » supplémentaires, douze récits chacun, un concept distinct par épisode ; sans prétention à une psychologie exhaustive ni à un diagnostic.',
            'source_proposals': PROPOSALS, 'published_on': characters[0]['published_on'],
            'characters': characters, 'rows': rows})
    else:
        require(not manifest_path.exists(), f'{MANIFEST} présent sans parcours enregistré dans releases.json')
    expected_posts = {path for path in outputs if path.is_relative_to(root / 'content/posts')}
    for path in (root / 'content/posts').rglob('*.md'):
        if f'editorial_pack: {PACK}' in path.read_text()[:2000] and path not in expected_posts:
            raise EditorialError(f'Article {PACK} sans publication enregistrée : {path.relative_to(root)}')
    # The rendered articles must pass the same checks as the v10 pack articles.
    for path in expected_posts:
        _, front, body = outputs[path].split('---', 2)
        issues, _ = check_article(path, yaml.safe_load(front), body.strip())
        require(not issues, f'{path.relative_to(root)} : ' + '; '.join(issues))
    stale = [path for path, value in outputs.items() if not path.exists() or path.read_text() != value]
    if check:
        require(not stale, 'Parcours psychologie à régénérer : ' + ', '.join(p.relative_to(root).as_posix() for p in stale))
    else:
        for path in stale:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(outputs[path])
    return [p.relative_to(root).as_posix() for p in stale]


def check_all(root: Path) -> list[str]:
    errors = []
    base = root / EDITORIAL
    for design in sorted((base / 'designs').glob('*.json')) if (base / 'designs').exists() else []:
        ident = design.stem
        try:
            validate_design(root, ident)
            if paths(root, ident)['manuscript'].exists():
                validate_manuscript(root, ident)
            if paths(root, ident)['review'].exists():
                validate_review(root, ident)
        except EditorialError as error:
            # Work in progress is reported, not published; only released arcs block.
            message = f'{ident} : {str(error).splitlines()[0]}'
            (errors.append if ident in releases(root)['arcs'] else print)(message)
    try:
        integrate(root, check=True)
    except EditorialError as error:
        errors.append(str(error))
    return errors


def brief(root: Path, ident: str, role: str) -> dict:
    files = {k: v.relative_to(root).as_posix() for k, v in paths(root, ident).items()}
    design = read(paths(root, ident)['design']) if paths(root, ident)['design'].exists() else None
    prenom = design['canon']['prenom'] if design else ident.capitalize()
    data = {
        'role': role, 'character_id': ident, 'card': 'editorial/agents/psychology.md',
        'model_article': 'content/posts/comprendre-et-maitriser-ses-emotions/alexithymie-en-couple.md',
        'model_mirror': 'data/article_mirrors/youssef.json',
        'write_only': {'creator': [files['design']], 'writer': [files['manuscript']], 'reviewer': [files['review']]}[role],
        'proposal': proposal_for(root, prenom),
        'validate': {'creator': f'python3 scripts/psychology_editorial.py design {ident}',
                     'writer': f'python3 scripts/psychology_editorial.py story {ident}  (puis --complete)',
                     'reviewer': f'python3 scripts/psychology_editorial.py review {ident}'}[role],
        'forbidden_formulas': list(FORBIDDEN_FORMULAS),
    }
    if role != 'creator':
        require(design, 'Le plan doit exister avant la rédaction ou la relecture')
        data['design_sha256'] = sha(paths(root, ident)['design'])
        data['design'] = design
    if role == 'writer' and paths(root, ident)['manuscript'].exists():
        data['saved_episodes'] = len(read(paths(root, ident)['manuscript']).get('episodes', []))
    if role == 'reviewer':
        data['manuscript_sha256'] = sha(paths(root, ident)['manuscript'])
        data['required_checks'] = list(REVIEW_CHECKS)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('design', 'review', 'proof-command'):
        sub.add_parser(name).add_argument('ident')
    story = sub.add_parser('story')
    story.add_argument('ident')
    story.add_argument('--complete', action='store_true')
    b = sub.add_parser('brief')
    b.add_argument('ident')
    b.add_argument('--role', choices=('creator', 'writer', 'reviewer'), required=True)
    r = sub.add_parser('record-release')
    r.add_argument('ident')
    r.add_argument('--date')
    i = sub.add_parser('integrate')
    i.add_argument('--check', action='store_true')
    sub.add_parser('check')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == 'design':
            design = validate_design(root, args.ident)
            print(f'Plan valide : {args.ident}, douze concepts distincts ; SHA256 {sha(paths(root, args.ident)["design"])}')
        elif args.command == 'story':
            _, manuscript = validate_manuscript(root, args.ident, args.complete)
            print(f'Manuscrit valide : {len(manuscript["episodes"])} épisode(s) ; SHA256 {sha(paths(root, args.ident)["manuscript"])}')
        elif args.command == 'review':
            review = validate_review(root, args.ident)
            print(f'Rapport valide : {review["status"]}')
        elif args.command == 'brief':
            print(json.dumps(brief(root, args.ident, args.role), ensure_ascii=False, indent=2))
        elif args.command == 'record-release':
            print(json.dumps(record_release(root, args.ident, args.date), ensure_ascii=False, indent=2))
        elif args.command == 'integrate':
            stale = integrate(root, args.check)
            print(f'{len(releases(root)["arcs"])} parcours psychologie {PACK} ; {len(stale)} fichier(s) {"à régénérer" if args.check else "écrits"}.')
            if not args.check and stale:
                print('Ensuite : python3 scripts/integrate_editorial_pack.py && python3 scripts/build_examples.py, '
                      'puis proof-command, puis python3 scripts/build_coverage_status.py et les contrôles.')
        elif args.command == 'proof-command':
            release = releases(root)['arcs'].get(args.ident)
            require(release, 'Parcours non enregistré')
            manuscript = read(paths(root, args.ident)['manuscript'])
            articles = ' '.join(article_path(e) for e in manuscript['episodes'])
            print(f'python3 scripts/refresh_original_proof.py {articles} --reviewer {release["reviewer"]} '
                  f'--corrector {release["author"]} --review-source {paths(root, args.ident)["review"].relative_to(root).as_posix()} '
                  f'--batch {PACK} --change "Épisode du pack {PACK} rédigé depuis le plan gelé et relu intégralement" '
                  '--regenerate --run-checks')
        else:
            errors = check_all(root)
            for error in errors:
                print(error)
            print(f'{len(releases(root)["arcs"])} parcours {PACK} publiés vérifiés ; {len(errors)} erreur(s).')
            return int(bool(errors))
    except EditorialError as error:
        print('Refus :\n' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
