"""Synthetic, deliberately artificial psychology-v11 files for tests only.

The prose is placeholder text that satisfies the mechanical contract; it is never
published and does not stand for an editorial review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

NOUNS = ['tasse', 'carnet', 'fenêtre', 'escalier', 'bureau', 'jardin', 'trajet', 'courrier', 'atelier', 'balcon',
         'marché', 'couloir', 'tableau', 'parapluie', 'cahier', 'placard', 'quai', 'lampe', 'banc', 'clavier']
VERBS = ['regarde', 'range', 'note', 'déplace', 'relit', 'prépare', 'ouvre', 'compte', 'pose', 'vérifie']


def sentence(seed: int, tag: str) -> str:
    noun, other = NOUNS[seed % len(NOUNS)], NOUNS[(seed * 7 + 3) % len(NOUNS)]
    verb = VERBS[(seed * 3) % len(VERBS)]
    return f'Ombeline {verb} le {noun} près du {other} pendant le passage {tag} numéro {seed}.'


def prose(seed: int, tag: str, count: int, name: str = 'Ombeline') -> str:
    text = ' '.join(sentence(seed * 31 + i, tag) for i in range(count))
    return text.replace('Ombeline', name)


def generic(seed: int, tag: str, count: int) -> str:
    return ' '.join(f'Ce repère général {tag} décrit une observation ordinaire numéro {seed * 17 + i} sans conclusion hâtive.'
                    for i in range(count))


def design(concepts: list[tuple[str, str]], creator: str = 'createur_ombeline') -> dict:
    keys = ['cat1_pensees', 'cat2_emotions', 'cat3_schemas']
    return {
        'schema_version': 1, 'character_id': 'ombeline', 'creator': creator,
        'canon': {'prenom': 'Ombeline', 'age': 41, 'genre': 'F', 'profession': 'bibliothécaire de quartier',
                  'situation_familiale': 'en couple avec Gaspard, une fille de dix ans', 'lieu': 'Tours',
                  'projet': 'ouvrir un club de lecture du samedi',
                  'proches': [{'prenom': 'Gaspard', 'lien': 'compagnon, menuisier', 'incise': 'Gaspard, son compagnon menuisier'},
                              {'prenom': 'Lison', 'lien': 'fille de dix ans', 'incise': 'Lison, sa fille de dix ans'}]},
        'presentation': 'Ombeline est bibliothécaire de quartier et prépare un club de lecture. Ses histoires suivent '
                        'des décisions ordinaires avec Gaspard et Lison, sans transformer chaque doute en diagnostic.',
        'tendances_psychologiques': ['Doute ordinaire', 'Organisation', 'Attention aux autres', 'Projets partagés', 'Curiosité'],
        'voice': 'Phrases courtes, gestes précis et humour discret.',
        'series_goal': 'Ouvrir le club de lecture sans renoncer aux soirées familiales.',
        'continuity_rules': ['Le club ouvre au dernier épisode.', 'Lison reste à l’école primaire.', 'Gaspard travaille à son atelier.'],
        'episodes': [{'number': n, 'concept_id': cid, 'concept': label, 'category_key': keys[n % 3],
                      'scene': f'Scène planifiée concrète pour le concept {n} à la bibliothèque.',
                      'development': f'Développement planifié du concept {n} avec ses limites.',
                      'ending': f'Fin planifiée avec une décision concrète {n}.',
                      'next': f'Raccord planifié vers la suite numéro {n}.',
                      'sources': ['https://example.org/source-a', 'https://example.org/source-b']}
                     for n, (cid, label) in enumerate(concepts, 1)],
    }


def episode(n: int, plan: dict) -> dict:
    tag = f'e{n}'
    ep = {
        'number': n, 'slug': f'ombeline-essai-{n}', 'title': f'Repère numéro {n} pour une décision ordinaire',
        'description': f'Ombeline rencontre une situation ordinaire numéro {n} à la bibliothèque.',
        'date': f'2026-12-{n:02d}', 'category_key': plan['category_key'], 'concept_id': plan['concept_id'],
        'sujet': plan['concept'], 'contexte': f'à la bibliothèque, situation {n}',
        'tags': [f'tag{n}a', f'tag{n}b', f'tag{n}c', f'tag{n}d', f'tag{n}e'],
        'topics': [f'Repère principal {n}', f'Repère second {n}'],
        'opening': ['Gaspard, son compagnon menuisier, pose une planche. Lison, sa fille de dix ans, lit à voix haute. '
                    + prose(n, tag + 'o1', 12), prose(n, tag + 'o2', 12)],
        'definition': {'heading': f'Qu’est-ce que le repère {n} ?',
                       'paragraphs': [generic(n, tag + 'd1', 8), generic(n, tag + 'd2', 8)]},
        'manifestations': {'heading': f'Comment le repère {n} se manifeste-t-il ?', 'points': [
            {'heading': f'Manifestation {n}.{k}', 'paragraphs': [prose(n * 10 + k, tag + f'm{k}', 10)],
             'examples': [f'Ombeline regarde le carnet du jour {n} et hésite au point {k}.',
                          f'Gaspard demande une précision sur le point {k} de la journée {n}.']}
            for k in range(1, 4)]},
        'techniques': {'heading': f'3 techniques pour le repère {n}', 'points': [
            {'heading': f'Technique {n}.{k}', 'objective': f'Clarifier le point {k} de la situation {n}.',
             'paragraphs': [prose(n * 20 + k, tag + f't{k}', 10)],
             'actions': [f'Noter le fait {k} observé le jour {n}.', f'Comparer le fait {k} avec une autre lecture.',
                         f'Choisir une petite action {k} réalisable.'],
             'benefit': f'Une décision plus lisible au point {k}, sans garantie.'} for k in range(1, 4)]},
        'ending': {'heading': f'Le samedi numéro {n}', 'paragraphs': [prose(n, tag + 'f1', 10), prose(n, tag + 'f2', 8)]},
        'conclusion': [generic(n, tag + 'c', 3) + ' Un psychologue peut aider si la difficulté persiste.'],
        'mirror': [f'Gaspard peut rester près d’Ombeline au moment numéro {n} et lui demander ce qui l’aiderait. '
                   + 'Il propose de porter les cartons du club, sans décider à sa place ni insister si elle préfère '
                   + 'attendre. Lison peut aussi lui montrer le livre choisi pour la séance suivante.'],
        'resume_narratif': f'Ombeline prépare la séance numéro {n} du club de lecture avec Gaspard.',
        'evolution': f'Elle choisit une petite action concrète le jour {n} et en observe la conséquence.',
        'sources': ['https://example.org/source-a', 'https://example.org/source-b'],
        'continuity': {'choice': f'Choix concret numéro {n}.', 'next': f'Suite prévue numéro {n}.'},
    }
    if n > 1:
        ep['continuity'].update({'callback': ep['opening'][0][:40], 'callback_episode': n - 1})
    # Pad to the 1800-word floor of the psychology packs with distinct sentences.
    ep['ending']['paragraphs'].append(prose(n, tag + 'pad', 70))
    return ep


def manuscript(plan: dict, design_sha: str, author: str = 'redacteur_ombeline', count: int = 12) -> dict:
    return {'schema_version': 1, 'character_id': 'ombeline', 'author': author, 'design_sha256': design_sha,
            'episodes': [episode(n, plan['episodes'][n - 1]) for n in range(1, count + 1)]}


def review(root: Path, plan: dict, script: dict, reviewer: str = 'relecteur_ombeline') -> dict:
    files = {f'editorial/psychology/designs/ombeline.json': None, 'editorial/psychology/ombeline.json': None}
    for name in files:
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {
        'schema_version': 1, 'character_id': 'ombeline', 'reviewer': reviewer, 'reviewed_on': '2026-11-12',
        'summary': 'Relecture synthétique de test : douze épisodes contrôlés mécaniquement, sans valeur éditoriale ni publication réelle.',
        'files': files, 'status': 'passed', 'findings': [],
        'checks': {k: True for k in ('proches_presentes_par_incise', 'scenes_concretes', 'sans_formules_procedurales',
                                     'raccords_exacts', 'concepts_distincts', 'conseils_proches_surs')},
        'episode_checks': [{'number': e['number'], 'concept': plan['episodes'][e['number'] - 1]['concept'],
                            'opening_quote': e['opening'][0][:40], 'ending_quote': e['ending']['paragraphs'][0][:40],
                            'mirror_quote': e['mirror'][0][:40],
                            **{key: f'Note de test {key} spécifique à la séance numéro {e["number"]} du club.'
                               for key in ('concept_note', 'continuity_note', 'readability_note', 'relatives_note')}}
                           for e in script['episodes']],
    }


def write(root: Path, relative: str, data) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return path
