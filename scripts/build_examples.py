"""Export the existing public articles as immutable onboarding references.

Does not modify articles, character pages or their URLs. The Worker validates every
chosen article against this generated catalogue; client-supplied excerpts are ignored.
"""
import json
import pathlib
import unicodedata
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Editorial labels derived from the existing fictional character registry.
# These are reading cues, never diagnoses or claims about a reader.
LABELS = {
    'overthinking': 'Suranalyse', 'surpensee': 'Suranalyse',
    'surpensée': 'Suranalyse', 'suranalyse': 'Suranalyse',
    'gestion-du-stress': 'Gestion du stress',
    'rumination mentale': 'Rumination',
    "syndrome de l'imposteur": 'Sentiment d’imposture',
    'le syndrome de l’imposteur': 'Sentiment d’imposture',
    'difficulté à lâcher prise': 'Lâcher prise',
    'difficulté à poser des limites': 'Poser ses limites',
    'sacrifice de soi': 'Oubli de soi', 'besoin de validation': 'Besoin de reconnaissance',
    'peur de décevoir': 'Peur de décevoir', 'people pleaser': 'Faire plaisir aux autres',
    'hyperréactivité émotionnelle': 'Émotions intenses', 'charge mentale excessive': 'Charge mentale',
    'culpabilité parentale': 'Culpabilité parentale', 'anticipation anxieuse': 'Anticipation inquiète',
    'comparaison sociale': 'Comparaison aux autres', "peur de l'échec": 'Peur de l’échec',
    'masque social': 'Image de soi', 'perfectionnisme professionnel': 'Perfectionnisme',
    'difficulté à déléguer': 'Déléguer', 'tendance au burn-out': 'Épuisement au travail',
    'besoin de contrôle': 'Besoin de contrôle', 'instabilité émotionnelle': 'Émotions changeantes',
    'dépendance affective': 'Dépendance affective', 'idéalisation des relations': 'Idéalisation des relations',
    'crise de mi-vie': 'Quête de sens', 'syndrome du nid vide': 'Départ des enfants',
    'remise en question existentielle': 'Remise en question', 'peur du changement': 'Peur du changement',
    'alexithymie partielle': 'Identifier ses émotions', 'intellectualisation des émotions': 'Intellectualisation',
    'difficulté à exprimer ses sentiments': 'Exprimer ses émotions', 'entre-deux culturel': 'Appartenance culturelle',
    'fatigue de compassion': 'Fatigue à aider', 'ancienne enfant parentifiée': 'Responsabilités précoces',
    "difficulté à recevoir de l'aide": 'Recevoir de l’aide', 'hypervigilance relationnelle': 'Vigilance dans les relations',
    'refoulement émotionnel': 'Émotions retenues', "difficulté à demander de l'aide": 'Demander de l’aide',
    'masque de force': 'Devoir être fort', 'peur du vieillissement': 'Avancer en âge',
    'codépendance': 'Surinvestissement relationnel', 'besoin de sauver les autres': 'Sauver les autres',
    "peur de l'abandon": 'Peur de l’abandon', 'idéalisation relationnelle': 'Idéalisation des relations',
    'difficulté à concilier travail et famille': 'Équilibre travail et famille',
    'impulsivité décisionnelle': 'Décisions impulsives', 'idéalisation du succès': 'Rapport à la réussite',
    'deuil non résolu': 'Vivre un deuil', 'peur de la solitude': 'Peur de la solitude',
    'culpabilité générationnelle': 'Culpabilité familiale', 'résistance au changement': 'Rapport au changement',
    'oscillation émotionnelle': 'Émotions changeantes', 'idéalisation/dévaluation': 'Relations contrastées',
    'évitement émotionnel': 'Éviter ses émotions', 'surinvestissement professionnel': 'Place du travail',
    'méfiance relationnelle': 'Confiance dans les relations', 'perfectionnisme médical': 'Perfectionnisme',
    'anxiété sociale masquée': 'Inquiétude dans les relations', "peur de l'échec professionnel": 'Peur de l’échec',
    'survivante de relation toxique': 'Reconstruire sa confiance', 'difficulté à faire confiance': 'Faire confiance',
    'tendance au sauvetage': 'Sauver les autres', 'burn-out latent': 'Épuisement au travail',
    'solitude assumée en façade': 'Solitude', 'deuil du couple': 'Après une séparation',
    'difficulté à reconstruire': 'Se reconstruire', 'anxiété sociale': 'Inquiétude sociale',
    'peur du jugement': 'Regard des autres', 'évitement des conflits': 'Éviter les conflits',
    'comparaison aux autres': 'Comparaison aux autres', 'deuil figé': 'Vivre un deuil',
    'difficulté à verbaliser': 'Trouver ses mots', 'isolement choisi': 'Isolement',
    'attachement aux objets et au passé': 'Attachement au passé',
    'perfectionniste': 'Perfectionnisme', 'introvertie': 'Introversion', 'introverti': 'Introversion',
    'analytique': 'Esprit analytique', 'autocritique': 'Autocritique',
    'empathique': 'Empathie', 'créatif': 'Créativité', 'résiliente': 'Résilience',
    'timide': 'Timidité', 'ambitieux': 'Ambition', 'en quête de sens': 'Quête de sens',
    'pudique': 'Pudeur émotionnelle', 'réfléchi': 'Réflexion', 'bienveillante': 'Bienveillance',
    'déterminée': 'Détermination', 'hypersensible': 'Sensibilité', 'sensible': 'Sensibilité',
}


def normalized(value):
    value = str(value).strip().lower().replace('’', "'")
    return ' '.join(value.split())


def label(value):
    # Resolve aliases after dropping explanatory parentheses, so a subject such
    # as 'overthinking (surpensée)' and either tag share one canonical label.
    base = str(value).strip().replace('_', ' ').split(' (', 1)[0].strip()
    key = normalized(base)
    result = LABELS.get(key, base)
    result = {'Les drivers/messages contraignants': 'Messages contraignants', 'les drivers/messages contraignants': 'Messages contraignants', 'Les jeux psychologiques': 'Jeux psychologiques', 'les jeux psychologiques': 'Jeux psychologiques'}.get(result, result)
    return result[:1].upper() + result[1:] if result else ''


def unique_labels(values, limit):
    result, seen = [], set()
    for value in values:
        text = label(value)
        key = normalized(text)
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result[:limit]


def character_catalog():
    characters = json.loads((ROOT / 'scripts/personnages.json').read_text())['personnages']
    data = {}
    for character in characters:
        # Four recurring tendencies + one strength/trait gives useful range.
        themes = unique_labels(character.get('tendances_psychologiques', []) + character.get('traits_personnalite', []), 5)
        data[character['prenom'].lower()] = {
            'id': character['id'], 'name': character['prenom'], 'keywords': themes,
            'source': 'traits_personnalite et tendances_psychologiques du registre des personnages fictifs',
        }
    return data


def build():
    examples = []
    characters = character_catalog()
    articles_data = {}
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
        character = characters.get(meta.get('personnage', '').lower(), {})
        theme_sources = [meta.get('sujet', '')] + meta.get('tags', [])
        # Generic site-wide tags add no information about affinity. The remaining
        # frontmatter tags naturally fill the space freed by synonym deduplication.
        keywords = unique_labels([value for value in theme_sources if normalized(value) != 'psychologie'], 3)
        articles_data[relative.as_posix()] = {
            'character': meta.get('personnage', ''), 'character_id': character.get('id', ''),
            'keywords': keywords, 'character_keywords': character.get('keywords', []),
        }
        examples.append({
            'id': relative.as_posix(), 'title': meta['title'],
            'url': '/' + relative.parent.as_posix() + '/' + slug + '/',
            'character': meta.get('personnage', ''),
            'description': meta.get('description', ''),
            'keywords': keywords, 'character_keywords': character.get('keywords', []),
            'sujet': meta.get('sujet', ''), 'category': (meta.get('categories') or [''])[0],
            'body': ('Résumé du personnage FICTIF : ' + summary + '\n\n' if summary else '') + excerpt,
        })
    if not examples or len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Empty or duplicate catalogue')
    (ROOT / 'data/character_keywords.json').write_text(json.dumps(characters, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'data/article_keywords.json').write_text(json.dumps(articles_data, ensure_ascii=False, indent=2) + '\n')
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
