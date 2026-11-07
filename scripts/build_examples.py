"""Export the existing public articles as immutable onboarding references.

Does not modify articles, character pages or their URLs. The Worker validates every
chosen article against this generated catalogue; client-supplied excerpts are ignored.
"""
import json
import hashlib
import re
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
    for path in sorted((ROOT / 'content/personnages').glob('*.md')):
        _, front, _ = path.read_text().split('---', 2)
        meta = yaml.safe_load(front)
        if not meta.get('clinical_code') or meta.get('draft'):
            continue
        name = meta['prenom']
        if name.lower() in data:
            raise ValueError(f'Character already exists: {name}')
        data[name.lower()] = {
            'id': path.stem, 'name': name, 'keywords': meta['reading_topics'],
            'source': 'repères des parcours cliniques de fiction',
        }
    return data


def extractive_metadata(body, meta):
    """Auditable extracts from the actual body, never unverified matrix summaries.

    Opening and final narrative paragraphs are retained with exact offsets. This
    intentionally does not claim to be an abstractive synopsis of every scene.
    """
    conclusion_at = body.rfind('\n\n---\n\n**Conclusion**\n\n') if meta.get('clinical_code') else -1
    paragraphs=[]
    for match in re.finditer(r'(?m)(?:^|\n\n)([^\n#][\s\S]*?)(?=\n\n|\Z)', body):
        value=match.group(1).strip()
        if not value or (conclusion_at == -1 and len(value)<80) or value.startswith(('---','![','<','*À lire','**À lire')):
            continue
        start=body.find(value,match.start())
        paragraphs.append({'quote':value,'start':start,'end':start+len(value)})
    name=str(meta.get('personnage','')).casefold()
    narrative = [p for p in paragraphs if name and name in p['quote'].casefold()]
    if conclusion_at >= 0:
        stages = list(re.finditer(r'^##[ \t]+.+$', body[:conclusion_at], re.MULTILINE))
        opening = [p for p in paragraphs if p['end'] <= stages[0].start()]
        ending = [p for p in paragraphs if stages[-1].end() <= p['start'] < conclusion_at]
        if not opening or not ending:
            raise ValueError(f'Missing opening or closing scene: {meta["slug"]}')
        opening_named = [p for p in opening if p in narrative and len(p['quote']) >= 80]
        ending_named = [p for p in ending if p in narrative and len(p['quote']) >= 80]
        selected = [(opening_named or opening)[0], (ending_named or ending)[-1]]
    else:
        selected=narrative or paragraphs
        selected=([selected[0],selected[-1]] if len(selected)>1 else selected)
    selected=list({p['start']:p for p in selected}.values())
    # Whole paragraphs preserve the conclusion rather than chopping mid-clause.
    summary='\n\n[…]\n\n'.join(p['quote'] for p in selected)
    headings=re.findall(r'^#{2,3}\s+(.+)$',body,re.MULTILINE)
    return {'catalogue_version':2,'content_hash':hashlib.sha256(body.encode()).hexdigest(),
            'summary':summary,'summary_method':'extractive_opening_and_final_scene',
            'summary_evidence':selected,'canonical_concept':label(meta.get('sujet','')),
            'context':str(meta.get('contexte','')),'tone':'récit fictif et explications pédagogiques',
            'resources':[h for h in headings if re.search(r'exercice|technique|pratique|outil',h,re.I)]}


def build(write=True):
    # Keep the extractive article overviews current alongside the public catalogue.
    from build_article_intros import build as build_article_intros, load_article_mirrors
    build_article_intros(write=write)
    mirrors = load_article_mirrors()
    examples = []
    characters = character_catalog()
    articles_data = {}
    article_topics = json.loads((ROOT / 'data/parcours_topics.json').read_text())
    # Editorial links belong to the public story, never to its reader. Only
    # reviewed rows from the complete pack can add controlled concept IDs.
    pack_path = ROOT / 'data/editorial_pack_v10.json'
    pack_rows = {row['path']: row for row in json.loads(pack_path.read_text())['rows']} if pack_path.exists() else {}
    pattern_ids = {row['id'] for row in json.loads((ROOT / 'data/psychology_catalog.json').read_text())['patterns']}
    coverage_rows = {}
    coverage_pairs = set()
    for row in json.loads((ROOT / 'data/psychology_coverage_review.json').read_text())['links']:
        pair = (row['article_id'], row['pattern_id'])
        if (row['pattern_id'] not in pattern_ids or row['role'] not in {'direct', 'secondary'}
                or pair in coverage_pairs or not row.get('evidence')):
            raise ValueError('Invalid reviewed psychology link: ' + str(pair))
        coverage_pairs.add(pair)
        coverage_rows.setdefault(row['article_id'], []).append(row)
    reviewed_articles = set()
    for path in sorted((ROOT / 'content/posts').rglob('*.md')):
        if path.name == '_index.md':
            continue
        _, front, body = path.read_text().split('---', 2)
        meta = yaml.safe_load(front)
        if meta.get('draft'):
            continue
        relative = path.relative_to(ROOT / 'content').with_suffix('')
        slug = meta.get('slug', path.stem)
        metadata = extractive_metadata(body, meta)
        if meta.get('editorial_pack') == 'psychology-v10':
            reviewed = pack_rows.get(path.relative_to(ROOT).as_posix())
            if not reviewed or reviewed['slug'] != slug or set(reviewed['pattern_ids']) - pattern_ids:
                raise ValueError('Missing or invalid editorial concept links: ' + str(path))
            metadata['pattern_ids'] = list(dict.fromkeys(reviewed['pattern_ids']))
            metadata['pattern_source'] = 'editorial_pack_v10_reviewed_story'
        for reviewed in coverage_rows.get(relative.as_posix(), []):
            if (reviewed['source_body_sha256'] != metadata['content_hash']
                    or any(not item.get('quote') or body[item['start']:item['end']] != item['quote']
                           for item in reviewed['evidence'])):
                raise ValueError('Stale psychology review evidence: ' + str(path))
            metadata.setdefault('pattern_ids', []).append(reviewed['pattern_id'])
            metadata['pattern_ids'] = list(dict.fromkeys(metadata['pattern_ids']))
            metadata['pattern_source'] = ('editorial_pack_v10_and_coverage_reviewed_story'
                                          if meta.get('editorial_pack') == 'psychology-v10'
                                          else 'psychology_coverage_reviewed_story')
            reviewed_articles.add(relative.as_posix())
        if meta.get('clinical_code'):
            metadata['clinical_code'] = meta['clinical_code']
            metadata['clinical_diagnosis'] = meta['clinical_diagnosis']
        summary = metadata['summary']
        # Opening scene + ending retain narrative approach at a bounded context cost.
        excerpt = body.strip()
        if len(excerpt) > 10000:
            excerpt = excerpt[:7500] + '\n\n[Extrait intermédiaire omis]\n\n' + excerpt[-2000:]
        character = characters.get(meta.get('personnage', '').lower(), {})
        topics = article_topics[relative.as_posix()]
        theme_sources = [meta.get('sujet', '')] + meta.get('tags', [])
        # Search still uses established aliases and context tags. Editorial
        # reading topics describe what this episode actually explores.
        keywords = unique_labels([value for value in theme_sources if normalized(value) != 'psychologie'], 3)
        articles_data[relative.as_posix()] = {
            'character': meta.get('personnage', ''), 'character_id': character.get('id', ''),
            'keywords': topics, 'character_keywords': character.get('keywords', []),
            **metadata,
        }
        # Keep the section intact so a condition or safety boundary at the end
        # cannot be separated from the suggestion. Ambre can omit it as a whole.
        support_excerpt = ' '.join(mirrors[relative.as_posix()])
        examples.append({
            'id': relative.as_posix(), 'title': meta['title'],
            'url': '/' + relative.parent.as_posix() + '/' + slug + '/',
            'character': meta.get('personnage', ''),
            'description': meta.get('description', ''),
            'keywords': keywords, 'reading_topics': topics, 'character_keywords': character.get('keywords', []),
            'sujet': meta.get('sujet', ''), 'category': (meta.get('categories') or [''])[0],
            **metadata,
            'support_excerpt': support_excerpt,
            'body': excerpt,
        })
    if not examples or len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Empty or duplicate catalogue')
    if reviewed_articles != set(coverage_rows):
        raise ValueError('Psychology review references an absent or unpublished article')
    outputs = {
        ROOT / 'data/character_keywords.json': json.dumps(characters, ensure_ascii=False, indent=2) + '\n',
        ROOT / 'data/article_keywords.json': json.dumps(articles_data, ensure_ascii=False, indent=2) + '\n',
        ROOT / 'journal/src/examples.mjs': '// Generated from the public catalogue. Run python scripts/build_examples.py.\n'
        + 'export const EXAMPLES = ' + json.dumps([{k:v for k,v in e.items() if k != 'body'} for e in examples], ensure_ascii=False, separators=(',', ':')) + ';\n',
    }
    for path, value in outputs.items():
        if write:
            path.write_text(value)
        elif not path.exists() or path.read_text() != value:
            raise ValueError(f'Stale catalogue projection: {path.relative_to(ROOT)}')
    return examples


if __name__ == '__main__':
    import sys
    examples = build(write='--check' not in sys.argv)
    print(f'{len(examples)} public articles; {len({e["character"] for e in examples})} characters.')
    if '--seed' in sys.argv:
        dest = pathlib.Path(sys.argv[sys.argv.index('--seed') + 1])
        dest.write_text(json.dumps(examples, ensure_ascii=False))
