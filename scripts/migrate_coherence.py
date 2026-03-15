#!/usr/bin/env python3
"""
Script de migration pour la cohérence narrative des personnages.

Phase 1 (programmatique, sans API) :
- Corrige le genre grammatical dans tous les articles
- Corrige les incohérences d'âge/profession dans les articles
- Ajoute les dates correctes dans la matrice et personnages.json

Phase 2 (avec API Mammouth) :
- Appelle Gemini pour réécrire les introductions avec des références croisées
- Extrait les résumés narratifs de chaque article pour enrichir la matrice
- Ancre chaque article dans une temporalité cohérente

Usage :
  python migrate_coherence.py --phase1          # Corrections programmatiques uniquement
  python migrate_coherence.py --phase2          # Enrichissement via API (nécessite MAMMOUTH_API_KEY)
  python migrate_coherence.py --all             # Les deux phases
"""

import json
import re
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONTENT_DIR = REPO_ROOT / "content" / "posts"
MATRIX_FILE = SCRIPT_DIR / "matrice_combinaisons.json"
PERSONNAGES_FILE = SCRIPT_DIR / "personnages.json"

sys.path.insert(0, str(SCRIPT_DIR))

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_article(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_article(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ============================================
# PHASE 1 : CORRECTIONS PROGRAMMATIQUES
# ============================================

# Dictionnaire de corrections genrées français
# Format : (masculin, féminin)
GENDER_REPLACEMENTS_IL_ELLE = [
    # Pronoms sujet
    (r'\bil\b', 'elle'),
    (r'\bIl\b', 'Elle'),
    # Pronoms compléments (attention au contexte)
    (r'\blui-même\b', 'elle-même'),
    (r'\bLui-même\b', 'Elle-même'),
]

GENDER_REPLACEMENTS_ELLE_IL = [
    (r'\belle\b', 'il'),
    (r'\bElle\b', 'Il'),
    (r'\belle-même\b', 'lui-même'),
    (r'\bElle-même\b', 'Lui-même'),
]

# Adjectifs et participes passés courants (masc -> fem et vice versa)
ADJECTIVES_M_TO_F = [
    # Participes passés et adjectifs fréquents dans les articles
    (r'\bassis\b', 'assise'),
    (r'\bAssis\b', 'Assise'),
    (r'\binstallé\b', 'installée'),
    (r'\bInstallé\b', 'Installée'),
    (r'\bsurpris\b', 'surprise'),
    (r'\bSurpris\b', 'Surprise'),
    (r'\bconvaincu\b', 'convaincue'),
    (r'\bperdu\b', 'perdue'),
    (r'\bseul\b', 'seule'),
    (r'\bSeul\b', 'Seule'),
    (r'\btouché\b', 'touchée'),
    (r'\bfatigué\b', 'fatiguée'),
    (r'\bstressé\b', 'stressée'),
    (r'\bénervé\b', 'énervée'),
    (r'\bangoisé\b', 'angoissée'),
    (r'\bangoissé\b', 'angoissée'),
    (r'\binquiet\b', 'inquiète'),
    (r'\bInquiet\b', 'Inquiète'),
    (r'\bheureux\b', 'heureuse'),
    (r'\bHeureux\b', 'Heureuse'),
    (r'\bmalheureux\b', 'malheureuse'),
    (r'\bplongé\b', 'plongée'),
    (r'\bPlongé\b', 'Plongée'),
    (r'\bsubmergé\b', 'submergée'),
    (r'\bSubmergé\b', 'Submergée'),
    (r'\bdébordé\b', 'débordée'),
    (r'\bépuisé\b', 'épuisée'),
    (r'\bÉpuisé\b', 'Épuisée'),
    (r'\btroublé\b', 'troublée'),
    (r'\bblessé\b', 'blessée'),
    (r'\bBlessé\b', 'Blessée'),
    (r'\babsorb[ée]\b', 'absorbée'),
    (r'\bconscient\b', 'consciente'),
    (r'\bConscient\b', 'Consciente'),
    (r'\binconscient\b', 'inconsciente'),
    (r'\bdéterminé\b', 'déterminée'),
    (r'\bDéterminé\b', 'Déterminée'),
    (r'\bdésemparé\b', 'désemparée'),
    (r'\bDésemparé\b', 'Désemparée'),
    (r'\bconfus\b', 'confuse'),
    (r'\bConfus\b', 'Confuse'),
    (r'\bbloqué\b', 'bloquée'),
    (r'\bBloqué\b', 'Bloquée'),
    (r'\bparalysé\b', 'paralysée'),
    (r'\bParalysé\b', 'Paralysée'),
    (r'\bprisonnier\b', 'prisonnière'),
    (r'\bPrisonnier\b', 'Prisonnière'),
    (r'\bsoulagé\b', 'soulagée'),
    (r'\bSoulagé\b', 'Soulagée'),
    (r'\bapaisé\b', 'apaisée'),
    (r'\bApaisé\b', 'Apaisée'),
    (r'\bcalmé\b', 'calmée'),
    (r'\bencouragé\b', 'encouragée'),
    (r'\btransformé\b', 'transformée'),
    (r'\blibéré\b', 'libérée'),
    (r'\bLibéré\b', 'Libérée'),
    (r'\bfrappé\b', 'frappée'),
    (r'\bFrappé\b', 'Frappée'),
    (r'\bprêt\b', 'prête'),
    (r'\bPrêt\b', 'Prête'),
    (r'\bcertain\b', 'certaine'),
    (r'\bCertain\b', 'Certaine'),
    (r'\bincapable\b', 'incapable'),  # invariable
    (r'\bhabitué\b', 'habituée'),
    (r'\bforcé\b', 'forcée'),
    (r'\btendu\b', 'tendue'),
    (r'\bcrispé\b', 'crispée'),
    (r'\bCrispé\b', 'Crispée'),
    (r'\bfigé\b', 'figée'),
    (r'\bFigé\b', 'Figée'),
    (r'\benvahi\b', 'envahie'),
    (r'\bEnvahi\b', 'Envahie'),
    (r'\btraversé\b', 'traversée'),
    (r'\benfermé\b', 'enfermée'),
    (r'\benfoncé\b', 'enfoncée'),
    (r'\bpenché\b', 'penchée'),
    (r'\bPenché\b', 'Penchée'),
    (r'\badossé\b', 'adossée'),
    (r'\bAdossé\b', 'Adossée'),
    (r'\ballongé\b', 'allongée'),
    (r'\brecroquevillé\b', 'recroquevillée'),
    (r'\bimmobilisé\b', 'immobilisée'),
    (r'\bconcentré\b', 'concentrée'),
    (r'\bConcentré\b', 'Concentrée'),
    (r'\bsatisfait\b', 'satisfaite'),
    (r'\binsatisfait\b', 'insatisfaite'),
    (r'\bdéçu\b', 'déçue'),
    (r'\bDéçu\b', 'Déçue'),
    (r'\bému\b', 'émue'),
    (r'\bÉmu\b', 'Émue'),
    (r'\bagacé\b', 'agacée'),
    (r'\birrité\b', 'irritée'),
    (r'\bfrustré\b', 'frustrée'),
    (r'\bFrustré\b', 'Frustrée'),
    (r'\bhumilié\b', 'humiliée'),
    (r'\bexclu\b', 'exclue'),
    (r'\brejeté\b', 'rejetée'),
    (r'\babandonné\b', 'abandonnée'),
    (r'\bpaniqué\b', 'paniquée'),
    (r'\beffrayé\b', 'effrayée'),
    (r'\btétanisé\b', 'tétanisée'),
    (r'\bTétanisé\b', 'Tétanisée'),
    (r'\bsidéré\b', 'sidérée'),
    (r'\bmotivé\b', 'motivée'),
    (r'\bchoqué\b', 'choquée'),
    (r'\binterloqué\b', 'interloquée'),
    (r'\brésigné\b', 'résignée'),
    (r'\baccablé\b', 'accablée'),
    (r'\bdésespéré\b', 'désespérée'),
    (r'\bimpuissant\b', 'impuissante'),
    (r'\bImpuissant\b', 'Impuissante'),
    (r'\bperfectionniste\b', 'perfectionniste'),  # invariable
    (r'\bcontraint\b', 'contrainte'),
    (r'\bdéstabilisé\b', 'déstabilisée'),
    (r'\bperplexe\b', 'perplexe'),  # invariable
    (r'\bdevenu\b', 'devenue'),
    (r'\bDevenu\b', 'Devenue'),
    (r'\bparvenu\b', 'parvenue'),
    (r'\bParvenu\b', 'Parvenue'),
    (r'\brentré\b', 'rentrée'),
    (r'\bRentré\b', 'Rentrée'),
    (r'\barrivé\b', 'arrivée'),
    (r'\bArrivé\b', 'Arrivée'),
    (r'\bparti\b', 'partie'),
    (r'\bsorti\b', 'sortie'),
    (r'\bresté\b', 'restée'),
    (r'\bResté\b', 'Restée'),
    (r'\bmonté\b', 'montée'),
    (r'\btombé\b', 'tombée'),
    (r'\bdescendu\b', 'descendue'),
    (r'\bentré\b', 'entrée'),
    (r'\bEntré\b', 'Entrée'),
    (r'\brevenu\b', 'revenue'),
    (r'\bRetourné\b', 'Retournée'),
    (r'\bPassé\b', 'Passée'),
    (r'\bpassé\b', 'passée'),
    (r'\bné\b', 'née'),
    (r'\bNé\b', 'Née'),
    (r'\bcontrarié\b', 'contrariée'),
    (r'\bexaspéré\b', 'exaspérée'),
    (r'\bdéprimé\b', 'déprimée'),
    (r'\bDéprimé\b', 'Déprimée'),
]

ADJECTIVES_F_TO_M = [(f, m) for m, f in ADJECTIVES_M_TO_F if m != f]


def detect_article_gender(content, prenom):
    """Détecte le genre grammatical utilisé dans l'article pour le personnage.
    Retourne 'M', 'F' ou 'unknown'."""
    # Chercher des patterns autour du prénom
    # Ex: "Nadia est assis" -> M, "Nadia est assise" -> F
    masc_count = 0
    fem_count = 0

    # Chercher les pronoms après le prénom
    lines = content.split('\n')
    for line in lines:
        if prenom not in line:
            continue
        # Chercher "il " ou "elle " dans la même phrase (approximation)
        parts = line.split(prenom)
        for part in parts[1:]:  # Après le prénom
            # Chercher dans les 100 caractères suivants
            snippet = part[:100].lower()
            if re.search(r'\bil\b', snippet):
                masc_count += 1
            if re.search(r'\belle\b', snippet):
                fem_count += 1

    # Chercher aussi les pronoms en début de phrase qui se réfèrent au personnage
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('Il ') or stripped.startswith('Il s') or stripped.startswith("Il n'"):
            masc_count += 1
        if stripped.startswith('Elle ') or stripped.startswith('Elle s') or stripped.startswith("Elle n'"):
            fem_count += 1

    if masc_count > fem_count * 2:
        return 'M'
    elif fem_count > masc_count * 2:
        return 'F'
    elif masc_count > fem_count:
        return 'M'
    elif fem_count > masc_count:
        return 'F'
    return 'unknown'


def fix_gender_in_article(content, prenom, target_gender, current_gender):
    """Corrige le genre grammatical dans un article.
    current_gender: le genre actuellement utilisé dans l'article
    target_gender: le genre correct du personnage"""

    if current_gender == target_gender:
        return content, 0

    changes = 0

    if current_gender == 'M' and target_gender == 'F':
        # Masculin → Féminin
        for pattern, replacement in GENDER_REPLACEMENTS_IL_ELLE:
            content_before = content
            content = re.sub(pattern, replacement, content)
            if content != content_before:
                changes += len(re.findall(pattern, content_before))
        for pattern, replacement in ADJECTIVES_M_TO_F:
            content_before = content
            content = re.sub(pattern, replacement, content)
            if content != content_before:
                changes += len(re.findall(pattern, content_before))

    elif current_gender == 'F' and target_gender == 'M':
        # Féminin → Masculin
        for pattern, replacement in GENDER_REPLACEMENTS_ELLE_IL:
            content_before = content
            content = re.sub(pattern, replacement, content)
            if content != content_before:
                changes += len(re.findall(pattern, content_before))
        for pattern, replacement in ADJECTIVES_F_TO_M:
            content_before = content
            content = re.sub(pattern, replacement, content)
            if content != content_before:
                changes += len(re.findall(pattern, content_before))

    return content, changes


def fix_age_in_article(content, prenom, correct_age):
    """Corrige les mentions d'âge incorrectes dans le front matter et le contenu."""
    changes = 0

    # Corriger le front matter si l'âge est mentionné incorrectement
    # Chercher des patterns comme "50 ans" qui ne correspondent pas à l'âge correct
    age_str = f"{correct_age} ans"

    # Dans le contenu (pas le front matter), chercher des mentions d'âge avec le prénom
    # Ex: "Nadia, 50 ans" → "Nadia, 38 ans"
    pattern = rf'({re.escape(prenom)}),?\s*(\d+)\s*ans'
    matches = re.finditer(pattern, content)
    for m in matches:
        found_age = int(m.group(2))
        if found_age != correct_age:
            old = m.group(0)
            new = old.replace(f"{found_age} ans", age_str)
            content = content.replace(old, new, 1)
            changes += 1
            print(f"      Âge corrigé : {found_age} → {correct_age}")

    return content, changes


def phase1_fix_articles(personnages_data, matrix):
    """Phase 1 : Corrections programmatiques sans API."""
    personnages = personnages_data["personnages"]
    perso_map = {p["prenom"]: p for p in personnages}

    # Trouver tous les fichiers articles
    article_files = []
    for cat_dir in CONTENT_DIR.iterdir():
        if cat_dir.is_dir():
            for md_file in cat_dir.glob("*.md"):
                if md_file.name == "_index.md":
                    continue
                article_files.append(md_file)

    print(f"\n{'='*60}")
    print(f"PHASE 1 : CORRECTIONS PROGRAMMATIQUES")
    print(f"{'='*60}")
    print(f"  {len(article_files)} articles trouvés\n")

    total_gender_fixes = 0
    total_age_fixes = 0
    gender_fixed_articles = 0

    for md_file in sorted(article_files):
        content = load_article(md_file)

        # Extraire le personnage du front matter
        m_perso = re.search(r'^personnage: "(.+?)"', content, re.MULTILINE)
        if not m_perso:
            continue

        prenom = m_perso.group(1)
        perso = perso_map.get(prenom)
        if not perso:
            print(f"  [SKIP] {md_file.stem} : personnage '{prenom}' inconnu")
            continue

        target_gender = perso.get("genre", "M")
        correct_age = perso["age"]
        modified = False

        # 1. Détecter et corriger le genre
        current_gender = detect_article_gender(content, prenom)
        if current_gender != 'unknown' and current_gender != target_gender:
            content, gender_changes = fix_gender_in_article(content, prenom, target_gender, current_gender)
            if gender_changes > 0:
                total_gender_fixes += gender_changes
                gender_fixed_articles += 1
                modified = True
                print(f"  [GENRE] {md_file.stem} : {current_gender} → {target_gender} ({gender_changes} corrections)")

        # 2. Corriger les âges incorrects
        content, age_changes = fix_age_in_article(content, prenom, correct_age)
        if age_changes > 0:
            total_age_fixes += age_changes
            modified = True

        if modified:
            save_article(md_file, content)

    print(f"\n  RÉSUMÉ PHASE 1 :")
    print(f"    {gender_fixed_articles} articles corrigés pour le genre ({total_gender_fixes} corrections)")
    print(f"    {total_age_fixes} corrections d'âge")

    # 3. Corriger les dates dans la matrice et personnages.json
    print(f"\n  Synchronisation des dates matrice ↔ personnages...")
    articles = matrix.get("articles", [])
    slug_to_date = {}
    for a in articles:
        if a.get("slug") and a.get("date") and a["date"] != "migré":
            slug_to_date[a["slug"]] = a["date"]

    dates_fixed = 0
    for perso in personnages:
        for h in perso.get("historique_articles", []):
            slug = h.get("slug", "")
            if slug and (not h.get("date") or h["date"] == ""):
                if slug in slug_to_date:
                    h["date"] = slug_to_date[slug]
                    dates_fixed += 1

    if dates_fixed > 0:
        save_json(PERSONNAGES_FILE, personnages_data)
        print(f"    {dates_fixed} dates synchronisées")

    return total_gender_fixes + total_age_fixes


def phase2_enrich_narrative(personnages_data, matrix):
    """Phase 2 : Enrichissement narratif via API Mammouth.
    - Extrait les résumés narratifs de chaque article
    - Réécrit les introductions avec des références croisées
    """
    api_key = os.environ.get("MAMMOUTH_API_KEY")
    if not api_key:
        print("\n  [ERREUR] MAMMOUTH_API_KEY non définie. Phase 2 nécessite l'API.")
        print("  Définissez la variable d'environnement et relancez avec --phase2")
        return 0

    # Import des fonctions API depuis generate_articles
    from generate_articles import call_mammouth_api, fix_json_trailing_commas, MODEL_ANALYST

    personnages = personnages_data["personnages"]
    perso_map = {p["prenom"]: p for p in personnages}
    articles = matrix.get("articles", [])

    print(f"\n{'='*60}")
    print(f"PHASE 2 : ENRICHISSEMENT NARRATIF (API)")
    print(f"{'='*60}")

    # 2a. Extraire les résumés narratifs pour les articles qui n'en ont pas
    articles_sans_resume = [a for a in articles if not a.get("resume_narratif") and a.get("slug")]
    print(f"\n  {len(articles_sans_resume)} articles sans résumé narratif\n")

    enriched_count = 0
    for a in articles_sans_resume:
        slug = a.get("slug", "")
        prenom = a.get("prenom", "")
        if not slug or not prenom:
            continue

        # Trouver le fichier article
        md_file = None
        for cat_dir in CONTENT_DIR.iterdir():
            if cat_dir.is_dir():
                candidate = cat_dir / f"{slug}.md"
                if candidate.exists():
                    md_file = candidate
                    break

        if not md_file:
            print(f"  [SKIP] Article non trouvé : {slug}")
            continue

        content = load_article(md_file)
        # Extraire le contenu après le front matter
        parts = content.split("---", 2)
        if len(parts) >= 3:
            article_body = parts[2].strip()
        else:
            article_body = content

        print(f"  [Narratif] Extraction pour {prenom} : {slug}...")

        system_prompt = (
            "Tu es un analyste narratif. On te donne un article de blog qui raconte l'histoire d'un personnage "
            "confronté à un concept psychologique. Tu dois extraire 3 informations.\n\n"
            "Réponds UNIQUEMENT en JSON valide, sans backticks."
        )

        user_prompt = (
            f"Personnage : {prenom}\n"
            f"Sujet : {a.get('sujet', '')}\n"
            f"Contexte : {a.get('contexte', '')}\n\n"
            f"Article (début) :\n{article_body[:2500]}\n\n"
            "Extrais en JSON :\n"
            "{\n"
            '  "resume_narratif": "2-3 phrases résumant les événements concrets vécus par le personnage",\n'
            '  "evolution": "1-2 phrases sur ce que le personnage a compris ou changé",\n'
            '  "elements_cles": "détails importants pour la continuité (personnes, lieux, décisions)"\n'
            "}"
        )

        raw = call_mammouth_api(
            model=MODEL_ANALYST,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=600,
            retries=2
        )

        if raw:
            cleaned = raw.strip()
            if "```" in cleaned:
                match = re.search(r'\{[\s\S]*\}', cleaned)
                if match:
                    cleaned = match.group(0)
            cleaned = fix_json_trailing_commas(cleaned)

            try:
                data = json.loads(cleaned)
                a["resume_narratif"] = data.get("resume_narratif", "")
                a["evolution"] = data.get("evolution", "")
                a["elements_cles"] = data.get("elements_cles", "")
                enriched_count += 1
                print(f"    → {a['resume_narratif'][:80]}...")
            except json.JSONDecodeError:
                print(f"    → Erreur parsing JSON")

        # Rate limiting
        import time
        time.sleep(1)

    if enriched_count > 0:
        save_json(MATRIX_FILE, matrix)
        print(f"\n  {enriched_count} résumés narratifs extraits et sauvegardés dans la matrice")

    # 2b. Réécrire les introductions avec références croisées
    print(f"\n  RÉÉCRITURE DES INTRODUCTIONS :")
    print(f"  Pour chaque personnage avec 2+ articles, réécrire l'intro des articles 2+ avec des références au passé.")

    # Regrouper les articles par personnage, triés par date
    perso_articles = {}
    for a in articles:
        prenom = a.get("prenom", "")
        if prenom and a.get("slug"):
            perso_articles.setdefault(prenom, []).append(a)

    for prenom in perso_articles:
        perso_articles[prenom].sort(key=lambda x: x.get("date", ""))

    rewritten_count = 0
    for prenom, arts in perso_articles.items():
        if len(arts) < 2:
            continue

        perso = perso_map.get(prenom)
        if not perso:
            continue

        genre = perso.get("genre", "M")
        pronom = "elle" if genre == "F" else "il"

        # Pour chaque article sauf le premier, réécrire l'introduction
        for idx, a in enumerate(arts):
            if idx == 0:
                continue  # Premier article, pas de référence à faire

            slug = a.get("slug", "")
            if not slug:
                continue

            # Trouver le fichier
            md_file = None
            for cat_dir in CONTENT_DIR.iterdir():
                if cat_dir.is_dir():
                    candidate = cat_dir / f"{slug}.md"
                    if candidate.exists():
                        md_file = candidate
                        break

            if not md_file:
                continue

            content = load_article(md_file)
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue

            front_matter = parts[1]
            article_body = parts[2].strip()

            # Extraire l'introduction (tout avant le premier ## H2)
            h2_match = re.search(r'^## ', article_body, re.MULTILINE)
            if not h2_match:
                continue

            intro = article_body[:h2_match.start()].strip()
            rest = article_body[h2_match.start():]

            # Construire le contexte des articles précédents
            previous_summary = []
            for prev in arts[:idx]:
                resume = prev.get("resume_narratif", "")
                if resume:
                    previous_summary.append(f"- [{prev.get('date', '?')}] \"{prev.get('sujet', '')}\" : {resume}")
                else:
                    previous_summary.append(f"- [{prev.get('date', '?')}] \"{prev.get('sujet', '')}\" en contexte \"{prev.get('contexte', '')}\"")

            previous_text = "\n".join(previous_summary)

            print(f"  [Intro] Réécriture pour {prenom} : {slug} (article {idx+1}/{len(arts)})...")

            system_prompt = (
                "Tu es un rédacteur expert en psychologie narrative. "
                "On te donne l'introduction d'un article de blog et l'historique du personnage. "
                "Tu dois RÉÉCRIRE l'introduction (3-4 paragraphes) en :\n"
                "1. Gardant le même personnage, la même situation, le même sujet\n"
                f"2. Utilisant le genre {'féminin' if genre == 'F' else 'masculin'} (accords corrects)\n"
                "3. Ajoutant 1-2 références NATURELLES et SUBTILES aux expériences passées du personnage\n"
                "4. Racontant au PRÉSENT de l'indicatif\n"
                "5. Situant la scène dans le présent, en France\n\n"
                "Les références au passé doivent être naturelles, comme : "
                "'Depuis cette conversation avec X il y a quelques semaines...', "
                "'Elle se souvient de cette période...', "
                "'Fort de ce qu'il a appris la dernière fois...'\n\n"
                "Retourne UNIQUEMENT l'introduction réécrite, rien d'autre."
            )

            user_prompt = (
                f"Personnage : {prenom} ({perso['age']} ans, {perso['profession']})\n"
                f"Genre : {'féminin' if genre == 'F' else 'masculin'}\n"
                f"Sujet actuel : {a.get('sujet', '')}\n"
                f"Contexte actuel : {a.get('contexte', '')}\n\n"
                f"HISTORIQUE DU PERSONNAGE :\n{previous_text}\n\n"
                f"INTRODUCTION ACTUELLE À RÉÉCRIRE :\n{intro}\n\n"
                "Réécris cette introduction en ajoutant des références au passé du personnage "
                "et en corrigeant le genre si nécessaire. Retourne uniquement le texte réécrit."
            )

            new_intro = call_mammouth_api(
                model=MODEL_ANALYST,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5,
                max_tokens=1500,
                retries=2
            )

            if new_intro:
                new_intro = new_intro.strip()
                # Vérifier que la réécriture est raisonnable
                if len(new_intro.split()) >= len(intro.split()) * 0.5:
                    new_content = f"---{front_matter}---\n\n{new_intro}\n\n{rest}"
                    save_article(md_file, new_content)
                    rewritten_count += 1
                    print(f"    → Introduction réécrite ({len(intro.split())} → {len(new_intro.split())} mots)")
                else:
                    print(f"    → Réécriture trop courte, conservé l'original")
            else:
                print(f"    → Gemini n'a pas répondu, conservé l'original")

            import time
            time.sleep(1)

    print(f"\n  {rewritten_count} introductions réécrites avec références croisées")
    return enriched_count + rewritten_count


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Migration cohérence narrative des personnages")
    parser.add_argument("--phase1", action="store_true", help="Corrections programmatiques (genre, âge, dates)")
    parser.add_argument("--phase2", action="store_true", help="Enrichissement narratif via API")
    parser.add_argument("--all", action="store_true", help="Les deux phases")
    args = parser.parse_args()

    if not args.phase1 and not args.phase2 and not args.all:
        parser.print_help()
        return

    personnages_data = load_json(PERSONNAGES_FILE)
    matrix = load_json(MATRIX_FILE)

    total_changes = 0

    if args.phase1 or args.all:
        total_changes += phase1_fix_articles(personnages_data, matrix)

    if args.phase2 or args.all:
        total_changes += phase2_enrich_narrative(personnages_data, matrix)

    print(f"\n{'='*60}")
    print(f"MIGRATION TERMINÉE : {total_changes} modifications au total")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
