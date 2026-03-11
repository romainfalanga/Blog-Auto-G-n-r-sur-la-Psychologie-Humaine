#!/usr/bin/env python3
"""
Génère 3 articles de blog quotidiens (1 par catégorie)
en appelant l'API Mammouth (Gemini pour la suggestion, GPT pour la rédaction)
et en créant des fichiers Markdown Hugo.

Flux :
1. Charger la matrice des combinaisons déjà réalisées (scripts/matrice_combinaisons.json)
2. Gemini analyse la matrice et suggère 3 nouvelles combinaisons uniques
3. GPT-4.1-mini rédige les articles sur les suggestions de Gemini
4. Gemini vérifie chaque article : si des tirets cadratins (—) sont détectés,
   Gemini reformule les passages concernés pour les supprimer
5. La matrice est mise à jour avec les nouvelles combinaisons
"""

import os
import sys
import json
import random
import requests
import re
from datetime import datetime, timezone
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONTENT_DIR = REPO_ROOT / "content" / "posts"
TRACKING_FILE = SCRIPT_DIR / "generated_topics.json"
MATRIX_FILE = SCRIPT_DIR / "matrice_combinaisons.json"

sys.path.insert(0, str(SCRIPT_DIR))

MAMMOUTH_API_KEY = os.environ.get("MAMMOUTH_API_KEY")
API_URL = "https://api.mammouth.ai/v1/chat/completions"
MODEL_WRITER = "gpt-4.1-mini"
MODEL_ANALYST = "gemini-3-flash-preview"
SITE_NAME = "Décode ton esprit"

from config import (
    CATEGORIES, CONTEXTES, PROFILS, ANGLES,
    PRENOMS, TRANCHES_AGE
)

# ============================================
# MATRICE DES COMBINAISONS
# ============================================

def load_matrix():
    """Charge la matrice des combinaisons déjà réalisées."""
    if MATRIX_FILE.exists():
        try:
            with open(MATRIX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "articles" not in data:
                    return {"articles": []}
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Attention: matrice corrompue, reinitialisation ({e})")
    return {"articles": []}


def save_matrix(data):
    """Sauvegarde la matrice des combinaisons."""
    with open(MATRIX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_to_matrix(matrix, combo, metadata):
    """Ajoute une combinaison à la matrice après génération de l'article."""
    entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "category_key": combo["category_key"],
        "category_name": combo["category_name"],
        "sujet": combo["sujet"],
        "contexte": combo["contexte"],
        "angle": combo["angle"],
        "profil": combo.get("profil"),
        "prenom": combo["prenom"],
        "age": combo["age"],
        "title": metadata.get("title", ""),
        "slug": metadata.get("slug", ""),
        "tags": metadata.get("tags", []),
    }
    matrix["articles"].append(entry)


def build_matrix_summary(matrix):
    """Construit un résumé de la matrice pour Gemini."""
    articles = matrix.get("articles", [])
    if not articles:
        return "Aucun article n'a encore été généré."

    lines = [f"ARTICLES DÉJÀ GÉNÉRÉS ({len(articles)} au total) :\n"]
    for i, a in enumerate(articles, 1):
        line = f"{i}. [{a['category_key']}] Sujet: \"{a['sujet']}\" | Contexte: \"{a['contexte']}\" | Angle: \"{a['angle']}\""
        if a.get("title"):
            line += f" | Titre: \"{a['title']}\""
        if a.get("profil"):
            line += f" | Profil: \"{a['profil']}\""
        lines.append(line)

    # Ajouter un résumé des sujets/contextes utilisés par catégorie
    lines.append("\nRÉSUMÉ PAR CATÉGORIE :")
    for cat_key in ["cat1_pensees", "cat2_emotions", "cat3_schemas"]:
        cat_articles = [a for a in articles if a["category_key"] == cat_key]
        if cat_articles:
            sujets_used = set(a["sujet"] for a in cat_articles)
            contextes_used = set(a["contexte"] for a in cat_articles)
            lines.append(f"  {cat_key}: {len(cat_articles)} articles, sujets utilisés: {sujets_used}, contextes utilisés: {contextes_used}")
        else:
            lines.append(f"  {cat_key}: 0 articles")

    return "\n".join(lines)


def migrate_tracking_to_matrix(matrix):
    """Migre les données de generated_topics.json vers la matrice si nécessaire."""
    if not TRACKING_FILE.exists():
        return

    try:
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            tracking = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    existing_combos = set()
    for a in matrix.get("articles", []):
        existing_combos.add(f"{a['category_key']}|{a['sujet']}|{a['contexte']}|{a['angle']}")

    migrated = 0
    for combo_str in tracking.get("generated", []):
        if combo_str in existing_combos:
            continue
        parts = combo_str.split("|", 3)
        if len(parts) == 4:
            matrix["articles"].append({
                "date": "migré",
                "category_key": parts[0],
                "category_name": CATEGORIES.get(parts[0], {}).get("name", parts[0]),
                "sujet": parts[1],
                "contexte": parts[2],
                "angle": parts[3],
                "profil": None,
                "prenom": "",
                "age": "",
                "title": "",
                "slug": "",
                "tags": [],
            })
            migrated += 1

    if migrated > 0:
        print(f"  {migrated} combinaisons migrees depuis generated_topics.json")


# ============================================
# GEMINI : SUGGESTION DE SUJETS
# ============================================

def call_mammouth_api(model, system_prompt, user_prompt, temperature=0.85, max_tokens=4500, retries=3):
    """Appelle l'API Mammouth avec le modèle spécifié."""
    headers = {
        "Authorization": f"Bearer {MAMMOUTH_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    for attempt in range(retries):
        try:
            print(f"    [API] Appel {model} (tentative {attempt + 1}/{retries})...")
            response = requests.post(API_URL, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            print(f"    [API] Erreur HTTP {response.status_code}: {response.text[:300]}")
            if attempt == retries - 1:
                raise
        except Exception as e:
            print(f"    [API] Tentative {attempt + 1}/{retries} echouee: {e}")
            if attempt == retries - 1:
                raise

    return None


def build_gemini_prompt(matrix_summary):
    """Construit le prompt pour Gemini avec des indices numériques pour fiabiliser le JSON."""

    # Construire les listes numérotées pour chaque dimension
    cat1_sujets = CATEGORIES["cat1_pensees"]["sujets"]
    cat2_sujets = CATEGORIES["cat2_emotions"]["sujets"]
    cat3_sujets = CATEGORIES["cat3_schemas"]["sujets"]

    cat1_list = "\n".join(f"  {i}: \"{s}\"" for i, s in enumerate(cat1_sujets))
    cat2_list = "\n".join(f"  {i}: \"{s}\"" for i, s in enumerate(cat2_sujets))
    cat3_list = "\n".join(f"  {i}: \"{s}\"" for i, s in enumerate(cat3_sujets))
    contextes_list = "\n".join(f"  {i}: \"{c}\"" for i, c in enumerate(CONTEXTES))
    angles_list = "\n".join(f"  {i}: \"{a}\"" for i, a in enumerate(ANGLES))
    profils_list = "\n".join(f"  {i}: \"{p}\"" for i, p in enumerate(PROFILS))

    system_prompt = """Tu es un directeur éditorial expert en psychologie. Tu analyses le contenu existant d'un blog et proposes des sujets complémentaires.

RÈGLES :
- Évite les sujets et combinaisons sujet+contexte déjà traités ou trop proches
- Privilégie la diversité thématique entre les 3 suggestions
- Choisis des combinaisons qui créent des articles intéressants et recherchés

IMPORTANT : Réponds UNIQUEMENT avec du JSON valide. Pas de texte avant ni après. Pas de backticks."""

    user_prompt = f"""{matrix_summary}

SUJETS DISPONIBLES PAR CATÉGORIE (utilise l'INDICE numérique) :

cat1_pensees:
{cat1_list}

cat2_emotions:
{cat2_list}

cat3_schemas:
{cat3_list}

CONTEXTES DISPONIBLES (utilise l'INDICE numérique) :
{contextes_list}

ANGLES DISPONIBLES (utilise l'INDICE numérique) :
{angles_list}

PROFILS (utilise l'INDICE numérique, ou -1 pour aucun profil) :
{profils_list}

Propose 3 combinaisons (1 par catégorie). Réponds en JSON :

[
  {{"cat": "cat1_pensees", "sujet_idx": 5, "contexte_idx": 2, "angle_idx": 0, "profil_idx": -1, "justification": "..."}},
  {{"cat": "cat2_emotions", "sujet_idx": 12, "contexte_idx": 7, "angle_idx": 3, "profil_idx": 2, "justification": "..."}},
  {{"cat": "cat3_schemas", "sujet_idx": 8, "contexte_idx": 1, "angle_idx": 5, "profil_idx": -1, "justification": "..."}}
]

Les indices doivent correspondre aux listes ci-dessus. Pas de texte autour du JSON."""

    return system_prompt, user_prompt


def parse_gemini_suggestions(raw_response):
    """Parse la réponse JSON de Gemini et résout les indices."""
    cleaned = raw_response.strip()

    # Enlever backticks markdown si présents
    if "```" in cleaned:
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if match:
            cleaned = match.group(0)
        else:
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        raw_suggestions = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  Erreur parsing JSON Gemini: {e}")
        print(f"  Reponse brute (500 premiers chars): {raw_response[:500]}")
        return None

    if not isinstance(raw_suggestions, list) or len(raw_suggestions) != 3:
        count = len(raw_suggestions) if isinstance(raw_suggestions, list) else "pas une liste"
        print(f"  Gemini n'a pas retourne 3 suggestions (recu: {count})")
        return None

    # Résoudre les indices vers les valeurs réelles
    resolved = []
    for s in raw_suggestions:
        cat_key = s.get("cat", "")
        if cat_key not in CATEGORIES:
            print(f"  Categorie inconnue: {cat_key}")
            return None

        cat_sujets = CATEGORIES[cat_key]["sujets"]
        sujet_idx = s.get("sujet_idx", -1)
        contexte_idx = s.get("contexte_idx", -1)
        angle_idx = s.get("angle_idx", -1)
        profil_idx = s.get("profil_idx", -1)

        # Validation des indices
        if not (0 <= sujet_idx < len(cat_sujets)):
            print(f"  Indice sujet invalide: {sujet_idx} (max {len(cat_sujets)-1}) pour {cat_key}")
            return None
        if not (0 <= contexte_idx < len(CONTEXTES)):
            print(f"  Indice contexte invalide: {contexte_idx} (max {len(CONTEXTES)-1})")
            return None
        if not (0 <= angle_idx < len(ANGLES)):
            print(f"  Indice angle invalide: {angle_idx} (max {len(ANGLES)-1})")
            return None

        profil = None
        if 0 <= profil_idx < len(PROFILS):
            profil = PROFILS[profil_idx]

        resolved.append({
            "category_key": cat_key,
            "sujet": cat_sujets[sujet_idx],
            "contexte": CONTEXTES[contexte_idx],
            "angle": ANGLES[angle_idx],
            "profil": profil,
            "justification": s.get("justification", ""),
        })

    return resolved


def is_combo_used(matrix, cat_key, sujet, contexte):
    """Vérifie si une combinaison catégorie+sujet+contexte a déjà été utilisée."""
    for a in matrix.get("articles", []):
        if a["category_key"] == cat_key and a["sujet"] == sujet and a["contexte"] == contexte:
            return True
    return False


def get_gemini_suggestions(matrix):
    """Appelle Gemini pour obtenir des suggestions de sujets."""
    matrix_summary = build_matrix_summary(matrix)
    system_prompt, user_prompt = build_gemini_prompt(matrix_summary)

    print("  Appel Gemini (analyse de la matrice)...")
    raw_response = call_mammouth_api(
        model=MODEL_ANALYST,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=2000
    )

    if not raw_response:
        print("  Gemini n'a pas repondu")
        return None

    suggestions = parse_gemini_suggestions(raw_response)
    if not suggestions:
        return None

    # Vérifier que les combinaisons ne sont pas déjà utilisées
    all_valid = True
    for s in suggestions:
        if is_combo_used(matrix, s["category_key"], s["sujet"], s["contexte"]):
            print(f"  Combinaison deja utilisee: {s['sujet']} / {s['contexte']}")
            all_valid = False
        else:
            print(f"  Suggestion validee [{s['category_key']}]: {s['sujet']} / {s['contexte']} / {s['angle']}")

    if not all_valid:
        print("  Certaines suggestions sont des doublons, retry necessaire")
        return None

    # Vérifier qu'on a bien 1 par catégorie
    cats = [s["category_key"] for s in suggestions]
    if sorted(cats) != ["cat1_pensees", "cat2_emotions", "cat3_schemas"]:
        print(f"  Les categories ne sont pas correctes: {cats}")
        return None

    return suggestions


# ============================================
# GÉNÉRATION ALÉATOIRE (FALLBACK)
# ============================================

def generate_random_combination(category_key, matrix):
    """Génère une combinaison unique aléatoire (fallback si Gemini échoue)."""
    category = CATEGORIES[category_key]
    max_attempts = 200

    for _ in range(max_attempts):
        sujet = random.choice(category["sujets"])
        contexte = random.choice(CONTEXTES)
        angle = random.choice(ANGLES)

        if not is_combo_used(matrix, category_key, sujet, contexte):
            profil = random.choice(PROFILS) if random.random() < 0.33 else None
            return {
                "category_key": category_key,
                "category_name": category["name"],
                "category_slug": category["slug"],
                "sujet": sujet,
                "contexte": contexte,
                "angle": angle,
                "profil": profil,
                "prenom": random.choice(PRENOMS),
                "age": random.choice(TRANCHES_AGE),
            }

    raise Exception(f"Impossible de trouver une combinaison unique pour {category_key}")


# ============================================
# GPT : RÉDACTION DES ARTICLES
# ============================================

def build_system_prompt():
    """Construit le system prompt pour GPT (rédaction)."""
    return """Tu es un rédacteur expert en psychologie vulgarisée et en SEO francophone. Tu rédiges des articles de blog pour le site "Décode ton esprit", dont la mission est d'aider les lecteurs à mieux se comprendre eux-mêmes grâce à la psychologie humaine.

MÉTHODE NARRATIVE OBLIGATOIRE :
- Chaque article raconte l'HISTOIRE d'un personnage fictif (prénom, âge, situation fournis)
- Le personnage est confronté à une situation concrète du quotidien liée au concept psychologique
- Le lecteur doit se reconnaître dans cette histoire
- L'histoire sert de fil conducteur pour expliquer le concept et les solutions
- Le personnage évolue au fil de l'article : il comprend son fonctionnement et commence à changer
- L'histoire doit être réaliste, touchante, avec des détails sensoriels et émotionnels
- TEMPS DE NARRATION : l'histoire DOIT être racontée au PRÉSENT de l'indicatif. Le personnage vit la scène en temps réel, comme si le lecteur assistait à la situation au moment où elle se produit (ex: "Sophie ouvre son ordinateur. Ses mains tremblent légèrement." et NON "Sophie a ouvert son ordinateur. Ses mains tremblaient."). Le présent crée une immersion immédiate et une connexion émotionnelle plus forte avec le lecteur.

RÈGLES DE RÉDACTION :
- Langue : français impeccable et fluide
- N'utilise JAMAIS le tiret long (—), le tiret cadratin ni le tiret semi-cadratin (–) comme ponctuation dans le texte. Utilise des virgules, des parenthèses ou reformule la phrase autrement. Les traits d'union (-) dans les mots composés (ex : moi-même, peut-être) sont autorisés. Ceci est une règle absolue.
- Ton : accessible, empathique, bienveillant, non-jugeant, chaleureux
- Vulgarise TOUJOURS les concepts complexes avec des mots simples et des métaphores du quotidien
- Inclus des exemples concrets auxquels le lecteur s'identifie
- Inclus TOUJOURS 2-3 techniques ou exercices pratiques actionnables
- Commence TOUJOURS par une scène narrative immersive (le personnage en situation)
- Termine par l'évolution du personnage + un message d'espoir pour le lecteur
- Longueur : entre 1800 et 2500 mots

RÈGLES SEO :
- Place le mot-clé principal naturellement dans le titre (H1), le premier paragraphe, au moins 2 sous-titres H2, et la conclusion
- Structure avec des H2 et H3 clairs
- Paragraphes courts (3-4 lignes max)
- Utilise des listes à puces ou numérotées pour les techniques
- Intègre 2-3 questions dans les H2 (ex: "Qu'est-ce que le biais de confirmation ?", "Comment se manifeste la peur de l'abandon au travail ?")

RÈGLES GEO (pour être cité par les IA) :
- Quand tu introduis un concept psychologique, donne UNE DÉFINITION CLAIRE EN UNE PHRASE au début de la section (ex: "Le biais de confirmation est la tendance à chercher et favoriser les informations qui confirment nos croyances existantes.")
- Cite le nom du chercheur ou psychologue associé au concept quand c'est pertinent
- Utilise des données chiffrées quand c'est possible (ex: "Selon une étude de 2019 publiée dans...")
- Structure les techniques en listes numérotées avec des titres clairs

RÈGLES ÉTHIQUES :
- Ne donne JAMAIS de diagnostic médical
- Inclus TOUJOURS un rappel bienveillant que consulter un professionnel est recommandé pour les difficultés persistantes
- Ne minimise jamais la souffrance du lecteur

FORMAT DE SORTIE :
Tu dois retourner EXACTEMENT ce format, rien d'autre :

TITRE_SEO: {titre optimisé SEO, max 65 caractères}
META_DESCRIPTION: {description unique, max 155 caractères, qui donne envie de cliquer}
SLUG: {slug-en-minuscules-avec-tirets-sans-accents, max 6 mots}
TAGS: {tag1, tag2, tag3, tag4, tag5}
---
{contenu complet de l'article en Markdown, commençant directement par le texte sans répéter le titre H1}"""


def build_article_prompt(combo):
    """Construit le prompt spécifique pour un article."""
    profil_text = f"\nPROFIL DU LECTEUR CIBLE : {combo['profil']}" if combo.get('profil') else ""

    return f"""Rédige un article de blog complet avec les paramètres suivants :

CATÉGORIE : {combo['category_name']}
SUJET PRINCIPAL : {combo['sujet']}
CONTEXTE DE VIE : {combo['contexte']}
ANGLE ÉDITORIAL : {combo['angle']}{profil_text}

PERSONNAGE DE L'HISTOIRE :
- Prénom : {combo['prenom']}
- Âge : {combo['age']}
- Situation : {combo['prenom']} vit une situation liée à "{combo['sujet']}" dans le contexte "{combo['contexte']}"

CONSIGNES SPÉCIFIQUES :
1. Commence par plonger le lecteur dans une scène de la vie de {combo['prenom']} (3-4 paragraphes immersifs). IMPORTANT : raconte l'histoire AU PRÉSENT, comme si elle se déroule sous les yeux du lecteur en ce moment même. {combo['prenom']} vit la situation en temps réel.
2. Fais le lien entre la situation de {combo['prenom']} et le concept de "{combo['sujet']}"
3. Explique le concept de manière simple et accessible, avec sa définition claire
4. Montre comment ce concept se manifeste dans le contexte "{combo['contexte']}" avec d'autres exemples
5. Propose 2-3 techniques/exercices concrets que le lecteur peut appliquer
6. Reviens à {combo['prenom']} pour montrer comment il/elle évolue en appliquant ces techniques
7. Conclus avec un message d'espoir et une invitation à réfléchir sur soi

MOT-CLÉ SEO À OPTIMISER : {combo['sujet']} {combo['contexte']}"""


# ============================================
# GEMINI : VÉRIFICATION DES TIRETS CADRATINS
# ============================================

EMDASH = "\u2014"  # —
ENDASH = "\u2013"  # –


def call_gemini_dash_fix(content):
    """Appelle Gemini pour reformuler les passages contenant des tirets cadratins ou semi-cadratins utilisés comme ponctuation."""
    system_prompt = (
        "Tu es un correcteur linguistique français expert. "
        "On te donne un article de blog en Markdown. "
        "Ton UNIQUE tâche : repérer tous les tirets cadratins (\u2014) et semi-cadratins (\u2013) "
        "utilisés comme PONCTUATION dans le texte "
        "et reformuler ces passages pour les remplacer par une ponctuation française standard "
        "(virgules, parenthèses, deux-points, points). "
        "ATTENTION : les traits d'union dans les mots composés (ex : moi-même, peut-être, c'est-à-dire) "
        "doivent être CONSERVÉS tels quels, ce sont des traits d'union normaux (-), pas des tirets de ponctuation. "
        "Tu dois retourner l'article COMPLET avec les corrections appliquées, "
        "sans rien supprimer ni modifier d'autre. "
        "Conserve tout le formatage Markdown intact (titres, listes, gras, italique, etc.). "
        "Ne rajoute AUCUN commentaire, AUCUNE explication. Retourne UNIQUEMENT l'article corrigé."
    )

    user_prompt = (
        "Voici l'article à corriger. Remplace chaque tiret cadratin (\u2014) et semi-cadratin (\u2013) "
        "utilisé comme ponctuation par une reformulation "
        "naturelle avec une ponctuation standard (virgule, parenthèse, deux-points, point). "
        "Ne touche PAS aux traits d'union dans les mots composés. "
        "Retourne l'article complet corrigé, rien d'autre :\n\n"
        f"{content}"
    )

    return call_mammouth_api(
        model=MODEL_ANALYST,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=4500,
        retries=2
    )


def _count_punctuation_dashes(content):
    """Compte les tirets cadratins et semi-cadratins utilisés comme ponctuation (entourés d'espaces)."""
    import re
    # Tirets entourés d'espaces = ponctuation (pas des traits d'union dans des mots composés)
    pattern = r' [–—] '
    return len(re.findall(pattern, content))


def verify_and_fix_emdashes(content, combo):
    """Vérifie et corrige les tirets cadratins et semi-cadratins utilisés comme ponctuation."""
    dash_count = _count_punctuation_dashes(content)

    if dash_count == 0:
        print(f"  [Vérification] OK : aucun tiret de ponctuation détecté")
        return content

    print(f"  [Vérification] {dash_count} tiret(s) de ponctuation détecté(s), appel Gemini pour correction...")

    try:
        fixed_content = call_gemini_dash_fix(content)

        if fixed_content:
            remaining = _count_punctuation_dashes(fixed_content)
            if remaining == 0:
                print(f"  [Vérification] Gemini a corrigé tous les tirets de ponctuation avec succès")
                return fixed_content
            else:
                print(f"  [Vérification] Gemini a laissé {remaining} tiret(s), fallback mécanique...")
                import re
                return re.sub(r' [–—] ', ', ', fixed_content)
        else:
            print(f"  [Vérification] Gemini n'a pas répondu, fallback mécanique...")
            import re
            return re.sub(r' [–—] ', ', ', content)

    except Exception as e:
        print(f"  [Vérification] Erreur Gemini: {e}, fallback mécanique...")
        import re
        return re.sub(r' [–—] ', ', ', content)


# ============================================
# PARSING ET CRÉATION HUGO
# ============================================

def parse_article_response(raw_response):
    """Parse la réponse de l'API pour extraire les métadonnées et le contenu."""
    lines = raw_response.strip().split("\n")

    metadata = {}
    content_start = 0

    for i, line in enumerate(lines):
        if line.startswith("TITRE_SEO:"):
            metadata["title"] = line.replace("TITRE_SEO:", "").strip()
        elif line.startswith("META_DESCRIPTION:"):
            metadata["description"] = line.replace("META_DESCRIPTION:", "").strip()
        elif line.startswith("SLUG:"):
            metadata["slug"] = line.replace("SLUG:", "").strip()
        elif line.startswith("TAGS:"):
            metadata["tags"] = [t.strip() for t in line.replace("TAGS:", "").split(",")]
        elif line.strip() == "---":
            content_start = i + 1
            break

    content = "\n".join(lines[content_start:]).strip()

    return metadata, content


def create_hugo_post(combo, metadata, content):
    """Crée le fichier Markdown Hugo pour l'article."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    date_short = datetime.now().strftime("%Y-%m-%d")

    word_count = len(content.split())
    reading_time = max(1, round(word_count / 250))

    tags_str = json.dumps(metadata.get("tags", []), ensure_ascii=False)

    front_matter = f"""---
title: "{metadata.get('title', 'Article du jour')}"
date: {date_str}
description: "{metadata.get('description', '')}"
categories: ["{combo['category_name']}"]
tags: {tags_str}
slug: "{metadata.get('slug', date_short)}"
readingTime: {reading_time}
wordCount: {word_count}
personnage: "{combo['prenom']}"
sujet: "{combo['sujet']}"
contexte: "{combo['contexte']}"
draft: false
---"""

    full_content = f"{front_matter}\n\n{content}"

    category_dir = CONTENT_DIR / combo["category_slug"]
    category_dir.mkdir(parents=True, exist_ok=True)

    file_path = category_dir / f"{metadata.get('slug', date_short)}.md"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"  Article cree : {file_path}")
    return file_path


# ============================================
# MAIN
# ============================================

def main():
    print(f"\n{'='*60}")
    print(f"GENERATION D'ARTICLES - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    if not MAMMOUTH_API_KEY:
        raise ValueError("MAMMOUTH_API_KEY non definie !")

    category_keys = ["cat1_pensees", "cat2_emotions", "cat3_schemas"]

    # ── ÉTAPE 1 : Charger la matrice des combinaisons ──
    print("ETAPE 1 : Chargement de la matrice des combinaisons...")
    matrix = load_matrix()

    # Migrer les anciennes données si nécessaire
    migrate_tracking_to_matrix(matrix)

    print(f"  {len(matrix['articles'])} combinaisons dans la matrice\n")

    # ── ÉTAPE 2 : Demander à Gemini des suggestions intelligentes ──
    print("ETAPE 2 : Analyse par Gemini et suggestion de sujets...")
    suggestions = None
    for attempt in range(3):
        try:
            suggestions = get_gemini_suggestions(matrix)
            if suggestions:
                break
        except Exception as e:
            print(f"  Erreur Gemini tentative {attempt + 1}: {e}")
        if attempt < 2:
            print(f"  Nouvelle tentative Gemini ({attempt + 2}/3)...")

    # Fallback : si Gemini échoue, utiliser la méthode aléatoire
    use_gemini = suggestions is not None
    if not use_gemini:
        print("  Gemini n'a pas pu fournir de suggestions valides, fallback aleatoire\n")

    # ── ÉTAPE 3 : Rédiger les articles avec GPT ──
    print("\nETAPE 3 : Redaction des articles par GPT...")
    system_prompt = build_system_prompt()

    for i, cat_key in enumerate(category_keys):
        print(f"\n  --- Categorie : {CATEGORIES[cat_key]['name']} ---")

        # Construire la combinaison (Gemini ou aléatoire)
        if use_gemini:
            s = suggestions[i]
            combo = {
                "category_key": s["category_key"],
                "category_name": CATEGORIES[s["category_key"]]["name"],
                "category_slug": CATEGORIES[s["category_key"]]["slug"],
                "sujet": s["sujet"],
                "contexte": s["contexte"],
                "angle": s["angle"],
                "profil": s.get("profil"),
                "prenom": random.choice(PRENOMS),
                "age": random.choice(TRANCHES_AGE),
            }
            print(f"  [Gemini] Sujet : {combo['sujet']}")
            print(f"  [Gemini] Contexte : {combo['contexte']}")
            print(f"  [Gemini] Angle : {combo['angle']}")
            if s.get("justification"):
                print(f"  [Gemini] Raison : {s['justification']}")
        else:
            combo = generate_random_combination(cat_key, matrix)
            print(f"  [Aleatoire] Sujet : {combo['sujet']}")
            print(f"  [Aleatoire] Contexte : {combo['contexte']}")
            print(f"  [Aleatoire] Angle : {combo['angle']}")

        print(f"  Personnage : {combo['prenom']} ({combo['age']})")
        if combo.get('profil'):
            print(f"  Profil : {combo['profil']}")

        print(f"  Appel GPT-4.1-mini pour redaction...")
        article_prompt = build_article_prompt(combo)
        raw_response = call_mammouth_api(
            model=MODEL_WRITER,
            system_prompt=system_prompt,
            user_prompt=article_prompt,
            temperature=0.85,
            max_tokens=4500
        )

        print(f"  Parsing de la reponse...")
        metadata, content = parse_article_response(raw_response)

        # Étape 3.5 : Vérification et correction des tirets cadratins (—)
        content = verify_and_fix_emdashes(content, combo)

        print(f"  Creation du fichier Hugo...")
        create_hugo_post(combo, metadata, content)

        # Ajouter à la matrice
        add_to_matrix(matrix, combo, metadata)

    # Sauvegarder la matrice
    save_matrix(matrix)
    print(f"\n{'='*60}")
    print(f"Generation terminee - {len(matrix['articles'])} combinaisons dans la matrice")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
