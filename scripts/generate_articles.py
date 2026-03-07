#!/usr/bin/env python3
"""
Génère 3 articles de blog quotidiens (1 par catégorie)
en appelant l'API Mammouth (Gemini pour la suggestion, GPT pour la rédaction)
et en créant des fichiers Markdown Hugo.

Flux :
1. Scan des articles existants sur le site
2. Gemini analyse le contenu existant et suggère 3 sujets uniques
3. GPT-4.1-mini rédige les articles sur les suggestions de Gemini
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

sys.path.insert(0, str(SCRIPT_DIR))

MAMMOUTH_API_KEY = os.environ.get("MAMMOUTH_API_KEY")
API_URL = "https://api.mammouth.ai/v1/chat/completions"
MODEL_WRITER = "gpt-4.1-mini"
MODEL_ANALYST = "gemini-2.5-flash-preview-05-20"
SITE_NAME = "Décode ton esprit"

from config import (
    CATEGORIES, CONTEXTES, PROFILS, ANGLES,
    PRENOMS, TRANCHES_AGE
)

# ============================================
# SCAN DES ARTICLES EXISTANTS
# ============================================

def scan_existing_articles():
    """Scanne tous les articles existants et extrait leurs métadonnées."""
    articles = []

    for category_dir in CONTENT_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        for md_file in category_dir.glob("*.md"):
            if md_file.name == "_index.md":
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                meta = parse_front_matter(content)
                if meta:
                    articles.append(meta)
            except Exception as e:
                print(f"  Erreur lecture {md_file}: {e}")

    return articles


def parse_front_matter(content):
    """Extrait les métadonnées du front matter YAML d'un fichier Markdown."""
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    front_matter = parts[1].strip()
    meta = {}

    for line in front_matter.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key in ("title", "sujet", "contexte", "personnage", "description", "slug"):
                meta[key] = value
            elif key == "categories":
                # Parse ["Catégorie"]
                match = re.findall(r'"([^"]+)"', value)
                if match:
                    meta["category"] = match[0]
            elif key == "tags":
                match = re.findall(r'"([^"]+)"', value)
                if match:
                    meta["tags"] = match

    return meta if meta.get("title") else None


def build_articles_summary(articles):
    """Construit un résumé textuel de tous les articles existants pour Gemini."""
    if not articles:
        return "Aucun article n'a encore été publié sur le site."

    summary_lines = [f"Le site compte actuellement {len(articles)} articles publiés :\n"]

    for i, art in enumerate(articles, 1):
        line = f"{i}. [{art.get('category', '?')}] \"{art.get('title', '?')}\""
        if art.get("sujet"):
            line += f" — Sujet: {art['sujet']}"
        if art.get("contexte"):
            line += f" — Contexte: {art['contexte']}"
        if art.get("tags"):
            line += f" — Tags: {', '.join(art['tags'])}"
        summary_lines.append(line)

    return "\n".join(summary_lines)

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
            response = requests.post(API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Tentative {attempt + 1}/{retries} echouee : {e}")
            if attempt == retries - 1:
                raise

    return None


def build_gemini_prompt(articles_summary, tracking_data):
    """Construit le prompt pour Gemini afin qu'il suggère des sujets uniques."""

    # Construire la liste des sujets disponibles par catégorie
    available_subjects = {}
    for cat_key, cat_data in CATEGORIES.items():
        available_subjects[cat_data["name"]] = cat_data["sujets"]

    # Construire la liste des combinaisons déjà utilisées
    used_combos = tracking_data.get("generated", [])
    used_text = "\n".join(f"- {c}" for c in used_combos) if used_combos else "Aucune combinaison utilisée."

    system_prompt = """Tu es un directeur éditorial expert en psychologie vulgarisée. Tu analyses le contenu existant d'un blog de psychologie et tu proposes des sujets d'articles pertinents, originaux et complémentaires à ce qui existe déjà.

Tu dois éviter :
- Les sujets déjà traités (même sous un angle différent, sauf si l'angle est VRAIMENT distinct)
- Les combinaisons sujet+contexte trop proches de ce qui existe (ex: "colère en couple" et "frustration en couple" sont trop proches)
- La redondance thématique (ne pas proposer 3 articles qui parlent tous de la même famille de concepts)

Tu dois privilégier :
- La diversité des sujets au sein de chaque catégorie
- Des combinaisons sujet+contexte qui créent des articles intéressants et recherchés
- Des angles éditoriaux variés
- Des contextes de vie différents de ceux déjà traités

IMPORTANT : Tu dois répondre UNIQUEMENT en JSON valide, sans aucun texte avant ou après."""

    user_prompt = f"""Voici l'état actuel du blog "Décode ton esprit" :

{articles_summary}

COMBINAISONS DÉJÀ UTILISÉES (format: catégorie|sujet|contexte|angle) :
{used_text}

SUJETS DISPONIBLES PAR CATÉGORIE :

1. CATÉGORIE "Reprendre le contrôle de ses pensées" (cat1_pensees) :
{json.dumps(available_subjects["Reprendre le contrôle de ses pensées"], ensure_ascii=False, indent=2)}

2. CATÉGORIE "Comprendre et maîtriser ses émotions" (cat2_emotions) :
{json.dumps(available_subjects["Comprendre et maîtriser ses émotions"], ensure_ascii=False, indent=2)}

3. CATÉGORIE "Sortir de ses schémas répétitifs" (cat3_schemas) :
{json.dumps(available_subjects["Sortir de ses schémas répétitifs"], ensure_ascii=False, indent=2)}

CONTEXTES DISPONIBLES :
{json.dumps(CONTEXTES, ensure_ascii=False, indent=2)}

ANGLES ÉDITORIAUX DISPONIBLES :
{json.dumps(ANGLES, ensure_ascii=False, indent=2)}

PROFILS DE LECTEURS (optionnel, à utiliser 1 fois sur 3) :
{json.dumps(PROFILS, ensure_ascii=False, indent=2)}

Propose exactement 3 sujets d'articles (1 par catégorie) au format JSON suivant :

```json
[
  {{
    "category_key": "cat1_pensees",
    "sujet": "le sujet exact de la liste ci-dessus",
    "contexte": "le contexte exact de la liste ci-dessus",
    "angle": "l'angle exact de la liste ci-dessus",
    "profil": "le profil exact de la liste ci-dessus ou null",
    "justification": "pourquoi ce sujet est pertinent et complémentaire"
  }},
  {{
    "category_key": "cat2_emotions",
    "sujet": "...",
    "contexte": "...",
    "angle": "...",
    "profil": "... ou null",
    "justification": "..."
  }},
  {{
    "category_key": "cat3_schemas",
    "sujet": "...",
    "contexte": "...",
    "angle": "...",
    "profil": "... ou null",
    "justification": "..."
  }}
]
```

RÈGLES STRICTES :
- Chaque "sujet" DOIT être un élément EXACT de la liste de sujets de sa catégorie
- Chaque "contexte" DOIT être un élément EXACT de la liste de contextes
- Chaque "angle" DOIT être un élément EXACT de la liste d'angles
- Chaque "profil" DOIT être un élément EXACT de la liste de profils, ou null
- Les combinaisons NE DOIVENT PAS exister dans les combinaisons déjà utilisées
- Réponds UNIQUEMENT avec le JSON, sans texte autour, sans backticks markdown"""

    return system_prompt, user_prompt


def parse_gemini_suggestions(raw_response):
    """Parse la réponse JSON de Gemini."""
    # Nettoyer la réponse (enlever backticks markdown si présents)
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        suggestions = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  Erreur parsing JSON Gemini: {e}")
        print(f"  Réponse brute: {raw_response[:500]}")
        return None

    if not isinstance(suggestions, list) or len(suggestions) != 3:
        print(f"  Gemini n'a pas retourné exactement 3 suggestions (reçu: {len(suggestions) if isinstance(suggestions, list) else 'pas une liste'})")
        return None

    return suggestions


def validate_suggestion(suggestion, tracking_data):
    """Vérifie qu'une suggestion de Gemini est valide."""
    cat_key = suggestion.get("category_key")
    if cat_key not in CATEGORIES:
        return False, f"Catégorie inconnue: {cat_key}"

    sujet = suggestion.get("sujet")
    if sujet not in CATEGORIES[cat_key]["sujets"]:
        return False, f"Sujet inconnu pour {cat_key}: {sujet}"

    contexte = suggestion.get("contexte")
    if contexte not in CONTEXTES:
        return False, f"Contexte inconnu: {contexte}"

    angle = suggestion.get("angle")
    if angle not in ANGLES:
        return False, f"Angle inconnu: {angle}"

    profil = suggestion.get("profil")
    if profil is not None and profil not in PROFILS:
        return False, f"Profil inconnu: {profil}"

    # Vérifier que la combinaison n'existe pas déjà
    combo_id = f"{cat_key}|{sujet}|{contexte}|{angle}"
    if combo_id in tracking_data.get("generated", []):
        return False, f"Combinaison déjà utilisée: {combo_id}"

    return True, "OK"


def get_gemini_suggestions(articles_summary, tracking_data):
    """Appelle Gemini pour obtenir des suggestions de sujets."""
    system_prompt, user_prompt = build_gemini_prompt(articles_summary, tracking_data)

    print("  Appel Gemini (analyse du contenu existant)...")
    raw_response = call_mammouth_api(
        model=MODEL_ANALYST,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=2000
    )

    suggestions = parse_gemini_suggestions(raw_response)
    if not suggestions:
        return None

    # Valider chaque suggestion
    valid_suggestions = []
    for s in suggestions:
        is_valid, msg = validate_suggestion(s, tracking_data)
        if is_valid:
            valid_suggestions.append(s)
            print(f"  Suggestion validée [{s['category_key']}]: {s['sujet']} / {s['contexte']}")
        else:
            print(f"  Suggestion rejetée: {msg}")

    return valid_suggestions if len(valid_suggestions) == 3 else None

# ============================================
# GÉNÉRATION ALÉATOIRE (FALLBACK)
# ============================================

def generate_combination(category_key, tracking_data):
    """Génère une combinaison unique sujet + contexte + angle (fallback si Gemini échoue)."""
    category = CATEGORIES[category_key]
    max_attempts = 100

    for _ in range(max_attempts):
        sujet = random.choice(category["sujets"])
        contexte = random.choice(CONTEXTES)
        angle = random.choice(ANGLES)

        combo_id = f"{category_key}|{sujet}|{contexte}|{angle}"

        if combo_id not in tracking_data["generated"]:
            profil = random.choice(PROFILS) if random.random() < 0.33 else None
            return {
                "category_key": category_key,
                "sujet": sujet,
                "contexte": contexte,
                "angle": angle,
                "profil": profil,
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

RÈGLES DE RÉDACTION :
- Langue : français impeccable et fluide
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
1. Commence par plonger le lecteur dans une scène de la vie de {combo['prenom']} (3-4 paragraphes immersifs)
2. Fais le lien entre la situation de {combo['prenom']} et le concept de "{combo['sujet']}"
3. Explique le concept de manière simple et accessible, avec sa définition claire
4. Montre comment ce concept se manifeste dans le contexte "{combo['contexte']}" avec d'autres exemples
5. Propose 2-3 techniques/exercices concrets que le lecteur peut appliquer
6. Reviens à {combo['prenom']} pour montrer comment il/elle évolue en appliquant ces techniques
7. Conclus avec un message d'espoir et une invitation à réfléchir sur soi

MOT-CLÉ SEO À OPTIMISER : {combo['sujet']} {combo['contexte']}"""

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

def load_tracking():
    """Charge le fichier de tracking des articles déjà générés."""
    if TRACKING_FILE.exists():
        try:
            with open(TRACKING_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "generated" not in data:
                    return {"generated": []}
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Attention: fichier de tracking corrompu, reinitialisation ({e})")
            return {"generated": []}
    return {"generated": []}


def save_tracking(data):
    """Sauvegarde le fichier de tracking."""
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print(f"\n{'='*60}")
    print(f"GENERATION D'ARTICLES - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    if not MAMMOUTH_API_KEY:
        raise ValueError("MAMMOUTH_API_KEY non definie !")

    tracking = load_tracking()
    category_keys = ["cat1_pensees", "cat2_emotions", "cat3_schemas"]

    # ── ÉTAPE 1 : Scanner les articles existants ──
    print("ETAPE 1 : Scan des articles existants...")
    existing_articles = scan_existing_articles()
    articles_summary = build_articles_summary(existing_articles)
    print(f"  {len(existing_articles)} articles trouves sur le site\n")

    # ── ÉTAPE 2 : Demander à Gemini des suggestions intelligentes ──
    print("ETAPE 2 : Analyse par Gemini et suggestion de sujets...")
    suggestions = None
    for attempt in range(2):
        suggestions = get_gemini_suggestions(articles_summary, tracking)
        if suggestions:
            break
        print(f"  Nouvelle tentative Gemini ({attempt + 2}/2)...")

    # Fallback : si Gemini échoue, utiliser la méthode aléatoire
    use_gemini = suggestions is not None
    if not use_gemini:
        print("  Gemini n'a pas pu fournir de suggestions valides, fallback aleatoire\n")

    # ── ÉTAPE 3 : Rédiger les articles avec GPT ──
    print("\nETAPE 3 : Redaction des articles par GPT...")
    system_prompt = build_system_prompt()

    for i, cat_key in enumerate(category_keys):
        print(f"\n  Categorie : {CATEGORIES[cat_key]['name']}")

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
            combo_id = f"{cat_key}|{combo['sujet']}|{combo['contexte']}|{combo['angle']}"
            tracking["generated"].append(combo_id)
            print(f"  [Gemini] Sujet : {combo['sujet']}")
            print(f"  [Gemini] Contexte : {combo['contexte']}")
            print(f"  [Gemini] Angle : {combo['angle']}")
            if s.get("justification"):
                print(f"  [Gemini] Justification : {s['justification']}")
        else:
            combo = generate_combination(cat_key, tracking)
            combo["category_name"] = CATEGORIES[cat_key]["name"]
            combo["category_slug"] = CATEGORIES[cat_key]["slug"]
            tracking["generated"].append(
                f"{cat_key}|{combo['sujet']}|{combo['contexte']}|{combo['angle']}"
            )
            combo["prenom"] = random.choice(PRENOMS)
            combo["age"] = random.choice(TRANCHES_AGE)
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

        print(f"  Creation du fichier Hugo...")
        create_hugo_post(combo, metadata, content)

    # Sauvegarder le tracking
    save_tracking(tracking)
    print(f"\n{'='*60}")
    print(f"Generation terminee - {len(tracking['generated'])} combinaisons tracees")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
