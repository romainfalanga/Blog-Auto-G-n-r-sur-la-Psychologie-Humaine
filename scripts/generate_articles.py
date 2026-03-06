#!/usr/bin/env python3
"""
Génère 3 articles de blog quotidiens (1 par catégorie)
en appelant l'API Mammouth et en créant des fichiers Markdown Hugo.
"""

import os
import json
import random
import requests
import re
from datetime import datetime, timezone
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

MAMMOUTH_API_KEY = os.environ.get("MAMMOUTH_API_KEY")
API_URL = "https://api.mammouth.ai/v1/chat/completions"
MODEL = "gpt-4.1-mini"
SITE_NAME = "Décode ton esprit"
BASE_URL = "https://decodetonsesprit.netlify.app"

# Résoudre les chemins par rapport à la racine du repo (pas le répertoire courant)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONTENT_DIR = REPO_ROOT / "content" / "posts"
TRACKING_FILE = SCRIPT_DIR / "generated_topics.json"

# Import des listes depuis config.py
from config import (
    CATEGORIES, CONTEXTES, PROFILS, ANGLES,
    PRENOMS, TRANCHES_AGE
)

# ============================================
# FONCTIONS
# ============================================

def load_tracking():
    """Charge le fichier de tracking des articles déjà générés."""
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"generated": []}

def save_tracking(data):
    """Sauvegarde le fichier de tracking."""
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_combination(category_key, tracking_data):
    """Génère une combinaison unique sujet + contexte + angle."""
    category = CATEGORIES[category_key]
    max_attempts = 100

    for _ in range(max_attempts):
        sujet = random.choice(category["sujets"])
        contexte = random.choice(CONTEXTES)
        angle = random.choice(ANGLES)

        combo_id = f"{category_key}|{sujet}|{contexte}|{angle}"

        if combo_id not in tracking_data["generated"]:
            # Profil : 1 fois sur 3
            profil = random.choice(PROFILS) if random.random() < 0.33 else None
            prenom = random.choice(PRENOMS)
            age = random.choice(TRANCHES_AGE)

            tracking_data["generated"].append(combo_id)

            return {
                "category_key": category_key,
                "category_name": category["name"],
                "category_slug": category["slug"],
                "sujet": sujet,
                "contexte": contexte,
                "angle": angle,
                "profil": profil,
                "prenom": prenom,
                "age": age
            }

    raise Exception(f"Impossible de trouver une combinaison unique pour {category_key}")

def build_system_prompt():
    """Construit le system prompt pour l'API."""
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
    profil_text = f"\nPROFIL DU LECTEUR CIBLE : {combo['profil']}" if combo['profil'] else ""

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

def call_mammouth_api(system_prompt, article_prompt, retries=3):
    """Appelle l'API Mammouth avec retry."""
    headers = {
        "Authorization": f"Bearer {MAMMOUTH_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "temperature": 0.85,
        "max_tokens": 4500,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": article_prompt}
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

    # Estimer le temps de lecture (250 mots/min)
    word_count = len(content.split())
    reading_time = max(1, round(word_count / 250))

    # Construire le front matter
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

    # Sauvegarder
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

    tracking = load_tracking()
    system_prompt = build_system_prompt()

    categories = ["cat1_pensees", "cat2_emotions", "cat3_schemas"]

    for cat_key in categories:
        print(f"\nCategorie : {CATEGORIES[cat_key]['name']}")
        print(f"   Selection d'une combinaison unique...")

        combo = generate_combination(cat_key, tracking)
        print(f"   Sujet : {combo['sujet']}")
        print(f"   Contexte : {combo['contexte']}")
        print(f"   Angle : {combo['angle']}")
        print(f"   Personnage : {combo['prenom']} ({combo['age']})")
        if combo['profil']:
            print(f"   Profil : {combo['profil']}")

        print(f"   Appel API Mammouth...")
        article_prompt = build_article_prompt(combo)
        raw_response = call_mammouth_api(system_prompt, article_prompt)

        print(f"   Parsing de la reponse...")
        metadata, content = parse_article_response(raw_response)

        print(f"   Creation du fichier Hugo...")
        create_hugo_post(combo, metadata, content)

    # Sauvegarder le tracking
    save_tracking(tracking)
    print(f"\n{'='*60}")
    print(f"Generation terminee - {len(tracking['generated'])} articles au total")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
