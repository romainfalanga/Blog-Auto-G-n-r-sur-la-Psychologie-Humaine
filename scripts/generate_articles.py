#!/usr/bin/env python3
"""
Génère 3 articles de blog quotidiens (1 par catégorie)
en appelant l'API Mammouth (Gemini pour la suggestion, GPT pour la rédaction)
et en créant des fichiers Markdown Hugo.

Flux :
1. Charger la matrice des combinaisons déjà réalisées (scripts/matrice_combinaisons.json)
2. Gemini analyse la matrice et suggère 3 nouvelles combinaisons uniques
3. GPT-5-mini rédige les articles sur les suggestions de Gemini
4. Gemini vérifie chaque article : si des tirets cadratins (—) sont détectés,
   Gemini reformule les passages concernés pour les supprimer
5. Gemini vérifie la structure H2/H3 et restructure si nécessaire
6. Gemini effectue une relecture qualité globale de chaque article
   (cohérence, suppression des tournures IA, qualité rédactionnelle)
7. La matrice est mise à jour avec les nouvelles combinaisons
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
MODEL_WRITER = "gpt-5-mini"
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

STRUCTURE OBLIGATOIRE DE L'ARTICLE (à respecter STRICTEMENT) :
Tu DOIS structurer chaque article avec des titres Markdown H2 (##) et H3 (###). Voici le squelette EXACT à suivre :

1. **Introduction narrative** (3-4 paragraphes SANS titre H2, directement le texte immersif du personnage)
2. **## Qu'est-ce que [concept] ?** (H2 avec le nom du concept, définition claire, contexte scientifique, chercheur associé)
3. **## Comment [concept] se manifeste-t-il [contexte] ?** (H2 sous forme de question, avec 2-3 sous-sections H3 montrant des manifestations concrètes)
4. **## [N] techniques pour [verbe d'action] face à [concept]** (H2, puis chaque technique en H3 numéroté avec titre en gras : ### 1. **Titre de la technique**)
5. **## [Prénom] commence à [verbe de transformation]** (H2, retour au personnage qui évolue grâce aux techniques)
6. **---** (séparateur horizontal obligatoire avant la conclusion)
7. **Conclusion** (3 paragraphes SANS titre H2 : bilan, message d'espoir, rappel professionnel)

CHAQUE article DOIT contenir au minimum 4 titres H2 et 3 titres H3. Ne JAMAIS écrire un article en prose continue sans titres. C'est une règle ABSOLUE et NON NÉGOCIABLE.

RÈGLES SEO :
- Place le mot-clé principal naturellement dans le titre (H1), le premier paragraphe, au moins 2 sous-titres H2, et la conclusion
- Paragraphes courts (3-4 lignes max)
- Utilise des listes à puces ou numérotées pour les techniques et les exemples
- Intègre 2-3 questions dans les H2 (ex: "Qu'est-ce que le biais de confirmation ?", "Comment se manifeste la peur de l'abandon au travail ?")
- Maillage sémantique : utilise des synonymes et variantes du mot-clé principal dans les H2/H3 et le corps du texte
- Chaque H2 doit idéalement contenir le mot-clé principal ou un synonyme proche

RÈGLES GEO (pour être cité par les IA) :
- Quand tu introduis un concept psychologique, donne UNE DÉFINITION CLAIRE EN UNE PHRASE au début de la section (ex: "Le biais de confirmation est la tendance à chercher et favoriser les informations qui confirment nos croyances existantes.")
- Cite le nom du chercheur ou psychologue associé au concept quand c'est pertinent
- Utilise des données chiffrées quand c'est possible (ex: "Selon une étude de 2019 publiée dans...")
- Structure les techniques en listes numérotées avec des titres clairs en H3

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

CONSIGNES SPÉCIFIQUES (structure OBLIGATOIRE avec titres H2/H3 Markdown) :
1. INTRODUCTION NARRATIVE (SANS titre H2) : 3-4 paragraphes immersifs plongeant le lecteur dans la vie de {combo['prenom']}. IMPORTANT : raconte AU PRÉSENT, comme si la scène se déroule sous les yeux du lecteur. {combo['prenom']} vit la situation en temps réel.
2. ## Qu'est-ce que {combo['sujet']} ? (titre H2 OBLIGATOIRE) : Fais le lien avec la situation de {combo['prenom']}, puis explique le concept avec une définition claire en une phrase, le nom du chercheur associé, et un contexte scientifique.
3. ## Comment {combo['sujet']} se manifeste dans le contexte "{combo['contexte']}" ? (titre H2 OBLIGATOIRE sous forme de question) : Détaille 2-3 manifestations concrètes avec des sous-titres H3 (###) pour chaque manifestation. Ajoute des exemples variés.
4. ## 3 techniques pour [verbe d'action] face à {combo['sujet']} (titre H2 OBLIGATOIRE) : Présente chaque technique avec un sous-titre H3 numéroté et en gras (### 1. **Nom de la technique**). Chaque technique doit être détaillée sur un paragraphe complet avec un exercice concret.
5. ## {combo['prenom']} commence à [verbe de transformation] (titre H2 OBLIGATOIRE) : Retour au personnage qui applique les techniques et évolue positivement.
6. --- (séparateur horizontal OBLIGATOIRE)
7. CONCLUSION (SANS titre H2) : 3 paragraphes avec bilan, message d'espoir, rappel bienveillant de consulter un professionnel.

RAPPEL CRITIQUE : L'article DOIT contenir au minimum 4 titres ## (H2) et 3 titres ### (H3). Un article sans cette structure sera rejeté.

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
# VÉRIFICATION DE LA STRUCTURE H2/H3
# ============================================

def call_gemini_quality_review(content, combo):
    """Appelle Gemini pour une relecture qualité globale de l'article.
    Vérifie la cohérence, supprime les tournures IA, et optimise la rédaction."""
    system_prompt = (
        "Tu es un relecteur professionnel spécialisé en articles de blog psychologie. "
        "On te donne un article de blog en Markdown. Tu dois effectuer une relecture complète :\n\n"
        "1. COHÉRENCE : vérifie que l'article est cohérent du début à la fin. "
        "Le personnage, son histoire, le concept psychologique et les techniques doivent former un tout logique. "
        "Corrige toute incohérence (contradictions, changements de prénom, de situation, de ton).\n\n"
        "2. STYLE IA : repère et reformule toutes les tournures qui sonnent artificielles ou générées par IA. "
        "Exemples de formulations à éliminer : \"Il est important de noter que\", \"Dans notre société actuelle\", "
        "\"Force est de constater\", \"Il convient de souligner\", \"En définitive\", \"Il est essentiel de\", "
        "\"N'hésitez pas à\", \"Il est crucial de\", \"Dans un monde où\", \"Qui n'a jamais\", "
        "\"Et si on vous disait que\", \"Vous l'aurez compris\". "
        "Remplace-les par des formulations naturelles, humaines et chaleureuses.\n\n"
        "3. QUALITÉ RÉDACTIONNELLE : corrige les maladresses de style, les répétitions excessives, "
        "les phrases trop longues ou alambiquées. Assure-toi que le ton reste accessible, empathique et bienveillant.\n\n"
        "4. PONCTUATION : vérifie qu'il n'y a aucun tiret cadratin (—) ni semi-cadratin (–) utilisé comme ponctuation. "
        "Si tu en trouves, reformule avec des virgules, parenthèses ou deux-points.\n\n"
        "RÈGLES ABSOLUES :\n"
        "- Conserve TOUTE la structure Markdown (titres H2, H3, listes, gras, italique, séparateur ---)\n"
        "- Ne supprime AUCUNE section, ne raccourcis PAS l'article\n"
        "- Conserve le même personnage, la même histoire, les mêmes techniques\n"
        "- Retourne l'article COMPLET corrigé, rien d'autre. Aucun commentaire, aucune explication."
    )

    user_prompt = (
        f"Voici un article sur \"{combo['sujet']}\" dans le contexte \"{combo['contexte']}\" "
        f"avec le personnage {combo['prenom']} ({combo['age']}). "
        f"Effectue une relecture qualité complète (cohérence, style IA, qualité rédactionnelle, ponctuation). "
        f"Retourne l'article complet corrigé :\n\n{content}"
    )

    return call_mammouth_api(
        model=MODEL_ANALYST,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=4500,
        retries=2
    )


def gemini_quality_review(content, combo):
    """Phase de relecture qualité par Gemini : cohérence, style IA, qualité rédactionnelle."""
    print(f"  [Qualité] Relecture globale par Gemini...")

    try:
        reviewed = call_gemini_quality_review(content, combo)

        if reviewed:
            # Vérifier que Gemini n'a pas cassé la structure
            h2_before = len(re.findall(r'^## ', content, re.MULTILINE))
            h2_after = len(re.findall(r'^## ', reviewed, re.MULTILINE))
            len_before = len(content.split())
            len_after = len(reviewed.split())

            # Accepter si la structure est préservée et la longueur raisonnable (pas moins de 70% de l'original)
            if h2_after >= h2_before and len_after >= len_before * 0.7:
                print(f"  [Qualité] Relecture terminée ({len_before} → {len_after} mots, structure préservée)")
                return reviewed
            else:
                print(f"  [Qualité] Gemini a altéré la structure (H2: {h2_before}→{h2_after}, mots: {len_before}→{len_after}), conservation de l'original")
                return content
        else:
            print(f"  [Qualité] Gemini n'a pas répondu, conservation de l'original")
            return content

    except Exception as e:
        print(f"  [Qualité] Erreur Gemini: {e}, conservation de l'original")
        return content


def verify_article_structure(content, combo):
    """Vérifie que l'article contient la structure H2/H3 obligatoire et le séparateur ---.
    Si la structure est insuffisante, demande à Gemini de restructurer l'article."""
    h2_count = len(re.findall(r'^## ', content, re.MULTILINE))
    h3_count = len(re.findall(r'^### ', content, re.MULTILINE))
    has_separator = '\n---\n' in content or content.strip().endswith('---')

    print(f"  [Structure] H2: {h2_count}, H3: {h3_count}, Séparateur: {'oui' if has_separator else 'non'}")

    if h2_count >= 4 and h3_count >= 3 and has_separator:
        print(f"  [Structure] OK : structure conforme")
        return content

    print(f"  [Structure] INSUFFISANT : restructuration par Gemini...")

    system_prompt = (
        "Tu es un rédacteur SEO expert. On te donne un article de blog en Markdown qui manque de structure. "
        "Tu dois le restructurer en ajoutant des titres H2 (##) et H3 (###) sans modifier le contenu textuel. "
        "RÈGLES STRICTES :\n"
        "- L'article DOIT contenir au minimum 4 titres ## (H2) et 3 titres ### (H3)\n"
        "- Les H2 doivent suivre ce schéma : définition du concept, manifestations, techniques, évolution du personnage\n"
        "- Les techniques doivent être en H3 numérotés : ### 1. **Titre**\n"
        "- Ajoute un séparateur --- avant la conclusion finale\n"
        "- NE MODIFIE PAS le texte existant, ajoute UNIQUEMENT les titres et le séparateur\n"
        "- Conserve tout le formatage Markdown existant (listes, gras, italique)\n"
        "- Retourne l'article complet restructuré, rien d'autre."
    )

    user_prompt = (
        f"Voici un article sur \"{combo['sujet']}\" dans le contexte \"{combo['contexte']}\" "
        f"avec le personnage {combo['prenom']}. "
        f"Il lui manque des titres H2/H3. Restructure-le en suivant les règles. "
        f"Retourne l'article complet restructuré :\n\n{content}"
    )

    try:
        fixed = call_mammouth_api(
            model=MODEL_ANALYST,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=4500,
            retries=2
        )

        if fixed:
            new_h2 = len(re.findall(r'^## ', fixed, re.MULTILINE))
            new_h3 = len(re.findall(r'^### ', fixed, re.MULTILINE))
            if new_h2 >= 4 and new_h3 >= 3:
                print(f"  [Structure] Gemini a restructuré : H2={new_h2}, H3={new_h3}")
                return fixed
            else:
                print(f"  [Structure] Restructuration insuffisante (H2={new_h2}, H3={new_h3}), conservation de l'original")
                return content
        else:
            print(f"  [Structure] Gemini n'a pas répondu, conservation de l'original")
            return content

    except Exception as e:
        print(f"  [Structure] Erreur Gemini: {e}, conservation de l'original")
        return content


# ============================================
# PARSING ET CRÉATION HUGO
# ============================================

def parse_article_response(raw_response):
    """Parse la réponse de l'API pour extraire les métadonnées et le contenu.
    Gère plusieurs formats possibles de sortie GPT."""
    cleaned = raw_response.strip()

    # Enlever les backticks markdown si GPT a wrappé la réponse
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:markdown)?\s*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)

    lines = cleaned.split("\n")

    metadata = {}
    content_start = 0

    # Patterns flexibles pour les métadonnées (gère espaces, **, #, etc.)
    title_pattern = re.compile(r'^\s*\**\s*TITRE[_ ]SEO\s*\**\s*:\s*(.+)', re.IGNORECASE)
    desc_pattern = re.compile(r'^\s*\**\s*META[_ ]DESCRIPTION\s*\**\s*:\s*(.+)', re.IGNORECASE)
    slug_pattern = re.compile(r'^\s*\**\s*SLUG\s*\**\s*:\s*(.+)', re.IGNORECASE)
    tags_pattern = re.compile(r'^\s*\**\s*TAGS\s*\**\s*:\s*(.+)', re.IGNORECASE)

    for i, line in enumerate(lines):
        m = title_pattern.match(line)
        if m:
            metadata["title"] = m.group(1).strip().strip('"').strip("'")
            continue
        m = desc_pattern.match(line)
        if m:
            metadata["description"] = m.group(1).strip().strip('"').strip("'")
            continue
        m = slug_pattern.match(line)
        if m:
            metadata["slug"] = m.group(1).strip().strip('"').strip("'")
            continue
        m = tags_pattern.match(line)
        if m:
            raw_tags = m.group(1).strip()
            metadata["tags"] = [t.strip().strip('"').strip("'") for t in raw_tags.split(",")]
            continue
        if line.strip() == "---":
            content_start = i + 1
            break

    content = "\n".join(lines[content_start:]).strip()

    # Si aucun séparateur --- trouvé mais des métadonnées extraites,
    # chercher le début du contenu après la dernière métadonnée
    if content_start == 0 and metadata:
        # Trouver la dernière ligne de métadonnée et prendre tout ce qui suit
        last_meta_line = 0
        for i, line in enumerate(lines):
            if any(p.match(line) for p in [title_pattern, desc_pattern, slug_pattern, tags_pattern]):
                last_meta_line = i
        content = "\n".join(lines[last_meta_line + 1:]).strip()

    return metadata, content


def validate_article(metadata, content, combo):
    """Valide que l'article parsé est complet et conforme.
    Retourne (True, message) si valide, (False, message) si invalide."""
    issues = []

    # Vérifier les métadonnées essentielles
    if not metadata.get("title") or metadata["title"] == "Article du jour":
        issues.append("titre manquant")
    if not metadata.get("slug"):
        issues.append("slug manquant")
    if not metadata.get("description"):
        issues.append("description manquante")
    if not metadata.get("tags") or len(metadata.get("tags", [])) == 0:
        issues.append("tags manquants")

    # Vérifier la longueur du contenu (minimum 1000 mots)
    word_count = len(content.split())
    if word_count < 1000:
        issues.append(f"contenu trop court ({word_count} mots, minimum 1000)")

    # Vérifier la structure H2
    h2_count = len(re.findall(r'^## ', content, re.MULTILINE))
    if h2_count < 3:
        issues.append(f"structure insuffisante ({h2_count} H2, minimum 3)")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


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

        # Rédaction avec retry si l'article est invalide
        metadata = None
        content = None
        max_redaction_attempts = 3

        for redaction_attempt in range(max_redaction_attempts):
            attempt_label = f" (tentative {redaction_attempt + 1}/{max_redaction_attempts})" if redaction_attempt > 0 else ""
            print(f"  Appel GPT-5-mini pour redaction{attempt_label}...")
            article_prompt = build_article_prompt(combo)
            raw_response = call_mammouth_api(
                model=MODEL_WRITER,
                system_prompt=system_prompt,
                user_prompt=article_prompt,
                temperature=0.85,
                max_tokens=4500
            )

            if not raw_response:
                print(f"  [Erreur] GPT n'a pas répondu, retry...")
                continue

            print(f"  Parsing de la reponse...")
            metadata, content = parse_article_response(raw_response)

            is_valid, validation_msg = validate_article(metadata, content, combo)
            if is_valid:
                print(f"  [Validation] Article valide : {len(content.split())} mots, {len(metadata.get('tags', []))} tags")
                break
            else:
                print(f"  [Validation] Article INVALIDE : {validation_msg}")
                if redaction_attempt < max_redaction_attempts - 1:
                    print(f"  Nouvelle tentative de rédaction...")
                else:
                    print(f"  [Erreur] Échec après {max_redaction_attempts} tentatives, article ignoré pour cette catégorie")
                    metadata = None
                    content = None

        if not metadata or not content:
            print(f"  [SKIP] Catégorie {cat_key} ignorée (impossible de générer un article valide)")
            continue

        # Étape 3.5 : Vérification et correction des tirets cadratins (—)
        content = verify_and_fix_emdashes(content, combo)

        # Étape 3.6 : Vérification de la structure H2/H3 obligatoire
        content = verify_article_structure(content, combo)

        # Étape 3.7 : Relecture qualité globale par Gemini (cohérence, style IA, rédaction)
        content = gemini_quality_review(content, combo)

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
