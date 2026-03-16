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
PERSONNAGES_FILE = SCRIPT_DIR / "personnages.json"

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


def add_to_matrix(matrix, combo, metadata, resume_narratif="", evolution="", elements_cles=""):
    """Ajoute une combinaison à la matrice après génération de l'article.
    Inclut les données narratives pour la continuité des personnages."""
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
        "resume_narratif": resume_narratif,
        "evolution": evolution,
        "elements_cles": elements_cles,
        "strategie_coherence": combo.get("strategie_coherence", ""),
        "apport_psychologique": combo.get("apport_psychologique", ""),
    }
    matrix["articles"].append(entry)


def extract_narrative_summary(content, combo):
    """Appelle Gemini pour extraire le résumé narratif, l'évolution et les éléments clés d'un article.
    Ces données sont stockées dans la matrice pour assurer la continuité narrative."""
    system_prompt = (
        "Tu es un analyste narratif. On te donne un article de blog qui raconte l'histoire d'un personnage "
        "confronté à un concept psychologique. Tu dois extraire 3 informations pour permettre la continuité "
        "narrative dans les prochains articles de ce personnage.\n\n"
        "Réponds UNIQUEMENT en JSON valide, sans backticks, sans texte autour."
    )

    user_prompt = (
        f"Personnage : {combo['prenom']}\n"
        f"Sujet : {combo['sujet']}\n"
        f"Contexte : {combo['contexte']}\n\n"
        f"Article :\n{content[:3000]}\n\n"
        "Extrais ces 3 informations en JSON :\n"
        "{\n"
        '  "resume_narratif": "2-3 phrases résumant ce qui est arrivé au personnage dans cet article (les événements concrets, les personnes impliquées)",\n'
        '  "evolution": "1-2 phrases décrivant ce que le personnage a compris/appris/changé grâce à cette expérience",\n'
        '  "elements_cles": "les détails importants à retenir pour la continuité (lieux, personnes mentionnées, décisions prises, techniques apprises)"\n'
        "}"
    )

    raw = call_mammouth_api(
        model=MODEL_ANALYST,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=800,
        retries=2
    )

    if not raw:
        return "", "", ""

    cleaned = raw.strip()
    if "```" in cleaned:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            cleaned = match.group(0)

    cleaned = fix_json_trailing_commas(cleaned)

    try:
        data = json.loads(cleaned)
        return (
            data.get("resume_narratif", ""),
            data.get("evolution", ""),
            data.get("elements_cles", ""),
        )
    except json.JSONDecodeError:
        print(f"  [Narratif] Erreur parsing JSON résumé narratif")
        return "", "", ""


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


def _to_str(value):
    """Convertit une valeur en string de manière sûre.
    Gère le cas où Gemini retourne un dict/list au lieu d'une string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value) if value else ""


def analyze_character_depth(perso, perso_articles):
    """Analyse en profondeur l'état psychologique d'un personnage.

    Retourne un dictionnaire structuré contenant :
    - themes_explored : catégories/sujets déjà couverts
    - techniques_learned : techniques acquises
    - emotional_trajectory : trajectoire émotionnelle
    - contradictions : contradictions potentielles détectées
    - unexplored_areas : zones de la psychologie du personnage pas encore creusées
    - depth_score : score de profondeur (0-100)
    - recommended_directions : directions recommandées pour approfondir
    """
    prenom = perso["prenom"]
    traits = perso.get("traits_personnalite", [])
    tendances = perso.get("tendances_psychologiques", [])
    histoire = perso.get("histoire_de_fond", "")
    affinites = perso.get("affinites_thematiques", {})

    analysis = {
        "themes_explored": {"cat1_pensees": [], "cat2_emotions": [], "cat3_schemas": []},
        "techniques_learned": [],
        "emotional_trajectory": [],
        "contradictions": [],
        "unexplored_affinities": {"cat1_pensees": [], "cat2_emotions": [], "cat3_schemas": []},
        "depth_score": 0,
        "relationship_evolution": [],
        "recommended_directions": [],
    }

    if not perso_articles:
        # Personnage vierge : tout est à explorer
        for cat_key, sujets in affinites.items():
            analysis["unexplored_affinities"][cat_key] = list(sujets)
        analysis["recommended_directions"] = [
            f"Premier article : explorer un trait central ({traits[0] if traits else 'à définir'})",
            f"Thème de fond à introduire : {tendances[0] if tendances else 'à définir'}",
        ]
        return analysis

    # 1. Thèmes explorés par catégorie
    for a in perso_articles:
        cat = a.get("category_key", "")
        if cat in analysis["themes_explored"]:
            analysis["themes_explored"][cat].append(a.get("sujet", ""))

    # 2. Techniques apprises (extraites des éléments clés)
    for a in perso_articles:
        elements = _to_str(a.get("elements_cles", ""))
        if elements and "technique" in elements.lower():
            analysis["techniques_learned"].append({
                "date": a.get("date", ""),
                "sujet": a.get("sujet", ""),
                "elements": elements,
            })

    # 3. Trajectoire émotionnelle
    for a in perso_articles:
        evolution = _to_str(a.get("evolution", ""))
        if evolution:
            analysis["emotional_trajectory"].append({
                "date": a.get("date", ""),
                "sujet": a.get("sujet", ""),
                "evolution": evolution,
            })

    # 4. Détection de contradictions potentielles
    evolutions_text = [_to_str(a.get("evolution", "")).lower() for a in perso_articles if a.get("evolution")]
    # Chercher des patterns contradictoires (a appris X puis échoue sur X)
    learned_concepts = set()
    for evo in evolutions_text:
        if "apprend" in evo or "comprend" in evo or "réalise" in evo or "découvre" in evo:
            learned_concepts.add(evo)

    # 5. Affinités non encore explorées
    explored_sujets = set()
    for cat_articles in analysis["themes_explored"].values():
        explored_sujets.update(s.lower() for s in cat_articles)

    for cat_key, sujets_affinites in affinites.items():
        for aff in sujets_affinites:
            if aff.lower() not in explored_sujets:
                # Vérifier aussi les correspondances partielles
                is_explored = any(aff.lower() in s or s in aff.lower() for s in explored_sujets)
                if not is_explored:
                    analysis["unexplored_affinities"][cat_key].append(aff)

    # 6. Évolution des relations
    relations = perso.get("relations", {})
    for a in perso_articles:
        elements = _to_str(a.get("elements_cles", ""))
        resume = _to_str(a.get("resume_narratif", ""))
        for role, nom in relations.items():
            if nom.lower() in (elements + " " + resume).lower():
                analysis["relationship_evolution"].append({
                    "date": a.get("date", ""),
                    "relation": f"{role}: {nom}",
                    "contexte": a.get("contexte", ""),
                })

    # 7. Score de profondeur (0-100)
    nb_articles = len(perso_articles)
    nb_categories = len([c for c in analysis["themes_explored"].values() if c])
    nb_techniques = len(analysis["techniques_learned"])
    nb_with_evolution = len(analysis["emotional_trajectory"])
    nb_with_resume = sum(1 for a in perso_articles if a.get("resume_narratif"))
    nb_relations_explored = len(set(r["relation"] for r in analysis["relationship_evolution"]))

    score = 0
    score += min(30, nb_articles * 5)  # Max 30 pts pour la quantité
    score += nb_categories * 10  # Max 30 pts pour la diversité catégorielle
    score += min(15, nb_with_evolution * 3)  # Max 15 pts pour la trajectoire documentée
    score += min(10, nb_techniques * 5)  # Max 10 pts pour les techniques acquises
    score += min(15, nb_relations_explored * 5)  # Max 15 pts pour l'exploration relationnelle
    analysis["depth_score"] = min(100, score)

    # 8. Directions recommandées
    all_themes = []
    for cat_articles in analysis["themes_explored"].values():
        all_themes.extend(cat_articles)

    # Catégorie la moins explorée
    cat_counts = {cat: len(arts) for cat, arts in analysis["themes_explored"].items()}
    least_explored_cat = min(cat_counts, key=cat_counts.get)
    cat_names = {
        "cat1_pensees": "pensées/biais cognitifs",
        "cat2_emotions": "émotions",
        "cat3_schemas": "schémas répétitifs"
    }

    if cat_counts[least_explored_cat] == 0:
        analysis["recommended_directions"].append(
            f"PRIORITÉ : Explorer la catégorie '{cat_names[least_explored_cat]}' (aucun article encore)"
        )

    # Affinités naturelles non explorées
    for cat_key, unexplored in analysis["unexplored_affinities"].items():
        if unexplored:
            analysis["recommended_directions"].append(
                f"Affinité naturelle inexploitée en {cat_names.get(cat_key, cat_key)} : {', '.join(unexplored[:3])}"
            )

    # Relations sous-exploitées
    all_relations = set(perso.get("relations", {}).keys())
    explored_relations = set(r["relation"].split(":")[0].strip() for r in analysis["relationship_evolution"])
    unexplored_relations = all_relations - explored_relations
    if unexplored_relations:
        analysis["recommended_directions"].append(
            f"Relations jamais explorées : {', '.join(unexplored_relations)}"
        )

    # Traits de personnalité sous-exploités
    explored_trait_themes = " ".join(all_themes).lower()
    unexplored_traits = [t for t in traits if t.lower() not in explored_trait_themes]
    if unexplored_traits:
        analysis["recommended_directions"].append(
            f"Traits de personnalité pas encore mis en situation : {', '.join(unexplored_traits[:3])}"
        )

    return analysis


def format_character_analysis_for_prompt(perso, analysis, perso_articles):
    """Formate l'analyse de profondeur d'un personnage pour l'inclure dans le prompt Gemini."""
    prenom = perso["prenom"]
    genre = perso.get("genre", "M")
    pronom = "Elle" if genre == "F" else "Il"

    lines = []
    lines.append(f"--- {prenom} ({perso['age']} ans, {perso['profession']}) [Profondeur : {analysis['depth_score']}/100] ---")
    lines.append(f"  Situation : {perso['situation_familiale']}")
    lines.append(f"  Traits : {', '.join(perso['traits_personnalite'])}")
    lines.append(f"  Tendances psy : {', '.join(perso['tendances_psychologiques'])}")
    lines.append(f"  Histoire : {perso['histoire_de_fond']}")

    if perso.get("relations"):
        rels = ", ".join(f"{r}: {n}" for r, n in perso["relations"].items())
        lines.append(f"  Relations : {rels}")

    if perso_articles:
        lines.append(f"  PARCOURS ({len(perso_articles)} articles) :")
        for i, a in enumerate(perso_articles, 1):
            date = a.get("date", "?")
            cat = a.get("category_key", "?")
            resume = _to_str(a.get("resume_narratif", ""))
            evolution = _to_str(a.get("evolution", ""))
            line = f"    {i}. [{date}] ({cat}) \"{a['sujet']}\" en contexte \"{a['contexte']}\""
            if a.get("title"):
                line += f" — \"{a['title']}\""
            lines.append(line)
            if resume:
                lines.append(f"       Résumé : {resume}")
            if evolution:
                lines.append(f"       Évolution : {evolution}")

        # Techniques acquises
        if analysis["techniques_learned"]:
            tech_list = [t["elements"] for t in analysis["techniques_learned"]]
            lines.append(f"  ACQUIS TECHNIQUES : {'; '.join(tech_list[:5])}")

        # Trajectoire émotionnelle résumée
        if analysis["emotional_trajectory"]:
            traj = " → ".join(t["evolution"][:80] for t in analysis["emotional_trajectory"][-4:])
            lines.append(f"  TRAJECTOIRE : {traj}")

        # Relations explorées vs non explorées
        if analysis["relationship_evolution"]:
            explored_rels = set(r["relation"] for r in analysis["relationship_evolution"])
            lines.append(f"  RELATIONS EXPLORÉES : {', '.join(explored_rels)}")

        # Contradictions détectées
        if analysis["contradictions"]:
            lines.append(f"  ⚠ CONTRADICTIONS DÉTECTÉES : {'; '.join(analysis['contradictions'][:3])}")

    else:
        lines.append(f"  PARCOURS : Aucun article encore. {pronom} attend sa première histoire.")

    # Directions recommandées (le cœur du système de cohérence active)
    if analysis["recommended_directions"]:
        lines.append(f"  DIRECTIONS POUR APPROFONDIR {prenom.upper()} :")
        for d in analysis["recommended_directions"][:4]:
            lines.append(f"    → {d}")

    # Affinités non explorées
    all_unexplored = []
    for cat_key, unexplored in analysis["unexplored_affinities"].items():
        all_unexplored.extend(unexplored[:2])
    if all_unexplored:
        lines.append(f"  AFFINITÉS NATURELLES INEXPLOITÉES : {', '.join(all_unexplored[:6])}")

    lines.append("")
    return "\n".join(lines)


def build_character_arcs_summary(personnages, matrix):
    """Construit un résumé enrichi de l'arc narratif de chaque personnage pour Gemini.

    Inclut une analyse de profondeur par personnage : thèmes explorés, lacunes,
    contradictions, directions recommandées pour renforcer la cohérence.
    """
    articles = matrix.get("articles", [])

    # Regrouper les articles par personnage, triés par date
    perso_articles = {}
    for a in articles:
        prenom = a.get("prenom", "")
        if prenom:
            perso_articles.setdefault(prenom, []).append(a)

    for prenom in perso_articles:
        perso_articles[prenom].sort(key=lambda x: x.get("date", ""))

    lines = ["ANALYSE DE L'ÉTAT NARRATIF DE CHAQUE PERSONNAGE :\n"]
    lines.append("Chaque personnage est un individu récurrent dont la psychologie doit devenir de plus en plus")
    lines.append("lisible, cohérente et convaincante au fil des articles. L'analyse ci-dessous identifie")
    lines.append("les lacunes et les directions d'approfondissement pour chaque personnage.\n")

    # Trier les personnages : les moins profonds en premier (priorité)
    perso_with_analysis = []
    for perso in personnages:
        prenom = perso["prenom"]
        articles_list = perso_articles.get(prenom, [])
        analysis = analyze_character_depth(perso, articles_list)
        perso_with_analysis.append((perso, analysis, articles_list))

    # Trier par score de profondeur croissant (les moins creusés en premier)
    perso_with_analysis.sort(key=lambda x: x[1]["depth_score"])

    for perso, analysis, articles_list in perso_with_analysis:
        lines.append(format_character_analysis_for_prompt(perso, analysis, articles_list))

    # Résumé global
    total_articles = len(articles)
    avg_depth = sum(a[1]["depth_score"] for a in perso_with_analysis) / len(perso_with_analysis) if perso_with_analysis else 0
    low_depth = [a[0]["prenom"] for a in perso_with_analysis if a[1]["depth_score"] < 30]
    lines.append(f"\nRÉSUMÉ GLOBAL : {total_articles} articles, profondeur moyenne {avg_depth:.0f}/100")
    if low_depth:
        lines.append(f"PERSONNAGES À PRIORISER (profondeur < 30) : {', '.join(low_depth)}")

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
# SYSTÈME DE PERSONNAGES RÉCURRENTS
# ============================================

def load_personnages():
    """Charge le registre des personnages depuis personnages.json."""
    if PERSONNAGES_FILE.exists():
        try:
            with open(PERSONNAGES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("personnages", [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Attention: personnages.json corrompu ({e})")
    return []


def save_personnages(personnages):
    """Sauvegarde le registre des personnages."""
    with open(PERSONNAGES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["personnages"] = personnages
    with open(PERSONNAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def select_best_personnage(personnages, category_key, sujet, contexte, matrix):
    """Sélectionne le personnage le plus pertinent pour un article donné.

    Algorithme :
    1. Calcul d'un score d'affinité thématique (le sujet apparaît-il dans les affinités du personnage ?)
    2. Bonus contextuel (le contexte de vie correspond-il à la situation du personnage ?)
    3. Malus de sur-utilisation (éviter qu'un personnage monopolise les articles)
    4. Bonus de diversité (favoriser les personnages peu utilisés)
    """
    articles = matrix.get("articles", [])

    # Compter les apparitions de chaque personnage
    apparitions = {}
    for a in articles:
        p = a.get("prenom", "")
        if p:
            apparitions[p] = apparitions.get(p, 0) + 1

    scores = []
    for perso in personnages:
        score = 0
        prenom = perso["prenom"]

        # 1. Score d'affinité thématique (0-30 points)
        affinites = perso.get("affinites_thematiques", {}).get(category_key, [])
        sujet_lower = sujet.lower()
        for affinite in affinites:
            affinite_lower = affinite.lower()
            # Correspondance exacte
            if sujet_lower == affinite_lower:
                score += 30
                break
            # Correspondance partielle (le sujet contient l'affinité ou vice versa)
            if sujet_lower in affinite_lower or affinite_lower in sujet_lower:
                score += 20
                break
            # Mots-clés communs
            mots_sujet = set(sujet_lower.split())
            mots_affinite = set(affinite_lower.split())
            communs = mots_sujet & mots_affinite - {"de", "du", "la", "le", "les", "des", "en", "et", "à", "l'", "d'", "un", "une"}
            if communs:
                score += min(15, len(communs) * 5)

        # 2. Bonus contextuel (0-10 points)
        contexte_lower = contexte.lower()
        situation = perso.get("situation_familiale", "").lower()
        profession = perso.get("profession", "").lower()
        traits = " ".join(perso.get("traits_personnalite", [])).lower()

        if "travail" in contexte_lower and ("entreprise" in profession or "manager" in profession or "cadre" in profession or "travail" in profession):
            score += 10
        elif "couple" in contexte_lower and ("marié" in situation or "couple" in situation or "compagne" in situation or "compagnon" in situation):
            score += 10
        elif "famille" in contexte_lower and ("enfant" in situation or "mère" in situation or "père" in situation or "parent" in situation):
            score += 10
        elif "parent" in contexte_lower and ("enfant" in situation or "mère" in situation or "père" in situation):
            score += 10
        elif "enfant" in contexte_lower and ("enfant" in situation or "mère" in situation or "père" in situation):
            score += 10
        elif "école" in contexte_lower and ("étudiant" in profession or "enseignant" in profession):
            score += 10
        elif "entretien" in contexte_lower and ("étudiant" in profession or "reconversion" in profession or "commercial" in profession):
            score += 8
        elif "manager" in contexte_lower and ("manager" in profession or "directrice" in profession or "cadre" in profession or "chef" in profession):
            score += 10
        elif "argent" in contexte_lower or "finances" in contexte_lower:
            score += 5
        elif "réseaux sociaux" in contexte_lower and (perso["age"] <= 30):
            score += 8
        elif "solitude" in contexte_lower and ("seul" in situation or "célibataire" in situation or "veuf" in situation or "veuve" in situation):
            score += 10
        elif "rupture" in contexte_lower and ("rupture" in perso.get("histoire_de_fond", "").lower() or "sépar" in situation or "divorc" in situation):
            score += 10
        elif "deuil" in contexte_lower and ("décédé" in str(perso.get("relations", {})).lower() or "veuf" in situation or "veuve" in situation):
            score += 10
        elif "reconversion" in contexte_lower and "reconversion" in profession:
            score += 10
        elif "vieillissement" in contexte_lower and perso["age"] >= 50:
            score += 10
        elif "compétition" in contexte_lower and ("compétitif" in traits or "ambitieux" in traits):
            score += 8
        elif "conflit" in contexte_lower:
            score += 3
        elif "intimité" in contexte_lower and ("couple" in situation or "compagne" in situation):
            score += 8

        # 3. Malus de sur-utilisation (-5 points par apparition au-delà de 3)
        nb_apparitions = apparitions.get(prenom, 0)
        if nb_apparitions > 3:
            score -= (nb_apparitions - 3) * 5

        # 4. Bonus de diversité (+5 si jamais utilisé, +3 si 1 seule fois)
        if nb_apparitions == 0:
            score += 5
        elif nb_apparitions == 1:
            score += 3

        scores.append((perso, score))

    # Trier par score décroissant
    scores.sort(key=lambda x: x[1], reverse=True)

    # Retourner le meilleur personnage
    best = scores[0][0]
    best_score = scores[0][1]
    print(f"  [Personnage] Sélection : {best['prenom']} (score: {best_score}, "
          f"affinité: {category_key}, {apparitions.get(best['prenom'], 0)} articles précédents)")
    return best


def update_personnage_history(personnages, prenom, article_info):
    """Met à jour l'historique d'un personnage après la génération d'un article."""
    for perso in personnages:
        if perso["prenom"] == prenom:
            perso["historique_articles"].append({
                "date": article_info.get("date", ""),
                "sujet": article_info.get("sujet", ""),
                "contexte": article_info.get("contexte", ""),
                "category": article_info.get("category_key", ""),
                "titre": article_info.get("title", ""),
                "slug": article_info.get("slug", ""),
            })
            break


def build_personnage_context(perso, matrix=None):
    """Construit le contexte narratif complet d'un personnage pour le prompt de rédaction.
    Inclut l'arc narratif chronologique avec résumés pour assurer la continuité."""
    historique = perso.get("historique_articles", [])
    relations = perso.get("relations", {})
    genre = perso.get("genre", "M")
    pronom = "elle" if genre == "F" else "il"
    pronom_maj = "Elle" if genre == "F" else "Il"

    context = f"""FICHE DU PERSONNAGE (à respecter ABSOLUMENT) :
- Prénom : {perso['prenom']}
- Genre : {"féminin" if genre == "F" else "masculin"} (utilise les accords grammaticaux correspondants)
- Âge : {perso['age']} ans (NE CHANGE JAMAIS)
- Profession : {perso['profession']}
- Situation : {perso['situation_familiale']}
- Traits de personnalité : {', '.join(perso['traits_personnalite'])}
- Tendances psychologiques : {', '.join(perso['tendances_psychologiques'])}
- Histoire de fond : {perso['histoire_de_fond']}
- Apparence : {perso['details_physiques']}
- Habitudes : {perso['habitudes']}"""

    if relations:
        relations_str = ", ".join(f"{role}: {nom}" for role, nom in relations.items())
        context += f"\n- Relations : {relations_str}"

    # Arc narratif enrichi avec résumés depuis la matrice
    if matrix:
        articles = matrix.get("articles", [])
        perso_articles = sorted(
            [a for a in articles if a.get("prenom") == perso["prenom"]],
            key=lambda x: x.get("date", "")
        )
        if perso_articles:
            context += f"\n\nARC NARRATIF DE {perso['prenom'].upper()} ({len(perso_articles)} chapitres précédents) :"
            context += f"\n{perso['prenom']} est un personnage récurrent dont les lecteurs suivent l'évolution. Tu DOIS faire référence à son passé :"
            for i, a in enumerate(perso_articles, 1):
                date = a.get("date", "")
                context += f"\n  Chapitre {i} ({date}) : \"{a.get('title', a.get('sujet', ''))}\" — {pronom_maj} explorait \"{a['sujet']}\" dans le contexte \"{a['contexte']}\"."
                if a.get("resume_narratif"):
                    context += f"\n    → {_to_str(a['resume_narratif'])}"
                if a.get("evolution"):
                    context += f"\n    → Évolution : {_to_str(a['evolution'])}"

            context += f"\n\nCONTINUITÉ OBLIGATOIRE :"
            context += f"\n- Fais référence à AU MOINS UN événement passé de {perso['prenom']} (mentionné ci-dessus)"
            context += f"\n- Montre que {pronom} a ÉVOLUÉ depuis ses expériences précédentes"
            context += f"\n- Les techniques apprises dans les articles précédents sont des acquis que {pronom} peut réutiliser"
            context += f"\n- L'histoire d'aujourd'hui est le PROCHAIN CHAPITRE de sa vie, pas une histoire isolée"
    elif historique:
        context += f"\n\nHISTORIQUE DU PERSONNAGE ({len(historique)} articles précédents) :"
        context += f"\n{perso['prenom']} est déjà apparu(e) dans ces articles. Fais des références à son parcours :"
        for h in historique[-5:]:
            context += f"\n  - \"{h.get('titre', h.get('sujet', ''))}\" ({h.get('contexte', '')})"

    return context


# ============================================
# GEMINI : SUGGESTION DE SUJETS
# ============================================

def call_mammouth_api(model, system_prompt, user_prompt, temperature=0.85, max_tokens=4500, retries=3):
    """Appelle l'API Mammouth avec le modèle spécifié."""
    import time

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
            response = requests.post(API_URL, headers=headers, json=data, timeout=240)
            response.raise_for_status()
            result = response.json()

            # Vérifier que la réponse contient bien du contenu
            choices = result.get("choices", [])
            if not choices:
                print(f"    [API] Reponse vide (pas de choices), retry...")
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
                continue

            content = choices[0].get("message", {}).get("content", "")
            if not content or not content.strip():
                print(f"    [API] Contenu vide dans la reponse, retry...")
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
                continue

            # Vérifier le finish_reason
            finish_reason = choices[0].get("finish_reason", "")
            if finish_reason == "length":
                print(f"    [API] Attention: reponse tronquee (max_tokens atteint)")

            return content

        except requests.exceptions.Timeout:
            print(f"    [API] Timeout apres 240s (tentative {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            elif attempt == retries - 1:
                return None
        except requests.exceptions.HTTPError:
            status = response.status_code
            print(f"    [API] Erreur HTTP {status}: {response.text[:300]}")
            if status == 429:
                # Rate limit : attendre plus longtemps
                wait = 2 ** (attempt + 2)
                print(f"    [API] Rate limit, attente {wait}s...")
                time.sleep(wait)
            elif status >= 500:
                # Erreur serveur : retry avec backoff
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
            else:
                # Erreur client (400, 401) : ne pas retry
                return None
        except requests.exceptions.ConnectionError as e:
            print(f"    [API] Erreur connexion: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
        except Exception as e:
            print(f"    [API] Tentative {attempt + 1}/{retries} echouee: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))

    print(f"    [API] Echec apres {retries} tentatives pour {model}")
    return None


def build_gemini_prompt(matrix_summary):
    """Construit le prompt pour Gemini avec des indices numériques pour fiabiliser le JSON.
    Version legacy (fallback si character-first échoue)."""

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


def build_gemini_character_first_prompt(character_arcs_summary, matrix_summary, personnages, today_str):
    """Construit le prompt Gemini 'character-first' orienté cohérence active.

    La question centrale n'est plus "quelle est la suite logique ?" mais
    "qu'est-ce qui viendrait rationaliser, approfondir ou enrichir ce qui a
    déjà été établi sur ces personnages ?"
    """

    cat1_sujets = CATEGORIES["cat1_pensees"]["sujets"]
    cat2_sujets = CATEGORIES["cat2_emotions"]["sujets"]
    cat3_sujets = CATEGORIES["cat3_schemas"]["sujets"]

    cat1_list = "\n".join(f"  {i}: \"{s}\"" for i, s in enumerate(cat1_sujets))
    cat2_list = "\n".join(f"  {i}: \"{s}\"" for i, s in enumerate(cat2_sujets))
    cat3_list = "\n".join(f"  {i}: \"{s}\"" for i, s in enumerate(cat3_sujets))
    contextes_list = "\n".join(f"  {i}: \"{c}\"" for i, c in enumerate(CONTEXTES))
    angles_list = "\n".join(f"  {i}: \"{a}\"" for i, a in enumerate(ANGLES))

    # Construire la liste des personnages avec index
    personnages_list = "\n".join(
        f"  {i}: \"{p['prenom']}\" ({p['age']} ans, {p['profession']})"
        for i, p in enumerate(personnages)
    )

    system_prompt = f"""Tu es un directeur éditorial, scénariste et psychologue expert en narration psychologique.
Tu gères un blog où 20 personnages récurrents vivent des histoires qui illustrent des concepts psychologiques.
Chaque personnage a une vie qui ÉVOLUE au fil des articles. Les lecteurs suivent leurs parcours comme une série.

Nous sommes le {today_str}. Les personnages vivent en France, dans le présent.

PRINCIPE FONDAMENTAL DE COHÉRENCE ACTIVE :
Plus il y a d'articles sur un personnage → plus sa psychologie doit être creusée →
plus son portrait devient lisible, cohérent et convaincant pour le lecteur.

Chaque nouvel article ne doit PAS simplement "se conformer" au passé du personnage.
Il doit ACTIVEMENT enrichir, rationaliser ou approfondir ce qui a déjà été établi.

TA MISSION : Pour chaque suggestion, réponds à cette question centrale :
"Qu'est-ce qui viendrait RATIONALISER, APPROFONDIR ou ENRICHIR ce qui a déjà été établi sur ce personnage ?"

STRATÉGIE DE SÉLECTION (par ordre de priorité) :
1. RATIONALISER : Un personnage a montré un comportement dans un article passé. Le nouveau sujet
   explique POURQUOI il réagit ainsi (ex: Sophie est perfectionniste → explorer le schéma d'exigences élevées
   de Young éclaire la racine de ce trait).
2. APPROFONDIR : Un personnage a effleuré un thème. Le nouveau sujet creuse plus profondément
   la même zone (ex: après l'irritabilité, explorer la colère refoulée qui se cache derrière).
3. ENRICHIR : Un personnage n'a été vu que dans un contexte (travail). Le montrer dans un autre contexte
   (famille, couple) révèle une nouvelle facette de sa personnalité, cohérente avec ce qu'on sait déjà.
4. CONNECTER : Le sujet crée un lien entre deux aspects déjà explorés du personnage,
   montrant que sa psychologie forme un tout cohérent.

RÈGLES NARRATIVES :
1. Consulte l'ANALYSE DE PROFONDEUR de chaque personnage (score, lacunes, directions recommandées)
2. Priorise les personnages avec un score de profondeur faible ou des lacunes identifiées
3. Le sujet choisi doit répondre à au moins UNE des 4 stratégies ci-dessus (rationaliser/approfondir/enrichir/connecter)
4. Le contexte doit correspondre à la vie actuelle du personnage (profession, situation, relations)
5. Évite les combinaisons sujet+contexte déjà traitées et les personnages des 3 derniers jours
6. 3 personnages DIFFÉRENTS, 3 histoires DIFFÉRENTES
7. Si un personnage a des affinités naturelles inexploitées, c'est une opportunité prioritaire

IMPORTANT : Réponds UNIQUEMENT avec du JSON valide. Pas de texte avant ni après. Pas de backticks."""

    user_prompt = f"""{character_arcs_summary}

{matrix_summary}

PERSONNAGES DISPONIBLES (utilise l'INDICE numérique) :
{personnages_list}

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

Propose 3 combinaisons personnage+sujet (1 par catégorie).
Pour chaque suggestion, explique en quoi elle RATIONALISE, APPROFONDIT, ENRICHIT ou CONNECTE
la psychologie déjà établie du personnage.

Réponds en JSON :

[
  {{
    "cat": "cat1_pensees",
    "personnage_idx": 0,
    "sujet_idx": 5,
    "contexte_idx": 2,
    "angle_idx": 0,
    "strategie_coherence": "rationaliser|approfondir|enrichir|connecter",
    "justification_narrative": "Sophie a exploré le biais rétrospectif et l'irritabilité. Explorer le biais d'ancrage RATIONALISE sa tendance perfectionniste en montrant comment elle s'accroche à ses premières impressions au travail...",
    "apport_psychologique": "Cet article révèle que le perfectionnisme de Sophie n'est pas qu'un trait isolé : il se nourrit de biais cognitifs spécifiques qui ancrent ses standards élevés...",
    "scene_envisagee": "Sophie est au bureau un lundi matin. Elle repense à la réunion de la semaine dernière où..."
  }},
  {{
    "cat": "cat2_emotions",
    "personnage_idx": 4,
    "sujet_idx": 12,
    "contexte_idx": 7,
    "angle_idx": 3,
    "strategie_coherence": "enrichir",
    "justification_narrative": "Nadia n'a été vue qu'au travail. L'explorer en famille ENRICHIT son portrait en révélant comment son besoin de contrôle se manifeste différemment avec ses enfants...",
    "apport_psychologique": "On découvre que le contrôle de Nadia au travail est une armure. En famille, la vulnérabilité affleure, créant un portrait plus nuancé et humain...",
    "scene_envisagee": "Nadia rentre chez elle après une journée de réunions. Sa fille Yasmine lui demande..."
  }},
  {{
    "cat": "cat3_schemas",
    "personnage_idx": 13,
    "sujet_idx": 8,
    "contexte_idx": 1,
    "angle_idx": 5,
    "strategie_coherence": "connecter",
    "justification_narrative": "Hugo a vécu la dette émotionnelle et l'attachement désorganisé. Le schéma d'échec CONNECTE ces deux expériences : sa peur de l'échec nourrit sa difficulté relationnelle...",
    "apport_psychologique": "Cet article tisse un fil rouge entre les expériences passées de Hugo, montrant que sa dépendance affective et son rapport à l'échec sont les deux faces d'une même blessure...",
    "scene_envisagee": "Hugo accorde sa guitare dans son petit appartement. Son téléphone vibre, c'est Chloé..."
  }}
]

Les indices doivent correspondre aux listes ci-dessus. Pas de texte autour du JSON."""

    return system_prompt, user_prompt


def parse_gemini_character_suggestions(raw_response, personnages):
    """Parse la réponse JSON de Gemini pour l'approche character-first."""
    cleaned = raw_response.strip()

    # Enlever backticks markdown si présents
    if "```" in cleaned:
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if match:
            cleaned = match.group(0)
        else:
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

    cleaned = fix_json_trailing_commas(cleaned)

    try:
        raw_suggestions = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  Erreur parsing JSON Gemini (character-first): {e}")
        print(f"  Reponse brute (500 premiers chars): {raw_response[:500]}")
        return None

    if not isinstance(raw_suggestions, list) or len(raw_suggestions) != 3:
        count = len(raw_suggestions) if isinstance(raw_suggestions, list) else "pas une liste"
        print(f"  Gemini n'a pas retourné 3 suggestions (reçu: {count})")
        return None

    resolved = []
    for s in raw_suggestions:
        cat_key = s.get("cat", "")
        if cat_key not in CATEGORIES:
            print(f"  Catégorie inconnue: {cat_key}")
            return None

        cat_sujets = CATEGORIES[cat_key]["sujets"]
        personnage_idx = s.get("personnage_idx", -1)
        sujet_idx = s.get("sujet_idx", -1)
        contexte_idx = s.get("contexte_idx", -1)
        angle_idx = s.get("angle_idx", -1)

        # Validation des indices
        if not (0 <= personnage_idx < len(personnages)):
            print(f"  Indice personnage invalide: {personnage_idx} (max {len(personnages)-1})")
            return None
        if not (0 <= sujet_idx < len(cat_sujets)):
            print(f"  Indice sujet invalide: {sujet_idx} (max {len(cat_sujets)-1}) pour {cat_key}")
            return None
        if not (0 <= contexte_idx < len(CONTEXTES)):
            print(f"  Indice contexte invalide: {contexte_idx} (max {len(CONTEXTES)-1})")
            return None
        if not (0 <= angle_idx < len(ANGLES)):
            print(f"  Indice angle invalide: {angle_idx} (max {len(ANGLES)-1})")
            return None

        perso = personnages[personnage_idx]
        resolved.append({
            "category_key": cat_key,
            "sujet": cat_sujets[sujet_idx],
            "contexte": CONTEXTES[contexte_idx],
            "angle": ANGLES[angle_idx],
            "personnage": perso,
            "justification_narrative": s.get("justification_narrative", ""),
            "scene_envisagee": s.get("scene_envisagee", ""),
            "strategie_coherence": s.get("strategie_coherence", ""),
            "apport_psychologique": s.get("apport_psychologique", ""),
        })

    return resolved


def fix_json_trailing_commas(json_str):
    """Corrige les virgules traînantes dans le JSON (fréquent avec Gemini).
    Ex: {"a": 1,} -> {"a": 1} et [{"a": 1},] -> [{"a": 1}]"""
    # Supprimer les virgules traînantes avant } ou ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    return json_str


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

    # Corriger les virgules traînantes (problème fréquent de Gemini)
    cleaned = fix_json_trailing_commas(cleaned)

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


def get_gemini_suggestions(matrix, personnages=None):
    """Appelle Gemini pour obtenir des suggestions.
    Si personnages est fourni, utilise l'approche character-first.
    Sinon, fallback sur l'approche legacy (sujet-first)."""

    matrix_summary = build_matrix_summary(matrix)

    # Approche character-first (prioritaire)
    if personnages:
        today_str = datetime.now().strftime("%d %B %Y")
        # Mois en français
        mois_fr = {
            "January": "janvier", "February": "février", "March": "mars",
            "April": "avril", "May": "mai", "June": "juin",
            "July": "juillet", "August": "août", "September": "septembre",
            "October": "octobre", "November": "novembre", "December": "décembre"
        }
        for en, fr in mois_fr.items():
            today_str = today_str.replace(en, fr)

        character_arcs = build_character_arcs_summary(personnages, matrix)
        system_prompt, user_prompt = build_gemini_character_first_prompt(
            character_arcs, matrix_summary, personnages, today_str
        )

        print("  Appel Gemini (analyse narrative character-first)...")
        raw_response = call_mammouth_api(
            model=MODEL_ANALYST,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=3000
        )

        if raw_response:
            suggestions = parse_gemini_character_suggestions(raw_response, personnages)
            if suggestions:
                # Valider les suggestions
                all_valid = True
                for s in suggestions:
                    if is_combo_used(matrix, s["category_key"], s["sujet"], s["contexte"]):
                        print(f"  Combinaison déjà utilisée: {s['sujet']} / {s['contexte']}")
                        all_valid = False
                    else:
                        print(f"  [Character-first] {s['personnage']['prenom']} → [{s['category_key']}] {s['sujet']} / {s['contexte']}")

                cats = [s["category_key"] for s in suggestions]
                if sorted(cats) != ["cat1_pensees", "cat2_emotions", "cat3_schemas"]:
                    print(f"  Les catégories ne sont pas correctes: {cats}")
                    all_valid = False

                # Vérifier 3 personnages différents
                prenoms = [s["personnage"]["prenom"] for s in suggestions]
                if len(set(prenoms)) != 3:
                    print(f"  Personnages non distincts: {prenoms}")
                    all_valid = False

                if all_valid:
                    return suggestions
                print("  Character-first a échoué la validation, fallback legacy...")
            else:
                print("  Parsing character-first échoué, fallback legacy...")
        else:
            print("  Gemini character-first n'a pas répondu, fallback legacy...")

    # Fallback : approche legacy (sujet-first)
    system_prompt, user_prompt = build_gemini_prompt(matrix_summary)

    print("  Appel Gemini (analyse legacy sujet-first)...")
    raw_response = call_mammouth_api(
        model=MODEL_ANALYST,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=2000
    )

    if not raw_response:
        print("  Gemini n'a pas répondu")
        return None

    suggestions = parse_gemini_suggestions(raw_response)
    if not suggestions:
        return None

    all_valid = True
    for s in suggestions:
        if is_combo_used(matrix, s["category_key"], s["sujet"], s["contexte"]):
            print(f"  Combinaison déjà utilisée: {s['sujet']} / {s['contexte']}")
            all_valid = False
        else:
            print(f"  Suggestion validée [{s['category_key']}]: {s['sujet']} / {s['contexte']} / {s['angle']}")

    if not all_valid:
        print("  Certaines suggestions sont des doublons, retry nécessaire")
        return None

    cats = [s["category_key"] for s in suggestions]
    if sorted(cats) != ["cat1_pensees", "cat2_emotions", "cat3_schemas"]:
        print(f"  Les catégories ne sont pas correctes: {cats}")
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
                "prenom": "",  # Sera rempli par select_best_personnage
                "age": "",     # Sera rempli par select_best_personnage
            }

    print(f"  [ERREUR] Impossible de trouver une combinaison unique pour {category_key} après {max_attempts} tentatives")
    return None


# ============================================
# GPT : RÉDACTION DES ARTICLES
# ============================================

def build_system_prompt():
    """Construit le system prompt pour GPT (rédaction)."""
    # Liste des prénoms réservés aux personnages récurrents (ne pas utiliser pour les secondaires)
    prenoms_interdits = ", ".join(PRENOMS)

    return f"""Tu es un rédacteur expert en psychologie vulgarisée et en SEO francophone. Tu rédiges des articles de blog pour le site "Décode ton esprit", dont la mission est d'aider les lecteurs à mieux se comprendre eux-mêmes grâce à la psychologie humaine.

MÉTHODE NARRATIVE OBLIGATOIRE :
- Chaque article raconte l'HISTOIRE d'un personnage fictif (prénom, âge, situation fournis)
- Le personnage est confronté à une situation concrète du quotidien liée au concept psychologique
- Le lecteur doit se reconnaître dans cette histoire
- L'histoire sert de fil conducteur pour expliquer le concept et les solutions
- Le personnage évolue au fil de l'article : il/elle comprend son fonctionnement et commence à changer
- L'histoire doit être réaliste, touchante, avec des détails sensoriels et émotionnels
- GENRE GRAMMATICAL : Respecte STRICTEMENT le genre du personnage (féminin ou masculin) indiqué dans la fiche. Utilise les accords corrects pour les adjectifs, participes passés, pronoms, etc. Un personnage féminin utilise "elle", "assise", "installée", etc. Un personnage masculin utilise "il", "assis", "installé", etc.
- CONTINUITÉ NARRATIVE : Les personnages sont RÉCURRENTS. Si le personnage a un historique d'articles, l'histoire d'aujourd'hui est la SUITE de son parcours. Fais référence naturellement à ses expériences passées. Montre son évolution. Les techniques apprises dans les articles précédents sont des acquis.
- PRÉNOMS INTERDITS POUR LES PERSONNAGES SECONDAIRES : Quand tu crées des personnages secondaires (exemples illustratifs, collègues, amis, proches mentionnés dans l'histoire), tu ne dois JAMAIS utiliser l'un des 20 prénoms suivants, car ils sont réservés aux personnages récurrents du blog : {prenoms_interdits}. Choisis des prénoms courants différents (ex : Julie, Antoine, Claire, Mathieu, Sarah, etc.). C'est une règle ABSOLUE.
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

TITRE_SEO: {{titre optimisé SEO, max 65 caractères}}
META_DESCRIPTION: {{description unique, max 155 caractères, qui donne envie de cliquer}}
SLUG: {{slug-en-minuscules-avec-tirets-sans-accents, max 6 mots}}
TAGS: {{tag1, tag2, tag3, tag4, tag5}}
---
{{contenu complet de l'article en Markdown, commençant directement par le texte sans répéter le titre H1}}"""


def build_article_prompt(combo):
    """Construit le prompt spécifique pour un article, avec ancrage temporel et continuité narrative."""
    profil_text = f"\nPROFIL DU LECTEUR CIBLE : {combo['profil']}" if combo.get('profil') else ""

    # Date du jour pour l'ancrage temporel
    today = datetime.now()
    today_str = today.strftime("%d %B %Y")
    mois_fr = {
        "January": "janvier", "February": "février", "March": "mars",
        "April": "avril", "May": "mai", "June": "juin",
        "July": "juillet", "August": "août", "September": "septembre",
        "October": "octobre", "November": "novembre", "December": "décembre"
    }
    for en, fr in mois_fr.items():
        today_str = today_str.replace(en, fr)

    # Scène envisagée par Gemini (si character-first)
    scene_section = ""
    if combo.get("scene_envisagee"):
        scene_section = f"""
SCÈNE DE DÉPART SUGGÉRÉE (tu peux t'en inspirer ou l'adapter) :
{combo['scene_envisagee']}
"""

    justification_section = ""
    if combo.get("justification_narrative"):
        justification_section = f"""
LOGIQUE NARRATIVE (pourquoi cette histoire maintenant) :
{combo['justification_narrative']}
"""

    coherence_section = ""
    if combo.get("strategie_coherence") or combo.get("apport_psychologique"):
        coherence_section = f"""
STRATÉGIE DE COHÉRENCE ACTIVE :
- Type : {combo.get('strategie_coherence', 'non spécifié')}
- Apport psychologique attendu : {combo.get('apport_psychologique', '')}
- INSTRUCTION : Cet article doit ACTIVEMENT enrichir le portrait psychologique de {combo['prenom']}.
  Il ne suffit pas de mentionner le passé du personnage : l'article doit RELIER ce nouveau sujet
  aux expériences passées pour créer un portrait plus profond et cohérent.
  Le lecteur qui a suivi {combo['prenom']} doit sentir que sa psychologie devient plus lisible et compréhensible.
"""

    # Contexte du personnage
    personnage_context = combo.get('personnage_context', '')
    genre_info = combo.get('genre', 'M')
    if personnage_context:
        personnage_section = f"""
{personnage_context}

IMPORTANT : Tu DOIS respecter l'identité, l'âge, le GENRE (accords {"féminins" if genre_info == "F" else "masculins"}), la profession et les relations du personnage tels que décrits ci-dessus. {combo['prenom']} a TOUJOURS {combo['age']}. Utilise ses traits de personnalité et son histoire pour rendre le récit cohérent avec les articles précédents."""
    else:
        personnage_section = f"""PERSONNAGE DE L'HISTOIRE :
- Prénom : {combo['prenom']}
- Âge : {combo['age']}
- Genre : {"féminin" if genre_info == "F" else "masculin"} (respecte les accords grammaticaux)
- Situation : {combo['prenom']} vit une situation liée à "{combo['sujet']}" dans le contexte "{combo['contexte']}" """

    return f"""Rédige un article de blog complet avec les paramètres suivants :

DATE DU JOUR : {today_str} (l'histoire se passe AUJOURD'HUI, en France)
CATÉGORIE : {combo['category_name']}
SUJET PRINCIPAL : {combo['sujet']}
CONTEXTE DE VIE : {combo['contexte']}
ANGLE ÉDITORIAL : {combo['angle']}{profil_text}

{personnage_section}
{scene_section}{justification_section}{coherence_section}
ANCRAGE TEMPOREL OBLIGATOIRE :
- L'histoire se déroule dans le PRÉSENT, en France, autour du {today_str}
- Si le personnage a un historique d'articles précédents, fais référence NATURELLEMENT à des événements passés
  (par exemple : "Depuis cette conversation avec [relation] il y a quelques semaines..." ou
  "Elle se souvient de cette période où elle avait compris que...")
- Le personnage ÉVOLUE : il/elle n'est pas le/la même qu'au début de son parcours

CONSIGNES SPÉCIFIQUES (structure OBLIGATOIRE avec titres H2/H3 Markdown) :
1. INTRODUCTION NARRATIVE (SANS titre H2) : 3-4 paragraphes immersifs plongeant le lecteur dans la vie de {combo['prenom']}. IMPORTANT : raconte AU PRÉSENT, comme si la scène se déroule sous les yeux du lecteur. {combo['prenom']} vit la situation en temps réel, aujourd'hui, en France.
2. ## Qu'est-ce que {combo['sujet']} ? (titre H2 OBLIGATOIRE) : Fais le lien avec la situation de {combo['prenom']}, puis explique le concept avec une définition claire en une phrase, le nom du chercheur associé, et un contexte scientifique.
3. ## Comment {combo['sujet']} se manifeste dans le contexte "{combo['contexte']}" ? (titre H2 OBLIGATOIRE sous forme de question) : Détaille 2-3 manifestations concrètes avec des sous-titres H3 (###) pour chaque manifestation. Ajoute des exemples variés.
4. ## 3 techniques pour [verbe d'action] face à {combo['sujet']} (titre H2 OBLIGATOIRE) : Présente chaque technique avec un sous-titre H3 numéroté et en gras (### 1. **Nom de la technique**). Chaque technique doit être détaillée sur un paragraphe complet avec un exercice concret.
5. ## {combo['prenom']} commence à [verbe de transformation] (titre H2 OBLIGATOIRE) : Retour au personnage qui applique les techniques et évolue positivement. Montre comment cette étape s'inscrit dans son parcours global.
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

    # Échapper les guillemets doubles dans le titre et la description pour éviter de casser le YAML
    safe_title = metadata.get('title', 'Article du jour').replace('"', '\\"')
    safe_description = metadata.get('description', '').replace('"', '\\"')

    front_matter = f"""---
title: "{safe_title}"
date: {date_str}
description: "{safe_description}"
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
# VALIDATION DE COHÉRENCE POST-GÉNÉRATION
# ============================================

def validate_coherence(content, combo, matrix, personnages):
    """Valide que l'article généré contribue activement à la cohérence du personnage.

    Appelle Gemini pour évaluer si l'article :
    1. Fait référence au passé du personnage de manière naturelle
    2. Approfondit réellement la psychologie du personnage
    3. Est cohérent avec les traits et l'histoire établis
    4. Apporte quelque chose de nouveau au portrait psychologique

    Retourne le contenu (éventuellement enrichi) et un score de cohérence.
    """
    prenom = combo["prenom"]

    # Trouver les articles passés du personnage
    perso_articles = [a for a in matrix.get("articles", []) if a.get("prenom") == prenom]

    if not perso_articles:
        # Premier article du personnage : pas de cohérence à valider
        print(f"  [Cohérence] Premier article de {prenom}, validation simplifiée")
        return content

    # Construire le résumé des articles passés pour la vérification
    past_summary = []
    for a in perso_articles[-6:]:  # 6 derniers articles max
        entry = f"- \"{a.get('sujet', '')}\" ({a.get('contexte', '')})"
        if a.get("resume_narratif"):
            entry += f" : {_to_str(a['resume_narratif'])}"
        if a.get("evolution"):
            entry += f" | Évolution : {_to_str(a['evolution'])}"
        past_summary.append(entry)

    system_prompt = (
        "Tu es un expert en cohérence narrative et en psychologie des personnages. "
        "On te donne un article de blog et l'historique du personnage. "
        "Tu dois vérifier que l'article contribue ACTIVEMENT à la cohérence du personnage.\n\n"
        "VÉRIFICATIONS :\n"
        "1. L'article fait-il référence au passé du personnage (événements, relations, techniques apprises) ?\n"
        "2. Le comportement du personnage est-il cohérent avec ses traits établis ?\n"
        "3. L'article apporte-t-il une NOUVELLE dimension au portrait psychologique ?\n"
        "4. Les acquis des articles précédents (techniques, prises de conscience) sont-ils respectés ?\n\n"
        "Si l'article manque de références au passé ou de cohérence, ENRICHIS-LE en ajoutant "
        "naturellement 2-3 passages qui :\n"
        "- Mentionnent un événement passé du personnage\n"
        "- Montrent que le personnage a évolué\n"
        "- Relient le sujet actuel aux expériences antérieures\n\n"
        "RÈGLES : Conserve toute la structure Markdown, ne raccourcis pas l'article, "
        "conserve le même ton. Retourne l'article complet (enrichi si nécessaire), rien d'autre."
    )

    user_prompt = (
        f"PERSONNAGE : {prenom} ({combo.get('age', '')})\n"
        f"SUJET ACTUEL : {combo['sujet']} en contexte \"{combo['contexte']}\"\n"
        f"STRATÉGIE DE COHÉRENCE : {combo.get('strategie_coherence', 'non spécifiée')}\n\n"
        f"HISTORIQUE DU PERSONNAGE ({len(perso_articles)} articles précédents) :\n"
        + "\n".join(past_summary)
        + f"\n\nARTICLE À VÉRIFIER :\n{content}"
    )

    print(f"  [Cohérence] Validation de la cohérence narrative pour {prenom}...")

    try:
        enriched = call_mammouth_api(
            model=MODEL_ANALYST,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=4500,
            retries=2
        )

        if enriched:
            # Vérifier que Gemini n'a pas cassé la structure
            h2_before = len(re.findall(r'^## ', content, re.MULTILINE))
            h2_after = len(re.findall(r'^## ', enriched, re.MULTILINE))
            len_before = len(content.split())
            len_after = len(enriched.split())

            if h2_after >= h2_before and len_after >= len_before * 0.8:
                if len_after > len_before:
                    print(f"  [Cohérence] Article enrichi ({len_before} → {len_after} mots)")
                else:
                    print(f"  [Cohérence] Cohérence validée ({len_after} mots)")
                return enriched
            else:
                print(f"  [Cohérence] Gemini a altéré la structure, conservation de l'original")
                return content
        else:
            print(f"  [Cohérence] Gemini n'a pas répondu, conservation de l'original")
            return content

    except Exception as e:
        print(f"  [Cohérence] Erreur: {e}, conservation de l'original")
        return content


# ============================================
# GÉNÉRATION DE LA PAGE DE SUIVI DES PERSONNAGES
# ============================================

def _build_article_url(article):
    """Construit l'URL Hugo d'un article à partir de sa catégorie et son slug."""
    slug = article.get("slug", "")
    if not slug:
        return ""
    cat_key = article.get("category_key", "")
    section_map = {
        "cat1_pensees": "reprendre-le-controle-de-ses-pensees",
        "cat2_emotions": "comprendre-et-maitriser-ses-emotions",
        "cat3_schemas": "sortir-de-ses-schemas-repetitifs",
    }
    section = section_map.get(cat_key, "")
    if not section:
        return ""
    return f"/posts/{section}/{slug}/"


def _cat_label(cat_key):
    """Retourne le label lisible d'une catégorie."""
    return {
        "cat1_pensees": "Biais cognitif",
        "cat2_emotions": "Émotion",
        "cat3_schemas": "Schéma répétitif",
    }.get(cat_key, cat_key)


def _perso_bio(perso, nb_articles):
    """Génère une courte bio pour la carte du personnage."""
    prenom = perso["prenom"]
    genre = perso.get("genre", "M")
    profession = perso["profession"]
    situation = perso["situation_familiale"]
    traits = perso["traits_personnalite"]
    il_elle = "Elle" if genre == "F" else "Il"

    traits_str = ", ".join(traits[:3])
    art_label = f"{nb_articles} article{'s' if nb_articles > 1 else ''}"

    return (
        f"{prenom} est {profession}, {situation}. "
        f"{il_elle} se distingue par un tempérament {traits_str}. "
        f"Découvrez son parcours à travers {art_label}."
    )


def generate_character_tracking_page(personnages, matrix):
    """Génère une page Hugo avec grille de cartes et détail dépliable par personnage.

    Niveau 1 : Grille de 20 cartes (avatar, nom, bio, bouton)
    Niveau 2 : Détail dépliable avec timeline des articles + résumé narratif
    """
    articles = matrix.get("articles", [])

    # Regrouper les articles par personnage
    perso_articles = {}
    for a in articles:
        prenom = a.get("prenom", "")
        if prenom:
            perso_articles.setdefault(prenom, []).append(a)

    for prenom in perso_articles:
        perso_articles[prenom].sort(key=lambda x: x.get("date", ""))

    # Préparer les données de chaque personnage
    perso_data = []
    for perso in personnages:
        prenom = perso["prenom"]
        arts = perso_articles.get(prenom, [])
        perso_data.append((perso, arts))

    # Trier par nombre d'articles décroissant
    perso_data.sort(key=lambda x: len(x[1]), reverse=True)

    # Générer la page
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    lines = []
    lines.append("---")
    lines.append('title: "Nos Personnages"')
    lines.append(f"date: {date_str}")
    lines.append('description: "Découvrez les 20 personnages récurrents du blog et suivez leur évolution psychologique au fil des articles."')
    lines.append('layout: "personnages"')
    lines.append('slug: "personnages"')
    lines.append("draft: false")
    lines.append("---\n")

    # ── GRILLE DE CARTES ──
    lines.append('<div class="perso-grid">')

    for perso, arts in perso_data:
        prenom = perso["prenom"]
        genre = perso.get("genre", "M")
        initiale = prenom[0].upper()
        nb = len(arts)
        bio = _perso_bio(perso, nb)
        perso_id = prenom.lower().replace(" ", "-")

        lines.append(f'  <div class="perso-card" id="card-{perso_id}">')

        # Avatar + identité
        lines.append('    <div class="perso-card-top">')
        lines.append(f'      <div class="perso-avatar">{initiale}</div>')
        lines.append(f'      <h2 class="perso-name">{prenom}, {perso["age"]} ans</h2>')
        lines.append(f'      <p class="perso-job">{perso["profession"].capitalize()}</p>')
        lines.append('    </div>')

        # Traits
        traits_html = "".join(f'<span class="trait-tag">{t}</span>' for t in perso["traits_personnalite"])
        lines.append(f'    <div class="perso-traits">{traits_html}</div>')

        # Bio courte
        lines.append(f'    <p class="perso-bio">{bio}</p>')

        # Stats
        lines.append(f'    <div class="perso-stat">{nb} article{"s" if nb > 1 else ""} publi\u00e9{"s" if nb > 1 else ""}</div>')

        # Détail dépliable
        if arts:
            lines.append(f'    <details class="perso-details">')
            lines.append(f'      <summary class="perso-toggle">Voir le parcours</summary>')
            lines.append(f'      <div class="perso-parcours">')

            for a in arts:
                date = a.get("date", "")
                title = a.get("title", a.get("sujet", ""))
                cat_key = a.get("category_key", "")
                resume = _to_str(a.get("resume_narratif", ""))
                evolution = _to_str(a.get("evolution", ""))
                url = _build_article_url(a)

                lines.append('        <div class="parcours-entry">')
                lines.append(f'          <div class="parcours-cat">{_cat_label(cat_key)}</div>')

                if url:
                    lines.append(f'          <h3 class="parcours-title"><a href="{url}">{title}</a></h3>')
                else:
                    lines.append(f'          <h3 class="parcours-title">{title}</h3>')

                lines.append(f'          <div class="parcours-date">{date}</div>')

                if resume:
                    lines.append(f'          <p class="parcours-resume">{resume}</p>')

                if evolution:
                    lines.append(f'          <div class="parcours-evolution">')
                    lines.append(f'            <span class="evolution-label">\u00c9volution</span>')
                    lines.append(f'            <p>{evolution}</p>')
                    lines.append(f'          </div>')

                if url:
                    lines.append(f'          <a href="{url}" class="parcours-link">Lire l\'article complet &rarr;</a>')

                lines.append('        </div>')

            lines.append('      </div>')
            lines.append('    </details>')
        else:
            il = "Elle" if genre == "F" else "Il"
            lines.append(f'    <p class="perso-waiting">{il} attend sa premi\u00e8re histoire...</p>')

        lines.append('  </div>')

    lines.append('</div>')

    # Écrire le fichier
    tracking_dir = REPO_ROOT / "content"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_file = tracking_dir / "personnages.md"

    full_content = "\n".join(lines)
    with open(tracking_file, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"  [Suivi] Page de suivi des personnages générée : {tracking_file}")
    return tracking_file


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

    # ── ÉTAPE 1 : Charger la matrice et les personnages ──
    print("ETAPE 1 : Chargement de la matrice et des personnages...")
    matrix = load_matrix()
    migrate_tracking_to_matrix(matrix)
    personnages = load_personnages()
    print(f"  {len(matrix['articles'])} combinaisons dans la matrice")
    print(f"  {len(personnages)} personnages récurrents chargés\n")

    # ── ÉTAPE 2 : Gemini analyse les arcs narratifs et propose personnage+sujet ──
    print("ETAPE 2 : Analyse narrative par Gemini (character-first)...")
    suggestions = None
    is_character_first = False

    for attempt in range(3):
        try:
            suggestions = get_gemini_suggestions(matrix, personnages if personnages else None)
            if suggestions:
                # Déterminer si c'est character-first (contient 'personnage') ou legacy
                is_character_first = "personnage" in suggestions[0]
                break
        except Exception as e:
            print(f"  Erreur Gemini tentative {attempt + 1}: {e}")
        if attempt < 2:
            print(f"  Nouvelle tentative Gemini ({attempt + 2}/3)...")

    use_gemini = suggestions is not None
    if not use_gemini:
        print("  Gemini n'a pas pu fournir de suggestions valides, fallback aléatoire\n")

    # ── ÉTAPE 3 : Rédiger les articles avec GPT ──
    print("\nETAPE 3 : Rédaction des articles par GPT...")
    system_prompt = build_system_prompt()

    for i, cat_key in enumerate(category_keys):
        if use_gemini:
            s = suggestions[i]
            print(f"\n  --- Catégorie : {CATEGORIES[s['category_key']]['name']} ---")
            combo = {
                "category_key": s["category_key"],
                "category_name": CATEGORIES[s["category_key"]]["name"],
                "category_slug": CATEGORIES[s["category_key"]]["slug"],
                "sujet": s["sujet"],
                "contexte": s["contexte"],
                "angle": s["angle"],
                "profil": s.get("profil"),
            }

            if is_character_first and s.get("personnage"):
                # Character-first : le personnage est déjà choisi par Gemini
                perso = s["personnage"]
                combo["prenom"] = perso["prenom"]
                combo["age"] = f"{perso['age']} ans"
                combo["genre"] = perso.get("genre", "M")
                combo["personnage_context"] = build_personnage_context(perso, matrix)
                combo["scene_envisagee"] = s.get("scene_envisagee", "")
                combo["justification_narrative"] = s.get("justification_narrative", "")
                combo["strategie_coherence"] = s.get("strategie_coherence", "")
                combo["apport_psychologique"] = s.get("apport_psychologique", "")
                print(f"  [Character-first] Personnage : {combo['prenom']} ({combo['age']})")
                print(f"  [Character-first] Sujet : {combo['sujet']}")
                print(f"  [Character-first] Contexte : {combo['contexte']}")
                if s.get("strategie_coherence"):
                    print(f"  [Cohérence] Stratégie : {s['strategie_coherence']}")
                if s.get("justification_narrative"):
                    print(f"  [Narratif] {s['justification_narrative'][:150]}...")
            else:
                # Legacy : on cherche le meilleur personnage après
                print(f"  [Legacy] Sujet : {combo['sujet']}")
                print(f"  [Legacy] Contexte : {combo['contexte']}")
                if personnages:
                    best_perso = select_best_personnage(personnages, combo["category_key"], combo["sujet"], combo["contexte"], matrix)
                    combo["prenom"] = best_perso["prenom"]
                    combo["age"] = f"{best_perso['age']} ans"
                    combo["genre"] = best_perso.get("genre", "M")
                    combo["personnage_context"] = build_personnage_context(best_perso, matrix)
                else:
                    combo["prenom"] = random.choice(PRENOMS)
                    combo["age"] = random.choice(TRANCHES_AGE)
                    combo["genre"] = "M"
        else:
            print(f"\n  --- Catégorie : {CATEGORIES[cat_key]['name']} ---")
            combo = generate_random_combination(cat_key, matrix)
            if not combo:
                print(f"  [SKIP] Catégorie {cat_key} ignorée (toutes les combinaisons épuisées)")
                continue
            print(f"  [Aléatoire] Sujet : {combo['sujet']}")
            if personnages:
                best_perso = select_best_personnage(personnages, combo["category_key"], combo["sujet"], combo["contexte"], matrix)
                combo["prenom"] = best_perso["prenom"]
                combo["age"] = f"{best_perso['age']} ans"
                combo["genre"] = best_perso.get("genre", "M")
                combo["personnage_context"] = build_personnage_context(best_perso, matrix)
            else:
                combo["prenom"] = random.choice(PRENOMS)
                combo["age"] = random.choice(TRANCHES_AGE)
                combo["genre"] = "M"

        print(f"  Personnage : {combo['prenom']} ({combo['age']})")

        # Rédaction avec retry
        metadata = None
        content = None
        max_redaction_attempts = 3

        for redaction_attempt in range(max_redaction_attempts):
            attempt_label = f" (tentative {redaction_attempt + 1}/{max_redaction_attempts})" if redaction_attempt > 0 else ""
            print(f"  Appel GPT-5-mini pour rédaction{attempt_label}...")
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

            print(f"  Parsing de la réponse...")
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
                    print(f"  [Erreur] Échec après {max_redaction_attempts} tentatives")
                    metadata = None
                    content = None

        if not metadata or not content:
            print(f"  [SKIP] Catégorie {cat_key} ignorée (impossible de générer un article valide)")
            continue

        # Étape 3.5 : Vérification et correction des tirets cadratins
        content = verify_and_fix_emdashes(content, combo)

        # Étape 3.6 : Vérification de la structure H2/H3
        content = verify_article_structure(content, combo)

        # Étape 3.7 : Relecture qualité globale par Gemini
        content = gemini_quality_review(content, combo)

        # Étape 3.8 : Validation de cohérence narrative
        content = validate_coherence(content, combo, matrix, personnages)

        print(f"  Création du fichier Hugo...")
        create_hugo_post(combo, metadata, content)

        # Étape 3.9 : Extraction du résumé narratif pour la continuité
        print(f"  [Narratif] Extraction du résumé narratif...")
        resume_narratif, evolution, elements_cles = extract_narrative_summary(content, combo)
        if resume_narratif:
            print(f"  [Narratif] Résumé : {resume_narratif[:100]}...")
        else:
            print(f"  [Narratif] Pas de résumé extrait (sera vide dans la matrice)")

        # Ajouter à la matrice avec les données narratives
        add_to_matrix(matrix, combo, metadata, resume_narratif, evolution, elements_cles)

        # Mettre à jour l'historique du personnage
        if personnages:
            update_personnage_history(personnages, combo["prenom"], {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "sujet": combo["sujet"],
                "contexte": combo["contexte"],
                "category_key": combo["category_key"],
                "title": metadata.get("title", ""),
                "slug": metadata.get("slug", ""),
            })

    # Sauvegarder la matrice et les personnages
    save_matrix(matrix)
    if personnages:
        save_personnages(personnages)
        print(f"  Historique des personnages mis à jour")

    # ── ÉTAPE 4 : Générer la page de suivi des personnages ──
    print("\nETAPE 4 : Génération de la page de suivi des personnages...")
    if personnages:
        generate_character_tracking_page(personnages, matrix)

    print(f"\n{'='*60}")
    print(f"Génération terminée - {len(matrix['articles'])} combinaisons dans la matrice")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
