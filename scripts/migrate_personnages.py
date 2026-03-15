#!/usr/bin/env python3
"""
Script de migration : attribue les 20 personnages récurrents aux articles existants.

Pour chaque article :
1. Détermine le personnage le plus pertinent selon le sujet et le contexte
2. Remplace l'ancien prénom par le nouveau dans tout le contenu
3. Met à jour le front matter (personnage, âge)
4. Met à jour la matrice des combinaisons
5. Met à jour l'historique des personnages
"""

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONTENT_DIR = REPO_ROOT / "content" / "posts"
MATRIX_FILE = SCRIPT_DIR / "matrice_combinaisons.json"
PERSONNAGES_FILE = SCRIPT_DIR / "personnages.json"

sys.path.insert(0, str(SCRIPT_DIR))
from config import CATEGORIES

# ============================================
# Chargement des données
# ============================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================
# Sélection du personnage (même algorithme que generate_articles.py)
# ============================================

def select_best_personnage(personnages, category_key, sujet, contexte, current_assignments):
    """Sélectionne le personnage le plus pertinent. current_assignments compte les attributions actuelles."""
    scores = []
    for perso in personnages:
        score = 0
        prenom = perso["prenom"]

        # 1. Score d'affinité thématique (0-30 points)
        affinites = perso.get("affinites_thematiques", {}).get(category_key, [])
        sujet_lower = sujet.lower()
        best_affinite_score = 0
        for affinite in affinites:
            affinite_lower = affinite.lower()
            if sujet_lower == affinite_lower:
                best_affinite_score = max(best_affinite_score, 30)
            elif sujet_lower in affinite_lower or affinite_lower in sujet_lower:
                best_affinite_score = max(best_affinite_score, 20)
            else:
                mots_sujet = set(sujet_lower.split())
                mots_affinite = set(affinite_lower.split())
                communs = mots_sujet & mots_affinite - {"de", "du", "la", "le", "les", "des", "en", "et", "à", "l'", "d'", "un", "une"}
                if communs:
                    best_affinite_score = max(best_affinite_score, min(15, len(communs) * 5))
        score += best_affinite_score

        # 2. Bonus contextuel (0-10 points)
        contexte_lower = contexte.lower()
        situation = perso.get("situation_familiale", "").lower()
        profession = perso.get("profession", "").lower()
        traits = " ".join(perso.get("traits_personnalite", [])).lower()

        if "travail" in contexte_lower and any(w in profession for w in ["entreprise", "manager", "cadre", "directrice", "chef", "commercial"]):
            score += 10
        elif "couple" in contexte_lower and any(w in situation for w in ["marié", "couple", "compagne", "compagnon"]):
            score += 10
        elif "famille" in contexte_lower and any(w in situation for w in ["enfant", "mère", "père", "parent"]):
            score += 10
        elif "parent" in contexte_lower and any(w in situation for w in ["enfant", "mère", "père"]):
            score += 10
        elif "enfant" in contexte_lower and any(w in situation for w in ["enfant", "mère", "père"]):
            score += 10
        elif "école" in contexte_lower and any(w in profession for w in ["étudiant", "enseignant"]):
            score += 10
        elif "entretien" in contexte_lower and any(w in profession for w in ["étudiant", "reconversion", "commercial"]):
            score += 8
        elif "manager" in contexte_lower and any(w in profession for w in ["manager", "directrice", "cadre", "chef"]):
            score += 10
        elif "solitude" in contexte_lower and any(w in situation for w in ["seul", "célibataire", "veuf", "veuve"]):
            score += 10
        elif "rupture" in contexte_lower and any(w in situation or w in perso.get("histoire_de_fond", "").lower() for w in ["rupture", "sépar", "divorc"]):
            score += 10
        elif "deuil" in contexte_lower and ("décédé" in str(perso.get("relations", {})).lower() or "veuf" in situation or "veuve" in situation):
            score += 10
        elif "reconversion" in contexte_lower and "reconversion" in profession:
            score += 10
        elif "vieillissement" in contexte_lower and perso["age"] >= 50:
            score += 10
        elif "compétition" in contexte_lower and ("compétitif" in traits or "ambitieux" in traits):
            score += 8
        elif "réseaux sociaux" in contexte_lower and perso["age"] <= 30:
            score += 8
        elif "intimité" in contexte_lower and ("couple" in situation or "compagne" in situation):
            score += 8
        elif "amitié" in contexte_lower:
            score += 3
        elif "conflit" in contexte_lower:
            score += 3

        # 3. Malus de sur-utilisation
        nb = current_assignments.get(prenom, 0)
        if nb > 5:
            score -= (nb - 5) * 8
        elif nb > 3:
            score -= (nb - 3) * 3

        # 4. Bonus de diversité
        if nb == 0:
            score += 5
        elif nb <= 2:
            score += 2

        scores.append((perso, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0], scores[0][1]


# ============================================
# Remplacement dans les articles
# ============================================

def replace_name_in_content(content, old_name, new_name):
    """Remplace un prénom par un autre dans le contenu, en respectant les cas possessifs et contextes."""
    if old_name == new_name:
        return content

    # Remplacement direct du prénom
    content = content.replace(old_name, new_name)

    return content


def update_article_file(file_path, old_name, new_perso):
    """Met à jour un fichier article avec le nouveau personnage."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_name = new_perso["prenom"]

    # Remplacer le prénom dans le contenu
    content = replace_name_in_content(content, old_name, new_name)

    # Mettre à jour le front matter - personnage
    content = re.sub(
        r'^personnage: ".*?"',
        f'personnage: "{new_name}"',
        content,
        count=1,
        flags=re.MULTILINE
    )

    # Mettre à jour le titre s'il contient l'ancien prénom
    content = re.sub(
        rf'^(title: ".*?){re.escape(old_name)}(.*?")',
        rf'\g<1>{new_name}\g<2>',
        content,
        count=1,
        flags=re.MULTILINE
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return True


# ============================================
# Script principal
# ============================================

def main():
    print("=" * 60)
    print("MIGRATION DES PERSONNAGES RÉCURRENTS")
    print("=" * 60)

    # Charger les données
    personnages_data = load_json(PERSONNAGES_FILE)
    personnages = personnages_data["personnages"]
    matrix = load_json(MATRIX_FILE)
    articles = matrix.get("articles", [])

    print(f"\n  {len(personnages)} personnages disponibles")
    print(f"  {len(articles)} articles dans la matrice")

    # Lister tous les fichiers articles
    all_article_files = {}
    for cat_dir in CONTENT_DIR.iterdir():
        if cat_dir.is_dir():
            for md_file in cat_dir.glob("*.md"):
                # Extraire le personnage et le sujet du front matter
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                m_perso = re.search(r'^personnage: "(.+?)"', content, re.MULTILINE)
                m_sujet = re.search(r'^sujet: "(.+?)"', content, re.MULTILINE)
                m_contexte = re.search(r'^contexte: "(.+?)"', content, re.MULTILINE)
                m_slug = re.search(r'^slug: "(.+?)"', content, re.MULTILINE)
                m_title = re.search(r'^title: "(.+?)"', content, re.MULTILINE)
                m_categories = re.search(r'^categories: \["(.+?)"\]', content, re.MULTILINE)

                if m_perso and m_sujet:
                    slug = m_slug.group(1) if m_slug else md_file.stem
                    all_article_files[slug] = {
                        "path": md_file,
                        "old_name": m_perso.group(1),
                        "sujet": m_sujet.group(1),
                        "contexte": m_contexte.group(1) if m_contexte else "",
                        "title": m_title.group(1) if m_title else "",
                        "category_name": m_categories.group(1) if m_categories else "",
                    }

    print(f"  {len(all_article_files)} articles trouvés dans les fichiers\n")

    # Déterminer la catégorie de chaque article
    cat_name_to_key = {}
    for key, cat in CATEGORIES.items():
        cat_name_to_key[cat["name"]] = key

    # Attribution des personnages
    current_assignments = {}  # Compteur d'attributions
    article_mapping = []  # Liste des (slug, old_name, new_perso, score)

    # Trier les articles par pertinence thématique pour optimiser la distribution
    articles_to_assign = []
    for slug, info in all_article_files.items():
        cat_key = cat_name_to_key.get(info["category_name"], "")
        if not cat_key:
            # Chercher par le chemin du répertoire
            path_str = str(info["path"])
            if "reprendre-le-controle" in path_str:
                cat_key = "cat1_pensees"
            elif "comprendre-et-maitriser" in path_str:
                cat_key = "cat2_emotions"
            elif "sortir-de-ses-schemas" in path_str:
                cat_key = "cat3_schemas"
        articles_to_assign.append((slug, info, cat_key))

    print("ATTRIBUTION DES PERSONNAGES :")
    print("-" * 60)

    for slug, info, cat_key in articles_to_assign:
        best_perso, score = select_best_personnage(
            personnages, cat_key, info["sujet"], info["contexte"], current_assignments
        )
        new_name = best_perso["prenom"]
        current_assignments[new_name] = current_assignments.get(new_name, 0) + 1

        article_mapping.append((slug, info, best_perso, score))

        changed = "✓ CHANGÉ" if info["old_name"] != new_name else "= MÊME"
        print(f"  [{changed}] {slug}")
        print(f"    {info['old_name']} → {new_name} (score: {score})")
        print(f"    Sujet: {info['sujet']} | Contexte: {info['contexte']}")

    # Afficher la distribution finale
    print(f"\n{'=' * 60}")
    print("DISTRIBUTION DES PERSONNAGES :")
    print("-" * 60)
    for name, count in sorted(current_assignments.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"  {name:12s} : {count:2d} articles {bar}")

    unused = [p["prenom"] for p in personnages if p["prenom"] not in current_assignments]
    if unused:
        print(f"\n  Non utilisés : {', '.join(unused)}")

    # Appliquer les modifications aux fichiers
    print(f"\n{'=' * 60}")
    print("APPLICATION DES MODIFICATIONS :")
    print("-" * 60)

    modified_count = 0
    for slug, info, new_perso, score in article_mapping:
        old_name = info["old_name"]
        new_name = new_perso["prenom"]

        if old_name != new_name:
            update_article_file(info["path"], old_name, new_perso)
            modified_count += 1
            print(f"  ✓ Modifié : {slug} ({old_name} → {new_name})")

    print(f"\n  {modified_count} articles modifiés sur {len(article_mapping)}")

    # Mettre à jour la matrice
    print(f"\nMISE À JOUR DE LA MATRICE :")
    # Créer un mapping slug → new_name
    slug_to_new_name = {slug: new_perso["prenom"] for slug, info, new_perso, score in article_mapping}

    for article in articles:
        slug = article.get("slug", "")
        if slug in slug_to_new_name:
            old = article.get("prenom", "")
            new = slug_to_new_name[slug]
            if old != new:
                article["prenom"] = new

    save_json(MATRIX_FILE, matrix)
    print(f"  Matrice mise à jour")

    # Mettre à jour l'historique des personnages
    print(f"\nMISE À JOUR DE L'HISTORIQUE DES PERSONNAGES :")
    # Réinitialiser les historiques
    for perso in personnages:
        perso["historique_articles"] = []

    # Reconstruire depuis le mapping
    for slug, info, new_perso, score in article_mapping:
        cat_key = ""
        path_str = str(info["path"])
        if "reprendre-le-controle" in path_str:
            cat_key = "cat1_pensees"
        elif "comprendre-et-maitriser" in path_str:
            cat_key = "cat2_emotions"
        elif "sortir-de-ses-schemas" in path_str:
            cat_key = "cat3_schemas"

        for perso in personnages:
            if perso["prenom"] == new_perso["prenom"]:
                perso["historique_articles"].append({
                    "date": "",
                    "sujet": info["sujet"],
                    "contexte": info["contexte"],
                    "category": cat_key,
                    "titre": info["title"],
                    "slug": slug,
                })
                break

    save_json(PERSONNAGES_FILE, {"description": personnages_data["description"], "personnages": personnages})
    print(f"  Historique des personnages mis à jour")

    # Résumé final
    print(f"\n{'=' * 60}")
    print(f"MIGRATION TERMINÉE")
    print(f"  {modified_count} articles modifiés")
    print(f"  {len(current_assignments)} personnages utilisés sur {len(personnages)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
