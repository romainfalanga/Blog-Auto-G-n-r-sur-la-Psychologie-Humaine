"""Reproduce the chapter 06 coverage from WHO's pinned French MMS tabulation.

The source ZIP stays outside the repository. Only identifiers, titles and
hierarchy are extracted. The editorial canons are read from the existing JSON.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data/clinical_catalog.json"
RELEASE = "2026-01"
SOURCE_URL = "https://icdcdn.who.int/static/releasefiles/2026-01/SimpleTabulation-ICD-11-MMS-fr.zip"
SOURCE_SHA256 = "c5c7c4a07ad007892a642afd1599314f303eac5dcee89c1d1e315fcf64fa2f0f"
TABLE_SHA256 = "a517d9b79a0005e2fa0d5b58e9e8d199a8c1419248193761b10306ac5dbe3dd6"
MANUAL = "https://www.who.int/publications/i/item/9789240077263"
BROWSER = "https://icd.who.int/browse/2026-01/mms/fr"
TABLE_NAME = "SimpleTabulation-ICD-11-MMS-fr.txt"

RULES = {
    "diagnostic_distinct": "Parcours canonique associé à ce diagnostic ou à un syndrome clinique distinct.",
    "groupe_code": "Catégorie qui regroupe plusieurs parcours cliniquement distincts.",
    "presentation": "Précision sur le profil, la manifestation, la gravité, l’évolution ou la rémission. Elle ne devient pas une nouvelle personne.",
    "presentation_transversale": "Précision liée à plusieurs diagnostics. Les liens indiquent où l’aborder sans en faire un diagnostic.",
    "classe_substance": "Famille qui distingue usage nocif, dépendance et syndromes induits.",
    "syndrome_substance": "Même type de problème clinique dans une autre classe de substances. Le renvoi ne pose aucune équivalence de risques ni de traitement.",
    "syndromes_substance": "Ensemble de syndromes distincts induits par une substance. Le groupe ne pose pas un diagnostic.",
    "code_residuel": "Catégorie résiduelle précisée ou non précisée. Le code seul ne définit pas un tableau stable pour un personnage.",
    "categorie_ouverte": "Libellé ouvert ou cause inconnue. Un arc avec une cause ou un tableau déterminé ne peut pas être inféré de ce code.",
    "facteur_associe": "Facteur associé à une autre maladie. Sujet de compréhension et d’accompagnement, sans diagnostic autonome dans ce projet.",
}

PRESENTATIONS = {
    "6A25": ("6A20", "6A21", "6A22", "6A23", "6A24"),
    "6A80": ("6A60", "6A61", "6A62", "6A70", "6A71", "6A72"),
    "6D11": ("6D10",),
    "6D86": ("6D80", "6D81", "6D82", "6D83", "6D84"),
}

SUBSTANCE_SYNDROMES = (
    ("episode of harmful", "6C40.0"),
    ("intoxication", "6C40.3"),
    ("withdrawal", "6C44.4"),
    ("delirium", "6C40.5"),
    ("psychotic disorder", "6C41.6"),
    ("mood disorder", "6C46.70"),
    ("anxiety disorder", "6C48.40"),
    ("obsessive-compulsive", "6C45.72"),
    ("impulse control", "6C45.73"),
)

RELATED_SUBJECTS = (
    ("7A00", "domaine_associe", "L’insomnie chronique appartient au chapitre veille-sommeil. Elle mérite un traitement éditorial propre."),
    ("7B01.2", "domaine_associe", "Les cauchemars relèvent du chapitre veille-sommeil. Des cauchemars isolés n’établissent pas un trouble."),
    ("8A05.00", "domaine_associe", "Le syndrome de la Tourette est classé parmi les maladies du système nerveux et traité aussi dans le guide clinique des troubles neurodéveloppementaux."),
    ("8A05.01", "domaine_associe", "Les tics moteurs chroniques ont leur propre code neurologique."),
    ("8A05.02", "domaine_associe", "Les tics phoniques chroniques ont leur propre code neurologique."),
    ("GA34.41", "domaine_associe", "Le trouble dysphorique prémenstruel est rattaché au chapitre génito-urinaire. L’évaluation du lien temporel relève du soin."),
    ("HA60", "sante_sexuelle", "Le code de santé sexuelle ne classe pas les identités trans ou la diversité de genre parmi les troubles mentaux."),
    ("HA61", "sante_sexuelle", "Les préférences d’un enfant et la diversité de genre ne définissent pas à elles seules un état pathologique."),
    ("QD85", "contexte_non_diagnostique", "Épuisement professionnel lié au travail, distinct d’un diagnostic de dépression ou d’un autre trouble."),
    ("QE62", "contexte_non_diagnostique", "Le deuil ordinaire mérite du soutien et se distingue du trouble du deuil prolongé."),
    ("QE50.7", "contexte_non_diagnostique", "La difficulté de la personnalité se distingue du trouble de la personnalité."),
    ("QE10", "contexte_non_diagnostique", "Un risque lié à la consommation d’alcool ne suffit pas à établir une dépendance."),
    ("QE11", "contexte_non_diagnostique", "Un risque lié à la consommation de drogues ne suffit pas à établir une dépendance."),
    ("QE12", "contexte_non_diagnostique", "Un risque lié à la nicotine ne suffit pas à établir une dépendance."),
    ("QE21", "contexte_non_diagnostique", "Le jeu d’argent à risque et le trouble du jeu d’argent ont des codes différents."),
    ("QE22", "contexte_non_diagnostique", "La pratique du jeu vidéo à risque et le trouble du jeu vidéo ont des codes différents."),
)

UNCODED_SUBJECTS = (
    ("variations-cognitives", "Diversité cognitive et besoins d’accessibilité", "La neurodiversité et le haut potentiel intellectuel ne constituent pas des diagnostics en tant que tels. Les besoins se discutent dans leur contexte."),
    ("identite-orientation", "Identité, orientation et expériences de discrimination", "Les récits peuvent explorer l’appartenance, le consentement et les effets de la discrimination sans pathologiser l’identité ou l’orientation."),
    ("relations-societe", "Relations, travail, vie sociale et inégalités", "Inclure les ruptures, le harcèlement, le racisme, le handicap, les conditions de travail, la pauvreté et le rôle d’aidant comme contextes. Une réaction à ces conditions ne suffit pas pour attribuer un diagnostic."),
)


def load_rows(source: Path) -> list[dict]:
    raw = source.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("Le ZIP ne correspond pas à la version OMS attendue.")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            raw = archive.read(TABLE_NAME)
    if hashlib.sha256(raw).hexdigest() != TABLE_SHA256:
        raise ValueError("Le tableau ne correspond pas à la version OMS attendue.")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"))


def title(row: dict) -> str:
    return re.sub(r"^(?:- )+", "", row["Title"]).strip()


def browser_url(row: dict) -> str:
    return f"{BROWSER}#{row['Linearization URI'].rsplit('/', 1)[-1]}"


def classify(code: str, row: dict, canons: dict) -> tuple[str, str, tuple[str, ...]]:
    root = code.split(".")[0]
    if code in canons:
        return "parcours", "diagnostic_distinct", (code,)
    if row["IsResidual"] == "True":
        return "hors_parcours", "code_residuel", ()
    if root == "6E40":
        return "hors_parcours", "facteur_associe", ()
    if root in PRESENTATIONS:
        return "rattache", "presentation_transversale", PRESENTATIONS[root]
    if code in {"6A03.3", "6C4G", "6D35"}:
        return "hors_parcours", "categorie_ouverte", ()
    if root.startswith("6C4"):
        if code[:-1] in canons:
            return "rattache", "presentation", (code[:-1],)
        if code == root:
            return "rattache", "classe_substance", tuple(c for c in canons if c.startswith(code + "."))
        descriptor = row["TitleEN"].casefold()
        if "harmful pattern" in descriptor:
            return "rattache", "syndrome_substance", (root + ".1" if root + ".1" in canons else "6C40.1",)
        if "dependence" in descriptor:
            return "rattache", "syndrome_substance", (root + ".2" if root + ".2" in canons else "6C40.2",)
        for phrase, representative in SUBSTANCE_SYNDROMES:
            if phrase in descriptor:
                return "rattache", "syndrome_substance", (representative,)
        if "specified" in descriptor and "mental or behavioural disorders" in descriptor:
            return "rattache", "syndromes_substance", ("6C46.70", "6C48.40", "6C45.72", "6C45.73")
        raise ValueError(f"Catégorie liée aux substances à examiner. {code} {descriptor}")
    for ancestor in (code[:-1], root):
        if ancestor in canons:
            return "rattache", "presentation", (ancestor,)
    children = tuple(c for c in canons if c.startswith(code + "."))
    if children:
        return "rattache", "groupe_code", children
    raise ValueError(f"Code du chapitre 06 sans décision éditoriale. {code}")


def build_catalogue(rows: list[dict], characters: list[dict]) -> dict:
    chapter = [row for row in rows if row["ChapterNo"] == "06"]
    by_uri = {row["Foundation URI"]: row for row in chapter}
    groups = [
        {"id": row["BlockId"], "label": title(row),
         "parent_id": by_uri[row["Parent"]]["BlockId"] or None, "url": browser_url(row)}
        for row in chapter if row["ClassKind"] == "block"
    ]
    canons = {character["code"]: character for character in characters}
    if len(canons) != len(characters) or len({c["id"] for c in characters}) != len(characters):
        raise ValueError("Les codes canoniques et les identifiants doivent être uniques.")
    by_code = {row["Code"]: row for row in chapter if row["ClassKind"] == "category"}
    if len(by_code) != 858 or len(groups) != 25 or set(canons) - by_code.keys():
        raise ValueError("Le chapitre 06 ne correspond pas au périmètre vérifié.")
    if any(by_code[c]["IsResidual"] == "True" for c in canons):
        raise ValueError("Un code résiduel ne peut pas servir de diagnostic canonique.")
    codes = []
    related = {character["id"]: [] for character in characters}
    for code, row in by_code.items():
        status, reason, target_codes = classify(code, row, canons)
        ids = [canons[target]["id"] for target in target_codes]
        if status == "rattache" and not ids:
            raise ValueError(f"Rattachement sans parcours. {code}")
        group = next((row[f"Grouping{i}"] for i in range(5, 0, -1) if row[f"Grouping{i}"]), None)
        parent = by_uri.get(row["Parent"])
        codes.append({
            "code": code,
            "label": title(row),
            "group_id": group,
            "parent_code": parent["Code"] or None if parent else None,
            "status": status,
            "reason": reason,
            "character_ids": ids,
            "url": browser_url(row),
        })
        if status == "rattache":
            for key in ids:
                related[key].append(code)
    registry = [
        {**character, "related_codes": related[character["id"]]}
        for character in characters
    ]
    outside = {row["Code"]: row for row in rows if row["Code"] and row["ChapterNo"] != "06"}
    subjects = [
        {"id": code, "code": code, "label": title(outside[code]),
         "chapter": outside[code]["ChapterNo"], "status": status,
         "editorial_note": note, "url": browser_url(outside[code])}
        for code, status, note in RELATED_SUBJECTS
    ]
    subjects.extend(
        {"id": slug, "label": label, "status": "contexte_non_diagnostique", "editorial_note": note}
        for slug, label, note in UNCODED_SUBJECTS
    )
    statuses = Counter(c["status"] for c in codes)
    by_timing = Counter(c["diagnosis"]["timing"] for c in registry)
    return {
        "schema_version": 1,
        "release": {
            "name": "CIM-11 MMS",
            "version": RELEASE,
            "language": "fr",
            "chapter": "06",
            "chapter_url": f"{BROWSER}#334423054",
            "source_url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "clinical_manual_url": MANUAL,
            "attribution": "International Classification of Diseases, Eleventh Revision (ICD-11), World Health Organization (WHO) 2019/2021, https://icd.who.int/browse11",
            "licence_url": "https://creativecommons.org/licenses/by-nd/3.0/igo/",
        },
        "counts": {
            "groups": len(groups),
            "codes": len(codes),
            "by_status": {status: statuses[status] for status in ("parcours", "rattache", "hors_parcours")},
            "characters": len(registry),
            "historical_wave_slots": len({c["wave"] for c in registry}),
            "diagnosis_timing": dict(by_timing),
        },
        "rules": RULES,
        "groups": groups,
        "codes": codes,
        "characters": registry,
        "related_subjects": subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ZIP OMS officiel ou fichier tabulé qu’il contient")
    parser.add_argument("--check", action="store_true", help="Vérifier le JSON sans l’écrire")
    args = parser.parse_args()
    current = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    rebuilt = build_catalogue(load_rows(args.source), current["characters"])
    if args.check:
        if rebuilt != current:
            raise SystemExit("Le catalogue ne correspond pas au tableau et aux canons.")
    else:
        CATALOGUE.write_text(json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rebuilt["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
