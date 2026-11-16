"""Assemble and check the public « Cartographie » dataset.

Inputs are editorial files written by distinct research agents:
``editorial/cartographie/facteurs.json`` (closed list of childhood factors) and
``editorial/cartographie/liens-*.json`` (factor → outcome links with sources).
The output ``data/cartographie.json`` is read by the Hugo page; nothing here
calls a model. ``--check`` fails if the committed output is stale or invalid.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "editorial" / "cartographie"
OUT = ROOT / "data" / "cartographie.json"
KINDS = {"trouble", "schema", "attachement", "tendance"}
STRENGTHS = {"forte": 3, "critere": 2, "moderee": 2, "limitee": 1, "theorique": 1}
MAX_QUESTIONS_PER_GROUP = 14
MAX_QUESTIONS = 72
MAX_LINK_SUMMARY = 350
SOURCE_TYPES = {"classification", "recommandation", "meta-analyse", "revue", "etude", "theorie"}
RELATION_TYPES = {"empirical_association", "classification_context", "clinical_model", "definitional"}
YOUNG_SCHEMA_IDS = {
    "sch_abandon", "sch_mefiance", "sch_carence_affective", "sch_imperfection", "sch_isolement",
    "sch_dependance", "sch_vulnerabilite", "sch_fusion", "sch_echec", "sch_droits_excessifs",
    "sch_controle_insuffisant", "sch_assujettissement", "sch_abnegation", "sch_recherche_approbation",
    "sch_negativite", "sch_surcontrole", "sch_exigences_elevees", "sch_punition",
}
# Conditions that research does not attribute to upbringing or adversity.
FORBIDDEN_OUTCOME_PREFIXES = ("icd_6a00", "icd_6a01", "icd_6a02", "icd_6a03", "icd_6a04", "icd_6a05")
DETERMINISTIC = re.compile(r"\b(provoque|cause directement|entraîne forcément|entraine forcement|garantit|toujours)\b", re.I)
PERCENT = re.compile(r"\d+\s?%")


def page_exists(url: str) -> bool:
    if url.startswith("/personnages/"):
        slug = url.strip("/").split("/")[1]
        return (ROOT / "content" / "personnages" / f"{slug}.md").exists()
    if url.startswith("/posts/"):
        parts = url.strip("/").split("/")
        if len(parts) != 3:
            return False
        section, slug = parts[1], parts[2]
        for path in (ROOT / "content" / "posts" / section).glob("*.md"):
            head = path.read_text(encoding="utf-8")[:1500]
            match = re.search(r'^slug:\s*"?([^"\n]+)"?\s*$', head, re.M)
            if (match and match.group(1).strip() == slug) or path.stem == slug:
                return True
    return False


DOMAINS = [
    ("perceptions", "Perceptions et dissociation", r"psychot|6a2|dissoci|6b6"),
    ("humeur", "Humeur", r"dep_|dysthym|bipolaire|suicid|6a7|6a6|deuil|6b42"),
    ("anxiete", "Anxiété, peur et traumatisme", r"anx|panique|phobi|agoraphob|6b0|stress|6b4[0-3]|hypervigil|hypocondr|vulnerabilite"),
    ("relations", "Relations et attachement", r"att_|attach|sauveur|overresponsib|codepend|dependance_aff|confiance|mefiance|abandon|intimit|couple|partenaire|relation|approbation|reassurance|violence|isolement|social|conflict|appeasement|assujett|abnegation|fusion|6b44|6b45|6d1"),
    ("soi", "Image de soi", r"estime|honte|shame|imperfection|culpab|guilt|imposture|impostor|perfection|exigence|echec|conditional|critic|negativ|punition|droits|self_crit|carence|valeur"),
    ("emotions", "Émotions", r"alexithym|emotion|regulation|colere|anger|suppress|detach|surcontrole"),
    ("comportements", "Corps et comportements", r"substance|alcool|6c4|tabac|drogue|aliment|6b8|poids|obes|sommeil|sleep|douleur|sante|somat|6c2|ecran|jeu|travail|work|procrastin|externalis|antisoc|scolaire|sexual|sexuel|controle_insuffisant|impulsiv"),
]


def domain_of(outcome: dict) -> str:
    if outcome.get("kind") == "attachement":
        return "relations"
    text = (outcome["id"] + " " + outcome.get("label", "")).lower()
    for key, _, pattern in DOMAINS:
        if re.search(pattern, text):
            return key
    return {"attachement": "relations", "schema": "soi"}.get(outcome.get("kind"), "soi")


def apply_corrections(corr: dict, factors: dict, outcomes: dict, links: dict, sources: dict) -> list[str]:
    """Apply the pilot's traceable corrections after the independent review."""
    errors: list[str] = []
    for fid, fields in corr.get("factor_overrides", {}).items():
        if fid not in factors:
            errors.append(f"correction : facteur inconnu {fid}")
        else:
            factors[fid].update({k: v for k, v in fields.items() if k != "id"})
    for oid, fields in corr.get("outcome_overrides", {}).items():
        if oid in outcomes:
            merged = {**outcomes[oid], **{k: v for k, v in fields.items() if k != "id"}}
            outcomes[oid] = {k: v for k, v in merged.items() if v is not None}
    for oid in corr.get("remove_outcomes", []):
        outcomes.pop(oid, None)
        for key in [k for k in links if k[1] == oid]:
            del links[key]
    for move in corr.get("retarget", []):
        if move["to"] not in factors:
            errors.append(f"correction : facteur cible inconnu {move['to']}")
        for oid in move["outcomes"]:
            link = links.pop((move["from"], oid), None)
            if link is None:
                errors.append(f"correction : lien absent {move['from']} → {oid}")
                continue
            links[(move["to"], oid)] = dict(link, factor=move["to"])
    for s in corr.get("add_sources", []):
        sources.setdefault(s["id"], s)
    for sid, fields in corr.get("source_overrides", {}).items():
        if sid in sources:
            sources[sid] = {**sources[sid], **fields}
    for old, extra in corr.get("add_source_where", {}).items():
        for key, link in links.items():
            if old in link["sources"] and extra not in link["sources"]:
                links[key] = dict(link, sources=[extra] + link["sources"])
    for change in corr.get("link_overrides", []):
        key = (change["factor"], change["outcome"])
        if key not in links:
            errors.append(f"correction : lien absent {key}")
            continue
        if change.get("remove"):
            del links[key]
            continue
        links[key] = {**links[key], **{k: v for k, v in change.items() if k in ("strength", "summary", "sources", "relation_type", "definitional")}}
    # A classification criterion is a precondition of the diagnosis, not a measured risk.
    for key, link in links.items():
        if link["strength"] == "forte" and all(sources.get(s, {}).get("type") == "classification" for s in link["sources"]):
            links[key] = dict(link, strength="critere")
    pref = corr.get("definitional_prefix")
    if pref:
        for key, link in links.items():
            ressenti = key[0].startswith(pref["factor"]) or factors.get(key[0], {}).get("group") == "ressentis"
            if ressenti and key[1].startswith(pref["outcome"]):
                links[key] = dict(link, definitional=True)
    labels: dict[str, str] = {}
    for oid, o in outcomes.items():
        if not any(k[1] == oid for k in links):
            continue
        if o["label"] in labels:
            errors.append(f"conséquences en double : {labels[o['label']]} et {oid} ({o['label']})")
        labels[o["label"]] = oid
    return errors


def build(*, check_audit: bool = True) -> tuple[dict, list[str]]:
    errors: list[str] = []
    base = json.loads((SRC / "facteurs.json").read_text(encoding="utf-8"))
    groups = {g["id"] for g in base["groups"]}
    factors = {f["id"]: f for f in base["factors"]}
    if len(groups) != 6:
        errors.append("le questionnaire doit conserver ses six thèmes")
    if len(factors) != len(base["factors"]):
        errors.append("identifiant de facteur dupliqué")
    if len(factors) > MAX_QUESTIONS:
        errors.append(f"questionnaire : plafond de {MAX_QUESTIONS} questions dépassé")
    for group in groups:
        if sum(f.get("group") == group for f in factors.values()) > MAX_QUESTIONS_PER_GROUP:
            errors.append(f"thème {group} : plafond de {MAX_QUESTIONS_PER_GROUP} questions dépassé")
    for f in base["factors"]:
        if f["group"] not in groups:
            errors.append(f"facteur {f['id']} : groupe inconnu {f['group']}")
    corr = json.loads((SRC / "corrections.json").read_text(encoding="utf-8")) if (SRC / "corrections.json").exists() else {}
    aliases = corr.get("aliases", {})
    canon = corr.get("outcome_overrides", {})
    outcomes: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    links: dict[tuple[str, str], dict] = {}
    protections: list[dict] = []
    notes: dict[str, dict] = {}
    parts = sorted(glob.glob(str(SRC / "liens-*.json")))
    if not parts:
        errors.append("aucun fragment liens-*.json")
    for part_path in parts:
        part = json.loads(Path(part_path).read_text(encoding="utf-8"))
        name = Path(part_path).name
        for s in part.get("sources", []):
            if s.get("type") not in SOURCE_TYPES or not str(s.get("url", "")).startswith("http"):
                errors.append(f"{name} : source invalide {s.get('id')}")
            sources.setdefault(s["id"], s)
        for o in part.get("outcomes", []):
            if o["id"] in aliases:
                target = aliases[o["id"]]
                o = dict(canon[target]) if "id" in canon.get(target, {}) else dict(o, id=target)
            if o.get("kind") not in KINDS:
                errors.append(f"{name} : type inconnu pour {o.get('id')}")
            if o["id"].startswith(FORBIDDEN_OUTCOME_PREFIXES):
                errors.append(f"{name} : {o['id']} ne doit pas être relié à l'enfance vécue")
            url = o.get("url")
            if url and not page_exists(url):
                errors.append(f"{name} : page absente pour {o['id']} ({url})")
                o = dict(o, url=None)
            outcomes.setdefault(o["id"], o)
        for p in part.get("protections", []):
            protections.append(p)
        # Only structured public notes (object keyed by topic) are published;
        # list-shaped notes are editorial remarks kept in the fragment.
        if isinstance(part.get("notes"), dict):
            notes.update(part["notes"])
        for link in part.get("links", []):
            link = dict(link, outcome=aliases.get(link.get("outcome"), link.get("outcome")))
            key = (link.get("factor"), link.get("outcome"))
            if key[0] not in factors:
                errors.append(f"{name} : facteur inconnu {key[0]}")
            if link.get("strength") not in STRENGTHS:
                errors.append(f"{name} : force invalide {key}")
            if not link.get("sources"):
                errors.append(f"{name} : lien sans source {key}")
            summary = link.get("summary", "")
            if DETERMINISTIC.search(summary) or PERCENT.search(summary):
                errors.append(f"{name} : formulation déterministe ou chiffrée {key}")
            if key in links:
                # Keep the better-evidenced version and merge sources.
                old = links[key]
                if STRENGTHS[link["strength"]] > STRENGTHS[old["strength"]]:
                    link = dict(link, sources=sorted(set(link["sources"]) | set(old["sources"])))
                else:
                    link = dict(old, sources=sorted(set(link["sources"]) | set(old["sources"])))
            links[key] = link
    errors.extend(apply_corrections(corr, factors, outcomes, links, sources))
    # Nature of a link is separate from its editorial evidence label. A diagnostic
    # context and a clinical model must never become an empirical association.
    for link in links.values():
        classification_only = all(sources.get(s, {}).get("type") == "classification" for s in link["sources"])
        link["relation_type"] = link.get("relation_type") or (
            "definitional" if link.get("definitional") else
            "classification_context" if link["strength"] == "critere" or classification_only else
            "clinical_model" if link["strength"] == "theorique" else "empirical_association")
        if link["relation_type"] not in RELATION_TYPES:
            errors.append(f"nature de lien invalide : {link['factor']} → {link['outcome']}")
        if link["relation_type"] == "classification_context":
            link["strength"] = "critere"
    # Canonicalize identical reference URLs rather than treating aliases as studies.
    canonical_sources: dict[str, str] = {}
    source_aliases: dict[str, str] = {}
    for sid, source in sources.items():
        url = source["url"].rstrip("/")
        source_aliases[sid] = canonical_sources.setdefault(url, sid)
    for item in [*links.values(), *protections, *[n for n in notes.values() if isinstance(n, dict)]]:
        item["sources"] = list(dict.fromkeys(source_aliases.get(s, s) for s in item.get("sources", [])))
    for (fid, oid), link in links.items():
        if len(link.get("summary", "")) > MAX_LINK_SUMMARY:
            errors.append(f"lien {fid} → {oid} : résumé de plus de {MAX_LINK_SUMMARY} caractères")
        if oid not in outcomes:
            errors.append(f"lien {fid} → {oid} : conséquence non définie")
        for sid in link.get("sources", []):
            if sid not in sources:
                errors.append(f"lien {fid} → {oid} : source inconnue {sid}")
    for p in protections:
        for sid in p.get("sources", []):
            if sid not in sources:
                errors.append(f"protection {p.get('id')} : source inconnue {sid}")
    used_sources = {s for link in links.values() for s in link["sources"]}
    used_sources |= {s for p in protections for s in p.get("sources", [])}
    used_sources |= {s for n in notes.values() if isinstance(n, dict) for s in n.get("sources", [])}
    notes.pop("adversite_cumul", None)  # merged into the displayed « cumul » note
    used_outcomes = {oid for _, oid in links}
    for oid, outcome in outcomes.items():
        own_links = [link for (_, target), link in links.items() if target == oid]
        if outcome.get("age_scope") == "young_child_only" or outcome.get("category") in {"health_outcome", "categorical_personality_diagnosis"} or (own_links and all(link["relation_type"] == "classification_context" for link in own_links)):
            outcome["contextual_only"] = True
    retired_outcomes = corr.get("retired_outcomes", [])
    retired_ids = [o.get("id") for o in retired_outcomes]
    if len(retired_ids) != len(set(retired_ids)):
        errors.append("identifiant historique retiré dupliqué")
    for outcome in retired_outcomes:
        if outcome.get("id") in used_outcomes or not outcome.get("retired") or not outcome.get("contextual_only"):
            errors.append(f"sujet historique encore actif ou non protégé : {outcome.get('id')}")
    data = {
        "schema_version": 2,
        "interpretation": {
            "purpose": "Exploration de rapprochements, sans dépistage ni diagnostic.",
            "validated_ace_instrument": False,
            "ranking": "Niveau maximal du lien ; jamais une somme de risques. L’intensité reste un contexte déclaré.",
            "convergence": "Au moins deux familles de vécus avec des associations empiriques modérées ou solides ; aucun gain de certitude individuelle.",
            "warning": "Des facteurs associés, même cumulés, ne suffisent pas à établir un trouble et ne sont pas nécessairement requis pour qu’il apparaisse."
        },
        "weights": STRENGTHS,
        "groups": base["groups"],
        "factors": base["factors"],
        "domains": [{"id": k, "label": label} for k, label, _ in DOMAINS],
        "outcomes": sorted(({**o, "domain": o.get("domain") or domain_of(o)} for o in outcomes.values() if o["id"] in used_outcomes), key=lambda o: o["label"]),
        # Immutable historical meanings accepted on save, excluded from the graph.
        # An adult recognition must never migrate to a newly distinct child card.
        "retired_outcomes": sorted(retired_outcomes, key=lambda o: o["id"]),
        "links": [links[k] for k in sorted(links)],
        "protections": protections,
        "exclusive": json.loads((SRC / "corrections.json").read_text(encoding="utf-8")).get("exclusive_factors", []),
        "notes": notes,
        "sources": sorted((s for s in sources.values() if s["id"] in used_sources), key=lambda s: s["id"]),
    }
    for outcome in data["outcomes"]:
        outcome.setdefault("category", {"trouble": "diagnosis", "schema": "schema", "attachement": "attachment_pattern", "tendance": "tendency"}[outcome["kind"]])
    clinical = json.loads((ROOT / "data/clinical_catalog.json").read_text(encoding="utf-8"))
    psychology = json.loads((ROOT / "data/psychology_catalog.json").read_text(encoding="utf-8"))
    exact_codes = sorted({o["code"] for o in data["outcomes"] if o.get("code") in {c["code"] for c in clinical["codes"]}})
    pattern_ids = {p["id"] for p in psychology["patterns"]}
    pattern_links = {o["id"][4:]: o["id"] for o in data["outcomes"] if o["id"].startswith("pat_") and o["id"][4:] in pattern_ids}
    data["coverage"] = {
        "scope": "Sélection documentée de vécus de l’enfance ; ni inventaire de la psychologie ni couverture de tous les diagnostics.",
        "outcomes": len(data["outcomes"]),
        "kinds": {kind: sum(o["kind"] == kind for o in data["outcomes"]) for kind in sorted(KINDS)},
        "icd11": {"reference": "CIM-11 MMS 2026-01, chapitre 06", "catalog_codes": len(clinical["codes"]), "exact_code_references": exact_codes, "exact_code_count": len(exact_codes), "aggregate_references": [{"outcome": o["id"], "codes": o["classification_codes"]} for o in data["outcomes"] if o.get("classification_codes")], "not_exactly_referenced_count": len(clinical["codes"]) - len(exact_codes), "complete": False, "note": "Un code cité ou agrégé n’atteste ni diagnostic ni validation de chacun de ses critères."},
        "dsm5tr": {"complete": False, "status": "diagnosis_by_diagnosis_comparison_pending", "note": "Aucun inventaire diagnostic DSM-5-TR complet ni équivalence automatique CIM/DSM."},
        "young_schemas": {"reference_count": len(YOUNG_SCHEMA_IDS), "represented_count": len(YOUNG_SCHEMA_IDS & used_outcomes), "missing_ids": sorted(YOUNG_SCHEMA_IDS - used_outcomes), "source": "young_schemas", "note": "Présence des concepts du modèle ; ni validation psychométrique ni causalité établie."},
        "psychology_catalog": {"catalog_patterns": len(pattern_ids), "explicit_prefix_links": pattern_links, "without_explicit_prefix_link": sorted(pattern_ids - pattern_links.keys()), "note": "Correspondances d’identifiants uniquement ; les recouvrements sémantiques et les ressources restent à examiner."},
        "known_exclusions": [{"scope": "icd_6a00–icd_6a05", "reason": "Aucun lien causal avec l’éducation ou les vécus listés n’est inféré pour les troubles neurodéveloppementaux."}],
        "remaining_review": ["Comparaison DSM-5-TR diagnostic par diagnostic", "Populations, âges, méthodes et limites pour chaque lien", "Recouvrements du catalogue psychologique au-delà des identifiants explicites", f"Évaluation clinique indépendante des {len(data['links'])} liens, au-delà des contrôles de structure"]
    }
    # The complete semantic review is a documentary artifact, not a causal-link
    # generator. Check its inputs and expose honest counters without importing
    # its editorial correspondences into the recommendation graph.
    audit_path = ROOT / "docs/audit-cartographie-correspondances-2026-11-17.json"
    if check_audit and not audit_path.exists():
        errors.append("audit des correspondances du 17 novembre absent : lancer scripts/audit_cartographie.py")
    if check_audit and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        core = {key: data[key] for key in ("factors", "outcomes", "retired_outcomes", "links", "protections")}
        fingerprint = hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if fingerprint != audit.get("input_fingerprints", {}).get("cartography_core_sha256"):
            errors.append("audit des correspondances périmé : les facteurs, sujets, liens ou ressources ont changé")
        for name in ("psychology_catalog", "clinical_catalog"):
            actual = hashlib.sha256((ROOT / f"data/{name}.json").read_bytes()).hexdigest()
            if actual != audit.get("input_fingerprints", {}).get(f"{name}_sha256"):
                errors.append(f"audit des correspondances périmé : {name} a changé")
        for key, filename in (("semantic_decisions_sha256", "correspondances-2026-11-17.json"), ("reference_registry_sha256", "couverture-referentiels.json")):
            path = SRC / filename
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != audit.get("input_fingerprints", {}).get(key):
                errors.append(f"audit des correspondances périmé : {filename} a changé")
        reviewed = audit.get("psychology_patterns", [])
        if len(reviewed) != len(pattern_ids) or {r.get("id") for r in reviewed} != pattern_ids:
            errors.append("audit des correspondances incomplet ou dupliqué")
        data["coverage"]["psychology_catalog"]["semantic_review"] = {
            "reviewed_patterns": len(reviewed), "status_counts": audit["counts"]["psychology_statuses"],
            "report": str(audit_path.relative_to(ROOT)),
            "causal_validation": False,
        }
        data["coverage"]["psychology_catalog"]["note"] = "Les 161 repères ont une décision sémantique documentée ; un identifiant commun, un regroupement ou un recouvrement partiel ne prouvent ni causalité ni couverture clinique complète."
        data["coverage"]["dsm5tr"].update({"chapters_reviewed": audit["counts"]["dsm_section_ii_chapters_reviewed"], "chapter_status_counts": audit["counts"]["dsm_chapter_statuses"], "level": "chapter_thematic_presence_only"})
        data["coverage"]["remaining_review"][2] = "Relecture clinique indépendante des correspondances sémantiques documentées, sans les transformer en liens causaux"
    registry_path = SRC / "couverture-referentiels.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entries = registry.get("entries", [])
        statuses = {status: sum(row.get("statut") == status for row in entries) for status in sorted({row["statut"] for row in entries})}
        data["coverage"]["reference_registry"] = {"report": str(registry_path.relative_to(ROOT)), "decisions": len(entries), "statuses": statuses, "exhaustive_diagnostic_criteria_review": False}
        data["coverage"]["known_exclusions"].append({"scope": "Registre documenté des neuf lots", "reason": "Les exclusions et recherches non concluantes sont distinguées des sujets reliés.", "report": str(registry_path.relative_to(ROOT))})
    return data, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data, errors = build()
    text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("data/cartographie.json est périmé : lancer python3 scripts/build_cartographie.py", file=sys.stderr)
            return 1
    else:
        OUT.write_text(text, encoding="utf-8")
        # Keep Ambre's compact copy in step with the public dataset.
        import subprocess
        subprocess.run(["node", str(ROOT / "scripts" / "build_cartographie_module.mjs")], check=True)
    print(f"{len(data['factors'])} facteurs, {len(data['outcomes'])} conséquences, {len(data['links'])} liens, {len(data['sources'])} sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
