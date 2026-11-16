"""Rebuild the documentary audit from explicit semantic decisions and the graph.

This script validates and materializes reviewed decisions. It never infers a
psychological equivalence from a label, nor creates a factor/outcome link.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import build_cartographie as graph

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "editorial/cartographie/correspondances-2026-11-17.json"
REGISTRY = ROOT / "editorial/cartographie/couverture-referentiels.json"
OUTPUT = ROOT / "docs/audit-cartographie-correspondances-2026-11-17.json"
STATUSES = {"reference_exact", "agrege", "partiel", "hors_perimetre_questionnaire", "a_examiner"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build():
    data, errors = graph.build(check_audit=False)
    if errors:
        raise ValueError("\n".join(errors))
    decisions = read(SOURCE)
    registry = read(REGISTRY)
    core_fingerprint = fingerprint({key: data[key] for key in ("factors", "outcomes", "retired_outcomes", "links", "protections")})
    final_review = decisions.get("final_validation", {})
    if decisions.get("review_summary", {}).get("final_graph_check_pending") is not False or final_review.get("status") != "passed" or final_review.get("graph_core_sha256") != core_fingerprint:
        raise ValueError("Les décisions sémantiques doivent être revalidées sur le graphe final avant de produire l’audit")
    psychology, clinical = read(ROOT / "data/psychology_catalog.json"), read(ROOT / "data/clinical_catalog.json")
    reference = {row["id"]: row for row in psychology["patterns"]}
    rows = decisions["psychology_patterns"]
    if len(rows) != len(reference) or {row["id"] for row in rows} != reference.keys():
        raise ValueError("Every psychology pattern requires one explicit reviewed decision")
    targets = {kind: {row["id"]: row for row in data[key]} for kind, key in (("outcome", "outcomes"), ("factor", "factors"), ("resource", "protections"))}
    materialized = []
    for row in rows:
        if row.get("status") not in STATUSES or not row.get("reason") or not row.get("reviewed_by") or row.get("reviewed_on") != "2026-11-17":
            raise ValueError("Missing semantic review: " + row["id"])
        pattern = reference[row["id"]]
        if row.get("source_description") != pattern.get("description"):
            raise ValueError("Pattern description changed since review: " + row["id"])
        resolved = []
        for target in row.get("targets", []):
            actual = targets.get(target["kind"], {}).get(target["id"])
            if actual is None:
                raise ValueError("Unknown reviewed target: " + target["id"])
            resolved.append({"id": actual["id"], "kind": target["kind"], "label": actual["label"], "description": actual.get("description", "")})
        if row["status"] in {"reference_exact", "agrege", "partiel"} and not resolved:
            raise ValueError("Mapping without a target: " + row["id"])
        materialized.append({**row, "label": pattern["label"], "domain": pattern.get("domain"), "targets": resolved, "causal_equivalence_claimed": False})
    exact = {row["code"]: row["id"] for row in data["outcomes"] if row.get("code")}
    aggregate = {code: row["id"] for row in data["outcomes"] for code in row.get("classification_codes", [])}
    registered = {}
    for entry in registry["entries"]:
        if "CIM" in entry.get("referentiel", "").upper() or "ICD" in entry.get("referentiel", "").upper():
            registered.setdefault(entry.get("code"), []).append(entry)
    codes = []
    for code in clinical["codes"]:
        value = code["code"]
        if value in exact:
            status, outcome = "exact_code_reference", exact[value]
        elif value in aggregate:
            status, outcome = "aggregate_reference", aggregate[value]
        elif value.startswith(("6A00", "6A01", "6A02", "6A03", "6A04", "6A05")):
            status, outcome = "excluded_from_upbringing_inference", None
        else:
            status, outcome = "not_explicitly_referenced", None
        item = {"code": value, "label": code.get("label"), "group_id": code.get("group_id"), "status": status, "outcome": outcome}
        if value in registered:
            item["registry_decisions"] = [{key: entry[key] for key in ("statut", "raison", "lot") if key in entry} for entry in registered[value]]
        codes.append(item)
    outcomes = [{**row, "link_count": sum(link["outcome"] == row["id"] for link in data["links"]), "relation_types": dict(Counter(link["relation_type"] for link in data["links"] if link["outcome"] == row["id"]))} for row in data["outcomes"]]
    factors = [{"id": row["id"], "label": row["label"], "link_count": sum(link["factor"] == row["id"] for link in data["links"]), "mapped_psychology_patterns": [review["id"] for review in rows if any(t["kind"] == "factor" and t["id"] == row["id"] for t in review.get("targets", []))]} for row in data["factors"]]
    dsm = decisions["dsm5tr"]
    dsm = {**dsm, "registry_decisions": [entry for entry in registry["entries"] if "DSM" in entry.get("referentiel", "").upper()]}
    counts = {"psychology_reviewed": len(rows), "psychology_statuses": dict(Counter(row["status"] for row in rows)), "cartography_outcomes": len(outcomes), "cartography_retired_outcomes": len(data["retired_outcomes"]), "cartography_factors": len(factors), "cartography_resources": len(data["protections"]), "icd11_codes_reviewed": len(codes), "icd11_code_statuses": dict(Counter(row["status"] for row in codes)), "dsm_section_ii_chapters_reviewed": len(dsm["chapters"]), "dsm_chapter_statuses": dict(Counter(row["status"] for row in dsm["chapters"])), "reference_registry_decisions": len(registry["entries"]), "dsm_registry_decisions": len(dsm["registry_decisions"])}
    return {"schema_version": 2, "reviewed_on": "2026-11-17", "scope": "Décisions explicites sur les 161 repères internes, présence exacte des codes du catalogue CIM-11 et registre documentaire des neuf lots. Pas de validation clinique individuelle ni de lecture complète du DSM-5-TR.", "method": decisions["method"], "input_fingerprints": {"cartography_core_sha256": fingerprint({key: data[key] for key in ("factors", "outcomes", "retired_outcomes", "links", "protections")}), **{name + "_sha256": hashlib.sha256((ROOT / ("data/" + name + ".json")).read_bytes()).hexdigest() for name in ("psychology_catalog", "clinical_catalog")}, "semantic_decisions_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "reference_registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest()}, "counts": counts, "psychology_patterns": materialized, "icd11_codes": codes, "cartography_outcomes": outcomes, "retired_outcomes": data["retired_outcomes"], "cartography_factors": factors, "dsm5tr": dsm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        audit = build()
        text = json.dumps(audit, ensure_ascii=False, indent=1) + "\n"
        if args.check:
            if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
                raise ValueError("Audit périmé : lancer python3 scripts/audit_cartographie.py")
        else:
            OUTPUT.write_text(text, encoding="utf-8")
        print(json.dumps(audit["counts"], ensure_ascii=False))
        return 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
