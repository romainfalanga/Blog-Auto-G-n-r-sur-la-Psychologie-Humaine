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
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "editorial" / "cartographie"
OUT = ROOT / "data" / "cartographie.json"
KINDS = {"trouble", "schema", "attachement", "tendance"}
STRENGTHS = {"forte": 3, "critere": 2, "moderee": 2, "limitee": 1, "theorique": 1}
SOURCE_TYPES = {"classification", "recommandation", "meta-analyse", "revue", "etude", "theorie"}
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


def apply_corrections(corr: dict, factors: dict, outcomes: dict, links: dict, sources: dict) -> list[str]:
    """Apply the pilot's traceable corrections after the independent review."""
    errors: list[str] = []
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
        links[key] = {**links[key], **{k: v for k, v in change.items() if k in ("strength", "summary", "sources")}}
    # A classification criterion is a precondition of the diagnosis, not a measured risk.
    for key, link in links.items():
        if link["strength"] == "forte" and all(sources.get(s, {}).get("type") == "classification" for s in link["sources"]):
            links[key] = dict(link, strength="critere")
    pref = corr.get("definitional_prefix")
    if pref:
        for key, link in links.items():
            if key[0].startswith(pref["factor"]) and key[1].startswith(pref["outcome"]):
                links[key] = dict(link, definitional=True)
    labels: dict[str, str] = {}
    for oid, o in outcomes.items():
        if not any(k[1] == oid for k in links):
            continue
        if o["label"] in labels:
            errors.append(f"conséquences en double : {labels[o['label']]} et {oid} ({o['label']})")
        labels[o["label"]] = oid
    return errors


def build() -> tuple[dict, list[str]]:
    errors: list[str] = []
    base = json.loads((SRC / "facteurs.json").read_text(encoding="utf-8"))
    groups = {g["id"] for g in base["groups"]}
    factors = {f["id"]: f for f in base["factors"]}
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
    for (fid, oid), link in links.items():
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
    data = {
        "schema_version": 1,
        "weights": STRENGTHS,
        "groups": base["groups"],
        "factors": base["factors"],
        "outcomes": sorted((o for o in outcomes.values() if o["id"] in used_outcomes), key=lambda o: o["label"]),
        "links": [links[k] for k in sorted(links)],
        "protections": protections,
        "notes": notes,
        "sources": sorted((s for s in sources.values() if s["id"] in used_sources), key=lambda s: s["id"]),
    }
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
    print(f"{len(data['factors'])} facteurs, {len(data['outcomes'])} conséquences, {len(data['links'])} liens, {len(data['sources'])} sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
