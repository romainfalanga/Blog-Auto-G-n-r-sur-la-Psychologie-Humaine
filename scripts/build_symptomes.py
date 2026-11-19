"""Check « Mes symptômes » content and generate its runtime copies.

Source of truth: ``editorial/symptomes/symptomes.json`` (instruments recopied
from the archived official forms, site landmarks paraphrasing ICD-11/CDDR,
safety rules, rule-based crosswalk to ``data/cartographie.json`` subjects).
Archives: ``editorial/symptomes/licences/MANIFEST.json`` (SHA-256 of every
licence page and form). Nothing here calls a model.

Outputs (published modules only):
- ``data/symptomes.json`` read by Hugo;
- ``journal/src/symptomes-data.generated.mjs`` for the private Worker.

``--check`` validates and fails if either output is stale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "editorial" / "symptomes" / "symptomes.json"
ARCHIVES = ROOT / "editorial" / "symptomes" / "licences"
MANIFEST = ARCHIVES / "MANIFEST.json"
CARTO = ROOT / "data" / "cartographie.json"
OUT_JSON = ROOT / "data" / "symptomes.json"
OUT_MJS = ROOT / "journal" / "src" / "symptomes-data.generated.mjs"

EXPECTED_MODULES = 23
SENSITIVE = {"perceptions", "dissociation", "personnalite", "sexualite", "alcool", "cannabis", "autres_substances"}
TEMPS1 = {"humeur", "anxiete"}
INSTRUMENTS = {"phq2", "phq9", "gad2", "gad7", "phq15", "itq"}
ROLES = {"repere", "entree", "prerequis", "contexte", "exclusion", "gene", "securite"}
RULE_KINDS = {"instrument", "entree", "repere", "repere_site", "contexte", "grappe", "orientation"}
REVIEW_STATES = {"en_attente", "relu", "retenu", "écarté"}
PUBLISHABLE = {"relu", "retenu"}
MAX_WORDS = 30
NEGATIONS = {"pas", "jamais", "aucun", "aucune", "rien", "sans", "ni", "nul", "nulle"}
FORBIDDEN = [
    r"\bvous avez (?:un|une|le|la|des) (?:trouble|maladie|dépression|tspt)",
    r"\bvous souffrez d", r"diagnostic (?:de|d’|probable)", r"\bprobablement\b", r"%",
    r"\bvous êtes (?:bipolaire|déprimée?|anorexique|boulimique|alcoolique|borderline|psychotique|schizophrène|toxicomane)",
    r"risque (?:faible|élevé)",
]
REQUIRED_RESOURCES = {"15", "112", "3114", "114", "3919", "119", "0 800 235 236"}
V11_PHRASES = {
    "positive_note": "Un repère au-dessus du seuil signale qu'un échange avec un professionnel est utile ; beaucoup de personnes dans ce cas n'ont pas de trouble après un entretien.",
    "negative_note": "Cela n'exclut pas une difficulté ; si quelque chose vous inquiète, parlez-en à un professionnel.",
    "crossing_title": "Présent dans vos deux questionnaires",
    "crossing_note": "Beaucoup de personnes ayant ces vécus n'ont pas ces difficultés, et inversement ; ce rapprochement ne montre pas une cause.",
}
RETIRED_V11 = {"ten_symptomes_premenstruels", "ten_rumination", "ten_inferences_negatives", "ten_nevrosisme", "ten_fibromyalgie", "ten_migraine"}
EMPIRICAL_STRENGTHS = {"forte", "moderee", "limitee"}


class Invalid(Exception):
    pass


def fail(msg: str) -> None:
    raise Invalid(msg)


def norm(text: str) -> str:
    return text.replace("’", "'")


def words(text: str) -> int:
    return len([w for w in re.split(r"[\s]+", text) if re.search(r"\w", w)])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relation_type(link: dict) -> str:
    if link.get("relation_type"):
        return link["relation_type"]
    if link.get("definitional"):
        return "definitional"
    if link.get("strength") == "critere":
        return "classification_context"
    if link.get("strength") == "theorique":
        return "clinical_model"
    return "empirical_association"


def associated_counts(carto: dict) -> dict[str, int]:
    """Distinct factors with an empirical link, as computeCartography counts them."""
    found: dict[str, set] = {}
    for link in carto["links"]:
        if link.get("definitional") or relation_type(link) != "empirical_association":
            continue
        if link.get("strength") in EMPIRICAL_STRENGTHS:
            found.setdefault(link["outcome"], set()).add(link["factor"])
    return {k: len(v) for k, v in found.items()}


def check_archives(data: dict) -> dict[str, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    entries = {}
    for entry in manifest.get("files", []):
        path = ARCHIVES / entry["path"]
        if not path.is_file():
            fail(f"archive manquante : {entry['path']}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            fail(f"empreinte invalide : {entry['path']}")
        if sha256(path) != entry["sha256"]:
            fail(f"empreinte différente : {entry['path']}")
        if entry.get("kind") == "original" and not str(entry.get("url", "")).startswith("https://"):
            fail(f"URL d’origine absente : {entry['path']}")
        if entry.get("kind") == "extrait_texte" and "derived_from" in entry and entry["derived_from"] not in {e["path"] for e in manifest["files"]}:
            fail(f"extrait sans original : {entry['path']}")
        entries[entry["path"]] = entry
    on_disk = {p.name for p in ARCHIVES.iterdir() if p.is_file() and p.name != "MANIFEST.json"}
    if on_disk != set(entries):
        fail(f"fichiers hors manifeste ou absents : {sorted(on_disk ^ set(entries))}")
    refs = [i.get("source_file") for i in data["instruments"].values() if i.get("source_file")]
    refs += [s["archive"] for s in data["sources"].values() if s.get("archive")]
    for lic in data["licences"].values():
        refs += [q["archive"] for q in lic.get("quotes", [])]
        if lic.get("terms_of_use"):
            refs.append(lic["terms_of_use"]["archive"])
    for ref in refs:
        if ref not in entries:
            fail(f"archive citée mais absente du manifeste : {ref}")
    return entries


def check_official_text(inst_id: str, inst: dict) -> None:
    """Every official item must appear in the archived text extract (spacing and apostrophes normalised)."""
    src = inst.get("source_file")
    if not src:
        return
    extract = ARCHIVES / (src.rsplit(".", 1)[0] + ".txt")
    text = extract.read_text(encoding="utf8")
    flat = text.replace("‟", "’").replace("„", "’")
    flat = re.sub(r"(?<!\S)[0-4](?!\S)", " ", flat)  # answer-column digits interleaved by the layout
    flat = re.sub(r"\s+", " ", flat)
    def present(s: str) -> bool:
        s = re.sub(r"\s+", " ", re.sub(r"(?<!\S)[0-4](?!\S)", " ", s)).strip()
        if s in flat:
            return True
        # Layout extraction interleaves column headers: every word must appear,
        # in order, with at most a few foreign words between two of them.
        ws, toks = s.split(" "), flat.split(" ")
        for start in (i for i, t in enumerate(toks) if t == ws[0]):
            pos = start
            for w in ws[1:]:
                window = toks[pos + 1:pos + 1 + 9]
                if w not in window:
                    break
                pos += 1 + window.index(w)
            else:
                return True
        return False
    for item in inst.get("items", []):
        if isinstance(item, dict) and not present(item["text"]):
            fail(f"{inst_id} : item {item['id']} absent du texte archivé {extract.name}")
    for key in ("stem", "notice"):
        if inst.get(key) and not present(inst[key]):
            fail(f"{inst_id} : {key} absent du texte archivé")


def check_bands(inst_id: str, inst: dict) -> None:
    sc = inst["scoring"]
    bands = inst.get("bands")
    if not bands:
        return
    if bands[0]["min"] != sc["min"] or bands[-1]["max"] != sc["max"]:
        fail(f"{inst_id} : paliers ne couvrent pas {sc['min']}–{sc['max']}")
    for a, b in zip(bands, bands[1:]):
        if b["min"] != a["max"] + 1:
            fail(f"{inst_id} : paliers non contigus")
    mins = [b["min"] for b in bands]
    if inst.get("screening_threshold") not in mins:
        fail(f"{inst_id} : repère hors paliers")
    item_max = max(o["value"] for o in inst["_scale"]["options"])
    n = len(inst["items"])
    if sc["max"] != n * min(item_max, 2 if inst_id == "phq15" else item_max):
        fail(f"{inst_id} : maximum {sc['max']} incohérent avec {n} items")


def check_expr(expr, qids: set, where: str) -> None:
    if not isinstance(expr, dict) or not expr:
        fail(f"{where} : expression vide")
    keys = set(expr)
    if keys == {"all"} or keys == {"any"}:
        items = expr[next(iter(keys))]
        if not isinstance(items, list) or not items:
            fail(f"{where} : liste vide")
        for e in items:
            check_expr(e, qids, where)
    elif keys == {"not"}:
        check_expr(expr["not"], qids, where)
    elif keys == {"atleast"}:
        a = expr["atleast"]
        if not (isinstance(a.get("n"), int) and 1 <= a["n"] <= len(a.get("of", []))):
            fail(f"{where} : atleast invalide")
        for e in a["of"]:
            check_expr(e, qids, where)
    elif "q" in keys:
        if expr["q"] not in qids:
            fail(f"{where} : question inconnue {expr['q']}")
        if not (keys <= {"q", "in", "gte"} and len(keys) == 2):
            fail(f"{where} : condition invalide")
    elif keys == {"score", "gte"}:
        if expr["score"] not in INSTRUMENTS - {"itq"}:
            fail(f"{where} : score inconnu")
    elif keys == {"itq_result"}:
        if expr["itq_result"] not in {"tspt", "tspt_c", "aucun"}:
            fail(f"{where} : résultat ITQ inconnu")
    elif keys == {"itq_cluster"}:
        pass
    else:
        fail(f"{where} : opérateur inconnu {sorted(keys)}")


def check_text(text: str, where: str) -> None:
    low = text.lower()
    for pat in FORBIDDEN:
        if re.search(pat, low):
            fail(f"{where} : formulation interdite ({pat})")


def validate(data: dict, carto: dict) -> dict:
    if data.get("schema_version") != 1 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.\d+", data.get("version", "")):
        fail("version du contenu absente")
    check_archives(data)
    scales, periods = data["scales"], data["periods"]
    for key, phrase in V11_PHRASES.items():
        if norm(data["fixed_texts"].get(key, "")) != phrase:
            fail(f"phrase fixe v1.1 modifiée : {key}")
    instruments = data["instruments"]
    if set(instruments) != INSTRUMENTS:
        fail(f"instruments attendus {sorted(INSTRUMENTS)}")
    all_qids: set[str] = set()
    for iid, inst in instruments.items():
        for k in ("name", "version", "scoring", "retest_days", "licence", "period"):
            if k not in inst:
                fail(f"{iid} : champ {k} manquant")
        if inst["licence"] not in data["licences"] or data["licences"][inst["licence"]]["status"] != "retenu":
            fail(f"{iid} : licence non retenue")
        if inst["period"] not in periods:
            fail(f"{iid} : période inconnue")
        if "derived_from" in inst:
            parent = instruments[inst["derived_from"]]
            ids = [i["id"] for i in parent["items"]]
            if inst["items"] != ids[:2]:
                fail(f"{iid} : items dérivés incorrects")
            if not (inst["scoring"]["min"] <= inst["entry_threshold"] <= inst["scoring"]["max"]):
                fail(f"{iid} : entrée hors échelle")
            if inst["entry_threshold"] != 2:
                fail(f"{iid} : entrée v1.1 = 2")
            continue
        if inst.get("scale") not in scales or scales[inst["scale"]]["owner"] != "officielle":
            fail(f"{iid} : échelle officielle manquante")
        inst["_scale"] = scales[inst["scale"]]
        for item in inst["items"]:
            if item["id"] in all_qids:
                fail(f"identifiant dupliqué {item['id']}")
            all_qids.add(item["id"])
            if item.get("scale") and item["scale"] not in scales:
                fail(f"{item['id']} : échelle inconnue")
        check_official_text(iid, inst)
        if iid != "itq":
            check_bands(iid, inst)
        del inst["_scale"]
    if instruments["phq9"]["screening_threshold"] != 10 or instruments["gad7"]["screening_threshold"] != 10:
        fail("repère PHQ-9 / GAD-7 = 10")
    if [b["min"] for b in instruments["phq9"]["bands"]] != [0, 5, 10, 15, 20]:
        fail("paliers PHQ-9 5/10/15/20")
    if [b["min"] for b in instruments["phq15"]["bands"]] != [0, 5, 10, 15]:
        fail("paliers PHQ-15 5/10/15")
    itq = instruments["itq"]
    clusters = itq["scoring"]["clusters"]
    covered = sorted(i for c in clusters.values() for i in c)
    if covered != sorted(i["id"] for i in itq["items"]) or len(covered) != 18:
        fail("ITQ : grappes ≠ 18 items")
    if not itq["scoring"]["exclusive"] or set(itq["scoring"]["tspt"]) | set(itq["scoring"]["pos"]) != set(clusters):
        fail("ITQ : règles TSPT / TSPT-C")
    for wording in itq["forbidden_wording"]:
        if wording in json.dumps(data["modules"], ensure_ascii=False).lower():
            fail(f"ITQ : « {wording} » repris")
    all_qids.add(itq["event"]["when"]["id"])
    safety = data["safety"]
    all_qids.add(safety["universal_question"]["id"])

    outcomes = {o["id"]: o for o in carto["outcomes"]}
    counts = associated_counts(carto)
    modules = data["modules"]
    if len(modules) != EXPECTED_MODULES:
        fail(f"{len(modules)} modules au lieu de {EXPECTED_MODULES}")
    if [m["order"] for m in modules] != list(range(1, EXPECTED_MODULES + 1)):
        fail("ordre des modules")
    mids = [m["id"] for m in modules]
    if len(set(mids)) != len(mids):
        fail("module dupliqué")
    for m in modules:
        for q in m["questions"]:
            if q["id"] in all_qids:
                fail(f"identifiant dupliqué {q['id']}")
            all_qids.add(q["id"])
    subject_modules: dict[str, list[str]] = {}
    for m in modules:
        mid = m["id"]
        for k in ("label", "temps", "sensible", "publie", "revue", "questions", "rules", "entry", "icd11"):
            if k not in m:
                fail(f"{mid} : champ {k} manquant")
        if m["sensible"] != (mid in SENSITIVE):
            fail(f"{mid} : drapeau sensible")
        if (m["temps"] == "temps1") != (mid in TEMPS1):
            fail(f"{mid} : temps")
        state = m["revue"].get("statut")
        if state not in REVIEW_STATES:
            fail(f"{mid} : statut de relecture inconnu")
        if m["publie"] and state not in PUBLISHABLE:
            fail(f"{mid} : publié sans relecture acceptée")
        if state in PUBLISHABLE and not m["revue"].get("rapport"):
            fail(f"{mid} : relecture sans rapport")
        genes = [q for q in m["questions"] if q["role"] == "gene"]
        if len(genes) != 1 or genes[0]["scale"] != "non_un_peu_nettement":
            fail(f"{mid} : une question de gêne Non / Un peu / Nettement attendue")
        reachable = set(m["entry"].get("questions", []))
        deepen = m.get("deepen", {})
        reachable |= set(deepen.get("questions", []))
        for b in deepen.get("branches", []):
            reachable |= set(b.get("questions", []))
            check_expr(b["if"], all_qids, f"{mid} branche")
        for inst_key in ("instrument",):
            for holder in (m["entry"], deepen):
                if inst_key in holder and holder[inst_key] not in INSTRUMENTS:
                    fail(f"{mid} : instrument inconnu")
        check_expr(m["entry"]["positive_if"], all_qids, f"{mid} entrée")
        retests = []
        for q in m["questions"]:
            where = f"{mid}.{q['id']}"
            for k in ("id", "text", "scale", "period", "role", "validated", "label", "basis"):
                if k not in q:
                    fail(f"{where} : champ {k} manquant")
            if q["validated"] is not False or q["label"] != "repere_non_valide":
                fail(f"{where} : repère du site non marqué non validé")
            if q["scale"] not in scales or scales[q["scale"]]["owner"] != "site":
                fail(f"{where} : échelle du site attendue")
            if q["period"] not in periods:
                fail(f"{where} : période inconnue")
            if q["role"] not in ROLES:
                fail(f"{where} : rôle inconnu")
            if q["basis"]["source"] not in data["sources"]:
                fail(f"{where} : source inconnue")
            if q["basis"]["source"] in {"cddr2024", "icd11_mms"} and not q["basis"].get("code"):
                fail(f"{where} : code CIM-11 manquant")
            if words(q["text"]) > MAX_WORDS:
                fail(f"{where} : {words(q['text'])} mots (> {MAX_WORDS})")
            toks = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", q["text"].lower())
            if sum(t in NEGATIONS for t in toks) >= 2:
                fail(f"{where} : double négation possible")
            if not q["text"].rstrip().endswith("?"):
                fail(f"{where} : question sans point d’interrogation")
            check_text(q["text"], where)
            if q["role"] not in {"gene", "securite"} and q["id"] not in reachable:
                fail(f"{where} : question jamais posée (ni entrée ni approfondissement)")
            p = periods[q["period"]]
            if p.get("retest_days"):
                retests.append(p["retest_days"])
        for holder in (m["entry"], deepen):
            if holder.get("instrument"):
                retests.append(instruments[holder["instrument"]]["retest_days"])
        for b in deepen.get("branches", []):
            if b.get("instrument"):
                retests.append(instruments[b["instrument"]]["retest_days"])
        m["_retest_days"] = min(retests) if retests else None
        if not m["rules"]:
            fail(f"{mid} : aucune règle")
        for r in m["rules"]:
            where = f"{mid}.{r['id']}"
            if r.get("kind") not in RULE_KINDS:
                fail(f"{where} : type de règle inconnu")
            check_expr(r["when"], all_qids, where)
            if r["kind"] not in {"grappe", "contexte"} and not r["result_positive"]:
                fail(f"{where} : texte positif manquant")
            for key in ("result_positive", "result_negative"):
                check_text(r.get(key, ""), where)
            for sid in r["subjects"]:
                o = outcomes.get(sid)
                if not o:
                    fail(f"{where} : sujet inconnu {sid}")
                if o.get("retired") or o.get("contextual_only"):
                    fail(f"{where} : sujet contextuel ou retiré {sid}")
                if counts.get(sid, 0) <= 0:
                    fail(f"{where} : sujet sans lien empirique {sid}")
                if sid in RETIRED_V11 or sid.startswith("sch_"):
                    fail(f"{where} : sujet retiré en v1.1 {sid}")
                if sid in data["crosswalk"]["safety_subjects"] or sid in data["crosswalk"]["not_asked"]:
                    fail(f"{where} : sujet de sécurité ou non demandé {sid}")
                subject_modules.setdefault(sid, [])
                if mid not in subject_modules[sid]:
                    subject_modules[sid].append(mid)
        if not any(r.get("result_negative") for r in m["rules"]) and not m.get("module_negative"):
            fail(f"{mid} : texte de résultat négatif manquant")
    # Safety
    uq = safety["universal_question"]
    if uq["scale"] != "non_oui_pref" or uq.get("scored") is not False:
        fail("question de sécurité universelle")
    numbers = {r["number"] for r in safety["resources"]}
    if not REQUIRED_RESOURCES <= numbers:
        fail(f"ressources manquantes : {sorted(REQUIRED_RESOURCES - numbers)}")
    kinds = {t["kind"] for t in safety["triggers"]}
    for t in safety["triggers"]:
        check_expr(t["when"], all_qids, f"sécurité {t['id']}")
    for r in safety["resources"]:
        if not set(r["show"]) <= kinds | {"mineur", "jeune_adulte"}:
            fail(f"ressource {r['id']} : règle d’affichage inconnue")
    if safety["display"]["risk_score"] is not False or safety["adult_gate"]["min_age"] != 18:
        fail("règles de sécurité v1.1")
    if "pcl5" in json.dumps(data["instruments"]).lower():
        fail("PCL-5 doit être exclu")
    return {"subject_modules": subject_modules}


def build(data: dict, extra: dict) -> dict:
    published = [m for m in data["modules"] if m["publie"]]
    keep_q = ("id", "text", "hint", "scale", "period", "role", "basis", "note")
    out_modules = []
    for m in published:
        out_modules.append({
            "id": m["id"], "order": m["order"], "label": m["label"], "temps": m["temps"], "sensible": m["sensible"],
            "icd11": m["icd11"], "retest_days": m["_retest_days"], "entry": m["entry"], **({"deepen": m["deepen"]} if m.get("deepen") else {}),
            "questions": [{k: q[k] for k in keep_q if k in q} for q in m["questions"]],
            "rules": [{k: r[k] for k in ("id", "kind", "when", "result_positive", "result_negative", "subjects", "instrument", "exclusive_group", "display") if k in r} for r in m["rules"]],
            **({"module_negative": m["module_negative"]} if m.get("module_negative") else {}),
            **({"medical_note": m["medical_note"]} if m.get("medical_note") else {}),
        })
    used = {m["entry"].get("instrument") for m in published} | {m.get("deepen", {}).get("instrument") for m in published}
    used |= {b.get("instrument") for m in published for b in m.get("deepen", {}).get("branches", [])}
    used.discard(None)
    if "phq2" in used:
        used.add("phq9")
    if "gad2" in used:
        used.add("gad7")
    instruments = {k: v for k, v in data["instruments"].items() if k in used}
    order = {m["id"]: m["order"] for m in published}
    subjects = {sid: sorted((x for x in mods if x in order), key=order.get) for sid, mods in extra["subject_modules"].items()}
    subjects = {k: v for k, v in sorted(subjects.items()) if v}
    return {
        "schema_version": data["schema_version"], "version": data["version"],
        "generated_from": "editorial/symptomes/symptomes.json",
        "periods": data["periods"], "scales": data["scales"], "fixed_texts": data["fixed_texts"],
        "safety": data["safety"], "instruments": instruments, "modules": out_modules,
        "subject_modules": subjects, "crossing": data["crossing"],
        "sources": {k: {"citation": v["citation"], **({"url": v["url"]} if v.get("url") else {})} for k, v in data["sources"].items()},
        "unpublished_modules": [m["id"] for m in data["modules"] if not m["publie"]],
    }


def render(payload: dict) -> tuple[str, str]:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False) + "\n"
    mjs = ("// Generated by scripts/build_symptomes.py from editorial/symptomes/symptomes.json. Do not edit.\n"
           "export const SYMPTOMES = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    return text, mjs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="valide et échoue si les fichiers générés sont périmés")
    args = parser.parse_args()
    try:
        data = json.loads(SRC.read_text(encoding="utf8"))
        carto = json.loads(CARTO.read_text(encoding="utf8"))
        extra = validate(data, carto)
        text, mjs = render(build(data, extra))
    except (Invalid, KeyError, TypeError, ValueError) as exc:
        print(f"symptomes invalide : {exc}", file=sys.stderr)
        return 1
    if args.check:
        stale = [p for p, c in ((OUT_JSON, text), (OUT_MJS, mjs)) if not p.is_file() or p.read_text(encoding="utf8") != c]
        if stale:
            print("fichiers périmés : " + ", ".join(str(p.relative_to(ROOT)) for p in stale) + " ; lancer python3 scripts/build_symptomes.py", file=sys.stderr)
            return 1
        print("symptomes à jour")
        return 0
    OUT_JSON.write_text(text, encoding="utf8")
    OUT_MJS.write_text(mjs, encoding="utf8")
    published = sum(1 for m in data["modules"] if m["publie"])
    print(f"symptomes : {published}/{len(data['modules'])} modules publiés")
    return 0


if __name__ == "__main__":
    sys.exit(main())
